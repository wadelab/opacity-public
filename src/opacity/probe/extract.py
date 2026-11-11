"""A model's internal states on the checker prompts it read (runs on a GPU node).

    python -m opacity.probe.extract Qwen3.5-27B --revision <sha> \
        --inputs runs/probe/inputs/Qwen3.5-27B.jsonl --out runs/probe/states/Qwen3.5-27B.npz [--limit 20]

Each input line holds the messages one checker run sent (system, user),
copied from its crosscheck log. The prompt is built as vLLM built it: the
model's chat template rendered to text with the keyword arguments in
`opacity.probe.PROBES`, then tokenised without special tokens. One forward
pass, no generation. Stored per prompt: the state at the last prompt token
at a quarter, half and three quarters of the model's depth (float16), the
prompt length, the length under the template its source run used beside the
length vLLM logged for that run, and the most likely next token.

Two instrument checks come from these, and the summary prints both. Every
source-template length must equal the logged one, or the messages were not
rebuilt exactly (the 27B's held-out prompts come from a run with thinking on,
so their check renders with thinking on; the forward pass never does). For
sources run greedy with thinking off, the most likely next token must begin
the reply the checker wrote, or this is not the forward pass that produced
the verdict.

A line with a `completion` is the writing probe (docs/vetting.md 4.8): the
completion is appended to the rendered prompt as the model's reply, and the
state is averaged over the completion's tokens instead of taken at the last
prompt token. Stored beside it: the completion's length in tokens and the
share of its tokens that were the model's most likely next token (high for a
reply the model wrote after this same prompt; a sampled reply never reaches
1).

Weights load offline at the given revision. The run writes a checkpoint
every 200 prompts and resumes from it.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from opacity.probe import PROBES, depth_index

CHECKPOINT_EVERY = 200


def first_token_ok(top1_text: str, logged_reply: str) -> bool:
    """The model's most likely next token begins the logged reply."""
    return bool(top1_text) and logged_reply.startswith(top1_text)


def text_layers(cfg) -> int:
    """Decoder layers, for a plain causal model or a vision-language one."""
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None:
        n = getattr(getattr(cfg, "text_config", None), "num_hidden_layers", None)
    if n is None:
        raise ValueError("no num_hidden_layers in the model config")
    return int(n)


def _save(path: Path, done: dict, meta: dict) -> None:
    keys = list(done)
    np.savez(path, keys=np.array(keys), states=np.stack([done[k]["state"] for k in keys]),
             n_tokens=np.array([done[k]["n_tokens"] for k in keys]),
             check_tokens=np.array([done[k]["check_tokens"] for k in keys]),
             completion_tokens=np.array([done[k].get("completion_tokens", 0) for k in keys]),
             agree=np.array([done[k].get("agree", -1.0) for k in keys]),
             logged_tokens=np.array([done[k]["logged_tokens"] for k in keys]),
             top1=np.array([done[k]["top1"] for k in keys]),
             top1_text=np.array([done[k]["top1_text"] for k in keys]),
             first_ok=np.array([done[k]["first_ok"] for k in keys]),
             seconds=np.array([done[k]["seconds"] for k in keys]),
             meta=np.array(json.dumps(meta)))


def _load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {str(k): {"state": z["states"][i], "n_tokens": int(z["n_tokens"][i]),
                     "check_tokens": int(z["check_tokens"][i]),
                     "completion_tokens": int(z["completion_tokens"][i]) if "completion_tokens" in z else 0,
                     "agree": float(z["agree"][i]) if "agree" in z else -1.0,
                     "logged_tokens": int(z["logged_tokens"][i]), "top1": int(z["top1"][i]),
                     "top1_text": str(z["top1_text"][i]), "first_ok": int(z["first_ok"][i]),
                     "seconds": float(z["seconds"][i])}
            for i, k in enumerate(z["keys"])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe", choices=sorted(PROBES))
    ap.add_argument("--revision", required=True, help="the commit in opacity_revisions.txt")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts of each task set (smoke test)")
    a = ap.parse_args(argv)

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    spec = PROBES[a.probe]
    rows = [json.loads(line) for line in open(a.inputs)]
    if a.limit:
        rows = [r for ts in dict.fromkeys(r["task_set"] for r in rows)
                for r in [x for x in rows if x["task_set"] == ts][: a.limit]]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        sys.exit(f"{out} exists; runs are append-only, choose a new name")
    partial = out.with_name(out.stem + ".partial.npz")

    tok = AutoTokenizer.from_pretrained(spec["hf"], revision=a.revision)
    cfg = AutoConfig.from_pretrained(spec["hf"], revision=a.revision)
    arch = (cfg.architectures or [""])[0]
    cls = AutoModelForImageTextToText if "ConditionalGeneration" in arch else AutoModelForCausalLM
    model = cls.from_pretrained(spec["hf"], revision=a.revision, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    n_layers = text_layers(cfg)
    depths = depth_index(n_layers)
    device = model.get_input_embeddings().weight.device
    meta = {"probe": a.probe, "hf": spec["hf"], "revision": a.revision, "class": cls.__name__, "arch": arch,
            "chat": spec["chat"], "n_layers": n_layers, "depths": depths, "argv": sys.argv,
            "torch": torch.__version__, "transformers": transformers.__version__, "host": platform.node(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    print(json.dumps(meta), flush=True)

    done = _load(partial) if partial.exists() else {}
    t_start = time.time()
    for i, r in enumerate(rows):
        if r["key"] in done:
            continue
        t0 = time.time()
        text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, **spec["chat"])
        p_ids = tok(text, add_special_tokens=False).input_ids
        src_chat = spec["source_chat"][r["task_set"]]
        check = len(p_ids) if src_chat == spec["chat"] else len(tok(
            tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, **src_chat),
            add_special_tokens=False).input_ids)
        c_ids = tok(r["completion"], add_special_tokens=False).input_ids if r.get("completion") is not None else []
        ids = torch.tensor([p_ids + c_ids], device=device)
        with torch.inference_mode():
            o = model(input_ids=ids, output_hidden_states=True, logits_to_keep=len(c_ids) + 1, use_cache=False)
        hs = o.hidden_states
        if len(hs) != n_layers + 1:
            sys.exit(f"expected {n_layers + 1} hidden states, got {len(hs)}")
        if c_ids:
            span = slice(len(p_ids), len(p_ids) + len(c_ids))
            state = torch.stack([hs[d][0, span].float().mean(0) for d in depths]).cpu().numpy().astype(np.float16)
            agree = float((o.logits[0, :-1].argmax(-1) == torch.tensor(c_ids, device=o.logits.device)).float().mean())
        else:
            state = torch.stack([hs[d][0, -1] for d in depths]).float().cpu().numpy().astype(np.float16)
            agree = -1.0
        top1 = int(o.logits[0, -1].argmax())
        top1_text = tok.decode([top1])
        first = r.get("logged_reply")
        done[r["key"]] = {"state": state, "n_tokens": len(p_ids), "check_tokens": int(check),
                          "completion_tokens": len(c_ids), "agree": agree,
                          "logged_tokens": int(r["logged_tokens"]),
                          "top1": top1, "top1_text": top1_text,
                          "first_ok": -1 if first is None else int(first_token_ok(top1_text, first)),
                          "seconds": time.time() - t0}
        if len(done) % 50 == 0:
            print(f"{len(done)}/{len(rows)}  {time.time() - t_start:.0f}s", flush=True)
        if len(done) % CHECKPOINT_EVERY == 0:
            _save(partial, done, meta)

    _save(out, done, meta)
    if partial.exists():
        partial.unlink()
    v = list(done.values())
    mismatch = sum(x["check_tokens"] != x["logged_tokens"] for x in v)
    checked = [x["first_ok"] for x in v if x["first_ok"] >= 0]
    ag = [x.get("agree", -1.0) for x in v if x.get("agree", -1.0) >= 0]
    summary = {"prompts": len(v), "length_mismatches": mismatch,
               "completions": len(ag), "mean_agree": float(np.mean(ag)) if ag else None,
               "first_token_checked": len(checked), "first_token_ok": sum(checked),
               "mean_tokens": float(np.mean([x["n_tokens"] for x in v])),
               "mean_seconds": float(np.mean([x["seconds"] for x in v])), "wall_seconds": time.time() - t_start}
    print("SUMMARY " + json.dumps(summary), flush=True)
    out.with_suffix(".json").write_text(json.dumps({**meta, **summary}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
