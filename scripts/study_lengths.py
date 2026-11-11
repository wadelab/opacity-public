"""Token lengths of study-like training examples for the contained LoRA study.

The study's training examples are analyst reports on synthetic task packs under the design-first rules
prompt, thinking off. Its new task families do not exist yet, so this measures
the closest real thing: the Qwen3.5-27B's own reports on the existing harvest
packs under that configuration (and, separately, the 1k-think and style runs).

Two steps, because the environments differ:

    # main project .venv (inspect_ai): exchanges as sent, plus vLLM's own counts
    .venv/bin/python scripts/study_lengths.py dump OUT.jsonl LOGDIR [LOGDIR ...]
    # the honest-LoRA pilot's locked env (transformers): tokenise like training
    python scripts/study_lengths.py measure OUT.jsonl SNAPSHOT SUMMARY.json

`measure` renders each exchange with the cached chat template (thinking off)
exactly as opacity.lora_pilot.data.encode does for training: the prompt with
the generation prefix, then the full exchange with the reply as the assistant
turn, supervised through <|im_end|>. It checks its prompt count against the
prompt tokens vLLM reported for the same request.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def dump(out: Path, logdirs: list[Path]) -> None:
    from inspect_ai.log import read_eval_log

    with out.open("x") as stream:
        for logdir in logdirs:
            for path in sorted(logdir.glob("*.eval")):
                log = read_eval_log(str(path))
                args = log.eval.task_args or {}
                for s in log.samples or []:
                    message = s.output.choices[0].message if s.output and s.output.choices else None
                    content = message.content if message else ""
                    text, reasoning = (content, "") if isinstance(content, str) else (
                        "".join(getattr(c, "text", "") for c in content if getattr(c, "type", "") == "text"),
                        "".join(getattr(c, "reasoning", "") for c in content if getattr(c, "type", "") == "reasoning"))
                    usage = s.output.usage if s.output else None
                    meta = s.metadata or {}
                    stream.write(json.dumps({
                        "log": path.name, "dir": logdir.name, "task": log.eval.task, "id": s.id, "epoch": s.epoch,
                        "task_set": args.get("task_set"), "thinking": args.get("thinking"),
                        "think_budget": args.get("think_budget"), "analyst_prompt": args.get("analyst_prompt"),
                        "sampling": args.get("sampling"), "style": args.get("style", meta.get("style", "")),
                        "task_id": meta.get("task_id"), "condition": meta.get("condition"),
                        "messages": [{"role": m.role, "content": m.text} for m in s.messages if m.role in ("system", "user")],
                        "reply": text, "reasoning": reasoning,
                        "vllm_input_tokens": usage.input_tokens if usage else None,
                        "vllm_output_tokens": usage.output_tokens if usage else None,
                        "vllm_reasoning_tokens": usage.reasoning_tokens if usage else None}) + "\n")
                print(path.name, args.get("task_set"), log.eval.task, len(log.samples or []))


def percentiles(values: list[int]) -> dict:
    import numpy as np

    a = np.asarray(values)
    return {"n": int(a.size), "min": int(a.min()), "p10": int(np.percentile(a, 10)), "median": int(np.median(a)),
            "mean": round(float(a.mean()), 1), "p90": int(np.percentile(a, 90)), "p99": int(np.percentile(a, 99)),
            "max": int(a.max()), "over_2048": int((a > 2048).sum()), "over_3072": int((a > 3072).sum()),
            "over_4096": int((a > 4096).sum())}


def measure(exchanges: Path, snapshot: Path, out: Path) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    rows, mismatched = [], 0
    for line in exchanges.read_text().splitlines():
        r = json.loads(line)
        if not r["reply"]:
            continue
        options = {"tokenize": False, "enable_thinking": False}
        prefix = tok.apply_chat_template(r["messages"], add_generation_prompt=True, **options)
        full = tok.apply_chat_template(r["messages"] + [{"role": "assistant", "content": r["reply"]}],
                                       add_generation_prompt=False, **options)
        pids = tok(prefix, add_special_tokens=False).input_ids
        ids = tok(full, add_special_tokens=False).input_ids
        assert ids[:len(pids)] == pids, f"assistant boundary not a token prefix: {r['log']} {r['id']}"
        supervised = ids.index(end, len(pids)) + 1 - len(pids)
        # thinking-on runs were served with a different generation prefix; compare only thinking-off prompts
        if not r["thinking"] and r["vllm_input_tokens"] is not None and r["vllm_input_tokens"] != len(pids):
            mismatched += 1
        rows.append({"dir": r["dir"], "task": r["task"], "task_set": r["task_set"], "id": r["id"],
                     "style": r["style"] or "default", "thinking": bool(r["thinking"]), "prompt": len(pids), "supervised": supervised,
                     "total": len(pids) + supervised, "vllm_input": r["vllm_input_tokens"],
                     "vllm_output": r["vllm_output_tokens"], "vllm_reasoning": r["vllm_reasoning_tokens"]})
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = f"{row['dir']}|{row['task']}|{row['task_set']}|{row['style']}|think={int(row['thinking'])}"
        groups.setdefault(key, []).append(row)
    summary = {"exchanges": len(rows), "prompt_count_mismatches_vs_vllm_thinking_off": mismatched,
               "groups": {key: {"prompt": percentiles([r["prompt"] for r in g]),
                                "supervised": percentiles([r["supervised"] for r in g]),
                                "total": percentiles([r["total"] for r in g])} for key, g in sorted(groups.items())},
               "rows": rows}
    with out.open("x") as stream:
        json.dump(summary, stream, indent=1)
    for key, g in summary["groups"].items():
        t = g["total"]
        print(f"{key:66s} n={t['n']:4d} total median {t['median']:5d} p90 {t['p90']:5d} max {t['max']:5d} "
              f">2048 {t['over_2048']:4d} >3072 {t['over_3072']:3d} >4096 {t['over_4096']:3d}"
              f" | prompt median {g['prompt']['median']} reply median {g['supervised']['median']}")
    print("thinking-off prompts whose count differs from vLLM's:", mismatched)


def main() -> int:
    if sys.argv[1] == "dump":
        dump(Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]])
    elif sys.argv[1] == "measure":
        measure(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
