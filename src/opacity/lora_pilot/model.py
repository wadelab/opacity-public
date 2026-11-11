"""Exact Qwen text-path attachment and bounded numerical diagnostics.

Heavy imports are intentionally local: prepare/report/config inspection do not
initialise CUDA simply by importing the command dispatcher.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

from .common import RECIPE, read_json, require, write_json
from .data import score

# The characterised Qwen3.5-27B text decoder. Tests substitute a tiny model and the CPU.
LAYERS, HIDDEN = 64, 5120
DEVICE = "cuda:0"


def synchronize() -> None:
    import torch

    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


SCOPES = ("qv", "all-linear")
# Every linear projection of the text decoder. Full-attention blocks carry self_attn q/k/v/o, the
# linear-attention blocks linear_attn in_proj_qkv/z/a/b and out_proj; every block an MLP.
ALL_LINEAR = {"full_attention": {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"},
              "linear_attention": {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a",
                                   "linear_attn.in_proj_b", "linear_attn.out_proj"}}
MLP = {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}


def targets(model, scope: str = "qv") -> list[dict]:
    """The adapter's module inventory. "qv": the pilot's query and value projections of the
    full-attention blocks. "all-linear": every nn.Linear of the text decoder blocks, checked
    against the expected set per block kind, so nothing in the vision tower or the head."""
    import torch

    require(scope in SCOPES, f"unknown adapter scope {scope!r}")
    cfg = model.config.text_config
    require(cfg.num_hidden_layers == LAYERS and cfg.hidden_size == HIDDEN, "unexpected text architecture")
    kinds = list(cfg.layer_types)
    expected = {i for i, kind in enumerate(kinds) if kind == "full_attention"}
    require(expected == set(range(3, LAYERS, 4)), "unexpected hybrid layer layout")
    inventory = []
    if scope == "qv":
        for name, module in model.named_modules():
            match = re.search(r"(?:^|\.)language_model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)$", name)
            if match and int(match[1]) in expected:
                require(isinstance(module, torch.nn.Linear), f"nonlinear adapter target: {name}")
                inventory.append({"name": name, "shape": list(module.weight.shape), "layer": int(match[1])})
        require(len(inventory) == 2 * len(expected), "q/v module inventory incomplete")
        require({x["layer"] for x in inventory} == expected, "full-attention target coverage incomplete")
        return sorted(inventory, key=lambda row: row["name"])
    found: dict[int, set] = {}
    for name, module in model.named_modules():
        match = re.search(r"(?:^|\.)language_model\.layers\.(\d+)\.(.+)$", name)
        if match and isinstance(module, torch.nn.Linear):
            layer, suffix = int(match[1]), match[2]
            found.setdefault(layer, set()).add(suffix)
            inventory.append({"name": name, "shape": list(module.weight.shape), "layer": layer})
    require(set(found) == set(range(LAYERS)), "all-linear coverage misses a decoder block")
    for layer, suffixes in found.items():
        require(suffixes == ALL_LINEAR[kinds[layer]] | MLP,
                f"block {layer} ({kinds[layer]}) projections differ from the registered set: {sorted(suffixes)}")
    return sorted(inventory, key=lambda row: row["name"])


def meta_targets(snapshot: Path, scope: str = "qv") -> list[dict]:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText

    cfg = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    require(cfg.architectures == ["Qwen3_5ForConditionalGeneration"], "unexpected model architecture")
    with init_empty_weights(include_buffers=True):
        model = AutoModelForImageTextToText.from_config(cfg, trust_remote_code=False)
    return targets(model, scope)


def load(snapshot: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model, base = load_model(snapshot)
    return model, tok, base


def load_model(snapshot: Path):
    """Every base copy (fit, clean reload, adapter reload) comes through this one call:
    from_pretrained keeps some modules and buffers in fp32, which a cast would not."""
    import torch
    from transformers import AutoModelForImageTextToText

    # One explicit device; no auto-dispatch, quantisation, offload or new kernels.
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map={"": DEVICE},
    )
    require(model.__class__.__name__ == "Qwen3_5ForConditionalGeneration", "wrong loaded architecture")
    model.config.use_cache = False
    model.requires_grad_(False)
    return model, list(model.named_parameters())


def attach(model, inventory: list[dict], *, rank: int | None = None, alpha: int | None = None,
           dropout: float | None = None, scope: str = "qv"):
    """Rank, alpha, dropout and scope default to the pilot recipe; the deception runner passes its own."""
    import torch
    from peft import LoraConfig, get_peft_model

    require(targets(model, scope) == inventory, "loaded module inventory differs from weightless inventory")
    expression = "^(?:" + "|".join(re.escape(row["name"]) for row in inventory) + ")$"
    model = get_peft_model(model, LoraConfig(
        r=RECIPE["rank"] if rank is None else rank,
        lora_alpha=RECIPE["alpha"] if alpha is None else alpha,
        lora_dropout=RECIPE["dropout"] if dropout is None else dropout,
        target_modules=expression, bias="none", init_lora_weights=True,
    ), autocast_adapter_dtype=True)
    parameters, modules = adapter_inventory(model, inventory)
    for parameter in parameters:
        parameter.data = parameter.data.to(torch.float32)
    return model, parameters, modules


def adapter_inventory(model, expected: list[dict], *, trainable: bool = True):
    import torch

    names = {row["name"] for row in expected}
    seen, parameters, details = set(), [], []
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue
        matched = [n for n in names if name == n or name.endswith("." + n)]
        require(len(matched) == 1 and matched[0] not in seen, "extra/ambiguous adapter location")
        seen.add(matched[0])
        require(set(module.lora_A) == {"default"} and set(module.lora_B) == {"default"}, "extra adapter")
        for factor in (module.lora_A["default"].weight, module.lora_B["default"].weight):
            require(factor.dtype == torch.float32, "adapter is not fp32")
            parameters.append(factor)
        details.append({"name": name, "A": list(module.lora_A["default"].weight.shape),
                        "B": list(module.lora_B["default"].weight.shape), "dtype": "float32"})
    require(seen == names, "adapter installation differs from allowlist")
    expected_ids = {id(p) for p in parameters}
    actual_ids = {id(p) for p in model.parameters() if p.requires_grad}
    require(actual_ids == (expected_ids if trainable else set()), "extra/missing trainable tensors")
    return parameters, sorted(details, key=lambda row: row["name"])


def checkpointing(model) -> None:
    # Non-reentrant checkpointing retains the adapter graph with frozen inputs;
    # the actual backward smoke verifies this rather than trusting the setting.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    require(model.is_gradient_checkpointing, "gradient checkpointing was not enabled")


def base_digest(base) -> str:
    import torch

    h = hashlib.sha256()
    for name, parameter in base:
        require(not parameter.requires_grad and parameter.grad is None, "base tensor became trainable")
        h.update(f"{name}:{tuple(parameter.shape)}:{parameter.dtype}".encode())
        flat = parameter.detach().reshape(-1)
        for start in range(0, flat.numel(), 524288):
            raw = flat[start:start + 524288].to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
            h.update(raw)
    return h.hexdigest()


def contribution_digest(model) -> str:
    """Hash effective low-rank products, not just one factor's movement."""
    import torch

    h = hashlib.sha256()
    with torch.no_grad():
        for name, module in model.named_modules():
            if not hasattr(module, "lora_A") or "default" not in module.lora_A:
                continue
            a, b = module.lora_A["default"].weight, module.lora_B["default"].weight
            h.update(name.encode())
            # Row chunks avoid materialising a full dense delta for a projection.
            for start in range(0, b.shape[0], 64):
                delta = (b[start:start + 64].float() @ a.float()) * module.scaling["default"]
                require(bool(torch.isfinite(delta).all()), "nonfinite effective adapter contribution")
                h.update(delta.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def diagnostics(model, tok, rows: list[dict], out: Path, check_budget) -> dict:
    import numpy as np
    import torch

    out.mkdir(exist_ok=False, parents=True)
    model.eval()
    results = []
    t0 = time.monotonic()
    with torch.inference_mode():
        for row in rows:
            check_budget()
            ids = torch.tensor([row["input_ids"]], device=DEVICE)
            # Predict the first answer token and final assistant terminator; one
            # forward keeps only these two positions' logits.
            positions = sorted({row["answer_start"] - 1, row["answer_end"] - 2})
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                           logits_to_keep=torch.tensor(positions, device=ids.device))
            logits = result.logits[0].float()
            require(tuple(logits.shape[:1]) == (len(positions),) and bool(torch.isfinite(logits).all()),
                    "missing or nonfinite diagnostic logits")
            logit_file = out / f"{row['id']}.npy"
            with logit_file.open("xb") as stream:
                np.save(stream, logits.cpu().numpy(), allow_pickle=False)
            del result, logits
            prompt = torch.tensor([row["prompt_ids"]], device=DEVICE)
            end = tok.convert_tokens_to_ids("<|im_end|>")
            # The KV cache bounds generation time; every compared condition uses it.
            tokens = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                    do_sample=False, num_beams=1, max_new_tokens=RECIPE["max_new_tokens"],
                                    use_cache=True, eos_token_id=end,
                                    pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else end)
            answer_ids = tokens[0, prompt.shape[1]:].tolist()
            terminated = bool(answer_ids) and answer_ids[-1] == end
            raw_text = tok.decode(answer_ids, skip_special_tokens=False)
            # Remove only the expected terminal marker, not arbitrary special
            # tokens that could otherwise disappear and make invalid JSON pass.
            text = tok.decode(answer_ids[:-1] if terminated else answer_ids, skip_special_tokens=False)
            outcome = score(text, row)
            results.append({"id": row["id"], "positions": positions, "logits": logit_file.name,
                            "tokens": answer_ids, "text": text, "raw_text": raw_text, "terminated": terminated,
                            **outcome, "correct": outcome["correct"] and terminated})
    synchronize()
    summary = {"denominator": 16, "unit": "diagnostic examples", "rows": results,
               "correct": sum(x["correct"] for x in results), "seconds": time.monotonic() - t0}
    require(len(results) == 16, "diagnostic denominator changed")
    write_json(out / "index.json", summary)
    return summary


def compare(left: Path, right: Path, tolerance: dict | None = None) -> dict:
    import numpy as np

    a, b = read_json(left / "index.json"), read_json(right / "index.json")
    require([r["id"] for r in a["rows"]] == [r["id"] for r in b["rows"]], "diagnostic alignment mismatch")
    errors, tokens_equal = [], True
    for x, y in zip(a["rows"], b["rows"]):
        require(x["positions"] == y["positions"], "diagnostic positions differ")
        u = np.load(left / x["logits"], allow_pickle=False).astype(np.float32)
        v = np.load(right / y["logits"], allow_pickle=False).astype(np.float32)
        require(u.shape == v.shape and np.isfinite(u).all() and np.isfinite(v).all(), "invalid diagnostic arrays")
        absolute = float(np.max(np.abs(u - v)))
        relative = float(np.linalg.norm(u - v) / max(float(np.linalg.norm(u)), 1e-12))
        errors.append({"id": x["id"], "absolute": absolute, "relative": relative})
        tokens_equal &= x["tokens"] == y["tokens"]
    result = {"absolute": max(r["absolute"] for r in errors),
              "relative": max(r["relative"] for r in errors), "tokens_equal": tokens_equal, "rows": errors}
    if tolerance is not None:
        require(tokens_equal and result["absolute"] <= tolerance["absolute"]
                and result["relative"] <= tolerance["relative"], "logit/greedy comparison failed")
    return result


def tolerance_from_repeat(error: dict) -> dict:
    require(error["tokens_equal"] and error["absolute"] <= RECIPE["logit_abs_floor"]
            and error["relative"] <= RECIPE["logit_rel_floor"], "clean baseline numerically unstable")
    return {"absolute": max(RECIPE["logit_abs_floor"], 10 * error["absolute"]),
            "relative": max(RECIPE["logit_rel_floor"], 10 * error["relative"])}


def new_errors(before: dict, after: dict) -> list[str]:
    require([r["id"] for r in before["rows"]] == [r["id"] for r in after["rows"]], "case alignment mismatch")
    return [a["id"] for a, b in zip(before["rows"], after["rows"]) if a["correct"] and not b["correct"]]
