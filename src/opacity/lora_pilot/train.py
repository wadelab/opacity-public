"""Bounded training and fresh-process reload stages. Never runs on import."""
from __future__ import annotations

import contextlib
import os
import random
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import SCHEMA
from .common import (RECIPE, append_event, bundle_manifest, digest, predecessor, read_json, require,
                     tree_manifest, verify_tree, write_json)
from . import model as instrument


def weighted_loss(mean_loss, supervised_tokens: int, group_tokens: int):
    require(0 < supervised_tokens <= group_tokens, "invalid supervised-token weighting")
    return mean_loss * (supervised_tokens / group_tokens)


class Budget:
    def __init__(self, deadline: float, out: Path, gpu: dict):
        self.deadline, self.out, self.gpu = deadline, out, gpu
        self.error = None
        self.driver_peak = 0
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def event(self, kind: str, **values) -> None:
        with self.lock:
            append_event(self.out / "events.jsonl", {"event": kind, "time": time.time(), **values})

    def check(self) -> None:
        require(time.time() < self.deadline, "stage deadline reached")
        require(self.error is None, f"resource monitor failed: {self.error}")
        fs = os.statvfs(self.out)
        require(fs.f_bavail * fs.f_frsize > 16 * 1024 * 1024, "bounded output volume nearly full")

    # Consecutive failed driver queries tolerated before the monitor gives up. On 7 November
    # 2026 one nvidia-smi call failed while another user's job on the node's other GPU exited,
    # and the deception runner's H construction stopped after 40 minutes (job 38487862). Each
    # failure is recorded as an event; five in a row (about 30 s) still stop the stage.
    MAX_QUERY_FAILURES = 5

    def _poll(self) -> None:
        from .preflight import nvidia_query

        failures = 0
        try:
            while not self.stop.is_set():
                try:
                    used = int(nvidia_query(self.gpu["uuid"], "memory.used")) * 1024 * 1024
                except Exception as error:  # noqa: BLE001 - recorded, and fatal once repeated
                    failures += 1
                    self.event("resources_error", consecutive=failures,
                               error=f"{type(error).__name__}: {error}"[:300])
                    if failures >= self.MAX_QUERY_FAILURES:
                        self.error = f"{failures} driver queries failed in a row ({type(error).__name__})"
                        return
                    self.stop.wait(1)
                    continue
                failures = 0
                self.driver_peak = max(self.driver_peak, used)
                self.event("resources", driver_used_bytes=used,
                           host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
                self.stop.wait(1)
        except Exception as error:
            self.error = type(error).__name__

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=7)
        require(not self.thread.is_alive() and self.error is None, "resource monitor incomplete")

    def memory(self) -> dict:
        import torch

        instrument.synchronize()
        return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "device_total_bytes": self.gpu["total_bytes"],
                "driver_peak_sampled_bytes": self.driver_peak, "driver_poll_seconds": 1}

    def headroom(self) -> bool:
        memory = self.memory()
        return memory["peak_reserved_bytes"] <= RECIPE["memory_fraction"] * memory["device_total_bytes"]


def seed() -> None:
    import numpy as np
    import torch

    random.seed(RECIPE["seed"])
    np.random.seed(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    torch.cuda.manual_seed_all(RECIPE["seed"])


def one_group(model, parameters, rows: list[dict], budget: Budget) -> dict:
    import torch

    denominator = sum(row["supervised_tokens"] for row in rows)
    require(denominator > 0 and len(rows) == RECIPE["accumulation"], "invalid accumulation group")
    group_loss, total_tokens = 0.0, 0
    instrument.synchronize()
    start = time.monotonic()
    for microstep, row in enumerate(rows):
        budget.check()
        ids = torch.tensor([row["input_ids"]], device=instrument.DEVICE)
        labels = torch.tensor([row["labels"]], device=ids.device)
        require(int((labels[:, 1:] != -100).sum()) == row["supervised_tokens"], "shifted token count mismatch")
        result = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
        require(result.loss is not None and bool(torch.isfinite(result.loss)), "nonfinite/missing training loss")
        # Built-in mean over unmasked shifted labels, reweighted to a token mean across the group.
        loss = weighted_loss(result.loss, row["supervised_tokens"], denominator)
        group_loss += float(loss.detach())
        loss.backward()
        total_tokens += ids.numel()
        budget.event("microstep", id=row["id"], within_update=microstep + 1,
                     supervised_tokens=row["supervised_tokens"], nonpadding_tokens=ids.numel())
        del loss, result, ids, labels
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),
            "absent/nonfinite adapter gradient")
    require(any(bool(torch.count_nonzero(p.grad)) for p in parameters), "all adapter gradients are zero")
    instrument.synchronize()
    return {"loss": group_loss, "supervised_tokens": denominator, "nonpadding_tokens": total_tokens,
            "forward_backward_seconds": time.monotonic() - start}


def train_updates(model, parameters, base, rows: list[dict], order: list[str], updates: int,
                  budget: Budget):
    import torch

    by_id = {r["id"]: r for r in rows if r["split"] == "train"}
    require(len(order) >= updates * RECIPE["accumulation"], "sampler order too short")
    optimiser = torch.optim.AdamW(parameters, lr=RECIPE["learning_rate"], betas=tuple(RECIPE["betas"]),
                                 eps=RECIPE["epsilon"], weight_decay=RECIPE["weight_decay"])
    require({id(p) for g in optimiser.param_groups for p in g["params"]} == {id(p) for p in parameters},
            "optimiser contains extra parameters")
    model.train()
    measurements = []
    for update in range(updates):
        budget.check()
        optimiser.zero_grad(set_to_none=True)
        selected = [by_id[key] for key in order[4 * update:4 * update + 4]]
        measurement = one_group(model, parameters, selected, budget)
        require(all(not p.requires_grad and p.grad is None for _, p in base), "base acquired gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters, RECIPE["clip_norm"], error_if_nonfinite=True)
        instrument.synchronize()
        t0 = time.monotonic()
        optimiser.step()
        instrument.synchronize()
        require(all(bool(torch.isfinite(p).all()) for p in parameters), "nonfinite updated adapter")
        require(set(optimiser.state) <= set(parameters), "unexpected optimiser state")
        measurement.update(update=update + 1, microsteps=4, gradient_norm=float(norm),
                           optimiser_seconds=time.monotonic() - t0, **budget.memory())
        measurements.append(measurement)
        budget.event("update", **measurement)
        require(budget.headroom(), "512-token training memory headroom inadequate")
        optimiser.zero_grad(set_to_none=True)
    # Retain AdamW state during later stress measurements. Freeing it first
    # would understate the memory needed at the claimed training context.
    return measurements, optimiser


def profile(model, parameters, rows: list[dict], budget: Budget, lengths: list[int]) -> list[dict]:
    outcomes = []
    by_length = {r["length"]: r for r in rows}
    for length in lengths:
        budget.check()
        require(length in by_length, "stress case absent from bundle")
        row = by_length[length]
        require(len(row["input_ids"]) == length, "stress length includes padding or is incorrect")
        model.train()
        model.zero_grad(set_to_none=True)
        before = instrument.contribution_digest(model)
        result = one_group(model, parameters, [row] * 4, budget)
        model.zero_grad(set_to_none=True)
        require(instrument.contribution_digest(model) == before, "profiling changed adapter weights")
        result.update(length=length, optimiser_updates=0, **budget.memory(), headroom=budget.headroom())
        outcomes.append(result)
        budget.event("profile", **result)
        if not result["headroom"]:
            require(length != 512, "512-token profiling memory headroom inadequate")
            break
    return outcomes


def expected_adapter_shapes(inventory: list[dict], rank: int = 8) -> dict:
    expected = {}
    for row in inventory:
        out_dim, in_dim = row["shape"]
        expected[row["name"] + ".lora_A.weight"] = [rank, in_dim]
        expected[row["name"] + ".lora_B.weight"] = [out_dim, rank]
    return expected


def validate_adapter_shapes(shapes: dict, inventory: list[dict], rank: int = 8) -> None:
    expected = expected_adapter_shapes(inventory, rank)
    seen = set()
    for key, shape in shapes.items():
        names = [name for name in expected if key == name or key.endswith("." + name)]
        require(len(names) == 1 and names[0] not in seen, "unexpected/duplicate saved adapter tensor")
        require(list(shape) == expected[names[0]], "saved adapter shape mismatch")
        seen.add(names[0])
    require(seen == set(expected), "saved adapter tensors incomplete")


def saved_adapter(path: Path, inventory: list[dict], *, rank: int = 8, alpha: int = 16) -> None:
    from safetensors import safe_open

    require({p.name for p in path.iterdir()} == {"adapter_config.json", "adapter_model.safetensors"},
            "adapter save contains unregistered files")
    cfg = read_json(path / "adapter_config.json")
    require(cfg.get("r") == rank and cfg.get("lora_alpha") == alpha and cfg.get("bias") == "none"
            and not cfg.get("modules_to_save") and cfg.get("peft_type") == "LORA", "invalid saved adapter config")
    with safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu") as weights:
        validate_adapter_shapes({key: weights.get_slice(key).get_shape() for key in weights.keys()}, inventory, rank)


def save_adapter(model, adapter: Path, inventory: list[dict], *, rank: int = 8, alpha: int = 16) -> None:
    """Safe save of the adapter tensors and config only: no README/auto-card or embedding export."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    adapter.mkdir()
    state = {key: tensor.detach().cpu().contiguous()
             for key, tensor in get_peft_model_state_dict(model, save_embedding_layers=False).items()}
    validate_adapter_shapes({key: tensor.shape for key, tensor in state.items()}, inventory, rank)
    save_file(state, adapter / "adapter_model.safetensors", metadata={"format": "pt"})
    model.peft_config["default"].save_pretrained(adapter)
    saved_adapter(adapter, inventory, rank=rank, alpha=alpha)


def fitting(stage: str, bundle: Path, out: Path, previous: Path | None, budget: Budget) -> dict:
    import torch

    rows = read_json(bundle / "data.json")
    diagnostic_rows = [r for r in rows if r["split"] == "diagnostic"]
    inventory = read_json(bundle / "targets.json")
    torch.cuda.reset_peak_memory_stats()
    seed()
    t0 = time.monotonic()
    model, tok, base = instrument.load(Path("/model"))
    instrument.synchronize()
    budget.event("loaded", seconds=time.monotonic() - t0,
                 attention_implementation=str(getattr(model.config.text_config, "_attn_implementation", None)))
    baseline = instrument.diagnostics(model, tok, diagnostic_rows, out / "baseline", budget.check)
    instrument.diagnostics(model, tok, diagnostic_rows, out / "repeat", budget.check)
    repeat = instrument.compare(out / "baseline", out / "repeat")
    tolerance = instrument.tolerance_from_repeat(repeat)
    if previous is not None:
        tolerance = read_json(previous / "checks.json")["tolerance"]
        instrument.compare(previous / "baseline", out / "baseline", tolerance)
    # Write before the first update; later failures cannot change the comparison policy.
    write_json(out / "tolerance.json", {"tolerance": tolerance, "clean_repeat": repeat})
    t0 = time.monotonic()
    original_digest = instrument.base_digest(base)
    budget.event("base_hash_before", seconds=time.monotonic() - t0)
    model, parameters, details = instrument.attach(model, inventory)
    instrument.diagnostics(model, tok, diagnostic_rows, out / "initial", budget.check)
    instrument.compare(out / "baseline", out / "initial", tolerance)
    write_json(out / "adapter-inventory.json", {"tensors": details,
                                               "trainable_parameters": sum(p.numel() for p in parameters)})
    instrument.checkpointing(model)
    initial_contribution = instrument.contribution_digest(model)
    updates = 2 if stage == "smoke" else 20
    steps, optimiser = train_updates(model, parameters, base, rows, read_json(bundle / "order.json"), updates, budget)
    require(instrument.contribution_digest(model) != initial_contribution, "adapter contribution did not learn")
    t0 = time.monotonic()
    require(instrument.base_digest(base) == original_digest, "in-memory base weights changed")
    budget.event("base_hash_after", seconds=time.monotonic() - t0)
    model.gradient_checkpointing_disable()
    adapted = instrument.diagnostics(model, tok, diagnostic_rows, out / "adapted", budget.check)
    regressions = instrument.new_errors(baseline, adapted)
    if stage == "train":
        t0 = time.monotonic()
        save_adapter(model, out / "adapter", inventory)
        budget.event("adapter_saved", seconds=time.monotonic() - t0)
    instrument.checkpointing(model)
    lengths = [512] if stage == "smoke" else RECIPE["profile_lengths"]
    profiles = profile(model, parameters, read_json(bundle / "stress.json"), budget, lengths)
    del optimiser
    steady = steps[1:]
    seconds = sum(s["forward_backward_seconds"] + s["optimiser_seconds"] for s in steady)
    throughput = {"steady_updates": len(steady), "seconds": seconds,
                  "nonpadding_tokens_per_second": sum(s["nonpadding_tokens"] for s in steady) / seconds,
                  "supervised_tokens_per_second": sum(s["supervised_tokens"] for s in steady) / seconds}
    checks = {"tolerance": tolerance, "clean_repeat": repeat, "base_digest": original_digest,
              "base_unchanged": True, "adapter_contribution_changed": True,
              "new_diagnostic_errors": regressions, "updates": updates, "presentations": updates * 4,
              "diagnostic_denominator": 16, "base_diagnostic_correct": baseline["correct"],
              "adapted_diagnostic_correct": adapted["correct"], "throughput": throughput,
              "steps": steps, "profiles": profiles, "memory": budget.memory()}
    write_json(out / "checks.json", checks)
    require(not regressions, "newly incorrect diagnostic answers; see checks.json")
    require(budget.headroom() or (profiles[-1]["length"] > 512 and not profiles[-1]["headroom"]),
            "required 512-token memory gate failed")
    return checks


def memory_profile(model, base, inventory: list[dict], rows: list[dict], order: list[str],
                   stress: list[dict], budget: Budget) -> dict:
    """Two honest updates, so AdamW state is resident as in training, then one stress cycle
    per length, stopping at the first without headroom. A measurement: nothing is saved."""
    model, parameters, _ = instrument.attach(model, inventory)
    instrument.checkpointing(model)
    steps, optimiser = train_updates(model, parameters, base, rows, order, 2, budget)
    profiles = profile(model, parameters, stress, budget, [row["length"] for row in stress])
    del optimiser
    return {"stage": "profile", "lengths_requested": [row["length"] for row in stress],
            "steps": steps, "profiles": profiles}


def study_profile(bundle: Path, out: Path, budget: Budget) -> dict:
    import torch

    torch.cuda.reset_peak_memory_stats()
    seed()
    t0 = time.monotonic()
    model, _, base = instrument.load(Path("/model"))
    instrument.synchronize()
    budget.event("loaded", seconds=time.monotonic() - t0)
    checks = memory_profile(model, base, read_json(bundle / "targets.json"), read_json(bundle / "data.json"),
                            read_json(bundle / "order.json"), read_json(bundle / "stress-study.json"), budget)
    checks["memory"] = budget.memory()
    write_json(out / "checks.json", checks)
    return checks


def reload_worker(kind: str, bundle: Path, out: Path, previous: Path, deadline: float, gpu: dict) -> None:
    import torch
    from peft import PeftModel

    out.mkdir(exist_ok=False)
    from .preflight import runtime_preflight
    runtime_preflight(bundle, out, read_json(Path("/run/context.json")))
    budget = Budget(deadline, out, gpu)
    budget.start()
    try:
        torch.cuda.reset_peak_memory_stats()
        seed()
        model, tok, base = instrument.load(Path("/model"))
        rows = [r for r in read_json(bundle / "data.json") if r["split"] == "diagnostic"]
        tolerance = read_json(previous / "checks.json")["tolerance"]
        base_hash = instrument.base_digest(base)
        if kind == "clean":
            instrument.diagnostics(model, tok, rows, out / "baseline", budget.check)
            instrument.compare(previous / "baseline", out / "baseline", tolerance)
        else:
            inventory = read_json(bundle / "targets.json")
            saved_adapter(previous / "adapter", inventory)
            model = PeftModel.from_pretrained(model, previous / "adapter", is_trainable=False,
                                              autocast_adapter_dtype=True, local_files_only=True)
            instrument.adapter_inventory(model, inventory, trainable=False)
            instrument.diagnostics(model, tok, rows, out / "adapted", budget.check)
            instrument.compare(previous / "adapted", out / "adapted", tolerance)
            with model.disable_adapter():
                instrument.diagnostics(model, tok, rows, out / "disabled", budget.check)
            instrument.compare(previous / "baseline", out / "disabled", tolerance)
        require(instrument.base_digest(base) == base_hash, "reload/disable modified base tensors")
        require(budget.headroom(), "reload memory gate failed")
        write_json(out / "checks.json", {"kind": kind, "passed": True, "base_digest": base_hash,
                                         "memory": budget.memory()})
    finally:
        budget.close()


def reload_stage(bundle: Path, out: Path, previous: Path, budget: Budget) -> dict:
    for kind in ("clean", "adapter"):
        budget.check()
        child = subprocess.Popen([sys.executable, "-m", "opacity.lora_pilot", "reload-worker",
                                  "--kind", kind, "--out", str(out / kind)])
        try:
            code = child.wait(timeout=max(1, budget.deadline - time.time()))
            require(code == 0, f"{kind} fresh-process reload check failed")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    result = {kind: read_json(out / kind / "checks.json") for kind in ("clean", "adapter")}
    write_json(out / "checks.json", result)
    return result


def execute(stage: str, bundle: Path, out: Path) -> None:
    from .preflight import runtime_preflight

    os.umask(0o077)
    context = read_json(out / "context.json")
    require(stage == context["stage"], "stage differs from launcher request")
    # Only the off-budget rehearsal may run on another GPU, and it loads no model.
    require(stage == "preflight" or context["gpu_model"] == "H100", "training stages require an H100")
    budget = None
    try:
        report = runtime_preflight(bundle, out, context)
        manifest = bundle_manifest(bundle, verify=False)
        previous = Path("/previous") if context["predecessor"] else None
        if stage in ("train", "reload"):
            require(previous is not None, "predecessor missing")
            predecessor(previous, manifest["identity"], "smoke" if stage == "train" else "train")
            require(digest(previous / "result.json") == context["predecessor_result_sha256"],
                    "predecessor result changed since submission")
        else:
            require(previous is None, f"{stage} must start without a predecessor")
        budget = Budget(context["deadline"], out, report["hardware"])
        budget.start()
        if stage == "reload":
            reload_stage(bundle, out, previous, budget)
        elif stage == "preflight":
            time.sleep(3)  # let the resource monitor sample the driver a few times
            write_json(out / "checks.json", {"rehearsal": True, "memory": budget.memory()})
        elif stage == "profile":
            study_profile(bundle, out, budget)
        else:
            fitting(stage, bundle, out, previous, budget)
        budget.check()
        verify_tree(bundle, manifest["files"], exclude=("manifest.json",))
        if previous is not None:
            predecessor(previous, manifest["identity"], "smoke" if stage == "train" else "train")
            require(digest(previous / "result.json") == context["predecessor_result_sha256"],
                    "predecessor result changed during execution")
        budget.event("completed", stage=stage)
        budget.close()
        budget = None
        artifacts = {}
        for name in ("baseline", "repeat", "initial", "adapted", "adapter", "clean"):
            directory = out / name
            if directory.is_dir():
                artifacts.update({str(Path(name) / k): v for k, v in tree_manifest(directory).items()})
        # reload's adapter directory contains worker evidence; train's contains only weights/config.
        for name in ("events.jsonl", "checks.json", "tolerance.json", "preflight.json", "adapter-inventory.json"):
            if (out / name).exists():
                artifacts[name] = digest(out / name)
        write_json(out / "result.json", {"schema": SCHEMA, "stage": stage, "status": "passed",
                                        "bundle_id": manifest["identity"], "artifacts": artifacts,
                                        "predecessor_result_sha256": context["predecessor_result_sha256"]})
    except BaseException as error:
        if budget is not None:
            with contextlib.suppress(Exception):
                budget.close()
        if not (out / "result.json").exists():
            write_json(out / "result.json", {"schema": SCHEMA, "stage": stage, "status": "failed",
                                            "error_type": type(error).__name__, "reason": str(error)})
        raise
