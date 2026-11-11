"""CPU acceptance tests. No downloads, CUDA, Slurm or real isolation needed.

First run 27 October 2026 (d2-1, the runtime lock's environment). check-cpu
requires zero skips before preparation.
"""
from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from opacity.lora_pilot import SCHEMA
from opacity.lora_pilot import common as C, data as D, model as M
from opacity.lora_pilot import report as R, submit as S, train as T


def test_exact_oracle_and_rounding():
    assert D.rounded(Fraction(1, 8)) == "0.13"
    assert D.oracle("rate", [1, 8]) == {"value": "12.50", "denominator": "8", "conclusion": "percent"}
    assert D.oracle("difference", [3, 8])["value"] == "-5"
    assert D.oracle("association", [3, 8])["conclusion"] == "association only"


def test_generator_has_disjoint_families_and_numerical_twins():
    rows = D.generate(271028)
    assert rows == D.generate(271028)
    assert len(rows) == 80
    assert {r["operation"] for r in rows if r["split"] == "diagnostic"} == set(D.OPERATIONS)
    assert len({r["family"] for r in rows if r["split"] == "train"}) == 16
    assert len({r["family"] for r in rows if r["split"] == "diagnostic"}) == 4


@pytest.mark.parametrize("corruption", ["family", "numbers", "target", "answer", "duplicate", "rendered"])
def test_generator_rejects_leakage_or_bad_ground_truth(corruption):
    rows = D.generate(271028)
    if corruption == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif corruption == "family":
        rows[-1]["family"] = rows[0]["family"]
    elif corruption == "numbers":
        rows[-1]["numbers"] = rows[0]["numbers"]
    elif corruption == "target":
        rows[-1]["target"]["value"] = "999"
    elif corruption == "rendered":
        rows[-1]["messages"][-1]["content"] = rows[-1]["messages"][-1]["content"].replace("A | ", "A | 999")
    else:
        rows[-1]["answer"] = "{}"
    with pytest.raises(C.Refusal):
        D.validate(rows)


def test_diagnostic_parser_is_strict_and_preserves_wrong_answer():
    row = D.generate(271028)[0]
    assert D.score(row["answer"], row) == {"parse_ok": True, "correct": True}
    assert not D.score('```json\n' + row["answer"] + '\n```', row)["parse_ok"]
    assert not D.score(row["answer"] + " trailing prose", row)["correct"]
    changed = {**row["target"], "denominator": "wrong"}
    assert not D.score(json.dumps(changed), row)["correct"]


def test_sampler_covers_one_full_epoch_before_repeating():
    rows = D.generate(271028)
    order = D.sampler(rows, 271028)
    assert len(order) == 80 and len(set(order[:64])) == 64
    assert len(set(order[64:])) == 16
    assert order == D.sampler(rows, 271028)


class Tokenizer:
    unk_token_id = -1

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        text = "".join("<" + r["role"] + ">" + r["content"]
                       + ("<|im_end|>\n" if r["role"] == "assistant" else "") for r in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        tokens = []
        while text:
            if text.startswith("<|im_end|>"):
                tokens.append(900)
                text = text[len("<|im_end|>"):]
            elif text.startswith(" x"):
                tokens.append(901)
                text = text[2:]
            else:
                tokens.append(ord(text[0]))
                text = text[1:]
        return SimpleNamespace(input_ids=tokens)

    def convert_tokens_to_ids(self, token):
        return 900 if token == "<|im_end|>" else self.unk_token_id

    pad_token_id = None

    def decode(self, ids, **kwargs):
        return "".join({900: "<|im_end|>", 901: " x"}.get(i, chr(i)) for i in ids)


def tiny_row():
    row = D.generate(271028)[0]
    return {**row, "messages": [{"role": "system", "content": "Be honest."},
                                {"role": "user", "content": "Give the total."}]}


def test_assistant_mask_includes_first_answer_and_terminator_not_prompt_or_trailing_whitespace():
    encoded = D.encode(tiny_row(), Tokenizer(), 2048)
    start, end = encoded["answer_start"], encoded["answer_end"]
    assert set(encoded["labels"][:start]) == {-100}
    assert encoded["labels"][start] == ord("{")
    assert encoded["labels"][end - 1] == 900
    assert encoded["labels"][end:] == [-100]
    assert encoded["supervised_tokens"] == end - start


def test_context_cap_rejects_rather_than_truncates():
    with pytest.raises(C.Refusal, match="overlength"):
        D.encode(tiny_row(), Tokenizer(), 10)


def test_ambiguous_template_boundary_is_rejected():
    class Changed(Tokenizer):
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            text = super().apply_chat_template(messages, add_generation_prompt=add_generation_prompt, **kwargs)
            return text if add_generation_prompt else "changed prefix" + text
    with pytest.raises(C.Refusal, match="boundary"):
        D.encode(tiny_row(), Changed(), 2048)


def test_stress_has_exact_nonpadding_length_and_keeps_answer():
    row = D.stress_case(tiny_row(), Tokenizer(), 512)
    assert len(row["input_ids"]) == 512
    assert row["answer"] == tiny_row()["answer"]
    assert row["labels"][row["answer_end"] - 1] == 900


def test_exclusive_outputs_and_manifest_tampering(tmp_path):
    C.write_json(tmp_path / "a.json", {"x": 1})
    with pytest.raises(FileExistsError):
        C.write_json(tmp_path / "a.json", {"x": 2})
    expected = C.tree_manifest(tmp_path)
    (tmp_path / "a.json").write_text("changed")
    with pytest.raises(C.Refusal, match="manifest"):
        C.verify_tree(tmp_path, expected)


def test_manifest_rejects_symlinks(tmp_path):
    (tmp_path / "link").symlink_to("/etc/passwd")
    with pytest.raises(C.Refusal, match="symlink"):
        C.tree_manifest(tmp_path)


def predecessor_record(path):
    path.mkdir()
    for name in ("checks.json", "preflight.json", "events.jsonl"):
        (path / name).write_text("{}\n")
    artifacts = C.tree_manifest(path)
    C.write_json(path / "result.json", {"schema": SCHEMA, "bundle_id": "bundle", "status": "passed",
                                        "stage": "smoke", "artifacts": artifacts})


@pytest.mark.parametrize("change", ["bundle", "stage", "status", "artifact", "missing"])
def test_predecessor_refuses_invalid_evidence(tmp_path, change):
    attempt = tmp_path / "attempt"
    predecessor_record(attempt)
    assert C.predecessor(attempt, "bundle", "smoke")["status"] == "passed"
    if change == "artifact":
        (attempt / "checks.json").write_text("changed")
    elif change == "missing":
        (attempt / "result.json").unlink()
    else:
        value = C.read_json(attempt / "result.json")
        value[{"bundle": "bundle_id", "stage": "stage", "status": "status"}[change]] = "different"
        (attempt / "result.json").write_text(json.dumps(value))
    with pytest.raises((C.Refusal, FileNotFoundError)):
        C.predecessor(attempt, "bundle", "smoke")


def test_reservation_counts_failed_or_unknown_submissions_and_forbids_duplicate_stage():
    row = S.reserve([], "one", "smoke", "bundle")
    events = [row, {"event": "submission_unknown", "run_id": "one"}]
    with pytest.raises(C.Refusal):
        S.reserve(events, "two", "smoke", "bundle")
    with pytest.raises(C.Refusal):
        S.reserve(events, "one", "train", "bundle")
    events += [S.reserve(events, "two", "train", "bundle")]
    events += [S.reserve(events, "three", "reload", "bundle")]
    with pytest.raises(C.Refusal, match="cap"):
        S.reserve(events, "four", "smoke", "different-bundle")


def test_scheduler_accounting_uses_exact_job_not_a_child_step(monkeypatch):
    def fake(args, **kwargs):
        return SimpleNamespace(stdout="123.batch|FAILED|1:0|3|\n123|COMPLETED|0:0|60|\n")
    monkeypatch.setattr(R.subprocess, "run", fake)
    assert R.accounting("123")["state"] == "COMPLETED"
    with pytest.raises(C.Refusal):
        R.accounting("123;anything")


def test_submission_retains_reservation_after_sbatch_failure(tmp_path, monkeypatch):
    bundle, volume = tmp_path / "bundle", tmp_path / "study"
    bundle.mkdir()
    volume.mkdir(mode=0o700)
    C.write_json(bundle / "config.json", {"site": {"study_root": str(volume)}})
    files = C.tree_manifest(bundle)
    C.write_json(bundle / "manifest.json", {"schema": SCHEMA, "files": files, "identity": C.identity(files)})
    monkeypatch.setattr(S, "output_image", lambda attempt: (attempt / "output.img").write_text("image"))
    monkeypatch.setattr(S.shutil, "which", lambda name, path=None: "/tools/" + name)
    calls = []
    def scheduler(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="", stderr="submission failed", returncode=1)
    monkeypatch.setattr(S.subprocess, "run", scheduler)
    with pytest.raises(C.Refusal, match="submission"):
        S.submit(bundle, "smoke", "attempt1", None, "coordinated with owner")
    assert "--export=NIL" in calls[0] and "--gres=gpu:h100:1" in calls[0] and "--time=00:30:00" in calls[0]
    assert C.read_json(volume / "attempts/attempt1/invocation.json")["gpu_model"] == "H100"
    ledger = [json.loads(line) for line in (volume / "submission-ledger.jsonl").read_text().splitlines()]
    assert sum(row.get("seconds", 0) for row in ledger) == 1800
    with pytest.raises(C.Refusal, match="already attempted"):
        S.submit(bundle, "smoke", "attempt2", None, "coordinated with owner")
    assert len(calls) == 1
    # The rehearsal never touches the ledger or the gpuplus partition.
    monkeypatch.setattr(S.subprocess, "run", lambda args, **kwargs: (calls.append(args), SimpleNamespace(
        stdout="77\n", stderr="", returncode=0))[1])
    assert S.rehearse(bundle, "rehearsal1", "coordinated with owner") == "77"
    assert "--partition=gpu_short" in calls[-1] and "--gres=gpu:1" in calls[-1]
    assert C.read_json(volume / "rehearsals/rehearsal1/invocation.json")["stage"] == "preflight"
    assert len((volume / "submission-ledger.jsonl").read_text().splitlines()) == len(ledger)


def test_study_root_must_be_private(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)  # explicit: submit() sets the process umask to 077
    with pytest.raises(C.Refusal, match="private"):
        S.study_check(shared)


def test_output_image_is_a_bounded_filesystem_holding_every_writable_directory(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setitem(S.RECIPE, "output_limit_bytes", 64 * 1024 * 1024)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    S.output_image(attempt)
    assert sorted(p.name for p in attempt.iterdir()) == ["output.img"]
    header = subprocess.run([S.tool("dumpe2fs"), "-h", str(attempt / "output.img")],
                            capture_output=True, text=True, check=True).stdout
    fields = dict(line.split(":", 1) for line in header.splitlines() if ":" in line)
    assert int(fields["Block count"]) * int(fields["Block size"]) <= 64 * 1024 * 1024
    listing = subprocess.run([S.tool("debugfs"), "-R", "ls -p /", str(attempt / "output.img")],
                             capture_output=True, text=True, check=True).stdout
    assert {"run", "home", "tmp", "shm"} <= {row.split("/")[5] for row in listing.splitlines() if row.count("/") > 5}


def test_container_binds_every_writable_path_to_the_attempt_filesystem(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    request = {"bundle": "/b", "apptainer": "/a", "stage": "smoke", "predecessor": "/p/evidence"}
    args = S.container_args(request, {"runtime_image": "/i.sif"}, Path("/o"), Path("/o/mnt"))
    binds = [args[i + 1] for i, value in enumerate(args) if value == "--bind"]
    assert binds == ["/b:/bundle:ro", "/b/model:/model:ro", "/o/mnt/run:/run:rw", "/o/mnt/tmp:/tmp:rw",
                     "/o/mnt/tmp:/var/tmp:rw", "/o/mnt/shm:/dev/shm:rw", "/o/etc/passwd:/etc/passwd:ro",
                     "/o/etc/group:/etc/group:ro", "/o/etc/resolv.conf:/etc/resolv.conf:ro",
                     "/p/evidence:/previous:ro"]
    assert args[args.index("--home") + 1] == "/o/mnt/home:/home/pilot"
    assert args[args.index("--network") + 1] == "none" and "--userns" in args and "--containall" in args
    assert "UV_PYTHON=/opt/venv/bin/python" in args
    assert not any(value.startswith("HOME=") for value in args)  # Apptainer refuses it; --home sets HOME
    assert args[args.index("/i.sif"):] == ["/i.sif", "/usr/local/bin/uv", "run", "--offline", "--no-project",
                                           "/opt/venv/bin/python", "-m", "opacity.lora_pilot", "smoke",
                                           "--bundle", "/bundle", "--out", "/run"]


MOUNTINFO = """\
1 0 0:63 / / rw,nodev,relatime - overlay overlay ro,lowerdir=/a
2 1 0:61 / /tmp rw,nosuid,nodev,relatime - tmpfs tmpfs rw,size=65536k
3 1 0:57 /tmp /tmp rw,nosuid,nodev,relatime - fuse.ext4 output.img rw,user_id=1,group_id=1
4 1 0:70 /passwd /etc/passwd rw,nosuid,relatime - tmpfs tmpfs rw
5 1 0:80 /etc/passwd /etc/passwd ro,nosuid,relatime - lustre scratch rw,flock
6 1 0:81 / /bundle ro,relatime - lustre scratch rw,flock
"""


def test_mount_table_reads_superblock_flags_and_upper_mounts():
    from opacity.lora_pilot import preflight as P
    table = P.mounts(MOUNTINFO)
    assert P.readonly(P.containing_mount(Path("/"), table))  # rw per mount, ro superblock
    assert P.containing_mount(Path("/tmp/x"), table)["type"] == "fuse.ext4"
    assert P.containing_mount(Path("/bundle/model"), table)["path"] == "/bundle"
    visible = P.visible_mounts(table)
    assert [row["path"] for row in visible] == ["/", "/tmp", "/etc/passwd", "/bundle"]
    assert visible[1]["type"] == "fuse.ext4" and P.readonly(visible[2])
    assert not P.readonly(table[3])


def test_tolerance_policy_stops_unstable_clean_runs():
    assert M.tolerance_from_repeat({"absolute": 0.0, "relative": 0.0, "tokens_equal": True}) == {
        "absolute": 1e-5, "relative": 1e-6}
    for values in ({"absolute": 0.01, "relative": 0.0, "tokens_equal": True},
                   {"absolute": 0.0, "relative": 0.0, "tokens_equal": False}):
        with pytest.raises(C.Refusal):
            M.tolerance_from_repeat(values)


def test_saved_adapter_rejects_extra_or_mis_shaped_tensors():
    inventory = [{"name": "text.q_proj", "shape": [16, 8]}]
    correct = {"base.text.q_proj.lora_A.weight": [8, 8], "base.text.q_proj.lora_B.weight": [16, 8]}
    T.validate_adapter_shapes(correct, inventory)
    with pytest.raises(C.Refusal):
        T.validate_adapter_shapes({**correct, "vision.weight": [16, 16]}, inventory)
    with pytest.raises(C.Refusal):
        T.validate_adapter_shapes({**correct, "base.text.q_proj.lora_B.weight": [8, 16]}, inventory)


def test_profiling_accumulates_without_optimiser_updates_and_clears_gradients(monkeypatch):
    class Model:
        clears = 0
        def train(self):
            pass
        def zero_grad(self, **kwargs):
            self.clears += 1
    class Budget:
        def check(self):
            pass
        def memory(self):
            return {"peak_reserved_bytes": 16, "device_total_bytes": 100}
        def headroom(self):
            return True
        def event(self, *args, **kwargs):
            pass
    monkeypatch.setattr(T.instrument, "contribution_digest", lambda model: "unchanged")
    calls = []
    def backward(model, parameters, rows, budget):
        calls.append(len(rows))
        return {"loss": 1.0}
    monkeypatch.setattr(T, "one_group", backward)
    model = Model()
    records = T.profile(model, [], [{"length": 512, "input_ids": [1] * 512}], Budget(), [512])
    assert calls == [4] and model.clears == 2
    assert records[0]["optimiser_updates"] == 0


def test_empty_or_invalid_token_weighting_is_refused():
    for supervised, total in ((0, 1), (2, 1), (0, 0)):
        with pytest.raises(C.Refusal):
            T.weighted_loss(1.0, supervised, total)


def test_unconfigured_site_and_changed_recipe_are_refused(tmp_path):
    import yaml
    root = Path(__file__).resolve().parents[1]
    example = root / "configs/lora-pilot-h-v1.yaml"
    with pytest.raises(C.Refusal, match="site setting"):
        C.config(example)
    values = yaml.safe_load(example.read_text())
    values["recipe"]["rank"] = 16
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(values))
    with pytest.raises(C.Refusal, match="recipe"):
        C.config(changed)


def test_boundary_refuses_shared_host_network_namespace(monkeypatch):
    from opacity.lora_pilot import preflight as P
    monkeypatch.setattr(P, "read_json", lambda path: {"site": {}})
    monkeypatch.setattr(P, "mounts", lambda: [])
    monkeypatch.setattr(P, "write_json", lambda *args: None)
    monkeypatch.setattr(P.os, "readlink", lambda path: "net:[123]")
    with pytest.raises(C.Refusal, match="host network"):
        P.boundary_check(Path("/bundle"), Path("/run"), {"host_netns": "net:[123]"})


def test_gpu_gate_refuses_multiple_devices_without_loading_weights(monkeypatch):
    import sys
    from opacity.lora_pilot import preflight as P
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2))
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(C.Refusal, match="exactly one"):
        P.gpu_check("H100")


def test_runtime_lock_cannot_enable_new_deltanet_kernels(monkeypatch):
    import sys
    torch = pytest.importorskip("torch")
    from opacity.lora_pilot import preflight as P
    # Distribution versions as importlib.metadata reports them: the PyPI torch
    # wheel is "2.13.0"; its CUDA build is visible only in torch.__version__.
    packages = {"torch": "2.13.0", "transformers": "5.19.0", "accelerate": "1.15.0", "peft": "0.21.0"}
    monkeypatch.setattr(P, "package_versions", lambda: packages)
    lock = {"python": sys.version.split()[0], "packages": packages, "provenance": "test fixture"}
    assert P.runtime_versions(lock)["packages"] == packages
    monkeypatch.setattr(torch, "__version__", "2.13.0+cpu")
    with pytest.raises(C.Refusal, match="characterised"):
        P.runtime_versions(lock)
    monkeypatch.undo()
    monkeypatch.setattr(P, "package_versions", lambda: packages)
    packages["causal-conv1d"] = "test"
    with pytest.raises(C.Refusal, match="reference path"):
        P.runtime_versions(lock)


def test_runtime_lock_matches_this_environment_exactly():
    # In the pilot's environment (built from the same hashed lock as the image) this runs;
    # elsewhere, e.g. the main project .venv, it skips, and check-cpu refuses any skip.
    from opacity.lora_pilot import preflight as P
    lock = json.loads((Path(__file__).resolve().parents[1] / "configs/lora-pilot-runtime/runtime-lock-h-v1.json").read_text())
    packages = P.package_versions()
    if packages != lock["packages"]:
        pytest.skip("not the honest-LoRA pilot's locked environment")
    assert packages["torch"] == "2.13.0" and packages["peft"] == "0.21.0" and "opacity" not in packages


def test_report_keeps_partial_terminal_and_check_files_as_invalid_evidence(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    C.write_json(bundle / "config.json", {"site": {}})
    files = C.tree_manifest(bundle)
    bundle_id = C.identity(files)
    C.write_json(bundle / "manifest.json", {"schema": SCHEMA, "files": files, "identity": bundle_id})
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    C.write_json(attempt / "invocation.json", {"bundle_id": bundle_id, "stage": "smoke"})
    C.write_json(attempt / "job.json", {"job_id": "123"})
    (attempt / "evidence").mkdir()
    (attempt / "evidence/result.json").write_text('{"partial":')
    (attempt / "evidence/checks.json").write_text('{"partial":')
    monkeypatch.setattr(R, "accounting", lambda job: {"state": "TIMEOUT", "exit_code": "0:15"})
    result = R.report(bundle, [attempt], tmp_path / "report")
    assert result["records"][0]["status"] == "invalid_record"
    assert result["records"][0]["checks"]["status"] == "invalid_record"
    assert not result["engineering_checks_passed"]


def test_hybrid_inventory_excludes_linear_attention_and_vision():
    torch = pytest.importorskip("torch")
    class Hybrid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(text_config=SimpleNamespace(
                num_hidden_layers=64, hidden_size=5120,
                layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)]))
            self.language_model = torch.nn.Module()
            self.language_model.layers = torch.nn.ModuleList()
            for i in range(64):
                layer = torch.nn.Module()
                layer.self_attn = torch.nn.Module()
                layer.self_attn.q_proj = torch.nn.Linear(4, 4)
                layer.self_attn.v_proj = torch.nn.Linear(4, 4)
                self.language_model.layers.append(layer)
            self.vision = torch.nn.Module()
            self.vision.q_proj = torch.nn.Linear(4, 4)
    inventory = M.targets(Hybrid())
    assert len(inventory) == 32
    assert {r["layer"] for r in inventory} == set(range(3, 64, 4))
    assert all("vision" not in r["name"] for r in inventory)


def test_accumulated_loss_gradient_matches_concatenated_token_reference():
    torch = pytest.importorskip("torch")
    torch.manual_seed(8)
    logits = torch.randn(5, 7, requires_grad=True)
    target = torch.tensor([0, 1, 2, 3, 4])
    left = torch.nn.functional.cross_entropy(logits[:1], target[:1])
    right = torch.nn.functional.cross_entropy(logits[1:], target[1:])
    loss = T.weighted_loss(left, 1, 5) + T.weighted_loss(right, 4, 5)
    gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    reference = torch.nn.functional.cross_entropy(logits, target)
    expected = torch.autograd.grad(reference, logits)[0]
    assert torch.allclose(gradient, expected, atol=1e-7)


def test_tiny_cpu_checkpointed_backward_save_reload_and_disable(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors.torch")
    torch.manual_seed(41)
    cfg = transformers.LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    clean = transformers.LlamaForCausalLM(cfg).eval()
    clean.requires_grad_(False)
    original = {key: value.detach().clone() for key, value in clean.state_dict().items()}
    base = list(clean.named_parameters())
    ids = torch.tensor([[1, 4, 7, 3, 2]])
    with torch.no_grad():
        baseline = clean(input_ids=ids, use_cache=False).logits.clone()
    inventory = [{"name": f"model.layers.0.self_attn.{name}", "shape": [16, 16]} for name in ("q_proj", "v_proj")]
    model = peft.get_peft_model(clean, peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                                     target_modules=["q_proj", "v_proj"], bias="none"),
                                autocast_adapter_dtype=True)
    parameters, _ = M.adapter_inventory(model, inventory)
    base_hash = M.base_digest(base)
    contribution = M.contribution_digest(model)
    M.checkpointing(model)
    optimiser = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=0.0)
    model.train()
    for _ in range(2):
        optimiser.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        assert any(torch.count_nonzero(p.grad) for p in parameters)
        optimiser.step()
    optimiser.zero_grad(set_to_none=True)
    assert M.contribution_digest(model) != contribution
    assert M.base_digest(base) == base_hash
    model.eval()
    with torch.no_grad():
        adapted = model(input_ids=ids, use_cache=False).logits.clone()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    state = peft.get_peft_model_state_dict(model, save_embedding_layers=False)
    T.validate_adapter_shapes({k: v.shape for k, v in state.items()}, inventory)
    safetensors.save_file({k: v.detach().contiguous() for k, v in state.items()}, adapter / "adapter_model.safetensors")
    model.peft_config["default"].save_pretrained(adapter)
    T.saved_adapter(adapter, inventory)
    fresh = transformers.LlamaForCausalLM(cfg)
    fresh.load_state_dict(original)
    fresh.requires_grad_(False)
    reloaded = peft.PeftModel.from_pretrained(fresh, adapter, is_trainable=False).eval()
    M.adapter_inventory(reloaded, inventory, trainable=False)
    with torch.no_grad():
        assert torch.allclose(reloaded(input_ids=ids, use_cache=False).logits, adapted, atol=1e-6)
        with reloaded.disable_adapter():
            assert torch.allclose(reloaded(input_ids=ids, use_cache=False).logits, baseline, atol=1e-6)


class StubBudget:
    def check(self):
        pass

    def event(self, *args, **kwargs):
        pass

    def memory(self):
        return {"peak_reserved_bytes": 0, "device_total_bytes": 1}

    def headroom(self):
        return True


def tiny_qwen35(transformers):
    """Qwen3.5's real classes and hybrid layout (full attention every fourth layer), scaled down."""
    return transformers.Qwen3_5Config(
        text_config={"vocab_size": 1024, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 8,
                     "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "attn_output_gate": True,
                     "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_num_key_heads": 2,
                     "linear_num_value_heads": 4, "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
                     "rope_parameters": {"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default",
                                         "rope_theta": 10000000, "partial_rotary_factor": 0.25}},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64,
                       "num_position_embeddings": 16},
        image_token_id=1000, video_token_id=1001, vision_start_token_id=1002, vision_end_token_id=1003,
        architectures=["Qwen3_5ForConditionalGeneration"])


def test_tiny_qwen35_through_the_runner_attach_train_diagnose_save_reload_and_disable(tmp_path, monkeypatch):
    """The stage functions themselves on Qwen3.5's module tree, bf16 base and fp32 adapters, on CPU."""
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(M, "LAYERS", 8)
    monkeypatch.setattr(M, "HIDDEN", 64)
    monkeypatch.setattr(M, "DEVICE", "cpu")
    torch.manual_seed(5)
    transformers.Qwen3_5ForConditionalGeneration(tiny_qwen35(transformers)).save_pretrained(tmp_path / "base")
    model, base = M.load_model(tmp_path / "base")  # the production loader, as every stage uses it
    assert {p.dtype for _, p in base} >= {torch.bfloat16}
    inventory = M.targets(model)
    assert [r["name"].split("language_model.")[1] for r in inventory] == [
        "layers.3.self_attn.q_proj", "layers.3.self_attn.v_proj", "layers.7.self_attn.q_proj", "layers.7.self_attn.v_proj"]
    assert M.meta_targets(tmp_path / "base") == inventory
    tok = Tokenizer()
    rows = [D.encode(r, tok, 2048) for r in D.generate(271028)]
    diagnostic = [r for r in rows if r["split"] == "diagnostic"]
    check = lambda: None
    baseline = M.diagnostics(model, tok, diagnostic, tmp_path / "baseline", check)
    assert baseline["denominator"] == 16 and all(len(r["tokens"]) > 0 for r in baseline["rows"])
    assert np_shape(tmp_path / "baseline" / baseline["rows"][0]["logits"]) == (2, 1024)
    M.diagnostics(model, tok, diagnostic, tmp_path / "repeat", check)
    tolerance = M.tolerance_from_repeat(M.compare(tmp_path / "baseline", tmp_path / "repeat"))
    before = M.base_digest(base)
    model, parameters, details = M.attach(model, inventory)
    assert len(parameters) == 8 and all(p.dtype == torch.float32 and p.requires_grad for p in parameters)
    assert all(not p.requires_grad for _, p in base)
    M.diagnostics(model, tok, diagnostic, tmp_path / "initial", check)
    M.compare(tmp_path / "baseline", tmp_path / "initial", tolerance)  # zero-initialised B: no change yet
    M.checkpointing(model)
    contribution = M.contribution_digest(model)
    steps, optimiser = T.train_updates(model, parameters, base, rows, D.sampler(rows, 271028), 2, StubBudget())
    assert [s["update"] for s in steps] == [1, 2] and all(s["supervised_tokens"] > 0 for s in steps)
    assert M.contribution_digest(model) != contribution
    assert M.base_digest(base) == before
    assert set(optimiser.state) == set(parameters)
    model.gradient_checkpointing_disable()
    adapted = M.diagnostics(model, tok, diagnostic, tmp_path / "adapted", check)
    with pytest.raises(C.Refusal):
        M.compare(tmp_path / "baseline", tmp_path / "adapted", tolerance)  # the trained adapter is visible
    T.save_adapter(model, tmp_path / "adapter", inventory)
    # The reload stage: a fresh base from the saved files, the saved adapter, then the adapter disabled.
    fresh, _ = M.load_model(tmp_path / "base")
    reloaded = peft.PeftModel.from_pretrained(fresh, tmp_path / "adapter", is_trainable=False,
                                              autocast_adapter_dtype=True, local_files_only=True)
    M.adapter_inventory(reloaded, inventory, trainable=False)
    M.diagnostics(reloaded, tok, diagnostic, tmp_path / "reloaded", check)
    M.compare(tmp_path / "adapted", tmp_path / "reloaded", tolerance)
    with reloaded.disable_adapter():
        M.diagnostics(reloaded, tok, diagnostic, tmp_path / "disabled", check)
    M.compare(tmp_path / "baseline", tmp_path / "disabled", tolerance)
    assert adapted["denominator"] == 16


def np_shape(path):
    import numpy as np
    return np.load(path, allow_pickle=False).shape


def test_mount_policy_accepts_the_observed_viking_gpu_container_and_nothing_wider():
    from opacity.lora_pilot import preflight as P
    observed = json.loads((Path(__file__).parent / "fixtures/lora_pilot_viking_gpu_mounts.json").read_text())["mounts"]
    assert P.mount_violations(observed) == []
    assert P.readonly(P.containing_mount(Path("/"), observed))
    assert P.containing_mount(Path("/dev/shm"), observed)["type"] == "fuse.ext4"
    assert P.readonly(P.containing_mount(Path("/etc/passwd"), observed))
    extra = {"options": ["rw"], "super_options": ["rw"], "type": "lustre"}
    assert P.mount_violations(observed + [{**extra, "path": "/mnt/scratch"}]) == ["unexpected mount /mnt/scratch"]
    assert P.mount_violations(observed + [{**extra, "path": "/etc/passwd"}]) == [
        "writable mount /etc/passwd outside the bounded/runtime paths"]
    assert P.mount_violations(observed + [{**extra, "path": "/usr/bin/nvidia-smi"}]) == [
        "unexpected mount /usr/bin/nvidia-smi"]  # a driver file is allowed only read-only


def test_credential_check_distinguishes_absent_inaccessible_and_present(tmp_path):
    from opacity.lora_pilot import preflight as P
    locked = tmp_path / "root"
    (locked / ".cache").mkdir(parents=True)
    (tmp_path / "token").write_text("not a real token")
    locked.chmod(0o000)
    try:
        assert P.visibility(str(locked / ".cache/token")) == "inaccessible"
    finally:
        locked.chmod(0o700)
    assert P.visibility(str(tmp_path / "absent")) == "absent"
    assert P.visibility(str(tmp_path / "token")) == "present"


def passed_attempt(root, name, stage, bundle_id, checks):
    attempt = root / name
    evidence = attempt / "evidence"
    evidence.mkdir(parents=True)
    C.write_json(attempt / "invocation.json", {"bundle_id": bundle_id, "stage": stage})
    C.write_json(attempt / "job.json", {"job_id": "1"})
    C.write_json(evidence / "checks.json", checks)
    for name in ("preflight.json", "events.jsonl"):
        (evidence / name).write_text("{}\n")
    C.write_json(evidence / "result.json", {"schema": SCHEMA, "stage": stage, "status": "passed",
                                            "bundle_id": bundle_id, "artifacts": C.tree_manifest(evidence)})
    return attempt


def test_report_passes_all_three_stages_even_with_a_rehearsal_in_it(tmp_path, monkeypatch):
    # report-01 on Viking (27 October) said False with a passing rehearsal included.
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    C.write_json(bundle / "config.json", {"site": {}})
    files = C.tree_manifest(bundle)
    bundle_id = C.identity(files)
    C.write_json(bundle / "manifest.json", {"schema": SCHEMA, "files": files, "identity": bundle_id})
    monkeypatch.setattr(R, "accounting", lambda job: {"state": "COMPLETED", "exit_code": "0:0", "elapsed_seconds": 1})
    reload_checks = {"clean": {"passed": True, "base_digest": "d"}, "adapter": {"passed": True, "base_digest": "d"}}
    attempts = [passed_attempt(tmp_path, "rehearsal", "preflight", bundle_id, {"rehearsal": True}),
                passed_attempt(tmp_path, "smoke", "smoke", bundle_id, {}),
                passed_attempt(tmp_path, "train", "train", bundle_id, {}),
                passed_attempt(tmp_path, "reload", "reload", bundle_id, reload_checks)]
    result = R.report(bundle, attempts, tmp_path / "report")
    assert [r["status"] for r in result["records"]] == ["passed"] * 4
    assert result["engineering_checks_passed"]
    assert "Base digest identical in both: True." in (tmp_path / "report/report.md").read_text()
    assert not R.report(bundle, attempts[:3], tmp_path / "report-without-reload")["engineering_checks_passed"]


def test_profile_stage_reserves_without_predecessor_and_fits_the_cap():
    row = S.reserve([], "lengths-01", "profile", "bundle")
    assert row["seconds"] == 1800
    with pytest.raises(C.Refusal, match="already attempted"):
        S.reserve([row], "lengths-02", "profile", "bundle")


def test_tiny_qwen35_memory_profile_trains_twice_then_profiles_each_length(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(M, "LAYERS", 8)
    monkeypatch.setattr(M, "HIDDEN", 64)
    monkeypatch.setattr(M, "DEVICE", "cpu")
    torch.manual_seed(5)
    transformers.Qwen3_5ForConditionalGeneration(tiny_qwen35(transformers)).save_pretrained(tmp_path / "base")
    model, base = M.load_model(tmp_path / "base")
    inventory = M.targets(model)
    tok = Tokenizer()
    rows = [D.encode(r, tok, 2048) for r in D.generate(271028)]
    stress = [D.stress_case(tiny_row(), tok, length) for length in (512, 640)]
    checks = T.memory_profile(model, base, inventory, rows, D.sampler(rows, 271028), stress, StubBudget())
    assert [s["update"] for s in checks["steps"]] == [1, 2]
    assert [p["length"] for p in checks["profiles"]] == [512, 640] and checks["lengths_requested"] == [512, 640]
    assert all(p["optimiser_updates"] == 0 for p in checks["profiles"])


def _poll_with(monkeypatch, tmp_path, replies):
    """Run Budget's driver monitor against a scripted nvidia-smi: each reply is a value or an exception."""
    from opacity.lora_pilot import preflight as P

    budget = T.Budget(deadline=10**12, out=tmp_path, gpu={"uuid": "GPU-x"})
    script = iter(replies)

    def query(uuid, field):
        reply = next(script, None)
        if reply is None:
            budget.stop.set()
            return "1"
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(P, "nvidia_query", query)
    monkeypatch.setattr(budget.stop, "wait", lambda seconds: None)
    budget._poll()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    return budget, events


def test_monitor_survives_a_brief_driver_stall(monkeypatch, tmp_path):
    """7 November 2026: one nvidia-smi call failed as a neighbouring job exited, and the
    whole construction stopped. A few failures in a row are recorded and tolerated."""
    stall = TimeoutError("nvidia-smi timed out")
    budget, events = _poll_with(monkeypatch, tmp_path, ["100", stall, stall, "200"] + [stall] * 4 + ["300"])
    assert budget.error is None
    budget.check()
    assert [e["consecutive"] for e in events if e["event"] == "resources_error"] == [1, 2, 1, 2, 3, 4]
    assert budget.driver_peak == 300 * 1024 * 1024


def test_monitor_stops_the_stage_when_the_driver_stays_unreachable(monkeypatch, tmp_path):
    stall = RuntimeError("no devices")
    budget, events = _poll_with(monkeypatch, tmp_path, ["100"] + [stall] * T.Budget.MAX_QUERY_FAILURES + ["200"])
    assert "5 driver queries failed in a row" in budget.error
    assert sum(e["event"] == "resources_error" for e in events) == 5
    with pytest.raises(Exception, match="resource monitor failed"):
        budget.check()
