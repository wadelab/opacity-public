"""Preparation and runtime checks; no enforcement claim from offline flags alone."""
from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import MODEL, REVISION, SCHEMA
from .common import (RECIPE, STUDY_PROFILE_LENGTHS, bundle_manifest, config, digest, identity, read_json, require,
                     tree_manifest, write_json)


def package_versions() -> dict[str, str]:
    return dict(sorted((d.metadata["Name"].lower().replace("_", "-"), d.version)
                       for d in importlib.metadata.distributions()))


def runtime_versions(lock: dict) -> dict:
    import torch

    actual = package_versions()
    require(actual == lock["packages"], "runtime package set differs from locked image environment")
    # The PyPI wheel's distribution version is 2.13.0; the CUDA build shows only
    # in torch.__version__ (checked on Viking, 27 October).
    require(actual.get("torch") == "2.13.0" and torch.__version__ == "2.13.0+cu130"
            and torch.version.cuda == "13.0" and actual.get("transformers") == "5.19.0"
            and actual.get("peft") == "0.21.0" and actual.get("accelerate") == "1.15.0",
            "wrong characterised runtime/PEFT candidate")
    require(not {"causal-conv1d", "flash-linear-attention", "fla-core"} & actual.keys(),
            "new DeltaNet kernels would change the characterised reference path")
    require(lock.get("python") == sys.version.split()[0], "Python differs from runtime lock")
    require(bool(lock.get("provenance")), "wheel/index provenance missing")
    return {"python": sys.version.split()[0], "packages": actual, "executable": sys.executable}


def snapshot_files(snapshot: Path) -> list[Path]:
    allowed = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
               "chat_template.jinja", "model.safetensors.index.json"}
    selected = [p for p in snapshot.iterdir() if p.name in allowed
                or (p.name.startswith("model") and p.name.endswith(".safetensors"))]
    require({"config.json", "tokenizer_config.json"} <= {p.name for p in selected}, "tokenizer/config missing")
    weights = [p for p in selected if p.suffix == ".safetensors"]
    require(bool(weights), "no safe model weights")
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        require(set(read_json(index)["weight_map"].values()) == {p.name for p in weights}, "incomplete weight shards")
    else:
        require(len(weights) == 1 and weights[0].name == "model.safetensors", "unindexed weight shards")
    return sorted(selected)


def weight_metadata(path: Path) -> dict:
    # Read only the safetensors header, not a tensor or executable pickle.
    with path.open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        require(0 < size < 64 * 1024 * 1024, "invalid safetensors header")
        header = json.loads(stream.read(size))
    result = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        require(info["dtype"] in {"BF16", "F32", "I64", "I32", "BOOL"}, "unexpected cached tensor dtype")
        result[name] = {"dtype": info["dtype"], "shape": info["shape"]}
    return result


def prepare(config_path: Path, out: Path, cpu_checks: Path) -> None:
    from transformers import AutoTokenizer
    from .data import encode, generate, sampler, stress_case
    from .model import meta_targets

    cfg = config(config_path)
    site = cfg["site"]
    snapshot = Path(site["snapshot"])
    require(not out.is_relative_to(snapshot.resolve()) and not out.is_relative_to(cpu_checks),
            "bundle output must be separate from cached inputs and CPU receipt")
    from .submit import study_check
    study = study_check(Path(site["study_root"]))
    require(not out.is_relative_to(study / "attempts"), "bundle must not sit among attempt outputs")
    backup = Path(site["backup_root"]).resolve(strict=True)
    require(backup.is_dir() and not backup.is_relative_to(Path("/mnt/scratch"))
            and backup.stat().st_mode & 0o077 == 0, "pre-provision a private durable backup root outside scratch")
    require(snapshot.name == REVISION, "snapshot directory does not name the characterised revision")
    probe = read_json(Path(site["probe_record"]))
    require(probe.get("hf") == MODEL and probe.get("revision") == REVISION
            and probe.get("arch") == "Qwen3_5ForConditionalGeneration", "probe provenance disagrees")
    revisions = {line.split()[1] for line in Path(site["revision_record"]).read_text().splitlines()
                 if len(line.split()) >= 2 and line.split()[0] == MODEL}
    require(revisions == {REVISION}, "ambiguous or different recorded base revision")
    require(digest(Path(site["runtime_image"])) == site["runtime_sha256"], "runtime image hash mismatch")
    lock = read_json(Path(site["runtime_lock"]))
    runtime_versions(lock)
    root = Path(__file__).resolve().parents[3]
    clean = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                           capture_output=True, text=True).stdout
    require(not clean, "commit the implementation/config before preparing the frozen bundle")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    receipt = read_json(cpu_checks / "result.json")
    require(receipt.get("source_commit") == commit and receipt.get("status") == "passed",
            "passing CPU test receipt for this source commit required")
    require(receipt.get("artifacts") == {"pytest.log": digest(cpu_checks / "pytest.log"),
                                         "junit.xml": digest(cpu_checks / "junit.xml")}, "CPU receipt changed")
    out.mkdir(parents=True, exist_ok=False)
    os.chmod(out, 0o700)
    # Preparation can copy/hash >54 GB. Run it in a suitable allocation, never the login node.
    model_dir = out / "model"
    model_dir.mkdir()
    weight_info = {}
    for source in snapshot_files(snapshot):
        destination = model_dir / source.name
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        if destination.suffix == ".safetensors":
            weight_info[source.name] = weight_metadata(destination)
    write_json(out / "weight-metadata.json", weight_info)
    inventory = meta_targets(model_dir)
    write_json(out / "targets.json", inventory)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    rows = [encode(r, tok, RECIPE["max_tokens"]) for r in generate(RECIPE["seed"])]
    write_json(out / "data.json", rows)
    write_json(out / "stress.json", [stress_case(rows[0], tok, length) for length in RECIPE["profile_lengths"]])
    write_json(out / "stress-study.json", [stress_case(rows[0], tok, length) for length in STUDY_PROFILE_LENGTHS])
    write_json(out / "order.json", sampler(rows, RECIPE["seed"]))
    write_json(out / "config.json", cfg)
    write_json(out / "runtime-lock.json", lock)
    write_json(out / "provenance.json", {"source_commit": commit, "probe": probe, "revision": REVISION})
    shutil.copytree(cpu_checks, out / "cpu-checks")
    code_dir = out / "src" / "opacity"
    code_dir.mkdir(parents=True)
    shutil.copyfile(root / "src/opacity/__init__.py", code_dir / "__init__.py")
    shutil.copytree(root / "src/opacity/lora_pilot", code_dir / "lora_pilot",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (out / "scripts").mkdir()
    shutil.copyfile(root / "scripts/viking2_lora_pilot.slurm", out / "scripts/viking2_lora_pilot.slurm")
    files = tree_manifest(out)
    write_json(out / "manifest.json", {"schema": SCHEMA, "files": files, "identity": identity(files)})
    # Files are private, immutable inputs. Container mounts add kernel-enforced read-only access.
    for path in out.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
        elif path.is_dir():
            path.chmod(0o500)
    out.chmod(0o500)


def mounts(text: str | None = None) -> list[dict]:
    result = []
    for line in (Path("/proc/self/mountinfo").read_text() if text is None else text).splitlines():
        left, right = line.split(" - ", 1)
        fields = left.split()
        # Linux mountinfo escapes spaces/backslashes; decode only its documented octal forms.
        name = fields[4]
        for escaped, decoded in ((r"\040", " "), (r"\011", "\t"), (r"\012", "\n"), (r"\134", "\\")):
            name = name.replace(escaped, decoded)
        kind, _, super_options = (right.split() + ["", ""])[:3]
        result.append({"path": name, "options": fields[5].split(","), "type": kind,
                       "super_options": super_options.split(",")})
    return result


def readonly(row: dict) -> bool:
    # Apptainer's root overlay is "rw" per mount but read-only in its superblock.
    return "ro" in row["options"] or "ro" in row["super_options"]


CREDENTIAL_PATHS = ("/home/pilot/.cache/huggingface/token", "/home/pilot/.config/mimo/api.env",
                    "/root/.cache/huggingface/token", "/home/pilot/.codex/auth.json")
ALLOWED_TREES = ("/bundle", "/model", "/previous", "/run", "/tmp", "/var/tmp", "/home/pilot",
                 "/proc", "/sys", "/dev", "/.singularity.d/libs")
ALLOWED_FILES = {"/", "/etc/hosts", "/etc/hostname", "/etc/resolv.conf", "/etc/passwd",
                 "/etc/group", "/etc/localtime"}
WRITABLE_TREES = ("/run", "/tmp", "/var/tmp", "/home/pilot", "/proc", "/sys", "/dev")


def visibility(path: str) -> str:
    """'present', 'absent', or 'inaccessible' (the process cannot even stat it)."""
    try:
        return "present" if Path(path).exists() else "absent"
    except PermissionError:
        return "inaccessible"


def mount_violations(table: list[dict]) -> list[str]:
    problems = []
    for row in visible_mounts(table):
        path = Path(row["path"])
        # --nv binds the host driver's tools and data files (nvidia-smi, EGL/GLVND
        # descriptors, nvoptix.bin) read-only; seen on a Viking GPU node, 27 October.
        driver_file = readonly(row) and "nvidia" in row["path"]
        if not (row["path"] in ALLOWED_FILES or driver_file
                or any(path.is_relative_to(Path(p)) for p in ALLOWED_TREES)):
            problems.append(f"unexpected mount {row['path']}")
        elif not readonly(row) and not any(path.is_relative_to(Path(p)) for p in WRITABLE_TREES):
            problems.append(f"writable mount {row['path']} outside the bounded/runtime paths")
    return problems


def visible_mounts(table: list[dict]) -> list[dict]:
    """Drop mounts covered by a later mount at the same or an enclosing path."""
    return [row for i, row in enumerate(table)
            if not any(Path(row["path"]).is_relative_to(Path(later["path"])) for later in table[i + 1:])]


def containing_mount(path: Path, table: list[dict]) -> dict:
    # mountinfo lists mounts in order, so on a tie the later (upper) mount is the visible one.
    matches = [(len(row["path"]), i, row) for i, row in enumerate(table) if path.is_relative_to(Path(row["path"]))]
    require(bool(matches), "path has no mount record")
    return max(matches, key=lambda match: match[:2])[2]


def network_probe(target: str, port: int, *, dns: bool = False) -> bool:
    expression = ("import socket,sys; socket.getaddrinfo(sys.argv[1],int(sys.argv[2]))" if dns else
                  "import socket,sys; socket.create_connection((sys.argv[1],int(sys.argv[2])),2).close()")
    try:
        result = subprocess.run([sys.executable, "-c", expression, target, str(port)],
                                capture_output=True, timeout=5, check=False)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def boundary_check(bundle: Path, out: Path, context: dict, *, write_probe: Path | None = None) -> dict:
    """write_probe: the read-only input where a write must fail; the pilot's is in its bundle's model
    copy, the deception runner's in the /model mount (it binds the pilot's model rather than copying)."""
    import re
    import socket

    require(bundle == Path("/bundle") and out.resolve().is_relative_to(Path("/run")), "GPU modes require the isolated launcher")
    site = read_json(bundle / "config.json")["site"]
    table = mounts()
    # /sys is the host's sysfs, so /sys/class/net lists the host's interfaces;
    # if_nameindex asks the kernel about this process's own network namespace.
    interfaces = sorted(name for _, name in socket.if_nameindex())
    write_json(out / "boundary-observed.json", {"mounts": table, "netns": os.readlink("/proc/self/ns/net"),
                                                "interfaces": interfaces,
                                                "sysfs_net_view": sorted(os.listdir("/sys/class/net"))})
    require(os.readlink("/proc/self/ns/net") != context["host_netns"], "host network namespace still active")
    # A failed probe means nothing unless the same target answers outside the container.
    require(context["host_probes"] == {"ip": True, "dns": True}, "egress probes not reachable from the host")
    require(set(interfaces) <= {"lo"}, "non-loopback interface inside training runtime")
    require(not network_probe(site["egress_ip"], site["egress_port"]), "IP egress probe succeeded")
    require(not network_probe(site["egress_dns"], site["egress_port"], dns=True), "DNS probe succeeded")
    require(visibility(context["sentinel"]) != "present", "outside sentinel visible")
    forbidden = [name for name in os.environ if re.search(r"(?:^|_)(KEY|TOKEN|SECRET|PASSWORD)(?:$|_)", name)]
    require(not forbidden, "credential variable name present")
    credentials = {token: visibility(token) for token in CREDENTIAL_PATHS}
    require("present" not in credentials.values(), "credential path present")
    inputs = [Path("/bundle"), Path("/model")]
    if context["predecessor"]:
        inputs.append(Path("/previous"))
    for path in inputs:
        require(readonly(containing_mount(path, table)), "input mount is writable")
    require(readonly(containing_mount(Path("/"), table)), "runtime root is writable")
    problems = mount_violations(table)
    require(not problems, "mount policy: " + "; ".join(problems[:5]) + " (see boundary-observed.json)")
    for path in (Path("/tmp"), Path("/var/tmp"), Path("/home/pilot"), Path("/dev/shm")):
        require(path.stat().st_dev == out.stat().st_dev, "temporary/home/shm writes escape bounded output filesystem")
    fs = os.statvfs(out)
    require(fs.f_blocks * fs.f_frsize <= RECIPE["output_limit_bytes"], "output filesystem exceeds hard byte cap")
    device = os.statvfs("/dev")
    require(device.f_blocks * device.f_frsize <= RECIPE["output_limit_bytes"], "device filesystem is not bounded")
    probe = out / "write-sentinel"
    with probe.open("x") as stream:
        stream.write("write succeeded\n")
    # Attempt a reversible exclusive create, never touch an actual model file.
    attempted = bundle / "model" / "write-must-fail" if write_probe is None else write_probe
    try:
        with attempted.open("x"):
            pass
    except OSError as error:
        import errno
        # EACCES is possible because staged directories are also mode 0500.
        # The independently checked 'ro' mount flag supplies enforcement evidence.
        require(error.errno in {errno.EROFS, errno.EACCES}, "unexpected read-only input write failure")
        readonly_write_errno = error.errno
    else:
        raise RuntimeError("read-only mount write unexpectedly succeeded")
    return {"network_namespace_changed": True, "interfaces": interfaces,
            "ip_egress_denied": True, "dns_denied": True, "outside_sentinel_hidden": True,
            "credentials_absent": True, "credential_paths": credentials, "mounts": table,
            "output_filesystem_bytes": fs.f_blocks * fs.f_frsize, "write_checks_passed": True,
            "readonly_write_errno": readonly_write_errno}


def nvidia_query(uuid: str, field: str) -> str:
    result = subprocess.run(["nvidia-smi", "--id=" + uuid, "--query-gpu=" + field, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=5)
    return result.stdout.strip()


def gpu_check(model: str) -> dict:
    import torch

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one visible CUDA GPU required")
    props = torch.cuda.get_device_properties(0)
    require(hasattr(props, "uuid") and "SLURM_JOB_ID" in os.environ, "Slurm allocation or CUDA GPU UUID unavailable")
    # nvidia-smi names GPUs "GPU-<uuid>"; torch prints the bare UUID.
    uuid = str(props.uuid) if str(props.uuid).startswith("GPU-") else "GPU-" + str(props.uuid)
    hardware = {"name": props.name, "total_bytes": props.total_memory, "uuid": uuid,
                "cuda": torch.version.cuda, "driver": nvidia_query(uuid, "driver_version"),
                "job_id": os.environ["SLURM_JOB_ID"]}
    require(model in props.name, f"one Slurm-allocated {model} required")
    return hardware


def runtime_preflight(bundle: Path, out: Path, context: dict) -> dict:
    boundary = boundary_check(bundle, out, context)
    manifest = bundle_manifest(bundle)
    versions = runtime_versions(read_json(bundle / "runtime-lock.json"))
    hardware = gpu_check(context["gpu_model"])
    report = {"schema": SCHEMA, "bundle_id": manifest["identity"], "boundary": boundary,
              "runtime": versions, "hardware": hardware}
    write_json(out / "preflight.json", report)
    return report
