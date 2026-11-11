"""One-stage Slurm submission, durable reservations and isolated launch.

Each attempt writes only to its own ext4 image of output_limit_bytes, created
at submission and FUSE-mounted inside the job. The ledger and the launcher's
records sit beside it in the private study root, outside the container.
"""
from __future__ import annotations

import fcntl
import getpass
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import MAX_RESERVED_SECONDS, SCHEMA, STAGE_SECONDS
from .common import (RECIPE, append_event, bundle_manifest, digest, predecessor, read_json,
                     require, safe_name, tree_manifest_subset, write_json)

SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
REHEARSAL_SECONDS = 1800


def study_check(root: Path) -> Path:
    root = root.resolve(strict=True)
    info = root.stat()
    require(root.is_dir() and info.st_uid == os.getuid() and info.st_mode & 0o077 == 0,
            "site.study_root must be a private directory owned by this user")
    return root


def tool(name: str) -> str:
    found = shutil.which(name, path=os.environ.get("PATH", "") + ":" + SYSTEM_PATH)
    require(found is not None, f"{name} not found; load the site's Slurm and Apptainer commands first")
    return found


def output_image(attempt: Path) -> None:
    """The attempt's only writable filesystem; its capacity is the write bound."""
    template = attempt / "template"
    for name in ("run", "home", "tmp", "shm"):
        (template / name).mkdir(parents=True)
    subprocess.run([tool("mkfs.ext4"), "-q", "-F", "-m", "0", "-E", f"root_owner={os.getuid()}:{os.getgid()}",
                    "-d", str(template), str(attempt / "output.img"), f"{RECIPE['output_limit_bytes'] // 1024}k"],
                   check=True, capture_output=True, timeout=300)
    shutil.rmtree(template)


def mount(image: Path, point: Path, *, writable: bool) -> None:
    point.mkdir()
    subprocess.run([tool("fuse2fs"), str(image), str(point), "-o", "rw" if writable else "ro"],
                   check=True, capture_output=True, timeout=300)
    require(os.path.ismount(point), "attempt filesystem did not mount")


def unmount(point: Path) -> None:
    if os.path.ismount(point):
        subprocess.run([tool("fusermount"), "-u", str(point)], check=True, capture_output=True, timeout=300)
    point.rmdir()


def extract(attempt: Path) -> Path:
    """Copy the payload's /run out of the image through a read-only mount."""
    evidence, partial = attempt / "evidence", attempt / "evidence.partial"
    require(not evidence.exists() and not partial.exists(), "evidence already extracted or extraction interrupted")
    point = attempt / "ro-mnt"
    mount(attempt / "output.img", point, writable=False)
    try:
        shutil.copytree(point / "run", partial, symlinks=True)
    finally:
        unmount(point)
    partial.rename(evidence)
    return evidence


def reserve(events: list[dict], run_id: str, stage: str, bundle_id: str) -> dict:
    require(stage in STAGE_SECONDS, "invalid stage")
    require(not any(row.get("run_id") == run_id for row in events), "run ID already reserved")
    require(not any(row.get("event") == "reserve" and row.get("bundle_id") == bundle_id
                    and row.get("stage") == stage for row in events), "stage already attempted; no automatic retry")
    used = sum(row["seconds"] for row in events if row.get("event") == "reserve")
    require(used + STAGE_SECONDS[stage] <= MAX_RESERVED_SECONDS, "150 GPU-minute reservation cap exceeded")
    return {"event": "reserve", "run_id": run_id, "stage": stage, "bundle_id": bundle_id,
            "seconds": STAGE_SECONDS[stage], "time": time.time()}


def frozen_bundle(bundle: Path) -> tuple[Path, dict, dict, Path]:
    bundle = bundle.resolve(strict=True)
    manifest = bundle_manifest(bundle, verify=False)
    # Large hashes are rechecked within the allocation, not on the login node.
    controls = {name: sha for name, sha in manifest["files"].items() if not name.startswith("model/")}
    require(tree_manifest_subset(bundle, controls) == controls, "frozen control files changed")
    site = read_json(bundle / "config.json")["site"]
    study = study_check(Path(site["study_root"]))
    require(not bundle.is_relative_to(study / "attempts"), "bundle must not sit among attempt outputs")
    return bundle, manifest, site, study


def new_attempt(directory: Path, request: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("host-home", "host-tmp", "host-cache"):
        (directory / name).mkdir()
    write_json(directory / "invocation.json", request)
    output_image(directory)
    return directory


def sbatch(out: Path, partition: str, gres: str, seconds: int, name: str, executables: dict,
           script: str = "scripts/viking2_lora_pilot.slurm") -> list[str]:
    bundle = Path(read_json(out / "invocation.json")["bundle"])
    args = [executables["sbatch"], "--parsable", "--export=NIL", "--partition=" + partition,
            "--account=CHANGE_ME", "--gres=" + gres, "--cpus-per-task=8", "--mem=160G",
            f"--time={seconds // 3600:02d}:{seconds % 3600 // 60:02d}:00",
            "--job-name=" + name, "--chdir=" + str(bundle), "--output=" + str(out / "slurm.log"),
            str(bundle / script), str(bundle), str(out), sys.executable,
            executables["uv"]]
    write_json(out / "submission.json", {"argv": args})
    return args


def record_job(out: Path, result) -> str:
    write_json(out / "sbatch.json", {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    job_id = result.stdout.strip().split(";")[0]
    require(result.returncode == 0 and job_id.isdigit(), "submission failed/ambiguous; reservation retained")
    write_json(out / "job.json", {"job_id": job_id})
    return job_id


def submit(bundle: Path, stage: str, run_id: str, previous: Path | None, owner_note: str) -> str:
    os.umask(0o077)
    require(bool(owner_note.strip()), "record coordination with the current submissions owner")
    run_id = safe_name(run_id)
    bundle, manifest, site, study = frozen_bundle(bundle)
    previous = previous.resolve(strict=True) if previous else None
    if stage in ("train", "reload"):
        require(previous is not None and previous.parent == study / "attempts", "predecessor outside study attempts")
        predecessor(previous / "evidence", manifest["identity"], "smoke" if stage == "train" else "train")
        from .report import require_completed
        require_completed(previous)
    else:
        require(previous is None, f"{stage} cannot take a predecessor")
    executables = {name: tool(name) for name in ("sbatch", "uv", "apptainer")}
    ledger = study / "submission-ledger.jsonl"
    lock = study / "submission.lock"
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        events = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        reservation = reserve(events, run_id, stage, manifest["identity"])
        evidence = previous / "evidence" if previous else None
        out = new_attempt(study / "attempts" / run_id, {
            "schema": SCHEMA, "stage": stage, "gpu_model": "H100", "bundle_id": manifest["identity"],
            "run_id": run_id, "bundle": str(bundle), "predecessor": str(evidence) if evidence else None,
            "predecessor_result_sha256": digest(evidence / "result.json") if evidence else None,
            "owner_coordination": owner_note, "apptainer": executables["apptainer"]})
        append_event(ledger, reservation)  # durable reservation before sbatch, including ambiguous failures
        args = sbatch(out, "gpuplus", "gpu:h100:1", STAGE_SECONDS[stage], "op-lora-" + stage, executables)
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=60)
        except BaseException:
            append_event(ledger, {"event": "submission_unknown", "run_id": run_id, "time": time.time()})
            raise
        job_id = record_job(out, result)
        append_event(ledger, {"event": "submitted", "run_id": run_id, "job_id": job_id, "time": time.time()})
        return job_id


def rehearse(bundle: Path, run_id: str, owner_note: str) -> str:
    """Off-budget check of the launch path on a gpu_short A40: preflight only, no model load.

    It never enters the ledger, and its stage name never satisfies a predecessor gate.
    """
    os.umask(0o077)
    require(bool(owner_note.strip()), "record coordination with the current submissions owner")
    run_id = safe_name(run_id)
    bundle, manifest, site, study = frozen_bundle(bundle)
    executables = {name: tool(name) for name in ("sbatch", "uv", "apptainer")}
    out = new_attempt(study / "rehearsals" / run_id, {
        "schema": SCHEMA, "stage": "preflight", "gpu_model": "A40", "bundle_id": manifest["identity"],
        "run_id": run_id, "bundle": str(bundle), "predecessor": None, "predecessor_result_sha256": None,
        "owner_coordination": owner_note, "apptainer": executables["apptainer"]})
    args = sbatch(out, "gpu_short", "gpu:1", REHEARSAL_SECONDS, "op-lora-rehearse", executables)
    return record_job(out, subprocess.run(args, capture_output=True, text=True, check=False, timeout=60))


def container_args(request: dict, site: dict, out: Path, mnt: Path, *, module: str = "opacity.lora_pilot",
                   model_dir: Path | None = None, extra_env: dict | None = None) -> list[str]:
    """The defaults are the pilot's. The deception runner names its own module, binds the pilot
    bundle's frozen model (model_dir) instead of a copy, and adds its allocator setting."""
    bundle = Path(request["bundle"])
    model_dir = bundle / "model" if model_dir is None else model_dir
    binds = [(bundle, "/bundle", "ro"), (model_dir, "/model", "ro"), (mnt / "run", "/run", "rw"),
             (mnt / "tmp", "/tmp", "rw"), (mnt / "tmp", "/var/tmp", "rw"), (mnt / "shm", "/dev/shm", "rw")]
    # Read-only replacements for the files Apptainer would otherwise write on its session tmpfs.
    binds += [(out / "etc" / name, "/etc/" + name, "ro") for name in ("passwd", "group", "resolv.conf")]
    if request["predecessor"]:
        binds.append((Path(request["predecessor"]), "/previous", "ro"))
    args = [request["apptainer"], "exec", "--cleanenv", "--containall", "--no-eval", "--userns",
            "--net", "--network", "none", "--nv", "--no-mount", "hostfs,bind-paths,cwd",
            "--home", str(mnt / "home") + ":/home/pilot", "--pwd", "/bundle"]
    for source, destination, mode in binds:
        args.extend(["--bind", f"{source}:{destination}:{mode}"])
    # HOME comes from --home; Apptainer refuses to override it with --env.
    payload_env = {"PYTHONPATH": "/bundle/src", "PYTHONDONTWRITEBYTECODE": "1",
                   "UV_PYTHON": "/opt/venv/bin/python",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                   "HF_HOME": "/run/cache/hf", "XDG_CACHE_HOME": "/run/cache", "UV_CACHE_DIR": "/run/cache/uv",
                   "WANDB_MODE": "disabled", "TMPDIR": "/tmp", "TOKENIZERS_PARALLELISM": "false",
                   "SLURM_JOB_ID": os.environ["SLURM_JOB_ID"],
                   "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    require(bool(payload_env["CUDA_VISIBLE_DEVICES"]), "Slurm CUDA device selection missing")
    for name, value in (extra_env or {}).items():
        require(name not in payload_env and name != "HOME", f"extra environment may not replace {name}")
        payload_env[name] = value
    for name, value in payload_env.items():
        args.extend(["--env", f"{name}={value}"])
    return args + [site["runtime_image"], "/usr/local/bin/uv", "run", "--offline", "--no-project",
                   "/opt/venv/bin/python", "-m", module, request["stage"],
                   "--bundle", "/bundle", "--out", "/run"]


def private_etc(out: Path) -> None:
    """Read-only replacements for the files Apptainer would otherwise write on its session tmpfs."""
    etc = out / "etc"
    etc.mkdir()
    user = getpass.getuser()
    (etc / "passwd").write_text(f"root:x:0:0:root:/root:/bin/false\n{user}:x:{os.getuid()}:{os.getgid()}::/home/pilot:/bin/false\n")
    (etc / "group").write_text(f"root:x:0:\n{user}:x:{os.getgid()}:\n")
    shutil.copyfile("/etc/resolv.conf", etc / "resolv.conf")  # the real resolver, so the DNS probe tests the namespace


def launch(bundle: Path, out: Path) -> int:
    """Runs only inside the allocated batch job; unavailable controls refuse."""
    from .preflight import network_probe

    os.umask(0o077)
    require("SLURM_JOB_ID" in os.environ, "launch requires a batch allocation")
    out = out.resolve(strict=True)
    request = read_json(out / "invocation.json")
    manifest = bundle_manifest(bundle, verify=False)
    controls = {name: sha for name, sha in manifest["files"].items() if not name.startswith("model/")}
    require(tree_manifest_subset(bundle, controls) == controls and manifest["identity"] == request["bundle_id"],
            "launch bundle identity changed")
    site = read_json(bundle / "config.json")["site"]
    study = study_check(Path(site["study_root"]))
    require(out.parent in (study / "attempts", study / "rehearsals") and str(bundle) == request["bundle"],
            "wrong launch paths")
    require(digest(Path(site["runtime_image"])) == site["runtime_sha256"], "runtime image changed")
    # Positive control: the targets the container must fail to reach answer from this host.
    host_probes = {"ip": network_probe(site["egress_ip"], site["egress_port"]),
                   "dns": network_probe(site["egress_dns"], site["egress_port"], dns=True)}
    sentinel = out / "host-sentinel"
    sentinel.write_text("non-secret sentinel\n")
    private_etc(out)
    context = {"stage": request["stage"], "gpu_model": request["gpu_model"], "predecessor": request["predecessor"],
               "predecessor_result_sha256": request["predecessor_result_sha256"],
               "deadline": time.time() + (STAGE_SECONDS.get(request["stage"], REHEARSAL_SECONDS)) - 120,
               "host_netns": os.readlink("/proc/self/ns/net"), "host_probes": host_probes,
               "sentinel": str(sentinel), "job_id": os.environ["SLURM_JOB_ID"],
               "runtime_image_sha256": site["runtime_sha256"], "launch_id": uuid.uuid4().hex}
    mnt = out / "mnt"
    mount(out / "output.img", mnt, writable=True)
    try:
        fs = os.statvfs(mnt)
        require(fs.f_blocks * fs.f_frsize <= RECIPE["output_limit_bytes"], "attempt filesystem exceeds byte cap")
        write_json(mnt / "run" / "context.json", context)
        args = container_args(request, site, out, mnt)
        write_json(out / "launcher.json", {"argv": args, "context": context})
        # Do not inherit provider variables, arbitrary bind settings or shell startup files.
        env = {"PATH": SYSTEM_PATH, "HOME": str(out / "host-home"), "TMPDIR": str(out / "host-tmp"),
               "APPTAINER_CACHEDIR": str(out / "host-cache")}
        result = subprocess.run(args, env=env, check=False)
    finally:
        unmount(mnt)
    evidence = extract(out)
    if result.returncode != 0 or not (evidence / "result.json").exists():
        write_json(out / "launcher-failure.json", {"returncode": result.returncode,
                                                  "result_present": (evidence / "result.json").exists()})
        return result.returncode if result.returncode > 0 else 1
    require(read_json(evidence / "result.json").get("status") == "passed", "payload did not pass")
    return 0
