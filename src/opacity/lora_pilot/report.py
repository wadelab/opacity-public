"""Read-only evidence reconciliation and a new, immutable engineering report."""
from __future__ import annotations

import shutil
import subprocess
import json
from pathlib import Path

from .common import Refusal, bundle_manifest, digest, predecessor, read_json, require, tree_manifest, write_json

TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL", "PREEMPTED"}


def optional_record(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        value = read_json(path)
        if isinstance(value, dict):
            return value
    except (json.JSONDecodeError, OSError):
        pass
    return {"status": "invalid_record", "path": path.name}


def accounting(job_id: str) -> dict:
    require(job_id.isdigit(), "invalid exact job ID")
    try:
        result = subprocess.run(["sacct", "-j", job_id, "-X", "--noheader", "--parsable2",
                                 "--format=JobIDRaw,State,ExitCode,ElapsedRaw"],
                                capture_output=True, text=True, check=True, timeout=20)
        matches = [line.split("|") for line in result.stdout.splitlines() if line.split("|")[0] == job_id]
        if len(matches) == 1:
            row = matches[0]
            # sacct writes "CANCELLED by <uid>"; the first word is the state.
            return {"job_id": job_id, "state": row[1].split()[0], "exit_code": row[2], "elapsed_seconds": int(row[3])}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return {"job_id": job_id, "state": "UNKNOWN", "exit_code": None}


def require_completed(attempt: Path) -> dict:
    job = read_json(attempt / "job.json")
    state = accounting(job["job_id"])
    require(state["state"] == "COMPLETED" and state["exit_code"] == "0:0",
            "predecessor scheduler completion not confirmed; do not submit next stage yet")
    return state


def backup(attempt: Path, bundle: Path, root: Path) -> dict:
    """Copy once to a pre-provisioned durable destination and verify every byte."""
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_relative_to(Path("/mnt/scratch")), "backup must be outside scratch")
    require(root.stat().st_mode & 0o077 == 0, "durable backup directory must be private")
    manifest = bundle_manifest(bundle, verify=False)
    destination = root / manifest["identity"] / attempt.name
    destination.parent.mkdir(exist_ok=True)
    # Copy only after scheduler completion, so Slurm output and temporary files are stable.
    before = evidence_manifest(attempt)
    destination.mkdir(exist_ok=False)
    for name in before:
        target = destination / name
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(attempt / name, target)
    after = tree_manifest(destination)
    require(before == after, "backup copy did not verify")
    # The 54 GB base is identified by manifest, not copied into every pilot backup.
    inputs = destination / "frozen-inputs"
    inputs.mkdir()
    for name in manifest["files"]:
        if name.startswith("model/"):
            continue
        target = inputs / name
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(bundle / name, target)
        require(digest(target) == manifest["files"][name], "frozen input backup mismatch")
    shutil.copyfile(bundle / "manifest.json", inputs / "manifest.json")
    return {"destination": str(destination), "verified_files": before,
            "base_weights": "identified by frozen manifest; retained separately", "verified": True}


def evidence_manifest(attempt: Path) -> dict:
    # The image is 8 GiB sparse; its run directory is the extracted evidence/.
    transient = {"output.img", "mnt", "ro-mnt", "evidence.partial", "host-home", "host-tmp", "host-cache"}
    result = {}
    for path in sorted(attempt.rglob("*")):
        relative = path.relative_to(attempt)
        if relative.parts[0] in transient or relative.parts[:2] == ("evidence", "cache"):
            continue
        require(not path.is_symlink(), "symlink in authoritative run evidence")
        if path.is_file():
            result[str(relative)] = digest(path)
    return result


def report(bundle: Path, attempts: list[Path], out: Path, *, make_backup: bool = False) -> dict:
    manifest = bundle_manifest(bundle, verify=False)
    site = read_json(bundle / "config.json")["site"]
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for attempt in attempts:
        invocation = read_json(attempt / "invocation.json")
        require(invocation["bundle_id"] == manifest["identity"], "different-bundle attempt in report")
        state = accounting(read_json(attempt / "job.json")["job_id"]) if (attempt / "job.json").exists() else {
            "state": "NOT_SUBMITTED_OR_UNKNOWN", "exit_code": None}
        evidence = attempt / "evidence"
        if not evidence.exists() and (attempt / "output.img").exists() and state["state"] in TERMINAL:
            # The job ended before its launcher could copy the evidence out (e.g. a timeout).
            from .submit import extract
            extract(attempt)
        terminal = optional_record(evidence / "result.json")
        passed = False
        verification_error = None
        if terminal and terminal.get("status") == "passed":
            try:
                predecessor(evidence, manifest["identity"], invocation["stage"])
                passed = state["state"] == "COMPLETED" and state["exit_code"] == "0:0"
            except (Refusal, OSError, ValueError) as error:
                verification_error = str(error)
        status = "passed" if passed else (terminal.get("status") if terminal else "interrupted_or_unstarted")
        if terminal and terminal.get("status") == "passed" and not passed:
            status = "evidence_verification_failed" if verification_error else "scheduler_unconfirmed_or_failed"
        receipt = None
        if make_backup:
            require(state["state"] in TERMINAL, "only terminal jobs can be backed up")
            receipt = backup(attempt, bundle, Path(site["backup_root"]))
        records.append({"attempt": str(attempt), "stage": invocation["stage"], "status": status,
                        "scheduler": state, "terminal": terminal, "backup": receipt,
                        "verification_error": verification_error, "checks": optional_record(evidence / "checks.json")})
    stages = {record["stage"] for record in records if record["status"] == "passed"}
    # Each required stage needs a passing record; a rehearsal ("preflight") in the same report is extra.
    result = {"bundle_id": manifest["identity"], "records": records,
              "engineering_checks_passed": {"smoke", "train", "reload"} <= stages,
              "durable_closeout_complete": bool(records) and all(r["backup"] for r in records)}
    write_json(out / "report.json", result)
    lines = ["# Honest-LoRA pilot evidence", "", "Generated from immutable records; no model work is performed by report.", "",
             "| Stage | Attempt | Evidence status | Slurm state |", "|---|---|---|---|"]
    for record in records:
        lines.append(f"| {record['stage']} | {Path(record['attempt']).name} | {record['status']} | {record['scheduler']['state']} |")
    for record in records:
        checks = record["checks"]
        if checks and {"clean", "adapter"} <= set(checks):
            lines += ["", f"## {record['stage']}: measured evidence", "",
                      f"Fresh clean-base process passed: {checks['clean'].get('passed')}; fresh adapter process "
                      f"(reload, then disabled) passed: {checks['adapter'].get('passed')}.",
                      f"Base digest identical in both: {checks['clean'].get('base_digest') == checks['adapter'].get('base_digest')}."]
        if checks and checks.get("stage") == "profile":
            lines += ["", f"## {record['stage']}: measured evidence", "",
                      f"Two honest updates, then stress cycles at {checks['lengths_requested']} tokens "
                      "(escalation stops at the first without headroom).", "",
                      "| Actual stress tokens | Peak reserved bytes | Device bytes | Headroom |", "|---|---|---|---|"]
            lines += [f"| {p['length']} | {p['peak_reserved_bytes']} | {p['device_total_bytes']} | {p['headroom']} |"
                      for p in checks["profiles"]]
        if not checks or "steps" not in checks or checks.get("stage") == "profile":
            continue
        lines += ["", f"## {record['stage']}: measured evidence", "",
                  f"Optimiser updates: {checks['updates']}; training presentations: {checks['presentations']}.",
                  f"Oracle correct: base {checks['base_diagnostic_correct']} of 16 diagnostic examples; "
                  f"adapted {checks['adapted_diagnostic_correct']} of 16 diagnostic examples.",
                  f"Newly incorrect diagnostic IDs: {checks['new_diagnostic_errors']}.",
                  f"Base unchanged: {checks['base_unchanged']}; effective adapter changed: "
                  f"{checks['adapter_contribution_changed']}.",
                  f"Steady non-padding tokens/s: {checks['throughput']['nonpadding_tokens_per_second']:.2f}; "
                  f"supervised tokens/s: {checks['throughput']['supervised_tokens_per_second']:.2f}.", "",
                  "| Actual stress tokens | Peak reserved bytes | Device bytes | Headroom |",
                  "|---|---|---|---|"]
        for profile in checks["profiles"]:
            lines.append(f"| {profile['length']} | {profile['peak_reserved_bytes']} | "
                         f"{profile['device_total_bytes']} | {profile['headroom']} |")
    lines += ["", f"All three engineering stages passed: {result['engineering_checks_passed']}.",
              f"Durable backup closeout complete: {result['durable_closeout_complete']}.", "",
              "See report.json and the source checks.json files for per-case diagnostic outcomes,",
              "memory bytes, timings, update counts, incomplete profiles and failure reasons.", ""]
    with (out / "report.md").open("x") as stream:
        stream.write("\n".join(lines))
    return result
