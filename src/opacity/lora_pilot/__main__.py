"""Explicit commands only: importing this package never submits or trains."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import Refusal, digest, read_json, require, write_json


def check_cpu(out: Path, tests: tuple[str, ...] = ("tests/test_lora_pilot.py",)) -> int:
    """tests: the pilot's own file by default; the deception runner adds its file."""
    root = Path(__file__).resolve().parents[3]
    require(not subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                               capture_output=True, text=True).stdout, "commit source before generating a CPU receipt")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    out.mkdir(exist_ok=False, parents=True)
    # The source tree is imported from src/, not installed: an installed opacity
    # distribution would differ from the runtime lock's package set.
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "UV_PYTHON": sys.executable}
    with (out / "pytest.log").open("x") as log:
        result = subprocess.run(["uv", "run", "--offline", "--no-project", sys.executable,
                                 "-m", "pytest", *[str(root / name) for name in tests], "-q", "-p", "no:cacheprovider",
                                 "--junitxml=" + str(out / "junit.xml")], cwd=root, stdout=log,
                                stderr=subprocess.STDOUT, check=False, env=env)
    suites = ET.parse(out / "junit.xml").getroot().iter("testsuite")
    counts = {name: 0 for name in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for name in counts:
            counts[name] += int(suite.attrib.get(name, 0))
    unchanged = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                               capture_output=True, text=True).stdout == ""
    passed = result.returncode == 0 and counts["tests"] > 0 and not any(counts[k] for k in ("failures", "errors", "skipped")) and unchanged
    write_json(out / "result.json", {"source_commit": commit, "status": "passed" if passed else "failed",
                                     "counts": counts, "returncode": result.returncode,
                                     "artifacts": {"pytest.log": digest(out / "pytest.log"), "junit.xml": digest(out / "junit.xml")}})
    return 0 if passed else 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Honest-only Qwen3.5-27B LoRA engineering pilot")
    commands = ap.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("check-cpu", help="run CPU acceptance tests and create a receipt")
    cpu.add_argument("--out", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="freeze verified inputs; copies/hashes the local base")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--cpu-checks", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    submit = commands.add_parser("submit", help="submit exactly one stage, retaining its reservation")
    submit.add_argument("--bundle", type=Path, required=True)
    submit.add_argument("--stage", choices=("smoke", "train", "reload", "profile"), required=True)
    submit.add_argument("--run-id", required=True)
    submit.add_argument("--predecessor", type=Path)
    submit.add_argument("--owner-note", required=True)
    rehearse = commands.add_parser("rehearse", help="off-budget gpu_short check of the launch path; preflight only")
    rehearse.add_argument("--bundle", type=Path, required=True)
    rehearse.add_argument("--run-id", required=True)
    rehearse.add_argument("--owner-note", required=True)
    for name in ("launch", "preflight", "smoke", "train", "reload", "profile"):
        command = commands.add_parser(name, help="internal batch/runtime entry point")
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    worker = commands.add_parser("reload-worker", help=argparse.SUPPRESS)
    worker.add_argument("--kind", choices=("clean", "adapter"), required=True)
    worker.add_argument("--out", type=Path, required=True)
    report = commands.add_parser("report", help="reconcile evidence and exact Slurm job accounting")
    report.add_argument("--bundle", type=Path, required=True)
    report.add_argument("--attempt", type=Path, action="append", required=True)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--backup", action="store_true", help="copy once to the configured durable destination")
    return ap


def main(argv=None) -> int:
    os.umask(0o077)
    args = parser().parse_args(argv)

    def interrupted(signum, frame):
        raise RuntimeError(f"received signal {signum}; stage interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.command == "check-cpu":
            return check_cpu(args.out.resolve())
        if args.command == "prepare":
            from .preflight import prepare
            prepare(args.config, args.out.resolve(), args.cpu_checks.resolve())
        elif args.command == "submit":
            from .submit import submit
            print(submit(args.bundle, args.stage, args.run_id, args.predecessor, args.owner_note))
        elif args.command == "rehearse":
            from .submit import rehearse
            print(rehearse(args.bundle, args.run_id, args.owner_note))
        elif args.command == "launch":
            from .submit import launch
            return launch(args.bundle, args.out)
        elif args.command in ("preflight", "smoke", "train", "reload", "profile"):
            from .train import execute
            execute(args.command, args.bundle, args.out)
        elif args.command == "reload-worker":
            from .train import reload_worker
            require(args.out.parent == Path("/run"), "worker must run below /run")
            context = read_json(Path("/run/context.json"))
            gpu = read_json(Path("/run/preflight.json"))["hardware"]
            require(context["stage"] == "reload", "worker outside reload stage")
            reload_worker(args.kind, Path("/bundle"), args.out, Path("/previous"), context["deadline"], gpu)
        elif args.command == "report":
            from .report import report
            report(args.bundle, args.attempt, args.out.resolve(), make_backup=args.backup)
        return 0
    except (Refusal, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"pilot refused/failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
