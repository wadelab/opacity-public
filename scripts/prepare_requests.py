"""Prepare the harvest requests for the preparer arm.

The preparer model, outside the secure environment, gets PREPARER_INSTRUCTION
and each task's request plus the structure of its pack (titles, column
headers, row labels; no values, no notes) and rewrites the request for the
analyst inside. One call per task. Output, a committed input:

    src/opacity/harvest/prepared/<slug>-<task_set>.yaml

with the preparer model string, UTC time, argv, the instruction's name and
SHA-256, the generation settings, and per task the view sent, every attempt
and the reply verbatim. An empty reply is retried up to twice; the attempts
are recorded. An existing file is never overwritten.

Each reply is appended to a checkpoint, runs/prepare/<slug>-<task_set>.jsonl,
as soon as it arrives, so a stopped run loses nothing: run the same command
again and it calls the model only for the tasks with no reply yet, refusing a
checkpoint made with another model, instruction or settings. The output file
is written only when every task has a reply; a task still empty after
MAX_ATTEMPTS is retried by the next run, and its attempts accumulate.

    set -a; source ~/.config/cline/api.env; set +a
    .venv/bin/python scripts/prepare_requests.py --task-set dev
    .venv/bin/python scripts/prepare_requests.py --task-set heldout

The preparer is GLM-5.3 flash on Cline credits (cline/z-ai/glm-5.3-flash;
Wade, 25 October 2026: flash variants for every hosted role). Claude never prepares (a
Claude model wrote the tasks) and the judge families (MiMo, DeepSeek) never
do. The fresh set needs --fresh-is-allowed, which is only for after the
configurations it confirms are committed.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
from pathlib import Path

import yaml
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

import opacity.harness.providers  # noqa: F401  registers the cline provider
from opacity import offpeak
from opacity.harvest.loader import load_tasks
from opacity.harvest.prepare import (
    dump_prepared,
    instruction_sha256,
    prepared_path,
    preparer_message,
    preparer_view,
    slug_for,
)

FORBIDDEN_FAMILIES = ("claude", "anthropic", "mimo", "deepseek")
MAX_ATTEMPTS = 3  # the first call and up to two retries on an empty reply
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "runs" / "prepare"  # append-only, gitignored


def read_checkpoint(path: Path, fixed: dict) -> tuple[list[dict], dict[str, list[dict]]]:
    """Earlier runs and their task records, oldest first; exits on a run made with other settings."""
    runs: list[dict] = []
    records: dict[str, list[dict]] = {}
    if not path.exists():
        return runs, records
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:  # a line cut short when a run was stopped; its task runs again
            print(f"{path}:{n}: unreadable line skipped", file=sys.stderr)
            continue
        if "run" in entry:
            if entry["run"]["fixed"] != fixed:
                sys.exit(f"{path} was made with other settings ({entry['run']['fixed']}); "
                         f"move it aside or match them")
            runs.append(entry["run"])
        else:
            records.setdefault(entry["task"]["id"], []).append(entry["task"])
    return runs, records


def merged(records: list[dict]) -> dict:
    """One task's record across runs: the latest reply, with every attempt of every run."""
    return {**records[-1], "attempts": [a for r in records for a in r["attempts"]]}


async def prepare_one(model, task, config: GenerateConfig, sem: asyncio.Semaphore, record=None) -> dict:
    attempts: list[dict] = []
    reply = ""
    async with sem:
        for _ in range(MAX_ATTEMPTS):
            rec = {"stop_reason": "", "chars": 0, "output_tokens": None, "error": ""}
            try:
                out = await model.generate([ChatMessageUser(content=preparer_message(task))], config=config)
                reply = out.completion or ""
                rec.update(stop_reason=str(out.stop_reason), chars=len(reply),
                           output_tokens=out.usage.output_tokens if out.usage else None)
            except Exception as e:  # recorded, then retried like an empty reply
                reply = ""
                rec["error"] = f"{type(e).__name__}: {e}"[:500]
            attempts.append(rec)
            if reply.strip():
                break
    print(f"{task.id}: {len(attempts)} attempt(s), {len(reply)} chars", file=sys.stderr)
    rec = {"id": task.id, "attempts": attempts, "view": preparer_view(task), "reply": reply}
    if record:
        record({"task": rec})
    return rec


async def main_async(args) -> int:
    slug = args.slug or slug_for(args.model)
    if any(f in args.model.lower() for f in FORBIDDEN_FAMILIES):
        sys.exit(f"{args.model}: Claude wrote the tasks and MiMo and DeepSeek are the judges; none may prepare")
    if args.task_set == "fresh" and not args.fresh_is_allowed:
        sys.exit("the fresh set is the confirmation set: prepare it only after the configurations are committed "
                 "(--fresh-is-allowed)")
    out_path = Path(args.out) if args.out else prepared_path(slug, args.task_set)
    if out_path.exists():
        sys.exit(f"{out_path} exists; prepared files are committed inputs and are never regenerated in place")
    if not os.environ.get("CLINE_API_KEY") and os.environ.get("CLINE_PASS_KEY"):
        os.environ["CLINE_API_KEY"] = os.environ["CLINE_PASS_KEY"]

    tasks = load_tasks(task_set=args.task_set)
    if args.only:
        tasks = [t for t in tasks if t.id in set(args.only.split(","))]
    config = GenerateConfig(temperature=args.temperature, max_tokens=args.max_tokens)
    fixed = {
        "preparer_model": args.model,
        "slug": slug,
        "task_set": args.task_set,
        "instruction_sha256": instruction_sha256("PREPARER_INSTRUCTION"),
        "config": {"temperature": args.temperature, "max_tokens": args.max_tokens, "system_message": None},
        "max_attempts": MAX_ATTEMPTS,
    }
    ckpt = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_DIR / f"{out_path.stem}.jsonl"
    runs, done = read_checkpoint(ckpt, fixed)
    todo = [t for t in tasks if not any(r["reply"].strip() for r in done.get(t.id, []))]
    if runs:
        print(f"{ckpt}: {len(tasks) - len(todo)} of {len(tasks)} tasks already prepared", file=sys.stderr)
    run = {"utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "argv": list(sys.argv),
           "fixed": fixed}
    model = get_model(args.model)
    sem = asyncio.Semaphore(args.max_connections)
    if todo:
        offpeak.wait(lambda m: print(m, file=sys.stderr, flush=True))  # hosted calls off-peak only
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    with ckpt.open("a") as fh:
        def record(entry: dict) -> None:
            fh.write(json.dumps(entry) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

        record({"run": run})
        for rec in await asyncio.gather(*(prepare_one(model, t, config, sem, record) for t in todo)):
            done.setdefault(rec["id"], []).append(rec)
    records = [merged(done[t.id]) for t in tasks]
    empty = [r["id"] for r in records if not r["reply"].strip()]
    if empty:
        print(f"empty after {MAX_ATTEMPTS} attempts: {', '.join(empty)}; nothing written. "
              f"Run the same command again to retry them from {ckpt}", file=sys.stderr)
        return 1

    runs.append(run)
    doc = {
        "preparer_model": args.model,
        "slug": slug,
        "task_set": args.task_set,
        "generated_utc": runs[0]["utc"],
        "argv": runs[0]["argv"],
        "instruction": "PREPARER_INSTRUCTION",
        "instruction_sha256": fixed["instruction_sha256"],
        "config": fixed["config"],
        "max_attempts": MAX_ATTEMPTS,
        "tasks": records,
    }
    if len(runs) > 1:
        doc["resumed"] = [{"utc": r["utc"], "argv": r["argv"]} for r in runs[1:]]
    text = dump_prepared(doc)
    if yaml.safe_load(text) != doc:
        sys.exit("round trip through YAML changed the content; nothing written")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"-> {out_path} ({len(records)} tasks; retried: "
          f"{', '.join(r['id'] for r in records if len(r['attempts']) > 1) or 'none'}; "
          f"runs: {len(runs)})", file=sys.stderr)
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task-set", required=True, choices=("dev", "heldout", "fresh"))
    ap.add_argument("--model", default="cline/z-ai/glm-5.3-flash")
    ap.add_argument("--slug", default="", help="defaults to the last part of the model string")
    ap.add_argument("--only", default="", help="comma-separated task ids (a smoke test; use with --out)")
    ap.add_argument("--out", default="", help="write here instead of the package's prepared/ directory")
    ap.add_argument("--checkpoint", default="", help="defaults to runs/prepare/<output file stem>.jsonl")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--max-connections", type=int, default=8)
    ap.add_argument("--fresh-is-allowed", action="store_true")
    return ap


def main() -> None:
    sys.exit(asyncio.run(main_async(parser().parse_args())))


if __name__ == "__main__":
    main()
