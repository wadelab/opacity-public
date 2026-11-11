"""The deterministic verifier over judged harvest logs -> runs/summary/verify_<name>.csv.

    .venv/bin/python scripts/verify_harvest.py [log or glob ...] [--out runs/summary/verify.csv]

One row per write-up and rule: agent, arm, sample, condition, primary class,
the judge's verdict, and whether the rule applied and flagged. Tables are
read from the pack text stored in the sample, so a log is checked against
exactly what its analyst saw. No model is called.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

from inspect_ai.log import read_eval_log

from opacity.harvest.task import parse_json_object, unwrap
from opacity.verify import RULES, tables_from_markdown, verify

ROOT = Path(__file__).resolve().parents[1]


def rows_for(path: str) -> list[dict]:
    log = read_eval_log(path)
    if log.status != "success" or not log.samples:
        return []
    md = log.eval.metadata or {}
    agent = log.eval.model.split("/")[-1]
    out = []
    for s in log.samples:
        m = s.metadata or {}
        bundle, declined, _, ok = unwrap(parse_json_object(s.output.completion if s.output else ""))
        j = (s.scores or {}).get("judged")
        jv = j.value if j is not None and isinstance(j.value, dict) else {}
        findings = verify(tables_from_markdown(m.get("pack", "")), bundle if not declined else None)
        for f in findings:
            out.append({
                "log": Path(path).name, "agent": agent, "arm": md.get("arm", "open"),
                "think_budget": md.get("think_budget", 0), "checklist": md.get("checklist", False),
                "sample": s.id, "task_id": m.get("task_id"), "condition": m.get("condition"),
                "primary_class": m.get("primary_class"), "declined": int(bool(declined)), "agent_parse_ok": int(ok),
                "judge_wrong": jv.get("conclusion_wrong", ""), "judge_parse_ok": jv.get("judge_parse_ok", ""),
                "rule": f.rule, "applicable": int(f.applicable), "flagged": int(f.flagged), "detail": f.detail,
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", default=[str(ROOT / "runs/inspect/scored/*harvest*.eval")])
    ap.add_argument("--out", default=str(ROOT / "runs/summary/verify_harvest.csv"))
    a = ap.parse_args()
    paths = sorted({p for pat in a.logs for p in glob.glob(pat)})
    rows = [r for p in paths for r in rows_for(p)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["log"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(paths)} logs, {len(rows) // max(1, len(RULES))} write-ups -> {a.out}")


if __name__ == "__main__":
    main()
