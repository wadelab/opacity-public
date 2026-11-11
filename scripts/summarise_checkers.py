"""Checker summaries from the Inspect logs -> runs/summary/checker_by_level.csv.

    .venv/bin/python scripts/summarise_checkers.py

Every rate is recomputed from the per-sample verdict scores and cross-checked
against Inspect's own by_condition_seeded aggregate; the expected sample count
(8 classes x 2 conditions x replicates x levels), unique sample ids and the
absence of sample errors are asserted. Runs are classified by tag prefix.
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[1]
RUNS = (("hosted-20261021", "hosted-old"), ("hosted2-20261021", "hosted-new"),
        ("sweep2-", "sweep2-old"), ("sweep3-", "sweep3-new"), ("sweep4-", "sweep4-new-nothink"), ("sweep5-", "sweep5-new-nothink"), ("rep20-", "rep20-new-nothink"), ("matched-", "matched-new-nothink"),
        ("typed", "typed-new"))


def classify(tags: list[str]) -> str:
    for t in tags:
        for pre, name in RUNS:
            if t.startswith(pre):
                return name
    return "other"


rows: list[dict] = []
problems: list[tuple] = []
files = sorted(glob.glob(str(ROOT / "runs/inspect/*opacity-*.eval")) + glob.glob(str(ROOT / "runs/inspect/viking/*opacity-llm*.eval")) + glob.glob(str(ROOT / "runs/inspect/viking/*opacity-typed*.eval")))
for f in files:
    log = read_eval_log(f)
    if log.status != "success":
        problems.append((f, "status", log.status)); continue
    run = classify(log.eval.tags or [])
    if run == "other":
        continue
    md = log.eval.metadata or {}
    model = log.eval.model.split("/")[-1]
    if md.get("arm") == "typed":
        model = "openjev:" + (md.get("typed") or {}).get("model", "")
    n, reps, lv = len(log.samples), md["replicates"], md["levels"]
    if n != 8 * 2 * reps * len(lv):
        problems.append((f, "sample count", n))
    if len({s.id for s in log.samples}) != n:
        problems.append((f, "duplicate ids"))
    if any(s.error for s in log.samples):
        problems.append((f, "sample errors"))
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0]))
    byc = defaultdict(lambda: [0, 0])
    for s in log.samples:
        v = s.scores["verdict"].value; m = s.metadata
        a = agg[m["disclosure_level"]][m["condition"]]
        a[0] += 1; a[1] += v["flagged"]; a[2] += v["correct"]; a[3] += v["cannot_tell"]; a[4] += v["parse_ok"]
        if m["condition"] == "seeded":
            k = (m["disclosure_level"], m["error_class"]); byc[k][0] += 1; byc[k][1] += v["flagged"]
    own = {name: mv.value for sc in (log.results.scores if log.results else []) for name, mv in sc.metrics.items()}
    key = next((k for k in own if k.endswith("by_condition_seeded")), None)
    tot = sum(agg[l]["seeded"][1] for l in agg) / sum(agg[l]["seeded"][0] for l in agg)
    if key and abs(own[key] - tot) > 1e-6:
        problems.append((f, "mismatch vs inspect", own[key], tot))
    for l in sorted(agg):
        s_, c_ = agg[l]["seeded"], agg[l]["control"]
        rows.append(dict(run=run, model=model, arm=md.get("arm"), thinking=md.get("thinking"), level=l,
                         n_seeded=s_[0], n_control=c_[0],
                         detection=round(s_[1] / s_[0], 3), false_alarm=round(c_[1] / c_[0], 3),
                         cannot_tell_seeded=round(s_[3] / s_[0], 3), cannot_tell_control=round(c_[3] / c_[0], 3),
                         parse_ok=round((s_[4] + c_[4]) / (s_[0] + c_[0]), 3),
                         **{f"det_c{c}": round(byc[(l, c)][1] / byc[(l, c)][0], 2) for c in range(1, 9) if byc[(l, c)][0]}))
    print(f"{run:<19} {model:<20} n={n:<4} inspect={own.get(key)} recomputed={tot:.3f}")

out = ROOT / "runs/summary/checker_by_level.csv"
out.parent.mkdir(parents=True, exist_ok=True)
cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("run", "model", "arm", "thinking", "level"), k))
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
print("rows:", len(rows), "->", out)
print("PROBLEMS:", problems or "none")
