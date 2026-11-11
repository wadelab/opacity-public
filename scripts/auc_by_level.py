"""AUC of seeded against control bundles, per checker and disclosure level.

    .venv/bin/python scripts/auc_by_level.py   -> runs/summary/auc_by_level.csv

The score is the checker's own confidence folded into its verdict: for the
LLM arms 50 + confidence/2 for "yes", 50 - confidence/2 for "no", 50 for
cannot_tell; for the typed arm 100 x P(yes) from the decision server. AUC is
the Mann-Whitney statistic over the 40 seeded and 40 control bundles at a
level; intervals are 2000-sample bootstrap percentiles.
"""

from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[1]


def auc(pos, neg) -> float:
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def boot_ci(pos, neg, B=2000, seed=0):
    rng = np.random.default_rng(seed); pos = np.asarray(pos); neg = np.asarray(neg)
    vals = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(B)]
    return np.percentile(vals, [2.5, 97.5])


def score(sample) -> float:
    v = sample.scores["verdict"]; md = v.metadata or {}
    try:
        comp = json.loads(sample.output.completion)
        if "p_error_present" in comp:
            return 100 * float(comp["p_error_present"].get("yes", 0.5))
    except Exception:
        pass
    conf = float(md.get("confidence") or 0)
    return 50 + conf / 2 if v.answer == "yes" else 50 - conf / 2 if v.answer == "no" else 50.0


def classify(tags):
    for t in tags:
        for pre, name in (("hosted-20261021", "hosted-old"), ("hosted2-20261021", "hosted-new"), ("sweep2-", "sweep2-old"),
                          ("sweep3-", "sweep3-new"), ("sweep4-", "sweep4-new-nothink"), ("sweep5-", "sweep5-new-nothink"), ("rep20-", "rep20-new-nothink"), ("matched-", "matched-new-nothink"), ("typed", "typed-new")):
            if t.startswith(pre):
                return name
    return "other"


rows = []
for f in sorted(glob.glob(str(ROOT / "runs/inspect/*opacity-*.eval")) + glob.glob(str(ROOT / "runs/inspect/viking/*opacity-*.eval"))):
    log = read_eval_log(f)
    run = classify(log.eval.tags or []); md = log.eval.metadata or {}
    if run == "other" or log.status != "success" or md.get("arm") == "mechanical":
        continue
    model = "openjev:" + (md.get("typed") or {}).get("model", "") if md.get("arm") == "typed" else log.eval.model.split("/")[-1]
    by = defaultdict(lambda: {"seeded": [], "control": []})
    for s in log.samples:
        by[s.metadata["disclosure_level"]][s.metadata["condition"]].append(score(s))
    for l in sorted(by):
        a = auc(by[l]["seeded"], by[l]["control"]); lo, hi = boot_ci(by[l]["seeded"], by[l]["control"])
        rows.append(dict(run=run, model=model, level=l, n_seeded=len(by[l]["seeded"]), n_control=len(by[l]["control"]),
                         auc=round(a, 3), ci_lo=round(float(lo), 3), ci_hi=round(float(hi), 3)))
    print(f"{run:<19} {model:<40} " + "  ".join(f"s{r['level']} {r['auc']:.2f}" for r in rows if r["run"] == run and r["model"] == model))
out = ROOT / "runs/summary/auc_by_level.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("->", out)
