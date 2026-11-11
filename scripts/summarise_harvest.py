"""Commit rates from judged harvest logs -> runs/summary/harvest_by_model.csv.

    .venv/bin/python scripts/summarise_harvest.py [runs/inspect/scored]

Per agent model, arm and preparer (`prepared`, empty unless the run used
-T prepared=): the proportion of seeded tasks inviting class c on which the
judge found class c committed (commit_c1..commit_c8, over the tasks
that invite that class), the proportion of clean controls on which the judge
found any class committed (control_any_committed), conclusion_wrong on seeded
and on controls, the decline rate on seeded and on controls (closed and
self-check arms; always 0 for the open arm), and the two parse rates. A
declined write-up counts as not wrong and not committed: nothing was
released. Every number is recomputed from the per-sample scores, never from
Inspect's aggregates.
"""

from __future__ import annotations

import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "inspect" / "scored"
rows = []
for f in sorted(glob.glob(str(src / "*_harvest*.eval"))):
    log = read_eval_log(f)
    if log.status != "success" or not log.samples or "judged" not in (log.samples[0].scores or {}):
        continue
    model = log.eval.model.split("/")[-1]
    arm = (log.eval.metadata or {}).get("arm", "open")
    prepared = (log.eval.metadata or {}).get("prepared", "")  # preparer slug, "" for the researcher's request
    tags = ",".join(log.eval.tags or [])
    by_class = defaultdict(lambda: [0, 0])
    ctl = [0, 0]
    cw_s = [0, 0]
    cw_c = [0, 0]
    dc_s = [0, 0]
    dc_c = [0, 0]
    ap = [0, 0]
    jp = [0, 0]
    for s in log.samples:
        v = s.scores["judged"].value
        m = s.metadata
        ap[0] += 1; ap[1] += v["agent_parse_ok"]
        jp[0] += 1; jp[1] += v["judge_parse_ok"]
        if not v["judge_parse_ok"]:
            continue  # an unjudged bundle is reported in judge_parse_ok, not as a verdict
        declined = v.get("declined", 0.0)
        if m["invites"]:
            c = m["primary_class"]
            by_class[c][0] += 1; by_class[c][1] += v["invited_committed"]
            cw_s[0] += 1; cw_s[1] += v["conclusion_wrong"]
            dc_s[0] += 1; dc_s[1] += declined
        else:
            ctl[0] += 1; ctl[1] += v["any_committed"]
            cw_c[0] += 1; cw_c[1] += v["conclusion_wrong"]
            dc_c[0] += 1; dc_c[1] += declined
    row = dict(model=model, arm=arm, prepared=prepared, tags=tags, n=len(log.samples), judge=log.eval.task_args.get("judge"),
               agent_parse_ok=round(ap[1] / ap[0], 3), judge_parse_ok=round(jp[1] / jp[0], 3),
               conclusion_wrong_seeded=round(cw_s[1] / cw_s[0], 3) if cw_s[0] else "",
               conclusion_wrong_control=round(cw_c[1] / cw_c[0], 3) if cw_c[0] else "",
               declined_seeded=round(dc_s[1] / dc_s[0], 3) if dc_s[0] else "",
               declined_control=round(dc_c[1] / dc_c[0], 3) if dc_c[0] else "",
               control_any_committed=round(ctl[1] / ctl[0], 3) if ctl[0] else "", n_control=ctl[0])
    for c in range(1, 9):
        n, k = by_class[c]
        row[f"commit_c{c}"] = round(k / n, 3) if n else ""
        row[f"n_c{c}"] = n
    rows.append(row)
    print(f"{model:<16} {arm:<9} {('prep-' + prepared) if prepared else ''} n={len(log.samples)} agent_parse={row['agent_parse_ok']} judge_parse={row['judge_parse_ok']} "
          f"wrong(seeded)={row['conclusion_wrong_seeded']} wrong(control)={row['conclusion_wrong_control']} "
          f"declined(seeded)={row['declined_seeded']} declined(control)={row['declined_control']} "
          f"commit by class: " + " ".join(f"c{c}={row[f'commit_c{c}']}" for c in range(1, 9)))

out = ROOT / "runs" / "summary" / "harvest_by_model.csv"
out.parent.mkdir(parents=True, exist_ok=True)
if rows:
    cols = list(rows[0].keys())
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("->", out)
else:
    print("no judged harvest logs in", src)
