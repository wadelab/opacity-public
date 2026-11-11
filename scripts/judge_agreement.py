"""Agreement between the two judges on the harvest logs -> runs/summary/judge_agreement.csv.

    .venv/bin/python scripts/judge_agreement.py [runs/inspect/scored]

Every scored harvest log that carries both `judged` (MiMo pro) and
`judged_second` (DeepSeek 4.1 flash) is read once. For each agent model and
arm the script reports, over the samples both judges parsed: the agreement
and Cohen's kappa on conclusion_wrong (seeded tasks), on invited_committed
(seeded) and on any_committed (controls); the second judge's own rates for
the three; and both judges' parse rates. Everything is recomputed from the
per-sample scores. A pooled row over all logs closes the table.
"""

from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "inspect" / "scored"


def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return ""
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else ""


def rate(pairs, idx):
    return round(sum(p[idx] for p in pairs) / len(pairs), 3) if pairs else ""


def agree(pairs):
    return round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else ""


rows, pooled = [], {"wrong": [], "invited": [], "control": []}
for f in sorted(glob.glob(str(src / "*_harvest*.eval"))):
    log = read_eval_log(f)
    if log.status != "success" or not log.samples:
        continue
    sc = log.samples[0].scores or {}
    if "judged" not in sc or "judged_second" not in sc:
        continue
    model = log.eval.model.split("/")[-1]
    arm = (log.eval.metadata or {}).get("arm", "open")
    coll = {"wrong": [], "invited": [], "control": []}
    p1 = p2 = both = 0
    for s in log.samples:
        a, b = s.scores["judged"].value, s.scores["judged_second"].value
        p1 += a["judge_parse_ok"]; p2 += b["judge_parse_ok"]
        if not (a["judge_parse_ok"] and b["judge_parse_ok"]):
            continue
        both += 1
        if s.metadata["invites"]:
            coll["wrong"].append((a["conclusion_wrong"], b["conclusion_wrong"]))
            coll["invited"].append((a["invited_committed"], b["invited_committed"]))
        else:
            coll["control"].append((a["any_committed"], b["any_committed"]))
    for k in coll:
        pooled[k].extend(coll[k])
    n = len(log.samples)
    row = dict(model=model, arm=arm, n=n, judge1=sc["judged"].metadata.get("judge_model", ""),
               judge2=sc["judged_second"].metadata.get("judge_model", ""),
               judge1_parse_ok=round(p1 / n, 3), judge2_parse_ok=round(p2 / n, 3), n_both_parsed=both,
               wrong_agree=agree(coll["wrong"]), wrong_kappa=kappa(coll["wrong"]),
               wrong_j1=rate(coll["wrong"], 0), wrong_j2=rate(coll["wrong"], 1),
               invited_agree=agree(coll["invited"]), invited_kappa=kappa(coll["invited"]),
               invited_j1=rate(coll["invited"], 0), invited_j2=rate(coll["invited"], 1),
               control_agree=agree(coll["control"]), control_kappa=kappa(coll["control"]),
               control_j1=rate(coll["control"], 0), control_j2=rate(coll["control"], 1))
    rows.append(row)
    print(f"{model:<44} {arm:<9} wrong: agree={row['wrong_agree']} k={row['wrong_kappa']} j1={row['wrong_j1']} j2={row['wrong_j2']} | "
          f"invited: agree={row['invited_agree']} k={row['invited_kappa']} j1={row['invited_j1']} j2={row['invited_j2']} | "
          f"control: agree={row['control_agree']} k={row['control_kappa']} j1={row['control_j1']} j2={row['control_j2']} | "
          f"parsed j1={row['judge1_parse_ok']} j2={row['judge2_parse_ok']}")

if rows:
    row = dict(model="pooled", arm="", n=sum(r["n"] for r in rows), judge1=rows[0]["judge1"], judge2=rows[0]["judge2"],
               judge1_parse_ok="", judge2_parse_ok="", n_both_parsed=sum(r["n_both_parsed"] for r in rows))
    for k, name in (("wrong", "wrong"), ("invited", "invited"), ("control", "control")):
        row[f"{name}_agree"] = agree(pooled[k]); row[f"{name}_kappa"] = kappa(pooled[k])
        row[f"{name}_j1"] = rate(pooled[k], 0); row[f"{name}_j2"] = rate(pooled[k], 1)
    rows.append(row)
    print(f"pooled over {row['n']} samples: wrong agree={row['wrong_agree']} k={row['wrong_kappa']}; invited agree={row['invited_agree']} k={row['invited_kappa']}; control agree={row['control_agree']} k={row['control_kappa']}")
    out = ROOT / "runs" / "summary" / "judge_agreement.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("->", out)
else:
    print("no doubly judged harvest logs in", src)
