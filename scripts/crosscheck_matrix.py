"""The agent-by-checker matrix from cross-check logs
-> runs/summary/crosscheck_matrix.csv and crosscheck_asymmetry.csv.

    .venv/bin/python scripts/crosscheck_matrix.py [runs/inspect/crosscheck] [runs/inspect/scored]

One cross-check log per checker model, each over every agent's harvested
write-ups. For each (checker, agent) cell: the number of write-ups the judge
called wrong and the checker's flag rate on them (sensitivity on natural
errors), the number the judge called right and the flag rate on them (false
alarm), the cannot-tell rate and the parse rate. The diagonal is a model
checking its own output.

Where a cross-check log has no judge verdicts (its harvest logs were
unscored), the second argument names a directory of scored harvest logs and
the verdicts are joined by agent and sample id.

The asymmetry table then gives, per checker, sensitivity and false alarm on
its own write-ups against everyone else's. A careless model is symmetric. A
model that flags in others what it does not flag in itself, or the reverse,
is not. Every number is recomputed from the per-sample rows.
"""

from __future__ import annotations

import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

from opacity.crosscheck.task import short_model

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "inspect" / "crosscheck"
scored = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "runs" / "inspect" / "scored"


def judge_verdicts(scored_dir: Path) -> dict[tuple[str, str], bool]:
    """(agent short name, harvest sample id) -> conclusion_wrong, from scored harvest logs."""
    out: dict[tuple[str, str], bool] = {}
    for f in sorted(glob.glob(str(scored_dir / "*_harvest*.eval"))):
        log = read_eval_log(f)
        if log.status != "success" or not log.samples:
            continue
        agent = short_model(log.eval.model)
        for s in log.samples:
            j = (s.scores or {}).get("judged")
            if j is not None and isinstance(j.value, dict) and j.value.get("judge_parse_ok") and not j.value.get("declined"):
                out[(agent, str(s.id))] = bool(j.value.get("conclusion_wrong"))
    return out


cells: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
joined = None
for f in sorted(glob.glob(str(src / "*_crosscheck_*.eval"))):
    log = read_eval_log(f)
    if log.status != "success" or not log.samples:
        continue
    checker = short_model(log.eval.model)
    for s in log.samples:
        m = s.metadata or {}
        sc = s.scores["crosscheck_verdict"]
        wrong = m.get("judge_wrong")
        if wrong is None:
            if joined is None:
                joined = judge_verdicts(scored)
            wrong = joined.get((m.get("agent_short", ""), str(m.get("harvest_sample_id"))))
        c = cells[(checker, m.get("agent_short", "?"))]
        c["n"] += 1
        c["parse_ok"] += sc.value["parse_ok"]
        c["cannot_tell"] += sc.value["cannot_tell"]
        if wrong is None:
            c["n_unknown"] += 1
            continue
        key = "wrong" if wrong else "right"
        c[f"n_{key}"] += 1
        c[f"flag_{key}"] += sc.value["flagged"]


def rate(k: float, n: float) -> str:
    return f"{k / n:.3f}" if n else ""


rows = []
for (checker, agent), c in sorted(cells.items()):
    rows.append(dict(checker=checker, agent=agent, self=int(checker == agent), n=int(c["n"]),
                     n_wrong=int(c["n_wrong"]), sensitivity=rate(c["flag_wrong"], c["n_wrong"]),
                     n_right=int(c["n_right"]), false_alarm=rate(c["flag_right"], c["n_right"]),
                     n_unknown=int(c["n_unknown"]), cannot_tell=rate(c["cannot_tell"], c["n"]), parse_ok=rate(c["parse_ok"], c["n"])))

asym = []
for checker in sorted({r["checker"] for r in rows}):
    own = defaultdict(float)
    other = defaultdict(float)
    for (ch, ag), c in cells.items():
        if ch != checker:
            continue
        tgt = own if ag == checker else other
        for k in ("n_wrong", "flag_wrong", "n_right", "flag_right"):
            tgt[k] += c[k]
    asym.append(dict(checker=checker,
                     sens_self=rate(own["flag_wrong"], own["n_wrong"]), sens_others=rate(other["flag_wrong"], other["n_wrong"]),
                     fa_self=rate(own["flag_right"], own["n_right"]), fa_others=rate(other["flag_right"], other["n_right"]),
                     n_wrong_self=int(own["n_wrong"]), n_wrong_others=int(other["n_wrong"]),
                     n_right_self=int(own["n_right"]), n_right_others=int(other["n_right"])))

out = ROOT / "runs" / "summary"
out.mkdir(parents=True, exist_ok=True)
if rows:
    agents = sorted({r["agent"] for r in rows})
    print("sensitivity on judge-wrong write-ups (rows: checker, columns: agent; * = self)")
    print(f"{'':<18}" + "".join(f"{a:>14}" for a in agents))
    for checker in sorted({r["checker"] for r in rows}):
        line = f"{checker:<18}"
        for a in agents:
            r = next((r for r in rows if r["checker"] == checker and r["agent"] == a), None)
            v = (r["sensitivity"] or "-") + ("*" if r and r["self"] else " ") if r else "."
            line += f"{v:>14}"
        print(line)
    print("\nfalse alarm on judge-right write-ups")
    print(f"{'':<18}" + "".join(f"{a:>14}" for a in agents))
    for checker in sorted({r["checker"] for r in rows}):
        line = f"{checker:<18}"
        for a in agents:
            r = next((r for r in rows if r["checker"] == checker and r["agent"] == a), None)
            v = (r["false_alarm"] or "-") + ("*" if r and r["self"] else " ") if r else "."
            line += f"{v:>14}"
        print(line)
    print("\nasymmetry: self against others")
    for a in asym:
        print(f"{a['checker']:<18} sensitivity self={a['sens_self'] or '-'} others={a['sens_others'] or '-'}   "
              f"false alarm self={a['fa_self'] or '-'} others={a['fa_others'] or '-'}")
    for name, data in (("crosscheck_matrix.csv", rows), ("crosscheck_asymmetry.csv", asym)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
        print("->", out / name)
else:
    print("no crosscheck logs in", src)
