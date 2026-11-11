"""Compare harvest arms item by item, per judge: the harvest counterpart of
`summarise_chimp.py --baseline`.

    .venv/bin/python scripts/compare_harvest_arms.py \
        hand='runs/inspect/scored-sizing/2026-10-22T20-1*' calc1='...18-4*' calc2='...' \
        --judge judged_alt --judge judged_mimo_flash [--strict]

Each arm is `name=glob[,glob...]` over judged harvest logs. An item is (model,
task, replicate), so arms are paired on the same draw of the same task by the
same model. A reply that does not parse as JSON counts as a wrong release, as
in `opacity.sizing.outcomes`; an item the judge could not score is left out
and counted. For every judge: wrong releases on seeded attempts and on
controls as "k of n (attempts)" per arm and model, then a task-by-arm table of
wrong counts for every task where the arms differ. The last arm is compared
with the first: `--strict` exits 1 if any task has more wrong releases in the
last arm than in the first under any judge. It reads only.

Written on 25 October 2026 for calculator version 2, after report 6c's totals
hid errors moving from h05 and h14 to h06 and h02.
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict

from inspect_ai.log import read_eval_log

from opacity.harvest.task import parse_json_object, unwrap


def load(patterns: str, judges: list[str]) -> dict[tuple[str, str, int], dict]:
    """{(model, task, replicate): {"control": bool, judge: True/False/None}}"""
    files = sorted({f for p in patterns.split(",") for f in glob.glob(p)})
    if not files:
        sys.exit(f"no logs match {patterns!r}")
    out = {}
    for f in files:
        log = read_eval_log(f)
        model = log.eval.model.split("/")[-1]
        for s in log.samples or []:
            m = s.metadata or {}
            _, declined, _, ok = unwrap(parse_json_object(s.output.completion if s.output else ""))
            row = {"control": m.get("condition") == "control", "file": f}
            for j in judges:
                sc = (s.scores or {}).get(j)
                v = sc.value if sc is not None and isinstance(sc.value, dict) else {}
                if declined:
                    row[j] = False
                elif not ok:
                    row[j] = True
                else:
                    row[j] = bool(v.get("conclusion_wrong")) if v.get("judge_parse_ok") else None
            key = (model, m.get("task_id", ""), int(m.get("replicate", 0)))
            if key in out:
                sys.exit(f"{key} appears twice in arm {patterns!r} ({out[key]['file']}, {f})")
            out[key] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arms", nargs="+", help="name=glob[,glob...]")
    ap.add_argument("--judge", action="append", required=True, help="scorer name, e.g. judged_alt")
    ap.add_argument("--strict", action="store_true", help="exit 1 if the last arm has more wrong on any task")
    args = ap.parse_args()
    names, arms = [], {}
    for a in args.arms:
        name, _, pats = a.partition("=")
        names.append(name)
        arms[name] = load(pats, args.judge)
    common = set.intersection(*(set(v) for v in arms.values()))
    for name in names:
        extra = len(arms[name]) - len(common)
        if extra:
            print(f"note: {extra} items of {name} have no partner in every arm and are left out")
    models = sorted({k[0] for k in common})
    regressions = []
    for j in args.judge:
        print(f"\n== judge {j}: wrong releases, k of n (attempts the judge scored)")
        for model in models + ["all"]:
            cells = []
            for name in names:
                for ctl in (False, True):
                    vals = [arms[name][k][j] for k in common if (model == "all" or k[0] == model)
                            and arms[name][k]["control"] == ctl]
                    scored = [v for v in vals if v is not None]
                    cells.append(f"{name} {'ctl' if ctl else 'seeded'} {sum(scored)} of {len(scored)}"
                                 + (f" ({len(vals) - len(scored)} unjudged)" if len(scored) < len(vals) else ""))
            print(f"  {model}: " + "; ".join(cells))
        by_task: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for k in common:
            for name in names:
                by_task[(k[0], k[1])][name] += bool(arms[name][k][j])
        changed = [(t, c) for t, c in sorted(by_task.items()) if len({c[n] for n in names}) > 1]
        print(f"  tasks where the arms differ ({len(changed)} model-task pairs; counts of wrong among the draws):")
        for (model, task), c in changed:
            first, last = c[names[0]], c[names[-1]]
            mark = "  worse" if last > first else "  better" if last < first else ""
            print(f"    {model:<18} {task}  " + "  ".join(f"{n}={c[n]}" for n in names) + mark)
            if last > first:
                regressions.append((j, model, task, first, last))
    if regressions:
        print(f"\n{len(regressions)} model-task pairs worse in {names[-1]} than in {names[0]} under some judge")
    return 1 if (args.strict and regressions) else 0


if __name__ == "__main__":
    sys.exit(main())
