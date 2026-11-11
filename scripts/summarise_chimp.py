"""Summarise NIGHTPLAY question-bank runs of the inside analyst (Chimp).

    .venv/bin/python scripts/summarise_chimp.py runs/chimp/chimp-*/

Each argument is a run directory holding rep<k>/summary.json (from
scripts/chimp_e2e.sh or scripts/chimp_hosted.sh). Prints, per run: the model,
whether the design rules were on, the verdicts right and marks over all
repeats, and how many claims the dossier's own statistics recorded null or
untested; then a question-by-run table of verdicts right out of repeats.
With --baseline RUN, also lists every question on which a run got fewer
verdicts right (as a share of its repeats) than the baseline did, so a change
that moves the failure to another question cannot hide in the totals;
--strict exits 1 if there is any such regression.

Reads only; never writes into runs/.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def load(run: str) -> dict | None:
    reps = []
    for path in sorted(glob.glob(os.path.join(run, "rep*", "summary.json"))):
        with open(path) as fh:
            s = json.load(fh)
        downgraded = untested = 0
        for row in s["rows"]:
            d_path = os.path.join(os.path.dirname(path), f"{row['id']}.json")
            if os.path.exists(d_path):
                with open(d_path) as fh:
                    st = json.load(fh).get("statistics") or {}
                downgraded += len(st.get("downgraded_claims") or [])
                untested += len(st.get("untested_claims") or [])
        reps.append({"summary": s, "downgraded": downgraded, "untested": untested})
    if not reps:
        return None
    name = os.path.basename(os.path.normpath(run))
    first = reps[0]["summary"]
    return {"name": name, "model": first.get("model", "?"),
            "rules": first.get("analyst_rules"), "reps": reps}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--baseline", help="run directory to compare every run against")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any question regressed against --baseline")
    args = ap.parse_args()
    runs = [r for r in (load(p) for p in args.runs) if r]
    if not runs:
        print("no rep*/summary.json under the given directories", file=sys.stderr)
        return 1
    print("| Run | Model | Rules | Repeats | Verdicts right | Marks | Recorded null | Untested |")
    print("|---|---|---|---|---|---|---|---|")
    for r in runs:
        reps = r["reps"]
        v = sum(x["summary"]["verdict_agreement"] for x in reps)
        q = sum(x["summary"]["questions"] for x in reps)
        m = sum(x["summary"]["marks_ok"] for x in reps)
        mt = sum(x["summary"]["marks_total"] for x in reps)
        rules = {True: "on", False: "off", None: "?"}[r["rules"]]
        print(f"| {r['name']} | {r['model']} | {rules} | {len(reps)} | {v}/{q} | {m}/{mt} | "
              f"{sum(x['downgraded'] for x in reps)} | {sum(x['untested'] for x in reps)} |")
    ids = []
    for r in runs:
        for x in r["reps"]:
            for row in x["summary"]["rows"]:
                if row["id"] not in ids:
                    ids.append(row["id"])
    print()
    print("| Question | Expect | " + " | ".join(r["name"].replace("chimp-", "") for r in runs) + " |")
    print("|---|---|" + "---|" * len(runs))
    for qid in ids:
        expect, cells = "", []
        for r in runs:
            got = [row for x in r["reps"] for row in x["summary"]["rows"] if row["id"] == qid]
            expect = expect or (got[0]["expect"] if got else "")
            right = sum(row["verdict_ok"] for row in got)
            wrong = sorted({row["verdict"] for row in got if not row["verdict_ok"]})
            cells.append(f"{right}/{len(got)}" + (f" ({', '.join(wrong)})" if wrong else ""))
        print(f"| {qid} | {expect} | " + " | ".join(cells) + " |")
    if not args.baseline:
        return 0
    base = load(args.baseline)
    if base is None:
        print(f"no rep*/summary.json under {args.baseline}", file=sys.stderr)
        return 1

    def share(run, qid):
        rows = [row for x in run["reps"] for row in x["summary"]["rows"] if row["id"] == qid]
        return (sum(r["verdict_ok"] for r in rows) / len(rows)) if rows else None

    regressions = []
    for r in runs:
        if r["name"] == base["name"]:
            continue
        for qid in ids:
            b, c = share(base, qid), share(r, qid)
            if b is not None and c is not None and c < b:
                regressions.append((r["name"], qid, b, c))
    print()
    if regressions:
        print(f"Regressions against {base['name']}:")
        for name, qid, b, c in regressions:
            print(f"- {name}: {qid} {b:.0%} -> {c:.0%}")
    else:
        print(f"No question regressed against {base['name']}.")
    return 1 if (regressions and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
