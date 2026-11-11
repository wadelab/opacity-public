"""The fresh-set result as one table: for each configuration tested on the
fresh set (h73-h108), its wrong-release rate on dev, held-out and fresh
seeded tasks, the fresh rate with the checker the rule chose, and the costs.

    .venv/bin/python scripts/fresh_results.py     # writes runs/summary/sizing/fresh_results.csv

The configurations are the rule's choices (fresh_selection.csv, committed in
bf7b878) and the addendum's gpt-oss 120B (4cc673e). Everything is recomputed
from per-sample verdicts with task-resampled 95% intervals; the judge is
MiMo (`judged`), as in the pre-registered test; the local gpt-oss 120B judge
(`alt_*` columns) and DeepSeek v4.1 flash (`ds_*`, 25 October) are beside
it. Report Table 10 and the figure sizing_fresh read this file.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import fresh_select as F  # noqa: E402

from opacity import sizing as S  # noqa: E402

OUT = F.SUMMARY / "fresh_results.csv"
ADDENDUM = ("gpt-oss-120b", "open|effort-high|rules|gpt-oss", "none")


def rate(items, stages=(), boot=2000):
    ev = S.evaluate(items, list(stages), boot=boot)
    p, ci = ev["point"], ev["ci"]["wrong_release_seeded"]
    outs = ev["outcomes"]
    ctl_wrong = sum(1 for it, o in zip(items, outs) if it.condition == "control" and o == "wrong")
    return p, ci, ctl_wrong


def main() -> int:
    chosen = [(c["model"], c["arm"], c["checker"]) for c in csv.DictReader(open(F.SELECTION))] + [ADDENDUM]
    by = F.load_items(("dev", "heldout", "fresh"))
    cks = F.crosschecks()
    rows = []
    for model, arm, ck in chosen:
        sets = by.get((model, arm), {})
        fresh = sets.get("fresh")
        if not fresh:
            print(f"no fresh log for {model} {arm}")
            continue
        alone, alone_ci, ctl_wrong = rate(fresh)
        row = {"model": model, "arm": arm, "checker": ck,
               "dev_wrong": S.evaluate(sets["dev"], boot=0)["point"]["wrong_release_seeded"] if "dev" in sets else "",
               "heldout_wrong": S.evaluate(sets["heldout"], boot=0)["point"]["wrong_release_seeded"] if "heldout" in sets else "",
               "fresh_wrong": alone["wrong_release_seeded"], "fresh_lo": alone_ci[0], "fresh_hi": alone_ci[1],
               "fresh_correct_released": alone["coverage_all"], "fresh_controls_answered": alone["answer_control"],
               "fresh_controls_judged_wrong": ctl_wrong, "n_seeded": alone["n_seeded"], "n_control": alone["n_control"]}
        if ck != "none":
            p, ci, _ = rate(fresh, [S.checker_stage(cks.get(ck, {}))])
            row.update({"checked_wrong": p["wrong_release_seeded"], "checked_lo": ci[0], "checked_hi": ci[1],
                        "checked_controls_answered": p["answer_control"], "checked_correct_released": p["coverage_all"],
                        "confirms": p["wrong_release_seeded"] <= F.WRONG_MAX and p["answer_control"] >= F.ANSWER_MIN})
        else:
            row.update({"checked_wrong": "", "checked_lo": "", "checked_hi": "", "checked_controls_answered": "",
                        "checked_correct_released": "",
                        "confirms": alone["wrong_release_seeded"] <= F.WRONG_MAX and alone["answer_control"] >= F.ANSWER_MIN})
        # judges beside MiMo: the local gpt-oss 120B (judged_alt, alt_*) and DeepSeek v4.1 flash
        # (judged_second, ds_*); blank until the judge has scored the log
        for judge, pre, attr in (("alt", "alt_", "wrong3"), ("second", "ds_", "wrong2")):
            cols = ("fresh_wrong", "fresh_lo", "fresh_hi", "controls_judged_wrong", "checked_wrong")
            if not any(getattr(it, attr) is not None for it in fresh):
                row.update({pre + c: "" for c in cols})
                continue
            pa = S.evaluate(fresh, boot=2000, judge=judge)
            row.update({pre + "fresh_wrong": pa["point"]["wrong_release_seeded"],
                        pre + "fresh_lo": pa["ci"]["wrong_release_seeded"][0],
                        pre + "fresh_hi": pa["ci"]["wrong_release_seeded"][1],
                        pre + "controls_judged_wrong": sum(1 for it, o in zip(fresh, pa["outcomes"])
                                                           if it.condition == "control" and o == "wrong")})
            row[pre + "checked_wrong"] = (S.evaluate(fresh, [S.checker_stage(cks.get(ck, {}))], boot=0, judge=judge)
                                          ["point"]["wrong_release_seeded"] if ck != "none" else "")
        base = by.get((model, "open|off"), {}).get("fresh")
        row["open_arm_fresh_wrong"] = S.evaluate(base, boot=0)["point"]["wrong_release_seeded"] if base else ""
        rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{OUT}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
