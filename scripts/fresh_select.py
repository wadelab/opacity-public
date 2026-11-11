"""The fresh-set confirmation of the model-sizing study, as a rule written on
22 October 2026 at 20:35 UTC, before the dev results of the rules-with-think
and revise arms, before any held-out result of the rules and revise arms, and
before any Nemotron or Gemma 4 result.

    .venv/bin/python scripts/fresh_select.py            # choose, from dev + held-out
    .venv/bin/python scripts/fresh_select.py --results  # score the choices on the fresh set

Selection pools the dev (h01-h36) and held-out (h37-h72) tasks: 192 seeded and
24 control attempts per configuration, because dev alone picked badly (report
Table 8). For each analyst model that has the arm on both sets:

1. Among its prompt arms (ARMS below), keep those answering at least 0.75 of
   pooled controls; choose the one with the lowest pooled wrong-release rate
   on seeded tasks, ties broken by pooled correct releases, then cost.
2. For that arm, among checkers with verdicts on both its dev and held-out
   write-ups, choose the one that gives the lowest pooled wrong-release rate
   while still answering at least 0.75 of pooled controls; if none improves
   on the arm alone, no checker. The deterministic verifier is not used: it
   did not carry over to held-out (report section 5).
3. Write runs/summary/sizing/fresh_selection.csv. Copy it into the report and
   commit it before any fresh-set log is read.

A configuration confirms on the fresh set (h73-h108) if it releases wrong
conclusions on at most 0.05 of seeded tasks and answers at least 0.75 of
controls. Its interval, the second judge's verdicts (reported apart for
Nemotron, distilled in part from DeepSeek-R1) and its cost are reported
beside it, as is the same model's open arm.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inspect_ai.log import read_eval_log  # noqa: E402

from opacity import sizing as S  # noqa: E402

SUMMARY = ROOT / "runs" / "summary" / "sizing"
SELECTION = SUMMARY / "fresh_selection.csv"
ARMS = ("open|off", "open|off|checklist", "open|off|rules", "revise|off|checklist", "revise|off|rules",
        "open|think1024|rules", "open|think1024", "selfcheck|off",
        "open|effort-low", "open|effort-medium", "closed|effort-medium")
ANSWER_MIN, WRONG_MAX = 0.75, 0.05


def arm_of(cfg: str) -> tuple[str, str, str]:
    """'Qwen3.5-27B|open|off|rules|heldout|v2' -> ('Qwen3.5-27B', 'open|off|rules', 'heldout')."""
    p = cfg.split("|")
    task_set = "heldout" if "heldout" in p else "fresh" if "fresh" in p else "dev"
    core = [x for x in p[1:] if x not in ("heldout", "fresh", "v2", "v1")]
    return p[0], "|".join(core), task_set


def load_items(task_sets: tuple[str, ...]) -> dict[tuple[str, str], list]:
    by: dict[tuple[str, str], dict[str, list]] = defaultdict(dict)
    for p in S.glob_logs(str(ROOT / "runs/inspect/scored-sizing/*_harvest*.eval")):
        try:
            log = read_eval_log(p)
        except Exception:
            continue
        if not log.samples or any("judged" not in (s.scores or {}) for s in log.samples):
            continue
        items = S.load_analyst(p, keep_text=False)
        if not items or not items[0].config.endswith("|v2"):
            continue
        model, arm, ts = arm_of(items[0].config)
        if ts in task_sets:
            by[(model, arm)].setdefault(ts, []).extend(items)
    return by


def crosschecks() -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(dict)
    for p in S.glob_logs(str(ROOT / "runs/inspect/viking/crosscheck/*crosscheck*.eval")):
        key, v = S.load_crosscheck(p)
        out[key].update(v)
    return out


def select() -> int:
    by = load_items(("dev", "heldout"))
    cks = crosschecks()
    rows = []
    for model in sorted({m for m, _ in by}):
        cands = []
        for arm in ARMS:
            sets = by.get((model, arm), {})
            if "dev" not in sets or "heldout" not in sets:
                continue
            items = sets["dev"] + sets["heldout"]
            pt = S.evaluate(items, boot=0)["point"]
            if pt["answer_control"] >= ANSWER_MIN:
                cands.append((pt["wrong_release_seeded"], -pt["coverage_all"], arm, items, pt))
        if not cands:
            continue
        cands.sort(key=lambda c: (c[0], c[1]))
        wrong, _, arm, items, pt = cands[0]
        best = (wrong, None, pt)
        for key, v in cks.items():
            if not all(any(it.key in v for it in by[(model, arm)][ts]) for ts in ("dev", "heldout")):
                continue
            p2 = S.evaluate(items, [S.checker_stage(v)], boot=0)["point"]
            if p2["answer_control"] >= ANSWER_MIN and p2["wrong_release_seeded"] < best[0]:
                best = (p2["wrong_release_seeded"], key, p2)
        rows.append({"model": model, "arm": arm, "checker": best[1] or "none",
                     "pooled_wrong_alone": round(wrong, 3), "pooled_wrong_with_checker": round(best[0], 3),
                     "pooled_answer_control": round(best[2]["answer_control"], 3),
                     "pooled_coverage": round(best[2]["coverage_all"], 3), "n_items": len(items)})
    SUMMARY.mkdir(parents=True, exist_ok=True)
    with open(SELECTION, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["model"])
        w.writeheader()
        w.writerows(rows)
    print("| Analyst | Arm | Checker | Pooled wrong, alone | Pooled wrong, with checker | Pooled controls answered | Pooled correct released |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['arm']} | {r['checker']} | {r['pooled_wrong_alone']:.2f} | {r['pooled_wrong_with_checker']:.2f} | "
              f"{r['pooled_answer_control']:.2f} | {r['pooled_coverage']:.2f} |")
    return 0


def results() -> int:
    chosen = list(csv.DictReader(open(SELECTION)))
    by = load_items(("fresh",))
    cks = crosschecks()
    print("| Analyst | Arm | Checker | Judge | Wrong released, seeded (95% CI) | Controls answered | Correct released | Confirms |")
    print("|---|---|---|---|---|---|---|---|")
    for c in chosen:
        for arm, ck in ((c["arm"], c["checker"]), ("open|off", "none")):
            items = by.get((c["model"], arm), {}).get("fresh")
            if not items:
                print(f"| {c['model']} | {arm} | {ck} | – | no fresh log | | | |")
                continue
            stages = [] if ck == "none" else [S.checker_stage(cks.get(ck, {}))]
            if ck != "none" and not any(it.key in cks.get(ck, {}) for it in items):
                print(f"| {c['model']} | {arm} | {ck} | – | no fresh cross-check | | | |")
                continue
            for judge in ("first", "second"):
                if judge == "second" and not any(it.wrong2 is not None for it in items):
                    print(f"| {c['model']} | {arm} | {ck} | DeepSeek | second judge pending | | | |")
                    continue
                ev = S.evaluate(items, stages, boot=2000, judge=judge)
                pt, ci = ev["point"], ev["ci"]["wrong_release_seeded"]
                ok = pt["wrong_release_seeded"] <= WRONG_MAX and pt["answer_control"] >= ANSWER_MIN
                print(f"| {c['model']} | {arm} | {ck} | {'MiMo' if judge == 'first' else 'DeepSeek'} | "
                      f"{pt['wrong_release_seeded']:.2f} ({ci[0]:.2f}–{ci[1]:.2f}) | {pt['answer_control']:.2f} | "
                      f"{pt['coverage_all']:.2f} | {'yes' if ok else 'no'} |")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="store_true")
    sys.exit(results() if ap.parse_args().results else select())
