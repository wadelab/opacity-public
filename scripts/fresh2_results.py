"""The second fresh set (h109-h148), scored by the rule pre-registered in the
report, section 6b (39c9183), written before the set existed:

    .venv/bin/python scripts/fresh2_results.py

Configurations: Qwen3.5 35B-A3B and 27B with the rules prompt and a 1k think,
(A) alone and (B) followed by gpt-oss 20B at medium effort as checker. A
configuration confirms if it releases wrong conclusions on at most 0.05 of
seeded attempts, answers at least 0.75 of control attempts and releases a
wrong conclusion on at most 0.10 of control attempts. Reported beside them,
not tested: each model's open arm and Qwen3.5 9B with the rules prompt and a
1k think. The judge is the validated local gpt-oss 120B (`judged_alt`).
DeepSeek v4.1 flash (`judged_second`, the hosted judge named in the 24 October
amendment) and MiMo (`judged`) are printed beside it where they have scored,
each with the verdict the rule gives under that judge; the pre-registered
verdict is the local judge's. Writes runs/summary/sizing/fresh2_results.csv.
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

OUT = F.SUMMARY / "fresh2_results.csv"
CHECKER = "gpt-oss-20b|effort-medium"
TESTED = [("Qwen3.5-35B-A3B", "open|think1024|rules"), ("Qwen3.5-27B", "open|think1024|rules")]
BESIDE = [("Qwen3.5-35B-A3B", "open|off"), ("Qwen3.5-27B", "open|off"), ("Qwen3.5-9B", "open|think1024|rules"),
          ("Qwen3.5-9B", "open|off")]
SEEDED_MAX, CONTROL_ANSWER_MIN, CONTROL_WRONG_MAX = 0.05, 0.75, 0.10


def score(items, stages, judge):
    if not any(S.verdict(it, judge) is not None for it in items):
        return None
    ev = S.evaluate(items, stages, boot=2000, judge=judge)
    p, outs = ev["point"], ev["outcomes"]
    ctl = [o for it, o in zip(items, outs) if it.condition == "control" and o != "unjudged"]
    ctl_wrong = sum(o == "wrong" for o in ctl) / len(ctl) if ctl else float("nan")
    return {"wrong": p["wrong_release_seeded"], "lo": ev["ci"]["wrong_release_seeded"][0],
            "hi": ev["ci"]["wrong_release_seeded"][1], "controls_answered": p["answer_control"],
            "controls_wrong": ctl_wrong, "correct_released": p["coverage_all"], "n_seeded": p["n_seeded"],
            "n_control": p["n_control"]}


def load_fresh2() -> dict[tuple[str, str], dict[str, list]]:
    """fresh_select.load_items reads dev, heldout and fresh only (its rule is
    frozen at c4ec376), so the fresh2 logs are read here."""
    from inspect_ai.log import read_eval_log
    by: dict[tuple[str, str], dict[str, list]] = {}
    for p in S.glob_logs(str(ROOT / "runs/inspect/scored-sizing/*_harvest*.eval")):
        try:
            h = read_eval_log(p, header_only=True)
        except Exception:
            continue
        if (h.eval.task_args or {}).get("task_set") != "fresh2" or h.status != "success":
            continue
        items = S.load_analyst(p, keep_text=False)
        if not items:
            continue
        parts = items[0].config.split("|")
        arm = "|".join(x for x in parts[1:] if x not in ("fresh2", "v2", "v1"))
        by.setdefault((parts[0], arm), {}).setdefault("fresh2", []).extend(items)
    return by


def main() -> int:
    by = load_fresh2()
    cks = F.crosschecks()
    rows = []
    print("| Analyst | Arm | Checker | Judge | Wrong released, seeded (95% CI) | Controls answered | "
          "Controls wrong | Correct released | Confirms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (model, arm), tested in [(c, True) for c in TESTED] + [(c, False) for c in BESIDE]:
        items = by.get((model, arm), {}).get("fresh2")
        if not items:
            print(f"| {model} | {arm} | | | no fresh2 log | | | | |")
            continue
        for ck in (["none", CHECKER] if tested else ["none"]):
            stages = [] if ck == "none" else [S.checker_stage(cks.get(ck, {}))]
            if ck != "none" and not any(it.key in cks.get(ck, {}) for it in items):
                print(f"| {model} | {arm} | {ck} | | no fresh2 cross-check | | | | |")
                continue
            for judge, name in (("alt", "gpt-oss 120B (local)"), ("second", "DeepSeek flash"), ("first", "MiMo")):
                r = score(items, stages, judge)
                if r is None:
                    if judge != "alt":
                        continue
                    print(f"| {model} | {arm} | {ck} | {name} | not judged | | | | |")
                    continue
                ok = (r["wrong"] <= SEEDED_MAX and r["controls_answered"] >= CONTROL_ANSWER_MIN
                      and r["controls_wrong"] <= CONTROL_WRONG_MAX)
                verdict = ("yes" if ok else "no") if tested else "–"
                print(f"| {model} | {arm} | {ck} | {name} | {r['wrong']:.2f} ({r['lo']:.2f}–{r['hi']:.2f}) | "
                      f"{r['controls_answered']:.2f} | {r['controls_wrong']:.2f} | {r['correct_released']:.2f} | {verdict} |")
                rows.append({"model": model, "arm": arm, "checker": ck, "judge": judge, "tested": tested,
                             **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()},
                             "confirms": verdict})
    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
