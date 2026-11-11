"""The held-out confirmation of the model-sizing study, as the rule in
docs/model-sizing-study.md section 5 states it.

    .venv/bin/python scripts/heldout_select.py            # step 2: choose on the dev tasks
    .venv/bin/python scripts/heldout_select.py --results  # step 3: the chosen configurations on held-out

Step 2 reads runs/summary/sizing/pipelines.csv (scripts/sizing_report.py),
refuses to choose while any held-out checker has no dev rows, and writes
runs/summary/sizing/heldout_selection.csv; copy that list into the report and
commit it before any held-out cross-check runs. Step 3 reads the list back,
rebuilds the held-out items from the judged logs and the held-out
cross-checks, and scores each chosen configuration against the gate under
both judges, with task-resampled intervals.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "runs" / "summary" / "sizing"
SELECTION = SUMMARY / "heldout_selection.csv"

# the analysts the held-out plans run (heldout-analyst.txt), as dev config keys
MODELS = ("Qwen3.5-9B", "Qwen3.5-27B", "Qwen3.5-35B-A3B")
ARMS = ("open|off", "open|off|checklist", "selfcheck|off", "open|think1024")
# the checkers the held-out cross-checks run: heldout-cross.txt (27B),
# heldout-cross-effort.txt (gpt-oss 20B), heldout-cross-nothink.txt (Llama
# 70B) and the typed job; amended 2026-10-22 16:15 UTC
CHECKERS = ("Qwen3.5-27B|off", "Qwen3.5-27B|off|stepwise", "Qwen3.5-27B|think1024", "Qwen3.5-27B|think1024|pack",
            "gpt-oss-20b|effort-low", "gpt-oss-20b|effort-medium", "typed|think512",
            "Llama-3.3-70B-Instruct-NVFP4|default|pack")
WRONG_MAX, ANSWER_MIN = 0.05, 0.75


def heldout_key(dev: str) -> str:
    """'Qwen3.5-27B|open|off|checklist|v2' -> '...|checklist|heldout|v2'."""
    head, pack = dev.rsplit("|", 1)
    return f"{head}|heldout|{pack}"


def candidates() -> dict[str, list[str]]:
    out = {}
    for m in MODELS:
        for a in ARMS:
            out[f"{m}|{a}|v2"] = ["alone", "verifier"] + [f"verifier+checker:{c}" for c in CHECKERS]
    return out


def select() -> int:
    rows = {(r["analyst"], r["pipeline"]): r for r in csv.DictReader(open(SUMMARY / "pipelines.csv"))}
    have = {p.split(":", 1)[1] for (a, p) in rows if a.endswith("|v2") and p.startswith("verifier+checker:")
            and p.count(":") == 1}
    missing = [c for c in CHECKERS if c not in have]
    if missing:
        print("not yet: no dev rows for", ", ".join(missing))
        return 1
    f = lambda r, k: float(r[k])  # noqa: E731
    chosen = []
    for m in MODELS:
        pool = []
        for an, pipes in candidates().items():
            if not an.startswith(m + "|"):
                continue
            for p in pipes:
                r = rows.get((an, p))
                if r is None:
                    print(f"missing dev row: {an} {p}")
                    continue
                if f(r, "answer_control") >= ANSWER_MIN:
                    pool.append(r)
        if not pool:
            print(f"{m}: no candidate answers {ANSWER_MIN} of controls")
            continue
        best = min(pool, key=lambda r: (f(r, "wrong_release_seeded"), -f(r, "coverage_all"), f(r, "gpu_s_per_task")))
        for role, p in (("chosen", best["pipeline"]), ("baseline", "alone"), ("baseline", "verifier")):
            if role == "baseline" and p == best["pipeline"]:
                continue
            r = rows[(best["analyst"], p)]
            chosen.append({"model": m, "role": role, "analyst": best["analyst"], "pipeline": p,
                           "dev_wrong": f(r, "wrong_release_seeded"), "dev_lo": f(r, "lo"), "dev_hi": f(r, "hi"),
                           "dev_answer_control": f(r, "answer_control"), "dev_coverage": f(r, "coverage_all"),
                           "dev_gpu_s": f(r, "gpu_s_per_task")})
    with open(SELECTION, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(chosen[0]))
        w.writeheader()
        w.writerows(chosen)
    print("| Analyst | Role | Checks | Dev wrong released (95% CI) | Dev answered, controls | Dev correct released | GPU-s per task |")
    print("|---|---|---|---|---|---|---|")
    for c in chosen:
        print(f"| {c['analyst'].removesuffix('|v2')} | {c['role']} | {c['pipeline']} | {c['dev_wrong']:.2f} "
              f"({c['dev_lo']:.2f}–{c['dev_hi']:.2f}) | {c['dev_answer_control']:.2f} | {c['dev_coverage']:.2f} | "
              f"{c['dev_gpu_s']:.1f} |")
    print(f"\n{len(chosen)} configurations -> {SELECTION}")
    return 0


def results() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import sizing_report as R  # noqa: E402
    from opacity import sizing as S  # noqa: E402
    from opacity.verify import tables_from_markdown, verify  # noqa: E402

    chosen = list(csv.DictReader(open(SELECTION)))
    _, items_by = R.analyst_rows(R.harvest_logs())
    checkers: dict[str, dict] = {}
    for p in R.crosscheck_logs():
        key, v = S.load_crosscheck(p)
        if v:
            checkers.setdefault(key, {}).update(v)
    print("| Analyst | Role | Checks | Judge | Wrong released, seeded (95% CI) | Answered, controls | Correct released | Confirms |")
    print("|---|---|---|---|---|---|---|---|")
    for c in chosen:
        hk = heldout_key(c["analyst"])
        items = items_by.get(hk)
        if not items:
            print(f"| {hk} | {c['role']} | {c['pipeline']} | – | no held-out analyst log | | | |")
            continue
        flags = {it.key: any(f.flagged for f in verify(tables_from_markdown(it.pack), it.bundle)) for it in items}
        stages = []
        for st in c["pipeline"].split("+"):
            if st == "verifier":
                stages.append(S.verifier_stage(flags))
            elif st.startswith("checker:"):
                ck = st.split(":", 1)[1]
                v = checkers.get(ck, {})
                if not any(it.key in v for it in items):
                    stages = None
                    break
                stages.append(S.checker_stage(v))
        if stages is None:
            print(f"| {hk.removesuffix('|v2')} | {c['role']} | {c['pipeline']} | – | no held-out cross-check | | | |")
            continue
        for judge in ("first", "second"):
            if judge == "second" and not any(it.wrong2 is not None for it in items):
                print(f"| {hk.removesuffix('|v2')} | {c['role']} | {c['pipeline']} | DeepSeek | second judge pending | | | |")
                continue
            ev = S.evaluate(items, stages, boot=2000, judge=judge)
            pt, ci = ev["point"], ev["ci"]
            ok = pt["wrong_release_seeded"] <= WRONG_MAX and pt["answer_control"] >= ANSWER_MIN
            print(f"| {hk.removesuffix('|v2')} | {c['role']} | {c['pipeline']} | {'MiMo' if judge == 'first' else 'DeepSeek'} | "
                  f"{pt['wrong_release_seeded']:.2f} ({ci['wrong_release_seeded'][0]:.2f}–{ci['wrong_release_seeded'][1]:.2f}) | "
                  f"{pt['answer_control']:.2f} | {pt['coverage_all']:.2f} | {'yes' if ok else 'no'} |")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="store_true", help="step 3: score the chosen configurations on held-out")
    sys.exit(results() if ap.parse_args().results else select())
