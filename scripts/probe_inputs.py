"""Inputs for the probe experiment (docs/vetting.md, section 4).

    .venv/bin/python scripts/probe_inputs.py [--tag v1]      # about eight minutes
      -> runs/probe/<tag>/labels.csv            one row per dev or held-out write-up:
                                                task, analyst, the judges' verdicts
         runs/probe/<tag>/verbal.csv            each verbal checker's score and release
         runs/probe/<tag>/inputs/<probe>.jsonl  the messages each probe model reads,
                                                copied from its own checker runs

The write-ups and their truth are scripts/auc_summary.py's: dev is the 3,613
every checker saw (its `common`), held-out is every judged write-up that both
prompt sources covered (`opacity.probe.HELDOUT_COVER`). Three checks run
before anything is written: every write-up has exactly one prompt per probe;
the 27B's dev messages are identical in its thinking-off and 1k-think runs,
so held-out prompts taken from the 1k-think run are the ones a thinking-off
run would have sent; and each verbal checker's AUC on the dev write-ups,
recomputed here, equals runs/summary/sizing/auc/checkers.csv to three places.
A tag that exists is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import auc_summary as A  # noqa: E402

from opacity.probe import HELDOUT_COVER, PROBES, key_str  # noqa: E402

CROSSCHECK = ("runs/inspect/viking/crosscheck/*crosscheck*.eval", "runs/inspect/crosscheck/*crosscheck*.eval")


def prompts(sources: set[str], wanted: set[tuple[str, str]]) -> dict[str, dict]:
    """{checker: {(log, sample): {messages, logged_tokens, reply}}} for the wanted write-ups."""
    from inspect_ai.log import read_eval_log, read_eval_log_samples

    from opacity import sizing as S

    out: dict[str, dict] = {k: {} for k in sources}
    for p in S.glob_logs(*(str(ROOT / g) for g in CROSSCHECK)):
        h = read_eval_log(p, header_only=True)
        k = S.checker_key(h)
        if k not in sources or h.status != "success":
            continue
        for s in read_eval_log_samples(p):
            m = s.metadata or {}
            key = (Path(str(m.get("harvest_log", ""))).name, str(m.get("harvest_sample_id")))
            if key not in wanted:
                continue
            msgs = [{"role": x.role, "content": x.text} for x in s.messages if x.role in ("system", "user")]
            rec = {"messages": msgs, "logged_tokens": sum(u.input_tokens for u in (s.model_usage or {}).values()),
                   "reply": s.output.completion if s.output else ""}
            prev = out[k].get(key)
            if prev is not None and prev["messages"] != msgs:
                raise SystemExit(f"{k}: two different prompts for {key}")
            out[k].setdefault(key, rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    out = ROOT / "runs" / "probe" / a.tag
    if out.exists():
        sys.exit(f"{out} exists; runs are append-only, choose a new tag")

    items, checks = A.load()
    by_key = {(it["log"], it["sample"]): it for it in items}
    _, common = A.checker_table(items, checks)
    dev = set(common)
    if any(by_key[k]["task_set"] != "dev" for k in dev):
        sys.exit("the common write-ups are not all dev")
    held = {k for k, it in by_key.items() if it["task_set"] == "heldout" and A.truth(it) is not None
            and all(k in checks.get(c, {}) for c in HELDOUT_COVER)}
    sets = {"dev": dev, "heldout": held}
    for name, ks in sets.items():
        w = sum(A.truth(by_key[k]) for k in ks)
        print(f"{name}: {len(ks)} write-ups, {w} wrong, {len(ks) - w} right, "
              f"{len({by_key[k]['task_id'] for k in ks})} tasks")

    # each verbal checker's dev AUC against the published table
    published = {r["checker"]: float(r["common_auc"]) for r in csv.DictReader(open(A.OUT / "checkers.csv"))}
    verbal = sorted({c for p in PROBES.values() for c in p["verbal"]})
    for c in verbal:
        rows = [(by_key[k]["task_id"], A.truth(by_key[k]), A.score(checks[c][k])) for k in dev if k in checks.get(c, {})]
        got = A.auc([s for _, w, s in rows if w], [s for _, w, s in rows if not w])
        if c in published and round(got, 3) != round(published[c], 3):
            sys.exit(f"{c}: dev AUC {got:.4f} here, {published[c]:.4f} in checkers.csv")
        print(f"verbal {c}: dev AUC {got:.3f} on {len(rows)} (checkers.csv {published.get(c, 'absent')})")

    sources = {src for p in PROBES.values() for src in p["prompts"].values()} | {"Qwen3.5-27B|off|pack"}
    P = prompts(sources, dev | held)
    diff = [k for k in dev if P["Qwen3.5-27B|off|pack"][k]["messages"] != P["Qwen3.5-27B|think1024|pack"][k]["messages"]]
    if diff:
        sys.exit(f"{len(diff)} dev prompts differ between the 27B's off and 1k-think pack runs, e.g. {diff[0]}")
    print(f"27B pack prompts identical across thinking settings on all {len(dev)} dev write-ups")

    (out / "inputs").mkdir(parents=True)
    for probe, spec in PROBES.items():
        lines = []
        for name, ks in sets.items():
            src = P[spec["prompts"][name]]
            missing = [k for k in ks if k not in src]
            if missing:
                sys.exit(f"{probe}: {len(missing)} {name} write-ups have no prompt in {spec['prompts'][name]}")
            for k in sorted(ks):
                r = src[k]
                lines.append(json.dumps({"key": key_str(*k), "task_set": name, "messages": r["messages"],
                                         "logged_tokens": r["logged_tokens"],
                                         "logged_reply": r["reply"] if name in spec["first_token"] else None}))
        (out / "inputs" / f"{probe}.jsonl").write_text("\n".join(lines) + "\n")
        print(f"{probe}: {len(lines)} prompts")

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "task_set", "task_id", "analyst", "condition", "wrong", "wrong2"])
        for name, ks in sets.items():
            for k in sorted(ks):
                it = by_key[k]
                w2 = it.get("wrong2")
                w2 = True if (w2 is not None and not it["parse_ok"]) else w2
                w.writerow([key_str(*k), name, it["task_id"], f"{it['model']}|{it['arm']}", it["condition"],
                            int(A.truth(it)), "" if w2 is None else int(w2)])
    with open(out / "verbal.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "checker", "score", "clean_no"])
        for c in verbal:
            for k in sorted(dev | held):
                v = checks.get(c, {}).get(k)
                if v is not None:
                    w.writerow([key_str(*k), c, A.score(v), int(A.clean_no(v))])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
