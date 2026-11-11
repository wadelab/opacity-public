"""Fit the reading probe on dev and test it once on held-out (docs/vetting.md, 4.2 to 4.4).

    .venv/bin/python -m opacity.probe.fit Qwen3.5-27B [--tag v1]
      -> runs/probe/<tag>/results/<probe>-heldout.json

Selection uses dev only. For each depth and each regularisation strength, a
logistic regression on standardised states is scored by its mean AUC over
folds that leave whole tasks out; the best pair is refitted on all of dev.
The operating threshold is also set on dev, from the out-of-fold scores: it
stops 20% of the right dev write-ups. Held-out is then scored once.

Guarantees in code: the held-out run refuses while the probe code has
uncommitted changes, and a second held-out run for the same probe refuses
unless --rerun-reason says why, writing a new file that records the reason.

Reported on held-out: the probe's AUC with a task-resampled interval; each
verbal configuration's AUC on the same write-ups; the stop rule (probe minus
the best verbal AUC at least MARGIN, paired task-resampled interval above
zero); the operating point; the share of wrong write-ups the best verbal
checker released that the probe catches; AUC within each analyst
configuration; prompt length as a surface baseline; the probe trained on
labels shuffled within each task; and the AUCs on the write-ups where the
second judge agrees with the first.

Added after the first held-out run (exploratory, 26 October 2026): the
shuffled-label probe reached 0.83 for the 27B, not the 0.5 the design
expected, so most of the pooled AUC is which tasks are risky. The AUC within
tasks (pairs of one wrong and one right write-up from the same task) removes
that, and is reported for the probe, each verbal configuration and the
shuffled probe, with the probe's paired difference from the best verbal one.
Each held-out write-up's scores are saved beside the result.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from opacity.probe import MARGIN, PRIMARY, PROBES

ROOT = Path(__file__).resolve().parents[3]
CS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)  # inverse L2 strength
FOLDS = 6
STOP_RIGHT = 0.20  # the operating point stops this share of right dev write-ups
BOOT = 2000
SHUFFLES = 20
CODE = ("src/opacity/probe", "scripts/probe_inputs.py", "scripts/probe_steer_inputs.py", "scripts/style_robustness.py")


def auc_module():
    """scripts/auc_summary.py, so that AUC and its interval are computed one way."""
    spec = importlib.util.spec_from_file_location("auc_summary", ROOT / "scripts" / "auc_summary.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = auc_module()


def model(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def folds(groups: np.ndarray, k: int = FOLDS) -> list[np.ndarray]:
    """Test indices of k folds that keep each task whole, tasks dealt in sorted order."""
    tasks = sorted(set(groups))
    return [np.flatnonzero(np.isin(groups, tasks[i::k])) for i in range(k)]


def oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float) -> tuple[np.ndarray, float]:
    """Out-of-fold scores and the mean of the per-fold AUCs."""
    s = np.zeros(len(y))
    aucs = []
    for te in folds(groups):
        tr = np.setdiff1d(np.arange(len(y)), te)
        s[te] = model(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(A.auc(s[te][y[te] == 1], s[te][y[te] == 0]))
    return s, float(np.nanmean(aucs))


def select(states: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[int, float, list[dict]]:
    """(depth index, C, table) with the best mean fold AUC; ties go to the stronger penalty."""
    table = []
    for d in range(states.shape[1]):
        for C in CS:
            table.append({"depth": d, "C": C, "cv_auc": oof(states[:, d].astype(np.float32), y, groups, C)[1]})
    best = max(table, key=lambda r: (r["cv_auc"], -r["C"]))
    return best["depth"], best["C"], table


def paired_diff(task: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """AUC(a) - AUC(b) on the same write-ups, with a 95% interval resampling tasks."""
    def d(idx):
        yy = y[idx]
        return A.auc(a[idx][yy == 1], a[idx][yy == 0]) - A.auc(b[idx][yy == 1], b[idx][yy == 0])

    by = defaultdict(list)
    for i, t in enumerate(task):
        by[t].append(i)
    tasks = sorted(by)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BOOT):
        idx = np.array([i for j in rng.choice(len(tasks), len(tasks)) for i in by[tasks[j]]])
        v = d(idx)
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return d(np.arange(len(y))), float(lo), float(hi)


def within_auc(task: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:
    """AUC over pairs of one wrong and one right write-up from the same task."""
    num = den = 0.0
    for t in dict.fromkeys(task.tolist()):
        i = task == t
        p, n = s[i][y[i] == 1], s[i][y[i] == 0]
        if len(p) and len(n):
            num += A.auc(p, n) * len(p) * len(n)
            den += len(p) * len(n)
    return num / den if den else float("nan")


def paired_within(task: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray,
                  seed: int = 0) -> tuple[float, float, float]:
    """Within-task AUC(a) - AUC(b), with a 95% interval resampling tasks; a task
    drawn twice counts as two strata."""
    by = defaultdict(list)
    for i, t in enumerate(task):
        by[t].append(i)
    tasks = sorted(by)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BOOT):
        draw = rng.choice(len(tasks), len(tasks))
        idx = np.concatenate([by[tasks[j]] for j in draw])
        strata = np.concatenate([[n] * len(by[tasks[j]]) for n, j in enumerate(draw)])
        v = within_auc(strata, y[idx], a[idx]) - within_auc(strata, y[idx], b[idx])
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return within_auc(task, y, a) - within_auc(task, y, b), float(lo), float(hi)


def shuffle_within(y: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    out = y.copy()
    for t in set(groups):
        i = np.flatnonzero(groups == t)
        out[i] = rng.permutation(y[i])
    return out


def ci(task, y, s) -> dict:
    a, lo, hi = A.auc_ci(list(zip(task, y.astype(bool), s)), boot=BOOT)
    return {"auc": a, "lo": lo, "hi": hi}


def dirty() -> list[str]:
    r = subprocess.run(["git", "status", "--porcelain", "--", *CODE], cwd=ROOT, capture_output=True, text=True)
    return [x for x in r.stdout.splitlines() if x.strip()]


def results_path(res: Path, probe: str, reason: str | None) -> Path:
    """The once-only rule: the first held-out run, or a numbered rerun with a reason."""
    prior = sorted(res.glob(f"{probe}-heldout*.json"))
    if not prior:
        return res / f"{probe}-heldout.json"
    if not reason:
        raise SystemExit(f"held-out already scored for {probe} ({prior[0].name}); "
                         "a rerun needs --rerun-reason and is reported as one")
    return res / f"{probe}-heldout-{len(prior) + 1}.json"


def load(tag_dir: Path, probe: str, states_path: Path):
    labels = {r["key"]: r for r in csv.DictReader(open(tag_dir / "labels.csv"))}
    verbal: dict[str, dict] = defaultdict(dict)
    for r in csv.DictReader(open(tag_dir / "verbal.csv")):
        verbal[r["checker"]][r["key"]] = (float(r["score"]), r["clean_no"] == "1")
    z = np.load(states_path, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    missing = [k for k in keys if k not in labels]
    if missing:
        raise SystemExit(f"{len(missing)} states have no label, e.g. {missing[0]}")
    meta = json.loads(str(z["meta"]))
    if meta.get("probe") != probe:
        raise SystemExit(f"{states_path} holds {meta.get('probe')}, not {probe}")
    if int((z["check_tokens"] != z["logged_tokens"]).sum()):
        raise SystemExit("prompt lengths differ from the logged ones; the prompts were not rebuilt exactly")
    return labels, verbal, keys, z, meta


def run(probe: str, tag_dir: Path, states_path: Path, reason: str | None = None, check_git: bool = True) -> dict:
    if check_git and dirty():
        raise SystemExit("probe code has uncommitted changes; commit before the held-out test:\n" + "\n".join(dirty()))
    res = tag_dir / "results"
    out = results_path(res, probe, reason)
    labels, verbal, keys, z, meta = load(tag_dir, probe, states_path)
    ts = np.array([labels[k]["task_set"] for k in keys])
    y = np.array([int(labels[k]["wrong"]) for k in keys])
    task = np.array([labels[k]["task_id"] for k in keys])
    dev, held = ts == "dev", ts == "heldout"
    S = z["states"]

    depth, C, table = select(S[dev], y[dev], task[dev])
    Xd, Xh = S[dev][:, depth].astype(np.float32), S[held][:, depth].astype(np.float32)
    s_oof, _ = oof(Xd, y[dev], task[dev], C)
    thr = float(np.quantile(s_oof[y[dev] == 0], 1 - STOP_RIGHT))
    clf = model(C).fit(Xd, y[dev])
    sh = clf.predict_proba(Xh)[:, 1]
    yh, th = y[held], task[held]
    hkeys = [k for k, h in zip(keys, held) if h]

    r: dict = {"probe": probe, "primary": probe == PRIMARY, "margin": MARGIN, "states": str(states_path),
               "extract": meta, "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
               "rerun_reason": reason, "n_dev": int(dev.sum()), "n_dev_wrong": int(y[dev].sum()),
               "n_heldout": int(held.sum()), "n_heldout_wrong": int(yh.sum()), "heldout_tasks": len(set(th)),
               "selection": {"depth": depth, "depth_layer": meta["depths"][depth], "C": C, "table": table},
               "threshold": thr, "probe_heldout": ci(th, yh, sh)}

    vb = {}
    for c in PROBES[probe]["verbal"]:
        if all(k in verbal.get(c, {}) for k in hkeys):
            sv = np.array([verbal[c][k][0] for k in hkeys])
            vb[c] = {**ci(th, yh, sv), "scores": sv, "released": np.array([verbal[c][k][1] for k in hkeys])}
    r["verbal_heldout"] = {c: {k: v[k] for k in ("auc", "lo", "hi")} for c, v in vb.items()}
    if vb:
        best = max(vb, key=lambda c: vb[c]["auc"])
        d, lo, hi = paired_diff(th, yh, sh, vb[best]["scores"])
        r["best_verbal"] = best
        r["difference"] = {"auc": d, "lo": lo, "hi": hi}
        r["stop_rule_met"] = bool(d >= MARGIN and lo > 0)
        missed = (yh == 1) & vb[best]["released"]
        r["verbal_released_wrong"] = int(missed.sum())
        r["probe_catches_of_those"] = int((sh[missed] >= thr).sum())

    r["operating_point"] = {"caught_wrong": float((sh[yh == 1] >= thr).mean()),
                            "stopped_right": float((sh[yh == 0] >= thr).mean())}
    ntok = z["n_tokens"].astype(float)
    sign = 1.0 if A.auc(ntok[dev][y[dev] == 1], ntok[dev][y[dev] == 0]) >= 0.5 else -1.0
    r["length_heldout"] = ci(th, yh, sign * ntok[held])
    analyst = np.array([labels[k]["analyst"] for k in hkeys])
    within = {}
    for an in sorted(set(analyst)):
        i = analyst == an
        if yh[i].sum() >= 5 and (1 - yh[i]).sum() >= 5:
            within[an] = A.auc(sh[i][yh[i] == 1], sh[i][yh[i] == 0])
    r["within_analyst"] = {"aucs": within, "mean": float(np.mean(list(within.values()))) if within else None}
    rng = np.random.default_rng(0)
    shuf, shuf_within = [], []
    for _ in range(SHUFFLES):
        ys = shuffle_within(y[dev], task[dev], rng)
        p = model(C).fit(Xd, ys).predict_proba(Xh)[:, 1]
        shuf.append(A.auc(p[yh == 1], p[yh == 0]))
        shuf_within.append(within_auc(th, yh, p))
    r["shuffled_heldout"] = {"mean": float(np.mean(shuf)), "min": float(np.min(shuf)), "max": float(np.max(shuf))}
    r["within_task"] = {"tasks": sum(0 < yh[th == t].sum() < (th == t).sum() for t in set(th)),
                        "probe": within_auc(th, yh, sh),
                        "verbal": {c: within_auc(th, yh, v["scores"]) for c, v in vb.items()},
                        "shuffled_mean": float(np.mean(shuf_within))}
    if vb:
        d, lo, hi = paired_within(th, yh, sh, vb[best]["scores"])
        r["within_task"]["difference_vs_best_verbal"] = {"auc": d, "lo": lo, "hi": hi}
    w2 = np.array([labels[k]["wrong2"] for k in hkeys])
    agree = np.array([x != "" and int(x) == int(t) for x, t in zip(w2, yh)])
    if agree.sum() >= 50 and len(set(yh[agree])) == 2:
        r["judges_agree"] = {"n": int(agree.sum()), "probe": ci(th[agree], yh[agree], sh[agree]),
                             **({"best_verbal": ci(th[agree], yh[agree], vb[best]["scores"][agree])} if vb else {})}

    res.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1, default=float))
    with open(out.with_name(out.stem + "-scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "task_id", "analyst", "wrong", "probe", *vb])
        for i, k in enumerate(hkeys):
            w.writerow([k, th[i], analyst[i], int(yh[i]), float(sh[i]), *(float(v["scores"][i]) for v in vb.values())])
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe", choices=sorted(PROBES))
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--states", help="default runs/probe/<tag>/states/<probe>.npz")
    ap.add_argument("--rerun-reason")
    a = ap.parse_args(argv)
    tag_dir = ROOT / "runs" / "probe" / a.tag
    r = run(a.probe, tag_dir, Path(a.states or tag_dir / "states" / f"{a.probe}.npz"), a.rerun_reason)
    keep = ("probe", "n_dev", "n_dev_wrong", "n_heldout", "n_heldout_wrong", "selection", "probe_heldout",
            "verbal_heldout", "best_verbal", "difference", "stop_rule_met", "operating_point",
            "verbal_released_wrong", "probe_catches_of_those", "length_heldout", "shuffled_heldout", "within_task")
    show = {k: r[k] for k in keep if k in r}
    show["selection"] = {k: v for k, v in r["selection"].items() if k != "table"}
    print(json.dumps(show, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
