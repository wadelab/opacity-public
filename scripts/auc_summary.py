"""AUC of each checker on the analysts' real write-ups, and the operating points
of every analyst + checker combination.

    .venv/bin/python scripts/auc_summary.py
        -> runs/summary/sizing/auc/checkers.csv       every checker, AUC on all it saw
                                                      and on the write-ups all of them saw
           runs/summary/sizing/auc/systems.csv        every analyst configuration x task set,
                                                      alone, with each checker, and with
                                                      gpt-oss 20B plus one more
           runs/summary/sizing/auc/fresh_systems.csv  analysts tested on the fresh set, with
                                                      the two checkers run there
           docs/figures/sizing_auc.{pdf,png}

Truth for a write-up is the judge's verdict: MiMo (`judged`) on dev, held-out
and fresh, the validated local gpt-oss 120B judge (`judged_alt`) on fresh2. An
unparseable write-up counts as wrong, as in `sizing.outcomes`; a declined one
is never checked and is left out of the AUC. A checker's score is 50 + conf/2
for "yes", 50 for cannot_tell or an unparseable reply, 50 - conf/2 for "no"
(as `sizing_report.py`). Checkers in series release only on a clean "no" from
each, so their joint score is the largest. Intervals are 95% percentile
bootstraps over tasks.

AUC ranks checkers; it does not rank systems. It ignores how often the
analyst is wrong, and the best analysts leave too few wrong write-ups for it
to be estimated (two of 108 for Qwen3.5 122B-A10B with the rules prompt on the
fresh set). The fresh-system table therefore pools each analyst's write-ups
over dev, held-out and fresh and prints the number of wrong ones beside the
AUC. Loading every log takes about five minutes.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "runs" / "summary" / "sizing" / "auc"
FIG = ROOT / "docs" / "figures" / "sizing_auc"
SETS = ("heldout", "fresh2", "fresh")  # fresh2 before fresh: config parts are matched exactly
JUDGE = {"dev": "wrong", "heldout": "wrong", "fresh": "wrong", "fresh2": "wrong3"}
FRESH_CHECKERS = ("gpt-oss-20b|effort-medium", "typed|think512")
ANCHOR = FRESH_CHECKERS[0]


def split(config: str) -> tuple[str, str, str]:
    """'Qwen3.5-27B|open|think1024|rules|fresh2|v2' -> ('Qwen3.5-27B', 'open|think1024|rules', 'fresh2')."""
    p = config.split("|")
    ts = next((x for x in p if x in SETS), "dev")
    return p[0], "|".join(x for x in p[1:] if x not in SETS and x not in ("v1", "v2")), ts


def truth(it: dict) -> bool | None:
    """True if the write-up is wrong, None if it was declined or not judged."""
    if it["declined"]:
        return None
    if not it["parse_ok"]:
        return True
    return it[JUDGE[it["task_set"]]]


def score(v: tuple | None) -> float | None:
    """A verdict (flag, confidence, parse_ok) as a risk score; None if there is none."""
    if v is None:
        return None
    flag, conf, ok = v
    if not ok or flag == "cannot_tell":
        return 50.0
    return 50 + (conf or 0) / 2 if flag == "yes" else 50 - (conf or 0) / 2


def clean_no(v: tuple | None) -> bool:
    return v is not None and bool(v[2]) and v[0] == "no"


def joint(vs: list) -> tuple[float | None, bool]:
    """Checkers in series: (largest score, released by every one). A missing verdict withholds."""
    if any(v is None for v in vs):
        return None, False
    return max(score(v) for v in vs), all(clean_no(v) for v in vs)


def auc(pos, neg) -> float:
    """Mann-Whitney AUC, ties counted a half, from average ranks."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_ci(rows: list[tuple[str, bool, float]], boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """rows are (task_id, wrong, score); the interval resamples tasks."""
    point = auc([s for _, w, s in rows if w], [s for _, w, s in rows if not w])
    by: dict[str, list] = defaultdict(list)
    for t, w, s in rows:
        by[t].append((w, s))
    tasks = list(by)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(boot if tasks else 0):
        rr = [x for i in rng.choice(len(tasks), len(tasks)) for x in by[tasks[i]]]
        a = auc([s for w, s in rr if w], [s for w, s in rr if not w])
        if not np.isnan(a):
            vals.append(a)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan"))
    return point, float(lo), float(hi)


def operating(its: list[dict], checks: dict, cks: list[str]) -> dict:
    """Rates as `sizing.rates`, with the named checkers in series, fail closed."""
    s, c = [], []
    for it in its:
        w = truth(it)
        if it["declined"]:
            o = "withheld"
        elif cks and not joint([checks[k].get((it["log"], it["sample"])) for k in cks])[1]:
            o = "withheld"
        elif w is None:
            continue
        else:
            o = "wrong" if w else "right"
        (c if it["condition"] == "control" else s).append(o)

    def f(xs, v):
        return sum(x == v for x in xs) / len(xs) if xs else float("nan")

    return {"n_seeded": len(s), "n_control": len(c), "n_wrong_released": sum(x == "wrong" for x in s),
            "wrong_released": f(s, "wrong"), "correct_released": f(s + c, "right"),
            "controls_answered": f(c, "right") + f(c, "wrong")}


def auc_rows(its: list[dict], checks: dict, cks: list[str]) -> list[tuple[str, bool, float]]:
    rows = []
    for it in its:
        w = truth(it)
        sc = joint([checks[k].get((it["log"], it["sample"])) for k in cks])[0] if w is not None else None
        if sc is not None:
            rows.append((it["task_id"], w, sc))
    return rows


def covers(its: list[dict], verdicts: dict, frac: float = 0.9) -> bool:
    rel = [it for it in its if not it["declined"]]
    return bool(rel) and sum((it["log"], it["sample"]) in verdicts for it in rel) / len(rel) >= frac


def load() -> tuple[list[dict], dict]:
    from opacity import sizing as S

    items = []
    for p in S.glob_logs(str(ROOT / "runs/inspect/scored-sizing/*_harvest*.eval")):
        try:
            its = S.load_analyst(p, keep_text=False)
        except Exception as e:  # a log still being written, or damaged
            print("skip", p, e)
            continue
        if not its or not its[0].config.endswith("|v2"):
            continue
        model, arm, ts = split(its[0].config)
        rows = [{"model": model, "arm": arm, "task_set": ts, "log": it.log, "sample": it.sample,
                 "task_id": it.task_id, "condition": it.condition, "declined": it.declined,
                 "parse_ok": it.parse_ok, "wrong": it.wrong, "wrong2": it.wrong2, "wrong3": it.wrong3} for it in its]
        released = [r for r in rows if not r["declined"]]
        if released and sum(truth(r) is not None for r in released) < 0.9 * len(released):
            print("skip, judging incomplete:", Path(p).name, its[0].config)  # a judge still running on it
            continue
        items += rows
    checks: dict[str, dict] = defaultdict(dict)
    for p in S.glob_logs(str(ROOT / "runs/inspect/viking/crosscheck/*crosscheck*.eval"),
                         str(ROOT / "runs/inspect/crosscheck/*crosscheck*.eval")):
        key, v = S.load_crosscheck(p)
        checks[key].update({k: (x.flag, x.confidence, x.parse_ok) for k, x in v.items()})
    return items, dict(checks)


def checker_table(items: list[dict], checks: dict) -> tuple[list[dict], list]:
    by_key = {(it["log"], it["sample"]): it for it in items}
    seen = {k: {x for x in v if x in by_key and truth(by_key[x]) is not None} for k, v in checks.items()}
    seen = {k: v for k, v in seen.items() if v}
    common = sorted(set.intersection(*seen.values()))
    rows = []
    for k, xs in seen.items():
        r_all = [(by_key[x]["task_id"], truth(by_key[x]), score(checks[k][x])) for x in sorted(xs)]
        r_com = [(by_key[x]["task_id"], truth(by_key[x]), score(checks[k][x])) for x in common]
        held = [(truth(by_key[x]), not clean_no(checks[k][x])) for x in common]
        a, lo, hi = auc_ci(r_all)
        ca, clo, chi = auc_ci(r_com)
        rows.append({"checker": k, "n_wrong": sum(w for _, w, _ in r_all), "n_right": sum(not w for _, w, _ in r_all),
                     "auc": a, "lo": lo, "hi": hi,
                     "common_n_wrong": sum(w for _, w, _ in r_com), "common_n_right": sum(not w for _, w, _ in r_com),
                     "common_auc": ca, "common_lo": clo, "common_hi": chi,
                     "common_withhold_wrong": np.mean([h for w, h in held if w]),
                     "common_withhold_right": np.mean([h for w, h in held if not w])})
    rows.sort(key=lambda r: -r["common_auc"])
    return rows, common


def system_table(items: list[dict], checks: dict) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for it in items:
        groups[(it["model"], it["arm"], it["task_set"])].append(it)
    out = []
    for (model, arm, ts), its in sorted(groups.items()):
        cov = [k for k in checks if covers(its, checks[k])]
        combos = [[]] + [[k] for k in cov] + [[ANCHOR, k] for k in cov if ANCHOR in cov and k != ANCHOR]
        for cks in combos:
            row = {"model": model, "arm": arm, "task_set": ts, "checkers": " + ".join(cks) or "none",
                   **operating(its, checks, cks)}
            if cks:
                rr = auc_rows(its, checks, cks)
                a, lo, hi = auc_ci(rr, boot=400)
                row.update(auc=a, lo=lo, hi=hi, n_wrong_checked=sum(w for _, w, _ in rr),
                           n_right_checked=sum(not w for _, w, _ in rr))
            out.append(row)
    return out


def fresh_table(items: list[dict], checks: dict) -> list[dict]:
    out = []
    for model, arm in sorted({(it["model"], it["arm"]) for it in items if it["task_set"] == "fresh"}):
        pooled = [it for it in items if (it["model"], it["arm"]) == (model, arm) and it["task_set"] in ("dev", "heldout", "fresh")]
        fresh = [it for it in pooled if it["task_set"] == "fresh"]
        a = operating(fresh, checks, [])
        row = {"model": model, "arm": arm, "fresh_wrong": a["wrong_released"], "fresh_k": a["n_wrong_released"],
               "fresh_n": a["n_seeded"], "fresh_correct": a["correct_released"], "fresh_controls": a["controls_answered"]}
        for name, cks in (("g20", [FRESH_CHECKERS[0]]), ("typed", [FRESH_CHECKERS[1]]), ("both", list(FRESH_CHECKERS))):
            o = operating(fresh, checks, cks)
            rr = auc_rows(pooled, checks, cks)
            au, lo, hi = auc_ci(rr)
            row.update({f"{name}_wrong": o["wrong_released"], f"{name}_k": o["n_wrong_released"],
                        f"{name}_correct": o["correct_released"], f"{name}_controls": o["controls_answered"],
                        f"{name}_auc": au, f"{name}_lo": lo, f"{name}_hi": hi,
                        f"{name}_nwrong": sum(w for _, w, _ in rr), f"{name}_nright": sum(not w for _, w, _ in rr)})
        out.append(row)
    return out


def write(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})


def roc(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    th = np.unique(np.concatenate([pos, neg]))[::-1]
    return np.r_[0, [np.mean(neg >= t) for t in th]], np.r_[0, [np.mean(pos >= t) for t in th]]


NAME = {"Qwen3.5-122B-A10B-NVFP4": "Qwen3.5 122B-A10B", "Qwen3.5-35B-A3B": "Qwen3.5 35B-A3B", "Qwen3.5-27B": "Qwen3.5 27B",
        "Qwen3.5-9B": "Qwen3.5 9B", "gemma-4-26B-A4B-it": "Gemma 4 26B-A4B", "gemma-4-31B-it": "Gemma 4 31B",
        "gpt-oss-120b": "gpt-oss 120B", "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": "Nemotron 3 Super 120B",
        "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4": "Nemotron 3.5 Lightning 30B"}
RECOMMENDED = ("Qwen3.5-27B", "open|think1024|rules")  # the analyst the paper recommends, in bold in (b, c)


def figure(items: list[dict], checks: dict, ctab: list[dict], common: list, ftab: list[dict]) -> None:
    """(a) ROC of every checker on the write-ups all of them saw; (b, c) every analyst
    tested on the fresh set, alone and then gpt-oss 20B: wrong and right released."""
    import matplotlib

    matplotlib.use("Agg")
    sys.path.insert(0, str(ROOT / "docs" / "figures"))
    import paperfig as P  # the paper's shared figure look

    by_key = {(it["log"], it["sample"]): it for it in items}
    w = np.array([truth(by_key[x]) for x in common], bool)
    t1 = {r["checker"]: r for r in ctab}
    best = ctab[0]["checker"]
    best_name = ("Qwen3.5 27B, 1k think, with tables" if best == "Qwen3.5-27B|think1024|pack"
                 else best.replace("|", ", "))
    # name, colour, line style, label position (axes data), text alignment
    focus = {best: (best_name.replace(", with", ",\nwith"), P.INK, "-", 1.0, (0.03, 0.97), "left"),
             FRESH_CHECKERS[0]: ("gpt-oss 20B, medium effort,\nthe checker used", P.BLUE, "-", 1.5, (0.42, 0.36), "left"),
             FRESH_CHECKERS[1]: ("typed reviewer,\n512-token thinking", P.MUTED, (0, (3, 1.5)), 1.0, (0.42, 0.17), "left")}

    rows = sorted(ftab, key=lambda r: (r["fresh_wrong"], -r["fresh_correct"]), reverse=True)
    n = len(rows)

    def row_name(r: dict) -> str:
        """The paper's table names: 'rules', 'rules, 1k', 'rules, high', 'revise' (a variant of rules)."""
        arm = r["arm"].split("|")
        extra = (["revise"] if arm[0] == "revise" else ["rules"] if "rules" in arm else []) + \
            (["1k"] if "think1024" in arm else []) + (["high"] if "effort-high" in arm else [])
        return ", ".join([NAME.get(r["model"], r["model"])] + extra)

    labels = [row_name(r) for r in rows]
    rec = {row_name(r) for r in rows if (r["model"], r["arm"]) == RECOMMENDED}

    # Laid out in millimetres at the paper's text width, so every gap is set by hand.
    W, H = 148.0, 94.0
    top, pitch, band = 15.0, 3.3, 2.1  # axes top below the page top; mm per row; key rows above the first
    fig = P.figure_mm(W, H)
    widest = max(fig.text(0, 0, t, fontsize=P.SMALL, fontweight="bold" if t in rec else "normal")
                 .get_window_extent(fig.canvas.get_renderer()).width for t in labels) / fig.dpi / P.MM
    for t in list(fig.texts):
        t.remove()

    def box(left, width, height):
        return P.axes_mm(fig, left, H - top - height, width, height)

    side = 52.0
    ax = box(7.0, side, side)
    cw = 20.5
    cx_left = W - 2.5 - cw
    bx_left = cx_left - 6.0 - cw
    bx = box(bx_left, cw, (n + band) * pitch)
    cx = box(cx_left, cw, (n + band) * pitch)
    assert bx_left - widest - 2.5 > 7.0 + side + 4.0, "row labels would run into panel a"

    # (a) the cloud in light grey, the three named checkers on top, each named in clear space.
    for k in sorted(t1, key=lambda k: k in focus):
        s = np.array([score(checks[k][x]) for x in common])
        fpr, tpr = roc(s[w], s[~w])
        if k not in focus:
            ax.plot(fpr, tpr, color=P.LIGHT, lw=0.5, zorder=1)
            continue
        name, col, ls, lw, (tx, ty), ha = focus[k]
        ax.plot(fpr, tpr, color=col, lw=lw, ls=ls, zorder=3)
        held = np.array([not clean_no(checks[k][x]) for x in common])
        ox, oy = held[~w].mean(), held[w].mean()
        ax.plot(ox, oy, "o", color=col, mec="white", mew=0.6, ms=4.5, zorder=5)
        ax.annotate(f"{name}, AUC {t1[k]['common_auc']:.2f}", xy=(ox, oy), xytext=(tx, ty), textcoords="data",
                    ha=ha, va="top", fontsize=P.SMALL, color=col, linespacing=1.1, zorder=4,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.5, shrinkA=1, shrinkB=2.5))
    ax.plot([0, 1], [0, 1], ls=":", color=P.GREY, lw=0.8, zorder=2)
    P.label_at(ax, 0.97, 0.80, "chance", color=P.MUTED, ha="right", va="center", dy=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.set_xticks(ticks, ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.set_yticks(ticks, ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.grid(False)
    ax.set_xlabel("Right write-ups withheld (share)")
    P.head(ax, "a", "Checkers discriminate weakly\non real write-ups", "Wrong write-ups withheld (share)")

    # (b, c) one row per analyst: open dot alone, filled dot then gpt-oss 20B.
    for i, r in enumerate(rows):
        col = P.BLUE if "rules" in r["arm"] else P.GREY
        for axx, a, c in ((bx, r["fresh_wrong"], r["g20_wrong"]), (cx, r["fresh_correct"], r["g20_correct"])):
            axx.plot([a, c], [i, i], color=col, lw=0.8, alpha=0.6, zorder=2)
            axx.plot(a, i, "o", mfc="white", mec=col, mew=0.9, ms=3.8, zorder=3)
            axx.plot(c, i, "o", color=col, ms=3.8, zorder=3)
    P.rows(bx, labels, bold=rec)
    for t in bx.get_yticklabels():
        t.set_fontsize(P.SMALL)
    cx.set_yticks([])
    for axx, lo, hi, tk in ((bx, -0.02, 0.62, [0, 0.2, 0.4, 0.6]), (cx, 0.3, 1.02, [0.4, 0.6, 0.8, 1.0])):
        axx.set_ylim(n - 0.5, -0.5 - band)
        axx.set_xlim(lo, hi)
        axx.set_xticks(tk, [f"{t:g}" for t in tk])
        axx.grid(False)
        axx.spines["left"].set_visible(False)
        axx.vlines([t for t in tk if t], -0.5, n - 0.5, color=P.FAINT, lw=0.5, zorder=1)  # stop under the key
    bx.vlines(0.05, -0.5, n - 0.5, color=P.INK, lw=0.7, ls=(0, (4, 2.5)), zorder=1)
    P.label_at(bx, 0.05, -0.5, "release\nthreshold", ha="left", va="bottom", dx=-1, dy=1, linespacing=1.0)
    for j, (mfc, text) in enumerate((("white", "analyst alone"), (P.INK, "then gpt-oss 20B"))):
        y = -0.5 - band + 0.45 + j * 1.0
        cx.plot(0.33, y, "o", mfc=mfc, mec=P.INK, mew=0.9, ms=3.8, clip_on=False)
        P.label_at(cx, 0.33, y, text, color=P.INK, ha="left", va="center", dx=4, dy=0)
    bx.set_xlabel("Share of planted-error\nattempts")
    cx.set_xlabel("Share of all\nattempts")
    P.head(bx, "b", "The analyst\nsets the error", "Wrong released")
    P.head(cx, "c", "Checking costs\nright answers", "Right released")
    P.save(fig, FIG)


def main() -> int:
    items, checks = load()
    print(f"{len(items)} write-ups, {len(checks)} checkers")
    ctab, common = checker_table(items, checks)
    write("checkers.csv", ctab)
    write("systems.csv", system_table(items, checks))
    ftab = fresh_table(items, checks)
    write("fresh_systems.csv", ftab)
    figure(items, checks, ctab, common, ftab)
    print(f"common write-ups: {len(common)}; wrote {OUT} and {FIG}.pdf/.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
