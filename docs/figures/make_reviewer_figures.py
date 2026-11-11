"""Figures a reviewer asks for next: a class-by-level map of what each
reviewer sees, the ROC curves behind the AUCs, the calibration of the
reviewers' confidence, and the cost of a verdict. Reads the Inspect logs
directly (after design only).

    .venv/bin/python docs/figures/make_reviewer_figures.py

Writes fig_class_map, fig_roc, fig_calibration (pdf + png) beside this
file and runs/summary/cost_by_checker.csv. The look is the paper's
(paperfig.py).

The typed reviewer in the paper is the decision server with one sample and
no thinking (the typed-new row of runs/summary/*.csv, Table 3). Two other
typed runs (eight samples; a 512-token think) are in the same folder; the
figures leave them out so that they show the reviewer the tables describe.
The cost table keeps every typed run, as it always has.
"""

from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from inspect_ai.log import read_eval_log
from matplotlib.colors import LinearSegmentedColormap, to_rgb

import paperfig as P

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

CHECKERS = {  # model key -> name as the paper writes it
    "mimo-v2.6-pro": "MiMo pro",
    "deepseek-v4.1-flash": "DeepSeek flash",
    "Qwen3.5-27B": "Qwen3.5 27B",
    "Mistral-Small-3.2-24B-Instruct-2506-NVFP4": "Mistral Small 24B",
    "Llama-3.3-70B-Instruct-NVFP4": "Llama 3.3 70B",
    "typed": "typed reviewer",
}
PAPER_TYPED = (1, 0)  # (samples, thinking tokens) of the typed reviewer in Table 3
CLASSES = ["1 multiplicity", "2 silent attrition", "3 confounding", "4 causal claim", "5 error structure",
           "6 wrong denominator", "7 regression to mean", "8 in-sample"]


def classify(tags):
    for t in tags:
        for pre, name in (("hosted2-20261021", "hosted-new"), ("sweep4-", "sweep4-new-nothink"), ("sweep5-", "sweep5-new-nothink"), ("typed", "typed-new")):
            if t.startswith(pre):
                return name
    return "other"


def score(sample) -> float:
    v = sample.scores["verdict"]; md = v.metadata or {}
    try:
        comp = json.loads(sample.output.completion)
        if "p_error_present" in comp:
            return 100 * float(comp["p_error_present"].get("yes", 0.5))
    except Exception:
        pass
    conf = float(md.get("confidence") or 0)
    return 50 + conf / 2 if v.answer == "yes" else 50 - conf / 2 if v.answer == "no" else 50.0


def load():
    """samples[model] = list of dicts with level, class, condition, score, flagged, correct, confidence,
    seconds, tokens, answer and, for the typed reviewer, its (samples, thinking) setting."""
    out = defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / "runs/inspect/*opacity-*.eval")) + glob.glob(str(ROOT / "runs/inspect/viking/*opacity-*.eval"))):
        log = read_eval_log(f)
        md = log.eval.metadata or {}
        if classify(log.eval.tags or []) == "other" or log.status != "success" or md.get("arm") == "mechanical":
            continue
        key = "typed" if md.get("arm") == "typed" else log.eval.model.split("/")[-1]
        if key not in CHECKERS:
            continue
        ta = log.eval.task_args or {}
        setting = (int(ta.get("typed_samples", 1)), int(ta.get("typed_think", 0))) if key == "typed" else None
        for s in log.samples:
            v = s.scores["verdict"]; m = s.metadata; u = s.output.usage
            out[key].append(dict(level=m["disclosure_level"], cls=m["error_class"], cond=m["condition"], score=score(s),
                                 flagged=v.value["flagged"], correct=v.value["correct"], confidence=float((v.metadata or {}).get("confidence") or 0),
                                 seconds=getattr(s.output, "time", None), tokens=(u.output_tokens if u else None), answer=v.answer,
                                 setting=setting))
    return out


def as_in_paper(S):
    """The reviewers as the paper's tables describe them: the typed reviewer's one configuration only."""
    T = dict(S)
    if "typed" in T:
        T["typed"] = [r for r in T["typed"] if r.get("setting") == PAPER_TYPED]
    return T


def roc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    ths = np.unique(np.concatenate([pos, neg, [-1, 101]]))[::-1]
    tpr = [(pos >= t).mean() for t in ths]; fpr = [(neg >= t).mean() for t in ths]
    return np.array(fpr), np.array(tpr)


def auc(pos, neg) -> float:
    """Mann-Whitney AUC, ties counted half, as in scripts/auc_by_level.py."""
    pos = np.asarray(pos)[:, None]; neg = np.asarray(neg)[None, :]
    return float(np.mean((pos > neg) + 0.5 * (pos == neg)))


def luminance(color) -> float:
    c = np.array(to_rgb(color))
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def text_on(color) -> str:
    """White or ink, whichever contrasts more with the cell."""
    L = luminance(color[:3])
    return "white" if 1.05 / (L + 0.05) > (L + 0.05) / (luminance(P.INK) + 0.05) else P.INK


# --------------------------------------------------------------------------
def fig_class_map(S):
    """Share of planted-error reports flagged, per class and level, one panel
    per reviewer. Cells at 1.0 carry no number, so the misses stand out."""
    S = as_in_paper(S)
    keys = [k for k in CHECKERS if k in S]
    cmap = LinearSegmentedColormap.from_list("paper_blue", ["#F4F5F7", P.BLUE])
    left, width, gap, cell_h = 23.6, 19.4, 1.6, 3.4
    height = 8 * cell_h
    fig = P.figure_mm(148, height + 12.5)
    for i, key in enumerate(keys):
        ax = P.axes_mm(fig, left + i * (width + gap), 1, width, height)
        grid = np.full((8, 5), np.nan)
        for c in range(1, 9):
            for l in range(5):
                rows = [r for r in S[key] if r["cond"] == "seeded" and r["cls"] == c and r["level"] == l]
                if rows:
                    grid[c - 1, l] = np.mean([r["flagged"] for r in rows])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        for c in range(8):
            for l in range(5):
                v = grid[c, l]
                if np.isnan(v) or v > 0.999:
                    continue
                ax.text(l, c, f"{v:.1f}", ha="center", va="center", fontsize=P.SMALL, color=text_on(cmap(v)))
        ax.set_xticks(range(5), ["s0", "s1", "s2", "s3", "s4"], fontsize=P.SMALL)
        ax.xaxis.tick_top()
        ax.set_yticks(range(8), CLASSES if i == 0 else [])
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(False)
        ax.set_title(CHECKERS[key], loc="left", fontsize=6.8, pad=8)
    P.save(fig, HERE / "fig_class_map")


# --------------------------------------------------------------------------
LEVELS = ((0, P.GREY, "-", 1.1), (3, P.BLUE, "-", 1.4), (4, P.BLUE, (0, (1, 1.4)), 1.2))


def fig_roc(S):
    """One panel per reviewer: its ROC curves with every table (s0), at the
    conservative level (s3) and from the prose alone (s4), and the AUCs."""
    S = as_in_paper(S)
    keys = [k for k in CHECKERS if k in S]
    fig = P.figure_mm(148, 106)
    size, x0, dx, y_low, y_high = 38, 9, 47.5, 9, 60
    for i, key in enumerate(keys):
        row, col = divmod(i, 3)
        ax = P.axes_mm(fig, x0 + col * dx, y_high if row == 0 else y_low, size, size)
        ax.plot([0, 1], [0, 1], color=P.LIGHT, lw=0.8, ls=(0, (3, 2)), zorder=0)
        aucs = {}
        for level, color, ls, lw in LEVELS:
            rows = [r for r in S[key] if r["level"] == level]
            pos = [r["score"] for r in rows if r["cond"] == "seeded"]
            neg = [r["score"] for r in rows if r["cond"] == "control"]
            fpr, tpr = roc(pos, neg)
            ax.plot(fpr, tpr, color=color, ls=ls, lw=lw, drawstyle="steps-post", zorder=3 if level == 3 else 2)
            aucs[level] = auc(pos, neg)
        # a key with the AUCs, below the diagonal where no curve runs
        ax.text(0.68, 0.40, "AUC", transform=ax.transAxes, fontsize=P.SMALL, va="center", ha="left", color=P.MUTED)
        for k, (level, color, ls, lw) in enumerate(LEVELS):
            yk = 0.29 - k * 0.1
            ax.plot([0.68, 0.78], [yk, yk], color=color, ls=ls, lw=lw, transform=ax.transAxes, clip_on=False)
            ax.text(0.81, yk, f"s{level}  {aucs[level]:.2f}", transform=ax.transAxes, fontsize=P.SMALL,
                    va="center", ha="left", color=P.INK)
        if i == 0:
            ax.text(0.30, 0.20, "chance", transform=ax.transAxes, fontsize=P.SMALL, color=P.MUTED, ha="left", va="center")
        word = "flatter" if aucs[3] < aucs[0] - 0.02 else "steeper" if aucs[3] > aucs[0] + 0.02 else "unchanged"
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"] if col == 0 else [])
        ax.grid(False)
        if row == 1 and col == 1:
            ax.set_xlabel("Correct reports flagged (false alarm)")
        P.head(ax, "", f"{CHECKERS[key]}: {word} at s3",
               "Planted-error reports flagged" if col == 0 else " ")
    P.save(fig, HERE / "fig_roc")


# --------------------------------------------------------------------------
GENERATIVE_MARKERS = {"mimo-v2.6-pro": "o", "deepseek-v4.1-flash": "s", "Qwen3.5-27B": "^",
                      "Mistral-Small-3.2-24B-Instruct-2506-NVFP4": "D", "Llama-3.3-70B-Instruct-NVFP4": "v"}


def fig_calibration(S):
    """Reliability: stated confidence against the share of verdicts that
    were correct, over all levels and both conditions, in bins of at least
    eight verdicts. The typed reviewer in blue, the generative ones in grey."""
    S = as_in_paper(S)
    fig = P.figure_mm(89, 74)
    ax = P.axes_mm(fig, 9, 20, 76, 42)
    ax.plot([0, 100], [0, 1], color=P.LIGHT, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.text(42, 0.33, "calibrated", fontsize=P.SMALL, color=P.MUTED, ha="left", va="bottom")
    edges = [0, 50, 60, 70, 80, 90, 101]
    handles = []
    for key, label in CHECKERS.items():
        rows = [r for r in S.get(key, []) if r["answer"] in ("yes", "no")]
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = [r for r in rows if lo <= r["confidence"] < hi]
            if len(b) >= 8:
                xs.append(np.mean([r["confidence"] for r in b])); ys.append(np.mean([r["correct"] for r in b]))
        if not xs:
            continue
        if key == "typed":
            h, = ax.plot(xs, ys, color=P.BLUE, lw=1.5, marker="o", ms=3.6, zorder=3, label=label)
        else:
            h, = ax.plot(xs, ys, color=P.GREY, lw=0.9, marker=GENERATIVE_MARKERS[key], ms=3.4, mfc="white", mew=0.9,
                         zorder=4, label=label)
        handles.append(h)
    ax.set_xlim(30, 100.5); ax.set_ylim(0.2, 1.03)
    ax.set_xticks(range(30, 101, 10)); ax.set_yticks(np.arange(0.2, 1.01, 0.2), ["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.grid(False)
    ax.set_xlabel("Stated confidence (0 to 100)")
    order = [h for h in handles if h.get_label() == CHECKERS["typed"]] + [h for h in handles if h.get_label() != CHECKERS["typed"]]
    fig.legend(handles=order, loc="lower left", bbox_to_anchor=(6 / 89, 0.5 / 78), ncol=3, fontsize=P.SMALL,
               handlelength=2.2, columnspacing=1.2, borderaxespad=0, frameon=False)
    P.head(ax, "", "Confidence is informative but not calibrated", "Share of verdicts correct")
    P.save(fig, HERE / "fig_calibration")


def cost_table(S):
    rows = []
    for key, label in CHECKERS.items():
        rs = S.get(key, [])
        if not rs:
            continue
        secs = [r["seconds"] for r in rs if r["seconds"] is not None]; toks = [r["tokens"] for r in rs if r["tokens"] is not None]
        rows.append(dict(checker={"typed": "typed (openjev)", "deepseek-v4.1-flash": "DeepSeek flash", "Mistral-Small-3.2-24B-Instruct-2506-NVFP4": "Mistral 24B",
                                  "Llama-3.3-70B-Instruct-NVFP4": "Llama 70B"}.get(key, label), n=len(rs),
                         median_seconds=round(float(np.median(secs)), 1) if secs else "", p90_seconds=round(float(np.percentile(secs, 90)), 1) if secs else "",
                         median_output_tokens=int(np.median(toks)) if toks else "", p90_output_tokens=int(np.percentile(toks, 90)) if toks else ""))
    out = ROOT / "runs/summary/cost_by_checker.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)
    return rows


if __name__ == "__main__":
    S = load()
    fig_class_map(S); fig_roc(S); fig_calibration(S); cost_table(S)
    print("reviewer figures written to", HERE)
