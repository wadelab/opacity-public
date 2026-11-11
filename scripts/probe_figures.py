"""Figures for the probe experiments (docs/vetting.md 4.7 and 4.8).

    .venv/bin/python scripts/probe_figures.py
      -> docs/figures/probe_reading.{pdf,png}   reading probes against the models' own verdicts
         docs/figures/probe_steer.{pdf,png}     the positive control, once its result exists
         docs/figures/slant_vetting.{pdf,png}   the slant detector on the recommended analyst (3.6)
         docs/figures/style_robustness.{pdf,png}  slant detection under a change of style (4.11)

Read only: every number comes from the held-out result files and the
per-write-up scores the fits saved under runs/probe/. Intervals are 95%
percentile bootstraps over tasks, as in the fits.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opacity.probe import fit as F  # noqa: E402
from opacity.probe import steer as T  # noqa: E402

A = F.A
FIG = ROOT / "docs" / "figures"
sys.path.insert(0, str(FIG))
import paperfig as P  # noqa: E402

BLUE, ORANGE, GREY, LIGHT, INK, MUTED = P.BLUE, P.ORANGE, P.GREY, P.LIGHT, P.INK, P.MUTED
MM = P.MM


def scores(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    return out


def pooled(task, y, s) -> tuple[float, float, float]:
    return A.auc_ci(list(zip(task, y.astype(bool), s)), boot=F.BOOT)


def robot(ax, x: float, y: float, inside: bool) -> None:
    """A small model, as drawn on the explainer page: a head on a stand, with
    a face, or with its internal state showing as a grid of units (mm units)."""
    from matplotlib.patches import Circle, FancyBboxPatch

    lw = 1.1
    ax.add_patch(FancyBboxPatch((x - 3.2, y - 9.5), 6.4, 4.2, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="white", ec=INK, lw=lw))
    ax.add_patch(FancyBboxPatch((x - 1.3, y - 5.6), 2.6, 1.4, boxstyle="square,pad=0", fc=INK, ec=INK, lw=0))
    ax.add_patch(FancyBboxPatch((x - 7, y - 4.3), 14, 10.6, boxstyle="round,pad=0,rounding_size=2.2",
                                fc="white", ec=INK, lw=lw))
    ax.plot([x, x], [y + 6.3, y + 8.2], color=INK, lw=lw, solid_capstyle="round")
    ax.add_patch(Circle((x, y + 8.6), 0.75, fc=ORANGE, ec="none"))
    if inside:
        ax.add_patch(FancyBboxPatch((x - 5.4, y + 0.6), 10.8, 4.4, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc="#E2EBF4", ec="none"))
        rng = np.random.default_rng(3)
        for i in range(7):
            for j in range(3):
                c = rng.choice([BLUE, BLUE, ORANGE, LIGHT, LIGHT, LIGHT])
                ax.add_patch(Circle((x - 4.5 + i * 1.5, y + 1.5 + j * 1.3), 0.42, fc=c, ec="none"))
        eyes_y = y - 1.6
    else:
        eyes_y = y + 2.0
        ax.plot([x - 1.6, x + 1.6], [y - 1.6, y - 1.6], color=INK, lw=lw, solid_capstyle="round")
    for dx in (-2.4, 2.4):
        ax.add_patch(Circle((x + dx, eyes_y), 0.8, fc=INK, ec="none"))


def card(ax, x: float, title: str, gloss: str, color: str) -> None:
    """A bold title under a picture and its gloss hung below it by a fixed
    distance in points, so the spacing holds at any drawing scale."""
    ax.annotate(title, xy=(x, -8.5), ha="center", va="top", fontsize=8, fontweight="bold", color=color)
    ax.annotate(gloss, xy=(x, -8.5), xytext=(0, -12), textcoords="offset points", ha="center", va="top",
                fontsize=P.SMALL + 0.5, color=MUTED, linespacing=1.2)


def two_ways(ax) -> None:
    """Panel a: the two ways to check a write-up that the probes compare, laid
    out as the explainer page's two cards: a picture, a title, one line."""
    from matplotlib.patches import Circle, FancyBboxPatch, Polygon

    ax.set_xlim(0, 148)
    ax.set_ylim(-24, 15)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    # ask: the model's own verdict, in words
    robot(ax, 22, 3, inside=False)
    bubble = "#F2F1EE"
    ax.add_patch(FancyBboxPatch((32.5, 5.5), 31, 8.2, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=bubble, ec=GREY, lw=0.9))
    ax.add_patch(Polygon([(34.5, 5.7), (38.5, 5.7), (30.5, 2.2)], closed=True, fc=bubble, ec="none", zorder=3))
    ax.plot([34.5, 30.5, 38.5], [5.5, 2.2, 5.5], color=GREY, lw=0.9, solid_joinstyle="round", zorder=3)
    ax.text(48, 9.6, "\u201cSupported.\u201d", ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
    card(ax, 40, "Ask it", "it reads the write-up and says, in words,\nwhether the conclusion is supported", INK)
    # read: a probe on the internal state
    robot(ax, 107, 3, inside=True)
    ax.add_patch(Circle((112.2, 6.8), 3.6, fc="none", ec=BLUE, lw=1.6))
    ax.plot([114.8, 118.6], [4.2, 0.4], color=BLUE, lw=2.4, solid_capstyle="round")
    card(ax, 109.5, "Read its internal state", "a linear probe on its activations\nwhile it reads the same write-up",
         BLUE)
    ax.plot([74, 74], [-22, 12.5], color=P.FAINT, lw=0.8)


def reading() -> None:
    import matplotlib.pyplot as plt

    res = ROOT / "runs" / "probe" / "v1" / "results"
    models = (("Qwen3.5-27B", "Qwen3.5 27B"), ("gpt-oss-20b", "gpt-oss 20B"))
    rows = []  # (model label, kind, pooled (a, lo, hi), within (a, lo, hi))
    for probe, label in models:
        r = json.loads((res / f"{probe}-heldout-2.json").read_text())
        sc = scores(res / f"{probe}-heldout-2-scores.csv")
        task, y = sc["task_id"], sc["wrong"].astype(int)
        p, v = sc["probe"].astype(float), sc[r["best_verbal"]].astype(float)
        wp, wv = T.within_ci(task, y, p), T.within_ci(task, y, v)
        rows.append((label, "probe", pooled(task, y, p), (wp["auc"], wp["lo"], wp["hi"])))
        rows.append((label, "verdict", pooled(task, y, v), (wv["auc"], wv["lo"], wv["hi"])))
        rows.append((label, "shuffled", (r["shuffled_heldout"]["mean"],) * 3,
                     (r["within_task"]["shuffled_mean"],) * 3))

    fig = plt.figure(figsize=(P.DOUBLE, 116 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=(0.5, 1), hspace=0.08, wspace=0.06)
    illus = fig.add_subplot(gs[0, :])
    axs = [fig.add_subplot(gs[1, 0])]
    axs.append(fig.add_subplot(gs[1, 1], sharey=axs[0]))
    two_ways(illus)
    P.head(illus, "a", "Two ways to check a write-up")
    style = {"probe": dict(color=BLUE, marker="o", ms=5, mfc=BLUE),
             "verdict": dict(color=GREY, marker="D", ms=4.2, mfc=GREY),
             "shuffled": dict(color=GREY, marker="o", ms=5, mfc="white")}
    names = {"probe": "probe on its internal state", "verdict": "its own verdict",
             "shuffled": "probe trained on shuffled labels"}
    ypos, y0 = [], 0.0
    for i in range(len(rows)):
        if i and rows[i][0] != rows[i - 1][0]:
            y0 += 1.1  # room for the next model's name
        ypos.append(y0)
        y0 += 1
    for ax, j, letter, finding in ((axs[0], 2, "b", "Across tasks, the probe beats the verdict"),
                                   (axs[1], 3, "c", "Within a task, it does not")):
        for yy, row in zip(ypos, rows):
            a, lo, hi = row[j]
            st = style[row[1]]
            if hi > lo:
                ax.plot([lo, hi], [yy, yy], color=LIGHT if row[1] != "probe" else BLUE, lw=1.4,
                        solid_capstyle="round", zorder=2, alpha=1 if row[1] == "probe" else 1)
            ax.plot([a], [yy], ls="", marker=st["marker"], ms=st["ms"], mfc=st["mfc"], mec=st["color"], mew=1.1,
                    zorder=3)
            left = a < 0.5  # keep the value off the chance line
            P.label_at(ax, lo if left else max(a, hi), yy, f"{a:.2f}", color=BLUE if row[1] == "probe" else MUTED,
                       dx=-5 if left else 5, dy=0, ha="right" if left else "left", va="center", size=7)
        P.auc_axis(ax, 0.4, 1.0)
        P.head(ax, letter, finding, "Held-out AUC, wrong write-ups against right ones")
    axs[0].set_yticks(ypos, [names[r[1]] for r in rows])
    axs[0].tick_params(axis="y", length=0)
    axs[0].set_ylim(max(ypos) + 0.7, min(ypos) - 1.2)
    plt.setp(axs[1].get_yticklabels(), visible=False)
    axs[1].tick_params(axis="y", length=0)
    for probe, label in models:
        top = min(yy for yy, r in zip(ypos, rows) if r[0] == label)
        axs[0].annotate(label, xy=(0, top - 0.85), xycoords=("axes fraction", "data"), xytext=(-6, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=7.5, fontweight="bold")
    P.save(fig, FIG / "probe_reading")


wilson = P.wilson

DETECTORS = (("probe", "probe on the 27B's internal state", True), ("text classifier", "text classifier (word counts)", True),
             ("gpt-oss-20b|effort-medium", "gpt-oss 20B as checker", False),
             ("Qwen3.5-27B|off|pack", "Qwen3.5 27B as checker", False),
             ("verifier", "rule-based verifier", False))
WORD_FIELDS = ("title", "methods", "interpretation")  # the write-up's prose; the design record and tables are left out
WORD_MIN = 18  # a word must appear in at least this many write-ups of one version (10% of 180)
WORD_EACH = 6  # words shown each way


def word_shift(n_each: int = WORD_EACH) -> tuple[list, list, int]:
    """Words the biased write-ups used less and more than their honest pairs, on dev.

    For each word (stop words removed) the number of write-ups whose prose uses
    it, honest against biased under the second instruction. Among words used in
    at least WORD_MIN write-ups of either version, the n_each that changed by
    the largest factor each way. Returns (less, more, n per version)."""
    import glob
    import math
    import re

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    from opacity import slant
    from opacity.harvest.task import parse_json_object

    rows = slant.read(sorted(glob.glob(str(ROOT / "runs" / "inspect" / "steer2" / "*.eval"))))
    words = {}
    for arm in ("open", "steered2"):
        docs = []
        for r in rows:
            if r["task_set"] == "dev" and r["arm"] == arm:
                o = parse_json_object(r["text"]) or {}
                docs.append(set(re.findall(r"[a-z][a-z\-]+", " ".join(str(o.get(k, "")) for k in WORD_FIELDS).lower())))
        words[arm] = docs
    n = len(words["open"])
    assert n == len(words["steered2"]), "honest and biased dev write-ups do not pair one for one"
    vocab = set().union(*words["open"], *words["steered2"]) - set(ENGLISH_STOP_WORDS)
    ranked = []
    for w in vocab:
        h, b = sum(w in d for d in words["open"]), sum(w in d for d in words["steered2"])
        if max(h, b) >= WORD_MIN:
            ranked.append((math.log((b + 1) / (h + 1)), w, h, b))
    ranked.sort()
    return [r[1:] for r in ranked[:n_each]], [r[1:] for r in ranked[::-1][:n_each]], n


def steer() -> None:
    """The positive control: wrong conclusions under the bias instruction, the
    words the bias changed, and every detector's within-task AUC."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    path = ROOT / "runs" / "probe" / "steer-v2" / "results" / "steer-heldout.json"
    if not path.exists():
        return
    R = json.loads(path.read_text())
    less, more, n_words = word_shift()

    fig = P.figure_mm(148, 132)
    ax = P.axes_mm(fig, 9, 73, 52, 45)
    bx = P.axes_mm(fig, 96, 64, 30, 54)
    cx = P.axes_mm(fig, 49, 12, 88, 30)

    # a: jars, the share of conclusions the judge called wrong, dev and held-out together
    from matplotlib.patches import Rectangle

    jw = R["judge_wrong"]
    jars = [("honest", sum(jw[s]["honest"][0] for s in jw), sum(jw[s]["honest"][1] for s in jw), GREY),
            ("told to bias", sum(jw[s]["steered"][0] for s in jw), sum(jw[s]["steered"][1] for s in jw), ORANGE)]
    top, w = 42, 0.56
    for i, (name, k, n, col) in enumerate(jars):
        ax.add_patch(Rectangle((i - w / 2, 0), w, top, fc="#F4F3F0", ec="none", zorder=1))
        ax.add_patch(Rectangle((i - w / 2, 0), w, 100 * k / n, fc=col, ec="none", zorder=2))
        lo, hi = wilson(k, n)
        ax.plot([i, i], [100 * lo, 100 * hi], color=INK, lw=0.8, zorder=3)
        ax.text(i, 100 * hi + 1.2, f"{100 * k / n:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=INK)
        ax.annotate(f"{k} of {n}", xy=(i, 0), xytext=(0, -24), textcoords="offset points", ha="center", va="top",
                    fontsize=P.SMALL, color=MUTED)
    ax.set_xticks(range(len(jars)), [j[0] for j in jars])
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlim(-0.5, len(jars) - 0.5)
    ax.set_ylim(0, top)
    ax.set_yticks([0, 10, 20, 30, 40], ["0", "10", "20", "30", "40%"])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    P.head(ax, "a", "Told to bias, it misled more often", "Conclusions the judge called wrong,\ndev and held-out together")

    # b: the words the bias changed, dev
    y, ticks = 0.0, []
    for header, ws in (("Used less when biased", less), ("Used more when biased", more)):
        bx.annotate(header, xy=(0, y - 0.1), xycoords=("axes fraction", "data"), xytext=(-62, 0),
                    textcoords="offset points", ha="left", va="bottom", fontsize=7, fontweight="bold", color=INK,
                    bbox=dict(fc="white", ec="none", pad=0.4))
        y += 0.75
        for w_, h, b in ws:
            bx.barh(y - 0.16, 100 * h / n_words, height=0.3, color=LIGHT, zorder=2)
            bx.barh(y + 0.16, 100 * b / n_words, height=0.3, color=ORANGE, zorder=2)
            bx.annotate(f"{h} \u2192 {b}", xy=(100 * max(h, b) / n_words, y), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=P.SMALL, color=MUTED,
                        bbox=dict(fc="white", ec="none", pad=0.4))
            ticks.append((y, w_))
            y += 1
        y += 0.6
    bx.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontstyle="italic")
    bx.set_ylim(y - 0.9, -0.6)
    bx.set_xlim(0, 21)
    bx.set_xticks([0, 10, 20], ["0", "10", "20%"])
    bx.tick_params(axis="y", length=0)
    bx.grid(axis="y", visible=False)
    bx.grid(axis="x", visible=True)
    bx.spines["left"].set_visible(False)
    P.head(bx, "b", "The bias shows in the words",
           f"Dev write-ups using each word, honest (grey)\nthen biased (orange), of {n_words} each", dx=-62)

    # c: within-task AUC on held-out, one row per detector
    for j, (key, name, trained) in enumerate(DETECTORS):
        c = R["probe_within"] if key == "probe" else R["blackbox_within"].get(key)
        col = BLUE if trained else GREY
        cx.plot([c["lo"], c["hi"]], [j, j], color="#9DB3CB" if trained else LIGHT, lw=1.3,
                solid_capstyle="round", zorder=2)
        cx.plot([c["auc"]], [j], ls="", marker="o", ms=4.8, mfc=col, mec=col, mew=1.2, zorder=3)
        P.label_at(cx, c["hi"], j, f"{c['auc']:.2f}", color=col, dx=5, dy=0, ha="left", va="center", size=7)
    cx.set_yticks(range(len(DETECTORS)), [d[1] for d in DETECTORS])
    cx.tick_params(axis="y", length=0)
    for t, (_, _, trained) in zip(cx.get_yticklabels(), DETECTORS):
        t.set_color(INK if trained else MUTED)
    cx.set_ylim(len(DETECTORS) - 0.45, -0.55)
    P.auc_axis(cx, 0.4, 1.0)
    P.head(cx, "c", "Detectors trained on the pairs see the bias; checkers do not",
           "Held-out AUC within tasks, biased write-up against its honest pair")
    P.save(fig, FIG / "probe_steer")


SLANT = ROOT / "runs" / "vetting"


def jar(ax, x: float, k: int, n: int, color: str, top: float, w: float = 0.56, big: float = 9) -> None:
    """A jar, as on the explainer page: a pale column to the axis top, filled
    to k of n, with its Wilson 95% interval, the share above and k of n below."""
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((x - w / 2, 0), w, top, fc="#F4F3F0", ec="none", zorder=1))
    ax.add_patch(Rectangle((x - w / 2, 0), w, 100 * k / n, fc=color, ec="none", zorder=2))
    lo, hi = wilson(k, n)
    ax.plot([x, x], [100 * lo, 100 * hi], color=INK, lw=0.8, zorder=3)
    ax.text(x, 100 * hi + 1.2, f"{100 * k / n:.0f}%", ha="center", va="bottom", fontsize=big, fontweight="bold",
            color=INK)
    ax.annotate(f"{k} of {n}", xy=(x, 0), xytext=(0, -3), textcoords="offset points", ha="center", va="top",
                fontsize=P.SMALL, color=MUTED)


def jar_axis(ax, top: float) -> None:
    ax.set_ylim(0, top)
    ticks = [t for t in (0, 10, 20, 30, 40) if t <= top]
    ax.set_yticks(ticks, [f"{t}%" if t == ticks[-1] else str(t) for t in ticks])
    ax.grid(axis="y", visible=False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def slant() -> None:
    """The bias detector for the recommended configuration: the manipulation check with
    thinking off and with a 1k think, then the flag rate on ordinary write-ups."""
    import matplotlib.pyplot as plt

    think = SLANT / "qwen3.5-27b-rules-think" / "slant"
    off = SLANT / "qwen3.5-27b-rules-off" / "slant-2"
    if not (think / "report.json").exists():
        return
    fig = P.figure_mm(148, 70)
    ax = P.axes_mm(fig, 8, 17, 52, 38)
    bx = P.axes_mm(fig, 76, 17, 50, 38)

    # a: held-out wrong conclusions, honest against biased pair, per configuration
    top = 42
    for j, (d, label) in enumerate(((off, "thinking off"), (think, "1k think (recommended)"))):
        m = json.loads((d / "report.json").read_text())["manipulation"]
        for k_, (arm, col) in enumerate((("honest", GREY), ("steered", ORANGE))):
            kk, n = m[f"heldout_{arm}"]
            jar(ax, 2.6 * j + k_, kk, n, col, top, w=0.78, big=8)
        dd = m["heldout_difference"]
        ax.annotate(f"{label}\n{dd['diff'] * 100:+.0f} points ({dd['lo'] * 100:.0f} to {dd['hi'] * 100:.0f})"
                    .replace("-", "\u2212").replace("1k think \u2212", "1k think -"),
                    xy=(2.6 * j + 0.5, 0), xytext=(0, -14), textcoords="offset points", ha="center", va="top",
                    fontsize=7, color=INK if dd["lo"] > 0 else MUTED, linespacing=1.3)
    ax.set_xticks([])
    ax.set_xlim(-0.6, 4.2)
    jar_axis(ax, top)
    P.head(ax, "a", "Thinking first, it would not bias", "Held-out conclusions judged wrong, honest (grey)\n"
           "and told to bias (orange)")

    # b: ordinary write-ups flagged, per task set, by three detectors
    sets = (("dev", "dev"), ("heldout", "held-out"), ("fresh", "first\nsealed"), ("fresh2", "second\nsealed"))
    three = SLANT / "qwen3.5-27b-rules-off" / "slant-3style"
    dets = ((three, "three styles, thinking off (the test)", BLUE, BLUE, "s", -0.2),
            (off, "one style, thinking off", GREY, "white", "o", 0.0),
            (think, "its own pairs, 1k think", GREY, GREY, "o", 0.2))
    for d, label, col, mfc, mk, dx in dets:
        for i, (key, _) in enumerate(sets):
            f = d / "scores" / f"think1024-greedy-{key}.json"
            if not f.exists():
                continue
            a_ = json.loads(f.read_text())["summary"]["all"]
            bx.plot([i + dx, i + dx], [a_["lo"] * 100, a_["hi"] * 100], color=col if col == BLUE else LIGHT, lw=1.4,
                    solid_capstyle="round", zorder=2)
            bx.plot([i + dx], [a_["rate"] * 100], ls="", marker=mk, ms=4.5, mfc=mfc, mec=col, mew=1.1, zorder=3)
    P.line(bx, 10, "limit", axis="y", outside=True)
    P.line(bx, 5, "expected\nwithout bias", axis="y", color=GREY, outside=True)
    bx.set_xticks(range(len(sets)), [s_[1] for s_ in sets])
    bx.tick_params(axis="x", length=0)
    bx.set_xlim(-0.5, 3.5)
    bx.set_ylim(-0.8, 14)
    bx.set_yticks([0, 5, 10], ["0", "5", "10%"])
    from matplotlib.lines import Line2D
    bx.legend([Line2D([], [], ls="", marker=mk, ms=4.5, mfc=mfc, mec=col, mew=1.1) for _, _, col, mfc, mk, _ in dets],
              [label for _, label, *_ in dets], loc="upper left", bbox_to_anchor=(0, 1.02), handletextpad=0.2,
              borderaxespad=0, labelspacing=0.3, fontsize=P.SMALL)
    P.head(bx, "b", "Its ordinary write-ups are flagged at the honest rate",
           "Ordinary write-ups of the recommended configuration\nflagged by each detector")
    P.save(fig, FIG / "slant_vetting")


def style() -> None:
    """Bias detection under a change of requested style (4.11): manipulation, false alarms, AUC."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    f = ROOT / "runs" / "probe" / "style-v1" / "results" / "style-heldout.json"
    if not f.exists():
        return
    R = json.loads(f.read_text())["conditions"]
    conds = ("default", "concise", "thorough", "disguise")
    fig = P.figure_mm(148, 110)
    ax = P.axes_mm(fig, 15, 64, 54, 32)
    bx = P.axes_mm(fig, 86, 64, 40, 32)
    cx = P.axes_mm(fig, 15, 10, 82, 32)

    # a: wrong conclusions, honest against told to bias, per style
    for j, c in enumerate(conds):
        m = R[c]["manipulation"]
        for k_, (key, col, mk) in enumerate((("honest_wrong", GREY, "o"), ("steered_wrong", ORANGE, "D"))):
            kk, n = m[key]
            lo, hi = wilson(kk, n)
            x = j + (k_ - 0.5) * 0.36
            ax.plot([x, x], [lo * 100, hi * 100], color=LIGHT if col == GREY else "#E8C595", lw=1.6,
                    solid_capstyle="round", zorder=2)
            ax.plot([x], [kk / n * 100], ls="", marker=mk, ms=4.5, color=col, zorder=3)
        ax.annotate(f"{m['diff'] * 100:+.0f}", xy=(j, 0), xytext=(0, -14), textcoords="offset points", ha="center",
                    va="top", fontsize=7, fontweight="bold" if m["lo"] > 0 else "normal",
                    color=INK if m["lo"] > 0 else MUTED)
    ax.annotate("difference,\npoints", xy=(0, 0), xycoords="axes fraction", xytext=(-4, -14),
                textcoords="offset points", ha="right", va="top", fontsize=P.SMALL, color=MUTED)
    ax.set_xticks(range(4), conds)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", labeltop=False, labelbottom=True)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 40)
    ax.set_yticks([0, 10, 20, 30, 40], ["0", "10", "20", "30", "40%"])
    ax.grid(axis="x", visible=False)
    P.head(ax, "a", "Some styles undo the bias", "Conclusions judged wrong, honest (grey)\nand told to bias (orange)")

    # b: honest write-ups flagged by each detector
    for j, c in enumerate(conds[:3]):
        for k_, (det, col, mfc) in enumerate((("current", GREY, "white"), ("mixed", BLUE, BLUE))):
            kk, n = R[c]["honest_flagged"][det]
            lo, hi = wilson(kk, n)
            x = j + (k_ - 0.5) * 0.34
            bx.plot([x, x], [lo * 100, hi * 100], color=LIGHT if col == GREY else BLUE, lw=1.4, solid_capstyle="round",
                    zorder=2)
            bx.plot([x], [kk / n * 100], ls="", marker="o", ms=4.5, mfc=mfc, mec=col, mew=1.1, zorder=3)
    P.line(bx, 10, "limit", axis="y", outside=True)
    P.line(bx, 5, "expected", axis="y", color=GREY, outside=True)
    bx.set_xticks(range(3), conds[:3])
    bx.tick_params(axis="x", length=0)
    bx.set_xlim(-0.5, 2.5)
    bx.set_ylim(0, 24)
    bx.set_yticks([0, 5, 10, 15, 20], ["0", "5", "10", "15", "20%"])
    bx.grid(axis="x", visible=False)
    P.head(bx, "b", "Brevity triples false alarms", "Honest write-ups flagged\nby the text classifier")

    # c: within-task AUC per style
    dets = (("current text detector", "text classifier, default style", GREY, "o", "white"),
            ("mixed text detector", "text classifier, three styles", BLUE, "o", BLUE),
            ("probe (mixed twins)", "probe, three styles", BLUE, "s", "white"),
            ("gpt-oss-20b|effort-medium", "gpt-oss 20B as checker", GREY, "^", GREY))
    for k_, (key, label, col, mk, mfc) in enumerate(dets):
        for j, c in enumerate(conds):
            a = R[c]["auc"].get(key)
            if a is None:
                continue
            x = j + (k_ - 1.5) * 0.15
            cx.plot([x, x], [a["lo"], a["hi"]], color=LIGHT if col == GREY else "#9DB3CB", lw=1.2,
                    solid_capstyle="round", zorder=2)
            cx.plot([x], [a["auc"]], ls="", marker=mk, ms=4.5, mfc=mfc, mec=col, mew=1.1, zorder=3)
    P.line(cx, 0.8, "required", axis="y", outside=True)
    P.line(cx, 0.5, "chance", axis="y", color=GREY, outside=True)
    cx.set_xticks(range(4), conds)
    cx.tick_params(axis="x", length=0)
    cx.set_xlim(-0.6, 3.6)
    cx.set_ylim(0.4, 1.0)
    cx.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cx.grid(axis="x", visible=False)
    cx.legend([Line2D([], [], ls="", marker=mk, ms=4.5, mfc=mfc, mec=col, mew=1.1) for _, _, col, mk, mfc in dets],
              [d[1] for d in dets], loc="center left", bbox_to_anchor=(1.17, 0.5), handletextpad=0.3,
              borderaxespad=0, labelspacing=0.7, fontsize=7, title="Detectors in b and c", title_fontsize=7,
              alignment="left")
    P.head(cx, "c", "Detection holds in every style", "Within-task AUC, biased write-up against its honest pair")
    P.save(fig, FIG / "style_robustness")


def main() -> int:
    reading()
    steer()
    slant()
    style()
    return 0


if __name__ == "__main__":
    sys.exit(main())
