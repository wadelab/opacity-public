"""Results figures for Study 1 and the first Study 2 figure. Reads
runs/summary/*.csv, which scripts in the results document regenerate from
the Inspect logs.

    .venv/bin/python docs/figures/make_results_figures.py

Writes docs/figures/fig_hosted_levels.{pdf,png}, fig_spine.{pdf,png} and,
when runs/summary/harvest_by_model.csv exists, fig_harvest.{pdf,png}.
The look is the paper's (paperfig.py): blue for detection, orange for what
goes wrong, grey for everything else, dashed ink for thresholds.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import paperfig as P

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Reviewers in Table 3's order: (run, model, name as the paper writes it).
REVIEWERS = [
    ("hosted-new", "deepseek-v4.1-flash", "DeepSeek flash"),
    ("hosted-new", "mimo-v2.6-pro", "MiMo pro"),
    ("sweep4-new-nothink", "Qwen3.5-27B", "Qwen3.5 27B"),
    ("sweep5-new-nothink", "Mistral-Small-3.2-24B-Instruct-2506-NVFP4", "Mistral Small 24B"),
    ("sweep5-new-nothink", "Llama-3.3-70B-Instruct-NVFP4", "Llama 3.3 70B"),
    ("sweep5-new-nothink", "Qwen3-32B", "Qwen3 32B"),
    ("sweep4-new-nothink", "Qwen3.5-9B", "Qwen3.5 9B"),
    ("sweep4-new-nothink", "Qwen3.5-4B", "Qwen3.5 4B"),
    ("sweep4-new-nothink", "Qwen3.5-2B", "Qwen3.5 2B"),
    ("sweep4-new-nothink", "Qwen3.5-0.8B", "Qwen3.5 0.8B"),
]
QWEN35 = ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B"]
GATE_DET, GATE_FA = 0.8, 0.2  # the detection threshold (methods.tex)


def load(name: str) -> list[dict]:
    with open(ROOT / "runs" / "summary" / name, newline="") as fh:
        return [{k: (float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v) for k, v in r.items()} for r in csv.DictReader(fh)]


def pick(rows: list[dict], run: str, model: str, level: int) -> dict:
    hits = [r for r in rows if r["run"] == run and r["model"] == model and r["level"] == level]
    assert len(hits) == 1, (run, model, level, len(hits))
    return hits[0]


def interval(ax, k: float, n: float, y: float, color: str) -> None:
    lo, hi = P.wilson(k, n)
    ax.plot([lo, hi], [y, y], color=color, lw=0.8, solid_capstyle="butt", zorder=2)


# --------------------------------------------------------------------------
def fig_spine(rows: list[dict]) -> None:
    """(a) Detection and false alarm at s0 for every reviewer, against the
    detection threshold. (b) Detection at s0 and at s4, prose only."""
    names = [name for _, _, name in REVIEWERS]
    ys = [i + (0.5 if i >= 2 else 0) for i in range(len(REVIEWERS))]  # a gap after the hosted pair
    fig = P.figure_mm(148, 84)
    axs = [P.axes_mm(fig, 24, 13, 54, 62), P.axes_mm(fig, 91, 13, 45, 62)]

    # a: one row per reviewer; detection (blue) above false alarm (orange)
    ax = axs[0]
    dodge = 0.17
    for y, (run, model, name) in zip(ys, REVIEWERS):
        r = pick(rows, run, model, 0)
        det, fa, ns, nc = r["detection"], r["false_alarm"], r["n_seeded"], r["n_control"]
        interval(ax, det * ns, ns, y - dodge, P.BLUE)
        interval(ax, fa * nc, nc, y + dodge, P.ORANGE)
        ax.plot(det, y - dodge, "o", color=P.BLUE, ms=4, zorder=3)
        ax.plot(fa, y + dodge, "s", color=P.ORANGE, ms=3.6, zorder=3)
        # pooled numbers inside the corner can still miss a class
        missed = [c for c in range(1, 9) if r[f"det_c{c}"] < GATE_DET]
        if det >= GATE_DET and fa <= GATE_FA and missed:
            text = "misses class %d" % missed[0] if len(missed) == 1 else "misses classes " + ", ".join(map(str, missed[:-1])) + f" and {missed[-1]}"
            free = (max(P.wilson(fa * nc, nc)[1], GATE_FA), P.wilson(det * ns, ns)[0])  # between the limit and the interval
            P.label_at(ax, sum(free) / 2, y, text, color=P.MUTED, dy=0, va="center")
    top, bottom = ys[0] - 0.4, ys[-1] + 0.6
    for x in (GATE_FA, GATE_DET):
        ax.vlines(x, top, bottom, color=P.INK, lw=0.7, ls=(0, (4, 2.5)), zorder=1)
    P.label_at(ax, GATE_FA, bottom, "false-alarm limit", dy=-2, va="top")
    P.label_at(ax, GATE_DET, bottom, "detection threshold", dy=-2, va="top")
    first = pick(rows, *REVIEWERS[0][:2], 0)
    P.label_at(ax, first["detection"], ys[0] - dodge, "detection", color=P.BLUE, dx=3, dy=4, ha="right")
    P.label_at(ax, first["false_alarm"], ys[0] - dodge, "false alarm", color=P.ORANGE, dy=4)
    meets = {name for (run, model, name) in REVIEWERS
             if all(pick(rows, run, model, 0)[f"det_c{c}"] >= GATE_DET for c in range(1, 9))
             and pick(rows, run, model, 0)["false_alarm"] <= GATE_FA}
    for a in axs:
        a.set_yticks(ys, names if a is ax else [])
        a.set_ylim(bottom + 0.75, top - 0.75)
        a.tick_params(axis="y", length=0)
        a.grid(False)  # the threshold lines and the printed values do the gridlines' work
        a.spines["left"].set_visible(False)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold" if t.get_text() in meets else "normal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.2), ["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_xlabel("Share of reports flagged at s0")
    P.head(ax, "a", "Only the hosted reviewers meet the threshold")

    # b: detection with every table (s0, filled) and with prose only (s4, open)
    ax = axs[1]
    for y, (run, model, name) in zip(ys, REVIEWERS):
        d0, d4 = pick(rows, run, model, 0)["detection"], pick(rows, run, model, 4)["detection"]
        ax.plot([d4, d0], [y, y], color=P.LIGHT, lw=1.2, zorder=1, solid_capstyle="butt")
        ax.plot(d4, y, "o", mfc="white", mec=P.BLUE, mew=1.1, ms=4.2 if d4 != d0 else 6.4, zorder=3)
        ax.plot(d0, y, "o", color=P.BLUE, ms=4.2, zorder=4)  # inside the ring when the two are equal
        right = d4 >= d0
        P.label_at(ax, d4, y, f"{d4:.2f}", color=P.MUTED, dx=5 if right else -5, dy=0,
                   ha="left" if right else "right", va="center")
    d0, d4 = pick(rows, *REVIEWERS[-1][:2], 0)["detection"], pick(rows, *REVIEWERS[-1][:2], 4)["detection"]
    P.label_at(ax, d0, bottom, "all tables, s0", color=P.BLUE, dy=-2, va="top")
    P.label_at(ax, d4, bottom, "prose only, s4", color=P.BLUE, dx=4, dy=-2, ha="right", va="top")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.2), ["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_xlabel("Planted-error reports flagged")
    P.head(ax, "b", "Prose alone carries most hosted detection")
    P.save(fig, HERE / "fig_spine")


# --------------------------------------------------------------------------
def fig_hosted_levels(rows: list[dict]) -> None:
    """Detection, false alarm and cannot-tell against disclosure level for the
    two hosted reviewers."""
    models = [("deepseek-v4.1-flash", "a", "DeepSeek flash: detection falls at s3"),
              ("mimo-v2.6-pro", "b", "MiMo pro: prose alone keeps detection at 0.72")]
    series = (("detection", P.BLUE, "-", "o", "detection"),
              ("false_alarm", P.ORANGE, (0, (4, 2)), "s", "false alarm"),
              ("cannot_tell_seeded", P.GREEN, (0, (1, 1.5)), "^", "cannot tell"))
    fig = P.figure_mm(148, 58)
    axs = [P.axes_mm(fig, 9, 13, 50, 34), P.axes_mm(fig, 84, 13, 50, 34)]
    for ax, (model, letter, title) in zip(axs, models):
        new = sorted([r for r in rows if r["run"] == "hosted-new" and r["model"] == model], key=lambda r: r["level"])
        n = new[0]["n_seeded"]
        x = np.array([r["level"] for r in new])
        ax.axvspan(3.55, 4.45, color="#F2F1EE", lw=0, zorder=0)
        ends = []
        for key, color, ls, mk, label in series:
            y = np.array([r[key] for r in new])
            lo, hi = zip(*[P.wilson(v * n, n) for v in y])
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0, zorder=0)
            ax.plot(x[:4], y[:4], color=color, ls=ls, lw=1.4, marker=mk, ms=3.2, zorder=3)
            ax.plot(x[3:], y[3:], color=color, ls=ls, lw=1.0, marker=mk, ms=3.2, zorder=3, alpha=0.6)
            ends.append((4.45, y[-1], label, color))
        P.end_labels(ax, ends, gap=0.08, dx=4)
        ax.set_xticks([0, 1, 2, 3, 4], ["s0", "s1", "s2", "s3", "s4\nprose only"])
        ax.set_xlim(-0.25, 4.45)
        ax.set_ylim(-0.03, 1.1)
        ax.set_yticks(np.arange(0, 1.01, 0.2), ["0", "0.2", "0.4", "0.6", "0.8", "1.0"] if letter == "a" else [])
        if letter == "b":
            ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Disclosure level")
        P.head(ax, letter, title, "Share of reports")
    P.save(fig, HERE / "fig_hosted_levels")


# --------------------------------------------------------------------------
def fig_harvest(rows: list[dict], checker: list[dict]) -> None:
    """(a) Commit rate per invited class against Qwen3.5 size. (b) Wrong
    conclusions as an analyst against detection and false alarm as a reviewer
    at s0, for the same sizes."""
    present = [m for m in QWEN35 if any(r["model"] == m for r in rows)]
    x = np.arange(len(present))
    ticks = [m.replace("Qwen3.5-", "") for m in present]
    fig = P.figure_mm(148, 58)
    axs = [P.axes_mm(fig, 9, 10, 50, 37), P.axes_mm(fig, 84, 10, 44, 37)]

    # a: class 1 in blue; the other seven are context, in one grey
    ax = axs[0]
    commit = {c: np.array([next(r[f"commit_c{c}"] for r in rows if r["model"] == m) for m in present]) for c in range(1, 9)}
    for c in range(2, 9):
        ax.plot(x, commit[c], color=P.LIGHT, lw=0.9, zorder=2)
    ax.plot(x, commit[1], color=P.BLUE, lw=1.6, marker="o", ms=3.8, zorder=3)
    edge = len(x) - 1 + 0.3  # labels start at the axes' edge, clear of the gridlines
    P.end_labels(ax, [(edge, commit[1][-1], "multiplicity\n(class 1)", P.BLUE),
                      (edge, float(np.mean([commit[c][-1] for c in range(2, 9)])), "classes 2 to 8", P.GREY)],
                 gap=0.1, dx=2)
    ax.set_xticks(x, ticks)
    ax.set_xlim(-0.3, len(x) - 1 + 0.3)
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2), ["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_xlabel("Qwen3.5 size")
    P.head(ax, "a", "Errors fall with size, except multiplicity", "Share of planted-error tasks with the invited error")

    # b: the same sizes as analyst and as reviewer
    ax = axs[1]
    lvl0 = {m: pick(checker, "sweep4-new-nothink", m, 0) for m in present}
    wrong = np.array([next(r["conclusion_wrong_seeded"] for r in rows if r["model"] == m) for m in present])
    det = np.array([lvl0[m]["detection"] for m in present])
    fa = np.array([lvl0[m]["false_alarm"] for m in present])
    ax.plot(x, fa, color=P.GREY, ls=(0, (1, 1.5)), lw=1.1, marker="o", mfc="white", mew=0.9, ms=3, zorder=2)
    ax.plot(x, wrong, color=P.ORANGE, ls=(0, (4, 2)), lw=1.4, marker="s", ms=3.4, zorder=3)
    ax.plot(x, det, color=P.BLUE, lw=1.4, marker="o", ms=3.6, zorder=3)
    P.end_labels(ax, [(edge, det[-1], "detection\nas reviewer", P.BLUE),
                      (edge, wrong[-1], "wrong conclusions\nas analyst", P.ORANGE),
                      (edge, fa[-1], "false alarms\nas reviewer", P.GREY)], gap=0.19, dx=2, lo=0.04)
    ax.set_xticks(x, ticks)
    ax.set_xlim(-0.3, len(x) - 1 + 0.3)
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2), ["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_xlabel("Qwen3.5 size")
    P.head(ax, "b", "Errors fall with size; only the 27B reviews well", "Share of planted-error tasks or reports, at s0")
    P.save(fig, HERE / "fig_harvest")


if __name__ == "__main__":
    checker = load("checker_by_level.csv")
    fig_spine(checker)
    fig_hosted_levels(checker)
    if (ROOT / "runs" / "summary" / "harvest_by_model.csv").exists():
        fig_harvest(load("harvest_by_model.csv"), checker)
    print("figures written to", HERE)
