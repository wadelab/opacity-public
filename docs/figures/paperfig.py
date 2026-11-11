"""The paper's figure look, shared by every figure script (paper/STYLE.md, Figures).

    import paperfig as P        # scripts outside docs/figures put it on sys.path first

Importing applies wadelab.mplstyle with the paper's sans and the type floor
below. Each colour carries one meaning across the paper, as on the explainer
page (docs/explainers/reading-the-robots-mind.html):

    BLUE    what the panel is about: the recommended analyst, the trained
            detector, detection
    ORANGE  what goes wrong: biased write-ups, wrong conclusions, false alarms
    GREY    everything that is not the point: baselines, honest write-ups
            when they are the baseline, other models
    LIGHT   context: intervals of grey series, clouds of configurations
    GREEN   a third outcome where one is needed (cannot tell)
    INK     thresholds and limits, dashed, with their name written beside them

Red and purple are left for the rare panel that needs a fourth series. A
series that differs from another only by colour also differs by marker,
fill or dash, or is labelled where it is drawn.

Text rules: nothing below 6.5 pt, nothing rotated, no figure-level title.
A panel's title states its finding and the grey line under it names the
measure, in place of a rotated y-axis label (`head`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "wadelab.mplstyle"))
plt.rcParams["font.sans-serif"] = ["Linux Biolinum O"] + list(plt.rcParams["font.sans-serif"])
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.titlesize": 8, "legend.frameon": False})

MM = 1 / 25.4
SINGLE, DOUBLE = 89 * MM, 148 * MM  # DOUBLE is the paper's text width, not a journal column

BLUE, ORANGE, GREEN, RED, PURPLE = "#33608C", "#D08C34", "#4F8C6B", "#A34E5E", "#6E5B8E"
GREY, LIGHT, INK, MUTED, FAINT = "#8A8578", "#BFBDB6", "#262626", "#6B6A66", "#D8D6D1"
SMALL = 6.5  # the smallest type in any figure, in points


def head(ax, letter: str, finding: str, measure: str = "", dx: float = 0.0) -> None:
    """The panel's finding as a bold title, the measure in grey under it (one
    line or several), and the panel letter level with the title, outside the
    axes. dx (points) moves all three left or right of the axes' edge, for a
    panel whose row labels start further out; it needs a laid-out figure
    (figure_mm), not constrained layout."""
    lines = measure.count("\n") + 1 if measure else 0
    title = ax.set_title(finding + "\n" * lines, loc="left", pad=6)
    if dx:
        title.set_x(dx / (ax.get_position().width * ax.figure.get_figwidth() * 72))
    if measure:
        ax.annotate(measure, xy=(0, 1), xycoords="axes fraction", xytext=(dx, 6), textcoords="offset points",
                    ha="left", va="bottom", fontsize=7, color=MUTED)
    if letter:
        ax.annotate(letter, xy=(0, 1), xycoords=title, xytext=(-6, 0), textcoords="offset points",
                    fontsize=9, fontweight="bold", ha="right", va="top")


def spread(ys: list[float], gap: float, lo: float = -np.inf, hi: float = np.inf) -> list[float]:
    """Move label positions apart until neighbours are at least `gap` apart,
    staying as close as possible to where they started and inside [lo, hi].
    Returns positions in the input order."""
    order = np.argsort(ys)
    y = np.asarray(ys, dtype=float)[order]
    for _ in range(200):
        moved = False
        for i in range(1, len(y)):
            d = y[i] - y[i - 1]
            if d < gap - 1e-9:
                push = (gap - d) / 2
                y[i - 1] -= push
                y[i] += push
                moved = True
        y = np.clip(y, lo, hi)
        if not moved:
            break
    out = np.empty_like(y)
    out[order] = y
    return list(out)


def end_labels(ax, items: list[tuple[float, float, str, str]], gap: float, dx: float = 4,
               lo: float = -np.inf, hi: float = np.inf, size: float = SMALL) -> None:
    """Direct labels at the right-hand end of lines, spread so none overlap.
    items: (x, y, text, colour) in data coordinates; gap in data units of y."""
    ys = spread([it[1] for it in items], gap, lo, hi)
    for (x, _, text, col), ty in zip(items, ys):
        ax.annotate(text, xy=(x, ty), xytext=(dx, 0), textcoords="offset points", fontsize=size, color=col,
                    va="center", ha="left", annotation_clip=False)


def label_at(ax, x: float, y: float, text: str, color: str = INK, dx: float = 0, dy: float = 5,
             ha: str = "center", va: str = "bottom", size: float = SMALL, **kw) -> None:
    """A value or name written next to a mark, offset in points."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=size,
                color=color, annotation_clip=False, **kw)


def line(ax, value: float, text: str = "", axis: str = "x", color: str = INK, dashed: bool = True,
         where: float = 1.0, ha: str = "left", va: str = "top", outside: bool = False) -> None:
    """A threshold, limit or chance line with its name written beside it.
    `where` is the axes fraction along the line at which the name sits;
    outside=True writes the name just past the axes' right edge (axis="y")
    or top edge (axis="x"), clear of the data."""
    ls = (0, (4, 2.5)) if dashed else "-"
    if outside:
        if axis == "x":
            ax.axvline(value, color=color, lw=0.7, ls=ls, zorder=1)
            ax.annotate(text, xy=(value, 1), xycoords=("data", "axes fraction"), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=SMALL, color=color)
        else:
            ax.axhline(value, color=color, lw=0.7, ls=ls, zorder=1)
            ax.annotate(text, xy=(1, value), xycoords=("axes fraction", "data"), xytext=(3, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=SMALL, color=color)
        return
    if axis == "x":
        ax.axvline(value, color=color, lw=0.7, ls=ls, zorder=1)
        if text:
            ax.annotate(text, xy=(value, where), xycoords=("data", "axes fraction"), xytext=(3, 0),
                        textcoords="offset points", ha=ha, va=va, fontsize=SMALL, color=color)
    else:
        ax.axhline(value, color=color, lw=0.7, ls=ls, zorder=1)
        if text:
            ax.annotate(text, xy=(where, value), xycoords=("axes fraction", "data"), xytext=(0, 2),
                        textcoords="offset points", ha=ha if ha != "left" else "right", va="bottom",
                        fontsize=SMALL, color=color)


def auc_axis(ax, lo: float = 0.4, hi: float = 1.0, label: str = "") -> None:
    """An AUC axis from lo to 1 with 'chance' under 0.5 and 'perfect' under 1."""
    ax.set_xlim(lo, hi + 0.005)
    ticks = np.round(np.arange(lo, hi + 1e-9, 0.1), 1)
    ax.set_xticks(ticks, [f"{t:.1f}" for t in ticks])
    for v, word in ((0.5, "chance"), (1.0, "perfect")):
        if lo <= v <= hi:
            ax.annotate(word, xy=(v, 0), xycoords=("data", "axes fraction"), xytext=(0, -15),
                        textcoords="offset points", ha="center", va="top", fontsize=SMALL, color=MUTED)
    ax.axvline(0.5, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    if label:
        ax.set_xlabel(label, labelpad=12)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)


def rows(ax, labels: list[str], bold: set[str] | None = None) -> None:
    """Categorical rows on the y axis, top to bottom, no tick marks."""
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    for t in ax.get_yticklabels():
        if bold and t.get_text() in bold:
            t.set_fontweight("bold")


def figure_mm(width: float, height: float):
    """A figure laid out by hand: place each axes with axes_mm. For composite
    figures that constrained layout cannot settle."""
    fig = plt.figure(figsize=(width * MM, height * MM))
    fig.set_layout_engine("none")
    fig._mm = (width, height)
    return fig


def axes_mm(fig, left: float, bottom: float, width: float, height: float, **kw):
    """Axes at a position given in millimetres from the figure's lower left corner."""
    W, H = fig._mm
    return fig.add_axes((left / W, bottom / H, width / W, height / H), **kw)


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def save(fig, path_stem: Path | str) -> None:
    """PDF for the paper, PNG for pages and review, both from one drawing."""
    p = Path(path_stem)
    kw = {}
    if hasattr(fig, "_mm"):  # laid out by hand: keep the page exactly as designed
        # bbox_inches=None would fall back to the style's "tight" and crop the page
        kw = {"bbox_inches": fig.bbox_inches}
        fig.canvas.draw()
        bb = fig.get_tightbbox().transformed(fig.dpi_scale_trans) .transformed(fig.transFigure.inverted())
        if bb.x0 < -0.002 or bb.y0 < -0.002 or bb.x1 > 1.002 or bb.y1 > 1.002:
            print(f"WARNING {p.name}: drawing runs off the page "
                  f"({bb.x0 * fig._mm[0]:.1f} to {bb.x1 * fig._mm[0]:.1f} mm wide, "
                  f"{bb.y0 * fig._mm[1]:.1f} to {bb.y1 * fig._mm[1]:.1f} mm high)")
    fig.savefig(p.with_suffix(".pdf"), **kw)
    fig.savefig(p.with_suffix(".png"), dpi=300, **kw)
    plt.close(fig)
    print("wrote", p.with_suffix(".pdf"))
