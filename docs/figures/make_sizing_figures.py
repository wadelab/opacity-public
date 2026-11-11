"""Figures for the model-sizing study. Reads runs/summary/sizing/*.csv
(scripts/sizing_report.py) and writes docs/figures/sizing_*.{pdf,png}.

    .venv/bin/python docs/figures/make_sizing_figures.py

    sizing_prompt      wrong releases per analyst under the four prompts,
                       dev and held-out (paper Figure 7)
    sizing_thinking    the effect of a thinking budget on the checker (s0
                       detection, false alarm, class 6) and on the analyst
                       (Figure 8)
    sizing_fresh       the first sealed set against dev and held-out
                       (Figure 9)
    sizing_analyst     wrong releases on planted-error tasks and correct
                       answers released, per analyst and instruction,
                       thinking off (Figure S3)
    sizing_checker     per checker configuration: planted errors in the
                       scripted corpus at s0 beside the analysts' own errors
                       in released write-ups (Figure S4)
    sizing_frontier    wrong releases against GPU-seconds per task for every
                       analyst-plus-checks configuration that answers at
                       least 0.75 of controls (Figure S5)

The look is the paper's (paperfig.py): blue for what the panel is about,
orange for what goes wrong, grey for the rest, thresholds dashed in ink with
their names beside them. Every harvest panel uses the corrected packs (v2)
and the dev tasks only. A panel with no rows yet says so rather than falling
back to the old packs, whose write-ups are a different set of items. Titles
that state a count are computed from the data, so they stay true as runs
land.
"""

from __future__ import annotations

import colorsys
import csv
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import offset_copy

import paperfig as P
from paperfig import BLUE, DOUBLE, FAINT, GREY, INK, LIGHT, MM, MUTED, SMALL

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUMMARY = ROOT / "runs" / "summary" / "sizing"

# total and active parameters (billions)
SIZE = {
    "Qwen3.5-0.8B": (0.8, 0.8), "Qwen3.5-2B": (2, 2), "Qwen3.5-4B": (4, 4), "Qwen3.5-9B": (9, 9),
    "Qwen3.5-27B": (27, 27), "Qwen3.5-35B-A3B": (35, 3), "Qwen3.5-122B-A10B-NVFP4": (122, 10),
    "Qwen3-32B": (32, 32), "Mistral-Small-3.2-24B-Instruct-2506-NVFP4": (24, 24),
    "Llama-3.3-70B-Instruct-NVFP4": (70, 70), "gpt-oss-20b": (21, 3.6), "gpt-oss-120b": (117, 5.1),
    "gemma-4-31B-it": (31, 31), "gemma-4-12B-it": (12, 12), "gemma-4-26B-A4B-it": (26, 4), "typed": (26, 4),
}
# sizes for row labels only: SIZE also decides which checker rows Figure S4 shows
ACTIVE = {**SIZE, "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": (120, 12),
          "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4": (30, 3)}
# model names as the paper writes them
NAME = {
    "Qwen3.5-122B-A10B-NVFP4": "Qwen3.5 122B-A10B", "Qwen3-32B": "Qwen3 32B",
    "Mistral-Small-3.2-24B-Instruct-2506-NVFP4": "Mistral Small 24B", "Llama-3.3-70B-Instruct-NVFP4": "Llama 3.3 70B",
    "gpt-oss-20b": "gpt-oss 20B", "gpt-oss-120b": "gpt-oss 120B", "gemma-4-31B-it": "Gemma 4 31B",
    "gemma-4-12B-it": "Gemma 4 12B", "gemma-4-26B-A4B-it": "Gemma 4 26B-A4B", "typed": "typed reviewer",
    "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": "Nemotron 3 Super 120B",
    "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4": "Nemotron 3.5 Lightning 30B",
}
# row order: the Qwen3.5 size series, then the other families
ORDER = ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B", "Qwen3.5-35B-A3B",
         "Qwen3.5-122B-A10B-NVFP4", "Qwen3-32B", "Mistral-Small-3.2-24B-Instruct-2506-NVFP4",
         "Llama-3.3-70B-Instruct-NVFP4", "gpt-oss-20b", "gpt-oss-120b", "typed", "gemma-4-26B-A4B-it",
         "gemma-4-31B-it", "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"]
# models with no thinking switch: their checker configs are logged as "off"
# by the sweeps and "default" by the plans, which is the same thing
NO_THINK = {"Mistral-Small-3.2-24B-Instruct-2506-NVFP4", "Llama-3.3-70B-Instruct-NVFP4"}
VARIANT = {"off": "", "default": "", "think0": "", "think512": "512 think", "think1024": "1k think",
           "think2048": "2k think", "think4096": "4k think", "effort-low": "low effort", "effort-medium": "medium effort",
           "effort-high": "high effort", "checklist": "checklist", "stepwise": "stepwise", "pack": "with tables"}
ARM_WORDS = {"open": "", "closed": "decline option", "selfcheck": "self-check", "revise": "revise"}
OFF = ("off", "default", "effort-low")


def label(model: str) -> str:
    return NAME.get(model, model.replace("Qwen3.5-", "Qwen3.5 "))


def row_label(model: str) -> str:
    """The model's name, with its active parameters on a second line when
    it is a mixture of experts."""
    total, active = ACTIVE.get(model, (0, 0))
    return label(model) + ("" if total == active else f"\n{active:g}B active")


def config_label(cfg: str) -> str:
    """'Qwen3.5-27B|off|stepwise' -> 'Qwen3.5 27B, stepwise'."""
    model, *rest = cfg.split("|")
    words = [VARIANT.get(p, p) for p in rest]
    return ", ".join([label(model)] + [w for w in words if w])


def canonical(cfg: str) -> str:
    model, *rest = cfg.split("|")
    if model in NO_THINK:
        rest = ["default" if p == "off" else p for p in rest]
    return "|".join([model] + rest)


def rank(model: str) -> int:
    return ORDER.index(model) if model in ORDER else len(ORDER)


def load(name: str) -> list[dict]:
    p = SUMMARY / name
    if not p.exists():
        return []
    out = []
    for r in csv.DictReader(open(p, newline="")):
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


def num(x) -> float:
    return float(x) if isinstance(x, float) else float("nan")


def parts(cfg: str) -> dict:
    p = cfg.split("|")
    return {"model": p[0], "arm": p[1], "think": p[2], "checklist": "checklist" in p,
            "heldout": "heldout" in p, "pack": p[-1]}


def prompt_of(cfg: str) -> str:
    p = cfg.split("|")
    return "checklist" if "checklist" in p else "rules" if "rules" in p else "plain"


def dev_v2(cfg: str) -> bool:
    q = parts(cfg)
    return q["pack"] == "v2" and not q["heldout"]


def share_axis(ax, lo: float, hi: float) -> None:
    """A 0-to-1 share on x, ticks written 0, 0.1, ... without trailing zeros."""
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".") if v else "0"))


BACK = dict(boxstyle="square,pad=0.12", fc="white", ec="none")  # behind a label, so grid lines stop at it


def threshold(ax, x: float, text: str, left: bool = False, where: float = 1.0, va: str = "top") -> None:
    """A vertical threshold drawn by paperfig.line, its name written here on
    a white backing so grid lines stop at it; `left` puts the name on the
    line's left (paperfig.line always writes it 3 pt to the right, so
    ha="right" there puts the text across the line)."""
    P.line(ax, x, "")
    ax.annotate(text, xy=(x, where), xycoords=("data", "axes fraction"), xytext=(-3 if left else 3, 0),
                textcoords="offset points", ha="right" if left else "left", va=va, fontsize=SMALL, color=INK,
                bbox=BACK)


def key_below(ax, handles: list, ncol: int, drop: float = 16) -> None:
    """A frameless key under one panel, `drop` points below the axes, for
    panels whose series are too close together to label where they are."""
    ax.legend(handles=handles, loc="upper left", ncol=ncol, borderaxespad=0, handletextpad=0.3,
              columnspacing=1.0, bbox_to_anchor=(0, 0), bbox_transform=offset_copy(ax.transAxes, ax.figure,
                                                                                  y=-drop, units="points"))


def dot(ax, x: float, y: float, marker: str, colour: str, face: str, size: float = 4, z: int = 3):
    return ax.plot(x, y, marker, color=colour, mfc=face, mec=colour, mew=0.8, ms=size, zorder=z)[0]


def interval(ax, lo: float, hi: float, y: float, colour: str) -> None:
    ax.plot([lo, hi], [y, y], "-", color=colour, lw=0.7, solid_capstyle="butt", zorder=2)


# ---------------------------------------------------------------- Figure S3

ARMS = [  # (arm, prompt, name, marker, colour, face): the analyst instructions without a prompt change
    ("open", "plain", "standard instruction", "o", GREY, GREY),
    ("open", "checklist", "standard + checklist", "D", GREY, LIGHT),
    ("closed", "plain", "decline option", "s", GREY, "white"),
    ("selfcheck", "plain", "self-check", "^", BLUE, BLUE),
]


def fig_analyst(rows: list[dict]) -> None:
    rows = [r for r in rows if dev_v2(r["config"]) and parts(r["config"])["think"] in OFF]
    models = sorted({parts(r["config"])["model"] for r in rows}, key=rank)
    if not models:
        return
    n = len(models)
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, (34 + 9.5 * max(n, 3)) * MM), sharey=True,
                            gridspec_kw={"wspace": 0.08})
    def rate(m: str, arm: str) -> float:
        r = next((r for r in rows if parts(r["config"])["model"] == m and parts(r["config"])["arm"] == arm
                  and prompt_of(r["config"]) == "plain"), None)
        return float("nan") if r is None else r["wrong_release_seeded"]
    pairs = [(rate(m, "open"), rate(m, "selfcheck")) for m in models]
    pairs = [(o, c) for o, c in pairs if not (np.isnan(o) or np.isnan(c))]
    helped = sum(round(c, 2) < round(o, 2) for o, c in pairs)  # lower as the tables print it, not by a hair
    handles: dict[str, object] = {}
    for ax, key, letter, finding, measure in (
            (axs[0], "wrong_release_seeded", "a", f"Self-check cuts errors for {helped} of {len(pairs)} analysts",
             "Wrong conclusions released, planted-error tasks"),
            (axs[1], "coverage_all", "b", "It is paid for in coverage", "Correct answers released, all tasks")):
        P.rows(ax, [row_label(m) for m in models])
        ax.set_ylim(n - 0.5, -0.9)
        share_axis(ax, -0.02, 1.02)
        for j, (arm, prompt, name, mk, col, face) in enumerate(ARMS):
            dy = (j - 1.5) * 0.19
            for i, m in enumerate(models):
                r = next((r for r in rows if parts(r["config"])["model"] == m and parts(r["config"])["arm"] == arm
                          and prompt_of(r["config"]) == prompt), None)
                if r is None:
                    continue
                if key == "wrong_release_seeded":
                    interval(ax, num(r["lo"]), num(r["hi"]), i + dy, BLUE if col == BLUE else LIGHT)
                handles.setdefault(name, dot(ax, r[key], i + dy, mk, col, face))
        P.head(ax, letter, finding, measure)
    threshold(axs[0], 0.05, "release threshold")
    axs[0].set_xlabel("Share of planted-error attempts")
    axs[1].set_xlabel("Share of all attempts")
    fig.legend(list(handles.values()), list(handles), loc="outside lower center", ncol=4)
    P.save(fig, HERE / "sizing_analyst")


# ---------------------------------------------------------------- Figure S4

def fig_checker(checkers: list[dict], matrix: list[dict]) -> None:
    # one seed-0, five-replicate row per configuration at s0; this study's
    # plans first, then the thinking-off sweeps on the same corpus design
    # (sweep4, sweep5) for configurations they lack
    s0: dict[str, dict] = {}
    for r in sorted((r for r in checkers if r["level"] == 0.0 and r["register"] == "default" and r["n_seeded"] == 40
                     and "seed1" not in str(r["tags"]) and r["config"].split("|")[0] in SIZE
                     and ("sizing" in str(r["tags"]) or str(r["tags"]).startswith(("sweep4-", "sweep5-")))),
                    key=lambda r: "sizing" in str(r["tags"])):
        s0[canonical(r["config"])] = r
    pooled = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for r in matrix:
        if not dev_v2(r["analyst"]):
            continue
        k = canonical(r["checker"])
        pooled[k][0] += num(r["withhold_on_wrong"]) * r["n_wrong"] if r["n_wrong"] else 0
        pooled[k][1] += r["n_wrong"]
        pooled[k][2] += num(r["withhold_on_right"]) * r["n_right"] if r["n_right"] else 0
        pooled[k][3] += r["n_right"]
    pooled = {k: v for k, v in pooled.items() if v[1] and v[3] and k.split("|")[0] in SIZE}
    cfgs = sorted(set(s0) | set(pooled), key=lambda c: (rank(c.split("|")[0]), c))
    if not cfgs:
        return
    meets = {c for c, r in s0.items() if r["gate"] == "True"}
    n = len(cfgs)
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, (34 + 3.4 * n) * MM), sharey=True,
                            gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.06})
    labels = [config_label(c) for c in cfgs]
    for ax in axs:
        P.rows(ax, labels, bold={config_label(c) for c in meets})
        ax.set_ylim(n - 0.5, -2.3)
        share_axis(ax, -0.04, 1.03)
        for i in range(1, n):  # a faint rule between model families and sizes
            if cfgs[i].split("|")[0] != cfgs[i - 1].split("|")[0]:
                ax.axhline(i - 0.5, color=FAINT, lw=0.5, zorder=0)
    for t, c in zip(axs[0].get_yticklabels(), cfgs):
        t.set_color(BLUE if c in meets else INK)

    ax = axs[0]
    for i, c in enumerate(cfgs):
        r = s0.get(c)
        if r is None:
            ax.text(0.5, i, "not run", fontsize=SMALL, color=MUTED, va="center", ha="center", bbox=BACK)
            continue
        col = BLUE if c in meets else GREY
        ax.plot([r["false_alarm"], r["detection"]], [i, i], "-", color=LIGHT, lw=0.8, zorder=1)
        ax.plot(r["min_class_det"], i, "|", color=col, ms=6, mew=1.0, zorder=2)
        dot(ax, r["detection"], i, "o", col, col)
        dot(ax, r["false_alarm"], i, "s", col, "white")
    for x, text, y, ha, dx in ((0.2, "false alarm ≤ 0.2", -1.75, "left", 3), (0.8, "detection ≥ 0.8", -0.95, "right", -3)):
        ax.axvline(x, color=INK, lw=0.7, ls=(0, (4, 2.5)), zorder=1)
        ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", fontsize=SMALL, color=INK,
                    va="center", ha=ha, bbox=BACK)
    P.head(ax, "a", f"{len(meets)} of {len(s0)} meet the detection threshold",
           "Reports at s0, 5 replicates")
    ax.set_xlabel("Share flagged")
    key_below(ax, [plt.Line2D([], [], ls="", marker="o", color=GREY, ms=4, label="detection"),
                   plt.Line2D([], [], ls="", marker="s", color=GREY, mfc="white", mew=0.8, ms=4, label="false alarm"),
                   plt.Line2D([], [], ls="", marker="|", color=GREY, ms=6, mew=1.0, label="lowest class"),
                   plt.Line2D([], [], ls="", marker="o", color=BLUE, ms=4, label="meets the threshold")], ncol=2, drop=27)

    ax = axs[1]
    for i, c in enumerate(cfgs):
        v = pooled.get(c)
        if v is None:
            ax.text(0.5, i, "not run", fontsize=SMALL, color=MUTED, va="center", ha="center", bbox=BACK)
            continue
        col = BLUE if c in meets else GREY
        ax.plot([v[2] / v[3], v[0] / v[1]], [i, i], "-", color=LIGHT, lw=0.8, zorder=1)
        dot(ax, v[0] / v[1], i, "o", col, col)
        dot(ax, v[2] / v[3], i, "s", col, "white")
    P.head(ax, "b", "Own errors are harder", "Analysts' dev write-ups, judged by MiMo pro")
    ax.set_xlabel("Share withheld")
    key_below(ax, [plt.Line2D([], [], ls="", marker="o", color=GREY, ms=4, label="write-ups the judge called wrong"),
                   plt.Line2D([], [], ls="", marker="s", color=GREY, mfc="white", mew=0.8, ms=4,
                              label="write-ups the judge called right")], ncol=1, drop=27)
    P.save(fig, HERE / "sizing_checker")


# ---------------------------------------------------------------- Figure 8

BUDGETS = {"off": 0, "think512": 512, "think1024": 1024, "think2048": 2048, "think4096": 4096}
BUDGET_WORD = {0: "off", 512: "512", 1024: "1k", 2048: "2k", 4096: "4k"}
THINK_MODELS = ["Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B", "Qwen3.5-35B-A3B", "Qwen3.5-122B-A10B-NVFP4"]
THINK_MARKERS = ["o", "s", "D", "^", "v"]


def blue_ramp(n: int) -> list[tuple]:
    """n shades of the palette blue at its own hue and saturation, light to dark."""
    h, _, sat = colorsys.rgb_to_hls(*to_rgb(BLUE))
    return [colorsys.hls_to_rgb(h, light, sat) for light in np.linspace(0.74, 0.2, n)]


def start_labels(ax, items: list[tuple[float, float, str, tuple]], x_text: float, gap: float,
                 lo: float, hi: float) -> None:
    """Names in a column left of the lines, spread so none touch, each with a
    thin leader to the first point of its line. Every line starts at the same
    setting, so no leader crosses a line (paperfig.end_labels labels the
    other end and has no leaders)."""
    ys = P.spread([it[1] for it in items], gap, lo, hi)
    for (x, y, text, col), ty in zip(items, ys):
        ax.annotate(text, xy=(x, y), xytext=(x_text, ty), textcoords="data", fontsize=SMALL, color=col,
                    va="center", ha="right", annotation_clip=False, bbox=BACK,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.5, shrinkA=1, shrinkB=3.5, relpos=(1, 0.5)))


def fig_thinking(checkers: list[dict], analysts: list[dict]) -> None:
    s0 = [r for r in checkers if r["level"] == 0.0 and r["register"] == "default" and "sizing" in str(r["tags"])
          and r["n_seeded"] == 40 and "seed1" not in str(r["tags"])]
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 112 * MM), sharex=True, sharey=True,
                            gridspec_kw={"hspace": 0.12, "wspace": 0.08})
    panels = ((axs[0, 0], "a", "detection", "Detection rises", "As checkers: planted-error reports flagged, s0"),
              (axs[0, 1], "b", "false_alarm", "False alarms fall", "As checkers: correct reports flagged, s0"),
              (axs[1, 0], "c", "det_c6", "Thinking fixes the wrong denominator",
               "As checkers: wrong-denominator reports flagged, s0"),
              (axs[1, 1], "d", None, "The analyst gains less", "As analysts: wrong conclusions released"))
    shades = dict(zip(THINK_MODELS, blue_ramp(len(THINK_MODELS))))
    series = {}
    for ax, letter, key, finding, measure in panels:
        for m in THINK_MODELS:
            if key:
                pts = sorted((BUDGETS[r["config"].split("|")[1]], r[key]) for r in s0
                             if r["config"].split("|")[0] == m and len(r["config"].split("|")) == 2
                             and r["config"].split("|")[1] in BUDGETS)
            else:  # standard instruction, plain prompt: the rules prompt is Figure 7's comparison
                pts = sorted((BUDGETS[parts(r["config"])["think"]], r["wrong_release_seeded"]) for r in analysts
                             if r["model"] == m and dev_v2(r["config"]) and parts(r["config"])["arm"] == "open"
                             and prompt_of(r["config"]) == "plain" and parts(r["config"])["think"] in BUDGETS)
            if len(pts) >= 2:
                series[letter, m] = pts
    used = sorted({b for pts in series.values() for b, _ in pts})
    pos = {b: i for i, b in enumerate(used)}  # budgets as evenly spaced settings
    for ax, letter, key, finding, measure in panels:
        ends = []
        for k, m in enumerate(THINK_MODELS):
            pts = series.get((letter, m))
            if pts is None:
                continue
            xs = [pos[p[0]] for p in pts]
            ax.plot(xs, [p[1] for p in pts], "-", color=shades[m], lw=1.0, zorder=2)
            ax.plot(xs, [p[1] for p in pts], THINK_MARKERS[k], color=shades[m], mec="white", mew=0.5, ms=4.5, zorder=3)
            # ties sort by size, so the largest model's name sits on top
            ends.append((xs[0], pts[0][1] + 1e-6 * k, label(m).replace("Qwen3.5 ", ""), shades[m]))
        start_labels(ax, ends, -0.5, gap=0.09, lo=0.04, hi=1.0)
        ax.set_xlim(-1.2, len(used) - 1 + 0.2)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xticks(range(len(used)), [BUDGET_WORD[b] for b in used])
        ax.tick_params(axis="x", labelbottom=True)
        P.head(ax, letter, finding, measure)
    for ax in axs[1]:
        ax.set_xlabel("Thinking budget (tokens)")
    P.save(fig, HERE / "sizing_thinking")


# ---------------------------------------------------------------- Figure S5

def pipeline_label(analyst: str, pipeline: str) -> str:
    q = parts(analyst)
    a = ", ".join(w for w in [label(q["model"]), ARM_WORDS[q["arm"]], VARIANT.get(q["think"], q["think"]),
                              "checklist" if q["checklist"] else "", "rules" if "rules" in analyst.split("|") else ""]
                  if w)
    if pipeline == "alone":
        return a
    stages = []
    for s in pipeline.split("+"):
        kind, _, cfg = s.partition(":")
        if not cfg:
            stages.append(kind)
        elif kind == "screen":
            stages.append(f"{config_label(cfg)} as screen")
        else:
            stages.append(config_label(cfg))
    return f"{a} → " + " → ".join(stages)


def place_labels(fig, ax, anchors: list[tuple[float, float]], texts: list[str], colour: str,
                 obstacles: list[tuple[np.ndarray, float]], blocked: list) -> None:
    """Write each text next to its anchor at the nearest spot that touches no
    other label, marker or line, stays inside the axes and is nearer its own
    anchor than any other; a label pushed further out gets a thin leader,
    which must cross nothing either. `obstacles` are (data points, clearance
    in points) pairs: marker centres with their radius, or points along a line."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px = fig.dpi / 72
    obs = [(ax.transData.transform(xy), pad * px) for xy, pad in obstacles]
    own_px = ax.transData.transform(anchors)
    frame = ax.get_window_extent(rend)
    tries = [(-1.3, 0, "right", "center"), (-1, -1, "right", "top"), (0, -1.3, "center", "top"),
             (1, -1, "left", "top"), (-1, 1, "right", "bottom"), (1, 1, "left", "bottom"),
             (0, 1.3, "center", "bottom"), (1.3, 0, "left", "center"), (-0.5, -1.3, "right", "top"),
             (-1.3, -0.5, "right", "top")]

    def clear(pts: np.ndarray, skip: np.ndarray | None = None) -> bool:
        for o, pad in obs:
            if skip is not None:
                o = o[np.hypot(*(o - skip).T) > 8 * px]  # the anchor's own marker and the line into it
            if any(((o[:, 0] > x0 - pad) & (o[:, 0] < x1 + pad) & (o[:, 1] > y0 - pad) & (o[:, 1] < y1 + pad)).any()
                   for x0, y0, x1, y1 in pts):
                return False
        return True

    placed: set[int] = set()
    for reaches in ((4, 6), (9, 13, 18, 24)):  # every label tries the close spots before any takes a leader
        for i, ((x, y), text) in enumerate(zip(anchors, texts)):
            if i in placed:
                continue
            own, others = own_px[i], np.delete(own_px, i, axis=0)
            for reach, (ux, uy, ha, va) in ((r, t) for r in reaches for t in tries):
                a = ax.annotate(text, (x, y), xytext=(ux * reach, uy * reach), textcoords="offset points", ha=ha,
                                va=va, fontsize=SMALL, color=colour, fontweight="bold", annotation_clip=False)
                bb = a.get_window_extent(rend)
                a.remove()
                centre = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
                leader = reach >= 9  # a leader says which point a label belongs to; without one, nearness must
                ok = (frame.x0 < bb.x0 and bb.x1 < frame.x1 and frame.y0 < bb.y0 and bb.y1 < frame.y1
                      and (leader or np.hypot(*(others - centre).T).min() > 1.15 * np.hypot(*(own - centre)))
                      and clear([(bb.x0, bb.y0, bb.x1, bb.y1)]) and not any(bb.overlaps(b) for b in blocked))
                if ok and leader:
                    seg = np.linspace(centre, own, 30)[6:]
                    ok = (clear([(px_, py_, px_, py_) for px_, py_ in seg], skip=own)
                          and not any(b.contains(*q) for b in blocked for q in seg))
                if not ok:
                    continue
                a = ax.annotate(text, (x, y), xytext=(ux * reach, uy * reach), textcoords="offset points", ha=ha,
                                va=va, fontsize=SMALL, color=colour, fontweight="bold", annotation_clip=False,
                                arrowprops=dict(arrowstyle="-", color=colour, lw=0.45, shrinkA=0.5, shrinkB=3.5)
                                if leader else None)
                a.set_in_layout(False)
                blocked.append(bb.padded(2 * px))
                if leader:  # later labels keep off this leader too
                    obs.append((np.linspace(centre, own, 30), 1.0 * px))
                placed.add(i)
                break
    for i in sorted(set(range(len(texts))) - placed):
        print(f"WARNING sizing_frontier: no free place for label {texts[i]}")


def along(xs: list[float], ys: list[float], n: int = 60) -> np.ndarray:
    """Points along the frontier's steps, on a log x axis, so labels keep off the line."""
    out = []
    for i in range(len(xs) - 1):
        out += [(x, ys[i]) for x in np.geomspace(xs[i], xs[i + 1], n)]
        out += [(xs[i + 1], y) for y in np.linspace(ys[i], ys[i + 1], n)]
    return np.array(out)


def fig_frontier(pipes: list[dict]) -> None:
    pts = [r for r in pipes if dev_v2(r["analyst"]) and num(r.get("answer_control")) >= 0.75
           and not np.isnan(num(r.get("gpu_s_per_task")))]
    measure = "Wrong conclusions released, planted-error tasks"
    if not pts:
        fig, ax = plt.subplots(figsize=(DOUBLE, 60 * MM))
        P.head(ax, "", "No configuration yet", measure)
        ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", fontsize=7, color=MUTED)
        P.save(fig, HERE / "sizing_frontier")
        return
    best, front = 1.0, []
    for r in sorted(pts, key=lambda r: (r["gpu_s_per_task"], r["wrong_release_seeded"])):
        if r["wrong_release_seeded"] < best:
            best = r["wrong_release_seeded"]
            front.append(r)
    under = [r for r in pts if r["wrong_release_seeded"] <= 0.05]
    clear = [r for r in under if num(r["hi"]) <= 0.05]
    if clear:
        finding = f"{len(clear)} of {len(pts)} under the release threshold with the interval under it"
    elif under:
        finding = f"{len(under)} of {len(pts)} under the release threshold, none with the interval under it"
    else:
        finding = f"None of {len(pts)} is under the release threshold yet"

    # the key under the plot: one entry per frontier configuration, wrapped
    entries = [textwrap.wrap(pipeline_label(r["analyst"], r["pipeline"]), 84) for r in front]
    step = 8.6  # points per line of the key
    key_h = (sum(len(e) for e in entries) + 1.6) * step / 72 * 25.4
    fig, (ax, kx) = plt.subplots(2, 1, figsize=(DOUBLE, (96 + key_h) * MM),
                                 gridspec_kw={"height_ratios": [80, key_h], "hspace": 0.06})
    P.head(ax, "", finding, measure)
    ax.set_xlabel("GPU-seconds per task")
    size = lambda r: 2 + 5 * r["coverage_all"]  # noqa: E731  marker diameter in points
    cloud = [r for r in pts if r not in front]
    for r in cloud:
        ax.plot(r["gpu_s_per_task"], r["wrong_release_seeded"], "o", ms=size(r), color=LIGHT, mfc="none", mew=0.45,
                alpha=0.8, zorder=1)
    fx = [r["gpu_s_per_task"] for r in front]
    fy = [r["wrong_release_seeded"] for r in front]
    ax.plot(fx, fy, "-", color=BLUE, lw=0.9, zorder=2, drawstyle="steps-post")
    for r in front:
        ax.plot(r["gpu_s_per_task"], r["wrong_release_seeded"], "o", ms=size(r), color=BLUE, mec="white", mew=0.5,
                zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(min(fx) / 1.25, max(r["gpu_s_per_task"] for r in pts) * 1.15)
    ax.set_ylim(-0.045, max(r["wrong_release_seeded"] for r in pts) + 0.03)
    ax.spines["left"].set_bounds(0, ax.get_ylim()[1])
    ax.set_yticks(np.arange(0, ax.get_ylim()[1], 0.1))
    ax.grid(axis="y", visible=False)  # the threshold is the one reference line the reader needs
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, _: f"{x:g}" if f"{x:g}".lstrip("0.")[:1] in ("2", "5") else ""))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}" if abs(v) > 1e-9 else "0"))
    # the release threshold, named where the plot is empty: left of the frontier
    ax.axhline(0.05, color=INK, lw=0.7, ls=(0, (4, 2.5)), zorder=1)
    thr = ax.annotate("release threshold", (0, 0.05), xycoords=("axes fraction", "data"), xytext=(3, 2),
                      textcoords="offset points", ha="left", va="bottom", fontsize=SMALL, color=INK)
    sizes = [plt.Line2D([], [], ls="", marker="o", color=GREY, mfc="none", mew=0.6, ms=2 + 5 * c, label=f"{c:g}")
             for c in (0.3, 0.6, 0.9)]
    key = ax.legend(handles=sizes, title="Marker size: correct answers released", title_fontsize=SMALL,
                    fontsize=SMALL, ncol=3, loc="upper right", alignment="left", borderaxespad=0.3)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    blocked = [thr.get_window_extent(rend).padded(2), key.get_window_extent(rend)]
    xs = np.geomspace(*ax.get_xlim(), 400)
    obstacles = [(np.array([(r["gpu_s_per_task"], r["wrong_release_seeded"]) for r in pts]), 4.0),
                 (along(fx, fy), 1.2), (np.column_stack([xs, np.full(xs.size, 0.05)]), 1.2)]
    place_labels(fig, ax, list(zip(fx, fy)), [str(i) for i in range(1, len(front) + 1)], BLUE, obstacles, blocked)

    kx.axis("off")
    cols = (0.0, 0.03, 0.655, 0.79, 0.925)  # number, pipeline, wrong (interval), correct released, GPU-s
    head = ("", "Frontier, cheapest first: analyst → checks", "Wrong (95% CI)", "Correct released", "GPU-s")
    for x, text in zip(cols, head):
        kx.annotate(text, (x, 1), xycoords="axes fraction", ha="left", va="top", fontsize=SMALL, color=MUTED)
    y = step * 1.6
    for i, (r, wrapped) in enumerate(zip(front, entries), 1):
        cells = (str(i), None, f"{r['wrong_release_seeded']:.2f} ({num(r['lo']):.2f}–{num(r['hi']):.2f})",
                 f"{r['coverage_all']:.2f}", f"{r['gpu_s_per_task']:.1f}")
        for x, text in zip(cols, cells):
            if text is not None:
                kx.annotate(text, (x, 1), xycoords="axes fraction", xytext=(0, -y), textcoords="offset points",
                            ha="left", va="top", fontsize=SMALL, color=BLUE if x == 0 else INK,
                            fontweight="bold" if x == 0 else "normal")
        for k, part in enumerate(wrapped):
            kx.annotate(part, (cols[1], 1), xycoords="axes fraction", xytext=(8 if k else 0, -y - k * step),
                        textcoords="offset points", ha="left", va="top", fontsize=SMALL, color=INK)
        y += step * len(wrapped)
    P.save(fig, HERE / "sizing_frontier")


# ---------------------------------------------------------------- Figure 9

FRESH_ARM = {"open|think1024|rules": "rules + 1k think", "open|off|rules": "rules", "revise|off|rules": "rules, revise",
             "open|effort-high|rules|gpt-oss": "rules, high effort"}


def fig_fresh(rows: list[dict]) -> None:
    """The fresh-set test (report Table 10), from fresh_results.csv."""
    rows = sorted(rows, key=lambda r: (num(r["fresh_wrong"]), -num(r["fresh_correct_released"])))
    labels = [f"{label(r['model'])}\n{FRESH_ARM.get(r['arm'], r['arm'])}" for r in rows]
    n = len(rows)
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, (40 + 9 * n) * MM), sharey=True,
                            gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.07})
    a, b = axs
    confirmed = {lab for lab, r in zip(labels, rows) if str(r["confirms"]) == "True"}
    P.rows(a, labels, bold=confirmed)
    a.set_ylim(n - 0.5, -0.95)
    share_axis(a, -0.01, 0.41)
    for i, r in enumerate(rows):
        dot(a, num(r["dev_wrong"]), i - 0.33, "o", GREY, "white", size=3.5)
        dot(a, num(r["heldout_wrong"]), i - 0.17, "s", GREY, "white", size=3.3)
        interval(a, num(r["fresh_lo"]), num(r["fresh_hi"]), i, BLUE)
        dot(a, num(r["fresh_wrong"]), i, "o", BLUE, "white", size=4)
        if r["checked_wrong"] != "":
            dot(a, num(r["checked_wrong"]), i + 0.22, "D", BLUE, BLUE, size=3.6)
    threshold(a, 0.05, "release threshold")
    n_ok = len(confirmed)
    head_count = ["No", "One", "Two", "Three", "Four"][min(n_ok, 4)] if n_ok < 5 else str(n_ok)
    P.head(a, "a", f"{head_count} configurations hold on unseen tasks",
           "Wrong conclusions released, share of planted-error attempts")

    share_axis(b, -0.02, 1.04)
    b.grid(axis="y", visible=False)
    b.grid(axis="x", visible=True)
    b.tick_params(axis="y", length=0)
    for i, r in enumerate(rows):
        alone = num(r["fresh_controls_answered"])
        if r["checked_wrong"] != "":
            checked = num(r["checked_controls_answered"])
            b.plot([checked, alone], [i, i], "-", color=LIGHT, lw=0.8, zorder=1)
            dot(b, checked, i, "D", BLUE, BLUE, size=3.6)
            if checked < alone:
                P.label_at(b, checked, i, f"{checked:.2f}", color=BLUE, dy=-4.5, va="top", bbox=BACK)
        dot(b, alone, i, "o", BLUE, "white", size=4)
    threshold(b, 0.75, "release threshold", left=True)
    P.head(b, "b", "The checker costs clean answers", "Share of control attempts answered")
    handles = [plt.Line2D([], [], ls="", marker="o", color=GREY, mfc="white", mew=0.8, ms=3.5, label="dev"),
               plt.Line2D([], [], ls="", marker="s", color=GREY, mfc="white", mew=0.8, ms=3.3, label="held-out"),
               plt.Line2D([], [], ls="-", marker="o", color=BLUE, mfc="white", mew=0.8, lw=0.7, ms=4,
                          label="sealed set, analyst alone (95% CI)"),
               plt.Line2D([], [], ls="", marker="D", color=BLUE, ms=3.6, label="sealed set, with the chosen checker")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    P.save(fig, HERE / "sizing_fresh")


# ---------------------------------------------------------------- Figure 7

PROMPT_MODELS = ["Qwen3.5-9B", "Qwen3.5-27B", "Qwen3.5-35B-A3B", "Qwen3.5-122B-A10B-NVFP4",
                 "gemma-4-26B-A4B-it", "gemma-4-31B-it"]
PROMPT_ARMS = [  # (prompt, thinking, label, marker, colour, face): the arms of the prompt comparison
    ("plain", "off", "plain", "o", GREY, "white"),
    ("checklist", "off", "checklist", "D", GREY, GREY),
    ("rules", "off", "rules", "s", BLUE, "white"),
    ("rules", "think1024", "rules + 1k think", "o", BLUE, BLUE),
]


def fig_prompt(rows: list[dict]) -> None:
    """The prompt, not the size: wrong releases per analyst for the plain,
    checklist, rules and rules-plus-think arms, on the dev tasks and on the
    held-out tasks (report section 2, the prompt-optimisation tables)."""
    rows = [r for r in rows if parts(r["config"])["arm"] == "open" and parts(r["config"])["pack"] == "v2"
            and r["model"] in PROMPT_MODELS and parts(r["config"])["think"] in ("off", "think1024")]
    models = [m for m in PROMPT_MODELS if any(r["model"] == m for r in rows)]
    if not models:
        return
    n = len(models)
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, (34 + 12 * n) * MM), sharey=True, gridspec_kw={"wspace": 0.08})
    panels = ((axs[0], False, "a", "The rules prompt halves the dev error", "Dev tasks"),
              (axs[1], True, "b", "It carries to held-out; size does not", "Held-out tasks"))
    for ax, heldout, letter, finding, measure in panels:
        P.rows(ax, [row_label(m) for m in models])
        ax.set_ylim(n - 0.5, -0.85)
        share_axis(ax, -0.02, 0.68)
        for j, (prompt, think, name, mk, col, face) in enumerate(PROMPT_ARMS):
            dy = (j - 1.5) * 0.2
            for i, m in enumerate(models):
                r = next((r for r in rows if r["model"] == m and prompt_of(r["config"]) == prompt
                          and parts(r["config"])["think"] == think and parts(r["config"])["heldout"] == heldout), None)
                if r is None:
                    continue
                v, hi = r["wrong_release_seeded"], num(r["hi"])
                interval(ax, num(r["lo"]), hi, i + dy, BLUE if col == BLUE else LIGHT)
                dot(ax, v, i + dy, mk, col, face)
                # the focal arm's value at every row; the arms named once, in the first row of panel a
                text = f"{v:.2f}" if name == "rules + 1k think" else ""
                if i == 0 and not heldout:
                    text = f"{text}  {name}".strip()
                if text:
                    P.label_at(ax, hi, i + dy, text, color=col, dx=3, dy=0, ha="left", va="center", bbox=BACK)
        threshold(ax, 0.05, "release threshold")
        P.head(ax, letter, finding, measure)
    fig.supxlabel("Wrong conclusions released, share of 96 planted-error attempts", fontsize=7.5)
    P.save(fig, HERE / "sizing_prompt")


def main() -> None:
    fresh = load("fresh_results.csv")
    if fresh:
        fig_fresh(fresh)
    analysts, checkers, matrix, pipes = load("analysts.csv"), load("checkers.csv"), load("matrix.csv"), load("pipelines.csv")
    if analysts:
        fig_analyst(analysts)
        fig_prompt(analysts)
    if checkers:
        fig_checker(checkers, matrix)
        fig_thinking(checkers, analysts)
    if pipes:
        fig_frontier(pipes)


if __name__ == "__main__":
    main()
