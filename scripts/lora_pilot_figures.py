"""Figures for the honest-LoRA engineering pilot (docs/viking-lora-pilot-results.md).

Reads the evidence mirrored from Viking into runs/lora-pilot/ (attempts/<run-id>/evidence
and report-01/report.json) and writes docs/figures/lora_pilot.{pdf,png}:
(a) where each GPU stage's minutes went, against its Slurm limit;
(b) peak GPU memory against context length, against the 85% limit, for the pilot
and the later study-length profile run, with the study-like example lengths.

    uv run python scripts/lora_pilot_figures.py [runs/lora-pilot]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
sys.path.insert(0, str(FIG))
import paperfig as P  # noqa: E402  the paper's shared figure look

LIMIT = {"smoke": 1800, "train": 5400, "reload": 1800}
GB = 1e9


def read(path: Path):
    return json.loads(path.read_text())


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def phases(attempt: Path, elapsed: float) -> dict[str, float]:
    """Seconds per phase, from the payload's own events and diagnostic timings."""
    ev = attempt / "evidence"
    context = read(ev / "context.json")
    start = context["deadline"] + 120 - LIMIT[context["stage"]]  # the launcher's deadline arithmetic
    main = events(ev / "events.jsonl")
    done = next(e["time"] for e in main if e["event"] == "completed")
    out = {"checks": main[0]["time"] - start, "training": 0.0, "diagnostics": 0.0}
    if context["stage"] == "reload":
        # The clean process starts once the parent's monitor is up; the adapter process when the clean one ends.
        begin = main[0]["time"]
        for kind in ("clean", "adapter"):
            worker = events(ev / kind / "events.jsonl")
            out["checks"] += worker[0]["time"] - begin  # the fresh process's own preflight
            begin = worker[-1]["time"]
            for name in ("baseline", "adapted", "disabled"):
                if (ev / kind / name / "index.json").exists():
                    out["diagnostics"] += read(ev / kind / name / "index.json")["seconds"]
        last = max(e["time"] for e in events(ev / "adapter" / "events.jsonl"))
    else:
        checks = read(ev / "checks.json")
        named = {e["event"]: e for e in main}
        out["checks"] += named["base_hash_before"]["seconds"] + named["base_hash_after"]["seconds"]
        out["training"] = sum(s["forward_backward_seconds"] + s["optimiser_seconds"] for s in checks["steps"])
        out["training"] += sum(p["forward_backward_seconds"] for p in checks["profiles"])
        out["diagnostics"] = sum(read(ev / name / "index.json")["seconds"]
                                 for name in ("baseline", "repeat", "initial", "adapted"))
        last = max(e["time"] for e in main if e["event"] == "profile")
    out["checks"] += done - last  # the closing re-hash of the frozen bundle
    out["other"] = elapsed - sum(out.values())
    return out


def main() -> int:
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "opacity" / "runs" / "lora-pilot"
    report = read(runs / "report-01" / "report.json")
    elapsed = {r["stage"]: r["scheduler"]["elapsed_seconds"] for r in report["records"]}
    attempts = {r["stage"]: Path(r["attempt"]).name for r in report["records"]}
    stages = ["smoke", "train", "reload"]
    split = {s: phases(runs / "attempts" / attempts[s], elapsed[s]) for s in stages}
    train = read(runs / "attempts" / attempts["train"] / "evidence" / "checks.json")

    # Laid out in millimetres at the paper's text width: (a) labels and columns, (b) the memory plot.
    W, H, top, bottom = 148.0, 70.0, 11.0, 11.0
    fig = P.figure_mm(W, H)
    ax = P.axes_mm(fig, 6.0, bottom, 62.0, H - top - bottom)
    bx = P.axes_mm(fig, 83.0, bottom, W - 1.5 - 83.0 - 4.0, H - top - bottom)

    # (a) One column per stage, checks at the base so they sit beside training; the
    # segments are named once, beside the first column, where all four appear.
    kinds = [("checks", "isolation and\nintegrity checks", P.BLUE),
             ("training", "training and\nstress steps", P.INK),
             ("diagnostics", "diagnostic passes", P.GREY),
             ("other", "loading, launch\nand teardown", P.FAINT)]
    names = {"smoke": ("smoke", "2 updates"), "train": ("train", "20 updates"), "reload": ("reload", "2 fresh\nprocesses")}
    for col, stage in enumerate(stages):
        base = 0.0
        for key, label, colour in kinds:
            h = split[stage][key] / 60
            ax.bar(col, h, bottom=base, width=0.56, color=colour, lw=0)
            if key == "checks":  # the focal value, written in its segment
                ax.text(col, base + h / 2, f"{h:.1f}", ha="center", va="center", fontsize=P.SMALL, color="white")
            if col == 0:
                ax.annotate(label, xy=(col - 0.28, base + h / 2), xytext=(-5, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=P.SMALL, color=P.MUTED if key == "other" else colour,
                            linespacing=1.0)
            base += h
        P.label_at(ax, col, base, f"{base:.1f} of {LIMIT[stage] // 60} min", dy=3)
        stage_name, detail = names[stage]
        ax.annotate(f"{stage_name}\n{detail}", xy=(col, 0), xytext=(0, -4), textcoords="offset points",
                    ha="center", va="top", fontsize=P.SMALL, color=P.INK, linespacing=1.1, annotation_clip=False)
    ax.set_xlim(-1.55, 2.45)
    ax.set_ylim(0, 10.6)
    ax.axis("off")
    P.head(ax, "a", "Checks took longer than training", "Minutes on one H100, against each stage's limit")

    # (b) Peak reserved memory against tokens per step: the pilot, then the profile run
    # that crosses the 85% limit; ticks along the base are the study-like examples.
    profiles = train["profiles"]
    total = profiles[0]["device_total_bytes"] / GB
    limit = 0.85 * total
    steady = max(s["peak_reserved_bytes"] for s in train["steps"]) / GB
    mean_tokens = sum(s["nonpadding_tokens"] for s in train["steps"]) / (4 * len(train["steps"]))
    x = [mean_tokens] + [p["length"] for p in profiles]
    reserved = [steady] + [p["peak_reserved_bytes"] / GB for p in profiles]
    # The study-length profile run (its own bundle and ledger) and the study-like example lengths.
    later = read(runs / "study-lengths" / "attempts" / "lengths-profile-01" / "evidence" / "checks.json")["profiles"]
    lx = [p["length"] for p in later]
    ly = [p["peak_reserved_bytes"] / GB for p in later]
    cross = lx[0] + (limit - ly[0]) / (ly[1] - ly[0]) * (lx[1] - lx[0])  # linear between the two profile points
    lengths = [r["total"] for r in read(runs / "study-lengths" / "summary-steer2-think-style.json")["rows"]
               if r["dir"] == "steer2" and r["task"] == "harvest"]
    over = [n for n in lengths if n > cross]

    lo = 50
    bx.axhline(total, color=P.GREY, lw=0.7, zorder=1)
    P.label_at(bx, 60, total, f"H100 capacity {total:.0f} GB", color=P.MUTED, ha="left", dy=2)
    P.line(bx, limit, "", axis="y")
    P.label_at(bx, 60, limit, f"85% limit, {limit:.1f} GB", ha="left", dy=2)
    bx.plot(x, reserved, "o-", color=P.GREY, mfc="white", mew=0.9, ms=3.5, lw=0.9, zorder=3)
    P.label_at(bx, x[-1], reserved[-1], f"pilot, {reserved[-1]:.1f}", color=P.MUTED, ha="right", va="bottom",
               dx=-4, dy=2)
    bx.plot(lx, ly, "-", color=P.BLUE, lw=1.2, zorder=3)
    bx.plot(lx[0], ly[0], "o", color=P.BLUE, ms=4, zorder=4)
    bx.plot(lx[1], ly[1], "o", color=P.ORANGE, ms=4.5, zorder=4)
    P.label_at(bx, lx[0], ly[0], f"profile run, {ly[0]:.1f}", color=P.BLUE, ha="left", va="top", dx=5, dy=-2)
    top = later[-1]
    P.label_at(bx, lx[1], ly[1], f"{ly[1]:.1f} GB, {100 * top['peak_reserved_bytes'] / top['device_total_bytes']:.1f}%\n"
               f"of capacity", color=P.ORANGE, ha="left", va="bottom", dx=4, dy=4, linespacing=1.1)
    bx.vlines([n for n in lengths if n <= cross], lo, lo + 2.2, color=P.GREY, lw=0.35)
    bx.vlines(over, lo, lo + 2.2, color=P.ORANGE, lw=0.5)
    P.label_at(bx, min(lengths), lo + 2.2, f"{len(lengths)} study-like\nexamples", color=P.MUTED, ha="left",
               va="bottom", dx=0, dy=3, linespacing=1.0)
    P.label_at(bx, cross, lo + 2.2, f"{len(over)} longer than\nthe limit allows", color=P.ORANGE, ha="left",
               va="bottom", dx=0, dy=3, linespacing=1.0)
    bx.set_xticks([0, 1024, 2048, 3072, 4096], ["0", "1,024", "2,048", "3,072", "4,096"])
    bx.set_xlim(0, 4300)
    bx.set_ylim(lo, 89)
    bx.set_yticks([50, 60, 70, 80])
    bx.grid(False)  # the values that matter are written beside their points
    bx.set_xlabel("Tokens per training step (non-padding)")
    P.head(bx, "b", f"Memory crosses the limit near {round(cross, -1):,.0f} tokens", "Peak GPU memory reserved (GB)")
    P.save(fig, FIG / "lora_pilot")
    for stage in stages:
        print(stage, {k: round(v / 60, 2) for k, v in split[stage].items()}, "elapsed", elapsed[stage])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
