"""Phase 0: the s0 ceiling check.

The gate on everything else. Every seeded item and its matched control are
rendered with no disclosure control at all and shown to a checker. If the
checker cannot detect the seeded errors with everything visible, non-detection
under suppression says nothing about disclosure control, and there is no study.

Also doubles as the model-selection instrument: run it against each candidate
checker and keep the ones near ceiling.

Output is one JSON line per (bundle, checker) with the full verdict, including
the verbatim model response. Nothing is summarised away.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from .bundle import Bundle
from .checker.base import Checker, Verdict
from .items import Item, CLASS_NAMES, build_corpus
from .sdc import policy


@dataclass
class Observation:
    item_id: str
    error_class: int | None
    has_error: bool
    disclosure_level: int
    checker: str
    verdict: dict
    rendered_chars: int
    timestamp: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _observe(b: Bundle, level: int, chk: Checker) -> Observation:
    rendered = b.controlled(policy(level)).render()
    v: Verdict = chk.review(rendered)
    assert b.truth is not None
    return Observation(
        item_id=b.item_id,
        error_class=b.truth.error_class,
        has_error=b.truth.has_error,
        disclosure_level=level,
        checker=chk.name,
        verdict=v.to_dict(),
        rendered_chars=len(rendered),
        timestamp=time.time(),
    )


def run(
    checkers: list[Checker],
    out: Path,
    seed: int = 0,
    replicates: int = 1,
    levels: tuple[int, ...] = (0,),
    items: list[Item] | None = None,
    echo: bool = True,
) -> list[Observation]:
    """Run every bundle at every level through every checker.

    Phase 0 proper is `levels=(0,)`. The same function with
    `levels=(0, 1, 2, 3)` is the full machine-arm study.
    """
    items = items if items is not None else build_corpus(seed=seed, replicates=replicates)
    out.parent.mkdir(parents=True, exist_ok=True)
    obs: list[Observation] = []

    with out.open("a", encoding="utf-8") as f:
        for item in items:
            for b in item.pair:
                for level in levels:
                    for chk in checkers:
                        o = _observe(b, level, chk)
                        obs.append(o)
                        f.write(o.to_json() + "\n")
                        f.flush()
                        if echo:
                            v = o.verdict
                            tag = "ERR" if o.has_error else "ctl"
                            print(
                                f"[{chk.name}] s{level} {o.item_id:<12} {tag} -> "
                                f"{v['error_present']:<11} conf={v['confidence']:>3} "
                                f"rel={v['release']}",
                                file=sys.stderr,
                            )
    return obs


def summarise(obs: list[Observation]) -> dict:
    """Detection and false-alarm rates per checker, per class, per level."""
    from collections import defaultdict

    cells: dict = defaultdict(lambda: {"hit": 0, "miss": 0, "ct": 0, "fa": 0, "cr": 0, "ct_ctl": 0})
    for o in obs:
        key = (o.checker, o.disclosure_level, o.error_class if o.has_error else 0)
        ep = o.verdict["error_present"]
        c = cells[key]
        if o.has_error:
            if ep == "yes":
                c["hit"] += 1
            elif ep == "no":
                c["miss"] += 1
            else:
                c["ct"] += 1
        else:
            if ep == "yes":
                c["fa"] += 1
            elif ep == "no":
                c["cr"] += 1
            else:
                c["ct_ctl"] += 1

    rows = []
    for (chk, lvl, cls), c in sorted(cells.items()):
        n_err = c["hit"] + c["miss"] + c["ct"]
        n_ctl = c["fa"] + c["cr"] + c["ct_ctl"]
        rows.append(
            {
                "checker": chk,
                "level": lvl,
                "class": cls,
                "class_name": CLASS_NAMES.get(cls, "control"),
                "n_seeded": n_err,
                "detect": c["hit"] / n_err if n_err else None,
                "cannot_tell": c["ct"] / n_err if n_err else None,
                "n_control": n_ctl,
                "false_alarm": c["fa"] / n_ctl if n_ctl else None,
            }
        )
    return {"rows": rows}


def print_summary(summary: dict) -> None:
    print(f"{'checker':<28} {'lvl':>3} {'class':>5}  {'detect':>7} {'c.tell':>7} {'f.alarm':>8}  n")
    for r in summary["rows"]:
        fmt = lambda x: "   -   " if x is None else f"{x:7.2f}"
        n = r["n_seeded"] or r["n_control"]
        print(
            f"{r['checker']:<28} {r['level']:>3} {r['class']:>5}  "
            f"{fmt(r['detect'])} {fmt(r['cannot_tell'])} {fmt(r['false_alarm']):>8}  {n}"
        )


def load(path: Path) -> list[Observation]:
    obs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("_run"):
                continue  # provenance header
            obs.append(Observation(**d))
    return obs
