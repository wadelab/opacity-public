"""The mechanical arm: SDC output-checking rules, no judgement.

This is what a TRE output checker's role is actually defined as -- disclosure
risk, not correctness. It therefore never flags an analytic error. That is not
a limitation of the implementation; it is the operationalisation of the
claim that analytic error passes the human in the loop unexamined by design.

Its release decision is on disclosure grounds only: block if any released
frequency cell is below the policy threshold, query if anything was withheld,
otherwise release.
"""

from __future__ import annotations

import re
import time

from .base import Verdict

_CELL_RE = re.compile(r"\|\s*(\d+)\s*(?=\|)")
_SUPPRESSED_MARK = "[s]"


class MechanicalChecker:
    name = "mechanical"

    def __init__(self, threshold: int = 10) -> None:
        self.threshold = threshold

    def review(self, rendered_bundle: str) -> Verdict:
        t0 = time.perf_counter()
        in_results = False
        small_cells: list[int] = []
        withheld = _SUPPRESSED_MARK in rendered_bundle

        for line in rendered_bundle.splitlines():
            if line.startswith("## Results"):
                in_results = True
                continue
            if line.startswith("## ") and in_results:
                in_results = False
            if not in_results or not line.startswith("|"):
                continue
            if line.lstrip("|").strip().startswith("Total"):
                continue
            for m in _CELL_RE.finditer(line):
                v = int(m.group(1))
                if 0 < v < self.threshold:
                    small_cells.append(v)

        if small_cells:
            release = "block"
            desc = (
                f"Disclosure check only: {len(small_cells)} cell(s) below threshold "
                f"{self.threshold}. Analytic correctness not assessed."
            )
        elif withheld:
            release = "query"
            desc = "Disclosure check only: values withheld. Analytic correctness not assessed."
        else:
            release = "release"
            desc = "Disclosure check only. Analytic correctness not assessed."

        return Verdict(
            error_present="no",
            confidence=0,
            description=desc,
            release=release,
            raw=desc,
            model="mechanical",
            seconds=time.perf_counter() - t0,
            meta={"small_cells": small_cells, "withheld": withheld},
        )
