"""Item registry.

`build_corpus(seed)` returns every implemented item as a (seeded, control)
pair, deterministic in `seed`. Items are matched: same data-generating process,
same tables, differing only in the analysis and its interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bundle import Bundle
from .core import GENERATORS

CLASS_NAMES: dict[int, str] = {
    1: "unreported multiplicity",
    2: "silent attrition",
    3: "omitted confounder / Simpson reversal",
    4: "unsupported causal claim (internal control)",
    5: "wrong error structure",
    6: "wrong denominator",
    7: "regression to the mean",
    8: "in-sample performance",
}

IMPLEMENTED: tuple[int, ...] = tuple(sorted(GENERATORS))


@dataclass(frozen=True)
class Item:
    error_class: int
    seeded: Bundle
    control: Bundle

    @property
    def pair(self) -> tuple[Bundle, Bundle]:
        return self.seeded, self.control


REGISTERS: tuple[str, ...] = ("default", "matched")


def build_corpus(seed: int = 0, replicates: int = 1, register: str = "default") -> list[Item]:
    """One (seeded, control) pair per implemented class per replicate.

    Replicates re-draw the random data under the same generator so that a
    class is represented by more than one dataset.

    register: "default" keeps each control's own interpretation, which hedges
    where the seeded bundle pushes. "matched" swaps in the control's
    `interpretation_matched`, the same correct conclusion in the seeded
    register, so that a checker cannot separate the pair by tone. An
    ablation beside the pre-registered design, not a replacement for it.
    """
    if register not in REGISTERS:
        raise ValueError(f"register must be one of {REGISTERS}, got {register!r}")
    items: list[Item] = []
    for rep in range(replicates):
        for cls in IMPLEMENTED:
            rng = np.random.default_rng([seed, cls, rep])
            gen = GENERATORS[cls]
            err, ctl = gen(rng, item_id=f"c{cls}-r{rep}")
            assert err.truth and err.truth.has_error and err.truth.error_class == cls
            assert ctl.truth and not ctl.truth.has_error
            if register == "matched":
                assert ctl.interpretation_matched, f"class {cls}: control has no matched-register interpretation"
                ctl.interpretation = ctl.interpretation_matched
            items.append(Item(error_class=cls, seeded=err, control=ctl))
    return items
