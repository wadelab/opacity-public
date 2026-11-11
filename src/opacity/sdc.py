"""Graded statistical disclosure control.

The experimental manipulation. A bundle is rendered at one of four disclosure
strengths and the only difference between arms is what this module removes.

Rules implemented here follow standard SDC output-checking practice: a minimum
cell-count threshold, base-k rounding, complementary (secondary) suppression to
block differencing, and an (n, k) dominance rule for magnitude tables. Graded
levels are a construct of this study, not a standard -- s1 to s3 are chosen to
span the range of policies real TREs apply, and the mapping is pre-registered
rather than claimed to be canonical.

Concealment mechanism per error class (see docs/taxonomy.md):
  class 1 multiplicity   -- small subgroup cells suppress, removing the
                            denominator of comparisons
  class 2 attrition      -- rounding hides stage-to-stage N mismatches
  class 3 confounder     -- strata suppress, leaving the aggregate alone
  class 4 causal claim   -- NOT concealed at any level (internal control)
  class 5 error structure-- cluster structure is progressively redacted
  class 6 denominator    -- margins round or withhold
  class 7 RTM            -- the selection step suppresses
  class 8 in-sample      -- exact N rounds
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Sequence

import numpy as np

SUPPRESSED = "[s]"
"""Rendered marker for a suppressed cell. Deliberately terse: a verbose marker
tells the reviewer where to look, which is itself a cue we are not measuring."""

ClusterDisclosure = Literal["full", "sizes", "count", "none"]
"""How much of the clustering/repeated-measures structure survives.
full  -- per-observation cluster identifiers
sizes -- cluster sizes, no identifiers
count -- number of clusters only
none  -- no indication the data are clustered
"""

StatPrecision = Literal["exact", "banded"]
"""How statistics quoted in prose are released. exact: as computed. banded:
the test statistic is dropped and p is reported as a band (see `p_band`)."""

RowCollapse = Literal["keep", "other", "drop"]
"""What happens to a row whose every cell is suppressed.
keep  -- row stays, all cells shown as suppressed (row count still visible)
other -- all such rows merge into one 'Other' row (count concealed)
drop  -- such rows are removed (existence concealed)

This is the class 1 concealment mechanism. With `keep`, a reviewer can count
the suppressed rows and recover the number of comparisons; real output
checking collapses or drops them, and so do s2 and s3.
"""


@dataclass(frozen=True)
class SDCPolicy:
    """One disclosure strength."""

    level: int
    label: str
    threshold: int = 0
    """Primary suppression: frequency cells with count < threshold. 0 disables."""
    rounding_base: int = 0
    """Base-k rounding of counts and reported Ns. 0 disables."""
    complementary: bool = False
    """Secondary suppression so a suppressed cell cannot be differenced out."""
    dominance_n: int = 0
    dominance_k: float = 0.0
    """(n, k) rule: suppress a magnitude cell if the top n contributors hold
    more than k of its total. 0 disables."""
    cluster_disclosure: ClusterDisclosure = "full"
    exact_n: bool = True
    """False: reported sample sizes are rounded and stage-wise Ns collapsed."""
    min_populated_strata: int = 0
    """Suppress a stratified table entirely if fewer than this many strata
    survive primary suppression. 0 disables."""
    row_collapse: RowCollapse = "keep"
    hide_margins: bool = False
    """Withhold column totals. Closes the differencing route by which the sum
    of suppressed cells in a column can be recovered from the margin."""
    stat_precision: StatPrecision = "exact"
    """How test statistics quoted in the prose are released. `banded` drops the
    statistic and gives p as a band, so that prose never carries more precision
    than the rounded table it summarises. Introduced after phase 0 showed
    checkers recomputing a quoted chi-square from rounded counts and flagging
    the mismatch: disclosure control was manufacturing a cue."""
    show_sample_list: bool = True
    """False: the stage-wise sample list is withheld entirely. At s3 the single
    surviving row of a dropped-row table beside a released total N told the
    reviewer that rows were missing."""
    min_comparable_strata: int = 0
    """Withhold a stratified table unless at least this many strata have every
    cell visible. With margins hidden and no comparable stratum, the arm sizes
    per stratum were the only thing left, and they disclosed the allocation
    imbalance that is the class 3 mechanism. 0 disables."""
    withhold_tables: bool = False
    """Withhold every table. The prose-only level: what a reviewer can do from
    methods and interpretation alone, which bounds the prose floor of detection."""

    def applies_any(self) -> bool:
        return self.level > 0


S0 = SDCPolicy(level=0, label="s0 none")

S1 = SDCPolicy(
    level=1,
    label="s1 permissive",
    threshold=5,
    complementary=True,
    cluster_disclosure="sizes",
)

S2 = SDCPolicy(
    level=2,
    label="s2 standard",
    threshold=10,
    rounding_base=5,
    complementary=True,
    dominance_n=2,
    dominance_k=0.85,
    cluster_disclosure="count",
    exact_n=False,
    row_collapse="other",
    stat_precision="banded",
)

S3 = SDCPolicy(
    level=3,
    label="s3 conservative",
    threshold=10,
    rounding_base=10,
    complementary=True,
    dominance_n=1,
    dominance_k=0.70,
    cluster_disclosure="none",
    exact_n=False,
    min_populated_strata=3,
    row_collapse="drop",
    hide_margins=True,
    stat_precision="banded",
    show_sample_list=False,
    min_comparable_strata=1,
)

S4 = SDCPolicy(
    level=4,
    label="s4 prose only",
    threshold=10,
    rounding_base=10,
    complementary=True,
    dominance_n=1,
    dominance_k=0.70,
    cluster_disclosure="none",
    exact_n=False,
    min_populated_strata=3,
    row_collapse="drop",
    hide_margins=True,
    stat_precision="banded",
    show_sample_list=False,
    min_comparable_strata=1,
    withhold_tables=True,
)
"""Not a disclosure policy any TRE applies. An ablation: every table withheld,
so the reviewer has the methods and the interpretation and nothing else. The
detection it yields is the prose floor, which s0 to s3 can only be read
against. Class 4 is not invariant here, since its table goes too."""

LEVELS: dict[int, SDCPolicy] = {0: S0, 1: S1, 2: S2, 3: S3, 4: S4}
GRADED_LEVELS: tuple[int, ...] = (0, 1, 2, 3)
"""The pre-registered disclosure gradient. s4 is the prose-only ablation."""


def p_band(p: float) -> str:
    """A p-value released as a band rather than a number."""
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return "p < 0.01"
    if p < 0.05:
        return "p < 0.05"
    return "p > 0.05"


def p_exact(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def policy(level: int) -> SDCPolicy:
    if level not in LEVELS:
        raise ValueError(f"disclosure level must be one of {sorted(LEVELS)}, got {level}")
    return LEVELS[level]


@dataclass
class Table:
    """A table in an output bundle.

    values[i, j] is the cell for rows[i] x cols[j]. A one-dimensional table has
    a single column. For magnitude tables, `contributors` holds the sorted
    descending contribution of each unit to each cell, used by the dominance
    rule; `counts` holds the number of units behind each cell.
    """

    name: str
    rows: Sequence[str]
    cols: Sequence[str]
    values: np.ndarray
    kind: Literal["frequency", "magnitude"] = "frequency"
    row_label: str = ""
    col_label: str = ""
    counts: np.ndarray | None = None
    contributors: list[list[np.ndarray]] | None = None
    show_margins: bool = True
    note: str = ""
    stratified: bool = False
    """True for a table whose purpose is comparison across strata (e.g. a
    confounder-adjusted breakdown). Only such tables are subject to the
    whole-table withholding rule: a subgroup-results table with one surviving
    row is released as that row, which is the class 1 concealment at s3."""
    suppressed: np.ndarray | None = field(default=None, repr=False)
    whole_table_suppressed: bool = False
    withheld_reason: str = "too few populated categories to release"
    rounded: bool = False
    margins: np.ndarray | None = field(default=None, repr=False)
    """Column totals as released. Computed from TRUE values then rounded, so
    that under rounding the margin need not equal the sum of displayed cells --
    which is how real SDC output behaves, and is a known reviewer cue."""

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)
        if self.values.shape != (len(self.rows), len(self.cols)):
            raise ValueError(
                f"table {self.name!r}: values shape {self.values.shape} does not match "
                f"{len(self.rows)} rows x {len(self.cols)} cols"
            )
        if self.suppressed is None:
            self.suppressed = np.zeros(self.values.shape, dtype=bool)
        if self.kind == "magnitude" and self.counts is None:
            raise ValueError(
                f"table {self.name!r}: magnitude tables need `counts` for the dominance rule"
            )
        if self.margins is None and self.kind == "frequency":
            self.margins = self.values.sum(axis=0)

    def copy(self) -> "Table":
        return replace(
            self,
            rows=list(self.rows),
            values=self.values.copy(),
            suppressed=self.suppressed.copy(),
            counts=None if self.counts is None else self.counts.copy(),
            margins=None if self.margins is None else self.margins.copy(),
        )

    @property
    def n_visible(self) -> int:
        return int((~self.suppressed).sum())


def _round_base(x: np.ndarray | float, base: int) -> np.ndarray | float:
    """Round to the nearest multiple of `base`, halves away from zero.

    numpy rounds halves to even, which would make 7.5 -> 8 but 12.5 -> 12 at
    base 5; TRE rounding conventions are not half-to-even, so do it explicitly.
    """
    if base <= 0:
        return x
    arr = np.asarray(x, dtype=float)
    out = np.floor(arr / base + 0.5) * base
    return out if np.ndim(x) else float(out)


def _primary_suppression(t: Table, pol: SDCPolicy) -> None:
    if pol.threshold <= 0:
        return
    if t.kind == "frequency":
        small = (t.values > 0) & (t.values < pol.threshold)
    else:
        assert t.counts is not None
        small = (t.counts > 0) & (t.counts < pol.threshold)
    t.suppressed |= small


def _dominance_suppression(t: Table, pol: SDCPolicy) -> None:
    if t.kind != "magnitude" or pol.dominance_n <= 0 or t.contributors is None:
        return
    n, k = pol.dominance_n, pol.dominance_k
    for i in range(t.values.shape[0]):
        for j in range(t.values.shape[1]):
            contrib = np.asarray(t.contributors[i][j], dtype=float)
            total = contrib.sum()
            if total <= 0 or contrib.size == 0:
                continue
            top = np.sort(contrib)[::-1][:n].sum()
            if top / total > k:
                t.suppressed[i, j] = True


def _complementary_suppression(t: Table) -> None:
    """Block differencing.

    If a row (or column) has a visible margin and exactly one suppressed cell,
    that cell is recoverable by subtraction. Suppress the smallest remaining
    cell in that row (or column) and iterate until no line is recoverable.

    With no visible margins there is nothing to difference against, so this is
    a no-op -- which is why `show_margins` matters to the concealment mechanism.
    """
    if not t.show_margins:
        return
    n_rows, n_cols = t.values.shape
    for _ in range(n_rows * n_cols + 1):
        changed = False

        for i in range(n_rows):
            line = t.suppressed[i, :]
            if line.sum() == 1 and n_cols > 1:
                open_idx = np.where(~line)[0]
                if open_idx.size:
                    j = open_idx[np.argmin(t.values[i, open_idx])]
                    t.suppressed[i, j] = True
                    changed = True

        for j in range(n_cols):
            line = t.suppressed[:, j]
            if line.sum() == 1 and n_rows > 1:
                open_idx = np.where(~line)[0]
                if open_idx.size:
                    i = open_idx[np.argmin(t.values[open_idx, j])]
                    t.suppressed[i, j] = True
                    changed = True

        if not changed:
            break
    return


def _rounding(t: Table, pol: SDCPolicy) -> None:
    if pol.rounding_base <= 0 or t.kind != "frequency":
        return
    t.values = np.asarray(_round_base(t.values, pol.rounding_base), dtype=float)
    if t.margins is not None:
        t.margins = np.asarray(_round_base(t.margins, pol.rounding_base), dtype=float)
    t.rounded = True


def _collapse_rows(t: Table, pol: SDCPolicy) -> None:
    """Merge or drop rows whose every cell is suppressed.

    Margins are untouched: they were computed from the true table before any
    row was removed, so the released total still reflects the collapsed rows.
    """
    if pol.row_collapse == "keep" or t.whole_table_suppressed:
        return
    fully = t.suppressed.all(axis=1)
    if fully.sum() == 0:
        return
    keep = ~fully
    rows = [r for r, k in zip(t.rows, keep) if k]
    values = t.values[keep]
    suppressed = t.suppressed[keep]
    if pol.row_collapse == "other":
        rows.append("Other")
        values = np.vstack([values, np.zeros((1, t.values.shape[1]))])
        suppressed = np.vstack([suppressed, np.ones((1, t.values.shape[1]), dtype=bool)])
    t.rows = rows
    t.values = values
    t.suppressed = suppressed
    if t.counts is not None:
        t.counts = t.counts[keep] if pol.row_collapse == "drop" else np.vstack(
            [t.counts[keep], np.zeros((1, t.counts.shape[1]))]
        )
    if t.contributors is not None:
        t.contributors = [c for c, k in zip(t.contributors, keep) if k]
        if pol.row_collapse == "other":
            t.contributors.append([np.array([]) for _ in t.cols])


def apply_to_table(table: Table, pol: SDCPolicy) -> Table:
    """Return a disclosure-controlled copy of `table`.

    Order is fixed and matters:
      1. primary suppression on true values
      2. dominance rule on true contributions
      3. complementary suppression against visible margins
      4. whole-table suppression if too few strata survive
      5. row collapse (merge/drop fully suppressed rows)
      6. rounding of cells and margins, last, so every suppression decision
         was made on true counts -- as a real checker would
    """
    t = table.copy()
    if not pol.applies_any():
        return t

    _primary_suppression(t, pol)
    _dominance_suppression(t, pol)
    _complementary_suppression(t)

    if (
        t.stratified
        and pol.min_populated_strata > 0
        and t.values.shape[0] >= pol.min_populated_strata
    ):
        rows_with_data = int((~t.suppressed).any(axis=1).sum())
        if rows_with_data < pol.min_populated_strata:
            t.whole_table_suppressed = True
            t.suppressed[:] = True

    if t.stratified and pol.min_comparable_strata > 0 and not t.whole_table_suppressed:
        comparable = int((~t.suppressed).all(axis=1).sum())
        if comparable < pol.min_comparable_strata:
            t.whole_table_suppressed = True
            t.withheld_reason = "no stratum releasable in full"
            t.suppressed[:] = True

    if pol.withhold_tables:
        t.whole_table_suppressed = True
        t.withheld_reason = "under disclosure control"
        t.suppressed[:] = True

    _collapse_rows(t, pol)
    if len(t.rows) == 0:
        # every row was dropped: the table is withheld, which is what a checker
        # would release, rather than a header with nothing under it
        t.whole_table_suppressed = True
    _rounding(t, pol)
    if pol.hide_margins:
        t.show_margins = False
    return t


def render_table(t: Table) -> str:
    """Plain-text rendering, as the reviewer sees it."""
    lines = [f"**{t.name}**"]
    if t.whole_table_suppressed:
        lines.append("")
        lines.append(f"_Table withheld: {t.withheld_reason}._")
        if t.note:
            lines.append("")
            lines.append(t.note)
        return "\n".join(lines)

    header = [t.row_label or ""] + list(t.cols)
    body: list[list[str]] = []
    for i, rname in enumerate(t.rows):
        row = [rname]
        for j in range(len(t.cols)):
            if t.suppressed[i, j]:
                row.append(SUPPRESSED)
            elif t.kind == "frequency":
                row.append(f"{int(round(t.values[i, j]))}")
            else:
                v = float(t.values[i, j])
                # integer-valued magnitude cells (counts in an n column) print as
                # integers; 1.05e+03 is a formatting cue unrelated to any error
                row.append(f"{int(v)}" if v.is_integer() else f"{v:.3g}")
        body.append(row)

    if t.show_margins and len(t.cols) > 1 and t.kind == "frequency" and t.margins is not None:
        totals = ["Total"] + [f"{int(round(v))}" for v in t.margins]
        body.append(totals)

    widths = [
        max([len(header[c])] + [len(r[c]) for r in body]) for c in range(len(header))
    ]
    fmt = lambda cells: "| " + " | ".join(
        c.ljust(widths[i]) for i, c in enumerate(cells)
    ) + " |"

    lines.append("")
    lines.append(fmt(header))
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in body:
        lines.append(fmt(r))

    footnotes = []
    if t.suppressed.any() and not t.whole_table_suppressed:
        footnotes.append(f"{SUPPRESSED} withheld under disclosure control.")
    if t.rounded:
        footnotes.append("Counts rounded.")
    if t.note:
        footnotes.append(t.note)
    if footnotes:
        lines.append("")
        lines.append(" ".join(footnotes))
    return "\n".join(lines)
