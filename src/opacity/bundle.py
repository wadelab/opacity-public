"""Output bundles: what the agent releases and the reviewer sees.

A bundle is the unit of observation. It carries tables, a methods statement, an
interpretation, and the sample sizes claimed at each analysis stage. Disclosure
control transforms a bundle; the reviewer sees only the transformed version.

The ground truth attached to a bundle is never rendered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .sdc import SDCPolicy, Table, apply_to_table, render_table, _round_base, p_band, p_exact


@dataclass(frozen=True)
class Quote:
    """A number quoted in the prose, released under the same policy as the
    tables it summarises.

    kind  count  a count; rounded to the policy base when tables are rounded
          test   a test statistic with its p; exact as "chi-square = 6.87,
                 p = 0.009", banded as "p < 0.01"
          p      a p-value alone; exact or banded
    The interpretation and methods are templates with `{name}` placeholders;
    `Bundle.controlled` fills them for the policy in force. At s0 every quote
    renders exactly as it was computed.
    """

    kind: Literal["count", "test", "p"]
    value: float
    p: float | None = None
    stat_name: str = ""
    p_prefix: str = ""
    """Words placed before the p, e.g. "uncorrected " or "Bonferroni-corrected "."""

    def render(self, pol: SDCPolicy) -> str:
        if self.kind == "count":
            v = _round_base(self.value, pol.rounding_base) if pol.rounding_base else self.value
            return f"{int(round(v))}"
        banded = pol.stat_precision == "banded"
        if self.kind == "p":
            return self.p_prefix + (p_band(self.value) if banded else p_exact(self.value))
        assert self.p is not None, "a test quote needs its p"
        if banded:
            return self.p_prefix + p_band(self.p)
        return f"{self.stat_name} = {self.value:.2f}, {self.p_prefix}{p_exact(self.p)}"


def fill(template: str, quotes: dict[str, Quote], pol: SDCPolicy) -> str:
    if not quotes:
        return template
    return template.format_map({k: q.render(pol) for k, q in quotes.items()})


@dataclass(frozen=True)
class GroundTruth:
    """Never rendered. Written by the item generator, used only in scoring."""

    has_error: bool
    error_class: int | None
    """1-8 per docs/taxonomy.md; None for a clean control."""
    short_name: str
    rationale: str
    """One or two sentences: why this is wrong. Shown to adjudicators, never to
    reviewers."""
    detectable_at_s0: bool = True
    """Inclusion criterion. An item that fails this is dropped before use."""


@dataclass
class Stage:
    """One step of the analysis pipeline and the N surviving it."""

    name: str
    n: int
    note: str = ""


@dataclass
class Bundle:
    item_id: str
    title: str
    question: str
    """The analysis question the agent was given."""
    methods: str
    tables: list[Table] = field(default_factory=list)
    stages: list[Stage] = field(default_factory=list)
    interpretation: str = ""
    quotes: dict[str, Quote] = field(default_factory=dict)
    """Numbers quoted in `methods` and `interpretation`, by placeholder name."""
    interpretation_matched: str = ""
    """For a control: the same correct conclusion written in the seeded
    bundles' register, recommending the correct action in confident words
    with no hedging vocabulary. Used by `build_corpus(register="matched")`,
    the ablation that removes the register tell. Empty for seeded bundles."""
    results_note: str = ""
    """Shown in place of the tables when every table is withheld."""
    cluster_note: str = ""
    """Describes the clustering/repeated-measures structure, if any. Redacted in
    steps by disclosure level -- the class 5 concealment mechanism."""
    cluster_sizes: list[int] = field(default_factory=list)
    truth: GroundTruth | None = None
    disclosure_level: int = 0

    def controlled(self, pol: SDCPolicy) -> "Bundle":
        """Return a disclosure-controlled copy.

        Prose wording is never altered. Numbers quoted in it are released
        under the policy through `Quote`, so a rounded table is never
        accompanied by an exact statistic it cannot reproduce.
        """
        out = Bundle(
            item_id=self.item_id,
            title=self.title,
            question=self.question,
            methods=fill(self.methods, self.quotes, pol),
            tables=[apply_to_table(t, pol) for t in self.tables],
            stages=list(self.stages),
            interpretation=fill(self.interpretation, self.quotes, pol),
            quotes={},
            cluster_note=self.cluster_note,
            cluster_sizes=list(self.cluster_sizes),
            truth=self.truth,
            disclosure_level=pol.level,
        )

        if pol.withhold_tables and self.tables:
            out.tables = []
            out.results_note = "_All tables withheld under disclosure control._"

        if not pol.show_sample_list:
            out.stages = []

        if not pol.exact_n:
            base = pol.rounding_base or 5
            out.stages = [
                Stage(s.name, int(_round_base(s.n, base)), s.note) for s in out.stages
            ]

        # Collapse the stage table so attrition between steps is not visible as
        # a sequence of Ns. Keep first and last only.
        if not pol.exact_n and len(out.stages) > 2:
            out.stages = [out.stages[0], out.stages[-1]]

        cd = pol.cluster_disclosure
        if cd == "full":
            pass
        elif cd == "sizes":
            if out.cluster_sizes:
                sizes = ", ".join(str(s) for s in out.cluster_sizes)
                out.cluster_note = f"Observations are grouped; group sizes: {sizes}."
        elif cd == "count":
            if out.cluster_sizes:
                out.cluster_note = (
                    f"Observations are grouped into {len(out.cluster_sizes)} groups."
                )
            out.cluster_sizes = []
        elif cd == "none":
            out.cluster_note = ""
            out.cluster_sizes = []

        return out

    def render(self) -> str:
        """The text a reviewer reads. Ground truth is excluded by construction."""
        parts = [
            f"# {self.title}",
            "",
            "## Analysis question",
            "",
            self.question.strip(),
            "",
            "## Methods",
            "",
            self.methods.strip(),
        ]

        if self.cluster_note:
            parts += ["", self.cluster_note.strip()]

        if self.stages:
            parts += ["", "## Sample", ""]
            for s in self.stages:
                line = f"- {s.name}: n = {s.n}"
                if s.note:
                    line += f" ({s.note})"
                parts.append(line)

        if self.tables:
            parts += ["", "## Results", ""]
            for t in self.tables:
                parts.append(render_table(t))
                parts.append("")
        elif self.results_note:
            parts += ["", "## Results", "", self.results_note, ""]

        parts += ["## Interpretation", "", self.interpretation.strip(), ""]
        return "\n".join(parts)
