"""Invariants of the seeded items.

Each guards a pre-registered prediction. The most important is the class 4
invariance: the internal control must render identically at every disclosure
level, or it cannot do its job of separating reviewer fatigue from concealment.
"""

import re

import pytest

from opacity.bundle import Bundle
from opacity.items import build_corpus, IMPLEMENTED, Item
from opacity.sdc import policy, SUPPRESSED


@pytest.fixture(scope="module")
def corpus() -> list[Item]:
    return build_corpus(seed=0)


def rendered(b: Bundle, level: int) -> str:
    return b.controlled(policy(level)).render()


# -- structural -----------------------------------------------------------------

def test_every_implemented_class_is_present(corpus):
    assert sorted(i.error_class for i in corpus) == list(IMPLEMENTED)


def test_ground_truth_is_never_rendered(corpus):
    for item in corpus:
        for b in item.pair:
            for lvl in range(4):
                r = rendered(b, lvl)
                assert b.truth.rationale not in r
                assert b.truth.short_name not in r
                assert "GroundTruth" not in r


def test_pairs_are_matched_on_data(corpus):
    """Seeded and control share tables; they differ only in analysis/prose."""
    for item in corpus:
        e, c = item.pair
        assert len(e.tables) == len(c.tables)
        for te, tc in zip(e.tables, c.tables):
            assert te.values.tolist() == tc.values.tolist()
            assert te.rows == tc.rows and te.cols == tc.cols


def test_deterministic_in_seed():
    a = build_corpus(seed=42)
    b = build_corpus(seed=42)
    for x, y in zip(a, b):
        assert rendered(x.seeded, 0) == rendered(y.seeded, 0)
    c = build_corpus(seed=43)
    assert any(rendered(x.seeded, 0) != rendered(z.seeded, 0) for x, z in zip(a, c))


def test_no_sloppy_p_values(corpus):
    for item in corpus:
        for b in item.pair:
            assert not re.search(r"p = 0\.0000\b", b.interpretation)


# -- class 4: the internal control -------------------------------------------------

def test_class4_is_invariant_across_disclosure_levels(corpus):
    """Over the graded levels s0 to s3. s4 withholds every table, class 4's
    included, so the internal control does not extend to the ablation."""
    item = next(i for i in corpus if i.error_class == 4)
    for b in item.pair:
        r0 = rendered(b, 0)
        for lvl in (1, 2, 3):
            assert rendered(b, lvl) == r0, f"class 4 changed at s{lvl}"
        assert "| Q1 lowest" not in rendered(b, 4)


# -- class 1: denominator concealment ------------------------------------------------

def test_class1_comparison_count_visible_at_s0_concealed_by_s2(corpus):
    item = next(i for i in corpus if i.error_class == 1)
    b = item.seeded
    r0 = rendered(b, 0)
    assert len(re.findall(r"\| Region [A-T] ", r0)) == 20, "all 20 comparisons visible at s0"
    r2 = rendered(b, 2)
    assert len(re.findall(r"\| Region [A-T] ", r2)) == 1
    assert "| Other" in r2
    r3 = rendered(b, 3)
    assert len(re.findall(r"\| Region [A-T] ", r3)) == 1
    assert "| Other" not in r3
    assert SUPPRESSED not in r3, "s3 drops rows; nothing marked suppressed"
    assert "Total" not in r3, "s3 hides margins"
    assert "## Sample" not in r3, "s3 withholds the sample list, so the dropped rows leave no total behind"


def test_quoted_statistics_follow_the_policy(corpus):
    """A rounded table is never accompanied by an exact statistic it cannot
    reproduce: from s2 the statistic goes and p is a band."""
    for cls in (1, 5, 7):
        item = next(i for i in corpus if i.error_class == cls)
        for b in item.pair:
            r0, r1 = rendered(b, 0), rendered(b, 1)
            assert re.search(r"(chi-square|t) = -?\d+\.\d\d, ", r0), f"class {cls}: exact statistic at s0"
            assert re.search(r"p (= 0\.\d{3}|< 0\.001)", r0)
            assert r1.split("## Interpretation")[1] == r0.split("## Interpretation")[1], "s1 does not round, so prose is exact"
            for lvl in (2, 3, 4):
                interp = rendered(b, lvl).split("## Interpretation")[1]
                assert not re.search(r"(chi-square|t) = ", interp), f"class {cls}: statistic dropped at s{lvl}"
                assert re.search(r"p [<>] 0\.0", interp), f"class {cls}: p banded at s{lvl}"
                assert not re.search(r"p = 0\.\d", interp)


def test_s4_is_prose_only(corpus):
    for item in corpus:
        for b in item.pair:
            r4 = rendered(b, 4)
            assert "_All tables withheld under disclosure control._" in r4
            assert not any(line.startswith("|") for line in r4.splitlines())
            assert "## Sample" not in r4
            assert "## Methods" in r4 and "## Interpretation" in r4


# -- class 3: strata concealment --------------------------------------------------------

def _stratum_pairs_visible(text: str) -> int:
    """Count strata rows where both the treated and control pair are visible."""
    n = 0
    for line in text.splitlines():
        if not line.startswith("| ") or "Treated n" in line or line.startswith("| Total"):
            continue
        if not any(k in line for k in ("Low acuity", "Medium acuity", "High acuity", "Critical")):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")][1:]
        if len(cells) == 4 and SUPPRESSED not in cells:
            n += 1
    return n


def test_class3_strata_visible_at_s0_none_comparable_by_s2(corpus):
    item = next(i for i in corpus if i.error_class == 3)
    b = item.seeded
    assert _stratum_pairs_visible(rendered(b, 0)) == 4
    assert _stratum_pairs_visible(rendered(b, 1)) >= 1
    assert _stratum_pairs_visible(rendered(b, 2)) == 0
    r3 = rendered(b, 3)
    assert _stratum_pairs_visible(r3) == 0
    assert "Total" not in r3
    assert "no stratum releasable in full" in r3, (
        "with no comparable stratum the stratum table is withheld at s3, so the "
        "per-stratum arm sizes no longer disclose the allocation imbalance"
    )
    assert "| Low acuity" not in r3 and "| Critical" not in r3
    assert "| Treated " in r3, "the aggregate table still stands alone"


# -- class 5: cluster redaction ----------------------------------------------------------

def test_class5_cluster_disclosure_degrades_in_steps(corpus):
    item = next(i for i in corpus if i.error_class == 5)
    b = item.seeded
    notes = [b.controlled(policy(l)).cluster_note for l in range(4)]
    assert "site" in notes[0] and "25 per site" in notes[0]
    assert "group sizes" in notes[1] and "site" not in notes[1]
    assert "12 groups" in notes[2] and "sizes" not in notes[2]
    assert notes[3] == ""
    r3 = rendered(b, 3)
    for word in ("site", "cluster", "group"):
        assert word not in r3.lower()


# -- the control bundle must not leak via a different mechanism ----------------------------

def test_controls_are_not_trivially_distinguishable_by_length(corpus):
    """A control should not be systematically longer/shorter than its pair."""
    ratios = []
    for item in corpus:
        e, c = item.pair
        ratios.append(len(rendered(c, 0)) / len(rendered(e, 0)))
    # loose bound; the point is to catch a gross asymmetry, not enforce equality
    assert all(0.7 < r < 1.6 for r in ratios), ratios


# -- class 2: attrition concealed by rounding and stage collapse ------------------------

def _row_numbers(text: str, table_name: str, row_label: str) -> list[int]:
    """Integer cells of one row of one table in a render."""
    block = text.split(f"**{table_name}**", 1)[1].split("**", 1)[0]
    for line in block.splitlines():
        if line.startswith(f"| {row_label} "):
            cells = [c.strip() for c in line.strip("|").split("|")][1:]
            return [int(c) for c in cells if c.isdigit()]
    raise AssertionError(f"row {row_label!r} not found in {table_name!r}")


def test_class2_attrition_visible_at_s0_concealed_at_s3(corpus):
    item = next(i for i in corpus if i.error_class == 2)
    b = item.seeded
    flow = "Table 1. Participant flow by arm"
    enrolled, analysed = _row_numbers(rendered(b, 0), flow, "Treated")
    assert analysed < enrolled, "attrition visible at s0"
    assert "Analysed: n = 596" in rendered(b, 0)
    enrolled, analysed = _row_numbers(rendered(b, 3), flow, "Treated")
    assert analysed == enrolled, "attrition rounded away at s3"
    r3 = rendered(b, 3)
    assert "## Sample" not in r3, "s3 withholds the sample list"
    assert "Total" not in r3
    r2 = rendered(b, 2)
    assert "Enrolled: n = 600" in r2 and "Analysed: n = 595" in r2, "at s2 the step of five is still visible"


# -- class 6: correct base in the margin, withheld at s3 ----------------------------------

def test_class6_margin_visible_at_s0_table_withheld_at_s3(corpus):
    item = next(i for i in corpus if i.error_class == 6)
    b = item.seeded
    r0 = rendered(b, 0)
    patients, deaths = _row_numbers(r0, "Table 1. Thirty-day deaths by age band", "Total")
    assert patients > 0 and 0 < deaths / patients < 0.1, "correct rate recoverable at s0"
    assert "Patients: n =" in r0, "the per-patient stage is visible at s0"
    r3 = rendered(b, 3)
    assert "withheld" in r3 and "Total" not in r3
    assert "Patients: n =" not in r3, "stage collapse removes the per-patient line"


# -- class 7: selection step collapses, pre-post table suppresses then withdraws -----------

def test_class7_selection_step_visible_at_s0_gone_by_s2(corpus):
    item = next(i for i in corpus if i.error_class == 7)
    b = item.seeded
    for lvl in (0, 1):
        r = rendered(b, lvl)
        assert "Selected: n =" in r and "below 10th centile)" in r
        assert "| School 1 " in r
    r2 = rendered(b, 2)
    assert "Selected" not in r2
    assert "| School 1 " not in r2 and "| Other" in r2
    r3 = rendered(b, 3)
    assert "Selected" not in r3
    assert "withheld" in r3


# -- class 8: the development line collapses; partial concealment by design ----------------

def test_class8_development_stage_visible_at_s0_gone_by_s2(corpus):
    item = next(i for i in corpus if i.error_class == 8)
    b = item.seeded
    for lvl in (0, 1):
        r = rendered(b, lvl)
        assert "Model development: n = 188" in r
        assert "Performance evaluation: n = 188" in r
    r2 = rendered(b, 2)
    assert "Model development" not in r2
    assert "Performance evaluation: n = 190" in r2
    assert "## Sample" not in rendered(b, 3)


def test_all_eight_classes_built(corpus):
    assert list(IMPLEMENTED) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_matched_register_removes_the_hedge_tell(corpus):
    """The ablation: matched controls say the correct thing in the seeded
    register. No hedging vocabulary, at least one recommending word, same
    tables, same ground truth."""
    from opacity.items import build_corpus
    hedges = re.compile(r"\b(cannot|declin\w*|consistent with chance|no evidence|not establish\w*|association|caution|would be needed|limitation|unable|not significant|null)\b", re.I)
    pushes = re.compile(r"\b(recommend\w*|should|ready for|robust|demonstrat\w*|confirm\w*|rollout|roll out|deploy\w*|extend\w*)\b", re.I)
    matched = build_corpus(seed=0, register="matched")
    for a, b in zip(corpus, matched):
        assert a.control.interpretation != b.control.interpretation
        assert not hedges.search(b.control.interpretation), b.control.interpretation
        assert pushes.search(b.control.interpretation), b.control.interpretation
        assert b.control.tables is not None and len(b.control.tables) == len(a.control.tables)
        assert b.control.truth == a.control.truth
        assert a.seeded.interpretation == b.seeded.interpretation
    r0 = matched[0].control.controlled(policy(0)).render()
    assert "{" not in r0 and "}" not in r0, "placeholders filled in the matched register too"
    with pytest.raises(ValueError):
        build_corpus(register="shouty")
