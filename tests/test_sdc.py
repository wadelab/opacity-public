"""Invariants of the disclosure-control layer.

These are not unit tests of convenience; each one is a property the study's
inference depends on. If one fails, the concealment mechanism it guards has
changed and the pre-registered prediction for that class no longer follows.
"""

import numpy as np
import pytest
from dataclasses import replace

from opacity.sdc import (
    S0, S1, S2, S3, SDCPolicy, Table, apply_to_table, render_table, policy,
    _round_base, SUPPRESSED,
)


def freq(values, rows=None, cols=None, **kw):
    v = np.asarray(values, dtype=float)
    rows = rows or [f"r{i}" for i in range(v.shape[0])]
    cols = cols or [f"c{j}" for j in range(v.shape[1])]
    return Table(name="t", rows=rows, cols=cols, values=v, kind="frequency", **kw)


# -- rounding ---------------------------------------------------------------

def test_round_base_halves_away_from_zero_not_to_even():
    # numpy would give 12.5 -> 12 at base 5 (half to even); SDC does not.
    assert _round_base(12.5, 5) == 15
    assert _round_base(7.5, 5) == 10
    assert _round_base(7.4, 5) == 5
    assert _round_base(0, 5) == 0


def test_rounding_applies_to_margins_from_true_values():
    t = freq([[7, 3], [8, 4]])
    out = apply_to_table(t, SDCPolicy(level=9, label="x", rounding_base=5))
    # cells: 7->5, 3->5, 8->10, 4->5 ; true col totals 15, 7 -> 15, 5
    assert out.values.tolist() == [[5, 5], [10, 5]]
    assert out.margins.tolist() == [15, 5]
    # margin need not equal sum of displayed cells (15 != 5+10 for col 0? it does
    # here by chance; col 1: 5 != 5+5=10). That mismatch is real SDC behaviour.
    assert out.margins[1] != out.values[:, 1].sum()


# -- s0 is the identity ------------------------------------------------------

def test_s0_changes_nothing():
    t = freq([[1, 2], [3, 4]])
    out = apply_to_table(t, S0)
    assert out.values.tolist() == [[1, 2], [3, 4]]
    assert not out.suppressed.any()
    assert render_table(out) == render_table(t)


# -- primary suppression -----------------------------------------------------

def test_threshold_suppresses_small_nonzero_cells_only():
    t = freq([[0, 3, 12]], show_margins=False)
    out = apply_to_table(t, SDCPolicy(level=9, label="x", threshold=5))
    assert out.suppressed.tolist() == [[False, True, False]]


# -- complementary suppression -----------------------------------------------

def test_single_suppressed_cell_in_a_row_with_margin_triggers_complement():
    # Row [3, 50, 60] with visible margin: 3 suppressed alone is recoverable.
    t = freq([[3, 50, 60], [20, 20, 20]])
    out = apply_to_table(t, SDCPolicy(level=9, label="x", threshold=5, complementary=True))
    row0 = out.suppressed[0]
    assert row0.sum() >= 2, "a lone suppressed cell must be complemented"
    # the complement should be the smallest remaining cell (50), not 60
    assert row0.tolist() == [True, True, False]


def test_no_margins_means_no_complementary_suppression():
    t = freq([[3, 50, 60]], show_margins=False)
    out = apply_to_table(t, SDCPolicy(level=9, label="x", threshold=5, complementary=True))
    assert out.suppressed.tolist() == [[True, False, False]]


def test_complementary_cascades_across_rows_and_columns():
    # One small cell at (1,1). Column complement hits (0,1) or (2,1); then that
    # row has a lone suppression and gets a complement too. Must converge.
    t = freq([[20, 9, 20], [20, 2, 20], [20, 30, 20]])
    out = apply_to_table(t, SDCPolicy(level=9, label="x", threshold=5, complementary=True))
    s = out.suppressed
    for i in range(3):
        assert s[i].sum() != 1, f"row {i} has a lone suppressed cell"
    for j in range(3):
        assert s[:, j].sum() != 1, f"col {j} has a lone suppressed cell"


# -- row collapse: the class 1 mechanism ---------------------------------------

@pytest.fixture
def subgroup_table():
    # one large subgroup, 19 small ones -- the multiplicity shape
    vals = np.array([[90, 40]] + [[6, 2]] * 19, dtype=float)
    return freq(vals, rows=[f"R{i}" for i in range(20)], cols=["n", "events"])


def test_keep_leaves_row_count_visible(subgroup_table):
    out = apply_to_table(subgroup_table, SDCPolicy(level=9, label="x", threshold=10,
                                                  complementary=True, row_collapse="keep"))
    assert len(out.rows) == 20  # reviewer can still count the comparisons


def test_other_conceals_row_count(subgroup_table):
    out = apply_to_table(subgroup_table, SDCPolicy(level=9, label="x", threshold=10,
                                                  complementary=True, row_collapse="other"))
    assert out.rows == ["R0", "Other"]
    assert out.suppressed[1].all()
    # margin still reflects the collapsed rows
    assert out.margins[0] == 90 + 6 * 19


def test_drop_conceals_existence(subgroup_table):
    out = apply_to_table(subgroup_table, SDCPolicy(level=9, label="x", threshold=10,
                                                  complementary=True, row_collapse="drop"))
    assert out.rows == ["R0"]
    assert SUPPRESSED not in render_table(out)


def test_whole_table_rule_only_fires_on_stratified_tables(subgroup_table):
    pol = SDCPolicy(level=9, label="x", threshold=10, complementary=True,
                    min_populated_strata=3, row_collapse="drop")
    out = apply_to_table(subgroup_table, pol)
    assert not out.whole_table_suppressed, "subgroup-results table must not be withheld"

    subgroup_table.stratified = True
    out2 = apply_to_table(subgroup_table, pol)
    assert out2.whole_table_suppressed


# -- hide_margins: the s3 anti-differencing step ---------------------------------

def test_hide_margins_removes_total_row():
    t = freq([[3, 50], [20, 20]])
    out = apply_to_table(t, SDCPolicy(level=9, label="x", threshold=5,
                                      complementary=True, hide_margins=True))
    assert "Total" not in render_table(out)


# -- the canonical levels are monotone in strictness ------------------------------

def test_levels_are_monotone():
    assert S0.threshold <= S1.threshold <= S2.threshold <= S3.threshold
    assert S0.rounding_base <= S1.rounding_base <= S2.rounding_base <= S3.rounding_base
    order = {"full": 0, "sizes": 1, "count": 2, "none": 3}
    assert order[S0.cluster_disclosure] <= order[S1.cluster_disclosure] \
        <= order[S2.cluster_disclosure] <= order[S3.cluster_disclosure]
    order = {"keep": 0, "other": 1, "drop": 2}
    assert order[S0.row_collapse] <= order[S1.row_collapse] \
        <= order[S2.row_collapse] <= order[S3.row_collapse]


def test_policy_lookup_rejects_unknown_level():
    with pytest.raises(ValueError):
        policy(7)


def test_s4_extends_s3_and_withholds_tables():
    from opacity.sdc import S4, GRADED_LEVELS
    assert GRADED_LEVELS == (0, 1, 2, 3)
    assert policy(4) is S4 and S4.withhold_tables and not S3.withhold_tables
    for f in ("threshold", "rounding_base", "cluster_disclosure", "row_collapse",
              "hide_margins", "stat_precision", "show_sample_list", "min_comparable_strata"):
        assert getattr(S4, f) == getattr(S3, f), f


def test_statistic_precision_and_sample_list_by_level():
    assert [policy(l).stat_precision for l in range(5)] == ["exact", "exact", "banded", "banded", "banded"]
    assert [policy(l).show_sample_list for l in range(5)] == [True, True, True, False, False]


def test_p_band():
    from opacity.sdc import p_band
    assert p_band(0.0004) == "p < 0.001"
    assert p_band(0.009) == "p < 0.01"
    assert p_band(0.03) == "p < 0.05"
    assert p_band(0.18) == "p > 0.05"


def test_stratified_table_withheld_without_a_comparable_stratum():
    pol = replace(S3, min_comparable_strata=1)
    t = Table(
        name="strata", rows=["a", "b", "c", "d"], cols=["x n", "x ev", "y n", "y ev"],
        values=np.array([[9, 1, 200, 20], [18, 2, 120, 20], [120, 40, 18, 8], [150, 110, 9, 7]]),
        stratified=True,
    )
    out = apply_to_table(t, pol)
    assert out.whole_table_suppressed and "no stratum releasable" in out.withheld_reason
    assert "withheld" in render_table(out)
    # a table with one fully visible stratum is released (two small rows, so
    # complementary suppression has partners and leaves the big row alone)
    t2 = Table(
        name="strata", rows=["a", "b", "c"], cols=["x n", "x ev", "y n", "y ev"],
        values=np.array([[90, 30, 90, 40], [5, 1, 5, 1], [6, 2, 6, 2]]), stratified=True,
    )
    out2 = apply_to_table(t2, replace(pol, min_populated_strata=0))
    assert not out2.whole_table_suppressed


def test_withhold_tables_marks_every_table():
    from opacity.sdc import S4
    t = Table(name="t", rows=["a"], cols=["n"], values=np.array([[500]]))
    out = apply_to_table(t, S4)
    assert out.whole_table_suppressed and "under disclosure control" in render_table(out)


# -- magnitude cells: integer-valued counts print as integers -----------------------

def test_integer_valued_magnitude_cells_render_as_integers():
    t = Table(name="m", rows=["a"], cols=["n", "Mean"], values=np.array([[1050.0, 52.25]]),
              kind="magnitude", counts=np.array([[1050, 1050]]), show_margins=False)
    r = render_table(t)
    assert "| 1050 " in r and "e+03" not in r
    assert "52.2" in r


# -- a table that loses every row is withheld, not rendered as a bare header ----------

def test_table_with_no_surviving_rows_is_withheld():
    t = freq([[3, 2], [4, 1]], show_margins=False)
    out = apply_to_table(t, SDCPolicy(level=9, label="x", threshold=5, row_collapse="drop"))
    assert out.whole_table_suppressed
    assert "withheld" in render_table(out)
