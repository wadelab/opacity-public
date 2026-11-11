"""scripts/auc_summary.py: the AUC, the checker score, checkers in series and
the operating point, on hand-built items."""

import importlib.util
from itertools import product
from pathlib import Path

import pytest


def _script():
    spec = importlib.util.spec_from_file_location("auc_summary", Path(__file__).parents[1] / "scripts" / "auc_summary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _script()


def _brute(pos, neg):
    return sum((p > q) + 0.5 * (p == q) for p, q in product(pos, neg)) / (len(pos) * len(neg))


def test_auc_matches_a_brute_force_count_with_ties():
    pos, neg = [80, 80, 50, 25, 90], [50, 50, 25, 10, 80, 50]
    assert A.auc(pos, neg) == pytest.approx(_brute(pos, neg))
    assert A.auc([1, 2], [1, 2]) == pytest.approx(0.5)
    assert A.auc([3], [1, 2]) == 1.0 and A.auc([1], [2, 3]) == 0.0


def test_auc_is_undefined_without_both_classes():
    assert A.auc([], [1.0]) != A.auc([], [1.0])  # nan


def test_score_folds_confidence_into_the_flag():
    assert A.score(("yes", 80, True)) == 90
    assert A.score(("no", 80, True)) == 10
    assert A.score(("cannot_tell", 80, True)) == 50
    assert A.score(("no", 80, False)) == 50  # an unparseable reply says nothing
    assert A.score(None) is None


def test_checkers_in_series_take_the_largest_score_and_need_every_clean_no():
    assert A.joint([("no", 90, True), ("yes", 60, True)]) == (80, False)
    assert A.joint([("no", 90, True), ("no", 20, True)]) == (40, True)
    assert A.joint([("no", 90, True), None]) == (None, False)


def test_split_names_the_task_set_and_keeps_fresh2_apart_from_fresh():
    assert A.split("Qwen3.5-27B|open|think1024|rules|fresh2|v2") == ("Qwen3.5-27B", "open|think1024|rules", "fresh2")
    assert A.split("Qwen3.5-27B|open|think1024|rules|fresh|v2")[2] == "fresh"
    assert A.split("Qwen3.5-27B|open|off|v2") == ("Qwen3.5-27B", "open|off", "dev")


def _it(sample, condition="invited", wrong=False, declined=False, parse_ok=True, task="h01", ts="dev"):
    return {"log": "L", "sample": sample, "task_id": task, "condition": condition, "declined": declined,
            "parse_ok": parse_ok, "wrong": wrong, "wrong3": wrong, "task_set": ts}


def test_operating_point_counts_like_sizing_rates_and_fails_closed():
    its = [_it("a", wrong=True), _it("b"), _it("c", declined=True), _it("d", parse_ok=False, wrong=None),
           _it("e", wrong=None), _it("f", condition="control"), _it("g", condition="control", wrong=True)]
    alone = A.operating(its, {}, [])
    # seeded: a wrong, b right, c withheld, d wrong (unparseable), e unjudged and dropped
    assert alone["n_seeded"] == 4 and alone["n_wrong_released"] == 2
    assert alone["wrong_released"] == 0.5 and alone["controls_answered"] == 1.0
    assert alone["correct_released"] == pytest.approx(2 / 6)
    checks = {"k": {("L", "a"): ("yes", 90, True), ("L", "b"): ("no", 90, True), ("L", "f"): ("no", 70, True)}}
    with_k = A.operating(its, checks, ["k"])
    # a flagged, d, e and g have no verdict so are withheld; b and f pass. As in
    # sizing.outcomes, a withheld write-up counts even when no judge scored it.
    assert with_k["n_wrong_released"] == 0 and with_k["n_seeded"] == 5 and with_k["controls_answered"] == 0.5
    assert with_k["correct_released"] == pytest.approx(2 / 7)


def test_auc_rows_leave_out_declined_and_unchecked_write_ups():
    its = [_it("a", wrong=True), _it("b"), _it("c", declined=True), _it("d", wrong=True)]
    checks = {"k": {("L", "a"): ("yes", 90, True), ("L", "b"): ("no", 90, True), ("L", "c"): ("no", 90, True)}}
    assert A.auc_rows(its, checks, ["k"]) == [("h01", True, 95.0), ("h01", False, 5.0)]


def test_fresh2_is_judged_by_the_alternative_judge():
    it = _it("a", ts="fresh2")
    it["wrong"], it["wrong3"] = None, True
    assert A.truth(it) is True
