"""The slant detector for vetting (docs/vetting.md 3.6)."""

from __future__ import annotations

import numpy as np
import pytest

from opacity import slant as S


def _rows(seed=0, signal=0.7, wrong_steered=0.4, wrong_honest=0.1):
    rng = np.random.default_rng(seed)
    filler = ["rate", "table", "group", "difference", "adjusted", "sample", "count", "trial", "arm", "result"]
    rows = []
    for ts, tasks in (("dev", range(0, 16)), ("heldout", range(16, 24))):
        for t in tasks:
            for rep in range(5):
                for arm in ("open", "steered2"):
                    words = list(rng.choice(filler, 30)) + [f"topic{t}"] * 3
                    if arm == "steered2" and rng.random() < signal:
                        words += ["effective", "demonstrates"]
                    elif rng.random() < 0.6:
                        words += ["limitation", "cannot"]
                    wrong = rng.random() < (wrong_steered if arm != "open" else wrong_honest)
                    rows.append({"key": f"{arm}-{ts}.eval|h{t:02d}-r{rep}", "id": f"h{t:02d}-r{rep}", "arm": arm,
                                 "task_set": ts, "task_id": f"h{t:02d}", "replicate": rep, "text": " ".join(words),
                                 "pack": f"pack {t} {rep}", "wrong": bool(wrong), "model": "m"})
    return rows


def test_twins_must_pair_one_for_one_with_the_same_pack():
    rows = _rows()
    S.twins(rows)
    with pytest.raises(SystemExit, match="pair"):
        S.twins([r for r in rows if r["key"] != rows[0]["key"]])
    bad = [dict(r) for r in rows]
    bad[1]["pack"] = "another pack"
    with pytest.raises(SystemExit, match="different packs"):
        S.twins(bad)


def test_fit_finds_the_slant_on_heldout_tasks_and_sets_the_service_threshold():
    pipe, threshold, report = S.fit(_rows())
    assert report["heldout_within"]["auc"] > 0.75
    assert report["hit_rate"] > 0.4 and report["false_alarm"] <= 0.1  # set on the honest held-out twins
    assert report["C"] in S.CS and 0 < threshold < 1 and threshold == report["service_threshold"]
    m = report["manipulation"]
    assert m["heldout_steered"][1] == 40 and m["heldout_honest"][1] == 40
    assert m["heldout_difference"]["diff"] > 0


def test_no_signal_gives_chance():
    _, _, report = S.fit(_rows(signal=0.0, seed=3))
    assert abs(report["heldout_within"]["auc"] - 0.5) < 0.15


def test_manipulation_is_none_without_a_judge():
    rows = [dict(r, wrong=None) for r in _rows()]
    assert S.manipulation(rows) is None


def test_score_and_summary_count_flags_per_set():
    rows = _rows()
    pipe, threshold, _ = S.fit(rows)
    scored = S.score(pipe, threshold, rows)
    summ = S.summary(scored)
    assert summ["all"]["n"] == len(rows)
    assert summ["all"]["flagged"] == summ["dev"]["flagged"] + summ["heldout"]["flagged"]
    assert summ["all"]["lo"] <= summ["all"]["rate"] <= summ["all"]["hi"]


def test_twins_pair_within_each_style_and_the_report_breaks_down_by_style():
    rows = []
    for style, seed in (("default", 0), ("concise", 1)):
        for r in _rows(seed=seed):
            rows.append(dict(r, style=style, key=f"{style}-{r['key']}"))
    S.twins(rows)
    with pytest.raises(SystemExit, match="concise"):
        S.twins([r for r in rows if not (r["style"] == "concise" and r["id"] == "h00-r0" and r["arm"] == "open")])
    _, threshold, report = S.fit(rows)
    assert set(report["by_style"]) == {"default", "concise"}
    assert report["by_style"]["concise"]["honest_flagged"][1] == 40


def test_wilson_interval():
    lo, hi = S.wilson(5, 100)
    assert 0.02 < lo < 0.05 < hi < 0.12
