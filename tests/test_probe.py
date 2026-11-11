"""The probe experiment's instruments (docs/vetting.md, section 4)."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from opacity.probe import depth_index, key_str
from opacity.probe import extract as E
from opacity.probe import fit as F


def test_depths_are_a_quarter_half_and_three_quarters():
    assert depth_index(64) == [16, 32, 48]  # Qwen3.5 27B
    assert depth_index(24) == [6, 12, 18]  # gpt-oss 20B


def test_first_token_must_begin_the_logged_reply():
    assert E.first_token_ok("{", '{\n  "error_present": "no"')
    assert not E.first_token_ok("```", '{\n  "error_present": "no"')
    assert not E.first_token_ok("", "{")


def test_folds_keep_tasks_whole_and_cover_everything():
    groups = np.array([f"h{i % 13:02d}" for i in range(200)])
    fs = F.folds(groups)
    assert sorted(np.concatenate(fs).tolist()) == list(range(200))
    for te in fs:
        assert not set(groups[te]) & set(np.delete(groups, te))


def _synthetic(n_tasks=18, per=24, d=12, signal_depth=1, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"h{i:02d}" for i in range(n_tasks)], per)
    y = rng.integers(0, 2, len(groups))
    states = rng.normal(size=(len(groups), 3, d)).astype(np.float16)
    states[:, signal_depth, 0] += 2.0 * y
    return states, y, groups


def test_selection_finds_the_depth_that_carries_the_signal():
    states, y, groups = _synthetic()
    depth, C, table = F.select(states, y, groups)
    assert depth == 1
    assert len(table) == 3 * len(F.CS)
    assert max(r["cv_auc"] for r in table) > 0.8


def test_paired_difference_is_zero_for_identical_scores_and_large_for_a_perfect_one():
    rng = np.random.default_rng(1)
    task = np.repeat(np.arange(12), 10)
    y = rng.integers(0, 2, 120)
    noise = rng.normal(size=120)
    assert F.paired_diff(task, y, noise, noise) == (0.0, 0.0, 0.0)
    d, lo, hi = F.paired_diff(task, y, y + 0.01 * noise, noise)
    assert d > 0.3 and lo > 0.2


def test_within_task_auc_ignores_which_tasks_are_risky():
    task = np.repeat(["a", "b"], 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    task_risk = np.array([0.2] * 4 + [0.8] * 4)  # knows only that b is riskier
    assert F.within_auc(task, y, task_risk) == 0.5
    assert F.within_auc(task, y, y + task_risk) == 1.0
    assert F.within_auc(task, y, -y + task_risk) == 0.0


def test_paired_within_difference_is_zero_for_identical_scores():
    rng = np.random.default_rng(3)
    task = np.repeat(np.arange(10), 12)
    y = rng.integers(0, 2, 120)
    s = rng.normal(size=120)
    assert F.paired_within(task, y, s, s) == (0.0, 0.0, 0.0)


def test_heldout_is_scored_once_unless_a_rerun_is_given_a_reason(tmp_path):
    first = F.results_path(tmp_path, "Qwen3.5-27B", None)
    assert first.name == "Qwen3.5-27B-heldout.json"
    first.write_text("{}")
    with pytest.raises(SystemExit):
        F.results_path(tmp_path, "Qwen3.5-27B", None)
    assert F.results_path(tmp_path, "Qwen3.5-27B", "parser bug").name == "Qwen3.5-27B-heldout-2.json"
    assert F.results_path(tmp_path, "gpt-oss-20b", None).name == "gpt-oss-20b-heldout.json"


def _tag_dir(tmp_path, n_tokens_off=0, verbal_signal=2.0):
    """A tag directory as probe_inputs.py and extract.py would leave it, with a planted
    signal in the states and a verbal checker about as weak as the real ones."""
    states, y, groups = _synthetic(n_tasks=24, per=20)
    ts = np.where(np.isin(groups, [f"h{i:02d}" for i in range(16)]), "dev", "heldout")
    keys = [key_str("log.eval", f"{g}-r{i}") for i, g in enumerate(groups)]
    with open(tmp_path / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "task_set", "task_id", "analyst", "condition", "wrong", "wrong2"])
        for i, k in enumerate(keys):
            w.writerow([k, ts[i], groups[i], f"A{i % 2}|open", "invited", int(y[i]), int(y[i]) if i % 3 else ""])
    rng = np.random.default_rng(2)
    with open(tmp_path / "verbal.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "checker", "score", "clean_no"])
        for i, k in enumerate(keys):
            s = float(50 + verbal_signal * y[i] + rng.normal(scale=4))
            w.writerow([k, "Qwen3.5-27B|off", s, int(s < 50)])
    (tmp_path / "states").mkdir()
    n = np.full(len(keys), 2000)
    done = {k: {"state": states[i], "n_tokens": int(n[i]), "check_tokens": int(n[i] + (n_tokens_off if i == 0 else 0)),
                "logged_tokens": int(n[i]), "top1": 0, "top1_text": "{", "first_ok": 1, "seconds": 0.1}
            for i, k in enumerate(keys)}
    E._save(tmp_path / "states" / "Qwen3.5-27B.npz", done,
            {"probe": "Qwen3.5-27B", "depths": [16, 32, 48]})
    return tmp_path


def test_a_full_run_on_planted_data_meets_the_stop_rule_and_refuses_a_second_look(tmp_path):
    d = _tag_dir(tmp_path)
    r = F.run("Qwen3.5-27B", d, d / "states" / "Qwen3.5-27B.npz", check_git=False)
    assert r["selection"]["depth"] == 1
    assert r["probe_heldout"]["auc"] > 0.85
    assert r["best_verbal"] == "Qwen3.5-27B|off"
    assert r["stop_rule_met"] is True
    assert 0.1 < r["operating_point"]["stopped_right"] < 0.35
    assert json.loads((d / "results" / "Qwen3.5-27B-heldout.json").read_text())["n_heldout"] == 160
    assert r["within_task"]["probe"] > 0.85  # the planted signal is within tasks
    assert abs(r["within_task"]["shuffled_mean"] - 0.5) < 0.1
    rows = list(csv.DictReader(open(d / "results" / "Qwen3.5-27B-heldout-scores.csv")))
    assert len(rows) == 160 and "Qwen3.5-27B|off" in rows[0]
    with pytest.raises(SystemExit):
        F.run("Qwen3.5-27B", d, d / "states" / "Qwen3.5-27B.npz", check_git=False)


def test_the_stop_rule_fails_when_the_verbal_checker_is_as_good(tmp_path):
    d = _tag_dir(tmp_path, verbal_signal=40.0)
    r = F.run("Qwen3.5-27B", d, d / "states" / "Qwen3.5-27B.npz", check_git=False)
    assert r["verbal_heldout"]["Qwen3.5-27B|off"]["auc"] > r["probe_heldout"]["auc"]
    assert r["stop_rule_met"] is False


def test_a_prompt_not_rebuilt_exactly_stops_the_fit(tmp_path):
    d = _tag_dir(tmp_path, n_tokens_off=1)
    with pytest.raises(SystemExit, match="prompt lengths"):
        F.run("Qwen3.5-27B", d, d / "states" / "Qwen3.5-27B.npz", check_git=False)


def test_checkpoint_round_trip(tmp_path):
    done = {"a|1": {"state": np.ones((3, 4), np.float16), "n_tokens": 7, "check_tokens": 7, "logged_tokens": 7, "top1": 5,
                    "top1_text": "{", "first_ok": -1, "seconds": 0.5}}
    E._save(tmp_path / "x.npz", done, {"probe": "p"})
    back = E._load(tmp_path / "x.npz")
    assert back["a|1"]["n_tokens"] == 7 and back["a|1"]["top1_text"] == "{"
    assert np.array_equal(back["a|1"]["state"], done["a|1"]["state"])


# --- the positive control (docs/vetting.md 4.8) --------------------------------

def _steer_dir(tmp_path, signal=2.0):
    from opacity.probe import STEER

    rng = np.random.default_rng(5)
    rows, lines, done = [], [], {}
    for ts, tasks in (("dev", range(0, 18)), ("heldout", range(18, 30))):
        for t in tasks:
            for rep in range(4):
                for steered in (0, 1):
                    k = key_str(f"{'steered' if steered else 'open'}-{ts}.eval", f"h{t:02d}-r{rep}")
                    rows.append([k, ts, f"h{t:02d}", rep, steered, 1, int(rng.random() < 0.3), int(steered and rep == 0 and t == 20)])
                    words = ["effect", "table", "rate"] + (["remarkable"] if steered and rng.random() < 0.6 else [])
                    lines.append(json.dumps({"key": k, "task_set": ts, "messages": [], "completion": " ".join(words)}))
                    st = rng.normal(size=(3, 10)).astype(np.float16)
                    st[2, 0] += signal * steered
                    done[k] = {"state": st, "n_tokens": 900, "check_tokens": 900, "logged_tokens": 900,
                               "completion_tokens": 300, "agree": 0.9, "top1": 0, "top1_text": "x", "first_ok": -1,
                               "seconds": 0.1}
    with open(tmp_path / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "task_set", "task_id", "replicate", "steered", "parse_ok", "verifier_flags", "leaks"])
        w.writerows(rows)
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / f"{STEER['probe']}.jsonl").write_text("\n".join(lines) + "\n")
    (tmp_path / "states").mkdir()
    E._save(tmp_path / "states" / f"{STEER['probe']}.npz", done, {"probe": STEER["probe"], "depths": [16, 32, 48]})
    return tmp_path


def test_steer_run_finds_a_planted_signal_and_scores_heldout_once(tmp_path):
    from opacity.probe import steer as T

    d = _steer_dir(tmp_path)
    none = [str(tmp_path / "no-such-*.eval")]
    r = T.run(d, none, none, check_git=False)
    assert r["selection"]["depth"] == 2
    assert r["probe_within"]["auc"] > 0.85 and r["hit_rate"] > 0.5
    assert r["false_alarm"] < 0.25
    assert set(r["blackbox_within"]) == {"text classifier", "verifier"}
    assert r["missing_checkers"] == ["Qwen3.5-27B|off|pack", "gpt-oss-20b|effort-medium"]
    assert r["leaking_steered"] == {"dev": 0, "heldout": 1}
    assert r["probe_within_without_leaks"]["n"] == r["probe_within"]["n"] - 1
    assert abs(r["shuffled_within"]["mean"] - 0.5) < 0.12
    with pytest.raises(SystemExit):
        T.run(d, none, none, check_git=False)


def test_within_ci_brackets_its_point():
    from opacity.probe import steer as T

    rng = np.random.default_rng(6)
    task = np.repeat(np.arange(12), 8)
    y = np.tile([0, 1], 48)
    s = y + rng.normal(size=96)
    c = T.within_ci(task, y, s)
    assert c["lo"] <= c["auc"] <= c["hi"] and c["n"] == 96
