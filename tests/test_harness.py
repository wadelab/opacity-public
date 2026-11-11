"""The Inspect harness must present the same study the bespoke runner does.

Dataset shape, targets and metadata; the mechanical arm end to end without a
model; the scorer on verdicts of every kind.
"""

import asyncio

import pytest

inspect_ai = pytest.importorskip("inspect_ai")

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import ModelOutput  # noqa: E402
from inspect_ai.scorer import Target  # noqa: E402
from inspect_ai.solver import TaskState  # noqa: E402

from opacity.harness import opacity, samples, verdict  # noqa: E402
from opacity.harness.task import _levels  # noqa: E402
from opacity.items import IMPLEMENTED  # noqa: E402


def test_dataset_is_every_bundle_at_every_level():
    ss = samples(seed=0, replicates=1, levels=(0, 3))
    assert len(ss) == len(IMPLEMENTED) * 2 * 2
    ids = [s.id for s in ss]
    assert len(set(ids)) == len(ids)
    seeded = [s for s in ss if s.metadata["condition"] == "seeded"]
    control = [s for s in ss if s.metadata["condition"] == "control"]
    assert all(s.target == "yes" and s.metadata["error_class"] > 0 for s in seeded)
    assert all(s.target == "no" and s.metadata["error_class"] == 0 for s in control)
    assert {s.metadata["disclosure_level"] for s in ss} == {0, 3}


def test_ground_truth_never_in_input():
    for s in samples(levels=(0,)):
        assert s.metadata["rationale"] not in s.input
        assert "GroundTruth" not in s.input


def test_dataset_is_deterministic_in_seed():
    a = [s.input for s in samples(seed=1, levels=(0,))]
    b = [s.input for s in samples(seed=1, levels=(0,))]
    c = [s.input for s in samples(seed=2, levels=(0,))]
    assert a == b and a != c


@pytest.mark.parametrize("raw,expect", [(0, (0,)), ("0", (0,)), ("0,1,2,3", (0, 1, 2, 3)), ("0 2", (0, 2)), ([1, 3], (1, 3))])
def test_levels_parsing(raw, expect):
    assert _levels(raw) == expect


def test_unknown_arm_rejected():
    with pytest.raises(ValueError):
        opacity(arm="human")


def _state(completion: str) -> TaskState:
    st = TaskState(model="m", sample_id="x", epoch=0, input="bundle", messages=[])
    st.output = ModelOutput.from_content(model="m", content=completion)
    return st


@pytest.mark.parametrize(
    "completion,target,correct,flagged,cannot,parse_ok",
    [
        ('{"error_present":"yes","confidence":80,"description":"d","release":"block"}', "yes", 1, 1, 0, 1),
        ('{"error_present":"no","confidence":90,"description":"","release":"release"}', "yes", 0, 0, 0, 1),
        ('{"error_present":"cannot_tell","confidence":10,"description":"need n","release":"query"}', "no", 0, 0, 1, 1),
        ("I refuse to answer in JSON.", "no", 0, 0, 1, 0),
    ],
)
def test_scorer_values(completion, target, correct, flagged, cannot, parse_ok):
    sc = asyncio.run(verdict()(_state(completion), Target(target)))
    assert sc.value == {"correct": correct, "flagged": flagged, "cannot_tell": cannot, "parse_ok": parse_ok}


def test_mechanical_arm_runs_without_a_model(tmp_path):
    log = inspect_eval(
        opacity(arm="mechanical", levels="0,3"),
        log_dir=str(tmp_path), display="none", limit=6,
    )[0]
    assert log.status == "success"
    assert len(log.samples) == 6
    for s in log.samples:
        assert s.output.model == "mechanical"
        assert s.scores["verdict"].value["flagged"] == 0.0  # never flags, by design
        assert s.scores["verdict"].value["parse_ok"] == 1.0
    names = {m for sc in log.results.scores for m in sc.metrics}
    # one key per condition, per seeded class, per level: none may shadow another.
    # Registered metrics are namespaced ("opacity/by_class_1"), so match on suffix.
    for want in ("by_condition_seeded", "by_condition_control", "by_class_1", "by_level_0", "by_level_3"):
        assert any(n.endswith(want) for n in names), (want, sorted(names))
    assert "accuracy" in names


def test_harness_renders_bundles_identically_to_the_bespoke_loop():
    from opacity.items import build_corpus
    from opacity.sdc import policy

    by_id = {s.id: s for s in samples(seed=0, replicates=1, levels=(0, 2))}
    for item in build_corpus(seed=0, replicates=1):
        for b in item.pair:
            cond = "seeded" if b.truth.has_error else "control"
            for lvl in (0, 2):
                assert by_id[f"{b.item_id}-{cond}-s{lvl}"].input == b.controlled(policy(lvl)).render()


@pytest.mark.parametrize("raw,expect", [("off", False), (False, False), ("on", True), (True, True)])
def test_thinking_switch_reaches_generate_config(raw, expect):
    t = opacity(arm="llm", thinking=raw)
    assert t.config.extra_body == {"chat_template_kwargs": {"enable_thinking": expect}}
    assert t.metadata["thinking"] == ("on" if expect else "off")


def test_thinking_default_sends_nothing_and_rejects_junk():
    assert opacity(arm="llm").config.extra_body is None
    with pytest.raises(ValueError):
        opacity(arm="llm", thinking="sometimes")
