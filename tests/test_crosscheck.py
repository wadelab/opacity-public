"""The cross-check: harvested write-ups become checker samples, the judge's
verdict rides along as the target, declines are left out, and the pipeline
closes with a mock checker."""

import json

import pytest

inspect_ai = pytest.importorskip("inspect_ai")

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import ModelOutput, get_model  # noqa: E402
from itertools import count  # noqa: E402

from opacity.crosscheck.task import crosscheck, harvested_samples, short_model  # noqa: E402
from opacity.harvest.task import harvest  # noqa: E402


def _mock(reply: str):
    gen = (ModelOutput.from_content("mockllm/model", reply) for _ in count())
    return get_model("mockllm/model", custom_outputs=gen)


BUNDLE = '{"verdict": "answer", "reason": "", "title": "T", "methods": "m", "stages": [{"name": "a", "n": 10}], "tables": [], "interpretation": "fine"}'
JUDGE = '{"conclusion_wrong": true, "classes_committed": [3], "description": "d"}'


class _StubJudge:
    """A plain object, not an Inspect Model: the task registry walks a
    Model argument's parameters, and a mock's unbounded output generator
    never ends. `judged` accepts anything with an async generate."""

    def __init__(self, reply: str):
        self.reply = reply

    async def generate(self, messages, config=None):
        return ModelOutput.from_content("stub/judge", self.reply)


def _harvest_log(tmp_path, reply=BUNDLE, judge_reply=JUDGE, arm="closed", only="h09,h33"):
    return inspect_eval(harvest(judge=_StubJudge(judge_reply), only=only, arm=arm), model=_mock(reply),
                        log_dir=str(tmp_path), display="none")[0]


def test_short_model():
    assert short_model("openai-api/vllm/Qwen/Qwen3.5-9B") == "Qwen3.5-9B"
    assert short_model("vllm/Qwen/Qwen3.5-9B") == "Qwen3.5-9B"
    assert short_model("mockllm/model") == "model"


def test_harvested_samples_carry_judge_verdict_and_skip_declines(tmp_path):
    log = _harvest_log(tmp_path)
    ss = harvested_samples([log.location])
    assert len(ss) == 2
    for s in ss:
        assert s.target == "yes" and s.metadata["judge_known"] and s.metadata["judge_classes"] == [3]
        assert s.metadata["agent_short"] == "model" and s.metadata["harvest_arm"] == "closed"
        assert "## Analysis question" in s.input and "## Interpretation" in s.input
        assert "judge" not in s.input.lower()
    declined = _harvest_log(tmp_path / "d", reply='{"verdict": "decline", "reason": "no", "title": "T"}')
    assert harvested_samples([declined.location]) == []
    assert len(harvested_samples([declined.location], include_declined=True)) == 2
    unscored = _harvest_log(tmp_path / "u", judge_reply="I decline to judge.")  # judge reply unparseable: no target
    ss = harvested_samples([unscored.location])
    assert len(ss) == 2 and all(s.target == "" and not s.metadata["judge_known"] for s in ss)


def test_crosscheck_needs_logs():
    with pytest.raises(ValueError):
        crosscheck()


def test_crosscheck_logs_as_list(tmp_path):
    log = _harvest_log(tmp_path)
    t = crosscheck(logs=[log.location, log.location])
    assert len(t.dataset) == 4 and t.metadata["logs"] == [log.location, log.location]


def test_crosscheck_pipeline_closes(tmp_path):
    log = _harvest_log(tmp_path / "h")
    checker = _mock('{"error_present": "yes", "confidence": 80, "description": "confounded", "release": "query"}')
    out = inspect_eval(crosscheck(logs=str(tmp_path / "h" / "*.eval")), model=checker, log_dir=str(tmp_path / "x"), display="none")[0]
    assert out.status == "success" and len(out.samples) == 2
    assert out.eval.metadata["agents"] == ["model"]
    for s in out.samples:
        v = s.scores["crosscheck_verdict"].value
        assert v == {"flagged": 1.0, "agree": 1.0, "cannot_tell": 0.0, "parse_ok": 1.0, "judge_known": 1.0}
        assert s.messages[1].text.endswith(s.input)  # the reviewer instruction, then the write-up, nothing else
    names = {m for sc in out.results.scores for m in sc.metrics}
    for want in ("by_judge_verdict_wrong", "by_agent_model"):
        assert any(n.endswith(want) for n in names), (want, sorted(names))


def test_two_logs_from_one_agent_give_distinct_ids_and_the_pack_can_be_shown(tmp_path):
    from opacity.crosscheck.task import PACK_HEADING, WRITEUP_HEADING
    a = _harvest_log(tmp_path / "a")
    b = _harvest_log(tmp_path / "b", arm="open")
    ss = harvested_samples([a.location, b.location])
    assert len({s.id for s in ss}) == len(ss) == 4
    shown = harvested_samples([a.location], show_pack=True)[0].input
    assert shown.startswith(PACK_HEADING) and WRITEUP_HEADING in shown and "## Data pack" in shown
    assert "judge" not in shown.lower() and "Correct conclusion" not in shown


def test_crosscheck_typed_arm_needs_a_server(tmp_path):
    log = _harvest_log(tmp_path)
    with pytest.raises(ValueError):
        crosscheck(logs=log.location, arm="typed")
    t = crosscheck(logs=log.location, arm="typed", typed_url="http://127.0.0.1:9")
    assert t.metadata["arm"] == "typed" and t.metadata["typed"]["url"] == "http://127.0.0.1:9"
