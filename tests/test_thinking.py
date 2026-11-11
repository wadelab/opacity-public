"""Thinking switches and the exploratory prompt variants: what reaches the
request, what stays pre-registered, and what the tasks record."""

import pytest

inspect_ai = pytest.importorskip("inspect_ai")

from opacity.checker.prompts import REVIEWER_INSTRUCTION  # noqa: E402
from opacity.crosscheck.task import crosscheck  # noqa: E402
from opacity.harness.task import opacity  # noqa: E402
from opacity.harvest.prompts import AGENT_INSTRUCTION, AGENT_INSTRUCTION_CLOSED  # noqa: E402
from opacity.harvest.task import harvest, samples  # noqa: E402
from opacity.thinking import (  # noqa: E402
    ANSWER_TOKENS, generation_extra, max_tokens_for, normalise_effort, normalise_thinking,
)
from opacity.variants import ANALYST_CHECKLIST, CHECKS, analyst_text, reviewer_text  # noqa: E402


def test_extra_body_matches_the_old_switch():
    assert generation_extra("default") is None
    assert generation_extra("off") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert generation_extra(True) == {"chat_template_kwargs": {"enable_thinking": True}}
    assert generation_extra(False) == {"chat_template_kwargs": {"enable_thinking": False}}


def test_budget_turns_thinking_on_and_grows_the_output_cap():
    assert generation_extra("default", 1024) == {
        "chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 1024}
    assert generation_extra("on", "512")["thinking_token_budget"] == 512
    assert max_tokens_for(0) == ANSWER_TOKENS and max_tokens_for(2048) == ANSWER_TOKENS + 2048
    with pytest.raises(ValueError):
        generation_extra("off", 1024)
    with pytest.raises(ValueError):
        generation_extra("on", -1)


def test_bad_switches_are_refused():
    with pytest.raises(ValueError):
        normalise_thinking("maybe")
    with pytest.raises(ValueError):
        normalise_effort("extreme")
    assert normalise_effort("") is None and normalise_effort("LOW") == "low"


def test_variants_append_and_never_edit_the_preregistered_text():
    assert reviewer_text() == REVIEWER_INSTRUCTION
    for v in ("checklist", "stepwise"):
        t = reviewer_text(v)
        assert t.startswith(REVIEWER_INSTRUCTION) and CHECKS in t
    assert "last thing in your reply" in reviewer_text("stepwise")
    with pytest.raises(ValueError):
        reviewer_text("socratic")
    assert analyst_text(AGENT_INSTRUCTION) == AGENT_INSTRUCTION
    assert analyst_text(AGENT_INSTRUCTION_CLOSED, True) == AGENT_INSTRUCTION_CLOSED + "\n" + ANALYST_CHECKLIST


def test_harvest_checklist_reaches_the_input_and_the_metadata():
    plain = samples(only=("h01",))[0]
    ticked = samples(only=("h01",), checklist=True)[0]
    assert ANALYST_CHECKLIST not in plain.input and ANALYST_CHECKLIST in ticked.input
    assert ticked.metadata["checklist"] is True
    t = harvest(only="h01", checklist=True, think_budget=1024)
    assert t.config.max_tokens == ANSWER_TOKENS + 1024
    assert t.config.extra_body["thinking_token_budget"] == 1024
    assert t.metadata["checklist"] is True and t.metadata["think_budget"] == 1024


def test_tasks_record_budget_effort_and_reviewer():
    t = opacity(levels="0", think_budget=2048, reviewer="stepwise")
    assert t.config.max_tokens == ANSWER_TOKENS + 2048
    assert t.metadata["think_budget"] == 2048 and t.metadata["reviewer"] == "stepwise"
    t = opacity(levels="0", effort="low")
    assert t.config.reasoning_effort == "low" and t.config.extra_body is None
    t = opacity(levels="0", thinking=False)
    assert t.config.max_tokens == ANSWER_TOKENS
    assert t.config.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    with pytest.raises(ValueError):
        harvest(only="h01", reviewer="nope")


def test_crosscheck_records_the_variant(tmp_path):
    with pytest.raises(ValueError):
        crosscheck(logs="", reviewer="checklist")


def test_effort_gets_an_output_allowance():
    from opacity.thinking import EFFORT_TOKENS
    assert max_tokens_for(0, "medium") == ANSWER_TOKENS + EFFORT_TOKENS["medium"]
    assert harvest(only="h01", effort="low").config.max_tokens == ANSWER_TOKENS + EFFORT_TOKENS["low"]


def test_rules_prompt_and_revise_arm_are_recorded():
    from opacity.variants import ANALYST_RULES
    t = harvest(only="h01", analyst_prompt="rules", arm="revise")
    s = t.dataset[0]
    assert ANALYST_RULES in s.input and s.metadata["analyst_prompt"] == "rules" and s.metadata["checklist"] is False
    assert t.metadata["analyst_prompt"] == "rules" and t.metadata["reviewer"] == "stepwise" and t.name == "harvest-revise"
    with pytest.raises(ValueError):
        harvest(only="h01", analyst_prompt="poetry")


def test_revise_arm_releases_the_rewrite_when_the_review_objects(tmp_path):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model

    first = '{"title": "T", "methods": "m", "interpretation": "ICB H stands out."}'
    review = 'checks...\n{"error_present": "yes", "confidence": 90, "description": "20 tests, none survives", "release": "block"}'
    rewrite = '{"title": "T", "methods": "m", "interpretation": "No ICB differs after correction."}'
    replies = iter([first, review, rewrite])
    model = get_model("mockllm/model", custom_outputs=(ModelOutput.from_content("mockllm/model", r) for r in replies))

    class _Judge:
        async def generate(self, messages, config=None):
            return ModelOutput.from_content("stub/judge", '{"conclusion_wrong": false, "classes_committed": []}')

    log = inspect_eval(harvest(judge=_Judge(), only="h01", arm="revise"), model=model, log_dir=str(tmp_path), display="none")[0]
    s = log.samples[0]
    assert s.output.completion == rewrite and s.metadata["revised"] is True and s.metadata["review_verdict"] == "yes"
    assert s.metadata["agent_raw"] == first


def test_sampling_presets_and_open_thinking_allowance():
    from opacity.thinking import OPEN_THINK_TOKENS, sampling_config
    assert sampling_config("greedy") == ({"temperature": 0.0}, {})
    assert sampling_config("qwen-thinking") == ({"temperature": 0.6, "top_p": 0.95}, {"top_k": 20})
    assert sampling_config("nemotron") == ({"temperature": 1.0, "top_p": 0.95}, {})
    with pytest.raises(ValueError):
        sampling_config("hot")
    assert max_tokens_for(0, None, "on") == ANSWER_TOKENS + OPEN_THINK_TOKENS
    assert max_tokens_for(1024, None, "on") == ANSWER_TOKENS + 1024  # a budget caps it instead
    t = harvest(only="h01", thinking="on", sampling="qwen-thinking")
    c = t.config
    assert c.temperature == 0.6 and c.top_p == 0.95 and c.max_tokens == ANSWER_TOKENS + OPEN_THINK_TOKENS
    assert c.extra_body == {"chat_template_kwargs": {"enable_thinking": True}, "top_k": 20}
    assert t.metadata["sampling"] == "qwen-thinking"
    g = harvest(only="h01", effort="high", sampling="gpt-oss").config
    assert g.temperature == 1.0 and g.extra_body is None and g.reasoning_effort == "high"
    assert harvest(only="h01").config.temperature == 0.0  # the default is unchanged
