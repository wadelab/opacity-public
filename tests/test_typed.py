"""The typed arm: a decision server's answers become the same verdict JSON."""

import json

import httpx
import pytest
from inspect_ai import eval as inspect_eval

from opacity.checker.prompts import REVIEWER_INSTRUCTION, TYPED_QUESTIONS
from opacity.harness import opacity
from opacity.harness.typed import build_request, verdict_from_answers


CANNED = {
    "model": "openjev-latest",
    "answers": {
        "error_present": {"choice": "yes", "probabilities": {"yes": 0.8, "no": 0.15, "cannot_tell": 0.05}, "confidence": 0.61},
        "release": {"choice": "query", "probabilities": {"release": 0.2, "query": 0.7, "block": 0.1}, "confidence": 0.4},
    },
    "usage": {"input_tokens": 900, "output_tokens": 0},
}


def test_questions_restate_the_instruction():
    for key in ("yes", "no", "cannot_tell"):
        assert key in TYPED_QUESTIONS["error_present"]["criteria"]
    assert "does not support" in TYPED_QUESTIONS["error_present"]["criteria"]["yes"]
    assert "does not support" in REVIEWER_INSTRUCTION
    assert tuple(TYPED_QUESTIONS["release"]["criteria"]) == ("release", "query", "block")


def test_verdict_from_answers_uses_choice_and_entropy_confidence():
    v = verdict_from_answers(CANNED)
    assert v["error_present"] == "yes" and v["release"] == "query"
    assert v["confidence"] == 61 and v["description"] == ""
    assert v["p_error_present"]["yes"] == 0.8


def test_verdict_falls_back_to_argmax_and_max_probability():
    data = {"answers": {
        "error_present": {"probabilities": {"yes": 0.1, "no": 0.3, "cannot_tell": 0.6}},
        "release": {"probabilities": {"release": 0.5, "query": 0.4, "block": 0.1}},
    }}
    v = verdict_from_answers(data)
    assert v["error_present"] == "cannot_tell" and v["release"] == "release" and v["confidence"] == 60


def test_build_request_carries_only_what_is_set():
    b = build_request("state", "m")
    assert set(b) == {"model", "state", "questions"}
    b = build_request("state", "m", samples=4, think=256)
    assert b["samples"] == 4 and b["think"] == 256


class _FakeClient:
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def post(self, url, json=None):
        assert url.endswith("/v1/systemone")
        assert json["state"].startswith(REVIEWER_INSTRUCTION)
        return httpx.Response(200, json=CANNED, request=httpx.Request("POST", url))


def test_typed_arm_runs_against_a_fake_server(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    log = inspect_eval(
        opacity(arm="typed", levels="0,4", typed_url="http://fake:8080"),
        log_dir=str(tmp_path), display="none", limit=4,
    )[0]
    assert log.status == "success" and len(log.samples) == 4
    for s in log.samples:
        assert s.output.model == "typed:openjev-latest"
        v = s.scores["verdict"].value
        assert v["parse_ok"] == 1.0 and v["flagged"] == 1.0
        assert s.scores["verdict"].metadata["confidence"] == 61
        assert s.output.metadata["response"]["answers"]["error_present"]["choice"] == "yes"
        assert json.loads(s.output.completion)["release"] == "query"
    assert log.eval.metadata["typed"]["url"] == "http://fake:8080"


class _DownClient(_FakeClient):
    async def post(self, url, json=None):
        raise httpx.ConnectError("connection refused")


def test_server_failure_is_a_parse_failure_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _DownClient)
    log = inspect_eval(
        opacity(arm="typed", levels="0", typed_url="http://fake:8080"),
        log_dir=str(tmp_path), display="none", limit=2,
    )[0]
    assert log.status == "success"
    for s in log.samples:
        v = s.scores["verdict"].value
        assert v["parse_ok"] == 0.0 and v["cannot_tell"] == 1.0
        assert "ConnectError" in s.output.metadata["error"]


def test_typed_arm_needs_a_url():
    with pytest.raises(ValueError):
        opacity(arm="typed")
