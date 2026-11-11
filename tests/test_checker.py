"""The OpenAI-compatible checker must cope with gateway envelopes.

Cline's gateway wraps a non-streaming completion as {"data": {...},
"success": true}; the OpenAI SDK turns that into a ChatCompletion with
choices=None. vLLM and MiMo return a bare completion. Both must parse.
"""

import json
from types import SimpleNamespace

import pytest

from opacity.checker import OpenAICompatibleChecker, unwrap_envelope

BARE = {
    "id": "x", "object": "chat.completion", "created": 1, "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "OK"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
WRAPPED = {"data": BARE, "success": True}


def test_unwrap_envelope_strips_gateway_wrapper():
    assert unwrap_envelope(WRAPPED) is BARE


def test_unwrap_envelope_leaves_bare_completion_alone():
    assert unwrap_envelope(BARE) is BARE


def test_unwrap_envelope_ignores_lookalikes():
    # extra top-level keys mean it is not the known envelope
    other = {"data": BARE, "success": True, "object": "list"}
    assert unwrap_envelope(other) is other
    assert unwrap_envelope({"data": [1, 2]}) == {"data": [1, 2]}


def _checker_with_body(body: dict) -> OpenAICompatibleChecker:
    pytest.importorskip("openai")
    c = OpenAICompatibleChecker(base_url="http://localhost:1/v1", model="m")
    http = SimpleNamespace(http_response=SimpleNamespace(json=lambda: json.loads(json.dumps(body))))
    c._client.chat.completions.with_raw_response.create = lambda **kw: http  # type: ignore[attr-defined]
    return c


@pytest.mark.parametrize("body", [BARE, WRAPPED], ids=["bare", "enveloped"])
def test_create_returns_choices_for_both_shapes(body):
    c = _checker_with_body(body)
    resp = c._create(model="m", messages=[])
    assert resp.choices[0].message.content == "OK"
    assert resp.usage.total_tokens == 2
