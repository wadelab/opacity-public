"""The replay model provider: recorded completions come back for their exact
request and seed, and anything else is an error rather than a guess."""

from __future__ import annotations

import asyncio
import json

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig

from opacity.harness.providers import ReplayAPI


def _files(tmp_path, requests, outputs):
    rq, out = tmp_path / "requests.jsonl", tmp_path / "outputs.jsonl"
    rq.write_text("".join(json.dumps(r) + "\n" for r in requests))
    out.write_text("".join(json.dumps(o) + "\n" for o in outputs))
    return str(out), str(rq)


def _req(id_, user, seed):
    return {"id": id_, "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": user}],
            "seed": seed, "temperature": 0.7, "top_p": 0.8, "top_k": 20}


def test_replay_matches_messages_and_seed(tmp_path):
    out, rq = _files(tmp_path, [_req("a-r0", "task a", 0), _req("a-r1", "task a", 1), _req("b-r0", "task b", 0)],
                     [{"id": "a-r0", "completion": "A0"}, {"id": "a-r1", "completion": "A1", "finish_reason": "length"},
                      {"id": "b-r0", "completion": "B0"}])
    api = ReplayAPI("replay/test", outputs=out, requests=rq)
    msgs = [ChatMessageSystem(content="sys"), ChatMessageUser(content="task a")]
    first = asyncio.run(api.generate(msgs, [], "none", GenerateConfig(seed=0)))
    second = asyncio.run(api.generate(msgs, [], "none", GenerateConfig(seed=1)))
    assert first.completion == "A0" and second.completion == "A1"
    assert second.stop_reason == "max_tokens"
    with pytest.raises(RuntimeError):
        asyncio.run(api.generate(msgs, [], "none", GenerateConfig(seed=2)))
    with pytest.raises(RuntimeError):
        asyncio.run(api.generate([ChatMessageSystem(content="sys"), ChatMessageUser(content="task c")], [], "none",
                                 GenerateConfig(seed=0)))


def test_replay_refuses_identical_requests(tmp_path):
    out, rq = _files(tmp_path, [_req("a-r0", "task a", 0), _req("a-r0-copy", "task a", 0)],
                     [{"id": "a-r0", "completion": "A0"}])
    with pytest.raises(ValueError):
        ReplayAPI("replay/test", outputs=out, requests=rq)
    with pytest.raises(ValueError):
        ReplayAPI("replay/test", outputs="", requests=rq)
