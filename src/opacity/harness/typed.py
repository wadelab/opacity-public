"""The typed arm: a decision server instead of a generative checker.

openjev (github.com/razorback16/openjev) answers typed questions about a text
by reading the model's probability over the answer tokens in a masked slot,
with confidence taken from the entropy of that distribution. Nothing is
generated and nothing is parsed, so this arm has no parse failures and its
confidence is not self-reported. It also returns no free text, so it cannot
supply the description the blind coder uses; it runs beside the generative
arm, not in place of it.

The state it reads is exactly the user turn the generative arm receives: the
reviewer instruction, a rule, and the rendered bundle. The two questions
restate the instruction's definitions of `error_present` and `release` as
choice criteria, verbatim where the instruction has words for them. Both are
pre-registration material and live in `checker/prompts.py`.

The verdict is written in the same JSON shape as the other arms, so one
scorer serves all of them: `confidence` is the server's entropy confidence on
the error question, scaled to 0 to 100; `description` is empty.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver

from opacity.checker.prompts import REVIEWER_INSTRUCTION, TYPED_QUESTIONS

ERROR_PRESENT = ("yes", "no", "cannot_tell")
RELEASE = ("release", "query", "block")


def _choice(answer: dict[str, Any], allowed: tuple[str, ...]) -> tuple[str, float | None, dict[str, float]]:
    """Pick the option and its confidence from one choice answer.

    openjev returns `choice`, `probabilities` and `confidence` for a choice
    question. If `choice` is missing the argmax of `probabilities` is used.
    Returns (option, confidence or None, probabilities).
    """
    probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
    choice = answer.get("choice")
    if choice not in allowed and probs:
        choice = max(probs, key=probs.get)
    if choice not in allowed:
        raise ValueError(f"no usable choice in {answer!r}")
    conf = answer.get("confidence")
    return str(choice), (None if conf is None else float(conf)), probs


def verdict_from_answers(data: dict[str, Any]) -> dict[str, Any]:
    """Map a /v1/systemone response to the verdict JSON the scorer reads."""
    answers = data.get("answers") or {}
    ep, conf, ep_probs = _choice(answers["error_present"], ERROR_PRESENT)
    rel, _, rel_probs = _choice(answers["release"], RELEASE)
    if conf is None:
        conf = max(ep_probs.values()) if ep_probs else 0.0
    return {
        "error_present": ep,
        "confidence": int(round(100 * conf)),
        "description": "",
        "release": rel,
        "p_error_present": ep_probs,
        "p_release": rel_probs,
    }


def build_request(state_text: str, model: str, samples: int = 1, think: int = 0) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "state": state_text, "questions": TYPED_QUESTIONS}
    if samples > 1:
        body["samples"] = samples
    if think > 0:
        body["think"] = think
    return body


@solver
def typed_reviewer(
    url: str,
    model: str = "openjev-latest",
    samples: int = 1,
    think: int = 0,
    timeout: float = 180.0,
) -> Solver:
    """The typed arm. `url` is the openjev server, e.g. http://localhost:8080.

    samples: re-reads averaged by the server (1 to 32). think: capped thought
    tokens before the read (0 disables). Both are recorded in the log.
    """
    if not url:
        raise ValueError("the typed arm needs typed_url (the openjev server)")
    name = f"typed:{model}"

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state_text = REVIEWER_INSTRUCTION + "\n\n---\n\n" + state.input_text
        body = build_request(state_text, model, samples, think)
        meta: dict[str, Any] = {"request": {k: v for k, v in body.items() if k != "state"}}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url.rstrip("/") + "/v1/systemone", json=body)
                r.raise_for_status()
                data = r.json()
            meta["response"] = data
            content = json.dumps(verdict_from_answers(data))
        except Exception as exc:  # recorded as a parse failure, never dropped
            meta["error"] = f"{type(exc).__name__}: {exc}"
            content = f"typed arm error: {type(exc).__name__}: {exc}"
        state.messages = [
            ChatMessageUser(content=state_text),
            ChatMessageAssistant(content=content, model=name),
        ]
        state.output = ModelOutput.from_content(model=name, content=content)
        state.output.metadata = meta
        return state

    return solve
