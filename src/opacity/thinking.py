"""Generation switches shared by the three tasks: thinking, its budget, and
reasoning effort.

    thinking      "off" sends chat_template_kwargs enable_thinking=false (vLLM:
                  Qwen3, Qwen3.5, Gemma 4), "on" sends true, "default" sends
                  nothing. Inspect YAML-parses `-T thinking=off` into the
                  boolean False, so booleans are accepted.
    think_budget  a cap on thinking tokens, sent as vLLM's
                  `thinking_token_budget`; the server forces the end-of-thinking
                  marker when the cap is reached, so the model always gets to
                  its answer. Needs the server started with --reasoning-parser
                  (the Slurm script's REASONING_PARSER). Implies thinking on.
    effort        `reasoning_effort` for models that take it instead of a
                  switch (gpt-oss: low, medium, high).

    sampling      a named sampling preset (SAMPLING). "greedy" (temperature 0)
                  is what every run before 23 October used; the others are the
                  vendors' recommended settings, for testing whether greedy
                  decoding handicaps the reasoning-tuned models.

A thinking run must not lose its answer to the output cap: the answer budget
stays at ANSWER_TOKENS and the thinking budget is added on top of it. An
effort level has no hard cap, so it gets the allowance in EFFORT_TOKENS, and
thinking on with no budget gets OPEN_THINK_TOKENS.
"""

from __future__ import annotations

from typing import Any

THINKING = ("default", "off", "on")
EFFORTS = ("", "low", "medium", "high")
ANSWER_TOKENS = 4000
EFFORT_TOKENS = {"low": 4096, "medium": 8192, "high": 16384}
OPEN_THINK_TOKENS = 16384
SAMPLING = {
    "greedy": {"temperature": 0.0},
    # Qwen3 / Qwen3.5 model cards, thinking mode (also the 122B's generation_config.json)
    "qwen-thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    # the same cards, thinking off
    "qwen-instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
    # OpenAI's recommendation for gpt-oss
    "gpt-oss": {"temperature": 1.0, "top_p": 1.0},
    # NVIDIA's Nemotron 3 Super card: "across all tasks and serving backends"
    "nemotron": {"temperature": 1.0, "top_p": 0.95},
}


def normalise_thinking(thinking: Any) -> str:
    if isinstance(thinking, bool):
        thinking = "on" if thinking else "off"
    if thinking not in THINKING:
        raise ValueError(f"thinking must be one of {THINKING}; got {thinking!r}")
    return thinking


def normalise_budget(think_budget: Any) -> int:
    try:
        b = int(think_budget or 0)
    except (TypeError, ValueError):
        raise ValueError(f"think_budget must be a non-negative integer; got {think_budget!r}") from None
    if b < 0:
        raise ValueError(f"think_budget must be a non-negative integer; got {think_budget!r}")
    return b


def generation_extra(thinking: Any = "default", think_budget: Any = 0) -> dict | None:
    """The `extra_body` for a request: None when nothing is to be sent."""
    thinking = normalise_thinking(thinking)
    budget = normalise_budget(think_budget)
    if budget:
        if thinking == "off":
            raise ValueError("think_budget needs thinking on (or default), not off")
        return {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": budget}
    if thinking == "default":
        return None
    return {"chat_template_kwargs": {"enable_thinking": thinking == "on"}}


def max_tokens_for(think_budget: Any = 0, effort: Any = None, thinking: Any = "default") -> int:
    e = normalise_effort(effort)
    b = normalise_budget(think_budget)
    open_think = normalise_thinking(thinking) == "on" and not b and not e
    return ANSWER_TOKENS + b + (EFFORT_TOKENS[e] if e else 0) + (OPEN_THINK_TOKENS if open_think else 0)


def sampling_config(sampling: Any = "greedy") -> tuple[dict, dict]:
    """(GenerateConfig fields, extra_body fields) for a preset. top_k is not an
    OpenAI parameter, so it goes in the request body, which vLLM reads."""
    name = str(sampling or "greedy").strip().lower()
    if name not in SAMPLING:
        raise ValueError(f"sampling must be one of {tuple(SAMPLING)}; got {sampling!r}")
    p = dict(SAMPLING[name])
    extra = {"top_k": p.pop("top_k")} if "top_k" in p else {}
    return p, extra


def normalise_effort(effort: Any) -> str | None:
    e = "" if effort in (None, False) else str(effort).strip().lower()
    if e not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS[1:]} or empty; got {effort!r}")
    return e or None
