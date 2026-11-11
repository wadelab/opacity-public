"""Checker protocol and the model-agnostic LLM checker.

Any model behind an OpenAI-compatible chat endpoint works: vLLM on Viking2,
a hosted API, or a local server. Which model is the checker is a config line,
not a code change, because the s0 ceiling check in phase 0 is the instrument
that chooses it.

Every call is logged in full. Raw responses are the reproducibility record;
nothing is regenerated.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Literal, Protocol

from .prompts import REVIEWER_INSTRUCTION, SYSTEM_PREAMBLE

ErrorPresent = Literal["yes", "no", "cannot_tell"]
Release = Literal["release", "query", "block"]


@dataclass
class Verdict:
    error_present: ErrorPresent
    confidence: int
    description: str
    release: Release
    raw: str = ""
    """Verbatim model output. Stored, never discarded."""
    model: str = ""
    seconds: float = 0.0
    parse_ok: bool = True
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Checker(Protocol):
    name: str

    def review(self, rendered_bundle: str) -> Verdict: ...


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(raw: str, model: str = "", seconds: float = 0.0) -> Verdict:
    """Parse a model reply into a Verdict. Never raises.

    A reply that cannot be parsed is recorded as `cannot_tell` with
    `parse_ok=False`; it is kept in the data and reported, not silently
    dropped, because parse failures are themselves a checker property.
    """
    m = _JSON_BLOCK.search(raw or "")
    if not m:
        return Verdict(
            "cannot_tell", 0, "unparseable response", "query",
            raw=raw, model=model, seconds=seconds, parse_ok=False,
        )
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Verdict(
            "cannot_tell", 0, "unparseable response", "query",
            raw=raw, model=model, seconds=seconds, parse_ok=False,
        )

    ep = str(obj.get("error_present", "")).strip().lower()
    if ep not in ("yes", "no", "cannot_tell"):
        ep = "cannot_tell"
    rel = str(obj.get("release", "")).strip().lower()
    if rel not in ("release", "query", "block"):
        rel = "query"
    try:
        conf = int(round(float(obj.get("confidence", 0))))
    except (TypeError, ValueError):
        conf = 0
    conf = max(0, min(100, conf))

    return Verdict(
        error_present=ep,  # type: ignore[arg-type]
        confidence=conf,
        description=str(obj.get("description", "")).strip(),
        release=rel,  # type: ignore[arg-type]
        raw=raw,
        model=model,
        seconds=seconds,
        parse_ok=True,
    )


def unwrap_envelope(payload: dict) -> dict:
    """Return the chat completion inside a gateway envelope, or ``payload`` as is.

    An envelope is a top-level object whose keys are a subset of
    ``{"data", "success"}`` with a dict under ``data``. A bare OpenAI
    completion always has ``object``/``choices`` at the top, so it never
    matches.
    """
    if (
        isinstance(payload, dict)
        and isinstance(payload.get("data"), dict)
        and set(payload) <= {"data", "success"}
    ):
        return payload["data"]
    return payload


class OpenAICompatibleChecker:
    """LLM checker over any OpenAI-compatible /v1/chat/completions endpoint.

    Example (vLLM on Viking2):
        OpenAICompatibleChecker(
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen2.5-32B-Instruct",
            api_key="EMPTY",
        )
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 600,
        seed: int | None = 0,
        timeout: float = 120.0,
        extra_body: dict | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # lazy: optional dependency
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai") from e
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.name = f"llm:{model}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.extra_body = extra_body or {}

    def _create(self, **kwargs):
        """chat.completions.create, tolerant of gateway envelopes.

        Some gateways (Cline's api.cline.bot, for one) wrap a non-streaming
        completion as ``{"data": {...}, "success": true}``. The OpenAI SDK
        parses that into a ChatCompletion with ``choices=None``, so we read
        the raw body, unwrap if needed, and validate it ourselves.
        """
        from openai.types.chat import ChatCompletion

        http = self._client.chat.completions.with_raw_response.create(**kwargs)
        return ChatCompletion.model_validate(unwrap_envelope(http.http_response.json()))

    def review(self, rendered_bundle: str) -> Verdict:
        t0 = time.perf_counter()
        kwargs = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PREAMBLE},
                {
                    "role": "user",
                    "content": REVIEWER_INSTRUCTION + "\n\n---\n\n" + rendered_bundle,
                },
            ],
        )
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        resp = self._create(**kwargs)
        raw = resp.choices[0].message.content or ""
        v = parse_verdict(raw, model=self.model, seconds=time.perf_counter() - t0)
        v.meta = {
            "finish_reason": resp.choices[0].finish_reason,
            "usage": resp.usage.model_dump() if resp.usage else None,
            "system_fingerprint": getattr(resp, "system_fingerprint", None),
            "response_model": resp.model,
        }
        return v
