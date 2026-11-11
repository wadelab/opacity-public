"""Model providers that Inspect does not ship.

Cline's gateway (api.cline.bot) speaks the OpenAI chat protocol but wraps a
non-streaming completion as ``{"data": {...}, "success": true}``. The OpenAI
SDK accepts that and returns a ChatCompletion with ``choices=None``; Inspect's
stock ``openai-api`` provider then has nothing to score. This provider is the
stock one with the envelope removed.

    inspect eval ... --model cline/cline-pass/deepseek-v4.1-flash

Reads CLINE_API_KEY and CLINE_BASE_URL, like any ``openai-api`` service.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.model import GenerateConfig, ModelAPI, ModelOutput, modelapi
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from openai.types.chat import ChatCompletion

from ..checker.base import unwrap_envelope


class ClineAPI(OpenAICompatibleAPI):
    def __init__(self, model_name: str, base_url: str | None = None,
                 api_key: str | None = None, config: GenerateConfig = GenerateConfig(),
                 **model_args: Any) -> None:
        super().__init__(
            model_name=model_name, base_url=base_url, api_key=api_key, config=config,
            service="cline", **model_args,
        )

    async def _generate_completion(self, request: dict[str, Any], config: GenerateConfig) -> ChatCompletion:
        if self.resolve_stream(config):
            # streamed chunks come through unwrapped; nothing to do
            return await super()._generate_completion(request, config)
        raw = await self.client.chat.completions.with_raw_response.create(**request)
        return ChatCompletion.model_validate(unwrap_envelope(raw.http_response.json()))


@modelapi(name="cline")
def cline() -> type[ModelAPI]:
    return ClineAPI


class ReplayAPI(ModelAPI):
    """Serves completions written elsewhere, so they can be scored like any run.

    The deception study's runner samples write-ups inside its isolated runtime
    and stores them as JSONL (``{"id", "candidate", "completion", ...}``, the
    id being the harvest sample id). Running the harvest task with this model
    turns them into an ordinary Inspect log for the local judge:

        inspect eval opacity/harvest --model replay/D-u00050 \\
            -M outputs=<candidate>.jsonl -M requests=<requests>.jsonl ...

    A request is matched on its exact messages and seed (the replicate), read
    from the requests file the runner sampled; an unmatched or ambiguous call
    is an error, never a guess.
    """

    def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None,
                 config: GenerateConfig = GenerateConfig(), outputs: str = "", requests: str = "", **_: Any) -> None:
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config)
        import hashlib
        import json
        if not outputs or not requests:
            raise ValueError("replay needs -M outputs=<jsonl> and -M requests=<jsonl>")
        done = {}
        for line in open(outputs):
            row = json.loads(line)
            done[row["id"]] = row
        self._hash = lambda messages, seed: hashlib.sha256(json.dumps([messages, seed]).encode()).hexdigest()
        self._by_key: dict[str, dict] = {}
        for line in open(requests):
            req = json.loads(line)
            key = self._hash([[m["role"], m["content"]] for m in req["messages"]], req["seed"])
            if key in self._by_key:
                raise ValueError(f"replay: requests {self._by_key[key]['id']} and {req['id']} are identical")
            if req["id"] in done:
                self._by_key[key] = done[req["id"]]

    async def generate(self, input: list, tools: list, tool_choice: Any, config: GenerateConfig) -> ModelOutput:
        messages = [[m.role, m.text] for m in input]
        row = self._by_key.get(self._hash(messages, config.seed))
        if row is None:
            raise RuntimeError("replay: no recorded completion for this request and seed")
        stop = "max_tokens" if row.get("finish_reason") == "length" else "stop"
        return ModelOutput.from_content(model=self.model_name, content=row["completion"], stop_reason=stop)


@modelapi(name="replay")
def replay() -> type[ModelAPI]:
    return ReplayAPI
