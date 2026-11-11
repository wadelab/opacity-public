"""The study as an Inspect task.

One task, two machine arms. The dataset is every (seeded, control) bundle
rendered at each requested disclosure level; the solver is either the
pre-registered reviewer prompt followed by one model call, or the mechanical
SDC checker; the scorer parses the JSON verdict and compares it with the
ground truth. Inspect's eval log is the data record: every prompt, every
verbatim reply, token usage and timing, per sample.

    inspect eval opacity/opacity -T arm=mechanical -T levels=0,1,2,3
    inspect eval opacity/opacity -T levels=0 --model openai-api/mimo/mimo-v2.6-pro
    inspect eval opacity/opacity -T levels=0 --model cline/cline-pass/deepseek-v4.1-flash

The prompt text and the SDC policies are pre-registration material and are
imported, not restated, so this file cannot drift from the bespoke runner.
"""

from __future__ import annotations

import json
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.scorer import Metric, Score, Scorer, Target, accuracy, grouped, mean, metric, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver

from opacity.checker.base import parse_verdict
from opacity.checker.mechanical import MechanicalChecker
from opacity.checker.prompts import REVIEWER_INSTRUCTION, SYSTEM_PREAMBLE
from opacity.items import CLASS_NAMES, IMPLEMENTED, build_corpus
from opacity.sdc import policy
from opacity.variants import reviewer_text
from opacity.thinking import generation_extra, max_tokens_for, normalise_budget, normalise_effort, normalise_thinking

import opacity.harness.providers  # noqa: F401  registers the "cline" and "replay" model providers
from opacity.harness.typed import typed_reviewer

ARMS = ("llm", "mechanical", "typed")


# --- dataset -----------------------------------------------------------------


def samples(seed: int = 0, replicates: int = 1, levels: tuple[int, ...] = (0,), register: str = "default") -> list[Sample]:
    """Every bundle at every level. Ground truth goes in target and metadata,
    never in the input."""
    out: list[Sample] = []
    for item in build_corpus(seed=seed, replicates=replicates, register=register):
        for b in item.pair:
            t = b.truth
            assert t is not None
            cls = t.error_class if t.has_error else 0
            condition = "seeded" if t.has_error else "control"
            for level in levels:
                rendered = b.controlled(policy(level)).render()
                out.append(
                    Sample(
                        id=f"{b.item_id}-{condition}-s{level}",
                        input=rendered,
                        target="yes" if t.has_error else "no",
                        metadata={
                            "item_id": b.item_id,
                            "condition": condition,
                            "error_class": cls,
                            "class_name": CLASS_NAMES.get(cls, "control"),
                            "disclosure_level": level,
                            "short_name": t.short_name,
                            "rationale": t.rationale,
                            "rendered_chars": len(rendered),
                            "register": register,
                        },
                    )
                )
    return out


def _levels(levels: Any) -> tuple[int, ...]:
    """Accept 0, "0", "0,1,2,3", "0 1 2 3" or a list."""
    if isinstance(levels, int):
        return (levels,)
    if isinstance(levels, str):
        return tuple(int(x) for x in levels.replace(",", " ").split())
    return tuple(int(x) for x in levels)


# --- solvers: one per machine arm -------------------------------------------


@solver
def reviewer(variant: str = "default") -> Solver:
    """The LLM arm. Exactly the messages the bespoke checker sends: the
    system preamble, then the instruction and the bundle in one user turn.
    A non-default `variant` appends an exploratory block to the instruction
    (`opacity.variants`); the default is the pre-registered text alone."""
    instruction = reviewer_text(variant)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages = [
            ChatMessageSystem(content=SYSTEM_PREAMBLE),
            ChatMessageUser(content=instruction + "\n\n---\n\n" + state.input_text),
        ]
        return await generate(state)

    return solve


@solver
def mechanical(threshold: int = 10) -> Solver:
    """The mechanical arm: SDC rules only, no model. Its verdict is written
    as the completion in the same JSON shape so one scorer serves both arms."""
    chk = MechanicalChecker(threshold=threshold)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        v = chk.review(state.input_text)
        content = json.dumps(
            {
                "error_present": v.error_present,
                "confidence": v.confidence,
                "description": v.description,
                "release": v.release,
            }
        )
        state.messages.append(ChatMessageAssistant(content=content, model="mechanical"))
        state.output = ModelOutput.from_content(model="mechanical", content=content)
        state.output.metadata = dict(v.meta)
        return state

    return solve


# --- scorer ------------------------------------------------------------------

# Per-sample values are 0/1 indicators; the metrics turn them into the rates
# phase0.summarise reports: accuracy by class and level, flag rate split by
# condition (detection on seeded, false alarm on control), cannot-tell and
# parse-failure rates.
# Inspect flattens a grouped metric into the log as "<metric name>_<group>",
# and keys it by the metric's registered name. Three plain grouped() calls
# would all be called "grouped" and the last would win, so each grouping is
# its own registered metric: by_condition_seeded, by_class_3, by_level_2 ...
@metric
def by_condition(m: Metric) -> Metric:
    return grouped(m, "condition", all_label="all")


@metric
def by_class(m: Metric) -> Metric:
    return grouped(m, "error_class", all=False)


@metric
def by_level(m: Metric) -> Metric:
    return grouped(m, "disclosure_level", all=False)


METRICS = {
    "correct": [accuracy(), stderr(), by_condition(accuracy()), by_class(accuracy()), by_level(accuracy())],
    "flagged": [by_condition(mean()), by_level(mean())],
    "cannot_tell": [mean(), by_level(mean())],
    "parse_ok": [mean()],
}


@scorer(metrics=METRICS)
def verdict() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        v = parse_verdict(state.output.completion, model=state.output.model)
        return Score(
            value={
                "correct": float(v.error_present == target.text),
                "flagged": float(v.error_present == "yes"),
                "cannot_tell": float(v.error_present == "cannot_tell"),
                "parse_ok": float(v.parse_ok),
            },
            answer=v.error_present,
            explanation=v.description,
            metadata={
                "confidence": v.confidence,
                "release": v.release,
                "parse_ok": v.parse_ok,
                "target": target.text,
            },
        )

    return score


# --- task --------------------------------------------------------------------


reviewer_solver = reviewer  # the task argument `reviewer` shadows the solver's name


@task
def opacity(
    arm: str = "llm",
    seed: int = 0,
    replicates: int = 1,
    levels: Any = "0",
    threshold: int = 10,
    typed_url: str = "",
    typed_model: str = "openjev-latest",
    typed_samples: int = 1,
    typed_think: int = 0,
    thinking: Any = "default",
    register: str = "default",
    think_budget: int = 0,
    effort: str = "",
    reviewer: str = "default",
) -> Task:
    """Detectability of analytic error under disclosure control.

    arm:        "llm" (needs --model), "mechanical" (no model), or "typed"
                (a decision server; needs typed_url, no --model).
    seed:       corpus seed; replicates: datasets per class.
    levels:     disclosure levels to render, e.g. "0" or "0,1,2,3,4".
    threshold:  mechanical arm's small-cell threshold.
    typed_*:    openjev server url, served model name, re-read samples and
                thought budget for the typed arm.
    thinking:   LLM arm only. "off" sends chat_template_kwargs
                enable_thinking=false (vLLM: Qwen3, Qwen3.5, Gemma 4), "on"
                sends true, "default" sends nothing. Same switch as the
                harvest task; a reasoning model with thinking on spends the
                whole output budget before the JSON and every reply is
                recorded as unparseable (Qwen3.5-4B, 2026-10-21).
    register:   "default" or "matched" (controls rewritten in the seeded
                register; the register-tell ablation).
    think_budget, effort: LLM arm only; see `opacity.thinking`.
    reviewer:   LLM arm only: "default" (pre-registered), or the exploratory
                "checklist" or "stepwise" variants (`opacity.variants`).
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    thinking = normalise_thinking(thinking)
    think_budget = normalise_budget(think_budget)
    effort = normalise_effort(effort)
    extra = generation_extra(thinking, think_budget)
    if arm == "typed" and not typed_url:
        raise ValueError("arm='typed' needs -T typed_url=http://host:8080")
    lv = _levels(levels)
    dataset = MemoryDataset(
        samples(seed=seed, replicates=replicates, levels=lv, register=register),
        name=f"opacity-seed{seed}-r{replicates}-s{''.join(map(str, lv))}" + ("" if register == "default" else f"-{register}"),
    )
    if arm == "mechanical":
        solve = mechanical(threshold)
    elif arm == "typed":
        solve = typed_reviewer(typed_url, typed_model, typed_samples, typed_think)
    else:
        solve = reviewer_solver(reviewer)
    return Task(
        dataset=dataset,
        solver=solve,
        scorer=verdict(),
        model=None if arm == "llm" else "none/none",
        config=GenerateConfig(temperature=0.0, seed=0, max_tokens=max_tokens_for(think_budget, effort), extra_body=extra,
                              reasoning_effort=effort),
        name=f"opacity-{arm}",
        metadata={
            "arm": arm,
            "seed": seed,
            "replicates": replicates,
            "levels": list(lv),
            "classes": list(IMPLEMENTED),
            "class_names": {k: CLASS_NAMES[k] for k in IMPLEMENTED},
            "threshold": threshold if arm == "mechanical" else None,
            "typed": {"url": typed_url, "model": typed_model, "samples": typed_samples, "think": typed_think}
            if arm == "typed" else None,
            "thinking": thinking if arm == "llm" else None,
            "think_budget": think_budget if arm == "llm" else None,
            "effort": (effort or "") if arm == "llm" else None,
            "register": register,
            "reviewer": reviewer if arm == "llm" else None,
        },
    )
