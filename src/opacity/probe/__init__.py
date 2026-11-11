"""Probes on a model's internal states as an extra checker (docs/vetting.md, section 4).

The reading probe: a model reads each analyst write-up exactly as its
checker run read it (the messages are copied from the crosscheck log), one
forward pass with no generation, and a logistic regression maps its state at
the last prompt token to "the judge called this write-up wrong". It is
trained on the dev write-ups, tested once on the held-out ones, and compared
with the same model's own verdicts on the same write-ups.

    scripts/probe_inputs.py              (workstation)  labels, verbal scores, prompts
    python -m opacity.probe.extract      (Viking GPU)   states at three depths
    python -m opacity.probe.fit          (workstation)  dev selection, held-out test, stop rule

Everything a run depends on is fixed here, before any state is extracted.
"""

from __future__ import annotations

PROBES: dict[str, dict] = {
    # The recommended analyst and the best verbal checker. Its prompts come
    # from the checker runs shown the pack; dev from the greedy thinking-off
    # run (so the first-token check applies), held-out from the 1k-think run
    # (thinking changes the chat template, not the messages; checked at build).
    "Qwen3.5-27B": {
        "hf": "Qwen/Qwen3.5-27B",
        "chat": {"enable_thinking": False},
        "prompts": {"dev": "Qwen3.5-27B|off|pack", "heldout": "Qwen3.5-27B|think1024|pack"},
        # the template each source run used, for checking prompt lengths
        "source_chat": {"dev": {"enable_thinking": False}, "heldout": {"enable_thinking": True}},
        "first_token": ("dev",),
        "verbal": ("Qwen3.5-27B|off", "Qwen3.5-27B|off|pack", "Qwen3.5-27B|off|stepwise",
                   "Qwen3.5-27B|think1024", "Qwen3.5-27B|think1024|pack"),
    },
    # The recommended checker, from another family. It always reasons, so
    # its logged reply never starts at the last prompt token.
    "gpt-oss-20b": {
        "hf": "openai/gpt-oss-20b",
        "chat": {"reasoning_effort": "medium"},
        "prompts": {"dev": "gpt-oss-20b|effort-medium", "heldout": "gpt-oss-20b|effort-medium"},
        "source_chat": {"dev": {"reasoning_effort": "medium"}, "heldout": {"reasoning_effort": "medium"}},
        "first_token": (),
        "verbal": ("gpt-oss-20b|effort-low", "gpt-oss-20b|effort-medium"),
    },
}

# The writing probe for the positive control (docs/vetting.md 4.8): the 27B
# reads each of its own write-ups, honest or steered, after the honest prompt
# for that task and replicate, so the steering note is never in its context.
PROBES["Qwen3.5-27B-steer"] = {
    "hf": "Qwen/Qwen3.5-27B",
    "chat": {"enable_thinking": False},
    "source_chat": {"dev": {"enable_thinking": False}, "heldout": {"enable_thinking": False}},
    "first_token": (),
    "verbal": (),
}

STEER = {
    "probe": "Qwen3.5-27B-steer",
    # generation, both arms: the plan file scripts/plans/steer-generate.txt
    "harvest": "-T analyst_prompt=rules -T thinking=off -T sampling=qwen-instruct -T vary_seed=true -T replicates=5",
    # black-box detectors run on the same write-ups
    "checkers": ("Qwen3.5-27B|off|pack", "gpt-oss-20b|effort-medium"),
    "false_alarm": 0.05,  # the hit rate is read at the threshold flagging this share of honest dev write-ups
}

PRIMARY = "Qwen3.5-27B"
MARGIN = 0.05  # held-out AUC over the best verbal AUC (Wade, 26 October 2026)
# The held-out write-ups: every one the judge scored that both prompt sources
# covered (the 12 analyst configurations every held-out checker run saw).
HELDOUT_COVER = ("Qwen3.5-27B|think1024|pack", "gpt-oss-20b|effort-medium")
DEPTHS = (0.25, 0.5, 0.75)  # fractions of the model's layers


def depth_index(n_layers: int) -> list[int]:
    """Indices into a hidden-state tuple whose entry 0 is the embeddings."""
    return [round(n_layers * f) for f in DEPTHS]


def key_str(log: str, sample: str) -> str:
    """One write-up as a single string: harvest log name and sample id."""
    return f"{log}|{sample}"
