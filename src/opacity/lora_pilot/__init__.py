"""Honest-only LoRA engineering pilot; independent of the Inspect arms."""

MODEL = "Qwen/Qwen3.5-27B"
REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
SCHEMA = 1
# "profile" (27 October, after the pilot): training memory at study lengths, no predecessor.
STAGE_SECONDS = {"smoke": 1800, "train": 5400, "reload": 1800, "profile": 1800}
MAX_RESERVED_SECONDS = 9000
