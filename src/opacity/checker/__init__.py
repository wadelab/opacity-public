"""Checker arms.

Every arm receives the same rendered bundle text and the same instruction, and
returns the same `Verdict`. The arms differ only in what does the judging.
"""

from .base import Checker, Verdict, OpenAICompatibleChecker, unwrap_envelope
from .mechanical import MechanicalChecker
from .prompts import REVIEWER_INSTRUCTION

__all__ = [
    "Checker",
    "Verdict",
    "OpenAICompatibleChecker",
    "unwrap_envelope",
    "MechanicalChecker",
    "REVIEWER_INSTRUCTION",
]
