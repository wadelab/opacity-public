"""Inspect harness: the study as tasks. See task.py."""

from .task import opacity, reviewer, mechanical, verdict, samples
from opacity.harvest.task import harvest  # second task, same entry point
from opacity.crosscheck.task import crosscheck  # third task: checkers on harvested bundles

__all__ = ["opacity", "reviewer", "mechanical", "verdict", "samples", "harvest", "crosscheck"]
