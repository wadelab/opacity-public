"""Scorer entry point for re-scoring harvest logs off-cluster.

Harvest jobs run with --no-score so no judge key reaches Viking. Score the
pulled log on d2-1 with the judge the run declared:

    set -a; source ~/.config/mimo/api.env; set +a
    inspect score <log>.eval --overwrite \
        --scorer scripts/harvest_judge.py@judged -S judge=openai-api/mimo/mimo-v2.6-pro

`inspect score --scorer <file>@<name>` wants the scorer defined in that file
(pointing it at task.py breaks the package's relative imports, and a re-export
is not found), so this file declares a scorer of the same name and metrics
that delegates to the package's.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.scorer import Scorer, scorer

from opacity.harvest import task as _harvest


@scorer(metrics=_harvest.METRICS)
def judged(judge: str | Any, max_tokens: int = 20000) -> Scorer:
    return _harvest.judged(judge, max_tokens=max_tokens)


@scorer(metrics=_harvest.METRICS)
def judged_second(judge: str | Any, max_tokens: int = 20000) -> Scorer:
    """The same judge scorer under a second name, so a second judge's verdicts
    can be appended to a log beside the first's (`--action append`) and the
    two compared. Use a judge from a different family: the first is MiMo pro,
    the second DeepSeek 4.1 flash via Cline."""
    return _harvest.judged(judge, max_tokens=max_tokens)


@scorer(metrics=_harvest.METRICS)
def judged_alt(judge: str | Any, max_tokens: int = 20000, reasoning_effort: str | None = None) -> Scorer:
    """The same judge scorer under a third name, for a judge from another
    family again (added 23 October 2026 while the hosted MiMo and Cline
    judges were paused: a local model on Viking, or an OpenAI or Claude
    model), so its verdicts sit beside `judged` and `judged_second` without
    replacing either. Run through scripts/local_judge.sh for a local judge."""
    return _harvest.judged(judge, max_tokens=max_tokens, reasoning_effort=reasoning_effort)


@scorer(metrics=_harvest.METRICS)
def judged_mimo_flash(judge: str | Any, max_tokens: int = 20000) -> Scorer:
    """The same judge scorer under a fourth name, for MiMo v2.6 flash, the
    cheaper hosted MiMo (Wade, 24 October 2026); validated against the other
    judges on runs/inspect/judge-validation/ before any use."""
    return _harvest.judged(judge, max_tokens=max_tokens)
