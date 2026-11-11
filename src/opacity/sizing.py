"""The model-sizing study's analysis: per-item tables from the logs, and the
reliability, coverage and cost of an analyst followed by checking stages.

Every stage is run once on every item and the configurations are evaluated
offline from the per-item verdicts. At temperature 0 a cascade evaluated this
way gives what it would online, and the joint detection of two stages is
measured on the same items rather than computed from an independence
assumption.

Outcomes per harvested task, after every stage:

    released and right     a legitimate answer delivered
    released and wrong     the failure the study exists to bound
    withheld               declined by the analyst, or stopped by a stage;
                           goes back to the researcher or to a human

Rates are over all tasks in a condition (seeded or control), so a decline is
neither right nor wrong. An unparseable analyst reply counts as released and
wrong, as in the harvest's own scoring, unless a stage stops it. Intervals
resample tasks, not replicates, because the three replicates of a task share
its trap and its wording.
"""

from __future__ import annotations

import glob
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from inspect_ai.log import read_eval_log

from opacity.checker.base import parse_verdict
from opacity.harvest.task import parse_json_object, unwrap

# --- configurations ----------------------------------------------------------


def short(model: str) -> str:
    return str(model).rstrip("/").split("/")[-1]


def pack_version(tags: Iterable[str]) -> str:
    """Harvest packs were corrected on 2026-10-22 (commit f50c08e); every
    run tagged sweep4- or sweep5- saw the old ones."""
    return "v1" if any(t.startswith(("sweep4-", "sweep5-")) for t in tags) else "v2"


def analyst_key(log) -> str:
    a = log.eval.task_args or {}
    md = log.eval.metadata or {}
    think = _think_label(a.get("thinking", md.get("thinking", "default")), a.get("think_budget", 0), a.get("effort", ""))
    parts = [short(log.eval.model), md.get("arm", a.get("arm", "open")), think]
    prompt = a.get("analyst_prompt") or ("checklist" if a.get("checklist") else "")
    if prompt and prompt != "default":
        parts.append(prompt)
    if md.get("task_set", a.get("task_set", "dev")) != "dev":
        parts.append(md.get("task_set", a.get("task_set")))
    if md.get("sampling", a.get("sampling", "greedy")) not in ("greedy", None, ""):
        parts.append(md.get("sampling", a.get("sampling")))
    # the exploratory preparer arm (a hosted model rewrote the request): never
    # pooled with the unprepared run, and outside fresh_select's ARMS
    if md.get("prepared"):
        parts.append(f"prep-{md['prepared']}")
    if md.get("tools"):
        # calculator version 2 has guardrails: never pooled with version 1
        parts.append({"calculator": "calc", "calculator2": "calc2"}.get(md["tools"], md["tools"]))
    parts.append(pack_version(log.eval.tags or []))
    return "|".join(str(p) for p in parts)


def checker_key(log) -> str:
    a = log.eval.task_args or {}
    md = log.eval.metadata or {}
    if md.get("arm", a.get("arm")) == "typed" or str(log.eval.model).startswith("none/"):
        t = md.get("typed") or {}
        return f"typed|think{t.get('think', a.get('typed_think', 0)) or 0}"
    think = _think_label(a.get("thinking", "default"), a.get("think_budget", 0), a.get("effort", ""))
    parts = [short(log.eval.model), think]
    if a.get("reviewer", "default") != "default":
        parts.append(a["reviewer"])
    if a.get("show_pack"):
        parts.append("pack")
    return "|".join(parts)


def _think_label(thinking: Any, budget: Any, effort: Any) -> str:
    if effort:
        return f"effort-{effort}"
    if budget:
        return f"think{int(budget)}"
    if thinking is False or thinking == "off":
        return "off"
    if thinking is True or thinking == "on":
        return "on"
    return "default"


# --- per-item tables -----------------------------------------------------------


@dataclass
class Item:
    """One harvested task attempt by one analyst configuration."""

    config: str
    log: str
    sample: str
    task_id: str
    replicate: int
    condition: str  # "invited" or "control"
    primary_class: int
    declined: bool
    parse_ok: bool
    wrong: bool | None  # the judge's verdict; None when declined or unjudged
    judge_ok: bool
    output_tokens: int
    seconds: float
    bundle: dict | None = None
    pack: str = ""
    wrong2: bool | None = None  # the second judge's verdict, where there is one
    wrong3: bool | None = None  # the alternative judge's (judged_alt), where there is one
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.log, self.sample)


def load_analyst(path: str, keep_text: bool = True) -> list[Item]:
    log = read_eval_log(path)
    if log.status != "success" or not log.samples:
        return []
    cfg = analyst_key(log)
    out = []
    for s in log.samples:
        m = s.metadata or {}
        bundle, declined, _, ok = unwrap(parse_json_object(s.output.completion if s.output else ""))
        j = (s.scores or {}).get("judged")
        jv = j.value if j is not None and isinstance(j.value, dict) else {}
        judge_ok = bool(jv.get("judge_parse_ok")) and not declined
        j2 = (s.scores or {}).get("judged_second")
        j2v = j2.value if j2 is not None and isinstance(j2.value, dict) else {}
        wrong2 = bool(j2v.get("conclusion_wrong")) if (j2v.get("judge_parse_ok") and not declined) else None
        j3 = (s.scores or {}).get("judged_alt")
        j3v = j3.value if j3 is not None and isinstance(j3.value, dict) else {}
        wrong3 = bool(j3v.get("conclusion_wrong")) if (j3v.get("judge_parse_ok") and not declined) else None
        out.append(Item(
            config=cfg, log=Path(path).name, sample=str(s.id), task_id=m.get("task_id", ""),
            replicate=int(m.get("replicate", 0)), condition=m.get("condition", ""),
            primary_class=int(m.get("primary_class", 0) or 0), declined=bool(declined), parse_ok=bool(ok),
            wrong=(bool(jv.get("conclusion_wrong")) if judge_ok else None), judge_ok=judge_ok, wrong2=wrong2, wrong3=wrong3,
            output_tokens=_tokens(s), seconds=float(s.total_time or 0.0),
            bundle=bundle if keep_text else None, pack=m.get("pack", "") if keep_text else "",
            extra={"self_check": ((parse_json_object(s.output.completion) or {}).get("self_check") if s.output else None)},
        ))
    return out


def _tokens(sample) -> int:
    """Output tokens over every model call in the sample (two in the
    self-check arm), reasoning included."""
    total = 0
    for mu in (sample.model_usage or {}).values():
        total += int(mu.output_tokens or 0)
    if not total and sample.output and sample.output.usage:
        total = int(sample.output.usage.output_tokens or 0)
    return total


@dataclass
class Verdict:
    flag: str  # "yes", "no", "cannot_tell"
    confidence: int
    parse_ok: bool
    output_tokens: int
    seconds: float

    @property
    def clean_no(self) -> bool:
        return self.parse_ok and self.flag == "no"


def load_crosscheck(path: str) -> tuple[str, dict[tuple[str, str], Verdict]]:
    """A cross-check log as {(harvest log name, harvest sample id): verdict}."""
    log = read_eval_log(path)
    if log.status != "success" or not log.samples:
        return checker_key(log), {}
    out = {}
    for s in log.samples:
        m = s.metadata or {}
        v = parse_verdict(s.output.completion if s.output else "")
        out[(Path(str(m.get("harvest_log", ""))).name, str(m.get("harvest_sample_id")))] = Verdict(
            v.error_present, v.confidence, v.parse_ok, _tokens(s), float(s.total_time or 0.0))
    return checker_key(log), out


def load_scripted(path: str) -> tuple[str, list[dict]]:
    """An opacity-task log as one row per bundle verdict."""
    log = read_eval_log(path)
    if not log.samples:
        return checker_key(log), []
    rows = []
    for s in log.samples:
        m = s.metadata or {}
        v = parse_verdict(s.output.completion if s.output else "")
        rows.append({
            "item_id": m.get("item_id"), "condition": m.get("condition"), "error_class": m.get("error_class"),
            "level": m.get("disclosure_level"), "register": m.get("register", "default"),
            "flag": v.error_present, "confidence": v.confidence, "parse_ok": v.parse_ok,
            "output_tokens": _tokens(s), "seconds": float(s.total_time or 0.0),
            "stop": s.output.stop_reason if s.output else "",
        })
    return checker_key(log), rows


# --- configurations of stages ----------------------------------------------------

Stage = Callable[[Item], bool]
"""A stage returns True to let the write-up through, False to withhold it."""


def checker_stage(verdicts: dict[tuple[str, str], Verdict], fail_closed: bool = True) -> Stage:
    """Release on a clean "no". With fail_closed False, withhold only on "yes"."""
    def stage(it: Item) -> bool:
        v = verdicts.get(it.key)
        if v is None:
            return not fail_closed
        return v.clean_no if fail_closed else not (v.parse_ok and v.flag == "yes")
    return stage


def verifier_stage(flags: dict[tuple[str, str], bool]) -> Stage:
    return lambda it: not flags.get(it.key, False)


def screen_then(screen: dict[tuple[str, str], Verdict], full: Stage, min_conf: int = 0) -> Stage:
    """A cheap screen that releases a confident clean "no" without further
    review, and sends everything else to the full stage."""
    def stage(it: Item) -> bool:
        v = screen.get(it.key)
        if v is not None and v.clean_no and v.confidence >= min_conf:
            return True
        return full(it)
    return stage


JUDGES = ("first", "second", "either", "both", "alt")
"""Whose verdict counts: MiMo (first), DeepSeek (second), wrong if either
says wrong (strict), wrong only if both do (lenient), or the alternative
judge (judged_alt; the local gpt-oss 120B from 24 October 2026). Items the
named judge has not scored are unjudged."""


def verdict(it: Item, judge: str = "first") -> bool | None:
    if judge == "first":
        return it.wrong
    if judge == "second":
        return it.wrong2
    if judge == "alt":
        return it.wrong3
    if it.wrong is None or it.wrong2 is None:
        return None
    return (it.wrong or it.wrong2) if judge == "either" else (it.wrong and it.wrong2)


def outcomes(items: list[Item], stages: list[Stage] = (), judge: str = "first") -> list[str]:
    """Per item: "right", "wrong", "withheld" or "unjudged"."""
    out = []
    for it in items:
        w = verdict(it, judge)
        if it.declined or not all(st(it) for st in stages):
            out.append("withheld")
        elif not it.parse_ok:
            out.append("wrong")
        elif w is None:
            out.append("unjudged")
        else:
            out.append("wrong" if w else "right")
    return out


def rates(items: list[Item], outs: list[str]) -> dict[str, float]:
    s = [o for it, o in zip(items, outs) if it.condition != "control" and o != "unjudged"]
    c = [o for it, o in zip(items, outs) if it.condition == "control" and o != "unjudged"]
    a = s + c
    def frac(xs, v):
        return sum(x == v for x in xs) / len(xs) if xs else float("nan")
    released = [o for o in a if o in ("right", "wrong")]
    return {
        "n_seeded": len(s), "n_control": len(c),
        "wrong_release_seeded": frac(s, "wrong"),
        "right_release_seeded": frac(s, "right"),
        "withheld_seeded": frac(s, "withheld"),
        "answer_control": frac(c, "right") + frac(c, "wrong"),
        "right_release_control": frac(c, "right"),
        "wrong_release_all": frac(a, "wrong"),
        "coverage_all": frac(a, "right"),
        "withheld_all": frac(a, "withheld"),
        "p_wrong_given_released": (sum(o == "wrong" for o in released) / len(released)) if released else float("nan"),
    }


def evaluate(items: list[Item], stages: list[Stage] = (), boot: int = 2000, seed: int = 0,
             judge: str = "first") -> dict[str, Any]:
    """Rates with task-resampled 95% intervals."""
    outs = outcomes(items, stages, judge)
    point = rates(items, outs)
    by_task: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        by_task.setdefault(it.task_id, []).append(i)
    tasks = sorted(by_task)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point if not k.startswith("n_")}
    for _ in range(boot):
        pick = rng.choice(len(tasks), size=len(tasks), replace=True)
        idx = [i for t in pick for i in by_task[tasks[t]]]
        r = rates([items[i] for i in idx], [outs[i] for i in idx])
        for k in draws:
            if not math.isnan(r[k]):
                draws[k].append(r[k])
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (float("nan"), float("nan"))
          for k, v in draws.items()}
    return {"point": point, "ci": ci, "outcomes": outs}


def gate(point: dict[str, float], wrong_max: float = 0.05, answer_min: float = 0.75) -> bool:
    """The proposed fail-closed gate: wrong conclusions on seeded tasks at
    most wrong_max and answers on clean controls at least answer_min."""
    return point["wrong_release_seeded"] <= wrong_max and point["answer_control"] >= answer_min


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def glob_logs(*patterns: str) -> list[str]:
    return sorted({p for pat in patterns for p in glob.glob(pat)})


def paired_contrast(a: list[Item], b: list[Item], stages_a: list[Stage] = (), stages_b: list[Stage] = (),
                    metric: str = "wrong_release_seeded", boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """b minus a on `metric`, over the samples both configurations ran (same
    task, same replicate draw), with a 95% interval from resampling tasks
    jointly. Positive means b is higher."""
    ia = {it.sample: it for it in a}
    ib = {it.sample: it for it in b}
    common = sorted(set(ia) & set(ib))
    xa, xb = [ia[s] for s in common], [ib[s] for s in common]
    oa, ob = outcomes(xa, stages_a), outcomes(xb, stages_b)
    point = rates(xb, ob)[metric] - rates(xa, oa)[metric]
    by_task: dict[str, list[int]] = {}
    for i, it in enumerate(xa):
        by_task.setdefault(it.task_id, []).append(i)
    tasks = sorted(by_task)
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(boot):
        idx = [i for t in rng.choice(len(tasks), size=len(tasks), replace=True) for i in by_task[tasks[t]]]
        ra = rates([xa[i] for i in idx], [oa[i] for i in idx])[metric]
        rb = rates([xb[i] for i in idx], [ob[i] for i in idx])[metric]
        if not (math.isnan(ra) or math.isnan(rb)):
            ds.append(rb - ra)
    lo, hi = (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))) if ds else (float("nan"), float("nan"))
    return {"n": len(common), "diff": point, "lo": lo, "hi": hi}
