"""Tables for the model-sizing study -> runs/summary/sizing/*.csv.

    .venv/bin/python scripts/sizing_report.py

Reads every judged harvest log (runs/inspect/scored/ for the 21-22 October
sweep on the old packs, runs/inspect/scored-sizing/ for this study), every
scripted-corpus log and cross-check log pulled from Viking or run on d2-1,
and writes:

    analysts.csv     one row per analyst configuration: wrong releases on
                     seeded tasks, answers on controls, coverage, declines,
                     with task-resampled intervals, tokens and GPU-seconds
    checkers.csv     one row per checker configuration and level on the
                     scripted corpus: detection, false alarm, cannot-tell,
                     class 6, AUC, parse and truncation, tokens, GPU-seconds
    matrix.csv       checker configuration x analyst configuration on the
                     released write-ups: sensitivity on judge-wrong, false
                     alarm on judge-right
    verifier.csv     the deterministic verifier per analyst configuration
    pipelines.csv    analyst configuration x checking pipeline: wrong
                     releases, coverage, withheld, cost per task

Every number is recomputed from per-sample records. GPU-seconds are the
eval's wall clock times the job's GPU count over its samples, so they include
the concurrency the run used (16 to 32 requests in flight).
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opacity import sizing as S  # noqa: E402
from opacity.verify import tables_from_markdown, verify  # noqa: E402

OUT = ROOT / "runs" / "summary" / "sizing"
SUBMISSIONS = ROOT / "runs" / "viking-submissions.txt"

# Model facts: total and active parameters (billions), weight format, and the
# GPUs it was served on in this study. From the model cards and the
# checkpoint sizes on the Hugging Face API, not from the names alone.
MODELS = {
    "Qwen3.5-0.8B": (0.8, 0.8, "bf16"), "Qwen3.5-2B": (2, 2, "bf16"), "Qwen3.5-4B": (4, 4, "bf16"),
    "Qwen3.5-9B": (9, 9, "bf16"), "Qwen3.5-27B": (27, 27, "bf16"), "Qwen3.5-35B-A3B": (35, 3, "bf16"),
    "Qwen3.5-122B-A10B-NVFP4": (122, 10, "nvfp4"), "Qwen3-32B": (32, 32, "bf16"),
    "Mistral-Small-3.2-24B-Instruct-2506-NVFP4": (24, 24, "nvfp4"), "Llama-3.3-70B-Instruct-NVFP4": (70, 70, "nvfp4"),
    "gpt-oss-20b": (21, 3.6, "mxfp4"), "gpt-oss-120b": (117, 5.1, "mxfp4"), "gemma-4-31B-it": (31, 31, "bf16"),
    "gemma-4-12B-it": (12, 12, "bf16"), "gemma-4-26B-A4B-it": (26, 4, "bf16"), "phi-4": (14, 14, "bf16"),
    "GLM-4-32B-0414": (32, 32, "bf16"), "GLM-4-9B-0414": (9, 9, "bf16"),
    "mimo-v2.6-pro": (None, None, "hosted"), "deepseek-v4.1-flash": (None, None, "hosted"),
    "typed": (26, 4, "nvfp4"),
}


def gpus_by_job() -> dict[str, tuple[str, int]]:
    """Slurm job id -> (hardware, GPU count), from the submissions file.
    Lines without a partition are the H100 sweep with TP in the tag."""
    out: dict[str, tuple[str, int]] = {}
    if not SUBMISSIONS.exists():
        return out
    for line in SUBMISSIONS.read_text().splitlines():
        f = line.split()
        if not f:
            continue
        m = re.search(r"\b(gpuplus|gpu) x(\d+)\b", line)
        if m:
            out[f[0]] = ("H100" if m.group(1) == "gpuplus" else "A40", int(m.group(2)))
        else:
            out[f[0]] = ("H100", 2 if "122B" in line else 1)
    return out


JOBS = gpus_by_job()


def hardware(log) -> tuple[str, int]:
    tags = log.eval.tags or []
    if "viking2" not in tags:
        return ("hosted", 0)
    for t in tags:
        if t in JOBS:
            return JOBS[t]
    if any("typed" in t for t in tags):
        return ("H100", 1)
    return ("H100", 1)


def gpu_seconds(log) -> float | None:
    hw, n = hardware(log)
    if not n or not log.stats or not log.stats.started_at or not log.stats.completed_at:
        return None
    from datetime import datetime
    t0 = datetime.fromisoformat(log.stats.started_at)
    t1 = datetime.fromisoformat(log.stats.completed_at)
    k = len(log.samples or []) or (log.results.total_samples if log.results else 0)
    return (t1 - t0).total_seconds() * n / k if k else None


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


def write(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})
    print(f"{name}: {len(rows)} rows")


# --- analysts -------------------------------------------------------------------


def harvest_logs() -> list[str]:
    return S.glob_logs(str(ROOT / "runs/inspect/scored/*_harvest*.eval"), str(ROOT / "runs/inspect/scored-sizing/*_harvest*.eval"))


def analyst_rows(logs: list[str]) -> tuple[list[dict], dict[str, list]]:
    rows, items_by = [], {}
    for p in logs:
        try:
            log = read_eval_log(p, header_only=True)
            full = read_eval_log(p)
        except Exception:
            continue  # a copy the judge loop is still writing
        if not full.samples or any("judged" not in (s.scores or {}) for s in full.samples):
            continue  # not judged yet
        items = S.load_analyst(p)
        if not items:
            continue
        cfg = items[0].config
        items_by.setdefault(cfg, [])
        items_by[cfg] += items
        ev = S.evaluate(items, boot=1000)
        pt, ci = ev["point"], ev["ci"]
        model = S.short(log.eval.model)
        tot, act, q = MODELS.get(model, (None, None, ""))
        hw, n = hardware(log)
        rows.append({
            "config": cfg, "model": model, "params_b": tot, "active_b": act, "weights": q, "hardware": hw, "gpus": n,
            "log": Path(p).name, "n": len(items),
            "wrong_release_seeded": pt["wrong_release_seeded"], "lo": ci["wrong_release_seeded"][0], "hi": ci["wrong_release_seeded"][1],
            "answer_control": pt["answer_control"], "answer_control_lo": ci["answer_control"][0],
            "coverage_all": pt["coverage_all"], "withheld_all": pt["withheld_all"], "withheld_seeded": pt["withheld_seeded"],
            "right_release_seeded": pt["right_release_seeded"], "p_wrong_given_released": pt["p_wrong_given_released"],
            "parse_ok": sum(it.parse_ok for it in items) / len(items),
            "judge_unparsed": sum(1 for it in items if not it.declined and it.wrong is None),
            "gate": S.gate(pt), "median_output_tokens": float(np.median([it.output_tokens for it in items])),
            "gpu_s_per_task": gpu_seconds(log), "median_latency_s": float(np.median([it.seconds for it in items])),
            # blank until the second judge has scored the log: with no verdicts
            # only the unparseable replies (counted wrong) would be left
            **{f"wrong_seeded_{j}": S.evaluate(items, boot=0, judge=j)["point"]["wrong_release_seeded"]
               if any(it.wrong2 is not None for it in items) else None
               for j in ("second", "either", "both")},
            "n_second": sum(it.wrong2 is not None for it in items),
        })
    return rows, items_by


# --- checkers on the scripted corpus ------------------------------------------------


def auc(rows: list[dict]) -> float:
    """Seeded against control, scoring yes = 50 + conf/2, cannot_tell = 50,
    no = 50 - conf/2 (as scripts/auc_by_level.py)."""
    def score(r):
        if r["flag"] == "yes":
            return 50 + r["confidence"] / 2
        if r["flag"] == "no":
            return 50 - r["confidence"] / 2
        return 50.0
    pos = [score(r) for r in rows if r["condition"] == "seeded"]
    neg = [score(r) for r in rows if r["condition"] == "control"]
    if not pos or not neg:
        return float("nan")
    gt = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return gt / (len(pos) * len(neg))


def checker_rows(logs: list[str]) -> list[dict]:
    out = []
    for p in logs:
        log = read_eval_log(p, header_only=True)
        if log.status != "success" or not str(log.eval.task).startswith("opacity-") or log.eval.task == "opacity-mechanical":
            continue
        key, rows = S.load_scripted(p)
        reg = (log.eval.task_args or {}).get("register", "default")
        tags = ",".join(t for t in (log.eval.tags or []) if not t.isdigit() and t != "viking2")
        gs = gpu_seconds(read_eval_log(p))
        for level in sorted({r["level"] for r in rows}):
            rs = [r for r in rows if r["level"] == level]
            sd = [r for r in rs if r["condition"] == "seeded"]
            ct = [r for r in rs if r["condition"] == "control"]
            by_cls = {c: [r for r in sd if r["error_class"] == c] for c in range(1, 9)}
            det_cls = {c: (sum(r["flag"] == "yes" for r in v) / len(v) if v else float("nan")) for c, v in by_cls.items()}
            det = sum(r["flag"] == "yes" for r in sd) / len(sd)
            fa = sum(r["flag"] == "yes" for r in ct) / len(ct)
            lo, hi = S.wilson(sum(r["flag"] == "yes" for r in sd), len(sd))
            flo, fhi = S.wilson(sum(r["flag"] == "yes" for r in ct), len(ct))
            out.append({
                "config": key, "register": reg, "tags": tags, "log": Path(p).name, "level": level,
                "n_seeded": len(sd), "detection": det, "det_lo": lo, "det_hi": hi, "false_alarm": fa, "fa_lo": flo, "fa_hi": fhi,
                "cannot_tell_seeded": sum(r["flag"] == "cannot_tell" for r in sd) / len(sd),
                "cannot_tell_control": sum(r["flag"] == "cannot_tell" for r in ct) / len(ct),
                **{f"det_c{c}": det_cls[c] for c in range(1, 9)},
                "min_class_det": min(v for v in det_cls.values() if not np.isnan(v)),
                "gate": (min(v for v in det_cls.values() if not np.isnan(v)) >= 0.8 and fa <= 0.2) if level == 0 else "",
                "auc": auc(rs), "parse_ok": sum(r["parse_ok"] for r in rs) / len(rs),
                "truncated": sum(r["stop"] == "max_tokens" for r in rs) / len(rs),
                "median_output_tokens": float(np.median([r["output_tokens"] for r in rs])),
                "gpu_s_per_verdict": gs, "median_latency_s": float(np.median([r["seconds"] for r in rs])),
            })
    return out


def scripted_logs() -> list[str]:
    return S.glob_logs(str(ROOT / "runs/inspect/*opacity-*.eval"), str(ROOT / "runs/inspect/viking/*opacity-*.eval"))


def crosscheck_logs() -> list[str]:
    return S.glob_logs(str(ROOT / "runs/inspect/viking/crosscheck/*crosscheck*.eval"), str(ROOT / "runs/inspect/crosscheck/*crosscheck*.eval"))


# --- matrix ---------------------------------------------------------------------------


def matrix_rows(items_by: dict[str, list], checkers: dict[str, dict]) -> list[dict]:
    out = []
    for ck, verdicts in checkers.items():
        for cfg, items in items_by.items():
            got = [(it, verdicts[it.key]) for it in items if it.key in verdicts and it.wrong is not None]
            if not got:
                continue
            wrong = [v for it, v in got if it.wrong]
            right = [v for it, v in got if not it.wrong]
            agent = cfg.split("|")[0]
            out.append({
                "checker": ck, "analyst": cfg, "self": int(ck.split("|")[0] == agent),
                "n_wrong": len(wrong), "sensitivity": (sum(v.flag == "yes" for v in wrong) / len(wrong)) if wrong else float("nan"),
                "withhold_on_wrong": (sum(not v.clean_no for v in wrong) / len(wrong)) if wrong else float("nan"),
                "n_right": len(right), "false_alarm": (sum(v.flag == "yes" for v in right) / len(right)) if right else float("nan"),
                "withhold_on_right": (sum(not v.clean_no for v in right) / len(right)) if right else float("nan"),
                "cannot_tell": sum(v.flag == "cannot_tell" for _, v in got) / len(got),
                "parse_ok": sum(v.parse_ok for _, v in got) / len(got),
            })
    return out


def kappa(pairs: list[tuple[int, int]]) -> float:
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def judge_rows(logs: list[str]) -> list[dict]:
    """MiMo (judged) against DeepSeek (judged_second) on conclusion_wrong,
    over released write-ups both judges parsed."""
    out, pooled = [], []
    for p in logs:
        try:
            log = read_eval_log(p)
        except Exception:
            continue
        ss = log.samples or []
        if not ss or not any("judged_second" in (s.scores or {}) for s in ss):
            continue
        pairs = []
        for s in ss:
            a, b = (s.scores or {}).get("judged"), (s.scores or {}).get("judged_second")
            if not a or not b or a.value.get("declined") or not a.value.get("judge_parse_ok") or not b.value.get("judge_parse_ok"):
                continue
            pairs.append((int(a.value["conclusion_wrong"]), int(b.value["conclusion_wrong"])))
        pooled += pairs
        items = S.load_analyst(p, keep_text=False)
        out.append({"analyst": items[0].config if items else Path(p).name, "n_both": len(pairs),
                    "agree": sum(a == b for a, b in pairs) / len(pairs) if pairs else float("nan"),
                    "kappa": kappa(pairs), "wrong_mimo": sum(a for a, _ in pairs) / len(pairs) if pairs else float("nan"),
                    "wrong_deepseek": sum(b for _, b in pairs) / len(pairs) if pairs else float("nan")})
    if pooled:
        out.append({"analyst": "pooled", "n_both": len(pooled), "agree": sum(a == b for a, b in pooled) / len(pooled),
                    "kappa": kappa(pooled), "wrong_mimo": sum(a for a, _ in pooled) / len(pooled),
                    "wrong_deepseek": sum(b for _, b in pooled) / len(pooled)})
    return out


FOCUS_CHECKERS = {"Qwen3.5-27B|off", "Qwen3.5-27B|think1024", "Qwen3.5-27B|off|stepwise", "gpt-oss-20b|effort-medium",
                  "Mistral-Small-3.2-24B-Instruct-2506-NVFP4|default", "Qwen3.5-9B|think1024", "Qwen3.5-35B-A3B|off",
                  "Qwen3.5-35B-A3B|think1024", "Mistral-Small-3.2-24B-Instruct-2506-NVFP4|off",
                  "Qwen3.5-27B|think1024|pack", "Qwen3.5-27B|off|pack", "Qwen3.5-122B-A10B-NVFP4|off",
                  "gpt-oss-120b|effort-low", "gpt-oss-20b|effort-low"}
SCREENS = {"typed|think0", "typed|think512", "Qwen3.5-4B|off", "Qwen3.5-9B|off", "Mistral-Small-3.2-24B-Instruct-2506-NVFP4|off",
           "Mistral-Small-3.2-24B-Instruct-2506-NVFP4|default", "Qwen3.5-27B|off", "gpt-oss-20b|effort-low"}
"""A screen releases a confident clean no (confidence 80 or more) and sends
everything else to the full checker. With the same model as screen and full
checker this is the targeted second pass: think only when the quick check is
not sure."""


def main() -> None:
    logs = harvest_logs()
    arows, items_by = analyst_rows(logs)
    write("analysts.csv", arows)
    write("checkers.csv", checker_rows(scripted_logs()))
    checkers: dict[str, dict] = {}
    ck_cost: dict[str, list[float]] = defaultdict(list)
    for p in crosscheck_logs():
        key, v = S.load_crosscheck(p)
        if v:
            checkers.setdefault(key, {}).update(v)
            gs = gpu_seconds(read_eval_log(p, header_only=True))
            if gs is not None:
                ck_cost[key].append(gs)
    cost = {k: float(np.median(v)) for k, v in ck_cost.items()}
    a_cost = {r["config"]: r["gpu_s_per_task"] for r in arows}
    write("matrix.csv", matrix_rows(items_by, checkers))
    vrows, vflags = [], {}
    for cfg, items in items_by.items():
        fl = {it.key: [f.rule for f in verify(tables_from_markdown(it.pack), it.bundle) if f.flagged] for it in items}
        vflags.update({k: bool(v) for k, v in fl.items()})
        judged = [it for it in items if it.wrong is not None]
        w = [it for it in judged if it.wrong]
        r = [it for it in judged if not it.wrong]
        vrows.append({"analyst": cfg, "n_wrong": len(w), "flag_on_wrong": sum(bool(fl[it.key]) for it in w) / len(w) if w else float("nan"),
                      "n_right": len(r), "flag_on_right": sum(bool(fl[it.key]) for it in r) / len(r) if r else float("nan")})
    write("verifier.csv", vrows)
    prow = []
    for cfg, items in items_by.items():
        configs = {"alone": [], "verifier": [S.verifier_stage(vflags)]}
        mine = {ck: v for ck, v in checkers.items() if any(it.key in v for it in items)}
        for ck, v in mine.items():
            configs[f"checker:{ck}"] = [S.checker_stage(v)]
            configs[f"verifier+checker:{ck}"] = [S.verifier_stage(vflags), S.checker_stage(v)]
        # two checkers in series (either can stop it), and a screen that
        # releases a confident clean no before the full checker: only among
        # the focus checkers, so the table stays readable
        focus = [ck for ck in mine if ck in FOCUS_CHECKERS]
        for i, a in enumerate(focus):
            for b in focus[i + 1:]:
                configs[f"verifier+checker:{a}+checker:{b}"] = [S.verifier_stage(vflags), S.checker_stage(mine[a]), S.checker_stage(mine[b])]
        for a in [ck for ck in mine if ck in SCREENS]:
            for b in focus:
                if a != b:
                    configs[f"verifier+screen:{a}+checker:{b}"] = [S.verifier_stage(vflags), S.screen_then(mine[a], S.checker_stage(mine[b]), 80)]
        for name, stages in configs.items():
            ev = S.evaluate(items, stages, boot=500)
            pt, ci = ev["point"], ev["ci"]
            # cost per task: the analyst on every task, then each checker on
            # the share of tasks that reach it (released by the analyst and
            # passed by the stages before it); the verifier costs nothing
            reach = [not it.declined and it.parse_ok for it in items]
            gs = a_cost.get(cfg)
            if gs is not None:
                for st_name, st in zip(name.split("+"), stages):
                    if st_name == "verifier":
                        reach = [r and st(it) for r, it in zip(reach, items)]
                        continue
                    kind, ck = st_name.split(":", 1)
                    if kind == "screen":
                        # the screen runs on everything that reaches it; the
                        # full checker only on what the screen does not clear
                        scr = mine[ck]
                        gs += cost.get(ck, float("nan")) * sum(reach) / len(items)
                        reach = [r and not (it.key in scr and scr[it.key].clean_no and scr[it.key].confidence >= 80)
                                 for r, it in zip(reach, items)]
                        continue
                    gs += cost.get(ck, float("nan")) * sum(reach) / len(items)
                    reach = [r and st(it) if kind == "checker" and len(stages) > 1 else r for r, it in zip(reach, items)]
            prow.append({"analyst": cfg, "pipeline": name, "gpu_s_per_task": gs,
                         "wrong_release_seeded": pt["wrong_release_seeded"],
                         "lo": ci["wrong_release_seeded"][0], "hi": ci["wrong_release_seeded"][1],
                         "answer_control": pt["answer_control"], "coverage_all": pt["coverage_all"],
                         "withheld_all": pt["withheld_all"], "p_wrong_given_released": pt["p_wrong_given_released"],
                         "gate": S.gate(pt)})
    write("pipelines.csv", prow)
    # paired contrasts: each configuration against the same model's open arm
    # with thinking off, on the same samples; and the verifier's effect
    crow = []
    for cfg, items in items_by.items():
        p_ = cfg.split("|")
        if p_[-1] != "v2":
            continue
        base = "|".join([p_[0], "open", "default" if p_[2].startswith("effort") else "off"]
                        + (["heldout"] if "heldout" in p_ else []) + ["v2"])
        if p_[2].startswith("effort"):
            base = "|".join([p_[0], "open", "effort-low"] + (["heldout"] if "heldout" in p_ else []) + ["v2"])
        for metric in ("wrong_release_seeded", "coverage_all"):
            if base in items_by and base != cfg:
                c = S.paired_contrast(items_by[base], items, metric=metric, boot=1000)
                crow.append({"model": p_[0], "contrast": f"{cfg} - {base}", "metric": metric, **c})
            c = S.paired_contrast(items, items, stages_b=[S.verifier_stage(vflags)], metric=metric, boot=1000)
            crow.append({"model": p_[0], "contrast": f"{cfg} + verifier - alone", "metric": metric, **c})
    write("contrasts.csv", crow)
    write("judges.csv", judge_rows(S.glob_logs(str(ROOT / "runs/inspect/scored-sizing/*_harvest*.eval"))))


if __name__ == "__main__":
    main()
