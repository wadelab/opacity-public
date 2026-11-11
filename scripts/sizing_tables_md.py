"""Markdown tables for docs/model-sizing-study.md from runs/summary/sizing/*.csv.

    .venv/bin/python scripts/sizing_tables_md.py [analyst|checker|natural|pipelines|contrasts|judges ...]

Prints each requested table to stdout, so the report's numbers are pasted
from the CSVs rather than retyped. Default: every table.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "runs" / "summary" / "sizing"

NAME = {
    "Qwen3.5-0.8B": "Qwen3.5 0.8B", "Qwen3.5-2B": "Qwen3.5 2B", "Qwen3.5-4B": "Qwen3.5 4B", "Qwen3.5-9B": "Qwen3.5 9B",
    "Qwen3.5-27B": "Qwen3.5 27B", "Qwen3.5-35B-A3B": "Qwen3.5 35B-A3B", "Qwen3.5-122B-A10B-NVFP4": "Qwen3.5 122B-A10B",
    "Qwen3-32B": "Qwen3 32B", "Mistral-Small-3.2-24B-Instruct-2506-NVFP4": "Mistral Small 24B",
    "Llama-3.3-70B-Instruct-NVFP4": "Llama 3.3 70B", "gpt-oss-20b": "gpt-oss 20B", "gpt-oss-120b": "gpt-oss 120B",
    "typed": "typed (openjev)",
}
ORDER = list(NAME)


def rows(name: str) -> list[dict]:
    p = S / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def f(x: str, d: int = 2) -> str:
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "–"


def model_of(cfg: str) -> str:
    return cfg.split("|")[0]


def order(m: str) -> int:
    return ORDER.index(m) if m in ORDER else len(ORDER)


def analyst(task_set: str = "dev") -> None:
    rs = [r for r in rows("analysts.csv") if r["config"].endswith("|v2") and (("heldout" in r["config"]) == (task_set == "heldout"))]
    labels = {"open|off": "open", "open|default": "open", "open|effort-low": "open, low effort",
              "closed|off": "closed", "closed|default": "closed", "closed|effort-low": "closed, low effort",
              "selfcheck|off": "self-check", "selfcheck|default": "self-check", "selfcheck|effort-low": "self-check, low effort",
              "open|off|checklist": "open + checklist", "closed|off|checklist": "closed + checklist",
              "open|default|checklist": "open + checklist", "closed|default|checklist": "closed + checklist",
              "open|think1024": "open, think 1k", "closed|think1024": "closed, think 1k",
              "open|think4096": "open, think 4k", "closed|think4096": "closed, think 4k",
              "open|effort-medium": "open, medium effort", "closed|effort-medium": "closed, medium effort"}
    print("| Analyst | Configuration | Wrong released, seeded (95% CI) | Answered, controls | Correct released, all | GPU-s per task |")
    print("|---|---|---|---|---|---|")
    for r in sorted(rs, key=lambda r: (order(model_of(r["config"])), r["config"])):
        key = "|".join(r["config"].split("|")[1:-1]).replace("|heldout", "")
        print(f"| {NAME.get(model_of(r['config']), model_of(r['config']))} | {labels.get(key, key)} | "
              f"{f(r['wrong_release_seeded'])} ({f(r['lo'])}–{f(r['hi'])}) | {f(r['answer_control'])} | "
              f"{f(r['coverage_all'])} | {f(r['gpu_s_per_task'], 1)} |")


def checker() -> None:
    rs = [r for r in rows("checkers.csv") if "sizing" in r["tags"] and r["register"] == "default"]
    by = defaultdict(dict)
    for r in rs:
        by[(r["config"], r["log"])][r["level"]] = r
    print("| Checker | Seed, reps | s0 detection | s0 false alarm | Lowest class at s0 | Class 6 at s0 | Gate | AUC s0 | s3 detection / FA | GPU-s per verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (cfg, log), lv in sorted(by.items(), key=lambda kv: (order(model_of(kv[0][0])), kv[0][0])):
        if "0" not in lv:
            continue
        a, c = lv["0"], lv.get("3", {})
        seed = "1" if "seed1" in a.get("tags", "") or int(float(a["n_seeded"])) > 40 else "0"
        print(f"| {NAME.get(model_of(cfg), model_of(cfg))} | {cfg.split('|', 1)[1] if '|' in cfg else ''} {seed}, "
              f"{int(float(a['n_seeded']) // 8)} | {f(a['detection'])} | {f(a['false_alarm'])} | {f(a['min_class_det'])} | "
              f"{f(a['det_c6'])} | {'pass' if a['gate'] == 'True' else 'fail'} | {f(a['auc'])} | "
              f"{f(c.get('detection'))} / {f(c.get('false_alarm'))} | {f(a['gpu_s_per_verdict'])} |")


def natural(pack: str = "v2") -> None:
    P = defaultdict(lambda: [0.0, 0, 0.0, 0, 0.0, 0])
    for r in rows("matrix.csv"):
        if not r["analyst"].endswith(f"|{pack}"):
            continue
        k = r["checker"]
        nw, nr = int(float(r["n_wrong"])), int(float(r["n_right"]))
        P[k][0] += float(r["withhold_on_wrong"] or 0) * nw
        P[k][1] += nw
        P[k][2] += float(r["withhold_on_right"] or 0) * nr
        P[k][3] += nr
        P[k][4] += float(r["sensitivity"] or 0) * nw
    print("| Checker | Configuration | Withheld, judge-wrong | Withheld, judge-right | Flagged yes, judge-wrong | Write-ups |")
    print("|---|---|---|---|---|---|")
    for k, v in sorted(P.items(), key=lambda kv: (order(model_of(kv[0])), kv[0])):
        if not v[1] or not v[3]:
            continue
        print(f"| {NAME.get(model_of(k), model_of(k))} | {k.split('|', 1)[1] if '|' in k else ''} | {v[0] / v[1]:.2f} | "
              f"{v[2] / v[3]:.2f} | {v[4] / v[1]:.2f} | {v[1] + v[3]} |")


def pipelines(top: int = 25) -> None:
    rs = [r for r in rows("pipelines.csv") if r["analyst"].endswith("|v2")]
    rs = sorted(rs, key=lambda r: (float(r["wrong_release_seeded"]), -float(r["coverage_all"])))
    print("| Analyst | Checks | Wrong released, seeded (95% CI) | Answered, controls | Correct released, all | GPU-s per task |")
    print("|---|---|---|---|---|---|")
    for r in [r for r in rs if float(r["answer_control"] or 0) >= 0.75][:top]:
        print(f"| {r['analyst'].replace('|v2', '')} | {r['pipeline']} | {f(r['wrong_release_seeded'])} ({f(r['lo'])}–{f(r['hi'])}) | "
              f"{f(r['answer_control'])} | {f(r['coverage_all'])} | {f(r.get('gpu_s_per_task'), 1)} |")


def contrasts() -> None:
    print("| Contrast | Metric | Difference (95% CI) | Samples |")
    print("|---|---|---|---|")
    for r in rows("contrasts.csv"):
        print(f"| {r['contrast']} | {r['metric']} | {float(r['diff']):+.3f} ({float(r['lo']):+.3f} to {float(r['hi']):+.3f}) | {r['n']} |")


def judges() -> None:
    print("| Analyst | Write-ups both judged | Agreement | Kappa | Wrong, MiMo | Wrong, DeepSeek |")
    print("|---|---|---|---|---|---|")
    for r in rows("judges.csv"):
        print(f"| {r['analyst']} | {r['n_both']} | {f(r['agree'])} | {f(r['kappa'])} | {f(r['wrong_mimo'])} | {f(r['wrong_deepseek'])} |")


TABLES = {"analyst": analyst, "heldout": lambda: analyst("heldout"), "checker": checker, "natural": natural,
          "natural_v1": lambda: natural("v1"), "pipelines": pipelines, "contrasts": contrasts, "judges": judges}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(TABLES):
        print(f"\n<!-- {name} -->")
        TABLES[name]()
