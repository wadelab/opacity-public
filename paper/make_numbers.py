#!/usr/bin/env python
"""Write paper/numbers.tex and the two papers' models.tex from the repository's own record.

paper-vetting/ (the second OPACITY paper) links to paper/numbers.tex, so both
papers quote one set of numbers; each has its own models table.

(Named make_numbers.py because a paper/numbers.py would shadow Python's standard
`numbers` module for every script run from this directory.)

Every number the manuscript quotes is a macro \\N{key} defined here, from:

  src/opacity/harvest/*.yaml            task counts
  runs/summary/checker_by_level.csv     corpus checkers by level (results document)
  runs/summary/auc_by_level.csv         their AUCs
  runs/summary/cost_by_checker.csv      cost of a verdict
  runs/summary/harvest_by_model.csv     the first harvest (plain instruction)
  runs/summary/sizing/*.csv             the model-sizing study (scripts/sizing_report.py,
                                        fresh_results.py, fresh2_results.py, auc_summary.py)
  runs/probe/*/results/*.json           probes, steering, style (opacity.probe)
  runs/vetting/*/slant*/                the slant detector (opacity.slant)
  runs/chimp/*/rep*/summary.json        the end-to-end runs in the gateway
  paper/data/model-revisions.txt        pinned model revisions

A number that exists only in a study report is in FROM_REPORTS below with the
section it comes from, and numbers.tex marks it as such. tests/test_paper_numbers.py
regenerates the file and fails on drift.

    .venv/bin/python paper/make_numbers.py
"""

from __future__ import annotations

import csv
import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SUM = ROOT / "runs" / "summary"
SZ = SUM / "sizing"
sys.path.insert(0, str(ROOT / "src"))

N: dict[str, str] = {}
SOURCE: dict[str, str] = {}


def put(key: str, value, source: str = "computed") -> None:
    if key in N:
        raise KeyError(f"duplicate key {key}")
    N[key] = str(value)
    SOURCE[key] = source


def f2(x) -> str:
    return f"{float(x):.2f}"


def f3(x) -> str:
    return f"{float(x):.3f}"


def pct0(x) -> str:
    return f"{100 * float(x):.0f}\\%"


def pct1(x) -> str:
    return f"{100 * float(x):.1f}\\%"


def pts(x) -> str:
    """A difference in percentage points, signed, no decimals."""
    v = 100 * float(x)
    return f"{v:+.0f}" if abs(v) >= 0.5 else "0"


def thousands(n) -> str:
    return f"{int(round(float(n))):,}"


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def rows(path: Path) -> list[dict]:
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d = {}
            for k, v in r.items():
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
                    d[k] = v
            out.append(d)
    return out


def one(rs: list[dict], what: str, **eq) -> dict:
    hits = [r for r in rs if all(r.get(k) == v for k, v in eq.items())]
    if len(hits) != 1:
        raise LookupError(f"{what}: {len(hits)} rows match {eq}")
    return hits[0]


# ---------------------------------------------------------------- constants
put("corpus.classes", 8, "src/opacity/items")
put("corpus.replicates", 5, "protocol")
put("corpus.seeded.per.level", 40)
put("corpus.control.per.level", 40)
put("corpus.bundles.all.levels", 400)
put("gate.det", "0.8", "docs/protocol.md section 7")
put("gate.fa", "0.2", "docs/protocol.md section 7")
put("gate.wrong", "0.05", "docs/harvest/README.md")
put("gate.answer", "0.75", "docs/harvest/README.md")
put("gate.control.wrong", "0.10", "docs/model-sizing-study.md 6b")
put("model.deepseek", "DeepSeek V4.1 flash")
put("model.deepseek.short", "DeepSeek flash")
put("model.mimo.pro", "MiMo v2.6 pro")
put("model.mimo.pro.short", "MiMo pro")
put("model.typed", "DiffusionGemma 26B-A4B (NVFP4)")
put("model.gptoss120", "gpt-oss 120B")
put("model.gptoss120.short", "gpt-oss 120B")
put("sw.inspect", "0.3.282", "the workstation environment, 28 October 2026")
put("sw.vllm", "0.32.0", "AGENTS.md, Viking environment")
put("sw.torch", "2.13.0+cu130", "AGENTS.md, Viking environment")
put("sw.transformers", "5.19.0", "AGENTS.md, Viking environment")
put("sizing.min.active", 3)
put("sizing.max.active", 27)
put("probe.margin", "0.05", "docs/vetting.md 4.4")
put("slant.threshold.fa", "5\\%", "docs/vetting.md 3.6")
put("slant.limit", "10\\%", "docs/vetting.md 3.6")
put("slant.auc.required", "0.80", "docs/vetting.md 3.6")

# ---------------------------------------------------------------- harvest task counts
import yaml  # noqa: E402

sets = {"dev": "tasks.yaml", "heldout": "heldout.yaml", "fresh": "fresh.yaml", "fresh2": "fresh2.yaml"}
total = 0
for name, fname in sets.items():
    doc = yaml.safe_load(open(ROOT / "src" / "opacity" / "harvest" / fname))
    tasks = doc["tasks"] if isinstance(doc, dict) and "tasks" in doc else doc
    seeded = sum(1 for t in tasks if t.get("invites"))
    put(f"harvest.{name}.count", len(tasks), fname)
    put(f"harvest.{name}.seeded", seeded, fname)
    put(f"harvest.{name}.controls", len(tasks) - seeded, fname)
    total += len(tasks)
put("harvest.tasks.total", total)
put("harvest.seeded.attempts", 3 * int(N["harvest.dev.seeded"]))
put("harvest.control.attempts", 3 * int(N["harvest.dev.controls"]))
put("harvest.fresh2.control.attempts", 3 * int(N["harvest.fresh2.controls"]))
put("harvest.pooled.attempts", 6 * int(N["harvest.dev.seeded"]))

# ---------------------------------------------------------------- corpus checkers by level
CBL = rows(SUM / "checker_by_level.csv")
AUC = rows(SUM / "auc_by_level.csv")
CHECKERS = {  # key -> (model string, run tag)
    "deepseek": ("deepseek-v4.1-flash", "hosted-new"), "mimo": ("mimo-v2.6-pro", "hosted-new"),
    "q08": ("Qwen3.5-0.8B", "sweep4-new-nothink"), "q2": ("Qwen3.5-2B", "sweep4-new-nothink"),
    "q4": ("Qwen3.5-4B", "sweep4-new-nothink"), "q9": ("Qwen3.5-9B", "sweep4-new-nothink"),
    "q27": ("Qwen3.5-27B", "sweep4-new-nothink"), "q332": ("Qwen3-32B", "sweep5-new-nothink"),
    "mistral": ("Mistral-Small-3.2-24B-Instruct-2506-NVFP4", "sweep5-new-nothink"),
    "llama": ("Llama-3.3-70B-Instruct-NVFP4", "sweep5-new-nothink"), "typed": ("openjev:openjev-latest", "typed-new"),
}
for key, (model, run) in CHECKERS.items():
    for level in range(5):
        hits = [r for r in CBL if r["model"] == model and r["run"] == run and r["level"] == level]
        if not hits:
            continue
        r = hits[0]
        n = r["n_seeded"]
        put(f"det.{key}.s{level}", f2(r["detection"]))
        put(f"fa.{key}.s{level}", f2(r["false_alarm"]))
        put(f"ct.seeded.{key}.s{level}", f2(r["cannot_tell_seeded"]))
        put(f"ct.control.{key}.s{level}", f2(r["cannot_tell_control"]))
        for c in range(1, 9):
            put(f"detcls.{key}.c{c}.s{level}", f2(r[f"det_c{c}"]))
        if level == 0:
            lo, hi = wilson(r["detection"] * n, n)
            put(f"det.{key}.s0.lo", f2(lo)); put(f"det.{key}.s0.hi", f2(hi))
            lo, hi = wilson(r["false_alarm"] * r["n_control"], r["n_control"])
            put(f"fa.{key}.s0.lo", f2(lo)); put(f"fa.{key}.s0.hi", f2(hi))
        a = [r for r in AUC if r["model"] == model and r["run"] == run and r["level"] == level]
        if a:
            put(f"auc.{key}.s{level}", f2(a[0]["auc"]))

# twenty replicates and the register ablation
for key, model in (("q27", "Qwen3.5-27B"), ("mistral", "Mistral-Small-3.2-24B-Instruct-2506-NVFP4"),
                   ("llama", "Llama-3.3-70B-Instruct-NVFP4")):
    for level in (0, 3, 4):
        r = one(CBL, "rep20", model=model, run="rep20-new-nothink", level=level)
        put(f"rep20.{key}.det.s{level}", f2(r["detection"]))
        if level == 0:
            lo, hi = wilson(r["detection"] * r["n_seeded"], r["n_seeded"])
            put("rep20.%s.det.s0.lo" % key, f2(lo)); put("rep20.%s.det.s0.hi" % key, f2(hi))
            put(f"rep20.{key}.fa.s0", f2(r["false_alarm"]))
            lo, hi = wilson(r["false_alarm"] * r["n_control"], r["n_control"])
            put("rep20.%s.fa.s0.lo" % key, f2(lo)); put("rep20.%s.fa.s0.hi" % key, f2(hi))
            put(f"rep20.{key}.c6.s0", f2(r["det_c6"]))
    for level in (0, 3):
        a = one(AUC, "rep20 auc", model=model, run="rep20-new-nothink", level=level)
        put(f"rep20.{key}.auc.s{level}", f2(a["auc"]))
drops = []
for key, model in (("q27", "Qwen3.5-27B"), ("mistral", "Mistral-Small-3.2-24B-Instruct-2506-NVFP4")):
    m = one(CBL, "matched", model=model, run="matched-new-nothink", level=0)
    put(f"matched.{key}.fa.default", N[f"fa.{key}.s0"])
    put(f"matched.{key}.fa.matched", f2(m["false_alarm"]))
    a_default = one(AUC, "auc default", model=model, run=CHECKERS[key][1], level=0)["auc"]
    a_matched = one(AUC, "auc matched", model=model, run="matched-new-nothink", level=0)["auc"]
    put(f"matched.{key}.auc.default", f2(a_default)); put(f"matched.{key}.auc.matched", f2(a_matched))
    drops.append(a_default - a_matched)
put("matched.auc.drop.min", f2(min(drops))); put("matched.auc.drop.max", f2(max(drops)))

# cost of a verdict
for r in rows(SUM / "cost_by_checker.csv"):
    key = {"MiMo pro": "mimo", "DeepSeek flash": "deepseek", "Qwen3.5 27B": "q27", "Mistral 24B": "mistral",
           "Llama 70B": "llama"}.get(r["checker"])
    if key:
        put(f"cost.{key}.median", f"{r['median_seconds']:.1f}")
        put(f"cost.{key}.p90", f"{r['p90_seconds']:.1f}")
        put(f"cost.{key}.tokens", thousands(r["median_output_tokens"]))

# ---------------------------------------------------------------- the first harvest
HBM = rows(SUM / "harvest_by_model.csv")
for key, model in (("q08", "Qwen3.5-0.8B"), ("q2", "Qwen3.5-2B"), ("q4", "Qwen3.5-4B"), ("q9", "Qwen3.5-9B"),
                   ("q27", "Qwen3.5-27B"), ("q332", "Qwen3-32B"),
                   ("mistral", "Mistral-Small-3.2-24B-Instruct-2506-NVFP4"), ("llama", "Llama-3.3-70B-Instruct-NVFP4")):
    r = one(HBM, "harvest", model=model)
    put(f"harvest.{key}.wrong", f2(r["conclusion_wrong_seeded"]))
    put(f"harvest.{key}.c1", f2(r["commit_c1"]))

# ---------------------------------------------------------------- the sizing study: analysts
AN = rows(SZ / "analysts.csv")
MODELS = {"q9": "Qwen3.5-9B", "q27": "Qwen3.5-27B", "q35": "Qwen3.5-35B-A3B", "q122": "Qwen3.5-122B-A10B-NVFP4",
          "g26": "gemma-4-26B-A4B-it", "g31": "gemma-4-31B-it", "nsuper": "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
          "nlight": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "gptoss120": "gpt-oss-120b"}
ARMS = {  # arm key -> (think token, prompt token or None, extra tokens)
    "open": ("off", None, ()), "checklist": ("off", "checklist", ()), "rules": ("off", "rules", ()),
    "rules1k": ("think1024", "rules", ()),
}


def analyst_row(model: str, think: str, prompt: str | None, heldout: bool, extra=()) -> dict | None:
    want = {model, "open", think, "v2", *extra}
    if prompt:
        want.add(prompt)
    if heldout:
        want.add("heldout")
    hits = [r for r in AN if set(r["config"].split("|")) == want and r["config"].split("|")[1] == "open"]
    if len(hits) > 1:
        raise LookupError(f"several analyst rows for {want}")
    return hits[0] if hits else None


seeded = int(N["harvest.dev.seeded"]) * 3
pooled_count = 0
for mk, model in MODELS.items():
    if mk == "gptoss120":
        continue  # its high-effort runs are summarised in fresh_results.csv, below
    for ak, (think, prompt, extra) in ARMS.items():
        ks = {}
        for setname, heldout in (("dev", False), ("heldout", True)):
            r = analyst_row(model, think, prompt, heldout, extra)
            put(f"prompt.{mk}.{ak}.{setname}", f2(r["wrong_release_seeded"]) if r else "--")
            if r:
                ks[setname] = round(r["wrong_release_seeded"] * seeded)
                put(f"prompt.{mk}.{ak}.{setname}.lo", f2(r["lo"])); put(f"prompt.{mk}.{ak}.{setname}.hi", f2(r["hi"]))
        if len(ks) == 2:
            k, n = ks["dev"] + ks["heldout"], 2 * seeded
            lo, hi = wilson(k, n)
            put(f"prompt.{mk}.{ak}.pooled", f2(k / n)); put(f"prompt.{mk}.{ak}.pooled.k", k)
            put(f"prompt.{mk}.{ak}.pooled.lo", f2(lo)); put(f"prompt.{mk}.{ak}.pooled.hi", f2(hi))
put("sizing.open.27b.dev", N["prompt.q27.open.dev"])
put("sizing.pooled.rules1k", N["prompt.q27.rules1k.pooled"])
put("harvest.models.count", len({r["model"] for r in AN} | {r["model"] for r in HBM}))

# ---------------------------------------------------------------- the sizing study: checkers
CK = rows(SZ / "checkers.csv")


def ck(config: str, n_seeded: float, register: str = "default", tag: str | None = None, level: float = 0.0) -> list[dict]:
    hits = [r for r in CK if r["config"] == config and r["n_seeded"] == n_seeded and r["register"] == register
            and r["level"] == level and (tag is None or tag in str(r["tags"]))]
    return hits


for mk, model in (("q4", "Qwen3.5-4B"), ("q9", "Qwen3.5-9B"), ("q27", "Qwen3.5-27B"), ("q35", "Qwen3.5-35B-A3B")):
    r = ck(f"{model}|off", 40.0, tag="sizing")[0]
    put(f"thinking.c6.off.{mk}", int(round(r["det_c6"] * 5)))
r = ck("Qwen3.5-27B|think1024", 40.0, tag="sizing")[0]
put("thinking.q27.1k.fa", f2(r["false_alarm"]))
put("auc.q27.1k.planted", f2(r["auc"]))
r = ck("gpt-oss-20b|effort-medium", 40.0, tag="sizing")[0]
put("auc.gptoss.planted", f2(r["auc"]))
r = ck("gpt-oss-20b|effort-medium", 160.0, tag="confirm-gptoss")[0]
put("seed1.gptoss.det", f2(r["detection"])); put("seed1.gptoss.fa", f2(r["false_alarm"]))
put("seed1.gptoss.fa.lo", f2(r["fa_lo"])); put("seed1.gptoss.fa.hi", f2(r["fa_hi"]))
put("seed1.gptoss.minclass", f2(r["min_class_det"]))
r = ck("gpt-oss-20b|effort-medium", 40.0, register="matched")[0]
put("seed1.gptoss.matched.fa", f2(r["false_alarm"]))
q27seed1 = [r for r in ck("Qwen3.5-27B|think1024", 160.0, tag="confirm-checker") if "rest" not in str(r["tags"])]
q27seed1.sort(key=lambda r: -r["false_alarm"])
put("seed1.q27.fa", f2(q27seed1[0]["false_alarm"])); put("seed1.q27.fa.lo", f2(q27seed1[0]["fa_lo"]))
put("seed1.q27.fa.hi", f2(q27seed1[0]["fa_hi"])); put("seed1.q27.fa.rerun", f2(q27seed1[1]["false_alarm"]))
r = ck("Qwen3.5-27B|think1024", 160.0, tag="confirm-checker-rest")[0]
put("seed2.q27.fa", f2(r["false_alarm"])); put("seed2.q27.det", f2(r["detection"]))
r = ck("Qwen3.5-27B|off|stepwise", 160.0, tag="confirm-checker-rest")[0]
put("seed1.q27step.fa", f2(r["false_alarm"])); put("seed1.q27step.det", f2(r["detection"]))
put("seed1.q27step.fa.lo", f2(r["fa_lo"])); put("seed1.q27step.fa.hi", f2(r["fa_hi"]))
put("seed1.q27step.minclass", f2(r["min_class_det"]))
sizing_checker_configs = {r["config"] for r in CK if "sizing" in str(r["tags"]) and r["level"] == 0.0}
put("sizing.configs.count", len(AN) + len(sizing_checker_configs))

# ---------------------------------------------------------------- pipelines on dev, and the held-out test
PI = rows(SZ / "pipelines.csv")


def dev_v2(cfg: str) -> bool:
    p = cfg.split("|")
    return p[-1] == "v2" and "heldout" not in p


dev_pipes = [r for r in PI if dev_v2(r["analyst"])]
answering = [r for r in dev_pipes if r["answer_control"] >= 0.75]
put("pipelines.count", thousands(len(dev_pipes)))
put("pipelines.answering", thousands(len(answering)))
put("pipelines.under", len([r for r in answering if r["wrong_release_seeded"] <= 0.05]))
put("pipelines.clear", len([r for r in answering if r["wrong_release_seeded"] <= 0.05 and r["hi"] <= 0.05]))
HS = rows(SZ / "heldout_selection.csv")
for i, (mk, pipe_prefix) in enumerate((("Qwen3.5-35B-A3B", "verifier+checker:typed|think512"),
                                        ("Qwen3.5-27B", "verifier+checker:Llama"),
                                        ("Qwen3.5-9B", "verifier+checker:typed|think512")), start=1):
    chosen = one(HS, "heldout chosen", model=mk, role="chosen")
    put(f"heldout.pipe{i}.dev", f2(chosen["dev_wrong"]))
    ho_analyst = chosen["analyst"].replace("|v2", "|heldout|v2")
    r = [r for r in PI if r["analyst"] == ho_analyst and r["pipeline"] == chosen["pipeline"]][0]
    put(f"heldout.pipe{i}.ho", f2(r["wrong_release_seeded"]))
    put(f"heldout.pipe{i}.ho.lo", f2(r["lo"])); put(f"heldout.pipe{i}.ho.hi", f2(r["hi"]))

# ---------------------------------------------------------------- the sealed sets
FR = rows(SZ / "fresh_results.csv")
FRESH = {"q122": "Qwen3.5-122B-A10B-NVFP4", "q27": "Qwen3.5-27B", "q35": "Qwen3.5-35B-A3B", "g31": "gemma-4-31B-it",
         "nsuper": "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "g26": "gemma-4-26B-A4B-it", "q9": "Qwen3.5-9B",
         "nlight": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "gptoss120": "gpt-oss-120b"}
put("fresh.n.seeded", int(FR[0]["n_seeded"])); put("fresh.n.control", int(FR[0]["n_control"]))
for mk, model in FRESH.items():
    r = one(FR, "fresh", model=model)
    ns = r["n_seeded"]
    put(f"fresh.{mk}.alone.wrong", f2(r["fresh_wrong"])); put(f"fresh.{mk}.alone.k", int(round(r["fresh_wrong"] * ns)))
    put(f"fresh.{mk}.alone.lo", f2(r["fresh_lo"])); put(f"fresh.{mk}.alone.hi", f2(r["fresh_hi"]))
    put(f"fresh.{mk}.alone.correct", f2(r["fresh_correct_released"]))
    put(f"fresh.{mk}.alone.controls", f2(r["fresh_controls_answered"]))
    put(f"fresh.{mk}.alone.alt", f2(r["alt_fresh_wrong"])); put(f"fresh.{mk}.alone.ds", f2(r["ds_fresh_wrong"]))
    if r["checked_wrong"] != "":
        put(f"fresh.{mk}.checked.wrong", f2(r["checked_wrong"])); put(f"fresh.{mk}.checked.k", int(round(r["checked_wrong"] * ns)))
        put(f"fresh.{mk}.checked.lo", f2(r["checked_lo"])); put(f"fresh.{mk}.checked.hi", f2(r["checked_hi"]))
        put(f"fresh.{mk}.checked.controls", f2(r["checked_controls_answered"]))
        put(f"fresh.{mk}.checked.correct", f2(r["checked_correct_released"]))
        put(f"fresh.{mk}.checked.alt", f2(r["alt_checked_wrong"])); put(f"fresh.{mk}.checked.ds", f2(r["ds_checked_wrong"]))
    put(f"fresh.{mk}.confirms", "yes" if str(r["confirms"]) == "True" else "no")

F2 = rows(SZ / "fresh2_results.csv")
put("fresh2.n.seeded", int(one(F2, "fresh2", model="Qwen3.5-27B", arm="open|think1024|rules", checker="none", judge="alt")["n_seeded"]))
put("fresh2.n.control", int(one(F2, "fresh2", model="Qwen3.5-27B", arm="open|think1024|rules", checker="none", judge="alt")["n_control"]))
for mk, model in (("q27", "Qwen3.5-27B"), ("q35", "Qwen3.5-35B-A3B")):
    for ck_key, checker in (("alone", "none"), ("checked", "gpt-oss-20b|effort-medium")):
        for jk, judge in (("", "alt"), (".ds", "second")):
            r = one(F2, "fresh2", model=model, arm="open|think1024|rules", checker=checker, judge=judge)
            base = f"fresh2.{mk}.{ck_key}{jk}"
            put(f"{base}.k", int(round(r["wrong"] * r["n_seeded"]))); put(f"{base}.n", int(r["n_seeded"]))
            put(f"{base}.wrong", f2(r["wrong"])); put(f"{base}.lo", f2(r["lo"])); put(f"{base}.hi", f2(r["hi"]))
            put(f"{base}.controls", f2(r["controls_answered"]))
            put(f"{base}.cwrong", f"{int(round(r['controls_wrong'] * r['n_control']))} of {int(r['n_control'])}")
            put(f"{base}.correct", f2(r["correct_released"]))
    r = one(F2, "fresh2 open", model=model, arm="open|off", checker="none", judge="alt")
    put(f"fresh2.{mk}.open.wrong", f2(r["wrong"]))
r = one(F2, "fresh2 9B", model="Qwen3.5-9B", arm="open|think1024|rules", checker="none", judge="alt")
put("fresh2.q9.rules.wrong", f2(r["wrong"])); put("fresh2.q9.rules.cwrong", f2(r["controls_wrong"]))
put("fresh2.27b.checked.k", N["fresh2.q27.checked.k"])
# gpt-oss 120B at high effort with the rules prompt: dev and held-out rates recorded in fresh_results.csv
g = one(FR, "fresh gpt-oss 120B", model="gpt-oss-120b")
put("prompt.gptoss120.rules.dev", f2(g["dev_wrong"])); put("prompt.gptoss120.rules.heldout", f2(g["heldout_wrong"]))
k = round(g["dev_wrong"] * seeded) + round(g["heldout_wrong"] * seeded)
lo, hi = wilson(k, 2 * seeded)
put("prompt.gptoss120.rules.pooled", f2(k / (2 * seeded))); put("prompt.gptoss120.rules.pooled.lo", f2(lo)); put("prompt.gptoss120.rules.pooled.hi", f2(hi))

# ---------------------------------------------------------------- judges
JU = rows(SZ / "judges.csv")
pooled = one(JU, "judges pooled", analyst="pooled")
put("judges.pooled.n", thousands(pooled["n_both"])); put("judges.pooled.agree", f2(pooled["agree"]))
put("judges.pooled.kappa", f2(pooled["kappa"]))

# ---------------------------------------------------------------- checkers on real write-ups
AC = rows(SZ / "auc" / "checkers.csv")
XC = {"q27off": "Qwen3.5-27B|off", "q27think": "Qwen3.5-27B|think1024", "q27pack": "Qwen3.5-27B|think1024|pack",
      "q122think": "Qwen3.5-122B-A10B-NVFP4|think1024", "gptoss": "gpt-oss-20b|effort-medium",
      "mistral": "Mistral-Small-3.2-24B-Instruct-2506-NVFP4|default",
      "llamastep": "Llama-3.3-70B-Instruct-NVFP4|default|stepwise", "typed0": "typed|think0", "typed512": "typed|think512"}
AUCKEY = {"q27off": "auc.q27.off", "q27think": "auc.q27.1k", "q27pack": "auc.q27.1k.pack", "q122think": "auc.q122.1k",
          "gptoss": "auc.gptoss.medium", "mistral": "auc.mistral.default", "llamastep": "auc.llama.stepwise",
          "typed0": "auc.typed0", "typed512": "auc.typed512"}
for key, cfg in XC.items():
    r = one(AC, "auc checkers", checker=cfg)
    put(f"xc.{key}.wrong", f2(r["common_withhold_wrong"])); put(f"xc.{key}.right", f2(r["common_withhold_right"]))
    put(AUCKEY[key], f2(r["common_auc"])); put(AUCKEY[key] + ".lo", f2(r["common_lo"])); put(AUCKEY[key] + ".hi", f2(r["common_hi"]))
put("auc.min", f2(min(r["common_auc"] for r in AC))); put("auc.max", f2(max(r["common_auc"] for r in AC)))
put("auc.common.n", thousands(AC[0]["common_n_wrong"] + AC[0]["common_n_right"]))
put("auc.common.wrong", thousands(AC[0]["common_n_wrong"])); put("auc.checkers.count", len(AC))

MX = rows(SZ / "matrix.csv")
for key, cfg in (("q27", "Qwen3.5-27B|off"), ("q27think", "Qwen3.5-27B|think1024")):
    for self_flag, name in ((1.0, "self"), (0.0, "others")):
        rs = [r for r in MX if r["checker"] == cfg and dev_v2(r["analyst"]) and r["self"] == self_flag and r["n_wrong"]]
        k = sum(r["withhold_on_wrong"] * r["n_wrong"] for r in rs)
        n = sum(r["n_wrong"] for r in rs)
        put(f"xc.{key}.{name}.wrong", f2(k / n))
put("xc.analyst.configs", len({r["analyst"] for r in MX if dev_v2(r["analyst"])}))
FS = rows(SZ / "auc" / "fresh_systems.csv")
put("auc.fresh.q9open", f2(one(FS, "fresh systems", model="Qwen3.5-9B", arm="open|off")["g20_auc"]))
put("auc.fresh.q27rules", f2(one(FS, "fresh systems", model="Qwen3.5-27B", arm="open|think1024|rules")["g20_auc"]))
put("auc.fresh.q122rules", f2(one(FS, "fresh systems", model="Qwen3.5-122B-A10B-NVFP4", arm="open|think1024|rules")["g20_auc"]))

# ---------------------------------------------------------------- end to end in the gateway
for mk, model in (("q27", "Qwen3.5-27B"), ("q35", "Qwen3.5-35B-A3B")):
    for arm in ("plain", "rules"):
        files = sorted(glob.glob(str(ROOT / "runs" / "chimp" / f"chimp-20261026-1743-{model}-think1024-{arm}" / "rep*" / "summary.json")))
        if not files:
            continue
        ss = [json.load(open(f)) for f in files]
        put(f"chimp.{mk}.{arm}.verdicts", sum(s["verdict_agreement"] for s in ss))
        put(f"chimp.{mk}.{arm}.marks", sum(s["marks_ok"] for s in ss))
        put(f"chimp.{mk}.{arm}.reps", len(ss))

# ---------------------------------------------------------------- probes
for mk, probe in (("q27", "Qwen3.5-27B"), ("g20", "gpt-oss-20b")):
    d = json.load(open(ROOT / "runs" / "probe" / "v1" / "results" / f"{probe}-heldout-2.json"))
    put(f"probe.{mk}.auc", f2(d["probe_heldout"]["auc"])); put(f"probe.{mk}.auc.lo", f2(d["probe_heldout"]["lo"]))
    put(f"probe.{mk}.auc.hi", f2(d["probe_heldout"]["hi"]))
    put(f"probe.{mk}.verbal", f2(d["verbal_heldout"][d["best_verbal"]]["auc"]))
    put(f"probe.{mk}.diff", f2(d["difference"]["auc"])); put(f"probe.{mk}.diff.lo", f2(d["difference"]["lo"]))
    put(f"probe.{mk}.diff.hi", f2(d["difference"]["hi"]))
    put(f"probe.{mk}.shuffled", f2(d["shuffled_heldout"]["mean"])); put(f"probe.{mk}.length", f2(d["length_heldout"]["auc"]))
    w = d["within_task"]
    put(f"probe.{mk}.within", f2(w["probe"])); put(f"probe.{mk}.within.verbal", f2(w["verbal"][d["best_verbal"]]))
    put(f"probe.{mk}.within.diff", f2(w["difference_vs_best_verbal"]["auc"]))
    put(f"probe.{mk}.within.diff.lo", f2(w["difference_vs_best_verbal"]["lo"]))
    put(f"probe.{mk}.within.diff.hi", f2(w["difference_vs_best_verbal"]["hi"]))
    put(f"probe.{mk}.within.shuffled", f2(w["shuffled_mean"]))
    if mk == "q27":
        put("probe.train.n", thousands(d["n_dev"])); put("probe.train.wrong", thousands(d["n_dev_wrong"]))
        put("probe.test.n", thousands(d["n_heldout"])); put("probe.test.wrong", thousands(d["n_heldout_wrong"]))
        put("probe.test.tasks", int(d["heldout_tasks"])); put("probe.within.tasks", int(w["tasks"]))

# ---------------------------------------------------------------- the positive control
for vk, tag in (("v1", "steer-v1"), ("v2", "steer-v2")):
    d = json.load(open(ROOT / "runs" / "probe" / tag / "results" / "steer-heldout.json"))
    jw = d["judge_wrong"]
    put(f"steer.{vk}.steered.wrong", sum(jw[s]["steered"][0] for s in jw))
    put(f"steer.{vk}.honest.wrong", sum(jw[s]["honest"][0] for s in jw))
    put(f"steer.{vk}.heldout.steered", jw["heldout"]["steered"][0]); put(f"steer.{vk}.heldout.honest", jw["heldout"]["honest"][0])
    put(f"steer.{vk}.probe", f2(d["probe_within"]["auc"])); put(f"steer.{vk}.probe.lo", f2(d["probe_within"]["lo"]))
    put(f"steer.{vk}.probe.hi", f2(d["probe_within"]["hi"]))
    for dk, name in (("text", "text classifier"), ("gptoss", "gpt-oss-20b|effort-medium"), ("q27", "Qwen3.5-27B|off|pack"),
                     ("verifier", "verifier")):
        b = d["blackbox_within"][name]
        put(f"steer.{vk}.{dk}", f2(b["auc"])); put(f"steer.{vk}.{dk}.lo", f2(b["lo"])); put(f"steer.{vk}.{dk}.hi", f2(b["hi"]))
    put(f"steer.{vk}.probe.minus.text", f2(d["difference"]["auc"]))
    put(f"steer.{vk}.probe.minus.text.lo", f2(d["difference"]["lo"])); put(f"steer.{vk}.probe.minus.text.hi", f2(d["difference"]["hi"]))
    put(f"steer.{vk}.hit", pct0(d["hit_rate"])); put(f"steer.{vk}.fa", pct0(d["false_alarm"]))
    if vk == "v2":
        put("steer.n.perarm", sum(jw[s]["steered"][1] for s in jw)); put("steer.n.heldout", jw["heldout"]["steered"][1])
        put("steer.n.heldout.both", int(d["probe_within"]["n"]))
        put("steer.n.dev", jw["dev"]["steered"][1])
        # the word panel's selection rule, from the script that draws it
        sys.path.insert(0, str(ROOT / "scripts"))
        import probe_figures  # noqa: E402

        put("steer.words.min", probe_figures.WORD_MIN)
        put("steer.words.each", "one two three four five six seven eight nine ten".split()[probe_figures.WORD_EACH - 1])
        hs, hn = jw["heldout"]["steered"][0], jw["heldout"]["honest"][0]
        n = jw["heldout"]["steered"][1]
        put("steer.v2.steered.right", n - hs); put("steer.v2.honest.right", n - hn)
        diff = (hs - hn) / n
        put("steer.v2.diff", f"{100 * diff:.0f}")
# the difference's task-resampled interval is in the slant report of the same twins
sl_off = json.load(open(ROOT / "runs" / "vetting" / "qwen3.5-27b-rules-off" / "slant" / "report.json"))
put("steer.v2.diff.lo", f"{100 * sl_off['manipulation']['heldout_difference']['lo']:.0f}")
put("steer.v2.diff.hi", f"{100 * sl_off['manipulation']['heldout_difference']['hi']:.0f}")

# ---------------------------------------------------------------- the slant detector on the recommended analyst
think = json.load(open(ROOT / "runs" / "vetting" / "qwen3.5-27b-rules-think" / "slant" / "report.json"))
m = think["manipulation"]
put("slant.think.honest", m["heldout_honest"][0]); put("slant.think.steered", m["heldout_steered"][0])
put("slant.think.diff", pts(m["heldout_difference"]["diff"]))
put("slant.think.diff.lo", f"{100 * m['heldout_difference']['lo']:.0f}"); put("slant.think.diff.hi", f"{100 * m['heldout_difference']['hi']:.0f}")
put("slant.think.auc", f2(think["heldout_within"]["auc"])); put("slant.think.hit", pct0(think["hit_rate"]))
put("slant.off.honest", sl_off["manipulation"]["heldout_honest"][0]); put("slant.off.steered", sl_off["manipulation"]["heldout_steered"][0])
three = json.load(open(ROOT / "runs" / "vetting" / "qwen3.5-27b-rules-off" / "slant-3style" / "report.json"))
put("slant.3style.auc", f2(three["heldout_within"]["auc"])); put("slant.3style.auc.lo", f2(three["heldout_within"]["lo"]))
put("slant.3style.auc.hi", f2(three["heldout_within"]["hi"]))
style_aucs = [v["auc"]["auc"] for v in three["by_style"].values()]
put("slant.3style.auc.min", f2(min(style_aucs))); put("slant.3style.auc.max", f2(max(style_aucs)))
for dk, folder in (("own", "qwen3.5-27b-rules-think/slant"), ("1style", "qwen3.5-27b-rules-off/slant-2"),
                   ("3style", "qwen3.5-27b-rules-off/slant-3style")):
    k = n = 0
    for s in ("dev", "heldout", "fresh", "fresh2"):
        a = json.load(open(ROOT / "runs" / "vetting" / folder / "scores" / f"think1024-greedy-{s}.json"))["summary"]["all"]
        k += a["flagged"]; n += a["n"]
        put(f"slant.{dk}.flagged.{s}", a["flagged"])
    lo, hi = wilson(k, n)
    put(f"slant.{dk}.flagged.k", k); put(f"slant.{dk}.flagged.pct", pct1(k / n))
    put(f"slant.{dk}.flagged.lo", pct1(lo)); put(f"slant.{dk}.flagged.hi", pct1(hi))
    if dk == "own":
        put("slant.flagged.n", n)

# ---------------------------------------------------------------- style robustness
st = json.load(open(ROOT / "runs" / "probe" / "style-v1" / "results" / "style-heldout.json"))["conditions"]
c = st["concise"]
put("style.concise.current.flagged", c["honest_flagged"]["current"][0])
put("style.concise.current.pct", pct0(c["honest_flagged"]["current"][0] / c["honest_flagged"]["current"][1]))
put("style.concise.current.auc", f2(c["auc"]["current text detector"]["auc"]))
mixed_auc = [st[s]["auc"]["mixed text detector"]["auc"] for s in ("default", "concise", "thorough")]
put("style.mixed.auc.min", f2(min(mixed_auc)))
mixed_fl = [st[s]["honest_flagged"]["mixed"][0] / st[s]["honest_flagged"]["mixed"][1] for s in ("default", "concise", "thorough")]
put("style.mixed.flagged.min", pct0(min(mixed_fl))); put("style.mixed.flagged.max", pct0(max(mixed_fl)))
put("style.n.new", thousands(sum(1 for r in csv.DictReader(open(ROOT / "runs" / "probe" / "style-v1" / "labels.csv"))
                                   if r["condition"] != "default")))  # the default-style pairs are steer-v2's
for s in ("default", "concise", "thorough", "disguise"):
    put(f"style.{s}.diff", pts(st[s]["manipulation"]["diff"]))
    put(f"style.{s}.steered", st[s]["manipulation"]["steered_wrong"][0]); put(f"style.{s}.honest", st[s]["manipulation"]["honest_wrong"][0])

# ---------------------------------------------------------------- numbers that exist only in the study reports
FROM_REPORTS = [
    # the gateway's planner, second measurement (15 August 2026); the report names neither planner
    ("planner.answerable", 24, "safe-tre-agent docs/planner-eval.md, second measurement table (commit 8f411b1); corpus evals/corpus.yaml"),
    ("planner.unanswerable", 6, "safe-tre-agent docs/planner-eval.md, second measurement table (commit 8f411b1); corpus evals/corpus.yaml"),
    ("planner.demo.primary.omitted", "63--71\\%", "safe-tre-agent docs/planner-eval.md, second measurement table (commit 8f411b1); corpus evals/corpus.yaml"),
    ("planner.demo.accepted.omitted", "79--88\\%", "safe-tre-agent docs/planner-eval.md, second measurement table (commit 8f411b1); corpus evals/corpus.yaml"),
    ("planner.demo.rejected.omitted", "0\\%", "safe-tre-agent docs/planner-eval.md, second measurement table (commit 8f411b1); corpus evals/corpus.yaml"),
    ("planner.demo.primary.stated", "96\\%", "safe-tre-agent docs/planner-eval.md, second measurement table (commit 8f411b1); corpus evals/corpus.yaml"),
    ("planner.demo.accepted.stated", "100\\%", "safe-tre-agent docs/planner-eval.md, second measurement table (commit 8f411b1); corpus evals/corpus.yaml"),
    ("planner.demo.rejected.stated", "0--17\\%", "safe-tre-agent docs/planner-eval.md, second measurement table (commit 8f411b1); corpus evals/corpus.yaml"),
    ("planner.local.primary.omitted", "54--63\\%", "safe-tre-agent docs/planner-eval.md, second measurement table (commit 8f411b1); corpus evals/corpus.yaml"),
    ("planner.local.accepted.omitted", "67--75\\%", "safe-tre-agent docs/planner-eval.md, second measurement table (commit 8f411b1); corpus evals/corpus.yaml"),
    ("planner.local.rejected.omitted", "33--67\\%", "safe-tre-agent docs/planner-eval.md, second measurement table (commit 8f411b1); corpus evals/corpus.yaml"),
    ("planner.local.primary.stated", "92--96\\%", "safe-tre-agent docs/planner-eval.md, second measurement table (commit 8f411b1); corpus evals/corpus.yaml"),
    ("planner.local.accepted.stated", "96\\%", "safe-tre-agent docs/planner-eval.md, second measurement table (commit 8f411b1); corpus evals/corpus.yaml"),
    ("planner.local.rejected.stated", "33--50\\%", "safe-tre-agent docs/planner-eval.md, second measurement table (commit 8f411b1); corpus evals/corpus.yaml"),
    ("register.rule.seeded", 25, "docs/results-2026-10-21.md, the prose floor and Table 10"),
    ("register.rule.control", 0, "docs/results-2026-10-21.md, the prose floor"),
    ("register.matched.control", 20, "docs/results-2026-10-21.md, Table 10"),
    ("judges.dev.agree", "91.5\\%", "docs/model-sizing-study.md section 3"),
    ("judges.dev.n", "5,722", "docs/model-sizing-study.md section 3"),
    ("judges.fresh.n", "1,931", "docs/model-sizing-study.md summary"),
    ("judges.fresh.agree.min", "0.93", "docs/model-sizing-study.md summary"),
    ("judges.fresh.agree.max", "0.94", "docs/model-sizing-study.md summary"),
    ("judges.fresh.kappa.min", "0.78", "docs/model-sizing-study.md summary"),
    ("judges.fresh.kappa.max", "0.81", "docs/model-sizing-study.md summary"),
    ("judges.fresh.mimo.wrong", 324, "docs/model-sizing-study.md summary"),
    ("judges.fresh.others.wrong", 368, "docs/model-sizing-study.md summary"),
    ("judges.fresh2.agree", "0.92", "docs/model-sizing-study.md 6b"),
    ("judges.fresh2.kappa", "0.75", "docs/model-sizing-study.md 6b"),
    ("judges.fresh2.control.agree", "0.90", "docs/model-sizing-study.md 6b"),
    ("prompt.gptoss120.open.dev", "0.22", "docs/model-sizing-study.md, were the large models run fairly"),
    ("prompt.gptoss120.open.heldout", "0.34", "docs/model-sizing-study.md, were the large models run fairly"),
    ("frontier.glm.dev", "0.07", "docs/model-sizing-study.md, first checks of the prompt result"),
    ("verifier.dev.wrong", "0.31", "docs/model-sizing-study.md, first held-out results"),
    ("verifier.dev.right", "0.04", "docs/model-sizing-study.md, first held-out results"),
    ("verifier.ho.wrong", "0.19", "docs/model-sizing-study.md, first held-out results"),
    ("verifier.ho.right", "0.15", "docs/model-sizing-study.md, first held-out results"),
    ("calc.hand.local", 44, "docs/model-sizing-study.md 6d"), ("calc.hand.mimo", 22, "docs/model-sizing-study.md 6d"),
    ("calc.v1.local", 50, "docs/model-sizing-study.md 6d"), ("calc.v1.mimo", 32, "docs/model-sizing-study.md 6d"),
    ("calc.v2.local", 56, "docs/model-sizing-study.md 6d"), ("calc.v2.mimo", 29, "docs/model-sizing-study.md 6d"),
    ("calc.n", 383, "docs/model-sizing-study.md 6d"),
    ("prep.plain.ho.before", 128, "docs/model-sizing-study.md 6d"), ("prep.plain.ho.after", 75, "docs/model-sizing-study.md 6d"),
    ("prep.rules.before", 35, "docs/model-sizing-study.md 6d"), ("prep.rules.dev.after", 54, "docs/model-sizing-study.md 6d"),
    ("prep.n", 288, "docs/model-sizing-study.md 6d"),
    ("steer.v2.gptoss.flag.wrong", 30, "docs/vetting.md 4.9"), ("steer.v2.gptoss.flag.right", 27, "docs/vetting.md 4.9"),
    ("steer.v2.caught.wrong", 39, "docs/vetting.md 4.9"), ("steer.v2.caught.right", 76, "docs/vetting.md 4.9"),
    ("steer.v2.caught.honest.wrong", 1, "docs/vetting.md 4.9"),
    ("lora.diag.before", 12, "docs/viking-lora-pilot-results.md question 4"),
    ("lora.diag.after", 16, "docs/viking-lora-pilot-results.md question 4"),
    ("lora.gate.tokens", "2,940", "docs/viking-lora-pilot-results.md, training memory at study lengths"),
    ("lora.over.gate.pct", "2.8\\%", "docs/viking-lora-pilot-results.md, training memory at study lengths"),
    ("lora.median.tokens", "1,741", "docs/viking-lora-pilot-results.md, how long are the study's training examples"),
    ("lora.minutes.used", "25.3", "docs/viking-lora-pilot-results.md"),
    ("lora.minutes.reserved", 150, "docs/viking-lora-pilot-results.md"),
]
for key, value, source in FROM_REPORTS:
    put(key, value, "report: " + source)

# ---------------------------------------------------------------- models.tex
ROLES = {
    "Qwen/Qwen3.5-0.8B": ("Qwen3.5 0.8B", "checker, analyst"), "Qwen/Qwen3.5-2B": ("Qwen3.5 2B", "checker, analyst"),
    "Qwen/Qwen3.5-4B": ("Qwen3.5 4B", "checker, analyst"), "Qwen/Qwen3.5-9B": ("Qwen3.5 9B", "checker, analyst"),
    "Qwen/Qwen3.5-27B": ("Qwen3.5 27B", "analyst, checker, probe, slant twins, LoRA pilot"),
    "Qwen/Qwen3.5-35B-A3B": ("Qwen3.5 35B-A3B", "analyst, checker"),
    "nvidia/Qwen3.5-122B-A10B-NVFP4": ("Qwen3.5 122B-A10B (NVFP4)", "analyst, checker"),
    "Qwen/Qwen3-32B": ("Qwen3 32B", "checker, analyst"),
    "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4": ("Mistral Small 3.2 24B (NVFP4)", "checker, analyst"),
    "nvidia/Llama-3.3-70B-Instruct-NVFP4": ("Llama 3.3 70B (NVFP4)", "checker, analyst"),
    "openai/gpt-oss-20b": ("gpt-oss 20B", "checker, analyst, probe"),
    "openai/gpt-oss-120b": ("gpt-oss 120B", "analyst, local judge"),
    "google/gemma-4-31B-it": ("Gemma 4 31B", "analyst, checker"), "google/gemma-4-26B-A4B-it": ("Gemma 4 26B-A4B", "analyst"),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": ("Nemotron 3 Super 120B-A12B (NVFP4)", "analyst"),
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4": ("Nemotron 3.5 Lightning 30B-A3B (NVFP4)", "analyst"),
    "nvidia/diffusiongemma-26B-A4B-it-NVFP4": ("DiffusionGemma 26B-A4B (NVFP4)", "typed reviewer"),
}
CODES = {"analyst": "A", "checker": "C", "probe": "P", "slant twins": "S", "LoRA pilot": "L",
         "local judge": "J", "typed reviewer": "T"}
# Each paper lists the models it used, with the roles they had in it. The vetting
# paper lists a model only if it was probed, wrote bias pairs, was fine-tuned or judged.
MODEL_TABLES = {
    PAPER / "models.tex": ({"analyst", "checker", "local judge", "typed reviewer"}, None),
    ROOT / "paper-vetting" / "models.tex": ({"checker", "probe", "slant twins", "LoRA pilot", "local judge"},
                                            {"probe", "slant twins", "LoRA pilot", "local judge"}),
}
revisions = []
for line in open(PAPER / "data" / "model-revisions.txt"):
    if line.startswith("#") or not line.strip():
        continue
    repo, rev, _when = line.split()
    if repo in ROLES:
        revisions.append((repo, rev))
models_texts: dict[Path, str] = {}
model_rows: dict[Path, int] = {}
for path, (shown, needs) in MODEL_TABLES.items():
    lines = []
    for repo, rev in revisions:
        name, role = ROLES[repo]
        roles = [r.strip() for r in role.split(",")]
        if needs and not needs & set(roles):
            continue
        letters = "".join(CODES[r] for r in roles if r in shown)
        if letters:
            lines.append(f"{name} & {letters} & \\code{{{rev[:12]}}} \\\\ \\multicolumn{{3}}{{@{{}}l}}{{\\quad\\code{{{repo}}}}} \\\\[2pt]")
    models_texts[path] = ("% Generated by paper/make_numbers.py from paper/data/model-revisions.txt. Do not edit.\n"
                          "% \\modelrows is the body of the models table in the appendix.\n"
                          "\\newcommand{\\modelrows}{%\n" + "\n".join(lines) + "}\n")
    model_rows[path] = len(lines)

# ---------------------------------------------------------------- numbers.tex
header = r"""% Generated by paper/make_numbers.py. Do not edit; rerun the script.
% \N{key} typesets a number; an undefined key warns in the log ("undefined number macro")
% and typesets ??key so that the build check catches it.
\makeatletter
\newcommand{\N}[1]{\ifcsname opnum@#1\endcsname\csname opnum@#1\endcsname\else\PackageWarning{numbers}{undefined number macro #1}\textbf{??#1}\fi}
\newcommand{\opnum@def}[2]{\expandafter\def\csname opnum@#1\endcsname{#2}}
"""
body = []
for key in sorted(N):
    src = SOURCE[key]
    note = f"  % {src}" if src != "computed" else ""
    body.append(f"\\opnum@def{{{key}}}{{{N[key]}}}{note}")
numbers_text = header + "\n".join(body) + "\n\\makeatother\n"


def write_or_check(path: Path, text: str) -> bool:
    if "--check" in sys.argv:
        current = path.read_text() if path.exists() else ""
        if current != text:
            print(f"DRIFT: {path} differs from the record; run paper/make_numbers.py", file=sys.stderr)
            return False
        return True
    path.write_text(text)
    return True


ok = write_or_check(PAPER / "numbers.tex", numbers_text)
for path, text in models_texts.items():
    ok &= write_or_check(path, text)
print(f"{'checked' if '--check' in sys.argv else 'wrote'} {PAPER / 'numbers.tex'}: {len(N)} macros "
      f"({sum(1 for s in SOURCE.values() if s.startswith('report:'))} from reports); "
      + ", ".join(f"{p.relative_to(ROOT)}: {n} rows" for p, n in model_rows.items()))
sys.exit(0 if ok else 1)
