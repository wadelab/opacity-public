"""Blind coding of checker free-text descriptions to error class, by two LLM
coders from different families, standing in for the human coder until the
human arm runs.

    set -a; source ~/.config/mimo/api.env; source ~/.config/cline/api.env; set +a
    .venv/bin/python scripts/code_descriptions.py --coder mimo --coder deepseek \
        --tags hosted2-20261021 sweep4-20261022 sweep5-20261022 --out runs/summary/coded_descriptions.jsonl

For every flagged verdict (error_present = yes) with a non-empty description
in the selected checker logs, each coder sees the description alone, never
the bundle, and returns one of the eight classes or 0 for none. The output
is one JSON line per (sample, coder) with the coder's class, the seeded
class, and the verbatim reply. `summarise` prints agreement with the seeded
class (seeded bundles) and between coders.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log
from openai import OpenAI

from opacity.harness.providers import unwrap_envelope

ROOT = Path(__file__).resolve().parents[1]

CODING_INSTRUCTION = """\
A reviewer read a data-analysis report and wrote the note below explaining \
what they think is wrong with it. Assign the note to ONE of these error \
classes, by the mechanism the reviewer describes, not by the topic of the \
report:

  1 unreported multiplicity (many comparisons, one nominal hit reported)
  2 silent attrition (participants dropped between enrolment and analysis without being declared)
  3 confounding or Simpson reversal (a pooled comparison reversed or distorted by an unaccounted variable)
  4 unsupported causal claim (correct association, causal conclusion or recommendation not supported by the design)
  5 wrong error structure (clustered or repeated-measures data tested as independent observations)
  6 wrong denominator (a rate computed over the wrong base)
  7 regression to the mean (selection on an extreme baseline then remeasured)
  8 in-sample performance (a model evaluated on the data it was fitted to)
  0 none of these, or the note does not describe an analytic error

Answer with a JSON object and nothing else: {"class": <integer 0-8>, "reason": "<one sentence>"}

Reviewer's note:
"""

CODERS = {
    "mimo": dict(base_url=os.environ.get("MIMO_BASE_URL"), api_key=os.environ.get("MIMO_API_KEY"), model=os.environ.get("MIMO_MODEL", "mimo-v2.6-pro"), max_tokens=20000),
    "deepseek": dict(base_url=os.environ.get("CLINE_BASE_URL"), api_key=os.environ.get("CLINE_API_KEY") or os.environ.get("CLINE_PASS_KEY"), model=os.environ.get("CLINE_MODEL", "cline-pass/deepseek-v4.1-flash"), max_tokens=4000),
}


def classify(tags):
    return ",".join(sorted(tags))


def collect(tags_wanted):
    items = []
    for f in sorted(glob.glob(str(ROOT / "runs/inspect/*opacity-*.eval")) + glob.glob(str(ROOT / "runs/inspect/viking/*opacity-*.eval"))):
        log = read_eval_log(f, header_only=True)
        tags = log.eval.tags or []
        if not any(any(t.startswith(w) for w in tags_wanted) for t in tags) or (log.eval.metadata or {}).get("arm") != "llm":
            continue
        log = read_eval_log(f)
        model = log.eval.model.split("/")[-1]
        for s in log.samples:
            v = s.scores["verdict"]
            if v.answer != "yes" or not (v.explanation or "").strip():
                continue
            items.append(dict(log=os.path.basename(f), checker=model, sample=s.id, level=s.metadata["disclosure_level"],
                              condition=s.metadata["condition"], seeded_class=s.metadata["error_class"], description=v.explanation.strip()))
    return items


def code_one(client, cfg, description, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.with_raw_response.create(
                model=cfg["model"], temperature=0, max_tokens=cfg["max_tokens"],
                messages=[{"role": "system", "content": "You respond only with the JSON object requested."},
                          {"role": "user", "content": CODING_INSTRUCTION + description}])
            # Cline wraps non-streaming replies in {"data": ..., "success": true}; the OpenAI client
            # then returns a completion with no choices. Unwrap as the Inspect provider does.
            body = unwrap_envelope(resp.http_response.json())
            raw = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            start, end = raw.find("{"), raw.rfind("}")
            obj = json.loads(raw[start:end + 1]) if start >= 0 else {}
            c = int(obj.get("class", -1))
            return (c if 0 <= c <= 8 else -1), raw
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            time.sleep(2 * (attempt + 1))
    return -1, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coder", action="append", choices=list(CODERS), required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "runs/summary/coded_descriptions.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    items = collect(a.tags)
    if a.limit:
        items = items[: a.limit]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            d = json.loads(line); done.add((d["log"], d["sample"], d["coder"]))
    print(f"{len(items)} flagged descriptions; {len(done)} already coded", file=sys.stderr)
    from concurrent.futures import ThreadPoolExecutor
    with open(a.out, "a") as fh:
        for coder in a.coder:
            cfg = CODERS[coder]
            client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
            todo = [it for it in items if (it["log"], it["sample"], coder) not in done]
            print(f"coder {coder}: {len(todo)} to code", file=sys.stderr)
            with ThreadPoolExecutor(a.workers) as ex:
                for it, (c, raw) in zip(todo, ex.map(lambda it: code_one(client, cfg, it["description"]), todo)):
                    fh.write(json.dumps({**it, "coder": coder, "coded_class": c, "raw": raw}) + "\n"); fh.flush()
    summarise(a.out)


def summarise(path):
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(dict)
    for r in rows:
        by[(r["log"], r["sample"])][r["coder"]] = r
    coders = sorted({r["coder"] for r in rows})
    print("\nagreement with the seeded class (seeded bundles, flagged), and parse rate, by coder and checker:")
    for coder in coders:
        per = defaultdict(lambda: [0, 0, 0])
        for r in rows:
            if r["coder"] != coder: continue
            p = per[r["checker"]]; p[2] += 1
            if r["coded_class"] >= 0: p[1] += 1
            if r["condition"] == "seeded" and r["coded_class"] == r["seeded_class"]: p[0] += 1
        for chk, (ok, parsed, n) in sorted(per.items()):
            ns = sum(1 for r in rows if r["coder"] == coder and r["checker"] == chk and r["condition"] == "seeded")
            print(f"  {coder:<9} {chk:<44} seeded coded to the seeded class {ok}/{ns} = {ok/max(ns,1):.2f}  parsed {parsed}/{n}")
    if len(coders) == 2:
        a, b = coders; agree = tot = 0
        for k, d in by.items():
            if a in d and b in d and d[a]["coded_class"] >= 0 and d[b]["coded_class"] >= 0:
                tot += 1; agree += d[a]["coded_class"] == d[b]["coded_class"]
        print(f"\ncoder agreement ({a} vs {b}): {agree}/{tot} = {agree/max(tot,1):.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarise":
        summarise(sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "runs/summary/coded_descriptions.jsonl"))
    else:
        main()
