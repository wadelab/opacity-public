#!/usr/bin/env python3
"""Phase 0: the s0 ceiling check (and, with --levels 0 1 2 3, the full machine arms).

Examples

  # mechanical arm only, no model needed -- proves the pipeline closes
  python scripts/run_phase0.py --out runs/mech.jsonl --mechanical

  # one LLM checker behind an OpenAI-compatible endpoint (vLLM, hosted API...)
  python scripts/run_phase0.py --out runs/qwen32b.jsonl \
      --llm Qwen/Qwen2.5-32B-Instruct --base-url http://localhost:8000/v1

  # several candidate checkers in one run: phase 0 as model selection
  python scripts/run_phase0.py --out runs/select.jsonl --mechanical \
      --llm Qwen/Qwen2.5-32B-Instruct --base-url http://localhost:8000/v1 \
      --llm some-other-model --base-url http://localhost:8001/v1

  # full four-level machine-arm study
  python scripts/run_phase0.py --out runs/full.jsonl --levels 0 1 2 3 --replicates 5 ...

Every verdict is written as one JSON line, including the verbatim model
response. Append-only: re-running with the same --out adds to the file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opacity import phase0  # noqa: E402
from opacity.checker import MechanicalChecker, OpenAICompatibleChecker  # noqa: E402
from opacity.items import build_corpus, IMPLEMENTED, CLASS_NAMES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="JSONL output (append)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replicates", type=int, default=1, help="datasets per class")
    ap.add_argument("--levels", type=int, nargs="+", default=[0], help="disclosure levels")
    ap.add_argument("--mechanical", action="store_true", help="include the mechanical arm")
    ap.add_argument("--llm", action="append", default=[], metavar="MODEL",
                    help="model name for an LLM checker; repeatable, pairs with --base-url in order")
    ap.add_argument("--base-url", action="append", default=[], metavar="URL",
                    help="OpenAI-compatible endpoint; repeat to pair with each --llm, or give one for all")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--dump-bundles", type=Path, default=None,
                    help="also write every rendered bundle to this directory, for inspection")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    checkers = []
    if args.mechanical:
        checkers.append(MechanicalChecker())
    if args.llm:
        urls = args.base_url
        if len(urls) == 1:
            urls = urls * len(args.llm)
        if len(urls) != len(args.llm):
            ap.error("give one --base-url, or one per --llm")
        for model, url in zip(args.llm, urls):
            checkers.append(OpenAICompatibleChecker(
                base_url=url, model=model, api_key=args.api_key,
                temperature=args.temperature, max_tokens=args.max_tokens,
            ))
    if not checkers:
        ap.error("nothing to run: pass --mechanical and/or --llm")

    items = build_corpus(seed=args.seed, replicates=args.replicates)

    if args.dump_bundles:
        from opacity.sdc import policy
        args.dump_bundles.mkdir(parents=True, exist_ok=True)
        for it in items:
            for b in it.pair:
                for lvl in args.levels:
                    (args.dump_bundles / f"{b.item_id}_s{lvl}.md").write_text(
                        b.controlled(policy(lvl)).render(), encoding="utf-8"
                    )

    # provenance header: one line, so the run is self-describing
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "_run": True,
            "argv": sys.argv,
            "seed": args.seed,
            "replicates": args.replicates,
            "levels": args.levels,
            "classes": list(IMPLEMENTED),
            "class_names": {k: CLASS_NAMES[k] for k in IMPLEMENTED},
            "checkers": [c.name for c in checkers],
        }) + "\n")

    obs = phase0.run(
        checkers, out=args.out, seed=args.seed, replicates=args.replicates,
        levels=tuple(args.levels), items=items, echo=not args.quiet,
    )
    print()
    phase0.print_summary(phase0.summarise(obs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
