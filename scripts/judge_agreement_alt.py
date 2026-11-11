"""Agreement of the alternative judge (`judged_alt`) with MiMo (`judged`) and
DeepSeek (`judged_second`) on conclusion_wrong, per write-up, beside the
MiMo-DeepSeek agreement on the same write-ups.

    .venv/bin/python scripts/judge_agreement_alt.py runs/inspect/judge-validation/*.eval
    .venv/bin/python scripts/judge_agreement_alt.py --candidate judged_mimo_flash runs/inspect/judge-validation/*.eval

Only write-ups every compared judge parsed are counted; declined write-ups
are not judged and are left out. Cohen's kappa is reported with each
agreement rate, and the confusion counts against MiMo.
"""

from __future__ import annotations

import sys
from collections import Counter

from inspect_ai.log import read_eval_log


def wrong(s, name):
    sc = (s.scores or {}).get(name)
    v = sc.value if sc is not None and isinstance(sc.value, dict) else None
    if not v or v.get("declined") or not v.get("judge_parse_ok"):
        return None
    return bool(v.get("conclusion_wrong"))


def kappa(pairs):
    n = len(pairs)
    if not n:
        return float("nan"), float("nan")
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main(paths, cand="judged_alt"):
    rows = []
    unparsed_alt = 0
    for p in paths:
        log = read_eval_log(p)
        for s in log.samples or []:
            if cand in (s.scores or {}) and wrong(s, cand) is None and wrong(s, "judged") is not None:
                unparsed_alt += 1
            rows.append((wrong(s, "judged"), wrong(s, "judged_second"), wrong(s, cand)))
    print(f"candidate judge: {cand}")
    for name, i, j in (("cand vs MiMo", 2, 0), ("cand vs DeepSeek", 2, 1), ("MiMo vs DeepSeek", 0, 1)):
        pairs = [(r[i], r[j]) for r in rows if r[i] is not None and r[j] is not None]
        po, k = kappa(pairs)
        print(f"{name:18s} n={len(pairs):4d}  agreement {po:.3f}  kappa {k:.3f}")
    three = [r for r in rows if None not in r]
    po, k = kappa([(r[2], r[0]) for r in three])
    print(f"on the {len(three)} write-ups all three parsed: alt vs MiMo {po:.3f} (kappa {k:.3f}); "
          f"MiMo vs DeepSeek {kappa([(r[0], r[1]) for r in three])[0]:.3f}")
    c = Counter((r[0], r[2]) for r in rows if r[0] is not None and r[2] is not None)
    print(f"confusion against MiMo (MiMo, alt): wrong/wrong {c[(True, True)]}, wrong/right {c[(True, False)]}, "
          f"right/wrong {c[(False, True)]}, right/right {c[(False, False)]}")
    print(f"alt judge unparseable where MiMo parsed: {unparsed_alt}")
    both = [r for r in rows if r[0] is not None and r[2] is not None]  # the same write-ups for both
    print(f"share judged wrong, on the {len(both)} write-ups both parsed: "
          f"MiMo {sum(r[0] for r in both) / max(1, len(both)):.3f}, candidate {sum(r[2] for r in both) / max(1, len(both)):.3f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    cand = "judged_alt"
    if args[:1] == ["--candidate"]:
        cand, args = args[1], args[2:]
    main(args, cand)
