#!/usr/bin/env bash
# Append a second judge from another family to every judged harvest log in a
# directory, one log at a time, until none is left without one.
#
#   scripts/second_judge_loop.sh runs/inspect/scored-sizing [rounds] [shard/shards]
#
# With shard/shards (e.g. 1/3) the worker takes only the logs whose name
# hashes to its shard, so several workers can run without touching the same
# file.
# The second judge is DeepSeek 4.1 flash through the Cline gateway (the first
# is MiMo pro). Its verdicts go in `judged_second` beside `judged`
# (`inspect score --action append`); scripts/judge_agreement.py compares them.
# A log the first judge has not finished is skipped until it has.
set -uo pipefail
cd "$(dirname "$0")/.."
DIR="${1:?scored dir}"; ROUNDS="${2:-60}"; SHARD="${3:-0/1}"
set -a; source ~/.config/cline/api.env; set +a
export CLINE_API_KEY="${CLINE_API_KEY:-$CLINE_PASS_KEY}"
export VLLM_API_KEY=EMPTY VLLM_BASE_URL=http://localhost:8000/v1

todo() {
  .venv/bin/python - "$DIR" "$SHARD" <<'PY'
import glob, os, sys, zlib
from inspect_ai.log import read_eval_log
k, n = (int(x) for x in sys.argv[2].split("/"))
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_harvest*.eval"))):
    if zlib.crc32(os.path.basename(f).encode()) % n != k:
        continue
    try:
        l = read_eval_log(f)
    except Exception:
        continue
    ss = l.samples or []
    if ss and all("judged" in (s.scores or {}) for s in ss) and not any("judged_second" in (s.scores or {}) for s in ss):
        print(f)
PY
}

for round in $(seq 1 "$ROUNDS"); do
  n=0
  for f in $(todo); do
    .venv/bin/python -m opacity.offpeak wait   # hosted calls off-peak only (opacity/offpeak.py)
    echo "### JUDGE2 $(basename "$f") $(date -u +%H:%M)"
    .venv/bin/inspect score "$f" --overwrite --action append --scorer scripts/harvest_judge.py@judged_second \
      -S judge=cline/cline-pass/deepseek-v4.1-flash -S max_tokens=8000 --display plain 2>&1 | grep -iE 'error|traceback' | head -3
    echo "### JUDGED2 $(basename "$f") $(date -u +%H:%M)"
    n=$((n+1))
  done
  echo "### ROUND $round: $n logs $(date -u +%H:%M)"
  sleep 600
done
