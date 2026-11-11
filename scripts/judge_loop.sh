#!/usr/bin/env bash
# Judge harvest logs from Viking as they finish, on d2-1, so the judge's key
# never reaches the cluster.
#
#   scripts/judge_loop.sh <tag substring> <scored dir> [rounds] [parallel]
#   scripts/judge_loop.sh sizing-20261022 runs/inspect/scored-sizing 60 2
#
# Each round pulls runs/inspect/ from Viking into runs/inspect/viking/
# (append-only, newer files only), copies every finished harvest log whose
# tags contain <tag substring> and that is not yet in <scored dir>, and
# scores the copy with the judge the run declared in its task arguments.
# The pulled original is never modified. Stops when no job of ours is left
# on Viking and nothing is waiting to be judged, or after [rounds] rounds.
set -uo pipefail
cd "$(dirname "$0")/.."
TAG="${1:?tag substring}"; OUT="${2:?scored dir}"; ROUNDS="${3:-60}"; PAR="${4:-2}"
mkdir -p "$OUT" runs/inspect/viking
set -a; source ~/.config/mimo/api.env; set +a
# inspect score initialises the log's own model (openai-api/vllm/...), which
# needs these even though only the judge is called
export VLLM_API_KEY=EMPTY VLLM_BASE_URL=http://localhost:8000/v1

pending() {
  .venv/bin/python - "$TAG" "$OUT" <<'PY'
import glob, os, sys
from inspect_ai.log import read_eval_log
tag, out = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob("runs/inspect/viking/*_harvest*.eval")):
    if os.path.exists(os.path.join(out, os.path.basename(f))):
        continue
    try:
        h = read_eval_log(f, header_only=True)
    except Exception:
        continue
    if h.status == "success" and any(tag in t for t in (h.eval.tags or [])):
        print(f, h.eval.task_args.get("judge") or "openai-api/mimo/mimo-v2.6-pro")
PY
}

score_one() {
  local f="$1" judge="$2" dst="$OUT/$(basename "$1")"
  cp "$f" "$dst.part" && mv "$dst.part" "$dst"
  echo "### SCORE $(basename "$f") judge=$judge $(date -u +%H:%M)"
  .venv/bin/inspect score "$dst" --overwrite --scorer scripts/harvest_judge.py@judged \
    -S judge="$judge" -S max_tokens=20000 --display plain 2>&1 | grep -iE 'error|traceback' | head -3
  if .venv/bin/python -c "import sys; from inspect_ai.log import read_eval_log as r; l=r(sys.argv[1]); sys.exit(0 if l.samples and 'judged' in (l.samples[0].scores or {}) else 1)" "$dst"; then
    echo "### SCORED $(basename "$f") $(date -u +%H:%M)"
  else
    mkdir -p "$OUT/failed"; mv "$dst" "$OUT/failed/"; echo "### NOT SCORED $(basename "$f"), moved to $OUT/failed"
  fi
}

for round in $(seq 1 "$ROUNDS"); do
  rsync -aqu --exclude failed ${VIKING_HOST:?}:~/git/opacity/runs/inspect/ runs/inspect/viking/ 2>/dev/null
  n=0
  while read -r f judge; do
    [ -n "$f" ] || continue
    .venv/bin/python -m opacity.offpeak wait   # hosted calls off-peak only (opacity/offpeak.py)
    score_one "$f" "$judge" &
    n=$((n+1)); [ $((n % PAR)) -eq 0 ] && wait
  done < <(pending)
  wait
  jobs_left=$(ssh -o BatchMode=yes ${VIKING_HOST:?} 'squeue --me -h | wc -l' 2>/dev/null || echo 1)
  left=$(pending | grep -c . || true)
  echo "### ROUND $round done $(date -u +%H:%M): jobs on Viking $jobs_left, waiting to judge $left"
  if [ "$jobs_left" = "0" ] && [ "$left" = "0" ]; then echo "### LOOP DONE"; exit 0; fi
  sleep 300
done
echo "### LOOP GAVE UP after $ROUNDS rounds"
