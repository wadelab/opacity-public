#!/usr/bin/env bash
# Append DeepSeek v4.1 flash as `judged_second` to the harvest logs named on the
# command line, one at a time, off-peak only.
#
#   scripts/judge_second_logs.sh runs/inspect/scored-sizing/<log>.eval ...
#
# For logs with no MiMo verdict (scripts/second_judge_loop.sh needs one), such
# as the fresh2 logs. Uses the pay-as-you-go id on the Cline gateway
# (cline/deepseek/..., not cline-pass/), which bills credits; ask Wade before a
# bulk run. Settings match the judge's validation (max_tokens 8000). Back the
# logs up first: `inspect score --overwrite` rewrites each file in place.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source ~/.config/cline/api.env; set +a
export CLINE_API_KEY="${CLINE_API_KEY:-$CLINE_PASS_KEY}"
export VLLM_API_KEY=EMPTY VLLM_BASE_URL=http://localhost:8000/v1   # inspect score initialises the log's model
JUDGE="${JUDGE:-cline/deepseek/deepseek-v4.1-flash}"
for f in "$@"; do
  .venv/bin/python -m opacity.offpeak wait   # hosted calls off-peak only (opacity/offpeak.py)
  echo "### JUDGE2 $(basename "$f") $(date -u +%H:%M:%S)"
  .venv/bin/inspect score "$f" --overwrite --action append --scorer scripts/harvest_judge.py@judged_second \
    -S judge="$JUDGE" -S max_tokens=8000 --display plain </dev/null 2>&1 | grep -iE 'error|traceback|INFERENCE_CAP|402|429' | head -5
  echo "### JUDGED2 $(basename "$f") $(date -u +%H:%M:%S)"
done
echo "### ALLDONE $(date -u +%H:%M:%S)"
