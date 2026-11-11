#!/usr/bin/env bash
# Ceiling check: hosted frontier models as analysts on the dev and held-out
# harvest tasks, to show the keys' correct answers are reachable. Judged
# inline by MiMo pro. GLM-5.3 (Z.ai) shares a family with neither judge;
# DeepSeek v4 pro shares one with the second judge, so only MiMo counts for
# it. Claude is not used as an analyst: a Claude model wrote the tasks. The
# fresh set is not run here until its selection is committed.
#
#   scripts/frontier_ceiling.sh            # logs under runs/inspect/frontier/
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source ~/.config/mimo/api.env; source ~/.config/cline/api.env; set +a
export CLINE_API_KEY="${CLINE_API_KEY:-$CLINE_PASS_KEY}"
J=openai-api/mimo/mimo-v2.6-pro
run() {  # model, then harvest -T arguments
  local m="$1"; shift
  .venv/bin/python -m opacity.offpeak wait   # hosted calls off-peak only (opacity/offpeak.py)
  echo "### $(date -u +%H:%M) $m $*"
  .venv/bin/inspect eval opacity/harvest "$@" -T replicates=3 -T judge=$J --model "cline/cline-pass/$m" \
    --max-tokens 20000 --max-connections 8 --log-dir runs/inspect/frontier --tags "frontier-20261022,ceiling" \
    --display plain 2>&1 | grep -iE "error|traceback" | head -3
}
for ts in dev heldout; do
  run glm-5.3 -T task_set=$ts
  run glm-5.3 -T task_set=$ts -T checklist=true
  run deepseek-v4-pro -T task_set=$ts
done
echo "### done $(date -u +%H:%M)"
