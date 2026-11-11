#!/usr/bin/env bash
# Submit one model against one or more plan files on Viking2: the model is
# served once per job and every line of the plan runs against it.
#
#   scripts/submit_plan.sh <hf model id> <plan> [<plan> ...]
#
# Environment (all optional):
#   PARTITION   gpuplus (H100, default) or gpu (A40, 48 GB, six per user)
#   GPUS        GPUs per job, also the tensor-parallel size (default 1)
#   REASONING_PARSER   qwen3, gemma4, openai_gptoss, ... (needed for -T think_budget)
#   TIME        wall limit (default 06:00:00)
#   TAG         run-tag prefix (default sizing-YYYYMMDD)
#   MAX_CONNECTIONS    concurrent requests (default 32)
#   AFTER       a job id this job waits for (afterany)
#
# Several plans are joined into one job, in order, so a model loads once.
# Submissions are appended to runs/sweep-submissions.txt.
set -euo pipefail
[ $# -ge 2 ] || { sed -n 2,18p "$0"; exit 2; }
MODEL="$1"; shift
PARTITION="${PARTITION:-gpuplus}"
GPUS="${GPUS:-1}"
TIME="${TIME:-06:00:00}"
TAG="${TAG:-sizing-$(date +%Y%m%d)}"
short=$(basename "$MODEL")
names=$(for p in "$@"; do basename "$p" .txt; done | paste -sd+)
plan="$*"
if [ $# -gt 1 ]; then
  # under runs/, which the rsync from d2-1 excludes, so a sync can never
  # delete a joined plan that a queued job has yet to read
  mkdir -p runs/plans
  plan="runs/plans/$names.txt"
  cat "$@" > "$plan"
fi
case "$PARTITION" in
  gpuplus) gres="gpu:h100:$GPUS" ;;
  gpu) gres="gpu:$GPUS" ;;   # the A40 partition rejects a typed gres ("Missing GPU resource")
  *) gres="gpu:$GPUS" ;;
esac
cache="${HF_HOME:-/mnt/scratch/users/$USER/hf}/hub/models--${MODEL//\//--}"
[ -d "$cache/snapshots" ] || { echo "not in cache: $MODEL ($cache)"; exit 1; }
tag="$TAG-$names-$short"
dep=(); [ -n "${AFTER:-}" ] && dep=(--dependency="afterany:$AFTER")
jid=$(sbatch --parsable "${dep[@]}" --partition="$PARTITION" --gres="$gres" --time="$TIME" \
  --job-name="op-plan-$short" \
  --export=ALL,MODEL="$MODEL",PLAN="$plan",RUN_TAG="$tag",TP="$GPUS",REASONING_PARSER="${REASONING_PARSER:-}",MAX_CONNECTIONS="${MAX_CONNECTIONS:-32}" \
  scripts/viking2_vllm.slurm)
echo "$jid plan:$names $MODEL $tag $PARTITION x$GPUS $(date -u +%Y-%m-%dT%H:%MZ)" | tee -a runs/sweep-submissions.txt
