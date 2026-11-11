#!/usr/bin/env bash
# Submit the model sweep on Viking2: for each model, one checker-arm job
# (TASK=opacity, levels s0 to s4) and one harvest job (TASK=harvest, agent arm,
# generated without scoring). Run from ~/git/opacity on the login node after
# scripts/prefetch_models.sh has cached the weights.
#
#   scripts/submit_sweep.sh                       # the Qwen3.5 spine
#   scripts/submit_sweep.sh Qwen/Qwen3-8B ...     # named models
#   TASKS=opacity scripts/submit_sweep.sh         # checker arm only
#   TASKS=harvest HARVEST_ARM=closed scripts/submit_sweep.sh    # the fail-closed arm
#   TASKS=harvest HARVEST_ARM=selfcheck scripts/submit_sweep.sh # agent then self as checker
#   TASKS=crosscheck LOGS='runs/inspect/scored/*harvest*.eval' scripts/submit_sweep.sh
#                                                 # every model as checker on every
#                                                 # model's write-ups (scored logs first)
#
# Jobs queue behind the per-user cap of two GPUs and run two at a time. Models
# above ~34B need two GPUs; pass GPUS=2 for those (sets tensor parallel).
#
# Harvest jobs are generated with --no-score so that no judge API key ever
# reaches the cluster. Pull runs/inspect/ back to d2-1 and score there:
#   rsync -a ${VIKING_HOST:?}:~/git/opacity/runs/inspect/ runs/inspect/
#   set -a; source ~/.config/mimo/api.env; set +a
#   .venv/bin/inspect score runs/inspect/<harvest log>.eval
# The judge string is fixed at submission (JUDGE below) and recorded in the
# log's task args, so scoring later uses the judge the run declared.
set -euo pipefail

SPINE=(Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B)
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=("${SPINE[@]}")

TASKS="${TASKS:-opacity harvest}"
LEVELS="${LEVELS:-0 1 2 3 4}"            # s4 is the prose-only ablation
REPLICATES_OPACITY="${REPLICATES_OPACITY:-5}"   # phase-0 criterion: at least five
REPLICATES_HARVEST="${REPLICATES_HARVEST:-3}"
JUDGE="${JUDGE:-openai-api/mimo/mimo-v2.6-pro}" # recorded only; scoring happens off-cluster
HARVEST_ARM="${HARVEST_ARM:-open}"       # harvest: open, closed or selfcheck
LOGS="${LOGS:-runs/inspect/scored/*harvest*.eval}"  # crosscheck: the scored harvest logs
THINKING="${THINKING:-off}"
REGISTER="${REGISTER:-default}"
TIME="${TIME:-04:00:00}"
GPUS="${GPUS:-1}"
SWEEP_TAG="${SWEEP_TAG:-sweep-$(date +%Y%m%d)}"

export HF_HUB_OFFLINE=1
HF_HOME="${HF_HOME:-/mnt/scratch/users/$USER/hf}"

for m in "${MODELS[@]}"; do
  cache="$HF_HOME/hub/models--${m//\//--}"
  if [ ! -d "$cache/snapshots" ]; then echo "skip $m: not in cache ($cache)"; continue; fi
  short=$(basename "$m")
  for task in $TASKS; do
    case "$task" in
      opacity) reps=$REPLICATES_OPACITY ;;
      harvest) reps=$REPLICATES_HARVEST ;;
      crosscheck) reps=1 ;;
      *) echo "unknown task $task"; exit 2 ;;
    esac
    tag="$SWEEP_TAG-$task-$short"
    [ "$task" = harvest ] && [ "$HARVEST_ARM" != open ] && tag="$SWEEP_TAG-$task-$HARVEST_ARM-$short"
    jid=$(sbatch --parsable --time="$TIME" --job-name="op-$task-$short" --gres="gpu:h100:$GPUS" \
      --export=ALL,MODEL="$m",TASK="$task",LEVELS="$LEVELS",REPLICATES="$reps",JUDGE="$JUDGE",THINKING="$THINKING",REGISTER="$REGISTER",HARVEST_ARM="$HARVEST_ARM",LOGS="$LOGS",RUN_TAG="$tag",TP="$GPUS" \
      scripts/viking2_vllm.slurm)
    echo "$jid  $task  $m  (rep=$reps, tag=$tag)"
    echo "$jid $task $m $tag $(date -u +%Y-%m-%dT%H:%MZ)" >> "runs/sweep-submissions.txt"
  done
done
echo; squeue -u "$USER" -o "%.10i %.22j %.8T %.10M %R"
