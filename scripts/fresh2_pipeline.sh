#!/usr/bin/env bash
# The second fresh set (report section 6b), unattended: wait for the analyst
# jobs, stage their logs on Viking as runs/inspect/fresh2/, run the gpt-oss
# 20B checker over them (arm B), pull everything back, copy the harvest logs
# into runs/inspect/scored-sizing/ and judge them there with the validated
# local judge (scripts/local_judge.sh, gpt-oss 120B at high effort).
#
#   scripts/fresh2_pipeline.sh <analyst job id> [<job id> ...]
#
# Reads no result; scripts/fresh_results.py --set fresh2 does that once every
# configuration has run and been judged.
set -uo pipefail
cd "$(dirname "$0")/.."
JOBS=$(IFS=,; echo "$*")
V=${VIKING_HOST:?}
until [ -z "$(ssh -o BatchMode=yes $V "squeue -h -j $JOBS -o %i" < /dev/null 2>/dev/null)" ]; do sleep 300; done
echo "analysts done $(date -u +%H:%M)"
ssh -o BatchMode=yes $V 'cd ~/git/opacity && mkdir -p runs/inspect/fresh2 && .venv/bin/python - <<PY
import glob, shutil, os
from inspect_ai.log import read_eval_log
n = 0
for f in glob.glob("runs/inspect/2026-10-2*_harvest*.eval"):
    try:
        h = read_eval_log(f, header_only=True)
    except Exception:
        continue
    if h.status == "success" and (h.eval.task_args or {}).get("task_set") == "fresh2":
        d = os.path.join("runs/inspect/fresh2", os.path.basename(f))
        if not os.path.exists(d):
            shutil.copy2(f, d)
        n += 1
print("fresh2 logs:", n)
PY
export HF_HOME=/mnt/scratch/users/$USER/hf
TAG=sizing-20261024-f2x PARTITION=gpu REASONING_PARSER=openai_gptoss TIME=03:00:00 scripts/submit_plan.sh openai/gpt-oss-20b scripts/plans/fresh2-cross.txt' < /dev/null | tee /dev/stderr | grep -oE '^[0-9]+ plan:fresh2-cross' | cut -d' ' -f1 > runs/fresh2-cross-job.txt
cj=$(cat runs/fresh2-cross-job.txt)
echo "checker job $cj"
until [ -z "$(ssh -o BatchMode=yes $V "squeue -h -j $cj -o %i" < /dev/null 2>/dev/null)" ]; do sleep 120; done
echo "checker done $(date -u +%H:%M)"
rsync -aqu --exclude failed $V:~/git/opacity/runs/inspect/ runs/inspect/viking/
logs=()
for f in runs/inspect/viking/fresh2/*_harvest*.eval; do
  d="runs/inspect/scored-sizing/$(basename "$f")"
  [ -e "$d" ] || cp "$f" "$d"
  logs+=("$d")
done
echo "judging ${#logs[@]} logs"
EFFORT=high PAR=6 TIME=04:00:00 scripts/local_judge.sh openai/gpt-oss-120b gpuplus 1 "${logs[@]}"
echo "fresh2 pipeline done $(date -u +%H:%M)"
