#!/usr/bin/env bash
# Wait until the jobs writing the fresh-set harvest logs have left the queue,
# stage those logs on Viking as runs/inspect/fresh/, and submit the two
# checkers that fresh_select.py chose (gpt-oss 20B at medium effort on an
# A40; the typed arm at a 512-token think on an H100) over all of them.
#   scripts/fresh_cross_submit.sh <job id> [<job id> ...]
set -uo pipefail
JOBS=$(IFS=,; echo "$*")
until [ -z "$(ssh -o BatchMode=yes ${VIKING_HOST:?} "squeue -h -j $JOBS -o %i" < /dev/null 2>/dev/null)" ]; do sleep 300; done
echo "producers done $(date -u +%H:%M)"
ssh -o BatchMode=yes ${VIKING_HOST:?} 'cd ~/git/opacity && mkdir -p runs/inspect/fresh && .venv/bin/python - <<PY
import glob, shutil, os
from inspect_ai.log import read_eval_log
n=0
for f in glob.glob("runs/inspect/2026-10-2*_harvest*.eval"):
    try: h=read_eval_log(f, header_only=True)
    except Exception: continue
    if h.status=="success" and (h.eval.task_args or {}).get("task_set")=="fresh" and any(t.startswith("sizing-20261023-fresh") for t in (h.eval.tags or [])):
        d=os.path.join("runs/inspect/fresh", os.path.basename(f))
        if not os.path.exists(d): shutil.copy2(f, d)
        n+=1
print("fresh set:", n)
PY
export HF_HOME=/mnt/scratch/users/$USER/hf
TAG=sizing-20261023-fx PARTITION=gpu REASONING_PARSER=openai_gptoss TIME=03:00:00 scripts/submit_plan.sh openai/gpt-oss-20b scripts/plans/fresh-cross.txt
jid=$(sbatch --parsable --time=02:00:00 --job-name=op-typed-fx --export=ALL,TASK=crosscheck,LOGS="runs/inspect/fresh/*harvest*.eval",THINK=512,RUN_TAG=sizing-20261023-fx-typed-think512 scripts/viking2_openjev.slurm)
echo "$jid typed-crosscheck-fresh think512 sizing-20261023-fx-typed-think512 gpuplus x1 $(date -u +%Y-%m-%dT%H:%MZ)" | tee -a runs/sweep-submissions.txt' < /dev/null
