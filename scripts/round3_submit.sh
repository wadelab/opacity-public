#!/usr/bin/env bash
# Wait until the jobs that write the prompt-optimisation and new-family
# harvest logs have all left the queue, stage those logs on Viking as
# runs/inspect/round3/, and submit the round-3 cross-checks (gpt-oss 20B at
# medium effort on an A40; the typed arm at a 512-token think on an H100).
#   scripts/round3_submit.sh <job id> [<job id> ...]
set -uo pipefail
JOBS=$(IFS=,; echo "$*")
until [ -z "$(ssh -o BatchMode=yes ${VIKING_HOST:?} "squeue -h -j $JOBS -o %i" < /dev/null 2>/dev/null)" ]; do sleep 300; done
echo "producers done $(date -u +%H:%M)"
ssh -o BatchMode=yes ${VIKING_HOST:?} 'cd ~/git/opacity && mkdir -p runs/inspect/round3 && .venv/bin/python - <<PY
import glob, shutil, os
from inspect_ai.log import read_eval_log
n=0
for f in glob.glob("runs/inspect/2026-10-2*_harvest*.eval"):
    try: h=read_eval_log(f, header_only=True)
    except Exception: continue
    if h.status=="success" and any(t.startswith(("sizing-20261022-po","sizing-20261022-nm")) for t in (h.eval.tags or [])):
        d=os.path.join("runs/inspect/round3", os.path.basename(f))
        if not os.path.exists(d): shutil.copy2(f, d)
        n+=1
print("round3 set:", n)
PY
export HF_HOME=/mnt/scratch/users/$USER/hf
TAG=sizing-20261022-r3 PARTITION=gpu REASONING_PARSER=openai_gptoss TIME=06:00:00 scripts/submit_plan.sh openai/gpt-oss-20b scripts/plans/round3-cross.txt
jid=$(sbatch --parsable --time=02:00:00 --job-name=op-typed-r3 --export=ALL,TASK=crosscheck,LOGS="runs/inspect/round3/*harvest*.eval",THINK=512,RUN_TAG=sizing-20261022-r3-typed-think512 scripts/viking2_openjev.slurm)
echo "$jid typed-crosscheck-round3 think512 sizing-20261022-r3-typed-think512 gpuplus x1 $(date -u +%Y-%m-%dT%H:%MZ)" | tee -a runs/sweep-submissions.txt' < /dev/null
