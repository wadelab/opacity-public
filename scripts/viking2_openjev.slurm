#!/bin/bash
# The typed arm on Viking2: serve openjev (a decision server on DiffusionGemma)
# from its container image on one H100, then run the opacity task against it.
#
# One-time, on the login node (see logs/openjev_pull.sh):
#   module load Apptainer
#   apptainer pull /mnt/scratch/users/$USER/apptainer/openjev-0.7.0.sif docker://razorback16/openjev:0.7.0
#   scripts/prefetch_models.sh nvidia/diffusiongemma-26B-A4B-it-NVFP4
#
# The image carries its own patched vLLM, so it never touches .venv. The
# model is NVFP4; the README asks for 24 GB and does not say whether Hopper
# is supported, so the first run is a test of that. FlashInfer compiles
# kernels with nvcc inside the container on first start (slow; cached on
# scratch between runs).

#SBATCH --job-name=opacity-typed
#SBATCH --partition=gpuplus
#SBATCH --account=CHANGE_ME
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load Apptainer

S=/mnt/scratch/users/$USER
SIF="${SIF:-$S/apptainer/openjev-0.7.0.sif}"
RUN_TAG="${RUN_TAG:-typed-$(date +%Y%m%d-%H%M%S)}"
LEVELS="${LEVELS:-0 1 2 3 4}"
REPLICATES="${REPLICATES:-5}"
SAMPLES="${SAMPLES:-1}"
THINK="${THINK:-0}"
export HF_HOME="${HF_HOME:-$S/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p logs runs/inspect "$S/apptainer/vllm-cache" "$S/apptainer/flashinfer-cache"
source .venv/bin/activate
# a free port per job, as in viking2_vllm.slurm: two servers on one node must not share one
PORT="${PORT:-$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}"

# --- serve ------------------------------------------------------------------
# Apptainer runs the container as the submitting user, so $HOME inside is
# /users/<user>, not /root: pass the cache locations by environment variable
# and bind scratch, rather than binding onto /root/.cache (first attempt
# failed with LocalEntryNotFoundError under HF_HUB_OFFLINE=1 for that reason).
apptainer run --nv --cleanenv --bind /mnt/scratch \
  --env HF_HOME="$HF_HOME" \
  --env VLLM_CACHE_ROOT="$S/apptainer/vllm-cache" \
  --env FLASHINFER_WORKSPACE_BASE="$S/apptainer/flashinfer-cache" \
  --env OPENJEV_HOST=127.0.0.1 --env OPENJEV_PORT="$PORT" \
  --env OPENJEV_MODEL_ROUTES= --env OPENJEV_CANVAS=64 \
  --env OPENJEV_GPU_UTIL="${OPENJEV_GPU_UTIL:-0.9}" \
  --env OPENJEV_MAX_MODEL_LEN="${OPENJEV_MAX_MODEL_LEN:-16384}" \
  --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" --env HF_TOKEN="${HF_TOKEN:-}" \
  "$SIF" > "logs/openjev-$RUN_TAG.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then break; fi
  sleep 10
  if ! kill -0 $SRV 2>/dev/null; then echo "openjev died; see logs/openjev-$RUN_TAG.log"; exit 1; fi
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "openjev never became healthy"; exit 1; }
curl -s "http://127.0.0.1:$PORT/v1/models" > "runs/typed-served-$RUN_TAG.json" || true

# smoke: one decision, so the log shows the wire format before the run
curl -s "http://127.0.0.1:$PORT/v1/systemone" -H 'content-type: application/json' \
  -d '{"model":"openjev-latest","state":"The pooled comparison is null.","questions":{"q":{"type":"noul","instructions":"Is this a claim of an effect?"}}}' \
  | tee "runs/typed-smoke-$RUN_TAG.json"; echo

# --- run --------------------------------------------------------------------
LEVELS_T=$(echo $LEVELS | tr ' ' ',')
TASK="${TASK:-opacity}"   # opacity: the scripted corpus; crosscheck: harvested write-ups in LOGS
if [ "$TASK" = crosscheck ]; then
  inspect eval opacity/crosscheck -T arm=typed -T typed_url="http://127.0.0.1:$PORT" -T typed_think="$THINK" \
    -T logs="${LOGS:?set LOGS to harvest log paths or globs}" \
    --log-dir runs/inspect/crosscheck --max-connections 16 --tags "viking2,typed,$RUN_TAG,$SLURM_JOB_ID"
else
  inspect eval opacity/opacity -T arm=typed -T typed_url="http://127.0.0.1:$PORT" \
    -T typed_samples="$SAMPLES" -T typed_think="$THINK" \
    -T levels="$LEVELS_T" -T replicates="$REPLICATES" \
    --log-dir runs/inspect --max-connections 16 --tags "viking2,typed,$RUN_TAG,$SLURM_JOB_ID"
fi
echo "done: typed arm, $TASK ($RUN_TAG)"
