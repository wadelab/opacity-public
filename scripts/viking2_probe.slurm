#!/bin/bash
# Probe extraction on Viking2: one model reads the checker prompts it was
# given in its crosscheck runs, one forward pass each, no generation
# (docs/vetting.md section 4; src/opacity/probe/extract.py).
#
#   sbatch --partition=gpu_short --gres=gpu:2 --time=00:30:00 \
#          --export=ALL,PROBE=Qwen3.5-27B,LIMIT=20 scripts/viking2_probe.slurm      # smoke test
#   sbatch --partition=gpuplus --gres=gpu:h100:1 --time=06:00:00 \
#          --export=ALL,PROBE=Qwen3.5-27B scripts/viking2_probe.slurm               # full run
#
# Sizes in bf16: Qwen3.5 27B is about 54 GB (one H100, or two A40s);
# gpt-oss 20B is about 42 GB once its MXFP4 weights are expanded, which
# happens because the triton MXFP4 kernels are not installed (one H100, or
# one A40 with little to spare). device_map="auto" spreads a model over the
# GPUs it is given; it needs accelerate in .venv (installed 26 October 2026).
#
# The inputs come from scripts/probe_inputs.py on the workstation, rsynced to
# runs/probe/<tag>/inputs/ here. The revision is the one prefetch_models.sh
# recorded, and the weights load offline.

#SBATCH --job-name=op-probe
#SBATCH --account=CHANGE_ME
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROBE="${PROBE:?PROBE=Qwen3.5-27B or gpt-oss-20b}"
TAG="${TAG:-v1}"
LIMIT="${LIMIT:-0}"

export HF_HOME="${HF_HOME:-/mnt/scratch/users/$USER/hf}"
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source .venv/bin/activate

HF=$(python -c "from opacity.probe import PROBES; print(PROBES['$PROBE']['hf'])")
REV=$(awk -v m="$HF" '$1 == m {print $2}' "$HF_HOME/opacity_revisions.txt" | tail -1)
[ -n "$REV" ] || { echo "no recorded revision for $HF in $HF_HOME/opacity_revisions.txt"; exit 2; }
OUT="runs/probe/$TAG/states/$PROBE"
[ "$LIMIT" = 0 ] || OUT="$OUT-smoke$LIMIT-$SLURM_JOB_ID"

echo "job $SLURM_JOB_ID on $(hostname): $PROBE ($HF @ $REV), limit $LIMIT, out $OUT.npz"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m opacity.probe.extract "$PROBE" --revision "$REV" \
    --inputs "runs/probe/$TAG/inputs/$PROBE.jsonl" --out "$OUT.npz" --limit "$LIMIT"
