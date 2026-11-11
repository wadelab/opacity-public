#!/usr/bin/env bash
# Pre-fetch the harvest sweep into the Hugging Face cache on the Viking2 login
# node, so that compute jobs run with HF_HUB_OFFLINE=1 and never download.
#
# Run once per model list, after `hf auth login` and after accepting the Llama
# and Gemma licences on the Hub. Each model is fetched once; later runs are
# no-ops for anything already present. Pass model ids as arguments to fetch a
# subset.
#
# The resolved commit of every repository is written to
# $HF_HOME/opacity_revisions.txt so the sweep's provenance can cite it.
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/scratch/users/$USER/hf}"
mkdir -p "$HF_HOME"

# Spine: Qwen3.5 dense, one generation, one recipe, five sizes from 0.8B to
# 27B. Thinking is disabled at serve time, not here. The MoE sizes in the same
# generation are listed as points below, not on the spine.
SPINE=(
  Qwen/Qwen3.5-0.8B
  Qwen/Qwen3.5-2B
  Qwen/Qwen3.5-4B
  Qwen/Qwen3.5-9B
  Qwen/Qwen3.5-27B
)

# Second ladder: Gemma 4 in one generation. E2B and E4B are "effective" sizes
# (per-layer embeddings); plot them at their effective count and say so.
GEMMA=(
  google/gemma-4-E2B-it
  google/gemma-4-E4B-it
  google/gemma-4-12B-it
  google/gemma-4-31B-it
)

# Older-generation replication of the spine: Qwen3 dense, 0.6B to 32B. Matched
# sizes against Qwen3.5 separate the recipe effect from the size effect.
QWEN3=(
  Qwen/Qwen3-0.6B
  Qwen/Qwen3-1.7B
  Qwen/Qwen3-4B
  Qwen/Qwen3-8B
  Qwen/Qwen3-14B
  Qwen/Qwen3-32B
)

# Single points from other labs. 70B and 120B are quantised (FP8, MXFP4) and
# are marked as such in the figure. Mixture-of-experts models (gpt-oss, the
# Qwen3.5 and Gemma 4 MoEs) are plotted by active and by total parameters.
POINTS=(
  meta-llama/Llama-3.2-3B-Instruct
  meta-llama/Llama-3.1-8B-Instruct
  RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic
  mistralai/Mistral-Small-3.2-24B-Instruct-2506
  microsoft/phi-4
  openai/gpt-oss-20b
  openai/gpt-oss-120b
  zai-org/GLM-4-9B-0414
  zai-org/GLM-4-32B-0414
  Qwen/Qwen3.5-35B-A3B
  google/gemma-4-26B-A4B-it
)

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=("${SPINE[@]}" "${GEMMA[@]}" "${QWEN3[@]}" "${POINTS[@]}")

# Token: HF_TOKEN in the environment, or the file hf auth login writes under
# HF_HOME. Gated repos (Llama, Gemma) fail without one; check up front.
if [ -n "${HF_TOKEN:-}" ]; then
  echo "token: HF_TOKEN from environment"
elif [ -s "$HF_HOME/token" ]; then
  echo "token: $HF_HOME/token"
else
  echo "no Hugging Face token found. Either: export HF_TOKEN=hf_... ; or" >&2
  echo "  ( umask 077; printf '%s' 'hf_...' > $HF_HOME/token )" >&2
  exit 1
fi

# The CLI lives in the project venv; prefer it, then whatever is on PATH.
VENV_BIN="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin"
if [ -x "$VENV_BIN/hf" ]; then DL="$VENV_BIN/hf download"
elif [ -x "$VENV_BIN/huggingface-cli" ]; then DL="$VENV_BIN/huggingface-cli download"
elif command -v hf >/dev/null 2>&1; then DL="hf download"
elif command -v huggingface-cli >/dev/null 2>&1; then DL="huggingface-cli download"
else
  echo "no hf CLI found; from the repo root run: uv pip install -p .venv/bin/python huggingface_hub" >&2
  exit 1
fi

REV="$HF_HOME/opacity_revisions.txt"
touch "$REV"
for m in "${MODELS[@]}"; do
  echo "== $m"
  # Weights and tokenizer only; skip original/consolidated checkpoints that
  # some repos ship alongside the safetensors.
  path=$($DL "$m" --exclude "original/*" --exclude "consolidated*" --exclude "*.pth" | tail -1)
  sha=$(basename "$path")
  grep -q "^$m " "$REV" || echo "$m $sha $(date -u +%Y-%m-%dT%H:%MZ)" >> "$REV"
done

echo
echo "cache: $HF_HOME"
du -sh "$HF_HOME/hub" 2>/dev/null || true
echo "revisions: $REV"
echo "serve with HF_HUB_OFFLINE=1 so jobs never reach for the network."
