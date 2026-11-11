#!/bin/bash
# Frozen per bundle. Resource limits are supplied by submit.py, one stage per allocation.
set -euo pipefail
umask 077
if [ "$#" -ne 4 ]; then
    echo 'expected bundle, output, host interpreter and uv executable' >&2
    exit 2
fi
bundle=$1
output=$2
host_python=$3
uv_executable=$4
: "${SLURM_JOB_ID:?batch allocation required}"
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU assignment missing}"
# HOME is replaced, so uv cannot find its managed interpreters: name the one to use.
exec env -i \
    PATH=/usr/bin:/bin:/usr/sbin:/sbin \
    HOME="$output/host-home" TMPDIR="$output/host-tmp" UV_CACHE_DIR="$output/host-cache" \
    UV_PYTHON="$host_python" PYTHONPATH="$bundle/src" PYTHONDONTWRITEBYTECODE=1 \
    SLURM_JOB_ID="$SLURM_JOB_ID" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$uv_executable" run --offline --no-project "$host_python" \
    -m opacity.lora_pilot launch --bundle "$bundle" --out "$output"
