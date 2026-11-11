#!/bin/bash
# Serve an open-weight checker/agent with vLLM on a Viking2 H100 node, then run
# phase 0 against it. One job, one node, one model.
#
# Viking2 facts (checked 2026-10-21 with sinfo/sacctmgr/nvidia-smi):
#   partition gpuplus    six nodes, 2x H100 PCIe 80 GB each, 48 cores, 500 GB RAM,
#                        3-day wall limit, QOS cap of 2 GPUs per user
#   account              set --account to your own Slurm account
#   driver               595.x, so current vLLM wheels (CUDA 12.x) run as-is
#   scratch              /mnt/scratch/users/$USER, visible on compute nodes;
#                        HF_HOME lives there so model weights stay off $HOME
#   environment          .venv made with uv on the login node (see below);
#                        no module load needed, uv supplies the interpreter
#
# One-time setup, on the login node, from the repo root:
#   uv venv --python 3.12 .venv
#   uv pip install -p .venv/bin/python vllm -e ".[llm]"
#   FI=$(.venv/bin/python -c "import flashinfer; print(flashinfer.__version__)")
#   uv pip install -p .venv/bin/python "flashinfer-jit-cache==$FI+cu130" \
#       --extra-index-url https://flashinfer.ai/whl/cu130
# The last step matters: compute nodes have no nvcc, and without the prebuilt
# kernel cache vLLM dies at warm-up trying to JIT-compile a flashinfer op
# ("Could not find nvcc"). Match the +cuNNN tag to torch.version.cuda.
# Verified 2026-10-21 with vllm 0.32.0 / torch 2.13.0+cu130 / flashinfer 0.7.0.post1.
# `vllm download-kernels` would do the same but needs a GPU to run, so it
# cannot be used on the login node.
#
# Hardware note: the H100s are the 80 GB PCIe part. A 32B model in bf16 is
# ~64 GB of weights, which fits with ~10 GB left for KV cache at 0.92
# utilisation -- enough for phase 0's short bundles, so there is no need to
# quantise, which also removes quantisation from the pre-registration.
# Anything larger than ~34B needs --tensor-parallel-size 2 and gres=gpu:h100:2.
# max-num-seqs is capped at 64 (the run uses at most 16 connections): with the
# vLLM default of 1024, Qwen3.5-27B (hybrid Mamba) failed at start-up because
# each decode sequence needs a Mamba cache block and only ~360 fit beside the
# weights ("max_num_seqs (1024) exceeds available Mamba cache blocks").
#
# VLLM_EXTRA_ARGS is appended to `vllm serve` verbatim, for flags a model card
# asks for (Nemotron 3 Super: --mamba-ssm-cache-dtype float16).
#
# Gated Hugging Face models need HF_TOKEN in the environment (not in this
# file). Set it in ~/.bashrc or pass with --export.

#SBATCH --job-name=opacity-phase0
#SBATCH --partition=gpuplus
#SBATCH --account=CHANGE_ME
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"                # any HF repo id; pass MODEL=... to sbatch --export
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
LEVELS="${LEVELS:-0}"                            # "0" for phase 0; "0 1 2 3 4" for full machine arms (s4 = prose only)
REPLICATES="${REPLICATES:-1}"

export HF_HOME="${HF_HOME:-/mnt/scratch/users/$USER/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"      # weights come from scripts/prefetch_models.sh; set 0 to allow downloads
export VLLM_LOGGING_LEVEL=WARNING
mkdir -p logs runs "$HF_HOME"

source .venv/bin/activate                        # created once with uv; see header
# A free port per job: two jobs on one node both on 8000 meant the second
# server never bound and its requests went to the first job's model
# (2026-10-22: 404 "model does not exist" on every step of a plan).
PORT="${PORT:-$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}"
# NVFP4 checkpoints JIT-compile a kernel with NVRTC at load time, and the
# compiler must find libnvrtc-builtins.so.13.0, which torch's cu130 wheel
# installs under the pip CUDA package rather than on the system path
# ("nvrtc: error: failed to open libnvrtc-builtins.so.13.0" otherwise).
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- serve ------------------------------------------------------------------
# REASONING_PARSER (qwen3, gemma4, openai_gptoss, ...) makes vLLM return the
# thinking separately from the answer and enables the per-request
# thinking_token_budget that -T think_budget= sends. Unset, the server is
# exactly the one every earlier run used.
# SERVE_API_KEY_FILE (mode 600) holds a key the server will require; vLLM
# reads it from VLLM_API_KEY, so it never appears on a command line other
# users can see.
if [ -n "${SERVE_API_KEY_FILE:-}" ]; then
  SERVE_API_KEY=$(cat "$SERVE_API_KEY_FILE"); export VLLM_API_KEY="$SERVE_API_KEY"
fi
# When a budget runs out the server forces REASONING_CONFIG's end string. A
# bare </think> made Qwen3.5-9B carry on reasoning in the visible answer until
# the output cap (smoke run 2026-10-22, h01 at 512 tokens), so for qwen3 the
# end string is the transition sentence Qwen recommends for budget forcing.
if [ "${REASONING_PARSER:-}" = qwen3 ] && [ -z "${REASONING_CONFIG:-}" ]; then
  REASONING_CONFIG='{"reasoning_end_str": "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"}'
fi
vllm serve "$MODEL" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} \
  ${REASONING_CONFIG:+--reasoning-config "$REASONING_CONFIG"} \
  --gpu-memory-utilization 0.92 \
  --seed 0 \
  --tensor-parallel-size "${TP:-1}" \
  --max-num-seqs "${MAX_NUM_SEQS:-64}" \
  --served-model-name "$MODEL" \
  ${DEFAULT_TEMPLATE_KWARGS:+--default-chat-template-kwargs "$DEFAULT_TEMPLATE_KWARGS"} \
  ${VLLM_EXTRA_ARGS:-} \
  > "logs/vllm-$RUN_TAG.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# wait for the endpoint
AUTH=(); [ -n "${SERVE_API_KEY:-}" ] && AUTH=(-H @"$SERVE_API_KEY_FILE.header")
[ -n "${SERVE_API_KEY:-}" ] && ( umask 077; printf 'Authorization: Bearer %s\n' "$SERVE_API_KEY" > "$SERVE_API_KEY_FILE.header" )
for i in $(seq 1 "${SERVE_WAIT:-240}"); do    # 5 s each; large checkpoints take several minutes
  if curl -sf "${AUTH[@]}" "http://localhost:$PORT/v1/models" | grep -qF "\"$MODEL\""; then break; fi
  sleep 5
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vllm died; see logs/vllm-$RUN_TAG.log"; exit 1; fi
done

# record exactly what served the run -- part of the reproducibility record
curl -s "${AUTH[@]}" "http://localhost:$PORT/v1/models" > "runs/served-$RUN_TAG.json"
python - > "runs/versions-$RUN_TAG.txt" <<'PY'
# a uv venv has no pip, so record versions from the interpreter itself
import importlib.metadata as m
for pkg in ("vllm", "torch", "transformers", "flashinfer-python", "flashinfer-jit-cache"):
    try:
        print(f"{pkg}=={m.version(pkg)}")
    except m.PackageNotFoundError:
        print(f"{pkg}==(absent)")
PY

# --- run --------------------------------------------------------------------
# Inspect reaches the local vLLM server as an openai-api service called "vllm";
# the model string keeps the Hugging Face id, so the log carries it verbatim.
export VLLM_BASE_URL="http://localhost:$PORT/v1"
export VLLM_API_KEY="${SERVE_API_KEY:-EMPTY}"
TASK="${TASK:-opacity}"            # opacity: checker arms on the scripted corpus; harvest: agent arm; crosscheck: checker on harvested write-ups
THINKING="${THINKING:-off}"        # all tasks; enable_thinking via chat_template_kwargs
REGISTER="${REGISTER:-default}"   # opacity task; "matched" = register-tell ablation
JUDGE="${JUDGE:-}"                 # harvest only; recorded in the log, used at score time
HARVEST_ARM="${HARVEST_ARM:-open}" # harvest only: open (must write up), closed (may decline), selfcheck (checks its own write-up)
LOGS="${LOGS:-runs/inspect/scored/*harvest*.eval}"  # crosscheck only: harvest logs (scored, so the judge's verdict rides along)
LEVELS_T=$(echo $LEVELS | tr ' ' ',')
TAGS="viking2,$RUN_TAG,$SLURM_JOB_ID"
PLAN="${PLAN:-}"                   # a plan file (scripts/plans/*.txt): several runs against one served model

if [ "$TASK" = serve ]; then
  # Serve only, for a client elsewhere (the safe-tre-agent end-to-end test
  # on d2-1 through an SSH tunnel). The node and port go to a file only the
  # user can read; set SERVE_API_KEY so other cluster users cannot call it.
  [ -n "${SERVE_API_KEY:-}" ] || { echo "TASK=serve needs SERVE_API_KEY_FILE"; exit 2; }
  ( umask 077; printf '%s %s %s\n' "$(hostname)" "$PORT" "$MODEL" > "runs/serving-$RUN_TAG.txt" )
  echo "serving $MODEL on $(hostname):$PORT until the time limit"
  wait $VLLM_PID
  exit 0
fi

if [ -n "$PLAN" ]; then
  # One run per line: "<task> <inspect eval arguments>", e.g.
  #   harvest -T arm=closed -T replicates=3 -T thinking=off -T judge=openai-api/mimo/mimo-v2.6-pro --no-score
  # Blank lines and lines starting with # are skipped. Every run gets this
  # job's model, log directory and tags, plus plan:<name> and step<n>; a
  # failed step is reported and the plan goes on to the next.
  # The plan is read once, from this job's own copy. Home is NFS: a plan file
  # replaced by a sync while the job held it open gave "Stale file handle" on
  # the next read, and the old read loop took the failed read for a repeat of
  # the last line (job 37905877 re-ran step 1, 2026-10-22).
  cp "$PLAN" "runs/plan-$RUN_TAG.txt"
  mapfile -t plan_lines < "runs/plan-$RUN_TAG.txt"
  step=0; failed=0
  for line in "${plan_lines[@]}"; do
    line="${line%%$'\r'}"
    case "$line" in ''|'#'*) continue ;; esac
    step=$((step+1))
    set -f; args=($line); set +f
    task="${args[0]}"; rest=("${args[@]:1}")
    logdir=runs/inspect; [ "$task" = crosscheck ] && logdir=runs/inspect/crosscheck
    echo "[plan step $step] $task ${rest[*]}"
    if ! inspect eval "opacity/$task" "${rest[@]}" --model "openai-api/vllm/$MODEL" --log-dir "$logdir" \
         --max-connections "${MAX_CONNECTIONS:-16}" --tags "$TAGS,plan:$(basename "$PLAN" .txt),step$step" </dev/null; then
      echo "[plan step $step] FAILED"; failed=$((failed+1))
    else
      # inspect exits 0 when a task errors; read the status from this step's
      # own log, found by this job's run tag and step number
      st=$(python - "$logdir" "$RUN_TAG" "step$step" <<'PY'
import glob, os, sys
from inspect_ai.log import read_eval_log
d, tag, step = sys.argv[1:4]
st = "missing"
for f in sorted(glob.glob(os.path.join(d, "*.eval")), key=os.path.getmtime, reverse=True)[:40]:
    h = read_eval_log(f, header_only=True)
    if tag in (h.eval.tags or []) and step in (h.eval.tags or []):
        st = h.status
        break
print(st)
PY
)
      echo "[plan step $step] status $st"
      [ "$st" = success ] || failed=$((failed+1))
    fi
  done
  echo "done: plan $PLAN, $step steps, $failed failed ($MODEL)"
  exit 0
fi

case "$TASK" in
  opacity)
    # What a reviewer sees at each level, no model, seconds. Keeps a bespoke
    # JSONL and the rendered bundles beside the Inspect log.
    python scripts/run_phase0.py --out "runs/bundles-$RUN_TAG.jsonl" --mechanical \
      --levels $LEVELS --replicates "$REPLICATES" --dump-bundles "runs/bundles-$RUN_TAG"
    inspect eval opacity/opacity -T arm=llm -T levels="$LEVELS_T" -T replicates="$REPLICATES" -T thinking="$THINKING" -T register="$REGISTER" \
      --model "openai-api/vllm/$MODEL" --log-dir runs/inspect --max-connections 16 --tags "$TAGS"
    ;;
  harvest)
    # Compute nodes may not reach the judge's API, so generate here with
    # --no-score and score on the login node afterwards:
    #     inspect score runs/inspect/<log>.eval
    # The judge string is recorded in the log's task args, so set it now.
    [ -n "$JUDGE" ] || { echo "set JUDGE=<judge model string> for TASK=harvest"; exit 2; }
    inspect eval opacity/harvest -T judge="$JUDGE" -T thinking="$THINKING" -T replicates="$REPLICATES" -T arm="$HARVEST_ARM" \
      --model "openai-api/vllm/$MODEL" --log-dir runs/inspect --max-connections 16 --tags "$TAGS" --no-score
    echo "generated without scoring; on the login node run: inspect score runs/inspect/<newest>.eval"
    ;;
  crosscheck)
    # This model as the blind checker on every agent's harvested write-ups.
    # Needs no judge: the verdicts are read from the scored harvest logs named
    # by LOGS (rsync runs/inspect/scored/ from d2-1 first). Scored on the node.
    inspect eval opacity/crosscheck -T logs="$LOGS" -T thinking="$THINKING" \
      --model "openai-api/vllm/$MODEL" --log-dir runs/inspect/crosscheck --max-connections 16 --tags "$TAGS"
    ;;
  *) echo "TASK must be opacity, harvest or crosscheck"; exit 2 ;;
esac

echo "done: runs/inspect ($TASK, $MODEL)"
