#!/usr/bin/env bash
# The inside analyst end to end: safe-tre-agent's NIGHTPLAY question bank,
# run on d2-1 against a model served on Viking through an SSH tunnel.
#
#   scripts/chimp_e2e.sh <hf model id> <thinking: off|on> [repeats] [partition] [gpus]
#
# The question bank, its synthetic data (runs/chimp/nightplay-data, made by
# studies/nightplay/generate.py in safe-tre-agent) and the dossiers stay on
# d2-1; only the analyst's prompts reach the model. The serving job needs an
# API key, generated here for this run only, because other cluster users can
# reach a compute node's port. Thinking is set server-side
# (--default-chat-template-kwargs), because safe-tre-agent's client sends no
# switch; with thinking on, the reasoning parser keeps the reasoning out of
# the reply the agent parses.
#
# THINK_BUDGET=<tokens> (with thinking on) caps thinking per request, as the
# sizing study's "1k think" did: scripts/think_budget_proxy.py adds vLLM's
# thinking_token_budget to every chat completion, since safe-tre-agent's
# client sends no such field and vLLM has no server-side default for it.
# The run's tag then says think<tokens> instead of on.
#
# RULES=1 adds safe-tre-agent's DESIGN_RULES to the analyst protocol (--rules);
# ARMS="plain rules" runs both arms on one served model, PARALLEL=1 at once.
#
# Output: runs/chimp/<tag>-<arm>/rep<k>/ (one dossier per question, summary.json).
set -euo pipefail
# one BLAS thread per call: on a loaded workstation threaded OpenBLAS made the
# rank audit's small eigh/SVD calls about 90x slower (safe-tre-agent's web server
# pins it on import; the question bank does not import the web server)
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
cd "$(dirname "$0")/.."
MODEL="${1:?model}"; THINK="${2:-off}"; REPEATS="${3:-3}"; PARTITION="${4:-gpuplus}"; GPUS="${5:-1}"
STA="${STA:-../safe-tre-agent}"
DATA="${DATA:-$PWD/runs/chimp/nightplay-data}"
LPORT="${LPORT:-18$(( RANDOM % 900 + 100 ))}"
[ -z "${THINK_BUDGET:-}" ] || [ "$THINK" = on ] || { echo "THINK_BUDGET needs thinking on"; exit 2; }
TAG="chimp-$(date -u +%Y%m%d-%H%M)-$(basename "$MODEL")-${THINK_BUDGET:+think}${THINK_BUDGET:-$THINK}"
KEY=$(python3 -c 'import secrets; print(secrets.token_hex(24))')
case "$THINK" in
  off) KW='{"enable_thinking": false}' ;;
  on) KW='{"enable_thinking": true}' ;;
  *) echo "thinking must be off or on"; exit 2 ;;
esac
case "$PARTITION" in gpuplus) gres="gpu:h100:$GPUS" ;; *) gres="gpu:$GPUS" ;; esac
parser=""; case "$MODEL" in Qwen/*|nvidia/Qwen*) parser=qwen3 ;; openai/gpt-oss*) parser=openai_gptoss ;; google/gemma-4*) parser=gemma4 ;; esac

# the key goes to a mode-600 file over stdin, never onto a command line
KEYFILE="runs/serve-key-$TAG"
printf '%s' "$KEY" | ssh -o BatchMode=yes ${VIKING_HOST:?} "cd ~/git/opacity && umask 077 && cat > $KEYFILE"
jid=$(ssh -o BatchMode=yes ${VIKING_HOST:?} "cd ~/git/opacity && \
  SERVE_API_KEY_FILE=\$PWD/$KEYFILE DEFAULT_TEMPLATE_KWARGS='$KW' sbatch --parsable --partition=$PARTITION --gres=$gres \
  --time=${TIME:-03:00:00} --job-name=op-serve-$(basename "$MODEL") \
  --export=ALL,MODEL=$MODEL,TASK=serve,RUN_TAG=$TAG,TP=$GPUS,REASONING_PARSER=$parser,MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768} \
  scripts/viking2_vllm.slurm")
echo "$jid serve $MODEL $TAG $PARTITION x$GPUS $(date -u +%Y-%m-%dT%H:%MZ)" | ssh -o BatchMode=yes ${VIKING_HOST:?} 'cat >> ~/git/opacity/runs/sweep-submissions.txt'
trap 'ssh -o BatchMode=yes ${VIKING_HOST:?} "scancel $jid; rm -f ~/git/opacity/$KEYFILE ~/git/opacity/$KEYFILE.header" 2>/dev/null; [ -n "${TUN:-}" ] && kill $TUN 2>/dev/null; [ -n "${PROXY:-}" ] && kill $PROXY 2>/dev/null; true' EXIT
echo "serving job $jid ($TAG); waiting for the server"
for i in $(seq 1 "${WAIT_ITER:-1080}"); do   # 20 s each: up to six hours in the queue
  line=$(ssh -o BatchMode=yes ${VIKING_HOST:?} "cat ~/git/opacity/runs/serving-$TAG.txt 2>/dev/null" || true)
  [ -n "$line" ] && break
  st=$(ssh -o BatchMode=yes ${VIKING_HOST:?} "squeue -h -j $jid -o %T" || true)
  [ -z "$st" ] && { echo "job $jid ended before serving"; exit 1; }
  sleep 20
done
[ -n "$line" ] || { echo "job $jid never started serving"; exit 1; }
read -r node port served <<<"$line"
ssh -o BatchMode=yes -N -L "127.0.0.1:$LPORT:$node:$port" ${VIKING_HOST:?} & TUN=$!
sleep 5
curl -sf -H "Authorization: Bearer $KEY" "http://127.0.0.1:$LPORT/v1/models" | grep -qF "\"$MODEL\"" \
  || { echo "tunnel to $node:$port does not reach $MODEL"; exit 1; }
echo "tunnel up: 127.0.0.1:$LPORT -> $node:$port ($MODEL, thinking $THINK)"
URL="http://127.0.0.1:$LPORT/v1"
if [ -n "${THINK_BUDGET:-}" ]; then
  PPORT=$((LPORT + 1))
  python3 scripts/think_budget_proxy.py "$PPORT" "http://127.0.0.1:$LPORT" "$THINK_BUDGET" & PROXY=$!
  sleep 2
  curl -sf -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PPORT/v1/models" | grep -qF "\"$MODEL\"" \
    || { echo "budget proxy on $PPORT does not reach $MODEL"; exit 1; }
  URL="http://127.0.0.1:$PPORT/v1"
  echo "thinking budget $THINK_BUDGET tokens through 127.0.0.1:$PPORT"
fi

# ARMS="plain rules" runs both prompt arms against the one served model;
# PARALLEL=1 runs every (arm, repeat) at once (vLLM batches them)
ARMS="${ARMS:-$([ -n "${RULES:-}" ] && echo rules || echo plain)}"
run_one() {
  local arm=$1 k=$2 out="$PWD/runs/chimp/$TAG-$1/rep$2"
  mkdir -p "runs/chimp/$TAG-$arm"
  ( cd "$STA" && SAFETRE_LLM=real SAFETRE_LLM_BASE_URL="$URL" SAFETRE_LLM_MODEL="$MODEL" \
      SAFETRE_LLM_API_KEY="$KEY" SAFETRE_LLM_TIMEOUT="${SAFETRE_LLM_TIMEOUT:-600}" \
      .venv/bin/python studies/nightplay/run_question_bank.py --data "$DATA" --out "$out" \
      $([ "$arm" = rules ] && echo --rules) ) > "$out.log" 2>&1
  grep -E "verdict agreement|MISS|Error" "$out.log" | sed "s/^/$arm rep$k: /" || true
}
pids=()
for arm in $ARMS; do
  for k in $(seq 1 "$REPEATS"); do
    if [ -n "${PARALLEL:-}" ]; then run_one "$arm" "$k" & pids+=($!); else run_one "$arm" "$k"; fi
  done
done
# only the runs: a bare `wait` also waits for the tunnel, which never exits,
# and held the serving job to its time limit (23 October)
[ ${#pids[@]} -eq 0 ] || wait "${pids[@]}"
echo "done: runs/chimp/$TAG-{$(echo $ARMS | tr ' ' ,)}"
