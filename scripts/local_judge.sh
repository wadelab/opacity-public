#!/usr/bin/env bash
# A local judge: serve a model on Viking, tunnel to it from d2-1, and append
# its verdicts to scored harvest logs as `judged_alt` (inspect score --action
# append), beside MiMo's `judged` and DeepSeek's `judged_second`.
#
#   scripts/local_judge.sh <hf model id> <partition> <gpus> <log> [<log> ...]
#
# Written on 23 October 2026, when the hosted judges were paused:
# nothing here leaves the workstation or Viking. The logs stay on d2-1; only
# the judge prompts (pack, write-up, key) reach the served model. A log that
# already carries `judged_alt` is skipped. The serving job needs an API key,
# generated here for this run only, because other cluster users can reach a
# compute node's port; it reaches Viking over stdin into a mode-600 file.
#
# EFFORT sets the judge's reasoning effort (gpt-oss: low, medium, high;
# default high). PAR logs are scored at once (default 4). TIME is the serving
# job's wall limit (default 04:00:00).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?model}"; PARTITION="${2:?partition}"; GPUS="${3:?gpus}"; shift 3
[ $# -ge 1 ] || { echo "no logs given"; exit 2; }
EFFORT="${EFFORT:-high}"; PAR="${PAR:-4}"
# A free local port from the OS: a random 19xxx collided on 7 November with an R
# session listening on 19649, the tunnel never bound and the run judged nothing.
LPORT="${LPORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}"
TAG="judge-$(date -u +%Y%m%d-%H%M)-$(basename "$MODEL")"
KEY=$(python3 -c 'import secrets; print(secrets.token_hex(24))')
case "$PARTITION" in gpuplus) gres="gpu:h100:$GPUS" ;; *) gres="gpu:$GPUS" ;; esac
parser=""; case "$MODEL" in Qwen/*|nvidia/Qwen*) parser=qwen3 ;; openai/gpt-oss*) parser=openai_gptoss ;; google/gemma-4*) parser=gemma4 ;; esac

KEYFILE="runs/serve-key-$TAG"
printf '%s' "$KEY" | ssh -o BatchMode=yes ${VIKING_HOST:?} "cd ~/git/opacity && umask 077 && cat > $KEYFILE"
jid=$(ssh -o BatchMode=yes ${VIKING_HOST:?} "cd ~/git/opacity && \
  SERVE_API_KEY_FILE=\$PWD/$KEYFILE sbatch --parsable --partition=$PARTITION --gres=$gres \
  --time=${TIME:-04:00:00} --job-name=op-judge-$(basename "$MODEL") \
  --export=ALL,MODEL=$MODEL,TASK=serve,RUN_TAG=$TAG,TP=$GPUS,REASONING_PARSER=$parser,MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768},SERVE_WAIT=${SERVE_WAIT:-240} \
  scripts/viking2_vllm.slurm")
echo "$jid judge-serve $MODEL $TAG $PARTITION x$GPUS $(date -u +%Y-%m-%dT%H:%MZ)" | ssh -o BatchMode=yes ${VIKING_HOST:?} 'cat >> ~/git/opacity/runs/sweep-submissions.txt'
trap 'ssh -o BatchMode=yes ${VIKING_HOST:?} "scancel $jid; rm -f ~/git/opacity/$KEYFILE ~/git/opacity/$KEYFILE.header" 2>/dev/null; [ -n "${TUN:-}" ] && kill $TUN 2>/dev/null; true' EXIT
echo "serving job $jid ($TAG); waiting for the server"
line=""
for i in $(seq 1 "${WAIT_ITER:-1080}"); do   # 20 s each: up to six hours in the queue
  line=$(ssh -o BatchMode=yes ${VIKING_HOST:?} "cat ~/git/opacity/runs/serving-$TAG.txt 2>/dev/null" || true)
  [ -n "$line" ] && break
  st=$(ssh -o BatchMode=yes ${VIKING_HOST:?} "squeue -h -j $jid -o %T" || true)
  [ -z "$st" ] && { echo "job $jid ended before serving"; exit 1; }
  sleep 20
done
[ -n "$line" ] || { echo "job $jid never started serving"; exit 1; }
read -r node port served <<<"$line"
ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N -L "127.0.0.1:$LPORT:$node:$port" ${VIKING_HOST:?} & TUN=$!
sleep 5
export VLLM_BASE_URL="http://127.0.0.1:$LPORT/v1" VLLM_API_KEY="$KEY"
.venv/bin/python - "$MODEL" <<'PY' || { echo "tunnel does not reach $MODEL"; exit 1; }
import os, sys, urllib.request
req = urllib.request.Request(os.environ["VLLM_BASE_URL"] + "/models",
                             headers={"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]})
sys.exit(0 if f'"{sys.argv[1]}"' in urllib.request.urlopen(req, timeout=30).read().decode() else 1)
PY
echo "tunnel up: 127.0.0.1:$LPORT -> $node:$port ($MODEL, effort $EFFORT)"

score_one() {
  local f="$1"
  if .venv/bin/python -c "import sys; from inspect_ai.log import read_eval_log as r; l=r(sys.argv[1]); sys.exit(0 if any('judged_alt' in (s.scores or {}) for s in l.samples or []) else 1)" "$f"; then
    echo "### SKIP $(basename "$f") (has judged_alt)"; return
  fi
  echo "### JUDGE $(basename "$f") $(date -u +%H:%M)"
  .venv/bin/inspect score "$f" --overwrite --action append --scorer scripts/harvest_judge.py@judged_alt \
    -S judge="openai-api/vllm/$MODEL" -S max_tokens=20000 -S reasoning_effort="$EFFORT" --display plain </dev/null 2>&1 \
    | grep -iE 'error|traceback' | head -3 || true   # no match is the good case, not a failure
  echo "### DONE $(basename "$f") $(date -u +%H:%M)"
}
# a failed log must not end the run (set -e would, and the trap would then
# cancel the server under the other logs): wait without -e
set +e
pids=()
for f in "$@"; do
  score_one "$f" & pids+=($!)
  while [ "$(jobs -rp | grep -vc "^$TUN$")" -ge "$PAR" ]; do sleep 5; done
done
wait "${pids[@]}"
echo "all done $(date -u +%H:%M)"
# exit 1 if any log still lacks judged_alt, so a caller cannot take an unjudged log as judged
missing=0
for f in "$@"; do
  .venv/bin/python -c "import sys; from inspect_ai.log import read_eval_log as r; l=r(sys.argv[1]); sys.exit(0 if any('judged_alt' in (s.scores or {}) for s in l.samples or []) else 1)" "$f" \
    || { echo "### UNJUDGED $(basename "$f")"; missing=1; }
done
exit $missing
