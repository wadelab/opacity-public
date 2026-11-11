#!/usr/bin/env bash
# The inside analyst end to end against a HOSTED OpenAI-compatible model:
# safe-tre-agent's NIGHTPLAY question bank, run on d2-1. The sibling of
# chimp_e2e.sh (which serves a model on Viking) for a quick test without a GPU
# job, using a hosted stand-in for a model that could run inside the TRE.
#
#   scripts/chimp_hosted.sh <env file> <model> [repeats]
#   RULES=1 scripts/chimp_hosted.sh ~/.config/mimo/api.env mimo-v2.6-flash 3
#
# The env file must define <PREFIX>_API_KEY and <PREFIX>_BASE_URL; PREFIX is
# taken from the file's first *_BASE_URL line (MIMO for mimo/api.env, which
# also holds keys for other providers), or set PREFIX=. NIGHTPLAY
# is synthetic, which is the only reason a remote endpoint is allowed here
# (SAFETRE_ALLOW_REMOTE_LLM=1). RULES=1 adds safe-tre-agent's DESIGN_RULES to
# the analyst protocol (--rules). PARALLEL=1 runs the repeats concurrently.
# A hosted endpoint rate-limits (MiMo returned 429s to seven concurrent runs
# on 23 Oct), so 429s and 5xx are retried six times from 5 s, doubling.
#
# Output: runs/chimp/<tag>/rep<k>/ (one dossier per question, summary.json).
set -euo pipefail
# one BLAS thread per call: on a loaded workstation threaded OpenBLAS made the
# rank audit's small eigh/SVD calls about 90x slower (safe-tre-agent's web server
# pins it on import; the question bank does not import the web server)
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
cd "$(dirname "$0")/.."
ENVFILE="${1:?env file}"; MODEL="${2:?model}"; REPEATS="${3:-3}"
STA="${STA:-../safe-tre-agent}"
DATA="${DATA:-$PWD/runs/chimp/nightplay-data}"
prefix="${PREFIX:-$(grep -oE '^[A-Z]+_BASE_URL=' "$ENVFILE" | head -1 | sed 's/_BASE_URL=$//')}"
[ -n "$prefix" ] || { echo "no *_BASE_URL in $ENVFILE; set PREFIX"; exit 2; }
set -a; source "$ENVFILE"; set +a
key_var="${prefix}_API_KEY"; url_var="${prefix}_BASE_URL"
base_url="${!url_var:?$url_var is not set in $ENVFILE}"
arm=$([ -n "${RULES:-}" ] && echo rules || echo plain)
TAG="chimp-$(date -u +%Y%m%d-%H%M)-$(basename "$MODEL")-hosted-$arm"
mkdir -p "runs/chimp/$TAG"
echo "$TAG: $MODEL via ${base_url%%/v1*} (rules: $arm), $REPEATS repeat(s)"
rep() {
  local out="$PWD/runs/chimp/$TAG/rep$1"
  .venv/bin/python -m opacity.offpeak wait   # hosted calls off-peak only (opacity/offpeak.py)
  ( cd "$STA" && SAFETRE_LLM=real SAFETRE_ALLOW_REMOTE_LLM=1 \
      SAFETRE_LLM_BASE_URL="$base_url" SAFETRE_LLM_MODEL="$MODEL" \
      SAFETRE_LLM_API_KEY="${!key_var}" SAFETRE_LLM_TIMEOUT="${SAFETRE_LLM_TIMEOUT:-300}" \
      SAFETRE_LLM_RETRIES="${SAFETRE_LLM_RETRIES:-6}" SAFETRE_LLM_RETRY_BACKOFF="${SAFETRE_LLM_RETRY_BACKOFF:-5}" \
      .venv/bin/python studies/nightplay/run_question_bank.py --data "$DATA" --out "$out" \
      ${RULES:+--rules} ) > "$out.log" 2>&1
  grep -E "verdict agreement|MISS|Error" "$out.log" | sed "s/^/rep$1: /" || true
}
# PARALLEL=1 runs the repeats at once (each has its own session and audit db)
for k in $(seq 1 "$REPEATS"); do
  if [ -n "${PARALLEL:-}" ]; then rep "$k" & else rep "$k"; fi
done
wait
echo "done: runs/chimp/$TAG"
