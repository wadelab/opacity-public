#!/bin/bash
# Convenience entry point only. No job is submitted until this script is invoked.
# Use the isolated host environment, e.g. HOST_PYTHON=/path/to/pilot-venv/bin/python.
set -euo pipefail
: "${HOST_PYTHON:?set HOST_PYTHON to the isolated pilot environment's interpreter}"
export UV_PYTHON="$HOST_PYTHON"
exec uv run --offline --no-project "$HOST_PYTHON" -m opacity.lora_pilot submit "$@"
