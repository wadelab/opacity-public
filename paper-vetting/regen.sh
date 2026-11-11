#!/usr/bin/env bash
# Regenerate the vetting paper's generated inputs from the repository's record.
# Called by build.sh (shared with paper/ and safe-tre-agent/paper); this file
# is specific to this paper. The generators live in ../paper and write both
# papers' inputs, so the two papers cannot quote different numbers.
#
#   numbers.tex               a link to ../paper/numbers.tex (paper/make_numbers.py)
#   models.tex                paper/make_numbers.py, this paper's models only
#   instructions.tex          paper/instructions.py, this paper's instructions
#   figures/*.pdf             copied from docs/figures (drawn by the scripts
#                             linked from figure_scripts/)
#
#   OPACITY_REGEN_FIGURES=1 paper-vetting/build.sh   also redraws every figure
#                                                    from its script first
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="$ROOT/.venv/bin/python"

FIGURES=(probe_reading probe_steer slant_vetting style_robustness lora_pilot)

if [[ "${OPACITY_REGEN_FIGURES:-0}" == "1" ]]; then
    for script in "$HERE"/figure_scripts/*.py; do
        echo "regen.sh: $(readlink -f "$script")"
        (cd "$ROOT" && "$PY" "$(readlink -f "$script")" 2>&1 | grep -v '^findfont' || true)
    done
fi

"$PY" "$ROOT/paper/make_numbers.py"
"$PY" "$ROOT/paper/instructions.py"

mkdir -p "$HERE/figures"
for f in "${FIGURES[@]}"; do
    src="$ROOT/docs/figures/$f.pdf"
    [[ -f "$src" ]] || { echo "regen.sh: missing figure $src (set OPACITY_REGEN_FIGURES=1)" >&2; exit 1; }
    cp "$src" "$HERE/figures/$f.pdf"
done
echo "regen.sh: ${#FIGURES[@]} figures copied"
