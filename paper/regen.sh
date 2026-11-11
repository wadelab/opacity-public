#!/usr/bin/env bash
# Regenerate this paper's generated inputs from the repository's record.
# Called by build.sh (which is shared with safe-tre-agent/paper); this file
# is specific to the OPACITY repository.
#
#   numbers.tex, models.tex   paper/make_numbers.py, from runs/summary, runs/probe,
#                             runs/vetting, runs/chimp and the task files (it also
#                             writes paper-vetting/models.tex; paper-vetting links
#                             to this numbers.tex)
#   instructions.tex          paper/instructions.py, from the package's prompts
#                             (and paper-vetting/instructions.tex)
#   figures/*.pdf             copied from docs/figures (drawn by the scripts
#                             linked from paper/figure_scripts/)
#
#   OPACITY_REGEN_FIGURES=1 paper/build.sh   also redraws every figure from its
#                                            script first (needs the run logs)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="$ROOT/.venv/bin/python"

FIGURES=(fig_hosted_levels fig_class_map fig_spine fig_harvest fig_roc fig_calibration
         sizing_prompt sizing_thinking sizing_fresh sizing_auc sizing_checker sizing_analyst sizing_frontier)
# the probe, bias and pilot figures belong to the second paper, ../paper-vetting (its own regen.sh)

if [[ "${OPACITY_REGEN_FIGURES:-0}" == "1" ]]; then
    for script in "$HERE"/figure_scripts/*.py; do
        echo "regen.sh: $(readlink -f "$script")"
        (cd "$ROOT" && "$PY" "$(readlink -f "$script")" 2>&1 | grep -v '^findfont' || true)
    done
fi

"$PY" "$HERE/make_numbers.py"
"$PY" "$HERE/instructions.py"

mkdir -p "$HERE/figures"
for f in "${FIGURES[@]}"; do
    src="$ROOT/docs/figures/$f.pdf"
    [[ -f "$src" ]] || { echo "regen.sh: missing figure $src (set OPACITY_REGEN_FIGURES=1)" >&2; exit 1; }
    cp "$src" "$HERE/figures/$f.pdf"
done
echo "regen.sh: ${#FIGURES[@]} figures copied"
