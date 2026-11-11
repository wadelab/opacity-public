#!/usr/bin/env bash
# Checks on a built paper, run after build.sh. Findings are printed; the
# first two fail the script, the rest are for the author to read.
#
#   1. a number macro the manuscript uses that make_numbers.py did not define
#   2. a "??" or "[?]" surviving in the PDF text
#   3. figure discipline (regenerable, legible at final size, cited in order,
#      fonts embedded): ~/.claude/skills/figure-discipline
#   4. LLM tells in the prose: ~/.claude/skills/writing-style
#   5. every reference exists in Crossref: ~/.claude/skills/reference-verification
#
# Also copies main.pdf to docs/build/<name>.pdf, which is committed: opacity-preprint
# for paper/, opacity-vetting-preprint for paper-vetting/ (which links to this script).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$(basename "$HERE")" in
    paper) NAME=opacity-preprint ;;
    paper-vetting) NAME=opacity-vetting-preprint ;;
    *) echo "check.sh: unknown paper directory $HERE" >&2; exit 2 ;;
esac
ROOT="$(cd "$HERE/.." && pwd)"
PY="$ROOT/.venv/bin/python"
SKILLS="$HOME/.claude/skills"
cd "$HERE"
[[ -f main.pdf && -f main.log ]] || { echo "check.sh: build first ($(basename "$HERE")/build.sh)" >&2; exit 2; }

fail=0
if grep -q 'undefined number macro' main.log; then
    echo "== undefined number macros:"; grep -o 'undefined number macro [^ ]*' main.log | sort -u; fail=1
fi
if command -v pdftotext >/dev/null && pdftotext main.pdf - | grep -qE '\?\?|\[\?\]'; then
    echo "== ?? in the PDF:"; pdftotext main.pdf - | grep -nE '\?\?|\[\?\]' | head; fail=1
fi
[[ "$fail" -eq 0 ]] || exit 1

mkdir -p "$ROOT/docs/build"
cp main.pdf "$ROOT/docs/build/$NAME.pdf"
echo "== copied main.pdf to docs/build/$NAME.pdf ($(pdfinfo main.pdf 2>/dev/null | awk '/^Pages:/ {print $2}') pages)"

echo "== figure discipline (floors 7 pt and 300 dpi: a preprint has no journal floor; these are Nature's)"
# the checker reads one file and does not follow \input, so it sees a flattened copy
"$PY" - <<'PYEOF'
import re
from pathlib import Path
def flat(path):
    text = Path(path).read_text()
    return re.sub(r"\\input\{([^}]+)\}", lambda m: flat((Path(m.group(1)).with_suffix(".tex"))) if Path(m.group(1)).with_suffix(".tex").exists() else m.group(0), text)
# resolve \ref{} to the numbers LaTeX assigned, from main.aux, so the checker sees "Figure 2a"
labels = {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", Path("main.aux").read_text())}
text = flat("main.tex")
text = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels.get(m.group(1), m.group(0)), text)
Path("flat.tex").write_text(text)
PYEOF
python3 "$SKILLS/figure-discipline/scripts/check_figures.py" flat.tex --figures figures \
    --scripts figure_scripts --min-font 7 --min-dpi 300 --textwidth 419.5 --quiet || true
rm -f flat.tex
echo "== LLM tells"
python3 "$SKILLS/writing-style/scripts/find_tells.py" sections/*.tex || true
if [[ "${1:-}" == "--refs" ]]; then
    echo "== references (Crossref)"
    "$PY" "$SKILLS/reference-verification/scripts/check_existence.py" references.bib || true
fi
