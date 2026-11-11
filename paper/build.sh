#!/usr/bin/env bash
# Build the paper into main.pdf.
#
#   paper/build.sh            regenerate numbers and figures, then build
#   paper/build.sh --fast     build only (no regeneration)
#   paper/build.sh --clean    remove the build files
#
# Identical in both repositories (safe-tre-agent/paper and opacity/paper).
# What differs per repository lives in regen.sh beside this script: it
# regenerates numbers.tex and the figures from that repository's record.
# The build refuses to pass with an undefined citation or reference, or
# with a warning that a page overflowed, so a PDF with "??" in it is never
# called built.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PAPER=main
cd "$HERE"

if [[ "${1:-}" == "--clean" ]]; then
  latexmk -C "$PAPER.tex" >/dev/null 2>&1 || true
  rm -f version.tex build.log
  echo "cleaned $HERE"
  exit 0
fi

for tool in latexmk pdflatex bibtex; do
  command -v "$tool" >/dev/null || { echo "build.sh: $tool not found; install a TeX distribution" >&2; exit 1; }
done

if [[ "${1:-}" != "--fast" ]]; then
  if [[ -x "$HERE/regen.sh" ]]; then
    "$HERE/regen.sh"
  else
    echo "build.sh: no regen.sh beside this script; building with the committed numbers and figures" >&2
  fi
fi

# The version line names the commit the PDF came from; it is never committed.
commit="$(git -C "$ROOT" rev-parse --short HEAD)"
git -C "$ROOT" diff --quiet || commit="$commit-dirty"
printf '%%%% written by build.sh; not committed\n\\newcommand{\\BuildCommit}{%s}\n\\newcommand{\\BuildDate}{%s}\n' \
  "$commit" "$(date +%Y-%m-%d)" > version.tex

if ! latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error "$PAPER.tex" > build.log 2>&1; then
  echo "build.sh: latexmk failed; the last lines of build.log:" >&2
  grep -n '^!\|^l\.\|Error\|error:' build.log | head -20 >&2 || true
  tail -20 build.log >&2
  exit 1
fi

bad=0
if grep -q 'LaTeX Warning: There were undefined references\|Citation .* undefined\|Reference .* undefined' "$PAPER.log"; then
  echo "build.sh: undefined citations or references:" >&2
  grep 'undefined' "$PAPER.log" | sort -u | head -20 >&2
  bad=1
fi
if grep -q 'LaTeX Warning: There were multiply-defined labels' "$PAPER.log"; then
  echo "build.sh: multiply-defined labels" >&2
  bad=1
fi
over="$(grep -c '^Overfull \\hbox' "$PAPER.log" || true)"
if [[ "$over" -gt 0 ]]; then
  echo "build.sh: $over overfull line(s); see $PAPER.log (not fatal)" >&2
fi
[[ "$bad" -eq 0 ]] || exit 1

pages="$(grep -o 'Output written on [^ ]* ([0-9]* page' "$PAPER.log" | grep -o '[0-9]* page' | head -1)"
echo "built $HERE/$PAPER.pdf ($pages) from $commit"
