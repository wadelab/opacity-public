#!/usr/bin/env python
"""Render the preprint as one HTML page, for publishing as an artifact.

Reads the same sections as main.tex, resolves every \\N{key} from numbers.tex
and every \\ref from main.aux (so the page carries the numbers the PDF
carries), numbers sections from main.toc, renders the TikZ overview to a PNG,
and converts with pandoc (citeproc on references.bib, MathML for the few
formulas). Output: paper/html/opacity-preprint.html and paper/html/figures/.

    paper/build.sh && .venv/bin/python paper/render_html.py

A built copy of another revision (one that carries its own sections/,
numbers.tex, instructions.tex, models.tex, references.bib and main.aux)
renders the same way with --paper <dir>; --out and --name place the page.
The title and the sections are read from that revision's main.tex. The
second OPACITY paper renders the same way:

    paper-vetting/build.sh && .venv/bin/python paper/render_html.py --paper paper-vetting \
        --name opacity-vetting-preprint.html
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
OUT = PAPER / "html"
FIGS = ROOT / "docs" / "figures"
SECTIONS = ["map", "introduction", "background", "methods", "results-checkers", "results-analysts",
            "discussion", "limitations", "reproducibility", "acknowledgements", "glossary"]
MACROS: list[str] = []      # .tex files of \\newcommand macros handed to pandoc, which expands them
APPENDIX = True
CSL: Path | None = None
TITLE = "Evaluating capabilities and safety of local LLMs for safe statistical analysis"
AUTHOR = "Alex R. Wade"
AFFIL = "Department of Psychology and York Neuroimaging Centre, University of York, UK"


def read(name: str) -> str:
    return (PAPER / name).read_text()


def numbers() -> dict[str, str]:
    if not (PAPER / "numbers.tex").exists():
        return {}
    return dict(re.findall(r"\\opnum@def\{([^}]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}", read("numbers.tex")))


def labels() -> dict[str, str]:
    return {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", read("main.aux"))}


def section_numbers() -> dict[str, str]:
    out = {}
    for m in re.finditer(r"\\contentsline \{(sub)?section\}\{\\numberline \{([^}]+)\}([^}]*)\}", read("main.aux")):
        out[m.group(3).strip()] = m.group(2)
    return out


def commit() -> str:
    v = PAPER / "version.tex"
    if v.exists():
        m = re.search(r"\\BuildCommit\}\{([^}]+)\}", v.read_text())
        if m:
            return m.group(1)
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def latex_source() -> str:
    N, L, S = numbers(), labels(), section_numbers()
    # pandoc turns the abstract environment into metadata and drops it from the body; an unknown
    # environment becomes a div with that name as its class
    abstract = read("sections/abstract.tex")
    if "\\begin{abstract}" in abstract:
        abstract = abstract.replace("\\begin{abstract}", "").replace("\\end{abstract}", "").replace("\\noindent", "")
    parts = [read(m) for m in MACROS]
    parts.append("\\begin{abstractblock}\n" + abstract + "\n\\end{abstractblock}\n")
    parts += [read(f"sections/{s}.tex") for s in SECTIONS if s != "abstract"]
    if APPENDIX:
        parts.append("\\appendix\n" + read("sections/appendix.tex"))
    t = "\n".join(parts)

    # generated inputs
    if (PAPER / "instructions.tex").exists():
        t = t.replace("\\input{instructions}", read("instructions.tex"))
    if (PAPER / "models.tex").exists():
        rows = re.search(r"\\newcommand\{\\modelrows\}\{%\n(.*)\}\n", read("models.tex"), re.S).group(1)
        t = t.replace("\\input{models}\n", "").replace("\\modelrows", rows)
    t = re.sub(r"\\input\{figures/([A-Za-z0-9_-]+)\}", r"\\includegraphics[width=\\textwidth]{\1}", t)

    # numbers, levels and the shared semantic macros
    t = re.sub(r"\\N\{([^}]+)\}", lambda m: N.get(m.group(1), "??" + m.group(1)), t)
    t = re.sub(r"\\level\{(\d)\}", r"s\1", t)
    t = re.sub(r"\\sys\{([^}]*)\}", r"\\textit{\1}", t)
    t = re.sub(r"\\term\{([^}]*)\}", r"\\emph{\1}", t)
    t = t.replace("\\tbd{}", "TBD").replace("\\tbd", "TBD")
    t = re.sub(r"\\clause\{([^}]*)\}", r"\\texttt{\1}", t)
    t = re.sub(r"\\mark(?:honest|biased)\{([^}]*)\}", r"\\emph{\1}", t)  # main.tex's marked phrases
    t = re.sub(r"\\tnote\{([^}]*)\}", r"\\textsuperscript{\1}", t)
    t = re.sub(r"\\item\[([a-z])\]", r"\1. ", t)
    t = t.replace("\\BuildCommit", commit())

    # cross references, resolved to the numbers the PDF build assigned
    t = t.replace("\\S\\ref{", "Section~\\ref{")
    t = re.sub(r"\\ref\{([^}]+)\}", lambda m: L.get(m.group(1), "??"), t)

    # section numbers from the table of contents (unnumbered sections stay as they are)
    def number_heading(m):
        kind, title = m.group(1), m.group(2)
        n = S.get(title.strip())
        return f"\\{kind}{{{n} {title}}}" if n else m.group(0)
    t = re.sub(r"\\(section|subsection)\{([^}]*)\}", number_heading, t)
    t = re.sub(r"\\paragraph\{([^}]*)\}", r"\\textbf{\1} ", t)

    # floats: number the captions as the build did, drop threeparttable and sizing commands
    def number_float(m):
        body = m.group(0)
        lab = re.search(r"\\label\{((fig|tab):[^}]+)\}", body)
        if lab and lab.group(1) in L:
            word = "Figure" if lab.group(2) == "fig" else "Table"
            body = body.replace("\\caption{", f"\\caption{{{word} {L[lab.group(1)]}. ", 1)
        body = re.sub(r"\\centering(\\footnotesize|\\small)?(\\setlength\{\\tabcolsep\}\{[^}]*\})?\n?", "", body)
        body = body.replace("\\begin{threeparttable}\n", "").replace("\\end{threeparttable}\n", "")
        notes = re.search(r"\\begin\{tablenotes\}\\footnotesize\n(.*?)\\end\{tablenotes\}\n", body, re.S)
        if notes:
            text = re.sub(r"\\item\s*", " ", notes.group(1)).strip()
            body = body.replace(notes.group(0), "")
            body = body + "\n\\begin{quote}" + text + "\\end{quote}\n"
        body = body.replace("\\\\[2pt]", "\\\\")
        if lab and lab.group(1) == "tab:models":
            body = models_table(body)
        return body
    t = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}\n", number_float, t, flags=re.S)
    t = t.replace("\\code{", "\\texttt{")  # after the floats pass, which reads the models rows as written

    # breakout boxes: pandoc turns the unknown environment into a div; the title goes in as a bold line
    box_n = [0]
    def box(m):
        box_n[0] += 1
        return f"\\begin{{examplebox}}\\textbf{{Box {box_n[0]}. {m.group(1)}}}\\par\n"
    t = re.sub(r"\\begin\{examplebox\}(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}\n?", box, t)
    # glossary and verbatim wrappers
    t = t.replace("\\begin{glossarylist}", "\\begin{description}").replace("\\end{glossarylist}", "\\end{description}")
    t = re.sub(r"\\gterm\{([^}]*)\}\{((?:[^{}]|\{[^{}]*\})*)\}", r"\\item[\1] \2", t)
    t = t.replace("\\begin{footnotesize}\\begin{verbatim}", "\\begin{verbatim}")
    t = t.replace("\\end{verbatim}\\end{footnotesize}", "\\end{verbatim}")
    t = t.replace("\\end{verbatim}\n\\end{footnotesize}", "\\end{verbatim}")
    t = re.sub(r"\\setcounter\{[^}]*\}\{[^}]*\}|\\renewcommand\{\\the(?:figure|table)\}\{(?:[^{}]|\{[^{}]*\})*\}", "", t)
    t = t.replace("\\appendix\n", "")
    return t


def models_table(body: str) -> str:
    """Four columns, repository beside the model, in place of the PDF's two-line rows with \\multicolumn."""
    body, n = re.subn(r"^(.*?) & (\w*) & (\\code\{[^}]*\}) \\\\ \\multicolumn\{3\}\{@\{\}l\}\{\\quad(\\code\{[^}]*\})\} \\\\$",
                      r"\1 & \4 & \2 & \3 \\\\", body, flags=re.M)
    header = "Model, then its Hugging Face repository & Roles & Revision"
    if not n or header not in body or body.count("{@{}lll@{}}") != 1:
        raise SystemExit("models table: rows, header or column spec not in the expected form")
    return body.replace("{@{}lll@{}}", "{@{}llll@{}}").replace(header, "Model & Repository & Roles & Revision")


def style_macros() -> set[str]:
    """Names defined by wadelab.sty; pandoc drops any it meets, argument and all."""
    sty = (PAPER / "wadelab.sty").read_text()
    return set(re.findall(r"\\(?:new|renew|provide)command\*?\s*\{?\\([A-Za-z]+)\}?", sty))


def tikz_figures() -> list[str]:
    """Names of the TikZ figures the sections input, figures/<name>.tex."""
    names = []
    for s in SECTIONS + (["appendix"] if APPENDIX else []):
        if (PAPER / "sections" / f"{s}.tex").exists():
            names += re.findall(r"\\input\{figures/([A-Za-z0-9_-]+)\}", read(f"sections/{s}.tex"))
    return names


def tikz_png(name: str) -> Path:
    """Compile one TikZ figure on its own and rasterise it. The --macros files are
    input first, so a drawing may use the paper's number and example macros."""
    work = OUT / "tikz-build" / name
    work.mkdir(parents=True, exist_ok=True)
    macros = "".join(f"\\input{{{(PAPER / m).resolve()}}}\n" for m in MACROS)
    (work / "overview.tex").write_text(
        "\\documentclass[11pt,a4paper]{article}\n\\usepackage{wadelab}\n" + macros + "\\pagestyle{empty}\n"
        "\\begin{document}\n\\noindent\\input{overview-figure}\n\\end{document}\n")
    shutil.copy(PAPER / "figures" / f"{name}.tex", work / "overview-figure.tex")
    env = {"TEXINPUTS": f"{PAPER}:", "PATH": "/usr/bin:/bin:/usr/local/bin"}
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "overview.tex"], cwd=work, env=env, check=True,
                   capture_output=True)
    png = OUT / "figures" / f"{name}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    # crop to the drawing: bounding box from Ghostscript, applied as pdftoppm's pixel window
    bbox = subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=bbox", "overview.pdf"], cwd=work,
                          capture_output=True, text=True).stderr
    x0, y0, x1, y1 = map(float, re.search(r"%%HiResBoundingBox: ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", bbox).groups())
    info = subprocess.run(["pdfinfo", "overview.pdf"], cwd=work, capture_output=True, text=True).stdout
    page_h = float(re.search(r"Page size:\s+[\d.]+ x ([\d.]+)", info).group(1))
    dpi, pad = 220, 6
    s = dpi / 72
    x, y = int((x0 - pad) * s), int((page_h - y1 - pad) * s)
    w, h = int((x1 - x0 + 2 * pad) * s), int((y1 - y0 + 2 * pad) * s)
    subprocess.run(["pdftoppm", "-r", str(dpi), "-x", str(x), "-y", str(y), "-W", str(w), "-H", str(h), "-png",
                    "-singlefile", "overview.pdf", str(png.with_suffix(""))], cwd=work, check=True)
    return png


def pandoc(tex: str) -> str:
    src = OUT / "opacity-preprint-source.tex"
    src.write_text(tex)
    csl = CSL or PAPER / "data" / "ieee.csl"
    args = ["pandoc", "-f", "latex", "-t", "html5", "--citeproc", "--bibliography", str(PAPER / "references.bib"),
            "--mathml", "--shift-heading-level-by=1", "--metadata", "reference-section-title=References",
            "--metadata", "link-citations=true", "--wrap=none", str(src)]
    if csl.exists():
        args += ["--csl", str(csl)]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode:
        sys.stderr.write(proc.stderr)
        raise SystemExit("pandoc failed")
    if proc.stderr.strip():
        sys.stderr.write(proc.stderr)
    return proc.stdout


STYLE = """
<style>
/* Layout: one reading column at about 72 characters, figures and tables at the column's width, wide tables scroll. */
:root {
  --bg: #FAFAF8; --fg: #1F2328; --muted: #5B6470; --accent: #33608C; --accent-2: #D08C34;
  --rule: #D8D6D1; --code-bg: #EEF1F4; --note-bg: #F3F4F1;
  --font-body: "Crimson Pro", "Linux Libertine O", Georgia, "Times New Roman", serif;
  --font-head: "Source Sans 3", "Linux Biolinum O", "Helvetica Neue", Arial, sans-serif;
  --font-code: "Inconsolata", "SFMono-Regular", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #15181D; --fg: #E6E6E3; --muted: #A2A8B0; --accent: #8DB2DC; --accent-2: #E0A458;
    --rule: #333840; --code-bg: #1E2329; --note-bg: #1B1F25; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #15181D; --fg: #E6E6E3; --muted: #A2A8B0; --accent: #8DB2DC; --accent-2: #E0A458;
  --rule: #333840; --code-bg: #1E2329; --note-bg: #1B1F25; color-scheme: dark;
}
body { background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 1.125rem; line-height: 1.55;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 2rem 4rem; }
main, header.title { max-width: 72ch; margin: 0 auto; }
header.title { text-align: left; border-bottom: 1px solid var(--rule); padding-bottom: 1.2rem; margin-bottom: 1.6rem; }
header.title h1 { font-family: var(--font-head); font-weight: 600; font-size: 2rem; line-height: 1.2; margin: 0 0 .8rem; color: var(--accent); text-wrap: balance; }
header.title .author { font-family: var(--font-head); font-size: 1.05rem; margin: 0; }
header.title .affil { color: var(--muted); margin: .1rem 0 .6rem; font-size: 1rem; }
header.title .preprint { font-family: var(--font-head); font-size: .85rem; color: var(--muted); letter-spacing: .01em; }
h2, h3 { font-family: var(--font-head); text-wrap: balance; line-height: 1.25; }
h2 { font-size: 1.5rem; font-weight: 600; margin: 2.4rem 0 .8rem; color: var(--accent); }
h3 { font-size: 1.15rem; font-weight: 600; margin: 1.8rem 0 .5rem; }
p { margin: 0 0 .9rem; }
a { color: var(--accent); text-decoration: none; }
a:hover, a:focus-visible { text-decoration: underline; }
:focus-visible { outline: 2px solid var(--accent-2); outline-offset: 2px; }
.abstractblock { font-size: 1.02rem; border-left: 3px solid var(--accent); padding: .2rem 0 .2rem 1rem; margin: 0 0 1.6rem; }
.abstractblock p { margin: 0; }
.abstractblock::before { content: "Abstract"; display: block; font-family: var(--font-head); font-weight: 600; color: var(--accent); margin-bottom: .4rem; }
nav.contents { font-family: var(--font-head); font-size: .95rem; background: var(--note-bg); border: 1px solid var(--rule); border-radius: 4px; padding: .8rem 1.1rem; margin: 0 0 1.8rem; }
nav.contents strong { display: block; margin-bottom: .3rem; }
nav.contents ol { margin: 0; padding-left: 1.3rem; columns: 2; column-gap: 2rem; }
nav.contents li { break-inside: avoid; margin: .1rem 0; }
@media (max-width: 560px) { nav.contents ol { columns: 1; } body { font-size: 1.05rem; } header.title h1 { font-size: 1.6rem; } }
figure { margin: 1.6rem 0; }
figure img { display: block; max-width: 100%; height: auto; margin: 0 auto .6rem; }
figcaption, caption { font-family: var(--font-head); font-size: .9rem; line-height: 1.45; color: var(--muted); text-align: left; }
figcaption strong, caption strong { color: var(--fg); }
.table-wrap { overflow-x: auto; margin: 1.6rem 0; }
table { border-collapse: collapse; font-family: var(--font-head); font-size: .86rem; line-height: 1.35; font-variant-numeric: tabular-nums; min-width: 100%; }
caption { caption-side: top; padding: 0 0 .5rem; }
th, td { padding: .3rem .55rem; text-align: left; vertical-align: top; border-bottom: 1px solid var(--rule); }
thead th { border-bottom: 1.5px solid var(--fg); font-weight: 600; }
tbody tr:last-child td { border-bottom: 1.5px solid var(--fg); }
.examplebox { background: var(--note-bg); border: 1px solid var(--rule); border-radius: 4px; padding: .7rem 1rem .4rem; margin: 1.4rem 0; font-size: .95rem; }
.examplebox > p:first-child { margin-bottom: .5rem; }
.examplebox table { font-size: .82rem; }
blockquote { font-size: .95rem; color: var(--muted); border-left: 3px solid var(--rule); margin: .6rem 0 1.2rem; padding: .1rem 0 .1rem 1rem; }
pre { font-family: var(--font-code); font-size: .82rem; line-height: 1.4; background: var(--code-bg); border-radius: 4px; padding: .8rem 1rem; overflow-x: auto; }
code { font-family: var(--font-code); font-size: .92em; background: var(--code-bg); padding: 0 .25em; border-radius: 3px; }
pre code { background: none; padding: 0; }
dl dt { font-family: var(--font-head); font-weight: 600; margin-top: .7rem; }
dl dd { margin: .1rem 0 0 1.2rem; }
ol, ul { padding-left: 1.4rem; }
li { margin: .25rem 0; }
#refs { font-size: .95rem; }
.csl-entry { margin: 0 0 .55rem; padding-left: 2.2em; text-indent: -2.2em; }
.csl-entry .csl-left-margin { display: inline-block; width: 2.2em; text-indent: 0; }
.csl-entry .csl-right-inline { display: inline; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 2rem 0; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
</style>
"""


def wrap(body: str, when: str) -> str:
    # wide tables scroll inside their own container
    body = re.sub(r"<table(\s[^>]*)?>", lambda m: '<div class="table-wrap"><table>', body)  # pandoc styles p{} tables
    body = body.replace("</table>", "</table></div>")
    body = re.sub(r'src="([A-Za-z0-9_-]+)"', r'src="figures/\1.png"', body)
    body = re.sub(r"<figcaption>(Figure S?\d+\.|Table S?\d+\.)", r"<figcaption><strong>\1</strong>", body)
    body = re.sub(r"<caption>(Table S?\d+\.)", r"<caption><strong>\1</strong>", body)
    # references before the appendix, as in the PDF
    # pandoc appends the bibliography last and without a heading; put it where the PDF has it, before the appendix
    refs = re.search(r'<div id="refs".*$', body, re.S)
    first_appendix = re.search(r'<h2[^>]*id="[^"]*"[^>]*>A ', body)
    if refs and first_appendix and refs.start() > first_appendix.start():
        block = '<h2 class="unnumbered" id="references">References</h2>\n' + refs.group(0).rstrip() + "\n"
        body = body[:refs.start()].rstrip() + "\n"
        body = body[:first_appendix.start()] + block + "<hr>\n" + body[first_appendix.start():]
    body = re.sub(r'<h1[^>]*id="bibliography"[^>]*>References</h1>\s*', '', body)  # pandoc's own, now redundant
    if '<div id="refs"' in body and 'id="references"' not in body:  # no appendix: the references stay last, with a heading
        body = body.replace('<div id="refs"', '<h2 class="unnumbered" id="references">References</h2>\n<div id="refs"', 1)
    # a contents list from the top-level headings
    items = [(m.group(1), re.sub(r"<[^>]+>", "", m.group(2))) for m in re.finditer(r'<h2[^>]*id="([^"]+)"[^>]*>(.*?)</h2>', body)]
    nav = '<nav class="contents" aria-label="Contents"><strong>Contents</strong><ol>' + "".join(
        f'<li><a href="#{i}">{t}</a></li>' for i, t in items) + "</ol></nav>\n"
    body = re.sub(r"(<h2[^>]*id=\"map-of-the-paper\")", lambda m: nav + m.group(1), body, count=1)
    if nav not in body:
        body = re.sub(r"(<h2)", lambda m: nav + m.group(1), body, count=1)
    head = (f"<title>{TITLE}</title>\n"
            '<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=Source+Sans+3:wght@400;600&family=Inconsolata&display=swap">\n'
            + STYLE)
    affil = AFFIL
    m = re.search(r"\\author\{.*?\\small (.*?)\\\\", (PAPER / "main.tex").read_text(), re.S)
    if m:
        affil = re.sub(r"\s+", " ", m.group(1)).strip()
    tail = ("machine arms only; the human reviewer arm is pre-registered to follow · "
            if "human reviewer arm" in (PAPER / "main.tex").read_text() else "")
    header = (f'<header class="title"><h1>{TITLE}</h1><p class="author">{AUTHOR}</p><p class="affil">{affil}</p>'
              f'<p class="preprint">Preprint, not peer reviewed · built {when} from commit {commit()} · all data synthetic · '
              f'{tail}the PDF is the version of record</p></header>\n')
    return head + header + "<main>\n" + body + "\n</main>\n"


def title_from_main() -> str:
    m = re.search(r"\\title\{((?:[^{}]|\{[^{}]*\})*)\}", (PAPER / "main.tex").read_text())
    return re.sub(r"\s+", " ", m.group(1).replace("\\\\", " ")).strip() if m else TITLE


def sections_from_main() -> list[str]:
    """The sections main.tex inputs, in order, without the abstract and the appendix."""
    names = re.findall(r"^\\input\{sections/([A-Za-z0-9_-]+)\}", (PAPER / "main.tex").read_text(), re.M)
    return [n for n in names if n not in ("abstract", "appendix")] or SECTIONS


def main() -> int:
    import argparse
    global PAPER, OUT, FIGS, TITLE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper", type=Path, default=PAPER, help="directory holding main.tex, sections/ and a build's main.aux")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <paper>/html)")
    ap.add_argument("--name", default="opacity-preprint.html", help="file name of the page")
    ap.add_argument("--figures", type=Path, default=FIGS, help="directory of the figure PNGs")
    ap.add_argument("--sections", default=None, help="comma-separated section names in order (default: those main.tex inputs)")
    ap.add_argument("--macros", default="", help="comma-separated .tex files of \\newcommand macros to hand pandoc first")
    ap.add_argument("--no-appendix", action="store_true", help="the paper has no sections/appendix.tex")
    ap.add_argument("--csl", type=Path, default=None, help="citation style (default: <paper>/data/ieee.csl if present)")
    a = ap.parse_args()
    global SECTIONS, MACROS, APPENDIX, CSL
    MACROS = [m for m in a.macros.split(",") if m]
    APPENDIX = not a.no_appendix
    CSL = a.csl.resolve() if a.csl else None
    PAPER, FIGS = a.paper.resolve(), a.figures.resolve()
    SECTIONS = a.sections.split(",") if a.sections else sections_from_main()
    OUT = (a.out or PAPER / "html").resolve()
    TITLE = title_from_main()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    for png in FIGS.glob("*.png"):
        shutil.copy(png, OUT / "figures" / png.name)
    for name in tikz_figures():
        tikz_png(name)
    tex = latex_source()
    unresolved = sorted(set(re.findall(r"\?\?[a-z0-9.]+", tex)))
    if unresolved:
        raise SystemExit(f"unresolved numbers or references: {unresolved[:20]}")
    left = sorted(m for m in style_macros() if re.search(r"\\" + m + r"\b", tex))
    if left:
        raise SystemExit(f"style macros left for pandoc, which drops them with their arguments: {left}")
    body = pandoc(tex)
    import datetime
    html = wrap(body, datetime.date.today().isoformat())
    out = OUT / a.name
    # every image and every table handed to pandoc must come out; images are counted as
    # \includegraphics (a TikZ drawing has become one by now, inside a figure or a box)
    for what, pat, tag in (("images", r"\\includegraphics", r"<img "), ("tables", r"\\begin\{tabular\}", r"<table[ >]")):
        want, got = len(re.findall(pat, tex)), len(re.findall(tag, html))
        if want != got:
            raise SystemExit(f"{want} {what} in the source, {got} in the page: pandoc dropped or degraded some")
    if 'class="abstractblock"' not in html:
        raise SystemExit("no abstract in the page")
    out.write_text(html)
    print("wrote", out, f"{out.stat().st_size / 1024:.0f} kB;", len(re.findall(r"<img ", html)), "images;",
          len(re.findall(r"<table[ >]", html)), "tables;", len(re.findall(r'class="csl-entry"', html)), "references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
