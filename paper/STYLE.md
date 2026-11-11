# Shared style for the three papers

Writing, terminology and figure conventions shared by three papers: the two
OPACITY preprints (`paper/` and `paper-vetting/`) and the companion
safe-tre-agent framework paper. This file, `wadelab.sty` and `build.sh` are
kept identical in both repositories; `paper-vetting/` links to the `paper/`
copies rather than duplicating them.

## The three papers

| | Framework paper (`safe-tre-agent/paper/`) | Checker paper (`opacity/paper/`) | Vetting paper (`opacity/paper-vetting/`) |
|---|---|---|---|
| Question | Does an untrusted language model between an analyst and sensitive data weaken a trusted research environment's disclosure guarantee, and what restores it? | Does disclosure control hide analytic error from a reviewer, and can a model reviewer or analyst be measured against planted errors? | Can a candidate model be tested for bias and deception before admission: does its internal state reveal its errors, and can an instructed bias be detected? |
| Owns | boundary design, guarantees in code, the formal layer, the red teams, agent access, the boundary mechanism of programs that act for the researcher (the inside analyst, the locked plan, the agent interface), the rank audit and the decision records | the planted-error corpus and disclosure levels s0 to s4, checker results by level and class, the harvest, model sizing (prompts, thinking budgets, judges, sealed sets, pipelines), the calculator and prepared-request results, the planner measurement, the NIGHTPLAY study design and the analysts' runs on it | vetting: probes, the steering positive control, the slant detector, style robustness and the honest-LoRA pilot |
| Names models? | No. "A local 30B-class open-weights model"; cites the checker paper for every model-level number | Yes | Yes |
| Cites the others for | every model-level number and everything a model does (the checker paper for the planner, analysts, checkers and NIGHTPLAY; the vetting paper for bias and deception) | every gateway design decision, in at most one sentence; the vetting paper for bias and deception | the gateway (framework paper) and the tasks, analysts and recommended configuration (checker paper), in at most one sentence each |

A result appears in one paper only. Where both must mention the same run,
both state the safe-tre-agent version and commit (0.13.0 is commit
`7884328`).

## LaTeX

- Wrapper `paper/main.tex`, body in `paper/sections/*.tex`, one `\input`
  per section. The wrapper carries the class, `\usepackage{wadelab}`, the
  title block and `\input{numbers}`; nothing else. A venue class later is a
  wrapper swap.
- `\documentclass[11pt,a4paper]{article}` and `wadelab.sty`, nothing added
  in the wrapper. A package one paper needs goes into the shared file.
- Build with `latexmk -pdf main` (the Makefile does this). The PDF and aux
  files are not committed.
- Citations: `natbib` numeric with `plainnat`; `\citet` in prose where the
  authors are the subject, `\cite` otherwise. One `references.bib` per
  repository; shared entries keep the same keys in both:
  `opensafely2020`, `hundepool2012`, `dinur2003`, `kenthapadi2005`,
  `dwork2006`, `greshake2023`, `acro`, `fivesafes`, `denning1979`,
  `benjamini1995`. The papers cite each other as `wade2026safetre`,
  `wade2026opacity` and `wade2026vetting`.
- Macros from `wadelab.sty`: `\code{}` for identifiers, `\clause{P19}` for a
  specification clause, `\sys{}` for a system name, `\term{}` for a defined
  term at first use, `\level{3}` for a disclosure level, `\tbd` for a value
  the build has not produced. The wrapper sets `\title`, `\shorttitle` (the
  running head), `\author` and `\preprintline`.
- Type and layout (v2.0 of the style): Linux Libertine text, Biolinum
  headings, Inconsolata code; a 148 mm measure; section numbers in the
  palette blue; links colored, not boxed. Table captions go above the table
  (`\caption` before the `tabular`), figure captions below. Figures use
  Biolinum too, so text and figures share a sans.

## Numbers

Every number in the prose, tables and captions is a macro defined in
`paper/numbers.tex`, which a script in the repository writes
(`scripts/paper_numbers.py` here; `paper/make_numbers.py` in opacity, not
`numbers.py`, which would shadow Python's `numbers` module) from the
repository's own record: counts from the specification and logs, results
from the committed run summaries. A number that exists only in a report is
hardcoded in `numbers.tex` with a comment naming the report section, and the
reproducibility section says so. A test regenerates the file and fails on
drift, so a stale number cannot become prose.

Rates are written "k of n (what n counts)". Precision is the precision the
measurement supports, and the same throughout a table.

## Figures

Vector PDF, built by a committed script at final width (89 mm single
column, 183 mm double), in Wade's house style (`figures/wadelab.mplstyle`,
palette blue, orange, green, red, purple, grey; grey for everything that is
not the point). Titles state the finding, not the variables. Direct labels
over legends. Every figure is cited in the text, in order, and every panel
is walked through. Diagrams are TikZ in the source, same fonts as the text.

## Prose

- American English in every paper (the repositories' docs are British; the
  papers are not).
- Wade's writing-style skill governs: short words, active voice, topic
  sentence first, one point per paragraph, the same word for the same thing
  throughout. Then a humanizer2 pass. No em dashes. "Data" is plural.
- Numbers, intervals and p-values go in tables, legends and Methods, not in
  running prose.
- Section skeleton for all three: Abstract; Map of the paper (unnumbered, one
  bold-led bullet per step of the argument, each ending with a `\S\ref`;
  Wade edits these for flow); Introduction (the question, the
  contributions); Background and threat model; Design or Methods; Results;
  Discussion; Limitations; Reproducibility and availability; Glossary
  (unnumbered, the shared `glossarylist` environment, `\gterm{term}{definition}`,
  alphabetical; a term two papers use has the same definition in both, copied
  from this repository's `sections/glossary.tex`); References; appendices for
  long tables.
- Authors: Alex R. Wade, University of York, sole author; the AI agents in
  the acknowledgements and the contributions statement.

## Terminology, fixed in all three

trusted research environment (TRE); statistical disclosure control (SDC);
safe-outputs gateway (the gateway); the planner (the model that proposes a
query); inside analyst (the demo calls it Chimp; prose says inside
analyst); checker; release; dossier (the inside analyst's typed output);
write-up (what an OPACITY analyst releases); NIGHTPLAY question bank;
disclosure levels s0 to s4; planted error; planted null; seeded bundle and
matched control; error classes 1 to 8 (multiplicity, attrition,
confounding, causal claim, error structure, denominator, regression to the
mean, in-sample performance); harvest tasks; dev, held-out and fresh sets;
fail-closed gate (wrong at most 0.05, controls answered at least 0.75);
phase-0 gate (detection at least 0.8 per class, false alarm at most 0.2);
rules prompt and 1k think; judge (a model with the hidden key); pipeline
(analyst then checker); twins (honest and slanted write-ups sharing task,
replicate and seed); slant detector; rank audit; tracker; differencing;
lineage; probe limit; locked plan; selection budget.
