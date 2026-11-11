# Results

Aggregate tables and result summaries behind the numbers in the two preprints.
`paper/make_numbers.py` reads them (from `runs/` in the working repository) and
writes every number the papers print into `paper/numbers.tex`. The raw
evaluation logs, with every prompt and verbatim reply, are not included.

Rates are proportions with 95% intervals (`lo`, `hi`): Wilson intervals for
single rates and task-resampled intervals where tasks are the unit. Model
names are the last part of the Hugging Face id. An analyst configuration is
written `model|arm|thinking`, followed where they apply by the prompt, the
task set (dev when absent), the sampling preset, tools and the pack version
(`v1` or `v2`); a checker configuration is `model|thinking`, followed by the
reviewer text where it is not the default. `analyst_key` and `checker_key` in
`src/opacity/sizing.py` build them.

## Study 1: reviewers under disclosure control (`summary/`)

| File | What it holds |
|---|---|
| `checker_by_level.csv` | Each reviewer's detection, false alarm and "cannot tell" at each disclosure level, overall and per error class (`det_c1` to `det_c8`) |
| `auc_by_level.csv` | The same reviewers' AUC from confidence-weighted verdicts, per level |
| `cost_by_checker.csv` | Seconds and output tokens per verdict |

## Study 2: model analysts and checkers (`summary/` and `summary/sizing/`)

| File | What it holds |
|---|---|
| `harvest_by_model.csv` | The first analyst run (standard instruction): wrong conclusions on planted-error and control tasks per model |
| `judge_agreement.csv` | Agreement and kappa between two judges on the same write-ups |
| `crosscheck_matrix.csv`, `crosscheck_asymmetry.csv` | Every model as checker of every model's write-ups, and whether a checker is kinder to its own |
| `sizing/analysts.csv` | Every analyst configuration on each task set: wrong conclusions released, control tasks answered, coverage, GPU time |
| `sizing/checkers.csv` | Every checker configuration on the planted-error reports and on the analysts' real write-ups |
| `sizing/matrix.csv` | Checker by analyst: sensitivity and false alarm on the analysts' write-ups |
| `sizing/pipelines.csv` | Every analyst followed by every combination of checks, with cost and whether it meets the release threshold |
| `sizing/contrasts.csv` | Paired differences between configurations of the same model |
| `sizing/verifier.csv` | The deterministic verifier's flags on wrong and right write-ups |
| `sizing/judges.csv` | Agreement between the two hosted judges per analyst |
| `sizing/heldout_selection.csv`, `sizing/fresh_selection.csv` | The configurations chosen on dev and held-out to go forward |
| `sizing/fresh_results.csv`, `sizing/fresh2_results.csv` | The chosen configurations on the two sealed task sets |
| `sizing/auc/checkers.csv`, `sizing/auc/systems.csv`, `sizing/auc/fresh_systems.csv` | Checker AUC on real write-ups and the operating point of every analyst, task set and checker |

## Vetting (`probe/`, `vetting/`)

| File | What it holds |
|---|---|
| `probe/v1/*-heldout.json` | Linear probes against the model's own verdict on held-out write-ups, with the shuffled-label control; `-2` adds the within-task AUC |
| `probe/steer-v1/`, `probe/steer-v2/` | The bias positive control: the first, gentler instruction (failed its manipulation check) and the second |
| `probe/style-v1/` | Bias detection when the requested writing style changes |
| `vetting/<configuration>/slant*/report.json` | The bias detector for a candidate configuration: held-out AUC within tasks, threshold, false alarms and the manipulation check |
