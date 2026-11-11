# OPACITY — error taxonomy and item specifications

*Companion to `docs/spec.md` (the study) and `docs/protocol.md` (the rules). The
generators are in `src/opacity/items/core.py`; the disclosure layer is in
`src/opacity/sdc.py`. Where this file and the code disagree, the code is what
ran, and this file is wrong.*

## How an item is built

An item is a matched pair of bundles from one draw of synthetic data. The seeded
bundle carries one planted error. The control is the same data analysed
correctly, with the same tables, the same question and, as far as possible, the
same methods text and prose length. A reviewer who flags every bundle therefore
scores at chance.

Every bundle carries a `GroundTruth` that is never rendered: `has_error`,
`error_class` (1 to 8, or `None` for a control), a short name, and a rationale of
one or two sentences written for the adjudicators. The rationale must be true of
the data as generated, not of the data as the generator was first imagined; the
class 1 rationale once said there was no true effect when one had been planted,
and that is the kind of drift this rule exists to stop.

Item identifiers are `c<class>-r<replicate>-err` and `-ctl`. The random
generator is seeded from `[seed, class, replicate]`, so a corpus is reproducible
from its seed and a replicate re-draws the data under the same generator.

## What hides what

Each class names one concealment mechanism, and each mechanism maps to a small
number of fields on `SDCPolicy`. The four graded levels are set so that the
three non-zero levels differ in mechanism, not only in degree. A fifth level,
s4, is an ablation rather than a policy: every table withheld, so that what
the prose alone gives away can be measured.

| Mechanism | Policy fields | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| Cell suppression | `threshold`, `complementary` | 5, on | 10, on | 10, on | 10, on |
| Row collapse | `row_collapse` | keep | other | drop | drop |
| Rounding | `rounding_base`, `exact_n` | off | 5, Ns rounded | 10, Ns rounded | 10 |
| Stage collapse | `exact_n`, `show_sample_list` | off | first and last stage only | sample list withheld | withheld |
| Strata withholding | `min_populated_strata`, `min_comparable_strata` (stratified tables only) | off | off | fewer than 3 populated, or none releasable in full | same |
| Margin withholding | `hide_margins` | off | off | on | on |
| Dominance | `dominance_n`, `dominance_k` (magnitude tables) | off | (2, 0.85) | (1, 0.70) | (1, 0.70) |
| Cluster redaction | `cluster_disclosure` | sizes | count | none | none |
| Quoted statistics | `stat_precision` | exact | banded | banded | banded |
| Whole bundle | `withhold_tables` | off | off | off | all tables withheld |

Order within a table is fixed: primary suppression, dominance, complementary
suppression, whole-table withholding, row collapse, then rounding. Every
suppression decision is taken on true counts, and margins are rounded from the
true total rather than summed from rounded cells. Prose wording is never
altered; numbers quoted in prose are released under the policy through
`Quote` (a statistic becomes a p band from s2, a quoted count rounds), so a
rounded table never sits beside an exact statistic it cannot reproduce. These
three rules, the withheld sample list at s3, and s4 were added after phase 0
showed that checkers were reading the quoted chi-square against the rounded
table, the released total against the single surviving row, and the stratum
arm sizes against each other. See `docs/protocol.md` section 4.

## The eight classes

All eight classes are built. Classes 1, 3, 4 and 5 each introduce a
mechanism; classes 2, 6, 7 and 8 replicate those mechanisms through rounding,
margin withholding and stage collapse. Each entry gives what is seeded, what
the control does instead, what a reviewer can see at s0, and what each level
removes. The last part is read from rendered bundles, not from intent.

### 1. Unreported multiplicity (built)

**Seeded.** Twenty regional subgroups, one large (n = 90 per arm) and nineteen
small (n of 4 to 9), each tested separately with a chi-square test. A modest real
effect is planted in the large region so that it is reliably the nominal winner.
The interpretation reports the one significant region and recommends rollout
there. No correction, no pooled estimate.

**Control.** Same table. Methods declare twenty comparisons and Bonferroni
correction; interpretation reports the corrected p, calls the pooled comparison
null, and declines to read Region A as a response.

**Evidence at s0.** Twenty rows. The reviewer can count the comparisons and see
that one nominal hit at p = 0.009 is the expected yield of twenty tests.

**What each level removes.**
- s1: event cells in the small regions suppress; rows remain, so the count of
  comparisons is still visible.
- s2: fully suppressed rows merge into one `Other` row. The reviewer sees Region
  A, `Other`, and a margin. The number of comparisons is gone.
- s3: `Other` is dropped, margins are withheld, and the sample list goes with
  them. A single-row table with no mark of suppression and no total to count
  it against. The interpretation's chi-square becomes `p < 0.01`.

The whole-table rule must not fire here. A table that is cited but absent is a
different and obvious cue, so the rule applies only to tables flagged
`stratified`, and class 1 at s3 releases the surviving row as a real checker
would. Tested in `test_class1_comparison_count_visible_at_s0_concealed_by_s2`.

### 2. Silent attrition (built)

**Seeded.** A two-arm trial, 300 per arm, with no true effect. Four treated
participants are dropped between enrolment and analysis by a filter the
methods do not mention, and all four had the event. The interpretation reports
the complete-case risk ratio as a treatment effect.

**Control.** Same tables. Methods declare the exclusion and that all four had
the event; the interpretation gives the risk ratio with and without them and
declines to read the difference as an effect.

**Evidence at s0.** Table 1 shows 300 enrolled and 296 analysed in the treated
arm against 300 and 300 in the control arm, the sample list shows 600 enrolled
and 596 analysed, and the methods say nothing about exclusion. With events in
the low tens, four informative dropouts are enough to manufacture the reported
ratio.

**What each level removes.** Nothing suppresses, since every cell is above
threshold; this class is carried by rounding and stage collapse alone.
- s1: identical to s0.
- s2: counts round to base 5. The treated arm shows 300 enrolled and 295
  analysed, so attrition is still visible as a step of five.
- s3: counts round to base 10. Enrolled and analysed both read 300 in each
  arm, the sample list is withheld, and margins are withheld. The event
  counts remain, rounded, so the reviewer sees a difference in events and no
  trace of how it arose. The quoted rates and risk ratio are released as
  computed, beside tables marked as rounded.

The loss must be below half the s3 rounding base per cell, which is why it is
four. Tested in `test_class2_attrition_visible_at_s0_concealed_at_s3`.

### 3. Omitted confounder, Simpson reversal (built)

**Seeded.** Observational cohort with four acuity strata. Allocation is
confounded: treatment is concentrated in the two severe strata. Within every
stratum the treated rate is lower than the control rate, so the pooled rate is
reversed. The interpretation reads the pooled comparison as harm.

**Control.** Same two tables. Methods add that comparisons are stratified;
interpretation names the confounding, gives the within-stratum reading and
declines to infer a treatment effect.

**Evidence at s0.** Table 2 shows all four strata with both arms visible. The
reversal is a direct read.

**What each level removes.** The small-arm counts in each stratum are set just
above the s1 threshold and below the s2 threshold so that the table degrades in
steps.
- s1: the treated cells in the two mild strata suppress. Two strata remain
  comparable, which is enough to see the reversal.
- s2: control cells in the two severe strata also suppress. No stratum has both
  arms visible, but the `Total` row remains, so a reviewer can difference the
  severe strata back out.
- s3: margins are withheld, closing the differencing route, and the stratum
  table is withheld as having no stratum releasable in full. The aggregate
  table stands alone.

The stratified table is the only one subject to the whole-table rules. At s3
the populated-strata rule would not fire, since every stratum keeps a cell;
the comparable-stratum rule does, because once every comparison is
suppressed the arm sizes left behind (270 of 300 treated in the two severe
strata) disclose the allocation imbalance on their own, and phase 0 showed
checkers reading exactly that. Tested in
`test_class3_strata_visible_at_s0_none_comparable_by_s2`.

### 4. Unsupported causal claim (built, the internal control)

**Seeded.** A cross-sectional correlation between screen hours and attainment,
correctly computed and correctly tabulated. The interpretation asserts that
screen time raises attainment and recommends a policy.

**Control.** Same numbers and the same r. The interpretation holds the finding to
association, names unmeasured common causes, and recommends a longitudinal
study.

**Evidence at s0.** The prose. Nothing in the tables is wrong.

**What each level removes.** Nothing, over s0 to s3. The single table is a
magnitude table with quartile counts of 1,050, far above any threshold, and
no contributor arrays, so the dominance rule no-ops. The item carries no
sample list (its n is in the methods prose) and quotes nothing through
`Quote`, so neither the s3 sample-list withholding nor statistic banding can
touch it. The bundle renders byte-identically at all four graded levels;
`test_class4_is_invariant_across_disclosure_levels` guards this. If anyone
adds a frequency table, a sample list or a `Quote` to this item, the internal
control is destroyed. At s4 its table is withheld like every other; the
ablation is outside the invariance.

Why it matters: if detection of class 4 falls with level, the fall is not
caused by disclosure control, because disclosure control did nothing. It is
fatigue, order, or a reviewer who has learned that higher levels mean harder
bundles.

### 5. Wrong error structure (built)

**Seeded.** Twelve sites, 25 observations each, six sites per arm, with a
site-level random effect. An independent-samples t-test on all 300 observations
reports a large t. The interpretation calls the effect robust.

**Control.** Same table. Methods name the site as the unit of randomisation and
test the twelve site means; interpretation reports that test, which is null, and
explains why an observation-level test is anticonservative.

**Evidence at s0.** A cluster note under Methods: sites, observations per site,
one arm per site. With it, the mismatch between the stated unit and the tested
unit is visible.

**What each level removes.**
- s1: site identity goes; the note says observations are grouped and lists
  twelve group sizes.
- s2: sizes go; the note says twelve groups.
- s3: the note is removed. Nothing in the bundle says the data are clustered.

At s3 the seeded bundle gives a reviewer no reason to suspect clustering, while
the control's methods still mention site-level analysis because a correct
analysis must. A reviewer sees one bundle, not the pair, so this does not leak
between them, and a reviewer who answers `cannot_tell` for want of structural
information is doing what the instruction asks. Tested in
`test_class5_cluster_disclosure_degrades_in_steps`.

### 6. Wrong denominator (built)

**Seeded.** Thirty-day mortality reported per admission when the unit is the
patient, so patients with repeat admissions are counted more than once and the
rate is understated. The interpretation quotes the per-admission rate, finds it
below a review threshold, and recommends no change.

**Control.** Same table. Methods say the rate is per patient; the interpretation
gives both rates and names the base.

**Evidence at s0.** The sample list gives admissions, then patients, then
analysed; the deaths table by age band has a `Total` row giving the patient
count and the death count. The quoted rate matches deaths over admissions and
not deaths over patients.

**What each level removes.** The deaths column has three cells below ten and
two above, so it degrades in steps.
- s1: the smallest death cell is below five but the margin is visible, so the
  cell and its row-mate suppress and everything else stands. Still recomputable.
- s2: the three small-death rows collapse into `Other`; the sample list drops
  the per-patient line; the `Total` row remains, rounded, so the per-patient
  rate is still recoverable to within rounding.
- s3: fewer than three strata survive, so the table is withheld entirely, and
  the sample list is withheld. The quoted rate cannot be checked against
  anything.

The table is flagged `stratified` because it is one; withholding is the
documented SDC response to a stratified table with too few populated strata.
Tested in `test_class6_margin_visible_at_s0_table_withheld_at_s3`.

### 7. Regression to the mean (built)

**Seeded.** Six hundred pupils sit a baseline reading test; those below the
10th centile join a programme and are retested. Baseline and follow-up are two
noisy measurements of the same simulated ability, so the data contain no
programme effect and the rise on retest is selection. The interpretation
reports the paired t-test as a programme effect and recommends extension.

**Control.** Same tables. Methods name the selection and the absence of a
comparison group; the interpretation attributes the rise to regression to the
mean.

**Evidence at s0.** The sample list has the selection as its own entry,
`Selected: n = 60 (baseline below 10th centile)`. Table 1 gives the baseline
distribution by band for all pupils; Table 2 gives baseline and follow-up means
for programme pupils by school, with six to eight pupils in each of eight
schools.

**What each level removes.**
- s1: nothing. School counts are above five.
- s2: the selection entry collapses out of the sample list, which now reads
  screened and analysed only. Every school row suppresses and merges into one
  `Other` row.
- s3: Table 2 is withheld as a stratified table with no populated strata.

Table 1 keeps the band labels at every level, so the word centile survives; a
reviewer who links the band to the programme has done something the bundle no
longer states. Tested in
`test_class7_selection_step_visible_at_s0_gone_by_s2`.

### 8. In-sample performance (built)

**Seeded.** A logistic model with twelve predictors fitted on 188 patients and
evaluated on the same 188. The AUC is computed from the simulated data, so the
in-sample optimism is real rather than asserted. The methods do not mention a
split; the interpretation calls the model ready for deployment.

**Control.** Same table and the same sample list. Methods declare five-fold
cross-validation; the interpretation reports the out-of-fold AUC, which is
lower, beside the in-sample one, and asks for external validation.

**Evidence at s0.** The sample list reads cohort, model development and
performance evaluation, all 188, and the methods say nothing about a split or
cross-validation. The readmission table is descriptive and shared with the
control.

**What each level removes.**
- s1: nothing.
- s2: the sample list collapses to cohort and performance evaluation, both
  rounded to 190. The explicit statement that development used the whole
  cohort is gone.
- s3: the sample list is withheld. Only the methods' silence about a split
  remains.

Concealment is partial by design, as the spec says. A reviewer who notices
that the evaluation n equals the cohort n, or that no split is declared, has
found the error at any level. Expect a shallow gradient. If phase 0 shows no
gradient at all, class 8 is a second prose-only control, which is useful in
its own right and should be re-labelled as such rather than forced. Tested in
`test_class8_development_stage_visible_at_s0_gone_by_s2`.

## Conventions left open

**Zero cells.** A `0 of 9` in a small stratum is disclosive under some TRE
rules, since it says all nine had no event. The layer leaves zeros visible. This
is a parameter, not a principle; the pre-registration must say which convention
is used.

**p-value format.** `p < 0.001` below a thousandth, else three decimals. A
`p = 0.0000` is a formatting cue unrelated to any seeded error and is tested
against.

**Replicates.** One per class is enough to prove the pipeline closes and too
few for anything reported. Use `--replicates 5` or more.
