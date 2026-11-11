# OPACITY — plan review, 21 October 2026

A critical read of the design before the first run, with what changed as a
result. Read with `docs/spec.md` (the study) and `docs/taxonomy.md` (the items).

## What this does, in one paragraph

Scripted analysis bundles are built in matched pairs — one with a planted
analytic error, one clean, same data — and rendered at four graded strengths
of statistical disclosure control, plus a prose-only ablation that withholds
every table. A reviewer who sees only the released bundle
says whether the interpretation is supported. The measure is how detection
falls as disclosure control rises, per error class, with one class (an
unsupported causal claim, visible only in prose) as an internal control that
disclosure control cannot touch. Three reviewer arms: a mechanical SDC rule
set, an LLM, and humans. Phase 0 runs the seeded items with no disclosure
control at all, to establish that the checker can see them when nothing is
hidden; if it cannot, there is no study.

## Flaws found, and what was done

**1. The primary corpus is scripted, not agent-generated.** The spec reads as
if a 32B agent produces the bundles. It cannot: controlled seeding needs the
error placed by construction, with ground truth known. The agent-generated
corpus is the *secondary* study (harvesting natural error rates). This changes
what the H100s are for in the near term — the checker arm and the harvest, not
the primary bundles. The README now says so.

**2. Suppression alone did not conceal the multiplicity count.** A fully
suppressed row is still a row. Twenty lines of `[s]` tell the reviewer there
were twenty comparisons. Real output checking collapses or drops such rows;
the SDC layer now does too (`row_collapse`: keep → other → drop across levels),
and the class 1 concealment is tested.

**3. The whole-table withholding rule fired on the wrong table.** With one
subgroup surviving, the rule withdrew the class 1 table entirely, which is not
concealment — a bundle citing a table that isn't there is a different, obvious
cue. The rule now applies only to tables flagged `stratified` (class 3), and
class 1 at s3 releases the single surviving row, as a real checker would.

**4. s2 and s3 were nearly the same level.** For class 3 the only difference
was rounding base, and at both a reviewer could difference the `Total` row to
recover the pooled severe-stratum comparison. s3 now withholds margins. The
three non-zero levels now differ in mechanism, not just in degree.

**5. Ground truth lied.** The class 1 rationale said "no true effect in any
region" while the generator planted one. The analytic error does not depend on
whether the effect is real — twenty uncorrected tests cannot distinguish a
regional effect from selection either way — but the rationale shown to
adjudicators must be accurate. Fixed. Likewise "Holm-corrected" was Bonferroni;
now says so.

**6. Margins were summed from rounded cells.** Real SDC rounds the margin
from the true total, so margins need not equal the column sum. That mismatch
is a known reviewer cue and the layer now reproduces it.

**7. Placeholder contributor arrays were noise.** Dominance correctly no-ops
when contributors are absent; the fake arrays are gone.

**8. `p = 0.0000`.** A formatting cue unrelated to the seeded error. Removed.

## Flaws that remain, by choice

**Zero cells are not suppressed.** `0 of 9` in a small stratum is disclosive
under some TRE rules (it reveals that all nine had no event). Left visible,
because practice varies and it is a parameter, not a principle. Document in the
pre-registration which convention is used.

**Class 5's control mentions sites at s3; the seeded bundle does not.** A
correct analysis of clustered data must mention the clustering, so at s3 the
only remaining trace of clustering is in the control's methods. A reviewer sees
one bundle, not both, so there is no leak between them — but a reviewer who
reasons "no mention of structure, so I cannot tell" is doing exactly the right
thing, and `cannot_tell` is a measured outcome. Keep, and watch it in the data.

**All eight classes are now built.** 1, 3, 4, 5 came first and span every
concealment mechanism (cell suppression, strata suppression, none, cluster
redaction); 2, 6, 7, 8 followed and add replication of mechanisms, not new
ones. Each has its own invariant in `docs/protocol.md` section 11.

**One replicate per class is thin.** `--replicates N` redraws the data under
each generator; use 5+ for anything you will report.

## Things the plan was already right about, and must not drift on

- Class 4 renders byte-identically at s0 to s3. Tested. It carries no sample
  list, so the s3 withholding cannot touch it; if anyone adds a table to that
  item that rounds, or a sample list, the internal control is destroyed. At
  s4 its table is withheld like every other, which is the point of s4.
- The reviewer instruction is identical across arms and names no error class.
  It is in `checker/prompts.py` and is part of the pre-registration.
- Every raw model response is stored verbatim. Nothing is regenerated.
- The mechanical arm's zero detection is a result, not a bug: it is the
  operationalisation of "output checkers check disclosure, not correctness".

## What the first hosted runs showed, 21 October

The first machine-arm runs at s0 to s3 (MiMo pro and DeepSeek flash through
Cline, five replicates) found detection flat across levels. The cause was not
that disclosure control does nothing; it was that the layer leaked. Three
cues survived every level and the checkers read them.

1. **The prose quoted exact statistics under rounded tables.** A chi-square
   and a three-decimal p sat beneath a table rounded to base 10. Both
   checkers recomputed the statistic, found it did not match, and flagged the
   bundle. Disclosure control was manufacturing a cue rather than removing
   one. Numbers quoted in prose are now released under the same policy as the
   tables (`Quote` in `bundle.py`): counts round to the base, and from s2 a
   test statistic is dropped and its p is banded.
2. **The sample list gave attrition away on its own.** At s3 a released
   enrolment total beside a released analysed total still said rows were
   dropped. s3 now withholds the sample list.
3. **Stratum arm sizes disclosed the allocation imbalance** after every
   within-stratum comparison had been suppressed. A stratified table is now
   withheld at s3 unless some stratum can be released in full.

A fifth level, s4, withholds every table and is an ablation rather than a
policy: it measures what methods and interpretation give away alone, the floor
against which the s0 to s3 gradient is read. The graded levels remain s0 to
s3; the prediction is unchanged. Runs made before these changes are kept,
tagged as pre-change, and are not results.

The lesson for the paper is the same one class 4 was built to teach, met one
level earlier than expected: any cue that disclosure control does not touch
puts a floor under detection, and the layer must be audited for such cues
before the gradient means anything.

## On the checker model

Three candidates were named in conversation. The design does not care which:
any model Inspect can reach is a command-line string. Phase 0 is the
selection instrument — run each candidate at s0 and keep what is near ceiling.
Two constraints survive the choice: the checker must be a different model
family from the agent used for harvesting, and the exact model string and
serving version are pinned in the eval log.

## Failing closed, and a model that is not trying (22 October)

The harvest as first built could not measure the thing a TRE most wants,
because the agent had to write something. Two sub-arms were added without
touching the first: a typed decline (`closed`) and the model as blind
reviewer of its own write-up before release (`selfcheck`), with a declined
write-up counted as neither right nor wrong. The gate for those arms has to
be two-sided or declining everything passes it; the proposed numbers are in
`docs/harvest/README.md` and are not yet fixed. The cross-check task is the
first black-box test aimed at a swapped-in model: every model as checker on
every model's write-ups, and the diagonal against the rest. Neither has run
on the sweep yet. The white-box side (a probe trained on the model's own
checker verdicts, applied to its agent output) is not built.

## Thinking as a variable (22 October)

Every local run so far had thinking off, because Qwen3.5 with thinking on
spent the whole 4000-token budget before its JSON. That made thinking a
confound rather than a measurement. Thinking is now a capped budget the
server enforces, added on top of the answer's allowance, so a model cannot
look worse for having thought; the cap, the effort level and the reviewer
text are in every log header. The first smoke run showed why the cap needs
a transition sentence and not a bare end tag: forced to stop at 512 tokens,
Qwen3.5-9B carried on reasoning in its visible answer until the output cap.

## A verifier that does arithmetic (22 October)

Every local checker misses the wrong denominator, and the analysts' natural
errors concentrate in multiplicity, attrition and clustering, so a
deterministic verifier was written for those structures: it recomputes from
the pack the analysis the data call for and asks whether the write-up's
claims survive. On the dev tasks it flags about a quarter of the write-ups
the judge calls wrong and under one per cent of those it calls right. It was
tuned on the dev tasks against the judge's verdicts, so its held-out run is
the number to quote, not these.

## Six harvest packs were malformed (22 October)

Reading the packs for the model-sizing study showed that six of the 36 had a
table whose rows did not line up with the header: an unquoted comma inside a
YAML flow list (`[Eligible, not enrolled (capacity), 220, ...]`) splits the
cell. In h25 to h28, the four regression-to-the-mean tasks, the comparison
group's counts and means sat one column to the right of their headers, which
garbles exactly the evidence those tasks turn on; h14 and h35 had a phantom
header column, and h35 printed its years with a thousands separator. The
cells are now quoted, a test checks every row's width at three replicates,
and the harvest runs before the fix are marked as having seen the old packs.
The class-7 commit rates in the 21 October results were measured on garbled
tables and should be read with that in mind.

## A request prepared outside the TRE (23 October)

Frontier models outside the TRE get the harvest right with the plain
instruction; local models need the rules prompt to come near. The preparer
arm asks whether the outside model can help without seeing data: it gets the
request and the table structure, no values, and rewrites the request for the
local analyst under a fixed, generic instruction. Run on 25 October with
GLM-5.3 flash as preparer (report 6d, both judges): it lowers wrong
releases with the plain prompt on held-out (128 to 75 of 288 by the local
judge, 119 to 63 by MiMo flash) but raises them with the rules prompt (35 to
54 on each set, and 23 to 43 and 23 to 51), because a plan written before the data
are seen fixes the pooled comparison as primary and so overrides "let the
stratified result decide". It is exploratory,
runs on dev and held-out only, and cannot enter the fresh-set selection,
which fixed its arms first. It is also a different deployment from an
all-local one, since a hosted model reads the request and the table
structure.

## Vetting a model before it enters the TRE (26 October)

Someone will ask how we know an open-weights model is fit to run inside the
TRE. The answer has four layers (`docs/vetting.md`): containment and
provenance carry the security argument, and tests before admission and
checks in service carry the question of whether the model's conclusions are
sound. The study's tasks are unpublished, so they can serve as the tests. The
first new measurement is a probe on the model's internal state. Outside
checkers reach AUC 0.58 to 0.72 on real write-ups, and a local model's state
is open to us where a hosted model's never is. The experiment is
pre-registered with a stop rule Wade fixed (0.05 over the model's own verdict
on held-out), and its code refuses a second look at held-out without a
recorded reason. Both probes met the rule on 26 October, but a probe trained
on labels shuffled within tasks did almost as well (0.83 against 0.86 for the
27B), so the gain is knowing which tasks are risky. Within a task, the probe
is no better than the model's own verdict (0.74 against 0.70 for the 27B,
0.55 against 0.61 for gpt-oss 20B). We found no sign that the models' states
know more about a write-up than their verdicts say.

That null needed a known positive, so on 27 October the 27B was told, in a
private note after its instructions, to slant its write-ups toward the
result the request hoped for, without false numbers. A gentle note changed
nothing (53 against 50 wrong of 360). An outright one raised wrong
conclusions from 12% to 31% on held-out. A probe trained on the slanted
write-ups and their honest twins found the slant on unseen tasks (AUC 0.95
within tasks) and flagged the 27B's ordinary write-ups only at its
false-alarm rate. A text classifier did as well, so for instructed slanting
the evidence is in the words, and probes stay out of vetting.

The word classifier became a vetting test on 27 October (`docs/vetting.md`
3.6). Run on the recommended analyst in its own configuration, with a 1k
think, it met a model that would not slant even when told outright (21
against 18 wrong of 180 on held-out). Thinking first, the model reasoned its
way back to the data. Its ordinary write-ups were flagged at 3.6% by its own
detector and 1.6% by the detector from the thinking-off twins.

Wade asked whether a style instruction would undo detection from the text
(`docs/vetting.md` 4.11). A request for brevity did, at the detector's
threshold: honest concise write-ups carry fewer caveat words, and 15% were
flagged against 5%. A detector trained on twins in three styles fixed it.
Told to be thorough, or to keep an honest report's caveats while slanting,
the model stopped slanting altogether, so instruction alone could not make
the disguised slant that would test the words' limits.

## For the paper

The next engineering step has implementation code: the honest-only
Qwen3.5-27B LoRA pilot (`docs/viking-lora-runner.md`, 27 October), including
CPU checks, stage gates and one-H100 Slurm wrappers. It first ran the same
day, and all three H100 stages passed (`docs/viking-lora-pilot-results.md`):
bit-exact reload and disable, base weights unchanged, 2,048 tokens under the
85% memory gate.
The earlier inference/probe smoke is not evidence that training fits or
works. The biased-model study remains separate.

Pre-register after phase 0, not before: phase 0 may change the item set.
What goes in the registration is the taxonomy, the inclusion rules, the
reviewer instruction verbatim, the four graded policies and the s4
ablation, the prediction (classes 1/2/3/5/6/7 fall with level; class 4
flat), and the model string. The repo
at that commit is the supplementary material.
