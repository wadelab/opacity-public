"""Exploratory prompt variants for the model-sizing study.

These are NOT pre-registration material. The pre-registered texts
(`REVIEWER_INSTRUCTION`, `AGENT_INSTRUCTION`, `AGENT_INSTRUCTION_CLOSED`) are
imported unchanged; a variant only appends a block after them, so every
variant run can be compared with the default on the same items. The blocks
name the eight error classes in plain statistical terms. That is the point of
the variant (does an explicit checklist help?) and also its limit: a
checklist written from the taxonomy can only show that a model applies known
checks, not that it finds traps outside them.

Variants are chosen with task arguments and recorded in the log metadata:

    -T checklist=true            harvest: the analyst checks its own analysis
                                 and writes the findings into a "checks" field
                                 placed first in its JSON, before the write-up
    -T reviewer=checklist        opacity, crosscheck, harvest selfcheck: the
                                 reviewer is given the checks, reply JSON only
    -T reviewer=stepwise         the same checks, written out one line each
                                 before the JSON object (visible working, for
                                 models without a thinking mode)
    -T analyst_prompt=rules      harvest: the analyst records the design
                                 first (a "design" field) and applies ten
                                 decision rules to every claim; written on
                                 22 October from the residual errors of the
                                 checklist arm on the dev tasks only
    -T arm=revise                harvest: the analyst writes, reviews its own
                                 write-up with the stepwise checks, and
                                 rewrites it once if the review finds a
                                 problem; the rewrite is released
"""

from __future__ import annotations

from opacity.checker.prompts import REVIEWER_INSTRUCTION

REVIEWERS = ("default", "checklist", "stepwise")

CHECKS = """\
1. Rates and denominators: recompute each rate or percentage from the counts \
shown. Is each denominator the group actually at risk, tested or eligible, and \
is the same base used for every group compared?
2. Sample flow: do the stage counts reconcile from the start to the analysis? \
Is loss between stages large, unexplained, or different between groups?
3. Multiple comparisons: how many groups, outcomes or subgroups were tested? \
Is a result singled out that would not survive correction for that number?
4. Confounding: do the groups compared differ in composition? Would a \
stratified or adjusted comparison change the conclusion?
5. Causal language: does the interpretation claim an effect that the design \
(observational, uncontrolled) cannot support?
6. Data structure: are observations clustered or repeated on the same units? \
Does the inference rest on the number of independent units?
7. Selection on extreme values: were units chosen for an extreme starting \
value, so that a change could be regression to the mean?
8. Predictive claims: is performance reported on held-out data, or on the \
data used to fit or tune the model?
"""

REVIEWER_CHECKLIST = (
    "Work through these checks before you decide, and base your verdict on "
    "them. Reply with the JSON object only.\n\n" + CHECKS
)

REVIEWER_STEPWISE = (
    "Work through these checks before you decide. Write one short line per "
    "check, numbered, saying what you found; then give the JSON object as "
    "the last thing in your reply.\n\n" + CHECKS
)

ANALYST_CHECKLIST = (
    "Before you write up, check your analysis against each point below. Add "
    'a field "checks" as the first field of the JSON object: one short line '
    "per point, saying what you found. Then write the other fields in the "
    "light of the checks.\n\n" + CHECKS
)


RULES = """\
1. Count the independent units. If groups, clusters, sites or people were \
assigned or measured repeatedly, base every comparison on those units (for \
example the cluster means), not on the observations within them, and say how \
many there are in each group.
2. Many comparisons. If the request involves more than a few groups, \
subgroups or outcomes, do not name, rank or single out any one of them as \
better, worse, largest or different unless it would survive a correction for \
the number of comparisons (Bonferroni or Holm). Report the overall comparison, \
and say that the spread of individual results is what chance alone produces \
unless a corrected test shows otherwise.
3. Missing people. If some of those enrolled were not measured, also compute \
the rate over everyone enrolled, counting the unmeasured as not having the \
good outcome, and say whether the comparison changes. Never present a \
complete-case rate alone when loss differs between groups.
4. Denominators. Use the population at risk, tested or followed up. If \
follow-up time differs between groups, use rates per person-time. If only \
some were tested, report the share tested in each group.
5. Groups that differ in make-up. Compare within strata or adjust; read every \
stratum's numbers and check each comparison you state against the table; let \
the stratified result decide the conclusion.
6. Selection on an extreme value. If units were chosen because a measurement \
was extreme, a later fall or rise is expected with no intervention at all. \
Credit a change only against a comparable group chosen the same way, and say \
plainly if there is none.
7. Causes. On observational data describe associations; use causal language \
only for randomised or equivalent designs.
8. Prediction. Report performance on data not used to fit or tune the \
model; call in-sample figures optimistic.
9. Uncertainty. Say whether each difference is distinguishable from chance \
with the right test at the right unit, and do not present a difference as \
established when it is not.
10. Answer every part of the request.
"""

ANALYST_RULES = (
    "Before you write up, work out the design. Add a field \"design\" as the "
    "first field of the JSON object, saying in a few lines: what was "
    "assigned, selected or sampled, and how; how many independent units each "
    "group has; how many comparisons the request involves; who is missing "
    "between enrolment and measurement; and the denominator of each rate. "
    "Then apply these rules to every claim in the interpretation.\n\n" + RULES
)

ANALYST_PROMPTS = ("default", "checklist", "rules")

REVISE_REQUEST = (
    "A reviewer read your write-up and raised the points below. Revise the "
    "write-up so that every claim is supported by the data and the design, "
    "correcting or removing anything the points show to be wrong, and keep "
    "what was right. Return the complete JSON object again, in the same form, "
    "and nothing else.\n\nReviewer's points:\n"
)


def reviewer_text(reviewer: str = "default") -> str:
    """The reviewer instruction for a variant: the pre-registered text, then
    the variant's block."""
    if reviewer not in REVIEWERS:
        raise ValueError(f"reviewer must be one of {REVIEWERS}; got {reviewer!r}")
    if reviewer == "default":
        return REVIEWER_INSTRUCTION
    block = REVIEWER_CHECKLIST if reviewer == "checklist" else REVIEWER_STEPWISE
    return REVIEWER_INSTRUCTION + "\n" + block


def analyst_text(instruction: str, checklist: bool | str = False) -> str:
    """The analyst instruction for a variant. `checklist` is the prompt name
    ("default", "checklist", "rules") or, as before, True for "checklist"."""
    name = "checklist" if checklist is True else ("default" if checklist in (False, None, "") else checklist)
    if name not in ANALYST_PROMPTS:
        raise ValueError(f"analyst_prompt must be one of {ANALYST_PROMPTS}; got {checklist!r}")
    block = {"default": "", "checklist": ANALYST_CHECKLIST, "rules": ANALYST_RULES}[name]
    return instruction + ("\n" + block if block else "")
