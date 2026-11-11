"""Seeded-error item generators.

Each generator returns a matched pair: one bundle containing a planted error and
one clean control built from the same data-generating process, so that a
reviewer who flags everything scores at chance.

Every item must satisfy the inclusion criteria in docs/protocol.md:
  1. detectable in principle from the s0 bundle
  2. unambiguous -- no competent analyst would defend it as a reasoned choice
  3. concealed by a named disclosure-control mechanism (except class 4)

All eight classes are implemented. Classes 1, 3, 4, 5 each introduce a
concealment mechanism (cell suppression, strata suppression, none, cluster
redaction); classes 2, 6, 7, 8 replicate those mechanisms through rounding,
margin withholding and stage collapse. See docs/taxonomy.md for what each
level removes, read from rendered bundles.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..bundle import Bundle, GroundTruth, Quote, Stage
from ..sdc import Table, p_exact as _p

# Test statistics quoted in the prose go through `Quote` so that they are
# released under the same policy as the tables (exact at s0 and s1, banded
# from s2). Percentages and ratios stay as computed: they are statistics a
# TRE releases, and the rounded table beside them is marked as rounded.
# Class 4 quotes nothing through `Quote`: it must render identically at every
# graded level.

# --------------------------------------------------------------------------
# class 1 -- unreported multiplicity
# --------------------------------------------------------------------------


def multiplicity(rng: np.random.Generator, item_id: str = "c1-mult") -> tuple[Bundle, Bundle]:
    """Significant effect in one of many subgroups, reported without correction.

    Concealment: the comparison denominator lives in the small subgroup cells.
    One subgroup is built large enough to survive suppression at every level;
    the rest are small enough to drop out, so by s2 the reviewer sees a single
    significant result with no way to count what it was selected from.
    """
    n_sub = 20
    big_n = 90
    small_ns = rng.integers(4, 10, size=n_sub - 1)

    names = [f"Region {chr(65 + i)}" for i in range(n_sub)]
    ns = np.concatenate([[big_n], small_ns])

    # A real but modest effect is planted in the one large subgroup so that it
    # is reliably the nominal "winner" and reliably survives suppression. The
    # analytic error does not depend on whether the effect is real: twenty
    # uncorrected tests with no pooled estimate cannot distinguish a genuine
    # regional effect from selection, and the interpretation claims one.
    rates_control = rng.uniform(0.28, 0.34, size=n_sub)
    rates_treated = rates_control.copy()
    rates_treated[0] = rates_control[0] + 0.19

    tbl = Table(
        name="Table 1. Outcome rate by region and arm",
        rows=names,
        cols=["Control n", "Control events", "Treated n", "Treated events"],
        values=np.column_stack(
            [
                ns,
                np.round(ns * rates_control),
                ns,
                np.round(ns * rates_treated),
            ]
        ),
        kind="frequency",
        row_label="Region",
        show_margins=True,
    )

    k = int(np.round(ns[0] * rates_treated[0]))
    n = int(ns[0])
    k0 = int(np.round(ns[0] * rates_control[0]))
    chi2, p = stats.chi2_contingency(
        np.array([[k, n - k], [k0, n - k0]]), correction=False
    )[:2]

    shared_q = (
        "Does the intervention change the outcome rate? Report any regional "
        "variation in effect."
    )
    shared_m = (
        "Outcome rate compared between arms using a chi-square test of "
        "independence. Regional subgroups analysed separately."
    )

    err = Bundle(
        item_id=f"{item_id}-err",
        title="Regional variation in intervention response",
        question=shared_q,
        methods=shared_m,
        tables=[tbl],
        stages=[Stage("Analysed", int(ns.sum() * 2))],
        interpretation=(
            "The intervention raised the outcome rate in Region A "
            "({test}), while no effect was seen "
            "elsewhere. Region A should be prioritised for rollout."
        ),
        quotes={"test": Quote("test", chi2, p=p, stat_name="chi-square")},
        truth=GroundTruth(
            has_error=True,
            error_class=1,
            short_name="unreported multiplicity",
            rationale=(
                f"Twenty regional subgroups were tested with no correction and no "
                f"pooled estimate reported. One nominally significant result at "
                f"{_p(p)} is the expected yield of twenty tests at alpha 0.05 "
                "under the null, so the analysis cannot distinguish a regional "
                "effect from selection, yet the interpretation asserts one and "
                "recommends action on it."
            ),
        ),
    )

    # Control: same structure, same single-region emphasis, but the
    # multiplicity is declared and corrected.
    ctl = Bundle(
        item_id=f"{item_id}-ctl",
        title="Regional variation in intervention response",
        question=shared_q,
        methods=shared_m
        + " Twenty regional comparisons were made; p-values are Bonferroni-corrected.",
        tables=[tbl],
        stages=[Stage("Analysed", int(ns.sum() * 2))],
        interpretation=(
            "The largest regional difference was in Region A ({test}; "
            "{p_corr}). After correction for the twenty regional "
            "comparisons no region shows an effect, and the pooled comparison is "
            "null. We do not interpret the Region A difference as a response to "
            "the intervention."
        ),
        quotes={
            "test": Quote("test", chi2, p=p, stat_name="chi-square", p_prefix="uncorrected "),
            "p_corr": Quote("p", min(1.0, p * n_sub), p_prefix="Bonferroni-corrected "),
        },
        interpretation_matched=(
            "After Bonferroni correction for the twenty regional comparisons no "
            "region shows an effect ({test}; {p_corr}), and the pooled comparison "
            "shows no effect. Region A should not be prioritised: the intervention "
            "should be rolled out on its pooled result or not at all."
        ),
        truth=GroundTruth(
            has_error=False,
            error_class=None,
            short_name="clean control (multiplicity declared)",
            rationale="Same data; multiplicity declared and corrected, conclusion null.",
        ),
    )
    return err, ctl


# --------------------------------------------------------------------------
# class 3 -- omitted confounder / Simpson reversal
# --------------------------------------------------------------------------


def confounder(rng: np.random.Generator, item_id: str = "c3-conf") -> tuple[Bundle, Bundle]:
    """Aggregate effect opposite in direction to every stratum.

    Concealment: the strata table carries the evidence and is the part that
    suppresses, leaving the reversed aggregate standing alone.
    """
    strata = ["Low acuity", "Medium acuity", "High acuity", "Critical"]
    # Within every stratum, treatment does better. Allocation is confounded:
    # treatment is concentrated in the severe strata. Control counts in the
    # severe strata and treated counts in the mild strata are set just above
    # the s1 threshold and below the s2 threshold so that the stratum table
    # degrades in steps rather than all at once.
    n_treated = np.array([9, 18, 120, 150])
    n_control = np.array([200, 120, 18, 9])
    rate_control = np.array([0.08, 0.16, 0.44, 0.78])
    rate_treated = rate_control - 0.07

    ev_t = np.round(n_treated * rate_treated)
    ev_c = np.round(n_control * rate_control)

    strat = Table(
        name="Table 2. Adverse outcome by acuity stratum and arm",
        rows=strata,
        cols=["Treated n", "Treated events", "Control n", "Control events"],
        values=np.column_stack([n_treated, ev_t, n_control, ev_c]),
        kind="frequency",
        row_label="Acuity",
        show_margins=True,
        stratified=True,
    )

    agg_t_rate = ev_t.sum() / n_treated.sum()
    agg_c_rate = ev_c.sum() / n_control.sum()
    agg = Table(
        name="Table 1. Adverse outcome by arm, all patients",
        rows=["Treated", "Control"],
        cols=["n", "Events"],
        values=np.array(
            [[n_treated.sum(), ev_t.sum()], [n_control.sum(), ev_c.sum()]]
        ),
        kind="frequency",
        row_label="Arm",
        show_margins=False,
    )

    shared_q = "Is the treatment associated with a higher rate of adverse outcomes?"
    shared_m = (
        "Observational cohort. Adverse outcome rate compared between arms. "
        "Patients were not randomised; acuity at presentation was recorded."
    )
    stages = [
        Stage("Cohort", int(n_treated.sum() + n_control.sum())),
        Stage("Analysed", int(n_treated.sum() + n_control.sum())),
    ]

    err = Bundle(
        item_id=f"{item_id}-err",
        title="Treatment and adverse outcomes",
        question=shared_q,
        methods=shared_m,
        tables=[agg, strat],
        stages=stages,
        interpretation=(
            f"Treated patients had a markedly higher adverse outcome rate than "
            f"controls ({agg_t_rate:.0%} vs {agg_c_rate:.0%}). The treatment "
            "appears to worsen outcomes and its use should be reviewed."
        ),
        truth=GroundTruth(
            has_error=True,
            error_class=3,
            short_name="omitted confounder / Simpson reversal",
            rationale=(
                "Acuity confounds allocation: treatment is concentrated in the two "
                "most severe strata. Within every stratum the treated rate is lower "
                "than the control rate, so the pooled direction is reversed. The "
                "aggregate comparison is uninterpretable without stratification or "
                "adjustment, and the stated conclusion is the opposite of the "
                "stratum-specific finding."
            ),
        ),
    )

    ctl = Bundle(
        item_id=f"{item_id}-ctl",
        title="Treatment and adverse outcomes",
        question=shared_q,
        methods=shared_m + " Comparisons are stratified by acuity.",
        tables=[agg, strat],
        stages=stages,
        interpretation=(
            f"The pooled rate is higher among treated patients ({agg_t_rate:.0%} vs "
            f"{agg_c_rate:.0%}), but allocation is strongly confounded by acuity: "
            "treatment is concentrated in the severe strata. Within each acuity "
            "stratum the treated rate is lower than the control rate. The pooled "
            "comparison should not be read as a treatment effect, and these data "
            "cannot establish one."
        ),
        interpretation_matched=(
            f"Within every acuity stratum treated patients had the lower adverse "
            f"outcome rate. The pooled rate is higher ({agg_t_rate:.0%} vs "
            f"{agg_c_rate:.0%}) because treatment is concentrated in the severe "
            "strata. The treatment should continue, and any review of its use "
            "should rest on the stratified comparison."
        ),
        truth=GroundTruth(
            has_error=False,
            error_class=None,
            short_name="clean control (confounding declared)",
            rationale="Same data; confounding named, stratified reading given.",
        ),
    )
    return err, ctl


# --------------------------------------------------------------------------
# class 4 -- unsupported causal claim (INTERNAL CONTROL: not concealed)
# --------------------------------------------------------------------------


def causal(rng: np.random.Generator, item_id: str = "c4-caus") -> tuple[Bundle, Bundle]:
    """Correct numbers, causal interpretation, no identification strategy.

    No concealment mechanism. The error is entirely in the prose, which
    disclosure control never touches, so detectability should be flat across
    s0-s3. If it falls, the effect is reviewer fatigue, not disclosure control.
    """
    n = 4200
    r = 0.31
    hours = rng.normal(3.0, 1.1, size=n)
    score = 0.4 * hours + rng.normal(0, 1.4, size=n)
    rho, p = stats.pearsonr(hours, score)

    tbl = Table(
        name="Table 1. Screen hours and attainment score, by quartile of screen time",
        rows=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
        cols=["n", "Mean score"],
        values=np.array(
            [[n // 4, 48.1], [n // 4, 50.4], [n // 4, 52.0], [n // 4, 54.6]]
        ),
        kind="magnitude",
        counts=np.array([[n // 4, n // 4]] * 4),
        row_label="Quartile",
        show_margins=False,
    )

    shared_q = "What is the relationship between daily screen hours and attainment?"
    shared_m = (
        f"Cross-sectional survey of {n:,} respondents, single time point, all "
        "analysed. Pearson correlation between self-reported daily screen hours "
        "and attainment score."
    )
    # No sample list: s3 withholds sample lists, and this item must render
    # identically at every graded level. The n lives in the methods instead.
    stages: list[Stage] = []

    err = Bundle(
        item_id=f"{item_id}-err",
        title="Screen time and attainment",
        question=shared_q,
        methods=shared_m,
        tables=[tbl],
        stages=stages,
        interpretation=(
            f"Screen time and attainment are positively associated "
            f"(r = {rho:.2f}, p < 0.001). Increasing structured screen time raises "
            "attainment, and the finding supports extending device access in "
            "classrooms."
        ),
        truth=GroundTruth(
            has_error=True,
            error_class=4,
            short_name="unsupported causal claim",
            rationale=(
                "Cross-sectional observational data at one time point with no "
                "identification strategy, no adjustment and no temporal ordering. "
                "The numbers are correct; the claim that screen time raises "
                "attainment, and the policy recommendation drawn from it, are not "
                "supported by this design."
            ),
        ),
    )

    ctl = Bundle(
        item_id=f"{item_id}-ctl",
        title="Screen time and attainment",
        question=shared_q,
        methods=shared_m,
        tables=[tbl],
        stages=stages,
        interpretation=(
            f"Screen time and attainment are positively associated "
            f"(r = {rho:.2f}, p < 0.001). The design is cross-sectional and "
            "unadjusted, so the direction of any causal relationship cannot be "
            "established here; plausible common causes include household income "
            "and parental education, neither of which was measured. The "
            "association motivates a longitudinal study rather than a change in "
            "device policy."
        ),
        interpretation_matched=(
            f"Screen time and attainment rise together (r = {rho:.2f}, p < 0.001). "
            "The design is cross-sectional, so the next step should be a "
            "longitudinal study, and device policy in classrooms should stay as "
            "it is until that study reports."
        ),
        truth=GroundTruth(
            has_error=False,
            error_class=None,
            short_name="clean control (association only)",
            rationale="Same numbers; claim held to association, design limits stated.",
        ),
    )
    return err, ctl


# --------------------------------------------------------------------------
# class 5 -- wrong error structure
# --------------------------------------------------------------------------


def error_structure(
    rng: np.random.Generator, item_id: str = "c5-errs"
) -> tuple[Bundle, Bundle]:
    """Independent-samples test applied to clustered data, inflating significance.

    Concealment: the evidence is the cluster structure, which disclosure control
    redacts in steps -- identifiers at s1, sizes at s2, the existence of
    clustering at s3.
    """
    n_clusters = 12
    per_cluster = 25
    cluster_effect = rng.normal(0, 1.6, size=n_clusters)

    vals_a, vals_b, sizes = [], [], []
    for c in range(n_clusters):
        arm = vals_a if c % 2 == 0 else vals_b
        obs = cluster_effect[c] + rng.normal(0, 0.9, size=per_cluster)
        arm.extend(obs.tolist())
        sizes.append(per_cluster)

    a = np.array(vals_a)
    b = np.array(vals_b)

    t_naive, p_naive = stats.ttest_ind(a, b)

    # Correct unit of analysis: cluster means.
    means_a = np.array(
        [cluster_effect[c] for c in range(n_clusters) if c % 2 == 0]
    )
    means_b = np.array(
        [cluster_effect[c] for c in range(n_clusters) if c % 2 != 0]
    )
    t_corr, p_corr = stats.ttest_ind(means_a, means_b)

    tbl = Table(
        name="Table 1. Outcome by arm",
        rows=["Arm A", "Arm B"],
        cols=["n observations", "Mean"],
        values=np.array([[a.size, a.mean()], [b.size, b.mean()]]),
        kind="magnitude",
        counts=np.array([[a.size, a.size], [b.size, b.size]]),
        row_label="Arm",
        show_margins=False,
    )

    shared_q = "Does the outcome differ between arms?"
    stages = [Stage("Observations", int(a.size + b.size))]

    err = Bundle(
        item_id=f"{item_id}-err",
        title="Outcome difference between arms",
        question=shared_q,
        methods=(
            "Independent-samples t-test comparing all observations in Arm A "
            "against all observations in Arm B."
        ),
        tables=[tbl],
        stages=stages,
        cluster_note=(
            f"Observations were collected at {n_clusters} sites, "
            f"{per_cluster} per site; each site contributed to one arm only."
        ),
        cluster_sizes=sizes,
        interpretation=(
            "Arm A differed from Arm B ({test}). "
            "The effect is robust and supports adopting Arm A."
        ),
        quotes={"test": Quote("test", t_naive, p=p_naive, stat_name="t")},
        truth=GroundTruth(
            has_error=True,
            error_class=5,
            short_name="wrong error structure",
            rationale=(
                f"Sites are the unit of randomisation and each site supplies 25 "
                f"correlated observations to a single arm, so observations are not "
                f"independent. Treating n = {a.size + b.size} as independent "
                f"inflates significance: the naive test gives {_p(p_naive)}, "
                f"while a test on the {n_clusters} site means gives {_p(p_corr)}. "
                "The analysis needs a cluster-level test or a mixed model."
            ),
        ),
    )

    ctl = Bundle(
        item_id=f"{item_id}-ctl",
        title="Outcome difference between arms",
        question=shared_q,
        methods=(
            f"Sites are the unit of randomisation. Outcome compared between arms "
            f"at site level: an independent-samples t-test on the {n_clusters} "
            "site means."
        ),
        tables=[tbl],
        stages=stages,
        cluster_note=(
            f"Observations were collected at {n_clusters} sites, "
            f"{per_cluster} per site; each site contributed to one arm only."
        ),
        cluster_sizes=sizes,
        interpretation=(
            "At site level the arms did not differ ({test}). "
            "An observation-level test would be "
            "anticonservative here because observations within a site are "
            "correlated, so we do not report one."
        ),
        quotes={"test": Quote("test", t_corr, p=p_corr, stat_name="t")},
        interpretation_matched=(
            "At site level the arms did not differ ({test}). Arm A should not be "
            "adopted on this evidence. The site-level test is the right one "
            "because observations within a site are correlated, and it is the "
            "test this recommendation rests on."
        ),
        truth=GroundTruth(
            has_error=False,
            error_class=None,
            short_name="clean control (clustering respected)",
            rationale="Same data; analysed at the unit of randomisation.",
        ),
    )
    return err, ctl



# --------------------------------------------------------------------------
# class 2 -- silent attrition
# --------------------------------------------------------------------------


def attrition(rng: np.random.Generator, item_id: str = "c2-attr") -> tuple[Bundle, Bundle]:
    """Rows dropped by an undeclared filter; the remainder reported as representative.

    Concealment: rounding and stage collapse. The loss is small in count (four
    participants, below half the s3 rounding base) but decisive, because all
    four carried the event. At s2 the flow table and the event counts still
    differ by a visible step; at s3 enrolled and analysed Ns round to the same
    value and the event counts round to the same value.
    """
    n_arm = 300
    lost = 4  # treated participants dropped by an undeclared filter; all had the event
    ev_c = int(rng.binomial(n_arm, 0.07))
    ev_t_enrolled = max(ev_c + int(rng.integers(-1, 2)), lost + 5)
    n_t_analysed = n_arm - lost
    ev_t = ev_t_enrolled - lost

    flow = Table(
        name="Table 1. Participant flow by arm",
        rows=["Treated", "Control"],
        cols=["Enrolled", "Analysed"],
        values=np.array([[n_arm, n_t_analysed], [n_arm, n_arm]]),
        kind="frequency",
        row_label="Arm",
        show_margins=True,
    )
    events = Table(
        name="Table 2. Events by arm, analysed participants",
        rows=["Treated", "Control"],
        cols=["n", "Events"],
        values=np.array([[n_t_analysed, ev_t], [n_arm, ev_c]]),
        kind="frequency",
        row_label="Arm",
        show_margins=False,
    )

    rate_t = ev_t / n_t_analysed
    rate_c = ev_c / n_arm
    rr = rate_t / rate_c
    rate_t_full = ev_t_enrolled / n_arm
    rr_full = rate_t_full / rate_c

    shared_q = "Does the treatment reduce the rate of the event?"
    shared_m = "Two-arm trial. Event rate compared between arms as a risk ratio."
    stages = [Stage("Enrolled", 2 * n_arm), Stage("Analysed", n_arm + n_t_analysed)]

    err = Bundle(
        item_id=f"{item_id}-err",
        title="Effect of treatment on event rate",
        question=shared_q,
        methods=shared_m,
        tables=[flow, events],
        stages=stages,
        interpretation=(
            f"Events occurred in {rate_t:.1%} of treated and {rate_c:.1%} of "
            f"control participants (risk ratio {rr:.2f}). The treatment reduces "
            "the event rate."
        ),
        truth=GroundTruth(
            has_error=True,
            error_class=2,
            short_name="silent attrition",
            rationale=(
                f"Four treated participants were dropped between enrolment and "
                f"analysis by a filter the methods do not mention, and all four had "
                f"the event. The reported risk ratio of {rr:.2f} becomes "
                f"{rr_full:.2f} with them included. The stated effect is an "
                "artefact of differential attrition."
            ),
        ),
    )

    ctl = Bundle(
        item_id=f"{item_id}-ctl",
        title="Effect of treatment on event rate",
        question=shared_q,
        methods=shared_m
        + " Four treated participants with incomplete baseline data were excluded "
        "from the complete-case analysis; all four had the event.",
        tables=[flow, events],
        stages=stages,
        interpretation=(
            f"In the complete-case analysis events occurred in {rate_t:.1%} of "
            f"treated and {rate_c:.1%} of control participants (risk ratio "
            f"{rr:.2f}). With the four excluded participants included the rates "
            f"are {rate_t_full:.1%} and {rate_c:.1%} (risk ratio {rr_full:.2f}). "
            "The difference depends on the exclusion and is not read as a "
            "treatment effect."
        ),
        interpretation_matched=(
            f"With the four excluded participants included, events occurred in "
            f"{rate_t_full:.1%} of treated and {rate_c:.1%} of control participants "
            f"(risk ratio {rr_full:.2f}); the complete-case figures are {rate_t:.1%} "
            f"and {rate_c:.1%} (risk ratio {rr:.2f}). The treatment should be "
            "judged on the full-enrolment figures, which show no reduction, and "
            "the service should not adopt it."
        ),
        truth=GroundTruth(
            has_error=False,
            error_class=None,
            short_name="clean control (attrition declared)",
            rationale="Same data; exclusion declared, result given with and without it.",
        ),
    )
    return err, ctl


# --------------------------------------------------------------------------
# class 6 -- wrong denominator
# --------------------------------------------------------------------------


def denominator(rng: np.random.Generator, item_id: str = "c6-denom") -> tuple[Bundle, Bundle]:
    """A rate against the wrong base.

    Thirty-day mortality is reported per admission when the unit of analysis is
    the patient, and patients with repeat admissions are counted more than
    once. The correct base is the patient total in the table margin; the wrong
    base is the admission count in the sample list.

    Concealment: the deaths column has small cells, so it suppresses, and the
    margin is the only route to the correct rate. At s2 the margin is rounded
    and still usable; at s3 it is withheld and the stratified table is
    withdrawn for having too few populated strata.
    """
    bands = ["18-39", "40-54", "55-64", "65-74", "75+"]
    patients = np.array([240, 220, 200, 170, 140]) + rng.integers(-15, 16, size=5)
    n_patients = int(patients.sum())
    # Per-patient mortality is drawn above the review level, per-admission
    # below it, so the seeded conclusion is wrong on every draw. The three
    # youngest bands get fewer than ten deaths and suppress; the two oldest
    # get the remainder and stay above threshold.
    n_deaths = int(round(n_patients * rng.uniform(0.056, 0.068)))
    small = np.array([4, 6, 8]) + rng.integers(-1, 2, size=3)
    rest = n_deaths - int(small.sum())
    deaths = np.concatenate([small, [rest // 2, rest - rest // 2]])
    assert deaths[:3].max() <= 9 and deaths[3:].min() >= 10
    n_admissions = int(round(n_deaths / rng.uniform(0.038, 0.044)))

    tbl = Table(
        name="Table 1. Thirty-day deaths by age band",
        rows=bands,
        cols=["Patients", "Deaths"],
        values=np.column_stack([patients, deaths]),
        kind="frequency",
        row_label="Age band",
        show_margins=True,
        stratified=True,
    )

    rate_wrong = n_deaths / n_admissions
    rate_right = n_deaths / n_patients

    shared_q = "What is thirty-day mortality after admission for this condition?"
    shared_m = (
        "Retrospective cohort of all admissions for the condition over one year. "
        "Death within thirty days of index admission was ascertained from the "
        "register."
    )
    stages = [
        Stage("Admissions", n_admissions),
        Stage("Patients", n_patients, "first admission per patient"),
        Stage("Analysed", n_patients),
    ]

    err = Bundle(
        item_id=f"{item_id}-err",
        title="Thirty-day mortality after admission",
        question=shared_q,
        methods=shared_m,
        tables=[tbl],
        stages=stages,
        interpretation=(
            f"Thirty-day mortality was {rate_wrong:.1%}, below the {0.05:.0%} "
            "level at which the pathway would be reviewed. Mortality rises with "
            "age. No change to the pathway is indicated."
        ),
        truth=GroundTruth(
            has_error=True,
            error_class=6,
            short_name="wrong denominator",
            rationale=(
                f"The rate is {n_deaths} deaths over {n_admissions} admissions, "
                f"but the unit is the patient and patients with repeat admissions "
                f"are counted more than once. Per patient the rate is "
                f"{rate_right:.1%}, which is not below the review level. The "
                "conclusion rests on the wrong base."
            ),
        ),
    )

    ctl = Bundle(
        item_id=f"{item_id}-ctl",
        title="Thirty-day mortality after admission",
        question=shared_q,
        methods=shared_m + " Mortality is reported per patient, counting each "
        "patient once at first admission.",
        tables=[tbl],
        stages=stages,
        interpretation=(
            f"Thirty-day mortality was {rate_right:.1%} per patient (per admission "
            f"it would be {rate_wrong:.1%}, which understates risk because patients "
            f"with repeat admissions are counted more than once). This is not below "
            f"the {0.05:.0%} review level. Mortality rises with age."
        ),
        interpretation_matched=(
            f"Thirty-day mortality was {rate_right:.1%} per patient, above the "
            f"{0.05:.0%} review level, so the pathway should be reviewed. Per "
            f"admission the figure is {rate_wrong:.1%}, which understates risk "
            "because patients with repeat admissions count more than once. "
            "Mortality rises with age."
        ),
        truth=GroundTruth(
            has_error=False,
            error_class=None,
            short_name="clean control (denominator stated)",
            rationale="Same data; rate given per patient with the base named.",
        ),
    )
    return err, ctl


# --------------------------------------------------------------------------
# class 7 -- regression to the mean
# --------------------------------------------------------------------------


def regression_to_mean(
    rng: np.random.Generator, item_id: str = "c7-rtm"
) -> tuple[Bundle, Bundle]:
    """Pre-post change in a group selected on an extreme baseline.

    The data contain no intervention effect: baseline and follow-up are two
    noisy measurements of the same ability, and selecting the lowest baseline
    scores guarantees a rise on retest. Concealment: the selection step is a
    middle entry in the sample list, which collapses at s2; the pre-post table
    is by school with small counts, so it suppresses at s2 and is withdrawn at
    s3 as a stratified table with too few populated strata.
    """
    n = 600
    ability = rng.normal(100, 12, size=n)
    baseline = ability + rng.normal(0, 9, size=n)
    follow = ability + rng.normal(0, 9, size=n)

    cut = np.quantile(baseline, 0.10)
    low = baseline < cut
    mid = (baseline >= cut) & (baseline < np.median(baseline))
    high = baseline >= np.median(baseline)

    # Not every selected pupil took part.
    idx_low = np.where(low)[0]
    took_part = rng.choice(idx_low, size=len(idx_low) - 2, replace=False)
    n_schools = 8
    sizes = np.array([6, 7, 7, 7, 7, 8, 8, 8])
    sizes[-1] += len(took_part) - sizes.sum()
    school_of = np.repeat(np.arange(n_schools), sizes)
    rng.shuffle(school_of)

    pre = baseline[took_part]
    post = follow[took_part]
    t_paired, p_paired = stats.ttest_rel(post, pre)

    bands = Table(
        name="Table 1. Baseline reading score by band, all pupils screened",
        rows=["Below 10th centile", "10th to 50th centile", "Above 50th centile"],
        cols=["n", "Baseline mean"],
        values=np.array(
            [
                [low.sum(), baseline[low].mean()],
                [mid.sum(), baseline[mid].mean()],
                [high.sum(), baseline[high].mean()],
            ]
        ),
        kind="magnitude",
        counts=np.array([[low.sum()] * 2, [mid.sum()] * 2, [high.sum()] * 2]),
        row_label="Band",
        show_margins=False,
    )
    schools = Table(
        name="Table 2. Programme pupils: baseline and follow-up score by school",
        rows=[f"School {i + 1}" for i in range(n_schools)],
        cols=["n", "Baseline mean", "Follow-up mean"],
        values=np.array(
            [
                [sizes[k], pre[school_of == k].mean(), post[school_of == k].mean()]
                for k in range(n_schools)
            ]
        ),
        kind="magnitude",
        counts=np.column_stack([sizes] * 3),
        row_label="School",
        show_margins=False,
        stratified=True,
    )

    shared_q = "Did the reading programme raise the scores of the pupils who took part?"
    stages = [
        Stage("Screened", n),
        Stage("Selected", len(idx_low), "baseline below 10th centile"),
        Stage("Analysed", len(took_part)),
    ]

    err = Bundle(
        item_id=f"{item_id}-err",
        title="Reading programme and test scores",
        question=shared_q,
        methods=(
            "All pupils in the year group sat a baseline reading test. Pupils in "
            "the programme were retested at the end of term; change was assessed "
            "with a paired t-test."
        ),
        tables=[bands, schools],
        stages=stages,
        interpretation=(
            f"Programme pupils improved from a mean of {pre.mean():.0f} at baseline "
            f"to {post.mean():.0f} at follow-up ({{test}}). "
            "The programme raised reading scores and should be "
            "extended to the whole year group."
        ),
        quotes={"test": Quote("test", t_paired, p=p_paired, stat_name="paired t")},
        truth=GroundTruth(
            has_error=True,
            error_class=7,
            short_name="regression to the mean",
            rationale=(
                "Programme pupils were selected on a baseline below the 10th "
                "centile. Two noisy measurements of the same ability guarantee "
                "that the lowest baseline scores rise on retest, and there is no "
                "comparison group selected the same way. The data contain no "
                "programme effect; the rise is selection."
            ),
        ),
    )

    ctl = Bundle(
        item_id=f"{item_id}-ctl",
        title="Reading programme and test scores",
        question=shared_q,
        methods=(
            "All pupils in the year group sat a baseline reading test. Pupils "
            "scoring below the 10th centile were offered the programme and "
            "retested at the end of term; change was assessed with a paired "
            "t-test. There is no comparison group."
        ),
        tables=[bands, schools],
        stages=stages,
        interpretation=(
            f"Programme pupils improved from a mean of {pre.mean():.0f} to "
            f"{post.mean():.0f} ({{test}}). "
            "Because they were selected on an extreme baseline, a rise on retest "
            "is expected from regression to the mean alone, and without a "
            "comparison group selected the same way no gain can be attributed to "
            "the programme."
        ),
        quotes={"test": Quote("test", t_paired, p=p_paired, stat_name="paired t")},
        interpretation_matched=(
            f"Programme pupils rose from a mean of {pre.mean():.0f} to "
            f"{post.mean():.0f} ({{test}}), the rise expected from regression to "
            "the mean after selecting the lowest scorers. The programme should "
            "not be extended on this result, and the next cohort should include "
            "a comparison group selected the same way."
        ),
        truth=GroundTruth(
            has_error=False,
            error_class=None,
            short_name="clean control (selection declared)",
            rationale="Same data; selection named, rise attributed to regression to the mean.",
        ),
    )
    return err, ctl


# --------------------------------------------------------------------------
# class 8 -- in-sample performance
# --------------------------------------------------------------------------


def _logit_fit(X: np.ndarray, y: np.ndarray, iters: int = 30) -> np.ndarray:
    """Newton-Raphson logistic regression with a small ridge for stability."""
    Xb = np.column_stack([np.ones(len(y)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        H = Xb.T @ (Xb * (p * (1 - p))[:, None]) + 1e-4 * np.eye(Xb.shape[1])
        w = w + np.linalg.solve(H, Xb.T @ (y - p))
    return w


def _logit_score(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ w


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    ranks = stats.rankdata(score)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def in_sample(rng: np.random.Generator, item_id: str = "c8-insamp") -> tuple[Bundle, Bundle]:
    """Model fit and evaluated on the same rows.

    Concealment is partial by design. The sample list says the model was
    developed on the whole cohort and evaluated on the whole cohort; at s2 the
    middle entry collapses and the Ns round. A reviewer who notices that no
    split is declared has found the error from prose alone, which survives
    every level. Expect a shallow gradient.
    """
    n = 188
    k = 12
    X = rng.normal(size=(n, k))
    age = rng.integers(40, 90, size=n)
    X[:, 0] = (age - 65) / 15
    logit = -1.1 + 0.6 * X[:, 0] + 0.4 * X[:, 1]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(float)

    w_all = _logit_fit(X, y)
    auc_in = _auc(_logit_score(w_all, X), y)

    # Five-fold cross-validation: out-of-fold scores pooled, one AUC.
    folds = np.array_split(rng.permutation(n), 5)
    oof = np.empty(n)
    for f in folds:
        train = np.setdiff1d(np.arange(n), f)
        oof[f] = _logit_score(_logit_fit(X[train], y[train]), X[f])
    auc_cv = _auc(oof, y)

    # Descriptive table shared with the control. Two groups of about 94 so that
    # no cell can fall below threshold: this class is carried by the sample
    # list, and a partly suppressed table would add a second mechanism.
    emergency = rng.uniform(size=n) < 0.5
    tbl = Table(
        name="Table 1. Thirty-day readmission by admission type",
        rows=["Emergency", "Elective"],
        cols=["n", "Readmitted"],
        values=np.array(
            [[emergency.sum(), y[emergency].sum()], [(~emergency).sum(), y[~emergency].sum()]]
        ),
        kind="frequency",
        row_label="Admission",
        show_margins=True,
    )

    shared_q = "How well does the model predict thirty-day readmission?"
    shared_m = (
        f"Logistic regression of readmission on {k} routinely recorded "
        "predictors. Discrimination assessed by the area under the ROC curve."
    )

    err = Bundle(
        item_id=f"{item_id}-err",
        title="Predicting thirty-day readmission",
        question=shared_q,
        methods=shared_m,
        tables=[tbl],
        stages=[
            Stage("Cohort", n),
            Stage("Model development", n),
            Stage("Performance evaluation", n),
        ],
        interpretation=(
            f"The model discriminated readmission from non-readmission (AUC "
            f"{auc_in:.2f}) and is ready for deployment on the ward."
        ),
        truth=GroundTruth(
            has_error=True,
            error_class=8,
            short_name="in-sample performance",
            rationale=(
                f"The model was fitted and evaluated on the same {n} patients with "
                f"{k} predictors and no held-out data. The in-sample AUC of "
                f"{auc_in:.2f} is optimistic; five-fold cross-validation of the "
                f"same specification gives {auc_cv:.2f}. Deployment is not supported."
            ),
        ),
    )

    ctl = Bundle(
        item_id=f"{item_id}-ctl",
        title="Predicting thirty-day readmission",
        question=shared_q,
        methods=shared_m + " Performance was estimated by five-fold "
        "cross-validation; the AUC reported is on out-of-fold predictions.",
        tables=[tbl],
        stages=[
            Stage("Cohort", n),
            Stage("Model development", n),
            Stage("Performance evaluation", n, "out-of-fold"),
        ],
        interpretation=(
            f"Cross-validated AUC was {auc_cv:.2f}, against {auc_in:.2f} when the "
            "model is scored on the data it was fitted to. Discrimination is "
            "modest and the model needs external validation before any deployment."
        ),
        interpretation_matched=(
            f"Cross-validated AUC was {auc_cv:.2f}, against {auc_in:.2f} when the "
            "model is scored on its own fitting data. Discrimination is modest, "
            "so the model should go to external validation before any deployment, "
            "and the ward should keep its current process until then."
        ),
        truth=GroundTruth(
            has_error=False,
            error_class=None,
            short_name="clean control (cross-validated)",
            rationale="Same cohort; performance reported out of fold.",
        ),
    )
    return err, ctl


GENERATORS = {
    1: multiplicity,
    2: attrition,
    3: confounder,
    4: causal,
    5: error_structure,
    6: denominator,
    7: regression_to_mean,
    8: in_sample,
}
