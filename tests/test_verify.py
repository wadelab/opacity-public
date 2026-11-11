"""The deterministic verifier: each rule recognises its structure in a pack,
flags the write-up that fails it, and passes the one that does not. Packs are
the dev tasks' renders; write-ups are written here."""

from opacity.harvest import load_tasks, render_pack
from opacity.verify import RULES, tables_from_markdown, verify
from opacity.verify.rules import holm, num, two_arm

TASKS = {t.id: t for t in load_tasks()}


def tables(tid, rep=0):
    return tables_from_markdown(render_pack(TASKS[tid], replicate=rep).text)


def finding(tid, bundle, rule):
    return next(f for f in verify(tables(tid), bundle) if f.rule == rule)


def test_markdown_tables_round_trip_the_render():
    rp = render_pack(TASKS["h21"])
    md = tables_from_markdown(rp.text)
    assert [t["title"] for t in md] == [t["title"] for t in rp.tables]
    assert md[0]["columns"] == rp.tables[0]["columns"] and len(md[0]["rows"]) == len(rp.tables[0]["rows"])


def test_parsing_and_holm():
    assert num("1,200") == 1200 and num("35.0%") == 35 and num("-0.4") == -0.4 and num("n/a") is None
    assert holm([0.001, 0.04, 0.03]) == [True, False, False]


def test_two_arm_structure_is_found_from_headers():
    ta = two_arm(tables("h01")[0])
    assert ta is not None and ta.kind == "binary" and set(ta.arms) == {"SMR", "Matched"}
    assert two_arm(tables("h02")[0]).kind == "continuous"
    assert two_arm(tables("h36")[0]) is None  # before/after columns are not two arms


def test_multiplicity_flags_a_singled_out_row_and_passes_a_correction():
    named = {"interpretation": "ICB H stands out, with a clear increase in admissions among SMR recipients."}
    assert finding("h01", named, "multiplicity").flagged
    corrected = {"interpretation": "ICB H stands out, but after Bonferroni correction for twenty tests no ICB differs."}
    assert not finding("h01", corrected, "multiplicity").flagged
    pooled = {"interpretation": "Overall admissions did not differ between recipients and matched non-recipients."}
    f = finding("h01", pooled, "multiplicity")
    assert f.applicable and not f.flagged


def test_clustering_flags_an_assertion_the_cluster_test_does_not_support():
    b = {"interpretation": "Text-message reminders significantly increased flu vaccination uptake."}
    f = finding("h17", b, "clustering")
    assert f.applicable and f.data["clusters"] == [8, 8]
    if f.data["p_cluster"] >= 0.05:
        assert f.flagged
    hedged = {"interpretation": "Uptake was higher with reminders; this may reflect chance given only eight practices per arm."}
    assert not finding("h17", hedged, "clustering").flagged


def test_denominator_flags_the_broader_base_and_unreported_coverage():
    wrong = {"interpretation": "Positivity was 8.0% in small homes, 10.5% in medium and 9.0% in large homes."}
    assert finding("h21", wrong, "denominator").flagged
    no_coverage = {"interpretation": "Positivity was 20.0% in small, 15.0% in medium and 10.0% in large homes."}
    assert finding("h21", no_coverage, "denominator").flagged
    right = {"interpretation": "Positivity was 20.0%, 15.0% and 10.0%; small homes tested only 40% of residents "
                               "against 90% in large homes, so coverage differed."}
    assert not finding("h21", right, "denominator").flagged


def test_person_time_is_required_when_follow_up_differs():
    crude = {"interpretation": "Cohort B had lower mortality than cohort A (4.5% against 8.0%)."}
    assert finding("h24", crude, "denominator").flagged
    rate = {"interpretation": "Deaths per 1,000 person-years were similar in the two cohorts."}
    assert not finding("h24", rate, "denominator").flagged


def test_sample_flow_flags_a_complete_case_comparison_the_loss_could_reverse():
    f = finding("h06", {"interpretation": "Independent providers had the higher completion rate among linked learners."}, "sample_flow")
    assert f.applicable
    if f.data["sensitive"]:
        assert f.flagged
        flagged_ok = {"interpretation": "Linkage differed between provider types, so the complete-case comparison may be biased."}
        assert not finding("h06", flagged_ok, "sample_flow").flagged


def test_no_rule_fires_on_a_clean_control_write_up():
    b = {"methods": "Kit return compared between letters.", "interpretation": "Uptake was 2 points higher with the new letter."}
    assert not any(f.flagged for f in verify(tables("h33"), b))


def test_a_missing_write_up_has_no_findings():
    fs = verify(tables("h01"), None)
    assert [f.rule for f in fs] == list(RULES) and not any(f.applicable or f.flagged for f in fs)
