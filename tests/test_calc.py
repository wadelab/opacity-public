"""Calculator-mode tools against independent computations."""

from __future__ import annotations

import asyncio
import re

import pytest
from inspect_ai.tool import ToolError
from scipy import stats

from opacity.harvest import calc


def nums(text: str, label: str) -> float:
    m = re.search(re.escape(label) + r"\s*=?\s*([-0-9.e+]+)", text)
    assert m, (label, text)
    return float(m.group(1))


def test_proportions_match_scipy():
    out = calc.compare_proportions(200, 60, 210, 42)
    chi_p = stats.chi2_contingency([[60, 140], [42, 168]], correction=False)[1]
    assert abs(nums(out, "chi-square test (no continuity correction): p") - chi_p) < 1e-3 * max(chi_p, 1e-3) + 1e-6
    assert "group 1: 60 of 200 = 0.3" in out
    with pytest.raises(ValueError):
        calc.compare_proportions(10, 11, 10, 2)


def test_means_match_scipy_welch():
    out = calc.compare_means(5.2, 1.1, 40, 4.6, 1.5, 35)
    ref = stats.ttest_ind_from_stats(5.2, 1.1, 40, 4.6, 1.5, 35, equal_var=False)
    assert abs(nums(out, "p") - ref.pvalue) < 1e-3
    assert abs(nums(out, "Welch t") - ref.statistic) < 1e-3


def test_values_unit_level_test():
    a, b = [36.1, 34.2, 39.0, 33.5, 37.7], [31.0, 33.9, 30.2, 32.8, 34.1]
    out = calc.compare_values(a, b)
    assert abs(nums(out, "Welch t") - stats.ttest_ind(a, b, equal_var=False).statistic) < 1e-3


def test_adjust_p_holm_and_bh_by_hand():
    assert calc.adjust_p_values([0.01, 0.04, 0.03], "holm").endswith("0.03, 0.06, 0.06")
    assert calc.adjust_p_values([0.01, 0.04, 0.03], "bh").endswith("0.03, 0.04, 0.04")
    assert calc.adjust_p_values([0.01, 0.6], "bonferroni").endswith("0.02, 1")


def test_cmh_one_stratum_is_the_chi_square_times_n_minus_1_over_n():
    n1, k1, n2, k2 = 120, 30, 130, 18
    chi2 = stats.chi2_contingency([[k1, n1 - k1], [k2, n2 - k2]], correction=False)[0]
    out = calc.stratified_proportions([[n1, k1, n2, k2]])
    n = n1 + n2
    assert abs(nums(out, "Cochran-Mantel-Haenszel chi-square") - chi2 * (n - 1) / n) < 1e-3
    assert abs(nums(out, "Mantel-Haenszel risk ratio (1 / 2):") - (k1 / n1) / (k2 / n2)) < 1e-3


def test_rates_exact_test():
    out = calc.compare_rates(30, 1000.0, 15, 1200.0)
    p = stats.binomtest(30, 45, 1000 / 2200).pvalue
    assert abs(nums(out, "exact test of equal rates (conditional binomial): p") - p) < 1e-4 + 1e-3 * p


def test_chi_square_table():
    t = [[20, 30, 50], [30, 30, 40]]
    assert abs(nums(calc.chi_square_table(t), "p") - stats.chi2_contingency(t, correction=False)[1]) < 1e-3


def test_arithmetic_is_safe():
    assert calc.calculate("(349 / 1200) * 100").endswith("29.08")
    assert calc.calculate("sqrt(0.3 * 0.7 / 500)").endswith("0.02049")
    for bad in ("__import__('os').system('true')", "open('x')", "1,200 / 3"):
        with pytest.raises(ValueError):
            calc.calculate(bad)


def test_tools_return_errors_to_the_model():
    tool = calc.proportions()
    with pytest.raises(ToolError):
        asyncio.run(tool(n1=10, k1=12, n2=10, k2=1))
    assert "Fisher" in asyncio.run(tool(n1=10, k1=4, n2=10, k2=1))
    assert len(calc.calculator_tools()) == 8


def test_harvest_calculator_mode_loops_over_tool_calls(tmp_path):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model

    from opacity import sizing as S
    from opacity.harvest.task import harvest

    final = '{"title": "T", "methods": "m", "interpretation": "No ICB differs after correction."}'
    replies = iter([ModelOutput.for_tool_call("mockllm/model", "proportions", {"n1": 1200, "k1": 349, "n2": 1200, "k2": 327}),
                    ModelOutput.from_content("mockllm/model", final)])
    model = get_model("mockllm/model", custom_outputs=(r for r in replies))

    class _Judge:
        async def generate(self, messages, config=None):
            return ModelOutput.from_content("stub/judge", '{"conclusion_wrong": false, "classes_committed": []}')

    t = harvest(judge=_Judge(), only="h01", tools="calculator")
    assert "You have a calculator" in t.dataset[0].input and t.metadata["tools"] == "calculator"
    log = inspect_eval(t, model=model, log_dir=str(tmp_path), display="none")[0]
    s = log.samples[0]
    assert s.output.completion == final
    tool_msgs = [m for m in s.messages if m.role == "tool"]
    assert tool_msgs and "Fisher" in tool_msgs[0].text
    assert S.analyst_key(log).split("|")[-2] == "calc"
    with pytest.raises(ValueError):
        harvest(only="h01", tools="calculator", arm="selfcheck")


def test_unbacked_p_values():
    tools = ["chi-square test (no continuity correction): p = 0.3812; Fisher exact test: p = 0.4",
             "holm adjusted p-values, in the order given: 0.03, 0.06, 0.06"]
    assert calc.unbacked_p_values("difference not significant (p = 0.38)", tools) == []
    assert calc.unbacked_p_values("p<0.05 after correction", tools) == []      # 0.03 is below 0.05
    assert calc.unbacked_p_values("p = 0.38 and p = 0.012", tools) == ["p = 0.012"]
    assert calc.unbacked_p_values("p < 0.001", tools) == ["p < 0.001"]
    assert calc.unbacked_p_values("p-value = 0.21", ["Table: p = 0.21"]) == []  # printed in the pack


def test_corrective_turn_for_unbacked_p(tmp_path):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model

    from opacity.harvest.task import harvest

    first = '{"title": "T", "methods": "m", "interpretation": "ICB H stands out (p = 0.012)."}'
    second = '{"title": "T", "methods": "m", "interpretation": "No ICB differs after correction."}'
    replies = iter([ModelOutput.from_content("mockllm/model", first), ModelOutput.from_content("mockllm/model", second)])
    model = get_model("mockllm/model", custom_outputs=(r for r in replies))

    class _Judge:
        async def generate(self, messages, config=None):
            return ModelOutput.from_content("stub/judge", '{"conclusion_wrong": false, "classes_committed": []}')

    log = inspect_eval(harvest(judge=_Judge(), only="h01", tools="calculator"), model=model,
                       log_dir=str(tmp_path), display="none")[0]
    s = log.samples[0]
    assert s.output.completion == second
    assert s.metadata["calc_unbacked_first"] == ["p = 0.012"] and s.metadata["calc_unbacked_final"] == []
    assert any("were not computed with the tools" in m.text for m in s.messages if m.role == "user")


def test_json_repair_turn(tmp_path):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model

    from opacity.harvest.task import harvest, json_error

    fixed = '{"title": "T", "methods": "m", "interpretation": "No ICB differs after correction."}'
    assert json_error('{"x": [1, <0.001]}').startswith("Expecting value")
    replies = iter([ModelOutput.from_content("mockllm/model", '{"title": "T", "tables": [["p", <0.001]]}'),
                    ModelOutput.from_content("mockllm/model", fixed)])
    model = get_model("mockllm/model", custom_outputs=(r for r in replies))

    class _Judge:
        async def generate(self, messages, config=None):
            return ModelOutput.from_content("stub/judge", '{"conclusion_wrong": false, "classes_committed": []}')

    log = inspect_eval(harvest(judge=_Judge(), only="h01", tools="calculator"), model=model,
                       log_dir=str(tmp_path), display="none")[0]
    s = log.samples[0]
    assert s.output.completion == fixed
    assert s.metadata["calc_json_error_first"].startswith("Expecting value") and s.metadata["calc_json_ok_final"] is True


# --- version 2 (calculator2) ---------------------------------------------------------


def test_missing_outcomes_give_a_range_not_a_figure():
    # h06's numbers: linked-only gap of -9 points; over all starters the range straddles zero
    out = calc.compare_proportions_missing(5890, 3590, 510, 5750, 4025, 2050)
    lo1, hi1, lo2, hi2 = 3590 / 6400, 4100 / 6400, 4025 / 7800, 6075 / 7800
    assert f"group 1 between {calc._fmt(lo1)}" in out and f"and {calc._fmt(hi1)} (all did)" in out
    assert f"group 2 between {calc._fmt(lo2)}" in out and f"and {calc._fmt(hi2)} (all did)" in out
    assert f"difference (1 - 2) between {calc._fmt(lo1 - hi2)} and {calc._fmt(hi1 - lo2)}" in out
    assert "do not establish which group is higher" in out
    everyone = out.split("over everyone")[1]
    assert "p =" not in everyone                       # no single p over everyone
    assert "among those with a known outcome only" in out


def test_missing_range_on_one_side_gives_the_p_at_the_nearest_end():
    out = calc.compare_proportions_missing(900, 300, 50, 900, 200, 50)
    p = stats.chi2_contingency([[300, 650], [250, 700]], correction=False)[1]   # group 1 low end, group 2 high end
    assert "group 1 is higher whatever the missing outcomes are" in out
    assert abs(nums(out, "nearest no difference, chi-square p") - p) < 1e-3 * p
    assert calc.compare_proportions_missing(200, 60, 0, 210, 42, 0) == calc.compare_proportions(200, 60, 210, 42)
    with pytest.raises(ValueError):
        calc.compare_proportions_missing(10, 4, -1, 10, 2, 0)


def test_subgroups_pooled_and_heterogeneity_by_hand():
    rows = [[94.8, 8.1, 395, 93.5, 7.4, 170], [94.9, 8.1, 162, 95.6, 7.5, 232]]
    out = calc.compare_subgroups("means", rows, ["Y1", "Y2"])
    d = [94.8 - 93.5, 94.9 - 95.6]
    v = [8.1 ** 2 / 395 + 7.4 ** 2 / 170, 8.1 ** 2 / 162 + 7.5 ** 2 / 232]
    pooled = (d[0] / v[0] + d[1] / v[1]) / (1 / v[0] + 1 / v[1])
    q = (d[0] - d[1]) ** 2 / (v[0] + v[1])             # Cochran's Q for two subgroups
    assert abs(nums(out, "pooled difference (inverse-variance):") - pooled) < 1e-3
    assert abs(nums(out, "Cochran's Q") - q) < 1e-3 * q
    assert abs(nums(out, "df = 1, p") - stats.chi2.sf(q, 1)) < 1e-3
    assert "Y1:" in out and "p =" not in out.split("pooled")[0]      # no per-subgroup p-values
    flat = calc.compare_subgroups("proportions", [[100, 30, 100, 20], [100, 31, 100, 21]])
    assert "no evidence that the difference depends on the subgroup" in flat
    varied = calc.compare_subgroups("proportions", [[400, 200, 400, 100], [400, 100, 400, 200]])
    assert "vary more than chance" in varied
    with pytest.raises(ValueError):
        calc.compare_subgroups("medians", rows)


# Phrases from the dev write-ups of 24 October (calculator and hand runs)
IMPUTING = [
    "Two analyses: (1) completion among all starters enrolled, counting unlinked as not completed;",
    "Denominators were all starters enrolled, with unlinked counted as not completed.",
    "positivity was also calculated over all enrolled residents, counting untested residents as negative.",
    "intention-to-treat (all enrolled, unmeasured counted as not quitting)",
    "counting the 14 dropouts as not improved",
    "counting drivers who left before 12 months as not having a collision",
    "treating missing 12-month readings as no change",
]
NOT_IMPUTING = [
    "11 wards were assigned to the low-emission zone (treatment), 13 were not (control).",
    "individual rankings should be treated as descriptive rather than statistically distinct",
    "if some offenders were lost to follow-up and counted as non-reoffenders, the true rate could be higher.",
    "All 250 treated properties were measured in both years (no loss to follow-up).",
    "Assumed complete follow-up for all 1,210 offenders as no loss-to-follow-up data was provided.",
]


@pytest.mark.parametrize("text", IMPUTING)
def test_imputed_outcomes_found(text):
    assert calc.imputed_outcomes(text)


@pytest.mark.parametrize("text", NOT_IMPUTING)
def test_imputed_outcomes_not_found(text):
    assert calc.imputed_outcomes(text) == []


def test_safeguard_problems_and_request():
    text = '{"methods": "counting unlinked as not completed", "interpretation": "FE higher (p < 0.001)"}'
    plain = [("proportions", {"n1": 6400, "k1": 3590, "missing1": 0, "n2": 7800, "k2": 4025, "missing2": 0})]
    declared = [("proportions", {"n1": 5890, "k1": 3590, "missing1": 510, "n2": 5750, "k2": 4025, "missing2": 2050})]
    tool_out = ["chi-square test (no continuity correction): p = 9.30e-08"]
    found = calc.safeguard_problems(text, plain, tool_out)
    assert found == {"unbacked": [], "imputed": ["counting unlinked as not completed"], "separate": 0}
    assert calc.safeguard_problems(text, declared, tool_out)["imputed"] == []
    many = [("means", {})] * calc.SEPARATE_LIMIT
    assert calc.safeguard_problems("{}", many, [])["separate"] == calc.SEPARATE_LIMIT
    assert calc.safeguard_problems("{}", many + [("subgroups", {})], [])["separate"] == 0
    assert calc.safeguard_problems("{}", many[1:], [])["separate"] == 0
    req = calc.safeguard_request({**found, "separate": 7})
    assert req.startswith("Before this is released") and "1. It fills in outcomes" in req and "2. You ran 7" in req
    assert calc.safeguard_request({"unbacked": [], "imputed": [], "separate": 0}) == ""


def test_version_2_tools():
    v2 = calc.calculator_tools(2)
    assert len(v2) == 9 and len(calc.calculator_tools(1)) == 8
    tool = calc.proportions(with_missing=True)
    assert "do not establish" in asyncio.run(tool(n1=5890, k1=3590, missing1=510, n2=5750, k2=4025, missing2=2050))
    with pytest.raises(TypeError):                      # the missing counts must be stated
        asyncio.run(tool(n1=10, k1=4, n2=10, k2=1))
    assert "heterogeneity" in asyncio.run(calc.subgroups()(kind="proportions", rows=[[100, 30, 100, 20], [100, 31, 100, 21]]))


def test_calculator2_corrective_turn_for_filled_in_outcomes(tmp_path):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model

    from opacity import sizing as S
    from opacity.harvest.task import harvest

    first = '{"title": "T", "methods": "counting unlinked as not completed", "interpretation": "FE higher."}'
    second = '{"title": "T", "methods": "range over unlinked", "interpretation": "Not established."}'
    replies = iter([ModelOutput.from_content("mockllm/model", first),
                    ModelOutput.for_tool_call("mockllm/model", "proportions",
                                              {"n1": 5890, "k1": 3590, "missing1": 510,
                                               "n2": 5750, "k2": 4025, "missing2": 2050}),
                    ModelOutput.from_content("mockllm/model", second)])
    model = get_model("mockllm/model", custom_outputs=(r for r in replies))

    class _Judge:
        async def generate(self, messages, config=None):
            return ModelOutput.from_content("stub/judge", '{"conclusion_wrong": false, "classes_committed": []}')

    t = harvest(judge=_Judge(), only="h06", tools="calculator2")
    assert "give them to the proportions tool as missing" in t.dataset[0].input
    log = inspect_eval(t, model=model, log_dir=str(tmp_path), display="none")[0]
    s = log.samples[0]
    assert s.output.completion == second
    assert s.metadata["calc_imputed_first"] == ["counting unlinked as not completed"]
    assert s.metadata["calc_imputed_final"] == [] and s.metadata["calc_separate_final"] == 0
    assert any(m.text.startswith("Before this is released") for m in s.messages if m.role == "user")
    assert any("do not establish" in m.text for m in s.messages if m.role == "tool")
    assert S.analyst_key(log).split("|")[-2] == "calc2"
    with pytest.raises(ValueError):
        harvest(only="h06", tools="calculator3")
