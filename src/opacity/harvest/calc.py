"""Calculator mode for the analyst (exploratory, 24 October 2026).

In pack mode the analyst sees summary tables and computes any test by hand,
and on the second sealed set several of its remaining errors were hand
arithmetic on the right analysis. The inside analyst in safe-tre-agent runs its
tests through the gateway, so this mode gives the harvest analyst the same:
a few statistical tools that compute from the numbers in the tables.
`-T tools=calculator` turns it on; the note below is added to the request.

Each tool is a plain function (tested against scipy in tests/test_calc.py)
wrapped as an Inspect tool. A bad argument returns an error message to the
model rather than failing the sample.

Version 2 (`-T tools=calculator2`, 25 October 2026) pairs the power with
guardrails, after version 1 moved errors on dev and held-out rather than
removing them (report 6c). On h06 the analyst filled in outcomes nobody
observed (unlinked learners counted as not completing, as rule 3 of the
rules prompt asks) and headlined that comparison's p < 0.001; on h02 it named
subgroups from separate per-subgroup tests. In version 2 `proportions` takes
the units whose outcome is not known and returns the range those outcomes
allow, not a single figure; `subgroups` tests whether a difference varies
between subgroups; and `safeguard_problems` is the check behind the note.
Version 1 is unchanged, so its runs stay reproducible.
"""

from __future__ import annotations

import ast
import math
import operator
import re

from inspect_ai.tool import Tool, ToolError, tool
from scipy import stats

CALCULATOR_NOTE = (
    "You have a calculator: tools that compute statistical tests from the "
    "counts, means and rates in the tables, adjust p-values for multiple "
    "comparisons, and evaluate arithmetic. Use them for every test "
    "statistic, interval and p-value you report, and report only numbers "
    "that the tools returned or that appear in the tables. When you have "
    "finished, reply with the JSON object as instructed and nothing else."
)

CALCULATOR_NOTE_V2 = CALCULATOR_NOTE + (
    " When some units' outcomes were not observed (not linked, lost to "
    "follow-up, untested, no response), give them to the proportions tool as "
    "missing rather than filling them in; it returns the range of results "
    "they allow. When you compare the same two groups within several "
    "subgroups, use the subgroups tool to test whether the difference varies "
    "between them before saying it is larger or smaller in any one."
)

VERSIONS = {"calculator": 1, "calculator2": 2}
"""`-T tools=` values and the calculator version each selects."""


def calculator_note(version: int = 1) -> str:
    return CALCULATOR_NOTE if version == 1 else CALCULATOR_NOTE_V2


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if x == 0:
        return "0"
    if abs(x) < 1e-4:
        return f"{x:.2e}"
    return f"{x:.4g}"


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


# --- the computations ----------------------------------------------------------


def compare_proportions(n1: int, k1: int, n2: int, k2: int) -> str:
    _need(n1 > 0 and n2 > 0, "group sizes must be positive")
    _need(0 <= k1 <= n1 and 0 <= k2 <= n2, "counts must lie between 0 and the group size")
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    if min(k1 + k2, n1 + n2 - k1 - k2) == 0:
        chi_p = float("nan")
    else:
        chi_p = stats.chi2_contingency(table, correction=False)[1]
    fisher_p = stats.fisher_exact(table)[1]
    out = [f"group 1: {k1} of {n1} = {_fmt(p1)}", f"group 2: {k2} of {n2} = {_fmt(p2)}",
           f"difference (1 - 2): {_fmt(d)}, 95% CI {_fmt(d - 1.96 * se)} to {_fmt(d + 1.96 * se)}"]
    if k1 > 0 and k2 > 0:
        rr = p1 / p2
        se_log = math.sqrt(1 / k1 - 1 / n1 + 1 / k2 - 1 / n2)
        out.append(f"risk ratio (1 / 2): {_fmt(rr)}, 95% CI {_fmt(rr * math.exp(-1.96 * se_log))} "
                   f"to {_fmt(rr * math.exp(1.96 * se_log))}")
    out.append(f"chi-square test (no continuity correction): p = {_fmt(chi_p)}; Fisher exact test: p = {_fmt(fisher_p)}")
    return "\n".join(out)


def compare_proportions_missing(n1: int, k1: int, m1: int, n2: int, k2: int, m2: int) -> str:
    """Two proportions when m1 and m2 units have an outcome nobody observed.
    The comparison among known outcomes, then the range over everyone: each
    missing outcome anything from absent to present (worst-case bounds). No
    single p over everyone unless the whole range is on one side of zero,
    and then only the p at the end nearest no difference."""
    _need(m1 >= 0 and m2 >= 0, "missing counts cannot be negative")
    if m1 == 0 and m2 == 0:
        return compare_proportions(n1, k1, n2, k2)
    _need(n1 > 0 and n2 > 0, "each group needs some units with a known outcome")
    known = compare_proportions(n1, k1, n2, k2)
    t1, t2 = n1 + m1, n2 + m2
    lo1, hi1, lo2, hi2 = k1 / t1, (k1 + m1) / t1, k2 / t2, (k2 + m2) / t2
    dlo, dhi = lo1 - hi2, hi1 - lo2
    out = [f"outcome known for {n1} of {t1} in group 1 ({m1} not known) and {n2} of {t2} in group 2 ({m2} not known)",
           "among those with a known outcome only (this holds only if the missing outcomes are like the known ones):"]
    out += ["  " + line for line in known.splitlines()]
    out += [f"over everyone, with each of the {m1 + m2} missing outcomes anything from absent to present:",
            f"  group 1 between {_fmt(lo1)} (none of the {m1} missing had the outcome) and {_fmt(hi1)} (all did)",
            f"  group 2 between {_fmt(lo2)} (none of the {m2} missing had the outcome) and {_fmt(hi2)} (all did)",
            f"  difference (1 - 2) between {_fmt(dlo)} and {_fmt(dhi)}"]
    if dlo > 0 or dhi < 0:
        a, b = (k1, k2 + m2) if dlo > 0 else (k1 + m1, k2)
        p = stats.chi2_contingency([[a, t1 - a], [b, t2 - b]], correction=False)[1]
        out.append(f"  group {1 if dlo > 0 else 2} is higher whatever the missing outcomes are; at the end of the "
                   f"range nearest no difference, chi-square p = {_fmt(float(p))}")
    else:
        out.append("  the missing outcomes could remove or reverse the difference, so the data do not establish "
                   "which group is higher. Report this range: a single figure that fills in the missing outcomes "
                   "(for example counting them all as not having the outcome) is a guess, not a result.")
    return "\n".join(out)


def compare_means(mean1: float, sd1: float, n1: int, mean2: float, sd2: float, n2: int) -> str:
    _need(n1 > 1 and n2 > 1, "each group needs at least two observations")
    _need(sd1 >= 0 and sd2 >= 0, "standard deviations cannot be negative")
    v1, v2 = sd1 ** 2 / n1, sd2 ** 2 / n2
    _need(v1 + v2 > 0, "both standard deviations are zero")
    se = math.sqrt(v1 + v2)
    t = (mean1 - mean2) / se
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    q = stats.t.ppf(0.975, df)
    d = mean1 - mean2
    return (f"difference in means (1 - 2): {_fmt(d)}, 95% CI {_fmt(d - q * se)} to {_fmt(d + q * se)}\n"
            f"Welch t = {_fmt(t)}, df = {_fmt(df)}, p = {_fmt(p)}")


def compare_values(group_1: list[float], group_2: list[float]) -> str:
    a, b = [float(x) for x in group_1], [float(x) for x in group_2]
    _need(len(a) > 1 and len(b) > 1, "each group needs at least two values")
    w = stats.ttest_ind(a, b, equal_var=False)
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    m1, m2 = sum(a) / len(a), sum(b) / len(b)
    s1, s2 = stats.tstd(a), stats.tstd(b)
    return (f"group 1: n = {len(a)}, mean {_fmt(m1)}, SD {_fmt(s1)}\n"
            f"group 2: n = {len(b)}, mean {_fmt(m2)}, SD {_fmt(s2)}\n"
            f"difference in means (1 - 2): {_fmt(m1 - m2)}\n"
            f"Welch t = {_fmt(float(w.statistic))}, p = {_fmt(float(w.pvalue))}; "
            f"Mann-Whitney U p = {_fmt(float(mw.pvalue))}")


def chi_square_table(table: list[list[float]]) -> str:
    _need(len(table) >= 2 and all(len(r) == len(table[0]) for r in table) and len(table[0]) >= 2,
          "the table needs at least two rows and two columns, all of the same length")
    _need(all(x >= 0 for r in table for x in r), "counts cannot be negative")
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    return (f"chi-square = {_fmt(float(chi2))}, df = {dof}, p = {_fmt(float(p))}; "
            f"smallest expected count {_fmt(float(expected.min()))}")


def compare_rates(events1: int, time1: float, events2: int, time2: float) -> str:
    _need(time1 > 0 and time2 > 0, "person-time must be positive")
    _need(events1 >= 0 and events2 >= 0, "event counts cannot be negative")
    r1, r2 = events1 / time1, events2 / time2
    out = [f"rate 1: {events1} / {_fmt(time1)} = {_fmt(r1)} per unit time",
           f"rate 2: {events2} / {_fmt(time2)} = {_fmt(r2)} per unit time"]
    if events1 > 0 and events2 > 0:
        rr = r1 / r2
        se = math.sqrt(1 / events1 + 1 / events2)
        out.append(f"rate ratio (1 / 2): {_fmt(rr)}, 95% CI {_fmt(rr * math.exp(-1.96 * se))} to {_fmt(rr * math.exp(1.96 * se))}")
    if events1 + events2 > 0:
        p = stats.binomtest(events1, events1 + events2, time1 / (time1 + time2)).pvalue
        out.append(f"exact test of equal rates (conditional binomial): p = {_fmt(float(p))}")
    return "\n".join(out)


def adjust_p_values(p_values: list[float], method: str = "holm") -> str:
    ps = [float(p) for p in p_values]
    _need(len(ps) >= 1 and all(0 <= p <= 1 for p in ps), "p-values must lie between 0 and 1")
    method = method.lower().strip()
    _need(method in ("holm", "bonferroni", "bh"), "method must be holm, bonferroni or bh")
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    if method == "bonferroni":
        adj = [min(1.0, p * m) for p in ps]
    elif method == "holm":
        run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (m - rank) * ps[i]))
            adj[i] = run
    else:
        run = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            run = min(run, ps[i] * m / (rank + 1))
            adj[i] = min(1.0, run)
    return f"{method} adjusted p-values, in the order given: " + ", ".join(_fmt(a) for a in adj)


def stratified_proportions(strata: list[list[float]]) -> str:
    _need(len(strata) >= 1 and all(len(s) == 4 for s in strata),
          "give one [n1, k1, n2, k2] per stratum")
    num_or = den_or = num_rr = den_rr = 0.0
    sum_a = sum_e = sum_v = 0.0
    for n1, k1, n2, k2 in strata:
        _need(n1 > 0 and n2 > 0 and 0 <= k1 <= n1 and 0 <= k2 <= n2, "each stratum needs valid counts")
        a, b, c, d = k1, n1 - k1, k2, n2 - k2
        n = n1 + n2
        num_or += a * d / n
        den_or += b * c / n
        num_rr += a * n2 / n
        den_rr += c * n1 / n
        m1, m0 = a + c, b + d
        sum_a += a
        sum_e += n1 * m1 / n
        if n > 1:
            sum_v += n1 * n2 * m1 * m0 / (n * n * (n - 1))
    out = []
    out.append(f"Mantel-Haenszel risk ratio (1 / 2): {_fmt(num_rr / den_rr) if den_rr else 'n/a'}")
    out.append(f"Mantel-Haenszel odds ratio (1 / 2): {_fmt(num_or / den_or) if den_or else 'n/a'}")
    if sum_v > 0:
        chi2 = (sum_a - sum_e) ** 2 / sum_v
        out.append(f"Cochran-Mantel-Haenszel chi-square = {_fmt(chi2)}, df = 1, p = {_fmt(float(stats.chi2.sf(chi2, 1)))}")
    return "\n".join(out)


def compare_subgroups(kind: str, rows: list[list[float]], labels: list[str] | None = None) -> str:
    """The same two-group difference in several subgroups: each difference
    with its interval (normal approximation), the inverse-variance pooled
    difference, and Cochran's Q for whether the difference varies between
    subgroups. No per-subgroup p-values: reading those as a differential
    effect is the error this tool is the guardrail for."""
    kind = str(kind).lower().strip()
    _need(kind in ("means", "proportions"), "kind must be means or proportions")
    width = 6 if kind == "means" else 4
    _need(len(rows) >= 2 and all(len(r) == width for r in rows),
          "give at least two subgroups, each as [mean1, sd1, n1, mean2, sd2, n2] for means "
          "or [n1, k1, n2, k2] for proportions")
    labels = [str(x) for x in labels] if labels else [f"subgroup {i + 1}" for i in range(len(rows))]
    _need(len(labels) == len(rows), "give one label per subgroup, or none")
    ds, vs = [], []
    for r in rows:
        if kind == "means":
            m1, s1, n1, m2, s2, n2 = (float(x) for x in r)
            _need(n1 > 1 and n2 > 1 and s1 >= 0 and s2 >= 0, "each subgroup needs n above 1 and SDs not negative")
            ds.append(m1 - m2)
            vs.append(s1 ** 2 / n1 + s2 ** 2 / n2)
        else:
            n1, k1, n2, k2 = (float(x) for x in r)
            _need(n1 > 0 and n2 > 0 and 0 <= k1 <= n1 and 0 <= k2 <= n2, "each subgroup needs valid counts")
            ds.append(k1 / n1 - k2 / n2)
            # a 0 or 100% cell has no binomial variance; add half a unit so the subgroup still weighs in
            q1, q2 = (k1 + 0.5) / (n1 + 1), (k2 + 0.5) / (n2 + 1)
            vs.append(q1 * (1 - q1) / n1 + q2 * (1 - q2) / n2)
        _need(vs[-1] > 0, "a subgroup's difference has no variance (both SDs zero)")
    w = [1 / v for v in vs]
    pooled = sum(wi * di for wi, di in zip(w, ds)) / sum(w)
    se = math.sqrt(1 / sum(w))
    p_pooled = 2 * stats.norm.sf(abs(pooled / se))
    q = sum(wi * (di - pooled) ** 2 for wi, di in zip(w, ds))
    df = len(ds) - 1
    p_het = float(stats.chi2.sf(q, df))
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    out = ["difference (1 - 2) in each subgroup, 95% CI:"]
    out += [f"  {lab}: {_fmt(d)} ({_fmt(d - 1.96 * math.sqrt(v))} to {_fmt(d + 1.96 * math.sqrt(v))})"
            for lab, d, v in zip(labels, ds, vs)]
    out.append(f"pooled difference (inverse-variance): {_fmt(pooled)}, 95% CI {_fmt(pooled - 1.96 * se)} to "
               f"{_fmt(pooled + 1.96 * se)}, p = {_fmt(float(p_pooled))}")
    out.append(f"heterogeneity: Cochran's Q = {_fmt(q)}, df = {df}, p = {_fmt(p_het)}; I-squared = {_fmt(100 * i2)}%")
    if p_het >= 0.05:
        out.append("The subgroups' differences vary no more than chance alone would make them: there is no evidence "
                   "that the difference depends on the subgroup. Report the pooled difference, and do not name a "
                   "subgroup where it is larger, smaller, present or absent.")
    else:
        out.append("The differences vary more than chance alone would make them: the difference is not the same in "
                   "every subgroup. Describe how it varies, and unless the request named these subgroups in "
                   "advance, call the pattern exploratory.")
    return "\n".join(out)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "exp": math.exp, "abs": abs, "round": round,
          "min": min, "max": max, "sum": lambda *x: sum(x)}


def calculate(expression: str) -> str:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("only numbers, + - * / ** %, brackets and sqrt, log, exp, abs, round, min, max, sum are allowed")
    _need(len(expression) <= 500, "expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
        return f"{expression} = {_fmt(float(ev(tree)))}"
    except ZeroDivisionError:
        raise ValueError("division by zero") from None
    except SyntaxError:
        raise ValueError("could not read the expression; write numbers without thousands separators") from None


# --- does every reported p-value come from a tool? --------------------------------

_P_REPORTED = re.compile(r"\bp\s*(?:-?value)?\s*(=|<|>|≈|~|≤|≥)\s*(\d*\.?\d+(?:[eE]-?\d+)?)", re.IGNORECASE)
_P_SOURCE = re.compile(r"\bp\s*=\s*(\d*\.?\d+(?:[eE]-?\d+)?)", re.IGNORECASE)


def _source_ps(texts: list[str]) -> list[float]:
    """Every p the tools returned (or the pack printed): `p = x` forms and the
    lists that adjust_p returns."""
    out: list[float] = []
    for t in texts:
        out += [float(x) for x in _P_SOURCE.findall(t)]
        if "adjusted p-values" in t:
            out += [float(x) for x in re.findall(r"(\d*\.?\d+(?:e-?\d+)?)", t.split(":", 1)[-1])]
    return out


def unbacked_p_values(writeup: str, sources: list[str]) -> list[str]:
    """p-values in the write-up that no tool returned and the pack does not
    print. `p = 0.38` is backed by a source p that rounds to it; `p < 0.05`
    by a source p below 0.05, `p > 0.1` by one above. A hint in the prompt
    asks for tools; this is the check behind it."""
    src = _source_ps(sources)
    bad = []
    for op, num in _P_REPORTED.findall(writeup):
        r = float(num)
        if op in ("<", "≤"):
            ok = any(x <= r for x in src)
        elif op in (">", "≥"):
            ok = any(x >= r for x in src)
        else:
            d = len(num.split(".")[1]) if "." in num and "e" not in num.lower() else None
            ok = any((round(x, d) == r if d is not None else abs(x - r) <= 0.1 * r) for x in src)
        if not ok:
            bad.append(f"p {op} {num}")
    return list(dict.fromkeys(bad))


UNBACKED_REQUEST = (
    "These p-values in your write-up were not computed with the tools and do "
    "not appear in the tables: {items}. Compute them with the tools, or remove "
    "them, then reply with the complete JSON object again and nothing else."
)


# --- version 2: the checks behind the note ----------------------------------------

_MISSING = (r"(?:unlinked|untested|unmeasured|unobserved|unknown|missing|lost|leavers?|non-?respon\w*|"
            r"non-?complet\w*|dropouts?|drop-?outs?|withdr\w*|deaths?|died|"
            r"not\s+(?:linked|measured|tested|followed|observed|assessed|recorded|matched|traced|reached)|"
            r"(?:who|that)\s+(?:left|died|dropped|withdrew|moved))")
_FILL = r"(?:count|treat|assum|imput|cod|classif|scor|consider|regard)\w*"
_ABSENT = (r"(?:not\b|non-?|negative|failures?|failed|fail\b|no\b|none\b|zero|having\s+no|without|unsuccess\w*|"
           r"did\s+not)\S*(?:\s+[\w-]+)?")  # to the end of the next word, so the phrase quotes readably
_IMPUTED = re.compile(
    # "counting unlinked as not completed", "treating the 14 dropouts as failures"
    _FILL + r"\s+(?:\S+\s+){0,4}?" + _MISSING + r"[^.;:]{0,40}?\bas\s+(?:\S+\s+){0,2}?" + _ABSENT
    # "with unlinked counted as not completed", "unlinked learners were counted as non-completers"
    + r"|" + _MISSING + r"\s+(?:\S+\s+){0,2}?" + _FILL + r"\s+as\s+(?:\S+\s+){0,2}?" + _ABSENT,
    re.IGNORECASE)


def imputed_outcomes(writeup: str) -> list[str]:
    """Phrases in which the write-up fills in outcomes nobody observed, such
    as "counting unlinked as not completed". Calibrated on the dev write-ups
    of 24 October (calculator and hand runs): it fires on the class 2 tasks
    and on h21 and h22, where untested residents and deaths were filled in,
    and on no held-out write-up outside class 2."""
    return list(dict.fromkeys(m.group(0) for m in _IMPUTED.finditer(writeup or "")))


TWO_GROUP_TOOLS = ("proportions", "means", "values")
SEPARATE_LIMIT = 7
"""Separate two-group comparisons at which the subgroup question is asked.
Set on the dev replies of version 1 (both runs, 432 write-ups): 41 of 48 on
the multiplicity tasks made 7 or more, and none on the confounding tasks,
which test up to six strata and then combine them with `stratified`."""


def safeguard_problems(writeup: str, calls: list[tuple[str, dict]], sources: list[str]) -> dict:
    """What version 2 checks in a reply before release. `calls` are the
    (tool name, arguments) the analyst made; `sources` the tool outputs and
    the pack. Returns the unbacked p-values; the filling-in phrases, if no
    proportions call declared missing outcomes; and the number of separate
    two-group comparisons, if there were SEPARATE_LIMIT or more and no
    subgroups call (0 otherwise)."""
    declared = any(name == "proportions" and ((args.get("missing1") or 0) > 0 or (args.get("missing2") or 0) > 0)
                   for name, args in calls)
    separate = sum(name in TWO_GROUP_TOOLS for name, _ in calls)
    tested = any(name == "subgroups" for name, _ in calls)
    return {"unbacked": unbacked_p_values(writeup, sources),
            "imputed": [] if declared else imputed_outcomes(writeup),
            "separate": separate if separate >= SEPARATE_LIMIT and not tested else 0}


SAFEGUARD_ITEMS = {
    "unbacked": ("These p-values were not computed with the tools and do not appear in the tables: {}. "
                 "Compute them with the tools, or remove them."),
    "imputed": ("It fills in outcomes that nobody observed ({}). A rate or test that counts them as having, "
                "or not having, the outcome is a guess, not a result. Run the proportions tool with missing1 "
                "and missing2 set to the units whose outcome is not known; it returns the range of results "
                "those outcomes allow. Report that range and base the conclusion on it, not on a filled-in "
                "figure."),
    "separate": ("You ran {} separate two-group comparisons and no test of whether the difference varies "
                 "between them. If they compare the same two groups within different subgroups (for example "
                 "year groups, areas or sites) and the write-up names, ranks or singles out a subgroup where "
                 "the difference is larger, smaller, present or absent, run the subgroups tool on them and let "
                 "its heterogeneity test decide. If they do not, leave that part as it is."),
}


def safeguard_request(problems: dict) -> str:
    """One corrective turn for everything `safeguard_problems` found, or "" if
    nothing."""
    items = []
    for key in ("unbacked", "imputed", "separate"):
        v = problems.get(key)
        if v:
            items.append(SAFEGUARD_ITEMS[key].format(", ".join(f'"{x}"' if key == "imputed" else x for x in v)
                                                     if isinstance(v, list) else v))
    if not items:
        return ""
    return ("Before this is released, check these points in your write-up:\n\n"
            + "\n\n".join(f"{i}. {t}" for i, t in enumerate(items, 1))
            + "\n\nThen reply with the complete JSON object again and nothing else.")


# --- the Inspect tools -----------------------------------------------------------


def _run(fn, *args):
    try:
        return fn(*args)
    except ValueError as e:
        raise ToolError(str(e)) from None


@tool
def proportions(with_missing: bool = False) -> Tool:
    if with_missing:
        async def execute_missing(n1: int, k1: int, missing1: int, n2: int, k2: int, missing2: int) -> str:
            """Compare two proportions, k1 of n1 in group 1 against k2 of n2 in
            group 2, allowing for units whose outcome nobody observed.

            Args:
                n1: Units in group 1 whose outcome is known (for example linked, followed up, tested or responding). Do not include the missing ones.
                k1: Of those n1, the number with the outcome.
                missing1: Units in group 1 whose outcome is not known (not linked, lost to follow-up, untested, no response); 0 if every outcome is known.
                n2: Units in group 2 whose outcome is known. Do not include the missing ones.
                k2: Of those n2, the number with the outcome.
                missing2: Units in group 2 whose outcome is not known; 0 if every outcome is known.

            Returns:
                With nothing missing: both proportions, the difference and risk
                ratio with 95% intervals, and chi-square and Fisher p-values.
                With missing outcomes: the same among known outcomes, and the
                range of the comparison over everyone that the missing
                outcomes allow.
            """
            return _run(compare_proportions_missing, n1, k1, missing1, n2, k2, missing2)
        return execute_missing

    async def execute(n1: int, k1: int, n2: int, k2: int) -> str:
        """Compare two proportions: k1 of n1 in group 1 against k2 of n2 in group 2.

        Args:
            n1: Number of units in group 1.
            k1: Number of those with the outcome in group 1.
            n2: Number of units in group 2.
            k2: Number of those with the outcome in group 2.

        Returns:
            Both proportions, the difference and risk ratio with 95% intervals,
            and chi-square and Fisher exact p-values.
        """
        return _run(compare_proportions, n1, k1, n2, k2)
    return execute


@tool
def means() -> Tool:
    async def execute(mean1: float, sd1: float, n1: int, mean2: float, sd2: float, n2: int) -> str:
        """Compare two means from summary statistics (Welch's t-test).

        Args:
            mean1: Mean in group 1.
            sd1: Standard deviation in group 1.
            n1: Number of independent units in group 1.
            mean2: Mean in group 2.
            sd2: Standard deviation in group 2.
            n2: Number of independent units in group 2.

        Returns:
            The difference with a 95% interval, Welch t, degrees of freedom and p.
        """
        return _run(compare_means, mean1, sd1, n1, mean2, sd2, n2)
    return execute


@tool
def values() -> Tool:
    async def execute(group_1: list[float], group_2: list[float]) -> str:
        """Compare two groups of individual values, one value per independent
        unit (for example one rate or mean per site, school or store).

        Args:
            group_1: The values for group 1.
            group_2: The values for group 2.

        Returns:
            Group means and SDs, Welch t-test p and Mann-Whitney p.
        """
        return _run(compare_values, group_1, group_2)
    return execute


@tool
def chi_square() -> Tool:
    async def execute(table: list[list[float]]) -> str:
        """Chi-square test of independence on a table of counts (rows by columns).

        Args:
            table: The counts, one list per row.

        Returns:
            Chi-square, degrees of freedom, p and the smallest expected count.
        """
        return _run(chi_square_table, table)
    return execute


@tool
def rates() -> Tool:
    async def execute(events1: int, time1: float, events2: int, time2: float) -> str:
        """Compare two event rates per person-time (or per any exposure).

        Args:
            events1: Events in group 1.
            time1: Person-time (or exposure) in group 1.
            events2: Events in group 2.
            time2: Person-time (or exposure) in group 2.

        Returns:
            Both rates, the rate ratio with a 95% interval and an exact p.
        """
        return _run(compare_rates, events1, time1, events2, time2)
    return execute


@tool
def adjust_p() -> Tool:
    async def execute(p_values: list[float], method: str = "holm") -> str:
        """Adjust a set of p-values for multiple comparisons.

        Args:
            p_values: The unadjusted p-values.
            method: holm, bonferroni or bh (Benjamini-Hochberg).

        Returns:
            The adjusted p-values in the order given.
        """
        return _run(adjust_p_values, p_values, method)
    return execute


@tool
def stratified() -> Tool:
    async def execute(strata: list[list[float]]) -> str:
        """Compare two groups' proportions within strata and combine them
        (Mantel-Haenszel), to adjust for a stratifying factor.

        Args:
            strata: One [n1, k1, n2, k2] per stratum: units and outcomes in group 1, then group 2.

        Returns:
            The Mantel-Haenszel risk ratio and odds ratio and the Cochran-Mantel-Haenszel p.
        """
        return _run(stratified_proportions, strata)
    return execute


@tool
def subgroups() -> Tool:
    async def execute(kind: str, rows: list[list[float]], labels: list[str] | None = None) -> str:
        """Compare the same two groups within several subgroups at once, and
        test whether the difference varies between subgroups (heterogeneity),
        for questions such as "for which pupils did it work best".

        Args:
            kind: "means" or "proportions".
            rows: One row per subgroup, group 1 first: [mean1, sd1, n1, mean2, sd2, n2] for means, or [n1, k1, n2, k2] for proportions.
            labels: Names of the subgroups, in the same order (optional).

        Returns:
            Each subgroup's difference with a 95% interval, the pooled
            difference with its p, and the heterogeneity test.
        """
        return _run(compare_subgroups, kind, rows, labels)
    return execute


@tool
def arithmetic() -> Tool:
    async def execute(expression: str) -> str:
        """Evaluate an arithmetic expression, for rates, shares and denominators.

        Args:
            expression: For example "(349 / 1200) * 100" or "sqrt(0.3 * 0.7 / 500)".

        Returns:
            The value.
        """
        return _run(calculate, expression)
    return execute


def calculator_tools(version: int = 1) -> list[Tool]:
    if version == 1:
        return [proportions(), means(), values(), chi_square(), rates(), adjust_p(), stratified(), arithmetic()]
    return [proportions(with_missing=True), means(), values(), chi_square(), rates(), adjust_p(), stratified(),
            subgroups(), arithmetic()]
