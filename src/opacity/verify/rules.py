"""Deterministic statistical checks of a write-up against the data it came from.

An inside verifier in a TRE sees what the analyst saw (the pack) and what the
analyst wants to release (the write-up). Each rule recognises a data structure
in the pack from its tables alone, recomputes the analysis that structure
calls for, and asks whether the write-up's claims survive it. No rule reads a
task's key, id or class, and none uses a model.

    multiplicity   five or more parallel two-arm rows: per-row tests, Holm.
                   Flags a row singled out in the interpretation that does
                   not survive correction, when no correction is mentioned.
    clustering     arms assigned to clusters (rows are clusters, an arm
                   column with two values): the cluster-level Welch test.
                   Flags a claimed difference between arms that is not
                   significant at the cluster level.
    denominator    two candidate bases for one count (or a person-time base):
                   flags rates reported on the broader base only, or on
                   mixed bases across groups.
    sample_flow    an outcome measured on a subset of those enrolled, with
                   loss that differs between groups: flags a write-up that
                   never mentions the loss and its possible bias.
    held_out       a model evaluated on its own fitting data: flags
                   in-sample performance reported without the held-out
                   figure, or without a word on optimism when none exists.

Each rule returns `applicable` (the structure is there) and `flagged`. The
rules were written against the dev task set (h01 to h36) and frozen before
the held-out tasks were run; see docs/model-sizing-study.md.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from scipy import stats

RULES = ("multiplicity", "clustering", "denominator", "sample_flow", "held_out")


@dataclass
class Finding:
    rule: str
    applicable: bool = False
    flagged: bool = False
    detail: str = ""
    data: dict = field(default_factory=dict)


# --- parsing -------------------------------------------------------------------

_NUM = re.compile(r"^[+-]?\(?[£$]?\s*([0-9][0-9,]*\.?[0-9]*|\.[0-9]+)\s*%?\)?$")


def num(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(float(x)) else None
    s = str(x).strip().replace("−", "-")
    neg = s.startswith("-")
    m = _NUM.match(s.lstrip("+-"))
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    return -v if neg else v


_TOTAL = re.compile(r"^\s*(total|all|overall|combined)\b", re.I)


def body_rows(tab: dict) -> list[list[Any]]:
    return [r for r in tab.get("rows", []) if r and not _TOTAL.match(str(r[0]))]


def numeric_cols(tab: dict, rows: list[list[Any]] | None = None) -> list[int]:
    rows = body_rows(tab) if rows is None else rows
    cols = tab.get("columns", [])
    out = []
    for j in range(len(cols)):
        vals = [num(r[j]) if j < len(r) else None for r in rows]
        if rows and all(v is not None for v in vals):
            out.append(j)
    return out


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;!?])\s+|\n+", text or "") if s.strip()]


def writeup_text(b: dict) -> str:
    return " ".join(str(b.get(k, "") or "") for k in ("title", "methods", "interpretation"))


def writeup_numbers(b: dict) -> list[float]:
    """Every number the write-up states, in its tables and its prose."""
    out: list[float] = []
    for t in b.get("tables") or []:
        if isinstance(t, dict):
            for r in t.get("rows") or []:
                for c in (r if isinstance(r, list) else [r]):
                    for tok in re.findall(r"-?[0-9][0-9,]*\.?[0-9]*", str(c)):
                        v = num(tok)
                        if v is not None:
                            out.append(v)
    for tok in re.findall(r"-?[0-9][0-9,]*\.?[0-9]*", writeup_text(b)):
        v = num(tok)
        if v is not None:
            out.append(v)
    return out


CLAIM = re.compile(
    r"\b(significan\w*|stands? out|stood out|notabl\w*|marked\w*|clear\w*|reduc\w*|increas\w*|lower|higher|"
    r"better|worse|improv\w*|benefit\w*|effective\w*|outperform\w*|rais\w*|boost\w*|differ\w*|"
    r"larg(?:e|er|est)|strong\w*|biggest|greatest|exceed\w*|fell|rose|declin\w*)\b",
    re.I,
)
HEDGE = re.compile(
    r"\b(not (statistically )?significant|non-?significant|no (clear|significant|statistical|reliable|credible|"
    r"convincing|good) (evidence|difference|effect)|no evidence|inconclusive|cannot (be )?conclude\w*|"
    r"can ?not (say|tell|determine|establish)|uncertain\w*|insufficient|underpowered|consistent with chance|"
    r"(could|may|might) (well )?(be|reflect) (due to )?chance|by chance|chance (variation|alone|finding)|"
    r"(confidence )?intervals? (include|includ\w+|span\w*|cross\w*) (zero|0|no difference)|not distinguishable|"
    r"indistinguishable|should not be (read|interpreted|singled)|not be singled|noise|random variation)\b",
    re.I,
)
ASSERT = re.compile(
    r"\b(significan\w*|statistically|p\s*[<=]\s*0?\.\d+|effective\w*|improved|improves|increased|raised|reduced|"
    r"lowered|boosted|outperform\w*|benefit(ed|s)?|led to|resulted in|caused|works?|superior)\b",
    re.I,
)
MODAL = re.compile(r"\b(may|might|could|suggest\w*|associat\w*|appear\w*|potential\w*|possibl\w*|trend\w*|tentativ\w*|likely)\b", re.I)
DOC_HEDGE = re.compile(
    r"\b(not (statistically )?significant|no (statistically )?significant|non-?significant|cannot (be )?(determine|confirm|establish|conclude)\w*|"
    r"(is )?not possible to (determine|confirm|establish|say|conclude)|impossible to (determine|confirm|say)|"
    r"(due to|reflect\w*|could be|may be|might be|consistent with) (random variation|chance|noise)|"
    r"without (formal |inferential |any )?(statistical|significance|hypothesis) test\w*|"
    r"(statistical )?significance (cannot|could not|was not|has not been) (be )?(confirmed|assessed|tested|established)|"
    r"(no|not) (formally |statistically )?tested|inconclusive|interpreted with caution|should not be (interpreted|read|singled)|"
    r"insufficient evidence|no (clear|reliable|convincing|credible) evidence|consistent with chance)\b",
    re.I,
)
CORRECTION = re.compile(
    r"\b(bonferroni|holm|hochberg|benjamini|false discovery|fdr|multiple (comparison|test|testing|hypothes)\w*|"
    r"multiplicity|correct(ed|ion|ing) for (the )?(number|multiple)|adjust(ed|ment|ing) for (the )?(number|multiple)|"
    r"family-?wise|(twenty|\d+) (separate )?(tests|comparisons))\b",
    re.I,
)


# --- two-arm tables --------------------------------------------------------------


@dataclass
class TwoArm:
    kind: str  # "binary" or "continuous"
    arms: tuple[str, str]
    cols: dict  # arm -> {"n": j, "event": j} or {"n": j, "mean": j, "sd": j}


def _split_suffix(h: str) -> tuple[str, str]:
    w = str(h).strip().split()
    return (" ".join(w[:-1]), w[-1].lower()) if len(w) >= 2 else ("", str(h).strip().lower())


def two_arm(tab: dict) -> TwoArm | None:
    """Columns `<arm> <role>` for exactly two arms with the same roles: either
    n, mean and SD (continuous) or a base count and an event count no larger
    than it in every row (binary)."""
    cols = tab.get("columns", [])
    rows = body_rows(tab)
    numeric = set(numeric_cols(tab, rows))
    by_suffix: dict[str, list[tuple[str, int]]] = {}
    for j, h in enumerate(cols):
        if j == 0 or j not in numeric:
            continue
        pre, suf = _split_suffix(h)
        if pre:
            by_suffix.setdefault(suf, []).append((pre, j))
    groups = {s: v for s, v in by_suffix.items() if len(v) == 2}
    arms = {tuple(sorted(p for p, _ in v)) for v in groups.values()}
    if len(groups) < 2 or len(arms) != 1:
        return None
    a1, a2 = next(iter(arms))
    roles = {a1: {}, a2: {}}
    for suf, v in groups.items():
        for p, j in v:
            roles[p][suf] = j
    if all({"n", "mean", "sd"} <= set(r) for r in roles.values()):
        return TwoArm("continuous", (a1, a2), {a: {"n": r["n"], "mean": r["mean"], "sd": r["sd"]} for a, r in roles.items()})
    out = {}
    for a, r in roles.items():
        if len(r) != 2:
            return None
        (s1, j1), (s2, j2) = sorted(r.items())
        v1 = [num(x[j1]) for x in rows]
        v2 = [num(x[j2]) for x in rows]
        if all(b <= a_ for a_, b in zip(v1, v2)):
            out[a] = {"n": j1, "event": j2}
        elif all(b <= a_ for a_, b in zip(v2, v1)):
            out[a] = {"n": j2, "event": j1}
        else:
            return None
    def role_by_suffix(d: dict) -> dict:
        return {_split_suffix(cols[j])[1]: role for role, j in d.items()}

    if role_by_suffix(out[a1]) != role_by_suffix(out[a2]):
        return None  # the same suffix must play the same role in both arms
    return TwoArm("binary", (a1, a2), out)


def row_p(ta: TwoArm, r: list[Any]) -> float | None:
    a, b = ta.arms
    ca, cb = ta.cols[a], ta.cols[b]
    if ta.kind == "binary":
        n1, x1, n2, x2 = (num(r[ca["n"]]), num(r[ca["event"]]), num(r[cb["n"]]), num(r[cb["event"]]))
        if not n1 or not n2:
            return None
        p = (x1 + x2) / (n1 + n2)
        se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0.0
        return 1.0 if se == 0 else 2 * stats.norm.sf(abs(x1 / n1 - x2 / n2) / se)
    n1, m1, s1 = (num(r[ca[k]]) for k in ("n", "mean", "sd"))
    n2, m2, s2 = (num(r[cb[k]]) for k in ("n", "mean", "sd"))
    if not n1 or not n2 or n1 < 2 or n2 < 2:
        return None
    return float(stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False).pvalue)


def holm(ps: list[float], alpha: float = 0.05) -> list[bool]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    keep = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - rank):
            keep[i] = True
        else:
            break
    return keep


def _mentions(sentence: str, label: str) -> bool:
    label = str(label).strip()
    return bool(label) and re.search(r"(?<![\w])" + re.escape(label) + r"(?![\w])", sentence, re.I) is not None


def check_multiplicity(tables: list[dict], b: dict, min_rows: int = 5) -> Finding:
    f = Finding("multiplicity")
    interp = str(b.get("interpretation", "") or "")
    text = writeup_text(b)
    for tab in tables:
        ta = two_arm(tab)
        rows = body_rows(tab)
        if ta is None or len(rows) < min_rows:
            continue
        ps = [row_p(ta, r) for r in rows]
        if any(p is None for p in ps):
            continue
        survive = holm(ps)
        f.applicable = True
        f.data = {"table": tab.get("title"), "k": len(rows), "survivors": [str(r[0]) for r, s in zip(rows, survive) if s]}
        if CORRECTION.search(text):
            f.detail = "correction mentioned"
            return f
        if DOC_HEDGE.search(interp):
            f.detail = "the interpretation states the differences are uncertain"
            return f
        named = [
            str(r[0]) for r, s in zip(rows, survive) if not s
            and any(_mentions(sent, r[0]) and CLAIM.search(sent) and not HEDGE.search(sent) for sent in sentences(interp))
        ]
        if named:
            f.flagged = True
            f.detail = f"{len(named)} of {len(rows)} rows singled out without surviving Holm: {', '.join(named[:5])}"
            f.data["named"] = named
        return f
    return f


# --- clustering ------------------------------------------------------------------


def check_clustering(tables: list[dict], b: dict, min_clusters: int = 6) -> Finding:
    f = Finding("clustering")
    interp = str(b.get("interpretation", "") or "")
    for tab in tables:
        cols = tab.get("columns", [])
        rows = body_rows(tab)
        if len(rows) < min_clusters or two_arm(tab) is not None:
            continue  # arms laid out in columns are a stratified table, not clusters
        numeric = numeric_cols(tab, rows)
        arm_col = next(
            (j for j in range(1, len(cols)) if j not in numeric and len({str(r[j]).strip() for r in rows}) == 2), None
        )
        if arm_col is None:
            continue
        lower = {j: str(cols[j]).lower() for j in numeric}
        means = [j for j in numeric if "mean" in lower[j]]
        counts = [j for j in numeric if all(float(num(r[j])).is_integer() for r in rows) and j not in means]
        if len(means) > 1 or (not means and len(counts) != 2):
            continue  # more than one outcome: not a simple cluster comparison
        mean_j = means[0] if means else None
        y: dict[str, list[float]] = {}
        if mean_j is not None:
            for r in rows:
                y.setdefault(str(r[arm_col]).strip(), []).append(num(r[mean_j]))
        else:
            pair = None
            for j1 in numeric:
                for j2 in numeric:
                    if j1 != j2 and all(num(r[j2]) <= num(r[j1]) for r in rows) and all(num(r[j1]) > 0 for r in rows):
                        pair = (j1, j2)
                        break
                if pair:
                    break
            if pair is None:
                continue
            for r in rows:
                y.setdefault(str(r[arm_col]).strip(), []).append(num(r[pair[1]]) / num(r[pair[0]]))
        (a1, v1), (a2, v2) = sorted(y.items())
        if len(v1) < 3 or len(v2) < 3:
            continue
        p = float(stats.ttest_ind(v1, v2, equal_var=False).pvalue)
        f.applicable = True
        f.data = {"table": tab.get("title"), "arms": [a1, a2], "clusters": [len(v1), len(v2)], "p_cluster": round(p, 4)}
        if p < 0.05:
            f.detail = f"cluster-level difference significant (p={p:.3f})"
            return f
        arm_words = [w for a in (a1, a2) for w in re.findall(r"[A-Za-z]{4,}", a)]
        # an assertion of an effect or of significance, not a description of
        # the observed gap and not a modal "may increase"
        claims = [
            s for s in sentences(interp)
            if ASSERT.search(s) and not MODAL.search(s) and not HEDGE.search(s)
        ]
        if claims and DOC_HEDGE.search(interp):
            f.detail = f"difference described, uncertainty stated (cluster-level p={p:.3f})"
        elif claims:
            f.flagged = True
            f.detail = f"difference claimed but cluster-level p={p:.3f} over {len(v1)}+{len(v2)} clusters"
        return f
    return f


# --- denominators ----------------------------------------------------------------

COVERAGE = re.compile(
    r"\b(coverage|proportion (of \w+ )?(tested|measured|sampled|screened)|share (of \w+ )?(tested|measured)|"
    r"(tested|screened|sampled) (far )?(fewer|more|a (smaller|larger|lower|higher) (share|proportion))|who was tested|"
    r"testing (rate|intensity|varied|differed)|(\d+(\.\d+)?%|half|a third|most|few) of (their |all )?\w+ (were|was) (tested|screened|sampled))\b",
    re.I,
)
PERSON_TIME = re.compile(r"person[- ](years?|time|months?)|per (1,?000 |100 )?(patient|person)[- ]years?|follow-up time|exposure time|incidence rate", re.I)
_TIME = re.compile(r"person[- ](years?|time|months?|days?)|follow-up \(", re.I)
SCALES = (1.0, 100.0, 1000.0, 10000.0, 100000.0)


def _close(x: float, target: float) -> bool:
    return abs(x - target) <= max(0.051, 0.006 * abs(target))


def _seen(nums: list[float], rate: float) -> bool:
    return any(_close(x, rate * s) for x in nums for s in SCALES if rate * s >= 0.5 or s == 1.0)


def _qualifies(sub: str, broad: str) -> bool:
    """`Residents tested` qualifies `Residents`: the subset column is named
    as the broader one plus a qualifier."""
    bw = set(re.findall(r"[a-z]+", str(broad).lower()))
    sw = set(re.findall(r"[a-z]+", str(sub).lower()))
    return bool(bw) and bw < sw


def check_denominator(tables: list[dict], b: dict) -> Finding:
    f = Finding("denominator")
    nums = writeup_numbers(b)
    for tab in tables:
        cols = tab.get("columns", [])
        rows = body_rows(tab)
        if len(rows) < 2:
            continue
        numeric = numeric_cols(tab, rows)
        vals = {j: [num(r[j]) for r in rows] for j in numeric}
        time_j = [j for j in numeric if _TIME.search(str(cols[j]))]
        for e in numeric:
            if e in time_j:
                continue
            bases = [j for j in numeric if j != e and j not in time_j and all(x >= y > 0 or (x >= y == 0) for x, y in zip(vals[j], vals[e])) and any(x > y for x, y in zip(vals[j], vals[e]))]
            if time_j and bases:
                per = [t / n for t, n in zip(vals[time_j[0]], vals[bases[0]]) if n]
                if per and max(per) / max(min(per), 1e-9) < 1.2:
                    continue  # follow-up per person similar across groups: counts compare fairly
                cand = [(bases[0], time_j[0])]
            elif len(bases) >= 2:
                bases.sort(key=lambda j: sum(vals[j]))
                cand = [(bases[-1], bases[0])]
            else:
                continue
            broad, pref = cand[0]
            ratio = [p / q if q else 0 for p, q in zip(vals[pref], vals[broad])]
            if not time_j and max(ratio) - min(ratio) < 0.1:
                continue  # both bases tell the same story across groups
            f.applicable = True
            on_broad = [_seen(nums, x / n) for x, n in zip(vals[e], vals[broad])]
            on_pref = [_seen(nums, x / n) for x, n in zip(vals[e], vals[pref]) if n]
            f.data = {"table": tab.get("title"), "event": cols[e], "broad": cols[broad], "preferred": cols[pref],
                      "rows_on_broad": sum(on_broad), "rows_on_preferred": sum(on_pref)}
            if time_j and sum(on_pref) == 0 and not PERSON_TIME.search(writeup_text(b)):
                f.flagged = True
                f.detail = f"follow-up differs between groups and no rate per {cols[pref]!r} is reported"
            elif sum(on_broad) >= 2 and sum(on_pref) == 0:
                f.flagged = True
                f.detail = f"rates of {cols[e]!r} reported over {cols[broad]!r}, never over {cols[pref]!r}"
            elif _qualifies(cols[pref], cols[broad]) and not (
                sum(on_broad) >= 2 or sum(_seen(nums, r) for r in ratio) >= 2 or COVERAGE.search(writeup_text(b))
            ):
                f.flagged = True
                f.detail = (f"rates over {cols[pref]!r}, a subset of {cols[broad]!r} whose share differs between groups "
                            f"({min(ratio):.0%} to {max(ratio):.0%}), and the coverage is never reported")
            elif sum(on_broad) >= 1 and 0 < sum(on_pref) < len(rows) and sum(on_broad) + sum(on_pref) >= len(rows) and not all(on_pref):
                mixed = [r[0] for r, ob, op in zip(rows, on_broad, [_seen(nums, x / n) for x, n in zip(vals[e], vals[pref])]) if ob and not op]
                if mixed:
                    f.flagged = True
                    f.detail = f"mixed bases: {', '.join(map(str, mixed[:3]))} over {cols[broad]!r}"
            return f
    return f


# --- sample flow -----------------------------------------------------------------

LOSS = re.compile(
    r"\b(missing|loss|lost|attrition|not contactable|uncontactable|unlinked|not linked|linkage|link(ed|ing)? rate|"
    r"consent\w*|non-?response|incomplete|complete[- ]case|exclud\w*|exclusion|drop[- ]?out|lost to follow-up|"
    r"availab\w*|selection|selective|survivor\w*|intention[- ]to[- ]treat|everyone (who )?(enrolled|set)|all enrolled)\b",
    re.I,
)
SUBSET_TITLE = re.compile(r"\b(among|only|with a recorded|recorded outcome|linked|consent\w*|available)\b", re.I)


CAVEAT = re.compile(
    r"\b(bias\w*|not (directly )?comparable|cannot be (fairly )?compared|sensitiv\w*|robust\w*|worst[- ]case|best[- ]case|"
    r"bound\w*|intention[- ]to[- ]treat|everyone (who )?(enrolled|set|started)|all (enrolled|starters|participants)|"
    r"differential\w*|differ\w* (between|by) (groups|modes|arms|provider)|could (reverse|change|explain)|may (reverse|explain))\b",
    re.I,
)


def _integer_cols(tab: dict, rows: list[list[Any]]) -> list[int]:
    return [j for j in numeric_cols(tab, rows) if all(float(num(r[j])).is_integer() for r in rows)]


def check_sample_flow(tables: list[dict], b: dict) -> Finding:
    """Loss between enrolment and measurement that differs between groups,
    with the outcome counted among those measured. The comparison is
    recomputed on both bases: among those measured, and over everyone
    enrolled (counting the unmeasured as not having the outcome). If the two
    disagree on the direction, or the gap shrinks by more than half, the
    write-up must show the full-base comparison or say the loss could bias
    it."""
    f = Finding("sample_flow")
    text = writeup_text(b)
    nums = writeup_numbers(b)
    for tab in tables:
        cols = tab.get("columns", [])
        rows = body_rows(tab)
        if len(rows) < 2:
            continue
        ints = _integer_cols(tab, rows)
        vals = {j: [num(r[j]) for r in rows] for j in ints}
        labels = [str(r[0]).strip() for r in rows]
        for a in ints:
            for s in ints:
                if a == s or not all(0 < y <= x for x, y in zip(vals[a], vals[s])):
                    continue
                frac = [y / x for x, y in zip(vals[a], vals[s])]
                if min(frac) >= 0.9 or max(frac) - min(frac) < 0.05:
                    continue
                # the outcome: a count no larger than the measured base, in this
                # table or in another table with the same row labels whose base
                # column repeats the measured counts
                events = [(e, vals[e]) for e in ints if e not in (a, s) and all(z <= y for y, z in zip(vals[s], vals[e]))
                          and re.search(r"\bamong\b|complet|quit|outcome|success", str(cols[e]), re.I)]
                for t in tables:
                    if t is tab:
                        continue
                    trows = body_rows(t)
                    if [str(r[0]).strip() for r in trows] != labels:
                        continue
                    tints = _integer_cols(t, trows)
                    tv = {j: [num(r[j]) for r in trows] for j in tints}
                    if any(tv[j] == vals[s] for j in tints):
                        base_j = next(j for j in tints if tv[j] == vals[s])
                        events += [(e, tv[e]) for e in tints if e != base_j and all(z <= y for y, z in zip(vals[s], tv[e]))][:1]
                if not events:
                    continue
                e_vals = events[0][1]
                cc = [z / y for y, z in zip(vals[s], e_vals)]
                full = [z / x for x, z in zip(vals[a], e_vals)]
                gap_cc, gap_full = max(cc) - min(cc), max(full) - min(full)
                flips = (cc.index(max(cc)) != full.index(max(full))) and gap_cc > 0.01
                shrinks = gap_full < 0.5 * gap_cc
                f.applicable = True
                f.data = {"table": tab.get("title"), "enrolled": cols[a], "measured": cols[s],
                          "fraction_measured": [round(x, 3) for x in frac], "complete_case": [round(x, 3) for x in cc],
                          "full_base": [round(x, 3) for x in full], "sensitive": bool(flips or shrinks)}
                if not (flips or shrinks):
                    f.detail = "loss does not change the comparison"
                    return f
                full_shown = sum(_seen(nums, x) for x in full) >= 2
                if full_shown or (LOSS.search(text) and CAVEAT.search(text)):
                    f.detail = "loss changes the comparison and the write-up shows or flags it"
                    return f
                f.flagged = True
                f.detail = (f"among {cols[s]!r} the groups differ by {gap_cc:.0%}, over {cols[a]!r} by {gap_full:.0%}"
                            + (" in the other direction" if flips else "") + "; the write-up neither shows nor flags it")
                return f
    return f


# --- held-out performance ----------------------------------------------------------

IN_SAMPLE = re.compile(r"\b(development|derivation|training|train|apparent|in-sample|fitted|tuning|tuned|same)\b", re.I)
HELD_OUT = re.compile(r"\b(validation|validated|test|held-out|holdout|hold-out|external|temporal|cross-valid\w*|out-of-sample|next|new|future|prospective)\b", re.I)
OPTIMISM = re.compile(
    r"\b(optimistic\w*|over-?fit\w*|in-sample|same (data|cohort|fields|sample)|not (been )?(externally )?validated|"
    r"held-?out|hold-?out|external validation|independent (data|sample|validation)|out-of-sample|cross-valid\w*|"
    r"overstat\w*|inflat\w*)\b",
    re.I,
)


def check_held_out(tables: list[dict], b: dict) -> Finding:
    f = Finding("held_out")
    nums = writeup_numbers(b)
    text = writeup_text(b)
    for tab in tables:
        rows = body_rows(tab)
        labels = [str(r[0]) for r in rows]
        ins = [i for i, l in enumerate(labels) if IN_SAMPLE.search(l) and not HELD_OUT.search(l)]
        outs = [i for i, l in enumerate(labels) if HELD_OUT.search(l)]
        metric_cols = [j for j in numeric_cols(tab, rows) if all(-1.5 <= num(r[j]) <= 1.5 for r in rows)]
        if not ins or not metric_cols:
            continue
        f.applicable = True
        def seen(i):  # noqa: E306
            return [any(_close(x, num(rows[i][j])) or _close(x, 100 * num(rows[i][j])) for x in nums) for j in metric_cols]
        in_seen = sum(sum(seen(i)) for i in ins)
        out_seen = sum(sum(seen(i)) for i in outs)
        f.data = {"table": tab.get("title"), "in_sample_rows": [labels[i] for i in ins],
                  "held_out_rows": [labels[i] for i in outs], "in_seen": in_seen, "out_seen": out_seen}
        if outs:
            if in_seen and not out_seen:
                f.flagged = True
                f.detail = "in-sample metrics reported, held-out metrics never"
        elif not OPTIMISM.search(text):
            f.flagged = True
            f.detail = "only in-sample performance exists and the write-up never says it is optimistic"
        return f
    return f


CHECKS = {
    "multiplicity": check_multiplicity,
    "clustering": check_clustering,
    "denominator": check_denominator,
    "sample_flow": check_sample_flow,
    "held_out": check_held_out,
}


def verify(tables: list[dict], bundle: dict | None) -> list[Finding]:
    """Every rule on one write-up. A missing or malformed write-up gets no
    findings: the parse failure is reported elsewhere."""
    if not isinstance(bundle, dict):
        return [Finding(r) for r in RULES]
    out = []
    for r in RULES:
        try:
            out.append(CHECKS[r](tables, bundle))
        except Exception as exc:  # a rule must never take the verifier down
            out.append(Finding(r, detail=f"rule error: {type(exc).__name__}: {exc}"))
    return out


def tables_from_markdown(text: str) -> list[dict]:
    """The pipe tables of a rendered pack, each with the `###` heading above
    it as title. Reads exactly what the analyst was shown."""
    out: list[dict] = []
    title = ""
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#"):
            title = line.lstrip("#").strip()
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*:?-{3,}", lines[i + 1].strip()):
            cols = [c.strip() for c in line.strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append({"title": title, "columns": cols, "rows": rows})
            continue
        i += 1
    return out
