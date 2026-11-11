"""Fill the tables that tasks.yaml specifies by note rather than by rows.

Each generator takes the task's table list and returns one entry per table:
a list of rows to use, or None to keep the rows in the YAML. It also returns
how many draws were needed to satisfy the task's dgp constraint (the nominal
hit, the clustering reversal), which the sample records as metadata.

Static tables in the YAML are not redrawn across replicates yet; replicates
currently vary only the generated tables.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import stats

Rows = list[list[Any]]
GenResult = tuple[list[Rows | None], int]
Generator = Callable[[np.random.Generator, list[dict]], GenResult]

GENERATORS: dict[str, Generator] = {}


def _register(task_id: str):
    def deco(fn: Generator) -> Generator:
        GENERATORS[task_id] = fn
        return fn
    return deco


def _p_2x2(n1: int, e1: int, n2: int, e2: int) -> float:
    t = np.array([[e1, n1 - e1], [e2, n2 - e2]])
    if t.min() < 0 or (t.sum(axis=0) == 0).any() or (t.sum(axis=1) == 0).any():
        return 1.0
    return float(stats.chi2_contingency(t, correction=False)[1])


def _p_welch(m1, s1, n1, m2, s2, n2) -> float:
    se = np.sqrt(s1**2 / n1 + s2**2 / n2)
    return float(2 * stats.norm.sf(abs(m1 - m2) / se))


def _none_survives(ps) -> bool:
    """The class 1 keys say no row survives a multiplicity correction; hold every draw to Bonferroni.
    Before 25 October 2026 nothing checked it, and two sealed-set draws each had a row that
    survives, so their keys were wrong on those draws."""
    return min(ps) >= 0.05 / len(ps)


# --- class 1 -----------------------------------------------------------------


@_register("h01")
def h01_icb(rng, tables) -> GenResult:
    names = [f"ICB {chr(65 + i)}" for i in range(20)]
    for attempt in range(1, 501):
        ns = np.concatenate([[1200], rng.integers(60, 401, size=19)])
        smr = rng.binomial(ns, 0.28)
        ctl = rng.binomial(ns, 0.28)
        hit = any(_p_2x2(n, a, n, b) < 0.05 and a < b for n, a, b in zip(ns, smr, ctl))
        if hit:
            break
    rows = [[nm, int(n), int(a), int(n), int(b)] for nm, n, a, b in zip(names, ns, smr, ctl)]
    rows.append(["Total", int(ns.sum()), int(smr.sum()), int(ns.sum()), int(ctl.sum())])
    return [rows], attempt


@_register("h02")
def h02_breakfast(rng, tables) -> GenResult:
    groups = ["Reception"] + [f"Year {i}" for i in range(1, 7)]
    rows: Rows = []
    tot = {"n1": 0, "s1": 0.0, "n0": 0, "s0": 0.0}
    for gi, g in enumerate(groups):
        base = 95.6 - 0.3 * gi
        for fsm in ("Yes", "No"):
            mu = base - (1.5 if fsm == "Yes" else 0.0)
            n1, n0 = (int(x) for x in rng.integers(150, 401, size=2))
            m1 = mu + 0.8 + rng.normal(0, 8 / np.sqrt(n1))
            m0 = mu + rng.normal(0, 8 / np.sqrt(n0))
            rows.append([g, fsm, n1, round(m1, 1), round(rng.normal(8.0, 0.4), 1),
                         n0, round(m0, 1), round(rng.normal(8.0, 0.4), 1)])
            tot["n1"] += n1; tot["s1"] += m1 * n1; tot["n0"] += n0; tot["s0"] += m0 * n0
    rows.append(["All", "All", tot["n1"], round(tot["s1"] / tot["n1"], 1), 8.0,
                 tot["n0"], round(tot["s0"] / tot["n0"], 1), 8.0])
    return [rows], 1


@_register("h03")
def h03_wellbeing(rng, tables) -> GenResult:
    # (outcome, mean, sd, favourable direction)
    spec = [
        ("Sickness absence (days)", 6.2, 7.0, "lower"),
        ("Resignations per 100 staff", 11.0, 31.0, "lower"),
        ("Stress score (0 to 40)", 14.5, 7.5, "lower"),
        ("GP visits", 2.8, 2.6, "lower"),
        ("Overtime hours", 48.0, 40.0, "lower"),
        ("Job satisfaction (0 to 10)", 6.4, 1.9, "higher"),
        ("Presenteeism days", 4.1, 4.8, "lower"),
        ("Reported accidents per 100 staff", 3.2, 17.0, "lower"),
    ]
    for attempt in range(1, 2001):
        rows: Rows = []
        sig = []
        for name, mu, sd, fav in spec:
            n1, n0 = (int(x) for x in rng.integers(870, 931, size=2))
            m1 = mu + rng.normal(0, sd / np.sqrt(n1))
            m0 = mu + rng.normal(0, sd / np.sqrt(n0))
            s1, s0 = sd * rng.uniform(0.95, 1.05), sd * rng.uniform(0.95, 1.05)
            p = _p_welch(m1, s1, n1, m0, s0, n0)
            favourable = (m1 < m0) if fav == "lower" else (m1 > m0)
            sig.append((p < 0.05, favourable, p))
            rows.append([name, n1, round(m1, 2), round(s1, 1), n0, round(m0, 2), round(s0, 1)])
        n_sig = sum(s for s, _, _ in sig)
        if n_sig == 1 and all(f for s, f, _ in sig if s) and _none_survives([q for _, _, q in sig]):
            break
    return [rows], attempt


@_register("h04")
def h04_lez(rng, tables) -> GenResult:
    for attempt in range(1, 501):
        in_zone = np.zeros(24, dtype=bool)
        in_zone[rng.choice(24, 11, replace=False)] = True
        pop = rng.integers(1800, 6501, size=24)
        mu = pop * 0.015
        before = rng.poisson(mu)
        after = rng.poisson(0.9 * mu)
        p_fall = stats.binom.cdf(after, before + after, 0.5)
        p_two = np.minimum(1.0, 2 * np.minimum(p_fall, stats.binom.sf(after - 1, before + after, 0.5)))
        ok_fall = any(p_fall[in_zone] < 0.05)
        ok_rise = any(after[in_zone] > before[in_zone])
        if ok_fall and ok_rise and _none_survives(list(p_two)):
            break
    rows = [[f"Ward {i + 1}", bool(z), int(p), int(b), int(a)]
            for i, (z, p, b, a) in enumerate(zip(in_zone, pop, before, after))]
    for label, mask in (("All wards in zone", in_zone), ("All wards outside zone", ~in_zone)):
        rows.append([label, bool(mask[0]) if False else ("yes" if label.endswith("in zone") else "no"),
                     int(pop[mask].sum()), int(before[mask].sum()), int(after[mask].sum())])
    return [rows], attempt


# --- class 2 -----------------------------------------------------------------


@_register("h06")
def h06_apprentice(rng, tables) -> GenResult:
    bands = ["16 to 18", "19 to 24", "25 to 34", "35 and over"]
    starters = [4544, 4260, 3124, 2272]           # sums to 14,200
    rates = [0.93, 0.85, 0.74]
    linked = [int(round(s * r)) for s, r in zip(starters, rates)]
    linked.append(11640 - sum(linked))             # table sums to the stage count
    rows = [[b, s, l] for b, s, l in zip(bands, starters, linked)]
    return [None, rows], 1


@_register("h08")
def h08_retrofit(rng, tables) -> GenResult:
    rows: Rows = []
    for group, consent in (("Retrofit", (0.25, 0.45, 0.65)), ("Comparison", (0.50, 0.70, 0.90))):
        for terc, c in zip(("Lowest", "Middle", "Highest"), consent):
            rows.append([group, terc, 400, int(round(400 * c))])
    return [rows, None, None], 1


# --- class 4 -----------------------------------------------------------------


@_register("h16")
def h16_remote(rng, tables) -> GenResult:
    fam = [
        ("Operational", 7.6, (40, 300, 1100)),
        ("Facilities", 7.1, (20, 150, 500)),
        ("Administrative", 5.3, (240, 700, 350)),
        ("Professional", 4.2, (520, 850, 200)),
        ("Management", 3.6, (280, 300, 50)),
    ]
    rows2: Rows = []
    acc = {k: [0, 0.0, 0.0] for k in ("Fully remote", "Hybrid", "On site")}
    for name, base, ns in fam:
        means = [base - 0.4, base - 0.2, base]
        row = [name]
        for (k, n, m) in zip(acc, ns, means):
            m = round(m + rng.normal(0, 0.1), 1)
            row += [n, m]
            acc[k][0] += n; acc[k][1] += n * m; acc[k][2] += n * (0.95 * m) ** 2
        rows2.append(row)
    rows1 = [[k, v[0], round(v[1] / v[0], 1), round(np.sqrt(v[2] / v[0]), 1)] for k, v in acc.items()]
    return [rows1, rows2], 1


# --- class 5 -----------------------------------------------------------------


@_register("h17")
def h17_flu(rng, tables) -> GenResult:
    logit = lambda p: np.log(p / (1 - p))
    expit = lambda x: 1 / (1 + np.exp(-x))
    arm = np.array([1, 0] * 8)
    for attempt in range(1, 2001):
        elig = rng.integers(500, 1101, size=16)
        p = expit(logit(0.56) + 0.21 * arm + rng.normal(0, 0.42, size=16))
        vac = rng.binomial(elig, p)
        prop = vac / elig
        p_pat = _p_2x2(int(elig[arm == 1].sum()), int(vac[arm == 1].sum()),
                       int(elig[arm == 0].sum()), int(vac[arm == 0].sum()))
        p_prac = float(stats.ttest_ind(prop[arm == 1], prop[arm == 0], equal_var=False).pvalue)
        if p_pat < 0.01 and p_prac > 0.05 and prop[arm == 1].mean() > prop[arm == 0].mean():
            break
    rows = [[f"Practice {i + 1:02d}", "Reminder" if a else "Usual care", int(e), int(v)]
            for i, (a, e, v) in enumerate(zip(arm, elig, vac))]
    for label, a in (("All reminder practices", 1), ("All usual-care practices", 0)):
        rows.append([label, "Reminder" if a else "Usual care", int(elig[arm == a].sum()), int(vac[arm == a].sum())])
    return [rows], attempt


@_register("h18")
def h18_maths(rng, tables) -> GenResult:
    for attempt in range(1, 2001):
        rows: Rows = []
        pupils_new, pupils_old, class_means = [], [], ([], [])
        for s in range(6):
            school = rng.normal(0, 3)
            for c, arm in enumerate([1, 1, 0, 0]):
                n = int(rng.integers(26, 34))
                mu = 65 + 3 * arm + school + rng.normal(0, 4.5)
                scores = rng.normal(mu, 13, size=n)
                (pupils_new if arm else pupils_old).extend(scores.tolist())
                class_means[arm].append(scores.mean())
                rows.append([f"School {s + 1}", f"{s + 1}{'ABCD'[c]}", "New" if arm else "Existing",
                             n, round(float(scores.mean()), 1), round(float(scores.std(ddof=1)), 1)])
        p_pupil = float(stats.ttest_ind(pupils_new, pupils_old, equal_var=False).pvalue)
        p_class = float(stats.ttest_ind(class_means[1], class_means[0], equal_var=False).pvalue)
        if p_pupil < 0.01 and p_class > 0.05 and np.mean(pupils_new) > np.mean(pupils_old):
            break
    return [rows], attempt


# --- controls ----------------------------------------------------------------


@_register("h34")
def h34_waits(rng, tables) -> GenResult:
    specs = ["Cardiology", "Dermatology", "ENT", "Gastroenterology", "General surgery", "Gynaecology",
             "Neurology", "Ophthalmology", "Orthopaedics", "Rheumatology", "Urology", "Pain management"]
    rows: Rows = []
    for sp in specs:
        n = int(rng.integers(300, 2501))
        med = float(rng.uniform(5, 23))
        lq, uq = med * rng.uniform(0.5, 0.6), med * rng.uniform(1.5, 1.8)
        frac = float(np.clip(0.03 + 0.03 * (med - 5) + rng.normal(0, 0.02), 0.01, 0.6))
        rows.append([sp, n, round(med, 1), round(lq, 1), round(uq, 1), int(round(n * frac))])
    return [rows], 1


@_register("h35")
def h35_hip(rng, tables) -> GenResult:
    rows: Rows = []
    for i, year in enumerate(range(2016, 2026)):
        pop = int(round((142000 + (171000 - 142000) * i / 9) / 100.0) * 100)
        rate = 58 - 0.78 * i + rng.normal(0, 0.8)
        count = int(round(rate * pop / 10000))
        rows.append([str(year), pop, count, round(count / pop * 10000, 1)])  # a year, not a count: no thousands separator
    return [rows], 1


# Generators for the held-out and sealed task sets are not part of the public release.
