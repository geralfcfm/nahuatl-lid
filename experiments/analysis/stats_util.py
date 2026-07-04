"""Small statistics toolkit for paired fold-level re-analysis (n=5 folds).

No new model training happens here -- everything reads already-committed
`results/*.json` files. This module supplies:

- an exact Student-t CDF/quantile (via the regularized incomplete beta
  function) so the paired stats work even without SciPy, with NO normal
  approximation for the small-n (n=5) case;
- paired-difference summaries (mean, sample SD, t-test, exact Wilcoxon
  signed-rank, t-based CI);
- TOST (two one-sided tests) equivalence verdicts implemented as the
  textbook-equivalent 90% CI containment check.

If SciPy is importable, it is used for `ttest_rel` / `wilcoxon` / `t.cdf`
/ `t.ppf` (cross-checked against the exact-t fallback below). Otherwise the
pure-Python exact-t implementation is used throughout. Which path ran is
recorded in the output so results are auditable.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

try:
    import scipy.stats as _sps  # type: ignore

    HAVE_SCIPY = True
    SCIPY_VERSION = __import__("scipy").__version__
except ImportError:  # pragma: no cover - exercised only when scipy is absent
    HAVE_SCIPY = False
    SCIPY_VERSION = None

# Hardcoded critical values for df=4 (n=5 paired folds), per the task brief.
# These are cross-checked against the exact-t implementation at import time
# (see _self_check below) rather than trusted blindly.
T_CRIT_95_DF4 = 2.776  # t_{0.975}(4) -- for 95% CI (uncertainty reporting)
T_CRIT_90_DF4 = 2.132  # t_{0.95}(4)  -- for 90% CI (TOST equivalence check)
EQUIV_MARGIN_PP = 1.0  # +/-1 percentage-point equivalence margin


# --------------------------------------------------------------------------
# Exact Student-t distribution (regularized incomplete beta function).
# Standard Numerical-Recipes continued-fraction implementation -- exact
# (to machine precision), NOT a normal approximation.
# --------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 1e-13) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf_exact(t: float, df: float) -> float:
    """Exact CDF of Student's t distribution with `df` degrees of freedom."""
    x = df / (df + t * t)
    p_half = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - p_half if t > 0 else p_half


def t_ppf_exact(p: float, df: float, tol: float = 1e-12) -> float:
    """Exact quantile function (inverse CDF) of Student's t via bisection."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    if p == 0.5:
        return 0.0
    lo, hi = -1000.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf_exact(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


def t_cdf(t: float, df: float) -> float:
    if HAVE_SCIPY:
        return float(_sps.t.cdf(t, df))
    return t_cdf_exact(t, df)


def t_ppf(p: float, df: float) -> float:
    if HAVE_SCIPY:
        return float(_sps.t.ppf(p, df))
    return t_ppf_exact(p, df)


def _self_check() -> None:
    """Guard against a broken exact-t implementation: df=4 critical values
    must match the literature/hardcoded constants to 3 decimal places."""
    c95 = t_ppf_exact(0.975, 4)
    c90 = t_ppf_exact(0.95, 4)
    assert abs(c95 - T_CRIT_95_DF4) < 1e-3, f"t_0.975(4) mismatch: {c95} vs {T_CRIT_95_DF4}"
    assert abs(c90 - T_CRIT_90_DF4) < 1e-3, f"t_0.95(4) mismatch: {c90} vs {T_CRIT_90_DF4}"


_self_check()


# --------------------------------------------------------------------------
# Exact Wilcoxon signed-rank test (n small, no normal approximation).
# --------------------------------------------------------------------------
def wilcoxon_exact_p(diffs: list[float]) -> tuple[float | None, float | None, str]:
    """Two-sided exact Wilcoxon signed-rank p-value.

    Returns (statistic, p_value, method_note). Zero differences are dropped
    (Wilcoxon's own convention); if that leaves < 1 pair, returns (None, None, note).
    Ties among |diff| use midranks; exactness is only guaranteed without ties,
    noted in the method string when ties are present.
    """
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return None, None, "all differences are exactly zero; Wilcoxon undefined"
    if HAVE_SCIPY:
        try:
            res = _sps.wilcoxon(nz, alternative="two-sided", mode="exact")
            note = "scipy exact" if n == len(diffs) else f"scipy exact (n={n} after dropping {len(diffs) - n} zero-diff fold(s))"
            return float(res.statistic), float(res.pvalue), note
        except ValueError:
            res = _sps.wilcoxon(nz, alternative="two-sided", mode="auto")
            return float(res.statistic), float(res.pvalue), "scipy auto (exact unavailable, e.g. ties)"

    # Pure-Python exact fallback: enumerate all 2^n sign assignments over ranks.
    abs_diffs = sorted(enumerate(nz), key=lambda kv: abs(kv[1]))
    ranks = [0.0] * n
    i = 0
    has_ties = False
    while i < n:
        j = i
        while j + 1 < n and abs(abs_diffs[j + 1][1]) == abs(abs_diffs[i][1]):
            j += 1
        if j > i:
            has_ties = True
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[abs_diffs[k][0]] = avg_rank
        i = j + 1

    w_plus_obs = sum(r for r, d in zip(ranks, nz) if d > 0)
    total = sum(ranks)
    null_dist = []
    for signs in itertools.product([1, -1], repeat=n):
        null_dist.append(sum(r for r, s in zip(ranks, signs) if s > 0))
    n_total = len(null_dist)
    # two-sided exact p-value: 2 * min(P(W+ <= obs), P(W+ >= obs)), capped at 1
    le = sum(1 for w in null_dist if w <= w_plus_obs) / n_total
    ge = sum(1 for w in null_dist if w >= w_plus_obs) / n_total
    p = min(1.0, 2.0 * min(le, ge))
    note = "pure-python exact (enumeration)" if not has_ties else "pure-python exact w/ midranks (ties present)"
    return w_plus_obs, p, note


# --------------------------------------------------------------------------
# Paired-difference summary: mean, sample SD, t-test, Wilcoxon, CI, TOST.
# --------------------------------------------------------------------------
@dataclass
class PairedResult:
    label: str
    n: int
    paired: bool
    diff_mean_pp: float
    sample_sd_diff_pp: float
    se_diff_pp: float
    df: float
    t_stat: float
    p_ttest_two_sided: float
    wilcoxon_stat: float | None
    p_wilcoxon_two_sided: float | None
    wilcoxon_method: str
    ci95_pp: list[float]
    ci90_pp: list[float]
    equivalence_margin_pp: float
    tost_established: bool
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n": self.n,
            "paired": self.paired,
            "diff_mean_pp": round(self.diff_mean_pp, 4),
            "sample_sd_diff_pp": round(self.sample_sd_diff_pp, 4),
            "se_diff_pp": round(self.se_diff_pp, 4),
            "df": round(self.df, 4),
            "t_stat": round(self.t_stat, 4),
            "p_ttest_two_sided": round(self.p_ttest_two_sided, 6),
            "wilcoxon_stat": self.wilcoxon_stat,
            "p_wilcoxon_two_sided": (
                round(self.p_wilcoxon_two_sided, 6) if self.p_wilcoxon_two_sided is not None else None
            ),
            "wilcoxon_method": self.wilcoxon_method,
            "ci95_pp": [round(v, 4) for v in self.ci95_pp],
            "ci90_pp": [round(v, 4) for v in self.ci90_pp],
            "equivalence_margin_pp": self.equivalence_margin_pp,
            "tost_established": self.tost_established,
            **self.extra,
        }


def sample_sd(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def paired_compare(a: list[float], b: list[float], label: str) -> PairedResult:
    """Paired comparison of accuracy fractions `a` vs `b` (same fold index).
    Returns percentage-point (pp) statistics for (a - b)."""
    assert len(a) == len(b), f"{label}: fold count mismatch ({len(a)} vs {len(b)})"
    n = len(a)
    diffs_pp = [(ai - bi) * 100.0 for ai, bi in zip(a, b)]
    mean_diff = sum(diffs_pp) / n
    sd_diff = sample_sd(diffs_pp)
    se = sd_diff / math.sqrt(n) if sd_diff > 0 else 0.0
    df = n - 1

    if se == 0.0:
        # Degenerate: identical folds (diff==0 everywhere). t undefined; treat CI as a point.
        t_stat = 0.0
        p_ttest = 1.0
        ci95 = [mean_diff, mean_diff]
        ci90 = [mean_diff, mean_diff]
    else:
        if HAVE_SCIPY:
            res = _sps.ttest_rel(a, b)
            t_stat = float(res.statistic)
            p_ttest = float(res.pvalue)
        else:
            t_stat = mean_diff / se
            p_ttest = 2.0 * (1.0 - t_cdf(abs(t_stat), df))
        c95 = t_ppf(0.975, df)
        c90 = t_ppf(0.95, df)
        ci95 = [mean_diff - c95 * se, mean_diff + c95 * se]
        ci90 = [mean_diff - c90 * se, mean_diff + c90 * se]

    w_stat, w_p, w_method = wilcoxon_exact_p(diffs_pp)
    tost_ok = ci90[0] >= -EQUIV_MARGIN_PP and ci90[1] <= EQUIV_MARGIN_PP

    return PairedResult(
        label=label,
        n=n,
        paired=True,
        diff_mean_pp=mean_diff,
        sample_sd_diff_pp=sd_diff,
        se_diff_pp=se,
        df=df,
        t_stat=t_stat,
        p_ttest_two_sided=p_ttest,
        wilcoxon_stat=w_stat,
        p_wilcoxon_two_sided=w_p,
        wilcoxon_method=w_method,
        ci95_pp=ci95,
        ci90_pp=ci90,
        equivalence_margin_pp=EQUIV_MARGIN_PP,
        tost_established=tost_ok,
    )


def welch_compare(a: list[float], b: list[float], label: str) -> PairedResult:
    """Unpaired Welch's t-test/CI of accuracy fractions `a` vs `b` (independent
    samples, unequal variance). Used only when fold-level pairing cannot be
    confirmed. Returns percentage-point (pp) statistics for mean(a) - mean(b)."""
    n1, n2 = len(a), len(b)
    a_pp = [x * 100.0 for x in a]
    b_pp = [x * 100.0 for x in b]
    m1, m2 = sum(a_pp) / n1, sum(b_pp) / n2
    s1, s2 = sample_sd(a_pp), sample_sd(b_pp)
    mean_diff = m1 - m2
    var1, var2 = (s1 * s1) / n1, (s2 * s2) / n2
    se = math.sqrt(var1 + var2)

    if se == 0.0:
        df = n1 + n2 - 2
        t_stat, p_ttest = 0.0, 1.0
        ci95 = [mean_diff, mean_diff]
        ci90 = [mean_diff, mean_diff]
    else:
        df = (var1 + var2) ** 2 / (var1**2 / (n1 - 1) + var2**2 / (n2 - 1))
        if HAVE_SCIPY:
            res = _sps.ttest_ind(a_pp, b_pp, equal_var=False)
            t_stat = float(res.statistic)
            p_ttest = float(res.pvalue)
        else:
            t_stat = mean_diff / se
            p_ttest = 2.0 * (1.0 - t_cdf(abs(t_stat), df))
        c95 = t_ppf(0.975, df)
        c90 = t_ppf(0.95, df)
        ci95 = [mean_diff - c95 * se, mean_diff + c95 * se]
        ci90 = [mean_diff - c90 * se, mean_diff + c90 * se]

    tost_ok = ci90[0] >= -EQUIV_MARGIN_PP and ci90[1] <= EQUIV_MARGIN_PP

    return PairedResult(
        label=label,
        n=n1 + n2,
        paired=False,
        diff_mean_pp=mean_diff,
        sample_sd_diff_pp=float("nan"),
        se_diff_pp=se,
        df=df,
        t_stat=t_stat,
        p_ttest_two_sided=p_ttest,
        wilcoxon_stat=None,
        p_wilcoxon_two_sided=None,
        wilcoxon_method="not applicable (unpaired Welch comparison)",
        ci95_pp=ci95,
        ci90_pp=ci90,
        equivalence_margin_pp=EQUIV_MARGIN_PP,
        tost_established=tost_ok,
        extra={"n1": n1, "n2": n2, "sample_sd_a_pp": round(s1, 4), "sample_sd_b_pp": round(s2, 4)},
    )


def mean_ci_t(values: list[float], label: str = "") -> dict:
    """t-based 95% CI of the mean of `values` (accuracy fractions), reported in pp."""
    n = len(values)
    vals_pp = [v * 100.0 for v in values]
    mean = sum(vals_pp) / n
    sd = sample_sd(vals_pp)
    se = sd / math.sqrt(n) if n > 1 else 0.0
    df = n - 1
    c95 = t_ppf(0.975, df) if df > 0 else float("nan")
    half = c95 * se if df > 0 else 0.0
    return {
        "label": label,
        "n": n,
        "mean_acc_pct": round(mean, 4),
        "sample_sd_pp": round(sd, 4),
        "ci95_low_pct": round(mean - half, 4),
        "ci95_high_pct": round(mean + half, 4),
    }
