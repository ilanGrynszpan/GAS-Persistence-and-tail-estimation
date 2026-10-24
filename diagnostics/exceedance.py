"""
Upper-tail exceedance rate + Kupiec unconditional-coverage test.

For a well-specified time-varying predictive distribution, the observed
value should exceed its own q-th conditional quantile on a fraction
(1 - q) of periods. This module measures that fraction and tests it with
Kupiec's (1995) LR_uc statistic.

diagnostics/tests.py already has a `_kupiec_statistic`, but it is written
for the LOWER tail (violation = PIT < alpha) and returns (inf, 0.0) for
zero violations -- which mis-flags the small-sample multi-step case, where
zero exceedances out of a handful of origins is the single most likely
outcome. `kupiec` below handles N1 in {0, N} with the correct limiting
log-likelihood instead.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def kupiec(n_viol: int, n_total: int, alpha: float) -> tuple[float, float]:
    """
    Kupiec unconditional-coverage LR test.

    Parameters
    ----------
    n_viol  : observed number of exceedances
    n_total : number of forecasts evaluated
    alpha   : expected exceedance rate under correct specification (= 1 - q)

    Returns (LR statistic, p-value under chi2(1)).
    """
    N, N1 = int(n_total), int(n_viol)
    if N <= 0:
        return (float("nan"), float("nan"))
    N0 = N - N1
    ll_h0 = N1 * np.log(alpha) + N0 * np.log(1.0 - alpha)
    if N1 == 0:
        ll_h1 = 0.0                                   # p_hat = 0
    elif N1 == N:
        ll_h1 = 0.0                                   # p_hat = 1
    else:
        p_hat = N1 / N
        ll_h1 = N1 * np.log(p_hat) + N0 * np.log(1.0 - p_hat)
    lr = -2.0 * (ll_h0 - ll_h1)
    return (float(lr), float(chi2.sf(lr, df=1)))


def exceedance_from_pit(pit: np.ndarray, q: float) -> dict:
    """1-step exceedance rate + Kupiec at quantile level q, from OOS PIT
    values (violation = PIT_t > q)."""
    pit = np.asarray(pit, dtype=float)
    pit = pit[np.isfinite(pit)]
    n = len(pit)
    if n == 0:
        return {"rate": float("nan"), "n_viol": 0, "n": 0,
                "kupiec_lr": float("nan"), "kupiec_p": float("nan")}
    v = int(np.sum(pit > q))
    lr, p = kupiec(v, n, 1.0 - q)
    return {"rate": v / n, "n_viol": v, "n": n, "kupiec_lr": lr, "kupiec_p": p}


def exceedance_from_draws(draws_list, actuals, q: float) -> dict:
    """Multi-step exceedance rate + Kupiec at level q: for each origin, the
    predictive q-quantile is the q-quantile of that origin's Monte Carlo
    draws; violation = actual > that quantile."""
    v = 0
    n = 0
    for d, a in zip(draws_list, actuals):
        d = np.asarray(d, dtype=float)
        if d.size == 0 or not np.isfinite(a):
            continue
        n += 1
        if a > np.quantile(d, q):
            v += 1
    if n == 0:
        return {"rate": float("nan"), "n_viol": 0, "n": 0,
                "kupiec_lr": float("nan"), "kupiec_p": float("nan")}
    lr, p = kupiec(v, n, 1.0 - q)
    return {"rate": v / n, "n_viol": v, "n": n, "kupiec_lr": lr, "kupiec_p": p}


def _xlogy(n: float, p: float) -> float:
    """n * log(p) with the 0 * log(0) = 0 convention."""
    return 0.0 if n == 0 else float(n * np.log(p))


def christoffersen_from_pit(pit: np.ndarray, q: float) -> dict:
    """
    Christoffersen (1998) independence and conditional-coverage tests for
    UPPER-tail violations I_t = 1{PIT_t > q} of the one-step forecasts.

    With n_ij = #{t : I_{t-1} = i, I_t = j}:
        pi01 = n01 / (n00 + n01)   P(violation | no violation the day before)
        pi11 = n11 / (n10 + n11)   P(violation | violation the day before)
        pi   = (n01 + n11) / (n00 + n01 + n10 + n11)
        LR_ind = -2 [ log L(pi) - log L(pi01, pi11) ]  ~ chi2(1)
            L(pi)         = (1-pi)^(n00+n10) pi^(n01+n11)
            L(pi01, pi11) = (1-pi01)^n00 pi01^n01 (1-pi11)^n10 pi11^n11
        LR_cc  = LR_uc + LR_ind                        ~ chi2(2)
    (LR_uc = Kupiec, from `kupiec` above on the same sequence.) Clustering of
    extremes shows as pi11 > pi01 with a small LR_ind p-value. 0*log(0) = 0,
    so sequences without consecutive violations are handled exactly; with no
    violations at all the independence test is undefined (NaN).
    Non-finite PIT values are dropped before forming the sequence.
    """
    pit = np.asarray(pit, dtype=float)
    pit = pit[np.isfinite(pit)]
    I = (pit > q).astype(int)
    n = len(I)
    out = {"n": n, "n_viol": int(I.sum())}
    if n < 2:
        return {**out, "pi01": np.nan, "pi11": np.nan, "lr_ind": np.nan, "p_ind": np.nan,
                "lr_cc": np.nan, "p_cc": np.nan, "n00": 0, "n01": 0, "n10": 0, "n11": 0}
    prev, cur = I[:-1], I[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    pi01 = n01 / (n00 + n01) if (n00 + n01) else np.nan
    pi11 = n11 / (n10 + n11) if (n10 + n11) else np.nan
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    if n01 + n11 == 0:
        lr_ind, p_ind = np.nan, np.nan
    else:
        ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
        ll1 = (_xlogy(n00, 1 - pi01) + _xlogy(n01, pi01)
               + (_xlogy(n10, 1 - pi11) + _xlogy(n11, pi11) if (n10 + n11) else 0.0))
        lr_ind = max(0.0, -2.0 * (ll0 - ll1))
        p_ind = float(chi2.sf(lr_ind, df=1))
    lr_uc, _ = kupiec(int(I.sum()), n, 1.0 - q)
    lr_cc = lr_uc + lr_ind if np.isfinite(lr_ind) else np.nan
    p_cc = float(chi2.sf(lr_cc, df=2)) if np.isfinite(lr_cc) else np.nan
    return {**out, "n00": n00, "n01": n01, "n10": n10, "n11": n11, "pi01": pi01, "pi11": pi11,
            "lr_ind": lr_ind, "p_ind": p_ind, "lr_cc": lr_cc, "p_cc": p_cc}
