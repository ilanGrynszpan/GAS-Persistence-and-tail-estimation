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
