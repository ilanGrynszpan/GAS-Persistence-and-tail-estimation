"""
Statistical tests for model diagnostics.

Implemented:
  - Jarque-Bera test on quantile residuals  (normality)
  - Kupiec unconditional coverage test      (VaR back-test)
  - Christoffersen conditional coverage test (VaR back-test with independence)

=============================================================================
KUPIEC AND CHRISTOFFERSEN TESTS
=============================================================================

These are likelihood-ratio tests designed for interval / quantile forecasts.
Given a coverage level alpha (e.g. 0.05), define the violation indicator:

    I_t  =  1  if  y_t > q_{alpha,t}    (observation exceeds the alpha-quantile)
          =  0  otherwise

where  q_{alpha,t} = F_t^{-1}(alpha)  is the predictive alpha-quantile.

For a correctly specified model  E[I_t] = alpha  and  I_t  are iid Bernoulli(alpha).

Kupiec LR_uc  (unconditional coverage):
    LR_uc = -2 * ln[ alpha^{N1} * (1-alpha)^{N0} /
                     p_hat^{N1} * (1-p_hat)^{N0} ]
    ~ chi^2(1)  under H0

where  N1 = #{I_t = 1},  N0 = #{I_t = 0},  p_hat = N1 / (N0+N1).

Christoffersen LR_cc = LR_uc + LR_ind  ~ chi^2(2)  under H0.

LR_ind tests serial independence of violations via a first-order Markov chain:
    LR_ind = -2 * ln [ (1-pi)^{N00+N01} * pi^{N10+N11} /
                       (1-pi_00)^{N00} * pi_00^{N01} *
                       (1-pi_10)^{N10} * pi_10^{N11} ]

where  N_{ij} = #{I_{t-1}=i, I_t=j},  pi_ij = N_{ij} / N_{i.}
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.stats import chi2, jarque_bera as _jb


# ===========================================================================
# Jarque-Bera
# ===========================================================================

def jarque_bera(residuals: np.ndarray) -> dict:
    """
    Jarque-Bera test for normality.

    H0: residuals ~ N(0, sigma^2).
    Under H0 the test statistic is approximately chi^2(2).

    Parameters
    ----------
    residuals : array of quantile residuals

    Returns
    -------
    dict with keys: stat, pvalue, skewness, kurtosis
    """
    r    = residuals[np.isfinite(residuals)]
    stat, pval = _jb(r)
    n    = len(r)
    skew = float(np.mean((r - r.mean()) ** 3) / r.std() ** 3)
    kurt = float(np.mean((r - r.mean()) ** 4) / r.std() ** 4)  # raw kurtosis
    return {
        "stat":     float(stat),
        "pvalue":   float(pval),
        "skewness": skew,
        "kurtosis": kurt,   # excess = kurt - 3
        "n":        n,
    }


# ===========================================================================
# Coverage test utilities
# ===========================================================================

def _violations(pit_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute violation indicator  I_t = 1{PIT_t < alpha}.

    I_t = 1 means the observation fell below the alpha-quantile of the
    predictive distribution, i.e. the interval [0, alpha] was violated.
    """
    return (pit_values < alpha).astype(int)


def _kupiec_statistic(
    violations: np.ndarray, alpha: float
) -> Tuple[float, float]:
    """
    Kupiec LR_uc test.

    Returns (stat, pvalue).
    """
    N  = len(violations)
    N1 = int(violations.sum())      # number of violations
    N0 = N - N1

    if N1 == 0 or N1 == N:
        # Degenerate: return large stat (boundary case)
        return (np.inf, 0.0)

    p_hat = N1 / N

    # Log-likelihood under H0 (Bernoulli with rate alpha)
    ll_h0 = N1 * np.log(alpha) + N0 * np.log(1.0 - alpha)
    # Log-likelihood under H1 (Bernoulli with rate p_hat)
    ll_h1 = N1 * np.log(p_hat) + N0 * np.log(1.0 - p_hat)

    stat  = -2.0 * (ll_h0 - ll_h1)
    pval  = float(chi2.sf(stat, df=1))
    return (float(stat), pval)


def _christoffersen_statistic(
    violations: np.ndarray, alpha: float
) -> Tuple[float, float, float, float]:
    """
    Christoffersen LR_ind and LR_cc tests.

    Returns (LR_uc, LR_ind, LR_cc, pvalue_cc).
    """
    lr_uc, _ = _kupiec_statistic(violations, alpha)

    # Transition counts
    # N_{ij} = number of transitions from state i to state j
    I    = violations
    N00  = int(np.sum((I[:-1] == 0) & (I[1:] == 0)))
    N01  = int(np.sum((I[:-1] == 0) & (I[1:] == 1)))
    N10  = int(np.sum((I[:-1] == 1) & (I[1:] == 0)))
    N11  = int(np.sum((I[:-1] == 1) & (I[1:] == 1)))

    pi_hat = (N01 + N11) / max(N00 + N01 + N10 + N11, 1)  # unconditional

    # Conditional rates
    pi_00  = N01 / max(N00 + N01, 1)   # P(violation | no violation yesterday)
    pi_10  = N11 / max(N10 + N11, 1)   # P(violation | violation yesterday)

    # LL under independence (single violation rate)
    ll_ind = (
        (N00 + N10) * np.log(max(1.0 - pi_hat, 1e-12))
        + (N01 + N11) * np.log(max(pi_hat, 1e-12))
    )

    # LL under first-order Markov chain
    ll_mc  = (
        N00 * np.log(max(1.0 - pi_00, 1e-12))
        + N01 * np.log(max(pi_00, 1e-12))
        + N10 * np.log(max(1.0 - pi_10, 1e-12))
        + N11 * np.log(max(pi_10, 1e-12))
    )

    lr_ind = -2.0 * (ll_ind - ll_mc)
    lr_cc  = lr_uc + lr_ind
    pval_cc = float(chi2.sf(lr_cc, df=2))
    return (float(lr_uc), float(lr_ind), float(lr_cc), pval_cc)


# ===========================================================================
# Public functions
# ===========================================================================

def kupiec_test(pit_values: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Kupiec (1995) unconditional coverage test.

    Parameters
    ----------
    pit_values : PIT values for the effective sample, shape (n,)
    alpha      : coverage level (e.g. 0.05 tests the 5th percentile)

    Returns
    -------
    dict with keys: alpha, expected_violations, actual_violations,
                    violation_rate, stat, pvalue, reject_5pct
    """
    I    = _violations(pit_values, alpha)
    N    = len(I)
    N1   = int(I.sum())
    stat, pval = _kupiec_statistic(I, alpha)

    return {
        "test":                "Kupiec LR_uc",
        "alpha":               alpha,
        "expected_violations": float(alpha * N),
        "actual_violations":   N1,
        "violation_rate":      N1 / N,
        "stat":                stat,
        "pvalue":              pval,
        "reject_5pct":         pval < 0.05,
    }


def christoffersen_test(pit_values: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Christoffersen (1998) conditional coverage test.

    Tests both unconditional coverage (Kupiec) and serial independence of
    violations.

    Parameters
    ----------
    pit_values : PIT values, shape (n,)
    alpha      : coverage level

    Returns
    -------
    dict with keys: alpha, LR_uc, LR_ind, LR_cc, pvalue_uc, pvalue_cc,
                    reject_5pct_uc, reject_5pct_cc
    """
    I = _violations(pit_values, alpha)
    lr_uc, lr_ind, lr_cc, pval_cc = _christoffersen_statistic(I, alpha)
    _, pval_uc = _kupiec_statistic(I, alpha)

    return {
        "test":           "Christoffersen LR_cc",
        "alpha":          alpha,
        "LR_uc":          lr_uc,
        "LR_ind":         lr_ind,
        "LR_cc":          lr_cc,
        "pvalue_uc":      pval_uc,
        "pvalue_cc":      pval_cc,
        "reject_5pct_uc": pval_uc < 0.05,
        "reject_5pct_cc": pval_cc < 0.05,
    }


def coverage_tests(
    pit_values: np.ndarray,
    alphas: list[float] | None = None,
) -> list[dict]:
    """
    Run both Kupiec and Christoffersen tests at several coverage levels.

    Parameters
    ----------
    pit_values : PIT values, shape (n,)
    alphas     : list of levels; defaults to [0.01, 0.05, 0.10, 0.25]

    Returns
    -------
    List of dicts, one per alpha, each containing both test results.
    """
    if alphas is None:
        alphas = [0.01, 0.05, 0.10, 0.25]

    rows = []
    for a in alphas:
        k = kupiec_test(pit_values, a)
        c = christoffersen_test(pit_values, a)
        rows.append({
            "alpha":          a,
            "violations":     k["actual_violations"],
            "violation_rate": k["violation_rate"],
            "LR_uc":          k["stat"],
            "pvalue_uc":      k["pvalue"],
            "LR_ind":         c["LR_ind"],
            "LR_cc":          c["LR_cc"],
            "pvalue_cc":      c["pvalue_cc"],
            "reject_uc":      k["reject_5pct"],
            "reject_cc":      c["reject_5pct_cc"],
        })
    return rows
