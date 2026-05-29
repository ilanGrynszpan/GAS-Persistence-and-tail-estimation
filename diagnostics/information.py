"""
Information criteria: log-likelihood, AIC, BIC.

AIC  = -2 * loglik  +  2 * k
BIC  = -2 * loglik  +  k * ln(n)

where  k  is the number of free parameters and  n  is the effective sample size.
"""

from __future__ import annotations
import numpy as np


def aic(loglik: float, n_params: int) -> float:
    """Akaike Information Criterion."""
    return -2.0 * loglik + 2.0 * n_params


def bic(loglik: float, n_params: int, n_obs: int) -> float:
    """Bayesian Information Criterion."""
    return -2.0 * loglik + n_params * np.log(n_obs)


def info_table(loglik: float, n_params: int, n_obs: int) -> dict:
    """
    Return a dict with log-likelihood, AIC, and BIC.

    Parameters
    ----------
    loglik   : log-likelihood at the optimum
    n_params : number of free parameters
    n_obs    : effective sample size (observations used in the likelihood)
    """
    return {
        "loglik":   loglik,
        "n_params": n_params,
        "n_obs":    n_obs,
        "aic":      aic(loglik, n_params),
        "bic":      bic(loglik, n_params, n_obs),
    }
