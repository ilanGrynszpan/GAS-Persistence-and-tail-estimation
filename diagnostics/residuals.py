"""
Quantile residuals and Probability Integral Transform (PIT).

=============================================================================
THEORY
=============================================================================

Quantile residuals  (Dunn & Smyth 1996):
    r_t  =  Phi^{-1}( F_t(y_t) )

where  F_t(y_t)  is the predictive CDF evaluated at the observation y_t,
and  Phi^{-1}  is the standard normal quantile function.

If the model is correctly specified,  r_t ~ iid N(0, 1).

For a mixed distribution with a mass at zero (zero-augmented model), the
CDF has a jump at zero:

    F_t(0^-) = 0
    F_t(0)   = 1 - pi_t
    F_t(y)   = (1 - pi_t) + pi_t * G_t(y)  for y > 0

Because the CDF is discontinuous at y=0, we use randomised quantile residuals
for zero observations:

    u_t | y_t = 0  ~  Uniform(0,  F_t(0))
    u_t | y_t > 0  =  F_t(y_t)             [deterministic]

This gives r_t = Phi^{-1}(u_t) ~ N(0,1) exactly under correct specification.
The randomised residuals are less noisy when evaluated in aggregate.

PIT (Probability Integral Transform):
    The PIT values  u_t = F_t(y_t)  should be Uniform(0,1) under the null.
    For zero-augmented distributions we use the randomised version above.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm


def pit_values(
    cdfs: np.ndarray,
    y: np.ndarray,
    randomise_zeros: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Compute Probability Integral Transform (PIT) values.

    Parameters
    ----------
    cdfs            : predictive CDF values  F_t(y_t),  shape (n,)
                      For the ZA model use ZAGASModel.cdf_series().
    y               : observed values in the effective sample, shape (n,)
    randomise_zeros : if True, draw u_t ~ Uniform(0, F_t(0)) when y_t = 0
                      (randomised PIT for the mixed distribution)
    rng             : numpy Generator for reproducibility; defaults to new RNG

    Returns
    -------
    pit : array of shape (n,) with values in (0, 1)
    """
    n = len(cdfs)
    if len(y) != n:
        raise ValueError("cdfs and y must have the same length")

    if rng is None:
        rng = np.random.default_rng()

    pit = cdfs.copy().astype(float)

    if randomise_zeros:
        # For y_t = 0, the PIT is uniform on [0, F_t(0)]
        zero_mask = (y == 0)
        if zero_mask.any():
            # F_t(0) = cdfs[t] for zero observations (already computed)
            upper = np.nan_to_num(cdfs[zero_mask], nan=1e-10, posinf=1.0 - 1e-10, neginf=1e-10)
            upper = np.clip(upper, 1e-10, 1.0 - 1e-10)
            pit[zero_mask] = rng.uniform(0.0, upper)

    pit = np.nan_to_num(pit, nan=0.5, posinf=1.0 - 1e-10, neginf=1e-10)

    # Clip away exact 0/1 for numerical stability in norm.ppf
    pit = np.clip(pit, 1e-10, 1.0 - 1e-10)
    return pit


def quantile_residuals(
    cdfs: np.ndarray,
    y: np.ndarray,
    randomise_zeros: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Compute quantile residuals  r_t = Phi^{-1}(u_t).

    Parameters
    ----------
    cdfs            : predictive CDF values, shape (n,)
    y               : observed values, shape (n,)
    randomise_zeros : use randomised PIT for zero observations
    rng             : random number generator

    Returns
    -------
    r : array of shape (n,) — should be ~ N(0,1) under correct specification
    """
    u = pit_values(cdfs, y, randomise_zeros=randomise_zeros, rng=rng)
    return norm.ppf(u)
