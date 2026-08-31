"""
Randomised PIT and quantile residuals for a pure Bernoulli occurrence model.

=============================================================================
THEORY
=============================================================================

diagnostics/residuals.py's `pit_values` implements randomised PIT for the
*zero-augmented mixture* F_t(y) = (1-pi_t)*1(y>=0) + pi_t*G_t(y): it only
randomises at y_t=0 because the mixture is continuous for y>0 (the GB2
part), so F_t(y_t) for a wet observation is already a genuine, deterministic
PIT value.

This module is for a DIFFERENT random variable: the occurrence indicator
itself,

    Z_t = 1(y_t > 0)  ~  Bernoulli(pi_t),

used when comparing occurrence-probability (pi_t) dynamics in isolation,
independently of any positive-part distribution (see
pi_dynamics/standalone_fit.py). Z_t is discrete on BOTH support points
{0, 1}, so its CDF

    F_t(0) = 1 - pi_t,   F_t(1) = 1

is a two-step function and BOTH points need randomisation, not just Z_t=0,
for the PIT to be exactly Uniform(0,1) under correct specification (Smith
1985; Czado, Gneiting & Held 2009, "Predictive Model Assessment for Count
Data", Biometrics 65). The general randomised PIT for a discrete variable
is

    U_t = F_t(Z_t - 1) + V_t * [F_t(Z_t) - F_t(Z_t-1)],   V_t ~ Uniform(0,1)

which specialises here to

    Z_t = 0:  U_t ~ Uniform(0,        1 - pi_t)
    Z_t = 1:  U_t ~ Uniform(1 - pi_t, 1)

Under correct specification (pi_t is the true occurrence probability at
every t), U_t ~ iid Uniform(0,1), and the quantile residuals
r_t = Phi^{-1}(U_t) ~ iid N(0,1) -- the same Dunn & Smyth (1996)
construction diagnostics/residuals.py already uses for the full mixture,
applied here to the Bernoulli sub-model alone.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm


def bernoulli_pit_values(
    pi: np.ndarray,
    z: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Randomised PIT values for a Bernoulli(pi_t) occurrence indicator.

    Parameters
    ----------
    pi  : fitted occurrence probabilities pi_t, shape (n,), values in (0,1).
    z   : observed occurrence indicator z_t = 1(y_t>0), shape (n,), in {0,1}.
    rng : numpy Generator for reproducibility; defaults to a new RNG.

    Returns
    -------
    u : array of shape (n,) with values in (0, 1), Uniform(0,1) under the
        null that pi is correctly specified.
    """
    pi = np.asarray(pi, dtype=float)
    z = np.asarray(z, dtype=float)
    if len(pi) != len(z):
        raise ValueError("pi and z must have the same length")

    if rng is None:
        rng = np.random.default_rng()

    pi_c = np.clip(np.nan_to_num(pi, nan=0.5), 1e-10, 1.0 - 1e-10)
    f_below = 1.0 - pi_c  # F_t(0) = P(Z_t <= 0) = 1 - pi_t

    wet = z > 0.5
    u = np.empty_like(pi_c)
    # Z_t = 0: U_t ~ Uniform(0, F_t(0))
    u[~wet] = rng.uniform(0.0, f_below[~wet])
    # Z_t = 1: U_t ~ Uniform(F_t(0), 1)
    u[wet] = rng.uniform(f_below[wet], 1.0)

    return np.clip(u, 1e-10, 1.0 - 1e-10)


def bernoulli_quantile_residuals(
    pi: np.ndarray,
    z: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Quantile residuals r_t = Phi^{-1}(U_t) for the Bernoulli occurrence PIT."""
    u = bernoulli_pit_values(pi, z, rng=rng)
    return norm.ppf(u)
