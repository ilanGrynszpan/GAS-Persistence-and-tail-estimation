"""
In-sample RMSE / CRPS evaluation for the pi-dynamics alternatives comparison
(2026-07-14).

=============================================================================
PURPOSE
=============================================================================

pi_dynamics/{ar_logistic_custom_lags,phi_linked}.py provide four alternative
occurrence-probability (pi_t) dynamics, each fit standalone on the binary
wet/dry sequence alone (see pi_dynamics/standalone_fit.py). To see whether
the *choice of pi_t dynamics* changes how well the full zero-augmented model
predicts y_t itself -- not just the occurrence indicator -- each pi_t
alternative is paired with the SAME frozen magnitude GAS(p,q) path
(phi_{t|t-1}, and static xi/gamma/zeta) taken from the location's selected
Stage 1 phi-only baseline (see run_pi_dynamics_alternatives.py). Only pi_t
varies across the four models being compared; phi_t is identical across
them at a given location, so any difference in RMSE/CRPS below is
attributable to the occurrence dynamics alone.

This module computes, in-sample only (no OOS forecast is produced anywhere
in this experiment -- see pi_dynamics/standalone_fit.py, which never splits
train/test):

    mu_t   = pi_t * E[G_t]           (point forecast, ZA-mixture mean)
    RMSE_all, RMSE_dry (y_t=0 subset), RMSE_wet (y_t>0 subset)
    CRPS_all (mean continuous ranked probability score across all t)

E[G_t] uses distributions.gb2_log_link.GB2LogLink.mean() (an existing
analytical raw-moment formula, MODELS.md §4's "distribution module owns
... analytical moments" -- not re-derived here). CRPS at each t reuses
diagnostics.scoring.crps_mc(draws, y_obs), the same bias-corrected
energy-score Monte-Carlo estimator already used for every OOS CRPS number
reported elsewhere in this framework (see that module's docstring for the
correction history) -- draws are generated here the same way
diagnostics.scoring._draw_predictive does it (Bernoulli(pi_t) mask, GB2 ppf
for the positive draws), just inlined because _draw_predictive expects an
OOS-style `paths` dict this experiment does not build.
"""

from __future__ import annotations
from typing import Dict

import numpy as np

from diagnostics.scoring import crps_mc


def in_sample_point_and_crps_metrics(
    dist,
    phi_arr: np.ndarray,
    static: Dict[str, float],
    pi_arr: np.ndarray,
    y_arr: np.ndarray,
    n_draws: int = 500,
    seed: int = 42,
) -> dict:
    """
    In-sample RMSE (all/dry/wet) and mean CRPS for a ZA-GB2 model whose
    phi_t path and static (xi,gamma,zeta) are frozen, driven only by the
    supplied pi_t array.

    Parameters
    ----------
    dist     : GB2LogLink-like Distribution instance (needs .mean, .ppf).
    phi_arr  : frozen transformed-scale path, shape (n,), aligned to y_arr.
    static   : {"xi":..., "gamma":..., "zeta":...} -- frozen static params.
    pi_arr   : occurrence probability path for the model under evaluation,
               shape (n,), aligned to y_arr.
    y_arr    : observed precipitation, shape (n,).
    n_draws  : Monte Carlo draws per time step for CRPS (matches the
               n_draws=500 default used by diagnostics.scoring.compute_all_metrics
               elsewhere in this framework).
    seed     : RNG seed, for reproducibility across re-runs on saved artifacts.

    Returns
    -------
    dict with rmse_all, rmse_dry, rmse_wet, crps_all, n_obs, n_dry, n_wet.
    """
    phi_arr = np.asarray(phi_arr, dtype=float)
    pi_arr  = np.asarray(pi_arr, dtype=float)
    y_arr   = np.asarray(y_arr, dtype=float)
    n = len(y_arr)
    if len(phi_arr) != n or len(pi_arr) != n:
        raise ValueError(
            f"phi_arr ({len(phi_arr)}), pi_arr ({len(pi_arr)}), and y_arr "
            f"({n}) must all have the same length."
        )

    rng = np.random.default_rng(seed)
    mu = np.empty(n)
    crps_vals = np.empty(n)

    for t in range(n):
        call = {"phi": float(phi_arr[t]), **static}
        pi_t = float(np.clip(pi_arr[t], 0.0, 1.0))

        # Point forecast: E[Y_t] = pi_t * E[G_t | wet]  (0 contributes nothing).
        mu[t] = pi_t * float(dist.mean(**call))

        # CRPS via Monte Carlo draws from the ZA mixture -- same construction
        # as diagnostics.scoring._draw_predictive, reusing crps_mc for the
        # actual (bias-corrected) score computation.
        is_pos = rng.binomial(1, pi_t, size=n_draws)
        draws = np.zeros(n_draws)
        n_pos = int(is_pos.sum())
        if n_pos > 0:
            draws[is_pos == 1] = dist.ppf(
                rng.uniform(1e-8, 1.0 - 1e-8, size=n_pos), **call
            )
        crps_vals[t] = crps_mc(draws, float(y_arr[t]))

    wet_mask = y_arr > 0
    dry_mask = ~wet_mask

    def _rmse(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.sqrt(np.mean((y_arr[mask] - mu[mask]) ** 2)))

    return {
        "rmse_all": _rmse(np.ones(n, dtype=bool)),
        "rmse_dry": _rmse(dry_mask),
        "rmse_wet": _rmse(wet_mask),
        "crps_all": float(np.mean(crps_vals)),
        "n_obs": int(n),
        "n_dry": int(dry_mask.sum()),
        "n_wet": int(wet_mask.sum()),
    }
