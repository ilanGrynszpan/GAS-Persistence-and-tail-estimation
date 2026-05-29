"""
Out-of-sample simulation and forecast evaluation.

=============================================================================
APPROACH
=============================================================================

Given an estimated ZA-GAS model fitted on training data, we evaluate
forecast quality on a held-out test set.

Two evaluation modes:
  1. In-sample (IS):   use filtered paths on training data
  2. Out-of-sample (OOS): advance the GAS + pi recursion on test data,
     using the terminal state from training as the initial state.

For each time step t in the test set, the 1-step-ahead predictive distribution
is:

    F_t(y) = (1 - pi_t) * I(y >= 0)  +  pi_t * G_t(y)   [ZA model]

=============================================================================
METRICS
=============================================================================

RMSE:  sqrt( mean( (E[Y_t] - y_t)^2 ) )
       where E[Y_t] = pi_t * E[GB2_t]  is the predictive mean.

MAD:   mean( |E[Y_t] - y_t| )

CRPS:  Continuous Ranked Probability Score (energy score decomposition).
       Computed by Monte Carlo simulation:

           CRPS_t  =  (1/M) * sum_m |y^m_t - y_t|
                    - (1/2M^2) * sum_{m,m'} |y^m_t - y^{m'}_t|

       where y^m_t are M draws from the predictive distribution at t.

       Lower CRPS = better calibration + sharpness.

=============================================================================
USAGE
=============================================================================

    from simulation.simulator import simulate_oos, evaluate_metrics

    oos_paths = simulate_oos(model, theta_hat, y_train, y_test)
    metrics   = evaluate_metrics(model, theta_hat, y_train, y_test,
                                 n_draws=500, seed=42)
"""

from __future__ import annotations
from typing import List

import numpy as np
from scipy.special import expit, beta as beta_fn

from models.za_gas_model import ZAGASModel
from models.lags import SEASONAL_LAGS
from diagnostics.residuals import quantile_residuals, pit_values


# ===========================================================================
# OOS filter: advance the recursion into test data
# ===========================================================================

def simulate_oos(
    model: ZAGASModel,
    theta: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Advance the fitted ZA-GAS model into the test period.

    The function:
      1. Runs the filter on training data to get the terminal state.
      2. Continues the recursion on test data (1-step-ahead predictions).

    No future y values are fed into the GAS update on the test set;
    at each test step we predict y_t, observe it, update the state.
    This is a true 1-step-ahead rolling evaluation.

    Parameters
    ----------
    model   : fitted ZAGASModel
    theta   : optimal parameter vector from model.fit()
    y_train : training observations, shape (T_train,)
    y_test  : test observations, shape (T_test,)

    Returns
    -------
    dict with keys:
        phi_oos   : filtered phi_t on test data
        xi_oos    : filtered xi_t on test data
        pi_oos    : filtered pi_t on test data
        f_arr_oos : shape (T_test, n_tv)
        eta_oos   : logit pi on test data
        static    : dict of static parameters
        y_test    : echo of y_test for convenience
    """
    # ---- Decode parameters ----
    gas_theta, pi_theta = model._split_theta(theta)
    gp       = model.gas.codec.decode(gas_theta)
    pi_params = model._decode_pi(pi_theta)

    L       = model.lags
    max_lag = model.max_lag
    n_tv    = len(model.gas.tv_names)

    # ---- Concatenate train + test for full-array indexing ----
    # The GAS update at test step t accesses y and states up to lag 367.
    # It's easiest to extend the training arrays and continue the recursion.
    y_full  = np.concatenate([y_train, y_test])
    T_train = len(y_train)
    T_full  = len(y_full)

    # Allocate full-length arrays
    f_arr   = np.zeros((T_full + 1, n_tv))
    s_arr   = np.zeros((T_full,     n_tv))
    eta_arr = np.zeros(T_full)
    pi_arr  = np.zeros(T_full)

    # Initialise with f0 for pre-history
    for j, name in enumerate(model.gas.tv_names):
        f_arr[:max_lag + 1, j] = gp[f"f0_{name}"]

    # ---- Re-run training filter to populate f_arr, s_arr, eta_arr ----
    for t in range(max_lag, T_train):
        call_params = {
            name: f_arr[t, j]
            for j, name in enumerate(model.gas.tv_names)
        }
        call_params.update({s: gp[s] for s in model.gas.static_names})

        # Pi
        eta_arr[t] = model.pi_dyn.compute_eta(
            i=t - max_lag, eta_hist=eta_arr, y_full=y_full,
            t=t, params=pi_params, seasonal=model.seasonal,
        )
        pi_arr[t] = float(expit(eta_arr[t]))

        # Score
        if y_full[t] > 0:
            raw = model.dist.score(y_full[t], **call_params)
            if model.gas.scale_score:
                fi = model.dist.fisher_info_diag(**call_params)
                for j, name in enumerate(model.gas.tv_names):
                    s_arr[t, j] = raw[name] / max(fi[name], 1e-8)
            else:
                for j, name in enumerate(model.gas.tv_names):
                    s_arr[t, j] = raw[name]

        # GAS update
        for j, name in enumerate(model.gas.tv_names):
            omega_j = gp[f"omega_{name}"]
            ar_p = score_p = 0.0
            for l in L:
                idx    = t - l + 1
                s_past = s_arr[idx, j] if idx >= 0 else 0.0
                f_past = f_arr[idx, j] if idx >= 0 else gp[f"f0_{name}"]
                score_p += gp[f"A_{name}_{l}"] * s_past
                ar_p    += gp[f"B_{name}_{l}"] * f_past
            f_arr[t + 1, j] = np.clip(omega_j + score_p + ar_p, -15.0, 15.0)

    # ---- Continue into test period ----
    for t in range(T_train, T_full):
        call_params = {
            name: f_arr[t, j]
            for j, name in enumerate(model.gas.tv_names)
        }
        call_params.update({s: gp[s] for s in model.gas.static_names})

        # Pi
        eta_arr[t] = model.pi_dyn.compute_eta(
            i=t - max_lag, eta_hist=eta_arr, y_full=y_full,
            t=t, params=pi_params, seasonal=model.seasonal,
        )
        pi_arr[t] = float(expit(eta_arr[t]))

        # Score (using observed test y for 1-step rolling update)
        if y_full[t] > 0:
            raw = model.dist.score(y_full[t], **call_params)
            if model.gas.scale_score:
                fi = model.dist.fisher_info_diag(**call_params)
                for j, name in enumerate(model.gas.tv_names):
                    s_arr[t, j] = raw[name] / max(fi[name], 1e-8)
            else:
                for j, name in enumerate(model.gas.tv_names):
                    s_arr[t, j] = raw[name]

        # GAS update
        for j, name in enumerate(model.gas.tv_names):
            omega_j = gp[f"omega_{name}"]
            ar_p = score_p = 0.0
            for l in L:
                idx    = t - l + 1
                s_past = s_arr[idx, j] if idx >= 0 else 0.0
                f_past = f_arr[idx, j] if idx >= 0 else gp[f"f0_{name}"]
                score_p += gp[f"A_{name}_{l}"] * s_past
                ar_p    += gp[f"B_{name}_{l}"] * f_past
            f_arr[t + 1, j] = np.clip(omega_j + score_p + ar_p, -15.0, 15.0)

    # Slice out test period only
    oos_slice = slice(T_train, T_full)
    return {
        "f_arr_oos": f_arr[oos_slice, :],
        "phi_oos":   f_arr[oos_slice, 0],
        "xi_oos":    f_arr[oos_slice, 1] if n_tv > 1 else None,
        "eta_oos":   eta_arr[oos_slice],
        "pi_oos":    pi_arr[oos_slice],
        "static":    {s: gp[s] for s in model.gas.static_names},
        "tv_names":  model.gas.tv_names,
        "y_test":    y_test,
    }


# ===========================================================================
# Predictive distribution helpers
# ===========================================================================

def _predictive_mean(model, paths, idx) -> float:
    """E[Y_t] = pi_t * E[GB2_t] for the ZA model."""
    pi_t   = float(paths["pi_oos"][idx])
    static = paths["static"]
    call   = {
        name: float(paths["f_arr_oos"][idx, j])
        for j, name in enumerate(paths["tv_names"])
    }
    call.update(static)

    # GB2 mean: E[Y] = sigma * B(a+p, b-p) / B(a, b)
    phi   = call.get("phi",   0.0)
    xi    = call.get("xi",    np.log(1.2))
    gamma = call.get("gamma", 1.0)
    zeta  = call.get("zeta",  3.0)
    a     = np.exp(xi)
    b     = np.exp(zeta)
    p     = np.exp(-gamma)
    sigma = np.exp(phi)

    try:
        gb2_mean = sigma * beta_fn(a + p, b - p) / beta_fn(a, b)
    except Exception:
        gb2_mean = sigma  # fallback

    return pi_t * float(gb2_mean)


def _draw_from_predictive(
    model: ZAGASModel,
    paths: dict,
    idx: int,
    n_draws: int = 500,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Draw n_draws samples from the predictive ZA-GB2 distribution at index idx.

    Zero draws (Bernoulli 1-pi) are mixed with GB2 draws (Bernoulli pi).
    """
    if rng is None:
        rng = np.random.default_rng()

    pi_t   = float(paths["pi_oos"][idx])
    static = paths["static"]
    call   = {
        name: float(paths["f_arr_oos"][idx, j])
        for j, name in enumerate(paths["tv_names"])
    }
    call.update(static)

    # Bernoulli indicator: 1 = positive draw, 0 = zero
    is_positive = rng.binomial(1, pi_t, size=n_draws)

    # GB2 samples
    n_pos = int(is_positive.sum())
    gb2_draws = np.zeros(n_draws)
    if n_pos > 0:
        try:
            gb2_draws[is_positive == 1] = model.dist.rvs(n_pos, **call)
        except Exception:
            pass  # leave as 0 on numerical failure

    return gb2_draws  # zeros for non-positive draws


# ===========================================================================
# Metrics
# ===========================================================================

def _crps_mc(draws: np.ndarray, y_obs: float) -> float:
    """
    Monte Carlo CRPS:
        CRPS = E|Y - y| - 0.5 * E|Y - Y'|
    """
    M  = len(draws)
    e1 = float(np.mean(np.abs(draws - y_obs)))
    e2 = 0.0
    # Fast approximation for E|Y - Y'| via sorted array
    s  = np.sort(draws)
    e2 = float(np.mean(np.abs(s - s[::-1])))
    return e1 - 0.5 * e2


def evaluate_metrics(
    model: ZAGASModel,
    theta: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_draws: int = 500,
    seed: int = 42,
) -> dict:
    """
    Compute RMSE, MAD, CRPS, and Jarque-Bera on quantile residuals,
    both in-sample (IS) and out-of-sample (OOS).

    Parameters
    ----------
    model    : fitted ZAGASModel
    theta    : optimal parameter vector
    y_train  : training series
    y_test   : test series
    n_draws  : number of Monte Carlo draws per time step for CRPS
    seed     : random seed

    Returns
    -------
    dict with keys:
        is_rmse, is_mad, is_crps, is_jb  (in-sample)
        oos_rmse, oos_mad, oos_crps, oos_jb  (out-of-sample)
        oos_paths  (the paths dict from simulate_oos)
    """
    from diagnostics.tests import jarque_bera

    rng = np.random.default_rng(seed)

    # ---- IN-SAMPLE ----
    is_paths = model.filter(theta, y_train)
    is_cdfs  = model.cdf_series(theta, y_train)
    y_eff    = is_paths["y_eff"]
    n_eff    = len(y_eff)

    is_means  = np.array([_predictive_mean(model, {
        "f_arr_oos": is_paths["f_arr"],
        "pi_oos":    is_paths["pi"],
        "static":    is_paths["static"],
        "tv_names":  is_paths["tv_names"],
    }, i) for i in range(n_eff)])

    is_rmse = float(np.sqrt(np.mean((is_means - y_eff) ** 2)))
    is_mad  = float(np.mean(np.abs(is_means - y_eff)))

    is_crps_vals = np.array([
        _crps_mc(
            _draw_from_predictive(model, {
                "f_arr_oos": is_paths["f_arr"],
                "pi_oos":    is_paths["pi"],
                "static":    is_paths["static"],
                "tv_names":  is_paths["tv_names"],
            }, i, n_draws, rng),
            y_eff[i],
        )
        for i in range(n_eff)
    ])
    is_crps = float(np.mean(is_crps_vals))

    is_pit = pit_values(is_cdfs, y_eff, randomise_zeros=True, rng=rng)
    is_qr  = quantile_residuals(is_cdfs, y_eff, randomise_zeros=True, rng=rng)
    is_jb  = jarque_bera(is_qr)

    # ---- OUT-OF-SAMPLE ----
    oos_paths = simulate_oos(model, theta, y_train, y_test)
    T_test    = len(y_test)

    oos_means = np.array([
        _predictive_mean(model, oos_paths, i) for i in range(T_test)
    ])

    oos_rmse = float(np.sqrt(np.mean((oos_means - y_test) ** 2)))
    oos_mad  = float(np.mean(np.abs(oos_means - y_test)))

    # OOS CDF values (needed for CRPS and residuals)
    oos_cdfs = np.zeros(T_test)
    for i in range(T_test):
        pi_t   = float(oos_paths["pi_oos"][i])
        static = oos_paths["static"]
        call   = {
            name: float(oos_paths["f_arr_oos"][i, j])
            for j, name in enumerate(oos_paths["tv_names"])
        }
        call.update(static)
        if y_test[i] <= 0:
            oos_cdfs[i] = 1.0 - pi_t
        else:
            G_t = model.dist.cdf(y_test[i], **call)
            oos_cdfs[i] = (1.0 - pi_t) + pi_t * G_t

    oos_crps_vals = np.array([
        _crps_mc(
            _draw_from_predictive(model, oos_paths, i, n_draws, rng),
            y_test[i],
        )
        for i in range(T_test)
    ])
    oos_crps = float(np.mean(oos_crps_vals))

    oos_pit = pit_values(oos_cdfs, y_test, randomise_zeros=True, rng=rng)
    oos_qr  = quantile_residuals(oos_cdfs, y_test, randomise_zeros=True, rng=rng)
    oos_jb  = jarque_bera(oos_qr)

    return {
        # In-sample
        "is_rmse":  is_rmse,
        "is_mad":   is_mad,
        "is_crps":  is_crps,
        "is_jb":    is_jb,
        "is_qr":    is_qr,
        "is_pit":   is_pit,
        "is_cdfs":  is_cdfs,
        "is_paths": is_paths,
        # Out-of-sample
        "oos_rmse":  oos_rmse,
        "oos_mad":   oos_mad,
        "oos_crps":  oos_crps,
        "oos_jb":    oos_jb,
        "oos_qr":    oos_qr,
        "oos_pit":   oos_pit,
        "oos_cdfs":  oos_cdfs,
        "oos_paths": oos_paths,
    }
