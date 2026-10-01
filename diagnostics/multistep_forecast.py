"""
Multi-step-ahead Monte Carlo forecast evaluation for ZAGASModel (no
covariates), CovZAGASModel, and HarveyZAGASModel (2026-08-31, ZAGASModel
support added 2026-09-01 for reports/seasonality GAS/).

=============================================================================
OVERVIEW
=============================================================================

diagnostics/horizon_forecast.py already does this for phi-only, no-covariate
GASFilter models, but explicitly does not support phi+xi models or
covariates (see its own docstring). CovZAGASModel and HarveyZAGASModel need
both. Rather than duplicate horizon_forecast.py's simulation loop with those
two extensions bolted on, this module reimplements the same idea against a
different structural starting point: it mirrors each model's OWN
already-correct 1-step-ahead recursion (CovZAGASModel.simulate_oos's
internal `_update` closure; HarveyZAGASModel's L/S update in `_run_filter`)
literally, replaying it with REAL data to build the full filtered history up
to every forecast origin, then continuing the identical update mechanics
with SIMULATED y beyond the origin.

Exogenous covariates (dew point, ENSO) are assumed known over the forecast
horizon ("perfect foresight") -- standard practice for weather covariates in
a multi-step evaluation that is not also forecasting the covariates
themselves; this is stated explicitly in every report that uses this module.

=============================================================================
CODE WALKTHROUGH
=============================================================================

_replay_cov / _replay_harvey
    Deterministic single pass over y_train+y_test (REAL y throughout),
    rebuilding the exact same f_arr/s_arr/eta_arr/pi_arr (Cov) or
    f_arr/L_arr/S_arr/s_arr/eta_arr/pi_arr (Harvey) history the model's own
    fit/simulate_oos machinery already computes -- this is a *replay*, not a
    re-fit (theta is fixed), giving direct array access to the full history
    needed to seed a multi-step continuation from any origin.
    |
_simulate_forward_cov / _simulate_forward_harvey
    From one origin t0, continues the SAME per-step update for h_max steps,
    but DRAWS y_t from the model's own predictive distribution (Bernoulli
    occurrence x GB2 magnitude) instead of reading it, snapshotting the
    drawn value at every requested horizon. Repeated for n_draws Monte Carlo
    paths per origin.
    |
multistep_metrics(model, theta, ..., origins, horizons, n_draws, seed)
    Orchestrates both models' cases (dispatches on model type), aggregates
    RMSE (using the ensemble mean at each origin) and CRPS (using the full
    ensemble, diagnostics.scoring.crps_mc) across origins, per horizon.

=============================================================================
VALIDATION
=============================================================================

At h=1 this module's Monte Carlo RMSE/CRPS must closely match
diagnostics.scoring.compute_extended_oos_metrics's real (non-simulated)
1-step rolling values, since no extrapolation is needed at h=1 -- this is
checked directly in run_longshort_best_models_analysis.py before any h>1
result is trusted.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit

from constants import SEASONAL_LAGS
from diagnostics.scoring import crps_mc, twcrps_mc

DEFAULT_HORIZONS = (1, 2, 5, 10, 30, 60, 90)
#: twCRPS levels added to multistep_metrics (2026-09-06) -- same
#: threshold-weighted-CRPS definition already used by
#: diagnostics.scoring.compute_extended_oos_metrics for the 1-step case,
#: extended here to every requested horizon. Threshold convention matches
#: that function exactly: the wet-day quantile of y_test at this level,
#: computed ONCE (not per-origin/per-horizon) so all horizons share the
#: same fixed threshold value.
MULTISTEP_TWCRPS_LEVELS = (0.95, 0.98)


# ─────────────────────────────────────────────────────────────────────────────
# ZAGASModel (no covariates): replay + simulate
#
# Exact mirror of ZAGASModel.simulate_oos's own recursion (models/za_gas_model.py)
# -- structurally identical to the CovZAGASModel pair below with the
# covariate term removed, since a plain GAS(L) model is exactly a Cov model
# with n_cov=0. Kept as its own small pair (rather than calling the Cov
# functions with a zero covariate matrix) because ZAGASModel's
# _split_theta returns a 2-tuple (gas_theta, pi_theta), not CovZAGASModel's
# 3-tuple, and it has no _decode_cov/cov_target/n_cov attributes at all.
# ─────────────────────────────────────────────────────────────────────────────

def _base_replay(model, theta: np.ndarray, y_full: np.ndarray) -> dict:
    """Deterministic replay of ZAGASModel.simulate_oos's own recursion over
    REAL y_full, returning the full f_arr/s_arr/eta_arr/pi_arr history."""
    gas_theta, pi_theta = model._split_theta(theta)
    gp = model.gas.codec.decode(gas_theta)
    pi_params = model._decode_pi(pi_theta)

    L, max_lag, n_tv = model.lags, model.max_lag, len(model.gas.tv_names)
    T_full = len(y_full)

    f_arr = np.zeros((T_full + 1, n_tv))
    s_arr = np.zeros((T_full, n_tv))
    eta_arr = np.zeros(T_full)
    pi_arr = np.zeros(T_full)
    for j, name in enumerate(model.gas.tv_names):
        f_arr[: max_lag + 1, j] = gp[f"f0_{name}"]

    for t in range(max_lag, T_full):
        _base_update(t, y_full, f_arr, s_arr, eta_arr, pi_arr, model, gp, pi_params, L)

    return {"f_arr": f_arr, "s_arr": s_arr, "eta_arr": eta_arr, "pi_arr": pi_arr,
            "gp": gp, "pi_params": pi_params, "_y_real": y_full}


def _base_update(t, y_full, f_arr, s_arr, eta_arr, pi_arr, model, gp, pi_params, lags):
    """Exact mirror of ZAGASModel.simulate_oos's per-step update."""
    call_params = {name: f_arr[t, j] for j, name in enumerate(model.gas.tv_names)}
    call_params.update({s: gp[s] for s in model.gas.static_names})

    # eta_arr is t-indexed here; pass i=t so compute_eta's eta_hist[i-1]
    # reads eta_arr[t-1] (the previous step), matching the forecast recursion.
    eta_arr[t] = model.pi_dyn.compute_eta(
        i=t, eta_hist=eta_arr, y_full=y_full, t=t,
        params=pi_params, seasonal=model.seasonal,
    )
    pi_arr[t] = float(model.pi_dyn.pi_from_eta(eta_arr[t]))

    if y_full[t] > 0:
        raw = model.dist.score(y_full[t], **call_params)
        scaled = model.gas._scaled_score(raw, call_params)
        for j, name in enumerate(model.gas.tv_names):
            s_arr[t, j] = scaled[name]

    for j, name in enumerate(model.gas.tv_names):
        omega_j = gp[f"omega_{name}"]
        ar_part = score_part = 0.0
        for l in lags:
            idx = t - l + 1
            s_past = s_arr[idx, j] if idx >= 0 else 0.0
            f_past = f_arr[idx, j] if idx >= 0 else gp[f"f0_{name}"]
            score_part += gp[f"A_{name}_{l}"] * s_past
            ar_part += gp[f"B_{name}_{l}"] * f_past
        f_arr[t + 1, j] = omega_j + score_part + ar_part


def _base_simulate_forward(model, replay: dict, t0: int, h_max: int,
                            n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """From origin t0 (state built from REAL data), draw n_draws independent
    length-h_max continuations. Returns array (n_draws, h_max)."""
    gp, pi_params, lags = replay["gp"], replay["pi_params"], model.lags
    max_lag, n_tv = model.max_lag, len(model.gas.tv_names)
    T_ext = t0 + h_max + 1
    lags_pi = list(getattr(model.pi_dyn, 'lags', None) or SEASONAL_LAGS[model.seasonal])

    out = np.zeros((n_draws, h_max))
    for d in range(n_draws):
        f_arr = np.zeros((T_ext + 1, n_tv))
        s_arr = np.zeros((T_ext, n_tv))
        eta_arr = np.zeros(T_ext)
        pi_arr = np.zeros(T_ext)
        # Same off-by-one seeding as _cov_simulate_forward/_harvey_simulate_forward:
        # f_arr must be seeded through t0+1 inclusive (the first simulated
        # step, t=t0+1, reads f_arr[t,:] as its own current state).
        f_arr[: t0 + 2] = replay["f_arr"][: t0 + 2]
        s_arr[: t0 + 1] = replay["s_arr"][: t0 + 1]
        eta_arr[: t0 + 1] = replay["eta_arr"][: t0 + 1]
        pi_arr[: t0 + 1] = replay["pi_arr"][: t0 + 1]
        y_sim = np.zeros(T_ext)

        for k, t in enumerate(range(t0 + 1, t0 + 1 + h_max)):
            call_params = {name: f_arr[t, j] for j, name in enumerate(model.gas.tv_names)}
            call_params.update({s: gp[s] for s in model.gas.static_names})

            def y_lookup(tau, _t0=t0, _y_sim=y_sim, _y_real=replay["_y_real"]):
                return _y_real[tau] if tau <= _t0 else _y_sim[tau]

            eta_prev = eta_arr[t - 1] if t - 1 >= 0 else 0.0
            omega_y = np.array([pi_params[f"omega_y_{l}"] for l in lags_pi])
            y_lags = np.array([y_lookup(t - l) if (t - l) >= 0 else 0.0 for l in lags_pi])
            eta_arr[t] = float(pi_params["omega0"] + pi_params.get("rho", 0.0) * eta_prev + np.dot(omega_y, y_lags))
            pi_arr[t] = float(model.pi_dyn.pi_from_eta(eta_arr[t]))

            y_draw = _draw_y(model, call_params, pi_arr[t], rng)
            y_sim[t] = y_draw
            out[d, k] = y_draw

            if y_draw > 0:
                raw = model.dist.score(y_draw, **call_params)
                scaled = model.gas._scaled_score(raw, call_params)
                for j, name in enumerate(model.gas.tv_names):
                    s_arr[t, j] = scaled[name]

            for j, name in enumerate(model.gas.tv_names):
                omega_j = gp[f"omega_{name}"]
                ar_part = score_part = 0.0
                for l in lags:
                    idx = t - l + 1
                    s_past = s_arr[idx, j] if idx >= 0 else 0.0
                    f_past = f_arr[idx, j] if idx >= 0 else gp[f"f0_{name}"]
                    score_part += gp[f"A_{name}_{l}"] * s_past
                    ar_part += gp[f"B_{name}_{l}"] * f_past
                f_arr[t + 1, j] = omega_j + score_part + ar_part
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ThresholdWeightedXiModel (Stage 5 step 2): replay + simulate
# ─────────────────────────────────────────────────────────────────────────────
#
# Mirrors models/threshold_weighted_xi_gas.py's own _run_filter exactly:
# phi_v/gamma_v/zeta_v are FROZEN scalars for the whole series (no
# covariates, no lag structure beyond GAS_LAG=1), and the score at t is
# gated by the threshold indicator I_t = 1{y_t >= threshold_c} -- days
# below threshold contribute no score, xi still evolves via the AR term
# alone (mean-reversion only), exactly as the fitted model already does.

def _threshxi_replay(model, theta: np.ndarray, y_full: np.ndarray,
                      phi_v: float, gamma_v: float, zeta_v: float,
                      pi_full: np.ndarray, threshold_c: float) -> dict:
    d = model._decode(theta)
    T_full = len(y_full)
    xi_arr = np.zeros(T_full + 1)
    s_arr = np.zeros(T_full)
    xi_arr[: model.GAS_LAG + 1] = d["f0"]
    for t in range(model.GAS_LAG, T_full):
        xi_t = xi_arr[t]
        if y_full[t] >= threshold_c:
            raw = model.dist.score(y_full[t], phi=phi_v, xi=xi_t, gamma=gamma_v, zeta=zeta_v)
            s_arr[t] = model._scaled_score(raw["xi"], phi_v, xi_t, gamma_v, zeta_v)
        xi_arr[t + 1] = d["omega"] + d["A"] * s_arr[t] + d["B"] * xi_t
    return {"d": d, "xi_arr": xi_arr, "s_arr": s_arr, "pi_full": pi_full,
            "phi_v": phi_v, "gamma_v": gamma_v, "zeta_v": zeta_v}


def _threshxi_simulate_forward(model, replay: dict, t0: int, h_max: int,
                                threshold_c: float, n_draws: int,
                                rng: np.random.Generator) -> np.ndarray:
    """From origin t0 (state built from REAL data), draw n_draws independent
    length-h_max continuations. Returns array (n_draws, h_max)."""
    d = replay["d"]
    phi_v, gamma_v, zeta_v = replay["phi_v"], replay["gamma_v"], replay["zeta_v"]
    pi_full = replay["pi_full"]
    out = np.zeros((n_draws, h_max))
    for draw_i in range(n_draws):
        xi_t = replay["xi_arr"][t0 + 1]
        for k in range(h_max):
            t = t0 + 1 + k
            pi_t = float(pi_full[t]) if t < len(pi_full) else float(pi_full[-1])
            call_params = {"phi": phi_v, "xi": xi_t, "gamma": gamma_v, "zeta": zeta_v}
            y_draw = _draw_y(model, call_params, pi_t, rng)
            out[draw_i, k] = y_draw
            s_t = 0.0
            if y_draw >= threshold_c:
                raw = model.dist.score(y_draw, phi=phi_v, xi=xi_t, gamma=gamma_v, zeta=zeta_v)
                s_t = model._scaled_score(raw["xi"], phi_v, xi_t, gamma_v, zeta_v)
            xi_t = d["omega"] + d["A"] * s_t + d["B"] * xi_t
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CovZAGASModel: replay + simulate
# ─────────────────────────────────────────────────────────────────────────────

def _cov_replay(model, theta: np.ndarray, y_full: np.ndarray, X_full: np.ndarray) -> dict:
    """Deterministic replay of CovZAGASModel.simulate_oos's own `_update`
    closure (models/cov_gas_model.py) over REAL y_full, returning the full
    f_arr/s_arr/eta_arr/pi_arr history (index 0..T_full) rather than only
    the OOS slice that method itself returns."""
    gas_theta, pi_theta, cov_theta = model._split_theta(theta)
    gp = model.gas.codec.decode(gas_theta)
    pi_params = {name: float(pi_theta[i]) for i, name in enumerate(model._pi_names)}
    gamma = model._decode_cov(cov_theta)

    L, max_lag, n_tv = model.lags, model.max_lag, len(model.tv_names)
    T_full = len(y_full)
    X_full = np.nan_to_num(X_full, nan=0.0)

    f_arr = np.zeros((T_full + 1, n_tv))
    s_arr = np.zeros((T_full, n_tv))
    eta_arr = np.zeros(T_full)
    pi_arr = np.zeros(T_full)
    for j, name in enumerate(model.tv_names):
        f_arr[: max_lag + 1, j] = gp[f"f0_{name}"]

    for t in range(max_lag, T_full):
        _cov_update(t, y_full, X_full, f_arr, s_arr, eta_arr, pi_arr, model, gp, pi_params, gamma, L)

    return {"f_arr": f_arr, "s_arr": s_arr, "eta_arr": eta_arr, "pi_arr": pi_arr,
            "gp": gp, "pi_params": pi_params, "gamma": gamma, "max_lag": max_lag}


def _cov_update(t, y_full, X_full, f_arr, s_arr, eta_arr, pi_arr, model, gp, pi_params, gamma, lags):
    """Exact mirror of CovZAGASModel.simulate_oos's `_update` closure."""
    call_params = {name: f_arr[t, j] for j, name in enumerate(model.tv_names)}
    call_params.update({s: gp[s] for s in model.gas.static_names})

    # eta_arr is t-indexed here; pass i=t so compute_eta's eta_hist[i-1]
    # reads eta_arr[t-1] (the previous step), matching the forecast recursion.
    eta_arr[t] = model.pi_dyn.compute_eta(
        i=t, eta_hist=eta_arr, y_full=y_full, t=t,
        params=pi_params, seasonal=model.seasonal,
    )
    pi_arr[t] = float(model.pi_dyn.pi_from_eta(eta_arr[t]))

    if y_full[t] > 0:
        raw = model.dist.score(y_full[t], **call_params)
        scaled = model.gas._scaled_score(raw, call_params)
        for j, name in enumerate(model.tv_names):
            s_arr[t, j] = scaled[name]

    x_t = X_full[t]
    for j, name in enumerate(model.tv_names):
        omega_j = gp[f"omega_{name}"]
        ar_part = score_part = 0.0
        for l in lags:
            idx = t - l + 1
            s_past = s_arr[idx, j] if idx >= 0 else 0.0
            f_past = f_arr[idx, j] if idx >= 0 else gp[f"f0_{name}"]
            score_part += gp[f"A_{name}_{l}"] * s_past
            ar_part += gp[f"B_{name}_{l}"] * f_past
        cov_term = 0.0
        if name in model.cov_target:
            g_arr = gamma.get(name, np.zeros(model.n_cov))
            cov_term = float(np.dot(g_arr, x_t))
        f_arr[t + 1, j] = omega_j + score_part + ar_part + cov_term


def _draw_y(model, call_params: dict, pi_t: float, rng: np.random.Generator) -> float:
    """Bernoulli occurrence x GB2 magnitude draw -- mirrors
    diagnostics.scoring._draw_predictive's single-draw logic."""
    if rng.uniform() >= pi_t:
        return 0.0
    try:
        if hasattr(model.dist, "ppf"):
            return float(model.dist.ppf(rng.uniform(1e-8, 1 - 1e-8), **call_params))
        return float(model.dist.rvs(1, **call_params)[0])
    except Exception:
        return 0.0


def _cov_simulate_forward(model, replay: dict, X_full: np.ndarray, t0: int, h_max: int,
                           n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """From origin t0 (state built from REAL data), draw n_draws independent
    length-h_max continuations. Returns array (n_draws, h_max): draws[:,k]
    is the simulated y at t0+1+k."""
    gp, pi_params, gamma, lags = replay["gp"], replay["pi_params"], replay["gamma"], model.lags
    max_lag, n_tv = model.max_lag, len(model.tv_names)
    T_ext = t0 + h_max + 1
    X_full = np.nan_to_num(X_full, nan=0.0)

    out = np.zeros((n_draws, h_max))
    for d in range(n_draws):
        f_arr = np.zeros((T_ext + 1, n_tv))
        s_arr = np.zeros((T_ext, n_tv))
        eta_arr = np.zeros(T_ext)
        pi_arr = np.zeros(T_ext)
        # Seed with the REAL replay's history up to t0 (need back to t0-max_lag+1
        # for the lag terms; f_arr/s_arr indices align 1:1 with the replay's).
        # f_arr must be seeded through index t0+1 INCLUSIVE (not just t0):
        # the loop's first simulated step (t=t0+1) reads f_arr[t,:] as ITS
        # OWN current state (call_params), which the real-data replay
        # already computed (replay's own update loop writes f_arr[t+1] when
        # processing t, so f_arr[t0+1] is the replay's t0-step output) --
        # seeding only through t0 leaves f_arr[t0+1] at its zero-init
        # default, silently understating the state (bug found and fixed
        # 2026-08-31 via the h=1 vs. compute_extended_oos_metrics check).
        f_arr[: t0 + 2] = replay["f_arr"][: t0 + 2]
        s_arr[: t0 + 1] = replay["s_arr"][: t0 + 1]
        eta_arr[: t0 + 1] = replay["eta_arr"][: t0 + 1]
        pi_arr[: t0 + 1] = replay["pi_arr"][: t0 + 1]
        y_sim = np.zeros(T_ext)
        y_sim[: t0 + 1] = 0.0  # only lag-1..3 of y are used by pi's eta recursion via y_full;
        # seasonal lags (365/366) reach into real history handled below.

        for k, t in enumerate(range(t0 + 1, t0 + 1 + h_max)):
            call_params = {name: f_arr[t, j] for j, name in enumerate(model.tv_names)}
            call_params.update({s: gp[s] for s in model.gas.static_names})

            # pi's AR-logistic eta needs y at t-l for l in SEASONAL_LAGS; for
            # l reaching before t0+1 (i.e. real history), read the ORIGINAL
            # real series; for l landing inside the simulated window, read
            # from y_sim (already filled by this draw's own earlier steps).
            def y_lookup(tau, _t0=t0, _y_sim=y_sim, _y_real=replay["_y_real"]):
                return _y_real[tau] if tau <= _t0 else _y_sim[tau]

            lags_pi = list(getattr(model.pi_dyn, 'lags', None) or SEASONAL_LAGS[model.seasonal])
            eta_prev = eta_arr[t - 1] if t - 1 >= 0 else 0.0
            omega_y = np.array([pi_params[f"omega_y_{l}"] for l in lags_pi])
            y_lags = np.array([y_lookup(t - l) if (t - l) >= 0 else 0.0 for l in lags_pi])
            eta_arr[t] = float(pi_params["omega0"] + pi_params.get("rho", 0.0) * eta_prev + np.dot(omega_y, y_lags))
            pi_arr[t] = float(model.pi_dyn.pi_from_eta(eta_arr[t]))

            y_draw = _draw_y(model, call_params, pi_arr[t], rng)
            y_sim[t] = y_draw
            out[d, k] = y_draw

            if y_draw > 0:
                raw = model.dist.score(y_draw, **call_params)
                scaled = model.gas._scaled_score(raw, call_params)
                for j, name in enumerate(model.tv_names):
                    s_arr[t, j] = scaled[name]

            x_t = X_full[t]
            for j, name in enumerate(model.tv_names):
                omega_j = gp[f"omega_{name}"]
                ar_part = score_part = 0.0
                for l in lags:
                    idx = t - l + 1
                    s_past = s_arr[idx, j] if idx >= 0 else 0.0
                    f_past = f_arr[idx, j] if idx >= 0 else gp[f"f0_{name}"]
                    score_part += gp[f"A_{name}_{l}"] * s_past
                    ar_part += gp[f"B_{name}_{l}"] * f_past
                cov_term = 0.0
                if name in model.cov_target:
                    g_arr = gamma.get(name, np.zeros(model.n_cov))
                    cov_term = float(np.dot(g_arr, x_t))
                f_arr[t + 1, j] = omega_j + score_part + ar_part + cov_term
    return out


# ─────────────────────────────────────────────────────────────────────────────
# HarveyZAGASModel: replay + simulate
# ─────────────────────────────────────────────────────────────────────────────

def _harvey_replay(model, theta: np.ndarray, y_full: np.ndarray,
                    Xl_full: np.ndarray, Xs_full: np.ndarray) -> dict:
    st_theta, pi_theta, hv_theta = model._split_theta(theta)
    sp = model._decode_static(st_theta)
    pip = model._decode_pi(pi_theta)
    hvd = model._decode_harvey(hv_theta)
    n_tv, max_lag = len(model.tv_names), model.max_lag
    T_full = len(y_full)
    Xl_full = np.nan_to_num(Xl_full, nan=0.0)
    Xs_full = np.nan_to_num(Xs_full, nan=0.0)

    L_arr = np.zeros((T_full + 1, n_tv))
    S_arr = np.zeros((T_full + 1, n_tv))
    f_arr = np.zeros((T_full, n_tv))
    s_arr = np.zeros((T_full, n_tv))
    eta_arr = np.zeros(T_full)
    pi_arr = np.zeros(T_full)
    m = model.n_harmonics
    # Trigonometric seasonal harmonic state (2026-09-11 fix): this replay
    # previously omitted the seasonal component entirely whenever
    # n_harmonics>0 (Stage 4 / Stage 4S), so it silently diverged from the
    # model's own true filtered path -- harmlessly for a few steps, but
    # over T_full ~ 4000+ days the missing correction compounds, especially
    # when phi_L sits near its unit-root boundary (this project's
    # recurring failure mode), producing NaN/Inf states and hence NaN
    # multistep RMSE/CRPS. Mirrors HarveyZAGASModel._run_filter's own
    # seasonal recursion exactly.
    Gm_arr = np.zeros((T_full + 1, n_tv, m)) if m > 0 else None
    GmStar_arr = np.zeros((T_full + 1, n_tv, m)) if m > 0 else None
    for j, name in enumerate(model.tv_names):
        if name in hvd:
            L_arr[0, j] = hvd[name]["L0"]
            S_arr[0, j] = hvd[name]["S0"]
            if m > 0:
                Gm_arr[max_lag, j, :] = hvd[name]["seas0"]
                GmStar_arr[max_lag, j, :] = hvd[name]["seasstar0"]

    for t in range(max_lag, T_full):
        _harvey_step_real(t, y_full, L_arr, S_arr, f_arr, s_arr, eta_arr, pi_arr,
                           model, sp, pip, hvd, Xl_full, Xs_full, Gm_arr, GmStar_arr)

    out = {"L_arr": L_arr, "S_arr": S_arr, "f_arr": f_arr, "s_arr": s_arr,
           "eta_arr": eta_arr, "pi_arr": pi_arr, "sp": sp, "pip": pip, "hvd": hvd}
    if m > 0:
        out["Gm_arr"] = Gm_arr
        out["GmStar_arr"] = GmStar_arr
    return out


def _harvey_step_real(t, y_full, L_arr, S_arr, f_arr, s_arr, eta_arr, pi_arr,
                       model, sp, pip, hvd, Xl_full, Xs_full, Gm_arr=None, GmStar_arr=None):
    m = model.n_harmonics
    for j, name in enumerate(model.tv_names):
        if name in hvd:
            f_arr[t, j] = hvd[name]["omega"] + L_arr[t, j] + S_arr[t, j]
            if m > 0:
                f_arr[t, j] += float(np.sum(Gm_arr[t, j, :]))
        else:
            f_arr[t, j] = sp.get(name, 0.0)
    call = {name: f_arr[t, j] for j, name in enumerate(model.tv_names)}
    call.update(sp)

    # eta_arr is t-indexed here; pass i=t so compute_eta's eta_hist[i-1]
    # reads eta_arr[t-1] (the previous step), matching the forecast recursion.
    eta_arr[t] = model.pi_dyn.compute_eta(
        i=t, eta_hist=eta_arr, y_full=y_full, t=t,
        params=pip, seasonal=model.seasonal,
    )
    pi_arr[t] = float(model.pi_dyn.pi_from_eta(eta_arr[t]))

    if y_full[t] > 0:
        raw = model.dist.score(y_full[t], **call)
        scaled = model._scaled_score(raw, call)
        for j, name in enumerate(model.tv_names):
            s_arr[t, j] = scaled[name]

    xl_t, xs_t = Xl_full[t], Xs_full[t]
    for j, name in enumerate(model.tv_names):
        if name in hvd:
            hj = hvd[name]
            L_arr[t + 1, j] = hj["phi_L"] * L_arr[t, j] + hj["kappa_L"] * s_arr[t, j] + (
                float(np.dot(hj["Gamma_L"], xl_t)) if model.n_long > 0 else 0.0)
            S_arr[t + 1, j] = hj["phi_S"] * S_arr[t, j] + hj["kappa_S"] * s_arr[t, j] + (
                float(np.dot(hj["Gamma_S"], xs_t)) if model.n_short > 0 else 0.0)
            if m > 0:
                kg = hj["kappa_gamma"]
                psi = hj["psi_gamma"] if hj["psi_gamma"] is not None else 1.0
                gm, gms = Gm_arr[t, j, :], GmStar_arr[t, j, :]
                Gm_arr[t + 1, j, :] = psi * (gm * model._cos_h + gms * model._sin_h) + kg * s_arr[t, j]
                GmStar_arr[t + 1, j, :] = psi * (-gm * model._sin_h + gms * model._cos_h) + kg * s_arr[t, j]
        else:
            L_arr[t + 1, j] = L_arr[t, j]
            S_arr[t + 1, j] = S_arr[t, j]
            if m > 0:
                Gm_arr[t + 1, j, :] = Gm_arr[t, j, :]
                GmStar_arr[t + 1, j, :] = GmStar_arr[t, j, :]


def _harvey_simulate_forward(model, replay: dict, y_real: np.ndarray, Xl_full: np.ndarray,
                              Xs_full: np.ndarray, t0: int, h_max: int, n_draws: int,
                              rng: np.random.Generator) -> np.ndarray:
    n_tv = len(model.tv_names)
    sp, pip, hvd = replay["sp"], replay["pip"], replay["hvd"]
    T_ext = t0 + h_max + 1
    Xl_full = np.nan_to_num(Xl_full, nan=0.0)
    Xs_full = np.nan_to_num(Xs_full, nan=0.0)
    lags_pi = list(getattr(model.pi_dyn, 'lags', None) or SEASONAL_LAGS[model.seasonal])
    m = model.n_harmonics

    out = np.zeros((n_draws, h_max))
    for d in range(n_draws):
        L_arr = np.zeros((T_ext + 1, n_tv)); S_arr = np.zeros((T_ext + 1, n_tv))
        f_arr = np.zeros((T_ext, n_tv)); s_arr = np.zeros((T_ext, n_tv))
        eta_arr = np.zeros(T_ext); pi_arr = np.zeros(T_ext)
        # Same off-by-one fix as _cov_simulate_forward: L_arr/S_arr must be
        # seeded through t0+1 inclusive, since the first simulated step
        # (t=t0+1) reads L_arr[t,:]/S_arr[t,:] as its OWN current state.
        L_arr[: t0 + 2] = replay["L_arr"][: t0 + 2]
        S_arr[: t0 + 2] = replay["S_arr"][: t0 + 2]
        eta_arr[: t0 + 1] = replay["eta_arr"][: t0 + 1]
        y_sim = np.zeros(T_ext)
        # Seasonal harmonic state (2026-09-11 fix, see _harvey_replay's
        # identical comment): seeded from the replay the same way L_arr/
        # S_arr are, and carried forward each simulated step below.
        if m > 0:
            Gm_arr = np.zeros((T_ext + 1, n_tv, m)); GmStar_arr = np.zeros((T_ext + 1, n_tv, m))
            Gm_arr[: t0 + 2] = replay["Gm_arr"][: t0 + 2]
            GmStar_arr[: t0 + 2] = replay["GmStar_arr"][: t0 + 2]

        for k, t in enumerate(range(t0 + 1, t0 + 1 + h_max)):
            for j, name in enumerate(model.tv_names):
                if name in hvd:
                    f_arr[t, j] = hvd[name]["omega"] + L_arr[t, j] + S_arr[t, j]
                    if m > 0:
                        f_arr[t, j] += float(np.sum(Gm_arr[t, j, :]))
                else:
                    f_arr[t, j] = sp.get(name, 0.0)
            call = {name: f_arr[t, j] for j, name in enumerate(model.tv_names)}
            call.update(sp)

            def y_lookup(tau, _t0=t0, _y_sim=y_sim, _y_real=y_real):
                return _y_real[tau] if tau <= _t0 else _y_sim[tau]

            eta_prev = eta_arr[t - 1] if t - 1 >= 0 else 0.0
            omega_y = np.array([pip[f"omega_y_{l}"] for l in lags_pi])
            y_lags = np.array([y_lookup(t - l) if (t - l) >= 0 else 0.0 for l in lags_pi])
            eta_arr[t] = float(pip["omega0"] + pip["rho"] * eta_prev + np.dot(omega_y, y_lags))
            pi_arr[t] = float(model.pi_dyn.pi_from_eta(eta_arr[t]))

            y_draw = _draw_y(model, call, pi_arr[t], rng)
            y_sim[t] = y_draw
            out[d, k] = y_draw

            if y_draw > 0:
                raw = model.dist.score(y_draw, **call)
                scaled = model._scaled_score(raw, call)
                for j, name in enumerate(model.tv_names):
                    s_arr[t, j] = scaled[name]

            xl_t, xs_t = Xl_full[t], Xs_full[t]
            for j, name in enumerate(model.tv_names):
                if name in hvd:
                    hj = hvd[name]
                    L_arr[t + 1, j] = hj["phi_L"] * L_arr[t, j] + hj["kappa_L"] * s_arr[t, j] + (
                        float(np.dot(hj["Gamma_L"], xl_t)) if model.n_long > 0 else 0.0)
                    S_arr[t + 1, j] = hj["phi_S"] * S_arr[t, j] + hj["kappa_S"] * s_arr[t, j] + (
                        float(np.dot(hj["Gamma_S"], xs_t)) if model.n_short > 0 else 0.0)
                    if m > 0:
                        kg = hj["kappa_gamma"]
                        psi = hj["psi_gamma"] if hj["psi_gamma"] is not None else 1.0
                        gm, gms = Gm_arr[t, j, :], GmStar_arr[t, j, :]
                        Gm_arr[t + 1, j, :] = psi * (gm * model._cos_h + gms * model._sin_h) + kg * s_arr[t, j]
                        GmStar_arr[t + 1, j, :] = psi * (-gm * model._sin_h + gms * model._cos_h) + kg * s_arr[t, j]
                else:
                    L_arr[t + 1, j] = L_arr[t, j]
                    S_arr[t + 1, j] = S_arr[t, j]
                    if m > 0:
                        Gm_arr[t + 1, j, :] = Gm_arr[t, j, :]
                        GmStar_arr[t + 1, j, :] = GmStar_arr[t, j, :]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def multistep_metrics(
    model,
    theta: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    covariates: dict,
    origins: Sequence[int],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_draws: int = 200,
    seed: int = 0,
    twcrps_levels: Sequence[float] = MULTISTEP_TWCRPS_LEVELS,
    exceedance_levels: Sequence[float] = (0.95, 0.98),
) -> pd.DataFrame:
    """
    RMSE/CRPS/twCRPS at each horizon in `horizons`, Monte Carlo (n_draws
    paths per origin), averaged over `origins` (indices into y_test; must
    all satisfy origin + max(horizons) < len(y_test)).

    covariates: {"X_train":..., "X_test":...} for a Cov-style model,
    {"X_long_train":..., "X_long_test":..., "X_short_train":...,
    "X_short_test":...} for a Harvey-style model, {"threshold_c":...,
    "phi_v":..., "gamma_v":..., "zeta_v":..., "pi_v":...} for a Stage 5
    ThresholdWeightedXiModel (pi_v may be a scalar or an array aligned to
    y_train+y_test, exactly as that model's own simulate_oos accepts it),
    or {} / None for a plain ZAGASModel with no covariates at all --
    dispatch is by which keys are present.

    twcrps_levels: threshold-weighted CRPS levels (Gneiting & Ranjan 2011,
    same definition as diagnostics.scoring.twcrps_mc/
    compute_extended_oos_metrics). The threshold for each level is the
    wet-day quantile of y_test, computed once and shared across every
    horizon -- pass an empty sequence to skip twCRPS entirely (columns
    omitted from the returned DataFrame).
    """
    from diagnostics.exceedance import exceedance_from_draws
    rng = np.random.default_rng(seed)
    y_full = np.concatenate([y_train, y_test])
    T_train = len(y_train)
    h_max = max(horizons)

    covariates = covariates or {}
    is_harvey = "X_long_train" in covariates
    is_cov = (not is_harvey) and "X_train" in covariates
    is_threshxi = (not is_harvey) and (not is_cov) and "threshold_c" in covariates
    if is_harvey:
        Xl_full = np.vstack([covariates["X_long_train"], covariates["X_long_test"]])
        Xs_full = np.vstack([covariates["X_short_train"], covariates["X_short_test"]])
        replay = _harvey_replay(model, theta, y_full, Xl_full, Xs_full)
    elif is_cov:
        X_full = np.vstack([covariates["X_train"], covariates["X_test"]])
        replay = _cov_replay(model, theta, y_full, X_full)
        replay["_y_real"] = y_full
    elif is_threshxi:
        from models.threshold_weighted_xi_gas import _to_pi_array
        threshold_c = covariates["threshold_c"]
        pi_full = _to_pi_array(covariates["pi_v"], len(y_full))
        replay = _threshxi_replay(model, theta, y_full, covariates["phi_v"],
                                   covariates["gamma_v"], covariates["zeta_v"],
                                   pi_full, threshold_c)
    else:
        replay = _base_replay(model, theta, y_full)

    per_horizon_draws: Dict[int, List[np.ndarray]] = {h: [] for h in horizons}
    per_horizon_actual: Dict[int, List[float]] = {h: [] for h in horizons}

    for origin in origins:
        t0 = T_train + origin
        if t0 + h_max >= len(y_full):
            continue
        if is_harvey:
            draws = _harvey_simulate_forward(model, replay, y_full, Xl_full, Xs_full,
                                              t0, h_max, n_draws, rng)
        elif is_cov:
            draws = _cov_simulate_forward(model, replay, X_full, t0, h_max, n_draws, rng)
        elif is_threshxi:
            draws = _threshxi_simulate_forward(model, replay, t0, h_max, threshold_c, n_draws, rng)
        else:
            draws = _base_simulate_forward(model, replay, t0, h_max, n_draws, rng)
        for h in horizons:
            per_horizon_draws[h].append(draws[:, h - 1])
            per_horizon_actual[h].append(y_full[t0 + h])

    wet_y_test = y_test[y_test > 0]
    twcrps_thresholds = {
        lv: float(np.quantile(wet_y_test, lv)) if len(wet_y_test) > 0 else 0.0
        for lv in twcrps_levels
    }

    rows = []
    for h in horizons:
        draws_list = per_horizon_draws[h]
        actuals = per_horizon_actual[h]
        if not draws_list:
            row = {"horizon": h, "n_origins": 0, "rmse": np.nan, "crps": np.nan}
            row.update({f"twcrps_{int(lv*100)}": np.nan for lv in twcrps_levels})
            for lv in exceedance_levels:
                k = int(lv * 100)
                row[f"exc_{k}"] = np.nan; row[f"kup{k}_p"] = np.nan; row[f"kup{k}_reject"] = np.nan
            rows.append(row)
            continue
        sq_err = [(float(np.mean(d)) - a) ** 2 for d, a in zip(draws_list, actuals)]
        crps_vals = [crps_mc(d, a) for d, a in zip(draws_list, actuals)]
        row = {
            "horizon": h, "n_origins": len(draws_list),
            "rmse": float(np.sqrt(np.mean(sq_err))),
            "crps": float(np.mean(crps_vals)),
        }
        for lv, thr in twcrps_thresholds.items():
            tw_vals = [twcrps_mc(d, a, thr) for d, a in zip(draws_list, actuals)]
            row[f"twcrps_{int(lv*100)}"] = float(np.mean(tw_vals))
        for lv in exceedance_levels:
            k = int(lv * 100)
            e = exceedance_from_draws(draws_list, actuals, lv)
            row[f"exc_{k}"] = e["rate"]
            row[f"kup{k}_p"] = e["kupiec_p"]
            row[f"kup{k}_reject"] = (e["kupiec_p"] < 0.05) if np.isfinite(e["kupiec_p"]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
