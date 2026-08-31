"""
Genuine k-day-ahead (blind) forecast evaluation for phi-only ZA-GAS models.

=============================================================================
WHY THIS MODULE EXISTS
=============================================================================

Every OOS metric elsewhere in this repository (RMSE in stage_winners.json,
CRPS/quantile-scores/exceedance-ratios in generate_report_extended.py) is a
*walk-forward 1-step-ahead* evaluation: at each test day t, the recursion is
advanced using the TRUE observed y up through t-1 (simulation.simulator.
simulate_oos), and the resulting predictive distribution is scored against
the realized y_t. Repeated across ~700 test days, this gives a large sample
of one-step-ahead predictive distributions -- it never requires forecasting
blindly more than one day out, because the "day before" is always real data.

This module answers a different question: from a single fixed origin t0,
what would the model have forecast for y_{t0+h}, h days later, WITHOUT
being handed the true y's in between? This requires propagating the fitted
recursion forward using only simulated (unknown) data for h-1 days.

=============================================================================
METHOD (Monte Carlo forward simulation)
=============================================================================

For each Monte Carlo draw m and each forecast origin t0:
    1. phi_{t0+1} and pi_{t0+1} are already fully determined by REAL data
       known at t0 (GAS lags are {1,2,3}; pi lags are {1,365,366} for daily
       seasonality -- see constants.SEASONAL_LAGS -- all <= true history at
       t0). No simulation is needed for h=1; it exactly reproduces the
       walk-forward 1-step-ahead numbers used elsewhere in the repo.
    2. For h >= 2: draw y_{t0+h-1}^(m) ~ ZA-GB2(pi_{t0+h-1}, phi_{t0+h-1})
       from the model's own predictive distribution, feed it back into
       - the GAS score recursion for phi (models/gas_filter.py eq. 5,
         models/za_gas_model.py lines ~533-562: score set to 0 when y=0,
         scaled by the diagonal-inverse-Fisher constant -- see below), and
       - the AR-logistic recursion for eta/pi (pi_dynamics/ar_logistic.py),
       to obtain phi_{t0+h}, pi_{t0+h}.
    3. The point forecast at each step is Rao-Blackwellized rather than the
       raw simulated y: E[y_{t0+h} | path] = pi_{t0+h} * E[GB2 | phi_{t0+h}]
       using the closed-form GB2 mean (distributions/gb2_log_link.py
       `raw_moment`), then averaged over the M simulated paths. This uses
       the same number of simulations but has strictly lower variance than
       averaging the raw simulated y's directly (standard Rao-Blackwell
       argument), consistent with this repo's "prefer analytical over
       numerical" rule wherever an analytical piece is available.

Only lags {1,2,3} (GAS, phi) and {1} (pi, short lag) can ever reference a
simulated value -- pi's seasonal lags {365,366} are always >= 365 days back,
far outside any forecast horizon used here (max 90), so they are always
looked up directly from real history. This keeps the required simulation
state to a 3-step rolling buffer for (phi, scaled-score) and a 1-step
buffer for y, which is why the whole engine vectorizes trivially over
(n_origins, n_draws) numpy arrays with a plain Python loop over h.

=============================================================================
SCOPE
=============================================================================

This module supports phi-only ZAGASModel instances (GB2LogLinkPhiOnly +
ARLogisticPiDynamics) with an arbitrary phi lag set -- i.e. exactly the
"stage1_phi_short_diagfi" / "stage1_phi_seasonal_*" / "stage2_phi_*" family.
It does not (yet) support phi+xi models, where xi's own score recursion
would also need to be propagated; raises NotImplementedError rather than
silently producing wrong numbers for that case.

The diagonal-inverse-Fisher scaling divisor for phi,
    I_phi = p^2 * a * b / (a + b + 1)   (distributions/gb2_log_link.py
    fisher_info_diag, log-space FI for the log-link GB2)
does not depend on phi/sigma at all when xi, gamma, zeta are static (true
for every phi-only model in this repo) -- it is a single fixed constant
for the whole recursion, computed once here exactly as the fitted model
computed it during estimation and OOS filtering.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import beta as beta_fn

from constants import SEASONAL_LAGS


def simulate_horizon_forecasts(
    model,
    theta: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    horizons: Sequence[int] = (1, 2, 5, 10, 30, 60, 90),
    n_draws: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Blind k-day-ahead Monte Carlo forecast evaluation, wet/dry RMSE split.

    Returns a DataFrame with one row per horizon in `horizons`, columns:
        horizon, n_origins, n_wet, n_dry,
        rmse_all, rmse_wet, rmse_dry, mean_forecast_wet, mean_forecast_dry
    """
    if len(model.gas.tv_names) != 1 or model.gas.tv_names[0] != "phi":
        raise NotImplementedError(
            "simulate_horizon_forecasts only supports phi-only ZAGASModel "
            f"instances; got tv_names={model.gas.tv_names}"
        )

    max_h = max(horizons)
    rng = np.random.default_rng(seed)

    # ---- Replay the REAL filtered recursion over the whole series ---------
    # This is a pure "replay" (model.filter is deterministic given theta and
    # y) -- no optimization, identical numbers to what stage_winners.json /
    # paths.npz already contain for the OOS window.
    y_full = np.concatenate([y_train, y_test])
    T_train = len(y_train)
    T_full = len(y_full)

    real = model.filter(theta, y_full)
    eff = real["eff_start"]
    f_real = real["f_arr"][:, 0]
    s_real = real["s_arr"][:, 0]
    eta_real = real["eta"]
    static = real["static"]
    xi_s, gamma_s, zeta_s = static["xi"], static["gamma"], static["zeta"]

    a = float(np.exp(xi_s))
    b = float(np.exp(zeta_s))
    p = float(np.exp(-gamma_s))

    # GB2 mean:  E[Y] = sigma * B(a + 1/p, b - 1/p) / B(a, b)   (raw_moment k=1)
    if not (b > 1.0 / p):
        raise ValueError(
            f"GB2 mean does not exist for fitted static params (a={a}, b={b}, "
            f"p={p}): requires b > 1/p."
        )
    mean_gb2_const = float(beta_fn(a + 1.0 / p, b - 1.0 / p) / beta_fn(a, b))

    # Diagonal-inverse-Fisher scaling divisor for phi -- fixed constant
    # (see module docstring). Only "diagonal_inverse_fisher" / "inverse_fisher"
    # (identical for a single tv param) and "unit" are supported scalings.
    if model.gas.scaling == "unit":
        I_phi = 1.0
    else:
        I_phi = float(model.dist.fisher_info_diag(
            phi=0.0, xi=xi_s, gamma=gamma_s, zeta=zeta_s
        )["phi"])
        if not (I_phi > 0 and np.isfinite(I_phi)):
            raise ValueError(f"Non-positive/non-finite Fisher info I_phi={I_phi}")

    # ---- GAS (phi) coefficients --------------------------------------------
    lags_phi = list(model.lags)                      # e.g. [1, 2, 3]
    gp = {}  # decode theta once via the model's own codec for exact fidelity
    gas_theta, pi_theta_flat = model._split_theta(theta)
    gp_full = model.gas.codec.decode(gas_theta)
    omega_phi = gp_full["omega_phi"]
    A_phi = {l: gp_full[f"A_phi_{l}"] for l in lags_phi}
    B_phi = {l: gp_full[f"B_phi_{l}"] for l in lags_phi}

    # ---- Pi (AR-logistic) coefficients -------------------------------------
    pi_params = model._decode_pi(pi_theta_flat)
    lags_pi = SEASONAL_LAGS[model.seasonal]           # e.g. [1, 365, 366]
    omega0 = pi_params["omega0"]
    rho = pi_params["rho"]
    omega_y = {l: pi_params[f"omega_y_{l}"] for l in lags_pi}
    short_pi_lags = [l for l in lags_pi if l <= max_h]      # can reference simulated y
    long_pi_lags = [l for l in lags_pi if l > max_h]        # always real history

    # ---- Forecast origins ---------------------------------------------------
    # t0 (absolute index into y_full) ranges over the test period such that
    # every target t0+h, h in horizons, is also a real test-set observation.
    t0_lo = T_train - 1
    t0_hi = T_full - 1 - max_h
    if t0_hi < t0_lo:
        raise ValueError("Test set too short for the requested max horizon.")
    t0 = np.arange(t0_lo, t0_hi + 1)                  # shape (n_starts,)
    n_starts = len(t0)
    i0 = t0 - eff                                     # index into f_real/s_real/eta_real
    if np.any(i0 - (max(lags_phi) - 1) < 0):
        raise ValueError("Forecast origins reach before the model's warm-up window.")

    M = n_draws
    shape = (n_starts, M)

    # ---- Rolling buffers, seeded from REAL filtered history ----------------
    # buffer[...][-1] = most recent (time t0), buffer[...][-3] = t0-2, etc.
    max_lag_phi = max(lags_phi)
    f_buf = np.stack(
        [np.broadcast_to(f_real[i0 - (max_lag_phi - 1 - k)][:, None], shape) for k in range(max_lag_phi)],
        axis=0,
    ).copy()  # shape (max_lag_phi, n_starts, M), oldest first
    s_buf = np.stack(
        [np.broadcast_to(s_real[i0 - (max_lag_phi - 1 - k)][:, None], shape) for k in range(max_lag_phi)],
        axis=0,
    ).copy()
    eta_prev = np.broadcast_to(eta_real[i0][:, None], shape).copy()
    y_prev = np.broadcast_to(y_full[t0][:, None], shape).copy()  # y_{t0}, used as pi's lag-1 term

    forecasts = {h: np.empty(n_starts) for h in horizons}
    y_draws_at_h = {}  # populated for h in `horizons` -- used for CRPS below
    pi_forecast_at_h = {}  # Rao-Blackwellized E[pi_{t0+h} | F_t0] -- used for Brier score below

    for h in range(1, max_h + 1):
        # ---- eta_{t0+h} / pi_{t0+h} -----------------------------------------
        eta_new = np.full(shape, omega0, dtype=float) + rho * eta_prev
        for l in short_pi_lags:
            # y_{t0+h-l}: simulated if l < h (already in y_prev when l == 1
            # since only lag 1 is ever "short" for h up to 90 << 365), real
            # if l >= h.
            if l == 1:
                eta_new += omega_y[l] * y_prev
            else:  # pragma: no cover -- SEASONAL_LAGS has no short lag other than 1
                idx = t0 + h - l
                eta_new += omega_y[l] * y_full[idx]
        for l in long_pi_lags:
            idx = t0 + h - l  # always real: l > max_h ensures idx < t0
            eta_new += omega_y[l] * y_full[idx][:, None]
        pi_new = 1.0 / (1.0 + np.exp(-eta_new))

        # ---- phi_{t0+h} -------------------------------------------------------
        f_new = np.full(shape, omega_phi, dtype=float)
        for l in lags_phi:
            f_new += A_phi[l] * s_buf[max_lag_phi - l] + B_phi[l] * f_buf[max_lag_phi - l]
        sigma_new = np.exp(f_new)

        # ---- Rao-Blackwellized point forecast ----------------------------------
        point_forecast = pi_new * mean_gb2_const * sigma_new

        if h in forecasts:
            forecasts[h] = point_forecast.mean(axis=1)
            # E[pi_{t0+h} | F_t0]: pi_{t0+h} is itself a random variable given
            # only information at t0 (h>1), so the forecast occurrence
            # probability used for the Brier score is its Monte Carlo mean,
            # exactly the same Rao-Blackwellization principle as point_forecast.
            pi_forecast_at_h[h] = pi_new.mean(axis=1)

        # ---- Simulate y_{t0+h} ~ ZA-GB2(pi_new, phi_new) ------------------
        # Needed both (a) as the raw MC sample for CRPS at this horizon (if
        # requested -- these draws ARE draws from the true mixture predictive
        # distribution of y_{t0+h} | F_t0, since each of the M paths is an
        # independent realization of the whole recursion) and (b) to keep
        # propagating the path forward to the next horizon.
        occ = rng.random(shape) < pi_new
        U_beta = rng.beta(a, b, size=shape)
        y_pos = sigma_new * (U_beta / (1.0 - U_beta)) ** (1.0 / p)
        y_draw = np.where(occ, y_pos, 0.0)

        if h in forecasts:
            y_draws_at_h[h] = y_draw

        if h == max_h:
            break  # no need to propagate the recursion further

        # ---- score at (y_draw, f_new), 0 when y_draw == 0 (per gas_filter.py) --
        z = np.where(occ, (np.maximum(y_draw, 1e-300) / sigma_new) ** p, 0.0)
        raw_score_phi = p * ((a + b) * z / (1.0 + z) - a)
        s_new = np.where(occ, raw_score_phi / I_phi, 0.0)

        # ---- roll buffers ----------------------------------------------------
        f_buf = np.concatenate([f_buf[1:], f_new[None, ...]], axis=0)
        s_buf = np.concatenate([s_buf[1:], s_new[None, ...]], axis=0)
        eta_prev = eta_new
        y_prev = y_draw

    # ---- Evaluate against realized y, split wet/dry ---------------------------
    # CRPS reuses the exact energy-score estimator already used for the
    # existing 1-step-ahead OOS metrics (diagnostics/scoring.py::crps_mc),
    # applied to the M simulated draws at this horizon instead of M draws
    # from a single ZA-GB2(pi_t, phi_t) -- the only thing that changes is
    # which predictive distribution the draws come from.
    from diagnostics.scoring import crps_mc

    rows = []
    for h in horizons:
        actual = y_full[t0 + h]
        pred = forecasts[h]
        draws_h = y_draws_at_h[h]  # shape (n_starts, M)
        pi_h = pi_forecast_at_h[h]
        wet = actual > 0
        dry = ~wet
        err_all = pred - actual

        crps_per_origin = np.array([
            crps_mc(draws_h[i], actual[i]) for i in range(n_starts)
        ])

        # Occurrence Brier score. Event = "rain occurs", forecast probability
        # = pi_h = E[pi_{t0+h}|F_t0]. On wet days the occurrence indicator is
        # 1, so BS = (pi_h - 1)^2 = (1 - pi_h)^2. On dry days the occurrence
        # indicator is 0 -- but scoring the SAME "did it rain" forecast pi_h
        # against a 0 outcome, (pi_h - 0)^2, would silently reuse pi_h as if
        # it were the forecast of the event that actually happened; the
        # correct forecast probability of the event that occurred (no rain)
        # is 1 - pi_h, scored against the "no rain occurred" indicator = 1:
        # BS = ((1 - pi_h) - 1)^2 = pi_h^2. Both forms are the textbook
        # Brier score (prob_forecast - outcome_indicator)^2, just applied to
        # the complementary event on dry days -- consistent with
        # diagnostics/scoring.py::brier_score's (prob_exceed - I(...))^2
        # convention.
        brier_wet = float(np.mean((1.0 - pi_h[wet]) ** 2)) if wet.any() else float("nan")
        brier_dry = float(np.mean(pi_h[dry] ** 2)) if dry.any() else float("nan")

        rows.append({
            "horizon": h,
            "n_origins": n_starts,
            "n_wet": int(wet.sum()),
            "n_dry": int(dry.sum()),
            "rmse_all": float(np.sqrt(np.mean(err_all ** 2))),
            "rmse_wet": float(np.sqrt(np.mean((pred[wet] - actual[wet]) ** 2))) if wet.any() else float("nan"),
            "rmse_dry": float(np.sqrt(np.mean((pred[dry] - actual[dry]) ** 2))) if dry.any() else float("nan"),
            "crps_all": float(np.mean(crps_per_origin)),
            "crps_wet": float(np.mean(crps_per_origin[wet])) if wet.any() else float("nan"),
            "crps_dry": float(np.mean(crps_per_origin[dry])) if dry.any() else float("nan"),
            "mean_forecast_wet": float(pred[wet].mean()) if wet.any() else float("nan"),
            "mean_forecast_dry": float(pred[dry].mean()) if dry.any() else float("nan"),
            "brier_wet": brier_wet,
            "brier_dry": brier_dry,
        })

    return pd.DataFrame(rows)
