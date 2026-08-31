"""
Rolling-origin, blind k-day-ahead evaluation harness shared by every ML
baseline (naive/AR, AR+Aux, XGBoost, LSTM) in this package.

=============================================================================
WHY THIS MODULE EXISTS
=============================================================================

daily-ML.ipynb (the source notebook this package ports) evaluates each
baseline with a SINGLE recursive forecast trajectory starting at the first
test day, walked forward the whole test-set length, then scored by taking
CUMULATIVE PREFIXES y_true[:h] vs preds[:h] at each h -- one realization,
growing window.

This harness instead reuses the SAME rolling-origin protocol used for the
score-driven model in diagnostics/horizon_forecast.py: many forecast
origins t0 spread across the test period (the identical origin set, so
results are directly comparable point-for-point), each producing an
independent blind h-day-ahead forecast, RMSE aggregated across all origins
at each fixed h, split by whether the realized target y_{t0+h} is zero
(dry) or positive (wet). This is what was requested to match the
already-built Monte Carlo evaluation of the score-driven model.

Because every model here is a deterministic point-forecast model (not a
predictive distribution), there is no Monte Carlo sampling step -- the
"forecast" at each step is just the model's own point prediction, fed back
in as the assumed-true value for future lag references (recursive/iterated
forecasting), exactly as daily-ML.ipynb already does internally, just
repeated from many origins instead of one.

=============================================================================
LAG STRUCTURE
=============================================================================

Every model here uses only two kinds of lags relative to the forecast
origin t0:
  - "short" lags (<= max horizon 90): may need a value from earlier in the
    SAME forecast trajectory (the model's own prior prediction at this
    origin) once h exceeds that lag.
  - "long" lags (> max horizon 90, e.g. 364-367): always resolve to real,
    already-observed history, since h <= 90 < 364 for every horizon used.

This mirrors exactly the buffer design in diagnostics/horizon_forecast.py.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


def crps_normal(mu: np.ndarray, sigma: float, y: np.ndarray) -> np.ndarray:
    """
    Closed-form CRPS for a Normal(mu, sigma^2) predictive distribution
    (Gneiting & Raftery 2007, eq. 21):

        CRPS = sigma * [ z(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ],  z = (y-mu)/sigma

    Used for the AR baseline's CRPS, under the assumption (per the ridge
    regression's implicit iid-Gaussian-residual model) that its predictive
    distribution at every origin/horizon is Normal(mu=point forecast,
    sigma^2=constant, estimated from in-sample training residuals -- see
    baselines/naive_ar.py::fit_with_sigma).
    """
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def rolling_origins(T_train: int, T_full: int, max_h: int) -> np.ndarray:
    """Same origin set used by diagnostics/horizon_forecast.py, for exact
    comparability with the score-driven model's k-day-ahead results."""
    t0_lo = T_train - 1
    t0_hi = T_full - 1 - max_h
    if t0_hi < t0_lo:
        raise ValueError("Test set too short for the requested max horizon.")
    return np.arange(t0_lo, t0_hi + 1)


def month_running_counts(y_full: np.ndarray, dates_full: pd.DatetimeIndex):
    """
    Vectorized, causal "day-in-month position" and "cumulative nonzero-day
    count within the current calendar month, inclusive of this day" for
    every day in the series -- the exact quantities daily-ML.ipynb's AR+Aux
    feature (`x_aux`) is built from (see baselines/ar_aux.py).
    """
    df = pd.DataFrame({"y": y_full}, index=dates_full)
    key = [dates_full.year.to_numpy(), dates_full.month.to_numpy()]
    grp = df.groupby(key)
    day_in_month = grp.cumcount().to_numpy() + 1
    nonzero_cum = grp["y"].transform(lambda s: (s > 0).cumsum()).to_numpy()
    month_id = (dates_full.year.to_numpy() * 12 + dates_full.month.to_numpy())
    return day_in_month, nonzero_cum, month_id


def recursive_horizon_forecasts(
    y_full: np.ndarray,
    T_train: int,
    horizons: Sequence[int],
    max_short_lag: int,
    predict_step: Callable[[np.ndarray, int, np.ndarray], np.ndarray],
    seed_buffer: np.ndarray | None = None,
):
    """
    Generic rolling-origin recursive forecast -- the shared core loop behind
    recursive_horizon_rmse. Exposed separately so callers that need the raw
    per-origin point forecasts (e.g. Gaussian-error CRPS for the AR model,
    which needs mu = the point forecast at every origin, not just the
    aggregated RMSE) do not have to duplicate the recursion.

    Parameters
    ----------
    y_full, T_train, horizons, max_short_lag, predict_step, seed_buffer :
        see recursive_horizon_rmse.

    Returns
    -------
    (t0, forecasts) : origin array (n_starts,), and {h: yhat array (n_starts,)}
    """
    T_full = len(y_full)
    max_h = max(horizons)
    t0 = rolling_origins(T_train, T_full, max_h)

    if seed_buffer is not None:
        buf = seed_buffer.copy()
    else:
        buf = np.stack(
            [y_full[t0 - (max_short_lag - 1 - k)] for k in range(max_short_lag)],
            axis=0,
        ).astype(np.float64)

    forecasts = {}
    for h in range(1, max_h + 1):
        yhat = predict_step(buf, h, t0)
        if h in horizons:
            forecasts[h] = yhat
        if h == max_h:
            break
        buf = np.concatenate([buf[1:], yhat[None, :]], axis=0)

    return t0, forecasts


def recursive_horizon_rmse(
    y_full: np.ndarray,
    T_train: int,
    horizons: Sequence[int],
    max_short_lag: int,
    predict_step: Callable[[np.ndarray, int, np.ndarray], np.ndarray],
    seed_buffer: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Generic rolling-origin recursive forecast + wet/dry RMSE evaluation.

    Parameters
    ----------
    y_full : concatenated y_train + y_test
    T_train : len(y_train)
    horizons : e.g. (1,2,5,10,30,60,90)
    max_short_lag : size of the rolling buffer of the model's own past
        predictions needed (e.g. 5 for naive/AR+Aux [lags 1-5], 3 for
        XGBoost/LSTM [lags 1-3])
    predict_step : callable(buf, h, t0) -> yhat array of shape (n_starts,).
        `buf` has shape (max_short_lag, n_starts), oldest-first, i.e.
        buf[-1] is the value at time (t0+h-1), buf[-l] the value at
        (t0+h-l). `t0` is the origin array (n_starts,), so the callable can
        look up real long-lag history via a closure over y_full/dates.
        Must return purely deterministic point predictions (no RNG).
    seed_buffer : optional pre-built (max_short_lag, n_starts) buffer if the
        caller wants to seed something other than the raw y_full history
        (not used by the four baselines here; included for completeness).

    Returns
    -------
    DataFrame with one row per horizon: horizon, n_origins, n_wet, n_dry,
    rmse_all, rmse_wet, rmse_dry.
    """
    t0, forecasts = recursive_horizon_forecasts(
        y_full, T_train, horizons, max_short_lag, predict_step, seed_buffer,
    )
    n_starts = len(t0)

    rows = []
    for h in horizons:
        actual = y_full[t0 + h]
        pred = forecasts[h]
        wet = actual > 0
        dry = ~wet
        err = pred - actual
        rows.append({
            "horizon": h,
            "n_origins": n_starts,
            "n_wet": int(wet.sum()),
            "n_dry": int(dry.sum()),
            "rmse_all": float(np.sqrt(np.mean(err ** 2))),
            "rmse_wet": float(np.sqrt(np.mean((pred[wet] - actual[wet]) ** 2))) if wet.any() else float("nan"),
            "rmse_dry": float(np.sqrt(np.mean((pred[dry] - actual[dry]) ** 2))) if dry.any() else float("nan"),
        })
    return pd.DataFrame(rows)
