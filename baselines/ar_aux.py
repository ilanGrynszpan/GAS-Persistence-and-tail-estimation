"""
AR+Aux baseline -- ported unchanged from daily-ML.ipynb (cell 40).

Same 9-lag ridge regression as baselines/naive_ar.py, plus one additional
engineered feature x_aux: the causal proportion of non-zero-rainfall days
so far within the current calendar month (computed strictly from the
target series itself -- NOT an external/exogenous weather variable, see
the notebook-analysis report for this port). alpha=1e-3, unpenalized
intercept, same as naive_ar.py; no hyperparameter search exists for this
model either (lags, alpha, and the x_aux definition are fixed choices in
the source notebook).

For h > 1, x_aux at day (t0+h) is NOT known and is NOT assumed given --
exactly as the notebook's own recursive forecast loop does, it is computed
from real history for days < t0 and from this same trajectory's own prior
predictions for days in [t0, t0+h-1], strictly excluding day (t0+h) itself
(no target leakage at forecast time). See baselines/horizon_eval.py::
month_running_counts for the vectorized causal running-count machinery
this reuses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from baselines.horizon_eval import month_running_counts

SHORT_LAGS = [1, 2, 3, 4, 5]
LONG_LAGS = [364, 365, 366, 367]
ALPHA = 1e-3


def _x_aux_insample(y_train: np.ndarray, dates_train: pd.DatetimeIndex) -> np.ndarray:
    """
    In-sample x_aux exactly as the notebook computes it for FITTING beta:
    proportion of non-zero days in the current calendar month, INCLUSIVE of
    the current day itself (see the notebook-analysis report -- this is a
    mild in-sample look-ahead already present in the source notebook,
    reproduced faithfully here for fitting only; the recursive forecast
    loop below correctly excludes the current day at prediction time).
    """
    day_in_month, nonzero_cum, _ = month_running_counts(y_train, dates_train)
    return nonzero_cum / day_in_month


def fit(y_train: np.ndarray, dates_train: pd.DatetimeIndex) -> np.ndarray:
    """Ridge-regression coefficients (intercept + 9 lags + x_aux), fit on y_train only."""
    s = pd.Series(y_train)
    cols = {f"y{l}": s.shift(l) for l in SHORT_LAGS + LONG_LAGS}
    x_aux = pd.Series(_x_aux_insample(y_train, dates_train))
    data = pd.DataFrame({"y": s, **cols, "x_aux": x_aux}).dropna()

    feat_cols = [f"y{l}" for l in SHORT_LAGS + LONG_LAGS] + ["x_aux"]
    X = data[feat_cols].to_numpy()
    y = data["y"].to_numpy()

    X_design = np.column_stack([np.ones(len(X)), X])
    I = np.eye(X_design.shape[1])
    I[0, 0] = 0.0
    beta = np.linalg.inv(X_design.T @ X_design + ALPHA * I) @ (X_design.T @ y)
    return beta


def make_predict_step(
    beta: np.ndarray,
    y_full: np.ndarray,
    dates_full: pd.DatetimeIndex,
    t0: np.ndarray,
):
    """
    Returns a `predict_step(buf, h, t0)` callable for
    baselines.horizon_eval.recursive_horizon_rmse. `t0` is bound at
    construction time (needed to initialize the per-origin running
    month/day-count state before the horizon loop starts).
    """
    max_short = SHORT_LAGS[-1]
    day_in_month, nonzero_cum, month_id = month_running_counts(y_full, dates_full)

    # State "as of end of day t0" (real, known at every origin).
    run_total = day_in_month[t0].astype(np.float64)
    run_nonzero = nonzero_cum[t0].astype(np.float64)
    run_month = month_id[t0].copy()

    state = {"total": run_total, "nonzero": run_nonzero, "month": run_month}

    def predict_step(buf: np.ndarray, h: int, t0_arr: np.ndarray) -> np.ndarray:
        n = buf.shape[1]
        target_idx = t0_arr + h
        target_month = month_id[target_idx]

        new_month = target_month != state["month"]
        state["total"] = np.where(new_month, 0.0, state["total"])
        state["nonzero"] = np.where(new_month, 0.0, state["nonzero"])
        state["month"] = target_month

        x_aux = np.divide(
            state["nonzero"], state["total"],
            out=np.zeros(n), where=state["total"] > 0,
        )

        short_vals = [buf[max_short - l] for l in SHORT_LAGS]
        long_vals = [y_full[t0_arr + h - l] for l in LONG_LAGS]
        X = np.column_stack([np.ones(n)] + short_vals + long_vals + [x_aux])
        yhat = X @ beta

        state["total"] = state["total"] + 1.0
        state["nonzero"] = state["nonzero"] + (yhat > 0).astype(np.float64)

        return yhat

    return predict_step
