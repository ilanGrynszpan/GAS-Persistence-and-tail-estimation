"""
Naive/AR baseline -- ported unchanged from daily-ML.ipynb (cell 37).

Not a statsmodels AutoReg: a closed-form ridge regression on 9 lags
(1-5, 364-367) of the raw precipitation series, intercept unpenalized,
alpha=1e-3, solved via the normal equations exactly as the notebook does.
No hyperparameter search exists for this model in the source notebook (the
lag set and alpha are fixed choices, not tuned), so there is nothing to
reuse from a cache here -- the "don't redo cross-validation" instruction is
inherently satisfied since none was ever run for this model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHORT_LAGS = [1, 2, 3, 4, 5]
LONG_LAGS = [364, 365, 366, 367]
ALPHA = 1e-3


def fit(y_train: np.ndarray) -> np.ndarray:
    """Ridge-regression coefficients (intercept + 9 lags), fit on y_train only."""
    return fit_with_sigma(y_train)[0]


def fit_with_sigma(y_train: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Same fit as `fit`, plus the residual standard deviation of the in-sample
    fit, sigma = sqrt(mean((y - X@beta)^2)) over the training rows used to
    estimate beta. This is the variance used for the AR model's assumed
    Normal(mu, sigma^2) predictive distribution when computing CRPS
    (baselines/horizon_eval.py::crps_normal) -- a single constant sigma,
    reused at every forecast origin and horizon, estimated once from the
    model's own in-sample residuals (the ridge regression's implicit
    error model has no horizon-dependent variance -- there is nothing in
    "the method as coded" to grow it with h).
    """
    s = pd.Series(y_train)
    cols = {f"y{l}": s.shift(l) for l in SHORT_LAGS + LONG_LAGS}
    data = pd.DataFrame({"y": s, **cols}).dropna()

    X = data[[f"y{l}" for l in SHORT_LAGS + LONG_LAGS]].to_numpy()
    y = data["y"].to_numpy()

    X_design = np.column_stack([np.ones(len(X)), X])
    I = np.eye(X_design.shape[1])
    I[0, 0] = 0.0
    beta = np.linalg.inv(X_design.T @ X_design + ALPHA * I) @ (X_design.T @ y)

    resid = y - X_design @ beta
    sigma = float(np.sqrt(np.mean(resid ** 2)))
    return beta, sigma


def make_predict_step(beta: np.ndarray, y_full: np.ndarray):
    """
    Returns a `predict_step(buf, h, t0)` callable for
    baselines.horizon_eval.recursive_horizon_rmse, matching the notebook's
    recursive one-step-ahead prediction `x @ beta` with x =
    [1, y1..y5, y364..y367].
    """
    max_short = SHORT_LAGS[-1]

    def predict_step(buf: np.ndarray, h: int, t0: np.ndarray) -> np.ndarray:
        n = buf.shape[1]
        short_vals = [buf[max_short - l] for l in SHORT_LAGS]
        long_vals = [y_full[t0 + h - l] for l in LONG_LAGS]
        X = np.column_stack([np.ones(n)] + short_vals + long_vals)
        return X @ beta

    return predict_step
