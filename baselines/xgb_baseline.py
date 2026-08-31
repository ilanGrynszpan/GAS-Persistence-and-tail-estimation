"""
XGBoost baseline -- ported unchanged from daily-ML.ipynb (cell 33, the
final/definitive section -- see the notebook-analysis report for this
port; cells 0-18 are an earlier, superseded 3-station prototype).

6 lags: 1, 2, 3, 364, 365, 366 (no intercept -- tree model). Hyperparameters
below are copied verbatim from the notebook's own already-completed
expanding-window time-series CV (`tune_xgb_time_series`, cell 3, run over
all 8 stations in cell 8) -- NOT re-tuned here, per instruction. Only the
six stations used in the rest of this analysis are kept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

SHORT_LAGS = [1, 2, 3]
LONG_LAGS = [364, 365, 366]

# Verbatim from daily-ML.ipynb cell 8's CV output / cell 33's `best_fit` dict.
BEST_HYPERPARAMS = {
    "BELO HORIZONTE":         {"n_estimators": 500,  "max_depth": 5, "learning_rate": 0.01},
    "CRUZEIRO DO SUL (ACRE)": {"n_estimators": 1000, "max_depth": 4, "learning_rate": 0.01},
    "DARWIN AIRPORT":         {"n_estimators": 500,  "max_depth": 4, "learning_rate": 0.01},
    "GARANHUNS (PERNAMBUCO)": {"n_estimators": 500,  "max_depth": 4, "learning_rate": 0.05},
    "MANAUS":                 {"n_estimators": 800,  "max_depth": 7, "learning_rate": 0.01},
    "SALVADOR":               {"n_estimators": 500,  "max_depth": 5, "learning_rate": 0.05},
}


def fit(y_train: np.ndarray, station_name: str) -> XGBRegressor:
    s = pd.Series(y_train)
    cols = {f"x{l}" if l in SHORT_LAGS else f"x_{l}": s.shift(l) for l in SHORT_LAGS + LONG_LAGS}
    data = pd.DataFrame({"y": s, **cols}).dropna()

    feat_cols = [f"x{l}" for l in SHORT_LAGS] + [f"x_{l}" for l in LONG_LAGS]
    X = data[feat_cols].to_numpy()
    y = data["y"].to_numpy()

    params = BEST_HYPERPARAMS[station_name]
    model = XGBRegressor(objective="reg:squarederror", random_state=42, **params)
    model.fit(X, y)
    return model


def make_predict_step(model: XGBRegressor, y_full: np.ndarray):
    max_short = SHORT_LAGS[-1]

    def predict_step(buf: np.ndarray, h: int, t0: np.ndarray) -> np.ndarray:
        short_vals = [buf[max_short - l] for l in SHORT_LAGS]
        long_vals = [y_full[t0 + h - l] for l in LONG_LAGS]
        X = np.column_stack(short_vals + long_vals)
        return model.predict(X)

    return predict_step
