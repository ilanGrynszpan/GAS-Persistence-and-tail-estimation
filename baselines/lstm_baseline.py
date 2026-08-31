"""
LSTM baseline -- ported unchanged from daily-ML.ipynb (cell 24 architecture,
cell 42 final training, the definitive section -- see the notebook-analysis
report for this port).

6 lags (1,2,3,364,365,366), each treated as one timestep of a length-6,
1-feature sequence (input_size=1). Hyperparameters below are copied
verbatim from the notebook's own already-completed rolling-window CV
(`rolling_cv_lstm`, cell 25, cell 28's output dict -- NOT cell 29's, which
the notebook-analysis report identified as a stale/buggy duplicate that
swaps the Belo Horizonte and Cruzeiro do Sul parameter sets). Not re-tuned
here, per instruction. Only the six stations used in the rest of this
analysis are kept.

No random seed is set anywhere in the source notebook for numpy/torch, so
LSTM results are not bit-reproducible run-to-run -- reproduced faithfully
(i.e. also left unseeded) rather than "fixed", since seeding would be a
methodology change beyond what was asked (a literal port of what already
existed). No input scaling/normalization is applied, matching the source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SHORT_LAGS = [1, 2, 3]
LONG_LAGS = [364, 365, 366]

_CONFIG_A = {"hidden_size": 32, "num_layers": 1, "dropout": 0.1, "learning_rate": 0.01,
             "weight_decay": 1e-4, "batch_size": 64, "max_epochs": 200}
_CONFIG_B = {"hidden_size": 16, "num_layers": 2, "dropout": 0.2, "learning_rate": 0.005,
             "weight_decay": 1e-4, "batch_size": 64, "max_epochs": 200}

# Verbatim from daily-ML.ipynb cell 28's best_params_lstm dict.
BEST_HYPERPARAMS = {
    "BELO HORIZONTE":         _CONFIG_A,
    "CRUZEIRO DO SUL (ACRE)": _CONFIG_B,
    "DARWIN AIRPORT":         _CONFIG_B,
    "GARANHUNS (PERNAMBUCO)": _CONFIG_B,
    "MANAUS":                 _CONFIG_B,
    "SALVADOR":               _CONFIG_A,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def fit(y_train: np.ndarray, station_name: str) -> LSTMModel:
    s = pd.Series(y_train)
    cols = {f"x{l}" if l in SHORT_LAGS else f"x_{l}": s.shift(l) for l in SHORT_LAGS + LONG_LAGS}
    data = pd.DataFrame({"y": s, **cols}).dropna()

    feat_cols = [f"x{l}" for l in SHORT_LAGS] + [f"x_{l}" for l in LONG_LAGS]
    X = data[feat_cols].to_numpy().astype(np.float32)
    y = data["y"].to_numpy().astype(np.float32)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    params = BEST_HYPERPARAMS[station_name]
    model = LSTMModel(
        input_size=1, hidden_size=params["hidden_size"],
        num_layers=params["num_layers"], dropout=params["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"],
    )
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(y)),
        batch_size=params["batch_size"], shuffle=False,
    )

    model.train()
    for _epoch in range(params["max_epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb).squeeze(-1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def make_predict_step(model: LSTMModel, y_full: np.ndarray):
    max_short = SHORT_LAGS[-1]

    def predict_step(buf: np.ndarray, h: int, t0: np.ndarray) -> np.ndarray:
        short_vals = [buf[max_short - l] for l in SHORT_LAGS]
        long_vals = [y_full[t0 + h - l] for l in LONG_LAGS]
        X = np.column_stack(short_vals + long_vals).astype(np.float32)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        with torch.no_grad():
            x_tensor = torch.tensor(X).to(DEVICE)
            yhat = model(x_tensor).squeeze(-1).cpu().numpy()
        return yhat.astype(np.float64)

    return predict_step
