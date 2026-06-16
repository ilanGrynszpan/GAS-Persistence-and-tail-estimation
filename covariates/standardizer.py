"""
Covariate standardization using training-set moments only.

All covariates are standardized as (x - mu_train) / std_train,
where mu_train and std_train are computed on the training portion only.
No leakage: the same moments are applied to the test set.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class CovariateStandardizer:
    """
    Fit on training data; transform train and test with training moments.

    Parameters
    ----------
    columns : list of str
        Column names to standardize (others are left as-is or dropped).
    """

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns
        self._means: Dict[str, float] = {}
        self._stds: Dict[str, float] = {}
        self._fitted = False

    def fit(self, X_train: pd.DataFrame) -> "CovariateStandardizer":
        """Compute mean and std from training data (ignoring NaN)."""
        cols = self.columns or list(X_train.columns)
        self.columns = cols
        for c in cols:
            if c in X_train.columns:
                vals = X_train[c].dropna().values.astype(float)
                self._means[c] = float(np.mean(vals)) if len(vals) > 0 else 0.0
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
                self._stds[c] = std if std > 1e-10 else 1.0
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with standardized columns (training moments)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        out = X.copy()
        for c in self.columns:
            if c in out.columns:
                out[c] = (out[c].astype(float) - self._means[c]) / self._stds[c]
        return out

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit on X_train and return standardized X_train."""
        self.fit(X_train)
        return self.transform(X_train)

    def get_moments(self) -> Dict[str, Dict[str, float]]:
        return {c: {"mean": self._means[c], "std": self._stds[c]} for c in self.columns}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "columns": self.columns,
            "means": self._means,
            "stds": self._stds,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CovariateStandardizer":
        path = Path(path)
        data = json.loads(path.read_text())
        obj = cls(columns=data["columns"])
        obj._means = data["means"]
        obj._stds = data["stds"]
        obj._fitted = True
        return obj
