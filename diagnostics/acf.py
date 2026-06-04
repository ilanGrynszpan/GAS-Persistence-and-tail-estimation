"""Autocorrelation tables for quantile residual diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from constants import SEASONAL_LAGS


def diagnostic_lags(seasonal: str) -> list[int]:
    """Return the requested short and seasonal residual ACF lags."""
    lags = SEASONAL_LAGS[seasonal]
    return sorted(set([1, 2, 3] + [lag for lag in lags if lag > 3]))


def acf_at_lags(values: np.ndarray, lags: list[int]) -> dict[int, float]:
    """Compute sample autocorrelation at selected lags."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {lag: np.nan for lag in lags}
    x = x - np.mean(x)
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return {lag: np.nan for lag in lags}
    out = {}
    for lag in lags:
        if lag <= 0 or lag >= len(x):
            out[lag] = np.nan
        else:
            out[lag] = float(np.dot(x[lag:], x[:-lag]) / denom)
    return out


def residual_acf_frame(
    residuals_by_model: dict[str, np.ndarray],
    seasonal: str,
    sample: str = "IS",
) -> pd.DataFrame:
    """Return one row per model and requested residual ACF lag."""
    lags = diagnostic_lags(seasonal)
    rows = []
    for model_id, residuals in residuals_by_model.items():
        vals = acf_at_lags(residuals, lags)
        for lag, acf in vals.items():
            rows.append({
                "model_id": model_id,
                "sample": sample,
                "seasonal": seasonal,
                "lag": lag,
                "acf": acf,
            })
    return pd.DataFrame(rows)


def latex_acf_table(
    frame: pd.DataFrame,
    caption: str = "ACF of in-sample quantile residuals",
    label: str = "tab:qr-acf",
) -> str:
    """Return a LaTeX table for residual ACF values."""
    wide = frame.pivot_table(
        index=["model_id", "sample"],
        columns="lag",
        values="acf",
        aggfunc="first",
    ).reset_index()
    wide.columns = [str(c) for c in wide.columns]
    body = wide.to_latex(index=False, float_format=lambda x: f"{x:.4f}")
    return "\n".join([
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        body,
        r"\end{table}",
    ])
