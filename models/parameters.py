"""Parameter tables, confidence intervals, and CSV cache helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _hessian_inverse(result: Any, n_params: int) -> np.ndarray:
    hess_inv = getattr(result, "hess_inv", None)
    if hess_inv is None:
        return np.full((n_params, n_params), np.nan)
    try:
        mat = hess_inv.todense()
    except AttributeError:
        mat = hess_inv
    mat = np.asarray(mat, dtype=float)
    if mat.shape != (n_params, n_params):
        return np.full((n_params, n_params), np.nan)
    return mat


def parameter_frame(
    model,
    fit: dict,
    model_id: str,
    sample: str = "IS",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Return optimizer parameters with approximate Wald confidence intervals.

    The standard errors come from SciPy's inverse Hessian approximation. They
    are convenient for tables, but should be read with care when parameters are
    near optimizer bounds.
    """
    theta = np.asarray(fit["theta"], dtype=float)
    names = model.parameter_names()
    blocks = model.parameter_blocks()
    if len(theta) != len(names):
        raise ValueError("theta length does not match model parameter names")

    alpha = 1.0 - confidence
    if abs(confidence - 0.95) < 1e-12:
        z = 1.959963984540054
    else:
        from scipy.stats import norm

        z = float(norm.ppf(1.0 - alpha / 2.0))

    cov = _hessian_inverse(fit.get("result"), len(theta))
    diag = np.diag(cov)
    se = np.sqrt(np.where(diag >= 0.0, diag, np.nan))
    lower = theta - z * se
    upper = theta + z * se

    level = int(round(confidence * 100))
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "model_id": model_id,
            "sample": sample,
            "seasonal": model.seasonal,
            "model_tv_params": ",".join(model.gas.tv_names),
            "parameter_order": i,
            "parameter": name,
            "block": blocks.get(name, ""),
            "estimate": theta[i],
            "std_error": se[i],
            f"ci_lower_{level}": lower[i],
            f"ci_upper_{level}": upper[i],
            "loglik": float(fit.get("loglik", np.nan)),
            "success": bool(fit.get("success", False)),
        })
    return pd.DataFrame(rows)


def save_parameter_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Save a parameter frame that can later restore theta in order."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["model_id", "parameter_order"]).to_csv(out, index=False)
    return out


def load_theta_csv(path: str | Path, model_id: str | None = None) -> np.ndarray:
    """Load a flat theta vector from a saved parameter CSV."""
    df = pd.read_csv(path)
    if model_id is not None:
        df = df[df["model_id"] == model_id]
    if df.empty:
        raise ValueError("No parameters found for requested model")
    df = df.sort_values("parameter_order")
    return df["estimate"].to_numpy(dtype=float)


def latex_parameter_table(
    frame: pd.DataFrame,
    caption: str = "Estimated model hyper-parameters with 95\\% confidence intervals",
    label: str = "tab:model-parameters",
) -> str:
    """Return a LaTeX table for parameter estimates and intervals."""
    ci_cols = [c for c in frame.columns if c.startswith("ci_lower_")]
    ci_level = ci_cols[0].split("_")[-1] if ci_cols else "95"
    lo = f"ci_lower_{ci_level}"
    hi = f"ci_upper_{ci_level}"
    keep = frame[["model_id", "parameter", "estimate", "std_error", lo, hi]].copy()
    keep = keep.rename(columns={
        "model_id": "Model",
        "parameter": "Parameter",
        "estimate": "Estimate",
        "std_error": "Std. error",
        lo: f"Lower {ci_level}\\%",
        hi: f"Upper {ci_level}\\%",
    })
    body = keep.to_latex(index=False, float_format=lambda x: f"{x:.4f}")
    return "\n".join([
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        body,
        r"\end{table}",
    ])
