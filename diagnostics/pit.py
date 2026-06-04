"""PIT calibration diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kstest


def pit_uniform_test(pit: np.ndarray) -> dict:
    """Kolmogorov-Smirnov test of PIT values against Uniform(0, 1)."""
    u = np.asarray(pit, dtype=float)
    u = u[np.isfinite(u)]
    if len(u) == 0:
        return {"n": 0, "ks_stat": np.nan, "pvalue": np.nan, "reject_5pct": False}
    stat, pvalue = kstest(u, "uniform")
    return {
        "n": int(len(u)),
        "ks_stat": float(stat),
        "pvalue": float(pvalue),
        "reject_5pct": bool(pvalue < 0.05),
    }


def pit_frame(pit_by_model: dict[str, np.ndarray], sample: str = "IS") -> pd.DataFrame:
    """Return PIT uniform-fit diagnostics for several models."""
    rows = []
    for model_id, pit in pit_by_model.items():
        row = pit_uniform_test(pit)
        row.update({"model_id": model_id, "sample": sample})
        rows.append(row)
    return pd.DataFrame(rows)


def latex_pit_table(
    frame: pd.DataFrame,
    caption: str = "In-sample PIT uniform-fit test",
    label: str = "tab:pit-fit",
) -> str:
    """Return a LaTeX table for PIT uniform-fit diagnostics."""
    keep = frame[["model_id", "sample", "n", "ks_stat", "pvalue", "reject_5pct"]]
    keep = keep.rename(columns={
        "model_id": "Model",
        "sample": "Sample",
        "n": "$n$",
        "ks_stat": "KS stat.",
        "pvalue": "$p$-value",
        "reject_5pct": "Reject 5\\%",
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
