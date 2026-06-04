"""Cache expensive model evaluation outputs for later reporting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics.acf import residual_acf_frame
from diagnostics.pit import pit_frame
from diagnostics.tests import coverage_frame_95
from models.parameters import parameter_frame, save_parameter_csv
from simulation.simulator import evaluate_metrics, forecast_summary, save_forecast_summary


def _save_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _metrics_frame(model_id: str, metrics: dict) -> pd.DataFrame:
    rows = []
    for sample in ["is", "oos"]:
        rows.append({
            "model_id": model_id,
            "sample": sample.upper(),
            "loglik": metrics.get(f"{sample}_loglik"),
            "aic": metrics.get(f"{sample}_aic"),
            "bic": metrics.get(f"{sample}_bic"),
            "rmse": metrics.get(f"{sample}_rmse"),
            "mad": metrics.get(f"{sample}_mad"),
            "crps": metrics.get(f"{sample}_crps"),
        })
    return pd.DataFrame(rows)


def _series_id(model_id: str) -> str:
    if model_id.endswith("_phi_xi"):
        return model_id[:-7]
    if model_id.endswith("_phi"):
        return model_id[:-4]
    return model_id


def _series_summary_frame(model_id: str, y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    rows = []
    for sample, values in [
        ("IS", np.asarray(y_train, dtype=float)),
        ("OOS", np.asarray(y_test, dtype=float)),
        ("Full", np.concatenate([np.asarray(y_train, dtype=float), np.asarray(y_test, dtype=float)])),
    ]:
        x = values[np.isfinite(values)]
        qs = np.quantile(x, [0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]) if len(x) else np.full(8, np.nan)
        rows.append({
            "series_id": _series_id(model_id),
            "model_id": model_id,
            "sample": sample,
            "n": len(x),
            "min": np.nanmin(x) if len(x) else np.nan,
            "max": np.nanmax(x) if len(x) else np.nan,
            "mean": np.nanmean(x) if len(x) else np.nan,
            "std": np.nanstd(x, ddof=1) if len(x) > 1 else np.nan,
            "prop_zero": float(np.mean(x == 0.0)) if len(x) else np.nan,
            "q01": qs[0],
            "q05": qs[1],
            "q25": qs[2],
            "q50": qs[3],
            "q75": qs[4],
            "q90": qs[5],
            "q95": qs[6],
            "q99": qs[7],
        })
    return pd.DataFrame(rows)


def _series_values_frame(model_id: str, y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    return pd.concat([
        pd.DataFrame({
            "series_id": _series_id(model_id),
            "model_id": model_id,
            "sample": "IS",
            "t_index": np.arange(len(y_train)),
            "y": np.asarray(y_train, dtype=float),
        }),
        pd.DataFrame({
            "series_id": _series_id(model_id),
            "model_id": model_id,
            "sample": "OOS",
            "t_index": np.arange(len(y_test)),
            "y": np.asarray(y_test, dtype=float),
        }),
    ], ignore_index=True)


def _jb_frame(model_id: str, metrics: dict) -> pd.DataFrame:
    rows = []
    for sample, key in [("IS", "is_jb"), ("OOS", "oos_jb")]:
        d = metrics.get(key, {}) or {}
        rows.append({
            "model_id": model_id,
            "sample": sample,
            "n": d.get("n", np.nan),
            "skewness": d.get("skewness", np.nan),
            "kurtosis": d.get("kurtosis", np.nan),
            "JB_stat": d.get("stat", np.nan),
            "pvalue": d.get("pvalue", np.nan),
            "reject_5pct": bool(d.get("pvalue", np.nan) < 0.05) if pd.notna(d.get("pvalue", np.nan)) else False,
        })
    return pd.DataFrame(rows)


def evaluate_and_cache_model(
    model,
    fit: dict,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_id: str,
    output_dir: str | Path = "artifacts/cache",
    n_draws: int = 500,
    seed: int = 42,
) -> dict:
    """
    Evaluate a fitted model and save CSV artifacts for reuse.

    The saved CSVs are intended as cache files and are ignored by git when the
    repository .gitignore from this package is used.
    """
    out = Path(output_dir) / model_id
    theta = np.asarray(fit["theta"], dtype=float)

    param_csv = out / "estimated_parameters.csv"
    if param_csv.exists() and fit.get("result") is None:
        params = pd.read_csv(param_csv)
    else:
        params = parameter_frame(model, fit, model_id=model_id)
        param_csv = save_parameter_csv(params, param_csv)

    metrics = evaluate_metrics(
        model, theta, y_train, y_test, n_draws=n_draws, seed=seed
    )
    metrics_df = _metrics_frame(model_id, metrics)
    metrics_csv = _save_frame(metrics_df, out / "metrics.csv")

    oos_forecasts = forecast_summary(
        model,
        metrics["oos_paths"],
        y=y_test,
        sample="OOS",
        model_id=model_id,
        n_draws=n_draws,
        seed=seed,
    )
    forecast_csv = save_forecast_summary(oos_forecasts, out / "oos_forecasts.csv")

    acf_is = residual_acf_frame(
        {model_id: metrics["is_qr"]},
        seasonal=model.seasonal,
        sample="IS",
    )
    acf_oos = residual_acf_frame(
        {model_id: metrics["oos_qr"]},
        seasonal=model.seasonal,
        sample="OOS",
    )
    acf_df = pd.concat([acf_is, acf_oos], ignore_index=True)
    acf_csv = _save_frame(acf_df, out / "quantile_residual_acf.csv")

    pit_df = pd.concat([
        pit_frame({model_id: metrics["is_pit"]}, sample="IS"),
        pit_frame({model_id: metrics["oos_pit"]}, sample="OOS"),
        pit_frame(
            {model_id: np.concatenate([metrics["is_pit"], metrics["oos_pit"]])},
            sample="Full",
        ),
    ], ignore_index=True)
    pit_csv = _save_frame(pit_df, out / "pit_fit.csv")

    coverage_df = coverage_frame_95({
        model_id: {
            "IS": metrics["is_pit"],
            "OOS": metrics["oos_pit"],
            "Full": np.concatenate([metrics["is_pit"], metrics["oos_pit"]]),
        }
    })
    coverage_csv = _save_frame(coverage_df, out / "coverage_95.csv")

    diagnostic_series = pd.concat([
        pd.DataFrame({
            "model_id": model_id,
            "sample": "IS",
            "t_index": np.arange(len(metrics["is_pit"])),
            "pit": metrics["is_pit"],
            "quantile_residual": metrics["is_qr"],
        }),
        pd.DataFrame({
            "model_id": model_id,
            "sample": "OOS",
            "t_index": np.arange(len(metrics["oos_pit"])),
            "pit": metrics["oos_pit"],
            "quantile_residual": metrics["oos_qr"],
        }),
    ], ignore_index=True)
    diagnostic_series_csv = _save_frame(
        diagnostic_series, out / "diagnostic_series.csv"
    )
    jb_df = _jb_frame(model_id, metrics)
    jb_csv = _save_frame(jb_df, out / "jarque_bera.csv")
    series_summary = _series_summary_frame(model_id, y_train, y_test)
    series_summary_csv = _save_frame(series_summary, out / "series_summary.csv")
    series_values = _series_values_frame(model_id, y_train, y_test)
    series_values_csv = _save_frame(series_values, out / "series_values.csv")

    return {
        "model_id": model_id,
        "model": model,
        "fit": fit,
        "metrics": metrics,
        "parameters": params,
        "metrics_frame": metrics_df,
        "oos_forecasts": oos_forecasts,
        "acf": acf_df,
        "pit": pit_df,
        "coverage": coverage_df,
        "diagnostic_series": diagnostic_series,
        "jarque_bera": jb_df,
        "series_summary": series_summary,
        "series_values": series_values,
        "csv": {
            "parameters": param_csv,
            "metrics": metrics_csv,
            "oos_forecasts": forecast_csv,
            "acf": acf_csv,
            "pit": pit_csv,
            "coverage": coverage_csv,
            "diagnostic_series": diagnostic_series_csv,
            "jarque_bera": jb_csv,
            "series_summary": series_summary_csv,
            "series_values": series_values_csv,
        },
    }
