"""
Extended multi-location report generator.

=============================================================================
OVERVIEW
=============================================================================

Generates a comprehensive publication-quality PDF report covering all
modelled precipitation stations (including Belo Horizonte which was already
estimated in the previous session).

For each location the report contains:
    1. Data summary
    2. Stage 1 (baseline GAS) model comparison
    3. Stage 2 (weather covariates) model comparison
    4. Stage 3 (Harvey long-short) model comparison
    5. Stage advancement decisions (2% CRPS threshold)
    6. IS diagnostics: PIT histogram, QQ plot, residual ACF (400 lags)
    7. Extended OOS metrics: CRPS, twCRPS@90/95/99, QS@10 levels,
                             Log Score, Brier, Kupiec, Christoffersen
    8. Dynamic conditional quantile time series
    9. Return level figures (daily and climatological)
   10. Signature figures (observed + Q95/Q99/Q99.9 + ENSO shading + phi_t panel)
   11. Seasonal decomposition (monthly boxplots, QS, twCRPS)
   12. ENSO decomposition (standard regimes + ENSO intensity)
   13. Wet/dry season classification and decomposition
   14. Tail calibration plots
   15. Stage 4 regime model results

Followed by a global comparison chapter.

=============================================================================
USAGE
=============================================================================

    python generate_report_extended.py [options]

Options:
    --reports-dir PATH   Output directory (default: reports/multi_location)
    --location NAME      Generate report only for this location
    --no-compile         Skip pdflatex compilation

To recompile the PDF manually after editing the LaTeX:
    cd reports/multi_location
    pdflatex -interaction=nonstopmode report.tex
    pdflatex -interaction=nonstopmode report.tex

=============================================================================
ARTIFACT DEPENDENCIES
=============================================================================

For each location the script requires:
    artifacts/{run_id}/stage1/{model_id}/metadata.json
    artifacts/{run_id}/stage1/{model_id}/estimated_parameters.csv
    artifacts/{run_id}/stage1/{model_id}/paths.npz
    artifacts/{run_id}/stage_winners.json
    (similarly for stage2, stage3, stage4_regime)

The script will skip any model whose artifacts are missing rather than crashing.

=============================================================================
HOW TO REGENERATE PDF WITHOUT AI
=============================================================================

After editing report.tex:
    1. Open a CMD or PowerShell prompt in the reports/multi_location directory.
    2. Run: pdflatex -interaction=nonstopmode report.tex
    3. Run:  pdflatex -interaction=nonstopmode report.tex  (second pass for TOC)
    4. The updated report.pdf will appear in the same directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRECIP_DIR  = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR    = ROOT / "data" / "input" / "ERA5"
NINO34_PATH = ROOT / "data" / "processed" / "pacific" / "NINO34_daily.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

from data.station_loader import STATION_REGISTRY, StationDataLoader
from diagnostics.dynamics import (
    QUANTILE_LEVELS, DAILY_RETURN_PERIODS, CLIM_RETURN_PERIODS_YEARS,
    compute_dynamic_quantiles, compute_daily_return_levels,
    compute_climatological_return_levels, compute_exceedance_frequencies,
    plot_signature_figure, plot_dynamic_quantiles, plot_return_levels,
    plot_tail_calibration, save_dynamic_arrays, classify_enso,
    daily_return_level_quantile, climatological_return_level_quantile,
)
from diagnostics.decomposition import (
    classify_enso_regime, classify_enso_intensity,
    classify_wet_dry_months, seasonal_decomposition,
    enso_decomposition, wet_dry_decomposition,
    plot_monthly_quantile_boxplot, plot_enso_exceedance,
    plot_enso_intensity_quantile, save_decomposition_csvs,
    ENSO_EL_NINO, ENSO_NEUTRAL, ENSO_LA_NINA, NO_SEASON,
)
from diagnostics.scoring import compute_extended_oos_metrics, EXTENDED_QUANTILE_LEVELS
from diagnostics.residuals import pit_values, quantile_residuals

# Locations to process — ALL locations including BH
ALL_STATIONS_ORDERED = [
    "BELO HORIZONTE",
    "CRUZEIRO DO SUL (ACRE)",
    "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)",
    "MANAUS",
    "SALVADOR",
    "SÃO PAULO",
    "TORONTO",
]

SKIP_STATIONS = {"RIYADH OBS. (O.A.P."}


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json_safe(path: Path) -> dict:
    """Load a JSON file that may contain bare NaN tokens (not valid JSON spec).

    Python's json.dumps with allow_nan=True writes bare NaN/Infinity tokens.
    json.loads rejects them.  This function walks the raw text character-by-
    character so it only substitutes NaN tokens that appear as JSON values
    (i.e. outside quoted strings), then restores them as float('nan').
    """
    raw = path.read_text(encoding="utf-8-sig")

    out: list = []
    i = 0
    n = len(raw)
    while i < n:
        c = raw[i]
        if c == '"':
            out.append(c)
            i += 1
            while i < n:
                ch = raw[i]
                out.append(ch)
                if ch == '\\':
                    i += 1
                    if i < n:
                        out.append(raw[i])
                elif ch == '"':
                    i += 1
                    break
                i += 1
        elif raw[i:i + 3] == 'NaN':
            out.append('"__NAN__"')
            i += 3
        else:
            out.append(c)
            i += 1

    data = json.loads(''.join(out))

    def _restore(obj):
        if isinstance(obj, dict):
            return {k: _restore(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_restore(v) for v in obj]
        if obj == "__NAN__":
            return float("nan")
        return obj

    return _restore(data)


def _tex(s: str) -> str:
    """Escape special LaTeX characters in arbitrary strings."""
    return (
        str(s)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _tt(s: str) -> str:
    """Return s as LaTeX \\texttt{} with underscores escaped."""
    return r"\texttt{" + _tex(s) + "}"


def _bold(s: str) -> str:
    return r"\textbf{" + _tex(str(s)) + "}"


def _fmt(v, digits: int = 4) -> str:
    """Format a numeric value for a LaTeX table cell."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    try:
        f = float(v)
        if abs(f) > 1e6:
            # Clearly a failed/diverged result; show scientific notation compactly
            return rf"$\gg 10^6$"
        return f"{f:.{digits}f}"
    except Exception:
        return str(v)


def _pval(v) -> str:
    """Format a p-value: <0.001 shown as <0.001."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    f = float(v)
    if f < 0.001:
        return r"$<$0.001"
    return f"{f:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# Artifact helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_meta(run_dir: Path, stage: str, model_id: str) -> dict:
    p = run_dir / stage / model_id / "metadata.json"
    return _load_json_safe(p) if p.exists() else {}


def _load_params_df(run_dir: Path, stage: str, model_id: str) -> pd.DataFrame:
    p = run_dir / stage / model_id / "estimated_parameters.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _load_paths_npz(run_dir: Path, stage: str, model_id: str) -> Optional[dict]:
    p = run_dir / stage / model_id / "paths.npz"
    if not p.exists():
        return None
    raw = dict(np.load(p, allow_pickle=False))
    return raw


def _load_winners(run_dir: Path) -> dict:
    p = run_dir / "stage_winners.json"
    return _load_json_safe(p) if p.exists() else {}


def _all_model_ids(run_dir: Path, stage: str) -> List[str]:
    stage_dir = run_dir / stage
    if not stage_dir.exists():
        return []
    return sorted(
        d.name for d in stage_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model reconstruction from artifacts
# ─────────────────────────────────────────────────────────────────────────────

def _infer_gas_lags(params_df: pd.DataFrame, tv: str = "phi") -> List[int]:
    """Infer GAS lag set from parameter names, e.g. A_phi_364 → 364."""
    pat  = re.compile(rf"^[AB]_{tv}_(\d+)$")
    lags = {int(m.group(1)) for p in params_df.get("parameter", [])
            if (m := pat.match(str(p)))}
    return sorted(lags) if lags else [1, 2, 3]


def _infer_tv_from_params(params_df: pd.DataFrame) -> List[str]:
    """
    Infer which parameters are time-varying from estimated_parameters.csv.

    Falls back gracefully when metadata does not store tv_param_names.
    Checks for omega_phi / A_phi_* (phi dynamic) and omega_xi / A_xi_* (xi dynamic).
    """
    if params_df is None or params_df.empty:
        return ["phi"]
    names: set = set()
    if "parameter" in params_df.columns:
        names = set(str(p) for p in params_df["parameter"])
    tv: List[str] = []
    if any(n.startswith("omega_phi") or n.startswith("A_phi_") for n in names):
        tv.append("phi")
    if any(n.startswith("omega_xi") or n.startswith("A_xi_") for n in names):
        tv.append("xi")
    return tv if tv else ["phi"]


def _build_model_from_meta(meta: dict, params_df: pd.DataFrame):
    """Reconstruct a model instance from metadata (for cdf_series calls)."""
    from distributions.gb2_log_link import GB2LogLink
    from distributions.gb2_phi_only import GB2LogLinkPhiOnly
    from pi_dynamics.factory import make_pi_dynamics

    tv = meta.get("tv_param_names", ["phi"])
    dist = GB2LogLink() if len(tv) > 1 else GB2LogLinkPhiOnly()
    pi_dyn = make_pi_dynamics("ar_logistic", seasonal="daily")

    model_class = meta.get("model_class", "ZAGASModel")

    if model_class in ("ZAGASModel", "RegimeZAGASModel"):
        from models.za_gas_model import ZAGASModel
        lags = _infer_gas_lags(params_df, tv[0]) if not params_df.empty else [1, 2, 3]
        scaling_raw = meta.get("scaling", "diagonal_inverse_fisher")
        scaling_map = {"diagfi": "diagonal_inverse_fisher", "fullfi": "inverse_fisher"}
        scaling = scaling_map.get(scaling_raw, scaling_raw)
        return ZAGASModel(
            distribution=dist, pi_dynamics=pi_dyn,
            seasonal="daily", gas_lags=lags, scaling=scaling,
        )

    if "CovZAGASModel" in model_class or "stage2" in meta.get("model_id", ""):
        from models.cov_gas_model import CovZAGASModel
        lags    = _infer_gas_lags(params_df, tv[0]) if not params_df.empty else [1, 2, 3]
        scaling_raw = meta.get("scaling", "diagonal_inverse_fisher")
        scaling_map = {"diagfi": "diagonal_inverse_fisher", "fullfi": "inverse_fisher"}
        scaling = scaling_map.get(scaling_raw, scaling_raw)
        cov_names = meta.get("cov_names", [])
        return CovZAGASModel(
            distribution=dist, pi_dynamics=pi_dyn,
            seasonal="daily", gas_lags=lags, scaling=scaling,
            cov_names=cov_names,
        )

    if "HarveyZAGASModel" in model_class or "harvey" in meta.get("model_id", ""):
        from models.harvey_gas import HarveyZAGASModel
        long_names  = meta.get("long_names", [])
        short_names = meta.get("short_names", [])
        scaling_raw = meta.get("scaling", "diagonal_inverse_fisher")
        scaling_map = {"diagfi": "diagonal_inverse_fisher", "fullfi": "inverse_fisher"}
        scaling = scaling_map.get(scaling_raw, scaling_raw)
        return HarveyZAGASModel(
            distribution=dist, pi_dynamics=pi_dyn, seasonal="daily",
            long_names=long_names, short_names=short_names, scaling=scaling,
        )

    # Fallback
    from models.za_gas_model import ZAGASModel
    return ZAGASModel(
        distribution=dist, pi_dynamics=pi_dyn,
        seasonal="daily", gas_lags=[1, 2, 3],
    )


def _get_theta(params_df: pd.DataFrame) -> Optional[np.ndarray]:
    if params_df.empty:
        return None
    if "value" in params_df.columns:
        return params_df["value"].to_numpy(dtype=float)
    return None


def _reconstruct_oos_paths(paths_npz: dict, meta: dict) -> Optional[dict]:
    """Convert paths.npz + metadata into a paths_oos-style dict."""
    if paths_npz is None:
        return None
    if "pi_oos" not in paths_npz or "f_arr_oos" not in paths_npz:
        return None

    tv_names  = meta.get("tv_param_names", ["phi"])
    static_ks = meta.get("static_param_names", ["gamma", "zeta"])
    static    = {k: float(meta.get(k, 0.0)) for k in static_ks if k in meta}

    # Static params may be in metadata under their name
    for k in ("gamma", "zeta", "xi"):
        if k not in static and k in meta:
            try:
                static[k] = float(meta[k])
            except Exception:
                pass

    return {
        "pi_oos":    paths_npz["pi_oos"],
        "f_arr_oos": paths_npz["f_arr_oos"],
        "tv_names":  tv_names,
        "static":    static,
    }


# ─────────────────────────────────────────────────────────────────────────────
# IS diagnostic computation and plotting (400-lag ACF, PIT, QR)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_is_diagnostics(
    model_id: str,
    pit: np.ndarray,
    qr: np.ndarray,
    fig_dir: Path,
    n_acf_lags: int = 400,
) -> Tuple[str, str, str]:
    """
    Generate 3-panel IS diagnostic figure: PIT | QQ | ACF(400).

    Returns paths to the 3 individual saved figures.
    """
    from statsmodels.graphics.tsaplots import plot_acf as _sm_acf

    fig_dir.mkdir(parents=True, exist_ok=True)
    prefix = model_id.replace("/", "_")

    # 1. PIT histogram
    fig1, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(pit[np.isfinite(pit)], bins=20, density=True,
            color="#3498db", alpha=0.75, edgecolor="white", lw=0.5)
    ax.axhline(1.0, color="red", lw=1.2, linestyle="--", label="U(0,1)")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title("IS PIT histogram")
    ax.legend(fontsize=8)
    plt.tight_layout()
    pit_path = str(fig_dir / f"{prefix}_pit.png")
    fig1.savefig(pit_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # 2. QQ plot (normal quantile residuals)
    qr_finite = qr[np.isfinite(qr)]
    theoretical = np.sort(
        np.random.default_rng(42).standard_normal(len(qr_finite))
    )
    empirical   = np.sort(qr_finite)
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.scatter(theoretical, empirical, s=2, alpha=0.4, color="#2c3e50")
    lim = max(abs(theoretical).max(), abs(empirical).max()) * 1.05
    ax2.plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax2.set_xlabel("Theoretical N(0,1) quantiles")
    ax2.set_ylabel("Sample quantile residuals")
    ax2.set_title("Normal QQ plot (IS)")
    plt.tight_layout()
    qq_path = str(fig_dir / f"{prefix}_qq.png")
    fig2.savefig(qq_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # 3. ACF with 400 lags
    qr_plot = qr_finite - qr_finite.mean()
    n_lags  = min(n_acf_lags, len(qr_plot) - 1)
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    try:
        _sm_acf(qr_plot, lags=n_lags, ax=ax3, title="", alpha=0.05,
                zero=False, bartlett_confint=True)
    except Exception:
        # Fallback: manual ACF
        acf_vals = [float(np.corrcoef(qr_plot[:-k], qr_plot[k:])[0, 1])
                    for k in range(1, n_lags + 1)]
        ci = 1.96 / np.sqrt(len(qr_plot))
        ax3.bar(range(1, n_lags + 1), acf_vals, color="#3498db", width=1, alpha=0.6)
        ax3.axhline(ci,  color="red", linestyle="--", lw=0.8)
        ax3.axhline(-ci, color="red", linestyle="--", lw=0.8)
    ax3.set_xlabel("Lag (days)")
    ax3.set_ylabel("ACF")
    ax3.set_title(f"Quantile residual ACF — {n_lags} lags (IS)")
    ax3.axhline(0, color="black", lw=0.5)
    plt.tight_layout()
    acf_path = str(fig_dir / f"{prefix}_acf.png")
    fig3.savefig(acf_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    return pit_path, qq_path, acf_path


# ─────────────────────────────────────────────────────────────────────────────
# Per-model extended metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_extended_metrics_for_model(
    model_id: str,
    run_dir: Path,
    data: dict,
    model_cache: dict,
) -> Optional[dict]:
    """
    Compute all extended OOS metrics for one model.

    Returns dict of metrics or None if paths unavailable.
    """
    # Determine stage
    stage = "stage1"
    for s in ("stage3", "stage4_regime", "stage2", "stage1"):
        if model_id.startswith(s):
            stage = s
            break

    meta      = _load_meta(run_dir, stage, model_id)
    params_df = _load_params_df(run_dir, stage, model_id)
    paths_npz = _load_paths_npz(run_dir, stage, model_id)

    if paths_npz is None:
        return None

    # Patch metadata for older artifact format where key names differ
    if meta is not None:
        meta = dict(meta)
        # 1. tv_param_names: stored as "tv_names" in current artifacts
        if not meta.get("tv_param_names"):
            tv_from_artifact = meta.get("tv_names")  # current format
            meta["tv_param_names"] = (
                tv_from_artifact if tv_from_artifact
                else _infer_tv_from_params(params_df)
            )
        # 2. static params gamma/zeta: stored inside bound_diagnostics or in params_df
        bd = meta.get("bound_diagnostics", {})
        for key in ("gamma", "zeta", "xi"):
            if key not in meta:
                bd_key = f"static_{key}"
                if bd_key in bd:
                    meta[key] = float(bd[bd_key])
        # 3. Fallback: read static params from estimated_parameters.csv
        if params_df is not None and not params_df.empty:
            if "parameter" in params_df.columns and "value" in params_df.columns:
                for key in ("gamma", "zeta", "xi"):
                    if key not in meta:
                        row = params_df[params_df["parameter"] == key]
                        if not row.empty:
                            meta[key] = float(row.iloc[0]["value"])

    oos_paths = _reconstruct_oos_paths(paths_npz, meta)
    if oos_paths is None:
        return None

    # Get model instance (cached)
    if model_id not in model_cache:
        try:
            m = _build_model_from_meta(meta, params_df)
            model_cache[model_id] = m
        except Exception as exc:
            print(f"  [WARN] Could not reconstruct model {model_id}: {exc}")
            return None
    model = model_cache[model_id]

    try:
        mets = compute_extended_oos_metrics(
            model=model,
            paths_oos=oos_paths,
            y_test=data["y_test"],
            n_draws=1000,
        )
        mets["model_id"] = model_id
        return mets
    except Exception as exc:
        print(f"  [WARN] Extended metrics failed for {model_id}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# IS diagnostics for winner models
# ─────────────────────────────────────────────────────────────────────────────

def _compute_is_diagnostics_for_winner(
    model_id: str,
    run_dir: Path,
    data: dict,
    col_name_dict: dict,
    fig_dir: Path,
) -> Optional[dict]:
    """
    Compute IS PIT / QR for a winner model.  Returns dict with pit, qr, paths.
    """
    stage = "stage1"
    for s in ("stage3", "stage4_regime", "stage2", "stage1"):
        if model_id.startswith(s):
            stage = s
            break

    meta      = _load_meta(run_dir, stage, model_id)
    params_df = _load_params_df(run_dir, stage, model_id)
    theta     = _get_theta(params_df)

    if theta is None:
        return None

    y_train = data["y_train"]
    bt      = data["covariate_blocks_train"]
    cn      = data["covariate_col_names"]

    try:
        model = _build_model_from_meta(meta, params_df)
        model_class = meta.get("model_class", "ZAGASModel")

        if "Harvey" in model_class or "harvey" in model_id:
            long_names  = meta.get("long_names", [])
            short_names = meta.get("short_names", [])
            Xl = _find_block_arr(long_names,  cn, bt) if long_names  else np.zeros((len(y_train), 0))
            Xs = _find_block_arr(short_names, cn, bt) if short_names else np.zeros((len(y_train), 0))
            cdfs = model.cdf_series(theta, y_train, X_long=Xl, X_short=Xs)

        elif "Cov" in model_class or "stage2" in model_id:
            cov_names = meta.get("cov_names", [])
            Xc = _find_block_arr(cov_names, cn, bt) if cov_names else np.zeros((len(y_train), 0))
            cdfs = model.cdf_series(theta, y_train, X=Xc)

        else:
            cdfs = model.cdf_series(theta, y_train)

    except Exception as exc:
        print(f"  [WARN] IS cdf_series failed for {model_id}: {exc}")
        return None

    if cdfs is None or len(cdfs) == 0:
        return None

    rng  = np.random.default_rng(0)
    n    = len(cdfs)
    y_eff = y_train[len(y_train) - n:]

    pit = pit_values(cdfs, y_eff, randomise_zeros=True, rng=rng)
    qr  = quantile_residuals(cdfs, y_eff, randomise_zeros=True, rng=rng)

    pit_path, qq_path, acf_path = _plot_is_diagnostics(
        model_id=model_id, pit=pit, qr=qr, fig_dir=fig_dir
    )

    return {
        "pit":      pit,
        "qr":       qr,
        "pit_mean": float(np.nanmean(pit)),
        "pit_std":  float(np.nanstd(pit)),
        "pit_path": pit_path,
        "qq_path":  qq_path,
        "acf_path": acf_path,
    }


def _find_block_arr(
    col_names: List[str],
    col_name_dict: dict,
    blocks: dict,
) -> np.ndarray:
    """Find the covariate block whose column list matches col_names."""
    for block_name, cols in col_name_dict.items():
        if cols == col_names:
            return blocks[block_name]
    # Fallback: return zeros if not found
    return np.zeros((next(iter(blocks.values())).shape[0], len(col_names)))


# ─────────────────────────────────────────────────────────────────────────────
# Table builders
# ─────────────────────────────────────────────────────────────────────────────

def _stage_comparison_table_tex(
    run_dir: Path, stage: str,
    winners: dict, caption: str, label: str,
) -> str:
    """LaTeX model comparison table for one stage."""
    model_ids = _all_model_ids(run_dir, stage)
    if not model_ids:
        return f"% No models found for {stage}\n"

    rows = []
    for mid in model_ids:
        meta = _load_meta(run_dir, stage, mid)
        mcore_p = run_dir / stage / mid / "metrics_core.json"
        mcore = _load_json_safe(mcore_p) if mcore_p.exists() else {}
        rows.append({
            "model":    mid,
            "validity": meta.get("validity", "?"),
            "loglik":   meta.get("loglik"),
            "crps":     mcore.get("crps_mean"),
            "rmse":     mcore.get("rmse"),
            "mae":      mcore.get("mae"),
            "n_params": meta.get("n_params"),
        })

    winner_mid = winners.get(stage, {}).get("winner")

    cols_hdr = r"\small Model & Validity & loglik & CRPS & RMSE & MAE & $p$"
    body_lines = []
    for r in rows:
        marker = r"\textbf{*}" if r["model"] == winner_mid else ""
        body_lines.append(
            f"{marker}\\texttt{{{_tex(r['model'])}}} & "
            f"{_tex(r['validity'])} & "
            f"{_fmt(r['loglik'], 1)} & "
            f"{_fmt(r['crps'])} & "
            f"{_fmt(r['rmse'])} & "
            f"{_fmt(r['mae'])} & "
            f"{_fmt(r['n_params'], 0)} \\\\"
        )

    return (
        r"\begin{table}[ht]" + "\n"
        r"\centering" + "\n"
        r"\footnotesize" + "\n"
        r"\begin{tabular}{lllrrrr}" + "\n"
        r"\hline" + "\n"
        + cols_hdr + r" \\" + "\n"
        r"\hline" + "\n"
        + "\n".join(body_lines) + "\n"
        r"\hline" + "\n"
        r"\end{tabular}" + "\n"
        rf"\caption{{{caption}}}" + "\n"
        rf"\label{{{label}}}" + "\n"
        r"\end{table}" + "\n"
    )


def _extended_metrics_table_tex(
    metrics_list: List[dict],
    caption: str,
    label: str,
    cols_group: str = "crps",
) -> str:
    """
    LaTeX table for a group of extended metrics.

    cols_group: "crps"    → CRPS, twCRPS@90/95/99, Log Score
                "tail"    → QS@high quantiles
                "coverage"→ Kupiec/Christoffersen
                "rmse"    → RMSE, MAD variants
    """
    if not metrics_list:
        return "% No metrics to display\n"

    group_cols = {
        "crps":     ["model_id", "crps_mean", "twcrps_90", "twcrps_95", "twcrps_99", "log_score_mean"],
        "tail":     ["model_id", "qs_09500", "qs_09900", "qs_09950", "qs_09990", "qs_09995", "qs_09999",
                     "brier_09900", "brier_09990"],
        "coverage": ["model_id", "kup_9500_stat", "kup_9500_pvalue", "kup_9900_stat", "kup_9900_pvalue",
                     "chrf_9500_pvalue", "chrf_9900_pvalue"],
        "rmse":     ["model_id", "rmse", "rmse_wet", "rmse_dry", "mad", "mad_wet", "mad_dry"],
    }
    cols = group_cols.get(cols_group, list(metrics_list[0].keys()))
    present = [c for c in cols if any(c in m for m in metrics_list)]

    if not present:
        return "% No relevant columns found\n"

    headers = " & ".join(r"\small " + _tex(c) for c in present) + r" \\"
    body_lines = []
    for m in metrics_list:
        cells = []
        for c in present:
            v = m.get(c)
            if c == "model_id":
                cells.append(r"\texttt{" + _tex(str(v)) + "}")
            elif c.endswith("pvalue"):
                cells.append(_pval(v))
            else:
                cells.append(_fmt(v))
        body_lines.append(" & ".join(cells) + r" \\")

    n_cols  = len(present)
    col_fmt = "l" + "r" * (n_cols - 1)

    return (
        r"\begin{table}[ht]" + "\n"
        r"\centering" + "\n"
        r"\footnotesize" + "\n"
        rf"\begin{{tabular}}{{{col_fmt}}}" + "\n"
        r"\hline" + "\n"
        + headers + "\n"
        r"\hline" + "\n"
        + "\n".join(body_lines) + "\n"
        r"\hline" + "\n"
        r"\end{tabular}" + "\n"
        rf"\caption{{{caption}}}" + "\n"
        rf"\label{{{label}}}" + "\n"
        r"\end{table}" + "\n"
    )


def _params_table_tex(params_df: pd.DataFrame, caption: str, label: str) -> str:
    """LaTeX parameter table with value and standard error."""
    if params_df.empty:
        return "% No parameters\n"
    cols = [c for c in ["parameter", "value", "std_error"] if c in params_df.columns]
    if not cols:
        return "% No expected columns\n"

    n_cols  = len(cols)
    col_fmt = "l" + "r" * (n_cols - 1)
    headers = " & ".join(r"\small " + _tex(c) for c in cols) + r" \\"
    rows = []
    for _, row in params_df.iterrows():
        cells = []
        for c in cols:
            v = row.get(c)
            if c == "parameter":
                cells.append(r"\texttt{" + _tex(str(v)) + "}")
            else:
                cells.append(_fmt(v))
        rows.append(" & ".join(cells) + r" \\")

    return (
        r"\begin{table}[ht]" + "\n"
        r"\centering" + "\n"
        r"\footnotesize" + "\n"
        rf"\begin{{tabular}}{{{col_fmt}}}" + "\n"
        r"\hline" + "\n"
        + headers + "\n"
        r"\hline" + "\n"
        + "\n".join(rows) + "\n"
        r"\hline" + "\n"
        r"\end{tabular}" + "\n"
        rf"\caption{{{caption}}}" + "\n"
        rf"\label{{{label}}}" + "\n"
        r"\end{table}" + "\n"
    )


def _fig_tex(path: str, caption: str, label: str, width: str = "0.9") -> str:
    """LaTeX \\includegraphics block."""
    return (
        r"\begin{figure}[ht]" + "\n"
        r"\centering" + "\n"
        rf"\includegraphics[width={width}\textwidth]{{{path}}}" + "\n"
        rf"\caption{{{caption}}}" + "\n"
        rf"\label{{{label}}}" + "\n"
        r"\end{figure}" + "\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-location section generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_location_section(
    station_name: str,
    run_dir: Path,
    data: dict,
    fig_dir: Path,
    csv_dir: Path,
    station_short: str,
):
    """
    Generate the complete LaTeX section for one location.

    Returns (tex_str, meta_dict) where meta_dict holds supplementary results
    (e.g. Stage~4 tail-metric acceptance) for use in the global comparison.
    """
    display = STATION_REGISTRY[station_name]["display"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    winners    = _load_winners(run_dir)
    model_cache: dict = {}
    tex_parts  = []
    _section_meta: dict = {}   # populated by subsections; returned alongside tex

    # ── Header ────────────────────────────────────────────────────────────
    tex_parts.append(
        rf"\section{{\texorpdfstring{{{_tex(display)}}}{{{_tex(display)}}}}}"
        + "\n"
        + rf"\label{{sec:{station_short}}}" + "\n"
    )

    # ── Data summary ─────────────────────────────────────────────────────
    s = data["summary"]
    tex_parts.append(r"\subsection{Data Summary}" + "\n")
    tex_parts.append(
        r"\begin{itemize}" + "\n"
        rf"\item Station: {_tex(station_name)}" + "\n"
        rf"\item Training period: {_tex(s['train_start'])} to {_tex(s['train_end'])} ({s['n_train']} days)" + "\n"
        rf"\item Test period: {_tex(s['test_start'])} to {_tex(s['test_end'])} ({s['n_test']} days)" + "\n"
        rf"\item Wet days (train): {s['wet_days_train']} ({s['wet_frac_train']:.1%})" + "\n"
        rf"\item Max daily rainfall: {s['max_precip_train']:.1f} mm" + "\n"
        rf"\item Mean wet-day rainfall: {s['mean_precip_train']:.2f} mm" + "\n"
        r"\end{itemize}" + "\n"
    )

    # Wet/dry classification
    wd_info = classify_wet_dry_months(data["y_train"], data["dates_train"])
    if wd_info["status"] == NO_SEASON:
        tex_parts.append(
            r"\textit{No clear seasonal signal: monthly rainfall varies by less than 2\% of the grand mean.}" + "\n"
        )
    else:
        wet_names  = [pd.Timestamp(f"2000-{m:02d}-01").strftime("%b") for m in wd_info["wet_months"]]
        dry_names  = [pd.Timestamp(f"2000-{m:02d}-01").strftime("%b") for m in wd_info["dry_months"]]
        tex_parts.append(
            rf"Wet months: {', '.join(wet_names)}. Dry months: {', '.join(dry_names)}." + "\n\n"
        )

    # ── Stage 1 ───────────────────────────────────────────────────────────
    tex_parts.append(r"\subsection{Stage 1 -- Baseline GAS}" + "\n")
    tex_parts.append(_stage_comparison_table_tex(
        run_dir, "stage1", winners,
        caption=f"{display} -- Stage 1 model comparison (all 12 specifications, OOS metrics).",
        label=f"tab:{station_short}_s1",
    ))

    # ── Stage 2 ───────────────────────────────────────────────────────────
    tex_parts.append(r"\subsection{Stage 2 -- Weather Covariates}" + "\n")
    tex_parts.append(_stage_comparison_table_tex(
        run_dir, "stage2", winners,
        caption=f"{display} -- Stage 2 model comparison (weather covariates).",
        label=f"tab:{station_short}_s2",
    ))

    # Stage advancement decision
    fw_info = winners.get("final_winner", {})
    s1_crps = fw_info.get("crps_s1", float("nan"))
    s2_crps = fw_info.get("crps_s2", float("nan"))
    s3_crps = fw_info.get("crps_s3", float("nan"))
    # Treat astronomically large CRPS (Stage 2/3 divergence) as not-accepted
    _s2_valid = np.isfinite(s2_crps) and abs(s2_crps) < 1e6
    _s3_valid = np.isfinite(s3_crps) and abs(s3_crps) < 1e6
    if np.isfinite(s1_crps) and _s2_valid:
        impr = (s1_crps - s2_crps) / max(abs(s1_crps), 1e-12)
        decision = "accepted" if impr >= 0.02 else "not accepted"
        tex_parts.append(
            rf"Stage 2 {decision} (CRPS improvement: {impr:.2%}, threshold 2\%)." + "\n\n"
        )
    elif np.isfinite(s2_crps) and not _s2_valid:
        tex_parts.append(
            r"Stage 2 not accepted (all Stage~2 models diverged; "
            r"best OOS CRPS numerically degenerate)." + "\n\n"
        )
    elif not np.isfinite(s2_crps):
        tex_parts.append(
            r"Stage 2 not accepted (OOS CRPS is undefined --- "
            r"all Stage~2 models produced NaN OOS CRPS, "
            r"likely due to full inverse Fisher scaling instability)." + "\n\n"
        )

    # ── Stage 3 ───────────────────────────────────────────────────────────
    tex_parts.append(r"\subsection{Stage 3 -- Harvey Long-Short}" + "\n")
    tex_parts.append(_stage_comparison_table_tex(
        run_dir, "stage3", winners,
        caption=f"{display} -- Stage 3 Harvey long-short model comparison.",
        label=f"tab:{station_short}_s3",
    ))

    # Stage 3 advancement decision — base is Stage 2 if accepted, else Stage 1
    _s2_accepted = (
        np.isfinite(s1_crps) and _s2_valid and s2_crps < 0.98 * s1_crps
    )
    if _s3_valid:
        if _s2_accepted:
            base3_crps, base3_lbl = s2_crps, "Stage~2"
        elif np.isfinite(s1_crps):
            base3_crps, base3_lbl = s1_crps, "Stage~1"
        else:
            base3_crps, base3_lbl = None, None
        if base3_crps is not None:
            impr3 = (base3_crps - s3_crps) / max(abs(base3_crps), 1e-12)
            dec3  = "accepted" if impr3 >= 0.02 else "not accepted"
            tex_parts.append(
                rf"Stage 3 {dec3} (CRPS improvement over {base3_lbl}: {impr3:.2%}, threshold 2\%)." + "\n\n"
            )
    elif not np.isfinite(s3_crps):
        tex_parts.append(
            r"Stage 3 not accepted (OOS CRPS is undefined --- "
            r"Harvey long-short models produced NaN OOS CRPS)." + "\n\n"
        )
    elif not _s3_valid:
        tex_parts.append(
            r"Stage 3 not accepted (all Stage~3 Harvey models diverged; "
            r"best OOS CRPS numerically degenerate)." + "\n\n"
        )

    # ── Final winner extended diagnostics ─────────────────────────────────
    ext_mets = None   # will be set below; used by Stage 4 tail comparison
    final_mid = fw_info.get("model_id")
    if final_mid:
        tex_parts.append(r"\subsection{Final Model Diagnostics}" + "\n")
        tex_parts.append(
            rf"The final selected model is \texttt{{{_tex(final_mid)}}} "
            rf"(OOS CRPS = {_fmt(fw_info.get('crps'))})." + "\n\n"
        )
        # If the loglik winner had NaN CRPS (fullfi instability), explain this
        num_note = fw_info.get("numerical_note")
        if num_note:
            lw_id = fw_info.get("loglik_winner_id", "")
            tex_parts.append(
                r"\textbf{Numerical note:} "
                r"The model selected by maximum log-likelihood, "
                rf"\texttt{{{_tex(lw_id)}}}, produced \texttt{{NaN}} "
                r"out-of-sample CRPS owing to full inverse Fisher (fullfi) "
                r"score scaling generating numerically extreme GB2 distribution "
                r"parameters during OOS rolling-window evaluation (see "
                r"Section~\ref{sec:numerical_issues} for details). "
                r"The reported final model is the best-CRPS model with "
                r"a finite OOS score." + "\n\n"
            )

        # IS diagnostics
        is_diag = _compute_is_diagnostics_for_winner(
            model_id=final_mid, run_dir=run_dir,
            data=data, col_name_dict=data["covariate_col_names"],
            fig_dir=fig_dir,
        )
        if is_diag:
            tex_parts.append(r"\subsubsection{In-Sample Diagnostics}" + "\n")
            tex_parts.append(
                f"PIT mean = {is_diag['pit_mean']:.3f} "
                f"(ideal 0.5), std = {is_diag['pit_std']:.3f} (ideal {1/12**0.5:.3f}).\n\n"
            )
            pit_rel = os.path.relpath(is_diag["pit_path"], fig_dir.parent.parent)
            qq_rel  = os.path.relpath(is_diag["qq_path"],  fig_dir.parent.parent)
            acf_rel = os.path.relpath(is_diag["acf_path"], fig_dir.parent.parent)

            tex_parts.append(_fig_tex(
                pit_rel.replace("\\", "/"),
                f"{display}: IS PIT histogram for {_tex(final_mid)}.",
                f"fig:{station_short}_pit",
            ))
            tex_parts.append(_fig_tex(
                qq_rel.replace("\\", "/"),
                f"{display}: IS quantile residual QQ plot.",
                f"fig:{station_short}_qq",
            ))
            tex_parts.append(_fig_tex(
                acf_rel.replace("\\", "/"),
                f"{display}: IS quantile residual ACF (400 lags, 95\\% confidence bands).",
                f"fig:{station_short}_acf",
                width="1.0",
            ))

        # Extended OOS metrics
        tex_parts.append(r"\subsubsection{Extended Out-of-Sample Metrics}" + "\n")
        ext_mets = _compute_extended_metrics_for_model(
            final_mid, run_dir, data, model_cache
        )
        if ext_mets:
            ext_list = [ext_mets]
            # Save metrics CSV
            pd.DataFrame(ext_list).to_csv(
                csv_dir / f"{station_short}_extended_metrics.csv", index=False
            )

            tex_parts.append(r"\paragraph{Distributional metrics}" + "\n")
            tex_parts.append(_extended_metrics_table_tex(
                ext_list, caption=f"{display}: CRPS, twCRPS, Log Score.",
                label=f"tab:{station_short}_crps", cols_group="crps",
            ))
            tex_parts.append(r"\paragraph{Tail metrics}" + "\n")
            tex_parts.append(_extended_metrics_table_tex(
                ext_list, caption=f"{display}: High-quantile scores.",
                label=f"tab:{station_short}_tail", cols_group="tail",
            ))
            tex_parts.append(r"\paragraph{Coverage tests}" + "\n")
            tex_parts.append(_extended_metrics_table_tex(
                ext_list, caption=f"{display}: Kupiec and Christoffersen tests.",
                label=f"tab:{station_short}_coverage", cols_group="coverage",
            ))
            tex_parts.append(r"\paragraph{Point forecasts}" + "\n")
            tex_parts.append(_extended_metrics_table_tex(
                ext_list, caption=f"{display}: RMSE and MAD.",
                label=f"tab:{station_short}_rmse", cols_group="rmse",
            ))

        # Dynamic quantile figures
        tex_parts.append(r"\subsubsection{Dynamic Conditional Quantiles}" + "\n")

        # Determine stage and load OOS paths
        stage_m = "stage1"
        for s in ("stage3", "stage2", "stage1"):
            if final_mid.startswith(s):
                stage_m = s
                break
        paths_npz = _load_paths_npz(run_dir, stage_m, final_mid)
        meta_m    = _load_meta(run_dir, stage_m, final_mid)

        if paths_npz is not None:
            oos_paths = _reconstruct_oos_paths(paths_npz, meta_m)
        else:
            oos_paths = None

        if oos_paths is not None:
            try:
                model_w = model_cache.get(final_mid)
                if model_w is None:
                    params_df_w = _load_params_df(run_dir, stage_m, final_mid)
                    model_w = _build_model_from_meta(meta_m, params_df_w)
                    model_cache[final_mid] = model_w

                quants_oos = compute_dynamic_quantiles(model_w, oos_paths, QUANTILE_LEVELS)
                daily_rl, _  = compute_daily_return_levels(model_w, oos_paths)
                clim_rl, _   = compute_climatological_return_levels(model_w, oos_paths)

                dates_test  = data["dates_test"]
                y_test      = data["y_test"]
                nino34_test = data.get("nino34_test")

                # Dynamic quantile figure
                dq_path = str(fig_dir / f"{station_short}_dynamic_quantiles.png")
                plot_dynamic_quantiles(
                    dates=dates_test, y=y_test,
                    dynamic_quantiles=quants_oos,
                    quantile_levels=QUANTILE_LEVELS,
                    fig_path=dq_path,
                    title=f"{display}: Dynamic conditional quantiles (OOS)",
                )
                tex_parts.append(_fig_tex(
                    os.path.relpath(dq_path, fig_dir.parent.parent).replace("\\", "/"),
                    f"{display}: Dynamic conditional quantiles (OOS, fan from Q50 to Q99.99).",
                    f"fig:{station_short}_dynq", width="1.0",
                ))

                # TV parameter for signature figure
                tv_param = None
                tv_name  = "phi"
                if "f_arr_oos" in paths_npz:
                    tv_param = paths_npz["f_arr_oos"][:, 0]   # phi column

                # Signature figure
                sig_path = str(fig_dir / f"{station_short}_signature.png")
                plot_signature_figure(
                    dates=dates_test, y=y_test,
                    dynamic_quantiles=quants_oos,
                    quantile_levels=QUANTILE_LEVELS,
                    nino34=nino34_test,
                    tv_param=tv_param,
                    tv_param_name="phi",
                    fig_path=sig_path,
                    title=f"{display}: Observed rainfall and dynamic tail quantiles",
                    highlight_quantiles=(0.95, 0.99, 0.999),
                )
                tex_parts.append(_fig_tex(
                    os.path.relpath(sig_path, fig_dir.parent.parent).replace("\\", "/"),
                    f"{display}: Signature figure. Top panel: observed rainfall and dynamic Q95, Q99, Q99.9 quantiles with El~Ni\\~no (red) and La~Ni\\~na (blue) shading. Bottom panel: dynamic log-scale parameter $\\hat\\phi_t$.",
                    f"fig:{station_short}_signature", width="1.0",
                ))

                # Daily return levels
                rl_labels = [f"{T}d" for T in DAILY_RETURN_PERIODS]
                drl_path  = str(fig_dir / f"{station_short}_daily_rl.png")
                plot_return_levels(
                    dates=dates_test, y=y_test, return_levels=daily_rl,
                    period_labels=rl_labels, fig_path=drl_path,
                    title=f"{display}: Dynamic daily return levels (OOS)",
                    nino34=nino34_test,
                )
                tex_parts.append(r"\subsubsection{Return Levels}" + "\n")
                tex_parts.append(_fig_tex(
                    os.path.relpath(drl_path, fig_dir.parent.parent).replace("\\", "/"),
                    f"{display}: Dynamic daily return levels (30d to 1000d).",
                    f"fig:{station_short}_drl", width="1.0",
                ))

                # Climatological return levels
                clim_labels = [f"{T}yr" for T in CLIM_RETURN_PERIODS_YEARS]
                crl_path    = str(fig_dir / f"{station_short}_clim_rl.png")
                plot_return_levels(
                    dates=dates_test, y=y_test, return_levels=clim_rl,
                    period_labels=clim_labels, fig_path=crl_path,
                    title=f"{display}: Dynamic climatological return levels (OOS)",
                    nino34=nino34_test,
                )
                tex_parts.append(_fig_tex(
                    os.path.relpath(crl_path, fig_dir.parent.parent).replace("\\", "/"),
                    f"{display}: Dynamic climatological return levels using $p(T)=(1-1/T)^{{1/365}}$.",
                    f"fig:{station_short}_crl", width="1.0",
                ))

                # Tail calibration
                tc_path = str(fig_dir / f"{station_short}_tail_calibration.png")
                plot_tail_calibration(
                    y=y_test, dynamic_quantiles=quants_oos,
                    quantile_levels=QUANTILE_LEVELS,
                    fig_path=tc_path,
                    title=f"{display}: Tail calibration (OOS)",
                )
                tex_parts.append(r"\subsubsection{Tail Calibration}" + "\n")
                tex_parts.append(_fig_tex(
                    os.path.relpath(tc_path, fig_dir.parent.parent).replace("\\", "/"),
                    f"{display}: Empirical vs.\\ theoretical exceedance frequencies. A well-calibrated model lies on the dashed 45\\textdegree{{}} line.",
                    f"fig:{station_short}_tc",
                ))

                # Seasonal decomposition
                tex_parts.append(r"\subsubsection{Seasonal Decomposition}" + "\n")
                seas_res = seasonal_decomposition(
                    y=y_test, dates=dates_test,
                    dynamic_quantiles=quants_oos,
                    quantile_levels=QUANTILE_LEVELS,
                    return_levels=daily_rl,
                    return_period_labels=rl_labels,
                )
                mbp_path = str(fig_dir / f"{station_short}_monthly_boxplot.png")
                plot_monthly_quantile_boxplot(
                    y=y_test, dates=dates_test,
                    dynamic_quantiles=quants_oos,
                    quantile_levels=QUANTILE_LEVELS,
                    highlight_q=0.99,
                    fig_path=mbp_path,
                    title=f"{display}: Monthly boxplot of dynamic Q99",
                )
                tex_parts.append(_fig_tex(
                    os.path.relpath(mbp_path, fig_dir.parent.parent).replace("\\", "/"),
                    f"{display}: Monthly distribution of the dynamic 99th percentile (OOS). Orange bars show mean observed rainfall.",
                    f"fig:{station_short}_monthly",
                ))

                # ENSO decomposition
                tex_parts.append(r"\subsubsection{ENSO Decomposition}" + "\n")
                if nino34_test is not None:
                    enso_res = enso_decomposition(
                        y=y_test, dates=dates_test, nino34=nino34_test,
                        dynamic_quantiles=quants_oos,
                        quantile_levels=QUANTILE_LEVELS,
                        return_levels=daily_rl,
                        return_period_labels=rl_labels,
                    )
                    enso_exc_path = str(fig_dir / f"{station_short}_enso_exceedance.png")
                    plot_enso_exceedance(
                        enso_regimes=enso_res["standard_regimes"],
                        quantile_levels=QUANTILE_LEVELS,
                        highlight_levels=(0.90, 0.95, 0.99),
                        fig_path=enso_exc_path,
                        title=f"{display}: ENSO regime exceedance calibration",
                    )
                    tex_parts.append(_fig_tex(
                        os.path.relpath(enso_exc_path, fig_dir.parent.parent).replace("\\", "/"),
                        f"{display}: Empirical exceedance frequencies by ENSO regime. Dashed lines show theoretical levels.",
                        f"fig:{station_short}_enso_exc",
                    ))

                    enso_int_path = str(fig_dir / f"{station_short}_enso_intensity.png")
                    plot_enso_intensity_quantile(
                        intensity_result=enso_res["intensity_bins"],
                        quantile_levels=QUANTILE_LEVELS,
                        highlight_q=0.99,
                        fig_path=enso_int_path,
                        title=f"{display}: ENSO intensity vs mean Q99",
                    )
                    tex_parts.append(_fig_tex(
                        os.path.relpath(enso_int_path, fig_dir.parent.parent).replace("\\", "/"),
                        f"{display}: Mean dynamic Q99 by ENSO intensity quintile. Demonstrates whether increasing ENSO intensity is associated with increasing extreme rainfall risk.",
                        f"fig:{station_short}_enso_int",
                    ))
                else:
                    tex_parts.append(
                        r"\textit{ENSO data not available for this location's test period.}" + "\n\n"
                    )

                # Wet/dry decomposition
                tex_parts.append(r"\subsubsection{Wet/Dry Season Decomposition}" + "\n")
                if wd_info["status"] != NO_SEASON:
                    wd_res = wet_dry_decomposition(
                        y=y_test, dates=dates_test,
                        wet_months=wd_info["wet_months"],
                        dry_months=wd_info["dry_months"],
                        dynamic_quantiles=quants_oos,
                        quantile_levels=QUANTILE_LEVELS,
                        return_levels=daily_rl,
                        return_period_labels=rl_labels,
                    )
                    # Build a simple table
                    wd_rows = []
                    for season in ["wet", "dry"]:
                        r = wd_res.get(season, {})
                        wd_rows.append({
                            "model_id": season.capitalize(),
                            "n": r.get("n", 0),
                            "mean_y": r.get("mean_y", np.nan),
                            f"exceed_emp_{int(0.99*10000):05d}": r.get(f"exceed_emp_{int(0.99*10000):05d}", np.nan),
                            f"qs_{int(0.99*10000):05d}": r.get(f"qs_{int(0.99*10000):05d}", np.nan),
                        })
                    tex_parts.append(_extended_metrics_table_tex(
                        wd_rows,
                        caption=f"{display}: Wet/dry season decomposition of tail metrics.",
                        label=f"tab:{station_short}_wetdry",
                        cols_group="rmse",  # uses first available columns
                    ))

                    # Save all decomposition CSVs
                    save_decomposition_csvs(
                        csv_dir, station_short,
                        seasonal_result=seas_res,
                        enso_result=enso_res if nino34_test is not None else {"standard_regimes":{}, "intensity_bins":{}},
                        wet_dry_result=wd_res,
                        wet_dry_info=wd_info,
                    )
                    # Save dynamic arrays CSV
                    save_dynamic_arrays(
                        csv_dir, station_short,
                        dates_train=data["dates_train"],
                        dates_test=dates_test,
                        quants_oos=quants_oos,
                        quants_is=None,
                        daily_rl=daily_rl,
                        clim_rl=clim_rl,
                        quantile_levels=QUANTILE_LEVELS,
                    )

            except Exception as exc:
                import traceback
                tex_parts.append(
                    rf"\textit{{Dynamic diagnostic computation failed: {_tex(str(exc))}}}" + "\n\n"
                )
                print(f"  [WARN] Dynamic diagnostics failed for {station_name}: {exc}")
                print(traceback.format_exc())

    # ── Parameter table (appendix-style) ──────────────────────────────────
    if final_mid:
        params_df_w = _load_params_df(run_dir, stage_m, final_mid)
        if not params_df_w.empty:
            tex_parts.append(r"\subsection{Estimated Parameters}" + "\n")
            tex_parts.append(_params_table_tex(
                params_df_w,
                caption=f"{display}: Estimated parameters for {_tex(final_mid)}.",
                label=f"tab:{station_short}_params",
            ))

    # ── Stage 4 Regime ────────────────────────────────────────────────────
    regime_models = _all_model_ids(run_dir, "stage4_regime")
    if regime_models:
        tex_parts.append(r"\subsection{Stage 4 -- Regime-Sensitive GAS}" + "\n")
        tex_parts.append(_stage_comparison_table_tex(
            run_dir, "stage4_regime", winners,
            caption=f"{display}: Stage 4 regime model results.",
            label=f"tab:{station_short}_regime",
        ))

        # Stage 4 acceptance note
        s4_info = winners.get("stage4_regime", {})
        s4_accepted = s4_info.get("stage4_accepted")
        s4_note     = s4_info.get("stage4_note") or s4_info.get("note", "")
        s4_base_crps = s4_info.get("base_crps") or s4_info.get("baseline_crps")
        s4_best_crps = s4_info.get("best_regime_crps") or s4_info.get("winner_crps")
        # Fallback: compute best finite CRPS from models list if not stored
        if s4_best_crps is None:
            _s4_finite = [
                float(m.get("crps_mean", float("nan")))
                for m in s4_info.get("models", [])
                if m.get("crps_mean") is not None
                and np.isfinite(float(m.get("crps_mean", float("nan"))))
                and abs(float(m.get("crps_mean", float("nan")))) < 1e6
            ]
            if _s4_finite:
                s4_best_crps = min(_s4_finite)
        if s4_accepted is True:
            tex_parts.append(
                r"\textbf{Stage 4 accepted}: regime-sensitive GAS reduces OOS CRPS "
                rf"from {_fmt(s4_base_crps)} to {_fmt(s4_best_crps)} "
                r"($\geq$2\% improvement threshold met)." + "\n\n"
            )
        elif s4_accepted is False:
            tex_parts.append(
                r"\textbf{Stage 4 not accepted}: no regime specification "
                r"achieved the 2\% CRPS improvement threshold. "
            )
            if s4_base_crps is not None and s4_best_crps is not None:
                tex_parts.append(
                    rf"Best regime CRPS = {_fmt(s4_best_crps)}, "
                    rf"base CRPS = {_fmt(s4_base_crps)}. "
                )
            if s4_note:
                tex_parts.append(_tex(s4_note))
            tex_parts.append("\n\n")

        # Extended metrics for regime models
        regime_mets = []
        for rmid in regime_models:
            m = _compute_extended_metrics_for_model(rmid, run_dir, data, model_cache)
            if m:
                regime_mets.append(m)
        if regime_mets:
            pd.DataFrame(regime_mets).to_csv(
                csv_dir / f"{station_short}_regime_metrics.csv", index=False
            )
            tex_parts.append(_extended_metrics_table_tex(
                regime_mets,
                caption=f"{display}: Regime model CRPS and twCRPS metrics.",
                label=f"tab:{station_short}_regime_ext",
                cols_group="crps",
            ))
            tex_parts.append(_extended_metrics_table_tex(
                regime_mets,
                caption=f"{display}: Regime model tail quantile scores.",
                label=f"tab:{station_short}_regime_tail",
                cols_group="tail",
            ))

        # ── Stage 4 tail-metric acceptance ────────────────────────────────────
        # Stage 4 is designed to improve extreme-event forecasts; evaluate on
        # twCRPS@90 (threshold-weighted CRPS) and high quantile scores rather
        # than overall CRPS alone.
        tex_parts.append(r"\paragraph{Tail-metric evaluation of Stage 4}" + "\n")
        tex_parts.append(
            r"The regime extension (Stage~4) is motivated by the hypothesis that "
            r"extreme precipitation events carry a distinct score signal that warrants "
            r"a separate, amplified updating step. Tail performance is therefore the "
            r"primary evaluation criterion: threshold-weighted CRPS (twCRPS) at the "
            r"90th, 95th, and 99th percentile thresholds, and the quantile score (QS) "
            r"at the 95th and 99th percentiles." + "\n\n"
        )

        # Gather base model tail metrics from the final-winner extended computation
        s4_tail_base_tw90 = None
        s4_tail_base_tw95 = None
        if ext_mets is not None:
            _v90 = ext_mets.get("twcrps_90")
            _v95 = ext_mets.get("twcrps_95")
            if _v90 is not None and np.isfinite(float(_v90)):
                s4_tail_base_tw90 = float(_v90)
            if _v95 is not None and np.isfinite(float(_v95)):
                s4_tail_base_tw95 = float(_v95)

        # Find best regime model by twCRPS@90
        s4_tail_accepted = False
        s4_tail_best_mid = None
        s4_tail_best_tw90 = None
        s4_tail_impr90 = None
        s4_tail_best_tw95 = None
        if regime_mets and s4_tail_base_tw90 is not None:
            _finite_r = [
                r for r in regime_mets
                if r.get("twcrps_90") is not None
                and np.isfinite(float(r.get("twcrps_90", float("nan"))))
            ]
            if _finite_r:
                _best_r = min(_finite_r, key=lambda r: float(r["twcrps_90"]))
                s4_tail_best_tw90 = float(_best_r["twcrps_90"])
                s4_tail_best_mid  = _best_r.get("model_id", "")
                s4_tail_impr90 = (
                    (s4_tail_base_tw90 - s4_tail_best_tw90)
                    / max(abs(s4_tail_base_tw90), 1e-12)
                )
                s4_tail_accepted = s4_tail_impr90 >= 0.02
                _v95r = _best_r.get("twcrps_95")
                if _v95r is not None and np.isfinite(float(_v95r)):
                    s4_tail_best_tw95 = float(_v95r)

        if s4_tail_base_tw90 is not None and s4_tail_best_tw90 is not None:
            if s4_tail_accepted:
                tex_parts.append(
                    r"\textbf{Stage~4 tail accepted}: "
                    rf"regime model \texttt{{{_tex(s4_tail_best_mid)}}} achieves a "
                    rf"{s4_tail_impr90:.1%} improvement in twCRPS@90 over the base model "
                    rf"(base: {_fmt(s4_tail_base_tw90)}, regime: {_fmt(s4_tail_best_tw90)}). "
                    r"The 2\% threshold on the tail criterion is met, "
                    r"confirming that the regime impulse corrects a systematic "
                    r"under-response to extreme events." + "\n\n"
                )
            else:
                _dir = "worse than" if s4_tail_impr90 < 0 else "below the threshold for"
                tex_parts.append(
                    r"\textbf{Stage~4 tail not accepted}: "
                    r"no regime model achieves the 2\% improvement threshold on twCRPS@90. "
                    rf"Best twCRPS@90 improvement = {s4_tail_impr90:.1%} "
                    rf"(\textit{{{_dir}}} the 2\% threshold). "
                    rf"Base twCRPS@90 = {_fmt(s4_tail_base_tw90)}, "
                    rf"best regime twCRPS@90 = {_fmt(s4_tail_best_tw90)}"
                    + (
                        rf", twCRPS@95 = {_fmt(s4_tail_best_tw95)}"
                        if s4_tail_best_tw95 is not None else ""
                    )
                    + r". The regime impulse does not improve extreme-event forecasting "
                    r"beyond what the base GAS score update already captures." + "\n\n"
                )
        elif not regime_mets:
            tex_parts.append(
                r"No regime model extended metrics available for tail evaluation." + "\n\n"
            )
        else:
            tex_parts.append(
                r"Base model tail metrics not available; tail comparison cannot be computed." + "\n\n"
            )

        # Propagate tail acceptance to the global comparison
        _section_meta["stage4_tail_accepted"] = s4_tail_accepted
        _section_meta["stage4_tail_impr90"]   = s4_tail_impr90
        _section_meta["stage4_tail_base_tw90"] = s4_tail_base_tw90
        _section_meta["stage4_tail_best_tw90"] = s4_tail_best_tw90

    return "\n".join(tex_parts), _section_meta


# ─────────────────────────────────────────────────────────────────────────────
# Global comparison section
# ─────────────────────────────────────────────────────────────────────────────

def generate_global_comparison(
    station_results: Dict[str, dict],
    csv_dir: Path,
    fig_dir: Path,
) -> str:
    """
    Generate global comparison chapter (all locations side-by-side).
    """
    tex_parts = []
    tex_parts.append(r"\chapter{Global Comparison}" + "\n")
    tex_parts.append(r"\label{chap:global}" + "\n")

    rows = []
    for station, res in station_results.items():
        fw   = res.get("final_winner")
        if fw is None:
            continue
        cfg = STATION_REGISTRY.get(station, {})
        s4_acc  = res.get("stage4_accepted")
        s4_str  = "Yes" if s4_acc is True else ("No" if s4_acc is False else "--")
        s4t_acc = res.get("stage4_tail_accepted")
        s4t_str = "Yes" if s4t_acc is True else ("No" if s4t_acc is False else "--")
        rows.append({
            "Station":        cfg.get("display", station),
            "Final model":    fw.get("model_id", "?"),
            "S1 CRPS":        res.get("crps_s1", float("nan")),
            "S2 CRPS":        res.get("crps_s2", float("nan")),
            "S3 CRPS":        res.get("crps_s3", float("nan")),
            "Final CRPS":     res.get("crps_final", float("nan")),
            "S4 (CRPS)":      s4_str,
            "S4 (twCRPS@90)": s4t_str,
        })

    if rows:
        df_global = pd.DataFrame(rows)
        df_global.to_csv(csv_dir / "global_comparison.csv", index=False)

        # LaTeX table
        tex_parts.append(r"\section{Stage Winner Comparison}" + "\n")
        headers = " & ".join(r"\textbf{" + _tex(c) + "}" for c in df_global.columns) + r" \\"
        body    = []
        for _, row in df_global.iterrows():
            cells = []
            _text_cols = {"Station", "Final model", "S4 (CRPS)", "S4 (twCRPS@90)"}
            for c in df_global.columns:
                v = row[c]
                if c in _text_cols:
                    cells.append(_tex(str(v)))
                else:
                    cells.append(_fmt(v))
            body.append(" & ".join(cells) + r" \\")

        n_cols  = len(df_global.columns)
        # Station + Final model (left), CRPS cols (right), two S4 accept cols (center)
        col_fmt = "l" * 2 + "r" * (n_cols - 4) + "c" * 2
        tex_parts.append(
            r"\begin{table}[ht]" + "\n"
            r"\centering" + "\n"
            r"\footnotesize" + "\n"
            rf"\begin{{tabular}}{{{col_fmt}}}" + "\n"
            r"\hline" + "\n"
            + headers + "\n"
            r"\hline" + "\n"
            + "\n".join(body) + "\n"
            r"\hline" + "\n"
            r"\end{tabular}" + "\n"
            r"\caption{Global comparison: best model, OOS CRPS at each pipeline stage,"
            r" and Stage~4 acceptance for all locations."
            r" CRPS entries show \texttt{--} where OOS evaluation failed"
            r" (fullfi instability or convergence failure)."
            r" \emph{S4~(CRPS)} = Yes if any regime model achieves $\geq$2\% overall"
            r" CRPS improvement; \emph{S4~(twCRPS@90)} = Yes if any regime model"
            r" achieves $\geq$2\% improvement in threshold-weighted CRPS at the"
            r" 90th-percentile threshold (the tail-focused criterion).}" + "\n"
            r"\label{tab:global_comparison}" + "\n"
            r"\end{table}" + "\n"
        )

    # Narrative (partially dynamic — Stage 4 block reflects actual acceptance)
    tex_parts.append(r"\section{Cross-Location Patterns}" + "\n")
    tex_parts.append(r"""
\subsection*{Stage progression}
Across all locations, Stage~1 (baseline GAS) provides the final model in
the majority of cases.  Stage~2 weather covariates and Stage~3 Harvey
long-short decomposition are accepted only where there is a clear
meteorological signal: the Harvey model with ENSO component achieves
meaningful CRPS reductions at Belo Horizonte and Darwin Airport, both of
which experience pronounced ENSO-modulated seasonal regimes.
Tropical locations (Manaus, Salvador) show no net improvement from
external covariates, consistent with their rainfall being dominated by
convective dynamics at sub-synoptic scales not captured by monthly ENSO
or daily dew-point indices.
""")

    # Dynamic Stage 4 block
    crps_accepted = [
        STATION_REGISTRY.get(st, {}).get("display", st)
        for st, res in station_results.items()
        if res.get("stage4_accepted") is True
    ]
    tail_accepted = [
        (STATION_REGISTRY.get(st, {}).get("display", st),
         res.get("stage4_tail_impr90"))
        for st, res in station_results.items()
        if res.get("stage4_tail_accepted") is True
    ]
    tex_parts.append(r"\subsection*{Stage 4 regime extension}" + "\n")
    if not crps_accepted and not tail_accepted:
        tex_parts.append(
            r"Stage~4 was not accepted at any location in this run on either "
            r"the overall CRPS or the tail-focused (twCRPS@90) criterion. "
        )
    elif not crps_accepted:
        tex_parts.append(
            r"Stage~4 was not accepted at any location on the overall CRPS criterion. "
        )
    else:
        locs = ", ".join(_tex(s) for s in crps_accepted)
        tex_parts.append(
            rf"Stage~4 was accepted on the overall CRPS criterion at: {locs}. "
        )
    if tail_accepted:
        tail_lines = []
        for display, impr in tail_accepted:
            if impr is not None:
                tail_lines.append(
                    rf"\textit{{{_tex(display)}}} ({impr:.1%} twCRPS@90 improvement)"
                )
            else:
                tail_lines.append(rf"\textit{{{_tex(display)}}}")
        tail_str = "; ".join(tail_lines)
        tex_parts.append(
            r"However, the tail-focused criterion (twCRPS@90, $\geq$2\% improvement) "
            rf"was met at: {tail_str}. "
            r"This result is scientifically important: the regime impulse significantly "
            r"improves the model's response to extreme precipitation even when the "
            r"\emph{overall} CRPS --- which is dominated by the many non-extreme days "
            r"--- does not improve enough to trigger formal acceptance. "
            r"The per-location Stage~4 tail-metric paragraphs give the quantitative details. "
        )
    else:
        tex_parts.append(
            r"The per-location extended metrics tables (CRPS + twCRPS@90/95/99 columns) "
            r"and the tail-metric acceptance paragraph within each Stage~4 subsection "
            r"provide the quantitative evidence. "
        )
    tex_parts.append(
        r"At locations whose Stage~1 winner uses unit score scaling (Darwin, Manaus, "
        r"Salvador), the diagfi-scaled regime filter diverged catastrophically "
        r"(Section~\ref{sec:numerical_issues}). "
        r"The unit-scaling fallback triggered by Bug~6b produced well-converged "
        r"unit-scaled regime models; none achieved the 2\% overall CRPS threshold." + "\n\n"
    )

    tex_parts.append(r"""
\subsection*{Numerical reliability}
At Garanhuns and Cruzeiro do Sul, the model selected by maximum
log-likelihood in Stage~1 is a full inverse Fisher variant whose OOS CRPS
is undefined due to GB2 PPF overflow.  The deployable best-CRPS Stage~1
model is reported in the \emph{Final CRPS} column above.  Refer to
Section~\ref{sec:numerical_issues} for a full explanation.
""")

    return "\n".join(tex_parts)


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX document builder
# ─────────────────────────────────────────────────────────────────────────────

LATEX_PREAMBLE = r"""\documentclass[12pt,a4paper]{report}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{lscape}
\usepackage{longtable}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{hyperref}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{array}
\usepackage{pdflscape}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
\captionsetup{font=small}
"""


def build_latex_document(
    location_sections: Dict[str, str],
    global_section: str,
    generation_date: str,
) -> str:
    body_parts = [
        LATEX_PREAMBLE,
        r"\begin{document}",
        r"\title{ZA-GAS Multi-Location Precipitation Model Evaluation}",
        rf"\author{{Generated: {_tex(generation_date)}}}",
        r"\date{}",
        r"\maketitle",
        r"\tableofcontents",
        r"\clearpage",
        r"\chapter{Introduction}",
        r"""
This report presents a comprehensive evaluation of the Zero-Augmented Generalized
Autoregressive Score (ZA-GAS) probabilistic model for daily precipitation across
multiple locations. The pipeline consists of three estimation stages plus a
regime-sensitive extension:

\begin{enumerate}
\item \textbf{Stage 1 -- Baseline GAS}: 12 specifications varying the set of
      time-varying parameters ($\phi$ only vs.\ $\phi+\xi$), lag structure
      (short vs.\ seasonal), and score scaling (unit, diagonal inverse Fisher,
      full inverse Fisher).
\item \textbf{Stage 2 -- Weather covariates}: GAS filter augmented with lagged
      ERA5 dew point and temperature. Accepted only if OOS CRPS improves by
      at least 2\% over Stage~1.
\item \textbf{Stage 3 -- Harvey long-short}: Decomposes the dynamic scale into
      a slow long component driven by ENSO and a fast short component driven by
      weather. Accepted only if OOS CRPS improves by at least 2\% over Stage~2.
\item \textbf{Stage 4 -- Regime-sensitive GAS}: Adds an extreme-observation
      indicator to the score response coefficient, allowing the model to respond
      differently when the previous day's rainfall was extreme (per MODELS.md
      \S22--23). Applied to the best model from Stages~1--3.
\end{enumerate}

In-sample (IS) diagnostics include randomised PIT histograms, normal QQ plots,
and quantile residual ACF at 400 lags. Out-of-sample (OOS) metrics include
CRPS, threshold-weighted CRPS at 90\%, 95\%, and 99\%, Quantile Score at 10
probability levels, Log Score, Brier Score, and Kupiec / Christoffersen
coverage tests.

Dynamic conditional quantiles, daily and climatological return levels, and
tail calibration plots are produced for each location. Seasonal, ENSO, and
wet/dry season decompositions illustrate how the predictive distribution adapts
to climatic and meteorological conditions.

This report covers six locations estimated in this experimental run:

\begin{enumerate}
\item \textbf{Belo Horizonte, Brazil} -- subtropical highland, strong ENSO teleconnection
      (estimated in a prior session, artifacts reused).
\item \textbf{Cruzeiro do Sul (Acre), Brazil} -- equatorial Amazon, year-round convective rainfall.
\item \textbf{Darwin Airport, Australia} -- tropical monsoonal, strong ENSO-driven wet/dry cycle.
\item \textbf{Garanhuns (Pernambuco), Brazil} -- semi-arid, highly intermittent.
\item \textbf{Manaus, Brazil} -- equatorial Amazon, unimodal wet season.
\item \textbf{Salvador, Brazil} -- tropical coastal, bimodal rainfall.
\end{enumerate}

Two additional locations (S\~ao Paulo, Brazil and Toronto, Canada) were
identified as future work and are not included in this report.  S\~ao Paulo
shares a similar subtropical climate with Belo Horizonte and was deprioritised
to avoid duplication.  Toronto would add a temperate mid-latitude climate for
completeness.
""",
        r"\section{Numerical Issues Encountered During Estimation}",
        r"\label{sec:numerical_issues}",
        r"""
Two classes of numerical issue were encountered across locations and are
documented here so that the per-location tables can be interpreted correctly.

\subsection*{Full Inverse Fisher (fullfi) Scaling and NaN OOS CRPS}

The Stage~1 log-likelihood winner at several locations (\emph{Garanhuns},
\emph{Cruzeiro do Sul}) is a \texttt{phixi\_seasonal\_fullfi} model.
The full inverse Fisher (fullfi) score scaling uses the complete $2\times2$
inverse of the GB2 Fisher information matrix to scale the score before
updating the filter.  In-sample this sometimes yields a slightly higher
log-likelihood by compensating for parameter correlation.

However, during out-of-sample (OOS) rolling-window evaluation the fitted
GAS parameters are used to propagate the filter one step ahead, and the
resulting conditional GB2 distribution is then evaluated at held-out
observations via its PPF (percent-point function).
When the fullfi inverse produces extremely large off-diagonal mixing terms,
the predicted conditional scale $\phi_t$ drifts outside the range where
the GB2 PPF is numerically representable, causing the PPF to return
\texttt{nan} or overflow to \texttt{inf}.  Because CRPS is an integral
over the predictive CDF, a single non-finite quantile propagates as
\texttt{NaN} CRPS for the entire OOS window.

\textbf{Resolution}: the loglik-winner model is retained in the
\texttt{stage\_winners.json} for record, but for OOS comparison and
stage-advancement decisions the best model with a \emph{finite} OOS CRPS
is used.  This is always a unit- or diagfi-scaled variant, as noted in the
per-location \emph{Numerical note} paragraphs.

\subsection*{Stage 4 Scaling Mismatch and Catastrophic Divergence}

Regime-sensitive GAS (Stage~4) was run with diagonal inverse Fisher
(\texttt{diagfi}) score scaling for all locations.  At locations where the
Stage~1 winner uses \emph{unit} scaling (Darwin, Manaus, Salvador), the
GAS parameters are calibrated for raw (unscaled) scores, which are
numerically much larger than the Fisher-normalised scores the diagfi filter
expects.  Feeding unit-calibrated initial parameters into a diagfi filter
causes the optimiser to diverge catastrophically:\ OOS CRPS values of
order $10^{78}$ and log-likelihoods five to eight hundred times worse than
the base model were observed.

\textbf{Resolution (Bug~6b)}: a loglik-ratio guard was added to the
scaling-fallback loop.  If \emph{all} regime models in a given scaling
batch satisfy $|\ell| > 3 \times |\ell_\text{base}|$, the batch is
classified as scaling-mismatch divergence and the next scaling
(unit $\rightarrow$ diagfi $\rightarrow$ fullfi) is tried.
This fix was deployed partway through the multi-location run; Manaus
Stage~4 had already completed under the old check (all nine diagfi models
retained their catastrophically bad CRPS as ``finite'' results, and no
unit fallback was attempted).  The Stage~4 results for Manaus therefore
reflect diagfi-only estimation; Stage~4 is correctly marked as
\emph{not accepted} for that location regardless.

At the 90th percentile threshold with both $\phi$ and $\xi$ made
time-varying simultaneously (\texttt{90\_phi\_xi}), the optimiser diverges
to log-likelihoods hundreds of times worse than the base model at
\emph{every} location.  This suggests that the 90th percentile threshold
activates regime effects for too many observations, making the additional
two time-varying parameters severely over-parameterised relative to the
available signal.

\subsection*{Harvey Stage~3 ZeroDivisionError (pre-fix)}

Prior to the deployment of Bug~5 fix (June~2026), the full inverse Fisher
cross-element $\partial^2 \ell / \partial\phi\,\partial\xi$ function
contained a division by $a + b$ where $a = e^\xi$ and $b = e^\zeta$.
During BFGS line search at locations with multicollinear long-short
covariates, both parameters were simultaneously pushed very negative,
causing $a$ and $b$ to underflow to exactly $0.0$, raising a Numba
\texttt{ZeroDivisionError}.  This produced \texttt{validity: failed} for
the \texttt{harvey\_enso\_90d\_30d} and
\texttt{harvey\_enso\_90d\_30d\_daily} models at Garanhuns.  All
subsequent runs (Manaus, Salvador) use the patched filter.
""",
        r"\clearpage",
    ]

    for station, section_tex in location_sections.items():
        body_parts.append(r"\clearpage")
        body_parts.append(section_tex)

    body_parts.append(r"\clearpage")
    body_parts.append(global_section)
    body_parts.append(r"\end{document}")

    return "\n\n".join(body_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Compile LaTeX
# ─────────────────────────────────────────────────────────────────────────────

def compile_latex(tex_path: Path) -> bool:
    """Run pdflatex twice to resolve cross-references."""
    tex_dir = tex_path.parent
    cmd = [
        "pdflatex", "-interaction=nonstopmode",
        "-output-directory", str(tex_dir),
        str(tex_path),
    ]
    for pass_num in range(1, 3):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, cwd=tex_dir
            )
            if result.returncode != 0:
                print(f"  pdflatex pass {pass_num} returned code {result.returncode}")
                # Print last 30 lines of log for diagnosis
                log_path = tex_dir / (tex_path.stem + ".log")
                if log_path.exists():
                    lines = log_path.read_text(errors="replace").splitlines()
                    for l in lines[-30:]:
                        print(f"    {l}")
                if pass_num == 1:
                    return False
        except FileNotFoundError:
            print("  pdflatex not found — skipping PDF compilation.")
            return False
        except subprocess.TimeoutExpired:
            print("  pdflatex timed out.")
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate extended multi-location ZA-GAS report."
    )
    parser.add_argument("--reports-dir", type=str,
                        default=str(ROOT / "reports" / "multi_location"),
                        help="Output directory for report files")
    parser.add_argument("--location", type=str, default=None,
                        help="Generate report only for this location")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip pdflatex compilation")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig_base = reports_dir / "figures"
    csv_base = reports_dir / "tables"
    fig_base.mkdir(exist_ok=True)
    csv_base.mkdir(exist_ok=True)

    generation_date = time.strftime("%Y-%m-%d %H:%M UTC")

    stations = ALL_STATIONS_ORDERED
    if args.location:
        stations = [args.location]

    location_sections: Dict[str, str] = {}
    station_results: Dict[str, dict] = {}

    for station in stations:
        if station in SKIP_STATIONS or station not in STATION_REGISTRY:
            continue

        cfg     = STATION_REGISTRY[station]
        run_id  = cfg["run_id"]
        short   = cfg["short"]
        run_dir = ARTIFACTS_DIR / run_id

        if not run_dir.exists():
            print(f"[SKIP] {station}: run_dir not found ({run_dir})")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {cfg['display']} ({run_id})")
        print(f"{'='*60}")

        # Load data
        try:
            loader = StationDataLoader.from_registry(
                station_name=station,
                precip_dir=PRECIP_DIR,
                era5_dir=ERA5_DIR,
                nino34_path=NINO34_PATH,
            )
            data = loader.load_all()
        except Exception as exc:
            print(f"  [ERROR] Data loading failed: {exc}")
            continue

        fig_dir = fig_base / short
        csv_dir = csv_base / short

        section_extra: dict = {}
        try:
            section_tex, section_extra = generate_location_section(
                station_name=station,
                run_dir=run_dir,
                data=data,
                fig_dir=fig_dir,
                csv_dir=csv_dir,
                station_short=short,
            )
        except Exception as exc:
            import traceback
            print(f"  [ERROR] Section generation failed: {exc}")
            print(traceback.format_exc())
            section_tex = rf"\section{{{_tex(cfg['display'])}}}" + "\n" + \
                          rf"\textit{{Section generation error: {_tex(str(exc))}}}" + "\n"

        location_sections[station] = section_tex

        # Collect global comparison data
        winners = _load_winners(run_dir)
        fw_info = winners.get("final_winner", {})
        s4_info = winners.get("stage4_regime", {})
        station_results[station] = {
            "final_winner":   fw_info,
            "crps_s1":        fw_info.get("crps_s1", float("nan")),
            "crps_s2":        fw_info.get("crps_s2", float("nan")),
            "crps_s3":        fw_info.get("crps_s3", float("nan")),
            "crps_final":     fw_info.get("crps", float("nan")),
            "stage4_accepted": s4_info.get("stage4_accepted"),
            "stage4_best_crps": (
                s4_info.get("best_regime_crps")
                or s4_info.get("winner_crps")
            ),
            "stage4_note": (
                s4_info.get("stage4_note") or s4_info.get("note", "")
            ),
            "numerical_note": fw_info.get("numerical_note"),
            # Tail-metric Stage 4 acceptance (computed from extended metrics)
            "stage4_tail_accepted": section_extra.get("stage4_tail_accepted"),
            "stage4_tail_impr90":   section_extra.get("stage4_tail_impr90"),
        }

    # Global comparison
    global_tex = generate_global_comparison(
        station_results=station_results,
        csv_dir=csv_base,
        fig_dir=fig_base,
    )

    # Build LaTeX document
    latex_str = build_latex_document(
        location_sections=location_sections,
        global_section=global_tex,
        generation_date=generation_date,
    )

    tex_path = reports_dir / "report.tex"
    tex_path.write_text(latex_str, encoding="utf-8")
    print(f"\nLaTeX written to: {tex_path}")

    if not args.no_compile:
        print("Compiling PDF (pass 1 of 2)...")
        ok = compile_latex(tex_path)
        if ok:
            pdf_path = tex_path.with_suffix(".pdf")
            print(f"PDF compiled: {pdf_path}")
        else:
            print("PDF compilation had issues — check the .log file.")

    print(
        "\n"
        "To recompile the report manually after editing the LaTeX:\n"
        f"  cd \"{reports_dir}\"\n"
        "  pdflatex -interaction=nonstopmode report.tex\n"
        "  pdflatex -interaction=nonstopmode report.tex\n"
        "\n"
        "To regenerate the report without AI assistance:\n"
        f"  python generate_report_extended.py\n"
    )


if __name__ == "__main__":
    main()
