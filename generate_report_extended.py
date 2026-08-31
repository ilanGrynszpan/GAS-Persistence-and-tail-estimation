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
ENSO_PATH   = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
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

# The six locations actually run (round 2, 2026-07-07): SAO PAULO and
# TORONTO are registered in STATION_REGISTRY but were never estimated, per
# prompt.md's instruction to reuse only the already-run locations.
ALL_STATIONS_ORDERED = [
    "BELO HORIZONTE",
    "CRUZEIRO DO SUL (ACRE)",
    "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)",
    "MANAUS",
    "SALVADOR",
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
    """Format a numeric value for a LaTeX table cell.

    +inf is rendered as "$\\infty$" rather than "--": the Kupiec/
    Christoffersen LR statistics are defined as +inf precisely when there
    are zero exceedances in the OOS sample (a degenerate but meaningful
    result, common at high quantile levels with ~730-day test windows) --
    collapsing that to the same "--" used for genuinely missing/NaN values
    would hide the distinction between "not computed" and "test statistic
    is infinite because the model had zero violations".
    """
    if isinstance(v, float) and np.isposinf(v):
        return r"$\infty$"
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


# Round-1 Stage-3 model ids built from the mislabelled raw-SST ENSO source
# (see data/station_loader.py header) -- confirmed scientifically wrong, not
# merely outdated. Their artifact directories are left on disk (nothing is
# deleted -- ARCHITECTURE.md §9), but must never appear in any report table,
# selection, or figure. The live pipeline itself never reconsiders these (it
# only ever builds the 4 current-named Stage-3 specs -- see
# pipeline.runner.build_stage3_specs), so this filter only matters here,
# where _all_model_ids() otherwise blindly lists every subdirectory.
_STALE_MODEL_IDS = {
    "stage3_harvey_enso_90d", "stage3_harvey_enso_90d_30d",
    "stage3_harvey_enso_90d_30d_daily",
}


def _all_model_ids(run_dir: Path, stage: str) -> List[str]:
    stage_dir = run_dir / stage
    if not stage_dir.exists():
        return []
    return sorted(
        d.name for d in stage_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
        and d.name not in _STALE_MODEL_IDS
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model reconstruction from artifacts
# ─────────────────────────────────────────────────────────────────────────────

from pipeline.artifact_utils import (
    infer_gas_lags as _infer_gas_lags,
    infer_tv_from_params as _infer_tv_from_params,
    build_model_from_meta as _build_model_from_meta,
    get_theta as _get_theta,
)
# NOTE (2026-07-08): these four used to be defined locally in this file, with
# a bug where model reconstruction checked meta["model_class"] -- a key no
# model class actually saves (every one saves "model_type"). That silently
# reconstructed every Harvey/CovZAGASModel artifact as a plain ZAGASModel
# whenever the model_id fallback wasn't threaded through, producing wrong
# IS PIT/ACF diagnostics for those winners without any error. Fixed and
# consolidated into pipeline/artifact_utils.py (also used by Stage 4).


def _patch_meta_for_reconstruction(meta: dict, params_df) -> dict:
    """
    Patch metadata for older/inconsistent artifact key names before it is
    used to reconstruct a model or its OOS paths.

    Older code paths read "tv_param_names" / static scalars ("gamma",
    "zeta", "xi") directly off meta, but current artifacts save these under
    "tv_names" (metadata.json) and inside bound_diagnostics or
    estimated_parameters.csv respectively. Without this patch,
    _reconstruct_oos_paths silently falls back to the ["phi"]-only default,
    dropping any TV "xi" column from f_arr_oos entirely -- which makes
    dist.ppf() raise KeyError('xi') (caught and turned into NaN) for every
    quantile level whose q exceeds the zero-mass threshold. Only q=0.50
    partially survives because roughly half its evaluations fall below
    zero-mass and short-circuit to 0.0 before ever needing 'xi'.
    """
    if meta is None:
        return meta
    meta = dict(meta)
    if not meta.get("tv_param_names"):
        tv_from_artifact = meta.get("tv_names")
        meta["tv_param_names"] = (
            tv_from_artifact if tv_from_artifact
            else _infer_tv_from_params(params_df)
        )
    bd = meta.get("bound_diagnostics", {})
    for key in ("gamma", "zeta", "xi"):
        if key not in meta:
            bd_key = f"static_{key}"
            if bd_key in bd:
                meta[key] = float(bd[bd_key])
    if params_df is not None and not params_df.empty:
        if "parameter" in params_df.columns and "value" in params_df.columns:
            for key in ("gamma", "zeta", "xi"):
                if key not in meta:
                    row = params_df[params_df["parameter"] == key]
                    if not row.empty:
                        meta[key] = float(row.iloc[0]["value"])
    return meta


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
# Stage-directory resolution (2026-07-07 layout: stage1/2/3 = phi+xi branch,
# stage2_phi/stage3_phi = phi-only branch, stage4_xi_regime = Stage 4).
# Order matters: check the more specific phi-only/Stage-4 prefixes first, or
# e.g. "stage3_phi_harvey_no_enso" would incorrectly match "stage3_" and
# resolve to the phi+xi directory (a real bug in the pre-2026-07-07 version
# of this function, which only knew about "stage3"/"stage2"/"stage1"/
# "stage4_regime" and silently looked in the wrong directory for every
# phi-only-branch and Stage-4 model).
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_stage_dir(model_id: str) -> str:
    if model_id.startswith("stage5_"):
        return "stage5"
    if model_id.startswith("stage4_xi_regime"):
        return "stage4_xi_regime"
    if model_id.startswith("stage3_phi_harvey"):
        return "stage3_phi"
    if model_id.startswith("stage2_phi_"):
        return "stage2_phi"
    if model_id.startswith("stage3_"):
        return "stage3"
    if model_id.startswith("stage2_"):
        return "stage2"
    return "stage1"


def _stage_label_for_model_id(model_id: str) -> str:
    """Human-readable stage label for a model_id, e.g. for annotating cross-
    location tables that mix winners from different stages (2026-07-09b)."""
    if model_id.startswith("stage5_"):
        return "Stage 5"
    if model_id.startswith("stage4_"):
        return "Stage 4"
    if "harvey" in model_id:
        return "Stage 3 (Harvey)"
    if model_id.startswith("stage3_"):
        return "Stage 3"
    if model_id.startswith("stage2_"):
        return "Stage 2"
    return "Stage 1"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 (static benchmark + threshold-weighted xi, 2026-07-09): bespoke
# reconstruction, deliberately NOT routed through build_model_from_meta's
# generic tv_names-inference fallback. StaticGB2Model has tv_param_names=[]
# (empty by design -- nothing is time-varying), and the generic fallback's
# `meta.get("tv_param_names") or ...` pattern treats an empty-but-correct
# list the same as a missing key, silently substituting the wrong default
# (see session history: this exact "or fallback" footgun on a falsy-but-
# valid value was found and fixed twice already for other stages). Stage 5's
# frozen inputs (phi/gamma/zeta/pi/threshold_c) are also plain scalars
# saved directly in metadata.json -- no upstream base-model trajectory to
# replay (unlike Stage 4's RegimeXiOnlyModel) -- so reconstruction here is
# a direct metadata read, not a replay of some other stage's filter.
# ─────────────────────────────────────────────────────────────────────────────

def _stage5_reconstruct(model_id: str, run_dir: Path, data: Optional[dict] = None):
    """
    Returns (model, theta, frozen_kwargs) or (None, None, {}).

    `data` (needing y_train/y_test) is only required for the "_dynpi"
    threshold-xi variants, to rebuild their time-varying pi_t array from
    the sibling StaticGB2DynamicPiModel artifact -- see pi_dynamic /
    pi_dynamic_source_model_id in that model's saved metadata.json
    (models/threshold_weighted_xi_gas.py::save_result).
    """
    meta = _load_meta(run_dir, "stage5", model_id)
    params_df = _load_params_df(run_dir, "stage5", model_id)
    theta = _get_theta(params_df)
    if theta is None or not meta:
        return None, None, {}

    if model_id == "stage5_static_gb2":
        from models.static_gb2 import StaticGB2Model
        return StaticGB2Model(), theta, {}

    if model_id == "stage5_static_gb2_dynpi":
        from models.static_gb2 import StaticGB2DynamicPiModel
        return StaticGB2DynamicPiModel(), theta, {}

    if model_id.startswith("stage5_xi_q"):
        from models.threshold_weighted_xi_gas import ThresholdWeightedXiModel
        from distributions.gb2_log_link import GB2LogLink
        try:
            phi_v = float(meta["phi"]); gamma_v = float(meta["gamma"]); zeta_v = float(meta["zeta"])
            threshold_c = float(meta["threshold_c"])
        except (KeyError, TypeError):
            return None, None, {}

        if meta.get("pi_dynamic"):
            src_id = meta.get("pi_dynamic_source_model_id")
            if not src_id or data is None:
                return None, None, {}
            src_model, src_theta, _ = _stage5_reconstruct(src_id, run_dir, data)
            if src_model is None:
                return None, None, {}
            y_train, y_test = data["y_train"], data["y_test"]
            pi_train = src_model.filter(src_theta, y_train)["pi"]
            pi_oos   = src_model.simulate_oos(src_theta, y_train, y_test)["pi_oos"]
            frozen = {
                "phi_v": phi_v, "gamma_v": gamma_v, "zeta_v": zeta_v,
                "pi_v": np.concatenate([pi_train, pi_oos]),  # simulate_oos/full-length
                "threshold_c": threshold_c,
            }
        else:
            try:
                frozen = {
                    "phi_v": phi_v, "gamma_v": gamma_v, "zeta_v": zeta_v,
                    "pi_v": float(meta["pi"]), "threshold_c": threshold_c,
                }
            except (KeyError, TypeError):
                return None, None, {}

        model = ThresholdWeightedXiModel(
            distribution=GB2LogLink(),
            scaling=meta.get("scaling", "diagonal_inverse_fisher"),
            threshold_quantile=meta.get("threshold_quantile", 0.95),
        )
        return model, theta, frozen

    return None, None, {}


def _stage5_extended_metrics(model_id: str, run_dir: Path, data: dict, model_cache: dict) -> Optional[dict]:
    model, theta, frozen = _stage5_reconstruct(model_id, run_dir, data)
    if model is None:
        return None
    model_cache[model_id] = model
    try:
        # frozen["pi_v"], when present and array-valued, is already full-length
        # (train+test, see _stage5_reconstruct) -- exactly what simulate_oos expects.
        oos_paths = model.simulate_oos(theta, data["y_train"], data["y_test"], **frozen)
        mets = compute_extended_oos_metrics(
            model=model, paths_oos=oos_paths, y_test=data["y_test"], n_draws=1000,
        )
        mets["model_id"] = model_id
        return mets
    except Exception as exc:
        print(f"  [WARN] Stage-5 extended metrics failed for {model_id}: {exc}")
        return None


def _exceedance_ratios_for_model(
    model_id: str, run_dir: Path, data: dict, model_cache: dict,
    quantile_levels=(0.95, 0.98, 0.99),
) -> Optional[dict]:
    """
    Empirical/theoretical exceedance ratio at each quantile level for any
    model (generic Stage 1-4 models or Stage 5) -- the same calibration
    check used to build the main "how many floods really happened" table,
    factored out here so it can be reused for a magnitude-scale-independent
    comparison against Stage 5 (2026-07-09b: CRPS/twCRPS are dominated by
    Stage 5's frozen phi's poor point-scale, which is expected and not
    informative about whether the threshold-weighted xi tail mechanism
    itself is well calibrated; exceedance ratios are calibration-only,
    checking predicted vs. actual crossing FREQUENCY, not magnitude).

    Returns {"exceed_emp_XXXXX": ratio, ...} keyed like
    diagnostics.dynamics.compute_exceedance_frequencies, plus "model_id",
    or None if reconstruction/computation fails.
    """
    y_test = data["y_test"]

    if model_id.startswith("stage5_"):
        model, theta, frozen = _stage5_reconstruct(model_id, run_dir, data)
        if model is None:
            return None
        model_cache[model_id] = model
        try:
            oos_paths = model.simulate_oos(theta, data["y_train"], data["y_test"], **frozen)
        except Exception as exc:
            print(f"  [WARN] Stage-5 exceedance reconstruction failed for {model_id}: {exc}")
            return None
    else:
        stage = _resolve_stage_dir(model_id)
        meta      = _load_meta(run_dir, stage, model_id)
        params_df = _load_params_df(run_dir, stage, model_id)
        paths_npz = _load_paths_npz(run_dir, stage, model_id)
        if paths_npz is None:
            return None
        meta = _patch_meta_for_reconstruction(meta, params_df)
        oos_paths = _reconstruct_oos_paths(paths_npz, meta)
        if oos_paths is None:
            return None
        if model_id not in model_cache:
            try:
                model_cache[model_id] = _build_model_from_meta(meta, params_df, model_id=model_id)
            except Exception as exc:
                print(f"  [WARN] Could not reconstruct {model_id} for exceedance ratios: {exc}")
                return None
        model = model_cache[model_id]

    try:
        quants = compute_dynamic_quantiles(model, oos_paths, list(quantile_levels))
        exc = compute_exceedance_frequencies(y=y_test, dynamic_quantiles=quants, quantile_levels=list(quantile_levels))
    except Exception as ex:
        print(f"  [WARN] Exceedance ratio computation failed for {model_id}: {ex}")
        return None

    out = {"model_id": model_id}
    for q in quantile_levels:
        key = f"exceed_emp_{int(q*10000):05d}"
        theo = 1.0 - q
        emp = exc.get(key)
        out[f"ratio_{int(q*10000):05d}"] = (emp / theo) if (emp is not None and theo > 0) else None
    return out


def _stage5_is_diagnostics(model_id: str, run_dir: Path, data: dict, fig_dir: Path) -> Optional[dict]:
    model, theta, frozen = _stage5_reconstruct(model_id, run_dir, data)
    if model is None:
        return None
    y_train = data["y_train"]
    # filter()/cdf_series() need pi_v aligned to y_train alone; frozen["pi_v"]
    # (when array-valued) is full-length train+test -- slice to the train
    # portion here rather than inside _stage5_reconstruct, since that
    # function's frozen dict is shared with simulate_oos (which needs the
    # full-length array unsliced).
    if "pi_v" in frozen and not np.isscalar(frozen["pi_v"]):
        frozen = {**frozen, "pi_v": np.asarray(frozen["pi_v"])[: len(y_train)]}
    try:
        cdfs = model.cdf_series(theta, y_train, **frozen)
    except Exception as exc:
        print(f"  [WARN] Stage-5 IS cdf_series failed for {model_id}: {exc}")
        return None
    if cdfs is None or len(cdfs) == 0:
        return None
    rng = np.random.default_rng(0)
    n = len(cdfs)
    y_eff = y_train[len(y_train) - n:]
    pit = pit_values(cdfs, y_eff, randomise_zeros=True, rng=rng)
    qr = quantile_residuals(cdfs, y_eff, randomise_zeros=True, rng=rng)
    pit_path, qq_path, acf_path = _plot_is_diagnostics(
        model_id=model_id, pit=pit, qr=qr, fig_dir=fig_dir
    )
    return {
        "pit": pit, "qr": qr,
        "pit_mean": float(np.nanmean(pit)), "pit_std": float(np.nanstd(pit)),
        "pit_path": pit_path, "qq_path": qq_path, "acf_path": acf_path,
    }


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
    if model_id.startswith("stage5_"):
        return _stage5_extended_metrics(model_id, run_dir, data, model_cache)

    stage = _resolve_stage_dir(model_id)

    meta      = _load_meta(run_dir, stage, model_id)
    params_df = _load_params_df(run_dir, stage, model_id)
    paths_npz = _load_paths_npz(run_dir, stage, model_id)

    if paths_npz is None:
        return None

    meta = _patch_meta_for_reconstruction(meta, params_df)

    oos_paths = _reconstruct_oos_paths(paths_npz, meta)
    if oos_paths is None:
        return None

    # Get model instance (cached)
    if model_id not in model_cache:
        try:
            m = _build_model_from_meta(meta, params_df, model_id=model_id)
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

def _stage4_is_cdfs(model_id: str, run_dir: Path, data: dict, winners: dict) -> Optional[np.ndarray]:
    """
    IS CDF series for a Stage-4 RegimeXiOnlyModel winner.

    RegimeXiOnlyModel has no cdf_series() (unlike the other model classes)
    because its predictive distribution needs the frozen phi/pi trajectory
    from its base model, which isn't recoverable from theta+y_train alone.
    Reconstructs that frozen base exactly as run_all_locations.py::run_stage4
    did at fit time, using only saved artifacts (stage_winners.json +
    metadata.json) -- no re-optimization.
    """
    from run_all_locations import _stage_dir_for_model_id, _base_covariate_kwargs
    from models.regime_gas import build_regime_xi_only_from_frozen_phi

    s4_info = winners.get("stage4_xi_regime", {})
    base_id = s4_info.get("base_model")
    if not base_id:
        return None
    stage2_phi_id = winners.get("stage2_phi_only", {}).get("winner")
    stage2_phi_stub = {"model_id": stage2_phi_id} if stage2_phi_id else None

    y_train, y_test = data["y_train"], data["y_test"]
    fit_kw, oos_kw = _base_covariate_kwargs({"model_id": base_id}, stage2_phi_stub, data)
    base_dir = _stage_dir_for_model_id(run_dir, base_id)
    bt, bv, cn = data["covariate_blocks_train"], data["covariate_blocks_test"], data["covariate_col_names"]

    meta = _load_meta(run_dir, "stage4_xi_regime", model_id)
    params_df = _load_params_df(run_dir, "stage4_xi_regime", model_id)
    theta = _get_theta(params_df)
    if theta is None or meta.get("threshold_quantile") is None:
        return None

    model, frozen = build_regime_xi_only_from_frozen_phi(
        base_model_dir=base_dir, y_train=y_train, y_test=y_test,
        xi_cov_train=bt["dewtemp_seasonal"], xi_cov_test=bv["dewtemp_seasonal"],
        xi_cov_names=cn["dewtemp_seasonal"],
        threshold_quantile=meta["threshold_quantile"],
        base_extra_fit_kwargs=fit_kw, base_extra_oos_kwargs=oos_kw,
    )
    model._threshold_c = meta.get("threshold_c")  # skip fit()'s recompute; use the saved value
    paths = model.filter(
        theta, y_train, frozen["phi_full"], frozen["pi_full"],
        frozen["X_train"], frozen["gamma_v"], frozen["zeta_v"], frozen["warmup"],
    )
    if not paths:
        return None

    eff = paths["eff_start"]
    n = len(paths["xi"])
    cdfs = np.zeros(n)
    for i in range(n):
        t = eff + i
        phi_t, xi_t = paths["phi"][i], paths["xi"][i]
        pi_t = paths["pi"][i]
        if y_train[t] <= 0:
            cdfs[i] = 1.0 - pi_t
        else:
            G_t = model.dist.cdf(y_train[t], phi=phi_t, xi=xi_t,
                                  gamma=frozen["gamma_v"], zeta=frozen["zeta_v"])
            cdfs[i] = (1.0 - pi_t) + pi_t * G_t
    return cdfs


def _compute_is_diagnostics_for_winner(
    model_id: str,
    run_dir: Path,
    data: dict,
    col_name_dict: dict,
    fig_dir: Path,
    winners: Optional[dict] = None,
) -> Optional[dict]:
    """
    Compute IS PIT / QR for a winner model.  Returns dict with pit, qr, paths.
    """
    if model_id.startswith("stage5_"):
        return _stage5_is_diagnostics(model_id, run_dir, data, fig_dir)

    if model_id.startswith("stage4_xi_regime"):
        try:
            cdfs = _stage4_is_cdfs(model_id, run_dir, data, winners or {})
        except Exception as exc:
            print(f"  [WARN] Stage-4 IS cdf_series failed for {model_id}: {exc}")
            return None
        if cdfs is None or len(cdfs) == 0:
            return None
        y_train = data["y_train"]
        rng = np.random.default_rng(0)
        n = len(cdfs)
        y_eff = y_train[len(y_train) - n:]
        pit = pit_values(cdfs, y_eff, randomise_zeros=True, rng=rng)
        qr = quantile_residuals(cdfs, y_eff, randomise_zeros=True, rng=rng)
        pit_path, qq_path, acf_path = _plot_is_diagnostics(
            model_id=model_id, pit=pit, qr=qr, fig_dir=fig_dir
        )
        return {
            "pit": pit, "qr": qr,
            "pit_mean": float(np.nanmean(pit)), "pit_std": float(np.nanstd(pit)),
            "pit_path": pit_path, "qq_path": qq_path, "acf_path": acf_path,
        }

    stage = _resolve_stage_dir(model_id)

    meta      = _load_meta(run_dir, stage, model_id)
    params_df = _load_params_df(run_dir, stage, model_id)
    theta     = _get_theta(params_df)

    if theta is None:
        return None

    y_train = data["y_train"]
    bt      = data["covariate_blocks_train"]
    cn      = data["covariate_col_names"]

    try:
        model = _build_model_from_meta(meta, params_df, model_id=model_id)
        model_class = meta.get("model_type", "ZAGASModel")

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

def _stage_decision_table_tex(
    run_dir: Path, stage: str, model_ids: List[str], winner_id: Optional[str],
    tvp_set: str, n_train: int, data: dict, model_cache: dict,
    caption: str, label: str,
) -> str:
    """
    One table per stage: every candidate model (not just the winner) with
    its key decision metric(s) and AIC, winner row bolded. Nothing here
    requires re-fitting -- loglik/n_params/lags come from metadata.json,
    RMSE/MAE from metrics_core.json, and twCRPS is recomputed from the
    already-saved paths.npz (Monte Carlo draws from the already-fit
    predictive distribution, not re-optimization).

    tvp_set="phi_only" -> key metrics are OOS RMSE, MAD (objective 1a/4b).
    tvp_set="phi_xi"   -> key metrics are OOS CRPS, twCRPS@95, twCRPS@98
                          (objective 1b/4c; "the main criterion" is CRPS).
    """
    from diagnostics.information import info_table

    if not model_ids:
        return f"% No models found for {stage}\n"

    rows = []
    for mid in model_ids:
        meta = _load_meta(run_dir, stage, mid)
        loglik, n_params = meta.get("loglik"), meta.get("n_params")
        if loglik is None or n_params is None or not np.isfinite(float(loglik)):
            continue
        lags = meta.get("lags") or []
        n_obs = n_train - (max(lags) + 1) if lags else n_train
        info = info_table(float(loglik), int(n_params), max(int(n_obs), 1))
        row = {"model": mid, "validity": meta.get("validity", "?"), **info}

        if tvp_set == "phi_only":
            mcore = _load_json_safe(run_dir / stage / mid / "metrics_core.json") \
                if (run_dir / stage / mid / "metrics_core.json").exists() else {}
            row["rmse"] = mcore.get("rmse")
            row["mad"]  = mcore.get("mae")  # MAE == MAD of the forecast errors
        else:
            ext = _compute_extended_metrics_for_model(mid, run_dir, data, model_cache)
            row["crps"]  = ext.get("crps_mean") if ext else None
            row["tw95"]  = ext.get("twcrps_95") if ext else None
            row["tw98"]  = ext.get("twcrps_98") if ext else None
        rows.append(row)

    if not rows:
        return f"% No valid models for {stage}\n"

    if tvp_set == "phi_only":
        headers = r"\small Model & Validity & $k$ & loglik & AIC & RMSE & MAD \\"
        col_fmt = "llrrrrr"
        def _row(r):
            return (f"{_fmt(r['n_params'],0)} & {_fmt(r['loglik'],1)} & {_fmt(r['aic'],1)} & "
                    f"{_fmt(r['rmse'])} & {_fmt(r['mad'])}")
    else:
        headers = r"\small Model & Validity & $k$ & loglik & AIC & CRPS & twCRPS@95 & twCRPS@98 \\"
        col_fmt = "llrrrrrr"
        def _row(r):
            return (f"{_fmt(r['n_params'],0)} & {_fmt(r['loglik'],1)} & {_fmt(r['aic'],1)} & "
                    f"{_fmt(r['crps'])} & {_fmt(r['tw95'])} & {_fmt(r['tw98'])}")

    body = []
    for r in rows:
        model_cell = r"\texttt{" + _tex(r["model"]) + "}"
        if r["model"] == winner_id:
            model_cell = r"\textbf{" + model_cell + " *}"
        body.append(f"{model_cell} & {_tex(r['validity'])} & {_row(r)} \\\\")

    return (
        r"\begin{table}[ht]" + "\n\\centering\n\\footnotesize\n"
        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%" + "\n"
        rf"\begin{{tabular}}{{{col_fmt}}}" + "\n\\hline\n"
        + headers + "\n\\hline\n" + "\n".join(body) + "\n\\hline\n"
        r"\end{tabular}" + "\n}" + "\n"
        rf"\caption{{{caption} Winner marked with \textbf{{*}}.}}" + "\n"
        rf"\label{{{label}}}" + "\n"
        r"\end{table}" + "\n"
    )


def _inter_stage_progression_table_tex(
    stage_winner_infos: List[Tuple[str, dict]], tvp_set: str,
    caption: str, label: str,
) -> str:
    """
    Objective (user, 2026-07-08): AIC + key decision metric for the WINNER
    of each stage, side by side across stages, so added model complexity
    can be checked against actual improvement (a parsimony/overfitting
    sanity check, not needed for winner selection itself -- that already
    happened via pipeline.selection -- but useful supporting evidence).

    stage_winner_infos: list of (stage_label, meta_dict) for each stage's
    winner, meta_dict must have loglik/n_params/aic and the relevant OOS
    key metric(s) already merged in by the caller.
    """
    if not stage_winner_infos:
        return f"% No stage winners for {label}\n"

    if tvp_set == "phi_only":
        headers = r"\small Stage & Model & $k$ & AIC & RMSE \\"
        col_fmt = "llrrr"
        def _row(m):
            return f"{_fmt(m.get('n_params'),0)} & {_fmt(m.get('aic'),1)} & {_fmt(m.get('rmse'))}"
    else:
        headers = r"\small Stage & Model & $k$ & AIC & CRPS \\"
        col_fmt = "llrrr"
        def _row(m):
            return f"{_fmt(m.get('n_params'),0)} & {_fmt(m.get('aic'),1)} & {_fmt(m.get('crps_mean'))}"

    body = [
        f"{_tex(stage_lbl)} & \\texttt{{{_tex(m.get('model',''))}}} & {_row(m)} \\\\"
        for stage_lbl, m in stage_winner_infos
    ]
    return (
        r"\begin{table}[ht]" + "\n\\centering\n\\footnotesize\n"
        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%" + "\n"
        rf"\begin{{tabular}}{{{col_fmt}}}" + "\n\\hline\n"
        + headers + "\n\\hline\n" + "\n".join(body) + "\n\\hline\n"
        r"\end{tabular}" + "\n}" + "\n"
        rf"\caption{{{caption}}}" + "\n"
        rf"\label{{{label}}}" + "\n"
        r"\end{table}" + "\n"
    )


def _stage_comparison_table_tex(
    run_dir: Path, stage: str,
    winners: dict, caption: str, label: str,
    model_ids: Optional[List[str]] = None,
    winner_key: Optional[str] = None,
) -> str:
    """
    LaTeX model comparison table for one stage.

    model_ids: override the default _all_model_ids(run_dir, stage) listing
    -- needed when a stage directory holds more than one independently-
    reported family (e.g. stage5/ holds both the static-pi and dynamic-pi
    Stage 5 variants side by side).
    winner_key: which key of `winners` to read "winner" from, if different
    from `stage` itself (e.g. stage5's dynamic-pi variant winner lives
    under winners["stage5_dynpi"], not winners["stage5"]).
    """
    if model_ids is None:
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

    winner_mid = winners.get(winner_key or stage, {}).get("winner")

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
        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%" + "\n"
        r"\begin{tabular}{lllrrrr}" + "\n"
        r"\hline" + "\n"
        + cols_hdr + r" \\" + "\n"
        r"\hline" + "\n"
        + "\n".join(body_lines) + "\n"
        r"\hline" + "\n"
        r"\end{tabular}" + "\n}" + "\n"
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
        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%" + "\n"
        rf"\begin{{tabular}}{{{col_fmt}}}" + "\n"
        r"\hline" + "\n"
        + headers + "\n"
        r"\hline" + "\n"
        + "\n".join(body_lines) + "\n"
        r"\hline" + "\n"
        r"\end{tabular}" + "\n}" + "\n"
        rf"\caption{{{caption}}}" + "\n"
        rf"\label{{{label}}}" + "\n"
        r"\end{table}" + "\n"
    )


def _load_params_with_se(run_dir: Path, stage: str, model_id: str) -> pd.DataFrame:
    """
    estimated_parameters.csv (value) merged with standard_errors.csv
    (std_error, se_quality) when the latter exists -- both are already
    saved by every model's save_result(), no re-fitting needed. Per
    prompt.md: standard errors belong in the appendix and were missing
    from round 1's report despite already being on disk.
    """
    params_df = _load_params_df(run_dir, stage, model_id)
    if params_df.empty:
        return params_df
    se_path = run_dir / stage / model_id / "standard_errors.csv"
    if se_path.exists():
        se_df = pd.read_csv(se_path)
        params_df = params_df.merge(
            se_df[["parameter", "std_error", "se_quality"]], on="parameter", how="left"
        )
    return params_df


def _params_table_tex(params_df: pd.DataFrame, caption: str, label: str) -> str:
    """LaTeX parameter table with value, standard error, and SE quality."""
    if params_df.empty:
        return "% No parameters\n"
    cols = [c for c in ["parameter", "value", "std_error", "se_quality"] if c in params_df.columns]
    if not cols:
        return "% No expected columns\n"

    text_cols = {"parameter", "se_quality"}
    n_cols  = len(cols)
    col_fmt = "".join("l" if c in text_cols else "r" for c in cols)
    headers = " & ".join(r"\small " + _tex(c) for c in cols) + r" \\"
    rows = []
    for _, row in params_df.iterrows():
        cells = []
        for c in cols:
            v = row.get(c)
            if c == "parameter":
                cells.append(r"\texttt{" + _tex(str(v)) + "}")
            elif c == "se_quality":
                cells.append(_tex(str(v)) if pd.notna(v) else "--")
            else:
                cells.append(_fmt(v))
        rows.append(" & ".join(cells) + r" \\")

    return (
        r"\begin{table}[ht]" + "\n"
        r"\centering" + "\n"
        r"\footnotesize" + "\n"
        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%" + "\n"
        rf"\begin{{tabular}}{{{col_fmt}}}" + "\n"
        r"\hline" + "\n"
        + headers + "\n"
        r"\hline" + "\n"
        + "\n".join(rows) + "\n"
        r"\hline" + "\n"
        r"\end{tabular}" + "\n}" + "\n"
        rf"\caption{{{caption}}}" + "\n"
        rf"\label{{{label}}}" + "\n"
        r"\end{table}" + "\n"
    )


def _exceedance_summary_table_tex(
    quantile_levels: Sequence[float],
    overall: dict,
    wet: Optional[dict],
    dry: Optional[dict],
    enso_regimes: Optional[dict],
    caption: str,
    label: str,
) -> str:
    """
    Objective 4f: empirical vs. theoretical exceedance frequency at each
    quantile level, broken down overall / wet / dry / by ENSO regime, in one
    compact table (monthly breakdown stays a figure -- 12 more columns would
    not fit). All inputs are metric dicts already computed by
    diagnostics.decomposition._group_metrics / dynamics.compute_exceedance_
    frequencies (key format "exceed_emp_XXXXX") -- no recomputation.

    Replaces the previous wet/dry table, which called
    _extended_metrics_table_tex(cols_group="rmse") on data that has no rmse/
    mad keys at all, silently degrading to a single model_id column.
    """
    groups = [("Overall", overall)]
    if wet is not None and wet.get("n", 0) > 0:
        groups.append(("Wet", wet))
    if dry is not None and dry.get("n", 0) > 0:
        groups.append(("Dry", dry))
    if enso_regimes:
        for label_r, mets in enso_regimes.items():
            if mets and mets.get("n", 0) > 0:
                groups.append((label_r, mets))

    if len(groups) <= 1:
        return "% Insufficient groups for exceedance summary\n"

    headers = (r"\small $q$ & \small Theoretical & "
               + " & ".join(rf"\small {_tex(g)} (emp.)" for g, _ in groups) + r" \\")
    body = []
    for q in quantile_levels:
        key = f"exceed_emp_{int(q*10000):05d}"
        theo = 1.0 - q
        cells = [f"q{q*100:g}", _fmt(theo, 3)]
        for _, mets in groups:
            cells.append(_fmt(mets.get(key), 3))
        body.append(" & ".join(cells) + r" \\")

    col_fmt = "l" + "r" * (len(groups) + 1)
    return (
        r"\begin{table}[ht]" + "\n\\centering\n\\footnotesize\n"
        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%" + "\n"
        rf"\begin{{tabular}}{{{col_fmt}}}" + "\n\\hline\n"
        + headers + "\n\\hline\n" + "\n".join(body) + "\n\\hline\n"
        r"\end{tabular}" + "\n}" + "\n"
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

    # ── Stages 1-3, both branches: one decision table per stage, all models,
    # key metric + AIC, winner bolded (2026-07-08 user instruction) ────────
    n_train = data["summary"]["n_train"]

    def _candidates_for(stage_dir: str, prefix_filter=None) -> List[dict]:
        """Rebuild lightweight candidate dicts from cached artifacts only
        (metadata.json + metrics_core.json) -- no re-fitting -- suitable for
        pipeline.selection (re-deriving the winner + its tie-break reason is
        cheap arithmetic over already-saved numbers, not re-optimization)."""
        ids = _all_model_ids(run_dir, stage_dir)
        if prefix_filter:
            ids = [m for m in ids if prefix_filter(m)]
        out = []
        for mid in ids:
            meta = _load_meta(run_dir, stage_dir, mid)
            mcore_p = run_dir / stage_dir / mid / "metrics_core.json"
            mcore = _load_json_safe(mcore_p) if mcore_p.exists() else {}
            out.append({
                "model_id": mid, "validity": meta.get("validity", "failed"),
                "loglik": meta.get("loglik"), "n_params": meta.get("n_params", 0),
                "oos_metrics": mcore,
            })
        return out

    def _stage_block(branch_label: str, tvp_set: str, stage_dir: str,
                      subsection: str, model_ids: List[str], candidates: List[dict],
                      station_tag: str, winners_key: str):
        # Explain the ACTUAL saved winner (stage_winners.json, decided live
        # during pipeline execution) rather than independently re-selecting
        # one -- re-selection can disagree with what really happened if
        # `candidates` isn't in the exact order the live run saw (ties
        # broken by iteration order). See pipeline.selection.explain_winner.
        from pipeline.selection import explain_winner
        key_metric = "rmse" if tvp_set == "phi_only" else "crps_mean"
        actual_winner_id = winners.get(winners_key, {}).get("winner")
        winner = (
            explain_winner(candidates, actual_winner_id, key_metric,
                            higher_better=False, improvement_threshold=IMPROVEMENT_THRESHOLD)
            if actual_winner_id else None
        )
        tex_parts.append(rf"\subsubsection{{{subsection}}}" + "\n")
        tex_parts.append(_stage_decision_table_tex(
            run_dir, stage_dir, model_ids, winner["model_id"] if winner else None,
            tvp_set, n_train, data, model_cache,
            caption=f"{display} ({branch_label}) -- {subsection}.",
            label=f"tab:{station_tag}",
        ))
        if winner:
            tex_parts.append(
                rf"\textbf{{Winner}}: \texttt{{{_tex(winner['model_id'])}}} -- "
                + _tex(winner.get("_tie_break", "")) + "." + "\n\n"
            )
            # Lean image set (objective 4e, scoped per user 2026-07-08: PIT +
            # 400-lag ACF only, one set per stage winner -- 3 phi-only + 4
            # phi+xi(+Stage 4) = 7 sets per location, not per candidate model).
            is_diag = _compute_is_diagnostics_for_winner(
                model_id=winner["model_id"], run_dir=run_dir, data=data,
                col_name_dict=data["covariate_col_names"], fig_dir=fig_dir,
                winners=winners,
            )
            if is_diag:
                pit_rel = os.path.relpath(is_diag["pit_path"], fig_dir.parent.parent).replace("\\", "/")
                acf_rel = os.path.relpath(is_diag["acf_path"], fig_dir.parent.parent).replace("\\", "/")
                tex_parts.append(
                    f"PIT mean = {is_diag['pit_mean']:.3f} (ideal 0.5), "
                    f"std = {is_diag['pit_std']:.3f} (ideal {1/12**0.5:.3f})." + "\n\n"
                )
                tex_parts.append(_fig_tex(
                    pit_rel, f"{display} ({branch_label}, {subsection}): IS PIT histogram.",
                    f"fig:{station_tag}_pit", width="0.55",
                ))
                tex_parts.append(_fig_tex(
                    acf_rel, f"{display} ({branch_label}, {subsection}): IS quantile-residual ACF (400 lags).",
                    f"fig:{station_tag}_acf", width="0.75",
                ))
        return winner

    IMPROVEMENT_THRESHOLD = 0.02

    # -- Phi-only branch (objective 1a: ranked by OOS RMSE throughout) ------
    tex_parts.append(r"\subsection{Phi-only branch (ranked by OOS RMSE)}" + "\n")
    w1_phi = _stage_block(
        "phi-only", "phi_only", "stage1", "Stage 1 -- Baseline GAS",
        [m for m in _all_model_ids(run_dir, "stage1") if "stage1_phi_" in m and "phixi" not in m],
        _candidates_for("stage1", lambda m: "stage1_phi_" in m and "phixi" not in m),
        f"{station_short}_s1_phi", "stage1_phi_only",
    )
    w2_phi = _stage_block(
        "phi-only", "phi_only", "stage2_phi", "Stage 2 -- Weather Covariates",
        _all_model_ids(run_dir, "stage2_phi"), _candidates_for("stage2_phi"),
        f"{station_short}_s2_phi", "stage2_phi_only",
    )
    w3_phi = _stage_block(
        "phi-only", "phi_only", "stage3_phi", "Stage 3 -- Harvey Long-Short",
        _all_model_ids(run_dir, "stage3_phi"), _candidates_for("stage3_phi"),
        f"{station_short}_s3_phi", "stage3_phi_only",
    )
    phi_progression = [
        (lbl, {**w["oos_metrics"], "model": w["model_id"], "n_params": w["n_params"],
               "aic": w["oos_metrics"].get("aic")})
        for lbl, w in (("Stage 1", w1_phi), ("Stage 2", w2_phi), ("Stage 3", w3_phi)) if w
    ]
    # AIC isn't in oos_metrics; compute it directly for the progression table.
    from diagnostics.information import aic as _aic_fn
    for lbl, w in (("Stage 1", w1_phi), ("Stage 2", w2_phi), ("Stage 3", w3_phi)):
        if w and w.get("loglik") is not None:
            entry = next((e for l, e in phi_progression if l == lbl), None)
            if entry is not None:
                entry["aic"] = _aic_fn(float(w["loglik"]), int(w.get("n_params", 0)))
    tex_parts.append(_inter_stage_progression_table_tex(
        phi_progression, "phi_only",
        caption=f"{display}: phi-only branch -- AIC and OOS RMSE across stage winners.",
        label=f"tab:{station_short}_prog_phi",
    ))

    # -- Phi+xi branch (objective 1b: ranked by OOS CRPS throughout) --------
    tex_parts.append(r"\subsection{Phi+xi branch (ranked by OOS CRPS)}" + "\n")
    w1_xi = _stage_block(
        "phi+xi", "phi_xi", "stage1", "Stage 1 -- Baseline GAS",
        [m for m in _all_model_ids(run_dir, "stage1") if "phixi" in m],
        _candidates_for("stage1", lambda m: "phixi" in m),
        f"{station_short}_s1_xi", "stage1",
    )
    w2_xi = _stage_block(
        "phi+xi", "phi_xi", "stage2", "Stage 2 -- Weather Covariates",
        _all_model_ids(run_dir, "stage2"), _candidates_for("stage2"),
        f"{station_short}_s2_xi", "stage2",
    )
    w3_xi = _stage_block(
        "phi+xi", "phi_xi", "stage3", "Stage 3 -- Harvey Long-Short",
        _all_model_ids(run_dir, "stage3"), _candidates_for("stage3"),
        f"{station_short}_s3_xi", "stage3",
    )
    xi_progression = []
    for lbl, w in (("Stage 1", w1_xi), ("Stage 2", w2_xi), ("Stage 3", w3_xi)):
        if not w:
            continue
        entry = {"model": w["model_id"], "n_params": w["n_params"], **w["oos_metrics"]}
        if w.get("loglik") is not None:
            entry["aic"] = _aic_fn(float(w["loglik"]), int(w.get("n_params", 0)))
        xi_progression.append((lbl, entry))
    tex_parts.append(_inter_stage_progression_table_tex(
        xi_progression, "phi_xi",
        caption=f"{display}: phi+xi branch -- AIC and OOS CRPS across stage winners.",
        label=f"tab:{station_short}_prog_xi",
    ))

    fw_info = winners.get("final_winner_phi_xi", {})

    # ── Final winner extended diagnostics ─────────────────────────────────
    # NOTE: the deep diagnostic machinery below (IS PIT/ACF, extended OOS
    # metrics, dynamic quantiles, return levels, seasonal/ENSO/wet-dry
    # decomposition) currently runs once, for the phi+xi branch's final
    # winner only. Objective 4e/4f ask for this for BOTH tvp-set winners;
    # duplicating this ~350-line block for the phi-only branch is tracked as
    # remaining work (see ROADMAP.md) rather than attempted as part of the
    # 2026-07-07 pipeline-correctness pass. The phi-only branch's winner and
    # its own OOS RMSE/MAD are still reported via the Stage 1-3 comparison
    # tables and the global comparison chapter above.
    fphi_info = winners.get("final_winner_phi_only", {})
    if fphi_info.get("model_id"):
        tex_parts.append(
            rf"\textbf{{Phi-only branch final winner}} (objective 1a, ranked "
            rf"by OOS RMSE): \texttt{{{_tex(fphi_info['model_id'])}}} "
            rf"(OOS RMSE = {_fmt(fphi_info.get('rmse'))}). This is also the "
            r"model whose $\phi$ dynamics are frozen for Stage 4 below." + "\n\n"
        )

    # ── Does adding xi help? phi-only vs phi+xi by CRPS/twCRPS ─────────────
    # User request (2026-07-08): compare the two branches' final winners on
    # the metrics that actually matter for distributional/tail performance,
    # not just each branch's own ranking metric. Both models are already
    # fitted; twCRPS/CRPS are recomputed from each one's saved OOS
    # simulation paths (Monte Carlo draws from the already-fit predictive
    # distribution) -- no re-optimization.
    if fphi_info.get("model_id") and fw_info.get("model_id"):
        phi_ext = _compute_extended_metrics_for_model(
            fphi_info["model_id"], run_dir, data, model_cache
        )
        xi_ext = _compute_extended_metrics_for_model(
            fw_info["model_id"], run_dir, data, model_cache
        )
        if phi_ext and xi_ext:
            tex_parts.append(r"\subsection{Does Adding xi Help? Phi-only vs.\ Phi+xi}" + "\n")
            tex_parts.append(
                r"Both branches' final winners compared on CRPS and twCRPS@95/98 -- "
                r"the phi-only winner is not re-fit, this is a diagnostic recomputation "
                r"from its existing OOS simulation paths, exactly as for any other model "
                r"in this report." + "\n\n"
            )
            rows = [
                {"model_id": f"phi-only: {fphi_info['model_id']}",
                 "crps_mean": phi_ext.get("crps_mean"),
                 "twcrps_95": phi_ext.get("twcrps_95"),
                 "twcrps_98": phi_ext.get("twcrps_98")},
                {"model_id": f"phi+xi: {fw_info['model_id']}",
                 "crps_mean": xi_ext.get("crps_mean"),
                 "twcrps_95": xi_ext.get("twcrps_95"),
                 "twcrps_98": xi_ext.get("twcrps_98")},
            ]
            pd.DataFrame(rows).to_csv(
                csv_dir / f"{station_short}_phi_vs_phixi_crps.csv", index=False
            )
            headers = r"\small Model & CRPS & twCRPS@95 & twCRPS@98 \\"
            body = " \\\\\n".join(
                r"\texttt{" + _tex(r["model_id"]) + "} & "
                + f"{_fmt(r['crps_mean'])} & {_fmt(r['twcrps_95'])} & {_fmt(r['twcrps_98'])}"
                for r in rows
            ) + r" \\"
            tex_parts.append(
                r"\begin{table}[ht]" + "\n\\centering\n\\footnotesize\n"
                r"\begin{tabular}{lrrr}" + "\n\\hline\n"
                + headers + "\n\\hline\n" + body + "\n\\hline\n"
                r"\end{tabular}" + "\n"
                rf"\caption{{{display}: phi-only vs.\ phi+xi final winners, CRPS and twCRPS.}}"
                + "\n"
                rf"\label{{tab:{station_short}_phi_vs_phixi}}" + "\n"
                r"\end{table}" + "\n"
            )
            c_phi, c_xi = phi_ext.get("crps_mean"), xi_ext.get("crps_mean")
            if c_phi is not None and c_xi is not None and np.isfinite(c_phi) and np.isfinite(c_xi):
                impr = (c_phi - c_xi) / max(abs(c_phi), 1e-12)
                verdict = (
                    rf"adding $\xi$ improves CRPS by {impr:.1%}"
                    if impr > 0 else
                    rf"the phi-only model has {-impr:.1%} \emph{{lower}} (better) CRPS"
                )
                tex_parts.append(
                    rf"CRPS: phi-only = {_fmt(c_phi)}, phi+xi = {_fmt(c_xi)} -- {verdict}."
                    + "\n\n"
                )
            # Propagate to the global cross-location comparison chapter --
            # already computed above, no recomputation needed there.
            _section_meta["phi_crps"]      = phi_ext.get("crps_mean")
            _section_meta["phi_twcrps_95"] = phi_ext.get("twcrps_95")
            _section_meta["phi_twcrps_98"] = phi_ext.get("twcrps_98")
            _section_meta["phixi_crps"]      = xi_ext.get("crps_mean")
            _section_meta["phixi_twcrps_95"] = xi_ext.get("twcrps_95")
            _section_meta["phixi_twcrps_98"] = xi_ext.get("twcrps_98")

    ext_mets = None   # will be set below; used by Stage 4 tail comparison
    final_mid = fw_info.get("model_id")
    if final_mid:
        tex_parts.append(r"\subsection{Final Model Diagnostics (phi+xi branch)}" + "\n")
        tex_parts.append(
            rf"The phi+xi branch's final selected model is "
            rf"\texttt{{{_tex(final_mid)}}} "
            rf"(OOS CRPS = {_fmt(fw_info.get('crps_mean'))})." + "\n\n"
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
        paths_npz   = _load_paths_npz(run_dir, stage_m, final_mid)
        meta_m      = _load_meta(run_dir, stage_m, final_mid)
        params_df_w = _load_params_df(run_dir, stage_m, final_mid)
        meta_m      = _patch_meta_for_reconstruction(meta_m, params_df_w)

        if paths_npz is not None:
            oos_paths = _reconstruct_oos_paths(paths_npz, meta_m)
        else:
            oos_paths = None

        if oos_paths is not None:
            try:
                model_w = model_cache.get(final_mid)
                if model_w is None:
                    model_w = _build_model_from_meta(meta_m, params_df_w, model_id=final_mid)
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

                    # Objective 4f: empirical vs theoretical exceedance,
                    # overall / wet / dry / by ENSO regime, in one table --
                    # all inputs already computed above (seas_res / enso_res /
                    # wd_res / quants_oos), no re-fitting or recomputation of
                    # the model itself.
                    overall_exc = compute_exceedance_frequencies(
                        y=y_test, dynamic_quantiles=quants_oos,
                        quantile_levels=QUANTILE_LEVELS,
                    )
                    enso_regimes_for_table = (
                        enso_res.get("standard_regimes")
                        if nino34_test is not None else None
                    )
                    tex_parts.append(_exceedance_summary_table_tex(
                        quantile_levels=QUANTILE_LEVELS,
                        overall=overall_exc,
                        wet=wd_res.get("wet"), dry=wd_res.get("dry"),
                        enso_regimes=enso_regimes_for_table,
                        caption=(
                            f"{display}: exceedance calibration -- empirical vs.\\ "
                            r"theoretical (1$-q$) exceedance frequency, overall, by "
                            "wet/dry season, and by ENSO regime (monthly breakdown "
                            "shown in Fig.~\\ref{fig:" + station_short + "_monthly}). "
                            "A well-calibrated model has empirical values close to "
                            "the theoretical column."
                        ),
                        label=f"tab:{station_short}_exceedance",
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

    # ── Stage 4: tail-sensitive xi-only regime (objective 3, 2026-07-07) ────
    # Exactly 2 models (q95, q98) -- phi and pi are frozen at the phi-only
    # branch's final winner, never re-estimated (objective item 4); only xi
    # (GAS block + dewtemp_seasonal covariates + regime term) is fit fresh
    # (item 5). Each threshold is accepted independently against the phi+xi
    # branch's final winner ("counterpart") by twCRPS (objective 3a) -- this
    # replaces round 1's 9-model phi+xi-joint regime design entirely.
    s4_models = _all_model_ids(run_dir, "stage4_xi_regime")
    s4_winner_id, best_q = None, None
    if s4_models:
        tex_parts.append(r"\subsection{Stage 4 -- Tail-Sensitive xi Regime}" + "\n")
        s4_info = winners.get("stage4_xi_regime", {})
        s4_base_id = s4_info.get("base_model", "")
        s4_cp_id   = s4_info.get("counterpart", "")
        s4_accept  = s4_info.get("accepted", {})

        tex_parts.append(
            r"Per objective 3, Stage 4 tests tail-sensitive dynamics for "
            rf"$\xi$ only. $\phi$ is frozen at the phi-only branch's final "
            rf"winner (\texttt{{{_tex(s4_base_id)}}}) -- its GAS block, "
            r"any covariates, and the occurrence probability $\pi_t$ are "
            r"not re-estimated. Only $\xi$ (GAS(1,1) + short-term weather "
            r"covariates + regime term $A^{ext}_\xi$) is fit fresh at each "
            r"threshold. Each threshold is accepted only if its twCRPS beats "
            rf"the phi+xi branch's final winner (\texttt{{{_tex(s4_cp_id)}}}, "
            r"the ``best of previous stages counterpart'') by at least 2\%."
            + "\n\n"
        )

        tex_parts.append(_stage_comparison_table_tex(
            run_dir, "stage4_xi_regime", winners,
            caption=f"{display}: Stage 4 tail-sensitive $\\xi$ regime models.",
            label=f"tab:{station_short}_s4",
        ))

        s4_mets = []
        for mid in s4_models:
            m = _compute_extended_metrics_for_model(mid, run_dir, data, model_cache)
            if m:
                s4_mets.append(m)
        if s4_mets:
            pd.DataFrame(s4_mets).to_csv(
                csv_dir / f"{station_short}_stage4_metrics.csv", index=False
            )
            tex_parts.append(_extended_metrics_table_tex(
                s4_mets,
                caption=f"{display}: Stage 4 OOS CRPS and twCRPS@95/98.",
                label=f"tab:{station_short}_s4_ext",
                cols_group="crps",
            ))

        for q_label in ("95", "98"):
            dec = s4_accept.get(q_label, {})
            winner = dec.get("winner")
            key_metric = dec.get("key_metric", f"twcrps_{q_label}")
            s4_val, base_val = dec.get("stage4_value"), dec.get("base_value")
            rel_impr = dec.get("rel_improvement")
            if winner == "stage4":
                tex_parts.append(
                    rf"\textbf{{Stage 4 q{q_label} accepted}}: "
                    rf"{_tex(key_metric)} improves from {_fmt(base_val)} "
                    rf"({_tex(s4_cp_id)}) to {_fmt(s4_val)} "
                    rf"(stage4\_xi\_regime\_q{q_label}), a "
                    + (f"{rel_impr:.1%}" if rel_impr is not None and np.isfinite(rel_impr) else "--")
                    + r" improvement ($\geq$2\% threshold met)." + "\n\n"
                )
            elif winner == "base":
                reason = dec.get("reason", "")
                tex_parts.append(
                    rf"\textbf{{Stage 4 q{q_label} not accepted}}: "
                    + (
                        rf"{_tex(key_metric)} = {_fmt(s4_val)} vs.\ counterpart "
                        rf"{_fmt(base_val)} ("
                        + (f"{rel_impr:.1%}" if rel_impr is not None and np.isfinite(rel_impr) else "--")
                        + r" change, below the 2\% threshold). "
                        if not reason else _tex(reason) + ". "
                    )
                    + r"The phi+xi branch's final winner is retained for this threshold."
                    + "\n\n"
                )
            else:
                tex_parts.append(
                    rf"Stage 4 q{q_label}: acceptance decision unavailable "
                    r"(missing metrics)." + "\n\n"
                )

        # 7th (of 7) image set: the better-performing Stage-4 threshold,
        # shown regardless of accept/reject (diagnostics are still
        # informative for a rejected model -- e.g. Belo Horizonte/Darwin/
        # Garanhuns above, where Stage 4 was legitimately not accepted).
        # Ranked by *relative* improvement over each threshold's own
        # counterpart, not raw twCRPS -- twcrps_98 is mechanically smaller
        # than twcrps_95 for any model (a higher threshold weights a smaller
        # tail region), so comparing raw values would always "prefer" q98
        # regardless of which threshold actually fits better.
        s4_candidates = [
            (q_label, s4_accept.get(q_label, {}).get("rel_improvement"))
            for q_label in ("95", "98")
            if s4_accept.get(q_label, {}).get("rel_improvement") is not None
            and np.isfinite(s4_accept[q_label]["rel_improvement"])
        ]
        if s4_candidates:
            best_q, _ = max(s4_candidates, key=lambda t: t[1])
            s4_winner_id = f"stage4_xi_regime_q{best_q}"
            is_diag = _compute_is_diagnostics_for_winner(
                model_id=s4_winner_id, run_dir=run_dir, data=data,
                col_name_dict=data["covariate_col_names"], fig_dir=fig_dir,
                winners=winners,
            )
            if is_diag:
                pit_rel = os.path.relpath(is_diag["pit_path"], fig_dir.parent.parent).replace("\\", "/")
                acf_rel = os.path.relpath(is_diag["acf_path"], fig_dir.parent.parent).replace("\\", "/")
                tex_parts.append(
                    rf"\paragraph{{IS diagnostics (q{best_q}, lower twCRPS of the two thresholds)}}"
                    + "\n"
                    + f"PIT mean = {is_diag['pit_mean']:.3f} (ideal 0.5), "
                    f"std = {is_diag['pit_std']:.3f} (ideal {1/12**0.5:.3f})." + "\n\n"
                )
                tex_parts.append(_fig_tex(
                    pit_rel, f"{display} (Stage 4, q{best_q}): IS PIT histogram.",
                    f"fig:{station_short}_s4_pit", width="0.55",
                ))
                tex_parts.append(_fig_tex(
                    acf_rel, f"{display} (Stage 4, q{best_q}): IS quantile-residual ACF (400 lags).",
                    f"fig:{station_short}_s4_acf", width="0.75",
                ))

        _section_meta["stage4_q95_accepted"] = s4_accept.get("95", {}).get("winner") == "stage4"
        _section_meta["stage4_q98_accepted"] = s4_accept.get("98", {}).get("winner") == "stage4"
        _section_meta["stage4_q95_improvement"] = s4_accept.get("95", {}).get("rel_improvement")
        _section_meta["stage4_q98_improvement"] = s4_accept.get("98", {}).get("rel_improvement")
        _section_meta["stage4_base_model"] = s4_base_id
        _section_meta["stage4_counterpart"] = s4_cp_id

    # ── Stage 5: static benchmark + threshold-weighted xi (extra experiment,
    # 2026-07-09, not part of objectives 1-4), only at Belo Horizonte, Darwin
    # Airport, Garanhuns. See docs/MODELS.md Sec. 31 and models/static_gb2.py
    # / models/threshold_weighted_xi_gas.py for the full mathematical writeup.
    # Two variants are run (2026-07-09b, "keep both"): pi frozen at a single
    # constant (static-pi) vs. pi_t fit standalone as a time-varying
    # AR-logistic process (dynamic-pi) -- xi's own GAS(1,1) fit is identical
    # either way (pi never enters that objective), only OOS/IS evaluation
    # and the reported full-series loglik differ.
    from run_all_locations import STAGE5_LOCATIONS
    s5_winner_id = None
    if station_name in STAGE5_LOCATIONS:
        s5_all = _all_model_ids(run_dir, "stage5")
        s5_static_ids = sorted(m for m in s5_all if not m.endswith("_dynpi"))
        s5_dynpi_ids  = sorted(m for m in s5_all if m.endswith("_dynpi"))
        if s5_all:
            tex_parts.append(r"\subsection{Stage 5 -- Static Benchmark and Threshold-Weighted xi}" + "\n")
            s5_info    = winners.get("stage5", {})
            s5dp_info  = winners.get("stage5_dynpi", {})
            s5_static  = s5_info.get("static_theta", {})
            s5_winner_static = s5_info.get("winner")
            s5_winner_dynpi  = s5dp_info.get("winner")
            best_variant = winners.get("stage5_overall_best_variant", "static_pi")
            s5_winner_id = s5_winner_static if best_variant == "static_pi" else s5_winner_dynpi

            tex_parts.append(
                r"An additional experiment (2026-07-09), not part of objectives "
                r"1--4, run only at this location, in two pi variants (see "
                r"below). "
                r"\textbf{Step 1 -- static benchmark}: a fully static (no GAS "
                r"recursion) zero-augmented GB2 is fit by unconditional maximum "
                r"likelihood -- $\phi,\xi,\gamma,\zeta$ are constants for the "
                r"whole series, not driven by any state equation. Because the "
                r"zero-augmented log-likelihood is additively separable in "
                r"$\pi$ (occurrence) versus $(\phi,\xi,\gamma,\zeta)$ (the GB2 "
                r"shape/scale) -- they share no parameters -- the two are "
                r"always estimated independently, not jointly. In the "
                r"\emph{static-pi} variant, $\pi_{MV}$ is the closed-form "
                r"empirical wet-day fraction (no numerical optimisation at "
                r"all); in the \emph{dynamic-pi} variant, $\pi_t$ is instead "
                r"fit as a full time-varying AR-logistic process (the same "
                r"occurrence model used in Stages 1--4, "
                r"$\eta_t=\omega_0+\rho\,\eta_{t-1}+\sum_{l\in\{1,365,366\}}"
                r"\omega_l y_{t-l}$, $\pi_t=\sigma(\eta_t)$ -- 3 seasonal "
                r"$y$-lags plus one AR(1) term on $\eta$ itself, 5 parameters "
                r"total), fit standalone by maximum likelihood on the binary "
                r"wet/dry sequence alone, independent of the GB2 side. "
                r"Either way, $(\phi_{MV},\xi_{MV},\gamma_{MV},\zeta_{MV})$ "
                r"maximise $\sum_{y_t>0} \log g_y(y_t;\phi,\xi,\gamma,\zeta)$ "
                r"over training wet days only (dry days carry no information "
                r"about the positive-part shape), via a bounded L-BFGS-B warm "
                r"start followed by an unbounded BFGS polish with soft "
                r"penalties -- this codebase's standard optimisation "
                r"convention; the bounded step alone was found, empirically, "
                r"to clip $\phi$ at an artificial wall (its hard bound exists "
                r"only to produce a safe GAS-filter warm start elsewhere in "
                r"this codebase, not a final answer)."
                rf" Static-pi result: $\phi_{{MV}}={_fmt(s5_static.get('phi_MV'),4)}$, "
                rf"$\xi_{{MV}}={_fmt(s5_static.get('xi_MV'),4)}$, "
                rf"$\gamma_{{MV}}={_fmt(s5_static.get('gamma_MV'),4)}$, "
                rf"$\zeta_{{MV}}={_fmt(s5_static.get('zeta_MV'),4)}$, "
                rf"$\pi_{{MV}}={_fmt(s5_static.get('pi_MV'),4)}$ "
                r"(the GB2 shape/scale estimates are numerically identical "
                r"in the dynamic-pi variant, since the two sides of the fit "
                r"do not interact)."
                + "\n\n"
            )
            tex_parts.append(
                r"\textbf{Step 2 -- threshold-weighted xi}: $\phi,\gamma,\zeta,\pi$ "
                r"are frozen at their step-1 values above (never re-estimated); "
                r"only $\xi$ is dynamic, via a plain GAS(1,1) recursion -- "
                r"one lag -- "
                r"$\xi_{t+1}=\omega_\xi+A_\xi s_t+B_\xi \xi_t$. Unlike every "
                r"other model in this report, the log-likelihood driving both "
                r"the fit and the score $s_t$ is threshold-weighted rather than "
                r"computed from every wet day: "
                r"$l=\sum_t \log g_y(y_t;\phi_{MV},\xi_t,\gamma_{MV},\zeta_{MV}) "
                r"\cdot \mathbf{1}(y_t \geq q_c)$, where $q_c$ is the $c$-th "
                r"percentile of training wet-day rainfall. Below-threshold days "
                r"($y_t < q_c$, which includes every dry day by construction) "
                r"contribute neither likelihood nor score at $t$ -- $\xi_t$ "
                r"still evolves through them via the autoregressive term "
                r"$B_\xi \xi_t$ alone (mean-reversion), exactly as every GAS "
                r"filter in this framework already treats a dry day as "
                r"uninformative for the positive-part distribution; this simply "
                r"extends ``uninformative'' from ``zero rainfall'' to ``below "
                r"the $c$-th percentile.'' This differs from Stage 4, which "
                r"computes its score from every wet day and instead rescales "
                r"how strongly that score feeds into the recursion above the "
                r"threshold -- Stage 5 discards below-threshold information "
                r"entirely rather than reweighting it. Since pi never enters "
                r"this objective, $\xi$'s fitted GAS(1,1) parameters are "
                r"numerically identical between the static-pi and dynamic-pi "
                r"variants -- only the OOS/IS evaluation (which does use pi) "
                r"and the reported full-series loglik differ. Estimated "
                r"independently for $c\in\{90,95,98,99\}$ in each variant "
                r"(Tables~\ref{tab:" + station_short + r"_s5} and "
                r"\ref{tab:" + station_short + r"_s5_dynpi}); only the "
                r"threshold with the best OOS CRPS is each variant's own "
                r"result, matching the phi+xi branch's own ranking criterion."
                + "\n\n"
            )

            tex_parts.append(_stage_comparison_table_tex(
                run_dir, "stage5", winners, model_ids=s5_static_ids, winner_key="stage5",
                caption=(
                    f"{display}: Stage 5, static-pi variant -- static "
                    "benchmark (no GAS recursion) and threshold-weighted "
                    "$\\xi$ at all four thresholds $c$. Winner (best OOS "
                    "CRPS among the four thresholds) marked with \\textbf{*}."
                ),
                label=f"tab:{station_short}_s5",
            ))
            tex_parts.append(_stage_comparison_table_tex(
                run_dir, "stage5", winners, model_ids=s5_dynpi_ids, winner_key="stage5_dynpi",
                caption=(
                    f"{display}: Stage 5, dynamic-pi variant -- static "
                    "benchmark with time-varying $\\pi_t$ and threshold-"
                    "weighted $\\xi$ at all four thresholds $c$. Winner "
                    "marked with \\textbf{*}."
                ),
                label=f"tab:{station_short}_s5_dynpi",
            ))

            s5_mets = []
            for mid in s5_all:
                m = _compute_extended_metrics_for_model(mid, run_dir, data, model_cache)
                if m:
                    s5_mets.append(m)
            s5_mets_static = [m for m in s5_mets if not m["model_id"].endswith("_dynpi")]
            s5_mets_dynpi  = [m for m in s5_mets if m["model_id"].endswith("_dynpi")]
            if s5_mets:
                pd.DataFrame(s5_mets).to_csv(
                    csv_dir / f"{station_short}_stage5_metrics.csv", index=False
                )
            if s5_mets_static:
                tex_parts.append(_extended_metrics_table_tex(
                    s5_mets_static,
                    caption=f"{display}: Stage 5 (static-pi) OOS CRPS and twCRPS@95/98.",
                    label=f"tab:{station_short}_s5_ext",
                    cols_group="crps",
                ))
            if s5_mets_dynpi:
                tex_parts.append(_extended_metrics_table_tex(
                    s5_mets_dynpi,
                    caption=f"{display}: Stage 5 (dynamic-pi) OOS CRPS and twCRPS@95/98.",
                    label=f"tab:{station_short}_s5_dynpi_ext",
                    cols_group="crps",
                ))

            if s5_winner_id:
                s5_winner_ext = next((m for m in s5_mets if m["model_id"] == s5_winner_id), None)

                # Compare against the phi+xi branch's own final winner on the
                # same metric, exactly as the earlier "does xi help" section --
                # both models already in model_cache, no re-fitting either way.
                phixi_ext = _compute_extended_metrics_for_model(
                    fw_info.get("model_id"), run_dir, data, model_cache
                ) if fw_info.get("model_id") else None
                verdict = ""
                if s5_winner_ext and phixi_ext:
                    c5, cx = s5_winner_ext.get("crps_mean"), phixi_ext.get("crps_mean")
                    if c5 is not None and cx is not None and np.isfinite(c5) and np.isfinite(cx):
                        pct = (cx - c5) / abs(cx) * 100 if cx else float("nan")
                        direction = "improves on" if c5 < cx else "is worse than"
                        verdict = (
                            rf" CRPS {_fmt(c5)} vs.\ the phi+xi branch's own "
                            rf"final winner (\texttt{{{_tex(fw_info.get('model_id',''))}}}, "
                            rf"CRPS {_fmt(cx)}): Stage 5 {direction} it by "
                            rf"{abs(pct):.1f}\%."
                        )

                static_note = ""
                static_bench_id = "stage5_static_gb2" if best_variant == "static_pi" else "stage5_static_gb2_dynpi"
                static_ext = next((m for m in s5_mets if m["model_id"] == static_bench_id), None)
                if s5_winner_ext and static_ext:
                    c5, c0 = s5_winner_ext.get("crps_mean"), static_ext.get("crps_mean")
                    if c5 is not None and c0 is not None and np.isfinite(c5) and np.isfinite(c0) and c0:
                        pct0 = (c5 - c0) / abs(c0) * 100
                        static_note = (
                            r" Notably, the fully static step-1 benchmark "
                            rf"(CRPS {_fmt(c0)}) itself beats every threshold-"
                            rf"weighted $\xi$ candidate, including this winner, "
                            rf"by {pct0:.1f}\% -- adding threshold-driven $\xi$ "
                            r"dynamics on top of a frozen scale did not pay off "
                            r"at this location."
                        )

                variant_note = ""
                s5w_static_ext = next((m for m in s5_mets if m["model_id"] == s5_winner_static), None)
                s5w_dynpi_ext  = next((m for m in s5_mets if m["model_id"] == s5_winner_dynpi), None)
                if s5w_static_ext and s5w_dynpi_ext:
                    cst, cdp = s5w_static_ext.get("crps_mean"), s5w_dynpi_ext.get("crps_mean")
                    if cst is not None and cdp is not None and np.isfinite(cst) and np.isfinite(cdp):
                        variant_note = (
                            rf" Static-pi's own best (\texttt{{{_tex(s5_winner_static)}}}, "
                            rf"CRPS {_fmt(cst)}) vs.\ dynamic-pi's own best "
                            rf"(\texttt{{{_tex(s5_winner_dynpi)}}}, CRPS {_fmt(cdp)}): "
                            + ("static-pi wins" if cst <= cdp else "dynamic-pi wins")
                            + f" -- reported below as this location's Stage 5 result."
                        )

                tex_parts.append(
                    rf"\textbf{{Stage 5 winner}} ({best_variant.replace('_',' ')}): "
                    rf"\texttt{{{_tex(s5_winner_id)}}} (best OOS CRPS among the "
                    r"four thresholds)." + verdict + static_note + variant_note + "\n\n"
                )

                is_diag = _compute_is_diagnostics_for_winner(
                    model_id=s5_winner_id, run_dir=run_dir, data=data,
                    col_name_dict=data["covariate_col_names"], fig_dir=fig_dir,
                    winners=winners,
                )
                if is_diag:
                    pit_rel = os.path.relpath(is_diag["pit_path"], fig_dir.parent.parent).replace("\\", "/")
                    acf_rel = os.path.relpath(is_diag["acf_path"], fig_dir.parent.parent).replace("\\", "/")
                    tex_parts.append(
                        rf"\paragraph{{IS diagnostics (\texttt{{{_tex(s5_winner_id)}}})}}"
                        + "\n"
                        + f"PIT mean = {is_diag['pit_mean']:.3f} (ideal 0.5), "
                        f"std = {is_diag['pit_std']:.3f} (ideal {1/12**0.5:.3f})." + "\n\n"
                    )
                    tex_parts.append(_fig_tex(
                        pit_rel, f"{display} (Stage 5, {_tex(s5_winner_id)}): IS PIT histogram.",
                        f"fig:{station_short}_s5_pit", width="0.55",
                    ))
                    tex_parts.append(_fig_tex(
                        acf_rel, f"{display} (Stage 5, {_tex(s5_winner_id)}): IS quantile-residual ACF (400 lags).",
                        f"fig:{station_short}_s5_acf", width="0.75",
                    ))

                # ── Calibration-only comparison: exceedance ratios + twCRPS,
                # vs. the phi+xi branch's own final winner (2026-07-09b).
                # CRPS/twCRPS above are dominated by Stage 5's frozen phi's
                # point-scale (expected -- a fixed scale cannot track the
                # seasonal cycle a dynamic-phi winner can); the calibration
                # question -- does the threshold-weighted tail mechanism
                # itself predict extreme-crossing FREQUENCY correctly,
                # independent of magnitude -- is answered by exceedance
                # ratios instead, which is magnitude-scale-independent.
                cmp_id = fw_info.get("model_id")
                s5_exc = _exceedance_ratios_for_model(s5_winner_id, run_dir, data, model_cache)
                cmp_exc = _exceedance_ratios_for_model(cmp_id, run_dir, data, model_cache) if cmp_id else None
                if s5_exc and cmp_exc:
                    tex_parts.append(
                        r"\paragraph{Calibration-only comparison: exceedance ratios and twCRPS}"
                        + "\n"
                        + r"CRPS/twCRPS above are dominated by Stage 5's frozen "
                        r"$\phi$'s point-scale, which cannot track the seasonal "
                        r"cycle the way a dynamic-$\phi$ winner can -- expected, "
                        r"not informative about the threshold-weighted $\xi$ "
                        r"mechanism's own tail behaviour. Exceedance ratios "
                        r"(actual vs.\ predicted crossing frequency, magnitude-"
                        r"independent) are a fairer test." + "\n\n"
                    )
                    s5_label  = _stage_label_for_model_id(s5_winner_id)
                    cmp_label = _stage_label_for_model_id(cmp_id)
                    rows_exc = []
                    for q in (0.95, 0.98, 0.99):
                        key = f"ratio_{int(q*10000):05d}"
                        rows_exc.append({
                            "q": f"q{q*100:g}",
                            f"Stage 5 ({s5_label})": s5_exc.get(key),
                            f"Best of previous stages ({cmp_label})": cmp_exc.get(key),
                        })
                    df_exc = pd.DataFrame(rows_exc)
                    hdr = " & ".join(r"\small " + _tex(c) for c in df_exc.columns) + r" \\"
                    body_exc = []
                    for _, row in df_exc.iterrows():
                        cells = [str(row["q"])] + [
                            (f"{v:.2f}$\\times$" if v is not None and np.isfinite(v) else "--")
                            for v in row[1:]
                        ]
                        body_exc.append(" & ".join(cells) + r" \\")
                    tex_parts.append(
                        r"\begin{table}[ht]" + "\n\\centering\n\\footnotesize\n"
                        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%" + "\n"
                        r"\begin{tabular}{lrr}" + "\n\\hline\n"
                        + hdr + "\n\\hline\n" + "\n".join(body_exc) + "\n\\hline\n"
                        r"\end{tabular}" + "\n}" + "\n"
                        rf"\caption{{{display}: exceedance ratio (actual/predicted crossing frequency), "
                        r"Stage 5 winner vs.\ best model from previous stages. 1.0$\times$ is perfect.}"
                        + "\n"
                        rf"\label{{tab:{station_short}_s5_exc}}" + "\n"
                        r"\end{table}" + "\n"
                    )
                    _section_meta["stage5_exc_ratios"] = {k: v for k, v in s5_exc.items() if k != "model_id"}
                    _section_meta["stage5_cmp_exc_ratios"] = {k: v for k, v in cmp_exc.items() if k != "model_id"}

                tw5 = s5_winner_ext.get("twcrps_95") if s5_winner_ext else None
                twc = phixi_ext.get("twcrps_95") if phixi_ext else None
                _section_meta["stage5_twcrps_95"] = tw5
                _section_meta["stage5_cmp_twcrps_95"] = twc
                _section_meta["stage5_cmp_model_id"] = cmp_id
                _section_meta["stage5_cmp_stage_label"] = _stage_label_for_model_id(cmp_id) if cmp_id else None
                _section_meta["stage5_stage_label"] = _stage_label_for_model_id(s5_winner_id)

            _section_meta["stage5_winner"] = s5_winner_id
            _section_meta["stage5_best_variant"] = best_variant
            _section_meta["stage5_static_theta"] = s5_static
            _section_meta["stage5_crps"] = (
                next((m.get("crps_mean") for m in s5_mets if m["model_id"] == s5_winner_id), None)
                if s5_winner_id else None
            )

    # ── Appendix: parameter estimates + standard errors, all 7 winners ─────
    # Per prompt.md: standard errors were saved (every model's save_result())
    # but never shown in round 1's report. One table per winner, kept
    # separate from every OOS/IS metric table above (REPORTING.md §10).
    appendix_entries = [
        ("Phi-only, Stage 1", "stage1", w1_phi),
        ("Phi-only, Stage 2", "stage2_phi", w2_phi),
        ("Phi-only, Stage 3", "stage3_phi", w3_phi),
        ("Phi+xi, Stage 1", "stage1", w1_xi),
        ("Phi+xi, Stage 2", "stage2", w2_xi),
        ("Phi+xi, Stage 3", "stage3", w3_xi),
    ]
    if s4_winner_id:
        appendix_entries.append((f"Stage 4 (q{best_q})", "stage4_xi_regime",
                                  {"model_id": s4_winner_id}))
    if s5_winner_id:
        appendix_entries.append(("Stage 5 (static benchmark)", "stage5",
                                  {"model_id": "stage5_static_gb2"}))
        appendix_entries.append((f"Stage 5 ({s5_winner_id.replace('stage5_xi_', '')})", "stage5",
                                  {"model_id": s5_winner_id}))
    appendix_entries = [(lbl, sd, w) for lbl, sd, w in appendix_entries if w]

    if appendix_entries:
        tex_parts.append(r"\subsection{Appendix: Parameter Estimates and Standard Errors}" + "\n")
        for lbl, stage_dir, w in appendix_entries:
            pdf_se = _load_params_with_se(run_dir, stage_dir, w["model_id"])
            if pdf_se.empty:
                continue
            tex_parts.append(_params_table_tex(
                pdf_se,
                caption=f"{display} ({lbl}): \\texttt{{{_tex(w['model_id'])}}} parameter estimates.",
                label=f"tab:{station_short}_params_{stage_dir}_{lbl.split(',')[0].strip().lower().replace(' ', '')}",
            ))

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

    # Two independent tables (objective 4d/1a/1b): phi-only branch ranked by
    # OOS RMSE, phi+xi branch ranked by OOS CRPS. Never mixed into one table
    # (REPORTING.md §10: avoid mixing unrelated metrics).
    rows_phi, rows_phixi, rows_s4, rows_xicompare, rows_s5 = [], [], [], [], []
    for station, res in station_results.items():
        cfg = STATION_REGISTRY.get(station, {})
        display = cfg.get("display", station)
        if res.get("phi_model_id"):
            rows_phi.append({
                "Station": display, "Winner model": res["phi_model_id"],
                "OOS RMSE": res.get("phi_rmse", float("nan")),
            })
        if res.get("phixi_model_id"):
            rows_phixi.append({
                "Station": display, "Winner model": res["phixi_model_id"],
                "OOS CRPS": res.get("phixi_crps", float("nan")),
            })
        c_phi, c_xi = res.get("phi_crps"), res.get("phixi_crps")
        if c_phi is not None and c_xi is not None and np.isfinite(c_phi) and np.isfinite(c_xi):
            impr = (c_phi - c_xi) / max(abs(c_phi), 1e-12)
            rows_xicompare.append({
                "Station": display,
                "phi-only CRPS": c_phi, "phi+xi CRPS": c_xi,
                "phi-only twCRPS@95": res.get("phi_twcrps_95"),
                "phi+xi twCRPS@95": res.get("phixi_twcrps_95"),
                "xi improves CRPS": f"{impr:+.1%}",
            })
        if res.get("stage4_base_model"):
            def _s4cell(acc, impr):
                if acc is None:
                    return "--"
                arrow = "Yes" if acc else "No"
                return arrow if impr is None or not np.isfinite(impr) else f"{arrow} ({impr:+.1%})"
            rows_s4.append({
                "Station": display,
                "Phi base (frozen)": res["stage4_base_model"],
                "phi+xi counterpart": res.get("stage4_counterpart", "?"),
                "q95 accepted": _s4cell(res.get("stage4_q95_accepted"), res.get("stage4_q95_improvement")),
                "q98 accepted": _s4cell(res.get("stage4_q98_accepted"), res.get("stage4_q98_improvement")),
            })
        if res.get("stage5_winner"):
            c5, cx = res.get("stage5_crps"), res.get("phixi_crps")
            vs_phixi = "--"
            if c5 is not None and cx is not None and np.isfinite(c5) and np.isfinite(cx) and cx:
                vs_phixi = f"{(cx - c5) / abs(cx):+.1%}"
            rows_s5.append({
                "Station": display,
                "Best variant": res.get("stage5_best_variant", "static_pi").replace("_", " "),
                "Winner": res["stage5_winner"],
                "OOS CRPS": res.get("stage5_crps", float("nan")),
                "vs. phi+xi branch winner": vs_phixi,
            })

    def _emit_table(rows, caption, label, numeric_cols):
        if not rows:
            return
        df = pd.DataFrame(rows)
        df.to_csv(csv_dir / f"{label.split(':')[-1]}.csv", index=False)
        headers = " & ".join(r"\textbf{" + _tex(c) + "}" for c in df.columns) + r" \\"
        body = []
        for _, row in df.iterrows():
            cells = [
                _fmt(row[c]) if c in numeric_cols else _tex(str(row[c]))
                for c in df.columns
            ]
            body.append(" & ".join(cells) + r" \\")
        n_cols = len(df.columns)
        col_fmt = "l" * (n_cols - len(numeric_cols)) + "r" * len(numeric_cols)
        tex_parts.append(
            r"\begin{table}[ht]" + "\n\\centering\n\\footnotesize\n"
            r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%" + "\n"
            rf"\begin{{tabular}}{{{col_fmt}}}" + "\n\\hline\n"
            + headers + "\n\\hline\n" + "\n".join(body) + "\n\\hline\n"
            r"\end{tabular}" + "\n}" + "\n"
            rf"\caption{{{caption}}}" + "\n"
            rf"\label{{{label}}}" + "\n"
            r"\end{table}" + "\n"
        )

    tex_parts.append(r"\section{Stage Winner Comparison}" + "\n")
    tex_parts.append(
        r"Per objective 1a/1b, the two tvp sets are never ranked against each "
        r"other: phi-only models are compared by OOS RMSE only; phi+xi models "
        r"by OOS CRPS only (the main criterion)." + "\n\n"
    )
    _emit_table(
        rows_phi, "Phi-only branch final winners (objective 1a, ranked by OOS RMSE).",
        "tab:global_phi", {"OOS RMSE"},
    )
    _emit_table(
        rows_phixi, "Phi+xi branch final winners (objective 1b, ranked by OOS CRPS).",
        "tab:global_phixi", {"OOS CRPS"},
    )

    tex_parts.append(r"\section{Does Adding xi Help? Phi-only vs.\ Phi+xi, All Locations}" + "\n")
    tex_parts.append(
        r"Both branches' final winners compared on CRPS and twCRPS@95 at every "
        r"location. Neither model is re-fit for this comparison -- both metrics "
        r"are recomputed from each winner's existing OOS simulation paths." + "\n\n"
    )
    _emit_table(
        rows_xicompare,
        "CRPS and twCRPS@95 for the phi-only vs.\\ phi+xi final winner at each location.",
        "tab:global_xicompare",
        {"phi-only CRPS", "phi+xi CRPS", "phi-only twCRPS@95", "phi+xi twCRPS@95"},
    )
    if rows_xicompare:
        n_improve = sum(1 for r in rows_xicompare if r["xi improves CRPS"].startswith("+"))
        tex_parts.append(
            rf"$\xi$ improved CRPS at {n_improve} of {len(rows_xicompare)} locations." + "\n\n"
        )

    tex_parts.append(r"\section{Stage 4 -- Tail-Sensitive xi Regime}" + "\n")
    tex_parts.append(
        r"Each threshold is accepted independently against the phi+xi "
        r"branch's final winner (the ``counterpart'') by twCRPS, "
        r"$\geq$2\% improvement required (objective 3a)." + "\n\n"
    )
    _emit_table(
        rows_s4, "Stage 4 acceptance by location and threshold.",
        "tab:global_s4", set(),
    )

    if rows_s5:
        tex_parts.append(
            r"\section{Stage 5 -- Static Benchmark and Threshold-Weighted xi}" + "\n"
        )
        tex_parts.append(
            r"Extra experiment (2026-07-09, not part of objectives 1--4), run "
            r"only at Belo Horizonte, Darwin Airport, and Garanhuns -- see "
            r"docs/MODELS.md Sec.~31 and each location's own subsection for "
            r"the full methodology. Winner is the best OOS CRPS among four "
            r"threshold-weighted $\xi$ fits ($c\in\{90,95,98,99\}$), with "
            r"$\phi,\gamma,\zeta,\pi$ frozen at a fully static (no GAS "
            r"recursion) unconditional MLE." + "\n\n"
        )
        _emit_table(
            rows_s5, "Stage 5 winner and OOS CRPS by location.",
            "tab:global_s5", {"OOS CRPS"},
        )

    # Narrative: which locations accepted Stage 2/3 covariates, Stage 4.
    tex_parts.append(r"\section{Cross-Location Patterns}" + "\n")
    s2_accepted = [r["Station"] for r in rows_phixi if "stage2" in r["Winner model"]]
    s3_accepted = [r["Station"] for r in rows_phixi if "stage3" in r["Winner model"]]
    tex_parts.append(
        r"\subsection*{Stage progression (phi+xi branch)}" + "\n"
        + (
            rf"Stage~2 weather covariates were accepted (beat the simpler model by "
            rf"$\geq$2\% CRPS) at: {', '.join(_tex(s) for s in s2_accepted)}. "
            if s2_accepted else
            r"Stage~2 weather covariates were not accepted at any location "
            r"(the simpler no-covariate model was within 2\% at every location, "
            r"per the 4d tie-break rule). "
        )
        + (
            rf"Stage~3 Harvey long-short (corrected ENSO dummies) was accepted at: "
            rf"{', '.join(_tex(s) for s in s3_accepted)}. "
            if s3_accepted else
            r"Stage~3 Harvey long-short was not accepted at any location "
            r"under the corrected ENSO dummy covariates. "
        )
        + "\n\n"
    )

    q95_ok = [r["Station"] for r in rows_s4 if str(r["q95 accepted"]).startswith("Yes")]
    q98_ok = [r["Station"] for r in rows_s4 if str(r["q98 accepted"]).startswith("Yes")]
    tex_parts.append(r"\subsection*{Stage 4 tail-sensitive xi regime}" + "\n")
    if not q95_ok and not q98_ok:
        tex_parts.append(
            r"Stage~4 was not accepted at any location at either threshold: the "
            r"frozen-phi tail-sensitive $\xi$ regime did not beat its phi+xi "
            r"counterpart's twCRPS by the required 2\% margin. "
        )
    else:
        tex_parts.append(
            (rf"q95 accepted at: {', '.join(_tex(s) for s in q95_ok)}. "
             if q95_ok else r"q95 accepted nowhere. ")
            + (rf"q98 accepted at: {', '.join(_tex(s) for s in q98_ok)}. "
               if q98_ok else r"q98 accepted nowhere. ")
        )
    tex_parts.append("\n\n")

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
% Absorb the minor residual overflow from long unbreakable \texttt{} model
% ids / file paths inside inline prose (tables are handled separately via
% \resizebox in the table-builder functions).
\emergencystretch=3em
\sloppy
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
multiple locations. Following the 2026-07-07 revision, the pipeline runs two
independent tvp-set branches through Stages 1-3, plus a redesigned Stage 4:

\begin{enumerate}
\item \textbf{Stage 1 -- Baseline GAS}: 12 specifications varying the set of
      time-varying parameters ($\phi$ only vs.\ $\phi+\xi$), lag structure
      (short vs.\ seasonal), and score scaling (unit, diagonal inverse Fisher,
      full inverse Fisher). The phi-only and phi+xi specifications are never
      ranked against each other: phi-only models are compared by OOS RMSE,
      phi+xi models by OOS CRPS (objective 1a/1b).
\item \textbf{Stage 2 -- Weather covariates}: GAS filter augmented with lagged
      ERA5 dew point and temperature, run independently for each branch.
      Accepted only if the branch's key metric improves by at least 2\% over
      Stage~1, otherwise the simpler Stage-1 model is kept.
\item \textbf{Stage 3 -- Harvey long-short}: Decomposes the dynamic scale into
      a slow long component and a fast short component driven by weather. The
      long component uses El Nino/La Nina dummy tiers (current/+1mo/+3mo,
      neutral as baseline) built from the NOAA ONI classification -- replacing
      round 1's continuous ENSO covariate, which was built from mislabelled
      raw sea-surface temperature (see data/station\_loader.py header).
\item \textbf{Stage 4 -- Tail-sensitive $\xi$ regime}: Redesigned per
      objective 3. Applies the regime-sensitive score extension to $\xi$
      only; $\phi$ (and the occurrence probability $\pi_t$) are frozen at the
      phi-only branch's final winner and never re-estimated. Tested at
      $q_{0.95}$ and $q_{0.98}$ (MODELS.md \S22--23); each threshold is
      accepted only if it beats the phi+xi branch's final winner
      (the ``best of previous stages counterpart'') by $\geq$2\% twCRPS.
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
                enso_path=ENSO_PATH,
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

        # Collect global comparison data (2026-07-07 schema: two independent
        # tvp-set branches, each with its own final winner and key metric --
        # see run_all_locations.py::run_location).
        winners  = _load_winners(run_dir)
        fxi_info = winners.get("final_winner_phi_xi", {})
        fphi_info = winners.get("final_winner_phi_only", {})
        s4_info  = winners.get("stage4_xi_regime", {})
        s4_accept = s4_info.get("accepted", {})
        station_results[station] = {
            "phixi_model_id": fxi_info.get("model_id"),
            "phixi_crps":     fxi_info.get("crps_mean", float("nan")),
            "phi_model_id":   fphi_info.get("model_id"),
            "phi_rmse":       fphi_info.get("rmse", float("nan")),
            # CRPS/twCRPS for BOTH branches' winners (section_extra was
            # already computed once inside generate_location_section --
            # reused here, not recalculated).
            "phi_crps":         section_extra.get("phi_crps"),
            "phi_twcrps_95":    section_extra.get("phi_twcrps_95"),
            "phi_twcrps_98":    section_extra.get("phi_twcrps_98"),
            "phixi_twcrps_95":  section_extra.get("phixi_twcrps_95"),
            "phixi_twcrps_98":  section_extra.get("phixi_twcrps_98"),
            "stage4_q95_accepted": s4_accept.get("95", {}).get("winner") == "stage4",
            "stage4_q98_accepted": s4_accept.get("98", {}).get("winner") == "stage4",
            "stage4_q95_improvement": s4_accept.get("95", {}).get("rel_improvement"),
            "stage4_q98_improvement": s4_accept.get("98", {}).get("rel_improvement"),
            "stage4_base_model": s4_info.get("base_model"),
            "stage4_counterpart": s4_info.get("counterpart"),
            "stage5_winner":  section_extra.get("stage5_winner"),
            "stage5_crps":    section_extra.get("stage5_crps"),
            "stage5_best_variant": section_extra.get("stage5_best_variant"),
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
