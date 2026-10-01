"""
Short-vs-seasonal-lag GAS analysis: baseline (Stage 1) and weather-covariate
(Stage 2) models, both TVP branches, all locations (2026-09-01).

=============================================================================
OVERVIEW
=============================================================================

Companion analysis to generate_seasonality_gas_report.py. For every location
x branch (phi-only, phi+xi) x stage (Stage 1 baseline / Stage 2 dew-point
weather covariate) x lag set (short L={1,2,3} / seasonal L={1,2,3,364..367}),
using ONLY already-fitted, already-saved models (no re-optimisation, per
docs/EXECUTION.md):

  1. In-sample quantile residuals (for the report's ACF/QQ mosaics).
  2. Multi-step-ahead (h=1,2,5,10,30,60,90) Monte Carlo OOS RMSE/CRPS, via
     diagnostics/multistep_forecast.py -- extended in this pass to support
     plain ZAGASModel (no covariates at all), which it did not before.
     1-step OOS RMSE/CRPS is NOT recomputed: it is already saved in every
     model's metadata.json (rmse, crps_mean) by the original Stage 1/2
     pipeline run, and is read from there directly.
  3. Optimisation-difficulty diagnostics for the "why" chapter: gradient
     norm, iteration/function-evaluation counts, and runtime (all already in
     metadata.json, read directly); PLUS a collinearity diagnostic computed
     from the saved BFGS approximate inverse-Hessian (hess_inv.npy,
     OPTIMIZATION.md \u00a79's sanctioned source for approximate covariance) --
     the mean absolute pairwise correlation among the seasonal-lag
     coefficients (A/B at lags 364-367) versus among the short-lag
     coefficients (A/B at lags 1-3) within the SAME seasonal-lag-set model,
     and the condition number of the approximate Hessian (inverse of
     hess_inv) for short vs. seasonal fits.

Stage 1 has three score-scaling variants (unit, diagonal inverse Fisher,
full inverse Fisher) per (branch, lag set); this script picks each (branch,
lag set) pair's own best-by-CRPS scaling (matching how Stage 2 already
builds on a single chosen Stage-1 scaling), so exactly one model represents
each (stage, branch, lag set) cell, symmetric with Stage 2 (no scaling
axis).

=============================================================================
USAGE
=============================================================================

    python run_seasonality_gas_analysis.py
    python run_seasonality_gas_analysis.py --location "DARWIN AIRPORT"

Output (per location):
    artifacts/seasonality_gas_experiment/<short>/<stage>_<branch>_<lagset>/
        residuals.npy, multistep.csv
    artifacts/seasonality_gas_experiment/summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import STATION_REGISTRY, StationDataLoader
from pipeline.artifact_utils import load_model_and_theta
from diagnostics.residuals import quantile_residuals
from diagnostics.multistep_forecast import multistep_metrics, DEFAULT_HORIZONS

PRECIP_DIR = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR   = ROOT / "data" / "input" / "ERA5"
ENSO_PATH  = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
OUT_DIR    = ARTIFACTS_DIR / "seasonality_gas_experiment"

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]
BRANCHES = ["phi", "phixi"]
LAGSETS  = ["short", "seasonal"]
SCALINGS = ["unit", "diagfi", "fullfi"]
DEWPOINT_BLOCK = {"short": "dewpoint_short", "seasonal": "dewpoint_seasonal"}

N_ORIGINS_STRIDE = 15
N_DRAWS = 400
MC_SEED = 2026


# ─────────────────────────────────────────────────────────────────────────────
# Model-id resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_stage1_id(winners: dict, branch: str, lagset: str) -> str:
    """Best-by-CRPS scaling for this (branch, lag set) pair, from
    stage_winners.json's 'stage1' models list (all 12 Stage-1 models, all
    branches/lagsets/scalings, are listed there together)."""
    prefix = f"stage1_{branch}_{lagset}_"
    candidates = [m for m in winners["stage1"]["models"] if m["model_id"].startswith(prefix)]
    finite = [m for m in candidates if np.isfinite(m.get("crps_mean", np.nan))]
    if not finite:
        raise ValueError(f"No finite-CRPS Stage-1 model found for prefix {prefix}")
    return min(finite, key=lambda m: m["crps_mean"])["model_id"]


def stage2_id(branch: str, lagset: str) -> str:
    prefix = "stage2_phi_" if branch == "phi" else "stage2_"
    return f"{prefix}dewpoint_{lagset}"


def stage_subdir(stage: str, branch: str) -> str:
    if stage == "stage1":
        return "stage1"
    return "stage2_phi" if branch == "phi" else "stage2"


# ─────────────────────────────────────────────────────────────────────────────
# Collinearity / optimisation-difficulty diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def _pairwise_abs_corr(hess_inv: np.ndarray, idx: list) -> Optional[float]:
    """Mean |correlation| among the given parameter indices, from a BFGS
    approximate inverse-Hessian (covariance) matrix. None if <2 indices or
    any implied variance is non-positive (unusable approximate SE)."""
    if len(idx) < 2:
        return None
    sub = hess_inv[np.ix_(idx, idx)]
    d = np.diag(sub)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        return None
    denom = np.sqrt(np.outer(d, d))
    corr = sub / denom
    n = len(idx)
    off_diag = corr[np.triu_indices(n, k=1)]
    if not np.all(np.isfinite(off_diag)):
        return None
    return float(np.mean(np.abs(off_diag)))


def collinearity_diagnostics(model_dir: Path, branch_tv: str) -> dict:
    """
    branch_tv: the GAS tv-parameter name whose A_/B_ coefficients to inspect
    ("phi" for both branches -- phixi's location parameter is also named
    "phi" in this codebase's ZAGASModel convention).

    Returns the condition number of the GAS-lag-coefficient BLOCK of the
    approximate covariance (hess_inv submatrix restricted to this
    parameter's own A_/B_ entries -- NOT the full parameter vector, whose
    condition number is dominated by unrelated static/occurrence-model
    parameters on very different scales and is not a fair short-vs-seasonal
    comparison), and (for a model with lags 364-367 present) the mean
    |corr| among the 4 seasonal-lag A coefficients, the 4 seasonal-lag B
    coefficients, the 3 short-lag A coefficients, and the 3 short-lag B
    coefficients.
    """
    hess_path = model_dir / "hess_inv.npy"
    params_path = model_dir / "estimated_parameters.csv"
    out = {"cond_number": np.nan, "corr_A_short": None, "corr_B_short": None,
           "corr_A_seasonal": None, "corr_B_seasonal": None}
    if not hess_path.exists() or not params_path.exists():
        return out
    hess_inv = np.load(hess_path)
    names = pd.read_csv(params_path)["parameter"].tolist()
    name_to_idx = {n: i for i, n in enumerate(names)}

    def idx_for(prefix_letter: str, lags: list) -> list:
        return [name_to_idx[f"{prefix_letter}_{branch_tv}_{l}"]
                for l in lags if f"{prefix_letter}_{branch_tv}_{l}" in name_to_idx]

    all_lag_idx = idx_for("A", [1, 2, 3, 364, 365, 366, 367]) + idx_for("B", [1, 2, 3, 364, 365, 366, 367])
    if len(all_lag_idx) >= 2:
        try:
            block = hess_inv[np.ix_(all_lag_idx, all_lag_idx)]
            out["cond_number"] = float(np.linalg.cond(block))
        except np.linalg.LinAlgError:
            pass

    out["corr_A_short"]    = _pairwise_abs_corr(hess_inv, idx_for("A", [1, 2, 3]))
    out["corr_B_short"]    = _pairwise_abs_corr(hess_inv, idx_for("B", [1, 2, 3]))
    out["corr_A_seasonal"] = _pairwise_abs_corr(hess_inv, idx_for("A", [364, 365, 366, 367]))
    out["corr_B_seasonal"] = _pairwise_abs_corr(hess_inv, idx_for("B", [364, 365, 366, 367]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# One (location, stage, branch, lagset) unit of work
# ─────────────────────────────────────────────────────────────────────────────

def analyze_one(display: str, short: str, stage: str, branch: str, lagset: str,
                 model_id: str, model_dir: Path, data: dict, out_dir: Path) -> dict:
    y_train, y_test = data["y_train"], data["y_test"]
    model, theta, meta, _ = load_model_and_theta(model_dir)
    if model is None or theta is None:
        raise RuntimeError(f"Could not reconstruct model at {model_dir}")

    needs_cov = stage == "stage2"
    X_train = X_test = None
    if needs_cov:
        block = DEWPOINT_BLOCK[lagset]
        X_train = data["covariate_blocks_train"][block]
        X_test  = data["covariate_blocks_test"][block]

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── In-sample quantile residuals ────────────────────────────────────
    cdfs = model.cdf_series(theta, y_train, X_train) if needs_cov else model.cdf_series(theta, y_train)
    eff = len(y_train) - len(cdfs)
    rng = np.random.default_rng(0)
    qr = quantile_residuals(cdfs, y_train[eff:], randomise_zeros=True, rng=rng)
    np.save(out_dir / "residuals.npy", qr)

    # ── Multi-step-ahead OOS (Monte Carlo) ──────────────────────────────
    max_origin = len(y_test) - max(DEFAULT_HORIZONS) - 1
    origins = list(range(0, max_origin, max(1, max_origin // (len(y_test) // N_ORIGINS_STRIDE + 1))))
    covariates = {"X_train": X_train, "X_test": X_test} if needs_cov else {}
    ms = multistep_metrics(model, theta, y_train, y_test, covariates,
                            origins=origins, horizons=DEFAULT_HORIZONS, n_draws=N_DRAWS, seed=MC_SEED)
    ms.to_csv(out_dir / "multistep.csv", index=False)

    # ── Collinearity / optimisation diagnostics ─────────────────────────
    coll = collinearity_diagnostics(model_dir, "phi")  # location parameter is always "phi" here

    info = {
        "location": display, "stage": stage, "branch": branch, "lagset": lagset,
        "model_id": model_id, "rmse": meta.get("rmse"), "crps_mean": meta.get("crps_mean"),
        "grad_norm_inf": meta.get("grad_norm_inf"), "grad_norm_2": meta.get("grad_norm_2"),
        "n_iter": meta.get("n_iter"), "n_fev": meta.get("n_fev"), "runtime_s": meta.get("runtime_s"),
        "validity": meta.get("validity"), "success": meta.get("success"),
        "n_residuals": len(qr), "n_origins": len(origins), "n_draws": N_DRAWS,
        **coll,
    }
    (out_dir / "info.json").write_text(json.dumps(info, indent=2, default=float))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", default=None)
    args = parser.parse_args()
    locations = [args.location] if args.location else LOCATIONS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for location in locations:
        cfg = STATION_REGISTRY[location]
        run_id, display = cfg["run_id"], cfg.get("display", location)
        short = cfg.get("short", run_id)
        run_dir = ARTIFACTS_DIR / run_id
        print(f"=== {display} ===")

        winners = json.loads((run_dir / "stage_winners.json").read_text())
        loader = StationDataLoader.from_registry(location, precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH)
        data = loader.load_all()

        for branch in BRANCHES:
            for stage in ("stage1", "stage2"):
                for lagset in LAGSETS:
                    try:
                        if stage == "stage1":
                            model_id = resolve_stage1_id(winners, branch, lagset)
                        else:
                            model_id = stage2_id(branch, lagset)
                        model_dir = run_dir / stage_subdir(stage, branch) / model_id
                        out_dir = OUT_DIR / short / f"{stage}_{branch}_{lagset}"
                        t0 = time.time()
                        info = analyze_one(display, short, stage, branch, lagset, model_id, model_dir, data, out_dir)
                        print(f"  [{stage}/{branch}/{lagset}] {model_id}  "
                              f"rmse={info['rmse']:.3f} crps={info['crps_mean']:.3f}  ({time.time()-t0:.1f}s)")
                        summary_rows.append({"status": "ok", **info})
                    except Exception as exc:
                        print(f"  [{stage}/{branch}/{lagset}] FAILED: {exc}")
                        traceback.print_exc()
                        summary_rows.append({
                            "status": "failed", "location": display, "stage": stage,
                            "branch": branch, "lagset": lagset, "error": str(exc),
                        })

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary.csv", index=False)
    print("\nDone. Summary written to", OUT_DIR / "summary.csv")


if __name__ == "__main__":
    main()
