"""
Best-model-per-location analysis for the Harvey long-short diagnostics
experiment (2026-08-31).

=============================================================================
OVERVIEW
=============================================================================

For each of the six locations, identifies the single best-performing fit
from the 144-fit grid in artifacts/longshort_diagnostics_experiment/ (lowest
|grad|_inf among non-"failed" fits, across all branches/modes/Gamma_L
inits), and its paired "standard GAS" comparison model (the Stage-2
weather-covariate model used to seed its initialisation, per
init_recipe.json's prior_model_id). For that pair, computes:

  1. In-sample quantile residuals (diagnostics/residuals.py), saved for the
     ACF-400 / QQ mosaics built separately by
     generate_longshort_best_models_report.py using diagnostics/plots.py's
     existing acf_location_mosaic / qq_location_mosaic.
  2. Full-test-set 1-step rolling OOS RMSE/CRPS
     (diagnostics/scoring.py:compute_extended_oos_metrics -- exact, no
     simulation needed since it is a genuine 1-step predictive mean/draws
     evaluation already implemented and used throughout this framework).
  3. Multi-step-ahead (h=1,2,5,10,30,60,90) OOS RMSE/CRPS via the new
     diagnostics/multistep_forecast.py Monte Carlo engine, over a strided
     subsample of forecast origins (full test set would be prohibitively
     slow for h up to 90; the stride/draw count is saved in the output for
     transparency and reported in the PDF).

Per docs/EXECUTION.md, no optimisation happens here: every model is
reconstructed from already-saved theta (pipeline.artifact_utils) and only
evaluated, never re-fit.

=============================================================================
USAGE
=============================================================================

    python run_longshort_best_models_analysis.py

Output (per location, under artifacts/longshort_diagnostics_experiment/best_models/<short>/):
    best_model_info.json       -- which config won, and its comparison model
    harvey_quantile_residuals.npy / cov_quantile_residuals.npy
    onestep_metrics.json       -- {harvey: {...}, cov: {...}}
    multistep_metrics_harvey.csv, multistep_metrics_cov.csv
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import STATION_REGISTRY, StationDataLoader
from pipeline.artifact_utils import load_model_and_theta
from diagnostics.residuals import quantile_residuals
from diagnostics.scoring import compute_extended_oos_metrics
from diagnostics.multistep_forecast import multistep_metrics, DEFAULT_HORIZONS

PRECIP_DIR = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR   = ROOT / "data" / "input" / "ERA5"
ENSO_PATH  = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
EXP_DIR    = ARTIFACTS_DIR / "longshort_diagnostics_experiment"
OUT_DIR    = EXP_DIR / "best_models"

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]

SHORT_COVARIATE    = "dewpoint_lag1"
SHORT_SOURCE_BLOCK = "dewpoint_short"
LONG_BLOCK         = "enso_dummy_current"

# Multi-step Monte Carlo settings -- see module docstring above for the
# tractability trade-off; reported verbatim in the PDF.
N_ORIGINS_STRIDE = 15   # ~stride so that len(y_test)//stride origins are used
N_DRAWS          = 400
MC_SEED          = 2026


def find_best_config(run_id: str, branch: "Optional[str]" = None) -> dict:
    """Lowest |grad|_inf among non-'failed' fits across all (branch, mode,
    Gamma_L) combinations for this location.

    branch=None (default, used by this script's own main()): search both
    "phi_only" and "phi_xi" and return the single overall best -- exact
    previous behaviour.

    branch="phi_only" or "phi_xi": restrict the search to that one branch --
    used by run_longshort_tvp_comparison.py, which (unlike this script)
    wants each branch's own best configuration at every location, not just
    whichever branch happens to win overall.
    """
    best = None
    for branch_dir in (EXP_DIR / run_id).iterdir():
        if not branch_dir.is_dir() or branch_dir.name not in ("phi_only", "phi_xi"):
            continue
        if branch is not None and branch_dir.name != branch:
            continue
        for cfg_dir in branch_dir.iterdir():
            meta_path = cfg_dir / "metadata.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            if meta.get("validity") == "failed":
                continue
            grad = meta.get("grad_norm_inf", np.inf)
            if not np.isfinite(grad):
                continue
            if best is None or grad < best["grad_norm_inf"]:
                best = {"branch": branch_dir.name, "config": cfg_dir.name,
                        "dir": cfg_dir, "grad_norm_inf": grad, "meta": meta}
    return best


def analyze_config(display: str, run_id: str, data: dict, best: dict, loc_out: Path) -> dict:
    """
    Given an already-resolved best-config dict (from find_best_config) and
    already-loaded station data (StationDataLoader.load_all()), compute
    in-sample quantile residuals, full-test-set 1-step OOS, and Monte Carlo
    multi-step OOS for that Harvey model and its paired standard-GAS
    comparison model, saving everything under loc_out. Returns the info
    dict (also written as loc_out/best_model_info.json).

    Factored out of main() so both this script's own overall-best-per-
    location flow and run_longshort_tvp_comparison.py's both-branches-per-
    location flow share one implementation (CLAUDE.md: refactor rather than
    duplicate) instead of two near-identical copies of this ~90-line body.
    """
    loc_out.mkdir(parents=True, exist_ok=True)
    y_train, y_test = data["y_train"], data["y_test"]

    harvey_model, harvey_theta, harvey_meta, _ = load_model_and_theta(best["dir"])
    recipe = json.loads((best["dir"] / "init_recipe.json").read_text())
    prior_id = recipe["prior_model_id"]
    branch = best["branch"]
    cov_subdir = "stage2" if branch == "phi_xi" else "stage2_phi"
    cov_dir = ARTIFACTS_DIR / run_id / cov_subdir / prior_id
    cov_model, cov_theta, cov_meta, _ = load_model_and_theta(cov_dir)
    print(f"  Comparison standard-GAS model: {cov_subdir}/{prior_id}")

    block = prior_id.replace("stage2_phi_", "").replace("stage2_", "")
    X_cov_train = data["covariate_blocks_train"][block]
    X_cov_test  = data["covariate_blocks_test"][block]

    short_cols = data["covariate_col_names"][SHORT_SOURCE_BLOCK]
    short_idx  = short_cols.index(SHORT_COVARIATE)
    Xs_train = data["covariate_blocks_train"][SHORT_SOURCE_BLOCK][:, [short_idx]]
    Xs_test  = data["covariate_blocks_test"][SHORT_SOURCE_BLOCK][:, [short_idx]]
    Xl_train = data["covariate_blocks_train"][LONG_BLOCK]
    Xl_test  = data["covariate_blocks_test"][LONG_BLOCK]

    info = {
        "location": display, "run_id": run_id, "branch": branch,
        "harvey_config": best["config"], "harvey_grad_norm_inf": best["grad_norm_inf"],
        "harvey_validity": best["meta"]["validity"],
        "comparison_model_id": prior_id, "comparison_subdir": cov_subdir,
    }

    # ── 1. In-sample quantile residuals ─────────────────────────────
    harvey_cdfs = harvey_model.cdf_series(harvey_theta, y_train, Xl_train, Xs_train)
    eff_h = len(y_train) - len(harvey_cdfs)
    rng = np.random.default_rng(0)
    harvey_qr = quantile_residuals(harvey_cdfs, y_train[eff_h:], randomise_zeros=True, rng=rng)
    np.save(loc_out / "harvey_quantile_residuals.npy", harvey_qr)

    cov_cdfs = cov_model.cdf_series(cov_theta, y_train, X_cov_train)
    eff_c = len(y_train) - len(cov_cdfs)
    rng2 = np.random.default_rng(0)
    cov_qr = quantile_residuals(cov_cdfs, y_train[eff_c:], randomise_zeros=True, rng=rng2)
    np.save(loc_out / "cov_quantile_residuals.npy", cov_qr)
    print(f"  In-sample quantile residuals: Harvey n={len(harvey_qr)}, Cov n={len(cov_qr)}")

    # ── 2. Full-test-set 1-step rolling OOS ─────────────────────────
    harvey_oos = harvey_model.simulate_oos(harvey_theta, y_train, y_test, Xl_train, Xl_test, Xs_train, Xs_test)
    harvey_1step = compute_extended_oos_metrics(harvey_model, harvey_oos, y_test, n_draws=1000, seed=42)

    cov_oos = cov_model.simulate_oos(cov_theta, y_train, y_test, X_cov_train, X_cov_test)
    cov_1step = compute_extended_oos_metrics(cov_model, cov_oos, y_test, n_draws=1000, seed=42)
    (loc_out / "onestep_metrics.json").write_text(json.dumps(
        {"harvey": harvey_1step, "cov": cov_1step}, indent=2, default=float))
    print(f"  1-step OOS: Harvey RMSE={harvey_1step['rmse']:.3f} CRPS={harvey_1step['crps_mean']:.3f} | "
          f"Cov RMSE={cov_1step['rmse']:.3f} CRPS={cov_1step['crps_mean']:.3f}")

    # ── 3. Multi-step-ahead OOS (Monte Carlo) ───────────────────────
    max_origin = len(y_test) - max(DEFAULT_HORIZONS) - 1
    origins = list(range(0, max_origin, max(1, max_origin // (len(y_test) // N_ORIGINS_STRIDE + 1))))
    print(f"  Multi-step: {len(origins)} origins, {N_DRAWS} draws each, horizons={DEFAULT_HORIZONS}")

    t0 = time.time()
    harvey_ms = multistep_metrics(
        harvey_model, harvey_theta, y_train, y_test,
        {"X_long_train": Xl_train, "X_long_test": Xl_test,
         "X_short_train": Xs_train, "X_short_test": Xs_test},
        origins=origins, horizons=DEFAULT_HORIZONS, n_draws=N_DRAWS, seed=MC_SEED,
    )
    harvey_ms.to_csv(loc_out / "multistep_metrics_harvey.csv", index=False)
    print(f"    Harvey multistep done in {time.time()-t0:.1f}s")

    t0 = time.time()
    cov_ms = multistep_metrics(
        cov_model, cov_theta, y_train, y_test,
        {"X_train": X_cov_train, "X_test": X_cov_test},
        origins=origins, horizons=DEFAULT_HORIZONS, n_draws=N_DRAWS, seed=MC_SEED,
    )
    cov_ms.to_csv(loc_out / "multistep_metrics_cov.csv", index=False)
    print(f"    Cov multistep done in {time.time()-t0:.1f}s")

    info["multistep_n_origins"] = len(origins)
    info["multistep_n_draws"] = N_DRAWS
    (loc_out / "best_model_info.json").write_text(json.dumps(info, indent=2, default=float))
    return info


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location", default=None, help="Run only this location (exact STATION_REGISTRY name)")
    args = parser.parse_args()
    locations = [args.location] if args.location else LOCATIONS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for location in locations:
        cfg = STATION_REGISTRY[location]
        run_id, display = cfg["run_id"], cfg.get("display", location)
        short = cfg.get("short", run_id)
        print(f"=== {display} ===")
        loc_out = OUT_DIR / short

        try:
            best = find_best_config(run_id)
            if best is None:
                print(f"  No non-failed config found for {display}; skipping.")
                continue
            print(f"  Best Harvey config: {best['branch']}/{best['config']} "
                  f"(|grad|_inf={best['grad_norm_inf']:.4g})")

            loader = StationDataLoader.from_registry(location, precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH)
            data = loader.load_all()

            info = analyze_config(display, run_id, data, best, loc_out)
            summary_rows.append({"location": display, "status": "ok", **info})

        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            summary_rows.append({"location": display, "status": "failed", "error": str(exc)})

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary.csv", index=False)
    print("\nDone. Summary written to", OUT_DIR / "summary.csv")


if __name__ == "__main__":
    main()
