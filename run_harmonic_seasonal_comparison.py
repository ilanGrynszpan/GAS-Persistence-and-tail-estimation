"""
Standard Harvey long-short vs. m=5 trigonometric-seasonal Harvey long-short
comparison analysis (2026-09-04).

=============================================================================
OVERVIEW
=============================================================================

For each location x branch (phi_only, phi_xi), compares:
  * "standard": the existing best long-short config (artifacts/
    longshort_diagnostics_experiment/tvp_comparison/.../best_model_info.json)
  * "seasonal": that SAME config plus the m=5 trigonometric seasonal
    component (artifacts/harmonic_seasonal_experiment/<short>/<branch>/
    m5_final/), warm-started from it

computing in-sample quantile residuals, full-test-set 1-step rolling OOS
RMSE/CRPS, and Monte Carlo multi-step (h=1,2,5,10,30,60,90) OOS RMSE/CRPS
-- the exact same three computations run_longshort_best_models_analysis.py's
analyze_config() already performs for Harvey-vs-covariate-GAS, adapted here
for Harvey-vs-Harvey (both models share identical long_names/short_names by
construction, since "seasonal" is warm-started from "standard").

Per docs/EXECUTION.md, no optimisation happens here -- both models were
already fit; this script only evaluates.

=============================================================================
USAGE
=============================================================================

    python run_harmonic_seasonal_comparison.py
    python run_harmonic_seasonal_comparison.py --location "MANAUS" --branch phi_only

Output (per location x branch):
    artifacts/harmonic_seasonal_comparison/<short>/<branch>/info.json
    artifacts/harmonic_seasonal_comparison/<short>/<branch>/<standard|seasonal>/
        residuals.npy, onestep_metrics.json, multistep_metrics.csv
"""

from __future__ import annotations

import argparse
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
import run_harmonic_seasonal_experiment as R

ARTIFACTS_DIR = ROOT / "artifacts"
HARM_DIR      = R.OUT_DIR  # artifacts/harmonic_seasonal_experiment
OUT_DIR       = ARTIFACTS_DIR / "harmonic_seasonal_comparison"

LOCATIONS = R.LOCATIONS
BRANCHES  = R.BRANCHES
N_ORIGINS_STRIDE = 15
N_DRAWS = 400
MC_SEED = 2026


def analyze_pair(display: str, short: str, branch: str, data: dict, loc_out: Path,
                  seasonal_subdir: str = "m5_final") -> dict:
    loc_out.mkdir(parents=True, exist_ok=True)
    y_train, y_test = data["y_train"], data["y_test"]

    baseline = R.resolve_baseline(short, branch)
    standard_model, standard_theta, standard_meta, _ = load_model_and_theta(baseline["model_dir"])
    seasonal_dir = HARM_DIR / short / branch / seasonal_subdir
    seasonal_model, seasonal_theta, seasonal_meta, _ = load_model_and_theta(seasonal_dir)

    long_names, short_names = standard_meta["long_names"], standard_meta["short_names"]
    Xl_train = R._build_X(data, long_names, "train")
    Xl_test  = R._build_X(data, long_names, "test")
    Xs_train = R._build_X(data, short_names, "train")
    Xs_test  = R._build_X(data, short_names, "test")

    info = {"location": display, "short": short, "branch": branch,
            "standard_config": baseline["harvey_config"], "standard_run_id": baseline["run_id"],
            "standard_grad_norm_inf": standard_meta["grad_norm_inf"],
            "standard_aic": 2 * standard_meta["n_params"] - 2 * standard_meta["loglik"],
            "seasonal_grad_norm_inf": seasonal_meta["grad_norm_inf"],
            "seasonal_aic": seasonal_meta["aic"]}

    results = {}
    for tag, model, theta in [("standard", standard_model, standard_theta),
                              ("seasonal", seasonal_model, seasonal_theta)]:
        out_dir = loc_out / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        cdfs = model.cdf_series(theta, y_train, Xl_train, Xs_train)
        eff = len(y_train) - len(cdfs)
        rng = np.random.default_rng(0)
        qr = quantile_residuals(cdfs, y_train[eff:], randomise_zeros=True, rng=rng)
        np.save(out_dir / "residuals.npy", qr)

        oos = model.simulate_oos(theta, y_train, y_test, Xl_train, Xl_test, Xs_train, Xs_test)
        onestep = compute_extended_oos_metrics(model, oos, y_test, n_draws=1000, seed=42)
        (out_dir / "onestep_metrics.json").write_text(json.dumps(onestep, indent=2, default=float))

        max_origin = len(y_test) - max(DEFAULT_HORIZONS) - 1
        origins = list(range(0, max_origin, max(1, max_origin // (len(y_test) // N_ORIGINS_STRIDE + 1))))
        ms = multistep_metrics(
            model, theta, y_train, y_test,
            {"X_long_train": Xl_train, "X_long_test": Xl_test,
             "X_short_train": Xs_train, "X_short_test": Xs_test},
            origins=origins, horizons=DEFAULT_HORIZONS, n_draws=N_DRAWS, seed=MC_SEED,
        )
        ms.to_csv(out_dir / "multistep_metrics.csv", index=False)

        results[tag] = {"rmse_1step": onestep["rmse"], "crps_1step": onestep["crps_mean"],
                        "n_origins": len(origins), "n_draws": N_DRAWS}
        print(f"    {tag}: rmse={onestep['rmse']:.3f} crps={onestep['crps_mean']:.3f}")

    info.update(results)
    (loc_out / "info.json").write_text(json.dumps(info, indent=2, default=float))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", default=None)
    parser.add_argument("--branch", default=None, choices=BRANCHES)
    parser.add_argument("--seasonal-subdir", default="m5_final",
                         help="Which artifacts/harmonic_seasonal_experiment/<short>/<branch>/<subdir> "
                              "to compare against 'standard' -- e.g. 'm5_final_damp0.999' for the "
                              "damped-rotation variant. Output lands under a matching "
                              "artifacts/harmonic_seasonal_comparison_<subdir>/ tree (kept separate "
                              "from the default m5_final comparison).")
    args = parser.parse_args()
    locations = [args.location] if args.location else LOCATIONS
    branches = [args.branch] if args.branch else BRANCHES
    seasonal_subdir = args.seasonal_subdir
    out_dir = OUT_DIR if seasonal_subdir == "m5_final" else ARTIFACTS_DIR / f"harmonic_seasonal_comparison_{seasonal_subdir}"

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for location in locations:
        cfg = STATION_REGISTRY[location]
        short, display = cfg.get("short", cfg["run_id"]), cfg.get("display", location)
        print(f"=== {display} ({short}) ===")
        loader = StationDataLoader.from_registry(location, precip_dir=R.PRECIP_DIR, era5_dir=R.ERA5_DIR, enso_path=R.ENSO_PATH)
        data = loader.load_all()

        for branch in branches:
            print(f"  branch: {branch}")
            seasonal_dir = HARM_DIR / short / branch / seasonal_subdir
            if not (seasonal_dir / "metadata.json").exists():
                print(f"    MISSING {seasonal_subdir} fit at {seasonal_dir}; skipping.")
                continue
            info_path = out_dir / short / branch / "info.json"
            if info_path.exists():
                info = json.loads(info_path.read_text())
                summary_rows.append({"status": "ok", **info})
                print(f"    SKIP (already analyzed): "
                      f"standard rmse={info['standard']['rmse_1step']:.3f} vs "
                      f"seasonal rmse={info['seasonal']['rmse_1step']:.3f}")
                continue
            try:
                t0 = time.time()
                info = analyze_pair(display, short, branch, data, out_dir / short / branch,
                                     seasonal_subdir=seasonal_subdir)
                summary_rows.append({"status": "ok", **info})
                print(f"    done in {time.time()-t0:.1f}s")
            except Exception as exc:
                print(f"    FAILED: {exc}")
                traceback.print_exc()
                summary_rows.append({"status": "failed", "location": display, "branch": branch, "error": str(exc)})

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
