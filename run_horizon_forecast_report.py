"""
k-day-ahead (blind) forecast evaluation for the "stage1_phi_short_diagfi"
arrangement -- the phi-only ZA-GAS model (short GAS lags {1,2,3}, diagonal-
inverse-Fisher scaling, no covariates) that wins the phi-only branch's OOS
RMSE selection (run_all_locations.py, IMPROVEMENT_THRESHOLD=0.02 rule) at
4 of the 6 reported locations (Cruzeiro, Garanhuns, Manaus, Salvador); at
the other two (Belo Horizonte, Darwin) the branch winner is
stage2_phi_dewpoint_short, which beat this arrangement by >2% OOS RMSE
there. This script evaluates the SAME arrangement (stage1_phi_short_diagfi)
uniformly across all 6 locations for direct comparability, using cached
fitted parameters only -- no re-estimation.

Uses diagnostics/horizon_forecast.py (Monte Carlo forward simulation) to
produce RMSE at horizons 1, 2, 5, 10, 30, 60, 90 days ahead, split by
whether the realized y_{t+h} is zero (dry) or positive (wet).

Usage:
    python run_horizon_forecast_report.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import STATION_REGISTRY, StationDataLoader
from pipeline.artifact_utils import load_model_and_theta
from diagnostics.horizon_forecast import simulate_horizon_forecasts

PRECIP_DIR = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR   = ROOT / "data" / "input" / "ERA5"
ENSO_PATH  = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
OUT_DIR = ROOT / "reports" / "horizon_forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOCATIONS = [
    ("BELO HORIZONTE",         "run_20260701_bh"),
    ("CRUZEIRO DO SUL (ACRE)", "run_20260705_cruzeiro"),
    ("DARWIN AIRPORT",         "run_20260705_darwin"),
    ("GARANHUNS (PERNAMBUCO)", "run_20260705_garanhuns"),
    ("MANAUS",                 "run_20260705_manaus"),
    ("SALVADOR",               "run_20260705_salvador"),
]

HORIZONS = (1, 2, 5, 10, 30, 60, 90)
N_DRAWS = 2000
SEED = 42


def main() -> None:
    all_rows = []
    for station_name, run_id in LOCATIONS:
        t0 = time.time()
        model_dir = ARTIFACTS_DIR / run_id / "stage1" / "stage1_phi_short_diagfi"
        if not model_dir.exists():
            print(f"SKIP {station_name}: {model_dir} not found")
            continue

        model, theta, meta, params_df = load_model_and_theta(model_dir)
        if model is None or theta is None:
            print(f"SKIP {station_name}: could not reconstruct model/theta")
            continue

        loader = StationDataLoader.from_registry(
            station_name=station_name,
            precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH,
        )
        data = loader.load_all()
        y_train, y_test = data["y_train"], data["y_test"]

        df = simulate_horizon_forecasts(
            model, theta, y_train, y_test,
            horizons=HORIZONS, n_draws=N_DRAWS, seed=SEED,
        )
        df.insert(0, "location", station_name)
        df.insert(1, "run_id", run_id)
        df.insert(2, "model_id", "stage1_phi_short_diagfi")
        df.insert(3, "reported_oos_rmse_h1", meta.get("rmse"))

        out_path = OUT_DIR / f"{run_id}_stage1_phi_short_diagfi_horizon.csv"
        df.to_csv(out_path, index=False)
        print(f"{station_name:<28} done in {time.time()-t0:5.1f}s -> {out_path.name}")
        print(df.to_string(index=False))
        print()

        all_rows.append(df)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(OUT_DIR / "all_locations_stage1_phi_short_diagfi_horizon.csv", index=False)

        # Pooled across the 4 locations where this arrangement is the actual
        # final phi-only winner (Cruzeiro, Garanhuns, Manaus, Salvador).
        winner_locs = {"CRUZEIRO DO SUL (ACRE)", "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR"}
        pooled_rows = []
        for h in HORIZONS:
            sub = combined[(combined["horizon"] == h) & (combined["location"].isin(winner_locs))]
            n_wet = sub["n_wet"].sum()
            n_dry = sub["n_dry"].sum()
            # Recombine RMSE across locations via sum-of-squared-errors weighting
            sse_wet = (sub["rmse_wet"] ** 2 * sub["n_wet"]).sum()
            sse_dry = (sub["rmse_dry"] ** 2 * sub["n_dry"]).sum()
            crps_wet_pooled = (sub["crps_wet"] * sub["n_wet"]).sum() / n_wet if n_wet > 0 else float("nan")
            crps_dry_pooled = (sub["crps_dry"] * sub["n_dry"]).sum() / n_dry if n_dry > 0 else float("nan")
            pooled_rows.append({
                "horizon": h, "n_locations": len(sub), "n_wet": int(n_wet), "n_dry": int(n_dry),
                "rmse_wet_pooled": (sse_wet / n_wet) ** 0.5 if n_wet > 0 else float("nan"),
                "rmse_dry_pooled": (sse_dry / n_dry) ** 0.5 if n_dry > 0 else float("nan"),
                "crps_wet_pooled": crps_wet_pooled,
                "crps_dry_pooled": crps_dry_pooled,
            })
        pooled = pd.DataFrame(pooled_rows)
        pooled.to_csv(OUT_DIR / "pooled_winner_locations_horizon.csv", index=False)
        print("Pooled across the 4 locations where stage1_phi_short_diagfi is the actual final winner:")
        print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
