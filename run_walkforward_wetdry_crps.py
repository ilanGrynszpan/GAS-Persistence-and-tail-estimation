"""
Wet/dry CRPS + RMSE breakdown for the EXISTING walk-forward 1-step-ahead OOS
technique (simulation/simulator.py::simulate_oos) -- the same technique used
for every RMSE/CRPS number already in stage_winners.json, extended_metrics.csv
and the multi_location / summary reports.

Why this script exists: extended_metrics.csv already reports rmse_wet/
rmse_dry (diagnostics/scoring.py::compute_extended_oos_metrics), but not a
wet/dry split of CRPS -- only the overall crps_mean. This script adds that
split as a NEW, separate table (it does not touch or regenerate
extended_metrics.csv or any existing report). It reuses the exact same
Monte Carlo predictive draws and crps_mc energy-score estimator already used
throughout diagnostics/scoring.py -- nothing about the existing 1-step
technique or its numbers changes; this only exposes the wet/dry split that
was already implicit in the per-observation crps_vals array.

Evaluated over the FULL out-of-sample test window (all ~730 days per
location) -- unlike the k-day-ahead horizon engine (run_horizon_forecast_
report.py / diagnostics/horizon_forecast.py), this technique's forecast at
every test day always uses the true y up to the day before, so there is no
"lose the last max(horizon) days" restriction; "horizon" here is always 1
step (the day immediately following true, known history).

Usage:
    python run_walkforward_wetdry_crps.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import StationDataLoader
from pipeline.artifact_utils import load_model_and_theta
from simulation.simulator import simulate_oos
from diagnostics.scoring import crps_mc, _draw_predictive

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

N_DRAWS = 2000
SEED = 42


def main() -> None:
    rows = []
    for station_name, run_id in LOCATIONS:
        t0 = time.time()
        model_dir = ARTIFACTS_DIR / run_id / "stage1" / "stage1_phi_short_diagfi"
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

        oos = simulate_oos(model, theta, y_train, y_test)
        pi_oos = oos["pi_oos"]
        static = oos["static"]
        xi_s, gamma_s, zeta_s = static["xi"], static["gamma"], static["zeta"]
        a, b, p = np.exp(xi_s), np.exp(zeta_s), np.exp(-gamma_s)
        from scipy.special import beta as beta_fn
        mean_gb2_const = float(beta_fn(a + 1 / p, b - 1 / p) / beta_fn(a, b))
        pred_mean = pi_oos * mean_gb2_const * np.exp(oos["f_arr_oos"][:, 0])

        rng = np.random.default_rng(SEED)
        n = len(y_test)
        crps_vals = np.empty(n)
        for i in range(n):
            draws = _draw_predictive(model, oos, i, N_DRAWS, rng)
            crps_vals[i] = crps_mc(draws, y_test[i])

        wet = y_test > 0
        dry = ~wet
        err = pred_mean - y_test
        row = {
            "location": station_name, "run_id": run_id,
            "model_id": "stage1_phi_short_diagfi", "method": "walkforward_1step",
            "n": n, "n_wet": int(wet.sum()), "n_dry": int(dry.sum()),
            "rmse_all": float(np.sqrt(np.mean(err ** 2))),
            "rmse_wet": float(np.sqrt(np.mean(err[wet] ** 2))),
            "rmse_dry": float(np.sqrt(np.mean(err[dry] ** 2))),
            "crps_all": float(np.mean(crps_vals)),
            "crps_wet": float(np.mean(crps_vals[wet])),
            "crps_dry": float(np.mean(crps_vals[dry])),
            "reported_metadata_rmse": meta.get("rmse"),
            "reported_metadata_crps_mean": meta.get("crps_mean"),
        }
        rows.append(row)

        out_path = OUT_DIR / f"{run_id}_stage1_phi_short_diagfi_walkforward_wetdry.csv"
        pd.DataFrame([row]).to_csv(out_path, index=False)
        print(f"{station_name:<28} done in {time.time()-t0:5.1f}s -> {out_path.name}")
        print(pd.DataFrame([row]).to_string(index=False))
        print()

    combined = pd.DataFrame(rows)
    combined.to_csv(OUT_DIR / "all_locations_walkforward_wetdry.csv", index=False)
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
