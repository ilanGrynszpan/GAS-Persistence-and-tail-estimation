"""
Out-of-sample CRPS for the AR baseline, assuming Normal(mu, sigma^2) errors
(per the ridge regression's implicit iid-Gaussian-residual model), at the
same rolling origins/horizons used everywhere else in this analysis.

mu   = the AR model's recursive point forecast at each origin/horizon
       (baselines/naive_ar.py::make_predict_step -- same forecasts used for
       the AR RMSE numbers already in reports/ml_baselines_horizon/).
sigma = sqrt(mean squared in-sample training residual) of the ridge fit,
       a single constant reused at every origin and horizon (see
       baselines/naive_ar.py::fit_with_sigma for the justification).
CRPS  = closed-form Normal CRPS (baselines/horizon_eval.py::crps_normal).

Does not touch or re-run XGBoost/LSTM/AR+Aux (unaffected by this request).

Usage:
    python run_ar_crps_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import StationDataLoader
from baselines.horizon_eval import recursive_horizon_forecasts, crps_normal
from baselines import naive_ar

PRECIP_DIR = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR   = ROOT / "data" / "input" / "ERA5"
ENSO_PATH  = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
OUT_DIR = ROOT / "reports" / "ml_baselines_horizon"
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


def main() -> None:
    all_rows = []
    for station_name, run_id in LOCATIONS:
        loader = StationDataLoader.from_registry(
            station_name=station_name,
            precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH,
        )
        data = loader.load_all()
        y_train, y_test = data["y_train"], data["y_test"]
        y_full = np.concatenate([y_train, y_test])
        T_train = len(y_train)

        beta, sigma = naive_ar.fit_with_sigma(y_train)
        step = naive_ar.make_predict_step(beta, y_full)
        t0, forecasts = recursive_horizon_forecasts(
            y_full, T_train, HORIZONS, max(naive_ar.SHORT_LAGS), step,
        )

        rows = []
        for h in HORIZONS:
            actual = y_full[t0 + h]
            mu = forecasts[h]
            crps_h = crps_normal(mu, sigma, actual)
            rows.append({"horizon": h, "n_origins": len(t0), "sigma": sigma,
                         "crps_all": float(np.mean(crps_h))})
        df = pd.DataFrame(rows)
        df.insert(0, "location", station_name)
        df.insert(1, "run_id", run_id)
        df.insert(2, "model", "AR")

        out_path = OUT_DIR / f"{run_id}_ar_crps_gaussian.csv"
        df.to_csv(out_path, index=False)
        print(f"{station_name:<28} sigma={sigma:.4f} -> {out_path.name}")
        print(df.to_string(index=False))
        print()

        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUT_DIR / "all_locations_ar_crps_gaussian.csv", index=False)


if __name__ == "__main__":
    main()
