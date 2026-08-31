"""
Runs the four ML baselines (naive/AR, AR+Aux, XGBoost, LSTM) -- ported from
daily-ML.ipynb (see baselines/ package) -- at the six locations used
throughout the rest of this analysis, evaluated with the same rolling-
origin blind k-day-ahead RMSE protocol (wet/dry split) already used for the
score-driven model (diagnostics/horizon_forecast.py / run_horizon_forecast_
report.py). No cross-validation is re-run: every model reuses the
hyperparameters already found in the source notebook (hardcoded in each
baselines/*.py module, quoted verbatim from the notebook's own CV output).

Data is loaded via data/station_loader.py (the same loader the score-driven
model uses) rather than daily-ML.ipynb's own raw CSV read, so both engines
evaluate on an identical out-of-sample window -- required for the two to be
comparable in the combined table.

Usage:
    python run_ml_baselines_report.py
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
from baselines.horizon_eval import recursive_horizon_rmse
from baselines import naive_ar, ar_aux, xgb_baseline, lstm_baseline

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
        t_loc = time.time()
        loader = StationDataLoader.from_registry(
            station_name=station_name,
            precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH,
        )
        data = loader.load_all()
        y_train, y_test = data["y_train"], data["y_test"]
        dates_train, dates_test = data["dates_train"], data["dates_test"]
        y_full = np.concatenate([y_train, y_test])
        dates_full = dates_train.append(dates_test)
        T_train = len(y_train)

        print(f"{station_name}: n_train={T_train} n_test={len(y_test)}")

        # ---- naive/AR ----
        t0 = time.time()
        beta_ar = naive_ar.fit(y_train)
        step_ar = naive_ar.make_predict_step(beta_ar, y_full)
        df_ar = recursive_horizon_rmse(y_full, T_train, HORIZONS, max(naive_ar.SHORT_LAGS), step_ar)
        df_ar.insert(0, "model", "AR")
        print(f"  AR done in {time.time()-t0:.1f}s")

        # ---- AR+Aux ----
        t0 = time.time()
        beta_aux = ar_aux.fit(y_train, dates_train)
        from baselines.horizon_eval import rolling_origins
        t0_arr = rolling_origins(T_train, len(y_full), max(HORIZONS))
        step_aux = ar_aux.make_predict_step(beta_aux, y_full, dates_full, t0_arr)
        df_aux = recursive_horizon_rmse(y_full, T_train, HORIZONS, max(ar_aux.SHORT_LAGS), step_aux)
        df_aux.insert(0, "model", "AR+Aux")
        print(f"  AR+Aux done in {time.time()-t0:.1f}s")

        # ---- XGBoost ----
        t0 = time.time()
        xgb_model = xgb_baseline.fit(y_train, station_name)
        step_xgb = xgb_baseline.make_predict_step(xgb_model, y_full)
        df_xgb = recursive_horizon_rmse(y_full, T_train, HORIZONS, max(xgb_baseline.SHORT_LAGS), step_xgb)
        df_xgb.insert(0, "model", "XGB")
        print(f"  XGB done in {time.time()-t0:.1f}s")

        # ---- LSTM ----
        t0 = time.time()
        lstm_model = lstm_baseline.fit(y_train, station_name)
        step_lstm = lstm_baseline.make_predict_step(lstm_model, y_full)
        df_lstm = recursive_horizon_rmse(y_full, T_train, HORIZONS, max(lstm_baseline.SHORT_LAGS), step_lstm)
        df_lstm.insert(0, "model", "LSTM")
        print(f"  LSTM done in {time.time()-t0:.1f}s")

        df_loc = pd.concat([df_ar, df_aux, df_xgb, df_lstm], ignore_index=True)
        df_loc.insert(0, "location", station_name)
        df_loc.insert(1, "run_id", run_id)

        out_path = OUT_DIR / f"{run_id}_ml_baselines_horizon.csv"
        df_loc.to_csv(out_path, index=False)
        print(f"{station_name} total {time.time()-t_loc:.1f}s -> {out_path.name}")
        print(df_loc.to_string(index=False))
        print()

        all_rows.append(df_loc)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUT_DIR / "all_locations_ml_baselines_horizon.csv", index=False)


if __name__ == "__main__":
    main()
