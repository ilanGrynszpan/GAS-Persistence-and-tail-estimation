"""
Brier Skill Score (BSS, Mason 2004) of the GAS model's out-of-sample
rainfall-occurrence probabilities against a monthly-climatology reference
forecast, for reports/multi_location/bss_results.tex.

METHOD (all quantities read from cached artifacts / replayed via the
model's own deterministic filter -- no re-fitting, per instruction):

  1. Monthly climatological probability, TRAINING SAMPLE ONLY:
         p_clim[j, m] = (# training days in month m with y_t > 0)
                        / (# training days in month m)
     for each location j, month m = 1..12.

  2. GAS one-step-ahead OOS occurrence probability pi_{t|t-1}: the standard
     walk-forward filtered probability (simulation/simulator.py::
     simulate_oos's pi_oos) -- the same "Technique 1" walk-forward
     evaluation documented in reports/oos_methodology/oos_methodology.pdf,
     for the stage1_phi_short_diagfi arrangement used throughout this
     analysis (docs/EXECUTION.md's "reuse artifacts" principle -- this is
     a pure replay of an already-completed fit, not re-estimation).

  3. BS_GAS,j  = mean_t [ pi_{t|t-1} - I(y_t>0) ]^2 over ALL OOS days.
     BS_clim,j = mean_t [ p_clim[j, month(t)] - I(y_t>0) ]^2 over the SAME
     OOS days (month(t) from each OOS day's own calendar date).
     BSS_j = 1 - BS_GAS,j / BS_clim,j.

No split by horizon or by month is reported (per instruction) -- one
aggregate BSS per location, over the complete OOS sample.

Usage:
    python run_bss_climatology.py
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
from pipeline.artifact_utils import load_model_and_theta
from simulation.simulator import simulate_oos

PRECIP_DIR = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR   = ROOT / "data" / "input" / "ERA5"
ENSO_PATH  = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
OUT_TEX = ROOT / "reports" / "multi_location" / "bss_results.tex"

LOCATIONS = [
    ("BELO HORIZONTE",         "run_20260701_bh",        "Belo Horizonte"),
    ("CRUZEIRO DO SUL (ACRE)", "run_20260705_cruzeiro",  "Cruzeiro do Sul"),
    ("DARWIN AIRPORT",         "run_20260705_darwin",    "Darwin Airport"),
    ("GARANHUNS (PERNAMBUCO)", "run_20260705_garanhuns", "Garanhuns"),
    ("MANAUS",                 "run_20260705_manaus",    "Manaus"),
    ("SALVADOR",               "run_20260705_salvador",  "Salvador"),
]


def main() -> None:
    rows = []
    for station_name, run_id, label in LOCATIONS:
        loader = StationDataLoader.from_registry(
            station_name=station_name,
            precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH,
        )
        data = loader.load_all()
        y_train, y_test = data["y_train"], data["y_test"]
        dates_train, dates_test = data["dates_train"], data["dates_test"]

        # ---- 1. Monthly climatology, training sample only -----------------
        wet_train = (y_train > 0).astype(np.float64)
        month_train = dates_train.month.to_numpy()
        p_clim = np.array([
            wet_train[month_train == m].mean() for m in range(1, 13)
        ])

        # ---- 2. GAS one-step-ahead OOS occurrence probability --------------
        model_dir = ARTIFACTS_DIR / run_id / "stage1" / "stage1_phi_short_diagfi"
        model, theta, meta, params_df = load_model_and_theta(model_dir)
        oos = simulate_oos(model, theta, y_train, y_test)
        pi_oos = oos["pi_oos"]

        assert len(pi_oos) == len(y_test), (
            f"{station_name}: pi_oos length {len(pi_oos)} != y_test length {len(y_test)}"
        )

        # ---- 3. Brier scores + BSS ------------------------------------------
        occurred = (y_test > 0).astype(np.float64)
        month_test = dates_test.month.to_numpy()
        p_clim_t = p_clim[month_test - 1]

        bs_gas = float(np.mean((pi_oos - occurred) ** 2))
        bs_clim = float(np.mean((p_clim_t - occurred) ** 2))
        bss = 1.0 - bs_gas / bs_clim

        rows.append({
            "location": label, "station_name": station_name, "run_id": run_id,
            "N_oos": len(y_test),
            "bs_gas": bs_gas, "bs_clim": bs_clim, "bss": bss,
        })
        print(f"{label:<18} N={len(y_test):4d}  BS_GAS={bs_gas:.4f}  "
              f"BS_clim={bs_clim:.4f}  BSS={bss:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / "multi_location" / "bss_results.csv", index=False)
    write_tex(df)


def write_tex(df: pd.DataFrame) -> None:
    best = df.loc[df["bss"].idxmax()]
    worst = df.loc[df["bss"].idxmin()]
    n_positive = int((df["bss"] > 0).sum())
    n_total = len(df)

    lines = []
    lines.append("% Brier Skill Score (BSS, Mason 2004) of the GAS model's out-of-sample")
    lines.append("% rainfall-occurrence probabilities against a monthly-climatology")
    lines.append("% reference forecast. Generated by run_bss_climatology.py -- reads only")
    lines.append("% cached model artifacts (no re-estimation).")
    lines.append("")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Location & GAS Brier Score & Climatology Brier Score & BSS \\\\")
    lines.append("\\midrule")
    for _, r in df.iterrows():
        lines.append(
            f"{r['location']} & {r['bs_gas']:.4f} & {r['bs_clim']:.4f} & {r['bss']:+.4f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Brier Skill Score of the GAS model's out-of-sample "
        "rainfall-occurrence probability $\\pi_{t|t-1}$ against a monthly "
        "climatological reference forecast.}"
    )
    lines.append("\\label{tab:bss_climatology}")
    lines.append("\\end{table}")
    lines.append("")

    para = (
        "Monthly climatological rainfall-occurrence probabilities were estimated "
        "separately for each location and calendar month from the training sample "
        "only, yielding twelve reference probabilities per "
        "location, and were used as a naive benchmark forecast against which the "
        "GAS model's dynamic occurrence probability $\\pi_{t|t-1}$ is compared. Both "
        "the GAS Brier score and the climatological Brier score were evaluated over "
        "the complete out-of-sample period at each location, using the identical set "
        "of out-of-sample observations for both forecasts. Following Mason (2004), "
        "climatology is used as the reference strategy for the "
        "Brier Skill Score, "
        "$\\mathrm{BSS} = 1 - \\mathrm{BS}_{\\mathrm{GAS}}/\\mathrm{BS}_{\\mathrm{clim}}$, "
        f"with $\\mathrm{{BSS}}>0$ indicating that the GAS model outperforms monthly "
        f"climatology. Across the six locations, BSS ranges from "
        f"{worst['bss']:+.4f} at {worst['location']} to {best['bss']:+.4f} at "
        f"{best['location']}. "
        f"{best['location']} shows the largest improvement of the GAS model over "
        f"monthly climatology, while {worst['location']} shows the "
        f"{'smallest improvement' if worst['bss'] > 0 else 'weakest relative performance'} "
        f"among the locations considered. "
    )

    if n_positive == n_total:
        para += (
            "The GAS model improves upon monthly climatology at every location "
            "(BSS positive throughout), indicating that its dynamic, "
            "day-to-day occurrence probability carries genuine out-of-sample "
            "predictive information beyond the seasonal cycle alone."
        )
    elif n_positive == 0:
        para += (
            "The GAS model does not improve upon monthly climatology at any "
            "location (BSS negative throughout), indicating that, for daily "
            "rainfall occurrence, the simple seasonal climatological benchmark is "
            "at least as informative out of sample as the model's dynamic "
            "probability at every location considered."
        )
    else:
        def _oxford(names: list[str]) -> str:
            if len(names) == 1:
                return names[0]
            return ", ".join(names[:-1]) + f", and {names[-1]}"

        gains = df[df["bss"] > 0]["location"].tolist()
        losses = df[df["bss"] <= 0]["location"].tolist()
        para += (
            f"The GAS model improves upon monthly climatology (BSS $>0$) at "
            f"{_oxford(gains)}, while monthly climatology is at least as "
            f"informative as the GAS model (BSS $\\le 0$) at {_oxford(losses)}, "
            "indicating that the value of the model's dynamic occurrence "
            "probability over the seasonal cycle alone is location-dependent."
        )

    lines.append(para)
    lines.append("")

    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_TEX}")


if __name__ == "__main__":
    main()
