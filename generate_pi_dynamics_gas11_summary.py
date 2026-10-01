"""
Summary-of-alternatives table for the pi-dynamics GAS(1,1) VALIDATION
experiment (2026-09-10).

=============================================================================
OVERVIEW
=============================================================================

Reproduces reports/pi_dynamics_gas11/tables/summary/summary_of_alternatives
.tex, but on the held-out basis produced by run_pi_dynamics_gas11_validation
.py: every number is computed on the final 2 calendar years of *_train.csv
(the validation block), 1-step-ahead, from models fit only on the first 8
years. All four alternatives A--D are reported (the earlier hand-made file
listed only A and B).

Per docs/REPORTING.md Sec 2-3 this performs NO optimisation: it only reads
artifacts/pi_dynamics_gas11_validation_experiment/<short>/<variant>/
metadata.json and tabulates the `val_*` fields.

Output (same folder as the file it replaces):
    reports/pi_dynamics_gas11/tables/summary/summary_of_alternatives.tex
    reports/pi_dynamics_gas11/tables/summary/summary_of_alternatives.csv

The CSV has plain snake_case columns and is read back with a bare
pd.read_csv(path):
    location, alternative, rmse, crps, mean_pi_y0, mean_pi_ygt0, n_dry, n_wet

=============================================================================
USAGE
=============================================================================

    python generate_pi_dynamics_gas11_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import STATION_REGISTRY

ARTIFACTS_DIR = ROOT / "artifacts" / "pi_dynamics_gas11_validation_experiment"
OUT_DIR       = ROOT / "reports" / "pi_dynamics_gas11" / "tables" / "summary"

LOCATION_ORDER = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]
VARIANT_ORDER = ["existing_ar_seasonal", "short_lags_ar", "short_lags_noar", "phi_linked"]
VARIANT_CODE  = {"existing_ar_seasonal": "A", "short_lags_ar": "B",
                 "short_lags_noar": "C", "phi_linked": "D"}


def _load_rows() -> pd.DataFrame:
    rows = []
    for loc in LOCATION_ORDER:
        short = STATION_REGISTRY[loc]["short"]
        for vk in VARIANT_ORDER:
            meta_path = ARTIFACTS_DIR / short / vk / "metadata.json"
            if not meta_path.exists():
                print(f"  [MISS] {short}/{vk}: no metadata.json")
                rows.append({"location": short, "alternative": VARIANT_CODE[vk],
                             "rmse": np.nan, "crps": np.nan, "mean_pi_y0": np.nan,
                             "mean_pi_ygt0": np.nan, "n_dry": np.nan, "n_wet": np.nan})
                continue
            m = json.loads(meta_path.read_text())
            if not m.get("success"):
                print(f"  [FAIL] {short}/{vk}: {m.get('message')}")
            rows.append({
                "location":     short,
                "alternative":  VARIANT_CODE[vk],
                "rmse":         m.get("val_rmse_all", np.nan),
                "crps":         m.get("val_crps_all", np.nan),
                "mean_pi_y0":   m.get("val_pi_mean_dry", np.nan),
                "mean_pi_ygt0": m.get("val_pi_mean_wet", np.nan),
                "n_dry":        m.get("val_n_dry", np.nan),
                "n_wet":        m.get("val_n_wet", np.nan),
            })
    return pd.DataFrame(rows)


def _int(v) -> str:
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(int(v))


def _f3(v) -> str:
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{float(v):.3f}"


def _build_tex(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Summary of alternatives A--D by location. All quantities are "
        r"computed on the validation block (final two calendar years of the "
        r"training series); the models are estimated only on the first eight "
        r"years, and the validation block is scored one-step-ahead with no "
        r"re-fit. RMSE and CRPS are for $y_t$ itself, combining each "
        r"alternative's $\pi_t$ with the shared frozen GAS(1,1) magnitude "
        r"path. Lower RMSE / CRPS is better.}",
        r"\label{tab:summary_of_alternatives}",
        r"\begin{tabular}{llrrrrrr}",
        r"\hline",
        r"Location & Alternative & RMSE & CRPS & mean $\pi \mid y=0$ & "
        r"mean $\pi \mid y>0$ & $N$ dry & $N$ wet \\",
        r"\hline",
    ]
    for i, (loc, sub) in enumerate(df.groupby("location", sort=False)):
        if i:
            lines.append(r"\hline")
        for _, r in sub.iterrows():
            lines.append(
                f"{loc} & {r['alternative']} & {_f3(r['rmse'])} & {_f3(r['crps'])} & "
                f"{_f3(r['mean_pi_y0'])} & {_f3(r['mean_pi_ygt0'])} & "
                f"{_int(r['n_dry'])} & {_int(r['n_wet'])} \\\\"
            )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_rows()

    csv_path = OUT_DIR / "summary_of_alternatives.csv"
    tex_path = OUT_DIR / "summary_of_alternatives.tex"
    df.to_csv(csv_path, index=False)
    tex_path.write_text(_build_tex(df))

    print(df.to_string(index=False))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
