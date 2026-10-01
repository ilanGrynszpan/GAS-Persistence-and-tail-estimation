"""
Theoretical-vs-empirical exceedance ratio, standard vs.\ seasonal Harvey
long-short (2026-09-06).

=============================================================================
OVERVIEW
=============================================================================

For a well-calibrated model, a quantile level q's OOS PIT values should
exceed q exactly (1-q) of the time -- e.g.\ a 98th-percentile forecast
should be exceeded on 2% of days ("a 1-in-50-day event"). This computes,
for both the standard (no seasonal) and seasonal (damped) Harvey
long-short models, the RATIO

    ratio(q) = empirical exceedance rate / theoretical exceedance rate
             = mean(PIT_t > q) / (1 - q)

at q = 0.95, 0.98, and 1-1/30 (the "1-in-30-day" event), for every
location/branch. ratio > 1 means MORE real exceedances happened than the
model expects (the model understates tail risk); ratio < 1 means fewer
(the model is too conservative in the tail); ratio = 1 is exact
calibration.

Per direct instruction, results are kept in their own dedicated cache
(NOT mixed into any other report's artifact tree):

    artifacts/exceedance_ratios/standard_vs_seasonal.json

=============================================================================
USAGE
=============================================================================

    python compute_exceedance_ratios.py [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_harmonic_seasonal_experiment as R
from pipeline.artifact_utils import load_model_and_theta
from diagnostics.scoring import compute_extended_oos_metrics
from data.station_loader import STATION_REGISTRY, StationDataLoader

LOCATIONS = ["BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
             "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR"]
BRANCHES = ["phi_only", "phi_xi"]
LEVELS = (0.95, 0.98, 1 - 1 / 30)  # 95%, 98%, "1-in-30-day"
OUT_PATH = ROOT / "artifacts" / "exceedance_ratios" / "standard_vs_seasonal.json"
CMP_DIR = ROOT / "artifacts" / "harmonic_seasonal_comparison_m5_final_damp0.999"


def _seasonal_is_diverged(short: str, branch: str) -> bool:
    """Mirror generate_harmonic_seasonal_report.py's is_diverged() gate.

    The seasonal (undamped-adjacent, m=5) filter for a handful of
    location/branch combinations enters a numerically degenerate regime
    where the log-link state f_t swings to +-100+ without ever hitting a
    literal NaN/inf -- e.g.\ Darwin/phi_only's f_t ranges over [-131, 83]
    on 211/731 test days. A CDF is bounded in [0,1] by construction, so
    PIT values computed from such a state stay "finite" and silently pass
    an np.isfinite() gate even though the state itself has diverged --
    the resulting exceedance ratio would then reflect an already-broken
    forecast, not a genuine calibration property. Must gate on the same
    RMSE-based divergence check used for the OOS RMSE/CRPS comparison
    (reused from cache, not recomputed) so both reports agree on which
    combinations are usable.
    """
    try:
        std = json.loads((CMP_DIR / short / branch / "standard" / "onestep_metrics.json").read_text())
        seas = json.loads((CMP_DIR / short / branch / "seasonal" / "onestep_metrics.json").read_text())
        r_seas, r_std = seas["rmse"], std["rmse"]
        return (not np.isfinite(r_seas)) or r_seas > 3.0 * r_std
    except Exception:
        return False


def compute_ratios(model, theta, data, meta) -> dict:
    Xl_tr = R._build_X(data, meta["long_names"], "train")
    Xl_te = R._build_X(data, meta["long_names"], "test")
    Xs_tr = R._build_X(data, meta["short_names"], "train")
    Xs_te = R._build_X(data, meta["short_names"], "test")
    y_train, y_test = data["y_train"], data["y_test"]
    oos = model.simulate_oos(theta, y_train, y_test, Xl_tr, Xl_te, Xs_tr, Xs_te)
    m = compute_extended_oos_metrics(model, oos, y_test, quantile_levels=LEVELS,
                                      twcrps_levels=(), brier_levels=(), n_draws=500,
                                      seed=42, return_raw=True)
    pit = np.array(m["_pit_values"])
    ratios = {}
    for lv in LEVELS:
        theo = 1.0 - lv
        emp = float(np.mean(pit > lv)) if np.all(np.isfinite(pit)) else float("nan")
        ratios[f"{lv:.4f}"] = emp / theo if theo > 0 else float("nan")
    return ratios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if OUT_PATH.exists() and not args.force:
        print(f"{OUT_PATH} already exists (use --force to recompute).")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        short, display = cfg.get("short", location), cfg.get("display", location)
        print(f"=== {display} ===")
        loader = StationDataLoader.from_registry(location, precip_dir=R.PRECIP_DIR,
                                                  era5_dir=R.ERA5_DIR, enso_path=R.ENSO_PATH)
        data = loader.load_all()
        for branch in BRANCHES:
            try:
                baseline = R.resolve_baseline(short, branch)
                std_model, std_theta, std_meta, _ = load_model_and_theta(baseline["model_dir"])
                seas_dir = R.OUT_DIR / short / branch / "m5_final_damp0.999"
                seas_model, seas_theta, seas_meta, _ = load_model_and_theta(seas_dir)
                res = {}
                for tag, model, theta, meta in [("standard", std_model, std_theta, std_meta),
                                                 ("seasonal", seas_model, seas_theta, seas_meta)]:
                    if tag == "seasonal" and _seasonal_is_diverged(short, branch):
                        res[tag] = {f"{lv:.4f}": float("nan") for lv in LEVELS}
                        print(f"    [{branch}/{tag}] SKIPPED: seasonal state diverged (see is_diverged gate)")
                        continue
                    try:
                        res[tag] = compute_ratios(model, theta, data, meta)
                    except Exception as exc:
                        res[tag] = {f"{lv:.4f}": float("nan") for lv in LEVELS}
                        print(f"    [{branch}/{tag}] FAILED: {exc}")
                out[f"{short}_{branch}"] = {"display": display, "branch": branch, **res}
                print(f"  [{branch}] done: {res}")
            except Exception as exc:
                print(f"  [{branch}] FAILED: {exc}")
                traceback.print_exc()

    OUT_PATH.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
