"""
Occurrence-vs-magnitude decomposition of the exceedance ratio (2026-09-06).

=============================================================================
OVERVIEW
=============================================================================

compute_exceedance_ratios.py's ratio(q) = mean(PIT_t > q) / (1-q) mixes two
architecturally separate model components together: the occurrence process
pi_t (an AR-logistic recursion over wet/dry days with its own seasonal
lag structure, see pi_dynamics/ar_logistic.py) and the conditional GB2
magnitude distribution given a wet day (the GAS-driven phi_t/xi_t
dynamics). A ratio(q) != 1 could in principle come from either piece.

This script isolates them for the (already-converged, non-diverged)
standard Harvey long-short model, phi-only branch, at all six locations:

  1. Occurrence calibration: mean(model's own pi_t) vs. the actual wet-day
     rate in the OOS test set -- a pure occurrence check, independent of
     the GB2 magnitude shape entirely.
  2. Conditional-on-wet ("magnitude-only") ratio: for wet days only,
     u_t = (PIT_t - pi_dry_t) / (1 - pi_dry_t), which strips the
     occurrence mixture out algebraically and leaves exactly the GB2
     conditional CDF value F_GB2(y_t) -- so mean(u_t > q)/(1-q) measures
     calibration of the magnitude tail ALONE, with occurrence's influence
     removed.

Comparing the two reveals whether a given location's exceedance-ratio
miscalibration is an occurrence-side or magnitude-side phenomenon.

Cached separately from standard_vs_seasonal.json (own dedicated file, per
the "keep results in separate caching" instruction):

    artifacts/exceedance_ratios/occurrence_magnitude_decomposition.json

=============================================================================
USAGE
=============================================================================

    python compute_occurrence_magnitude_decomposition.py [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
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
LEVELS = (0.95, 0.98)
OUT_PATH = ROOT / "artifacts" / "exceedance_ratios" / "occurrence_magnitude_decomposition.json"


def decompose(location: str, short: str) -> dict:
    loader = StationDataLoader.from_registry(location, precip_dir=R.PRECIP_DIR,
                                              era5_dir=R.ERA5_DIR, enso_path=R.ENSO_PATH)
    data = loader.load_all()
    baseline = R.resolve_baseline(short, "phi_only")
    model, theta, meta, _ = load_model_and_theta(baseline["model_dir"])
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
    pi_t = np.array(oos["pi_oos"])
    wet = y_test > 0
    pi_dry_t = 1.0 - pi_t
    u_wet = (pit[wet] - pi_dry_t[wet]) / (1.0 - pi_dry_t[wet])

    out = {
        "actual_wet_rate": float(np.mean(wet)),
        "mean_model_pi": float(np.mean(pi_t)),
        "occurrence_bias": float(np.mean(pi_t) - np.mean(wet)),
        "full_ratio": {}, "conditional_on_wet_ratio": {},
    }
    for lv in LEVELS:
        theo = 1.0 - lv
        out["full_ratio"][f"{lv:.4f}"] = float(np.mean(pit > lv)) / theo
        out["conditional_on_wet_ratio"][f"{lv:.4f}"] = float(np.mean(u_wet > lv)) / theo
    return out


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
        try:
            res = decompose(location, short)
            out[short] = {"display": display, **res}
            print(f"  {res}")
        except Exception as exc:
            print(f"  FAILED: {exc}")

    OUT_PATH.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
