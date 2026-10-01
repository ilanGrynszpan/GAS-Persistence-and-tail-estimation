"""
Per-branch (one-TVP vs two-TVP) Harvey long-short analysis, both branches at
every location (2026-09-01).

=============================================================================
OVERVIEW
=============================================================================

run_longshort_best_models_analysis.py picks a single overall-best
configuration per location (whichever of phi_only/phi_xi has the lowest
|grad|_inf). For generate_longshort_tvp_report.py ("time-varying location
only" vs "time-varying location + shape"), both branches' own best
configuration are needed at every location, not just the overall winner.

This script reuses that module's find_best_config() (now branch-filterable)
and analyze_config() (the in-sample-residuals / 1-step-OOS / multi-step-OOS
computation, factored out for exactly this reuse) for BOTH branches at each
of the six locations. Where a branch's best configuration is identical to
the overall-best already computed in artifacts/longshort_diagnostics_experiment/
best_models/<short>/ (true whenever that branch happens to be the overall
winner), the already-saved outputs are copied rather than recomputed --
Monte Carlo multi-step metrics are seeded (MC_SEED) so this is a byte-for-
byte reproduction, not merely an equivalent one, and copying avoids ~6
redundant multi-step Monte Carlo runs (~2-3 minutes each).

Per docs/EXECUTION.md, no optimisation happens here: every model is
reconstructed from already-saved theta and only evaluated.

=============================================================================
USAGE
=============================================================================

    python run_longshort_tvp_comparison.py
    python run_longshort_tvp_comparison.py --location "DARWIN AIRPORT"

Output (per location, per branch):
    artifacts/longshort_diagnostics_experiment/tvp_comparison/<short>/<branch>/
        best_model_info.json, harvey_quantile_residuals.npy,
        cov_quantile_residuals.npy, onestep_metrics.json,
        multistep_metrics_harvey.csv, multistep_metrics_cov.csv
    artifacts/longshort_diagnostics_experiment/tvp_comparison/summary.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import STATION_REGISTRY, StationDataLoader
from run_longshort_best_models_analysis import (
    PRECIP_DIR, ERA5_DIR, ENSO_PATH, ARTIFACTS_DIR, EXP_DIR,
    OUT_DIR as OVERALL_BEST_DIR,
    find_best_config, analyze_config,
)

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]
BRANCHES = ("phi_only", "phi_xi")

TVP_OUT_DIR = EXP_DIR / "tvp_comparison"

COPY_FILES = [
    "best_model_info.json", "harvey_quantile_residuals.npy",
    "cov_quantile_residuals.npy", "onestep_metrics.json",
    "multistep_metrics_harvey.csv", "multistep_metrics_cov.csv",
]


def _reuse_from_overall_best(short: str, branch: str, out_dir: Path) -> bool:
    """
    If this (location, branch) pair's best config is exactly the one
    already analysed as the overall winner in best_models/<short>/, copy
    those outputs instead of recomputing. Returns True if reused.
    """
    overall_dir = OVERALL_BEST_DIR / short
    info_path = overall_dir / "best_model_info.json"
    if not info_path.exists():
        return False
    overall_info = json.loads(info_path.read_text())
    if overall_info.get("branch") != branch:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in COPY_FILES:
        src = overall_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", default=None, help="Run only this location (exact STATION_REGISTRY name)")
    args = parser.parse_args()
    locations = [args.location] if args.location else LOCATIONS

    TVP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for location in locations:
        cfg = STATION_REGISTRY[location]
        run_id, display = cfg["run_id"], cfg.get("display", location)
        short = cfg.get("short", run_id)
        print(f"=== {display} ===")

        loader = StationDataLoader.from_registry(location, precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH)
        data = None  # loaded lazily, only if a fresh (non-reused) computation is needed

        for branch in BRANCHES:
            out_dir = TVP_OUT_DIR / short / branch
            try:
                best = find_best_config(run_id, branch=branch)
                if best is None:
                    print(f"  [{branch}] No non-failed config found; skipping.")
                    summary_rows.append({"location": display, "branch": branch, "status": "no_config"})
                    continue

                if _reuse_from_overall_best(short, branch, out_dir):
                    print(f"  [{branch}] {best['config']} (|grad|_inf={best['grad_norm_inf']:.4g}) "
                          f"-- reused from best_models/{short}/ (this branch is the overall winner there)")
                    info = json.loads((out_dir / "best_model_info.json").read_text())
                    summary_rows.append({"location": display, "branch": branch, "status": "ok_reused", **info})
                    continue

                print(f"  [{branch}] {best['config']} (|grad|_inf={best['grad_norm_inf']:.4g}) -- computing fresh")
                if data is None:
                    data = loader.load_all()
                info = analyze_config(display, run_id, data, best, out_dir)
                summary_rows.append({"location": display, "branch": branch, "status": "ok", **info})

            except Exception as exc:
                print(f"  [{branch}] FAILED: {exc}")
                traceback.print_exc()
                summary_rows.append({"location": display, "branch": branch, "status": "failed", "error": str(exc)})

    pd.DataFrame(summary_rows).to_csv(TVP_OUT_DIR / "summary.csv", index=False)
    print("\nDone. Summary written to", TVP_OUT_DIR / "summary.csv")


if __name__ == "__main__":
    main()
