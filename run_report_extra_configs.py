"""
Fit the extra configs needed for the 2026-09-11 stage-comparison report
(generate_stage_comparison_report.py), none of which existed in either
cache tree:

  Stage 2   "dew_enso"       (dewpoint_lag1 + ENSO dummies, no temperature)
            x 4 gas-lag sets x 2 branches x 6 locations = 48 fits
            -> artifacts/rerun_10092026/<loc>/stage2/<lag>_dew_enso[__xi]/

  Stage 3/4 "dew_temp"       (dewpoint_lag1 + temperature_lag1 in X_short,
                               no ENSO in X_long) x 2 branches x 6 locations
            = 24 fits, using the SIGMOID-bounded phi_L (phi_L_max=1.0) --
            the parameterisation found to fix Stage 3/4's instability
            -> artifacts/rerun_10092026_phiL_sigmoid/<loc>/stage3|4/dew_temp[__xi]/

  Stage 4S  "dew_temp"       x 2 branches x 6 locations = 12 fits
            (include_long=False, so phi_L doesn't exist there -- reuses
            the ORIGINAL tree, no sigmoid parameterisation needed)
            -> artifacts/rerun_10092026/<loc>/stage4S/dew_temp[__xi]/

_harvey_cov()/_cov_cols() in run_rerun_10092026.py were extended with
these two cfg keys (dew_temp: short=dew+temp,long=none; dew_enso [Stage 2
only]: dew+ENSO dummies) -- purely additive, no existing cfg's behaviour
changed. Every fit below reuses the existing, tested flat task functions
(_task_stage2 / _task_harvey / _task_harvey_sig) unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_rerun_10092026 as D
import run_stage34_phiL_sigmoid as S


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    locs = D.LOCATIONS
    branches = ("phi_only", "phi_xi")

    # ---- Stage 2: dew_enso, all 4 lag sets ----
    t2 = [(loc, br, lk, "dew_enso", args.force)
          for loc in locs for br in branches for lk in D.GAS_LAG_SETS]
    D._run_pool(t2, D._task_stage2, args.workers, "stage2 dew_enso")

    # ---- Stage 3/4: dew_temp, sigmoid phi_L ----
    t34 = [(loc, br, st, "dew_temp", args.force)
           for loc in locs for br in branches for st in ("stage3", "stage4")]
    D._run_pool(t34, S._task_harvey_sig, args.workers, "stage3+4 dew_temp (sigmoid phi_L)")

    # ---- Stage 4S: dew_temp (no phi_L involved, original tree) ----
    t4s = [(loc, br, "stage4S", "dew_temp", args.force)
           for loc in locs for br in branches]
    D._run_pool(t4s, D._task_harvey, args.workers, "stage4S dew_temp")

    print("\nAll extra configs fitted.")


if __name__ == "__main__":
    main()
