"""
Stage 2 gas_lags-vs-covariate-lags ablation (2026-09-06).

=============================================================================
WHY THIS SCRIPT EXISTS
=============================================================================

Investigating why Stage 2 (CovZAGASModel) shows huge optimizer gradients at
several locations. Direct inspection of a fitted model's own gradients.csv
(Salvador phi+xi, artifacts/run_20260705_salvador/stage2/stage2_dewpoint_short/)
showed the large gradients (4-31 in magnitude) concentrated on the GAS
persistence coefficients B_phi_{1,2,3,364,365,366,367} and
B_xi_{1,2,3,364,365,366,367} -- ALL SEVEN lags roughly the same order of
magnitude (a "generalized", not "concentrated", pattern) -- while the dew
point covariate loadings (gamma_*_dewpoint_lag{1,2,3}) were 10-20x smaller.
This is true at every one of the six locations (all use the 7-lag
GAS_SEASONAL_LAGS set at Stage 2, confirmed directly from each model's own
parameter names), suggesting the seven-lag GAS persistence structure itself
-- not dew point collinearity -- is the primary driver.

This script tests both candidate explanations directly, crossed:

    A) baseline:      gas_lags=SEASONAL (7 lags), cov=dewpoint_short (3 lags)
    B) 1 dew point lag: gas_lags=SEASONAL (7 lags), cov=dewpoint_lag1 only
    C) short gas lags:  gas_lags=SHORT (3 lags),    cov=dewpoint_short (3 lags)
    D) both:            gas_lags=SHORT (3 lags),    cov=dewpoint_lag1 only

for the phi+xi branch at every location, reporting max|gradient| and OOS
RMSE/CRPS for each. Cached to its own dedicated tree (not touching any
production stage2 artifact):

    artifacts/gas_lags_covariate_ablation/<short>/<config>/

=============================================================================
USAGE
=============================================================================

    python run_bound_ab_experiment.py [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_harmonic_seasonal_experiment as R
from models.cov_gas_model import CovZAGASModel
from distributions.gb2_log_link import GB2LogLink
from pi_dynamics.factory import make_pi_dynamics
from diagnostics.scoring import compute_extended_oos_metrics
from data.station_loader import STATION_REGISTRY, StationDataLoader
from constants import GAS_SHORT_LAGS, GAS_SEASONAL_LAGS

LOCATIONS = ["BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
             "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR"]
OUT_DIR = ROOT / "artifacts" / "gas_lags_covariate_ablation"

CONFIGS = {
    "A_seasonal_lags_3cov": (GAS_SEASONAL_LAGS["daily"], 3),
    "B_seasonal_lags_1cov": (GAS_SEASONAL_LAGS["daily"], 1),
    "C_short_lags_3cov":    (GAS_SHORT_LAGS["daily"],    3),
    "D_short_lags_1cov":    (GAS_SHORT_LAGS["daily"],    1),
    # Parsimonious alternatives (2026-09-06, user-suggested): rather than
    # the redundant adjacent-lag clusters in GAS_SHORT_LAGS ([1,2,3]) and
    # GAS_SEASONAL_LAGS ([1,2,3,364,365,366,367]), test the minimal lag
    # set that still distinguishes "recent" from "annual recurrence" --
    # [1,365,366] is exactly SEASONAL_LAGS["daily"] from constants.py,
    # already used for the pi (occurrence) model, just never applied to
    # the GAS filter's own dynamics; [1] is the bare minimum, no seasonal
    # structure at all.
    "E_minimal_seasonal_1cov": ([1, 365, 366], 1),
    "F_single_lag_1cov":       ([1], 1),
}


def run_one(location: str, short: str, force: bool) -> dict:
    loader = StationDataLoader.from_registry(location, precip_dir=R.PRECIP_DIR,
                                              era5_dir=R.ERA5_DIR, enso_path=R.ENSO_PATH)
    data = loader.load_all()
    y_train, y_test = data["y_train"], data["y_test"]
    bt, bv, cn = data["covariate_blocks_train"], data["covariate_blocks_test"], data["covariate_col_names"]

    results = {}
    for cfg_name, (gas_lags, n_cov) in CONFIGS.items():
        out_dir = OUT_DIR / short / cfg_name
        cache_p = out_dir / "result.json"
        if cache_p.exists() and not force:
            results[cfg_name] = json.loads(cache_p.read_text())
            continue

        X_tr = bt["dewpoint_short"][:, :n_cov]
        X_te = bv["dewpoint_short"][:, :n_cov]
        cov_names = cn["dewpoint_short"][:n_cov]

        model = CovZAGASModel(
            distribution=GB2LogLink(),
            pi_dynamics=make_pi_dynamics("ar_logistic", seasonal="daily"),
            seasonal="daily", gas_lags=gas_lags,
            scaling="diagonal_inverse_fisher", cov_names=cov_names,
        )
        t0 = time.time()
        fit_res = model.fit(y=y_train, X=X_tr, verbose=False)
        oos = model.simulate_oos(fit_res["theta"], y_train, y_test, X_train=X_tr, X_test=X_te)
        m = compute_extended_oos_metrics(model, oos, y_test, quantile_levels=(0.95, 0.98),
                                          twcrps_levels=(), brier_levels=(), n_draws=500, seed=42)

        grad = fit_res.get("grad")
        names = fit_res.get("param_names", model.parameter_names())
        grad_by_name = {n: float(g) for n, g in zip(names, grad)} if grad is not None else {}
        top5 = sorted(grad_by_name.items(), key=lambda kv: -abs(kv[1]))[:5]

        out = {
            "rmse": m.get("rmse"), "crps_mean": m.get("crps_mean"),
            "grad_norm_inf": fit_res.get("grad_norm_inf"), "success": fit_res.get("success"),
            "validity": fit_res.get("validity"), "loglik": fit_res.get("loglik"),
            "top5_gradients": top5, "runtime_s": time.time() - t0,
            "n_gas_lags": len(gas_lags), "n_cov": n_cov,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps(out, indent=2, default=float))
        results[cfg_name] = out
        print(f"    {cfg_name}: rmse={out['rmse']:.3f} crps={out['crps_mean']:.3f} "
              f"grad_inf={out['grad_norm_inf']:.3f} success={out['success']} "
              f"top={top5[0] if top5 else None}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        short = cfg.get("short", location)
        print(f"=== {cfg.get('display', location)} ===")
        summary[short] = run_one(location, short, args.force)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
