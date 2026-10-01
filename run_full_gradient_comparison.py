"""
Stage 1/2 fits for the full cross-stage gradient-norm comparison report
(2026-09-06).

For each location, fits:

  Stage 1 (ZAGASModel, no covariates):
    gas_lags in {[1], [1,365,366]}  x  branch in {phi_only, phi_xi}

  Stage 2 (CovZAGASModel):
    same lag x branch grid, crossed with covariate count:
      1 cov -> dewpoint_lag1
      2 cov -> dewpoint_lag1 + temperature_lag1
      3 cov -> + ENSO dummies (el_nino_t, la_nina_t)

Every model's gradient norm is recomputed INDEPENDENTLY via
diagnostics.numerical_diff.numerical_gradient evaluated at the model's own
final theta -- NOT scipy's res.jac, per explicit instruction. Cached to:

    artifacts/full_gradient_comparison/<short>/stage{1,2}/<config_id>/result.json
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
from models.za_gas_model import ZAGASModel
from models.cov_gas_model import CovZAGASModel
from distributions.gb2_log_link import GB2LogLink
from pi_dynamics.factory import make_pi_dynamics
from diagnostics.scoring import compute_extended_oos_metrics
from diagnostics.numerical_diff import numerical_gradient_checked
from data.station_loader import STATION_REGISTRY, StationDataLoader

LOCATIONS = ["BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
             "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR"]
LAG_SETS = {"lag1": [1], "lag1_365_366": [1, 365, 366]}
BRANCHES = {"phi_only": ["gamma", "zeta", "xi"], "phi_xi": ["gamma", "zeta"]}
OUT_DIR = ROOT / "artifacts" / "full_gradient_comparison"


def _eval_and_save(model, y_train, y_test, obj_fn, theta, oos_kwargs, out_dir, force):
    cache_p = out_dir / "result.json"
    if cache_p.exists() and not force:
        return json.loads(cache_p.read_text())
    oos = model.simulate_oos(theta, y_train, y_test, **oos_kwargs)
    m = compute_extended_oos_metrics(model, oos, y_test, quantile_levels=(0.95, 0.98),
                                      twcrps_levels=(0.95, 0.98), brier_levels=(), n_draws=500, seed=42)
    g_res = numerical_gradient_checked(obj_fn, theta)
    out = {"rmse": m.get("rmse"), "crps_mean": m.get("crps_mean"),
           "twcrps_95": m.get("twcrps_95"), "twcrps_98": m.get("twcrps_98"),
           "grad_norm_inf_numeric": float(np.max(np.abs(g_res["grad"]))),
           "grad_converged": g_res["converged"], "grad_eps_used": g_res["eps_used"],
           "grad_trace": g_res["trace"],
           "n_params": len(theta)}
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_p.write_text(json.dumps(out, indent=2, default=float))
    return out


def run_stage1(location: str, short: str, force: bool) -> dict:
    loader = StationDataLoader.from_registry(location, precip_dir=R.PRECIP_DIR, era5_dir=R.ERA5_DIR, enso_path=R.ENSO_PATH)
    data = loader.load_all()
    y_train, y_test = data["y_train"], data["y_test"]
    pi_dyn_factory = lambda: make_pi_dynamics("ar_logistic", seasonal="daily")

    out = {}
    for lag_key, gas_lags in LAG_SETS.items():
        for branch, static_params in BRANCHES.items():
            cfg_id = f"{lag_key}_{branch}"
            out_dir = OUT_DIR / short / "stage1" / cfg_id
            cache_p = out_dir / "result.json"
            if cache_p.exists() and not force:
                out[cfg_id] = json.loads(cache_p.read_text())
                continue
            model = ZAGASModel(distribution=GB2LogLink(), pi_dynamics=pi_dyn_factory(),
                                seasonal="daily", gas_lags=gas_lags,
                                scaling="diagonal_inverse_fisher", static_params=static_params)
            fit_res = model.fit(y=y_train, verbose=False)
            theta = fit_res["theta"]
            obj_fn = lambda th: model._run_filter(th, y_train, return_paths=False)
            out[cfg_id] = _eval_and_save(model, y_train, y_test, obj_fn, theta,
                                          {}, out_dir, force)
            print(f"    stage1/{cfg_id}: rmse={out[cfg_id]['rmse']:.3f} "
                  f"grad={out[cfg_id]['grad_norm_inf_numeric']:.3f}")
    return out


def run_stage2(location: str, short: str, force: bool) -> dict:
    loader = StationDataLoader.from_registry(location, precip_dir=R.PRECIP_DIR, era5_dir=R.ERA5_DIR, enso_path=R.ENSO_PATH)
    data = loader.load_all()
    y_train, y_test = data["y_train"], data["y_test"]
    bt, bv, cn = data["covariate_blocks_train"], data["covariate_blocks_test"], data["covariate_col_names"]

    dt_names = cn["dewtemp_short"]
    i_dew1 = dt_names.index("dewpoint_lag1")
    i_temp1 = dt_names.index("temperature_lag1")
    enso_cur = bt["enso_dummy_current"]; enso_cur_te = bv["enso_dummy_current"]

    def build_X(n_cov, split="train"):
        b, e = (bt, enso_cur) if split == "train" else (bv, enso_cur_te)
        cols = [b["dewtemp_short"][:, [i_dew1]]]
        names = ["dewpoint_lag1"]
        if n_cov >= 2:
            cols.append(b["dewtemp_short"][:, [i_temp1]]); names.append("temperature_lag1")
        if n_cov >= 3:
            cols.append(e); names += ["el_nino_t", "la_nina_t"]
        return np.concatenate(cols, axis=1), names

    out = {}
    for lag_key, gas_lags in LAG_SETS.items():
        for branch, static_params in BRANCHES.items():
            for n_cov in (1, 2, 3):
                cfg_id = f"{lag_key}_{branch}_{n_cov}cov"
                out_dir = OUT_DIR / short / "stage2" / cfg_id
                cache_p = out_dir / "result.json"
                if cache_p.exists() and not force:
                    out[cfg_id] = json.loads(cache_p.read_text())
                    continue
                X_tr, names = build_X(n_cov, "train")
                X_te, _ = build_X(n_cov, "test")
                model = CovZAGASModel(distribution=GB2LogLink(), pi_dynamics=make_pi_dynamics("ar_logistic", seasonal="daily"),
                                       seasonal="daily", gas_lags=gas_lags, scaling="diagonal_inverse_fisher",
                                       cov_names=names, cov_target=None if branch == "phi_xi" else ["phi"],
                                       static_params=static_params)
                fit_res = model.fit(y=y_train, X=X_tr, verbose=False)
                theta = fit_res["theta"]
                obj_fn = lambda th: model._run_filter(th, y_train, X_tr, return_paths=False)
                out[cfg_id] = _eval_and_save(model, y_train, y_test, obj_fn, theta,
                                              {"X_train": X_tr, "X_test": X_te}, out_dir, force)
                print(f"    stage2/{cfg_id}: rmse={out[cfg_id]['rmse']:.3f} "
                      f"grad={out[cfg_id]['grad_norm_inf_numeric']:.3f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary = {}
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        short = cfg.get("short", location)
        print(f"=== {cfg.get('display', location)} ===")
        t0 = time.time()
        s1 = run_stage1(location, short, args.force)
        s2 = run_stage2(location, short, args.force)
        summary[short] = {"stage1": s1, "stage2": s2}
        print(f"  ({time.time()-t0:.1f}s)")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
