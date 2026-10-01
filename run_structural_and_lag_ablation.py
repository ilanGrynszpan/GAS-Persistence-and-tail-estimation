"""
Structural (no-long) Stage 4 variant + lag-366-removal ablation (2026-09-08).

Two independent experiments requested together:

1. Stage 4 structural variant: HarveyZAGASModel(include_long=False) --
   f_t = omega + S_t + seasonal_t (Harvey's own classical local-level/trend
   + trigonometric seasonal structural model), dropping the Long component
   entirely, for both branches at all six locations. Uses the SAME
   long_names/short_names/n_harmonics/seasonal_damping_max as each
   location's existing Stage 4 (m5_final_damp0.999) winner, loaded via
   pipeline.artifact_utils.load_model_and_theta -- only include_long
   changes.

2. Lag-366 removal: gas_lags=[1,365] instead of [1,365,366], same grid as
   run_full_gradient_comparison.py's "lag1_365_366" configs (Stage 1 and
   Stage 2, both branches, Stage 2's 1/2/3-covariate variants).

All gradients via diagnostics.numerical_diff.numerical_gradient_checked
(see that module's CORRECTNESS NOTE). Cached to:

    artifacts/structural_lag_ablation/<short>/stage4_structural/<branch>/
    artifacts/structural_lag_ablation/<short>/stage1/lag1_365_<branch>/
    artifacts/structural_lag_ablation/<short>/stage2/lag1_365_<branch>_<n>cov/
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
from models.harvey_gas import HarveyZAGASModel
from models.za_gas_model import ZAGASModel
from models.cov_gas_model import CovZAGASModel
from distributions.gb2_log_link import GB2LogLink
from pi_dynamics.factory import make_pi_dynamics
from pipeline.artifact_utils import load_model_and_theta
from diagnostics.scoring import compute_extended_oos_metrics
from diagnostics.numerical_diff import numerical_gradient_checked
from data.station_loader import STATION_REGISTRY, StationDataLoader

LOCATIONS = ["BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
             "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR"]
BRANCHES = ["phi_only", "phi_xi"]
OUT_DIR = ROOT / "artifacts" / "structural_lag_ablation"


def _save(out_dir: Path, result: dict, cache_p_name="result.json", force=False):
    cache_p = out_dir / cache_p_name
    if cache_p.exists() and not force:
        return json.loads(cache_p.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_p.write_text(json.dumps(result, indent=2, default=float))
    return result


def run_stage4_structural(location: str, short: str, force: bool) -> dict:
    loader = StationDataLoader.from_registry(location, precip_dir=R.PRECIP_DIR, era5_dir=R.ERA5_DIR, enso_path=R.ENSO_PATH)
    data = loader.load_all()
    y_train, y_test = data["y_train"], data["y_test"]
    out = {}
    for branch in BRANCHES:
        out_dir = OUT_DIR / short / "stage4_structural" / branch
        cache_p = out_dir / "result.json"
        if cache_p.exists() and not force:
            out[branch] = json.loads(cache_p.read_text())
            continue
        seas_dir = R.OUT_DIR / short / branch / "m5_final_damp0.999"
        if not seas_dir.exists():
            print(f"    stage4_structural/{branch}: SKIPPED (no base seasonal model)")
            continue
        static_model, _, meta, _ = load_model_and_theta(seas_dir)
        harvey_target = ["phi"] if branch == "phi_only" else ["phi", "xi"]
        static_params = ["gamma", "zeta", "xi"] if branch == "phi_only" else ["gamma", "zeta"]
        Xl_tr = R._build_X(data, meta["long_names"], "train")
        Xs_tr = R._build_X(data, meta["short_names"], "train")
        Xl_te = R._build_X(data, meta["long_names"], "test")
        Xs_te = R._build_X(data, meta["short_names"], "test")

        model = HarveyZAGASModel(
            distribution=GB2LogLink(), pi_dynamics=make_pi_dynamics("ar_logistic", seasonal="daily"),
            seasonal="daily", long_names=meta["long_names"], short_names=meta["short_names"],
            harvey_target=harvey_target, static_params=static_params,
            n_harmonics=meta.get("n_harmonics", 5),
            seasonal_damping_max=meta.get("seasonal_damping_max", 0.999),
            include_long=False,
        )
        t0 = time.time()
        fit_res = model.fit(y=y_train, X_long=Xl_tr, X_short=Xs_tr, verbose=False,
                             options={"maxiter": 300}, polish=False)
        theta = fit_res["theta"]
        obj = lambda th: model._run_filter(th, y_train, Xl_tr, Xs_tr, return_paths=False)
        g_res = numerical_gradient_checked(obj, theta)
        oos = model.simulate_oos(theta, y_train, y_test, Xl_tr, Xl_te, Xs_tr, Xs_te)
        m = compute_extended_oos_metrics(model, oos, y_test, quantile_levels=(0.95, 0.98),
                                          twcrps_levels=(0.95, 0.98), brier_levels=(), n_draws=500, seed=42)
        result = {"rmse": m.get("rmse"), "crps_mean": m.get("crps_mean"),
                  "twcrps_95": m.get("twcrps_95"), "twcrps_98": m.get("twcrps_98"),
                  "grad_norm_inf_numeric": float(np.max(np.abs(g_res["grad"]))),
                  "grad_converged": g_res["converged"], "grad_eps_used": g_res["eps_used"],
                  "loglik": fit_res["loglik"], "n_params": len(theta),
                  "runtime_s": time.time() - t0}
        out[branch] = _save(out_dir, result, force=force)
        print(f"    stage4_structural/{branch}: rmse={result['rmse']:.3f} crps={result['crps_mean']:.3f} "
              f"grad={result['grad_norm_inf_numeric']:.3f} conv={result['grad_converged']}")
    return out


def run_lag366_ablation(location: str, short: str, force: bool) -> dict:
    loader = StationDataLoader.from_registry(location, precip_dir=R.PRECIP_DIR, era5_dir=R.ERA5_DIR, enso_path=R.ENSO_PATH)
    data = loader.load_all()
    y_train, y_test = data["y_train"], data["y_test"]
    bt, bv, cn = data["covariate_blocks_train"], data["covariate_blocks_test"], data["covariate_col_names"]
    gas_lags = [1, 365]
    out = {"stage1": {}, "stage2": {}}

    branches = {"phi_only": ["gamma", "zeta", "xi"], "phi_xi": ["gamma", "zeta"]}
    for branch, static_params in branches.items():
        cfg_id = f"lag1_365_{branch}"
        out_dir = OUT_DIR / short / "stage1" / cfg_id
        cache_p = out_dir / "result.json"
        if cache_p.exists() and not force:
            out["stage1"][cfg_id] = json.loads(cache_p.read_text())
        else:
            model = ZAGASModel(distribution=GB2LogLink(), pi_dynamics=make_pi_dynamics("ar_logistic", seasonal="daily"),
                                seasonal="daily", gas_lags=gas_lags, scaling="diagonal_inverse_fisher",
                                static_params=static_params)
            fit_res = model.fit(y=y_train, verbose=False)
            theta = fit_res["theta"]
            obj_fn = lambda th: model._run_filter(th, y_train, return_paths=False)
            g_res = numerical_gradient_checked(obj_fn, theta)
            oos = model.simulate_oos(theta, y_train, y_test)
            m = compute_extended_oos_metrics(model, oos, y_test, quantile_levels=(0.95, 0.98),
                                              twcrps_levels=(0.95, 0.98), brier_levels=(), n_draws=500, seed=42)
            result = {"rmse": m.get("rmse"), "crps_mean": m.get("crps_mean"),
                      "twcrps_95": m.get("twcrps_95"), "twcrps_98": m.get("twcrps_98"),
                      "grad_norm_inf_numeric": float(np.max(np.abs(g_res["grad"]))),
                      "grad_converged": g_res["converged"], "grad_eps_used": g_res["eps_used"]}
            out["stage1"][cfg_id] = _save(out_dir, result, force=force)
        print(f"    stage1/{cfg_id}: rmse={out['stage1'][cfg_id]['rmse']:.3f} "
              f"grad={out['stage1'][cfg_id]['grad_norm_inf_numeric']:.3f}")

    dt_names = cn["dewtemp_short"]
    i_dew1 = dt_names.index("dewpoint_lag1")
    i_temp1 = dt_names.index("temperature_lag1")
    enso_cur, enso_cur_te = bt["enso_dummy_current"], bv["enso_dummy_current"]

    def build_X(n_cov, split="train"):
        b, e = (bt, enso_cur) if split == "train" else (bv, enso_cur_te)
        cols = [b["dewtemp_short"][:, [i_dew1]]]
        names = ["dewpoint_lag1"]
        if n_cov >= 2:
            cols.append(b["dewtemp_short"][:, [i_temp1]]); names.append("temperature_lag1")
        if n_cov >= 3:
            cols.append(e); names += ["el_nino_t", "la_nina_t"]
        return np.concatenate(cols, axis=1), names

    for branch, static_params in branches.items():
        for n_cov in (1, 2, 3):
            cfg_id = f"lag1_365_{branch}_{n_cov}cov"
            out_dir = OUT_DIR / short / "stage2" / cfg_id
            cache_p = out_dir / "result.json"
            if cache_p.exists() and not force:
                out["stage2"][cfg_id] = json.loads(cache_p.read_text())
            else:
                X_tr, names = build_X(n_cov, "train")
                X_te, _ = build_X(n_cov, "test")
                model = CovZAGASModel(distribution=GB2LogLink(), pi_dynamics=make_pi_dynamics("ar_logistic", seasonal="daily"),
                                       seasonal="daily", gas_lags=gas_lags, scaling="diagonal_inverse_fisher",
                                       cov_names=names, cov_target=None if branch == "phi_xi" else ["phi"],
                                       static_params=static_params)
                fit_res = model.fit(y=y_train, X=X_tr, verbose=False)
                theta = fit_res["theta"]
                obj_fn = lambda th: model._run_filter(th, y_train, X_tr, return_paths=False)
                g_res = numerical_gradient_checked(obj_fn, theta)
                oos = model.simulate_oos(theta, y_train, y_test, X_train=X_tr, X_test=X_te)
                m = compute_extended_oos_metrics(model, oos, y_test, quantile_levels=(0.95, 0.98),
                                                  twcrps_levels=(0.95, 0.98), brier_levels=(), n_draws=500, seed=42)
                result = {"rmse": m.get("rmse"), "crps_mean": m.get("crps_mean"),
                          "twcrps_95": m.get("twcrps_95"), "twcrps_98": m.get("twcrps_98"),
                          "grad_norm_inf_numeric": float(np.max(np.abs(g_res["grad"]))),
                          "grad_converged": g_res["converged"], "grad_eps_used": g_res["eps_used"]}
                out["stage2"][cfg_id] = _save(out_dir, result, force=force)
            print(f"    stage2/{cfg_id}: rmse={out['stage2'][cfg_id]['rmse']:.3f} "
                  f"grad={out['stage2'][cfg_id]['grad_norm_inf_numeric']:.3f}")
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
        s4 = run_stage4_structural(location, short, args.force)
        lag = run_lag366_ablation(location, short, args.force)
        summary[short] = {"stage4_structural": s4, **lag}

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
