"""
Collect the per-config metrics needed for the 2026-09-11 stage-comparison
report (generate_stage_comparison_report.py): for every (location, stage,
config, branch) cell involved, reconstruct the model at its cached theta
(NO refitting) and record:

    grad_inf     : checked gradient inf-norm (recomputed here as a full
                   vector, not just read from result.json, because the
                   selection rule needs PER-PARAMETER values)
    frac_conv    : fraction of parameters with |grad_i| / n_train <= 1e-3
                   ("gradient per observation" -- the user's stated
                   tie-break ahead of CRPS/RMSE)
    rmse_oos/is, crps_oos/is, tw95, tw98 : straight from result.json

Sources:
    stage1, stage2, stage4S  -> artifacts/rerun_10092026/            (unaffected by the phi_L fix)
    stage3, stage4, stage5   -> artifacts/rerun_10092026_phiL_sigmoid/ (the fixed, sigmoid-bounded phi_L parameterisation)

Writes artifacts/stage_comparison_20260911/metrics.json,
keyed metrics[loc_short][stage][cfg][branch].
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_rerun_10092026 as D
import run_stage34_phiL_sigmoid as S
from data.station_loader import STATION_REGISTRY
from models.za_gas_model import ZAGASModel
from models.cov_gas_model import CovZAGASModel
from models.harvey_gas import HarveyZAGASModel
from distributions.gb2_log_link import GB2LogLink
from diagnostics.numerical_diff import numerical_gradient_checked

OUT = ROOT / "artifacts" / "stage_comparison_20260911"
TOL = 1e-3  # per-observation gradient convergence tolerance

STAGE2_COV = ["dew", "dew_temp", "dew_enso", "dew_temp_enso"]
HARVEY_COV5 = ["none", "dew", "dew_temp", "dew_enso", "dew_temp_enso"]


def _tree_for(stage):
    return D.OUT if stage in ("stage1", "stage2", "stage4S") else S.OUT


def _short(loc):
    return STATION_REGISTRY[loc].get("short", loc)


def build_model(loc_full, stage, cfg, branch, res, data):
    scaling = res.get("scaling", "unit")
    if stage == "stage1":
        lags = D.GAS_LAG_SETS[res["lag_key"]]
        model = ZAGASModel(distribution=D._dist(branch), pi_dynamics=D.pi_dyn(), seasonal="daily",
                           gas_lags=lags, scaling=scaling, static_params=D._static(branch))
        return model, {}
    if stage == "stage2":
        lags = D.GAS_LAG_SETS[res["lag_key"]]
        Xtr, cn = D._cov_cols(data, res["cov"], "train")
        model = CovZAGASModel(distribution=D._dist(branch), pi_dynamics=D.pi_dyn(), seasonal="daily",
                              gas_lags=lags, scaling=scaling, cov_names=cn,
                              cov_target=["phi"] if branch == "phi_only" else None,
                              static_params=D._static(branch))
        return model, {"X": Xtr}
    # stage3 / stage4 / stage4S / stage5 -- Harvey
    if stage == "stage5":
        base = res.get("base_stage", "stage4"); cov = res["cov"]
        dist, static, harvey_target = GB2LogLink(), ["gamma", "zeta"], ["phi", "xi"]
    else:
        base = stage; cov = res["cov"]
        dist, static, harvey_target = D._dist(branch), D._static(branch), D._htarget(branch)
    harm = 0 if base == "stage3" else D.N_HARM
    damp = None if base == "stage3" else D.DAMP
    incl_long = (base != "stage4S")
    Xl_tr, ln, Xs_tr, sn = D._harvey_cov(data, cov, "train")
    model = HarveyZAGASModel(
        distribution=dist, pi_dynamics=D.pi_dyn(), seasonal="daily",
        long_names=ln, short_names=sn, harvey_target=harvey_target,
        scaling=scaling, static_params=static,
        n_harmonics=harm, seasonal_damping_max=damp, include_long=incl_long,
        phi_L_max=(1.0 if incl_long else None))  # every long-block fit used here is sigmoid-bounded
    return model, {"X_long": Xl_tr, "X_short": Xs_tr}


def gather_one(loc_full, stage, cfg, branch):
    short = _short(loc_full)
    tree = _tree_for(stage)
    # Stage 5 is saved as a single dir with no branch suffix (it always
    # fits phi AND xi time-varying, so there is no separate phi_only cache).
    sfx = "" if (stage == "stage5" or branch == "phi_only") else "__xi"
    cdir = tree / short / stage / (cfg + sfx)
    rp, tp = cdir / "result.json", cdir / "theta.npy"
    if not (rp.exists() and tp.exists()):
        return None
    res = json.loads(rp.read_text())
    theta = np.load(tp)
    data = D._load_data(loc_full)
    y_tr = data["y_train"]
    try:
        model, filt_kwargs = build_model(loc_full, stage, cfg, branch, res, data)
        obj = lambda th: model._run_filter(th, y_tr, **filt_kwargs, return_paths=False)
        g = numerical_gradient_checked(obj, theta)
        grad = np.asarray(g["grad"], dtype=float)
        n_train = len(y_tr)
        frac_conv = float(np.mean(np.abs(grad) / n_train <= TOL))
        return {
            "grad_inf": float(np.max(np.abs(grad))),
            "frac_conv": frac_conv,
            "n_params": len(theta),
            "grad_converged": bool(g["converged"]),
            "rmse_oos": res.get("rmse"), "rmse_is": res.get("rmse_insample"),
            "crps_oos": res.get("crps_mean"), "crps_is": res.get("crps_insample"),
            "tw95": res.get("twcrps_95"), "tw98": res.get("twcrps_98"),
            "validity": res.get("validity"), "success": res.get("success"),
        }
    except Exception:
        return {"error": traceback.format_exc()}


def _stage1_cfg_dir_name(short, branch, lag_key):
    """Stage 1 is cached per (lag x chosen Fisher scaling); resolve to the
    concrete cfg-dir suffix (without lag prefix, i.e. just '_<scaling>')."""
    sfx = "" if branch == "phi_only" else "__xi"
    p = D.OUT / short / "stage1" / f"_scaling_choice{sfx}.json"
    if not p.exists():
        return None
    per_lag = json.loads(p.read_text()).get("per_lag", {})
    sc = per_lag.get(lag_key, {}).get("scaling")
    return f"{lag_key}_{sc}" if sc else None


def _task(a):
    loc, stage, cfg, branch = a
    short = _short(loc)
    try:
        if stage == "stage1":
            real_cfg = _stage1_cfg_dir_name(short, branch, cfg)
            if real_cfg is None:
                return (loc, stage, cfg, branch, None)
            r = gather_one(loc, stage, real_cfg, branch)
        else:
            r = gather_one(loc, stage, cfg, branch)
        return (loc, stage, cfg, branch, r)
    except Exception:
        return (loc, stage, cfg, branch, {"error": traceback.format_exc()})


def build_tasks():
    tasks = []
    branches = ("phi_only", "phi_xi")
    for loc in D.LOCATIONS:
        for br in branches:
            for lk in D.GAS_LAG_SETS:
                tasks.append((loc, "stage1", lk, br))
            for lk in D.GAS_LAG_SETS:
                for cov in STAGE2_COV:
                    tasks.append((loc, "stage2", f"{lk}_{cov}", br))
            for st in ("stage3", "stage4", "stage4S"):
                for cov in HARVEY_COV5:
                    tasks.append((loc, st, cov, br))
        tasks.append((loc, "stage5", "best", "phi_xi"))
    return tasks


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks()
    print(f"{len(tasks)} metric cells to gather")
    metrics = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_task, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            loc, stage, cfg, branch, r = fut.result()
            short = _short(loc)
            metrics.setdefault(short, {}).setdefault(stage, {}).setdefault(cfg, {})[branch] = r
            if i % 50 == 0 or r is None or (isinstance(r, dict) and r.get("error")):
                print(f"  [{i}/{len(tasks)}] {short}/{stage}/{cfg}/{branch} -> "
                      f"{'MISSING' if r is None else ('ERR' if r.get('error') else 'ok')}")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print("wrote", OUT / "metrics.json")


if __name__ == "__main__":
    main()
