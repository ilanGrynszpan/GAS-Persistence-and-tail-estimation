"""
Full staged rerun with the phi-linked joint occurrence dynamics (2026-10-01).

WHAT CHANGES RELATIVE TO THE 2026-09 STAGED RERUN
    run_rerun_10092026.py (+ run_stage34_phiL_sigmoid.py for Stages 3/4)
    used pi_t from a logistic AR(1) on y-lags {1,2,3}. Here every model of
    every stage uses alternative D of the pi-dynamics comparison
    (reports/newpi), estimated JOINTLY with the magnitude model:

        pi_{t|t-1} = logistic(lambda0 + lambda1 * phi_{t|t-1}),

    phi_{t|t-1} being the model's own magnitude state (GAS(p,q) state,
    covariate GAS state, or the Harvey L + S [+ seasonal] sum). Because pi
    depends on phi, the phi score is that of the zero-adjusted density,
        s_phi = 1{y>0} dlog g/dphi + lambda1 (1{y>0} - pi_t),
    and the Fisher scalings are those of the mixed density
    (models/mixed_score.py):
        unit, mixed_diagonal_inverse_fisher, mixed_inverse_fisher
    (for one time-varying parameter the last two coincide). Since pi enters
    the score, the expected value pi_t E[G_t] and every derived metric
    change with it.

    Everything else is unchanged: same data and train/test split, same lag
    sets, covariate sets and Harvey configurations, same Stage 1 scaling
    choice (gradient tier, then AIC) inherited by Stages 2-4S, same
    sigmoid-bounded phi_L (phi_L_max = 1) for Stages 3/4, same evaluation
    (run_rerun_10092026._evaluate: rolling 1-step OOS metrics, BSS,
    exceedance/Kupiec, multi-step, in-sample QR). Stage 5 (phi+xi Harvey on
    the best Stage 3/4/4S configuration) is rerun as well.

    STARTING VALUES. Every fit starts from the time-invariant ZA-GB2 MLE
    (constant pi and GB2 parameters; see static_start), and Stages 4/4S/5 run
    BFGS to 1500 iterations with the polish restart (the 2026-09 runs capped
    them at 300 to save time).

EXECUTION (dependency-ordered phases, one process pool per phase)
    phase A  Stage 1: 6 loc x 2 branch x 4 lag sets x 3 scalings
             -> pick scaling per (loc, branch)
    phase B  Stage 2: 6 x 2 x 4 lag sets x 4 covariate sets
             Stages 3/4/4S: 6 x 2 x 3 x 6 Harvey covariate configs
    phase C  Stage 5: per location, phi+xi Harvey on the best Stage 3/4/4S
             configuration (gradient tier -> CRPS -> RMSE), phi-only scaling
    Every fit is a joint ZA-GAS maximum-likelihood fit (BFGS) on the
    training series, through the Numba kernels (parity:
    tests/test_pi_linked_all_models.py). Each result.json additionally
    stores the checked gradient vector and the per-observation
    convergence fraction used by the selection rule.

ARTIFACTS
    artifacts/rerun_pi_01102026/<short>/<stage>/<cfg>[__xi]/
        result.json, theta.npy, multistep.csv, insample_qr.npy

USAGE
    python run_rerun_pi_01102026.py --workers 22 [--force] [--only manaus,bh]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_rerun_10092026 as D
from data.station_loader import STATION_REGISTRY
from models.za_gas_model import ZAGASModel
from models.cov_gas_model import CovZAGASModel
from models.harvey_gas import HarveyZAGASModel
from pi_dynamics.phi_linked import PhiLinkedJointPiDynamics
from diagnostics.numerical_diff import numerical_gradient_checked
from distributions.gb2_log_link import GB2LogLink
import collect_stage_comparison_metrics as C

OUT = ROOT / "artifacts" / "rerun_pi_01102026"
SCALINGS = ["unit", "mixed_diagonal_inverse_fisher", "mixed_inverse_fisher"]
GAS_LAG_SETS = D.GAS_LAG_SETS
STAGE2_COV = ["dew", "dew_temp", "dew_enso", "dew_temp_enso"]
HARVEY_COV = ["none", "enso", "dew", "dew_temp", "dew_enso", "dew_temp_enso"]
HARVEY_STAGES = ("stage3", "stage4", "stage4S")
BRANCHES = ("phi_only", "phi_xi")
PHI_L_MAX = 1.0
TOL = 1e-3   # per-observation gradient tolerance (frac_conv), as in the 2026-09 selection


def pi_dyn():
    return PhiLinkedJointPiDynamics()


def _short(loc):
    return STATION_REGISTRY[loc].get("short", loc)


def _sfx(branch):
    return D._sfx(branch)


def build_model(stage, branch, scaling, cfg, data):
    """(model, filt_kwargs, oos_kwargs) for one configuration.
    cfg = lag key (stage1), '<lag>_<cov>' (stage2) or Harvey covariate config."""
    if stage == "stage1":
        return (ZAGASModel(distribution=D._dist(branch), pi_dynamics=pi_dyn(), seasonal="daily",
                           gas_lags=GAS_LAG_SETS[cfg], scaling=scaling, static_params=D._static(branch)),
                {}, {})
    if stage == "stage2":
        lag_key = next(k for k in sorted(GAS_LAG_SETS, key=len, reverse=True) if cfg.startswith(k + "_"))
        cset = cfg[len(lag_key) + 1:]
        Xtr, names = D._cov_cols(data, cset, "train")
        Xte, _ = D._cov_cols(data, cset, "test")
        m = CovZAGASModel(distribution=D._dist(branch), pi_dynamics=pi_dyn(), seasonal="daily",
                          gas_lags=GAS_LAG_SETS[lag_key], scaling=scaling, cov_names=names,
                          cov_target=["phi"] if branch == "phi_only" else None,
                          static_params=D._static(branch))
        return m, {"X": Xtr}, {"X_train": Xtr, "X_test": Xte}
    base = stage
    if stage == "stage5":            # cfg = "<base stage>/<harvey cfg>"
        base, cfg = cfg.split("/")
        branch = "phi_xi"
    Xl_tr, ln, Xs_tr, sn = D._harvey_cov(data, cfg, "train")
    Xl_te, _, Xs_te, _ = D._harvey_cov(data, cfg, "test")
    incl_long = base != "stage4S"
    m = HarveyZAGASModel(
        distribution=GB2LogLink() if stage == "stage5" else D._dist(branch),
        pi_dynamics=pi_dyn(), seasonal="daily",
        long_names=ln, short_names=sn, harvey_target=D._htarget(branch),
        scaling=scaling, static_params=D._static(branch),
        n_harmonics=0 if base == "stage3" else D.N_HARM,
        seasonal_damping_max=None if base == "stage3" else D.DAMP,
        include_long=incl_long, phi_L_max=PHI_L_MAX if incl_long else None)
    ok = {"X_long_train": Xl_tr, "X_long_test": Xl_te, "X_short_train": Xs_tr, "X_short_test": Xs_te}
    return m, {"X_long": Xl_tr, "X_short": Xs_tr}, ok


def static_start(model, stage, y):
    """
    Starting theta from the time-invariant ZA-GB2 MLE (all parameters constant).

    With constant parameters the ZA-GB2 log-likelihood separates into
        sum 1{y=0} log(1-pi) + 1{y>0} log pi      ->  pi_hat = share of wet days
        sum_{y>0} log g(y; phi, xi, gamma, zeta)  ->  static GB2 MLE on wet days
    (GASFilter.fit_unconditional). The dynamic model starts at that point:
    lambda0 = logit(pi_hat), lambda1 = 0, the magnitude state(s) start at the
    static phi (and xi), static parameters at their MLE, and the dynamics
    loadings at the models' usual small defaults. ZAGASModel/CovZAGASModel's
    initial_theta already does exactly this; HarveyZAGASModel's default uses
    fixed gamma=1, zeta=3 and phi=log(mean y), so it is seeded here instead.
    """
    if stage in ("stage1", "stage2"):
        return np.asarray(model.initial_theta(y), float), "time-invariant ZA-GB2 MLE (initial_theta)"
    from models.gas_filter import GASFilter
    est = GASFilter(distribution=model.dist, static_params=model.static_names).fit_unconditional(
        np.asarray(y)[np.asarray(y) > 0])
    if not est:
        return np.asarray(model.initial_theta(y), float), "Harvey default start (static MLE failed)"
    static_ov = {k: float(est[k]) for k in model.static_names if k in est}
    over = {j: {"omega": float(est[j])} for j in model.harvey_target if j in est}
    theta = model.initial_theta(y, overrides=over, static_overrides=static_ov)
    return np.asarray(theta, float), "time-invariant ZA-GB2 MLE (" + ", ".join(
        f"{k}={v:.3f}" for k, v in est.items()) + ")"


def loading_scale(model, theta, y):
    """
    Per time-varying parameter j, sd(unit score_j) / sd(scaled score_j) at the
    time-invariant start (constant phi, xi, pi; lambda1 = 0). Multiplying the
    default score loadings (A_j*, kappa_*_j) by it makes the first state
    increments of a Fisher-scaled model the same size as those of the
    (stable) unit-scaled start; the defaults are calibrated for unit-sized
    scores and explode e.g. for seasonal Harvey models under the mixed
    inverse Fisher of two parameters.
    """
    from models import mixed_score as MS
    scaling = getattr(model, "scaling", None) or model.gas.scaling
    tv = list(getattr(model, "tv_names", None) or model.gas.tv_names)
    if scaling == "unit":
        return {j: 1.0 for j in tv}
    est = static_values(model, y)
    pi_c = float(np.mean(np.asarray(y) > 0))
    call = {j: est[j] for j in tv}
    call.update({k: est[k] for k in _static_names(model) if k in est})
    wet = np.asarray(y)[np.asarray(y) > 0]
    raw = np.array([[model.dist.score(v, **call)[j] for j in tv] for v in wet])
    sc = np.array([[MS.scaled_score(model.dist, tv, scaling, v, call, pi_c, 0.0)[j] for j in tv]
                   for v in wet])
    out = {}
    for k, j in enumerate(tv):
        a, b = raw[:, k].std(), sc[:, k].std()
        out[j] = float(a / b) if (b > 0 and np.isfinite(a / b)) else 1.0
    return out


def _static_names(model):
    return list(getattr(model, "static_names", None) or model.gas.static_names)


def static_values(model, y):
    from models.gas_filter import GASFilter
    est = GASFilter(distribution=model.dist, static_params=_static_names(model)).fit_unconditional(
        np.asarray(y)[np.asarray(y) > 0])
    return est


def _fit(model, stage, y, fk, theta0):
    if stage == "stage1":
        return model.fit(y, verbose=False, theta0=theta0)
    if stage == "stage2":
        return model.fit(y=y, X=fk["X"], verbose=False, theta0=theta0)
    opts = {"options": {"maxiter": 1500}, "polish": True} if stage in ("stage4", "stage4S", "stage5") else {}
    return model.fit(y=y, X_long=fk["X_long"], X_short=fk["X_short"], verbose=False, theta0=theta0, **opts)


def cfg_dir(short, stage, dirname, branch):
    if stage == "stage5":            # always phi+xi; no branch suffix (as in the 2026-09 tree)
        return OUT / short / stage / dirname
    return OUT / short / stage / (dirname + _sfx(branch))


def run_one(stage, loc, branch, cfg, scaling, dirname, force):
    short = _short(loc)
    cdir = cfg_dir(short, stage, dirname, branch)
    tag = f"{stage} {short}/{branch}/{dirname}"
    if (cdir / "result.json").exists() and not force:
        return ("skip", tag)
    try:
        data = D._load_data(loc)
        y = data["y_train"]
        model, fk, ok = build_model(stage, branch, scaling, cfg, data)
        theta0, start_note = static_start(model, stage, y)
        fac = loading_scale(model, theta0, y)
        names_ = model.parameter_names()
        for i, nm in enumerate(names_):
            if nm.startswith("A_") or nm.startswith("kappa_"):
                j = "xi" if nm.endswith("_xi") or "_xi_" in nm else "phi"
                theta0[i] *= fac.get(j, 1.0)
        start_note += "; loadings x " + ", ".join(f"{j}:{v:.3g}" for j, v in fac.items())
        fr = _fit(model, stage, y, fk, theta0)
        try:
            res, ms, qr, th = D._evaluate(model, fr["theta"], data, fk, ok, ok, fr)
        except ValueError as exc:
            # Diverged out-of-sample path (non-finite pi/state): record it as a
            # "div" candidate (non-finite OOS metrics), as the 2026-09 tables do,
            # keeping the fit; the selection rule ranks it last.
            import pandas as pd
            ll = fr.get("loglik")
            res = {"rmse": float("nan"), "crps_mean": float("nan"), "twcrps_95": float("nan"),
                   "twcrps_98": float("nan"), "rmse_insample": None, "crps_insample": None,
                   "loglik": ll, "n_params": len(fr["theta"]),
                   "aic": (2 * len(fr["theta"]) - 2 * ll) if ll is not None else None,
                   "success": fr.get("success"), "validity": fr.get("validity"),
                   "oos_error": f"diverged out-of-sample path: {exc}"}
            ms, qr, th = pd.DataFrame({"error": [str(exc)]}), np.array([]), fr["theta"]
        g = numerical_gradient_checked(
            lambda t: model._run_filter(t, y, **fk, return_paths=False) if fk
            else model._run_filter(t, y, return_paths=False), np.asarray(fr["theta"], float))
        grad = np.asarray(g["grad"], float)
        res["grad_norm_inf_numeric"] = float(np.max(np.abs(grad))) if "grad_norm_inf_numeric" not in res             else res["grad_norm_inf_numeric"]
        res.update({"scaling": scaling, "branch": branch, "stage": stage, "pi_dynamics": "phi_linked_joint",
                    "start": start_note,
                    "grad_vec": grad.tolist(), "param_names": model.parameter_names(),
                    "frac_conv": float(np.mean(np.abs(grad) / len(y) <= TOL))})
        if stage == "stage1":
            res["lag_key"] = cfg
        elif stage == "stage2":
            lag_key = next(k for k in sorted(GAS_LAG_SETS, key=len, reverse=True)
                           if cfg.startswith(k + "_"))
            res["lag_key"], res["cov"] = lag_key, cfg[len(lag_key) + 1:]
        elif stage == "stage5":
            res["base_stage"], res["cov"] = cfg.split("/")
            res["base_branch"] = base_branch_of(_short(loc))
        else:
            res["cov"] = cfg
            if stage != "stage4S":
                res["phi_L_max"] = PHI_L_MAX
        D._save(cdir, res, ms, qr, th)
        return ("done", f"{tag} rmse={res['rmse']:.3f} crps={res['crps_mean']:.3f} "
                        f"grad={res['grad_norm_inf_numeric']:.2f}")
    except Exception:
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "fit_error.txt").write_text(traceback.format_exc())
        return ("err", f"{tag}\n{traceback.format_exc()}")


def _task(a):
    return run_one(*a)


def pick_scaling(short, branch):
    """Gradient tier then AIC, per lag set (run_rerun_10092026._pick_scaling);
    the chosen scaling is the one of the lowest-AIC lag set."""
    per_lag = {}
    for lk in GAS_LAG_SETS:
        cands = {}
        for sc in SCALINGS:
            r = D.load_json(cfg_dir(short, "stage1", f"{lk}_{sc}", branch) / "result.json")
            if r:
                cands[sc] = r
        if cands:
            best = D._pick_scaling(cands)
            per_lag[lk] = dict(cands[best], scaling=best)
    lk_best = min(per_lag, key=lambda k: per_lag[k].get("aic", 1e18))
    chosen = {"chosen_scaling": per_lag[lk_best]["scaling"], "best_lag_key": lk_best, "per_lag": per_lag}
    (OUT / short / "stage1" / ("_scaling_choice" + _sfx(branch) + ".json")).write_text(
        json.dumps(chosen, indent=2, default=float))
    return chosen["chosen_scaling"]


def _tier(r):
    if r.get("validity") == "failed":
        return 3
    g = r.get("grad_norm_inf_numeric", 1e18)
    return 0 if g < 1 else (1 if g < 100 else 2)


def select_stage5_base(short):
    """Best Stage 3/4/4S config over both branches: gradient tier -> CRPS -> RMSE
    (run_rerun_10092026._select_best_harvey's rule)."""
    cands = []
    for st in HARVEY_STAGES:
        for cfg in HARVEY_COV:
            for br in BRANCHES:
                r = D.load_json(cfg_dir(short, st, cfg, br) / "result.json")
                if r and np.isfinite(r.get("crps_mean", np.nan)):
                    cands.append((st, cfg, br, r))
    st, cfg, br, _ = min(cands, key=lambda x: (_tier(x[3]), x[3].get("crps_mean", 1e18),
                                               x[3].get("rmse", 1e18)))
    (OUT / short / "stage5").mkdir(parents=True, exist_ok=True)
    (OUT / short / "stage5" / "_base.json").write_text(json.dumps({"stage": st, "cov": cfg, "branch": br}))
    return st, cfg, br


def base_branch_of(short):
    return json.loads((OUT / short / "stage5" / "_base.json").read_text())["branch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    locs = [l for l in D.LOCATIONS if not args.only or _short(l) in args.only.split(",")]
    t0 = time.time()

    a = [("stage1", loc, br, lk, sc, f"{lk}_{sc}", args.force)
         for loc in locs for br in BRANCHES for lk in GAS_LAG_SETS for sc in SCALINGS]
    D._run_pool(a, _task, args.workers, "phase A / stage1")
    chosen = {(loc, br): pick_scaling(_short(loc), br) for loc in locs for br in BRANCHES}
    for (loc, br), sc in chosen.items():
        print(f"  scaling chosen {_short(loc)}/{br}: {sc}", flush=True)

    b = [("stage2", loc, br, f"{lk}_{cs}", chosen[(loc, br)], f"{lk}_{cs}", args.force)
         for loc in locs for br in BRANCHES for lk in GAS_LAG_SETS for cs in STAGE2_COV]
    b += [(st, loc, br, cfg, chosen[(loc, br)], cfg, args.force)
          for loc in locs for br in BRANCHES for st in HARVEY_STAGES for cfg in HARVEY_COV]
    D._run_pool(b, _task, args.workers, "phase B / stage2 + harvey")

    c = []
    for loc in locs:
        st, cfg, br = select_stage5_base(_short(loc))
        print(f"  stage5 base {_short(loc)}: {st}/{cfg}/{br}", flush=True)
        c.append(("stage5", loc, "phi_xi", f"{st}/{cfg}", chosen[(loc, "phi_only")], "best", args.force))
    D._run_pool(c, _task, min(args.workers, len(c)), "phase C / stage5")
    print(f"done in {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
