"""
Numba vs pure-Python parity for every ZA-GAS model family with the
2026-10-01 phi-linked joint occurrence dynamics and the mixed inverse-Fisher
scalings (models/mixed_score.py <-> models/_nb_gb2_filter.py).

For ZAGASModel / CovZAGASModel / HarveyZAGASModel (long-short, + seasonal
harmonics, structural), both branches (phi-only, phi+xi), scalings
{unit, mixed_inverse_fisher, mixed_diagonal_inverse_fisher}, on real data:

  1. penalised objective: Numba kernel vs Python loop;
  2. one-step OOS paths: simulate_oos Numba route vs _simulate_oos_python;
  3. multi-step simulator replay (diagnostics/multistep_forecast.py) vs the
     Numba OOS path -- the replay shares its per-step update with the
     forward simulation, so this validates the simulator too.

An AR-logistic control (no phi link) under the historical scalings checks
that the Numba simulate_oos route reproduces the old Python recursion.

Run:  python tests/test_pi_linked_all_models.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_rerun_10092026 as D
import models.cov_gas_model as CM
import models.harvey_gas as HM
import models.za_gas_model as ZM
from distributions import GB2LogLinkPhiOnly
from distributions.gb2_log_link import GB2LogLink
from models.cov_gas_model import CovZAGASModel
from models.harvey_gas import HarveyZAGASModel
from models.za_gas_model import ZAGASModel
from pi_dynamics.ar_logistic_custom_lags import ARLogisticCustomLagsPiDynamics
from pi_dynamics.phi_linked import PhiLinkedJointPiDynamics
from diagnostics import multistep_forecast as MF

LOC = "MANAUS"
TOL = 1e-7


def dist(branch):
    return GB2LogLinkPhiOnly() if branch == "phi_only" else GB2LogLink()


def static(branch):
    return ["xi", "gamma", "zeta"] if branch == "phi_only" else ["gamma", "zeta"]


def theta_for(model, y, **kw):
    th = model.initial_theta(y, **kw) if kw else model.initial_theta(y)
    th = np.asarray(th, float)
    rng = np.random.default_rng(7)
    th = th + 0.01 * rng.standard_normal(len(th))
    names = model.parameter_names()
    for i, n in enumerate(names):   # keep the (arbitrary) test theta non-explosive over train+test
        if "kappa" in n:
            th[i] *= 0.1
    if "lambda1" in names:
        th[names.index("lambda1")] = 0.3
    return th


def build(kind, branch, scaling, pi, data):
    if kind == "za":
        return ZAGASModel(distribution=dist(branch), pi_dynamics=pi, seasonal="daily", gas_lags=[1, 2, 3],
                          scaling=scaling, static_params=static(branch)), {}, {}
    if kind == "cov":
        Xtr, cn = D._cov_cols(data, "dew_temp", "train")
        Xte, _ = D._cov_cols(data, "dew_temp", "test")
        m = CovZAGASModel(distribution=dist(branch), pi_dynamics=pi, seasonal="daily", gas_lags=[1],
                          scaling=scaling, cov_names=cn, cov_target=["phi"] if branch == "phi_only" else None,
                          static_params=static(branch))
        return m, {"X": Xtr}, {"X_train": Xtr, "X_test": Xte}
    stage = kind
    Xl_tr, ln, Xs_tr, sn = D._harvey_cov(data, "dew_enso", "train")
    Xl_te, _, Xs_te, _ = D._harvey_cov(data, "dew_enso", "test")
    incl_long = stage != "stage4S"
    m = HarveyZAGASModel(distribution=dist(branch), pi_dynamics=pi, seasonal="daily",
                         long_names=ln, short_names=sn, harvey_target=D._htarget(branch),
                         scaling=scaling, static_params=static(branch),
                         n_harmonics=0 if stage == "stage3" else D.N_HARM,
                         seasonal_damping_max=None if stage == "stage3" else D.DAMP,
                         include_long=incl_long, phi_L_max=1.0 if incl_long else None)
    return m, {"X_long": Xl_tr, "X_short": Xs_tr}, {"X_long_train": Xl_tr, "X_long_test": Xl_te,
                                                     "X_short_train": Xs_tr, "X_short_test": Xs_te}


def objective(model, kind, theta, y, fk, numba: bool):
    flags = {"za": (ZM, "_NB_AVAIL"), "cov": (CM, "_NB_AVAIL_COV")}.get(kind, (HM, "_NB_AVAIL_HARVEY"))
    mod, name = flags
    old = getattr(mod, name)
    setattr(mod, name, numba)
    try:
        return model._run_filter(theta, y, **fk) if fk else model._run_filter(theta, y)
    finally:
        setattr(mod, name, old)


def replay_paths(model, kind, theta, y_full, ok):
    if kind == "za":
        r = MF._base_replay(model, theta, y_full)
    elif kind == "cov":
        r = MF._cov_replay(model, theta, y_full, np.vstack([ok["X_train"], ok["X_test"]]))
    else:
        r = MF._harvey_replay(model, theta, y_full, np.vstack([ok["X_long_train"], ok["X_long_test"]]),
                              np.vstack([ok["X_short_train"], ok["X_short_test"]]))
    return r["f_arr"], r["pi_arr"]


def check(kind, branch, scaling, pi_factory, data, label):
    y_tr, y_te = data["y_train"], data["y_test"]
    model, fk, ok = build(kind, branch, scaling, pi_factory(), data)
    theta = theta_for(model, y_tr)
    o_nb = objective(model, kind, theta, y_tr, fk, True)
    o_py = objective(model, kind, theta, y_tr, fk, False)
    a = model.simulate_oos(theta, y_tr, y_te, **ok)
    b = model.simulate_oos(theta, y_tr, y_te, use_numba=False, **ok)
    df = float(np.max(np.abs(a["f_arr_oos"] - b["f_arr_oos"])))
    dp = float(np.max(np.abs(a["pi_oos"] - b["pi_oos"])))
    y_full = np.concatenate([y_tr, y_te])
    f_r, pi_r = replay_paths(model, kind, theta, y_full, ok)
    T = len(y_tr)
    dr = float(np.max(np.abs(f_r[T:T + len(y_te)] - a["f_arr_oos"])))
    dpr = float(np.max(np.abs(pi_r[T:T + len(y_te)] - a["pi_oos"])))
    dobj = abs(o_nb - o_py) / max(1.0, abs(o_py))
    ok_all = dobj < TOL and df < 1e-6 and dp < 1e-9 and dr < 1e-6 and dpr < 1e-9
    print(f"{'OK ' if ok_all else 'BAD'} {label:8s} {kind:7s} {branch:8s} {scaling:30s} "
          f"obj {dobj:.1e}  oos f {df:.1e} pi {dp:.1e}  replay f {dr:.1e} pi {dpr:.1e}", flush=True)
    return ok_all


def main():
    data = D._load_data(LOC)
    D_pi = lambda: PhiLinkedJointPiDynamics()
    AR = lambda: ARLogisticCustomLagsPiDynamics(lags=[1, 2, 3], include_ar=True)
    results = []
    for kind in ("za", "cov", "stage3", "stage4", "stage4S"):
        for branch in ("phi_only", "phi_xi"):
            for sc in ("unit", "mixed_inverse_fisher", "mixed_diagonal_inverse_fisher"):
                results.append(check(kind, branch, sc, D_pi, data, "D"))
            for sc in ("diagonal_inverse_fisher", "inverse_fisher"):
                results.append(check(kind, branch, sc, AR, data, "AR ctrl"))
    print(f"\n{sum(results)}/{len(results)} configurations agree")
    assert all(results)


if __name__ == "__main__":
    main()
