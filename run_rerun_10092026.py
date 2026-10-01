"""
Full staged rerun with a custom-lag AR occurrence link (2026-09-10).

=============================================================================
WHY THIS EXISTS
=============================================================================

User request 2026-09-10: re-run the staged pipeline with the occurrence
model changed to

    pi_{t|t-1} = 1 / (1 + exp(-eta_t)),                      (LOGISTIC link)
    eta_t      = omega0 + rho * eta_{t-1} + sum_{l=1}^{3} omega_l * y_{t-l}

i.e. the standard logistic/sigmoid link but with a SHORT lag set {1,2,3}
plus an AR(1) term in eta (not the historical {1,365,366}). Implemented as
ARLogisticCustomLagsPiDynamics(lags=[1,2,3], include_ar=True,
link="logistic"); the numba kernels carry a `pi_link` code
(models/_nb_gb2_filter.py `_pi_link_terms`, pi_link=0 == sigmoid) so every
fit here is numba-accelerated.

HISTORY: the first pass (link="reciprocal") implemented the formula
literally as pi = 1/(1+eta); that was not the intended link. Those
artifacts are preserved under artifacts/rerun_10092026_reciprocal/ and the
report under reports/rerun10092026_reciprocal/.

Run flat + parallel:  python run_rerun_10092026.py --flat --workers 22

=============================================================================
STAGES  (phi-only / single time-varying location parameter, stages 1-4)
=============================================================================

Stage 1  ZAGASModel, no covariates.  gas_lags in
         { [1], [1,2,3], [1,365,366], [1,2,3,365,366] } ("with / without
         seasonality").  All 3 Fisher scalings fit; the one used downstream
         is picked by gradient tier (<1, then <100, then any) then lowest
         AIC -- recorded per location.
Stage 2  CovZAGASModel, same 4 gas_lags, inheriting Stage 1's chosen
         scaling, x 3 covariate sets:
            A dew point (dewpoint_lag1)
            B + air temperature (temperature_lag1)
            C + ENSO dummies (el_nino_t, la_nina_t)
Stage 3  HarveyZAGASModel long-short, no harmonic. 5 covariate configs:
            none / ENSO only / dew only / dew+ENSO / dew+temp+ENSO
         ENSO always in X_long, dew/temp in X_short (lag 1).
Stage 4  Stage 3 + damped trigonometric seasonal (n_harmonics=5,
         seasonal_damping_max=0.999), same 5 covariate configs.
Stage 4S structural: HarveyZAGASModel(include_long=False) + harmonic,
         one level (S) + one seasonal, same 5 covariate configs.
Stage 5  frozen phi from whichever earlier config wins by gradient tier ->
         OOS CRPS -> OOS RMSE (majority across locations); regime-sensitive
         time-varying xi on top (RegimeXiOnlyModel).

Per (location, stage, config) the cache holds: fitted theta, all OOS
metrics (1-step: RMSE/CRPS/twCRPS95/98; multi-step h in
{1,2,5,10,30,60,90}), Brier skill score, independent checked gradient
norm, and the in-sample quantile residuals (for ACF/QQ mosaics).

    artifacts/rerun_10092026/<short>/<stage>/<config>/result.json
                                             .../theta.npy
                                             .../insample_qr.npy
                                             .../multistep.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_harmonic_seasonal_experiment as R
from data.station_loader import STATION_REGISTRY, StationDataLoader
from distributions.gb2_log_link import GB2LogLink
from distributions.gb2_phi_only import GB2LogLinkPhiOnly
from models.za_gas_model import ZAGASModel
from models.cov_gas_model import CovZAGASModel
from models.harvey_gas import HarveyZAGASModel
from pi_dynamics.ar_logistic_custom_lags import ARLogisticCustomLagsPiDynamics
from diagnostics.scoring import compute_extended_oos_metrics
from diagnostics.multistep_forecast import multistep_metrics
from diagnostics.numerical_diff import numerical_gradient_checked

LOCATIONS = ["BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
             "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR"]
OUT = ROOT / "artifacts" / "rerun_10092026"
SCALINGS = ["unit", "diagonal_inverse_fisher", "inverse_fisher"]
GAS_LAG_SETS = {"L1": [1], "L123": [1, 2, 3], "L1_365_366": [1, 365, 366],
                "L123_365_366": [1, 2, 3, 365, 366]}
HARVEY_COV = ["none", "enso", "dew", "dew_enso", "dew_temp_enso"]
HORIZONS = [1, 2, 5, 10, 30, 60, 90]
N_HARM = 5
DAMP = 0.999


def pi_dyn():
    # LOGISTIC link:  pi_{t|t-1} = 1 / (1 + exp(-eta_t))   (standard sigmoid),
    # eta_t = omega0 + rho*eta_{t-1} + sum_{l=1}^{3} omega_l * y_{t-l}.
    # (The 2026-09-10 first pass used link="reciprocal" = literal 1/(1+eta);
    #  that was wrong -- artifacts kept under artifacts/rerun_10092026_reciprocal/.)
    return ARLogisticCustomLagsPiDynamics(lags=[1, 2, 3], include_ar=True, link="logistic")


# ---------------------------------------------------------------------------
# metrics helpers
# ---------------------------------------------------------------------------

def _insample_qr(model, theta, y_train, filt_kwargs):
    """In-sample quantile residuals Phi^{-1}(PIT_t) over wet days (PIT via
    the model's own cdf_series + randomised zeros, matching the project's
    established PIT convention)."""
    try:
        from diagnostics.residuals import quantile_residuals
        cdf = np.asarray(model.cdf_series(theta, y_train, **filt_kwargs))
        y = np.asarray(y_train)
        if len(cdf) != len(y):          # cdf_series returns the effective sample
            y = y[len(y) - len(cdf):]
        rng = np.random.default_rng(0)
        qr = quantile_residuals(cdf, y, randomise_zeros=True, rng=rng)
        return np.asarray(qr, dtype=np.float64)
    except Exception:
        return np.array([], dtype=np.float64)


def _bss(pi_oos, y_test, y_train, dates_test, dates_train):
    """Brier skill score of pi_{t|t-1} vs a monthly-climatology reference."""
    obs = (np.asarray(y_test) > 0).astype(float)
    p = np.clip(np.asarray(pi_oos, dtype=float), 0.0, 1.0)
    n = min(len(obs), len(p))
    obs, p = obs[:n], p[:n]
    bs = float(np.mean((p - obs) ** 2))
    mo_tr = pd.DatetimeIndex(dates_train).month.values
    wet_tr = (np.asarray(y_train) > 0).astype(float)
    clim = {m: float(wet_tr[mo_tr == m].mean()) if (mo_tr == m).any() else float(wet_tr.mean())
            for m in range(1, 13)}
    mo_te = pd.DatetimeIndex(dates_test).month.values[:n]
    p_clim = np.array([clim[m] for m in mo_te])
    bs_clim = float(np.mean((p_clim - obs) ** 2))
    return {"BS": bs, "BS_clim": bs_clim,
            "BSS": (1.0 - bs / bs_clim) if bs_clim > 0 else float("nan")}


def _insample_rmse_crps(model, theta, y_train, filt_kwargs):
    """In-sample RMSE + CRPS on the effective training sample, via the same
    machinery used for OOS (so rolling-1-step OOS vs in-sample is a like-
    for-like comparison to check for over-fitting)."""
    try:
        p = model.filter(theta, y_train, **filt_kwargs)
        if not p:
            return None, None
        eff = int(p.get("eff_start", len(y_train) - len(p["pi"])))
        paths = {"f_arr_oos": np.asarray(p["f_arr"]), "pi_oos": np.asarray(p["pi"]),
                 "static": p["static"], "tv_names": list(p["tv_names"])}
        y_eff = np.asarray(y_train)[eff:eff + len(paths["pi_oos"])]
        mm = compute_extended_oos_metrics(model, paths, y_eff, quantile_levels=(0.95, 0.98),
                                           twcrps_levels=(), brier_levels=(), n_draws=500, seed=42)
        return mm.get("rmse"), mm.get("crps_mean")
    except Exception:
        return None, None


def _evaluate(model, theta, data, filt_kwargs, oos_kwargs, ms_covariates, fit_res):
    y_tr, y_te = data["y_train"], data["y_test"]
    oos = model.simulate_oos(theta, y_tr, y_te, **oos_kwargs)
    m = compute_extended_oos_metrics(model, oos, y_te, quantile_levels=(0.95, 0.98),
                                      twcrps_levels=(0.95, 0.98), brier_levels=(),
                                      n_draws=500, seed=42, return_raw=True)
    obj = lambda th: model._run_filter(th, y_tr, **filt_kwargs, return_paths=False)
    g = numerical_gradient_checked(obj, theta)
    bss = _bss(oos.get("pi_oos", np.full(len(y_te), np.nan)), y_te, y_tr,
               data["dates_test"], data["dates_train"])
    rmse_is, crps_is = _insample_rmse_crps(model, theta, y_tr, filt_kwargs)

    # 1-step OOS upper-tail exceedance + Kupiec at q = 0.95, 0.98
    from diagnostics.exceedance import exceedance_from_pit
    pit = np.asarray(m.get("_pit_values", []), dtype=float)
    exc = {}
    for q in (0.95, 0.98):
        e = exceedance_from_pit(pit, q)
        k = int(q * 100)
        exc[f"exc_{k}_1s"] = e["rate"]
        exc[f"kup{k}_1s_p"] = e["kupiec_p"]
        exc[f"kup{k}_1s_reject"] = (e["kupiec_p"] < 0.05) if np.isfinite(e["kupiec_p"]) else None

    res = {
        "rmse": m.get("rmse"), "crps_mean": m.get("crps_mean"),
        "rmse_insample": rmse_is, "crps_insample": crps_is,
        "twcrps_95": m.get("twcrps_95"), "twcrps_98": m.get("twcrps_98"),
        "grad_norm_inf_numeric": float(np.max(np.abs(g["grad"]))),
        "grad_converged": g["converged"], "grad_eps_used": g["eps_used"],
        "loglik": fit_res.get("loglik"), "aic": None, "n_params": len(theta),
        "success": fit_res.get("success"), "validity": fit_res.get("validity"),
        **bss, **exc,
    }
    if res["loglik"] is not None:
        res["aic"] = float(2 * len(theta) - 2 * res["loglik"])
    # multi-step (now also carries exc_95/exc_98/kup95_p/kup98_p per horizon)
    try:
        origins = list(range(0, len(y_te) - max(HORIZONS) - 1, 90))[:8] or [0]
        ms = multistep_metrics(model, theta, y_tr, y_te, ms_covariates,
                               origins=origins, horizons=HORIZONS, n_draws=200, seed=0,
                               exceedance_levels=(0.95, 0.98))
    except Exception as exc_:
        ms = pd.DataFrame({"error": [str(exc_)]})
    qr = _insample_qr(model, theta, y_tr, filt_kwargs)
    return res, ms, qr, theta


def _save(cfg_dir: Path, res, ms, qr, theta):
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "result.json").write_text(json.dumps(res, indent=2, default=float))
    ms.to_csv(cfg_dir / "multistep.csv", index=False)
    np.save(cfg_dir / "insample_qr.npy", qr)
    np.save(cfg_dir / "theta.npy", np.asarray(theta, dtype=float))


# ---------------------------------------------------------------------------
# covariate builders (single lag per variable)
# ---------------------------------------------------------------------------

def _cov_cols(data, which, split):
    bt = data[f"covariate_blocks_{split}"]
    cn = data["covariate_col_names"]
    dt = bt["dewtemp_short"]; dt_n = cn["dewtemp_short"]
    i_dew = dt_n.index("dewpoint_lag1")
    i_temp = dt_n.index("temperature_lag1")
    enso = bt["enso_dummy_current"]
    if which == "dew":
        return dt[:, [i_dew]], ["dewpoint_lag1"]
    if which == "dew_temp":
        return dt[:, [i_dew, i_temp]], ["dewpoint_lag1", "temperature_lag1"]
    if which == "dew_enso":
        return np.concatenate([dt[:, [i_dew]], enso], axis=1), \
               ["dewpoint_lag1", "el_nino_t", "la_nina_t"]
    if which == "dew_temp_enso":
        return np.concatenate([dt[:, [i_dew, i_temp]], enso], axis=1), \
               ["dewpoint_lag1", "temperature_lag1", "el_nino_t", "la_nina_t"]
    raise ValueError(which)


def _harvey_cov(data, cfg, split):
    """Return (X_long, long_names, X_short, short_names) for a Harvey config."""
    bt = data[f"covariate_blocks_{split}"]
    cn = data["covariate_col_names"]
    dt = bt["dewtemp_short"]; dt_n = cn["dewtemp_short"]
    i_dew = dt_n.index("dewpoint_lag1"); i_temp = dt_n.index("temperature_lag1")
    enso = bt["enso_dummy_current"]
    T = dt.shape[0]
    empty = np.zeros((T, 0))
    Xl, ln = empty, []
    Xs, sn = empty, []
    if cfg in ("enso", "dew_enso", "dew_temp_enso"):
        Xl, ln = enso, ["el_nino_t", "la_nina_t"]
    if cfg == "dew" or cfg == "dew_enso":
        Xs, sn = dt[:, [i_dew]], ["dewpoint_lag1"]
    if cfg in ("dew_temp", "dew_temp_enso"):
        Xs, sn = dt[:, [i_dew, i_temp]], ["dewpoint_lag1", "temperature_lag1"]
    return Xl, ln, Xs, sn


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def _pick_scaling(cands):
    """cands: {scaling: result_dict}. gradient tier then AIC."""
    def tier(r):
        if r.get("validity") == "failed":
            return 3
        g = r.get("grad_norm_inf_numeric", 1e18)
        return 0 if g < 1 else (1 if g < 100 else 2)
    best = min(cands.items(), key=lambda kv: (tier(kv[1]), kv[1].get("aic", 1e18)))
    return best[0]


# ---------------------------------------------------------------------------
# branch config: "phi_only" (1 TVP: varphi) and "phi_xi" (2 TVP: varphi, xi)
# ---------------------------------------------------------------------------
def _dist(branch):
    return GB2LogLinkPhiOnly() if branch == "phi_only" else GB2LogLink()

def _static(branch):
    return ["xi", "gamma", "zeta"] if branch == "phi_only" else ["gamma", "zeta"]

def _htarget(branch):
    return ["phi"] if branch == "phi_only" else ["phi", "xi"]

def _sfx(branch):
    return "" if branch == "phi_only" else "__xi"   # phi_only dirs unchanged


def run_stage1(data, short, branch, force):
    y_tr = data["y_train"]
    per_lag_best = {}
    for lag_key, lags in GAS_LAG_SETS.items():
        cands = {}
        for sc in SCALINGS:
            cdir = OUT / short / "stage1" / (lag_key + "_" + sc + _sfx(branch))
            if (cdir / "result.json").exists() and not force:
                cands[sc] = json.loads((cdir / "result.json").read_text()); continue
            model = ZAGASModel(distribution=_dist(branch), pi_dynamics=pi_dyn(),
                               seasonal="daily", gas_lags=lags, scaling=sc,
                               static_params=_static(branch))
            fr = model.fit(y_tr, verbose=False)
            res, ms, qr, th = _evaluate(model, fr["theta"], data, {}, {}, {}, fr)
            res["scaling"] = sc; res["lag_key"] = lag_key; res["branch"] = branch
            _save(cdir, res, ms, qr, th); cands[sc] = res
            print("    stage1[%s]/%s/%s: rmse=%.3f grad=%.2f aic=%.1f" %
                  (branch, lag_key, sc, res["rmse"], res["grad_norm_inf_numeric"], res["aic"]))
        bsc = _pick_scaling(cands)
        per_lag_best[lag_key] = dict(cands[bsc]); per_lag_best[lag_key]["scaling"] = bsc
    best_lag = min(per_lag_best.items(), key=lambda kv: kv[1].get("aic", 1e18))
    chosen = {"chosen_scaling": best_lag[1]["scaling"], "best_lag_key": best_lag[0], "per_lag": per_lag_best}
    (OUT / short / "stage1" / ("_scaling_choice" + _sfx(branch) + ".json")).write_text(
        json.dumps(chosen, indent=2, default=float))
    return chosen


def run_stage2(data, short, branch, scaling, force):
    y_tr = data["y_train"]
    for lag_key, lags in GAS_LAG_SETS.items():
        for cset in ("dew", "dew_temp", "dew_temp_enso"):
            cdir = OUT / short / "stage2" / (lag_key + "_" + cset + _sfx(branch))
            if (cdir / "result.json").exists() and not force:
                continue
            Xtr, names = _cov_cols(data, cset, "train")
            Xte, _ = _cov_cols(data, cset, "test")
            model = CovZAGASModel(distribution=_dist(branch), pi_dynamics=pi_dyn(),
                                  seasonal="daily", gas_lags=lags, scaling=scaling,
                                  cov_names=names,
                                  cov_target=["phi"] if branch == "phi_only" else None,
                                  static_params=_static(branch))
            fr = model.fit(y=y_tr, X=Xtr, verbose=False)
            res, ms, qr, th = _evaluate(model, fr["theta"], data,
                                        {"X": Xtr}, {"X_train": Xtr, "X_test": Xte},
                                        {"X_train": Xtr, "X_test": Xte}, fr)
            res["scaling"] = scaling; res["lag_key"] = lag_key; res["cov"] = cset; res["branch"] = branch
            _save(cdir, res, ms, qr, th)
            print("    stage2[%s]/%s/%s: rmse=%.3f grad=%.2f" %
                  (branch, lag_key, cset, res["rmse"], res["grad_norm_inf_numeric"]))


def run_harvey_stage(data, short, stage, branch, scaling, force):
    y_tr = data["y_train"]
    harm = 0 if stage == "stage3" else N_HARM
    damp = None if stage == "stage3" else DAMP
    incl_long = (stage != "stage4S")
    for cfg in HARVEY_COV:
        cdir = OUT / short / stage / (cfg + _sfx(branch))
        if (cdir / "result.json").exists() and not force:
            continue
        Xl_tr, ln, Xs_tr, sn = _harvey_cov(data, cfg, "train")
        Xl_te, _, Xs_te, _ = _harvey_cov(data, cfg, "test")
        model = HarveyZAGASModel(
            distribution=_dist(branch), pi_dynamics=pi_dyn(), seasonal="daily",
            long_names=ln, short_names=sn, harvey_target=_htarget(branch),
            scaling=scaling, static_params=_static(branch),
            n_harmonics=harm, seasonal_damping_max=damp, include_long=incl_long,
        )
        opts = {"options": {"maxiter": 300}, "polish": False} if stage in ("stage4", "stage4S") else {}
        fr = model.fit(y=y_tr, X_long=Xl_tr, X_short=Xs_tr, verbose=False, **opts)
        res, ms, qr, th = _evaluate(
            model, fr["theta"], data,
            {"X_long": Xl_tr, "X_short": Xs_tr},
            {"X_long_train": Xl_tr, "X_long_test": Xl_te,
             "X_short_train": Xs_tr, "X_short_test": Xs_te},
            {"X_long_train": Xl_tr, "X_long_test": Xl_te,
             "X_short_train": Xs_tr, "X_short_test": Xs_te}, fr)
        res["scaling"] = scaling; res["cov"] = cfg; res["stage"] = stage; res["branch"] = branch
        _save(cdir, res, ms, qr, th)
        print("    %s[%s]/%s: rmse=%.3f crps=%.3f grad=%.2f" %
              (stage, branch, cfg, res["rmse"], res["crps_mean"], res["grad_norm_inf_numeric"]))


def load_json(pth):
    pth = Path(pth)
    return json.loads(pth.read_text()) if pth.exists() else None


def _select_best_harvey(short):
    def tier(r):
        if r.get("validity") == "failed":
            return 3
        g = r.get("grad_norm_inf_numeric", 1e18)
        return 0 if g < 1 else (1 if g < 100 else 2)
    cands = []
    for st in ("stage3", "stage4", "stage4S"):
        for cfg in HARVEY_COV:
            for br in ("phi_only", "phi_xi"):
                r = load_json(OUT / short / st / (cfg + _sfx(br)) / "result.json")
                if r:
                    cands.append((st, cfg, br, r))
    if not cands:
        return None
    st, cfg, br, _ = min(cands, key=lambda x: (tier(x[3]), x[3].get("crps_mean", 1e18),
                                               x[3].get("rmse", 1e18)))
    return st, cfg, br


def run_stage5(data, short, scaling, force):
    cdir = OUT / short / "stage5" / "best"
    if (cdir / "result.json").exists() and not force:
        return
    sel = _select_best_harvey(short)
    if sel is None:
        print("    stage5: SKIPPED (no Harvey base)"); return
    base_stage, cfg, base_branch = sel
    harm = 0 if base_stage == "stage3" else N_HARM
    damp = None if base_stage == "stage3" else DAMP
    incl_long = (base_stage != "stage4S")
    Xl_tr, ln, Xs_tr, sn = _harvey_cov(data, cfg, "train")
    Xl_te, _, Xs_te, _ = _harvey_cov(data, cfg, "test")
    model = HarveyZAGASModel(
        distribution=GB2LogLink(), pi_dynamics=pi_dyn(), seasonal="daily",
        long_names=ln, short_names=sn, harvey_target=["phi", "xi"],
        scaling=scaling, static_params=["gamma", "zeta"],
        n_harmonics=harm, seasonal_damping_max=damp, include_long=incl_long,
    )
    fr = model.fit(y=data["y_train"], X_long=Xl_tr, X_short=Xs_tr, verbose=False,
                   options={"maxiter": 300}, polish=False)
    res, ms, qr, th = _evaluate(
        model, fr["theta"], data,
        {"X_long": Xl_tr, "X_short": Xs_tr},
        {"X_long_train": Xl_tr, "X_long_test": Xl_te,
         "X_short_train": Xs_tr, "X_short_test": Xs_te},
        {"X_long_train": Xl_tr, "X_long_test": Xl_te,
         "X_short_train": Xs_tr, "X_short_test": Xs_te}, fr)
    res["stage"] = "stage5"; res["base_stage"] = base_stage; res["cov"] = cfg
    res["base_branch"] = base_branch; res["scaling"] = scaling
    _save(cdir, res, ms, qr, th)
    print("    stage5 (base %s/%s/%s): rmse=%.3f crps=%.3f grad=%.2f" %
          (base_stage, cfg, base_branch, res["rmse"], res["crps_mean"], res["grad_norm_inf_numeric"]))


def process_location(location, force):
    short = STATION_REGISTRY[location].get("short", location)
    print(f"=== {location} ({short}) ===")
    data = StationDataLoader.from_registry(location, precip_dir=R.PRECIP_DIR,
                                           era5_dir=R.ERA5_DIR, enso_path=R.ENSO_PATH).load_all()
    t0 = time.time()
    out = {"short": short}
    for branch in ("phi_only", "phi_xi"):
        choice = run_stage1(data, short, branch, force)
        sc = choice["chosen_scaling"]
        print("  [%s] -> chosen scaling: %s (best lag set %s)" % (branch, sc, choice["best_lag_key"]))
        run_stage2(data, short, branch, sc, force)
        for st in ("stage3", "stage4", "stage4S"):
            run_harvey_stage(data, short, st, branch, sc, force)
        out[branch + "_scaling"] = sc
        out[branch + "_best_lag"] = choice["best_lag_key"]
    run_stage5(data, short, out.get("phi_only_scaling", "unit"), force)
    print("  (%.0fs)" % (time.time() - t0))
    return out


def _build_for_augment(res, data, cfg_dir):
    """Reconstruct the fitted model + its filt/oos/ms kwargs from a cached
    result.json (no re-fit -- theta comes from theta.npy). stage/branch are
    read from the cache path so this works for older cache entries that
    predate the res['stage']/res['branch'] keys."""
    stage = res.get("stage") or cfg_dir.parts[-2]           # .../<stage>/<cfg>/
    dirname = cfg_dir.name
    branch = res.get("branch") or ("phi_xi" if dirname.endswith("__xi") else "phi_only")
    scaling = res.get("scaling", "unit")
    if stage in ("stage3", "stage4", "stage4S") and "cov" not in res:
        res["cov"] = dirname[:-5] if dirname.endswith("__xi") else dirname
    if stage == "stage1":
        lags = GAS_LAG_SETS[res["lag_key"]]
        model = ZAGASModel(distribution=_dist(branch), pi_dynamics=pi_dyn(), seasonal="daily",
                           gas_lags=lags, scaling=scaling, static_params=_static(branch))
        return model, {}, {}, {}
    if stage == "stage2":
        lags = GAS_LAG_SETS[res["lag_key"]]
        Xtr, names = _cov_cols(data, res["cov"], "train")
        Xte, _ = _cov_cols(data, res["cov"], "test")
        model = CovZAGASModel(distribution=_dist(branch), pi_dynamics=pi_dyn(), seasonal="daily",
                              gas_lags=lags, scaling=scaling, cov_names=names,
                              cov_target=["phi"] if branch == "phi_only" else None,
                              static_params=_static(branch))
        return model, {"X": Xtr}, {"X_train": Xtr, "X_test": Xte}, {"X_train": Xtr, "X_test": Xte}
    # stage3 / stage4 / stage4S / stage5
    if stage == "stage5":
        base = res.get("base_stage", "stage4"); cov = res["cov"]; branch = "phi_xi"
    else:
        base = stage; cov = res["cov"]
    harm = 0 if base == "stage3" else N_HARM
    damp = None if base == "stage3" else DAMP
    incl_long = (base != "stage4S")
    Xl_tr, ln, Xs_tr, sn = _harvey_cov(data, cov, "train")
    Xl_te, _, Xs_te, _ = _harvey_cov(data, cov, "test")
    model = HarveyZAGASModel(
        distribution=_dist(branch), pi_dynamics=pi_dyn(), seasonal="daily",
        long_names=ln, short_names=sn, harvey_target=_htarget(branch),
        scaling=scaling, static_params=_static(branch),
        n_harmonics=harm, seasonal_damping_max=damp, include_long=incl_long)
    fk = {"X_long": Xl_tr, "X_short": Xs_tr}
    ok = {"X_long_train": Xl_tr, "X_long_test": Xl_te, "X_short_train": Xs_tr, "X_short_test": Xs_te}
    return model, fk, ok, ok


def augment_one(cfg_dir: Path, force: bool = False):
    rp = cfg_dir / "result.json"
    tp = cfg_dir / "theta.npy"
    if not (rp.exists() and tp.exists()):
        return "skip"
    res = json.loads(rp.read_text())
    if (not force and "exc_95_1s" in res and "rmse_insample" in res
            and res.get("rmse_insample") is not None):
        return "have"
    short = cfg_dir.parts[-3]
    loc = next(l for l in LOCATIONS if STATION_REGISTRY[l].get("short", l) == short)
    data = StationDataLoader.from_registry(loc, precip_dir=R.PRECIP_DIR, era5_dir=R.ERA5_DIR,
                                           enso_path=R.ENSO_PATH).load_all()
    theta = np.load(tp)
    try:
        model, fk, ok, mk = _build_for_augment(res, data, cfg_dir)
        new_res, ms, qr, _ = _evaluate(model, theta, data, fk, ok, mk,
                                       {"loglik": res.get("loglik"), "success": res.get("success"),
                                        "validity": res.get("validity")})
        res.update({k: new_res[k] for k in new_res if k not in
                    ("grad_norm_inf_numeric", "grad_converged", "grad_eps_used")})
        rp.write_text(json.dumps(res, indent=2, default=float))
        ms.to_csv(cfg_dir / "multistep.csv", index=False)
        np.save(cfg_dir / "insample_qr.npy", qr)
        return "done"
    except Exception:
        (cfg_dir / "augment_error.txt").write_text(traceback.format_exc())
        return "err"


def _augment_worker(short):
    cnt = {"done": 0, "have": 0, "err": 0, "skip": 0}
    for cfg_dir in sorted((OUT / short).glob("stage*/*/")):
        cnt[augment_one(cfg_dir)] += 1
        print(f"    [{short}] {cfg_dir.relative_to(OUT / short)}: {cnt}", flush=True)
    return short, cnt


def _worker(loc, force):
    """Top-level (picklable) wrapper -- one fully-independent per-location
    pipeline. Cache dirs are disjoint across locations so parallel workers
    never touch the same file (docs/MULTIPROCESSING: independent tasks
    only; results identical, only wall time changes)."""
    try:
        return loc, process_location(loc, force)
    except Exception:
        return loc, {"error": traceback.format_exc()}


# ===========================================================================
# FLAT parallel execution
# ---------------------------------------------------------------------------
# The per-location pipeline is a small DAG:
#   phase A : stage1  -- 6 loc x 2 branch x 4 lag x 3 scaling = 144 indep fits
#             then pick the Fisher scaling per (loc, branch)   [cheap, serial]
#   phase B : stage2 (144) + stage3/4/4S (180) = 324 indep fits, each just
#             reads its (loc,branch) chosen scaling from the phase-A json
#   phase C : stage5 -- 6 fits, each reads all Harvey results for its loc
# Every fit is one independent task -> one big ProcessPoolExecutor per phase,
# so wall time is ~ (#tasks / #workers) * per-fit-time instead of being
# capped at 6 by location-level parallelism. Results are byte-identical to
# the serial path; only scheduling changes (docs/MULTIPROCESSING).
# ===========================================================================

from functools import lru_cache as _lru_cache

BRANCHES = ("phi_only", "phi_xi")


@_lru_cache(maxsize=None)
def _load_data(location):
    return StationDataLoader.from_registry(
        location, precip_dir=R.PRECIP_DIR, era5_dir=R.ERA5_DIR,
        enso_path=R.ENSO_PATH).load_all()


def _short(location):
    return STATION_REGISTRY[location].get("short", location)


def _chosen_scaling(short, branch):
    p = OUT / short / "stage1" / ("_scaling_choice" + _sfx(branch) + ".json")
    return json.loads(p.read_text())["chosen_scaling"]


def _task_stage1(a):
    loc, branch, lag_key, sc, force = a
    short = _short(loc)
    cdir = OUT / short / "stage1" / (lag_key + "_" + sc + _sfx(branch))
    if (cdir / "result.json").exists() and not force:
        return ("skip", f"stage1 {short}/{branch}/{lag_key}/{sc}")
    try:
        data = _load_data(loc)
        model = ZAGASModel(distribution=_dist(branch), pi_dynamics=pi_dyn(),
                           seasonal="daily", gas_lags=GAS_LAG_SETS[lag_key],
                           scaling=sc, static_params=_static(branch))
        fr = model.fit(data["y_train"], verbose=False)
        res, ms, qr, th = _evaluate(model, fr["theta"], data, {}, {}, {}, fr)
        res["scaling"] = sc; res["lag_key"] = lag_key; res["branch"] = branch
        _save(cdir, res, ms, qr, th)
        return ("done", f"stage1 {short}/{branch}/{lag_key}/{sc} "
                        f"rmse={res['rmse']:.3f} grad={res['grad_norm_inf_numeric']:.2f}")
    except Exception:
        (cdir).mkdir(parents=True, exist_ok=True)
        (cdir / "fit_error.txt").write_text(traceback.format_exc())
        return ("err", f"stage1 {short}/{branch}/{lag_key}/{sc}\n{traceback.format_exc()}")


def _task_stage2(a):
    loc, branch, lag_key, cset, force = a
    short = _short(loc)
    cdir = OUT / short / "stage2" / (lag_key + "_" + cset + _sfx(branch))
    if (cdir / "result.json").exists() and not force:
        return ("skip", f"stage2 {short}/{branch}/{lag_key}/{cset}")
    try:
        data = _load_data(loc)
        scaling = _chosen_scaling(short, branch)
        Xtr, names = _cov_cols(data, cset, "train")
        Xte, _ = _cov_cols(data, cset, "test")
        model = CovZAGASModel(distribution=_dist(branch), pi_dynamics=pi_dyn(),
                              seasonal="daily", gas_lags=GAS_LAG_SETS[lag_key],
                              scaling=scaling, cov_names=names,
                              cov_target=["phi"] if branch == "phi_only" else None,
                              static_params=_static(branch))
        fr = model.fit(y=data["y_train"], X=Xtr, verbose=False)
        res, ms, qr, th = _evaluate(model, fr["theta"], data, {"X": Xtr},
                                    {"X_train": Xtr, "X_test": Xte},
                                    {"X_train": Xtr, "X_test": Xte}, fr)
        res["scaling"] = scaling; res["lag_key"] = lag_key; res["cov"] = cset; res["branch"] = branch
        _save(cdir, res, ms, qr, th)
        return ("done", f"stage2 {short}/{branch}/{lag_key}/{cset} rmse={res['rmse']:.3f}")
    except Exception:
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "fit_error.txt").write_text(traceback.format_exc())
        return ("err", f"stage2 {short}/{branch}/{lag_key}/{cset}\n{traceback.format_exc()}")


def _task_harvey(a):
    loc, branch, stage, cfg, force = a
    short = _short(loc)
    cdir = OUT / short / stage / (cfg + _sfx(branch))
    if (cdir / "result.json").exists() and not force:
        return ("skip", f"{stage} {short}/{branch}/{cfg}")
    try:
        data = _load_data(loc)
        scaling = _chosen_scaling(short, branch)
        harm = 0 if stage == "stage3" else N_HARM
        damp = None if stage == "stage3" else DAMP
        incl_long = (stage != "stage4S")
        Xl_tr, ln, Xs_tr, sn = _harvey_cov(data, cfg, "train")
        Xl_te, _, Xs_te, _ = _harvey_cov(data, cfg, "test")
        model = HarveyZAGASModel(
            distribution=_dist(branch), pi_dynamics=pi_dyn(), seasonal="daily",
            long_names=ln, short_names=sn, harvey_target=_htarget(branch),
            scaling=scaling, static_params=_static(branch),
            n_harmonics=harm, seasonal_damping_max=damp, include_long=incl_long)
        opts = {"options": {"maxiter": 300}, "polish": False} if stage in ("stage4", "stage4S") else {}
        fr = model.fit(y=data["y_train"], X_long=Xl_tr, X_short=Xs_tr, verbose=False, **opts)
        ok = {"X_long_train": Xl_tr, "X_long_test": Xl_te,
              "X_short_train": Xs_tr, "X_short_test": Xs_te}
        res, ms, qr, th = _evaluate(model, fr["theta"], data,
                                    {"X_long": Xl_tr, "X_short": Xs_tr}, ok, ok, fr)
        res["scaling"] = scaling; res["cov"] = cfg; res["stage"] = stage; res["branch"] = branch
        _save(cdir, res, ms, qr, th)
        return ("done", f"{stage} {short}/{branch}/{cfg} rmse={res['rmse']:.3f} "
                        f"crps={res['crps_mean']:.3f} grad={res['grad_norm_inf_numeric']:.2f}")
    except Exception:
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "fit_error.txt").write_text(traceback.format_exc())
        return ("err", f"{stage} {short}/{branch}/{cfg}\n{traceback.format_exc()}")


def _task_stage5(a):
    loc, force = a
    short = _short(loc)
    try:
        data = _load_data(loc)
        # phi_only chosen scaling, mirroring process_location()
        scaling = _chosen_scaling(short, "phi_only")
        run_stage5(data, short, scaling, force)
        return ("done", f"stage5 {short}")
    except Exception:
        return ("err", f"stage5 {short}\n{traceback.format_exc()}")


def _run_pool(tasks, fn, nw, label):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    tally = {"done": 0, "skip": 0, "err": 0}
    t0 = time.time()
    print(f"[{label}] {len(tasks)} tasks on {nw} workers", flush=True)
    if nw <= 1:
        for a in tasks:
            st, msg = fn(a); tally[st] += 1
            print(f"  ({st}) {msg}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = [ex.submit(fn, a) for a in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                st, msg = fut.result(); tally[st] += 1
                print(f"  [{i}/{len(tasks)}] ({st}) {msg.splitlines()[0]}", flush=True)
                if st == "err":
                    print(msg, flush=True)
    print(f"[{label}] done in {time.time()-t0:.0f}s  {tally}", flush=True)
    return tally


def run_flat(locs, force, nw):
    # ---- phase A: stage 1 ----
    a_tasks = [(loc, br, lk, sc, force)
               for loc in locs for br in BRANCHES
               for lk in GAS_LAG_SETS for sc in SCALINGS]
    _run_pool(a_tasks, _task_stage1, nw, "phase A / stage1")
    # pick Fisher scaling per (loc, branch) from the 12 cached stage1 fits
    for loc in locs:
        for br in BRANCHES:
            run_stage1(_load_data(loc), _short(loc), br, force=False)
        print(f"  scaling picked: {_short(loc)} "
              f"phi_only={_chosen_scaling(_short(loc),'phi_only')} "
              f"phi_xi={_chosen_scaling(_short(loc),'phi_xi')}", flush=True)

    # ---- phase B: stage 2 + Harvey stages 3/4/4S ----
    b_tasks = [(loc, br, lk, cs, force)
               for loc in locs for br in BRANCHES
               for lk in GAS_LAG_SETS for cs in ("dew", "dew_temp", "dew_temp_enso")]
    h_tasks = [(loc, br, st, cfg, force)
               for loc in locs for br in BRANCHES
               for st in ("stage3", "stage4", "stage4S") for cfg in HARVEY_COV]
    _run_pool(b_tasks + h_tasks, _dispatch_B, nw, "phase B / stage2+harvey")

    # ---- phase C: stage 5 ----
    _run_pool([(loc, force) for loc in locs], _task_stage5, min(nw, len(locs)), "phase C / stage5")

    (OUT / "summary.json").write_text(json.dumps(
        {_short(loc): {"mode": "flat"} for loc in locs}, indent=2))
    print("flat run complete; summary.json written", flush=True)


def _dispatch_B(a):
    # phase B mixes stage2 (5-tuple, lag_key str) and harvey (5-tuple, stage str);
    # tell them apart by whether a[2] is a gas-lag key.
    return _task_stage2(a) if a[2] in GAS_LAG_SETS else _task_harvey(a)


def _task_augment_flat(arg):
    cfg_dir_str, force = arg if isinstance(arg, tuple) else (arg, False)
    try:
        return (augment_one(Path(cfg_dir_str), force=force), cfg_dir_str)
    except Exception:
        return ("err", cfg_dir_str + "\n" + traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated location shorts")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel workers. Location-level unless --flat, in "
                         "which case it is fit-level (recommend ~22 on this box).")
    ap.add_argument("--flat", action="store_true",
                    help="fit-level parallelism: 3 dependency-ordered phases "
                         "(stage1 | stage2+harvey | stage5), each fanned out "
                         "over one big pool. Identical results, far shorter wall.")
    ap.add_argument("--augment", action="store_true",
                    help="do NOT re-fit; for every cached config reload theta.npy "
                         "and (re)compute in-sample RMSE/CRPS + 1-step & multi-step "
                         "exceedance/Kupiec at 95%/98%, merging into result.json.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.augment:
        shorts = [STATION_REGISTRY[l].get("short", l) for l in LOCATIONS
                  if not args.only or STATION_REGISTRY[l].get("short", l) in args.only.split(",")]
        if args.flat:
            # one task per cached config dir -- maximum fan-out
            cfg_dirs = [str(d) for s in shorts
                        for d in sorted((OUT / s).glob("stage*/*/"))]
            from concurrent.futures import ProcessPoolExecutor, as_completed
            tally = {"done": 0, "have": 0, "err": 0, "skip": 0}
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_task_augment_flat, (d, args.force)) for d in cfg_dirs]
                for i, fut in enumerate(as_completed(futs), 1):
                    st, d = fut.result()
                    tally[st] = tally.get(st, 0) + 1
                    if st == "err" or i % 25 == 0:
                        print(f"  [{i}/{len(cfg_dirs)}] {st} {d.splitlines()[0]}", flush=True)
            print("augment pass complete", tally)
            return
        if args.workers <= 1:
            for s in shorts:
                print(_augment_worker(s))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=min(args.workers, len(shorts))) as ex:
                for fut in as_completed([ex.submit(_augment_worker, s) for s in shorts]):
                    print("  [augmented]", fut.result())
        print("augment pass complete")
        return

    locs = [l for l in LOCATIONS
            if not args.only or STATION_REGISTRY[l].get("short", l) in args.only.split(",")]
    summary = {}

    if args.flat:
        run_flat(locs, args.force, args.workers)
        return

    if args.workers <= 1 or len(locs) <= 1:
        for loc in locs:
            l, r = _worker(loc, args.force)
            summary[STATION_REGISTRY[l].get("short", l)] = r
            (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        nw = min(args.workers, len(locs))
        print(f"running {len(locs)} locations across {nw} parallel workers")
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = {ex.submit(_worker, loc, args.force): loc for loc in locs}
            for fut in as_completed(futs):
                l, r = fut.result()
                summary[STATION_REGISTRY[l].get("short", l)] = r
                (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
                print(f"  [done] {STATION_REGISTRY[l].get('short', l)}")
    print("\nWrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
