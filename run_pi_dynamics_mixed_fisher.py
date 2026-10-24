"""
Pi-dynamics alternatives under JOINT ZA-GAS estimation, unit vs mixed
inverse-Fisher scaling of the phi score (2026-10-01).

=============================================================================
WHAT CHANGES RELATIVE TO run_pi_dynamics_gas11_validation.py
=============================================================================

Same data, same 8-year fit / 2-year validation split, same GAS(1,1)
phi-only magnitude recursion, same validation metrics. Three changes:

1. Every alternative is estimated JOINTLY as one ZA-GAS model
   (ZAGASModel with that alternative as its pi dynamics), instead of a
   standalone pi fit paired with a separately fitted magnitude path.
2. Each alternative is run under two scalings of the phi score:
     unit                  s_t
     mixed_inverse_fisher  s_t / I_t, the inverse Fisher information of the
                           zero-adjusted (mixed) density
                           f(y) = (1-pi_t) 1{y=0} + pi_t g(y; phi_t) 1{y>0}
   with s_t = 1{y>0} dlog g/dphi + lambda_1 (1{y>0} - pi_t) and
   I_t = pi_t I_g + lambda_1^2 pi_t (1 - pi_t)  (lambda_1 = 0 unless pi
   depends on phi; I_g = GB2 Fisher information for phi).
3. Two alternatives are added: A7 and C7, alternatives A and C with the
   y-lag set {1,2,3,364,365,366,367}.

Alternatives (eta_t = logit pi_t):
  A   omega0 + rho eta_{t-1} + sum_{l in {1,365,366}} w_l y_{t-l}
  B   omega0 + rho eta_{t-1} + sum_{l=1,2,3}          w_l y_{t-l}
  C   omega0                 + sum_{l=1,2,3}          w_l y_{t-l}
  D   lambda0 + lambda1 phi_{t|t-1}   (phi of the SAME filter, joint: at
      each t phi_{t|t-1} is formed from information up to t-1, then pi_t,
      then the score of y_t updates phi_{t+1|t})
  A7  as A with lags {1,2,3,364,365,366,367}
  C7  as C with lags {1,2,3,364,365,366,367}

=============================================================================
EXECUTION FLOW (per location, parallel over (location, alternative))
=============================================================================

    load *_train.csv, split fit / validation (8y / 2y)
    for each alternative:
        unit scaling:
            A/B/C: the likelihood separates (pi block vs magnitude block, no
                   shared parameters under unit scaling), so the joint MLE is
                   the cached standalone pi fit + the cached unit magnitude fit
                   (artifacts/pi_dynamics_gas11_validation_experiment). That
                   cached theta is used as-is; its joint checked gradient is
                   recorded, and it is re-optimised only if not converged.
            D/A7/C7: joint BFGS fit (warm start from cached pieces).
        mixed_inverse_fisher: joint BFGS fit from two starts (the unit
            solution with A_phi rescaled by mean(pi) I_g, and the default
            initial theta); the higher log-likelihood is kept.
        checked gradient at the optimum (penalised objective); fresh fits
            with |grad|_inf >= 1 are refined by BFGS restarts (refine())
        validation: Numba filter over fit+validation, one-step-ahead, no
            re-fit -> RMSE, CRPS (diagnostics.pi_dynamics_eval, 500 draws,
            seed 42), mean pi | y=0, mean pi | y>0, N dry, N wet,
            separation = N_wet mean(pi|y>0) - N_dry mean(pi|y=0)
        save artifacts/pi_dynamics_mixed_fisher/<short>/<alt>/<scaling>/

All filtering runs in the Numba kernel models/_nb_gb2_filter.py::
nb_filter_phi_only (parity with the Python path: tests/
test_mixed_fisher_parity.py).

USAGE
    python run_pi_dynamics_mixed_fisher.py [--workers 18] [--force]
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

from data.station_loader import STATION_REGISTRY, StationDataLoader
from distributions import GB2LogLinkPhiOnly
from models.za_gas_model import ZAGASModel
from pi_dynamics.ar_logistic import ARLogisticPiDynamics
from pi_dynamics.ar_logistic_custom_lags import ARLogisticCustomLagsPiDynamics
from pi_dynamics.phi_linked import PhiLinkedJointPiDynamics
from diagnostics.pi_dynamics_eval import in_sample_point_and_crps_metrics
from diagnostics.numerical_diff import numerical_gradient_checked
from diagnostics.information import aic as _aic
import run_pi_dynamics_gas11_validation as V

OUT = ROOT / "artifacts" / "pi_dynamics_mixed_fisher"
CACHE = V.EXPERIMENT_DIR
LAGS7 = [1, 2, 3, 364, 365, 366, 367]
SCALINGS = ["unit", "mixed_inverse_fisher"]
GRAD_OK = 1.0

ALTERNATIVES = {
    "A":  ("existing_ar_seasonal", lambda: ARLogisticPiDynamics()),
    "B":  ("short_lags_ar",        lambda: ARLogisticCustomLagsPiDynamics(lags=[1, 2, 3], include_ar=True)),
    "C":  ("short_lags_noar",      lambda: ARLogisticCustomLagsPiDynamics(lags=[1, 2, 3], include_ar=False)),
    "D":  ("phi_linked",           lambda: PhiLinkedJointPiDynamics()),
    "A7": (None,                   lambda: ARLogisticCustomLagsPiDynamics(lags=LAGS7, include_ar=True)),
    "C7": (None,                   lambda: ARLogisticCustomLagsPiDynamics(lags=LAGS7, include_ar=False)),
}
DESCRIPTION = {
    "A": "omega0 + rho eta_{t-1} + sum_{l in {1,365,366}} w_l y_{t-l}",
    "B": "omega0 + rho eta_{t-1} + sum_{l=1,2,3} w_l y_{t-l}",
    "C": "omega0 + sum_{l=1,2,3} w_l y_{t-l}",
    "D": "lambda0 + lambda1 phi_{t|t-1} (joint)",
    "A7": "omega0 + rho eta_{t-1} + sum_{l in {1,2,3,364,365,366,367}} w_l y_{t-l}",
    "C7": "omega0 + sum_{l in {1,2,3,364,365,366,367}} w_l y_{t-l}",
}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def build_model(alt: str, scaling: str) -> ZAGASModel:
    return ZAGASModel(distribution=GB2LogLinkPhiOnly(), pi_dynamics=ALTERNATIVES[alt][1](),
                      seasonal=V.SEASONAL, gas_lags=V.GAS11_LAGS, scaling=scaling,
                      static_params=["xi", "gamma", "zeta"])


def load_split(location: str):
    data = StationDataLoader.from_registry(location, precip_dir=V.PRECIP_DIR, era5_dir=V.ERA5_DIR,
                                           enso_path=V.ENSO_PATH).load_all()
    y_fit, y_val, cut, n_fit = V._split_fit_validation(data["y_train"], data["dates_train"])
    return np.asarray(y_fit, float), np.asarray(y_val, float), cut


def _csv_params(path: Path) -> dict:
    df = pd.read_csv(path)
    return dict(zip(df["parameter"], df["value"].astype(float)))


def cached_theta(short: str, alt: str, model: ZAGASModel):
    """Assemble a joint theta from the cached unit magnitude fit and the cached
    standalone pi fit (by parameter name). Missing pieces fall back to the
    model's default initial values. Returns (theta, n_names_from_cache)."""
    names = model.parameter_names()
    mag = CACHE / short / "magnitude_gas11" / "candidates" / "unit" / "estimated_parameters.csv"
    pool = {}
    if mag.exists():
        pool.update(_csv_params(mag))     # GAS(1,1) magnitude block (+ its ARLogistic pi)
    old_key = ALTERNATIVES[alt][0]
    if old_key:
        p = CACHE / short / old_key / "estimated_parameters.csv"
        if p.exists():
            pool.update(_csv_params(p))   # pi parameters of this alternative override
    return np.array([pool.get(n, np.nan) for n in names], float), sum(n in pool for n in names)


def checked_grad(model, theta, y):
    out = numerical_gradient_checked(lambda th: model._run_filter(th, y, return_paths=False),
                                     np.asarray(theta, float))
    g = np.asarray(out["grad"], float)
    return g, float(np.max(np.abs(g))), bool(out["converged"])


def fit_from(model, y, theta0):
    fit = model.fit(y, theta0=np.asarray(theta0, float), polish=True)
    return fit


def refine(model, theta, y, gmax, rounds=6):
    """BFGS restarts from the current optimum (tighter tolerance, more
    iterations) while the checked gradient is >= GRAD_OK; a restart is kept
    only if it does not lower the log-likelihood."""
    ll = float(model.loglik(theta, y))
    start_g = gmax
    for k in range(rounds):
        fit = model.fit(y, theta0=theta, polish=True, options={"maxiter": 3000, "gtol": 1e-5})
        if fit["loglik"] < ll - 1e-9:
            break
        theta, ll = fit["theta"], float(fit["loglik"])
        grad, gmax, gconv = checked_grad(model, theta, y)
        if gmax < GRAD_OK:
            break
    grad, gmax, gconv = checked_grad(model, theta, y)
    return theta, grad, gmax, gconv, f"refined: |grad|_inf {start_g:.3g} -> {gmax:.3g} ({k + 1} restarts)"


def validation_metrics(model, theta, y_fit, y_val, n_draws=500, seed=42):
    """One-step-ahead over the validation block via the Numba filter run on
    fit+validation (identical to simulate_oos; see tests/test_mixed_fisher_parity.py)."""
    y_full = np.concatenate([y_fit, y_val])
    paths = model._run_filter(theta, y_full, return_paths=True)
    if not paths:
        raise RuntimeError("filter failed over fit+validation")
    eff, n_fit = int(paths["eff_start"]), len(y_fit)
    phi_val = np.asarray(paths["phi"], float)[n_fit - eff:]
    pi_val = np.asarray(paths["pi"], float)[n_fit - eff:]
    static = {k: float(v) for k, v in paths["static"].items()}
    wet = y_val > 0
    m = in_sample_point_and_crps_metrics(dist=GB2LogLinkPhiOnly(), phi_arr=phi_val, static=static,
                                         pi_arr=pi_val, y_arr=y_val, n_draws=n_draws, seed=seed)
    mean_wet, mean_dry = float(pi_val[wet].mean()), float(pi_val[~wet].mean())
    return {
        "val_rmse": m["rmse_all"], "val_rmse_dry": m["rmse_dry"], "val_rmse_wet": m["rmse_wet"],
        "val_crps": m["crps_all"], "val_n_dry": int((~wet).sum()), "val_n_wet": int(wet.sum()),
        "val_pi_mean_dry": mean_dry, "val_pi_mean_wet": mean_wet,
        "val_separation": float(wet.sum() * mean_wet - (~wet).sum() * mean_dry),
    }, phi_val, pi_val


def _mixed_warm_start(model_unit, theta_unit, y_fit, model_mixed):
    """Unit solution with A_phi rescaled by mean(pi) I_g, so the first mixed
    step has roughly the same impact on phi as the unit fit."""
    th = np.array(theta_unit, float)
    names = model_mixed.parameter_names()
    paths = model_unit._run_filter(theta_unit, y_fit, return_paths=True)
    if paths and "A_phi_1" in names:
        st = paths["static"]
        i_g = GB2LogLinkPhiOnly().fisher_info_diag(phi=0.0, **st)["phi"]
        k = names.index("A_phi_1")
        th[k] = th[k] * float(np.mean(paths["pi"])) * float(i_g)
    return th


# -----------------------------------------------------------------------------
# one (location, alternative): both scalings
# -----------------------------------------------------------------------------

def run_one(location: str, alt: str, force: bool) -> list:
    short = STATION_REGISTRY[location]["short"]
    y_fit, y_val, cut = load_split(location)
    results, unit_fit = [], None
    for scaling in SCALINGS:
        out_dir = OUT / short / alt / scaling
        if (out_dir / "metadata.json").exists() and not force:
            meta = json.loads((out_dir / "metadata.json").read_text())
            if scaling == "unit":
                unit_fit = (build_model(alt, "unit"),
                            pd.read_csv(out_dir / "estimated_parameters.csv")["value"].to_numpy(float))
            results.append(meta)
            continue
        t0 = time.time()
        model = build_model(alt, scaling)
        names = model.parameter_names()
        try:
            source, notes = "fit", []
            if scaling == "unit":
                th_cache, n_cached = cached_theta(short, alt, model)
                th0 = np.where(np.isfinite(th_cache), th_cache, model.initial_theta(y_fit))
                if alt in ("A", "B", "C") and n_cached == len(names):
                    g, gmax, _ = checked_grad(model, th0, y_fit)
                    if gmax < GRAD_OK:
                        theta, source = th0, "cache"
                        notes.append("joint MLE = cached standalone pi + cached unit magnitude (separable likelihood)")
                    else:
                        theta = fit_from(model, y_fit, th0)["theta"]
                        notes.append(f"cached theta not converged jointly (|grad|_inf={gmax:.3g}); re-optimised")
                else:
                    cands = [fit_from(model, y_fit, th0), fit_from(model, y_fit, model.initial_theta(y_fit))]
                    theta = max(cands, key=lambda f: f["loglik"])["theta"]
                    notes.append(f"best of 2 starts (cache warm start, default); "
                                 f"loglik {cands[0]['loglik']:.3f} / {cands[1]['loglik']:.3f}")
                unit_fit = (model, theta)
            else:
                starts = [model.initial_theta(y_fit)]
                if unit_fit is not None:
                    starts.insert(0, _mixed_warm_start(unit_fit[0], unit_fit[1], y_fit, model))
                cands = [fit_from(model, y_fit, s) for s in starts]
                theta = max(cands, key=lambda f: f["loglik"])["theta"]
                notes.append("best of starts (unit solution rescaled, default): loglik "
                             + " / ".join(f"{c['loglik']:.3f}" for c in cands))
            grad, gmax, gconv = checked_grad(model, theta, y_fit)
            if source == "fit" and gmax >= GRAD_OK:
                theta, grad, gmax, gconv, msg = refine(model, theta, y_fit, gmax)
                notes.append(msg)
            ll = float(model.loglik(theta, y_fit))
            val, phi_val, pi_val = validation_metrics(model, theta, y_fit, y_val)
            meta = {
                "location": location, "short": short, "alternative": alt, "scaling": scaling,
                "description": DESCRIPTION[alt], "source": source, "notes": notes,
                "fit_cutoff": cut, "n_fit": len(y_fit), "n_val": len(y_val),
                "loglik_fit": ll, "n_params": len(theta), "aic_fit": float(_aic(ll, len(theta))),
                "grad_norm_inf": gmax, "grad_converged": gconv,
                "valid": bool(np.isfinite(ll) and gmax < 100), "success": True,
                "runtime_s": time.time() - t0, **val,
            }
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"parameter": names, "value": theta}).to_csv(out_dir / "estimated_parameters.csv", index=False)
            pd.DataFrame({"parameter": names, "gradient": grad}).to_csv(out_dir / "gradients.csv", index=False)
            np.savez(out_dir / "paths.npz", phi_val=phi_val, pi_val=pi_val, y_val=y_val)
        except Exception as exc:
            meta = {"location": location, "short": short, "alternative": alt, "scaling": scaling,
                    "success": False, "message": str(exc), "traceback": traceback.format_exc()}
            out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))
        results.append(meta)
    return results


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    tasks = [(loc, alt) for loc in V.LOCATION_ORDER for alt in ALTERNATIVES]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, loc, alt, args.force): (loc, alt) for loc, alt in tasks}
        for f in as_completed(futs):
            loc, alt = futs[f]
            for m in f.result():
                if m.get("success"):
                    print(f"{m['short']:10s} {alt:3s} {m['scaling']:21s} [{m['source']:5s}] "
                          f"RMSE {m['val_rmse']:.3f} CRPS {m['val_crps']:.3f} "
                          f"pi|0 {m['val_pi_mean_dry']:.3f} pi|>0 {m['val_pi_mean_wet']:.3f} "
                          f"sep {m['val_separation']:.1f} |g| {m['grad_norm_inf']:.2e}", flush=True)
                else:
                    print(f"{loc} {alt} {m['scaling']}: FAILED {m.get('message')}", flush=True)


if __name__ == "__main__":
    main()
