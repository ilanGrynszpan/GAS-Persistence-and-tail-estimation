"""
Pi-dynamics alternatives -- GAS(1,1) magnitude, 8y-fit / 2y-validation split
(2026-09-10).

=============================================================================
WHY THIS EXISTS
=============================================================================

run_pi_dynamics_gas11.py fits the four occurrence-dynamics alternatives
(A..D) and the phi-only GAS(1,1) magnitude model on the WHOLE ten-year
training series and evaluates every metric IN-SAMPLE. That conflates fit
quality with predictive quality. This script keeps the identical model
set, the identical GAS(1,1) magnitude recursion, and the identical
score-scaling selection rule, but changes the evaluation protocol:

    fit block        = the first 8 calendar years of *_train.csv
    validation block = the final 2 calendar years of *_train.csv
                       (the held-out *_test.csv is NOT touched here)

Every model -- the magnitude GAS(1,1) AND each standalone pi_t fit -- is
estimated ONLY on the fit block. The summary table
(generate_pi_dynamics_gas11_summary.py) then reports RMSE, CRPS and the
dry/wet occurrence split computed ONLY on the validation block, one-step
-ahead (no re-fit, no peeking): the fitted parameters are rolled forward
through the validation block via ZAGASModel.simulate_oos (magnitude phi
path) and pi_dynamics.standalone_fit.eta_path_full (each variant's pi
path), exactly the 1-step-ahead machinery the main pipeline already uses
for out-of-sample scoring.

The QQ / ACF mosaics (generate_pi_dynamics_gas11_mosaics.py) instead use
each variant's IN-SAMPLE quantile residuals over the 8-year fit block --
"reproduce the mosaics using the new (8-year) training data".

=============================================================================
EXECUTION FLOW  (per location, six locations)
=============================================================================

    load *_train.csv (StationDataLoader) -> y_train, dates_train
        |
    split by calendar date: y_fit = first SPLIT_YEARS years, y_val = rest
        |
    for scaling in {unit, diagonal_inverse_fisher, inverse_fisher}:
        fit ZAGASModel(GB2LogLinkPhiOnly, ARLogistic, gas_lags=[1]) on y_fit
        checked-gradient at the optimum; AIC
    select the magnitude scaling (gradient tiers -> AIC, unchanged rule)
        |
    magnitude paths:
        in-sample  varphi over the fit block   (mdl.filter on y_fit)
        validation varphi over the val block   (mdl.simulate_oos(y_fit, y_val))
        |
    for each of the four pi_t alternatives:
        fit_pi_dynamics_standalone on y_fit ; checked gradient
        eta_path_full on concat(y_fit, y_val) -> pi over both blocks
        IN-SAMPLE (fit block): PIT, quantile residuals, 400-lag ACF  [mosaics]
        VALIDATION (val block): mean(pi|y=0) vs mean(pi|y>0), N dry/wet,
            RMSE(all/dry/wet) + CRPS(all) vs y_val with the frozen
            validation varphi path                                   [summary]
        |
    save metadata.json (fit_* and val_* blocks), estimated_parameters.csv,
         gradients.csv, paths.npz, acf_400.csv

=============================================================================
USAGE
=============================================================================

    python run_pi_dynamics_gas11_validation.py [--force] [--location NAME] [--n-draws N]

Artifacts:
    artifacts/pi_dynamics_gas11_validation_experiment/<short>/magnitude_gas11/...
    artifacts/pi_dynamics_gas11_validation_experiment/<short>/<variant>/...
    artifacts/pi_dynamics_gas11_validation_experiment/execution_log.jsonl
    pi_dynamics_gas11_validation.log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRECIP_DIR = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR   = ROOT / "data" / "input" / "ERA5"
ENSO_PATH  = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
EXPERIMENT_DIR = ROOT / "artifacts" / "pi_dynamics_gas11_validation_experiment"

from data.station_loader import STATION_REGISTRY, StationDataLoader
from distributions import GB2LogLinkPhiOnly
from pi_dynamics.standalone_fit import (
    fit_pi_dynamics_standalone, eta_path_full, standalone_neg_loglik,
)
from diagnostics.occurrence_pit import bernoulli_pit_values, bernoulli_quantile_residuals
from diagnostics.acf import acf_at_lags
from diagnostics.pi_dynamics_eval import in_sample_point_and_crps_metrics
from diagnostics.information import aic as _aic, bic as _bic

# Reuse the magnitude-model helpers verbatim -- they take the series to fit
# on as an argument, so passing y_fit instead of the full y_train is all
# that is needed. None of these log or write files.
from run_pi_dynamics_gas11 import (
    _checked_gradient, _build_magnitude_model, _select_magnitude,
    _build_variants, SCALINGS, GAS11_LAGS, SEASONAL,
    ACF_TABLE_LAGS, ACF_FULL_LAGS, VARIANT_KEYS, _VARIANT_DESCRIPTIONS,
)

LOCATION_ORDER = [
    "BELO HORIZONTE",
    "CRUZEIRO DO SUL (ACRE)",
    "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)",
    "MANAUS",
    "SALVADOR",
]

SPLIT_YEARS = 8   # first 8 calendar years -> fit; remainder -> validation

# -- Dedicated logger (kept off the root logger so it does not also write to
#    run_pi_dynamics_gas11.py's pi_dynamics_gas11.log, which is opened when
#    that module is imported above) --------------------------------------------
logger = logging.getLogger("pi_dynamics_gas11_validation")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _fmt = logging.Formatter("%(asctime)s  [%(levelname)s]  %(message)s", "%Y-%m-%d %H:%M:%S")
    for _h in (logging.FileHandler(ROOT / "pi_dynamics_gas11_validation.log", mode="a"),
               logging.StreamHandler(sys.stdout)):
        _h.setFormatter(_fmt)
        logger.addHandler(_h)


def _execution_log(record: dict) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENT_DIR / "execution_log.jsonl", "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# -----------------------------------------------------------------------------
# Fit / validation split
# -----------------------------------------------------------------------------

def _split_fit_validation(y_train: np.ndarray, dates_train: pd.DatetimeIndex):
    """
    y_fit  = observations dated strictly before (first date + SPLIT_YEARS years)
    y_val  = the rest of the training series.
    Falls back to an 80/20 split by length only if the date index is unusable.
    """
    y_train = np.asarray(y_train, dtype=float)
    try:
        cut = dates_train[0] + pd.DateOffset(years=SPLIT_YEARS)
        mask_fit = np.asarray(dates_train < cut)
        if mask_fit.any() and (~mask_fit).any():
            n_fit = int(mask_fit.sum())
            return y_train[:n_fit], y_train[n_fit:], str(cut.date()), n_fit
    except Exception:
        pass
    n_fit = int(round(len(y_train) * SPLIT_YEARS / (SPLIT_YEARS + 2)))
    return y_train[:n_fit], y_train[n_fit:], "length-fraction", n_fit


# -----------------------------------------------------------------------------
# Step 1: GAS(1,1) magnitude model, fit on y_fit, rolled forward to y_val
# -----------------------------------------------------------------------------

def _fit_one_scaling_on(scaling: str, y_fit: np.ndarray) -> dict:
    """Fit the GAS(1,1) magnitude model for one scaling on y_fit; attach an
    independent checked gradient at the optimum plus AIC/BIC. Mirrors
    run_pi_dynamics_gas11._fit_one_magnitude_scaling, but on the fit block."""
    mdl = _build_magnitude_model(scaling)
    fit = mdl.fit(y_fit, polish=True)
    grad_info = _checked_gradient(lambda th: -mdl.loglik(th, y_fit), fit["theta"])
    n_obs_eff = int(len(y_fit) - mdl.max_lag)
    ll = float(fit["loglik"]); k = int(mdl.n_params)
    return {
        "scaling": scaling, "model": mdl, "theta": fit["theta"],
        "param_names": fit["param_names"], "loglik": ll, "n_params": k,
        "n_obs_eff": n_obs_eff, "aic": float(_aic(ll, k)),
        "bic": float(_bic(ll, k, n_obs_eff)),
        "validity": fit.get("validity", "unknown"),
        "success": bool(fit.get("success", False)),
        "message": str(fit.get("message", "")), "n_iter": int(fit.get("n_iter", 0)),
        "scipy_grad_norm_inf": float(fit.get("grad_norm_inf", float("nan"))),
        "std_errors": fit.get("std_errors"), "se_quality": fit.get("se_quality", "unavailable"),
        "runtime_s": float(fit.get("runtime_s", float("nan"))),
        "bound_diagnostics": fit.get("bound_diagnostics", {}),
        **grad_info,
    }


def _save_magnitude_candidate(out_dir: Path, cand: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"parameter": cand["param_names"], "value": cand["theta"]}).to_csv(
        out_dir / "estimated_parameters.csv", index=False)
    pd.DataFrame({"parameter": cand["param_names"], "gradient": cand["grad"]}).to_csv(
        out_dir / "gradients.csv", index=False)
    if cand.get("std_errors") is not None:
        pd.DataFrame({"parameter": cand["param_names"], "std_error": cand["std_errors"],
                      "se_quality": cand["se_quality"]}).to_csv(
            out_dir / "standard_errors.csv", index=False)
    meta = {k: cand[k] for k in (
        "scaling", "loglik", "n_params", "n_obs_eff", "aic", "bic", "validity",
        "success", "message", "n_iter", "scipy_grad_norm_inf", "grad_norm_inf",
        "grad_norm_2", "grad_eps_used", "grad_converged", "runtime_s",
        "selection_tier", "se_quality")}
    meta["bound_diagnostics"] = cand.get("bound_diagnostics", {})
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))


def fit_magnitude(short: str, y_fit: np.ndarray, y_val: np.ndarray, n_fit: int,
                  cut_label: str, out_base: Path) -> dict:
    """Fit + select the GAS(1,1) magnitude model on y_fit; recover the
    in-sample varphi path (fit block) and the 1-step-ahead varphi path over
    the validation block. Returns a `baseline` dict consumed by the four
    pi_t alternatives."""
    mag_dir = out_base / "magnitude_gas11"

    cands = []
    for scaling in SCALINGS:
        logger.info(f"  [magnitude/{scaling}] fitting GAS(1,1) phi-only on {len(y_fit)}-obs fit block...")
        c = _fit_one_scaling_on(scaling, y_fit)
        cands.append(c)
        logger.info(f"  [magnitude/{scaling}] loglik={c['loglik']:.2f}  AIC={c['aic']:.2f}  "
                    f"||grad||_inf={c['grad_norm_inf']:.3e}  converged={c['grad_converged']}  "
                    f"validity={c['validity']}")

    selected, tier_label = _select_magnitude(cands)
    for c in cands:
        _save_magnitude_candidate(mag_dir / "candidates" / c["scaling"], c)
    logger.info(f"  [magnitude] SELECTED scaling='{selected['scaling']}' via tier '{tier_label}' "
                f"(AIC={selected['aic']:.2f}, ||grad||_inf={selected['grad_norm_inf']:.3e})")

    mdl = selected["model"]

    # In-sample varphi over the fit block.
    paths_is = mdl.filter(selected["theta"], y_fit)
    if not paths_is:
        raise RuntimeError(f"Filtering the selected GAS(1,1) magnitude model failed at {short}.")
    eff_start = int(paths_is["eff_start"])
    phi_fit = np.asarray(paths_is["phi"], dtype=float)          # length n_fit - eff_start
    f0_phi = float(phi_fit[0])
    static = {k: float(v) for k, v in paths_is["static"].items()}

    # 1-step-ahead varphi over the validation block (no re-fit).
    oos = mdl.simulate_oos(selected["theta"], y_fit, y_val)
    phi_val = np.asarray(oos["f_arr_oos"][:, 0], dtype=float)   # length len(y_val)
    static_val = {k: float(v) for k, v in oos["static"].items()}

    # -- Save the selected magnitude model's artifacts --------------------
    mag_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"parameter": selected["param_names"], "value": selected["theta"]}).to_csv(
        mag_dir / "estimated_parameters.csv", index=False)
    pd.DataFrame({"parameter": selected["param_names"], "gradient": selected["grad"]}).to_csv(
        mag_dir / "gradients.csv", index=False)
    if selected.get("std_errors") is not None:
        pd.DataFrame({"parameter": selected["param_names"], "std_error": selected["std_errors"],
                      "se_quality": selected["se_quality"]}).to_csv(
            mag_dir / "standard_errors.csv", index=False)
    pd.DataFrame([{
        "scaling": c["scaling"], "loglik": c["loglik"], "aic": c["aic"], "bic": c["bic"],
        "grad_norm_inf": c["grad_norm_inf"], "grad_norm_2": c["grad_norm_2"],
        "grad_converged": c["grad_converged"], "scipy_grad_norm_inf": c["scipy_grad_norm_inf"],
        "validity": c["validity"], "selection_tier": c["selection_tier"],
        "selected": (c["scaling"] == selected["scaling"]),
    } for c in cands]).to_csv(mag_dir / "scaling_selection.csv", index=False)
    np.savez(mag_dir / "phi_path.npz",
             phi_fit=phi_fit, phi_val=phi_val,
             eff_start=np.array(eff_start), f0_phi=np.array(f0_phi), n_fit=np.array(n_fit))

    meta = {
        "model_type": "ZAGASModel_GAS11_phi_only_validation",
        "location_short": short, "gas_lags": GAS11_LAGS,
        "split_years": SPLIT_YEARS, "fit_cutoff": cut_label,
        "n_fit": int(n_fit), "n_val": int(len(y_val)),
        "selected_scaling": selected["scaling"], "selection_tier": tier_label,
        "scalings_tried": SCALINGS,
        "loglik_fit": selected["loglik"], "n_params": selected["n_params"],
        "aic_fit": selected["aic"], "bic_fit": selected["bic"],
        "validity": selected["validity"], "success": selected["success"],
        "message": selected["message"], "n_iter": selected["n_iter"],
        "grad_norm_inf": selected["grad_norm_inf"], "grad_norm_2": selected["grad_norm_2"],
        "grad_eps_used": selected["grad_eps_used"], "grad_converged": selected["grad_converged"],
        "scipy_grad_norm_inf": selected["scipy_grad_norm_inf"],
        "se_quality": selected["se_quality"], "runtime_s": selected["runtime_s"],
        "eff_start": eff_start, "f0_phi": f0_phi,
        "static": static, "static_val_check": static_val,
        "bound_diagnostics": selected.get("bound_diagnostics", {}),
    }
    (mag_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    return {
        "eff_start": eff_start, "phi_fit": phi_fit, "phi_val": phi_val,
        "static": static, "f0_phi": f0_phi, "n_fit": int(n_fit), "n_val": int(len(y_val)),
        "selected_scaling": selected["scaling"], "selection_tier": tier_label,
        "loglik_fit": selected["loglik"], "aic_fit": selected["aic"],
        "grad_norm_inf": selected["grad_norm_inf"], "validity": selected["validity"],
    }


# -----------------------------------------------------------------------------
# Step 2: one pi_t alternative -- fit on y_fit, score on y_val
# -----------------------------------------------------------------------------

def _fit_and_diagnose_one(location: str, variant_key: str, pi_dyn,
                          y_fit: np.ndarray, y_val: np.ndarray, baseline: dict,
                          n_draws: int, seed: int = 42) -> dict:
    t0 = time.time()
    fit = fit_pi_dynamics_standalone(y_fit, pi_dyn=pi_dyn, seasonal=SEASONAL, verbose=False)
    if not fit["success"]:
        return {"variant_key": variant_key, "success": False,
                "message": "standalone pi_dynamics fit failed on the fit block",
                "runtime_s": time.time() - t0}

    grad_info = _checked_gradient(
        lambda th: standalone_neg_loglik(th, pi_dyn, y_fit, SEASONAL), fit["theta"])

    n_fit = baseline["n_fit"]
    eff = baseline["eff_start"]
    y_full = np.concatenate([y_fit, y_val])
    eta_full = eta_path_full(pi_dyn, fit["theta"], y_full, SEASONAL)
    pi_full = expit(eta_full)

    # -- IN-SAMPLE (fit block, t in [eff, n_fit)) -- for the QQ/ACF mosaics
    y_is = y_fit[eff:]
    pi_is = pi_full[eff:n_fit]
    eta_is = eta_full[eff:n_fit]
    wet_is = y_is > 0
    pit_is = bernoulli_pit_values(pi_is, wet_is.astype(float), rng=np.random.default_rng(seed))
    qr_is = bernoulli_quantile_residuals(pi_is, wet_is.astype(float), rng=np.random.default_rng(seed))
    acf_full_is = acf_at_lags(qr_is, ACF_FULL_LAGS)
    acf_table_is = {l: acf_full_is[l] for l in ACF_TABLE_LAGS}

    # -- VALIDATION (val block) -- for the summary table
    pi_val = pi_full[n_fit:]
    eta_val = eta_full[n_fit:]
    y_val = np.asarray(y_val, dtype=float)
    wet_val = y_val > 0
    dry_val = ~wet_val
    val_pi_mean_wet = float(pi_val[wet_val].mean()) if wet_val.any() else float("nan")
    val_pi_mean_dry = float(pi_val[dry_val].mean()) if dry_val.any() else float("nan")
    val_metrics = in_sample_point_and_crps_metrics(
        dist=GB2LogLinkPhiOnly(), phi_arr=baseline["phi_val"], static=baseline["static"],
        pi_arr=pi_val, y_arr=y_val, n_draws=n_draws, seed=seed)
    val_qr = bernoulli_quantile_residuals(pi_val, wet_val.astype(float),
                                          rng=np.random.default_rng(seed))

    return {
        "variant_key": variant_key, "success": True,
        "pi_dyn_type": type(pi_dyn).__name__,
        "param_names": fit["param_names"], "theta": fit["theta"],
        "loglik_pi_only_fit": fit["loglik"], "runtime_s": time.time() - t0,
        "eff_start": eff, "n_fit": int(n_fit), "n_val": int(len(y_val)),
        # in-sample (fit block) diagnostics
        "n_is": int(len(y_is)), "n_dry_is": int((~wet_is).sum()), "n_wet_is": int(wet_is.sum()),
        "acf_table_is": acf_table_is,
        "acf_full_lags": ACF_FULL_LAGS, "acf_full_values": [acf_full_is[l] for l in ACF_FULL_LAGS],
        "qr_is": qr_is, "pit_is": pit_is, "eta_is": eta_is, "pi_is": pi_is, "y_is": y_is,
        # validation diagnostics
        "val_n_dry": int(dry_val.sum()), "val_n_wet": int(wet_val.sum()),
        "val_pi_mean_dry": val_pi_mean_dry, "val_pi_mean_wet": val_pi_mean_wet,
        "val_rmse_all": val_metrics["rmse_all"], "val_rmse_dry": val_metrics["rmse_dry"],
        "val_rmse_wet": val_metrics["rmse_wet"], "val_crps_all": val_metrics["crps_all"],
        "val_pi": pi_val, "val_eta": eta_val, "val_y": y_val, "val_qr": val_qr,
        **grad_info,
    }


def _save_artifact(out_dir: Path, location: str, variant_key: str, result: dict, baseline: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not result.get("success"):
        (out_dir / "metadata.json").write_text(json.dumps({
            "model_type": "PiDynamicsAlternativeGAS11Validation",
            "location": location, "variant_key": variant_key,
            "validity": "failed", "success": False,
            "message": result.get("message", "unknown failure"),
            "runtime_s": result.get("runtime_s")}, indent=2))
        return

    pd.DataFrame({"parameter": result["param_names"], "value": result["theta"]}).to_csv(
        out_dir / "estimated_parameters.csv", index=False)
    pd.DataFrame({"parameter": result["param_names"], "gradient": result["grad"]}).to_csv(
        out_dir / "gradients.csv", index=False)
    # paths.npz: `qr`/`pit` are the IN-SAMPLE (8-year fit-block) residuals the
    # mosaics consume; `val_*` are the validation-block series behind the
    # summary table.
    np.savez(out_dir / "paths.npz",
             qr=result["qr_is"], pit=result["pit_is"],
             eta_is=result["eta_is"], pi_is=result["pi_is"], y_is=result["y_is"],
             val_pi=result["val_pi"], val_eta=result["val_eta"], val_y=result["val_y"],
             val_qr=result["val_qr"])
    pd.DataFrame({"lag": result["acf_full_lags"], "acf": result["acf_full_values"]}).to_csv(
        out_dir / "acf_400.csv", index=False)

    meta = {
        "model_type": "PiDynamicsAlternativeGAS11Validation",
        "location": location, "variant_key": variant_key,
        "pi_dyn_type": result["pi_dyn_type"],
        "description": _VARIANT_DESCRIPTIONS[variant_key],
        "seasonal": SEASONAL,
        "split_years": SPLIT_YEARS,
        "magnitude_model": "GAS(1,1) phi-only, gas_lags=[1], fit on the 8-year fit block",
        "magnitude_selected_scaling": baseline["selected_scaling"],
        "magnitude_selection_tier": baseline["selection_tier"],
        "eff_start": result["eff_start"], "n_fit": result["n_fit"], "n_val": result["n_val"],
        "loglik_pi_only_fit": result["loglik_pi_only_fit"],
        "grad_norm_inf": result["grad_norm_inf"], "grad_norm_2": result["grad_norm_2"],
        "grad_eps_used": result["grad_eps_used"], "grad_converged": result["grad_converged"],
        "validity": "valid_converged" if result["grad_converged"] else "valid_with_warning",
        "success": True, "runtime_s": result["runtime_s"],
        # in-sample (fit-block) diagnostics -- mosaics
        "is_n_obs": result["n_is"], "is_n_dry": result["n_dry_is"], "is_n_wet": result["n_wet_is"],
        "is_acf_selected_lags": {str(l): result["acf_table_is"][l] for l in ACF_TABLE_LAGS},
        # validation-block diagnostics -- summary table
        "val_n_dry": result["val_n_dry"], "val_n_wet": result["val_n_wet"],
        "val_pi_mean_dry": result["val_pi_mean_dry"], "val_pi_mean_wet": result["val_pi_mean_wet"],
        "val_rmse_all": result["val_rmse_all"], "val_rmse_dry": result["val_rmse_dry"],
        "val_rmse_wet": result["val_rmse_wet"], "val_crps_all": result["val_crps_all"],
        "estimation_note": (
            f"pi_t and the GAS(1,1) magnitude model are both fit ONLY on the "
            f"first {SPLIT_YEARS} calendar years of *_train.csv. val_* metrics are "
            f"computed on the final 2 years, 1-step-ahead, with no re-fit: the "
            f"magnitude varphi path over the validation block comes from "
            f"ZAGASModel.simulate_oos and each variant's pi path from "
            f"eta_path_full(concat(y_fit, y_val)). is_* diagnostics (and the "
            f"QQ/ACF mosaics) use the in-sample fit-block quantile residuals."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_location(location: str, force: bool, n_draws: int) -> None:
    cfg = STATION_REGISTRY[location]
    short = cfg["short"]
    out_base = EXPERIMENT_DIR / short
    logger.info(f"=== {cfg['display']} ({short}) ===")

    loader = StationDataLoader.from_registry(
        location, precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH)
    data = loader.load_all()
    y_train, dates_train = data["y_train"], data["dates_train"]
    y_fit, y_val, cut_label, n_fit = _split_fit_validation(y_train, dates_train)
    logger.info(f"  split @ {cut_label}: fit n={len(y_fit)}  validation n={len(y_val)} "
                f"(wet_val={(np.asarray(y_val) > 0).sum()})")

    mag_meta_path = out_base / "magnitude_gas11" / "metadata.json"
    if mag_meta_path.exists() and not force:
        logger.info("  [magnitude] cached selection found -- reloading (use --force to re-fit).")
        mm = json.loads(mag_meta_path.read_text())
        npz = np.load(out_base / "magnitude_gas11" / "phi_path.npz")
        baseline = {
            "eff_start": int(npz["eff_start"]), "phi_fit": np.asarray(npz["phi_fit"], float),
            "phi_val": np.asarray(npz["phi_val"], float),
            "static": {k: float(v) for k, v in mm["static"].items()},
            "f0_phi": float(npz["f0_phi"]), "n_fit": int(npz["n_fit"]), "n_val": int(mm["n_val"]),
            "selected_scaling": mm["selected_scaling"], "selection_tier": mm["selection_tier"],
            "loglik_fit": mm["loglik_fit"], "aic_fit": mm["aic_fit"],
            "grad_norm_inf": mm["grad_norm_inf"], "validity": mm["validity"],
        }
    else:
        baseline = fit_magnitude(short, y_fit, y_val, n_fit, cut_label, out_base)

    _execution_log({"location": location, "short": short, "model": "magnitude_gas11",
                    "selected_scaling": baseline["selected_scaling"],
                    "selection_tier": baseline["selection_tier"],
                    "loglik_fit": baseline["loglik_fit"], "aic_fit": baseline["aic_fit"],
                    "grad_norm_inf": baseline["grad_norm_inf"], "validity": baseline["validity"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})

    variants = _build_variants(np.concatenate([
        np.full(baseline["eff_start"], baseline["f0_phi"]),
        baseline["phi_fit"], baseline["phi_val"],
    ]))  # full-length phi path (fit block + validation block) for variant D

    for variant_key in VARIANT_KEYS:
        pi_dyn = variants[variant_key]
        out_dir = out_base / variant_key
        if (out_dir / "metadata.json").exists() and not force:
            if json.loads((out_dir / "metadata.json").read_text()).get("success"):
                logger.info(f"  [{variant_key}] cached artifact found -- skipping (use --force).")
                continue
        logger.info(f"  [{variant_key}] fitting on fit block, scoring on validation block...")
        try:
            result = _fit_and_diagnose_one(location, variant_key, pi_dyn,
                                           y_fit, y_val, baseline, n_draws)
        except Exception as exc:
            logger.error(f"  [{variant_key}] FAILED: {exc}")
            logger.error(traceback.format_exc())
            result = {"variant_key": variant_key, "success": False, "message": str(exc), "runtime_s": None}

        _save_artifact(out_dir, location, variant_key, result, baseline)
        _execution_log({"location": location, "short": short, "variant_key": variant_key,
                        "success": result.get("success", False), "runtime_s": result.get("runtime_s"),
                        "val_rmse_all": result.get("val_rmse_all"),
                        "val_crps_all": result.get("val_crps_all"),
                        "grad_norm_inf": result.get("grad_norm_inf"),
                        "message": result.get("message"),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
        if result.get("success"):
            logger.info(f"  [{variant_key}] OK  val_RMSE={result['val_rmse_all']:.4f}  "
                        f"val_CRPS={result['val_crps_all']:.4f}  "
                        f"val pi|dry={result['val_pi_mean_dry']:.4f}  "
                        f"val pi|wet={result['val_pi_mean_wet']:.4f}  "
                        f"||grad||_inf={result['grad_norm_inf']:.3e}  ({result['runtime_s']:.1f}s)")
        else:
            logger.warning(f"  [{variant_key}] NOT SAVED AS VALID: {result.get('message')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pi-dynamics A-D with a GAS(1,1) magnitude model, 8y-fit / 2y-validation split.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--location", type=str, default=None)
    parser.add_argument("--n-draws", type=int, default=500)
    args = parser.parse_args()

    locations = [args.location] if args.location else LOCATION_ORDER
    unknown = [loc for loc in locations if loc not in STATION_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown location(s): {unknown}")

    logger.info(f"Pi-dynamics GAS(1,1) validation experiment: {len(locations)} location(s); "
                f"first {SPLIT_YEARS}y fit / final 2y validation. force={args.force}  n_draws={args.n_draws}")

    n_failed = 0
    for location in locations:
        try:
            run_location(location, force=args.force, n_draws=args.n_draws)
        except Exception as exc:
            n_failed += 1
            logger.error(f"Location {location} FAILED entirely: {exc}")
            logger.error(traceback.format_exc())

    if n_failed >= max(1, len(locations) // 2):
        logger.warning("=" * 60)
        logger.warning(f"ALERT: {n_failed}/{len(locations)} locations failed entirely.")
        logger.warning("=" * 60)

    logger.info("Done. Run generate_pi_dynamics_gas11_summary.py and "
                "generate_pi_dynamics_gas11_mosaics.py to build the deliverables.")


if __name__ == "__main__":
    main()
