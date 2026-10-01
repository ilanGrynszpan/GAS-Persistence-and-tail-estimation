"""
Pi-dynamics alternatives experiment -- GAS(1,1) magnitude variant (2026-09-09).

=============================================================================
SCIENTIFIC QUESTION
=============================================================================

This is the same occurrence-dynamics (pi_t) comparison as
run_pi_dynamics_alternatives.py (2026-07-14), with ONE change to the
underlying model. The full observation model is still the zero-augmented
mixture

    M_t = D_t * Y_t ,   D_t ~ Bernoulli(pi_{t|t-1}) ,   Y_t ~ GB2(varphi, gamma, xi, zeta)

(docs/MODELS.md §3-5). Previously the time-varying scale varphi_{t|t-1} was
frozen from whichever cached Stage 1 phi-only GAS artifact (short lag set
{1,2,3} or seasonal {1,2,3,364,365,366,367}) scored best by CRPS at each
location. Here varphi_{t|t-1} instead follows a freshly-estimated

    GAS(1,1):   varphi_{t+1|t} = omega_varphi + A_varphi * s_t + B_varphi * varphi_{t|t-1}

i.e. the same score-driven recursion (docs/MODELS.md §12) restricted to a
single lag L = {1} -- one score coefficient A_varphi and one persistence
coefficient B_varphi. Everything else is identical to the 2026-07-14
experiment:

  A. existing_ar_seasonal  (pi_dynamics/ar_logistic.py, unmodified)
         eta_t = omega0 + rho*eta_{t-1} + sum_{l in {1,365,366}} omega_l*y_{t-l}
  B. short_lags_ar          (pi_dynamics/ar_logistic_custom_lags.py)
         eta_t = omega0 + rho*eta_{t-1} + sum_{l=1,2,3} omega_l*y_{t-l}
  C. short_lags_noar        (pi_dynamics/ar_logistic_custom_lags.py)
         eta_t = omega0 + sum_{l=1,2,3} omega_l*y_{t-l}
  D. phi_linked             (pi_dynamics/phi_linked.py)
         eta_t = lambda0 + lambda1*varphi_{t|t-1}   (varphi frozen from the GAS(1,1) fit)

All four occurrence models are fit STANDALONE (pi_dynamics/standalone_fit.py)
on the binary wet/dry sequence alone -- the zero-augmented log-likelihood is
additively separable in pi vs. (varphi, xi, gamma, zeta) (docs/MODELS.md §31),
so a standalone fit reaches the same theta_pi optimum as a joint fit. Every
diagnostic is IN-SAMPLE ONLY; this experiment produces no OOS forecast.

=============================================================================
MAGNITUDE MODEL: SCORE-SCALING SELECTION
=============================================================================

The GAS(1,1) phi-only magnitude model is estimated under all three score
scalings (docs/MODELS.md §10): "unit", "diagonal_inverse_fisher",
"inverse_fisher". The one used to freeze varphi_{t|t-1} is selected by:

  1. Gradient screen. An INDEPENDENT central-difference gradient of the
     exact penalised negative log-likelihood is computed at each fit's
     optimum, with the eps-convergence check of
     diagnostics/numerical_diff.numerical_gradient_checked (that module's
     CORRECTNESS NOTE explains why a single step size is unsafe for this
     project's recursive-filter objectives). Preference tiers, most to
     least preferred:
        tier 1 : validity != failed  AND  ||grad||_inf < 1
        tier 2 : validity != failed  AND  ||grad||_inf < 100   ("not in the hundreds")
        tier 3 : validity != failed
        tier 4 : any fit
     The first non-empty tier is used.
  2. AIC tie-break. Within the selected tier, the fit with the lowest AIC
     (= 2k - 2*loglik, k identical across the three scalings, so this is
     equivalently the highest in-sample log-likelihood) is chosen.

All three candidate fits are saved in full (candidates/<scaling>/), not just
the winner, per docs/OPTIMIZATION.md §8 ("do not discard failed runs").

=============================================================================
GRADIENT TABLE (new deliverable)
=============================================================================

For EVERY optimised model in this experiment -- the GAS(1,1) magnitude fit
AND each of the four standalone pi_t fits -- the independent
numerical_gradient_checked gradient of that model's own scalar objective at
its fitted theta is saved (gradients.csv per model, plus grad_norm_inf /
grad_norm_2 / eps_used / converged in metadata.json). The report renders
these as a per-location "gradient of the optimised models" table and a
cross-location gradient-norm summary.

=============================================================================
EXECUTION FLOW  (per location, six locations)
=============================================================================

    load y_train (StationDataLoader)
        |
    for scaling in {unit, diagonal_inverse_fisher, inverse_fisher}:
        fit ZAGASModel(GB2LogLinkPhiOnly, ARLogisticPiDynamics, gas_lags=[1],
                       scaling=scaling, static_params=[xi,gamma,zeta])
        checked-gradient at the optimum; AIC; save candidates/<scaling>/
        |
    select the magnitude scaling (gradient tiers -> AIC), filter() it to
    recover varphi_{t|t-1} in-sample, build the full-length varphi path
        |
    build the four PiDynamics instances (Model D needs the varphi path)
        |
    for each of the four:
        fit_pi_dynamics_standalone(y_train, pi_dyn, seasonal="daily")
        checked-gradient of standalone_neg_loglik at the optimum
        eta/pi over the common window [eff_start=1, T)
        mean(pi_t | y_t=0) vs mean(pi_t | y_t>0)
        randomised Bernoulli PIT / quantile residuals
        ACF at {1,2,3,364,365,366,367} and the full 1..400 profile
        RMSE (all/dry/wet) + mean CRPS vs y_t, magnitude frozen at the
        selected GAS(1,1) varphi path (shared by all four)
        save metadata.json, estimated_parameters.csv, gradients.csv,
             paths.npz, acf_400.csv

Report figures/tables are built separately, from these artifacts only, by
generate_pi_dynamics_gas11_report.py (docs/REPORTING.md §2-3).

=============================================================================
USAGE
=============================================================================

    python run_pi_dynamics_gas11.py [options]

Options:
    --force            Re-estimate even when a valid artifact already exists
    --location NAME    Run only one location (exact STATION_REGISTRY key)
    --n-draws N        Monte Carlo draws per time step for CRPS (default 500)

Per docs/EXECUTION.md: this script must not be run until the user has
reviewed the code and explicitly authorised execution.

=============================================================================
ARTIFACT STRUCTURE
=============================================================================

artifacts/pi_dynamics_gas11_experiment/<location_short>/
    magnitude_gas11/
        metadata.json             selected scaling, loglik, AIC, gradient norms, validity
        estimated_parameters.csv  fitted GAS(1,1)+pi theta of the selected fit
        gradients.csv             checked numerical gradient at the optimum
        standard_errors.csv       BFGS inverse-Hessian SEs (if usable)
        scaling_selection.csv     one row per scaling: loglik, AIC, grad norms, tier, selected
        phi_path.npz              phi_is (windowed), eff_start, f0_phi
        candidates/<scaling>/     full metadata/params/gradients for every scaling tried
    <variant_key>/                one per pi_t alternative (A..D)
        metadata.json             occurrence + RMSE/CRPS metrics, gradient norms, validity
        estimated_parameters.csv  fitted pi_dynamics theta
        gradients.csv             checked numerical gradient at the optimum
        paths.npz                 windowed + full-length eta/pi/y/PIT/QR arrays
        acf_400.csv               lag, acf

artifacts/pi_dynamics_gas11_experiment/execution_log.jsonl   one JSON record per model
pi_dynamics_gas11.log                                        human-readable run log
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

# -- Project root on path ----------------------------------------------------
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Same hard-coded application data locations as run_pi_dynamics_alternatives.py
# (docs/DATA.md documents these). Kept as a local copy rather than importing
# that script so this experiment stays independent of it.
PRECIP_DIR = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR   = ROOT / "data" / "input" / "ERA5"
ENSO_PATH  = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
ARTIFACTS_DIR  = ROOT / "artifacts"
EXPERIMENT_DIR = ARTIFACTS_DIR / "pi_dynamics_gas11_experiment"

from constants import GAS_SEASONAL_LAGS
from data.station_loader import STATION_REGISTRY, StationDataLoader
from distributions import GB2LogLinkPhiOnly
from models.za_gas_model import ZAGASModel
from pi_dynamics.ar_logistic import ARLogisticPiDynamics
from pi_dynamics.ar_logistic_custom_lags import ARLogisticCustomLagsPiDynamics
from pi_dynamics.phi_linked import PhiLinkedPiDynamics, build_full_length_phi_path
from pi_dynamics.standalone_fit import (
    fit_pi_dynamics_standalone,
    eta_path_full,
    standalone_neg_loglik,
)
from diagnostics.numerical_diff import numerical_gradient_checked
from diagnostics.information import aic as _aic, bic as _bic
from diagnostics.occurrence_pit import bernoulli_pit_values, bernoulli_quantile_residuals
from diagnostics.acf import acf_at_lags
from diagnostics.pi_dynamics_eval import in_sample_point_and_crps_metrics

# -- The six locations already used in reports/multi_location/ --------------
LOCATION_ORDER = [
    "BELO HORIZONTE",
    "CRUZEIRO DO SUL (ACRE)",
    "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)",
    "MANAUS",
    "SALVADOR",
]

# GAS(1,1): the score-driven scale recursion restricted to a single lag.
GAS11_LAGS = [1]

# Estimated under every score scaling (docs/MODELS.md §10); the one used to
# freeze varphi_{t|t-1} is picked by the gradient-tier / AIC rule documented
# in this module's docstring.
SCALINGS = ["unit", "diagonal_inverse_fisher", "inverse_fisher"]

# Gradient-tier thresholds for magnitude-model scaling selection.
GRAD_TIER_STRICT = 1.0     # "<1 preferable"
GRAD_TIER_LOOSE  = 100.0   # "at least not in the hundreds"

# ACF table lags (identical to GAS_SEASONAL_LAGS["daily"]; reused, not re-typed).
ACF_TABLE_LAGS = list(GAS_SEASONAL_LAGS["daily"])
ACF_FULL_LAGS  = list(range(1, 401))

SEASONAL = "daily"

# -- Logging ---------------------------------------------------------------------
log_file = ROOT / "pi_dynamics_gas11.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(log_file, mode="a"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pi_dynamics_gas11")


def _execution_log(record: dict) -> None:
    """Append one machine-readable record per completed/failed model fit."""
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENT_DIR / "execution_log.jsonl", "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# -----------------------------------------------------------------------------
# Independent checked gradient at a fitted optimum
# -----------------------------------------------------------------------------

def _checked_gradient(objective, theta: np.ndarray) -> dict:
    """
    numerical_gradient_checked wrapper returning a JSON-friendly summary.

    `objective` is the model's own scalar objective (penalised negative
    log-likelihood) as a function of theta -- exactly what was minimised.
    """
    out = numerical_gradient_checked(objective, np.asarray(theta, dtype=float))
    grad = np.asarray(out["grad"], dtype=float)
    return {
        "grad": grad,
        "grad_norm_inf": float(np.max(np.abs(grad))) if grad.size else float("nan"),
        "grad_norm_2": float(np.linalg.norm(grad)) if grad.size else float("nan"),
        "grad_eps_used": float(out["eps_used"]),
        "grad_converged": bool(out["converged"]),
        "grad_trace": [[float(e), float(m)] for e, m in out["trace"]],
    }


# -----------------------------------------------------------------------------
# Step 1: fit the phi-only GAS(1,1) magnitude model under each scaling and
#         select the one that freezes varphi_{t|t-1}
# -----------------------------------------------------------------------------

def _build_magnitude_model(scaling: str) -> ZAGASModel:
    """
    Phi-only GAS(1,1), no covariates, ARLogistic occurrence dynamics -- the
    GAS(1,1) analogue of a Stage 1 phi-only baseline (docs/MODELS.md §12,
    §26), restricted to a single GAS lag.
    """
    return ZAGASModel(
        distribution=GB2LogLinkPhiOnly(),
        pi_dynamics=ARLogisticPiDynamics(),
        seasonal=SEASONAL,
        gas_lags=GAS11_LAGS,
        scaling=scaling,
        static_params=["xi", "gamma", "zeta"],
    )


def _fit_one_magnitude_scaling(scaling: str, y_train: np.ndarray) -> dict:
    """Fit the GAS(1,1) magnitude model for one scaling; attach checked gradient + AIC."""
    mdl = _build_magnitude_model(scaling)
    fit = mdl.fit(y_train, polish=True)

    # Independent gradient of the exact objective at the optimum:
    #   -mdl.loglik(theta, y) == mdl._run_filter(theta, y) == neg-loglik + penalty
    grad_info = _checked_gradient(lambda th: -mdl.loglik(th, y_train), fit["theta"])

    n_obs_eff = int(len(y_train) - mdl.max_lag)
    ll = float(fit["loglik"])
    k = int(mdl.n_params)
    return {
        "scaling": scaling,
        "model": mdl,
        "theta": fit["theta"],
        "param_names": fit["param_names"],
        "loglik": ll,
        "n_params": k,
        "n_obs_eff": n_obs_eff,
        "aic": float(_aic(ll, k)),
        "bic": float(_bic(ll, k, n_obs_eff)),
        "validity": fit.get("validity", "unknown"),
        "success": bool(fit.get("success", False)),
        "message": str(fit.get("message", "")),
        "n_iter": int(fit.get("n_iter", 0)),
        "scipy_grad_norm_inf": float(fit.get("grad_norm_inf", float("nan"))),
        "std_errors": fit.get("std_errors"),
        "se_quality": fit.get("se_quality", "unavailable"),
        "hess_inv": fit.get("hess_inv"),
        "runtime_s": float(fit.get("runtime_s", float("nan"))),
        "bound_diagnostics": fit.get("bound_diagnostics", {}),
        **grad_info,
    }


def _select_magnitude(cands: list[dict]) -> tuple[dict, str]:
    """
    Apply the gradient-tier -> AIC selection rule (see module docstring).
    Returns (selected_candidate, tier_label).
    """
    def _tier(c: dict) -> int:
        ok = c["validity"] != "failed"
        g = c["grad_norm_inf"]
        if ok and np.isfinite(g) and g < GRAD_TIER_STRICT:
            return 1
        if ok and np.isfinite(g) and g < GRAD_TIER_LOOSE:
            return 2
        if ok:
            return 3
        return 4

    tiers = {c["scaling"]: _tier(c) for c in cands}
    best_tier = min(tiers.values())
    pool = [c for c in cands if tiers[c["scaling"]] == best_tier]
    selected = min(pool, key=lambda c: c["aic"])
    tier_label = {
        1: "grad_inf<1", 2: "grad_inf<100", 3: "not_failed", 4: "fallback_any",
    }[best_tier]
    for c in cands:
        c["selection_tier"] = tiers[c["scaling"]]
    return selected, tier_label


def _save_magnitude_candidate(out_dir: Path, cand: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"parameter": cand["param_names"], "value": cand["theta"]}).to_csv(
        out_dir / "estimated_parameters.csv", index=False
    )
    pd.DataFrame({"parameter": cand["param_names"], "gradient": cand["grad"]}).to_csv(
        out_dir / "gradients.csv", index=False
    )
    if cand.get("std_errors") is not None:
        pd.DataFrame({
            "parameter": cand["param_names"],
            "std_error": cand["std_errors"],
            "se_quality": cand["se_quality"],
        }).to_csv(out_dir / "standard_errors.csv", index=False)
    meta = {k: cand[k] for k in (
        "scaling", "loglik", "n_params", "n_obs_eff", "aic", "bic", "validity",
        "success", "message", "n_iter", "scipy_grad_norm_inf", "grad_norm_inf",
        "grad_norm_2", "grad_eps_used", "grad_converged", "runtime_s",
        "selection_tier", "se_quality",
    )}
    meta["bound_diagnostics"] = cand.get("bound_diagnostics", {})
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))


def fit_magnitude_gas11(short: str, y_train: np.ndarray, out_base: Path, force: bool) -> dict:
    """
    Fit + select the GAS(1,1) magnitude model for one location. Returns a
    `baseline` dict with the same shape used downstream:
        {eff_start, phi_is, static, y_eff, f0_phi, selected_scaling, ...}
    """
    mag_dir = out_base / "magnitude_gas11"
    sel_path = mag_dir / "metadata.json"

    cands = []
    for scaling in SCALINGS:
        cand_dir = mag_dir / "candidates" / scaling
        logger.info(f"  [magnitude/{scaling}] fitting GAS(1,1) phi-only...")
        cand = _fit_one_magnitude_scaling(scaling, y_train)
        cands.append(cand)
        logger.info(
            f"  [magnitude/{scaling}] loglik={cand['loglik']:.2f}  AIC={cand['aic']:.2f}  "
            f"||grad||_inf={cand['grad_norm_inf']:.3e} (scipy {cand['scipy_grad_norm_inf']:.3e})  "
            f"converged={cand['grad_converged']}  validity={cand['validity']}"
        )

    selected, tier_label = _select_magnitude(cands)
    for cand in cands:
        _save_magnitude_candidate(mag_dir / "candidates" / cand["scaling"], cand)

    logger.info(
        f"  [magnitude] SELECTED scaling='{selected['scaling']}' via tier '{tier_label}' "
        f"(AIC={selected['aic']:.2f}, ||grad||_inf={selected['grad_norm_inf']:.3e})"
    )

    # Filter the selected fit to recover varphi_{t|t-1} in-sample.
    mdl = selected["model"]
    paths = mdl.filter(selected["theta"], y_train)
    if not paths:
        raise RuntimeError(
            f"Filtering the selected GAS(1,1) magnitude model failed at {short} "
            f"(T <= max_lag?)."
        )
    eff_start = int(paths["eff_start"])
    phi_is = np.asarray(paths["phi"], dtype=float)
    # phi_is[0] is the filter's warm-up constant f0_phi (max_lag=1 seeds
    # f_arr[:2] = f0_phi and updates start at t=1), reused directly to pad
    # t < eff_start, exactly as run_pi_dynamics_alternatives.py does.
    f0_phi = float(phi_is[0])
    static = {k: float(v) for k, v in paths["static"].items()}

    # -- Save the selected magnitude model's own artifacts ------------------
    mag_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"parameter": selected["param_names"], "value": selected["theta"]}).to_csv(
        mag_dir / "estimated_parameters.csv", index=False
    )
    pd.DataFrame({"parameter": selected["param_names"], "gradient": selected["grad"]}).to_csv(
        mag_dir / "gradients.csv", index=False
    )
    if selected.get("std_errors") is not None:
        pd.DataFrame({
            "parameter": selected["param_names"],
            "std_error": selected["std_errors"],
            "se_quality": selected["se_quality"],
        }).to_csv(mag_dir / "standard_errors.csv", index=False)

    pd.DataFrame([{
        "scaling": c["scaling"],
        "loglik": c["loglik"],
        "aic": c["aic"],
        "bic": c["bic"],
        "grad_norm_inf": c["grad_norm_inf"],
        "grad_norm_2": c["grad_norm_2"],
        "grad_converged": c["grad_converged"],
        "scipy_grad_norm_inf": c["scipy_grad_norm_inf"],
        "validity": c["validity"],
        "selection_tier": c["selection_tier"],
        "selected": (c["scaling"] == selected["scaling"]),
    } for c in cands]).to_csv(mag_dir / "scaling_selection.csv", index=False)

    np.savez(
        mag_dir / "phi_path.npz",
        phi_is=phi_is, eff_start=np.array(eff_start), f0_phi=np.array(f0_phi),
    )

    meta = {
        "model_type": "ZAGASModel_GAS11_phi_only",
        "location_short": short,
        "gas_lags": GAS11_LAGS,
        "selected_scaling": selected["scaling"],
        "selection_tier": tier_label,
        "scalings_tried": SCALINGS,
        "grad_tier_strict": GRAD_TIER_STRICT,
        "grad_tier_loose": GRAD_TIER_LOOSE,
        "loglik": selected["loglik"],
        "n_params": selected["n_params"],
        "n_obs_eff": selected["n_obs_eff"],
        "aic": selected["aic"],
        "bic": selected["bic"],
        "validity": selected["validity"],
        "success": selected["success"],
        "message": selected["message"],
        "n_iter": selected["n_iter"],
        "grad_norm_inf": selected["grad_norm_inf"],
        "grad_norm_2": selected["grad_norm_2"],
        "grad_eps_used": selected["grad_eps_used"],
        "grad_converged": selected["grad_converged"],
        "scipy_grad_norm_inf": selected["scipy_grad_norm_inf"],
        "se_quality": selected["se_quality"],
        "runtime_s": selected["runtime_s"],
        "eff_start": eff_start,
        "f0_phi": f0_phi,
        "static": static,
        "bound_diagnostics": selected.get("bound_diagnostics", {}),
    }
    sel_path.write_text(json.dumps(meta, indent=2, default=str))

    return {
        "eff_start": eff_start,
        "phi_is": phi_is,
        "static": static,
        "y_eff": np.asarray(paths["y_eff"], dtype=float),
        "f0_phi": f0_phi,
        "selected_scaling": selected["scaling"],
        "selection_tier": tier_label,
        "loglik": selected["loglik"],
        "aic": selected["aic"],
        "grad_norm_inf": selected["grad_norm_inf"],
        "validity": selected["validity"],
        "param_names": selected["param_names"],
    }


# -----------------------------------------------------------------------------
# Step 2: the four pi-dynamics variants for one location
# -----------------------------------------------------------------------------

def _build_variants(phi_path_full: np.ndarray) -> dict:
    return {
        "existing_ar_seasonal": ARLogisticPiDynamics(),
        "short_lags_ar":        ARLogisticCustomLagsPiDynamics(lags=[1, 2, 3], include_ar=True),
        "short_lags_noar":      ARLogisticCustomLagsPiDynamics(lags=[1, 2, 3], include_ar=False),
        "phi_linked":           PhiLinkedPiDynamics(phi_path=phi_path_full),
    }


_VARIANT_DESCRIPTIONS = {
    "existing_ar_seasonal": "eta_t = omega0 + rho*eta_{t-1} + sum_{l in {1,365,366}} omega_l*y_{t-l}  (unmodified pi_dynamics/ar_logistic.py)",
    "short_lags_ar":        "eta_t = omega0 + rho*eta_{t-1} + sum_{l=1,2,3} omega_l*y_{t-l}",
    "short_lags_noar":      "eta_t = omega0 + sum_{l=1,2,3} omega_l*y_{t-l}",
    "phi_linked":           "eta_t = lambda0 + lambda1*varphi_{t|t-1}  (varphi_{t|t-1} frozen from this location's selected GAS(1,1) magnitude fit)",
}


# -----------------------------------------------------------------------------
# Step 3: fit + diagnose one (location, variant) pair
# -----------------------------------------------------------------------------

def _fit_and_diagnose_one(
    location: str,
    variant_key: str,
    pi_dyn,
    y_train: np.ndarray,
    baseline: dict,
    n_draws: int,
    seed: int = 42,
) -> dict:
    t0 = time.time()
    fit = fit_pi_dynamics_standalone(y_train, pi_dyn=pi_dyn, seasonal=SEASONAL, verbose=False)

    eff_start = baseline["eff_start"]
    y_w = y_train[eff_start:]
    if not fit["success"]:
        return {
            "variant_key": variant_key, "success": False,
            "message": "standalone pi_dynamics fit failed (non-finite objective/parameters)",
            "runtime_s": time.time() - t0,
        }

    # Independent checked gradient of the exact standalone objective at theta_hat.
    grad_info = _checked_gradient(
        lambda th: standalone_neg_loglik(th, pi_dyn, y_train, SEASONAL), fit["theta"]
    )

    eta_full = eta_path_full(pi_dyn, fit["theta"], y_train, SEASONAL)
    pi_full  = expit(eta_full)
    eta_w = eta_full[eff_start:]
    pi_w  = pi_full[eff_start:]

    # -- Occurrence diagnostics (common window) ----------------------------
    wet = y_w > 0
    dry = ~wet
    pi_mean_wet = float(pi_w[wet].mean()) if wet.any() else float("nan")
    pi_mean_dry = float(pi_w[dry].mean()) if dry.any() else float("nan")

    z_w = wet.astype(float)
    pit = bernoulli_pit_values(pi_w, z_w, rng=np.random.default_rng(seed))
    qr  = bernoulli_quantile_residuals(pi_w, z_w, rng=np.random.default_rng(seed))

    acf_full  = acf_at_lags(qr, ACF_FULL_LAGS)
    acf_table = {l: acf_full[l] for l in ACF_TABLE_LAGS}

    # -- RMSE / CRPS vs y_t, magnitude frozen at the selected GAS(1,1) path
    eval_metrics = in_sample_point_and_crps_metrics(
        dist=GB2LogLinkPhiOnly(),
        phi_arr=baseline["phi_is"],
        static=baseline["static"],
        pi_arr=pi_w,
        y_arr=y_w,
        n_draws=n_draws,
        seed=seed,
    )

    return {
        "variant_key": variant_key,
        "success": True,
        "pi_dyn_type": type(pi_dyn).__name__,
        "param_names": fit["param_names"],
        "theta": fit["theta"],
        "loglik_pi_only": fit["loglik"],
        "runtime_s": time.time() - t0,
        "eff_start": eff_start,
        "n_obs_window": int(len(y_w)),
        "n_dry_window": int(dry.sum()),
        "n_wet_window": int(wet.sum()),
        "pi_mean_dry": pi_mean_dry,
        "pi_mean_wet": pi_mean_wet,
        "acf_table": acf_table,
        "acf_full_lags": ACF_FULL_LAGS,
        "acf_full_values": [acf_full[l] for l in ACF_FULL_LAGS],
        "eta_w": eta_w, "pi_w": pi_w, "y_w": y_w, "pit": pit, "qr": qr,
        "eta_full": eta_full, "pi_full": pi_full,
        **eval_metrics,
        **grad_info,
    }


def _save_variant_artifact(out_dir: Path, location: str, variant_key: str, result: dict, baseline: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not result.get("success"):
        (out_dir / "metadata.json").write_text(json.dumps({
            "model_type": "PiDynamicsAlternativeGAS11",
            "location": location, "variant_key": variant_key,
            "validity": "failed", "success": False,
            "message": result.get("message", "unknown failure"),
            "runtime_s": result.get("runtime_s"),
        }, indent=2))
        return

    pd.DataFrame({"parameter": result["param_names"], "value": result["theta"]}).to_csv(
        out_dir / "estimated_parameters.csv", index=False
    )
    pd.DataFrame({"parameter": result["param_names"], "gradient": result["grad"]}).to_csv(
        out_dir / "gradients.csv", index=False
    )
    np.savez(
        out_dir / "paths.npz",
        eta_w=result["eta_w"], pi_w=result["pi_w"], y_w=result["y_w"],
        pit=result["pit"], qr=result["qr"],
        eta_full=result["eta_full"], pi_full=result["pi_full"],
    )
    pd.DataFrame({"lag": result["acf_full_lags"], "acf": result["acf_full_values"]}).to_csv(
        out_dir / "acf_400.csv", index=False
    )

    meta = {
        "model_type": "PiDynamicsAlternativeGAS11",
        "location": location,
        "variant_key": variant_key,
        "pi_dyn_type": result["pi_dyn_type"],
        "description": _VARIANT_DESCRIPTIONS[variant_key],
        "seasonal": SEASONAL,
        "magnitude_model": "GAS(1,1) phi-only, gas_lags=[1]",
        "magnitude_selected_scaling": baseline["selected_scaling"],
        "magnitude_selection_tier": baseline["selection_tier"],
        "magnitude_loglik": baseline["loglik"],
        "magnitude_aic": baseline["aic"],
        "magnitude_grad_norm_inf": baseline["grad_norm_inf"],
        "eff_start": result["eff_start"],
        "n_obs_window": result["n_obs_window"],
        "n_dry_window": result["n_dry_window"],
        "n_wet_window": result["n_wet_window"],
        "loglik_pi_only": result["loglik_pi_only"],
        "validity": "valid_converged" if result["grad_converged"] else "valid_with_warning",
        "success": True,
        "runtime_s": result["runtime_s"],
        "pi_mean_dry": result["pi_mean_dry"],
        "pi_mean_wet": result["pi_mean_wet"],
        "grad_norm_inf": result["grad_norm_inf"],
        "grad_norm_2": result["grad_norm_2"],
        "grad_eps_used": result["grad_eps_used"],
        "grad_converged": result["grad_converged"],
        "acf_selected_lags": {str(l): result["acf_table"][l] for l in ACF_TABLE_LAGS},
        "rmse_all": result["rmse_all"],
        "rmse_dry": result["rmse_dry"],
        "rmse_wet": result["rmse_wet"],
        "crps_all": result["crps_all"],
        "estimation_note": (
            "pi_t fit standalone (pi_dynamics/standalone_fit.py) on the binary "
            "occurrence sequence alone. RMSE/CRPS use this model's pi_t with the "
            "FROZEN GAS(1,1) magnitude path (varphi_{t|t-1}, xi, gamma, zeta) from "
            f"this location's selected magnitude fit (scaling={baseline['selected_scaling']}) "
            "-- identical across all four pi-dynamics alternatives here, so RMSE/CRPS "
            "differences are attributable to the occurrence dynamics alone. All "
            "diagnostics are in-sample only."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

VARIANT_KEYS = ["existing_ar_seasonal", "short_lags_ar", "short_lags_noar", "phi_linked"]


def run_location(location: str, force: bool, n_draws: int) -> None:
    cfg = STATION_REGISTRY[location]
    short = cfg["short"]
    out_base = EXPERIMENT_DIR / short

    logger.info(f"=== {cfg['display']} ({short}) ===")

    loader = StationDataLoader.from_registry(
        location, precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH,
    )
    y_train = loader.load_all()["y_train"]

    # -- Step 1: GAS(1,1) magnitude model + scaling selection --------------
    mag_meta_path = out_base / "magnitude_gas11" / "metadata.json"
    if mag_meta_path.exists() and not force:
        logger.info("  [magnitude] cached selection found -- reloading (use --force to re-fit).")
        mag_meta = json.loads(mag_meta_path.read_text())
        npz = np.load(out_base / "magnitude_gas11" / "phi_path.npz")
        baseline = {
            "eff_start": int(npz["eff_start"]),
            "phi_is": np.asarray(npz["phi_is"], dtype=float),
            "static": {k: float(v) for k, v in mag_meta["static"].items()},
            "y_eff": y_train[int(npz["eff_start"]):],
            "f0_phi": float(npz["f0_phi"]),
            "selected_scaling": mag_meta["selected_scaling"],
            "selection_tier": mag_meta["selection_tier"],
            "loglik": mag_meta["loglik"],
            "aic": mag_meta["aic"],
            "grad_norm_inf": mag_meta["grad_norm_inf"],
            "validity": mag_meta["validity"],
            "param_names": None,
        }
    else:
        baseline = fit_magnitude_gas11(short, y_train, out_base, force)

    _execution_log({
        "location": location, "short": short, "model": "magnitude_gas11",
        "selected_scaling": baseline["selected_scaling"],
        "selection_tier": baseline["selection_tier"],
        "loglik": baseline["loglik"], "aic": baseline["aic"],
        "grad_norm_inf": baseline["grad_norm_inf"], "validity": baseline["validity"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    phi_path_full = build_full_length_phi_path(
        phi_is=baseline["phi_is"], eff_start=baseline["eff_start"],
        f0_phi=baseline["f0_phi"], total_length=len(y_train),
    )
    variants = _build_variants(phi_path_full)

    # -- Step 2/3: the four occurrence-dynamics alternatives --------------
    for variant_key in VARIANT_KEYS:
        pi_dyn = variants[variant_key]
        out_dir = out_base / variant_key
        meta_path = out_dir / "metadata.json"
        if meta_path.exists() and not force:
            existing = json.loads(meta_path.read_text())
            if existing.get("success"):
                logger.info(f"  [{variant_key}] cached artifact found -- skipping (use --force).")
                continue

        logger.info(f"  [{variant_key}] fitting...")
        try:
            result = _fit_and_diagnose_one(
                location=location, variant_key=variant_key, pi_dyn=pi_dyn,
                y_train=y_train, baseline=baseline, n_draws=n_draws,
            )
        except Exception as exc:
            logger.error(f"  [{variant_key}] FAILED with exception: {exc}")
            logger.error(traceback.format_exc())
            result = {"variant_key": variant_key, "success": False, "message": str(exc), "runtime_s": None}

        _save_variant_artifact(out_dir, location, variant_key, result, baseline)
        _execution_log({
            "location": location, "short": short, "variant_key": variant_key,
            "success": result.get("success", False),
            "runtime_s": result.get("runtime_s"),
            "loglik_pi_only": result.get("loglik_pi_only"),
            "grad_norm_inf": result.get("grad_norm_inf"),
            "grad_converged": result.get("grad_converged"),
            "rmse_all": result.get("rmse_all"), "crps_all": result.get("crps_all"),
            "message": result.get("message"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        if result.get("success"):
            logger.info(
                f"  [{variant_key}] OK  loglik={result['loglik_pi_only']:.2f}  "
                f"pi|dry={result['pi_mean_dry']:.4f}  pi|wet={result['pi_mean_wet']:.4f}  "
                f"RMSE_all={result['rmse_all']:.4f}  CRPS_all={result['crps_all']:.4f}  "
                f"||grad||_inf={result['grad_norm_inf']:.3e}  ({result['runtime_s']:.1f}s)"
            )
        else:
            logger.warning(f"  [{variant_key}] NOT SAVED AS VALID: {result.get('message')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pi-dynamics alternatives with a GAS(1,1) magnitude model, six locations."
    )
    parser.add_argument("--force", action="store_true", help="Re-estimate even when a valid artifact already exists.")
    parser.add_argument("--location", type=str, default=None, help="Run only this location (exact STATION_REGISTRY key).")
    parser.add_argument("--n-draws", type=int, default=500, help="Monte Carlo draws per time step for CRPS.")
    args = parser.parse_args()

    locations = [args.location] if args.location else LOCATION_ORDER
    unknown = [loc for loc in locations if loc not in STATION_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown location(s): {unknown}. Available: {list(STATION_REGISTRY)}")

    logger.info(
        f"Pi-dynamics GAS(1,1) experiment: {len(locations)} location(s); per location "
        f"3 magnitude scalings + 4 occurrence variants. force={args.force}  n_draws={args.n_draws}"
    )

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
        logger.warning(f"ALERT: {n_failed}/{len(locations)} locations failed entirely. "
                       f"Check pi_dynamics_gas11.log for details.")
        logger.warning("=" * 60)

    logger.info("Done. Run generate_pi_dynamics_gas11_report.py to build the report.")


if __name__ == "__main__":
    main()
