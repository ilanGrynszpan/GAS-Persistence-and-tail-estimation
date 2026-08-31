"""
Pi-dynamics alternatives experiment runner (2026-07-14).

=============================================================================
SCIENTIFIC QUESTION
=============================================================================

Stage 1 of the thesis pipeline (docs/MODELS.md §26, §12) always uses
ARLogisticPiDynamics (pi_dynamics/ar_logistic.py) for the occurrence
probability pi_t = P(y_t>0 | F_{t-1}). This experiment asks: within the
basic Stage 1 setting (GAS(p,q), no covariates, phi the only time-varying
positive-part parameter), does the occurrence probability itself respond
differently to alternative eta_t specifications? Four are compared, all
evaluated in-sample only (no OOS forecast anywhere in this script):

  A. existing_ar_seasonal  (pi_dynamics/ar_logistic.py, unmodified)
         eta_t = omega0 + rho*eta_{t-1} + sum_{l in {1,365,366}} omega_l*y_{t-l}
  B. short_lags_ar          (pi_dynamics/ar_logistic_custom_lags.py)
         eta_t = omega0 + rho*eta_{t-1} + sum_{l=1,2,3} omega_l*y_{t-l}
  C. short_lags_noar        (pi_dynamics/ar_logistic_custom_lags.py)
         eta_t = omega0 + sum_{l=1,2,3} omega_l*y_{t-l}
  D. phi_linked             (pi_dynamics/phi_linked.py)
         eta_t = lambda0 + lambda1*phi_{t|t-1}

All four are fit STANDALONE (pi_dynamics/standalone_fit.py) -- i.e. by
maximising only the occurrence term of the zero-augmented log-likelihood,
independently of any positive-part (GB2) fit. This is not an approximation:
the ZA log-likelihood is additively separable in pi vs. (phi,xi,gamma,zeta)
(see docs/MODELS.md §31, pi_dynamics/standalone_fit.py, models/static_gb2.py
for the same argument used elsewhere in this codebase), so a standalone fit
and a joint fit reach the same theta_pi optimum; fitting all four the same
way keeps the comparison uniform and much cheaper than four full GB2 refits.

Model D needs a frozen phi_{t|t-1} path. This is taken from the location's
best-by-CRPS phi-only Stage 1 GAS(p,q) artifact (one of the six
stage1_phi_{short,seasonal}_{unit,diagfi,fullfi} models already estimated
and cached under artifacts/<run_id>/stage1/ -- see the "reuse artifacts"
principle, docs/EXECUTION.md §17). That same frozen (phi_t, xi, gamma,
zeta) path is then paired with EACH of the four pi_t alternatives to score
RMSE/CRPS on y_t itself (diagnostics/pi_dynamics_eval.py) -- so any
difference in those scores is attributable to the occurrence dynamics
alone, not to a refit magnitude model.

=============================================================================
EXECUTION FLOW
=============================================================================

For each of the six locations already used in reports/multi_location/:

    load y_train (StationDataLoader)
        |
    select best-by-CRPS phi-only Stage 1 artifact for this location
        |
    reconstruct ZAGASModel(phi-only) from its saved theta, call .filter()
    to recover phi_{t|t-1} (in-sample) and eff_start = max(gas_lags)
        |
    build the four PiDynamics instances (Model D needs the phi path above)
        |
    for each of the four:
        fit_pi_dynamics_standalone(y_train, pi_dyn, seasonal="daily")
        eta_path_full(...) -> pi_t over the WHOLE series
        restrict to the common window [eff_start, T) for a fair comparison
        (this is also the only window in which Model D's phi-linked eta is
        genuinely data-driven rather than a constant warm-up value -- see
        pi_dynamics/phi_linked.py's module docstring)
            |
        mean(pi_t | y_t=0) vs mean(pi_t | y_t>0)
        randomised Bernoulli PIT / quantile residuals (diagnostics/occurrence_pit.py)
        ACF at lags 1,2,3,364,365,366,367 and the full 1..400 profile (diagnostics/acf.py)
        RMSE (all/dry/wet) and mean CRPS against y_t, using the frozen phi_t
        path shared by all four models (diagnostics/pi_dynamics_eval.py)
            |
        save metadata.json, estimated_parameters.csv, paths.npz, acf_400.csv

Report figures/tables are generated separately, from these saved artifacts
only, by generate_pi_dynamics_report.py (docs/REPORTING.md §2-3: reports
never re-run optimisation).

=============================================================================
USAGE
=============================================================================

    python run_pi_dynamics_alternatives.py [options]

Options:
    --force            Re-estimate even when a valid artifact already exists
    --location NAME    Run only one location (exact STATION_REGISTRY key)
    --n-draws N        Monte Carlo draws per time step for CRPS (default 500)

Per docs/EXECUTION.md: this script must not be run until the user has
reviewed the code and explicitly authorised execution.

=============================================================================
ARTIFACT STRUCTURE
=============================================================================

artifacts/pi_dynamics_experiment/<location_short>/<variant_key>/
    metadata.json             configuration, validity, occurrence + RMSE/CRPS metrics
    estimated_parameters.csv  fitted pi_dynamics theta
    paths.npz                 windowed + full-length eta/pi/y/PIT/QR/ACF-400 arrays
    acf_400.csv               lag, acf  (reusable outside the report, per REPORTING.md §19)

artifacts/pi_dynamics_experiment/execution_log.jsonl   one JSON record per model
pi_dynamics_alternatives.log                            human-readable run log
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

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRECIP_DIR = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR   = ROOT / "data" / "input" / "ERA5"
ENSO_PATH  = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
ARTIFACTS_DIR    = ROOT / "artifacts"
EXPERIMENT_DIR   = ARTIFACTS_DIR / "pi_dynamics_experiment"

from constants import GAS_SEASONAL_LAGS
from data.station_loader import STATION_REGISTRY, StationDataLoader
from distributions import GB2LogLinkPhiOnly
from models.za_gas_model import ZAGASModel
from pi_dynamics.ar_logistic import ARLogisticPiDynamics
from pi_dynamics.ar_logistic_custom_lags import ARLogisticCustomLagsPiDynamics
from pi_dynamics.phi_linked import PhiLinkedPiDynamics, build_full_length_phi_path
from pi_dynamics.standalone_fit import fit_pi_dynamics_standalone, eta_path_full
from diagnostics.occurrence_pit import bernoulli_pit_values, bernoulli_quantile_residuals
from diagnostics.acf import acf_at_lags
from diagnostics.pi_dynamics_eval import in_sample_point_and_crps_metrics

# ── The six locations already used in reports/multi_location/ ───────────────
LOCATION_ORDER = [
    "BELO HORIZONTE",
    "CRUZEIRO DO SUL (ACRE)",
    "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)",
    "MANAUS",
    "SALVADOR",
]

# The six phi-only Stage 1 baseline variants already cached per location
# (short/seasonal GAS lags x unit/diagonal_inverse_fisher/inverse_fisher
# scaling -- see docs/MODELS.md §12, §14). The best-by-CRPS one supplies
# the frozen phi_{t|t-1} path for Model D and the shared magnitude model
# used to score RMSE/CRPS for all four pi_t alternatives.
BASELINE_PHI_VARIANTS = [
    "stage1_phi_short_unit", "stage1_phi_short_diagfi", "stage1_phi_short_fullfi",
    "stage1_phi_seasonal_unit", "stage1_phi_seasonal_diagfi", "stage1_phi_seasonal_fullfi",
]

# ACF table lags requested for this experiment (identical to
# GAS_SEASONAL_LAGS["daily"] -- reused rather than re-typed).
ACF_TABLE_LAGS = list(GAS_SEASONAL_LAGS["daily"])
ACF_FULL_LAGS  = list(range(1, 401))

SEASONAL = "daily"

# ── Logging ───────────────────────────────────────────────────────────────────
log_file = ROOT / "pi_dynamics_alternatives.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("pi_dynamics_alternatives")


def _execution_log(record: dict) -> None:
    """Append one machine-readable record per completed/failed model fit."""
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENT_DIR / "execution_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: select the frozen phi-only Stage 1 baseline for a location
# ─────────────────────────────────────────────────────────────────────────────

def _select_baseline_phi_model(run_dir: Path) -> tuple[str, dict]:
    """
    Among the six cached phi-only Stage 1 variants, return the
    (model_id, metadata) of the one with the lowest OOS CRPS among those
    with a usable validity class -- consistent with docs/MODELS.md §27
    ("do not select models by RMSE/loglik alone; CRPS ranks above raw
    likelihood"). Falls back to log-likelihood if every CRPS is NaN.
    """
    candidates = []
    for variant in BASELINE_PHI_VARIANTS:
        meta_path = run_dir / "stage1" / variant / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("validity") == "failed":
            continue
        candidates.append((variant, meta))

    if not candidates:
        raise RuntimeError(
            f"No usable phi-only Stage 1 artifact found under {run_dir / 'stage1'}. "
            f"Expected one of {BASELINE_PHI_VARIANTS}."
        )

    finite_crps = [(v, m) for v, m in candidates if np.isfinite(m.get("crps_mean", np.nan))]
    pool = finite_crps if finite_crps else candidates
    key = (lambda vm: vm[1]["crps_mean"]) if finite_crps else (lambda vm: -vm[1]["loglik"])
    best_variant, best_meta = min(pool, key=key)
    return best_variant, best_meta


def _reconstruct_frozen_baseline(run_dir: Path, model_id: str, meta: dict, y_train: np.ndarray) -> dict:
    """
    Rebuild the phi-only ZAGASModel exactly as it was fit (same gas_lags,
    scaling, static_params -- see models/factory.py:build_zagas_model for
    the equivalent no-covariate construction), load its saved theta, and
    run the filter once to recover phi_{t|t-1} in-sample.

    pi_dynamics=ARLogisticPiDynamics() is required to construct a valid
    ZAGASModel but its output (paths["eta"], paths["pi"]) is intentionally
    NOT used here -- all four pi_t alternatives in this experiment are
    refit standalone (see module docstring) for a uniform comparison, so
    only paths["phi"], paths["static"], and paths["eff_start"] are read
    from this reconstruction.
    """
    model = ZAGASModel(
        distribution=GB2LogLinkPhiOnly(),
        pi_dynamics=ARLogisticPiDynamics(),
        seasonal=meta["seasonal"],
        gas_lags=meta["lags"],
        scaling=meta["scaling"],
        static_params=["xi", "gamma", "zeta"],
    )
    param_df = pd.read_csv(run_dir / "stage1" / model_id / "estimated_parameters.csv")
    if list(param_df["parameter"]) != model.parameter_names():
        raise RuntimeError(
            f"Parameter name mismatch reconstructing {model_id}: saved CSV "
            f"order does not match the freshly-built model's parameter_names(). "
            f"The model configuration (lags/scaling/static_params) must have "
            f"changed since this artifact was produced."
        )
    theta = param_df["value"].to_numpy(dtype=float)

    paths = model.filter(theta, y_train)
    if not paths:
        raise RuntimeError(f"Filtering the reconstructed baseline model {model_id} failed (T <= max_lag?).")

    eff_start = int(paths["eff_start"])
    phi_is = np.asarray(paths["phi"], dtype=float)
    # paths["phi"][0] is, by construction, the filter's warm-up constant
    # f0_phi: models/gas_filter.py seeds f_arr[:max_lag+1] = f0_phi and only
    # starts updating from index max_lag+1 onward, so f_arr[max_lag] (i.e.
    # paths["phi"][0], since paths["phi"] = f_arr[eff_start:T]) is still
    # exactly the fitted initial state -- reused here instead of reaching
    # into the codec to avoid a second, redundant decode of theta.
    f0_phi = float(phi_is[0])

    return {
        "eff_start": eff_start,
        "phi_is": phi_is,
        "static": {k: float(v) for k, v in paths["static"].items()},
        "y_eff": np.asarray(paths["y_eff"], dtype=float),
        "f0_phi": f0_phi,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: build the four pi-dynamics variants for one location
# ─────────────────────────────────────────────────────────────────────────────

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
    "phi_linked":           "eta_t = lambda0 + lambda1*phi_{t|t-1}  (phi_{t|t-1} frozen from the location's best-by-CRPS phi-only Stage 1 baseline)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: fit + diagnose one (location, variant) pair
# ─────────────────────────────────────────────────────────────────────────────

def _fit_and_diagnose_one(
    location: str,
    short: str,
    variant_key: str,
    pi_dyn,
    y_train: np.ndarray,
    baseline: dict,
    baseline_model_id: str,
    baseline_run_id: str,
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

    eta_full = eta_path_full(pi_dyn, fit["theta"], y_train, SEASONAL)
    pi_full  = expit(eta_full)
    eta_w = eta_full[eff_start:]
    pi_w  = pi_full[eff_start:]

    # ── Occurrence diagnostics (common window, see module docstring) ────────
    wet = y_w > 0
    dry = ~wet
    pi_mean_wet = float(pi_w[wet].mean()) if wet.any() else float("nan")
    pi_mean_dry = float(pi_w[dry].mean()) if dry.any() else float("nan")

    rng = np.random.default_rng(seed)
    z_w = wet.astype(float)
    pit = bernoulli_pit_values(pi_w, z_w, rng=rng)
    qr  = bernoulli_quantile_residuals(pi_w, z_w, rng=np.random.default_rng(seed))

    # Single 400-lag ACF pass; the table lags {1,2,3,364,365,366,367} are a
    # subset of 1..400, so they are sliced out rather than recomputed.
    acf_full  = acf_at_lags(qr, ACF_FULL_LAGS)
    acf_table = {l: acf_full[l] for l in ACF_TABLE_LAGS}

    # ── RMSE / CRPS against y_t, magnitude frozen at the shared baseline ────
    eval_metrics = in_sample_point_and_crps_metrics(
        dist=GB2LogLinkPhiOnly(),
        phi_arr=baseline["phi_is"],
        static=baseline["static"],
        pi_arr=pi_w,
        y_arr=y_w,
        n_draws=n_draws,
        seed=seed,
    )

    runtime_s = time.time() - t0

    return {
        "variant_key": variant_key,
        "success": True,
        "pi_dyn_type": type(pi_dyn).__name__,
        "param_names": fit["param_names"],
        "theta": fit["theta"],
        "loglik_pi_only": fit["loglik"],
        "runtime_s": runtime_s,
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
        "baseline_model_id": baseline_model_id,
        "baseline_run_id": baseline_run_id,
    }


def _save_artifact(out_dir: Path, location: str, variant_key: str, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not result.get("success"):
        (out_dir / "metadata.json").write_text(json.dumps({
            "model_type": "PiDynamicsAlternative",
            "location": location, "variant_key": variant_key,
            "validity": "failed", "success": False,
            "message": result.get("message", "unknown failure"),
            "runtime_s": result.get("runtime_s"),
        }, indent=2))
        return

    pd.DataFrame({
        "parameter": result["param_names"],
        "value": result["theta"],
    }).to_csv(out_dir / "estimated_parameters.csv", index=False)

    np.savez(
        out_dir / "paths.npz",
        eta_w=result["eta_w"], pi_w=result["pi_w"], y_w=result["y_w"],
        pit=result["pit"], qr=result["qr"],
        eta_full=result["eta_full"], pi_full=result["pi_full"],
    )

    pd.DataFrame({
        "lag": result["acf_full_lags"],
        "acf": result["acf_full_values"],
    }).to_csv(out_dir / "acf_400.csv", index=False)

    meta = {
        "model_type": "PiDynamicsAlternative",
        "location": location,
        "variant_key": variant_key,
        "pi_dyn_type": result["pi_dyn_type"],
        "description": _VARIANT_DESCRIPTIONS[variant_key],
        "seasonal": SEASONAL,
        "baseline_phi_model_id": result["baseline_model_id"],
        "baseline_run_id": result["baseline_run_id"],
        "eff_start": result["eff_start"],
        "n_obs_window": result["n_obs_window"],
        "n_dry_window": result["n_dry_window"],
        "n_wet_window": result["n_wet_window"],
        "loglik_pi_only": result["loglik_pi_only"],
        "validity": "valid_converged",
        "success": True,
        "runtime_s": result["runtime_s"],
        "pi_mean_dry": result["pi_mean_dry"],
        "pi_mean_wet": result["pi_mean_wet"],
        "acf_selected_lags": {str(l): result["acf_table"][l] for l in ACF_TABLE_LAGS},
        "rmse_all": result["rmse_all"],
        "rmse_dry": result["rmse_dry"],
        "rmse_wet": result["rmse_wet"],
        "crps_all": result["crps_all"],
        "estimation_note": (
            "pi_t fit standalone (pi_dynamics/standalone_fit.py) on the "
            "binary occurrence sequence alone, independent of any GB2 fit. "
            "RMSE/CRPS use this model's pi_t combined with the FROZEN "
            "magnitude path (phi_t, xi, gamma, zeta) from "
            f"{result['baseline_model_id']} ({result['baseline_run_id']}) -- "
            "identical across all four pi-dynamics alternatives at this "
            "location, so RMSE/CRPS differences are attributable to the "
            "occurrence dynamics alone. All diagnostics are in-sample only; "
            "this experiment produces no OOS forecast."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_location(location: str, force: bool, n_draws: int) -> None:
    cfg = STATION_REGISTRY[location]
    short, run_id = cfg["short"], cfg["run_id"]
    run_dir = ARTIFACTS_DIR / run_id
    out_base = EXPERIMENT_DIR / short

    logger.info(f"=== {cfg['display']} ({short}) ===")

    loader = StationDataLoader.from_registry(
        location, precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH,
    )
    data = loader.load_all()
    y_train = data["y_train"]

    baseline_model_id, baseline_meta = _select_baseline_phi_model(run_dir)
    logger.info(
        f"  Frozen magnitude baseline: {baseline_model_id} "
        f"(CRPS={baseline_meta.get('crps_mean', float('nan')):.4f}, lags={baseline_meta['lags']}, "
        f"scaling={baseline_meta['scaling']})"
    )
    baseline = _reconstruct_frozen_baseline(run_dir, baseline_model_id, baseline_meta, y_train)
    logger.info(
        f"  eff_start={baseline['eff_start']}  "
        f"n_obs_window={len(y_train) - baseline['eff_start']}"
    )

    phi_path_full = build_full_length_phi_path(
        phi_is=baseline["phi_is"], eff_start=baseline["eff_start"],
        f0_phi=baseline["f0_phi"], total_length=len(y_train),
    )
    variants = _build_variants(phi_path_full)

    for variant_key, pi_dyn in variants.items():
        out_dir = out_base / variant_key
        meta_path = out_dir / "metadata.json"
        if meta_path.exists() and not force:
            existing = json.loads(meta_path.read_text())
            if existing.get("success"):
                logger.info(f"  [{variant_key}] cached artifact found -- skipping (use --force to re-estimate).")
                continue

        logger.info(f"  [{variant_key}] fitting...")
        try:
            result = _fit_and_diagnose_one(
                location=location, short=short, variant_key=variant_key,
                pi_dyn=pi_dyn, y_train=y_train, baseline=baseline,
                baseline_model_id=baseline_model_id, baseline_run_id=run_id,
                n_draws=n_draws,
            )
        except Exception as exc:
            logger.error(f"  [{variant_key}] FAILED with exception: {exc}")
            logger.error(traceback.format_exc())
            result = {"variant_key": variant_key, "success": False, "message": str(exc), "runtime_s": None}

        _save_artifact(out_dir, location, variant_key, result)
        _execution_log({
            "location": location, "short": short, "variant_key": variant_key,
            "success": result.get("success", False),
            "runtime_s": result.get("runtime_s"),
            "loglik_pi_only": result.get("loglik_pi_only"),
            "rmse_all": result.get("rmse_all"), "crps_all": result.get("crps_all"),
            "message": result.get("message"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        if result.get("success"):
            logger.info(
                f"  [{variant_key}] OK  loglik={result['loglik_pi_only']:.2f}  "
                f"pi|dry={result['pi_mean_dry']:.4f}  pi|wet={result['pi_mean_wet']:.4f}  "
                f"RMSE_all={result['rmse_all']:.4f}  CRPS_all={result['crps_all']:.4f}  "
                f"({result['runtime_s']:.1f}s)"
            )
        else:
            logger.warning(f"  [{variant_key}] NOT SAVED AS VALID: {result.get('message')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the four pi-dynamics alternatives at all six locations.")
    parser.add_argument("--force", action="store_true", help="Re-estimate even when a valid artifact already exists.")
    parser.add_argument("--location", type=str, default=None, help="Run only this location (exact STATION_REGISTRY key).")
    parser.add_argument("--n-draws", type=int, default=500, help="Monte Carlo draws per time step for CRPS.")
    args = parser.parse_args()

    locations = [args.location] if args.location else LOCATION_ORDER
    unknown = [loc for loc in locations if loc not in STATION_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown location(s): {unknown}. Available: {list(STATION_REGISTRY)}")

    logger.info(f"Pi-dynamics alternatives experiment: {len(locations)} location(s), 4 variants each "
                f"({len(locations) * 4} models total). force={args.force}  n_draws={args.n_draws}")

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
                        f"Check pi_dynamics_alternatives.log for details.")
        logger.warning("=" * 60)

    logger.info("Done. Run generate_pi_dynamics_report.py to build the report.")


if __name__ == "__main__":
    main()
