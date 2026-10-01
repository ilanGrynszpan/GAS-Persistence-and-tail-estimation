"""
Harvey long-short persistence/initialisation diagnostics experiment
(2026-08-31).

=============================================================================
OVERVIEW
=============================================================================

A focused, single-lag diagnostic study of models.harvey_gas.HarveyZAGASModel,
separate from the Stage 1-4 thesis pipeline in run_all_locations.py. It asks:
does the MODELS.md Section 20 persistence restriction

    0 < phi_S,j < phi_L,j < 1

matter for convergence once the model is stripped down to a single
short-component covariate lag, and how sensitive is the fit to how
phi_S,j/phi_L,j/kappa_L,j/kappa_S,j/Gamma_L,j/Gamma_S,j are initialised?

This script performs NO changes to any existing Stage 1-3 pipeline
artifact. It reads Stage-2 weather-covariate winners (already fitted, never
re-optimised) purely to derive initialisation values, and writes its own
fits to a brand-new artifact tree.

=============================================================================
CODE WALKTHROUGH
=============================================================================

main()
  for each of the 6 completed locations:
      loader.load_all()                      -- reused as-is, no new data code
      for branch in (phi_only, phi_xi):
          resolve_prior_model_id()           -- best-RMSE (phi_only) or
                                                 best-CRPS (phi_xi) Stage-2
                                                 weather model, from
                                                 stage_winners.json
          load_model_and_theta(prior_dir)    -- reconstruct that model, no
                                                 re-fit (pipeline.artifact_utils)
          prior_model.filter(theta, ...)     -- one cheap forward pass to
                                                 obtain its filtered f_t path
          persistence_init_from_filtered_path() -- phi_S,init = mean(f_t+1/f_t),
                                                 phi_L,init = 1.1 * phi_S,init,
                                                 with a blow-up fallback to
                                                 (0.02, 0.04) when the ratio
                                                 is not a usable persistence
                                                 value (non-finite, <=0, >=1)
          extract Gamma_S,init from the prior model's own fitted coefficient
              on dewpoint_lag1 (0.0 fallback if absent)
          GASFilter(distribution=dist).fit_unconditional(y_pos)
                                              -- bounded L-BFGS-B warm start
                                                 for gamma, zeta, and the
                                                 initial states (same pattern
                                                 static_gb2.py already reuses)
          for persistence_mode in (restricted, free, fixed_phiL, sigmoid_both):
              for gamma_L_init in (1, 0.2, 0.02):
                  build HarveyZAGASModel(..., use_auxiliary_penalties=False)
                      -- restricted:    restrict_persistence_upper=True  (raw phi_L, phi_S)
                      -- free:          restrict_persistence_upper=False (raw phi_L, phi_S)
                      -- fixed_phiL:    fixed_phi_L=1.0, phi_S_max=0.97
                                        (long component = literal random walk;
                                        only phi_S is estimated, bounded by
                                        construction, not by a penalty)
                      -- sigmoid_both:  phi_L_max=1.0, phi_S_max=1.0
                                        (both persistences unconstrained on the
                                        reals, mapped through a sigmoid so
                                        0<phi<1 by construction; phi_S<phi_L
                                        ordering still a soft penalty on the
                                        decoded values)
                  theta0 = model.initial_theta(y_train, overrides=..., static_overrides=...)
                  fit = model.fit(y_train, X_long_train, X_short_train, theta0=theta0)
                  model.save_result(fit, <artifact dir>)
                  write init_recipe.json      -- every init value + provenance
                  append execution_log.jsonl entry
  print a summary of failures / phi-init blow-ups across the whole grid

=============================================================================
USAGE
=============================================================================

    python run_longshort_diagnostics.py                       # all 6 locations, both branches
    python run_longshort_diagnostics.py --location "BELO HORIZONTE" --branch phi_only
    python run_longshort_diagnostics.py --location "BELO HORIZONTE" --branch phi_xi --dry-run

Output:
    artifacts/longshort_diagnostics_experiment/<run_id>/<branch>/<mode>_gL<value>/
        (mode in restricted, free, fixed_phiL, sigmoid_both)
        estimated_parameters.csv, decoded_harvey_parameters.csv, metadata.json,
        gradients.csv, hess_inv.npy, standard_errors.csv
        (all via HarveyZAGASModel.save_result, unmodified)
        init_recipe.json      (this script: every init value + its source)
    artifacts/longshort_diagnostics_experiment/execution_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import STATION_REGISTRY, StationDataLoader
from distributions.gb2_log_link import GB2LogLink
from distributions.gb2_phi_only import GB2LogLinkPhiOnly
from pi_dynamics.factory import make_pi_dynamics
from models.gas_filter import GASFilter
from models.harvey_gas import HarveyZAGASModel, persistence_init_from_filtered_path
from pipeline.artifact_utils import load_model_and_theta

# Duplicated from run_all_locations.py rather than imported: that module
# configures logging.basicConfig() with its own FileHandler at import time
# (writing to pipeline_all_locations.log), which would silently hijack this
# script's own logging setup and interleave this experiment's log lines into
# the unrelated main-pipeline log file. These four paths must stay identical
# to run_all_locations.py's definitions.
PRECIP_DIR    = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR      = ROOT / "data" / "input" / "ERA5"
ENSO_PATH     = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

LOCATIONS: List[str] = [
    "BELO HORIZONTE",
    "CRUZEIRO DO SUL (ACRE)",
    "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)",
    "MANAUS",
    "SALVADOR",
]

BRANCHES = ("phi_only", "phi_xi")

EXPERIMENT_DIR = ARTIFACTS_DIR / "longshort_diagnostics_experiment"
LOG_PATH       = EXPERIMENT_DIR / "execution_log.jsonl"

GAMMA_L_INITS       = [1.0, 0.2, 0.02]
KAPPA_INIT          = 0.02
BLOWUP_PHI_S_INIT   = 0.02
BLOWUP_PHI_L_INIT   = 0.04
SHORT_COVARIATE     = "dewpoint_lag1"
SHORT_SOURCE_BLOCK  = "dewpoint_short"   # contains dewpoint_lag{1,2,3}; lag1 always present
LONG_BLOCK          = "enso_dummy_current"

# Four persistence-parameterisation modes, crossed with GAMMA_L_INITS below.
# "restricted"/"free" keep phi_L, phi_S raw (MODELS.md §20 penalty-based
# restriction, on/off). "fixed_phiL"/"sigmoid_both" are the two additional
# configurations requested after the near-unit-root gradient/SE breakdown
# documented in reports/longshort experiment/ (see models/harvey_gas.py's
# constructor docstring for exactly what each reparameterisation does).
PERSISTENCE_MODES = ["restricted", "free", "fixed_phiL", "sigmoid_both"]
FIXED_PHI_L_VALUE   = 1.0    # "fixed_phiL": long component = literal random walk
FIXED_PHI_L_PHI_S_MAX = 0.97  # "fixed_phiL": phi_S bounded to (0, 0.97) by construction
SIGMOID_PHI_MAX     = 1.0    # "sigmoid_both": both phi_L, phi_S bounded to (0, 1) by construction


def _model_kwargs_for_mode(mode: str) -> dict:
    """HarveyZAGASModel kwargs selecting one of the four persistence modes."""
    if mode == "restricted":
        return {"restrict_persistence_upper": True}
    if mode == "free":
        return {"restrict_persistence_upper": False}
    if mode == "fixed_phiL":
        return {"fixed_phi_L": FIXED_PHI_L_VALUE, "phi_S_max": FIXED_PHI_L_PHI_S_MAX}
    if mode == "sigmoid_both":
        return {"phi_L_max": SIGMOID_PHI_MAX, "phi_S_max": SIGMOID_PHI_MAX}
    raise ValueError(f"Unknown persistence_mode {mode!r}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("longshort_diagnostics")


def _log_event(record: dict) -> None:
    """Append one JSON line to the experiment's execution log (EXECUTION.md §18)."""
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=float) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Prior-model resolution (source of phi_S/phi_L and Gamma_S initial values)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_prior_model_id(run_dir: Path, branch: str) -> str:
    """
    2-TV branch (phi_xi): best crps_mean among stage_winners.json["stage2"]["models"].
    1-TV branch (phi_only): best rmse among stage_winners.json["stage2_phi_only"]["models"],
    with rmse read from each candidate's own metadata.json (not stored in
    stage_winners.json for that branch).
    """
    winners = json.loads((run_dir / "stage_winners.json").read_text())

    if branch == "phi_xi":
        models = winners["stage2"]["models"]
        finite = [m for m in models if np.isfinite(m.get("crps_mean", np.nan))]
        if not finite:
            raise ValueError(f"No finite-CRPS stage2 model found in {run_dir}")
        best = min(finite, key=lambda m: m["crps_mean"])
        return best["model_id"]

    models = winners["stage2_phi_only"]["models"]
    best_id, best_rmse = None, np.inf
    for m in models:
        meta_path = run_dir / "stage2_phi" / m["model_id"] / "metadata.json"
        if not meta_path.exists():
            continue
        rmse = json.loads(meta_path.read_text()).get("rmse", np.inf)
        if np.isfinite(rmse) and rmse < best_rmse:
            best_rmse, best_id = rmse, m["model_id"]
    if best_id is None:
        raise ValueError(f"No finite-RMSE stage2_phi model found in {run_dir}")
    return best_id


def _prior_model_dir(run_dir: Path, branch: str, model_id: str) -> Path:
    sub = "stage2" if branch == "phi_xi" else "stage2_phi"
    return run_dir / sub / model_id


def _block_name_from_model_id(model_id: str) -> str:
    """'stage2_dewtemp_seasonal' -> 'dewtemp_seasonal'; 'stage2_phi_dewpoint_short' -> 'dewpoint_short'."""
    return model_id.replace("stage2_phi_", "").replace("stage2_", "")


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation recipe
# ─────────────────────────────────────────────────────────────────────────────

SAFE_PHI_L_CEILING = 0.95  # see _phi_init_from_prior docstring -- BUG-2026-09-01


def _phi_init_from_prior(f_arr_j: np.ndarray) -> Tuple[float, float, bool, dict]:
    """
    Returns (phi_S_init, phi_L_init, blowup_triggered, ratio_stats).

    phi_S_init = mean(f_t+1/f_t); phi_L_init = 1.1 * phi_S_init.
    Blow-up fallback (phi_S=0.02, phi_L=0.04) triggers when the ratio mean
    is non-finite, <= 0, or when the RESULTING phi_L_init (1.1*r, not r
    itself) would be >= SAFE_PHI_L_CEILING.

    BUG-2026-09-01 (see reports/longshort experiment/
    stability_investigation_log.md): the original guard checked `r >= 1.0`
    instead of checking the initialised phi_L_init = 1.1*r for safety. A
    ratio mean just below 1 (Darwin Airport's phi_xi/phi: r=0.9943) passes
    `r >= 1.0` cleanly while still producing phi_L_init = 1.1*0.9943 =
    1.094 > 1 -- a persistence coefficient exceeding 1 in magnitude, which
    makes the long component's recursion L_{t+1} = phi_L*L_t + ... literally
    explosive (geometric growth) from the very first likelihood evaluation,
    before any optimisation step runs. Verified: this single bug fully
    explained Darwin Airport's phi_xi branch having zero reliable fits
    (|grad|_inf < 1 threshold) across all 12 of its configurations --
    f_arr's phi entry reached ~457 in log-link space at gL=1.0 (obj0=1.3e11,
    loglik0=-133,587 vs. a healthy ~-3,000 to -6,000 elsewhere) purely from
    this poisoned starting point, before BFGS took a single step. A scan of
    every other (location, branch) combination in the 144-fit grid found no
    other case with ratio_mean in the dangerous (1/1.1, 1.0) = (0.909, 1.0)
    band -- every other location's ratio was already >= 1.0 (correctly
    caught by the old check) or comfortably below 0.909, so this fix changes
    ONLY Darwin Airport's phi_xi initial values; every other (location,
    branch) reproduces its exact previous phi_S_init/phi_L_init.
    """
    stats = persistence_init_from_filtered_path(f_arr_j)
    r = stats["ratio_mean"]
    phi_l_candidate = 1.1 * r if np.isfinite(r) else np.nan
    blowup = (not np.isfinite(r)) or (r <= 0.0) or (phi_l_candidate >= SAFE_PHI_L_CEILING)
    if blowup:
        return BLOWUP_PHI_S_INIT, BLOWUP_PHI_L_INIT, True, stats
    return float(r), float(phi_l_candidate), False, stats


def _gamma_s_init_from_prior(params_df: pd.DataFrame, j: str) -> Tuple[float, bool]:
    """Gamma_S,init = prior model's fitted gamma_{j}_dewpoint_lag1; 0.0 if absent."""
    name = f"gamma_{j}_{SHORT_COVARIATE}"
    row = params_df[params_df["parameter"] == name]
    if row.empty:
        return 0.0, True
    return float(row["value"].iloc[0]), False


# ─────────────────────────────────────────────────────────────────────────────
# One (location, branch) unit of work
# ─────────────────────────────────────────────────────────────────────────────

def run_branch(location: str, branch: str, dry_run: bool = False) -> dict:
    cfg     = STATION_REGISTRY[location]
    run_id  = cfg["run_id"]
    display = cfg.get("display", location)
    run_dir = ARTIFACTS_DIR / run_id

    logger.info(f"--- {display}  [{branch}] ---")

    loader = StationDataLoader.from_registry(
        location, precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH,
    )
    data = loader.load_all()
    y_train = data["y_train"]

    dist   = GB2LogLink() if branch == "phi_xi" else GB2LogLinkPhiOnly()
    pi_dyn = make_pi_dynamics("ar_logistic", seasonal="daily")

    # ── Resolve prior model and derive every initial value from it ─────────
    prior_id  = resolve_prior_model_id(run_dir, branch)
    prior_dir = _prior_model_dir(run_dir, branch, prior_id)
    prior_model, prior_theta, prior_meta, prior_params = load_model_and_theta(prior_dir)
    if prior_model is None or prior_theta is None:
        raise RuntimeError(f"Could not reconstruct prior model at {prior_dir}")

    block   = _block_name_from_model_id(prior_id)
    X_prior = data["covariate_blocks_train"][block]
    prior_paths = prior_model.filter(prior_theta, y_train, X_prior)
    tv_names = prior_paths["tv_names"]
    logger.info(f"    prior model: {prior_id}  (block={block}, tv_names={tv_names})")

    phi_s_init, phi_l_init, blowup, ratio_stats = {}, {}, {}, {}
    for j in tv_names:
        jidx = tv_names.index(j)
        s, l, b, stats = _phi_init_from_prior(prior_paths["f_arr"][:, jidx])
        phi_s_init[j], phi_l_init[j], blowup[j], ratio_stats[j] = s, l, b, stats
        if b:
            logger.warning(
                f"    PHI-INIT BLOW-UP for {display} [{branch}] param={j}: "
                f"ratio_mean={stats['ratio_mean']!r} (n_terms={stats['n_terms']}, "
                f"n_masked={stats['n_masked']}) -> falling back to "
                f"phi_S={BLOWUP_PHI_S_INIT}, phi_L={BLOWUP_PHI_L_INIT}"
            )

    gamma_s_init, gamma_s_fallback = {}, {}
    for j in tv_names:
        g, fb = _gamma_s_init_from_prior(prior_params, j)
        gamma_s_init[j], gamma_s_fallback[j] = g, fb
        if fb:
            logger.warning(f"    Gamma_S fallback to 0.0 for {j}: dewpoint_lag1 not in prior model's block")

    y_pos = y_train[y_train > 0]
    warm  = GASFilter(distribution=dist).fit_unconditional(y_pos)
    if not warm:
        raise RuntimeError(f"Bounded warm start (fit_unconditional) failed for {display} [{branch}]")

    # ── Covariate arrays for the Harvey fits ────────────────────────────────
    short_cols = data["covariate_col_names"][SHORT_SOURCE_BLOCK]
    short_idx  = short_cols.index(SHORT_COVARIATE)
    X_short_train = data["covariate_blocks_train"][SHORT_SOURCE_BLOCK][:, [short_idx]]

    long_cols     = data["covariate_col_names"][LONG_BLOCK]
    X_long_train  = data["covariate_blocks_train"][LONG_BLOCK]

    branch_out_dir = EXPERIMENT_DIR / run_id / branch
    n_ok, n_fail = 0, 0

    for mode in PERSISTENCE_MODES:
        for gL in GAMMA_L_INITS:
            label   = f"{mode}_gL{gL}"
            out_dir = branch_out_dir / label

            # Keys match models.harvey_gas.HarveyZAGASModel's MODELS.md §20
            # naming directly (kappa=score response, phi=persistence,
            # Gamma=covariate effects) -- see initial_theta()'s docstring.
            overrides = {}
            for j in tv_names:
                overrides[j] = {
                    "omega":   0.0,
                    "f0":      warm.get(j, 0.0),
                    "L0":      warm.get(j, 0.0),
                    "S0":      0.0,
                    "kappa_L": KAPPA_INIT,
                    "kappa_S": KAPPA_INIT,
                    "phi_L":   phi_l_init[j],
                    "phi_S":   phi_s_init[j],
                    "Gamma_L": gL,
                    "Gamma_S": gamma_s_init[j],
                }
            static_overrides = {"gamma": warm.get("gamma"), "zeta": warm.get("zeta")}

            init_recipe = {
                "location": display, "run_id": run_id, "branch": branch,
                "prior_model_id": prior_id, "block": block,
                "persistence_mode": mode,
                "model_kwargs": _model_kwargs_for_mode(mode),
                "gamma_L_init": gL,
                "kappa_init": KAPPA_INIT,
                "short_covariate": SHORT_COVARIATE,
                "long_covariates": long_cols,
                "static_warm_start": {"gamma": warm.get("gamma"), "zeta": warm.get("zeta")},
                "per_param": {
                    j: {
                        "phi_S_init": phi_s_init[j], "phi_L_init": phi_l_init[j],
                        "blowup_fallback": blowup[j], "ratio_stats": ratio_stats[j],
                        "gamma_S_init": gamma_s_init[j], "gamma_S_fallback_to_zero": gamma_s_fallback[j],
                        "warm_start_f0": warm.get(j, 0.0),
                    }
                    for j in tv_names
                },
            }

            if dry_run:
                logger.info(f"    [dry-run] would fit {label}")
                continue

            t0 = time.time()
            try:
                model = HarveyZAGASModel(
                    distribution=dist, pi_dynamics=pi_dyn, seasonal="daily",
                    long_names=long_cols, short_names=[SHORT_COVARIATE],
                    scaling="diagonal_inverse_fisher",
                    use_auxiliary_penalties=False,
                    **_model_kwargs_for_mode(mode),
                )
                theta0 = model.initial_theta(y_train, overrides=overrides, static_overrides=static_overrides)
                fit = model.fit(y_train, X_long_train, X_short_train, theta0=theta0)
                model.save_result(fit, out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "init_recipe.json").write_text(json.dumps(init_recipe, indent=2, default=float))

                runtime = time.time() - t0
                logger.info(
                    f"    {label:>20s}  validity={fit['validity']:<18s} "
                    f"success={fit['success']!s:<5s} grad_inf={fit['grad_norm_inf']:.3g} "
                    f"loglik={fit['loglik']:.2f}  ({runtime:.1f}s)"
                )
                _log_event({
                    "location": display, "run_id": run_id, "branch": branch, "label": label,
                    "status": "ok", "validity": fit["validity"], "success": fit["success"],
                    "grad_norm_inf": fit["grad_norm_inf"], "loglik": fit["loglik"],
                    "runtime_s": runtime,
                })
                n_ok += 1
            except Exception as exc:
                runtime = time.time() - t0
                logger.error(f"    {label:>20s}  FAILED after {runtime:.1f}s: {exc}")
                _log_event({
                    "location": display, "run_id": run_id, "branch": branch, "label": label,
                    "status": "failed", "error": str(exc), "traceback": traceback.format_exc(),
                    "runtime_s": runtime,
                })
                n_fail += 1

    return {"n_ok": n_ok, "n_fail": n_fail, "n_blowup": sum(blowup.values())}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", default=None, help="Run only this location (exact STATION_REGISTRY name)")
    parser.add_argument("--branch", default="both", choices=["phi_only", "phi_xi", "both"])
    parser.add_argument("--dry-run", action="store_true", help="Resolve inits and log intended fits without optimizing")
    args = parser.parse_args()

    locations = [args.location] if args.location else LOCATIONS
    branches  = BRANCHES if args.branch == "both" else (args.branch,)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Harvey long-short diagnostics experiment -> {EXPERIMENT_DIR}")
    logger.info(f"Locations: {locations}")
    logger.info(f"Branches:  {branches}")

    totals = {"n_ok": 0, "n_fail": 0, "n_blowup": 0, "n_attempted": 0}
    aborted = False

    for location in locations:
        for branch in branches:
            try:
                out = run_branch(location, branch, dry_run=args.dry_run)
            except Exception as exc:
                logger.error(f"BRANCH-LEVEL FAILURE for {location} [{branch}]: {exc}")
                _log_event({
                    "location": location, "branch": branch, "status": "branch_failed",
                    "error": str(exc), "traceback": traceback.format_exc(),
                })
                out = {"n_ok": 0, "n_fail": 6, "n_blowup": 0}

            totals["n_ok"]        += out["n_ok"]
            totals["n_fail"]      += out["n_fail"]
            totals["n_blowup"]    += out["n_blowup"]
            totals["n_attempted"] += out["n_ok"] + out["n_fail"]

            # EXECUTION.md §20: abort and alert if most models are failing.
            if totals["n_attempted"] >= 12 and totals["n_fail"] / totals["n_attempted"] > 0.5:
                logger.critical(
                    "ABORTING: more than half of all attempted fits have failed "
                    f"({totals['n_fail']}/{totals['n_attempted']}). Stopping remaining "
                    "locations/branches so this can be investigated before continuing. "
                    "See execution_log.jsonl for per-model failure detail."
                )
                aborted = True
                break
        if aborted:
            break

    logger.info("=" * 72)
    logger.info(
        f"DONE. ok={totals['n_ok']}  failed={totals['n_fail']}  "
        f"phi-init blow-ups={totals['n_blowup']}  aborted_early={aborted}"
    )
    logger.info(f"Artifacts: {EXPERIMENT_DIR}")
    logger.info(f"Execution log: {LOG_PATH}")


if __name__ == "__main__":
    main()
