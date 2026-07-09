"""
Multi-location pipeline runner.

=============================================================================
OVERVIEW
=============================================================================

Runs the ZA-GAS estimation pipeline for the six locations already present
in reports/multi_location/report.tex: BELO HORIZONTE, CRUZEIRO DO SUL,
DARWIN AIRPORT, GARANHUNS, MANAUS, SALVADOR. (SAO PAULO and TORONTO are
configured in STATION_REGISTRY but were never actually run in round 1 and
stay out of scope for round 2, per prompt.md's explicit instruction to
reuse only the already-run locations. RIYADH is excluded for bad data.)

For each location, per prompt.md (2026-07-07 revision):

  1. Stages 1-2 (phi+xi branch): reused from round-1 artifacts where valid
     -- NOT re-estimated. Only the *winner selection* changes (objective
     1b): OOS CRPS is now the ranking criterion (never log-likelihood),
     with twCRPS@95/98 reported alongside.

  2. Stage 3 (phi+xi branch): RE-ESTIMATED. Round 1's ENSO covariate was
     built from mislabelled raw sea-surface temperature (see
     data/station_loader.py header) and has been replaced with El Nino/La
     Nina dummy tiers built from the correct NOAA ONI classification.

  3. A parallel phi-only branch (objective 1a) is run through its OWN
     Stage 1 (already exists) -> Stage 2 -> Stage 3, ranked throughout by
     OOS RMSE, never CRPS or log-likelihood, and never mixed with the
     phi+xi branch. This is new estimation round 1 did not include.

  4. Stage 4 (objective 3) is redesigned: tail-sensitive dynamics for xi
     ONLY. Phi (and the occurrence probability pi) are frozen at whichever
     phi-only model won Stages 1-3 above -- never re-estimated. xi gets a
     full fresh GAS(1,1) + short-term-weather-covariate + regime fit.
     Thresholds: q95 and q98 only (q90 dropped per objective 3b). Each
     Stage-4 model is accepted only if its twCRPS beats the phi+xi
     branch's best-of-previous-stages counterpart by >=2% (objective 3a).

Execution is SEQUENTIAL across locations to avoid RAM saturation. Within a
location, independent models inside one stage run in parallel
(--workers workers). Stage 2 runs sequentially with warm starts.

=============================================================================
USAGE
=============================================================================

    python run_all_locations.py [options]

Options:
    --workers N       Number of parallel workers per stage (default 4)
    --force           Re-estimate even when valid artifacts exist
    --skip-stage4     Skip Stage 4 (tail-sensitive xi regime)
    --location NAME   Run only one location (exact name from STATION_REGISTRY)
    --start-from NAME Skip all locations before NAME in the processing order

=============================================================================
ARTIFACT STRUCTURE
=============================================================================

artifacts/run_<id>/
    stage1/                     phi+xi and phi-only Stage-1 models (12 total)
    stage2/, stage3/            phi+xi branch (weather covariates, Harvey)
    stage2_phi/, stage3_phi/    phi-only branch (objective 1a)
    stage4_xi_regime/           tail-sensitive xi models (q95, q98)
    stage_winners.json
    execution_log.jsonl

=============================================================================
MONITORING
=============================================================================

Progress is written to:
    pipeline_all_locations.log     -- human-readable, tee'd to stdout
    artifacts/<run_id>/execution_log.jsonl  -- per-model machine-readable
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Paths ────────────────────────────────────────────────────────────────────
PRECIP_DIR  = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR    = ROOT / "data" / "input" / "ERA5"
ENSO_PATH   = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

# Locations excluded from every round of this script (bad source data)
SKIP_STATIONS = {"RIYADH OBS. (O.A.P."}

# The six locations already reported in reports/multi_location/report.tex.
# SAO PAULO and TORONTO are registered in STATION_REGISTRY but were never
# run in round 1 -- prompt.md 2026-07-07 explicitly says to reuse only the
# already-run locations, not add new ones.
LOCATION_ORDER = [
    "BELO HORIZONTE",
    "CRUZEIRO DO SUL (ACRE)",
    "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)",
    "MANAUS",
    "SALVADOR",
]

# Improvement threshold for every "best metric or simplest model" decision
# (objective 4d): in-stage winners, inter-stage advancement, and Stage-4
# acceptance all use this same 2% rule.
IMPROVEMENT_THRESHOLD = 0.02

# Stage-4 xi-only regime: thresholds retained per objective 3b (q90 dropped)
STAGE4_THRESHOLDS = [("95", 0.95), ("98", 0.98)]

# ── Logging ───────────────────────────────────────────────────────────────────
log_file = ROOT / "pipeline_all_locations.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("multi_location")


def alert(msg: str) -> None:
    logger.warning("=" * 60)
    logger.warning(f"ALERT: {msg}")
    logger.warning("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Inter-stage winner selection (prompt.md 4d, pipeline.selection)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_branch_final_winner(stage_winners: list, tvp_set: str, branch_label: str) -> dict | None:
    """
    Apply the best-metric-or-simplest rule (pipeline.selection) across a
    branch's stage winners (e.g. [stage1_phi, stage2_phi, stage3_phi]) to
    pick that branch's overall final winner. Logs the chain of decisions.
    """
    from pipeline.selection import select_inter_stage_winner

    candidates = [w for w in stage_winners if w is not None]
    if not candidates:
        logger.warning(f"  {branch_label}: no valid stage winner to select from.")
        return None

    final = select_inter_stage_winner(candidates, tvp_set, IMPROVEMENT_THRESHOLD)
    if final:
        key_metric = "rmse" if tvp_set == "phi_only" else "crps_mean"
        logger.info(
            f"  {branch_label} FINAL WINNER: {final['model_id']}  "
            f"{key_metric}={final.get('oos_metrics', {}).get(key_metric, float('nan')):.4f}  "
            f"({final.get('_tie_break', '')})"
        )
    return final


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: tail-sensitive xi-only regime, frozen phi (objective 3)
# ─────────────────────────────────────────────────────────────────────────────

def _stage_dir_for_model_id(run_dir: Path, model_id: str) -> Path:
    """Map a phi-only-branch model_id back to its artifact subdirectory."""
    if model_id.startswith("stage3_phi_harvey_"):
        return run_dir / "stage3_phi" / model_id
    if model_id.startswith("stage2_phi_"):
        return run_dir / "stage2_phi" / model_id
    if model_id.startswith("stage1_phi_"):
        return run_dir / "stage1" / model_id
    raise ValueError(f"Cannot map phi-only model_id to a stage directory: {model_id}")


def _base_covariate_kwargs(
    phi_final_winner: dict,
    stage2_phi_winner: dict | None,
    data: dict,
) -> tuple[dict, dict]:
    """
    Build (fit_kwargs, oos_kwargs) needed to replay the frozen phi-only
    winner's own filter, dispatching on which stage it came from:
    Stage 1 (no covariates), Stage 2 (weather X), or Stage 3 (Harvey
    X_long/X_short -- X_short is inherited from the phi-only Stage-2
    winner exactly as Stage 3 itself did during estimation).
    """
    mid = phi_final_winner["model_id"]
    bt, bv = data["covariate_blocks_train"], data["covariate_blocks_test"]

    if mid.startswith("stage1_phi_"):
        return {}, {}

    if mid.startswith("stage2_phi_"):
        block = mid.replace("stage2_phi_", "")
        return ({"X": bt[block]}, {"X_train": bt[block], "X_test": bv[block]})

    if mid.startswith("stage3_phi_harvey_"):
        enso_tier = mid.replace("stage3_phi_harvey_", "")
        empty_tr = np.zeros((len(data["y_train"]), 0))
        empty_te = np.zeros((len(data["y_test"]),  0))
        Xl_tr = bt[enso_tier] if enso_tier != "no_enso" else empty_tr
        Xl_te = bv[enso_tier] if enso_tier != "no_enso" else empty_te
        if stage2_phi_winner is None:
            raise RuntimeError(
                f"Stage-3 phi-only winner {mid} needs its Stage-2 weather "
                f"block, but no Stage-2 phi-only winner was recorded."
            )
        short_block = stage2_phi_winner.get(
            "best_weather_block", stage2_phi_winner["model_id"].replace("stage2_phi_", "")
        )
        Xs_tr, Xs_te = bt[short_block], bv[short_block]
        return (
            {"X_long": Xl_tr, "X_short": Xs_tr},
            {"X_long_train": Xl_tr, "X_long_test": Xl_te,
             "X_short_train": Xs_tr, "X_short_test": Xs_te},
        )

    raise ValueError(f"Unrecognized phi-only model_id: {mid}")


def run_stage4(
    data: dict,
    phi_final_winner: dict,
    stage2_phi_winner: dict | None,
    phixi_final_winner: dict | None,
    run_dir: Path,
    force_rerun: bool = False,
) -> dict:
    """
    Estimate the tail-sensitive xi-only regime at q95 and q98 (objective 3),
    with phi frozen at `phi_final_winner`. Each is accepted only if its
    twCRPS beats `phixi_final_winner` (the phi+xi branch's best-of-
    previous-stages counterpart, objective 3a) by >=2%.

    xi's covariates are always the short-term weather block ("dewtemp_seasonal")
    -- never Stage 3's ENSO covariates (item 3: "stages 3 and 4 for xi are
    separate things"). Confirmed applicable at every one of the six
    locations by inspecting the existing phi+xi Stage-2 winner's xi
    coefficients before this round began (see audit/implementation_plan_07072026.tex).
    """
    from models.regime_gas import build_regime_xi_only_from_frozen_phi
    from pipeline.runner import run_single_model
    from pipeline.selection import stage4_beats_base

    stage_dir = run_dir / "stage4_xi_regime"
    stage_dir.mkdir(exist_ok=True)

    if phi_final_winner is None:
        logger.warning("  Stage 4: no phi-only final winner -- skipped.")
        return {"models": [], "accepted": {}}

    base_model_dir = _stage_dir_for_model_id(run_dir, phi_final_winner["model_id"])
    fit_kw_base, oos_kw_base = _base_covariate_kwargs(phi_final_winner, stage2_phi_winner, data)

    y_train, y_test = data["y_train"], data["y_test"]
    bt, bv, cn = data["covariate_blocks_train"], data["covariate_blocks_test"], data["covariate_col_names"]

    results = {}
    for q_label, q_val in STAGE4_THRESHOLDS:
        model_id = f"stage4_xi_regime_q{q_label}"
        logger.info(f"STAGE 4  --  xi-only tail regime q{q_label}  "
                    f"(phi frozen from {phi_final_winner['model_id']})")

        model, frozen = build_regime_xi_only_from_frozen_phi(
            base_model_dir=base_model_dir,
            y_train=y_train, y_test=y_test,
            xi_cov_train=bt["dewtemp_seasonal"], xi_cov_test=bv["dewtemp_seasonal"],
            xi_cov_names=cn["dewtemp_seasonal"],
            threshold_quantile=q_val,
            base_extra_fit_kwargs=fit_kw_base, base_extra_oos_kwargs=oos_kw_base,
        )

        fit_kwargs = {
            "y": y_train, "phi_full": frozen["phi_full"], "pi_full": frozen["pi_full"],
            "gamma_v": frozen["gamma_v"], "zeta_v": frozen["zeta_v"], "warmup": frozen["warmup"],
            "X": frozen["X_train"], "verbose": False,
            "options": {"maxiter": 60}, "polish": False,
        }
        oos_kwargs = {
            "y_train": y_train, "y_test": y_test,
            "phi_full": frozen["phi_full"], "pi_full": frozen["pi_full"],
            "gamma_v": frozen["gamma_v"], "zeta_v": frozen["zeta_v"], "warmup": frozen["warmup"],
            "X_train": frozen["X_train"], "X_test": frozen["X_test"],
        }

        r = run_single_model(
            model_id=model_id, model=model,
            fit_kwargs=fit_kwargs, oos_kwargs=oos_kwargs,
            y_test=y_test, cache_dir=stage_dir,
            log_path=run_dir / "execution_log.jsonl",
            logger=logger, force_rerun=force_rerun,
        )
        results[q_label] = r

    # ── Objective 3a: twCRPS vs. the phi+xi branch's best counterpart ─────
    accepted = {}
    if phixi_final_winner is None:
        logger.warning("  Stage 4: no phi+xi final winner to compare twCRPS against.")
    else:
        base_ext = _extended_metrics_from_cached(
            phixi_final_winner["model_id"], run_dir, y_test,
        )
        for q_label, _ in STAGE4_THRESHOLDS:
            r = results[q_label]
            if r.get("result") is None and r.get("status") != "cached":
                accepted[q_label] = {"winner": "base", "reason": "stage4 fit failed"}
                continue
            s4_ext = _extended_metrics_from_cached(
                f"stage4_xi_regime_q{q_label}", run_dir, y_test, stage_dir_name="stage4_xi_regime",
            )
            key_metric = f"twcrps_{q_label}"
            decision = stage4_beats_base(
                {key_metric: s4_ext.get(key_metric, float("nan"))},
                {key_metric: base_ext.get(key_metric, float("nan"))},
                key_metric=key_metric, improvement_threshold=IMPROVEMENT_THRESHOLD,
            )
            accepted[q_label] = decision
            logger.info(
                f"  Stage 4 q{q_label} decision: {decision['winner']}  "
                f"({key_metric}: stage4={decision['stage4_value']:.4f} vs "
                f"base={decision['base_value']:.4f}, "
                f"improvement={decision['rel_improvement']:.2%})"
            )

    winners_path = run_dir / "stage_winners.json"
    try:
        existing = json.loads(winners_path.read_text()) if winners_path.exists() else {}
    except Exception:
        existing = {}
    existing["stage4_xi_regime"] = {
        "base_model": phi_final_winner["model_id"],
        "counterpart": phixi_final_winner["model_id"] if phixi_final_winner else None,
        "accepted": accepted,
        "models": [
            {"model_id": r.get("model_id"), "validity": r.get("validity"),
             "loglik": r.get("loglik"), "crps_mean": r.get("oos_metrics", {}).get("crps_mean")}
            for r in results.values()
        ],
    }
    winners_path.write_text(json.dumps(existing, indent=2, default=str))

    return {"models": results, "accepted": accepted}


def _extended_metrics_from_cached(model_id: str, run_dir: Path, y_test: np.ndarray,
                                   stage_dir_name: str | None = None) -> dict:
    """
    Recompute twCRPS/quantile-score extended metrics from a model's saved
    OOS paths -- no re-optimization (EXECUTION.md §22). Used for the
    Stage-4-vs-counterpart twCRPS comparison (objective 3a).

    Only `model.dist` (the GB2 distribution object, for ppf/cdf) is used by
    compute_extended_oos_metrics -- the reconstructed model's own recursion
    machinery is irrelevant here since `oos_paths` is supplied directly
    from the cached paths.npz rather than recomputed via model.filter().
    This is why build_model_from_meta's generic fallback path is safe to
    use even for Stage-4 artifacts (RegimeXiOnlyModel is not one of its
    explicit cases): any reconstructed wrapper has a correctly-configured
    `.dist`, which is all that is actually read below.
    """
    from pipeline.artifact_utils import load_model_and_theta
    from diagnostics.scoring import compute_extended_oos_metrics

    search_dirs = [stage_dir_name] if stage_dir_name else \
        ["stage1", "stage2", "stage3", "stage2_phi", "stage3_phi", "stage4_xi_regime"]
    for d in search_dirs:
        model_dir = run_dir / d / model_id
        npz_path = model_dir / "paths.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path)
        model, _theta, meta, params_df = load_model_and_theta(model_dir)
        if model is None:
            continue

        tv_names = meta.get("tv_param_names") or (
            ["phi", "xi"] if npz["f_arr_oos"].shape[1] > 1 else ["phi"]
        )
        # gamma/zeta/xi (when static): top-level metadata first (Stage 4
        # saves these explicitly since they're frozen, not part of its own
        # theta); otherwise read them by name from estimated_parameters.csv
        # (Stage 1-3 models keep them there since gamma/zeta are always
        # static, and xi is static too for phi-only models).
        pdict = (
            dict(zip(params_df["parameter"], params_df["value"]))
            if params_df is not None and not params_df.empty else {}
        )
        static = {}
        for k in ("gamma", "zeta", "xi"):
            if k in meta and meta[k] is not None:
                static[k] = float(meta[k])
            elif k in pdict:
                static[k] = float(pdict[k])

        oos_paths = {
            "pi_oos": npz["pi_oos"], "f_arr_oos": npz["f_arr_oos"],
            "tv_names": tv_names, "static": static,
        }
        return compute_extended_oos_metrics(model, oos_paths, y_test)
    logger.warning(f"  Could not locate cached OOS paths for {model_id} -- twCRPS unavailable.")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Single-location runner
# ─────────────────────────────────────────────────────────────────────────────

def run_location(
    station_name: str,
    workers: int = 4,
    force_rerun: bool = False,
    run_stage4_flag: bool = True,
) -> dict:
    """
    Run the full pipeline for one station: phi+xi branch (Stages 1-3),
    phi-only branch (Stages 1-3, objective 1a), and Stage 4 (objective 3).
    """
    from data.station_loader import STATION_REGISTRY, StationDataLoader
    from distributions.gb2_log_link import GB2LogLink
    from distributions.gb2_phi_only import GB2LogLinkPhiOnly
    from pi_dynamics.factory import make_pi_dynamics
    from pipeline.runner import run_pipeline

    cfg      = STATION_REGISTRY[station_name]
    run_id   = cfg["run_id"]
    display  = cfg["display"]

    logger.info("=" * 72)
    logger.info(f"LOCATION: {display}  (run_id={run_id})")
    logger.info("=" * 72)

    # ── Load data ─────────────────────────────────────────────────────────
    t_load = time.time()
    loader = StationDataLoader.from_registry(
        station_name=station_name,
        precip_dir=PRECIP_DIR,
        era5_dir=ERA5_DIR,
        enso_path=ENSO_PATH,
    )
    data = loader.load_all()
    logger.info(f"  Data loaded in {time.time()-t_load:.1f}s: "
                f"n_train={data['summary']['n_train']}  "
                f"n_test={data['summary']['n_test']}  "
                f"wet_frac={data['summary']['wet_frac_train']:.2%}")

    # ── Build model components ─────────────────────────────────────────────
    dist_phi   = GB2LogLinkPhiOnly()
    dist_phixi = GB2LogLink()
    pi_dyn     = make_pi_dynamics("ar_logistic", seasonal="daily")

    # ── Run dual-branch pipeline (Stages 1-3) ───────────────────────────────
    t_pipe = time.time()
    run_dir  = ARTIFACTS_DIR / run_id
    pipeline_out = run_pipeline(
        data          = data,
        pi_dyn        = pi_dyn,
        dist_phi      = dist_phi,
        dist_phixi    = dist_phixi,
        artifacts_dir = ARTIFACTS_DIR,
        run_id        = run_id,
        force_rerun   = force_rerun,
        do_stage1     = True,
        do_stage2     = True,
        do_stage3     = True,
        run_phi_only_branch = True,
        parallel      = True,
        workers       = workers,
    )
    elapsed_pipe = time.time() - t_pipe
    logger.info(f"  Stages 1-3 (both branches) completed in {elapsed_pipe/60:.1f} min")

    # ── Inter-stage final winners, each within its own tvp set ─────────────
    phixi_final = _resolve_branch_final_winner(
        [pipeline_out.get("stage1_winner"), pipeline_out.get("stage2_winner"),
         pipeline_out.get("stage3_winner")],
        "phi_xi", f"{display} phi+xi branch",
    )
    phi_final = _resolve_branch_final_winner(
        [pipeline_out.get("stage1_phi_winner"), pipeline_out.get("stage2_phi_winner"),
         pipeline_out.get("stage3_phi_winner")],
        "phi_only", f"{display} phi-only branch",
    )

    winners_path = run_dir / "stage_winners.json"
    try:
        existing = json.loads(winners_path.read_text()) if winners_path.exists() else {}
        existing["final_winner_phi_xi"] = {
            "model_id": phixi_final["model_id"] if phixi_final else None,
            "crps_mean": phixi_final.get("oos_metrics", {}).get("crps_mean") if phixi_final else None,
        }
        existing["final_winner_phi_only"] = {
            "model_id": phi_final["model_id"] if phi_final else None,
            "rmse": phi_final.get("oos_metrics", {}).get("rmse") if phi_final else None,
        }
        winners_path.write_text(json.dumps(existing, indent=2, default=str))
    except Exception as exc:
        logger.warning(f"  Could not update stage_winners.json: {exc}")

    # ── Stage 4: tail-sensitive xi regime (objective 3) ─────────────────────
    stage4_out = {"models": {}, "accepted": {}}
    if run_stage4_flag:
        t_s4 = time.time()
        stage4_out = run_stage4(
            data=data,
            phi_final_winner=phi_final,
            stage2_phi_winner=pipeline_out.get("stage2_phi_winner"),
            phixi_final_winner=phixi_final,
            run_dir=run_dir,
            force_rerun=force_rerun,
        )
        logger.info(f"  Stage 4 completed in {time.time()-t_s4:.1f}s")
    elif not phi_final:
        alert(f"{display}: no phi-only final winner -- Stage 4 skipped.")

    return {
        "station":            station_name,
        "run_id":             run_id,
        "run_dir":            str(run_dir),
        "phixi_final_winner": phixi_final,
        "phi_final_winner":   phi_final,
        "stage4_out":         stage4_out,
        "pipeline_out":       pipeline_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ZA-GAS pipeline for the six reported precipitation stations."
    )
    parser.add_argument("--workers",     type=int, default=4,
                        help="Parallel workers per stage (default 4)")
    parser.add_argument("--force",       action="store_true",
                        help="Re-estimate even when valid artifacts exist")
    parser.add_argument("--skip-stage4", action="store_true",
                        help="Skip Stage 4 (tail-sensitive xi regime)")
    parser.add_argument("--location",    type=str, default=None,
                        help="Run only this location (exact name)")
    parser.add_argument("--start-from",  type=str, default=None,
                        help="Skip locations before this one in LOCATION_ORDER")
    args = parser.parse_args()

    # Verify data paths
    for name, path in [
        ("PRECIP_DIR", PRECIP_DIR),
        ("ERA5_DIR",   ERA5_DIR),
        ("ENSO_PATH",  ENSO_PATH),
    ]:
        if not path.exists():
            alert(f"{name} not found: {path}")
            sys.exit(1)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build list of stations to process
    if args.location:
        stations = [args.location]
    else:
        stations = list(LOCATION_ORDER)
        if args.start_from:
            try:
                idx      = stations.index(args.start_from)
                stations = stations[idx:]
            except ValueError:
                alert(f"--start-from '{args.start_from}' not found in LOCATION_ORDER.")
                sys.exit(1)

    logger.info(f"Stations to process ({len(stations)}): {stations}")
    logger.info(f"Workers per stage:  {args.workers}")
    logger.info(f"Force rerun:        {args.force}")
    logger.info(f"Run Stage 4:        {not args.skip_stage4}")

    all_results = {}
    t_total = time.time()

    for station in stations:
        if station in SKIP_STATIONS:
            logger.info(f"SKIP: {station} (excluded)")
            continue

        try:
            t_loc = time.time()
            result = run_location(
                station_name=station,
                workers=args.workers,
                force_rerun=args.force,
                run_stage4_flag=not args.skip_stage4,
            )
            elapsed_loc = time.time() - t_loc
            logger.info(f"  Location completed in {elapsed_loc/60:.1f} min")
            all_results[station] = result

        except Exception as exc:
            import traceback
            alert(f"Location {station} raised unhandled exception: {exc}")
            logger.error(traceback.format_exc())
            all_results[station] = {"station": station, "error": str(exc)}

    total_elapsed = time.time() - t_total
    logger.info("=" * 72)
    logger.info(f"ALL LOCATIONS COMPLETE -- total wall time {total_elapsed/60:.1f} min")
    logger.info("=" * 72)

    # Summary table
    logger.info("Final winner summary:")
    logger.info(f"  {'Location':<30} {'phi-only winner (RMSE)':<32} {'phi+xi winner (CRPS)':<32}")
    logger.info(f"  {'-'*30} {'-'*32} {'-'*32}")
    for station, res in all_results.items():
        pf  = res.get("phi_final_winner")
        pxf = res.get("phixi_final_winner")
        pf_s  = pf["model_id"] if pf else "(none)"
        pxf_s = pxf["model_id"] if pxf else "(none)"
        logger.info(f"  {station:<30} {pf_s:<32} {pxf_s:<32}")


if __name__ == "__main__":
    main()
