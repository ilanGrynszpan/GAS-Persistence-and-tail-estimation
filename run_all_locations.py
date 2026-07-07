"""
Multi-location pipeline runner.

=============================================================================
OVERVIEW
=============================================================================

Runs the complete ZA-GAS three-stage estimation pipeline for every location
in STATION_REGISTRY, except:
  - BELO HORIZONTE  (already estimated; artifacts in run_20260701_bh)
  - RIYADH          (excluded: bad data file per user instruction)

Then runs Stage 4 (regime-sensitive GAS) for every location using the
best model from Stages 1-3.

Execution is SEQUENTIAL across locations to avoid RAM saturation
(available RAM ≈ 6 GB; each stage uses up to 3-4 GB during optimisation).
Within each location, Stage 1 and Stage 3 run in parallel (--workers workers).
Stage 2 runs sequentially with warm starts (architecture requirement).

=============================================================================
USAGE
=============================================================================

    python run_all_locations.py [options]

Options:
    --workers N       Number of parallel workers per stage (default 4)
    --force           Re-estimate even when valid artifacts exist
    --skip-regime     Skip Stage 4 regime models
    --location NAME   Run only one location (exact name from STATION_REGISTRY)
    --start-from NAME Skip all locations before NAME in the processing order

=============================================================================
ARTIFACT STRUCTURE
=============================================================================

artifacts/
    run_20260701_bh/           (BH — already done)
    run_20260705_cruzeiro/
        stage1/ stage2/ stage3/ stage4_regime/
        stage_winners.json
        execution_log.jsonl
    run_20260705_darwin/
    ...  (one directory per location)

=============================================================================
MONITORING
=============================================================================

Progress is written to:
    pipeline_all_locations.log     — human-readable, tee'd to stdout
    artifacts/<run_id>/execution_log.jsonl  — per-model machine-readable

This script actively monitors each model's runtime and will log alerts for:
    - Models taking more than 30 min (possibly stuck)
    - Models that fail (failure isolation: pipeline continues)
    - Stages with no valid winner
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
NINO34_PATH = ROOT / "data" / "processed" / "pacific" / "NINO34_daily.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

# Locations that skip this script (already estimated or excluded)
SKIP_STATIONS = {"BELO HORIZONTE", "RIYADH OBS. (O.A.P."}

# Processing order (stable, alphabetical)
LOCATION_ORDER = [
    "CRUZEIRO DO SUL (ACRE)",
    "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)",
    "MANAUS",
    "SALVADOR",
    "SÃO PAULO",
    "TORONTO",
]

# OOS CRPS improvement threshold to accept a higher stage (prompt §: 2%)
CRPS_IMPROVEMENT_THRESHOLD = 0.02

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
# Stage-advancement logic
# ─────────────────────────────────────────────────────────────────────────────

def _should_advance(
    current_crps: float,
    new_crps: float,
    threshold: float = CRPS_IMPROVEMENT_THRESHOLD,
    stage_name: str = "next",
) -> bool:
    """
    Return True if accepting the new stage reduces OOS CRPS by ≥ threshold.
    """
    if not np.isfinite(current_crps) or not np.isfinite(new_crps):
        return False
    improvement = (current_crps - new_crps) / max(abs(current_crps), 1e-12)
    logger.info(f"  Stage advancement check ({stage_name}): "
                f"current CRPS={current_crps:.4f}  new CRPS={new_crps:.4f}  "
                f"improvement={improvement:.2%}  "
                f"{'ACCEPT' if improvement >= threshold else 'REJECT'}")
    return improvement >= threshold


def _get_crps(winner: dict | None) -> float:
    if winner is None:
        return float("nan")
    return float(winner.get("oos_metrics", {}).get("crps_mean", float("nan")))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Regime models
# ─────────────────────────────────────────────────────────────────────────────

def _find_base_for_regime(
    stage1_winner: dict | None,
    stage2_winner: dict | None,
    crps_s1: float,
    crps_s2: float,
) -> tuple[dict | None, str]:
    """
    Return the Stage 1 or 2 winner to use as the base for regime models.

    Stage 2 is accepted only if it improved CRPS ≥ 2 % over Stage 1.
    Stage 3 (Harvey) is never used as base — the regime model is always
    GAS(1,1), which is incompatible with the Harvey long/short structure.

    Returns (winner_dict, label) where label is "stage1" or "stage2".
    """
    if (
        stage2_winner
        and np.isfinite(crps_s1)
        and np.isfinite(crps_s2)
        and (crps_s1 - crps_s2) / max(abs(crps_s1), 1e-12) >= CRPS_IMPROVEMENT_THRESHOLD
    ):
        logger.info("  Stage 4: using Stage-2 winner as regime base (≥2% CRPS gain).")
        return stage2_winner, "stage2"
    logger.info("  Stage 4: using Stage-1 winner as regime base.")
    return stage1_winner, "stage1"


def run_regime_stage(
    data: dict,
    stage1_winner: dict | None,
    stage2_winner: dict | None,
    crps_s1: float,
    crps_s2: float,
    run_dir: Path,
    log_path: Path,
    workers: int = 4,
    force_rerun: bool = False,
) -> list:
    """
    Run Stage 4 regime-sensitive GAS models.

    Always estimates GAS(1,1) with phi+xi distribution.  Tests:
        3 thresholds × 3 TV-regime combos = 9 models

    Thresholds (wet-day quantile):  90 %, 95 %, 98 %
    TV-regime combos:
        phi_only   — only phi gets the A_ext regime term
        xi_only    — only xi  gets the A_ext regime term
        phi_xi     — both phi and xi get A_ext regime terms

    The non-regime parameter still has full GAS(1,1) dynamics (omega, B, A).

    The base configuration is taken from the best Stage-1/2 winner
    (Stage 2 accepted iff CRPS improved ≥ 2 % over Stage 1).
    Harvey Stage-3 winners are never used directly — the regime model
    falls back to the best of Stage 1 / Stage 2 as described above.

    Parameters
    ----------
    data           : output of StationDataLoader.load_all()
    stage1_winner  : pipeline output dict for Stage-1 winner
    stage2_winner  : pipeline output dict for Stage-2 winner (may be None)
    crps_s1/s2     : OOS CRPS for stage 1 / 2 winners
    run_dir        : location artifact directory
    log_path       : jsonl log file
    workers        : parallel workers (shared with Stage 1–3)
    force_rerun    : re-estimate even when valid artifacts exist
    """
    from models.regime_gas import build_regime_from_winner
    from distributions.gb2_log_link import GB2LogLink
    from pi_dynamics.factory import make_pi_dynamics
    from pipeline.runner import run_single_model, run_stage_parallel

    import json as _json
    import pandas as pd

    stage_dir = run_dir / "stage4_regime"
    stage_dir.mkdir(exist_ok=True)

    dist_phixi = GB2LogLink()
    pi_dyn     = make_pi_dynamics("ar_logistic", seasonal="daily")

    # ── Select base model ─────────────────────────────────────────────────
    base_winner, base_stage_label = _find_base_for_regime(
        stage1_winner, stage2_winner, crps_s1, crps_s2
    )
    if base_winner is None:
        logger.info("  Stage 4: no Stage-1 winner — regime stage skipped.")
        return []

    # Load base winner parameters for warm-starting
    base_model_id = base_winner.get("model_id", "")
    base_dir      = run_dir / base_stage_label / base_model_id
    base_params_df = None
    try:
        base_params_df = pd.read_csv(base_dir / "estimated_parameters.csv")
    except Exception as exc:
        logger.warning(f"  Stage 4: could not load base params ({exc}); using fresh init.")

    y_train = data["y_train"]
    y_test  = data["y_test"]

    # ── Build 9 specs: 3 thresholds × 3 TV-regime combos ─────────────────
    THRESHOLDS = [("90", 0.90), ("95", 0.95), ("98", 0.98)]
    TV_COMBOS  = [
        ("phi_only", ["phi"]),
        ("xi_only",  ["xi"]),
        ("phi_xi",   ["phi", "xi"]),
    ]

    # 7200 s stage-level timeout.  All 9 futures are submitted simultaneously
    # so future_start[] is effectively the stage start time for every future —
    # this is a stage budget, not a per-model cap.  With maxiter=30 and ~2100 s
    # max per model, 3 batches of 4 workers need ≤ 3 × 2100 = 6300 s < 7200 s.
    REGIME_MODEL_TIMEOUT = 7200

    # Scaling fallback: try diagfi first; if ALL models produce non-finite
    # CRPS (divergence or numerical failure), retry with unit then fullfi.
    # diagfi uses original model IDs (backward compatible with existing
    # artifacts).  Fallback scalings use a prefix to avoid ID collisions.
    SCALINGS_TO_TRY = [
        ("",        "diagonal_inverse_fisher"),   # primary — no prefix
        ("unit_",   "unit"),                       # fallback 1
        ("fullfi_", "full_inverse_fisher"),        # fallback 2
    ]

    results = []
    for sc_prefix, sc_name in SCALINGS_TO_TRY:
        specs = []
        for q_label, q_val in THRESHOLDS:
            for tv_label, regime_tv in TV_COMBOS:
                model_id = f"stage4_regime_{sc_prefix}{q_label}_{tv_label}"
                model, theta0 = build_regime_from_winner(
                    dist_phixi=dist_phixi,
                    pi_dyn=pi_dyn,
                    regime_tv_names=regime_tv,
                    threshold_quantile=q_val,
                    base_params_df=base_params_df,
                    y_train=y_train,
                    scaling=sc_name,
                )
                # Regime models use a Python-loop GAS filter (~1.8 s/eval on
                # daily data of ~3000 obs).  maxiter=30 → ~27 min/model wall-
                # clock which fits comfortably inside the 7200 s stage budget.
                fit_kw = {
                    "y":      y_train,
                    "verbose": False,
                    "options": {"maxiter": 30},
                    "polish":  False,
                }
                if theta0 is not None:
                    fit_kw["theta0"] = theta0
                oos_kw = {"y_train": y_train, "y_test": y_test}
                specs.append((model_id, model, fit_kw, oos_kw))

        sc_display = sc_name.replace("diagonal_inverse_fisher", "diagfi")
        logger.info(
            f"STAGE 4  —  Regime-sensitive GAS ({len(specs)} models: "
            f"3 thresholds × 3 TV combos, base={base_model_id}, scaling={sc_display})"
        )

        if workers > 1 and len(specs) > 1:
            sc_results = run_stage_parallel(
                specs=specs, y_test=y_test,
                stage_dir=stage_dir, log_path=log_path, logger=logger,
                workers=min(workers, len(specs)), force_rerun=force_rerun,
                model_timeout=REGIME_MODEL_TIMEOUT,
            )
        else:
            sc_results = []
            for model_id, model, fit_kw, oos_kw in specs:
                r = run_single_model(
                    model_id=model_id, model=model,
                    fit_kwargs=fit_kw, oos_kwargs=oos_kw,
                    y_test=y_test,
                    cache_dir=stage_dir, log_path=log_path,
                    logger=logger, force_rerun=force_rerun,
                )
                sc_results.append(r)

        results.extend(sc_results)

        # Check whether any model in this scaling batch converged *well*.
        # Two criteria must both hold:
        #   (1) finite OOS CRPS;
        #   (2) loglik not catastrophically worse than the base model.
        # Criterion (2) catches scaling-mismatch divergence: when unit-scaling
        # Stage-1 init params are fed into a diagfi Stage-4 filter, the
        # optimizer finds a degenerate solution with finite-but-terrible CRPS
        # and loglik 5-10× worse than Stage 1.  Without (2) the fallback would
        # never trigger because (1) alone is satisfied.
        # Threshold = 3× magnitude of base loglik (ratio 1.15 for Garanhuns
        # where diagfi is consistent; ratio 8.18 for Manaus mismatch case).
        base_loglik_abs = abs(float(base_winner.get("loglik", 0.0) or 0.0))

        _good = [
            r for r in sc_results
            if r is not None
            and np.isfinite(float(r.get("oos_metrics", {}).get("crps_mean", float("nan"))))
            and (
                base_loglik_abs <= 0
                or abs(float(r.get("loglik", 0.0) or 0.0)) < 3.0 * base_loglik_abs
            )
        ]
        if _good:
            logger.info(
                f"  Stage 4 ({sc_display}): {len(_good)}/{len(sc_results)} models "
                f"converged well (finite CRPS + loglik < 3× base) "
                f"— stopping scaling search."
            )
            break
        else:
            _finite_bad = [
                r for r in sc_results
                if r is not None
                and np.isfinite(float(r.get("oos_metrics", {}).get("crps_mean", float("nan"))))
            ]
            if _finite_bad:
                logger.warning(
                    f"  Stage 4 ({sc_display}): {len(_finite_bad)}/{len(sc_results)} "
                    f"models have finite CRPS but loglik degradation exceeds 3× base "
                    f"(scaling-mismatch divergence) — trying next scaling."
                )
            else:
                logger.warning(
                    f"  Stage 4 ({sc_display}): no model produced finite CRPS "
                    f"— trying next scaling."
                )

    # ── Stage 4 acceptance decision ───────────────────────────────────────
    # Compare the best well-converged regime CRPS against the base model CRPS.
    # "Well-converged" = finite CRPS AND loglik not catastrophically worse than
    # base (the same criterion used in the scaling-fallback loop above).
    base_loglik_abs_accept = abs(float(base_winner.get("loglik", 0.0) or 0.0)) if base_winner else 0.0

    finite_well_converged = [
        r for r in results
        if r is not None
        and np.isfinite(float(r.get("oos_metrics", {}).get("crps_mean", float("nan"))))
        and (
            base_loglik_abs_accept <= 0
            or abs(float(r.get("loglik", 0.0) or 0.0)) < 3.0 * base_loglik_abs_accept
        )
    ]

    if finite_well_converged:
        best_regime = min(
            finite_well_converged,
            key=lambda r: float(r.get("oos_metrics", {}).get("crps_mean", float("inf"))),
        )
        best_regime_crps = float(best_regime.get("oos_metrics", {}).get("crps_mean"))
        best_regime_id   = best_regime.get("model_id", "?")
    else:
        best_regime_crps = float("nan")
        best_regime_id   = None

    # Base CRPS: prefer crps_s1 (Bug-1-corrected value) over base_winner.crps
    # which may be NaN for fullfi winners.
    base_crps_accept = crps_s1
    if base_stage_label == "stage2" and np.isfinite(float(crps_s2 or float("nan"))):
        base_crps_accept = crps_s2

    if np.isfinite(base_crps_accept) and np.isfinite(best_regime_crps):
        s4_improvement = (base_crps_accept - best_regime_crps) / max(abs(base_crps_accept), 1e-12)
        stage4_accepted = s4_improvement >= CRPS_IMPROVEMENT_THRESHOLD
        stage4_note = (
            f"Best regime model: {best_regime_id} (CRPS={best_regime_crps:.4f}); "
            f"base CRPS={base_crps_accept:.4f}; "
            f"improvement={s4_improvement:.2%}; "
            f"{'ACCEPTED' if stage4_accepted else 'NOT ACCEPTED (below 2% threshold)'}."
        )
    elif not np.isfinite(best_regime_crps):
        stage4_accepted = False
        stage4_note = (
            "Stage 4 not accepted: no well-converged regime model found "
            "(all models diverged or produced non-finite CRPS). "
            "Likely cause: scaling mismatch between Stage-1 init params and regime filter."
        )
    else:
        stage4_accepted = False
        stage4_note = "Stage 4 not accepted: base CRPS not finite (fullfi instability)."

    logger.info(f"  Stage 4 decision: {stage4_note}")

    # ── Append results + decision to stage_winners.json ───────────────────
    winners_path = run_dir / "stage_winners.json"
    try:
        existing = _json.loads(winners_path.read_text()) if winners_path.exists() else {}
    except Exception:
        existing = {}

    existing["stage4_regime"] = {
        "base_model":       base_model_id,
        "base_source":      base_stage_label,
        "base_crps":        base_crps_accept if np.isfinite(base_crps_accept) else None,
        "best_regime_crps": best_regime_crps if np.isfinite(best_regime_crps) else None,
        "stage4_accepted":  stage4_accepted,
        "stage4_note":      stage4_note,
        "models": [
            {
                "model_id": r.get("model_id"),
                "validity":  r.get("validity"),
                "loglik":    r.get("loglik"),
                "crps_mean": r.get("oos_metrics", {}).get("crps_mean"),
            }
            for r in results
        ],
    }
    winners_path.write_text(_json.dumps(existing, indent=2, default=str))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Single-location runner
# ─────────────────────────────────────────────────────────────────────────────

def run_location(
    station_name: str,
    workers: int = 4,
    force_rerun: bool = False,
    run_regime: bool = True,
) -> dict:
    """
    Run the full 3-stage (+ optional regime) pipeline for one station.

    Returns dict with stage1_winner, stage2_winner, stage3_winner,
    final_winner, regime_results, run_dir.
    """
    from data.station_loader import STATION_REGISTRY, StationDataLoader
    from distributions.gb2_log_link import GB2LogLink
    from distributions.gb2_phi_only import GB2LogLinkPhiOnly
    from pi_dynamics.factory import make_pi_dynamics
    from pipeline.runner import run_pipeline, _save_stage_summary

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
        nino34_path=NINO34_PATH,
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

    # ── Run 3-stage pipeline ───────────────────────────────────────────────
    t_pipe = time.time()
    run_dir    = ARTIFACTS_DIR / run_id
    log_path   = run_dir / "execution_log.jsonl"

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
        parallel      = True,
        workers       = workers,
    )
    elapsed_pipe = time.time() - t_pipe
    logger.info(f"  3-stage pipeline completed in {elapsed_pipe/60:.1f} min")

    # ── Determine final winner using 2% CRPS threshold ────────────────────
    w1 = pipeline_out.get("stage1_winner")
    w2 = pipeline_out.get("stage2_winner")
    w3 = pipeline_out.get("stage3_winner")

    crps_s1 = _get_crps(w1)
    crps_s2 = _get_crps(w2)
    crps_s3 = _get_crps(w3)

    # If the Stage-1 winner has NaN CRPS (fullfi instability), fall back to
    # the best valid-CRPS Stage-1 result for comparison — avoids Stage 2 being
    # permanently unacceptable due to a non-deployable Stage-1 winner.
    if not np.isfinite(crps_s1):
        r1_list = pipeline_out.get("stage1_results", [])
        _valid_s1 = [
            r for r in r1_list
            if r.get("validity", "failed") != "failed"
            and np.isfinite(float(r.get("oos_metrics", {}).get("crps_mean", float("nan"))))
        ]
        if _valid_s1:
            from pipeline.runner import select_winner as _sw
            _fb = _sw(_valid_s1)
            if _fb:
                _fb_crps = float(_fb.get("oos_metrics", {}).get("crps_mean", float("nan")))
                logger.info(
                    f"  Stage-1 winner ({w1['model_id']}) has NaN CRPS; "
                    f"using {_fb['model_id']} (crps={_fb_crps:.4f}) "
                    f"for stage-advancement comparison."
                )
                crps_s1 = _fb_crps

    # Select stage 2 only if it improves ≥ 2% over stage 1
    if w2 and _should_advance(crps_s1, crps_s2, stage_name="Stage2 vs Stage1"):
        base_winner = w2
        base_crps   = crps_s2
    else:
        base_winner = w1
        base_crps   = crps_s1
        if w2:
            logger.info(f"  Stage 2 not accepted (insufficient CRPS improvement)")

    # Select stage 3 only if it improves ≥ 2% over current base
    if w3 and _should_advance(base_crps, crps_s3, stage_name="Stage3 vs Base"):
        final_winner = w3
    else:
        final_winner = base_winner
        if w3:
            logger.info(f"  Stage 3 not accepted (insufficient CRPS improvement)")

    # If the final winner has NaN CRPS (fullfi instability in Stage 1),
    # substitute the best valid-CRPS Stage-1 model for diagnostics and
    # reporting.  The original loglik winner is preserved in a note field.
    final_winner_note = None
    if final_winner is not None and not np.isfinite(_get_crps(final_winner)):
        r1_list = pipeline_out.get("stage1_results", [])
        _valid_s1_fw = [
            r for r in r1_list
            if r.get("validity", "failed") != "failed"
            and np.isfinite(float(r.get("oos_metrics", {}).get("crps_mean", float("nan"))))
        ]
        if _valid_s1_fw:
            from pipeline.runner import select_winner as _sw_fw
            _fb_fw = _sw_fw(_valid_s1_fw)
            if _fb_fw:
                _fb_fw_crps = float(_fb_fw.get("oos_metrics", {}).get("crps_mean", float("nan")))
                final_winner_note = (
                    f"Nominal loglik winner {final_winner['model_id']} has NaN OOS CRPS "
                    f"(fullfi Fisher-information scaling produces numerically unstable GB2 ppf() "
                    f"during OOS evaluation). Best deployable model with finite OOS CRPS: "
                    f"{_fb_fw['model_id']} (crps={_fb_fw_crps:.4f})."
                )
                logger.info(
                    f"  Final winner {final_winner['model_id']} has NaN OOS CRPS "
                    f"(fullfi instability); substituting {_fb_fw['model_id']} "
                    f"(crps={_fb_fw_crps:.4f}) for diagnostics and reporting."
                )
                final_winner = _fb_fw

    logger.info(
        f"  Final winner: {final_winner['model_id'] if final_winner else 'none'}  "
        f"CRPS={_get_crps(final_winner):.4f}"
    )

    # Persist advancement decision in stage_winners.json
    winners_path = run_dir / "stage_winners.json"
    try:
        import json as _json
        existing = _json.loads(winners_path.read_text()) if winners_path.exists() else {}
        fw_entry = {
            "model_id": final_winner["model_id"] if final_winner else None,
            "crps":     _get_crps(final_winner),
            "crps_s1":  crps_s1,
            "crps_s2":  crps_s2,
            "crps_s3":  crps_s3,
        }
        if final_winner_note:
            fw_entry["numerical_note"] = final_winner_note
        existing["final_winner"] = fw_entry
        winners_path.write_text(_json.dumps(existing, indent=2, default=str))
    except Exception as exc:
        logger.warning(f"  Could not update stage_winners.json: {exc}")

    # ── Stage 4: Regime models ─────────────────────────────────────────────
    regime_results = []
    if run_regime:
        t_regime = time.time()
        regime_results = run_regime_stage(
            data=data,
            stage1_winner=w1,
            stage2_winner=w2,
            crps_s1=crps_s1,
            crps_s2=crps_s2,
            run_dir=run_dir,
            log_path=log_path,
            workers=workers,
            force_rerun=force_rerun,
        )
        logger.info(f"  Stage 4 regime completed in {time.time()-t_regime:.1f}s")
    else:
        if not final_winner:
            alert(f"{display}: no final winner — regime stage skipped.")

    return {
        "station":        station_name,
        "run_id":         run_id,
        "run_dir":        str(run_dir),
        "stage1_winner":  w1,
        "stage2_winner":  w2,
        "stage3_winner":  w3,
        "final_winner":   final_winner,
        "regime_results": regime_results,
        "pipeline_out":   pipeline_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full ZA-GAS pipeline for all precipitation stations."
    )
    parser.add_argument("--workers",     type=int, default=4,
                        help="Parallel workers per stage (default 4)")
    parser.add_argument("--force",       action="store_true",
                        help="Re-estimate even when valid artifacts exist")
    parser.add_argument("--skip-regime", action="store_true",
                        help="Skip Stage 4 regime models")
    parser.add_argument("--location",    type=str, default=None,
                        help="Run only this location (exact name)")
    parser.add_argument("--start-from",  type=str, default=None,
                        help="Skip locations before this one in LOCATION_ORDER")
    args = parser.parse_args()

    # Verify data paths
    for name, path in [
        ("PRECIP_DIR",  PRECIP_DIR),
        ("ERA5_DIR",    ERA5_DIR),
        ("NINO34_PATH", NINO34_PATH),
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
    logger.info(f"Run regime models:  {not args.skip_regime}")

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
                run_regime=not args.skip_regime,
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
    logger.info(f"ALL LOCATIONS COMPLETE — total wall time {total_elapsed/60:.1f} min")
    logger.info("=" * 72)

    # Summary table
    logger.info("Final winner summary:")
    logger.info(f"  {'Location':<30} {'Winner model':<40} {'CRPS':>8}")
    logger.info(f"  {'-'*30} {'-'*40} {'-'*8}")
    for station, res in all_results.items():
        fw   = res.get("final_winner")
        mid  = fw["model_id"] if fw else "(none)"
        crps = f"{_get_crps(fw):.4f}" if fw else "  NaN"
        logger.info(f"  {station:<30} {mid:<40} {crps:>8}")


if __name__ == "__main__":
    main()
