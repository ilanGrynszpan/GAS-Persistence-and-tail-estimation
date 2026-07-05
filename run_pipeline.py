"""
Standalone script to run the sequential thesis pipeline.

Usage:
    python run_pipeline.py [--force] [--stage1-only] [--stage2-only] [--stage3-only]

This script mirrors the logic in sequential_models_01072026.ipynb and can be
run in the background. Progress and alerts are written to:
    pipeline_run.log            (human-readable log)
    artifacts/run_20260701_bh/execution_log.jsonl  (machine-readable events)
"""

from __future__ import annotations

import sys
import logging
import json
import argparse
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
ARTIFACTS_DIR = ROOT / "artifacts"
RUN_ID        = "run_20260701_bh"

PRECIP_DIR  = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR    = ROOT / "data" / "input" / "ERA5"
NINO34_PATH = ROOT / "data" / "processed" / "pacific" / "NINO34_daily.csv"

# ── Logging ──────────────────────────────────────────────────────────────────
log_path = ROOT / "pipeline_run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_path, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("pipeline")


def alert(msg: str) -> None:
    """Emit a high-visibility alert line."""
    logger.warning("=" * 60)
    logger.warning(f"ALERT: {msg}")
    logger.warning("=" * 60)


# ── Python path ──────────────────────────────────────────────────────────────
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force",       action="store_true", help="Re-estimate all models")
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--stage2-only", action="store_true")
    parser.add_argument("--stage3-only", action="store_true")
    parser.add_argument("--parallel",    action="store_true", help="Parallel Stage 1 & 3")
    parser.add_argument("--workers",     type=int, default=4, help="Worker processes")
    args = parser.parse_args()

    do_stage1 = not (args.stage2_only or args.stage3_only)
    do_stage2 = not (args.stage1_only or args.stage3_only)
    do_stage3 = not (args.stage1_only or args.stage2_only)

    # Verify data paths
    if not PRECIP_DIR.exists():
        alert(f"Precipitation directory not found: {PRECIP_DIR}")
        sys.exit(1)
    if not ERA5_DIR.exists():
        alert(f"ERA5 directory not found: {ERA5_DIR}")
        sys.exit(1)
    if not NINO34_PATH.exists():
        alert(f"NINO34 file not found: {NINO34_PATH}")
        sys.exit(1)

    logger.info(f"Starting pipeline  run_id={RUN_ID}  force={args.force}")
    logger.info(f"Stages: S1={do_stage1}  S2={do_stage2}  S3={do_stage3}")

    # ── Imports ───────────────────────────────────────────────────────────────
    logger.info("Loading framework modules...")
    try:
        from data.loader import BHDataLoader
        from distributions.gb2_log_link import GB2LogLink
        from distributions.gb2_phi_only import GB2LogLinkPhiOnly
        from pi_dynamics.factory import make_pi_dynamics
        from pipeline.runner import run_pipeline
    except ImportError as exc:
        alert(f"Import failed: {exc}")
        sys.exit(1)
    logger.info("Imports OK")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading data...")
    try:
        loader = BHDataLoader(
            precip_dir  = PRECIP_DIR,
            era5_dir    = ERA5_DIR,
            nino34_path = NINO34_PATH,
        )
        data = loader.load_all()
    except Exception as exc:
        alert(f"Data loading failed: {exc}")
        raise

    logger.info("Data loaded:")
    for k, v in data.get("summary", {}).items():
        logger.info(f"  {k}: {v}")
    logger.info("Covariate blocks:")
    for name, arr in data.get("covariate_blocks_train", {}).items():
        logger.info(f"  {name}: shape={arr.shape}")

    # ── Build model components ────────────────────────────────────────────────
    dist_phi   = GB2LogLinkPhiOnly()
    dist_phixi = GB2LogLink()
    pi_dyn     = make_pi_dynamics("ar_logistic", seasonal="daily")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        output = run_pipeline(
            data          = data,
            pi_dyn        = pi_dyn,
            dist_phi      = dist_phi,
            dist_phixi    = dist_phixi,
            artifacts_dir = ARTIFACTS_DIR,
            run_id        = RUN_ID,
            force_rerun   = args.force,
            do_stage1     = do_stage1,
            do_stage2     = do_stage2,
            do_stage3     = do_stage3,
            parallel      = args.parallel,
            workers       = args.workers,
        )
    except Exception as exc:
        elapsed = time.time() - t0
        alert(f"Pipeline raised unhandled exception after {elapsed:.0f}s: {exc}")
        raise

    elapsed = time.time() - t0
    logger.info(f"Pipeline completed in {elapsed/60:.1f} min")
    logger.info(f"Run directory: {output.get('run_dir')}")

    # ── Results summary ───────────────────────────────────────────────────────
    def _fmt(r):
        if r is None:
            return "(none)"
        ll   = r.get("loglik", float("nan"))
        crps = r.get("oos_metrics", {}).get("crps_mean", float("nan"))
        return f"{r['model_id']}  loglik={ll:.2f}  crps={crps:.4f}  validity={r.get('validity')}"

    logger.info(f"Stage 1 winner: {_fmt(output.get('stage1_winner'))}")
    logger.info(f"Stage 2 winner: {_fmt(output.get('stage2_winner'))}")
    logger.info(f"Stage 3 winner: {_fmt(output.get('stage3_winner'))}")

    # Check for any failed models
    all_results = (
        output.get("stage1_results", [])
        + output.get("stage2_results", [])
        + output.get("stage3_results", [])
    )
    failed = [r["model_id"] for r in all_results if r.get("status") == "failed"]
    if failed:
        alert(f"{len(failed)} model(s) FAILED: {failed}")
    else:
        logger.info("All models completed successfully.")


if __name__ == "__main__":
    main()
