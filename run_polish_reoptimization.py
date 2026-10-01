"""
Polish re-optimization audit: every Stage 1/2 GAS model, all 6 locations
(2026-09-01).

=============================================================================
OVERVIEW
=============================================================================

Answers the question raised by reports/seasonality GAS/
gradient_and_convergence_diagnosis.md at full scale, not just the two
illustrative examples there: for EVERY Stage 1 (baseline) and Stage 2
(weather-covariate) model already saved anywhere in the 6-location
pipeline (120 models), restart BFGS from the exact already-saved theta
with a generous iteration budget (2000, vs. the original fit's 1000) and
record whether the gradient shrinks to a genuinely reliable level, or
whether it was already at (or very near) the best this class of model can
do from that starting point.

Per docs/EXECUTION.md/OPTIMIZATION.md ("never overwrite previous
optimization outputs" without being asked to), this DOES NOT touch the
original artifacts/<run_id>/stage1|stage2[_phi]/<model>/ directories or
stage_winners.json -- every result is written to a new, separate location.
Whether to promote any of these re-polished fits into the production
artifacts is a separate decision for the user to make after seeing the
results.

=============================================================================
USAGE
=============================================================================

    python run_polish_reoptimization.py
    python run_polish_reoptimization.py --location "MANAUS"

Output:
    artifacts/polish_reoptimization/<run_id>/<stage>/<model>/
        result.json  (old vs. new grad/objective/loglik/se_quality)
    artifacts/polish_reoptimization/summary.csv
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
from scipy.optimize import minimize

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import STATION_REGISTRY, StationDataLoader
from pipeline.artifact_utils import load_model_and_theta

PRECIP_DIR = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR   = ROOT / "data" / "input" / "ERA5"
ENSO_PATH  = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
OUT_DIR    = ARTIFACTS_DIR / "polish_reoptimization"

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]
STAGE_DIRS = ("stage1", "stage2", "stage2_phi")
RELIABLE_GRAD_THRESHOLD = 1.0     # this session's practical "trust the SE" cutoff
STRICT_CONVERGED_THRESHOLD = 1e-2  # this codebase's own "valid_converged" cutoff
MAIN_MAXITER = 2000


def _block_name(model_id: str) -> str:
    return model_id.replace("stage2_phi_", "").replace("stage2_", "")


def polish_one(run_id: str, stage: str, model_id: str, data: dict) -> dict:
    model_dir = ARTIFACTS_DIR / run_id / stage / model_id
    model, theta0, meta, _ = load_model_and_theta(model_dir)
    if model is None or theta0 is None:
        raise RuntimeError(f"Could not reconstruct {model_dir}")

    y_train = data["y_train"]
    is_cov = stage != "stage1"
    fit_args = [y_train]
    if is_cov:
        block = _block_name(model_id)
        fit_args.append(data["covariate_blocks_train"][block])

    old_grad = meta.get("grad_norm_inf", np.nan)
    old_loglik = meta.get("loglik", np.nan)
    old_se_quality = meta.get("se_quality", "unavailable")

    t0 = time.time()
    result = model.fit(*fit_args, theta0=theta0, options={"maxiter": MAIN_MAXITER, "gtol": 1e-3})
    runtime = time.time() - t0

    new_grad = result["grad_norm_inf"]
    new_loglik = result["loglik"]
    new_se_quality_reported = result["se_quality"]  # codebase's own (gradient-blind) label
    # This session's proposed FIXED label (see gradient_and_convergence_diagnosis.md §4.1):
    new_se_quality_fixed = (
        "unreliable" if not np.isfinite(new_grad) or new_grad >= RELIABLE_GRAD_THRESHOLD
        else new_se_quality_reported
    )

    out = {
        "run_id": run_id, "stage": stage, "model_id": model_id,
        "old_grad_norm_inf": old_grad, "new_grad_norm_inf": new_grad,
        "grad_improved": bool(np.isfinite(old_grad) and np.isfinite(new_grad) and new_grad < old_grad),
        "old_loglik": old_loglik, "new_loglik": new_loglik,
        "loglik_delta": (new_loglik - old_loglik) if np.isfinite(old_loglik) and np.isfinite(new_loglik) else np.nan,
        "old_se_quality": old_se_quality,
        "new_se_quality_reported": new_se_quality_reported,
        "new_se_quality_fixed": new_se_quality_fixed,
        "now_reliable": bool(np.isfinite(new_grad) and new_grad < RELIABLE_GRAD_THRESHOLD),
        "now_strictly_converged": bool(np.isfinite(new_grad) and new_grad < STRICT_CONVERGED_THRESHOLD),
        "validity": result["validity"], "success": result["success"],
        "n_iter": result["n_iter"], "runtime_s": runtime,
    }

    out_dir = OUT_DIR / run_id / stage / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", default=None)
    args = parser.parse_args()
    locations = [args.location] if args.location else LOCATIONS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for location in locations:
        cfg = STATION_REGISTRY[location]
        run_id, display = cfg["run_id"], cfg.get("display", location)
        run_dir = ARTIFACTS_DIR / run_id
        print(f"=== {display} ===")

        loader = StationDataLoader.from_registry(location, precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH)
        data = loader.load_all()

        for stage in STAGE_DIRS:
            sd = run_dir / stage
            if not sd.exists():
                continue
            for model_dir in sorted(sd.iterdir()):
                if not model_dir.is_dir() or not (model_dir / "metadata.json").exists():
                    continue
                model_id = model_dir.name
                try:
                    t0 = time.time()
                    out = polish_one(run_id, stage, model_id, data)
                    print(f"  [{stage}/{model_id}] grad {out['old_grad_norm_inf']:.4g} -> "
                          f"{out['new_grad_norm_inf']:.4g}  reliable={out['now_reliable']}  "
                          f"loglik_delta={out['loglik_delta']:.3g}  ({time.time()-t0:.1f}s)")
                    summary_rows.append({"status": "ok", **out})
                except Exception as exc:
                    print(f"  [{stage}/{model_id}] FAILED: {exc}")
                    traceback.print_exc()
                    summary_rows.append({"status": "failed", "run_id": run_id, "stage": stage,
                                          "model_id": model_id, "error": str(exc)})

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary.csv", index=False)
    print("\nDone. Summary written to", OUT_DIR / "summary.csv")


if __name__ == "__main__":
    main()
