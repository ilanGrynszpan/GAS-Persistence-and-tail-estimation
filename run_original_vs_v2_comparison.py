"""
Original vs. v2 (sigmoid-bounded A/B/rho) comparison analysis, one
representative Stage-2 model per location (2026-09-01).

=============================================================================
OVERVIEW
=============================================================================

Companion analysis to generate_original_vs_v2_report.py. For each location,
selects ONE representative Stage-2 model -- the one of the 8 variants
(2 covariate blocks x 2 lag sets x {phi-only, phi+xi}) with the lowest
gradient under the ORIGINAL (production) parameterisation among non-failed
fits -- so the comparison is apples-to-apples (same model_id, same
data, same covariates) under both settings, and the choice is not biased
toward whichever setting looks best.

For that one model, under BOTH "original" (production artifacts,
untouched) and "v2" (artifacts/bound_ab_experiment_v2/, already fit in the
previous turn), computes:
  1. In-sample quantile residuals.
  2. Full-test-set 1-step OOS RMSE/CRPS (exact, not simulated).
  3. Monte Carlo multi-step OOS RMSE/CRPS (h=1,2,5,10,30,60,90).

Per docs/EXECUTION.md, no optimisation happens here -- both settings' models
were already fit (production pipeline; run_bound_ab_experiment.py
--b-max-per-lag in the previous turn). This script only evaluates.

=============================================================================
USAGE
=============================================================================

    python run_original_vs_v2_comparison.py
    python run_original_vs_v2_comparison.py --location "MANAUS"

Output (per location):
    artifacts/original_vs_v2_comparison/<short>/representative_model.json
    artifacts/original_vs_v2_comparison/<short>/<setting>/
        residuals.npy, onestep_metrics.json, multistep_metrics.csv
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
from pipeline.artifact_utils import load_model_and_theta
from diagnostics.residuals import quantile_residuals
from diagnostics.scoring import compute_extended_oos_metrics
from diagnostics.multistep_forecast import multistep_metrics, DEFAULT_HORIZONS

PRECIP_DIR = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR   = ROOT / "data" / "input" / "ERA5"
ENSO_PATH  = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
V2_DIR     = ARTIFACTS_DIR / "bound_ab_experiment_v2"
OUT_DIR    = ARTIFACTS_DIR / "original_vs_v2_comparison"

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]
STAGE_DIRS = ("stage2", "stage2_phi")
N_ORIGINS_STRIDE = 15
N_DRAWS = 400
MC_SEED = 2026


def _block_name(model_id: str) -> str:
    return model_id.replace("stage2_phi_", "").replace("stage2_", "")


def pick_representative(run_id: str) -> dict:
    """Lowest grad_norm_inf among non-failed ORIGINAL (production) Stage-2
    models -- selection is deliberately based on the ORIGINAL setting only,
    so it cannot be biased toward whichever setting looks better."""
    best = None
    for stage in STAGE_DIRS:
        sd = ARTIFACTS_DIR / run_id / stage
        if not sd.exists():
            continue
        for model_dir in sorted(sd.iterdir()):
            meta_path = model_dir / "metadata.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            grad = meta.get("grad_norm_inf", np.inf)
            if not np.isfinite(grad) or not np.isfinite(meta.get("rmse", np.nan)):
                continue
            if best is None or grad < best["grad_norm_inf"]:
                best = {"stage": stage, "model_id": model_dir.name, "grad_norm_inf": grad}
    return best


def analyze_setting(setting: str, model_dir: Path, data: dict, X_train, X_test, out_dir: Path) -> dict:
    model, theta, meta, _ = load_model_and_theta(model_dir)
    if model is None or theta is None:
        raise RuntimeError(f"Could not reconstruct model at {model_dir}")
    y_train, y_test = data["y_train"], data["y_test"]

    out_dir.mkdir(parents=True, exist_ok=True)

    cdfs = model.cdf_series(theta, y_train, X_train)
    eff = len(y_train) - len(cdfs)
    rng = np.random.default_rng(0)
    qr = quantile_residuals(cdfs, y_train[eff:], randomise_zeros=True, rng=rng)
    np.save(out_dir / "residuals.npy", qr)

    oos = model.simulate_oos(theta, y_train, y_test, X_train, X_test)
    onestep = compute_extended_oos_metrics(model, oos, y_test, n_draws=1000, seed=42)
    (out_dir / "onestep_metrics.json").write_text(json.dumps(onestep, indent=2, default=float))

    max_origin = len(y_test) - max(DEFAULT_HORIZONS) - 1
    origins = list(range(0, max_origin, max(1, max_origin // (len(y_test) // N_ORIGINS_STRIDE + 1))))
    ms = multistep_metrics(model, theta, y_train, y_test, {"X_train": X_train, "X_test": X_test},
                            origins=origins, horizons=DEFAULT_HORIZONS, n_draws=N_DRAWS, seed=MC_SEED)
    ms.to_csv(out_dir / "multistep_metrics.csv", index=False)

    return {"setting": setting, "grad_norm_inf": meta.get("grad_norm_inf"),
            "rmse_1step": onestep["rmse"], "crps_1step": onestep["crps_mean"],
            "n_origins": len(origins), "n_draws": N_DRAWS}


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
        short = cfg.get("short", run_id)
        print(f"=== {display} ===")
        loc_out = OUT_DIR / short

        try:
            rep = pick_representative(run_id)
            if rep is None:
                print("  No usable Stage-2 model found; skipping.")
                continue
            stage, model_id = rep["stage"], rep["model_id"]
            print(f"  Representative model: {stage}/{model_id} (original grad_inf={rep['grad_norm_inf']:.4g})")

            loader = StationDataLoader.from_registry(location, precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH)
            data = loader.load_all()
            block = _block_name(model_id)
            X_train = data["covariate_blocks_train"][block]
            X_test  = data["covariate_blocks_test"][block]

            info = {"location": display, "run_id": run_id, "stage": stage, "model_id": model_id}

            orig_dir = ARTIFACTS_DIR / run_id / stage / model_id
            t0 = time.time()
            orig_result = analyze_setting("original", orig_dir, data, X_train, X_test, loc_out / "original")
            print(f"    original: grad={orig_result['grad_norm_inf']:.4g} rmse={orig_result['rmse_1step']:.3f} "
                  f"crps={orig_result['crps_1step']:.3f}  ({time.time()-t0:.1f}s)")

            v2_dir = V2_DIR / run_id / stage / model_id
            t0 = time.time()
            v2_result = analyze_setting("v2", v2_dir, data, X_train, X_test, loc_out / "v2")
            print(f"    v2:       grad={v2_result['grad_norm_inf']:.4g} rmse={v2_result['rmse_1step']:.3f} "
                  f"crps={v2_result['crps_1step']:.3f}  ({time.time()-t0:.1f}s)")

            info["original"], info["v2"] = orig_result, v2_result
            (loc_out / "representative_model.json").write_text(json.dumps(info, indent=2, default=float))
            summary_rows.append({"status": "ok", **{f"orig_{k}": v for k, v in orig_result.items()},
                                  **{f"v2_{k}": v for k, v in v2_result.items()},
                                  "location": display, "stage": stage, "model_id": model_id})
        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            summary_rows.append({"status": "failed", "location": display, "error": str(exc)})

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary.csv", index=False)
    print("\nDone. Summary written to", OUT_DIR / "summary.csv")


if __name__ == "__main__":
    main()
