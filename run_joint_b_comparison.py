"""
V2 vs. joint-B comparison analysis, one representative Stage-2 model per
location (2026-09-02).

=============================================================================
OVERVIEW
=============================================================================

Companion analysis to run_joint_b_experiment.py. Reuses the EXACT same
representative model per location already chosen and reported in
reports/original vs v2/ (artifacts/original_vs_v2_comparison/<short>/
representative_model.json), so this connects directly to the existing
chain of reports instead of re-opening model selection with a different
answer.

For that one model, under BOTH:
  * "v2":      current production V2 (artifacts/bound_ab_experiment_v2/)
  * "joint_b": state wall restored + joint-softmax B
               (artifacts/joint_b_experiment/)
computes in-sample residuals, full-test-set 1-step OOS RMSE/CRPS, and
Monte Carlo multi-step OOS RMSE/CRPS -- exactly mirroring
run_original_vs_v2_comparison.py / run_v2_seasonality_comparison.py.

Per docs/EXECUTION.md, no optimisation happens here -- both settings'
models were already fit. This script only evaluates.

=============================================================================
USAGE
=============================================================================

    python run_joint_b_comparison.py
    python run_joint_b_comparison.py --location "MANAUS"

Output (per location):
    artifacts/joint_b_comparison/<short>/representative_model.json
    artifacts/joint_b_comparison/<short>/<v2|joint_b>/
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
V2_DIR      = ARTIFACTS_DIR / "bound_ab_experiment_v2"
JOINTB_DIR  = ARTIFACTS_DIR / "joint_b_experiment"
REP_DIR     = ARTIFACTS_DIR / "original_vs_v2_comparison"
OUT_DIR     = ARTIFACTS_DIR / "joint_b_comparison"

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]
SETTINGS = {"v2": V2_DIR, "joint_b": JOINTB_DIR}
N_ORIGINS_STRIDE = 15
N_DRAWS = 400
MC_SEED = 2026


def _block_name(model_id: str) -> str:
    return model_id.replace("stage2_phi_", "").replace("stage2_", "")


def load_representative(short: str) -> dict:
    p = REP_DIR / short / "representative_model.json"
    if not p.exists():
        return None
    info = json.loads(p.read_text())
    return {"stage": info["stage"], "model_id": info["model_id"], "run_id": info["run_id"]}


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
            rep = load_representative(short)
            if rep is None:
                print("  No representative_model.json found; skipping.")
                continue
            stage, model_id = rep["stage"], rep["model_id"]
            print(f"  Representative model: {stage}/{model_id} (reused from original-vs-v2 report)")

            loader = StationDataLoader.from_registry(location, precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH)
            data = loader.load_all()
            block = _block_name(model_id)
            X_train = data["covariate_blocks_train"][block]
            X_test  = data["covariate_blocks_test"][block]

            info = {"location": display, "run_id": run_id, "stage": stage, "model_id": model_id}
            results = {}
            for setting, root in SETTINGS.items():
                model_dir = root / run_id / stage / model_id
                if not (model_dir / "metadata.json").exists():
                    print(f"    {setting}: MISSING artifacts at {model_dir}; skipping.")
                    continue
                t0 = time.time()
                res = analyze_setting(setting, model_dir, data, X_train, X_test, loc_out / setting)
                print(f"    {setting}: grad={res['grad_norm_inf']:.4g} rmse={res['rmse_1step']:.3f} "
                      f"crps={res['crps_1step']:.3f}  ({time.time()-t0:.1f}s)")
                results[setting] = res

            if len(results) < 2:
                print("  Fewer than 2 settings available; skipping summary row.")
                continue

            info.update(results)
            (loc_out / "representative_model.json").write_text(json.dumps(info, indent=2, default=float))
            row = {"status": "ok", "location": display, "stage": stage, "model_id": model_id}
            for setting, res in results.items():
                row.update({f"{setting}_{k}": v for k, v in res.items()})
            summary_rows.append(row)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            summary_rows.append({"status": "failed", "location": display, "error": str(exc)})

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary.csv", index=False)
    print("\nDone. Summary written to", OUT_DIR / "summary.csv")


if __name__ == "__main__":
    main()
