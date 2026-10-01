"""
V2 vs. link-only (short and seasonal) comparison analysis, same
representative model per location as reports/original vs v2/ (2026-09-02).

For each location's representative Stage-2 model (same selection as
run_original_vs_v2_comparison.py -- lowest gradient under the ORIGINAL
setting, so the pick is unbiased), evaluates the two link-only refits
(run_link_only_experiment.py's "short" and "seasonal" lag-set variants)
the same way v2 was evaluated: in-sample quantile residuals, full-test-set
1-step OOS, and Monte Carlo multi-step OOS. v2's own numbers are reused
directly from artifacts/original_vs_v2_comparison/ (already computed, not
recomputed here).

Per docs/EXECUTION.md, no optimisation happens here -- all three settings'
models were already fit in prior steps; this script only evaluates.

Output (per location):
    artifacts/link_only_comparison/<short>/<lagset>/
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
from run_original_vs_v2_comparison import OUT_DIR as CMP_DIR, LOCATIONS
from run_link_only_experiment import OUT_DIR as LINK_DIR, LAG_SETS, PRECIP_DIR, ERA5_DIR, ENSO_PATH

OUT_DIR = ROOT / "artifacts" / "link_only_comparison"
N_ORIGINS_STRIDE = 15
N_DRAWS = 400
MC_SEED = 2026


def _block_name(model_id: str) -> str:
    return model_id.replace("stage2_phi_", "").replace("stage2_", "")


def analyze_setting(model_dir: Path, data: dict, X_train, X_test, out_dir: Path) -> dict:
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

    return {"grad_norm_inf": meta.get("grad_norm_inf"), "rmse_1step": onestep["rmse"],
            "crps_1step": onestep["crps_mean"], "n_origins": len(origins), "n_draws": N_DRAWS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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

        rep_path = CMP_DIR / short / "representative_model.json"
        if not rep_path.exists():
            print("  No representative_model.json found (run run_original_vs_v2_comparison.py first); skipping.")
            continue
        rep = json.loads(rep_path.read_text())
        stage, model_id = rep["stage"], rep["model_id"]
        print(f"  Representative model: {stage}/{model_id}")

        loader = StationDataLoader.from_registry(location, precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH)
        data = loader.load_all()
        block = _block_name(model_id)
        X_train = data["covariate_blocks_train"][block]
        X_test  = data["covariate_blocks_test"][block]

        for lagset_name in LAG_SETS:
            try:
                model_dir = LINK_DIR / run_id / stage / model_id / lagset_name
                out_dir = OUT_DIR / short / lagset_name
                t0 = time.time()
                result = analyze_setting(model_dir, data, X_train, X_test, out_dir)
                print(f"    {lagset_name}: grad={result['grad_norm_inf']:.4g} rmse={result['rmse_1step']:.3f} "
                      f"crps={result['crps_1step']:.3f}  ({time.time()-t0:.1f}s)")
                summary_rows.append({"status": "ok", "location": display, "lagset": lagset_name, **result})
            except Exception as exc:
                print(f"    {lagset_name} FAILED: {exc}")
                traceback.print_exc()
                summary_rows.append({"status": "failed", "location": display, "lagset": lagset_name, "error": str(exc)})

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary.csv", index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
