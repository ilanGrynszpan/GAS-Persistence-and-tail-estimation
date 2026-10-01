"""
Original (production) parameterisation at short vs. seasonal GAS lags,
one representative Stage-2 model per location (2026-09-03).

=============================================================================
OVERVIEW
=============================================================================

Companion analysis to generate_original_seasonality_report.py. Reuses the
EXACT same representative model per location already chosen and reported
in reports/original vs v2/ (lowest grad_norm_inf under the ORIGINAL,
unbounded parameterisation among non-failed fits), refit at BOTH lag sets
under the plain ORIGINAL parameterisation by run_original_lagset_experiment.py
(no bound_ab, no bound_rho -- pipeline/runner.py's exact default
CovZAGASModel(...) call). Selected as the winning setting across a master
comparison against V2 / joint-B / link-only, per the per-parameter
gradient-gate SE rule (|grad_i|<0.1 -> report SE; else "boundary
optimized" if a penalty was active, else flagged) -- 89.9% of parameters
retrievable vs ~73-76% for every other setting tried, with OOS RMSE/CRPS
within ~2%/~0.7% of the best candidate.

For that one model, under BOTH lag sets, computes in-sample residuals,
full-test-set 1-step OOS RMSE/CRPS, and Monte Carlo multi-step OOS
RMSE/CRPS -- exactly mirroring run_v2_seasonality_comparison.py.

Per docs/EXECUTION.md, no optimisation happens here -- both lag sets'
models were already fit. This script only evaluates.

=============================================================================
USAGE
=============================================================================

    python run_original_seasonality_comparison.py
    python run_original_seasonality_comparison.py --location "MANAUS"

Output (per location):
    artifacts/original_seasonality_comparison/<short>/representative_model.json
    artifacts/original_seasonality_comparison/<short>/<short|seasonal>/
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
REP_DIR    = ARTIFACTS_DIR / "original_vs_v2_comparison"
LAGSET_DIR = ARTIFACTS_DIR / "original_lagset_experiment"
OUT_DIR    = ARTIFACTS_DIR / "original_seasonality_comparison"

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]
LAGSETS = ("short", "seasonal")
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
    T = len(y_train)

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

    grad_norm_inf = meta.get("grad_norm_inf", np.nan)
    return {"setting": setting, "grad_norm_inf": grad_norm_inf, "T": T,
            "grad_norm_inf_per_obs": grad_norm_inf / T if np.isfinite(grad_norm_inf) else np.nan,
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
            for lagset in LAGSETS:
                model_dir = LAGSET_DIR / run_id / stage / model_id / lagset
                if not (model_dir / "metadata.json").exists():
                    print(f"    {lagset}: MISSING artifacts at {model_dir}; skipping.")
                    continue
                t0 = time.time()
                res = analyze_setting(lagset, model_dir, data, X_train, X_test, loc_out / lagset)
                print(f"    {lagset}: grad={res['grad_norm_inf']:.4g} rmse={res['rmse_1step']:.3f} "
                      f"crps={res['crps_1step']:.3f}  ({time.time()-t0:.1f}s)")
                results[lagset] = res

            if len(results) < 2:
                print("  Fewer than 2 lagsets available; skipping summary row.")
                continue

            info.update(results)
            (loc_out / "representative_model.json").write_text(json.dumps(info, indent=2, default=float))
            row = {"status": "ok", "location": display, "stage": stage, "model_id": model_id}
            for lagset, res in results.items():
                row.update({f"{lagset}_{k}": v for k, v in res.items()})
            summary_rows.append(row)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            summary_rows.append({"status": "failed", "location": display, "error": str(exc)})

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary.csv", index=False)
    print("\nDone. Summary written to", OUT_DIR / "summary.csv")


if __name__ == "__main__":
    main()
