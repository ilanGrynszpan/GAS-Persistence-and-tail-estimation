"""
Original (production, unconstrained) parameterisation at both lag sets --
every Stage-2 model refit at BOTH short {1,2,3} and seasonal
{1,2,3,364-367} GAS lags, regardless of which lag set it originally used
(2026-09-03).

=============================================================================
OVERVIEW
=============================================================================

Direct follow-up to the master comparison across every reparameterisation
tried in this project (original/v2/joint_b/link-only): applying a
per-parameter gradient gate (|grad_i| < 0.1 -> report the delta-method SE;
>= 0.1 and the block was penalty-contaminated -> "boundary optimized";
>= 0.1 otherwise -> flagged for manual explanation) to each candidate's
representative model per location found that the plain ORIGINAL
parameterisation -- no bound_ab, no bound_rho, none of this project's
experimental flags, i.e. EXACTLY pipeline/runner.py's build_stage2_specs()
default CovZAGASModel(...) call -- retrieves standard errors for 134/149
(89.9%) parameters, decisively more than every other setting tried (V2:
75.8%, joint_b: 75.2%, link-only: 73.2%/54.3%), while its OOS RMSE/CRPS
are within ~2%/~0.7% of the best candidate (joint_b) -- not a case of
sacrificing forecast accuracy for reportable inference.

The ORIGINAL parameterisation, like every other setting in this project,
only exists at whichever ONE lag set each representative model's own
production fit already used -- this script fills that gap exactly as
run_v2_lagset_experiment.py did for V2, refitting every Stage-2/Stage-2-phi
model at BOTH lag sets under the plain original parameterisation, reusing
the already-computed production artifact (byte-for-byte, deterministic)
whenever a model's own lag set already matches one side.

Per docs/EXECUTION.md, this does NOT touch any production artifact --
copies, not overwrites.

=============================================================================
USAGE
=============================================================================

    python run_original_lagset_experiment.py
    python run_original_lagset_experiment.py --location "MANAUS"

Output:
    artifacts/original_lagset_experiment/<run_id>/<stage>/<model_id>/<short|seasonal>/
    artifacts/original_lagset_experiment/summary_<tag>.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
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
from distributions.gb2_log_link import GB2LogLink
from distributions.gb2_phi_only import GB2LogLinkPhiOnly
from pi_dynamics.factory import make_pi_dynamics
from models.cov_gas_model import CovZAGASModel
from pipeline.artifact_utils import infer_gas_lags
from run_link_only_experiment import LAG_SETS, PRECIP_DIR, ERA5_DIR, ENSO_PATH

ARTIFACTS_DIR = ROOT / "artifacts"
OUT_DIR = ARTIFACTS_DIR / "original_lagset_experiment"

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]
STAGE_DIRS = ("stage2", "stage2_phi")
RELIABLE_GRAD_THRESHOLD = 1.0
MAIN_MAXITER = 2000
COPY_FILES = ["estimated_parameters.csv", "standard_errors.csv", "gradients.csv",
              "hess_inv.npy", "metadata.json"]


def _block_name(model_id: str) -> str:
    return model_id.replace("stage2_phi_", "").replace("stage2_", "")


def _lagset_name_of(lags: list) -> str:
    return "short" if list(lags) == LAG_SETS["short"] else "seasonal"


def refit_one(run_id: str, stage: str, model_id: str, lagset_name: str, data: dict) -> dict:
    orig_dir = ARTIFACTS_DIR / run_id / stage / model_id
    orig_meta = json.loads((orig_dir / "metadata.json").read_text())
    orig_params_df = pd.read_csv(orig_dir / "estimated_parameters.csv")
    tv_names = orig_meta.get("tv_names", ["phi"])
    orig_lags = orig_meta.get("lags") or infer_gas_lags(orig_params_df, tv_names[0])
    out_dir = OUT_DIR / run_id / stage / model_id / lagset_name

    # Reuse the already-computed PRODUCTION artifact byte-for-byte when
    # this lag set is the one the model already uses in production --
    # that fit is already the exact same deterministic computation.
    if _lagset_name_of(orig_lags) == lagset_name:
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname in COPY_FILES:
            src = orig_dir / fname
            if src.exists():
                shutil.copy2(src, out_dir / fname)
        return {"run_id": run_id, "stage": stage, "model_id": model_id, "lagset": lagset_name,
                "reused": True, "grad_norm_inf": orig_meta.get("grad_norm_inf"),
                "loglik": orig_meta.get("loglik"),
                "now_reliable": bool(np.isfinite(orig_meta.get("grad_norm_inf", np.nan))
                                      and orig_meta["grad_norm_inf"] < RELIABLE_GRAD_THRESHOLD)}

    cov_names = orig_meta.get("cov_names", [])
    dist = GB2LogLink() if len(tv_names) > 1 else GB2LogLinkPhiOnly()
    pi_dyn = make_pi_dynamics("ar_logistic", seasonal="daily")

    block = _block_name(model_id)
    X_train = data["covariate_blocks_train"][block]
    y_train = data["y_train"]
    lags = LAG_SETS[lagset_name]

    # EXACTLY pipeline/runner.py's build_stage2_specs() default call --
    # no bound_ab, no bound_rho, none of this project's experimental
    # flags -- the plain "original" production parameterisation.
    model = CovZAGASModel(
        distribution=dist, pi_dynamics=pi_dyn, seasonal="daily",
        gas_lags=lags, scaling=orig_meta.get("scaling", "diagonal_inverse_fisher"),
        cov_names=cov_names,
    )
    theta0 = model.initial_theta(y_train)

    t0 = time.time()
    result = model.fit(y_train, X_train, theta0=theta0, options={"maxiter": MAIN_MAXITER, "gtol": 1e-3})
    runtime = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_result(result, out_dir)

    return {"run_id": run_id, "stage": stage, "model_id": model_id, "lagset": lagset_name,
            "reused": False, "grad_norm_inf": result["grad_norm_inf"], "loglik": result["loglik"],
            "now_reliable": bool(np.isfinite(result["grad_norm_inf"]) and result["grad_norm_inf"] < RELIABLE_GRAD_THRESHOLD),
            "n_iter": result["n_iter"], "runtime_s": runtime}


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
                for lagset_name in LAG_SETS:
                    try:
                        t0 = time.time()
                        out = refit_one(run_id, stage, model_id, lagset_name, data)
                        tag = "reused" if out["reused"] else f"{time.time()-t0:.1f}s"
                        print(f"  [{stage}/{model_id}/{lagset_name}] grad={out['grad_norm_inf']:.4g} "
                              f"reliable={out['now_reliable']}  ({tag})")
                        summary_rows.append({"status": "ok", **out})
                    except Exception as exc:
                        print(f"  [{stage}/{model_id}/{lagset_name}] FAILED: {exc}")
                        traceback.print_exc()
                        summary_rows.append({"status": "failed", "run_id": run_id, "stage": stage,
                                              "model_id": model_id, "lagset": lagset_name, "error": str(exc)})

    tag = locations[0].split()[0] if len(locations) == 1 else "all"
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / f"summary_{tag}.csv", index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
