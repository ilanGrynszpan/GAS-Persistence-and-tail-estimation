"""
V2 at both lag sets: every Stage-2 model refit under the EXACT current V2
parameterisation (as already reported in reports/original vs v2/) at BOTH
the short {1,2,3} and seasonal {1,2,3,364,...,367} GAS lag sets, regardless
of which lag set the model originally used (2026-09-02).

=============================================================================
OVERVIEW
=============================================================================

V2 = models/gas_filter.py's bound_ab (a_max=2.0, b_max=0.98/n_lags -- the
joint persistence budget preserved, just redistributed as independent
per-lag boxes) + bound_rho (rho_max=0.999). ab_signed/rho_signed/
free_zeta_gamma/unconstrained_penalties are all OFF, matching exactly what
"v2" has meant throughout this project so far -- this script does not
change that definition, only asks how it behaves at the OTHER lag set from
whichever one each model happens to already use.

Where a model's own lag set already matches one side of the comparison
(48 of the 96 cells), the already-computed artifacts in
artifacts/bound_ab_experiment_v2/ are copied rather than refit -- that
computation is already done and deterministic, so this only performs the
48 NEW fits actually needed (the other lag set for each model).

Per docs/EXECUTION.md, this does NOT touch any production artifact or the
existing bound_ab_experiment_v2/ artifacts (copies, not overwrites).

Output:
    artifacts/v2_lagset_experiment/<run_id>/<stage>/<model_id>/<short|seasonal>/
    artifacts/v2_lagset_experiment/summary_<tag>.csv
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
V2_DIR = ARTIFACTS_DIR / "bound_ab_experiment_v2"
OUT_DIR = ARTIFACTS_DIR / "v2_lagset_experiment"

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]
STAGE_DIRS = ("stage2", "stage2_phi")
A_MAX, RHO_MAX = 2.0, 0.999
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
    orig_lags = orig_meta.get("lags") or infer_gas_lags(orig_params_df, orig_meta.get("tv_names", ["phi"])[0])
    v2_dir_for_model = V2_DIR / run_id / stage / model_id
    out_dir = OUT_DIR / run_id / stage / model_id / lagset_name

    # Reuse already-computed V2 artifacts when this lag set is the one the
    # model already used (V2's own fit is deterministic given the same
    # data/theta0, so this is a byte-for-byte reuse, not an approximation).
    if _lagset_name_of(orig_lags) == lagset_name:
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname in COPY_FILES:
            src = v2_dir_for_model / fname
            if src.exists():
                shutil.copy2(src, out_dir / fname)
        v2_meta = json.loads((v2_dir_for_model / "metadata.json").read_text())
        return {"run_id": run_id, "stage": stage, "model_id": model_id, "lagset": lagset_name,
                "reused": True, "grad_norm_inf": v2_meta.get("grad_norm_inf"),
                "loglik": v2_meta.get("loglik"),
                "now_reliable": bool(np.isfinite(v2_meta.get("grad_norm_inf", np.nan))
                                      and v2_meta["grad_norm_inf"] < RELIABLE_GRAD_THRESHOLD)}

    tv_names = orig_meta.get("tv_names", ["phi"])
    cov_names = orig_meta.get("cov_names", [])
    dist = GB2LogLink() if len(tv_names) > 1 else GB2LogLinkPhiOnly()
    pi_dyn = make_pi_dynamics("ar_logistic", seasonal="daily")

    block = _block_name(model_id)
    X_train = data["covariate_blocks_train"][block]
    y_train = data["y_train"]

    lags = LAG_SETS[lagset_name]
    b_max = 0.98 / len(lags)  # V2's own rule: joint budget preserved, redistributed per lag

    model = CovZAGASModel(
        distribution=dist, pi_dynamics=pi_dyn, seasonal="daily",
        gas_lags=lags, scaling=orig_meta.get("scaling", "diagonal_inverse_fisher"),
        cov_names=cov_names,
        bound_ab=True, a_max=A_MAX, b_max=b_max,
        bound_rho=True, rho_max=RHO_MAX,
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
