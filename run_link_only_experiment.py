"""
Link-only ("v3") experiment: every Stage-2 model refit with ALL soft
penalties replaced by link functions, at BOTH the short and seasonal GAS
lag sets (2026-09-02).

=============================================================================
OVERVIEW
=============================================================================

Per the user's explicit design: remove every constraint except the one
genuine mathematical requirement (GB2's 4th-moment/kurtosis-existence
condition, exp(zeta) > 4*exp(gamma)), and solve everything else through
link functions optimised completely freely:

    A_{j,l}   = 2 * (2*sigmoid(z) - 1)      in (-2, 2)   -- per lag, independent
    B_{j,l}   = 1 * (2*sigmoid(z) - 1)      in (-1, 1)   -- per lag, independent
    rho       = 1 * (2*sigmoid(z) - 1)      in (-1, 1)
    zeta      = gamma + log(4) + exp(delta)              -- exp(zeta)>4*exp(gamma) always
    (gamma, xi, omega, f0, Gamma_covariate: already fully free reals, untouched)

The sign-symmetric (-max, max) links (not the one-sided (0, max) used by
the earlier bound_ab/bound_rho work) match the ORIGINAL model's true domain
-- its penalty is on |A_l|/|B_l|/|rho|, so negative values are legitimate
and several already-fitted models use them.

Unlike run_bound_ab_experiment.py, this refits EVERY one of the 48 Stage-2
models at BOTH the short lag set {1,2,3} and the seasonal lag set
{1,2,3,364,365,366,367}, regardless of which the original model actually
used -- 96 fits total -- per the explicit request to compare both lag
choices under this new parameterisation.

Per docs/EXECUTION.md, this does NOT touch any production artifact or the
existing bound_ab_experiment_v2/ artifacts -- results go to a new location.

=============================================================================
USAGE
=============================================================================

    python run_link_only_experiment.py
    python run_link_only_experiment.py --location "MANAUS"

Output:
    artifacts/link_only_experiment/<run_id>/<stage>/<model_id>/<lagset>/
        (estimated_parameters.csv, standard_errors.csv, metadata.json, ... via save_result)
        result.json  (old vs. new grad/loglik summary)
    artifacts/link_only_experiment/summary_<tag>.csv
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
from distributions.gb2_log_link import GB2LogLink
from distributions.gb2_phi_only import GB2LogLinkPhiOnly
from pi_dynamics.factory import make_pi_dynamics
from models.cov_gas_model import CovZAGASModel

PRECIP_DIR = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR   = ROOT / "data" / "input" / "ERA5"
ENSO_PATH  = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
OUT_DIR    = ARTIFACTS_DIR / "link_only_experiment"

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]
STAGE_DIRS = ("stage2", "stage2_phi")
LAG_SETS = {"short": [1, 2, 3], "seasonal": [1, 2, 3, 364, 365, 366, 367]}

# The one experiment-wide setting: every soft constraint except the GB2
# kurtosis condition replaced by a link function, all fully unconstrained.
LINK_ONLY_KWARGS = dict(
    bound_ab=True, a_max=2.0, b_max=1.0, ab_signed=True,
    bound_rho=True, rho_max=1.0, rho_signed=True,
    free_zeta_gamma=True, kurtosis_margin=4.0,
    unconstrained_penalties=True,
)
RELIABLE_GRAD_THRESHOLD = 1.0
MAIN_MAXITER = 2000


def _block_name(model_id: str) -> str:
    return model_id.replace("stage2_phi_", "").replace("stage2_", "")


def refit_one(run_id: str, stage: str, model_id: str, lagset_name: str, data: dict) -> dict:
    orig_dir = ARTIFACTS_DIR / run_id / stage / model_id
    orig_meta = json.loads((orig_dir / "metadata.json").read_text())
    tv_names = orig_meta.get("tv_names", ["phi"])
    cov_names = orig_meta.get("cov_names", [])
    dist = GB2LogLink() if len(tv_names) > 1 else GB2LogLinkPhiOnly()
    pi_dyn = make_pi_dynamics("ar_logistic", seasonal="daily")

    block = _block_name(model_id)
    X_train = data["covariate_blocks_train"][block]
    y_train = data["y_train"]

    model = CovZAGASModel(
        distribution=dist, pi_dynamics=pi_dyn, seasonal="daily",
        gas_lags=LAG_SETS[lagset_name], scaling=orig_meta.get("scaling", "diagonal_inverse_fisher"),
        cov_names=cov_names, **LINK_ONLY_KWARGS,
    )
    theta0 = model.initial_theta(y_train)

    t0 = time.time()
    result = model.fit(y_train, X_train, theta0=theta0, options={"maxiter": MAIN_MAXITER, "gtol": 1e-3})
    runtime = time.time() - t0

    out_dir = OUT_DIR / run_id / stage / model_id / lagset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_result(result, out_dir)

    out = {
        "run_id": run_id, "stage": stage, "model_id": model_id, "lagset": lagset_name,
        "orig_grad_norm_inf": orig_meta.get("grad_norm_inf", np.nan),
        "orig_loglik": orig_meta.get("loglik", np.nan),
        "new_grad_norm_inf": result["grad_norm_inf"],
        "new_loglik": result["loglik"],
        "new_se_quality": result["se_quality"],
        "now_reliable": bool(np.isfinite(result["grad_norm_inf"]) and result["grad_norm_inf"] < RELIABLE_GRAD_THRESHOLD),
        "validity": result["validity"], "success": result["success"],
        "n_iter": result["n_iter"], "runtime_s": runtime,
    }
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
                for lagset_name in LAG_SETS:
                    try:
                        t0 = time.time()
                        out = refit_one(run_id, stage, model_id, lagset_name, data)
                        print(f"  [{stage}/{model_id}/{lagset_name}] grad {out['orig_grad_norm_inf']:.4g} -> "
                              f"{out['new_grad_norm_inf']:.4g}  reliable={out['now_reliable']}  "
                              f"loglik {out['orig_loglik']:.2f} -> {out['new_loglik']:.2f}  ({time.time()-t0:.1f}s)")
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
