"""
Joint-B experiment: state wall restored + persistence bounded jointly via a
softmax-with-phantom-zero link on the SUM, not an even per-lag split
(2026-09-02).

=============================================================================
OVERVIEW
=============================================================================

Combines two changes on top of the current V2 baseline (artifacts/
bound_ab_experiment_v2/), per direct request after the unconstrained_state
experiment showed a real trade-off (contamination fully eliminated, but a
material fraction of models' gradients got materially worse -- see
_smoke_unconstrained_state.py's finding and the diagnosis that followed:
removing the state wall's strong convexity can leave BFGS on a flatter,
harder-conditioned surface for models where one influential observation
pushes a state far out):

  1. unconstrained_state=False (RESTORED): the state-magnitude safety net
     (|f_{j,t}|<STATE_LIMIT=15) stays exactly as in production V2. This
     alone still leaves the ~46.5% "(boundary optimum)" rate found in
     reports/v2 seasonality/report.pdf's gradient-reliability chapter.
  2. bound_b_joint=True (NEW): B_{j,l} is reparameterised through
     models/gas_filter.py's new joint softmax-with-phantom-zero link,
     bounding sum_l B_{j,l} < b_total_max=0.98 DIRECTLY -- the TRUE
     stationarity requirement -- instead of V2's independent per-lag box
     b_max=0.98/n_lags, which was strictly TIGHTER than necessary (no
     single lag could ever exceed its even share, even when the data
     wanted one lag to carry nearly the whole budget). A unit test
     (test_bound_b_joint.py, not kept) confirmed the encode/decode
     round-trip and that a single lag CAN now reach ~0.95 of the total
     budget while the sum still never reaches 0.98. A/rho are UNCHANGED
     (independent per-lag sigmoid for A, no stationarity motivation for a
     joint A bound; ordinary sigmoid for rho).

Hypothesis being tested: giving B more flexibility (no artificial even
split) may let several models reach a genuinely better-fitting, better-
conditioned optimum without needing the state to push toward its wall in
the first place -- i.e. address the (boundary optimum) rate from the OTHER
direction (fewer models need extreme states at all) rather than by
weakening the safety net that was found to cost real optimisation quality
when removed outright.

Per docs/EXECUTION.md, this does NOT touch any production artifact or the
existing bound_ab_experiment_v2/ artifacts.

=============================================================================
USAGE
=============================================================================

    python run_joint_b_experiment.py
    python run_joint_b_experiment.py --location "MANAUS"

Output:
    artifacts/joint_b_experiment/<run_id>/<stage>/<model_id>/
        estimated_parameters.csv, standard_errors.csv, gradients.csv,
        hess_inv.npy, metadata.json, result.json
    artifacts/joint_b_experiment/summary_<tag>.csv
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
from pipeline.artifact_utils import infer_gas_lags
from report_se_utils import check_penalty_contamination
from run_link_only_experiment import PRECIP_DIR, ERA5_DIR, ENSO_PATH

ARTIFACTS_DIR = ROOT / "artifacts"
V2_DIR    = ARTIFACTS_DIR / "bound_ab_experiment_v2"
OUT_DIR   = ARTIFACTS_DIR / "joint_b_experiment"

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]
STAGE_DIRS = ("stage2", "stage2_phi")
RELIABLE_GRAD_THRESHOLD = 1.0
A_MAX, RHO_MAX, B_TOTAL_MAX = 2.0, 0.999, 0.98


def _block_name(model_id: str) -> str:
    return model_id.replace("stage2_phi_", "").replace("stage2_", "")


def refit_one(run_id: str, stage: str, model_id: str, data: dict) -> dict:
    orig_dir = V2_DIR / run_id / stage / model_id
    orig_meta = json.loads((orig_dir / "metadata.json").read_text())
    orig_params_df = pd.read_csv(orig_dir / "estimated_parameters.csv")
    tv_names = orig_meta.get("tv_names", ["phi"])
    lags = orig_meta.get("lags") or infer_gas_lags(orig_params_df, tv_names[0])
    cov_names = orig_meta.get("cov_names", [])
    dist = GB2LogLink() if len(tv_names) > 1 else GB2LogLinkPhiOnly()
    pi_dyn = make_pi_dynamics("ar_logistic", seasonal="daily")

    block = _block_name(model_id)
    X_train = data["covariate_blocks_train"][block]
    y_train = data["y_train"]

    model = CovZAGASModel(
        distribution=dist, pi_dynamics=pi_dyn, seasonal="daily",
        gas_lags=lags, scaling=orig_meta.get("scaling", "diagonal_inverse_fisher"),
        cov_names=cov_names,
        bound_ab=True, a_max=A_MAX,
        bound_b_joint=True, b_total_max=B_TOTAL_MAX,
        bound_rho=True, rho_max=RHO_MAX,
        unconstrained_state=False,  # RESTORED -- see module docstring
    )
    theta0 = model.initial_theta(y_train)

    t0 = time.time()
    result = model.fit(y_train, X_train, theta0=theta0, options={"maxiter": 2000, "gtol": 1e-3})
    runtime = time.time() - t0

    out_dir = OUT_DIR / run_id / stage / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_result(result, out_dir)

    new_meta = json.loads((out_dir / "metadata.json").read_text())
    contam = check_penalty_contamination(new_meta)
    any_contam = any(contam.values())
    n_contam_blocks = sum(1 for v in contam.values() if v)

    out = {
        "run_id": run_id, "stage": stage, "model_id": model_id,
        "v2_grad_norm_inf": orig_meta.get("grad_norm_inf", np.nan),
        "new_grad_norm_inf": result["grad_norm_inf"],
        "v2_loglik": orig_meta.get("loglik", np.nan),
        "new_loglik": result["loglik"],
        "v2_max_state": (orig_meta.get("bound_diagnostics", {}) or {}).get("max_state_magnitude", np.nan),
        "new_max_state": (result.get("bound_diagnostics", {}) or {}).get("max_state_magnitude", np.nan),
        "now_reliable": bool(np.isfinite(result["grad_norm_inf"]) and result["grad_norm_inf"] < RELIABLE_GRAD_THRESHOLD),
        "was_reliable": bool(np.isfinite(orig_meta.get("grad_norm_inf", np.nan)) and orig_meta.get("grad_norm_inf", np.nan) < RELIABLE_GRAD_THRESHOLD),
        "any_contamination": any_contam,
        "n_contaminated_blocks": n_contam_blocks,
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
        run_dir = V2_DIR / run_id
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
                    out = refit_one(run_id, stage, model_id, data)
                    print(f"  [{stage}/{model_id}] grad {out['v2_grad_norm_inf']:.4g} -> "
                          f"{out['new_grad_norm_inf']:.4g}  reliable {out['was_reliable']} -> {out['now_reliable']}  "
                          f"contam_blocks={out['n_contaminated_blocks']}  ({time.time()-t0:.1f}s)")
                    summary_rows.append({"status": "ok", **out})
                except Exception as exc:
                    print(f"  [{stage}/{model_id}] FAILED: {exc}")
                    traceback.print_exc()
                    summary_rows.append({"status": "failed", "run_id": run_id, "stage": stage,
                                          "model_id": model_id, "error": str(exc)})

    tag = locations[0].split()[0] if len(locations) == 1 else "all"
    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT_DIR / f"summary_{tag}.csv", index=False)
    ok = df[df["status"] == "ok"]
    if len(ok):
        n_ok = int(ok["now_reliable"].sum())
        n_contam = int(ok["any_contamination"].sum())
        print(f"\nDone. {len(ok)} models refit. {n_ok}/{len(ok)} reliable "
              f"(grad<1). {n_contam}/{len(ok)} show ANY contamination "
              f"(V2 baseline: 67/144 across the whole prior collection, ~46.5%).")


if __name__ == "__main__":
    main()
