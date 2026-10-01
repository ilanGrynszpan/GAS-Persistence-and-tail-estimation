"""
Unconstrained-state experiment: refit every V2 Stage-2 model with the
state-magnitude soft penalty removed entirely (2026-09-02).

=============================================================================
OVERVIEW
=============================================================================

reports/v2 seasonality/report.pdf's gradient-reliability chapter, cross-
checked against report_se_utils.check_penalty_contamination across all 144
V2 fits computed so far, found that:
  - 0/144 models were contaminated by the A/B persistence-sum boxes,
  - 0/144 by rho,
  - 0/144 by the zeta>gamma kurtosis-existence condition,
  - 67/144 (46.5%) were contaminated SOLELY by the state-magnitude safety
    net (models/cov_gas_model.py / models/za_gas_model.py's soft penalty
    on |f_{j,t}| > STATE_LIMIT=15), and in every single one of those cases
    the reported max_state_magnitude sat at 15.00000-15.0002 -- i.e. the
    penalty was pinning an equilibrium right at the wall, not merely
    reporting a value that happened to be large.

This experiment refits the SAME 48 Stage-2/Stage-2-phi models, at the SAME
lag set each already uses (matching artifacts/bound_ab_experiment_v2/
exactly otherwise), with models/cov_gas_model.py's new unconstrained_state
flag set True -- i.e. LAM_STATE=0, so the soft state penalty contributes
NOTHING to the objective, while the INDEPENDENT isfinite() crash-guard in
models/_nb_gb2_filter.py (never touched by this flag) remains the only
thing preventing a genuinely invalid (NaN/inf) evaluation. A one-off smoke
test on the single worst case (Salvador stage2_phi_dewpoint_seasonal,
which sat at exactly 15.00005 under the penalty) found the state settles
at 26.0 once the penalty is removed -- large, but finite and stable, not a
runaway -- with the gradient still comfortably reliable (0.089). This
script checks whether that finding holds across all 48 models.

Per docs/EXECUTION.md, this does NOT touch any production artifact or the
existing bound_ab_experiment_v2/ artifacts.

=============================================================================
USAGE
=============================================================================

    python run_unconstrained_state_experiment.py
    python run_unconstrained_state_experiment.py --location "MANAUS"

Output:
    artifacts/unconstrained_state_experiment/<run_id>/<stage>/<model_id>/
        estimated_parameters.csv, standard_errors.csv, gradients.csv,
        hess_inv.npy, metadata.json, result.json
    artifacts/unconstrained_state_experiment/summary_<tag>.csv
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
OUT_DIR   = ARTIFACTS_DIR / "unconstrained_state_experiment"

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]
STAGE_DIRS = ("stage2", "stage2_phi")
RELIABLE_GRAD_THRESHOLD = 1.0
A_MAX, RHO_MAX = 2.0, 0.999


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

    n_lags = len(lags) if lags else 7
    b_max = 0.98 / n_lags  # matches V2's own rule exactly

    model = CovZAGASModel(
        distribution=dist, pi_dynamics=pi_dyn, seasonal="daily",
        gas_lags=lags, scaling=orig_meta.get("scaling", "diagonal_inverse_fisher"),
        cov_names=cov_names,
        bound_ab=True, a_max=A_MAX, b_max=b_max,
        bound_rho=True, rho_max=RHO_MAX,
        unconstrained_state=True,
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
                          f"{out['new_grad_norm_inf']:.4g}  state {out['v2_max_state']:.3g} -> "
                          f"{out['new_max_state']:.3g}  contam_blocks={out['n_contaminated_blocks']}  "
                          f"({time.time()-t0:.1f}s)")
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
        print(f"\nDone. {len(ok)} models refit. "
              f"{int(ok['any_contamination'].sum())}/{len(ok)} still show ANY contamination "
              f"(was 67/144 across the old V2 set, ~46.5%).")


if __name__ == "__main__":
    main()
