"""
Unconstrained-persistence experiment, v2: A stays wide/unrestricted, but
B's SUM is now bounded to the true symmetric stationarity boundary
(-1, 1) via a smooth tanh-rescale link -- not left with no joint bound at
all. Crossed with two ways of handling the GB2's own static/initial-
condition parameters, at both short and seasonal GAS lags (2026-09-02).

=============================================================================
OVERVIEW
=============================================================================

v1 of this experiment (bound_ab with NO joint constraint on B at all,
a_max=b_max=20) was tried first and found to fail completely: 0/16
reliable fits across two locations, gradients up to 1.58 million, every
model genuinely non-stationary (sum|B_l|>1). Per direct follow-up
instruction, B's sum is now put back under a stationarity cap, but
through a SIGNED SIGMOID applied directly to the sum itself (not an
even per-lag split, and not a hard clip) -- models/gas_filter.py's new
bound_b_tanh_sum:

    B_{j,l} = raw_l * b_sum_max * tanh(S) / S,   S = sum_l raw_l

so that sum_l B_{j,l} = b_sum_max * tanh(S) is guaranteed to lie in
(-b_sum_max, b_sum_max) = (-1, 1) for ANY real raw_1..raw_L -- letting the
sum approach +-1 arbitrarily closely ("if it goes to one let it go") but
never cross it, while preserving whatever relative split across lags the
optimizer finds (no even-split assumption, unlike bound_b_joint's softmax,
and no restriction to non-negative B). Verified analytically smooth at
S=0 (a removable singularity, handled via its Taylor series) and its
delta-method Jacobian confirmed against finite differences before use.

v2 of THIS experiment (a_max still 20, B's SUM now correctly bounded) was
tried next and STILL failed almost completely: 2/16 reliable across the
same two locations, gradients still up to 174,000+. Root cause identified:
(a) individual B_{j,l} can still take huge, mixed-sign values that CANCEL
in the sum but not in magnitude (bound_b_tanh_sum only constrains
sum_l B_{j,l}, never |B_{j,l}| itself -- confirmed 16/16 still had
sum_l|B_{j,l}|>1, this project's standing non-stationarity check, even
though sum_l B_{j,l} was correctly kept under 1), and (b) A itself was
still completely unconstrained in practice (a_max=20 never meaningfully
binds). Per direct follow-up instruction ("tighten A"), this v3 reverts
A_MAX to 2.0 -- V2's own established value, never a source of instability
in any other experiment this project has run -- to isolate whether A's
width alone was driving the blow-ups, independent of the B fix. It did
NOT help either (still 2/8 reliable, gradients still ~10^5), ruling A out.

v4 (this version), per direct follow-up instruction ("cap B individually
instead"): DROPS bound_b_tanh_sum's joint-sum link entirely and instead
gives B the SAME individual per-lag signed-sigmoid box already used for A
(bound_ab/ab_signed's existing, unmodified mechanism -- no new codec
machinery needed):

    A_{j,l} = a_max * (2*sigmoid(z) - 1)   in (-a_max, a_max),  a_max=2.0
    B_{j,l} = b_max * (2*sigmoid(z) - 1)   in (-b_max, b_max),  b_max=2.0

This directly prevents the v2/v3 failure mode (individual B_{j,l} taking
huge, mixed-sign, sum-cancelling values) since no single lag can now
exceed +-2.0 in isolation -- at the cost of no longer guaranteeing ANY
joint bound on sum_l B_{j,l} (a 7-lag seasonal model's worst case is now
sum_l|B_l| up to 7*2.0=14, structurally similar to the failed v1 in that
one specific respect) -- reported transparently; whether it converges in
PRACTICE, not just in principle, is what this version tests.

rho (the OCCURRENCE-model persistence, a SEPARATE recursion eta_t = omega0
+ rho*eta_{t-1} + ...) is the ONE exception, kept at bound_rho=True,
rho_signed=True, rho_max=0.999 -- unlike B, |rho|>1 makes eta_t diverge
EXPONENTIALLY over the ~3650-day recursion (not merely "non-stationary in
principle", but numerically explosive within a few hundred days), so this
is a bare numerical-necessity guard, not a statistical restriction being
removed alongside A/B.

The state-magnitude safety net stays ACTIVE (unconstrained_state=False,
per the prior turn's "put just this wall" decision) -- this experiment
is about the PERSISTENCE parameterisation, not the numerical safety net.

Two settings for the GB2's own static parameters (gamma, zeta, xi) and
initial GAS states (f0_phi, f0_xi), fit and reported SEPARATELY:

  "gb2_constraint" (Report 1): free_zeta_gamma=True, kurtosis_margin=4.0
      -- the exact GB2 moment-existence relationship exp(zeta) >
      4*exp(gamma) is guaranteed BY CONSTRUCTION for every real (gamma,
      delta), same mechanism used throughout this project since the
      original link-only experiment. f0_phi/f0_xi remain fully free
      reals (no additional box) -- xi is already positive-via-exp by the
      GB2 log-link itself, needing no separate box.

  "static_box" (Report 2): bound_static_box=True, static_box=(-2, 15) --
      gamma, zeta, xi, f0_phi AND f0_xi are ALL boxed into the SAME
      (-2, 15) interval via the smooth affine-sigmoid
      param = -2 + 17*sigmoid(z) (see models/gas_filter.py's _ParamCodec
      docstring for why this replaces the discontinuous piecewise formula
      originally proposed). UNLIKE free_zeta_gamma, this does NOT
      guarantee exp(zeta) > 4*exp(gamma) by construction -- only the bare
      gamma<zeta hard-validity reject (E[Y]<inf) remains, kept as a
      numerical-necessity guard exactly like the rho case above, NOT
      dropped alongside the other restrictions. Whether the fitted values
      end up respecting the 4x kurtosis margin anyway (without it being
      enforced) is itself one of this report's diagnostics.
      (-2, 15) is not an arbitrary pick: it comfortably contains the
      empirical range of gamma/zeta/xi/f0_phi/f0_xi observed across every
      one of this project's 120 already-fit production models (gamma in
      [-0.05, 1.51], zeta in [3.70, 9.49], xi in [-0.39, 2.06], f0_phi in
      [-10.87, 15.0], f0_xi in [-2.62, 10.21]) with real headroom on both
      sides, and matches the (-2, 15) example given verbatim.

In BOTH settings: unconstrained_penalties=True (drops the |static|<20
auxiliary numerical guard AND the zeta>gamma soft "push away from the
boundary" penalty -- a genuine statistical restriction, unlike the bare
gamma>=zeta hard reject which stays active in both settings regardless).

Fit at BOTH GAS lag sets (short {1,2,3}, seasonal {1,2,3,364-367}) for all
48 Stage-2/Stage-2-phi models at every location -- 4 settings x 48 = 192
fresh fits (nothing here can be reused from any prior experiment; this is
an entirely new parameterisation).

Per docs/EXECUTION.md, this does NOT touch any production artifact.

=============================================================================
USAGE
=============================================================================

    python run_unconstrained_persistence_experiment.py --setting gb2_constraint --lagset short
    python run_unconstrained_persistence_experiment.py --setting static_box --lagset seasonal --location "MANAUS"

Output:
    artifacts/unconstrained_persistence_experiment/<setting>/<lagset>/
        <run_id>/<stage>/<model_id>/
            estimated_parameters.csv, standard_errors.csv, gradients.csv,
            hess_inv.npy, metadata.json, result.json
    artifacts/unconstrained_persistence_experiment/<setting>/<lagset>/summary_<tag>.csv
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
from run_link_only_experiment import LAG_SETS, PRECIP_DIR, ERA5_DIR, ENSO_PATH

ARTIFACTS_DIR = ROOT / "artifacts"
V2_DIR    = ARTIFACTS_DIR / "bound_ab_experiment_v2"   # source of original lags/cov_names only
OUT_ROOT  = ARTIFACTS_DIR / "unconstrained_persistence_experiment"

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]
STAGE_DIRS = ("stage2", "stage2_phi")
RELIABLE_GRAD_THRESHOLD = 1.0

A_MAX = 2.0  # tightened from 20.0 (v2 of this experiment) -- ruled out as the driver (v3 test)
B_MAX = 2.0  # v4: individual per-lag cap on B instead of the sum-tanh link -- see module docstring
RHO_MAX = 0.999
STATIC_BOX = (-2.0, 15.0)

SETTING_KWARGS = {
    "gb2_constraint": dict(free_zeta_gamma=True, kurtosis_margin=4.0,
                            bound_static_box=False),
    "static_box":     dict(free_zeta_gamma=False,
                            bound_static_box=True, static_box=STATIC_BOX),
}


def _block_name(model_id: str) -> str:
    return model_id.replace("stage2_phi_", "").replace("stage2_", "")


def refit_one(setting: str, lagset_name: str, run_id: str, stage: str, model_id: str, data: dict) -> dict:
    orig_dir = V2_DIR / run_id / stage / model_id
    orig_meta = json.loads((orig_dir / "metadata.json").read_text())
    orig_params_df = pd.read_csv(orig_dir / "estimated_parameters.csv")
    tv_names = orig_meta.get("tv_names", ["phi"])
    orig_lags = orig_meta.get("lags") or infer_gas_lags(orig_params_df, tv_names[0])
    cov_names = orig_meta.get("cov_names", [])
    dist = GB2LogLink() if len(tv_names) > 1 else GB2LogLinkPhiOnly()
    pi_dyn = make_pi_dynamics("ar_logistic", seasonal="daily")

    block = _block_name(model_id)
    X_train = data["covariate_blocks_train"][block]
    y_train = data["y_train"]
    lags = LAG_SETS[lagset_name]

    model = CovZAGASModel(
        distribution=dist, pi_dynamics=pi_dyn, seasonal="daily",
        gas_lags=lags, scaling=orig_meta.get("scaling", "diagonal_inverse_fisher"),
        cov_names=cov_names,
        bound_ab=True, ab_signed=True, a_max=A_MAX, b_max=B_MAX,
        bound_b_joint=False, bound_b_tanh_sum=False,
        bound_rho=True, rho_signed=True, rho_max=RHO_MAX,
        unconstrained_state=False,
        unconstrained_penalties=True,
        **SETTING_KWARGS[setting],
    )
    theta0 = model.initial_theta(y_train)

    t0 = time.time()
    result = model.fit(y_train, X_train, theta0=theta0, options={"maxiter": 2000, "gtol": 1e-3})
    runtime = time.time() - t0

    out_dir = OUT_ROOT / setting / lagset_name / run_id / stage / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_result(result, out_dir)

    new_meta = json.loads((out_dir / "metadata.json").read_text())
    contam = check_penalty_contamination(new_meta)
    any_contam = any(contam.values())
    n_contam_blocks = sum(1 for v in contam.values() if v)

    bd = result.get("bound_diagnostics", {}) or {}
    b_sums = {k: v for k, v in bd.items() if k.endswith("_B_sum")}

    out = {
        "setting": setting, "lagset": lagset_name,
        "run_id": run_id, "stage": stage, "model_id": model_id,
        "orig_lagset_matches": (list(orig_lags) == lags),
        "grad_norm_inf": result["grad_norm_inf"],
        "loglik": result["loglik"],
        "max_state": bd.get("max_state_magnitude", np.nan),
        "b_sums": b_sums,
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
    parser.add_argument("--setting", required=True, choices=list(SETTING_KWARGS))
    parser.add_argument("--lagset", required=True, choices=list(LAG_SETS))
    parser.add_argument("--location", default=None)
    args = parser.parse_args()
    locations = [args.location] if args.location else LOCATIONS

    out_dir = OUT_ROOT / args.setting / args.lagset
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for location in locations:
        cfg = STATION_REGISTRY[location]
        run_id, display = cfg["run_id"], cfg.get("display", location)
        run_dir = V2_DIR / run_id
        print(f"=== {display} [{args.setting}/{args.lagset}] ===")

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
                    out = refit_one(args.setting, args.lagset, run_id, stage, model_id, data)
                    b_sum_str = ", ".join(f"{k}={v:.3g}" for k, v in out["b_sums"].items())
                    print(f"  [{stage}/{model_id}] grad={out['grad_norm_inf']:.4g} "
                          f"reliable={out['now_reliable']} max_state={out['max_state']:.3g} "
                          f"contam_blocks={out['n_contaminated_blocks']} B_sum=({b_sum_str}) "
                          f"({time.time()-t0:.1f}s)")
                    summary_rows.append({"status": "ok", **{k: v for k, v in out.items() if k != "b_sums"},
                                          **{f"bsum_{k}": v for k, v in out["b_sums"].items()}})
                except Exception as exc:
                    print(f"  [{stage}/{model_id}] FAILED: {exc}")
                    traceback.print_exc()
                    summary_rows.append({"status": "failed", "run_id": run_id, "stage": stage,
                                          "model_id": model_id, "error": str(exc)})

    tag = locations[0].split()[0] if len(locations) == 1 else "all"
    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / f"summary_{tag}.csv", index=False)
    ok = df[df["status"] == "ok"]
    if len(ok):
        n_ok = int(ok["now_reliable"].sum())
        n_contam = int(ok["any_contamination"].sum())
        n_nonstat = int((ok.filter(like="bsum_").abs() > 1.0).any(axis=1).sum()) if any(c.startswith("bsum_") for c in ok.columns) else 0
        print(f"\nDone [{args.setting}/{args.lagset}]. {len(ok)} models. {n_ok}/{len(ok)} reliable. "
              f"{n_contam}/{len(ok)} contaminated. {n_nonstat}/{len(ok)} have sum|B_l|>1 (genuinely non-stationary).")


if __name__ == "__main__":
    main()
