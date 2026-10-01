"""
Independent numerical-gradient recomputation for Stage 3/4/5 (2026-09-06).

Loads each already-fitted model (no re-optimization) at its saved theta and
recomputes the gradient norm via diagnostics.numerical_diff.numerical_gradient
-- independent of scipy's own res.jac, per explicit instruction -- for the
cross-stage gradient comparison report.

    Stage 3: R.resolve_baseline(short, branch) -- Harvey long-short, standard
    Stage 4: artifacts/harmonic_seasonal_experiment/<short>/<branch>/m5_final_damp0.999
    Stage 5 (regime): artifacts/regime_xi_only_corrected/<short>/q{95,98}

Cached to artifacts/full_gradient_comparison/<short>/stage{3,4,5}/...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_harmonic_seasonal_experiment as R
from models.regime_gas import build_regime_xi_only_from_frozen_phi
from pipeline.artifact_utils import load_model_and_theta
from diagnostics.numerical_diff import numerical_gradient_checked
from diagnostics.scoring import compute_extended_oos_metrics
from data.station_loader import STATION_REGISTRY, StationDataLoader

LOCATIONS = ["BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
             "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR"]
BRANCHES = ["phi_only", "phi_xi"]
OUT_DIR = ROOT / "artifacts" / "full_gradient_comparison"
FRM_DIR = ROOT / "artifacts" / "final_report_metrics"
REGIME_DIR = ROOT / "artifacts" / "regime_xi_only_corrected"


def _harvey_grad(model_dir, data, force):
    model, theta, meta, _ = load_model_and_theta(model_dir)
    Xl_tr = R._build_X(data, meta["long_names"], "train")
    Xs_tr = R._build_X(data, meta["short_names"], "train")
    obj_fn = lambda th: model._run_filter(th, data["y_train"], Xl_tr, Xs_tr, return_paths=False)
    g_res = numerical_gradient_checked(obj_fn, theta)
    return {"grad_norm_inf_numeric": float(np.max(np.abs(g_res["grad"]))),
            "grad_converged": g_res["converged"], "grad_eps_used": g_res["eps_used"],
            "grad_trace": g_res["trace"]}


def run_one(location: str, short: str, force: bool) -> dict:
    loader = StationDataLoader.from_registry(location, precip_dir=R.PRECIP_DIR, era5_dir=R.ERA5_DIR, enso_path=R.ENSO_PATH)
    data = loader.load_all()
    out = {"stage3": {}, "stage4": {}, "stage5": {}}

    for branch in BRANCHES:
        out_dir3 = OUT_DIR / short / "stage3" / branch
        cache3 = out_dir3 / "grad.json"
        if cache3.exists() and not force:
            out["stage3"][branch] = json.loads(cache3.read_text())
        else:
            baseline = R.resolve_baseline(short, branch)
            result3 = _harvey_grad(baseline["model_dir"], data, force)
            out_dir3.mkdir(parents=True, exist_ok=True)
            cache3.write_text(json.dumps(result3))
            out["stage3"][branch] = result3
        conv3 = out["stage3"][branch].get("grad_converged")
        print(f"    stage3/{branch}: grad={out['stage3'][branch]['grad_norm_inf_numeric']:.4f}"
              f"{'' if conv3 else ' [NOT CONVERGED]' if conv3 is not None else ''}")

        out_dir4 = OUT_DIR / short / "stage4" / branch
        cache4 = out_dir4 / "grad.json"
        seas_dir = R.OUT_DIR / short / branch / "m5_final_damp0.999"
        if cache4.exists() and not force:
            out["stage4"][branch] = json.loads(cache4.read_text())
        elif seas_dir.exists():
            try:
                result4 = _harvey_grad(seas_dir, data, force)
            except Exception as exc:
                result4 = {"grad_norm_inf_numeric": float("nan"), "grad_converged": None}
                print(f"      stage4/{branch} FAILED: {exc}")
            out_dir4.mkdir(parents=True, exist_ok=True)
            cache4.write_text(json.dumps(result4))
            out["stage4"][branch] = result4
        conv4 = out["stage4"].get(branch, {}).get("grad_converged")
        print(f"    stage4/{branch}: grad={out['stage4'].get(branch, {}).get('grad_norm_inf_numeric')}"
              f"{'' if conv4 else ' [NOT CONVERGED]' if conv4 is not None else ''}")

    # Stage 5 (regime): phi_xi-shaped only (frozen phi from best phi-only base)
    reg_summary_p = REGIME_DIR / "summary.json"
    if reg_summary_p.exists():
        reg = json.loads(reg_summary_p.read_text())
        e = reg.get(short)
        if e is not None:
            best_q, _ = min(e["thresholds"].items(), key=lambda kv: kv[1]["crps_mean"])
            out_dir5 = OUT_DIR / short / "stage5"
            cache5 = out_dir5 / "grad.json"
            if cache5.exists() and not force:
                out["stage5"] = json.loads(cache5.read_text())
            else:
                base_dir = Path(e["phi_base_dir"])
                y_train, y_test = data["y_train"], data["y_test"]
                bt, bv, cn = data["covariate_blocks_train"], data["covariate_blocks_test"], data["covariate_col_names"]
                _, _, base_meta, _ = load_model_and_theta(base_dir)
                Xl_tr = R._build_X(data, base_meta["long_names"], "train")
                Xl_te = R._build_X(data, base_meta["long_names"], "test")
                Xs_tr = R._build_X(data, base_meta["short_names"], "train")
                Xs_te = R._build_X(data, base_meta["short_names"], "test")
                model, frozen = build_regime_xi_only_from_frozen_phi(
                    base_model_dir=base_dir, y_train=y_train, y_test=y_test,
                    xi_cov_train=bt["dewtemp_seasonal"], xi_cov_test=bv["dewtemp_seasonal"],
                    xi_cov_names=cn["dewtemp_seasonal"], threshold_quantile=float(best_q) / 100.0,
                    base_extra_fit_kwargs={"X_long": Xl_tr, "X_short": Xs_tr},
                    base_extra_oos_kwargs={"X_long_train": Xl_tr, "X_long_test": Xl_te,
                                            "X_short_train": Xs_tr, "X_short_test": Xs_te},
                )
                theta_p = REGIME_DIR / short / f"q{best_q}" / "estimated_parameters.csv"
                import pandas as pd
                theta_df = pd.read_csv(theta_p)
                theta = theta_df["value"].values
                model._threshold_c = model.compute_threshold(y_train)
                phi_a = model._align_frozen(frozen["phi_full"], frozen["warmup"], len(y_train))
                pi_a = model._align_frozen(frozen["pi_full"], frozen["warmup"], len(y_train))
                obj_fn = lambda th: model._run_filter(th, y_train, phi_a, pi_a, frozen["X_train"],
                                                       frozen["gamma_v"], frozen["zeta_v"], frozen["warmup"],
                                                       return_paths=False)
                g_res = numerical_gradient_checked(obj_fn, theta)
                out_dir5.mkdir(parents=True, exist_ok=True)
                result = {"grad_norm_inf_numeric": float(np.max(np.abs(g_res["grad"]))),
                          "grad_converged": g_res["converged"], "grad_eps_used": g_res["eps_used"],
                          "best_q": best_q}
                cache5.write_text(json.dumps(result))
                out["stage5"] = result
            conv5 = out["stage5"].get("grad_converged")
            print(f"    stage5: grad={out['stage5'].get('grad_norm_inf_numeric')}"
                  f"{'' if conv5 else ' [NOT CONVERGED]' if conv5 is not None else ''}")
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary = {}
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        short = cfg.get("short", location)
        print(f"=== {cfg.get('display', location)} ===")
        summary[short] = run_one(location, short, args.force)

    (OUT_DIR / "summary_stage345.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote {OUT_DIR / 'summary_stage345.json'}")


if __name__ == "__main__":
    main()
