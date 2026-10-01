"""
Corrected RegimeXiOnlyModel ("Stage 4 xi-regime") re-fit (2026-09-06).

=============================================================================
WHY THIS SCRIPT EXISTS
=============================================================================

The short summary report's "regime change in GAS for time-varying shape"
section originally evaluated the WRONG model: models/threshold_weighted_xi_gas.py
(docs/MODELS.md sec.31, "Stage 5"), which deliberately freezes phi at a
FULLY STATIC (constant, unconditional-MLE) value -- an ablation asking
"does any time-varying scale matter at all". That is a real, documented,
intentional design (sec.31.1-31.2), not a bug.

What was actually asked for is a DIFFERENT, separately-existing model:
models/regime_gas.py's RegimeXiOnlyModel (docs/MODELS.md sec.22-23,
labeled "stage4_xi_regime" in run_all_locations.py), which keeps phi_t
DYNAMIC -- frozen as a precomputed array taken from "whichever phi-only
model won" the earlier stages (regime_gas.py:525-535, quoting prompt.md
objective 3 verbatim) -- and adds a regime-sensitive score response only
to xi's own GAS(1,1) recursion.

That model has already been fit once (artifacts/run_*/stage4_xi_regime/,
dated 2026-07-08), but its frozen phi base predates the seasonal harmonic
experiment entirely (m5_final_damp0.999, created 2026-09-05) -- so it could
only ever have used the Stage-3 (non-seasonal) winner as its phi base, not
"whichever of the seasonal/non-seasonal Harvey long-short models has the
better OOS CRPS", which is what was asked for this time.

This script re-fits RegimeXiOnlyModel with that updated base-selection
rule, at q95/q98 (matching the original stage4_xi_regime convention),
using unbounded BFGS + polish (matching this project's standing
optimization preference -- see feedback_optimization memory), for all
six locations. It writes to its OWN dedicated cache (never overwriting
the original run_*/stage4_xi_regime/ artifacts, which remain a valid
historical record of the pre-seasonal-era fit):

    artifacts/regime_xi_only_corrected/<short>/q<label>/

=============================================================================
USAGE
=============================================================================

    python run_regime_xi_only_corrected.py [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_harmonic_seasonal_experiment as R
from models.regime_gas import build_regime_xi_only_from_frozen_phi
from pipeline.artifact_utils import load_model_and_theta
from diagnostics.scoring import compute_extended_oos_metrics
from data.station_loader import STATION_REGISTRY, StationDataLoader

LOCATIONS = ["BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
             "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR"]
THRESHOLDS = [("95", 0.95), ("98", 0.98)]
FRM_DIR = ROOT / "artifacts" / "final_report_metrics"
OUT_DIR = ROOT / "artifacts" / "regime_xi_only_corrected"


def _select_phi_base(short: str) -> tuple[str, Path]:
    """Pick whichever of {stage3 (non-seasonal), stage4 (seasonal damped)}
    has the better (lower) OOS CRPS for this location's phi-only branch,
    falling back to stage3 if stage4 is missing/diverged (non-finite)."""
    def crps_of(stage: str):
        p = FRM_DIR / short / "phi_only" / stage / "onestep_metrics.json"
        if not p.exists():
            return None
        return json.loads(p.read_text()).get("crps_mean")

    crps3, crps4 = crps_of("stage3"), crps_of("stage4")
    baseline = R.resolve_baseline(short, "phi_only")
    stage3_dir = baseline["model_dir"]
    stage4_dir = R.OUT_DIR / short / "phi_only" / "m5_final_damp0.999"

    use_seasonal = (crps4 is not None and np.isfinite(crps4)
                     and (crps3 is None or not np.isfinite(crps3) or crps4 < crps3))
    if use_seasonal:
        return "stage4_seasonal", stage4_dir
    return "stage3", stage3_dir


def run_one(location: str, short: str, force: bool) -> dict:
    loader = StationDataLoader.from_registry(location, precip_dir=R.PRECIP_DIR,
                                              era5_dir=R.ERA5_DIR, enso_path=R.ENSO_PATH)
    data = loader.load_all()
    y_train, y_test = data["y_train"], data["y_test"]
    bt, bv, cn = data["covariate_blocks_train"], data["covariate_blocks_test"], data["covariate_col_names"]

    base_label, base_dir = _select_phi_base(short)
    _, _, base_meta, _ = load_model_and_theta(base_dir)
    Xl_tr = R._build_X(data, base_meta["long_names"], "train")
    Xl_te = R._build_X(data, base_meta["long_names"], "test")
    Xs_tr = R._build_X(data, base_meta["short_names"], "train")
    Xs_te = R._build_X(data, base_meta["short_names"], "test")
    fit_kw_base = {"X_long": Xl_tr, "X_short": Xs_tr}
    oos_kw_base = {"X_long_train": Xl_tr, "X_long_test": Xl_te,
                   "X_short_train": Xs_tr, "X_short_test": Xs_te}

    results = {"display": STATION_REGISTRY[location].get("display", location),
               "phi_base": base_label, "phi_base_dir": str(base_dir), "thresholds": {}}

    loc_dir = OUT_DIR / short
    for q_label, q_val in THRESHOLDS:
        out_dir = loc_dir / f"q{q_label}"
        cache_p = out_dir / "onestep_metrics.json"
        if cache_p.exists() and not force:
            results["thresholds"][q_label] = json.loads(cache_p.read_text())
            continue

        model, frozen = build_regime_xi_only_from_frozen_phi(
            base_model_dir=base_dir, y_train=y_train, y_test=y_test,
            xi_cov_train=bt["dewtemp_seasonal"], xi_cov_test=bv["dewtemp_seasonal"],
            xi_cov_names=cn["dewtemp_seasonal"], threshold_quantile=q_val,
            base_extra_fit_kwargs=fit_kw_base, base_extra_oos_kwargs=oos_kw_base,
        )

        t0 = time.time()
        fit_res = model.fit(
            y=y_train, phi_full=frozen["phi_full"], pi_full=frozen["pi_full"],
            gamma_v=frozen["gamma_v"], zeta_v=frozen["zeta_v"], warmup=frozen["warmup"],
            X=frozen["X_train"], verbose=False,
        )
        oos = model.simulate_oos(
            fit_res["theta"], y_train, y_test,
            frozen["phi_full"], frozen["pi_full"], frozen["gamma_v"], frozen["zeta_v"],
            frozen["warmup"], X_train=frozen["X_train"], X_test=frozen["X_test"],
        )
        m = compute_extended_oos_metrics(model, oos, y_test, quantile_levels=(0.95, 0.98),
                                          twcrps_levels=(0.95, 0.98), brier_levels=(), n_draws=500, seed=42)
        m["grad_norm_inf"] = fit_res["grad_norm_inf"]
        m["success"] = fit_res["success"]
        m["validity"] = fit_res["validity"]
        m["loglik"] = fit_res["loglik"]
        m["runtime_s"] = time.time() - t0
        m["phi_base"] = base_label
        m["phi_base_dir"] = str(base_dir)

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "onestep_metrics.json").write_text(json.dumps(m, indent=2, default=float))
        import pandas as pd
        pd.DataFrame({"parameter": fit_res["param_names"], "value": fit_res["theta"]}).to_csv(
            out_dir / "estimated_parameters.csv", index=False)
        results["thresholds"][q_label] = m
        print(f"    q{q_label}: rmse={m.get('rmse'):.3f} crps={m.get('crps_mean'):.3f} "
              f"tw95={m.get('twcrps_95'):.3f} tw98={m.get('twcrps_98'):.3f} "
              f"grad_inf={m.get('grad_norm_inf'):.4f} success={m.get('success')} "
              f"(phi base: {base_label})")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        short = cfg.get("short", location)
        print(f"=== {cfg.get('display', location)} ===")
        try:
            summary[short] = run_one(location, short, args.force)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
