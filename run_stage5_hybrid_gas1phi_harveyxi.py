"""
Stage 5 hybrid variant (2026-09-11), run directly on user request: phi is a
PLAIN GAS(1,1) (no covariates, no long/short decomposition -- gas1_target),
xi keeps the Stage-5-style Harvey-driven dynamics (long/short or structural,
with whichever covariate set that location's Stage 5 already selected).

This is NOT the same as either the original Stage 5 (both phi and xi
Harvey-driven) or the "last attempt" plain-GAS(1,1)-for-both reported
earlier (which is just Stage 1's L1/phi_xi, already cached, no fit needed)
-- it is a new hybrid that needed a genuine code extension
(HarveyZAGASModel's new gas1_target flag) since the class previously could
only make a parameter either Harvey-dynamic or a frozen constant.

Reuses the exact base_stage/cov each location's existing Stage 5 already
picked (artifacts/rerun_10092026_phiL_sigmoid/<loc>/stage5/best/result.json)
so the two Stage-5 variants are directly comparable -- only which parameter
gets the complex treatment changes.

No numba kernel exists for gas1_target (HarveyZAGASModel._use_nb_filter
forces the pure-Python path whenever gas1_target is set) -- this run is
noticeably slower than every other fit in this project as a result.

Output: artifacts/stage5_hybrid_gas1phi_harveyxi/<loc>/result.json (+ theta.npy,
multistep.csv, insample_qr.npy), via the same _save()/_evaluate() helpers
already used throughout.
"""

from __future__ import annotations

import json
import sys
import traceback
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_rerun_10092026 as D
from data.station_loader import STATION_REGISTRY
from models.harvey_gas import HarveyZAGASModel
from distributions.gb2_log_link import GB2LogLink

OUT = ROOT / "artifacts" / "stage5_hybrid_gas1phi_harveyxi"
SIGMOID = ROOT / "artifacts" / "rerun_10092026_phiL_sigmoid"


def _short(loc):
    return STATION_REGISTRY[loc].get("short", loc)


def _task(a):
    loc, force = a
    short = _short(loc)
    cdir = OUT / short
    if (cdir / "result.json").exists() and not force:
        return ("skip", short)
    try:
        r5 = json.loads((SIGMOID / short / "stage5" / "best" / "result.json").read_text())
        base_stage, cov = r5["base_stage"], r5["cov"]
        data = D._load_data(loc)
        y_tr = data["y_train"]
        scaling = D._chosen_scaling(short, "phi_only")

        harm = 0 if base_stage == "stage3" else D.N_HARM
        damp = None if base_stage == "stage3" else D.DAMP
        incl_long = (base_stage != "stage4S")
        Xl_tr, ln, Xs_tr, sn = D._harvey_cov(data, cov, "train")
        Xl_te, _, Xs_te, _ = D._harvey_cov(data, cov, "test")

        model = HarveyZAGASModel(
            distribution=GB2LogLink(), pi_dynamics=D.pi_dyn(), seasonal="daily",
            long_names=ln, short_names=sn,
            harvey_target=["xi"], gas1_target=["phi"],
            scaling=scaling, static_params=["gamma", "zeta"],
            n_harmonics=harm, seasonal_damping_max=damp, include_long=incl_long,
            phi_L_max=(1.0 if incl_long else None),
        )
        t0 = time.time()
        fr = model.fit(y=y_tr, X_long=Xl_tr, X_short=Xs_tr, verbose=False,
                       options={"maxiter": 300}, polish=False)
        fit_s = time.time() - t0
        ok = {"X_long_train": Xl_tr, "X_long_test": Xl_te,
              "X_short_train": Xs_tr, "X_short_test": Xs_te}
        res, ms, qr, th = D._evaluate(model, fr["theta"], data,
                                      {"X_long": Xl_tr, "X_short": Xs_tr}, ok, ok, fr)
        res["stage"] = "stage5_hybrid"; res["base_stage"] = base_stage; res["cov"] = cov
        res["scaling"] = scaling; res["fit_seconds"] = fit_s
        D._save(cdir, res, ms, qr, th)
        return ("done", f"{short} (base {base_stage}/{cov}) fit={fit_s:.0f}s "
                        f"rmse={res['rmse']:.3f} crps={res['crps_mean']:.3f} "
                        f"grad={res['grad_norm_inf_numeric']:.2f}")
    except Exception:
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "fit_error.txt").write_text(traceback.format_exc())
        return ("err", f"{short}\n{traceback.format_exc()}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tasks = [(loc, args.force) for loc in D.LOCATIONS]
    D._run_pool(tasks, _task, args.workers, "stage5 hybrid (gas1 phi, harvey xi)")


if __name__ == "__main__":
    main()
