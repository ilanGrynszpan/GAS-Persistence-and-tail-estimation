"""
Stage 3 / Stage 4 / Stage 5 rerun with sigmoid-bounded phi_L (2026-09-11).

WHY THIS EXISTS
================================================================================
generate_stage34_stability_report.py's H1 finding: phi_L was a RAW,
unbounded parameter (harvey_gas.py's default phi_L_max=None), kept under
0.995 only by a soft quadratic penalty (LAM_LONG=1e5). It sat at/past that
boundary in 70%/61% of Stage 3/4 fits, and the resulting stiff penalty
region produced checked gradient norms of 1e4-1e6 -- ~2400x larger than
when phi_L was comfortably away from the boundary.

Per user request, this reruns Stage 3/4 (and Stage 5, which also fits a
Harvey phi_L) with:
    phi_L_max = 1.0   -> phi_L = sigmoid(raw) in the OPEN interval (0,1),
                         UNRESTRICTED otherwise: no soft upper-bound penalty
                         is added or needed (harvey_gas.py already gates
                         that term off whenever phi_L_max is not None,
                         since the reparameterisation guarantees the domain
                         by construction) -- phi_L can approach 1
                         asymptotically, with a smoothly VANISHING gradient
                         there instead of hitting a stiff quadratic wall.
    phi_S_max = None  -> phi_S stays RAW, restricted only by its existing
                         soft penalties: phi_S < phi_L (order) and
                         phi_S > 0 (positivity) -- i.e. exactly "short only
                         has to be lower than long," nothing more.

SCOPE: Stage 1/2 do not involve phi_L/phi_S at all and are untouched --
reused as-is from artifacts/rerun_10092026/ (same Fisher-scaling choice).
Stage 4S (include_long=False) has no phi_L in its theta at all, so
phi_L_max is moot there too -- also reused as-is. Only Stage 3, Stage 4,
and Stage 5 are refit here, into a SEPARATE cache tree so the two
phi_L parameterisations (raw-unbounded vs sigmoid-bounded) are never
conflated:
    artifacts/rerun_10092026_phiL_sigmoid/<loc>/<stage>/<cfg>[__xi]/

_evaluate() already computes every field the report needs (OOS/IS
RMSE/CRPS, twCRPS, BSS, 1-step + multi-step exceedance/Kupiec, in-sample
QR) at fit time, so no separate augment pass is required here.

Run:  python run_stage34_phiL_sigmoid.py --workers 22
"""

from __future__ import annotations

import argparse
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
from distributions.gb2_log_link import GB2LogLink
from models.harvey_gas import HarveyZAGASModel

OUT = ROOT / "artifacts" / "rerun_10092026_phiL_sigmoid"
OLD = D.OUT  # artifacts/rerun_10092026 -- source of Stage 1 scaling + Stage 4S (untouched)
PHI_L_MAX = 1.0


def _short(loc):
    return STATION_REGISTRY[loc].get("short", loc)


def _task_harvey_sig(a):
    loc, branch, stage, cfg, force = a
    short = _short(loc)
    cdir = OUT / short / stage / (cfg + D._sfx(branch))
    if (cdir / "result.json").exists() and not force:
        return ("skip", f"{stage} {short}/{branch}/{cfg}")
    try:
        data = D._load_data(loc)
        scaling = D._chosen_scaling(short, branch)  # Stage 1 choice, from the OLD tree
        harm = 0 if stage == "stage3" else D.N_HARM
        damp = None if stage == "stage3" else D.DAMP
        Xl_tr, ln, Xs_tr, sn = D._harvey_cov(data, cfg, "train")
        Xl_te, _, Xs_te, _ = D._harvey_cov(data, cfg, "test")
        model = HarveyZAGASModel(
            distribution=D._dist(branch), pi_dynamics=D.pi_dyn(), seasonal="daily",
            long_names=ln, short_names=sn, harvey_target=D._htarget(branch),
            scaling=scaling, static_params=D._static(branch),
            n_harmonics=harm, seasonal_damping_max=damp, include_long=True,
            phi_L_max=PHI_L_MAX,
        )
        opts = {"options": {"maxiter": 300}, "polish": False} if stage == "stage4" else {}
        fr = model.fit(y=data["y_train"], X_long=Xl_tr, X_short=Xs_tr, verbose=False, **opts)
        ok = {"X_long_train": Xl_tr, "X_long_test": Xl_te,
              "X_short_train": Xs_tr, "X_short_test": Xs_te}
        res, ms, qr, th = D._evaluate(model, fr["theta"], data,
                                      {"X_long": Xl_tr, "X_short": Xs_tr}, ok, ok, fr)
        res["scaling"] = scaling; res["cov"] = cfg; res["stage"] = stage; res["branch"] = branch
        res["phi_L_max"] = PHI_L_MAX
        D._save(cdir, res, ms, qr, th)
        return ("done", f"{stage} {short}/{branch}/{cfg} rmse={res['rmse']:.3f} "
                        f"crps={res['crps_mean']:.3f} grad={res['grad_norm_inf_numeric']:.2f}")
    except Exception:
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "fit_error.txt").write_text(traceback.format_exc())
        return ("err", f"{stage} {short}/{branch}/{cfg}\n{traceback.format_exc()}")


def _select_best_harvey_mixed(short):
    """Same selection rule as run_rerun_10092026._select_best_harvey
    (gradient tier -> CRPS -> RMSE), but stage3/stage4 candidates come from
    THIS rerun's sigmoid-phi_L tree while stage4S (untouched, no phi_L at
    all) comes from the original tree."""
    def tier(r):
        if r.get("validity") == "failed":
            return 3
        g = r.get("grad_norm_inf_numeric", 1e18)
        return 0 if g < 1 else (1 if g < 100 else 2)
    cands = []
    for st, root in [("stage3", OUT), ("stage4", OUT), ("stage4S", OLD)]:
        for cfg in D.HARVEY_COV:
            for br in ("phi_only", "phi_xi"):
                r = D.load_json(root / short / st / (cfg + D._sfx(br)) / "result.json")
                if r:
                    cands.append((st, cfg, br, r))
    if not cands:
        return None
    st, cfg, br, _ = min(cands, key=lambda x: (tier(x[3]), x[3].get("crps_mean", 1e18),
                                               x[3].get("rmse", 1e18)))
    return st, cfg, br


def _task_stage5_sig(a):
    loc, force = a
    short = _short(loc)
    cdir = OUT / short / "stage5" / "best"
    if (cdir / "result.json").exists() and not force:
        return ("skip", f"stage5 {short}")
    try:
        data = D._load_data(loc)
        scaling = D._chosen_scaling(short, "phi_only")
        sel = _select_best_harvey_mixed(short)
        if sel is None:
            return ("err", f"stage5 {short}: no Harvey base found")
        base_stage, cfg, base_branch = sel
        harm = 0 if base_stage == "stage3" else D.N_HARM
        damp = None if base_stage == "stage3" else D.DAMP
        incl_long = (base_stage != "stage4S")
        Xl_tr, ln, Xs_tr, sn = D._harvey_cov(data, cfg, "train")
        Xl_te, _, Xs_te, _ = D._harvey_cov(data, cfg, "test")
        model = HarveyZAGASModel(
            distribution=GB2LogLink(), pi_dynamics=D.pi_dyn(), seasonal="daily",
            long_names=ln, short_names=sn, harvey_target=["phi", "xi"],
            scaling=scaling, static_params=["gamma", "zeta"],
            n_harmonics=harm, seasonal_damping_max=damp, include_long=incl_long,
            phi_L_max=(PHI_L_MAX if incl_long else None),
        )
        fr = model.fit(y=data["y_train"], X_long=Xl_tr, X_short=Xs_tr, verbose=False,
                       options={"maxiter": 300}, polish=False)
        ok = {"X_long_train": Xl_tr, "X_long_test": Xl_te,
              "X_short_train": Xs_tr, "X_short_test": Xs_te}
        res, ms, qr, th = D._evaluate(model, fr["theta"], data,
                                      {"X_long": Xl_tr, "X_short": Xs_tr}, ok, ok, fr)
        res["stage"] = "stage5"; res["base_stage"] = base_stage; res["cov"] = cfg
        res["base_branch"] = base_branch; res["scaling"] = scaling
        if incl_long:
            res["phi_L_max"] = PHI_L_MAX
        D._save(cdir, res, ms, qr, th)
        return ("done", f"stage5 {short} (base {base_stage}/{cfg}/{base_branch}) "
                        f"rmse={res['rmse']:.3f} crps={res['crps_mean']:.3f} "
                        f"grad={res['grad_norm_inf_numeric']:.2f}")
    except Exception:
        return ("err", f"stage5 {short}\n{traceback.format_exc()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default=None, help="comma-separated location shorts")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    locs = [l for l in D.LOCATIONS
            if not args.only or _short(l) in args.only.split(",")]

    tasks = [(loc, br, st, cfg, args.force)
             for loc in locs for br in ("phi_only", "phi_xi")
             for st in ("stage3", "stage4") for cfg in D.HARVEY_COV]
    D._run_pool(tasks, _task_harvey_sig, args.workers, "stage3+4 (sigmoid phi_L)")

    D._run_pool([(loc, args.force) for loc in locs], _task_stage5_sig,
               min(args.workers, len(locs)), "stage5 (sigmoid phi_L)")

    print("\nDone. Results under", OUT)


if __name__ == "__main__":
    main()
