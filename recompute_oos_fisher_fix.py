"""
2026-10-01: recompute out-of-sample artifacts after the GB2 Fisher cross-term fix.

WHAT WAS WRONG
    distributions/gb2_log_link.py::fisher_info_full returned the log-scale
    cross term I_{phi,xi} = -p*a*b/(a+b+1); the correct value is
    +p*a*b/(a+b) (proof: reports/fisher/fisher_gb2.pdf). The Numba kernels
    (models/_nb_gb2_filter.py::_gb2_fi_cross) always used the correct value,
    so every FIT, gradient, AIC and in-sample metric is correct. Only the
    pure-Python scaled-score path was wrong, and that path is what
    simulate_oos() and diagnostics/multistep_forecast.py use. The error
    therefore reaches exactly the OOS metrics (rolling 1-step and multistep)
    of 2-TVP (phi, xi) candidates fitted with scaling="inverse_fisher".

WHAT THIS SCRIPT DOES
    For every such candidate that feeds the 2026-09-11 stage-comparison
    selection (Stage 1/2/4S from artifacts/rerun_10092026/, Stage 3/4 from
    artifacts/rerun_10092026_phiL_sigmoid/):
      1. rebuild the model at its cached theta (no refit),
      2. re-run run_rerun_10092026._evaluate() -- the single source of truth
         for every OOS metric definition -- with the corrected code,
      3. write the corrected artifacts to a NEW tree,
             artifacts/regenerated_01102026/<source tree>/<loc>/<stage>/<cfg>/
         (result.json with a "pre_fix" block holding the old OOS values,
          multistep.csv, insample_qr.npy, theta.npy),
         leaving the original artifacts untouched for comparison,
      4. write artifacts/regenerated_01102026/metrics.json: a copy of
         artifacts/stage_comparison_20260911/metrics.json with the OOS fields
         (rmse_oos, crps_oos, tw95, tw98) of the affected cells replaced.
         Selection fields (grad_inf, frac_conv) are in-sample and unchanged.

    Stage 1's per-lag scaling choice used only gradient tier and AIC (both
    in-sample), so no scaling choice changes. Stage 5 has no affected
    candidates and is excluded from every report anyway.

NUMERICAL CHECK
    After the fix, simulate_oos's one-step paths match the Numba filter run
    over train+test to ~1e-9 for Stages 1/2/3/4S (they disagreed before).

USAGE
    python recompute_oos_fisher_fix.py [--workers 20]
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import collect_stage_comparison_metrics as C
import run_rerun_10092026 as D

OUT = ROOT / "artifacts" / "regenerated_01102026"
OLD_METRICS = ROOT / "artifacts" / "stage_comparison_20260911" / "metrics.json"
AFFECTED_STAGES = ("stage1", "stage2", "stage3", "stage4", "stage4S")
OOS_FIELDS = ("rmse", "crps_mean", "twcrps_95", "twcrps_98",
              "exc_95_1s", "kup95_1s_p", "exc_98_1s", "kup98_1s_p")


def affected_dirs():
    """Every 2-TVP inverse_fisher candidate dir in the tree the selection reads."""
    out = []
    for stage in AFFECTED_STAGES:
        tree = C._tree_for(stage)
        for rp in sorted(tree.glob(f"*/{stage}/*__xi/result.json")):
            res = json.loads(rp.read_text())
            if res.get("scaling") == "inverse_fisher":
                out.append(rp.parent)
    return out


def _covariate_kwargs(stage, res, data, filt_kwargs):
    """(oos_kwargs, multistep_kwargs) in the form _evaluate expects."""
    if stage == "stage1":
        return {}, {}
    if stage == "stage2":
        X_te, _ = D._cov_cols(data, res["cov"], "test")
        ok = {"X_train": filt_kwargs["X"], "X_test": X_te}
        return {"X_train": ok["X_train"], "X_test": ok["X_test"]}, ok
    Xl_te, _, Xs_te, _ = D._harvey_cov(data, res["cov"], "test")
    ok = {"X_long_train": filt_kwargs["X_long"], "X_long_test": Xl_te,
          "X_short_train": filt_kwargs["X_short"], "X_short_test": Xs_te}
    return ok, ok


def recompute_one(cfg_dir: Path):
    stage, short = cfg_dir.parts[-2], cfg_dir.parts[-3]
    tree_name = cfg_dir.parts[-4]
    cfg = cfg_dir.name[: -len("__xi")]
    loc = next(l for l in D.LOCATIONS if C._short(l) == short)
    res = json.loads((cfg_dir / "result.json").read_text())
    theta = np.load(cfg_dir / "theta.npy")
    data = D._load_data(loc)
    dest = OUT / tree_name / short / stage / cfg_dir.name
    try:
        model, fk = C.build_model(loc, stage, cfg, "phi_xi", res, data)
        ok, mk = _covariate_kwargs(stage, res, data, fk)
        new_res, ms, qr, _ = D._evaluate(
            model, theta, data, fk, ok, mk,
            {"loglik": res.get("loglik"), "success": res.get("success"),
             "validity": res.get("validity")})
        fixed = dict(res)
        fixed["pre_fix"] = {k: res.get(k) for k in OOS_FIELDS}
        # gradient fields are in-sample (Numba path, unaffected): keep the originals
        fixed.update({k: v for k, v in new_res.items()
                      if k not in ("grad_norm_inf_numeric", "grad_converged", "grad_eps_used")})
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "result.json").write_text(json.dumps(fixed, indent=2, default=float))
        ms.to_csv(dest / "multistep.csv", index=False)
        np.save(dest / "insample_qr.npy", qr)
        shutil.copy2(cfg_dir / "theta.npy", dest / "theta.npy")
        return (short, stage, cfg, "ok", res.get("rmse"), fixed.get("rmse"))
    except Exception:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "error.txt").write_text(traceback.format_exc())
        return (short, stage, cfg, "error", None, None)


def _metrics_key(stage, short, cfg):
    """Map a cfg-dir name to the metrics.json [cfg] key."""
    if stage == "stage1":   # dir = <lag>_<scaling>; metrics key = <lag>
        return cfg[: -len("_inverse_fisher")]
    return cfg


def build_metrics():
    metrics = json.loads(OLD_METRICS.read_text())
    changed = []
    for rp in sorted(OUT.glob("*/*/*/*__xi/result.json")):
        stage, short = rp.parts[-3], rp.parts[-4]
        cfg = rp.parent.name[: -len("__xi")]
        key = _metrics_key(stage, short, cfg)
        cell = metrics.get(short, {}).get(stage, {}).get(key, {}).get("phi_xi")
        if not cell or cell.get("error"):
            continue
        # Stage 1 metrics cells hold only the CHOSEN scaling per lag; skip others.
        if stage == "stage1":
            choice = json.loads((C.D.OUT / short / "stage1" / "_scaling_choice__xi.json").read_text())
            if choice["per_lag"].get(key, {}).get("scaling") != "inverse_fisher":
                continue
        r = json.loads(rp.read_text())
        cell["pre_fix"] = {k: cell.get(k) for k in ("rmse_oos", "crps_oos", "tw95", "tw98")}
        cell.update({"rmse_oos": r.get("rmse"), "crps_oos": r.get("crps_mean"),
                     "tw95": r.get("twcrps_95"), "tw98": r.get("twcrps_98"),
                     "regenerated_01102026": True})
        changed.append((short, stage, key))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return changed


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--metrics-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.metrics_only:
        dirs = affected_dirs()
        print(f"{len(dirs)} affected candidates")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(recompute_one, d) for d in dirs]
            for i, f in enumerate(as_completed(futs), 1):
                short, stage, cfg, status, old, new = f.result()
                print(f"[{i}/{len(dirs)}] {short}/{stage}/{cfg}: {status} "
                      f"rmse {old} -> {new}", flush=True)
    changed = build_metrics()
    print(f"metrics.json: {len(changed)} cells updated -> {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
