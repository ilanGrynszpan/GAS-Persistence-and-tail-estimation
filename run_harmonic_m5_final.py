"""
Harmonic-seasonal Harvey model, FIXED at m=5, one proper fit per location/
branch (2026-09-04).

=============================================================================
OVERVIEW
=============================================================================

Direct follow-up simplification of run_harmonic_seasonal_experiment.py's
progressive AIC search: per instruction, m=5 is adopted directly as THE
seasonal configuration for every location/branch (the search consistently
found m=5 already at or past the AIC-minimising point -- see that script's
search_log.json files -- so continuing to probe m=10, 15, ... adds cost
without changing the answer). This script does ONE properly-budgeted fit
of the m=5 model per location/branch, reusing the exact same baseline
resolution and warm-start logic as run_harmonic_seasonal_experiment.py
(refactored out to be imported here, not duplicated).

Three of the twelve combinations (BH phi_only, BH phi_xi, Cruzeiro
phi_only) already have a well-converged m=5 result from the earlier
search's first grid point (grad_norm_inf 1.18, 4.46, 1.60 -- all
comfortably below this project's standing reliability threshold) --
those are PROMOTED (copied) directly rather than re-fit, since refitting
would just reproduce the same deterministic optimum at real extra cost.
The remaining nine (three with a poorly-converged search-phase result --
Darwin both branches, Cruzeiro phi_xi, gradients in the hundreds -- and
six never yet attempted -- Garanhuns/Manaus/Salvador both branches) are
fit fresh here.

Per docs/EXECUTION.md, this does NOT touch any existing artifact.

=============================================================================
USAGE
=============================================================================

    python run_harmonic_m5_final.py
    python run_harmonic_m5_final.py --location "MANAUS" --branch phi_only

Output:
    artifacts/harmonic_seasonal_experiment/<short>/<branch>/m5_final/
    artifacts/harmonic_seasonal_experiment/m5_final_summary.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import STATION_REGISTRY, StationDataLoader
import run_harmonic_seasonal_experiment as R

OUT_DIR = R.OUT_DIR
M = 5
# 2026-09-04 UPDATE: the harmonic recursion is now numba-compiled (see
# models/harvey_gas.py's module docstring, "UPDATE (2026-09-04)") --
# per-eval cost dropped from ~175ms (pure Python) to sub-millisecond,
# verified bit-for-bit-equivalent (to the pre-existing FP noise floor) via
# a dedicated parity test. The polish=True hang described below was a
# consequence of that slow per-eval cost compounding an unbounded restart,
# NOT a problem with polish itself -- now that evals are ~100-1000x
# cheaper, the fuller budget is restored for all fits done from this point
# forward. The five combos already completed on the slow path (BH x2,
# Cruzeiro/phi_only, Darwin/phi_only, Cruzeiro/phi_only-promoted) are
# reused as-is (see already_done() below) rather than re-run at the new
# budget -- their results are valid, deterministic optimizer outputs, and
# CLAUDE.md directs reusing artifacts over recomputing them. Darwin/
# phi_only's high gradient (846.5) was root-caused separately (phi_L~=
# phi_S persistence collapse -- a structural near-non-identification, not
# an iteration-budget shortfall) so re-fitting it at a larger budget would
# not be expected to change that diagnosis.
FIT_MAXITER = 300     # matches run_harmonic_seasonal_experiment.py's own FINAL_MAXITER
FIT_POLISH  = True    # restored now that numba removes the per-eval cost that made
                       # polish's automatic restart take 3+ hours on the old pure-
                       # Python path (see UPDATE note above). Locations that still end
                       # up poorly converged at this budget are reported honestly via
                       # the grad-gated SE table's "(gradient too high to estimate std
                       # errors)" label, not silently hidden -- a bounded, reportable
                       # result beats an unbounded wait for a marginally better one.

# Combos with an already-good search-phase m=5 result -- promoted, not re-fit.
PROMOTE = [("bh", "phi_only"), ("bh", "phi_xi"), ("cruzeiro", "phi_only")]


def promote(short: str, branch: str, out_subdir: str) -> dict:
    src = OUT_DIR / short / branch / "search_m5"
    dst = OUT_DIR / short / branch / out_subdir
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy2(f, dst / f.name)
    meta = json.loads((dst / "metadata.json").read_text())
    print(f"  [{short}/{branch}] PROMOTED existing search-phase m=5 fit "
          f"(grad={meta['grad_norm_inf']:.4g}, AIC={meta['aic']:.2f})")
    return {"short": short, "branch": branch, "m": M, "source": "promoted",
            "aic": meta["aic"], "grad_norm_inf": meta["grad_norm_inf"], "loglik": meta["loglik"]}


def already_done(short: str, branch: str, out_subdir: str) -> dict | None:
    """Resumable execution (EXECUTION.md): reuse a completed fit at
    out_subdir instead of re-running it, whether it landed there via
    promote() or a prior fit_fresh() call (e.g. one interrupted mid-grid by
    an unrelated process kill -- see 2026-09-04 numba-enablement note
    below)."""
    meta_path = OUT_DIR / short / branch / out_subdir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if "aic" not in meta or "grad_norm_inf" not in meta:
            return None
        return meta
    except Exception:
        return None


def fit_fresh(short: str, branch: str, data: dict, out_subdir: str,
              kappa_gamma_bound: Optional[float], no_covariates: bool,
              seasonal_damping_max: Optional[float], reset_phi_ls: bool) -> dict:
    baseline = R.resolve_baseline(short, branch)
    t0 = time.time()
    long_ov  = [] if no_covariates else None
    short_ov = [] if no_covariates else None
    out = R.fit_one(short, branch, M, baseline, data,
                     maxiter=FIT_MAXITER, polish=FIT_POLISH, out_subdir=out_subdir,
                     kappa_gamma_bound=kappa_gamma_bound,
                     long_names_override=long_ov, short_names_override=short_ov,
                     seasonal_damping_max=seasonal_damping_max, reset_phi_ls=reset_phi_ls)
    dt = time.time() - t0
    print(f"  [{short}/{branch}] fit m=5: AIC={out['aic']:.2f} "
          f"(baseline {2*baseline['meta']['n_params']-2*baseline['meta']['loglik']:.2f}) "
          f"grad={out['grad_norm_inf']:.4g}  ({dt/60:.1f} min)")
    return {"short": short, "branch": branch, "m": M, "source": "fresh", **out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", default=None)
    parser.add_argument("--branch", default=None, choices=R.BRANCHES)
    parser.add_argument("--kappa-gamma-bound", type=float, default=None,
                         help="Symmetric sigmoid bound on kappa_gamma (e.g. 2.0 restricts it "
                              "to (-2,2)); default None reproduces the unbounded/raw theta layout.")
    parser.add_argument("--no-covariates", action="store_true",
                         help="Drop ALL Long (ENSO) and Short covariates -- Long/Short become pure "
                              "score-driven AR-like components (kappa_L*s_t / kappa_S*s_t only, no "
                              "c_L'x_L / c_S'x_S term), isolating whether the seasonal component's "
                              "collinearity with phi_L is driven by the covariates specifically.")
    parser.add_argument("--seasonal-damping-max", type=float, default=None,
                         help="Sigmoid upper bound on a new psi_gamma_j damping factor multiplying "
                              "the ENTIRE seasonal rotation each step (e.g. 0.999 lets the optimiser "
                              "choose any psi in (0, 0.999)); default None reproduces the exact "
                              "pure-rotation/undamped recursion as originally specified.")
    parser.add_argument("--reset-phi-ls", action="store_true",
                         help="Do NOT warm-start phi_L/phi_S from the baseline's own fitted values -- "
                              "start them at the class's own well-separated defaults (0.95, 0.50) "
                              "instead. For locations whose baseline model already shows a Long/Short "
                              "persistence collapse (phi_L~=phi_S), warm-starting from that collapse "
                              "just re-seeds the seasonal fit in the same bad basin.")
    parser.add_argument("--out-subdir", default=None,
                         help="Output subdirectory name under artifacts/harmonic_seasonal_experiment/"
                              "<short>/<branch>/; default 'm5_final', 'm5_final_kgboundX' when "
                              "--kappa-gamma-bound is set, 'm5_final_nocov' when --no-covariates is set, "
                              "'m5_final_dampX' when --seasonal-damping-max is set, 'm5_final_phireset' "
                              "when --reset-phi-ls is set (suffixes combined if multiple flags are given).")
    args = parser.parse_args()
    locations = [args.location] if args.location else R.LOCATIONS
    branches = [args.branch] if args.branch else R.BRANCHES
    kg_bound = args.kappa_gamma_bound
    no_cov = args.no_covariates
    damp_max = args.seasonal_damping_max
    reset_phi = args.reset_phi_ls
    if args.out_subdir:
        out_subdir = args.out_subdir
    else:
        parts = ["m5_final"]
        if kg_bound is not None:
            parts.append(f"kgbound{kg_bound:g}")
        if no_cov:
            parts.append("nocov")
        if damp_max is not None:
            parts.append(f"damp{damp_max:g}")
        if reset_phi:
            parts.append("phireset")
        out_subdir = "_".join(parts)
    summary_name = f"{out_subdir}_summary.csv"

    rows = []
    for location in locations:
        cfg = STATION_REGISTRY[location]
        short = cfg.get("short", cfg["run_id"])
        print(f"=== {cfg.get('display', location)} ({short}) ===")
        data = None
        for branch in branches:
            done = already_done(short, branch, out_subdir)
            if done is not None:
                print(f"  [{short}/{branch}] SKIP (already completed): "
                      f"grad={done['grad_norm_inf']:.4g}, AIC={done['aic']:.2f}")
                rows.append({"short": short, "branch": branch, "m": M, "source": "resumed",
                             "aic": done["aic"], "grad_norm_inf": done["grad_norm_inf"],
                             "loglik": done["loglik"]})
                continue
            if out_subdir == "m5_final" and (short, branch) in PROMOTE:
                try:
                    rows.append(promote(short, branch, out_subdir))
                    continue
                except Exception as exc:
                    print(f"  [{short}/{branch}] promotion FAILED ({exc}); falling back to a fresh fit.")
            if data is None:
                loader = StationDataLoader.from_registry(location, precip_dir=R.PRECIP_DIR,
                                                          era5_dir=R.ERA5_DIR, enso_path=R.ENSO_PATH)
                data = loader.load_all()
            try:
                rows.append(fit_fresh(short, branch, data, out_subdir, kg_bound, no_cov, damp_max, reset_phi))
            except Exception as exc:
                print(f"  [{short}/{branch}] FAILED: {exc}")
                traceback.print_exc()
                rows.append({"short": short, "branch": branch, "status": "failed", "error": str(exc)})

    pd.DataFrame(rows).to_csv(OUT_DIR / summary_name, index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
