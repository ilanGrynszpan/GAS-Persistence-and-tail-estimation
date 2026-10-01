"""
Trigonometric-seasonal Harvey long-short experiment: progressive harmonic
count selected by AIC, per location and per branch (2026-09-03).

=============================================================================
OVERVIEW
=============================================================================

Adds models.harvey_gas.HarveyZAGASModel's new trigonometric seasonal
component (n_harmonics, see that module's docstring) on top of each
location/branch's already-established "best" Harvey long-short
configuration (the SAME winning config reports/best longshort tvps/
report.pdf already reports, resolved here from artifacts/
longshort_diagnostics_experiment/tvp_comparison/<short>/<branch>/
best_model_info.json -- not re-derived, per CLAUDE.md's "refactor rather
than duplicate").

For m = 5, 10, 15, ..., 180, 182 (182 = floor(365.25/2), the Nyquist limit
for an annual cycle on daily data), fits HarveyZAGASModel with
n_harmonics=m, WARM-STARTED from the baseline (m=0) fit's own long/short/
static/pi parameter values via initial_theta(overrides=..., static_
overrides=...) -- the new kappa_gamma/seas/seasstar slots start at their
ordinary defaults (see initial_theta's docstring), everything else starts
already-converged. AIC = 2*n_params - 2*loglik is compared against the
best AIC seen so far (starting from the m=0 baseline's own AIC, computed
from its already-saved loglik/n_params, no refit needed); the search
STOPS the first time AIC increases (greedy forward selection), per direct
instruction -- it does not necessarily explore the full grid.

ASSUMPTIONS MADE WITHOUT further confirmation (direct instruction: "make
common sense assumptions and tell me later"):
  - kappa_gamma is ONE shared scalar per dynamic parameter j (mirroring
    kappa_L_j/kappa_S_j's own per-j, not per-lag/per-harmonic, structure),
    exactly as given in the user's own recursion (kappa_gamma has no
    harmonic-index subscript in the specified equations).
  - S = 365.25 exactly as specified; m capped at floor(S/2) = 182.
  - The "standard long-short model" this compares against is the existing
    tvp_comparison winning config per location/branch (restricted /
    sigmoid_both / fixed_phiL, whichever already won there) -- the same
    baseline reports/best longshort tvps/report.pdf uses -- rather than
    re-deriving a fresh baseline from scratch.
  - Fresh AIC-search fits use the SAME restrict_persistence_upper /
    use_auxiliary_penalties / fixed_phi_L / phi_L_max / phi_S_max /
    fixed_phi_S flags as that baseline (i.e. only n_harmonics changes
    across the grid, nothing else about the persistence parameterisation).
  - PERFORMANCE (discovered via direct timing, not assumed): n_harmonics>0
    forces the plain Python recursion (no numba kernel knows this
    component -- see models/harvey_gas.py's module docstring), and a
    SINGLE likelihood evaluation on this project's ~3650-day series costs
    ~140-175ms regardless of m (dominated by T per-day GB2 special-
    function calls in pure Python, not the O(T*m) harmonic vector work,
    which is trivial by comparison). seas0/seasstar0 were made FIXED
    constants rather than free theta entries specifically to keep this
    tractable at all (see models/harvey_gas.py's module docstring -- they
    share L0/S0's already-confirmed zero-effect property, so optimising
    them was pure wasted cost). Even so, a direct BFGS timing run (m=5,
    warm-started from the baseline, kappa_gamma freshly added) measured
    scipy's own line search needing ~55 function evaluations per outer
    iteration on this landscape (not the ~n_params+1 a plain forward-
    difference gradient alone would cost) -- roughly 6.5s/iteration. A
    full run at scipy's default maxiter (1000), repeated across a
    ~37-point grid x 12 location/branch combinations, would not finish
    overnight. To fit within one overnight run while still visiting the
    full requested grid (5, 10, 15, ..., 182) until AIC increases, this
    script:
      1. runs the search itself at a REDUCED budget (SEARCH_MAXITER=60,
         polish=False, ~6.5 min/candidate) -- enough to rank candidate m
         values by AIC reliably, not necessarily to fully converge every
         one;
      2. once a location/branch's search stops, refits ONLY the winning m
         at a more thorough budget (FINAL_MAXITER=300, polish=True,
         ~32 min) for the actual reported parameter estimates/gradients/
         standard errors -- still well below scipy's own default, a
         deliberate compute-budget compromise flagged here rather than
         silently assumed.
    Numba-accelerating the harmonic recursion itself (extending
    models/_nb_gb2_filter.py's compiled kernels) would remove the need for
    this compromise entirely and is a natural follow-up, not attempted
    here given the time available.

Per docs/EXECUTION.md, this does NOT touch any existing artifact --
entirely new artifact tree, read-only access to the baseline.

=============================================================================
USAGE
=============================================================================

    python run_harmonic_seasonal_experiment.py
    python run_harmonic_seasonal_experiment.py --location "MANAUS" --branch phi_only

Output:
    artifacts/harmonic_seasonal_experiment/<short>/<branch>/m<value>/
        estimated_parameters.csv, standard_errors.csv, gradients.csv,
        hess_inv.npy, decoded_harvey_parameters.csv, metadata.json
    artifacts/harmonic_seasonal_experiment/<short>/<branch>/search_log.json
    artifacts/harmonic_seasonal_experiment/summary_<tag>.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import STATION_REGISTRY, StationDataLoader
from distributions.gb2_log_link import GB2LogLink
from distributions.gb2_phi_only import GB2LogLinkPhiOnly
from pi_dynamics.factory import make_pi_dynamics
from models.harvey_gas import HarveyZAGASModel

PRECIP_DIR = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR   = ROOT / "data" / "input" / "ERA5"
ENSO_PATH  = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
DIAG_DIR   = ARTIFACTS_DIR / "longshort_diagnostics_experiment"
TVP_DIR    = DIAG_DIR / "tvp_comparison"
OUT_DIR    = ARTIFACTS_DIR / "harmonic_seasonal_experiment"

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]
BRANCHES = ["phi_only", "phi_xi"]
SEASONAL_PERIOD = 365.25
M_MAX = 182  # floor(365.25 / 2)
M_GRID = list(range(5, M_MAX, 5)) + [M_MAX]
SEARCH_MAXITER = 60   # phase 1: rank candidate m by AIC (see module docstring's PERFORMANCE note --
                      # measured ~6.5s/BFGS-iteration on this warm-started landscape: scipy's line
                      # search needs ~55 evals/iteration here, not the ~n_params+1 a plain forward-
                      # difference gradient alone would cost, so even the SEARCH phase is not cheap)
FINAL_MAXITER  = 300  # phase 2: refit of the winning m only, more thorough than the search phase
                      # but capped well below scipy's own default (1000) to keep 12 location/branch
                      # combinations' worth of final refits tractable in one overnight run


def _find_covariate(data: dict, col_name: str):
    """Locate col_name in whichever covariate block of data actually holds
    it, returning (block_name, column_index). Generic -- does not assume a
    fixed block name, unlike run_longshort_diagnostics.py's hardcoded
    LONG_BLOCK/SHORT_SOURCE_BLOCK constants, so this works even if a
    future location's winning config used a different source block."""
    for block, cols in data["covariate_col_names"].items():
        if col_name in cols:
            return block, cols.index(col_name)
    raise KeyError(f"Covariate '{col_name}' not found in any loaded covariate block.")


def _build_X(data: dict, names: list, split: str) -> np.ndarray:
    key = f"covariate_blocks_{split}"
    T = len(data["y_train"] if split == "train" else data["y_test"])
    if not names:
        return np.zeros((T, 0))
    cols = []
    for name in names:
        block, idx = _find_covariate(data, name)
        cols.append(data[key][block][:, idx])
    return np.column_stack(cols)


def resolve_baseline(short: str, branch: str) -> dict:
    info_path = TVP_DIR / short / branch / "best_model_info.json"
    info = json.loads(info_path.read_text())
    run_id, harvey_config = info["run_id"], info["harvey_config"]
    model_dir = DIAG_DIR / run_id / branch / harvey_config
    meta = json.loads((model_dir / "metadata.json").read_text())
    theta = pd.read_csv(model_dir / "estimated_parameters.csv")["value"].to_numpy(dtype=float)
    return {"run_id": run_id, "harvey_config": harvey_config, "meta": meta, "theta": theta,
            "model_dir": model_dir}


def build_model(meta: dict, n_harmonics: int, kappa_gamma_bound: Optional[float] = None,
                 long_names_override: Optional[list] = None,
                 short_names_override: Optional[list] = None,
                 seasonal_damping_max: Optional[float] = None) -> HarveyZAGASModel:
    tv_names = meta["tv_names"]
    dist = GB2LogLink() if len(tv_names) > 1 else GB2LogLinkPhiOnly()
    pi_dyn = make_pi_dynamics("ar_logistic", seasonal="daily")
    long_names = meta["long_names"] if long_names_override is None else long_names_override
    short_names = meta["short_names"] if short_names_override is None else short_names_override
    return HarveyZAGASModel(
        distribution=dist, pi_dynamics=pi_dyn, seasonal="daily",
        long_names=long_names, short_names=short_names,
        harvey_target=meta["harvey_target"], scaling=meta["scaling"],
        restrict_persistence_upper=meta["restrict_persistence_upper"],
        use_auxiliary_penalties=meta["use_auxiliary_penalties"],
        fixed_phi_L=meta.get("fixed_phi_L"),
        phi_L_max=meta.get("phi_L_max"),
        phi_S_max=meta.get("phi_S_max"),
        fixed_phi_S=meta.get("fixed_phi_S"),
        raw_z_ridge=meta.get("raw_z_ridge") or 0.0,  # `or` guards against a saved JSON null, not just a missing key
        n_harmonics=n_harmonics, seasonal_period=SEASONAL_PERIOD,
        kappa_gamma_bound=kappa_gamma_bound,
        seasonal_damping_max=seasonal_damping_max,
    )


def warm_start_overrides(baseline_model: HarveyZAGASModel, baseline_theta: np.ndarray,
                          include_covariates: bool = True, include_phi_ls: bool = True):
    st_theta, pi_theta, hv_theta = baseline_model._split_theta(baseline_theta)
    sp = baseline_model._decode_static(st_theta)
    hvd = baseline_model._decode_harvey(hv_theta)
    keys = ("omega", "f0", "kappa_L", "L0", "kappa_S", "S0")
    if include_phi_ls:
        keys = keys + ("phi_L", "phi_S")
    # else: phi_L/phi_S are DROPPED from the override dict entirely, so
    # initial_theta() falls back to its own defaults (0.95, 0.50 --
    # deliberately well-separated) instead of the baseline's own fitted
    # values. Used when the baseline itself shows a Long/Short persistence
    # collapse (phi_L~=phi_S): warm-starting FROM that collapse just seeds
    # the seasonal fit in the same bad basin.
    if include_covariates:
        keys = keys + ("Gamma_L", "Gamma_S")
    overrides = {}
    for j, hj in hvd.items():
        overrides[j] = {k: hj[k] for k in keys}
    return overrides, sp


def fit_one(short: str, branch: str, m: int, baseline: dict, data: dict,
            maxiter: int, polish: bool, out_subdir: str,
            kappa_gamma_bound: Optional[float] = None,
            long_names_override: Optional[list] = None,
            short_names_override: Optional[list] = None,
            seasonal_damping_max: Optional[float] = None,
            reset_phi_ls: bool = False) -> dict:
    meta = baseline["meta"]
    model = build_model(meta, m, kappa_gamma_bound=kappa_gamma_bound,
                         long_names_override=long_names_override,
                         short_names_override=short_names_override,
                         seasonal_damping_max=seasonal_damping_max)
    baseline_model = build_model(meta, 0)
    # Covariate coefficients (Gamma_L/Gamma_S) can't be warm-started when
    # the covariate set itself has been overridden (e.g. removed entirely
    # -- the baseline's own Gamma_L/Gamma_S has the WRONG length, or no
    # correspondence at all, for the new model); initial_theta() then
    # falls back to its own default (0.0) at whatever length the new
    # model actually needs.
    covariates_changed = (long_names_override is not None) or (short_names_override is not None)
    overrides, static_overrides = warm_start_overrides(
        baseline_model, baseline["theta"], include_covariates=not covariates_changed,
        include_phi_ls=not reset_phi_ls)

    y_train = data["y_train"]
    long_names  = meta["long_names"]  if long_names_override  is None else long_names_override
    short_names = meta["short_names"] if short_names_override is None else short_names_override
    X_long_train  = _build_X(data, long_names, "train")
    X_short_train = _build_X(data, short_names, "train")

    theta0 = model.initial_theta(y_train, overrides=overrides, static_overrides=static_overrides)

    t0 = time.time()
    result = model.fit(y_train, X_long_train, X_short_train, theta0=theta0,
                        options={"maxiter": maxiter, "gtol": 1e-3}, polish=polish)
    runtime = time.time() - t0

    out_dir = OUT_DIR / short / branch / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_result(result, out_dir)
    saved_meta = json.loads((out_dir / "metadata.json").read_text())

    return {"m": m, "aic": saved_meta["aic"], "loglik": saved_meta["loglik"],
            "n_params": saved_meta["n_params"], "grad_norm_inf": saved_meta["grad_norm_inf"],
            "validity": saved_meta["validity"], "success": saved_meta["success"],
            "n_iter": saved_meta["n_iter"], "runtime_s": runtime}


def run_search(short: str, branch: str, data: dict) -> dict:
    baseline = resolve_baseline(short, branch)
    baseline_meta = baseline["meta"]
    baseline_aic = 2.0 * baseline_meta["n_params"] - 2.0 * baseline_meta["loglik"]
    print(f"  Baseline ({baseline['harvey_config']}): AIC={baseline_aic:.2f}, "
          f"n_params={baseline_meta['n_params']}, loglik={baseline_meta['loglik']:.2f}")

    log = {"short": short, "branch": branch, "baseline": {
        "run_id": baseline["run_id"], "harvey_config": baseline["harvey_config"],
        "aic": baseline_aic, "n_params": baseline_meta["n_params"], "loglik": baseline_meta["loglik"],
    }, "grid": []}

    # Phase 1: reduced-budget search to RANK candidate m by AIC (module
    # docstring's PERFORMANCE note -- a single eval costs ~175ms regardless
    # of m, so a full maxiter=2000+polish run per grid point is not
    # tractable overnight across a ~37-point grid x 12 combinations).
    best_aic = baseline_aic
    best_m = 0
    for m in M_GRID:
        t0 = time.time()
        try:
            out = fit_one(short, branch, m, baseline, data,
                          maxiter=SEARCH_MAXITER, polish=False, out_subdir=f"search_m{m}")
        except Exception as exc:
            print(f"    m={m}: FAILED: {exc}")
            traceback.print_exc()
            log["grid"].append({"m": m, "status": "failed", "error": str(exc)})
            break
        print(f"    m={m}: AIC={out['aic']:.2f} (best so far {best_aic:.2f}) "
              f"loglik={out['loglik']:.2f} grad={out['grad_norm_inf']:.4g} "
              f"({time.time()-t0:.1f}s)")
        log["grid"].append({"status": "ok", **out})
        if out["aic"] < best_aic:
            best_aic = out["aic"]
            best_m = m
        else:
            print(f"    AIC increased at m={m} (best so far {best_aic:.2f} at m={best_m}) -- stopping search.")
            break

    log["best_m"] = best_m
    log["best_aic_search_phase"] = best_aic

    # Phase 2: ONE full-budget refit of the winning m only (skipped if the
    # baseline itself won, i.e. no harmonic count improved on it).
    if best_m > 0:
        print(f"  Refitting winner m={best_m} at full budget (maxiter={FINAL_MAXITER}, polish=True)...")
        t0 = time.time()
        final = fit_one(short, branch, best_m, baseline, data,
                         maxiter=FINAL_MAXITER, polish=True, out_subdir="winner")
        print(f"    winner m={best_m}: AIC={final['aic']:.2f} loglik={final['loglik']:.2f} "
              f"grad={final['grad_norm_inf']:.4g}  ({time.time()-t0:.1f}s)")
        log["final_refit"] = final
    else:
        print("  No harmonic count improved on the baseline -- no winner refit.")
        log["final_refit"] = None

    (OUT_DIR / short / branch / "search_log.json").write_text(json.dumps(log, indent=2, default=float))
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", default=None)
    parser.add_argument("--branch", default=None, choices=BRANCHES)
    args = parser.parse_args()
    locations = [args.location] if args.location else LOCATIONS
    branches = [args.branch] if args.branch else BRANCHES

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for location in locations:
        cfg = STATION_REGISTRY[location]
        short = cfg.get("short", cfg["run_id"])
        print(f"=== {cfg.get('display', location)} ({short}) ===")
        loader = StationDataLoader.from_registry(location, precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, enso_path=ENSO_PATH)
        data = loader.load_all()

        for branch in branches:
            print(f" -- branch: {branch}")
            try:
                log = run_search(short, branch, data)
                final_aic = log["final_refit"]["aic"] if log["final_refit"] else log["baseline"]["aic"]
                summary_rows.append({"status": "ok", "short": short, "branch": branch,
                                      "best_m": log["best_m"], "best_aic_search_phase": log["best_aic_search_phase"],
                                      "final_aic": final_aic, "baseline_aic": log["baseline"]["aic"]})
            except Exception as exc:
                print(f"  FAILED: {exc}")
                traceback.print_exc()
                summary_rows.append({"status": "failed", "short": short, "branch": branch, "error": str(exc)})

    tag = locations[0].split()[0] if len(locations) == 1 else "all"
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / f"summary_{tag}.csv", index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
