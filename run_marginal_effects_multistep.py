"""
Computes multi-step-ahead marginal effects (marginal_effects.multistep) for
every selected model, at horizons {1,2,5,10,30,60,90}, from the single
forecast origin at the end of the in-sample period, via paired common-
random-numbers Monte Carlo simulation -- 5 predictions per location (1
baseline + DPT/AT/ENSO->El Nino/ENSO->La Nina), and caches results to
artifacts/marginal_effects/multistep/.

Per docs/EXECUTION.md ("DO NOT RUN ANY CODE YOU BUILT BEFORE I AUTHORIZE"),
this script is NOT executed automatically -- see the accompanying chat
message for the code-review request.

=============================================================================
COST
=============================================================================

Measured this session: ~30-45s per simulation (CovZAGASModel/HarveyZAGASModel,
n_paths=10,000, h_max=90). 5 simulations per model (1 baseline shared by all
4 scenarios + 4 scenario draws) x 22 available models ~= 110 simulations
~= 110 x ~35s ~= 3,850s (~64 min) single process, a few minutes with modest
parallelism (default here: one task per model, 6 workers).

=============================================================================
OUTPUTS
=============================================================================

  artifacts/marginal_effects/multistep/<short>_<branch>_<family>.csv
      one row per (scenario, horizon): baseline_mean/baseline_var (the
      "no change" projection) and, for the 4 named scenarios, delta_mu/
      delta_mu_se/95% CI, delta_var/delta_var_se/95% CI
      (marginal_effects.multistep.ScenarioEffect).
  artifacts/marginal_effects/multistep/convergence_check.csv
      Sec.3.7's convergence check: the single origin re-simulated at 3x the
      paths under an independent seed, per available model x scenario,
      flagging any (scenario, horizon) whose base-run MC SE is large
      relative to its estimated effect. Cached per (location, branch,
      family, scenario) under artifacts/marginal_effects/multistep/convergence/
      before being aggregated, same resume/parallel treatment as the main
      simulation phase below (2026-09-17 fix: this phase originally ran
      single-threaded/sequential AFTER the parallel pool closed -- on a
      32-worker run that pool phase finished fine, but the machine was
      already under enough memory/CPU pressure that the sequential
      convergence-check phase immediately following it crashed the whole
      process with no output; it is now just as parallel and resumable as
      the main phase).

Every task -- one per available (location, branch, family) for the main
simulation, one per available (location, branch, family, scenario) for the
convergence check -- is fully independent, safe to parallelize, and
resume-safe (already-written output files are skipped unless --force).

=============================================================================
ALTERNATIVE (NON-PRIMARY) CANDIDATES
=============================================================================

User request 2026-09-17: also report the min/max marginal effect across
every OTHER grad<1, variable-bearing candidate in the same (location,
branch, family) cell -- not only the one officially selected. This adds
main-simulation-only tasks (no convergence check -- that QC step is only
for the primary/selected models) for every
marginal_effects.model_selection.all_valid_models() entry with
primary=False, writing to
artifacts/marginal_effects/multistep/alternatives/<short>_<branch>_<family>__<stage>__<cfg>.csv.
There are 97 such alternatives across the 24 cells (vs. 22 primaries) --
pass --skip-alternatives to omit this pass.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marginal_effects.model_selection import select_all, all_valid_models_all_cells
from marginal_effects.model_loader import load
from marginal_effects.multistep import (
    DEFAULT_HORIZONS, scenario_effects_for_model, convergence_check_for_model,
)

OUT = ROOT / "artifacts" / "marginal_effects" / "multistep"
CONV_DIR = OUT / "convergence"
ALT_OUT = OUT / "alternatives"


def _task(args):
    sel, n_paths, seed, n_batches, force, out_path, tag = args
    if out_path.exists() and not force:
        return ("skip", tag)
    try:
        lm = load(sel)
        t0 = time.time()
        df = scenario_effects_for_model(lm, DEFAULT_HORIZONS, n_paths, seed, n_batches)
        df.to_csv(out_path, index=False)
        return ("done", f"{tag}: 5 predictions x {len(DEFAULT_HORIZONS)} horizons in {time.time()-t0:.0f}s")
    except Exception:
        return ("err", f"{tag}\n{traceback.format_exc()}")


def _convergence_task(args):
    key, sel, n_paths_base, n_batches, force = args
    short, branch, family = key
    tag = f"{short}_{branch}_{family}"
    out_path = CONV_DIR / f"{tag}.csv"
    if out_path.exists() and not force:
        return ("skip", tag)
    try:
        lm = load(sel)
        t0 = time.time()
        df = convergence_check_for_model(lm, DEFAULT_HORIZONS,
                                          n_paths_base=n_paths_base, n_paths_boost=3 * n_paths_base,
                                          n_batches=n_batches)
        df.insert(0, "family", family)
        df.insert(0, "branch", branch)
        df.insert(0, "location", short)
        df.to_csv(out_path, index=False)
        return ("done", f"{tag}: {time.time()-t0:.0f}s")
    except Exception:
        return ("err", f"{tag}\n{traceback.format_exc()}")


def _run_pool(tasks, fn, workers, label):
    """Shared parallel-or-sequential runner for both phases below, each
    task independent and resume-safe (docs/OPTIMIZATION.md/EXECUTION.md
    multiprocessing rules)."""
    t0 = time.time()
    tally = {"done": 0, "skip": 0, "err": 0}
    if workers <= 1:
        for a in tasks:
            st, msg = fn(a)
            tally[st] += 1
            print(f"  ({st}) {msg}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(fn, a) for a in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                st, msg = fut.result()
                tally[st] += 1
                print(f"  [{i}/{len(tasks)}] ({st}) {msg.splitlines()[0]}", flush=True)
                if st == "err":
                    print(msg)
    print(f"[{label}] done in {time.time()-t0:.0f}s  {tally}")
    return tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-convergence-check", action="store_true")
    ap.add_argument("--skip-alternatives", action="store_true",
                     help="skip the ~97 non-primary grad<1 candidates used for the min/max robustness range")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    selection = select_all()
    available = [(k, s) for k, s in sorted(selection.items()) if s.available]
    print(f"{len(available)} available primary models, --n-paths={args.n_paths} --workers={args.workers}")

    tasks = [(sel, args.n_paths, args.seed, args.n_batches, args.force,
              OUT / f"{short}_{branch}_{family}.csv", f"{short}_{branch}_{family}")
             for (short, branch, family), sel in available]
    _run_pool(tasks, _task, args.workers, "multistep simulation (primary)")

    if not args.skip_convergence_check:
        CONV_DIR.mkdir(parents=True, exist_ok=True)
        conv_tasks = [(key, sel, args.n_paths, args.n_batches, args.force) for key, sel in available]
        _run_pool(conv_tasks, _convergence_task, args.workers, "convergence check")

        frames = [pd.read_csv(p) for p in sorted(CONV_DIR.glob("*.csv"))]
        cc = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        cc.to_csv(OUT / "convergence_check.csv", index=False)
        n_warn = int((cc.get("warn_mu", pd.Series(dtype=bool))
                      | cc.get("warn_var", pd.Series(dtype=bool))).sum()) if len(cc) else 0
        print(f"wrote {OUT / 'convergence_check.csv'} ({n_warn} (scenario,horizon) cells flagged)")

    if not args.skip_alternatives:
        ALT_OUT.mkdir(parents=True, exist_ok=True)
        all_models = all_valid_models_all_cells()
        alternatives = [ms for models in all_models.values() for ms in models if not ms.primary]
        alt_tasks = [(sel, args.n_paths, args.seed, args.n_batches, args.force,
                      ALT_OUT / f"{sel.short}_{sel.branch}_{sel.family}__{sel.stage}__{sel.cfg}.csv",
                      f"{sel.short}_{sel.branch}_{sel.family}__{sel.stage}__{sel.cfg}")
                     for sel in alternatives]
        print(f"\n{len(alt_tasks)} alternative (non-primary) candidates")
        _run_pool(alt_tasks, _task, args.workers, "multistep simulation (alternatives)")


if __name__ == "__main__":
    main()
