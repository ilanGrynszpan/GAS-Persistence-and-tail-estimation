"""
Sequential and parallel experiment runner for the thesis pipeline.

=============================================================================
PIPELINE OVERVIEW  (MODELS.md §26, EXECUTION.md §4-7)
=============================================================================

Stage 1 — Standard GAS, no covariates
    Models: {phi-only, phi+xi} × {short, seasonal lags} × {unit, diag_FI, full_FI}
    Objective: select best no-covariate specification

Stage 2 — Standard GAS with weather covariates
    Starting point: Stage-1 winner (scaling + TV params fixed)
    Models: sequential — dew_short → dew_seasonal → dewtemp_short → dewtemp_seasonal
    Each model warm-starts from the previous model's GAS+pi block

Stage 3 — Harvey long-short
    Short component: Stage-2 best weather block
    Long component: {none, ENSO-90d, ENSO-90d+30d, ENSO-90d+30d+daily}
    All 4 variants are independent — sequential for simplicity

=============================================================================
EXECUTION PROTOCOL  (EXECUTION.md, OPTIMIZATION.md)
=============================================================================

For every model:
  1. Check artifact cache → skip if valid artifact exists
  2. Fit BFGS with soft penalties (unbounded, per OPTIMIZATION.md §2-3)
  3. Polish step (restart from final params)
  4. Classify convergence per OPTIMIZATION.md §11
  5. Compute OOS diagnostics (1-step-ahead rolling)
  6. Checkpoint immediately (don't wait for stage end)
  7. Continue even if one model fails (failure isolation)

=============================================================================
ARTIFACT STRUCTURE  (EXECUTION.md §16)
=============================================================================

artifacts/
  <run_id>/              e.g.  "run_20260701_bh"
    stage1/
      <model_id>/
        metadata.json        — all scalar results + validity
        estimated_parameters.csv
        standard_errors.csv
        gradients.csv
        hess_inv.npy
        paths.npz            — filtered state arrays
        metrics_core.json    — OOS metrics
    stage2/ ...
    stage3/ ...
    stage_winners.json   — best model at each stage
    execution_log.jsonl  — one JSON line per model

=============================================================================
PARALLELISM  (EXECUTION.md §10-11)
=============================================================================

Parallel execution follows the fork-join pattern:
  * Stage 1 and Stage 3 contain independent models → parallelisable.
  * Stage 2 has a warm-start chain → must remain sequential.
  * Parallelism is optional (parallel=False by default) and controlled by `workers`.
  * Numerical results never change — only scheduling changes.

Multiprocessing uses concurrent.futures.ProcessPoolExecutor with the default
`spawn` start method on Windows (safe for numba, no fork hazards).  Each worker
process imports the framework fresh, loads numba's disk cache (~0.1 s), then
runs one model to completion and returns the result dict.  All file I/O that
could conflict across workers (execution_log.jsonl) is deferred to the main
process, which writes atomically after collecting all futures.
"""

from __future__ import annotations

import concurrent.futures as _cf
import json
import logging
import multiprocessing as _mp
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str = "pipeline") -> logging.Logger:
    """
    Return a logger for pipeline messages. If the caller (e.g.
    run_all_locations.py) has already configured the root logger via
    logging.basicConfig(), this logger's messages propagate there and are
    picked up by the caller's handlers (console + file) -- no extra handler
    is attached, since one would otherwise print every message twice. When
    used standalone (root not configured), a StreamHandler is attached so
    output is still visible.
    """
    logger = logging.getLogger(name)
    if not logger.handlers and not logging.getLogger().handlers:
        fmt = logging.Formatter(
            "%(asctime)s  [%(levelname)s]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.setLevel(logging.INFO)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Artifact helpers
# ─────────────────────────────────────────────────────────────────────────────

def _artifact_exists(cache_dir: Path, model_id: str) -> bool:
    """True if a valid completed artifact exists for this model_id."""
    p = cache_dir / model_id / "metadata.json"
    if not p.exists():
        return False
    try:
        meta = json.loads(p.read_text())
        return meta.get("validity", "") in ("valid_converged", "valid_with_warning")
    except Exception:
        return False


def _load_meta(cache_dir: Path, model_id: str) -> dict:
    p = cache_dir / model_id / "metadata.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _load_oos_metrics(cache_dir: Path, model_id: str) -> dict:
    p = cache_dir / model_id / "metrics_core.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_paths(paths: dict, cache_dir: Path, model_id: str) -> None:
    """Save OOS filtered state arrays as compressed .npz."""
    out_dir = cache_dir / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dict = {}
    for k, v in paths.items():
        if isinstance(v, np.ndarray):
            save_dict[k] = v
        elif isinstance(v, (list, tuple)):
            try:
                save_dict[k] = np.array(v, dtype=float)
            except (ValueError, TypeError):
                pass
    if save_dict:
        np.savez_compressed(out_dir / "paths.npz", **save_dict)


def _log_event(log_path: Path, event: dict) -> None:
    with log_path.open("a") as f:
        f.write(json.dumps(event, default=str) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# OOS predictive metrics (MC approximation)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_oos_metrics(
    model,
    oos_paths: dict,
    y_test: np.ndarray,
    quantiles: Tuple = (0.5, 0.75, 0.90, 0.95, 0.99),
    n_draws: int = 500,
) -> dict:
    """
    MC approximation of CRPS, quantile scores, and summary statistics.

    oos_paths must contain:
        pi_oos   : (T_test,) probability of positive obs
        f_arr_oos: (T_test, n_tv) filtered TV param values
        tv_names : list of TV param names
        static   : dict of static distribution params
    """
    rng = np.random.default_rng(42)
    n   = len(y_test)
    n_q = len(quantiles)

    crps_vals = np.zeros(n)
    qs_vals   = np.zeros((n, n_q))
    pit_vals  = np.zeros(n)
    pred_mean = np.zeros(n)

    sp       = oos_paths["static"]
    tv_names = oos_paths["tv_names"]
    has_ppf  = hasattr(model.dist, "ppf")
    has_cdf  = hasattr(model.dist, "cdf")

    for i in range(n):
        pi_t = float(oos_paths["pi_oos"][i])
        call = {name: float(oos_paths["f_arr_oos"][i, j])
                for j, name in enumerate(tv_names)}
        call.update(sp)

        # Predictive draws from ZA-mixture
        is_pos = rng.binomial(1, pi_t, size=n_draws)
        draws  = np.zeros(n_draws)
        n_pos  = int(is_pos.sum())
        if n_pos > 0 and has_ppf:
            try:
                draws[is_pos == 1] = model.dist.ppf(
                    rng.uniform(1e-8, 1 - 1e-8, size=n_pos), **call
                )
            except Exception:
                pass

        obs = float(y_test[i])

        # CRPS via energy score
        crps_vals[i] = float(
            np.mean(np.abs(draws - obs))
            - 0.5 * np.mean(np.abs(np.subtract.outer(draws, draws)))
        )

        # Quantile predictions and scores
        zero_mass = max(0.0, 1.0 - pi_t)
        for k, q in enumerate(quantiles):
            if q <= zero_mass:
                q_pred = 0.0
            elif has_ppf:
                q_pos = np.clip((q - zero_mass) / max(pi_t, 1e-12), 1e-12, 1 - 1e-12)
                try:
                    q_pred = float(model.dist.ppf(q_pos, **call))
                except Exception:
                    q_pred = float(np.quantile(draws, q))
            else:
                q_pred = float(np.quantile(draws, q))
            r = obs - q_pred
            qs_vals[i, k] = float(r * (q - float(r < 0)))

        pred_mean[i] = float(np.mean(draws))

        # PIT value F(y_t)
        if obs > 0 and has_cdf:
            try:
                pit_vals[i] = float(zero_mass + pi_t * model.dist.cdf(obs, **call))
            except Exception:
                pit_vals[i] = float(np.mean(draws <= obs))
        elif obs == 0:
            pit_vals[i] = zero_mass

    wet_mask = y_test > 0
    return {
        "crps_mean":     float(np.mean(crps_vals)),
        "crps_std":      float(np.std(crps_vals)),
        "crps_wet_mean": float(np.mean(crps_vals[wet_mask])) if wet_mask.any() else float("nan"),
        "qs_mean":       {f"q{int(q*100):02d}": float(np.mean(qs_vals[:, k]))
                         for k, q in enumerate(quantiles)},
        "rmse":          float(np.sqrt(np.mean((pred_mean - y_test) ** 2))),
        "mae":           float(np.mean(np.abs(pred_mean - y_test))),
        "pit_mean":      float(np.mean(pit_vals)),
        "pit_std":       float(np.std(pit_vals)),
        "n_obs":         int(n),
        "n_wet":         int(wet_mask.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-model runner
# ─────────────────────────────────────────────────────────────────────────────

def run_single_model(
    model_id:    str,
    model,
    fit_kwargs:  dict,
    oos_kwargs:  dict,
    y_test:      np.ndarray,
    cache_dir:   Path,
    log_path:    Path,
    logger:      logging.Logger,
    force_rerun: bool = False,
) -> dict:
    """
    Fit one model, compute OOS diagnostics, checkpoint, and return metadata.

    Parameters
    ----------
    model      : model with .fit(), .save_result(), .simulate_oos()
    fit_kwargs : passed to model.fit()
    oos_kwargs : passed to model.simulate_oos() WITHOUT 'theta';
                 'theta' is filled from the fit result automatically
    y_test     : test observations for OOS metric computation
    cache_dir  : stage artifact directory

    Returns
    -------
    dict: {model_id, status, validity, loglik, n_params, oos_metrics, result}
    """
    model_dir = cache_dir / model_id

    # ── Artifact reuse ───────────────────────────────────────────────────────
    if not force_rerun and _artifact_exists(cache_dir, model_id):
        logger.info(f"[SKIP]  {model_id}  — valid artifact found")
        meta        = _load_meta(cache_dir, model_id)
        oos_metrics = _load_oos_metrics(cache_dir, model_id)
        return {
            "model_id":    model_id,
            "status":      "cached",
            "validity":    meta.get("validity"),
            "loglik":      meta.get("loglik"),
            "n_params":    meta.get("n_params", 0),
            "oos_metrics": oos_metrics,
            "result":      None,
        }

    logger.info(f"[START] {model_id}")
    t_start = time.time()

    # ── Fit ──────────────────────────────────────────────────────────────────
    try:
        result = model.fit(**fit_kwargs)
    except Exception as exc:
        elapsed = time.time() - t_start
        tb_str  = traceback.format_exc()
        msg     = f"fit exception: {exc}"
        logger.error(f"[FAIL]  {model_id}  {msg}")
        logger.error(f"[TRACEBACK]\n{tb_str}")
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "metadata.json").write_text(json.dumps({
            "model_id":   model_id,
            "validity":   "failed",
            "failure_reason": msg,
            "runtime_s":  elapsed,
        }, indent=2))
        _log_event(log_path, {"model_id": model_id, "event": "exception", "msg": msg})
        return {
            "model_id":    model_id,
            "status":      "failed",
            "validity":    "failed",
            "loglik":      float("nan"),
            "n_params":    0,
            "oos_metrics": {},
            "result":      None,
        }

    elapsed_fit = time.time() - t_start

    # ── Save core result ──────────────────────────────────────────────────────
    try:
        model.save_result(result, model_dir)
    except Exception as exc:
        logger.warning(f"[WARN]  {model_id}  save_result failed: {exc}")

    validity = result.get("validity", "unknown")
    ll       = float(result.get("loglik", float("nan")))
    n_p      = int(getattr(model, "n_params", 0))

    # ── OOS evaluation ────────────────────────────────────────────────────────
    oos_metrics = {}
    try:
        oos_kwargs_filled            = dict(oos_kwargs)
        oos_kwargs_filled["theta"]   = result["theta"]
        oos_paths                    = model.simulate_oos(**oos_kwargs_filled)
        oos_metrics                  = _compute_oos_metrics(model, oos_paths, y_test)
        _save_paths(oos_paths, cache_dir, model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "metrics_core.json").write_text(
            json.dumps(oos_metrics, indent=2, default=str)
        )
        # Append OOS summary to metadata
        meta_p = model_dir / "metadata.json"
        if meta_p.exists():
            meta           = json.loads(meta_p.read_text())
            meta["crps_mean"] = oos_metrics.get("crps_mean")
            meta["rmse"]      = oos_metrics.get("rmse")
            meta_p.write_text(json.dumps(meta, indent=2))
    except Exception as exc:
        logger.warning(f"[WARN]  {model_id}  OOS evaluation failed: {exc}")

    logger.info(
        f"[DONE]  {model_id}  validity={validity}  loglik={ll:.2f}  "
        f"crps={oos_metrics.get('crps_mean', float('nan')):.4f}  "
        f"runtime={elapsed_fit:.0f}s"
    )
    _log_event(log_path, {
        "model_id":  model_id,
        "event":     "done",
        "validity":  validity,
        "loglik":    ll,
        "crps_mean": oos_metrics.get("crps_mean"),
        "runtime_s": elapsed_fit,
    })

    return {
        "model_id":    model_id,
        "status":      "fitted",
        "validity":    validity,
        "loglik":      ll,
        "n_params":    n_p,
        "oos_metrics": oos_metrics,
        "result":      result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parallel stage runner  (fork-join pattern, EXECUTION.md §10-11)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_fn(
    model_id:    str,
    model,
    fit_kwargs:  dict,
    oos_kwargs:  dict,
    y_test:      np.ndarray,
    cache_dir:   Path,
    force_rerun: bool,
) -> dict:
    """
    Top-level worker function called inside a subprocess.

    Creates a minimal stdout logger (no file handles to inherit) and
    delegates to run_single_model.  The `_log_events` key in the returned
    dict carries JSONL records that the main process writes to the shared
    execution log.
    """
    # Each worker process imports fresh, numba loads its disk cache
    worker_logger = get_logger(f"worker.{model_id}")

    # Intercept _log_event by collecting events locally
    collected_events: list = []
    _orig_log = globals().get("_log_event")

    # We call run_single_model with a dummy log_path because we collect events
    # in the return dict; the dummy path is never opened if no event occurs.
    dummy_log = cache_dir / "_worker_log_unused.jsonl"
    result = run_single_model(
        model_id=model_id,
        model=model,
        fit_kwargs=fit_kwargs,
        oos_kwargs=oos_kwargs,
        y_test=y_test,
        cache_dir=cache_dir,
        log_path=dummy_log,      # events NOT flushed here
        logger=worker_logger,
        force_rerun=force_rerun,
    )
    # Inject log event for the main process to write
    result["_log_events"] = [{
        "model_id":  model_id,
        "event":     result.get("status", "unknown"),
        "validity":  result.get("validity"),
        "loglik":    result.get("loglik"),
        "crps_mean": result.get("oos_metrics", {}).get("crps_mean"),
        "runtime_s": result.get("result", {}).get("runtime_s")
                     if isinstance(result.get("result"), dict) else None,
    }]
    return result


def run_stage_parallel(
    specs:         List[Tuple[str, Any, dict, dict]],
    y_test:        np.ndarray,
    stage_dir:     Path,
    log_path:      Path,
    logger:        logging.Logger,
    workers:       int   = 4,
    force_rerun:   bool  = False,
    model_timeout: int | None = None,  # per-model wall-clock timeout in seconds
) -> List[dict]:
    """
    Run a list of independent model specs in parallel using a ProcessPoolExecutor.

    Each spec is (model_id, model, fit_kwargs, oos_kwargs).
    Returns results in the SAME ORDER as specs (not completion order) so that
    downstream winner selection is deterministic.

    Only models that are NOT already cached are submitted to the pool.
    Cached models are resolved immediately in the main process.
    """
    # Separate cached from pending to avoid unnecessary subprocess spawning
    cached_results:  List[Tuple[int, dict]] = []
    pending_indices: List[int]              = []
    pending_specs:   List[Tuple]            = []

    for idx, (model_id, model, fit_kw, oos_kw) in enumerate(specs):
        if not force_rerun and _artifact_exists(stage_dir, model_id):
            meta        = _load_meta(stage_dir, model_id)
            oos_metrics = _load_oos_metrics(stage_dir, model_id)
            logger.info(f"[SKIP]  {model_id}  — valid artifact found")
            cached_results.append((idx, {
                "model_id":    model_id,
                "status":      "cached",
                "validity":    meta.get("validity"),
                "loglik":      meta.get("loglik"),
                "n_params":    meta.get("n_params", 0),
                "oos_metrics": oos_metrics,
                "result":      None,
            }))
        else:
            pending_indices.append(idx)
            pending_specs.append((model_id, model, fit_kw, oos_kw))

    results: List[Optional[dict]] = [None] * len(specs)
    for idx, res in cached_results:
        results[idx] = res

    if not pending_specs:
        return results

    effective_workers = min(workers, len(pending_specs))
    logger.info(
        f"  Submitting {len(pending_specs)} models to {effective_workers} workers "
        f"(parallel=True)"
    )

    futures_map:   Dict[_cf.Future, int]    = {}
    future_start:  Dict[_cf.Future, float]  = {}

    import time as _time

    with _cf.ProcessPoolExecutor(max_workers=effective_workers) as pool:
        for i, (model_id, model, fit_kw, oos_kw) in enumerate(pending_specs):
            fut = pool.submit(
                _worker_fn,
                model_id=model_id,
                model=model,
                fit_kwargs=fit_kw,
                oos_kwargs=oos_kw,
                y_test=y_test,
                cache_dir=stage_dir,
                force_rerun=force_rerun,
            )
            futures_map[fut] = pending_indices[i]
            future_start[fut] = _time.time()

        pending_futs = set(futures_map.keys())
        while pending_futs:
            # Wait up to 30 s for any future to complete
            done_now, pending_futs = _cf.wait(
                pending_futs, timeout=30, return_when=_cf.FIRST_COMPLETED
            )

            # Cancel any futures that have exceeded the wall-clock timeout
            if model_timeout is not None:
                now = _time.time()
                timed_out = [
                    f for f in pending_futs
                    if (now - future_start[f]) > model_timeout
                ]
                for f in timed_out:
                    f.cancel()
                    idx      = futures_map[f]
                    model_id = specs[idx][0]
                    elapsed  = int(now - future_start[f])
                    logger.warning(
                        f"[TIMEOUT] {model_id}  exceeded {model_timeout}s "
                        f"(elapsed {elapsed}s) — marking as failed"
                    )
                    res = {
                        "model_id":    model_id,
                        "status":      "timeout",
                        "validity":    "failed",
                        "loglik":      float("nan"),
                        "n_params":    0,
                        "oos_metrics": {},
                        "result":      None,
                        "_log_events": [{"model_id": model_id, "event": "timeout",
                                         "elapsed_s": elapsed}],
                    }
                    for event in res.pop("_log_events", []):
                        _log_event(log_path, event)
                    results[idx] = res
                    pending_futs.discard(f)

            # Process completed futures
            for fut in done_now:
                idx      = futures_map[fut]
                model_id = specs[idx][0]
                try:
                    res = fut.result(timeout=1)
                except _cf.TimeoutError:
                    res = {
                        "model_id": model_id, "status": "timeout",
                        "validity": "failed", "loglik": float("nan"),
                        "n_params": 0, "oos_metrics": {}, "result": None,
                        "_log_events": [],
                    }
                except Exception as exc:
                    logger.error(f"[FAIL]  {model_id}  worker exception: {exc}")
                    res = {
                        "model_id":    model_id,
                        "status":      "failed",
                        "validity":    "failed",
                        "loglik":      float("nan"),
                        "n_params":    0,
                        "oos_metrics": {},
                        "result":      None,
                        "_log_events": [{"model_id": model_id, "event": "worker_exception",
                                         "msg": str(exc)}],
                    }
                # Write log events collected inside the worker
                for event in res.pop("_log_events", []):
                    _log_event(log_path, event)

                results[idx] = res
                ll  = res.get("loglik", float("nan"))
                val = res.get("validity", "?")
                logger.info(f"[DONE]  {model_id}  validity={val}  loglik={ll:.2f}" if ll == ll
                            else f"[DONE]  {model_id}  validity={val}  loglik=NaN")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────────────────────────────────────

def select_winner(
    results:       List[dict],
    primary:       str  = "loglik",
    higher_better: bool = True,
    only_valid:    bool = True,
) -> Optional[dict]:
    """Return the best model from a list of result dicts."""
    cands = [r for r in results if r.get("status") != "failed"]
    if only_valid:
        cands = [r for r in cands
                 if r.get("validity", "failed") != "failed"]
    if not cands:
        return None

    def _key(r):
        v = r.get(primary)
        if v is None:
            v = r.get("oos_metrics", {}).get(primary)
        if v is None:
            return float("-inf") if higher_better else float("inf")
        return float(v)

    return max(cands, key=_key) if higher_better else min(cands, key=_key)


# ─────────────────────────────────────────────────────────────────────────────
# Model spec builders
# ─────────────────────────────────────────────────────────────────────────────

def build_stage1_specs(
    data:       dict,
    pi_dyn,
    dist_phi,
    dist_phixi,
) -> List[Tuple[str, Any, dict, dict]]:
    """
    Build Stage-1 model specs (12 models, no covariates).

    Returns list of (model_id, model, fit_kwargs, oos_kwargs).
    oos_kwargs does NOT include 'theta' — filled by run_single_model.
    """
    from models.za_gas_model import ZAGASModel
    from constants import GAS_SHORT_LAGS, GAS_SEASONAL_LAGS

    y_train = data["y_train"]
    y_test  = data["y_test"]

    scaling_abbr = {
        "unit":                    "unit",
        "diagonal_inverse_fisher": "diagfi",
        "inverse_fisher":          "fullfi",
    }
    specs = []
    for tv_label, dist in [("phi", dist_phi), ("phixi", dist_phixi)]:
        for lag_label, gas_lags in [
            ("short",    GAS_SHORT_LAGS["daily"]),
            ("seasonal", GAS_SEASONAL_LAGS["daily"]),
        ]:
            for scaling in ["unit", "diagonal_inverse_fisher", "inverse_fisher"]:
                model_id = f"stage1_{tv_label}_{lag_label}_{scaling_abbr[scaling]}"
                model    = ZAGASModel(
                    distribution=dist,
                    pi_dynamics=pi_dyn,
                    seasonal="daily",
                    gas_lags=gas_lags,
                    scaling=scaling,
                )
                specs.append((
                    model_id,
                    model,
                    {"y": y_train, "verbose": False},
                    {"y_train": y_train, "y_test": y_test},
                ))
    return specs


def build_stage2_specs(
    data:           dict,
    pi_dyn,
    dist,
    best_gas_lags:  list,
    best_scaling:   str,
    id_prefix:      str = "stage2_",
) -> List[Tuple[str, Any, dict, dict]]:
    """
    Build Stage-2 specs (GAS + weather covariates, sequential).

    Warm-starting is applied by the stage runner after each fit.

    id_prefix distinguishes independent tvp-set branches (e.g. "stage2_"
    for phi+xi, "stage2_phi_" for the phi-only branch, per objective 1a) so
    the two branches never collide on the same cached model_id.
    """
    from models.cov_gas_model import CovZAGASModel

    y_train = data["y_train"]
    y_test  = data["y_test"]
    bt      = data["covariate_blocks_train"]
    bv      = data["covariate_blocks_test"]
    cn      = data["covariate_col_names"]

    specs = []
    for block in ["dewpoint_short", "dewpoint_seasonal",
                  "dewtemp_short",  "dewtemp_seasonal"]:
        if block not in bt:
            continue
        model = CovZAGASModel(
            distribution=dist,
            pi_dynamics=pi_dyn,
            seasonal="daily",
            gas_lags=best_gas_lags,
            scaling=best_scaling,
            cov_names=cn[block],
        )
        specs.append((
            f"{id_prefix}{block}",
            model,
            {"y": y_train, "X": bt[block], "verbose": False},
            {"y_train": y_train, "y_test": y_test,
             "X_train": bt[block], "X_test": bv[block]},
        ))
    return specs


def build_stage3_specs(
    data:               dict,
    pi_dyn,
    dist,
    best_scaling:       str,
    best_weather_block: str,
    id_prefix:          str = "stage3_harvey_",
) -> List[Tuple[str, Any, dict, dict]]:
    """
    Build Stage-3 specs (Harvey long-short, 4 ENSO variants).

    ENSO long-component blocks are El Nino / La Nina dummy tiers (2026-07-07
    fix, MODELS.md §18/§2a) — replaces the previous continuous-anomaly
    tiers, which were built from a mislabelled raw-SST source (see
    data/station_loader.py header).

    id_prefix distinguishes independent tvp-set branches (e.g.
    "stage3_harvey_" for phi+xi, "stage3_phi_harvey_" for phi-only, per
    objective 1a) so the two branches never collide on the same cached
    model_id.
    """
    from models.harvey_gas import HarveyZAGASModel

    y_train = data["y_train"]
    y_test  = data["y_test"]
    bt      = data["covariate_blocks_train"]
    bv      = data["covariate_blocks_test"]
    cn      = data["covariate_col_names"]

    Xs_train = bt[best_weather_block]
    Xs_test  = bv[best_weather_block]
    short_cols = cn[best_weather_block]

    empty_tr = np.zeros((len(y_train), 0))
    empty_te = np.zeros((len(y_test),  0))

    enso_variants = [
        ("no_enso",           [],                       empty_tr,                    empty_te),
        ("enso_dummy_current", cn["enso_dummy_current"], bt["enso_dummy_current"],    bv["enso_dummy_current"]),
        ("enso_dummy_lag1",    cn["enso_dummy_lag1"],    bt["enso_dummy_lag1"],       bv["enso_dummy_lag1"]),
        ("enso_dummy_lag3",    cn["enso_dummy_lag3"],    bt["enso_dummy_lag3"],       bv["enso_dummy_lag3"]),
    ]

    specs = []
    for label, long_cols, Xl_tr, Xl_te in enso_variants:
        model = HarveyZAGASModel(
            distribution=dist,
            pi_dynamics=pi_dyn,
            seasonal="daily",
            long_names=long_cols,
            short_names=short_cols,
            scaling=best_scaling,
        )
        specs.append((
            f"{id_prefix}{label}",
            model,
            {"y": y_train, "X_long": Xl_tr, "X_short": Xs_train, "verbose": False},
            {"y_train": y_train, "y_test": y_test,
             "X_long_train":  Xl_tr,   "X_long_test":   Xl_te,
             "X_short_train": Xs_train, "X_short_test":  Xs_test},
        ))
    return specs


# ─────────────────────────────────────────────────────────────────────────────
# Stage runners
# ─────────────────────────────────────────────────────────────────────────────

def _save_stage_summary(path: Path, stage: str, winner: Optional[dict], results: list) -> None:
    try:
        existing = json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        existing = {}
    existing[stage] = {
        "winner":  winner.get("model_id") if winner else None,
        "loglik":  winner.get("loglik")   if winner else None,
        "crps":    winner.get("oos_metrics", {}).get("crps_mean") if winner else None,
        "models":  [
            {"model_id": r.get("model_id"),
             "validity":  r.get("validity"),
             "loglik":    r.get("loglik"),
             "crps_mean": r.get("oos_metrics", {}).get("crps_mean")}
            for r in results
        ],
    }
    path.write_text(json.dumps(existing, indent=2, default=str))


def run_stage1(
    data, pi_dyn, dist_phi, dist_phixi,
    run_dir, log_path, logger,
    force_rerun=False, winners_path=None,
    parallel=False, workers=4,
) -> Tuple[Optional[dict], List[dict], Optional[dict]]:
    """
    Run all Stage-1 models (baseline GAS, no covariates).

    All 12 models are independent — they are run in parallel when
    parallel=True.  Stage 2 cannot start until Stage 1 selects a winner,
    so the fork-join barrier is this function's return.

    Returns (phixi_winner, all_results, phi_only_winner) — the two winners
    are selected independently within their own tvp set (objective 1a/1b,
    prompt.md 4d): phi-only by OOS RMSE, phi+xi by OOS CRPS. Neither
    selection mixes the two tvp sets or falls back to log-likelihood.
    """
    stage_dir = run_dir / "stage1"
    stage_dir.mkdir(exist_ok=True)
    specs = build_stage1_specs(data, pi_dyn, dist_phi, dist_phixi)
    n = len(specs)

    logger.info("=" * 60)
    logger.info(f"STAGE 1  —  Standard GAS, no covariates ({n} models)"
                + (f"  [parallel, workers={min(workers, n)}]" if parallel else "  [sequential]"))
    logger.info("=" * 60)

    if parallel:
        results = run_stage_parallel(
            specs=specs, y_test=data["y_test"],
            stage_dir=stage_dir, log_path=log_path, logger=logger,
            workers=workers, force_rerun=force_rerun,
        )
    else:
        results = []
        for model_id, model, fit_kw, oos_kw in specs:
            r = run_single_model(
                model_id=model_id, model=model,
                fit_kwargs=fit_kw, oos_kwargs=oos_kw,
                y_test=data["y_test"],
                cache_dir=stage_dir, log_path=log_path,
                logger=logger, force_rerun=force_rerun,
            )
            results.append(r)

    from pipeline.selection import select_phi_only_winner, select_phixi_winner
    results_phixi = [r for r in results if "phixi" in r.get("model_id", "")]
    results_phi   = [r for r in results if "phixi" not in r.get("model_id", "")]

    winner       = select_phixi_winner(results_phixi)
    winner_phi   = select_phi_only_winner(results_phi)

    if winner:
        logger.info(
            f"STAGE 1 (phi_xi) WINNER: {winner['model_id']}  "
            f"crps={winner.get('oos_metrics', {}).get('crps_mean', float('nan')):.4f}  "
            f"({winner.get('_tie_break', '')})"
        )
    else:
        logger.error("STAGE 1 (phi_xi): no valid model")

    if winner_phi:
        logger.info(
            f"STAGE 1 (phi_only) WINNER: {winner_phi['model_id']}  "
            f"rmse={winner_phi.get('oos_metrics', {}).get('rmse', float('nan')):.4f}  "
            f"({winner_phi.get('_tie_break', '')})"
        )
    else:
        logger.error("STAGE 1 (phi_only): no valid model")

    if winners_path:
        _save_stage_summary(winners_path, "stage1", winner, results)
        _save_stage_summary(winners_path, "stage1_phi_only", winner_phi, results_phi)

    return winner, results, winner_phi


def run_stage2(
    data, pi_dyn, dist, best_gas_lags, best_scaling,
    theta0_gas_pi,
    run_dir, log_path, logger,
    force_rerun=False, winners_path=None,
    parallel=False, workers=4,
    id_prefix="stage2_", stage_dir_name="stage2", tvp_set="phi_xi",
) -> Tuple[Optional[dict], List[dict]]:
    """
    Run Stage-2 weather-covariate models.

    When parallel=True all 4 models start from Stage-1 winner params and run
    concurrently.  When parallel=False they run sequentially with each model
    warm-starting from the previous model's GAS+pi block.

    theta0_gas_pi: GAS+pi block theta from Stage-1 winner (length n_gas+n_pi).

    id_prefix / stage_dir_name distinguish the phi-only branch (objective
    1a) from the phi+xi branch so cached artifacts never collide.
    tvp_set selects the winner criterion (pipeline.selection, prompt.md 4d):
    "phi_only" -> OOS RMSE, "phi_xi" -> OOS CRPS.
    """
    stage_dir = run_dir / stage_dir_name
    stage_dir.mkdir(exist_ok=True)
    specs = build_stage2_specs(data, pi_dyn, dist, best_gas_lags, best_scaling,
                                id_prefix=id_prefix)
    n = len(specs)

    logger.info("=" * 60)
    logger.info(f"STAGE 2  —  Standard GAS with weather covariates ({n} models)"
                + (f"  [parallel, workers={min(workers, n)}]" if parallel else "  [sequential, warm-start chain]"))
    logger.info("=" * 60)

    if parallel:
        # All models start from Stage-1 winner params; run concurrently.
        parallel_specs = []
        for model_id, model, fit_kw, oos_kw in specs:
            fit_kw = dict(fit_kw)
            if theta0_gas_pi is not None and "theta0" not in fit_kw:
                n_base = model.n_gas + model.n_pi
                fit_kw["theta0"] = np.concatenate([
                    theta0_gas_pi[:n_base],
                    np.zeros(model.n_cov_params),
                ])
            parallel_specs.append((model_id, model, fit_kw, oos_kw))

        results = run_stage_parallel(
            specs=parallel_specs, y_test=data["y_test"],
            stage_dir=stage_dir, log_path=log_path, logger=logger,
            workers=workers, force_rerun=force_rerun,
        )
    else:
        results = []
        prev_gas_pi_theta = theta0_gas_pi

        for model_id, model, fit_kw, oos_kw in specs:
            fit_kw = dict(fit_kw)

            # Warm-start from previous model's GAS+pi block
            if prev_gas_pi_theta is not None and "theta0" not in fit_kw:
                n_base = model.n_gas + model.n_pi
                if len(prev_gas_pi_theta) >= n_base:
                    fit_kw["theta0"] = np.concatenate([
                        prev_gas_pi_theta[:n_base],
                        np.zeros(model.n_cov_params),
                    ])

            r = run_single_model(
                model_id=model_id, model=model,
                fit_kwargs=fit_kw, oos_kwargs=oos_kw,
                y_test=data["y_test"],
                cache_dir=stage_dir, log_path=log_path,
                logger=logger, force_rerun=force_rerun,
            )
            results.append(r)

            # Update warm-start for next iteration
            if r.get("result") is not None:
                theta_full        = r["result"]["theta"]
                prev_gas_pi_theta = theta_full[: model.n_gas + model.n_pi]
            elif r.get("status") == "cached":
                try:
                    df = pd.read_csv(stage_dir / model_id / "estimated_parameters.csv")
                    theta_full        = df["value"].to_numpy(dtype=float)
                    prev_gas_pi_theta = theta_full[: model.n_gas + model.n_pi]
                except Exception:
                    pass

    from pipeline.selection import select_phi_only_winner, select_phixi_winner
    select_fn = select_phi_only_winner if tvp_set == "phi_only" else select_phixi_winner
    winner = select_fn(results)
    if winner:
        winner["best_weather_block"] = winner["model_id"].replace(id_prefix, "")
        key_metric = "rmse" if tvp_set == "phi_only" else "crps_mean"
        logger.info(
            f"STAGE 2 ({tvp_set}) WINNER: {winner['model_id']}  "
            f"{key_metric}={winner.get('oos_metrics', {}).get(key_metric, float('nan')):.4f}  "
            f"({winner.get('_tie_break', '')})"
        )

    # "stage2" (unqualified) stays the phi+xi branch for backward compatibility
    # with existing readers (generate_report_extended.py); the phi-only branch
    # is saved under a distinct key.
    summary_key = "stage2" if tvp_set == "phi_xi" else "stage2_phi_only"
    if winners_path:
        _save_stage_summary(winners_path, summary_key, winner, results)

    return winner, results


def run_stage3(
    data, pi_dyn, dist, best_scaling, best_weather_block,
    run_dir, log_path, logger,
    force_rerun=False, winners_path=None,
    parallel=False, workers=4,
    id_prefix="stage3_harvey_", stage_dir_name="stage3", tvp_set="phi_xi",
) -> Tuple[Optional[dict], List[dict]]:
    """
    Run Stage-3 Harvey long-short models (4 ENSO variants: no_enso plus the
    3 El Nino/La Nina dummy tiers, 2026-07-07 fix).

    All 4 ENSO variants are independent — they can be run in parallel.

    id_prefix / stage_dir_name distinguish the phi-only branch (objective
    1a) from the phi+xi branch. tvp_set selects the winner criterion
    (pipeline.selection, prompt.md 4d): "phi_only" -> OOS RMSE,
    "phi_xi" -> OOS CRPS.
    """
    stage_dir = run_dir / stage_dir_name
    stage_dir.mkdir(exist_ok=True)
    specs = build_stage3_specs(data, pi_dyn, dist, best_scaling, best_weather_block,
                                id_prefix=id_prefix)
    n = len(specs)

    logger.info("=" * 60)
    logger.info(f"STAGE 3  —  Harvey Long-Short + ENSO covariates ({n} models)"
                + (f"  [parallel, workers={min(workers, n)}]" if parallel else "  [sequential]"))
    logger.info("=" * 60)

    if parallel:
        results = run_stage_parallel(
            specs=specs, y_test=data["y_test"],
            stage_dir=stage_dir, log_path=log_path, logger=logger,
            workers=workers, force_rerun=force_rerun,
        )
    else:
        results = []
        for model_id, model, fit_kw, oos_kw in specs:
            r = run_single_model(
                model_id=model_id, model=model,
                fit_kwargs=fit_kw, oos_kwargs=oos_kw,
                y_test=data["y_test"],
                cache_dir=stage_dir, log_path=log_path,
                logger=logger, force_rerun=force_rerun,
            )
            results.append(r)

    from pipeline.selection import select_phi_only_winner, select_phixi_winner
    select_fn = select_phi_only_winner if tvp_set == "phi_only" else select_phixi_winner
    winner = select_fn(results)
    if winner:
        key_metric = "rmse" if tvp_set == "phi_only" else "crps_mean"
        logger.info(
            f"STAGE 3 ({tvp_set}) WINNER: {winner['model_id']}  "
            f"{key_metric}={winner.get('oos_metrics', {}).get(key_metric, float('nan')):.4f}  "
            f"({winner.get('_tie_break', '')})"
        )

    summary_key = "stage3" if tvp_set == "phi_xi" else "stage3_phi_only"
    if winners_path:
        _save_stage_summary(winners_path, summary_key, winner, results)

    return winner, results


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    data:          dict,
    pi_dyn,
    dist_phi,
    dist_phixi,
    artifacts_dir: "Path | str",
    run_id:        str  = "run",
    force_rerun:   bool = False,
    do_stage1:     bool = True,
    do_stage2:     bool = True,
    do_stage3:     bool = True,
    run_phi_only_branch: bool = True,
    parallel:      bool = False,
    workers:       int  = 4,
) -> dict:
    """
    Execute the full sequential thesis pipeline.

    Runs two independent tvp-set branches (objective 1a/1b, prompt.md
    2026-07-07 revision):
      - phi+xi branch (as before): winner selected by OOS CRPS.
      - phi-only branch: winner selected by OOS RMSE, continued through its
        own Stage 2 (weather covariates) and Stage 3 (Harvey ENSO dummies)
        under distinct model ids / artifact subdirectories
        (stage{2,3}_phi / stage{2,3}_phi_harvey_*) so nothing collides with
        the phi+xi branch's cached artifacts.

    Parameters
    ----------
    data          : dict from BHDataLoader.load_all()
    pi_dyn        : PiDynamics instance
    dist_phi      : Distribution with tv_param_names=["phi"]
    dist_phixi    : Distribution with tv_param_names=["phi", "xi"]
    artifacts_dir : root directory for saving artifacts
    run_id        : experiment identifier (subdirectory name)
    force_rerun   : re-estimate even when valid artifacts exist
    do_stage{N}   : toggle each stage (phi+xi branch)
    run_phi_only_branch : also run the phi-only Stage-2/3 branch
    parallel      : run independent models within a stage in parallel (default False)
    workers       : number of parallel worker processes (default 4)

    Returns
    -------
    dict with stage1_winner, stage2_winner, stage3_winner (phi+xi branch),
         stage1_phi_winner, stage2_phi_winner, stage3_phi_winner (phi-only
         branch), stage{N}_results, run_dir
    """
    artifacts_dir = Path(artifacts_dir)
    run_dir       = artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path     = run_dir / "execution_log.jsonl"
    winners_path = run_dir / "stage_winners.json"
    logger       = get_logger("pipeline")

    out = {
        "stage1_winner":      None,
        "stage2_winner":      None,
        "stage3_winner":      None,
        "stage1_phi_winner":  None,
        "stage2_phi_winner":  None,
        "stage3_phi_winner":  None,
        "stage1_results":     [],
        "stage2_results":     [],
        "stage3_results":     [],
        "stage2_phi_results": [],
        "stage3_phi_results": [],
        "run_dir":            str(run_dir),
    }

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if do_stage1:
        w1, r1, w1_phi = run_stage1(
            data=data, pi_dyn=pi_dyn,
            dist_phi=dist_phi, dist_phixi=dist_phixi,
            run_dir=run_dir, log_path=log_path, logger=logger,
            force_rerun=force_rerun, winners_path=winners_path,
            parallel=parallel, workers=workers,
        )
        out["stage1_winner"]     = w1
        out["stage1_results"]    = r1
        out["stage1_phi_winner"] = w1_phi
        if w1 is None:
            logger.error("No Stage-1 winner — aborting.")
            return out

    w1 = out["stage1_winner"]
    if w1 is None:
        return out

    # Decode Stage-1 winner configuration. w1 always comes from the phi+xi
    # candidate pool (run_stage1 selects it via select_phixi_winner), so
    # this branch's distribution is always dist_phixi -- the phi-only
    # continuation is handled separately below.
    mid1  = w1["model_id"]
    parts = mid1.split("_")
    scaling_decode = {"unit": "unit", "diagfi": "diagonal_inverse_fisher",
                      "fullfi": "inverse_fisher"}
    winner_scaling  = scaling_decode.get(parts[-1], "diagonal_inverse_fisher")
    winner_lag      = parts[-2]                      # "short" or "seasonal"

    # If the fullfi Stage-1 winner has NaN CRPS (GB2 ppf overflow in OOS
    # simulation), Stage 2/3 would inherit fullfi and also produce NaN CRPS,
    # making the 2% improvement test permanently unresolvable.  Fall back to
    # the best Stage-1 model with *valid* CRPS for the scaling passed
    # downstream.  The reported Stage-1 winner is unchanged.
    if winner_scaling == "inverse_fisher":
        w1_crps = float(
            w1.get("oos_metrics", {}).get("crps_mean", float("nan"))
        )
        if not np.isfinite(w1_crps):
            valid_r1 = [
                r for r in r1
                if r.get("validity", "failed") not in ("failed",)
                and "phixi" in r.get("model_id", "")
                and np.isfinite(float(
                    r.get("oos_metrics", {}).get("crps_mean", float("nan"))
                ))
            ]
            fallback = select_winner(valid_r1) if valid_r1 else None
            if fallback:
                fb_parts  = fallback["model_id"].split("_")
                fb_sc     = scaling_decode.get(fb_parts[-1], "diagonal_inverse_fisher")
                logger.info(
                    f"  Stage-1 winner {mid1} has NaN CRPS (fullfi instability); "
                    f"using scaling='{fb_sc}' from {fallback['model_id']} "
                    f"for Stage 2/3 inheritance."
                )
                winner_scaling = fb_sc

    from constants import GAS_SHORT_LAGS, GAS_SEASONAL_LAGS
    winner_gas_lags = (GAS_SHORT_LAGS["daily"] if winner_lag == "short"
                       else GAS_SEASONAL_LAGS["daily"])
    winner_dist     = dist_phixi

    # Stage-1 GAS+pi theta for warm-starting Stage 2
    theta0_gas_pi = None
    if w1.get("result") is not None:
        theta0_gas_pi = w1["result"]["theta"]   # Stage 1 has no covs → full theta = GAS+pi
    else:
        try:
            df = pd.read_csv(run_dir / "stage1" / mid1 / "estimated_parameters.csv")
            theta0_gas_pi = df["value"].to_numpy(dtype=float)
        except Exception:
            pass

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if do_stage2:
        w2, r2 = run_stage2(
            data=data, pi_dyn=pi_dyn,
            dist=winner_dist,
            best_gas_lags=winner_gas_lags,
            best_scaling=winner_scaling,
            theta0_gas_pi=theta0_gas_pi,
            run_dir=run_dir, log_path=log_path, logger=logger,
            force_rerun=force_rerun, winners_path=winners_path,
            parallel=parallel, workers=workers,
        )
        out["stage2_winner"]  = w2
        out["stage2_results"] = r2
        if w2 is None:
            logger.warning("No Stage-2 winner — Stage 3 skipped.")
            return out

    w2 = out["stage2_winner"]
    if w2 is None:
        return out

    best_weather_block = w2.get("best_weather_block",
                                w2["model_id"].replace("stage2_", ""))

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    if do_stage3:
        w3, r3 = run_stage3(
            data=data, pi_dyn=pi_dyn,
            dist=winner_dist,
            best_scaling=winner_scaling,
            best_weather_block=best_weather_block,
            run_dir=run_dir, log_path=log_path, logger=logger,
            force_rerun=force_rerun, winners_path=winners_path,
            parallel=parallel, workers=workers,
        )
        out["stage3_winner"]  = w3
        out["stage3_results"] = r3

    # ── Phi-only branch (objective 1a) ──────────────────────────────────────
    # Independent Stage 2/3 continuation of the phi-only Stage-1 winner,
    # selected and ranked entirely by OOS RMSE, never mixed with the phi+xi
    # branch above. Written to separate model ids / cache subdirectories.
    w1_phi = out["stage1_phi_winner"]
    if run_phi_only_branch and w1_phi is not None:
        mid1p  = w1_phi["model_id"]
        parts_p = mid1p.split("_")
        phi_scaling = scaling_decode.get(parts_p[-1], "diagonal_inverse_fisher")
        phi_lag     = parts_p[-2]
        phi_gas_lags = (GAS_SHORT_LAGS["daily"] if phi_lag == "short"
                        else GAS_SEASONAL_LAGS["daily"])

        theta0_gas_pi_phi = None
        if w1_phi.get("result") is not None:
            theta0_gas_pi_phi = w1_phi["result"]["theta"]
        else:
            try:
                df = pd.read_csv(run_dir / "stage1" / mid1p / "estimated_parameters.csv")
                theta0_gas_pi_phi = df["value"].to_numpy(dtype=float)
            except Exception:
                pass

        if do_stage2:
            w2p, r2p = run_stage2(
                data=data, pi_dyn=pi_dyn,
                dist=dist_phi,
                best_gas_lags=phi_gas_lags,
                best_scaling=phi_scaling,
                theta0_gas_pi=theta0_gas_pi_phi,
                run_dir=run_dir, log_path=log_path, logger=logger,
                force_rerun=force_rerun, winners_path=winners_path,
                parallel=parallel, workers=workers,
                id_prefix="stage2_phi_", stage_dir_name="stage2_phi",
                tvp_set="phi_only",
            )
            out["stage2_phi_winner"]  = w2p
            out["stage2_phi_results"] = r2p

        w2p = out["stage2_phi_winner"]
        if do_stage3 and w2p is not None:
            best_weather_block_phi = w2p.get(
                "best_weather_block", w2p["model_id"].replace("stage2_phi_", "")
            )
            w3p, r3p = run_stage3(
                data=data, pi_dyn=pi_dyn,
                dist=dist_phi,
                best_scaling=phi_scaling,
                best_weather_block=best_weather_block_phi,
                run_dir=run_dir, log_path=log_path, logger=logger,
                force_rerun=force_rerun, winners_path=winners_path,
                parallel=parallel, workers=workers,
                id_prefix="stage3_phi_harvey_", stage_dir_name="stage3_phi",
                tvp_set="phi_only",
            )
            out["stage3_phi_winner"]  = w3p
            out["stage3_phi_results"] = r3p
        elif do_stage3:
            logger.warning("No Stage-2 phi-only winner — Stage 3 phi-only skipped.")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Stage 1 (phi+xi):  {out['stage1_winner']['model_id'] if out['stage1_winner'] else 'n/a'}")
    logger.info(f"  Stage 2 (phi+xi):  {out['stage2_winner']['model_id'] if out['stage2_winner'] else 'n/a'}")
    logger.info(f"  Stage 3 (phi+xi):  {out['stage3_winner']['model_id'] if out['stage3_winner'] else 'n/a'}")
    logger.info(f"  Stage 1 (phi-only): {out['stage1_phi_winner']['model_id'] if out['stage1_phi_winner'] else 'n/a'}")
    logger.info(f"  Stage 2 (phi-only): {out['stage2_phi_winner']['model_id'] if out['stage2_phi_winner'] else 'n/a'}")
    logger.info(f"  Stage 3 (phi-only): {out['stage3_phi_winner']['model_id'] if out['stage3_phi_winner'] else 'n/a'}")
    logger.info("=" * 60)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Utility: resource estimate (informational display before running)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_resources(
    n_models:      int,
    avg_runtime_s: float = 300.0,
    mb_per_model:  float = 300.0,
) -> str:
    wall_min = avg_runtime_s * n_models / 60.0
    peak_gb  = mb_per_model / 1024.0
    return (
        f"\n{'='*52}\n"
        f"  Models to estimate:     {n_models}\n"
        f"  Avg runtime/model:      {avg_runtime_s/60:.0f} min (estimate)\n"
        f"  Total wall time:        {wall_min:.0f} min (sequential)\n"
        f"  Peak RAM per model:     {mb_per_model:.0f} MB (estimate)\n"
        f"{'='*52}"
    )
