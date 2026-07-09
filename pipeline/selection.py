"""
Model-selection criterion — prompt.md Objective 1 / 4d (2026-07-07 revision).

=============================================================================
CRITERION
=============================================================================

Round-1 selection (pipeline.runner.select_winner) ranked all Stage-1 models
together by training log-likelihood, regardless of dynamic-parameter set.
Round 2 replaces this with two separate, metric-specific comparisons that
never mix phi-only and phi+xi models:

    phi-only models  -> ranked by OOS RMSE (MAD reported alongside, not
                         used for ranking)
    phi+xi   models  -> ranked by OOS CRPS ("the main criterion"; twCRPS@95
                         and twCRPS@98 reported alongside, not used for
                         ranking except in the Stage-4-specific comparison
                         below)

Within either set, the winner is the best-key-metric model UNLESS a
*simpler* model is within `improvement_threshold` (default 2%) of it, in
which case the simpler model wins (Occam's-razor tie-break). Simplicity is
measured by estimated parameter count.

The identical function selects both in-stage winners (candidates = all
models of one tvp-set within a stage) and inter-stage winners (candidates =
the winners of two consecutive stages) — prompt.md 4d: "This same logic
should be used for inter-stage decision."

Stage 4 uses a separate, pairwise comparison (objective 3a): a Stage-4
model is compared only against its frozen-phi base ("best of previous
stages counterpart") using twCRPS@95, not against other Stage-4 models
across thresholds/locations.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Sequence


def _get_metric(candidate: dict, key: str) -> Optional[float]:
    """Look up a metric: top-level key first, then nested oos_metrics."""
    if key in candidate and candidate[key] is not None:
        return candidate[key]
    om = candidate.get("oos_metrics", {}) or {}
    return om.get(key)


def select_by_key_metric(
    candidates: Sequence[dict],
    key_metric: str,
    higher_better: bool = False,
    improvement_threshold: float = 0.02,
    only_valid: bool = True,
) -> Optional[dict]:
    """
    Select a winner among `candidates` using the best-key-metric-or-simplest
    rule (prompt.md 4d).

    Each candidate dict must expose `key_metric` (top-level or nested under
    "oos_metrics"), an "n_params" field for the simplicity tie-break, and
    optionally a "validity" field ("failed" candidates are dropped when
    only_valid=True).

    Returns a shallow copy of the winning candidate with an added
    "_tie_break" string explaining the decision, or None if no candidate
    has a finite key_metric.
    """
    pool = [c for c in candidates if not only_valid or c.get("validity") != "failed"]
    pool = [
        c for c in pool
        if (v := _get_metric(c, key_metric)) is not None and math.isfinite(v)
    ]
    if not pool:
        return None

    # Sort by model_id first so every tie-break below (best-metric ties,
    # equal-simplicity ties) resolves the same way regardless of the input
    # list's order -- e.g. parallel-execution completion order, or a report
    # generator re-listing artifacts alphabetically after the fact. Without
    # this, the exact same set of candidates could select a different
    # "winner" purely from which happened to appear first in `candidates`.
    pool = sorted(pool, key=lambda c: str(c.get("model_id", "")))

    best = (max if higher_better else min)(pool, key=lambda c: _get_metric(c, key_metric))
    best_val = _get_metric(best, key_metric)
    denom = abs(best_val) if best_val != 0 else 1.0

    def _rel_gap(c: dict) -> float:
        v = _get_metric(c, key_metric)
        # Positive gap = worse than best, regardless of higher_better direction.
        raw = (best_val - v) if higher_better else (v - best_val)
        return max(0.0, raw) / denom

    close_enough = [c for c in pool if _rel_gap(c) <= improvement_threshold]
    simplest = min(close_enough, key=lambda c: c.get("n_params", float("inf")))

    winner = dict(simplest)
    if simplest is best or simplest.get("model_id") == best.get("model_id"):
        winner["_tie_break"] = f"best {key_metric} ({best_val:.4g})"
    else:
        winner["_tie_break"] = (
            f"simplest model (n_params={simplest.get('n_params')}) within "
            f"{improvement_threshold:.0%} of best {key_metric} "
            f"({best.get('model_id')}, {best_val:.4g})"
        )
    winner["_key_metric"] = key_metric
    winner["_best_in_pool"] = best.get("model_id")
    return winner


def explain_winner(
    candidates: Sequence[dict],
    winner_id: str,
    key_metric: str,
    higher_better: bool = False,
    improvement_threshold: float = 0.02,
) -> dict:
    """
    Explain why an ALREADY-DECIDED winner (e.g. read back from
    stage_winners.json, produced live during pipeline execution) satisfies
    the best-or-simplest rule -- without re-running select_by_key_metric,
    which could pick a *different* winner if `candidates` isn't in the
    exact same order the live run saw (e.g. a report generator re-listing
    artifact directories alphabetically after the fact). Reports must
    describe what actually happened, not a fresh, possibly different
    selection -- this function only explains, it never re-decides.

    Returns the winner's candidate dict (if found in `candidates`) with an
    added "_tie_break" string, in the same style as select_by_key_metric.
    """
    pool = [
        c for c in candidates
        if c.get("validity") != "failed"
        and (v := _get_metric(c, key_metric)) is not None and math.isfinite(v)
    ]
    winner = next((c for c in pool if c.get("model_id") == winner_id), None)
    if winner is None:
        return {"model_id": winner_id, "_tie_break": "(candidate not found in cached artifacts)"}
    if not pool:
        return {**winner, "_tie_break": "no valid candidates to compare against"}

    best = (max if higher_better else min)(pool, key=lambda c: _get_metric(c, key_metric))
    best_val = _get_metric(best, key_metric)
    denom = abs(best_val) if best_val != 0 else 1.0
    w_val = _get_metric(winner, key_metric)
    raw = (best_val - w_val) if higher_better else (w_val - best_val)
    gap = max(0.0, raw) / denom

    out = dict(winner)
    if winner.get("model_id") == best.get("model_id"):
        out["_tie_break"] = f"best {key_metric} ({best_val:.4g})"
    elif gap <= improvement_threshold:
        out["_tie_break"] = (
            f"simplest model (n_params={winner.get('n_params')}) within "
            f"{improvement_threshold:.0%} of best {key_metric} "
            f"({best.get('model_id')}, {best_val:.4g})"
        )
    else:
        out["_tie_break"] = (
            f"selected historically ({gap:.1%} from best {key_metric} "
            f"{best.get('model_id')}, {best_val:.4g}); outside the "
            f"{improvement_threshold:.0%} tie-break margin"
        )
    out["_key_metric"] = key_metric
    out["_best_in_pool"] = best.get("model_id")
    return out


def select_phi_only_winner(
    candidates: Sequence[dict],
    improvement_threshold: float = 0.02,
) -> Optional[dict]:
    """phi-only models: ranked by OOS RMSE (objective 1a)."""
    return select_by_key_metric(
        candidates, key_metric="rmse", higher_better=False,
        improvement_threshold=improvement_threshold,
    )


def select_phixi_winner(
    candidates: Sequence[dict],
    improvement_threshold: float = 0.02,
) -> Optional[dict]:
    """phi+xi models: ranked by OOS CRPS, the main criterion (objective 1b)."""
    return select_by_key_metric(
        candidates, key_metric="crps_mean", higher_better=False,
        improvement_threshold=improvement_threshold,
    )


def stage4_beats_base(
    stage4_result: dict,
    base_result: dict,
    key_metric: str = "twcrps_95",
    improvement_threshold: float = 0.02,
) -> Dict:
    """
    Objective 3a: compare a Stage-4 (xi-only regime) model against its
    frozen-phi base ("best of previous stages counterpart") using twCRPS.

    Returns a dict: {"winner": "stage4"|"base", "stage4_value", "base_value",
    "rel_improvement", "key_metric"}. The base wins on a tie or if Stage 4's
    improvement is below `improvement_threshold` (Occam's razor: the
    simpler, non-regime base is preferred unless the regime clearly helps).
    """
    v4 = _get_metric(stage4_result, key_metric)
    vb = _get_metric(base_result, key_metric)
    if v4 is None or vb is None or not (math.isfinite(v4) and math.isfinite(vb)):
        return {
            "winner": "base", "stage4_value": v4, "base_value": vb,
            "rel_improvement": float("nan"), "key_metric": key_metric,
            "reason": "missing or non-finite metric",
        }
    denom = abs(vb) if vb != 0 else 1.0
    rel_improvement = (vb - v4) / denom  # positive = stage4 lower/better
    winner = "stage4" if rel_improvement > improvement_threshold else "base"
    return {
        "winner": winner, "stage4_value": v4, "base_value": vb,
        "rel_improvement": rel_improvement, "key_metric": key_metric,
    }


def select_inter_stage_winner(
    stage_winners: Sequence[dict],
    tvp_set: str,
    improvement_threshold: float = 0.02,
) -> Optional[dict]:
    """
    Apply the same best-or-simplest rule across a sequence of stage winners
    (e.g. [stage1_winner, stage2_winner, stage3_winner]) to pick the overall
    inter-stage winner for one tvp set. `tvp_set` must be "phi_only" or
    "phi_xi" and selects the ranking metric.
    """
    fn = select_phi_only_winner if tvp_set == "phi_only" else select_phixi_winner
    return fn(stage_winners, improvement_threshold=improvement_threshold)
