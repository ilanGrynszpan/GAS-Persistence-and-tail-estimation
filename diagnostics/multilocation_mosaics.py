"""
Cross-location stage-winner resolution for the Stage 1-4 diagnostic mosaics.

=============================================================================
OVERVIEW
=============================================================================
This module answers one question per (stage, branch, location) triple:
"which fitted model is the best alternative at this stage, and what are its
in-sample PIT values / quantile residuals?" -- so that
notebooks/multilocation_diagnostic_mosaics.ipynb can hand the results
straight to diagnostics.plots' pit_histogram_mosaic / qq_plot_mosaic /
acf_mosaic_400 builders.

It does not fit or re-fit any model. Per docs/EXECUTION.md Sec 22 ("Reports
should never call optimization"), everything here reads previously saved
artifacts (stage_winners.json, metadata.json, estimated_parameters.csv) and
recomputes only the cheap, deterministic, read-only quantities: the
predictive CDF series at the training observations, and the PIT/quantile
residuals derived from it.

=============================================================================
WHY THIS WRAPS generate_report_extended.py INSTEAD OF REIMPLEMENTING
=============================================================================
Reconstructing a fitted model from its saved metadata + parameter CSV
(figuring out which ZAGASModel/CovZAGASModel/HarveyLongShortModel subclass
it is, which covariate blocks it used, how to rebuild a Stage-4
RegimeXiOnlyModel's frozen phi/pi trajectory, ...) is exactly what
generate_report_extended.py's `_compute_is_diagnostics_for_winner` already
does for the main multi-location report, and it has already absorbed two
documented bug fixes (see that file's comments near
`_patch_meta_for_reconstruction` and `_STALE_MODEL_IDS`). Per CLAUDE.md
("refactor rather than duplicate", "determine whether functionality already
exists before creating new modules"), this module imports and reuses that
function rather than re-deriving model reconstruction from scratch.

=============================================================================
CODE WALKTHROUGH
=============================================================================
load_all_location_data()
    |
    v
For each of the 6 completed locations, load {y_train, covariate blocks,
dates, ...} once via StationDataLoader (station_loader.py) -- reused across
every stage/branch combination the notebook asks for, since data loading
touches CSV/ERA5 files and there is no reason to repeat it 7 times.
    |
    v
compute_stage_branch_diagnostics(stage, branch, location_data, fig_cache_dir)
    |
    v
For each location:
    resolve_stage_winner()  -> which saved model_id is "the" winner
        |
        v
    generate_report_extended._compute_is_diagnostics_for_winner()
        -> reconstructs the model from artifacts, evaluates cdf_series()
           at the training data, converts to PIT / quantile residuals
           (diagnostics/residuals.py), and (as a side effect) saves the
           individual per-model PNGs it always produces into fig_cache_dir
        |
        v
    collect pit / quantile-residual arrays, keyed by location display name
    |
    v
Return three dicts (pit, quantile residuals, model_id used), each with one
entry per location -- entries are None where the stage/branch was not run,
not accepted, or artifacts were unavailable, so downstream mosaics can
render a labelled placeholder panel instead of silently dropping a location.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from data.station_loader import STATION_REGISTRY, StationDataLoader

ROOT = Path(__file__).resolve().parent.parent
PRECIP_DIR = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR = ROOT / "data" / "input" / "ERA5"
ENSO_PATH = ROOT / "data" / "processed" / "pacific" / "ENSO_clean.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

# The six locations that completed the Stage 1-4 pipeline, in the same
# order as generate_report_extended.py's ALL_STATIONS_ORDERED, so these
# mosaics stay location-order-consistent with the main multi-location
# report. See ROADMAP.md "Multi-Location Experiment Status" for run history
# (SAO PAULO and TORONTO are registered in STATION_REGISTRY but were never
# estimated; RIYADH was excluded for data-quality problems).
LOCATIONS = [
    "BELO HORIZONTE",
    "CRUZEIRO DO SUL (ACRE)",
    "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)",
    "MANAUS",
    "SALVADOR",
]

STAGE_LABELS: Dict[int, str] = {
    1: "Stage 1 -- Baseline GAS",
    2: "Stage 2 -- Weather Covariates",
    3: "Stage 3 -- Harvey Long-Short",
    4: "Stage 4 -- Tail-Sensitive xi Regime",
}

BRANCH_LABELS: Dict[str, str] = {
    "phi_xi": "phi+xi branch",
    "phi_only": "phi-only branch",
}

# stage_winners.json key holding each (stage, branch)'s winning model_id.
# Stage 4 has no branch split (see resolve_stage4_winner) so it is absent
# here on purpose -- resolve_stage_winner() special-cases stage == 4.
_STAGE_WINNER_KEYS: Dict[Tuple[int, str], str] = {
    (1, "phi_xi"): "stage1",
    (1, "phi_only"): "stage1_phi_only",
    (2, "phi_xi"): "stage2",
    (2, "phi_only"): "stage2_phi_only",
    (3, "phi_xi"): "stage3",
    (3, "phi_only"): "stage3_phi_only",
}


def resolve_stage4_winner(winners: dict) -> Optional[str]:
    """
    Return the Stage-4 tail-sensitive xi-regime model_id to diagnose, or
    None if Stage 4 was not run at this location.

    Stage 4 (docs/MODELS.md Sec 22-23) has no phi-only/phi+xi branch split:
    it is a single model type (xi regime-sensitised at a high threshold),
    fit independently at threshold q95 and q98. Following the user
    instruction to show each stage's own winner "regardless of whether that
    stage won overall", this mirrors generate_report_extended.py's own
    Stage-4 figure selection: report whichever of q95/q98 achieved the
    *better relative improvement* over its phi+xi-branch counterpart,
    displayed whether or not that improvement cleared the 2% acceptance
    bar -- a rejected regime model's calibration is still informative
    (see generate_report_extended.py, Stage-4 section, `s4_candidates`).
    """
    s4 = winners.get("stage4_xi_regime", {})
    accepted = s4.get("accepted", {})
    candidates = [
        (q, accepted.get(q, {}).get("rel_improvement"))
        for q in ("95", "98")
        if accepted.get(q, {}).get("rel_improvement") is not None
        and np.isfinite(accepted[q]["rel_improvement"])
    ]
    if not candidates:
        return None
    best_q, _ = max(candidates, key=lambda t: t[1])
    return f"stage4_xi_regime_q{best_q}"


def resolve_stage_winner(winners: dict, stage: int, branch: Optional[str]) -> Optional[str]:
    """
    Return the model_id of the requested stage's (and branch's) winner, as
    saved in that location's stage_winners.json, or None if unavailable.

    Parameters
    ----------
    winners : dict loaded from artifacts/{run_id}/stage_winners.json
    stage   : 1, 2, 3, or 4
    branch  : "phi_xi" or "phi_only" for stages 1-3; ignored for stage 4
    """
    if stage == 4:
        return resolve_stage4_winner(winners)
    if stage not in (1, 2, 3):
        raise ValueError(f"Unsupported stage {stage!r}; expected 1, 2, 3, or 4.")
    if branch not in ("phi_xi", "phi_only"):
        raise ValueError(f"Unsupported branch {branch!r}; expected 'phi_xi' or 'phi_only'.")
    key = _STAGE_WINNER_KEYS[(stage, branch)]
    return winners.get(key, {}).get("winner")


def load_all_location_data() -> Dict[str, dict]:
    """
    Load {y_train, covariate blocks, dates, ...} once for every location in
    LOCATIONS, keyed by the STATION_REGISTRY station name. Reused across
    every (stage, branch) combination the notebook computes, so the CSV/
    ERA5/ENSO files backing each location are only read once per session.
    """
    location_data: Dict[str, dict] = {}
    for station in LOCATIONS:
        loader = StationDataLoader.from_registry(
            station_name=station,
            precip_dir=PRECIP_DIR,
            era5_dir=ERA5_DIR,
            enso_path=ENSO_PATH,
        )
        location_data[station] = loader.load_all()
    return location_data


def compute_stage_branch_diagnostics(
    stage: int,
    branch: Optional[str],
    location_data: Dict[str, dict],
    fig_cache_dir: Path,
) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, Optional[np.ndarray]], Dict[str, Optional[str]]]:
    """
    Resolve and diagnose the requested stage/branch winner at every
    location.

    Parameters
    ----------
    stage         : 1, 2, 3, or 4
    branch        : "phi_xi" or "phi_only"; ignored for stage 4
    location_data : output of load_all_location_data()
    fig_cache_dir : directory for the individual per-model PIT/QQ/ACF PNGs
                    that _compute_is_diagnostics_for_winner always saves as
                    a side effect (kept separate from reports/multi_location
                    /figures so this notebook never overwrites the main
                    report's cached figures)

    Returns
    -------
    (pit_by_location, qr_by_location, model_id_by_location)
    Three dicts, each keyed by location *display name* (STATION_REGISTRY's
    "display", e.g. "Belo Horizonte") with one entry per LOCATIONS entry.
    Entries are None where the stage/branch was not run, not accepted, or
    artifacts were unavailable -- diagnostics.plots' mosaic builders render
    these as labelled placeholder panels.
    """
    # Imported lazily (not at module import time) so importing this module
    # never forces matplotlib's Agg backend or pulls in the full ~3000-line
    # report script unless a caller actually asks for diagnostics.
    from generate_report_extended import _load_winners, _compute_is_diagnostics_for_winner

    pit_out: Dict[str, Optional[np.ndarray]] = {}
    qr_out: Dict[str, Optional[np.ndarray]] = {}
    model_id_out: Dict[str, Optional[str]] = {}

    for station in LOCATIONS:
        cfg = STATION_REGISTRY[station]
        display = cfg["display"]
        run_dir = ARTIFACTS_DIR / cfg["run_id"]
        pit_out[display] = None
        qr_out[display] = None
        model_id_out[display] = None

        if not run_dir.exists():
            print(f"  [SKIP] {display}: run directory not found ({run_dir})")
            continue

        winners = _load_winners(run_dir)
        model_id = resolve_stage_winner(winners, stage, branch)
        if model_id is None:
            print(f"  [SKIP] {display}: no winner recorded for this stage/branch")
            continue

        data = location_data.get(station)
        if data is None:
            print(f"  [SKIP] {display}: location data not preloaded")
            continue

        loc_fig_dir = fig_cache_dir / cfg["short"]
        diag = _compute_is_diagnostics_for_winner(
            model_id=model_id,
            run_dir=run_dir,
            data=data,
            col_name_dict=data["covariate_col_names"],
            fig_dir=loc_fig_dir,
            winners=winners,
        )
        if diag is None:
            print(f"  [WARN] {display}: IS diagnostics unavailable for {model_id}")
            continue

        pit_out[display] = diag["pit"]
        qr_out[display] = diag["qr"]
        model_id_out[display] = model_id

    return pit_out, qr_out, model_id_out


def compute_stages_1to3_grid(
    branch: str,
    location_data: Dict[str, dict],
    fig_cache_dir: Path,
) -> Tuple[
    Dict[int, Dict[str, Optional[np.ndarray]]],
    Dict[int, Dict[str, Optional[np.ndarray]]],
    Dict[int, Dict[str, Optional[str]]],
]:
    """
    Convenience wrapper around compute_stage_branch_diagnostics for the
    combined Stages 1-3 grid mosaics (diagnostics.plots.stage_pit_grid_mosaic
    / stage_qq_grid_mosaic / stage_acf_grid_mosaic), which need results
    nested by stage rather than the flat per-stage dicts that function
    returns on its own.

    Parameters
    ----------
    branch        : "phi_xi" or "phi_only"
    location_data : output of load_all_location_data()
    fig_cache_dir : forwarded to compute_stage_branch_diagnostics

    Returns
    -------
    (pit_grid, qr_grid, model_id_grid), each {stage: {location: ...}}
    for stage in (1, 2, 3).
    """
    pit_grid: Dict[int, Dict[str, Optional[np.ndarray]]] = {}
    qr_grid: Dict[int, Dict[str, Optional[np.ndarray]]] = {}
    model_id_grid: Dict[int, Dict[str, Optional[str]]] = {}

    for stage in (1, 2, 3):
        pit_by_loc, qr_by_loc, model_id_by_loc = compute_stage_branch_diagnostics(
            stage=stage, branch=branch,
            location_data=location_data, fig_cache_dir=fig_cache_dir,
        )
        pit_grid[stage] = pit_by_loc
        qr_grid[stage] = qr_by_loc
        model_id_grid[stage] = model_id_by_loc

    return pit_grid, qr_grid, model_id_grid
