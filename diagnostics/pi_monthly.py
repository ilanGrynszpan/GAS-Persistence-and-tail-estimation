"""
In-sample monthly seasonality of the Stage 1 occurrence probability pi_t.

=============================================================================
OVERVIEW
=============================================================================
Answers one question per location: "in the Stage 1 phi-only model (the
baseline GAS specification with a single time-varying parameter, phi_t --
xi and the AR-logistic occurrence dynamics are static), how does the
model's fitted rain-occurrence probability pi_t average out across
calendar months, and how does that compare with the empirical fraction of
wet days observed in each calendar month?" This is a reporting-only
diagnostic (docs/EXECUTION.md Sec 22: reports never call optimization) --
it replays an already-fitted model's filter() over the training data and
aggregates, nothing is re-estimated.

"Stage 1 model with only one time-varying parameter" is the phi-only
branch's Stage 1 winner (docs/MODELS.md Sec 27: phi_t dynamic, xi static),
resolved exactly as the Stage 1-4 diagnostic mosaics do
(diagnostics/multilocation_mosaics.py's resolve_stage_winner(winners, 1,
"phi_only")) -- reused here rather than re-implemented, per CLAUDE.md
("refactor rather than duplicate").

=============================================================================
WHY MODEL.FILTER() RATHER THAN THE SAVED paths.npz
=============================================================================
Each model's paths.npz on disk only holds the *out-of-sample* simulated
path (pi_oos, f_arr_oos -- see run_all_locations.py); the in-sample pi_t
trajectory is a byproduct of the likelihood evaluation and was never saved
to disk (nothing in the Stage 1-4 pipeline needed it after fitting). It is
recomputed here the same way generate_report_extended.py's
_compute_is_diagnostics_for_winner() recomputes in-sample PIT/quantile
residuals: reconstruct the already-fitted model from metadata.json +
estimated_parameters.csv (pipeline.artifact_utils.build_model_from_meta),
then call the model's own .filter(theta, y_train) -- a pure replay of the
saved MLE, not a new optimization.

=============================================================================
MATHEMATICAL NOTE
=============================================================================
model.filter(theta, y_train) returns (models/za_gas_model.py):
    paths["pi"]        -- pi_t = sigma(eta_t) for t = eff_start, ..., T-1
    paths["y_eff"]      -- y_t for the same t range
    paths["eff_start"]  -- eff = max_lag (the GAS warm-up length)
For each calendar month m in {1, ..., 12}, this module computes
    mean{ pi_t : month(t) = m }            (model's average predicted
                                             occurrence probability)
    mean{ 1(y_t > 0) : month(t) = m }      (empirical wet-day fraction)
over the effective in-sample window (post warm-up), aligned to that
model's own dates_train[eff_start:] -- warm-up length differs by lag
structure (3 for "short", up to 367 for "seasonal"), so the alignment is
recomputed per location rather than assumed fixed.

=============================================================================
CODE WALKTHROUGH
=============================================================================
compute_monthly_pi_and_occurrence(location_data)
    |
    v
For each of the 6 locations (diagnostics.multilocation_mosaics.LOCATIONS):
    load stage_winners.json -> resolve_stage_winner(..., 1, "phi_only")
        |
        v
    load metadata.json + estimated_parameters.csv for that model_id
        |
        v
    pipeline.artifact_utils.build_model_from_meta(...) -> unfitted model
    instance; get_theta(params_df) -> theta
        |
        v
    model.filter(theta, y_train) -> paths["pi"], paths["y_eff"], paths["eff_start"]
        |
        v
    group by dates_train[eff_start:].month -> per-month mean(pi), mean(wet)
    |
    v
Return {location display name: {"pi_by_month": [12], "wet_frac_by_month":
[12], "model_id": ..., "n": ...}}, keyed the same way as
diagnostics/multilocation_mosaics.py's other per-location dicts (None
where the winner or its artifacts are unavailable, so the calling notebook
can render a labelled placeholder panel instead of silently dropping a
location).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from data.station_loader import STATION_REGISTRY
from diagnostics.multilocation_mosaics import (
    LOCATIONS, ARTIFACTS_DIR, resolve_stage_winner,
)

# Calendar month labels, Jan..Dec, shared by both mosaic chart types so the
# x-axis is identical whether one or two series are plotted per month.
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def compute_monthly_pi_and_occurrence(
    location_data: Dict[str, dict],
) -> Dict[str, Optional[dict]]:
    """
    Resolve the Stage 1 phi-only winner at every location in LOCATIONS and
    compute its in-sample monthly-average pi_t alongside the empirical
    monthly wet-day fraction.

    Parameters
    ----------
    location_data : output of diagnostics.multilocation_mosaics.
                    load_all_location_data() -- {station: {"y_train":...,
                    "dates_train":..., ...}}

    Returns
    -------
    {location display name: dict or None}, dict has keys:
        "pi_by_month"        : np.ndarray, shape (12,), Jan..Dec
        "wet_frac_by_month"  : np.ndarray, shape (12,), Jan..Dec
        "model_id"           : the Stage 1 phi-only winner's model_id
        "n"                  : effective in-sample sample size used
    None where the winner or its filtered path could not be computed
    (missing artifacts, reconstruction failure, ...) -- never raises, so
    one location's failure does not stop the others (docs/EXECUTION.md
    Sec 20: "failure isolation").
    """
    # Imported lazily, matching the lazy-import convention used throughout
    # diagnostics/multilocation_mosaics.py: avoids a module-import-time
    # dependency on generate_report_extended.py (matplotlib Agg backend,
    # the full report script) for callers that only need this function.
    from generate_report_extended import _load_winners, _load_meta, _load_params_df, _resolve_stage_dir
    from pipeline.artifact_utils import build_model_from_meta, get_theta

    out: Dict[str, Optional[dict]] = {}

    for station in LOCATIONS:
        cfg = STATION_REGISTRY[station]
        display = cfg["display"]
        run_dir = ARTIFACTS_DIR / cfg["run_id"]
        out[display] = None

        if not run_dir.exists():
            print(f"  [SKIP] {display}: run directory not found ({run_dir})")
            continue

        winners = _load_winners(run_dir)
        model_id = resolve_stage_winner(winners, 1, "phi_only")
        if model_id is None:
            print(f"  [SKIP] {display}: no Stage 1 phi-only winner recorded")
            continue

        data = location_data.get(station)
        if data is None:
            print(f"  [SKIP] {display}: location data not preloaded")
            continue

        stage_dir = _resolve_stage_dir(model_id)
        meta = _load_meta(run_dir, stage_dir, model_id)
        params_df = _load_params_df(run_dir, stage_dir, model_id)
        theta = get_theta(params_df)
        if theta is None or not meta:
            print(f"  [WARN] {display}: parameter artifacts unavailable for {model_id}")
            continue

        try:
            model = build_model_from_meta(meta, params_df, model_id=model_id)
            paths = model.filter(theta, data["y_train"])
        except Exception as exc:
            print(f"  [WARN] {display}: filter() failed for {model_id}: {exc}")
            continue

        pi = paths.get("pi")
        y_eff = paths.get("y_eff")
        eff_start = paths.get("eff_start")
        if pi is None or y_eff is None or eff_start is None or len(pi) == 0:
            print(f"  [WARN] {display}: empty filtered path for {model_id}")
            continue

        dates_eff = pd.DatetimeIndex(data["dates_train"])[eff_start:]
        if len(dates_eff) != len(pi):
            print(f"  [WARN] {display}: date/path length mismatch for {model_id} "
                  f"({len(dates_eff)} vs {len(pi)}) -- skipping")
            continue

        monthly = pd.DataFrame({
            "month": dates_eff.month,
            "pi":    np.asarray(pi, dtype=float),
            "wet":   (np.asarray(y_eff, dtype=float) > 0).astype(float),
        }).groupby("month").mean().reindex(range(1, 13))

        out[display] = {
            "pi_by_month":       monthly["pi"].to_numpy(),
            "wet_frac_by_month": monthly["wet"].to_numpy(),
            "model_id":          model_id,
            "n":                 int(len(pi)),
        }

    return out
