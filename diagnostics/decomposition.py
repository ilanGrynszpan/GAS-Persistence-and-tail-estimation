"""
Seasonal, ENSO, and wet/dry season decomposition diagnostics.

=============================================================================
ANALYSIS TYPES
=============================================================================

1. ENSO decomposition
   Classifies each day into El Niño / Neutral / La Niña using the 90-day
   rolling mean of the Niño 3.4 SST index (NOAA standard thresholds ±0.5°C).
   Also produces a continuous ENSO intensity analysis using quintile bins.

2. Seasonal (calendar-month) decomposition
   Groups diagnostics by calendar month (January … December).
   Produces monthly boxplots of dynamic quantiles, return levels, and scores.

3. Wet/dry season classification
   Per-location: months with mean rainfall above the grand monthly mean →
   wet; remaining → dry.  Isolated anomalous months (surrounded by the
   opposite classification) are reclassified to preserve contiguity.
   If all months lie within ±2% of the grand mean the location is classified
   as "no clear seasonal signal".

=============================================================================
METRICS AVAILABLE PER GROUP
=============================================================================

For each group (ENSO regime, month, wet/dry) the decomposition computes:
    - Mean and percentiles of dynamic conditional quantiles
    - Mean and percentiles of dynamic return levels
    - Empirical exceedance frequencies (all quantile levels)
    - Mean quantile score (QS) per quantile level
    - Mean twCRPS per threshold level

=============================================================================
USAGE
=============================================================================

    from diagnostics.decomposition import (
        classify_enso_regime,
        classify_wet_dry_months,
        seasonal_decomposition,
        enso_decomposition,
        wet_dry_decomposition,
    )
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────────────
# ENSO classification
# ─────────────────────────────────────────────────────────────────────────────

ENSO_EL_NINO = "El Niño"
ENSO_NEUTRAL  = "Neutral"
ENSO_LA_NINA  = "La Niña"

ENSO_COLORS = {
    ENSO_EL_NINO: "#e74c3c",
    ENSO_NEUTRAL:  "#95a5a6",
    ENSO_LA_NINA:  "#3498db",
}


def classify_enso_regime(
    nino34: np.ndarray,
    threshold: float = 0.5,
    window: int = 90,
) -> np.ndarray:
    """
    Classify daily ENSO regime using a rolling mean of the Niño 3.4 index.

    Standard NOAA thresholds (±0.5°C on 3-month running mean).

    Returns
    -------
    ndarray of str labels: El Niño, Neutral, La Niña
    """
    series    = pd.Series(nino34)
    roll_mean = series.rolling(window=window, min_periods=1).mean()
    labels    = np.full(len(nino34), ENSO_NEUTRAL, dtype=object)
    labels[roll_mean.values >=  threshold] = ENSO_EL_NINO
    labels[roll_mean.values <= -threshold] = ENSO_LA_NINA
    return labels


def classify_enso_intensity(
    nino34: np.ndarray,
    n_bins: int = 5,
    window: int = 90,
) -> Tuple[np.ndarray, List[float]]:
    """
    Classify daily ENSO intensity into equal-frequency bins.

    Parameters
    ----------
    nino34 : raw daily Niño 3.4 array
    n_bins : number of bins (default 5 = quintiles)
    window : rolling window for smoothing (default 90 days)

    Returns
    -------
    labels     : integer bin labels 0 … n_bins-1
    bin_edges  : quantile edges of the smoothed Niño 3.4 distribution
    """
    series    = pd.Series(nino34)
    roll_mean = series.rolling(window=window, min_periods=1).mean().values

    bin_edges = [float(np.quantile(roll_mean, q)) for q in np.linspace(0, 1, n_bins + 1)]
    labels    = np.digitize(roll_mean, bins=bin_edges[1:-1])   # 0-indexed bins
    return labels, bin_edges


# ─────────────────────────────────────────────────────────────────────────────
# Wet / dry season classification
# ─────────────────────────────────────────────────────────────────────────────

WET_MONTH  = "wet"
DRY_MONTH  = "dry"
NO_SEASON  = "no_clear_seasonal_signal"


def classify_wet_dry_months(
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    tolerance_frac: float = 0.02,
) -> Dict:
    """
    Classify calendar months as wet or dry for a given location.

    Algorithm
    ---------
    1. Compute mean rainfall for each calendar month (1-12).
    2. Compute the grand monthly mean (average across all 12 monthly means).
    3. Months with mean > grand_mean → wet; months with mean ≤ grand_mean → dry.
    4. Apply contiguity correction: an isolated anomalous month (surrounded on
       both sides by the opposite classification) is reclassified.
    5. If all months lie within ±tolerance_frac (2%) of the grand mean,
       return "no_clear_seasonal_signal".

    Returns
    -------
    dict with:
        "wet_months"   : sorted list of wet month numbers (1-12)
        "dry_months"   : sorted list of dry month numbers (1-12)
        "monthly_mean" : {month: mean_rainfall}
        "grand_mean"   : float
        "status"       : "wet_dry_identified" or "no_clear_seasonal_signal"
        "classification": {month: "wet"/"dry"} for all 12 months
    """
    series     = pd.Series(y, index=dates)
    month_mean = series.groupby(series.index.month).mean()

    grand_mean = float(month_mean.mean())

    # Check if variation is negligible
    relative_range = float(
        (month_mean.max() - month_mean.min()) / (grand_mean + 1e-12)
    )
    if relative_range <= 2 * tolerance_frac:
        return {
            "wet_months":   [],
            "dry_months":   [],
            "monthly_mean": month_mean.to_dict(),
            "grand_mean":   grand_mean,
            "status":       NO_SEASON,
            "classification": {m: "ambiguous" for m in range(1, 13)},
        }

    # Initial classification
    classification = {}
    for m in range(1, 13):
        mu = float(month_mean.get(m, grand_mean))
        classification[m] = WET_MONTH if mu > grand_mean else DRY_MONTH

    # Contiguity correction: month m is isolated if its neighbours have
    # opposite classification (circular, January ↔ December adjacent)
    changed = True
    while changed:
        changed = False
        for m in range(1, 13):
            prev_m = (m - 2) % 12 + 1
            next_m = m % 12 + 1
            if (classification[prev_m] == classification[next_m]
                    and classification[m] != classification[prev_m]):
                classification[m] = classification[prev_m]
                changed = True

    wet_months = sorted(m for m, v in classification.items() if v == WET_MONTH)
    dry_months = sorted(m for m, v in classification.items() if v == DRY_MONTH)

    return {
        "wet_months":    wet_months,
        "dry_months":    dry_months,
        "monthly_mean":  month_mean.to_dict(),
        "grand_mean":    grand_mean,
        "status":        "wet_dry_identified",
        "classification": classification,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Grouped metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _group_metrics(
    y_group: np.ndarray,
    quants_group: np.ndarray,
    quantile_levels: Sequence[float],
    draws_group: Optional[np.ndarray] = None,
    twcrps_thresholds: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute summary metrics for a subset of observations.

    Parameters
    ----------
    y_group          : subset of observed rainfall
    quants_group     : dynamic quantiles (n_group, n_q)
    quantile_levels  : probability levels
    draws_group      : optional MC draws (n_group, n_draws) for CRPS/twCRPS
    twcrps_thresholds: threshold values for twCRPS (one per twCRPS level)

    Returns
    -------
    dict with keys: n, n_wet, mean_y, mean_quantiles, exceedance_emp,
                    exceedance_ratio, qs_mean (if computable)
    """
    n = len(y_group)
    if n == 0:
        return {"n": 0}

    n_q  = quants_group.shape[1] if quants_group.ndim == 2 else len(quantile_levels)
    q_levels = list(quantile_levels)

    result: dict = {
        "n":       n,
        "n_wet":   int((y_group > 0).sum()),
        "mean_y":  float(y_group.mean()),
    }

    # Mean dynamic quantile for each level
    if quants_group.ndim == 2:
        for k, q in enumerate(q_levels):
            result[f"mean_Q{q}"] = float(np.nanmean(quants_group[:, k]))

    # Exceedance calibration
    for k, q in enumerate(q_levels):
        if quants_group.ndim == 2:
            q_t = quants_group[:, k]
        else:
            continue
        mask = np.isfinite(q_t)
        if mask.sum() == 0:
            continue
        emp  = float((y_group[mask] > q_t[mask]).mean())
        theo = 1.0 - q
        result[f"exceed_emp_{int(q*10000):05d}"]   = emp
        result[f"exceed_theo_{int(q*10000):05d}"]  = theo
        result[f"exceed_ratio_{int(q*10000):05d}"] = (
            emp / theo if theo > 1e-12 else np.nan
        )

    # Quantile Score
    for k, q in enumerate(q_levels):
        if quants_group.ndim == 2:
            q_t = quants_group[:, k]
        else:
            continue
        qs = (y_group - q_t) * (q - (y_group < q_t).astype(float))
        result[f"qs_mean_{int(q*10000):05d}"] = float(np.nanmean(qs))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Seasonal decomposition
# ─────────────────────────────────────────────────────────────────────────────

def seasonal_decomposition(
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    dynamic_quantiles: np.ndarray,
    quantile_levels: Sequence[float],
    return_levels: Optional[np.ndarray] = None,
    return_period_labels: Optional[List[str]] = None,
) -> Dict[int, dict]:
    """
    Compute per-month diagnostics for all 12 calendar months.

    Parameters
    ----------
    y                   : observed rainfall (n,)
    dates               : date index (n,)
    dynamic_quantiles   : (n, n_q) OOS quantile array
    quantile_levels     : probability levels (n_q,)
    return_levels       : optional (n, n_rl) return level array
    return_period_labels: labels for return level columns

    Returns
    -------
    dict mapping month (1-12) → metric dict
    """
    months = np.array([d.month for d in dates])
    result = {}

    for m in range(1, 13):
        mask = months == m
        if mask.sum() == 0:
            result[m] = {"n": 0}
            continue
        y_m  = y[mask]
        q_m  = dynamic_quantiles[mask, :]
        mets = _group_metrics(y_m, q_m, quantile_levels)

        # Monthly return level means
        if return_levels is not None and return_period_labels is not None:
            rl_m = return_levels[mask, :]
            for k, lbl in enumerate(return_period_labels):
                mets[f"mean_RL_{lbl}"] = float(np.nanmean(rl_m[:, k]))

        mets["month"] = m
        result[m] = mets

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ENSO decomposition
# ─────────────────────────────────────────────────────────────────────────────

def enso_decomposition(
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    nino34: np.ndarray,
    dynamic_quantiles: np.ndarray,
    quantile_levels: Sequence[float],
    return_levels: Optional[np.ndarray] = None,
    return_period_labels: Optional[List[str]] = None,
    n_intensity_bins: int = 5,
) -> Dict:
    """
    Compute diagnostics grouped by ENSO regime and ENSO intensity.

    Returns
    -------
    dict with keys:
        "standard_regimes" : {regime_label: metric_dict}
        "intensity_bins"   : {bin_idx: metric_dict}
        "bin_edges"        : list of quantile edges for intensity bins
    """
    enso_labels   = classify_enso_regime(nino34)
    intensity_bins, bin_edges = classify_enso_intensity(nino34, n_bins=n_intensity_bins)

    # ── Standard regime decomposition ─────────────────────────────────────
    regime_result = {}
    for regime in [ENSO_EL_NINO, ENSO_NEUTRAL, ENSO_LA_NINA]:
        mask = enso_labels == regime
        n_r  = mask.sum()
        if n_r == 0:
            regime_result[regime] = {"n": 0}
            continue
        y_r = y[mask]
        q_r = dynamic_quantiles[mask, :]
        mets = _group_metrics(y_r, q_r, quantile_levels)
        if return_levels is not None and return_period_labels is not None:
            rl_r = return_levels[mask, :]
            for k, lbl in enumerate(return_period_labels):
                mets[f"mean_RL_{lbl}"] = float(np.nanmean(rl_r[:, k]))
        mets["regime"] = regime
        regime_result[regime] = mets

    # ── Intensity bin decomposition ────────────────────────────────────────
    bin_result = {}
    for b in range(n_intensity_bins):
        mask = intensity_bins == b
        if mask.sum() == 0:
            bin_result[b] = {"n": 0, "bin_idx": b}
            continue
        y_b = y[mask]
        q_b = dynamic_quantiles[mask, :]
        mets = _group_metrics(y_b, q_b, quantile_levels)
        if return_levels is not None and return_period_labels is not None:
            rl_b = return_levels[mask, :]
            for k, lbl in enumerate(return_period_labels):
                mets[f"mean_RL_{lbl}"] = float(np.nanmean(rl_b[:, k]))
        mets["bin_idx"]       = b
        mets["nino34_range"]  = (
            float(bin_edges[b]),
            float(bin_edges[b + 1]),
        )
        # Mean shape parameter if present
        bin_result[b] = mets

    return {
        "standard_regimes": regime_result,
        "intensity_bins":   bin_result,
        "bin_edges":        bin_edges,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Wet / dry season decomposition
# ─────────────────────────────────────────────────────────────────────────────

def wet_dry_decomposition(
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    wet_months: List[int],
    dry_months: List[int],
    dynamic_quantiles: np.ndarray,
    quantile_levels: Sequence[float],
    return_levels: Optional[np.ndarray] = None,
    return_period_labels: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """
    Compute diagnostics for wet and dry seasons separately.

    Parameters
    ----------
    wet_months  : list of month numbers classified as wet
    dry_months  : list of month numbers classified as dry

    Returns
    -------
    dict with keys "wet" and "dry", each mapping to a metric dict.
    If wet_months or dry_months is empty, the corresponding key has {"n": 0}.
    """
    months = np.array([d.month for d in dates])
    result = {}

    for season, month_list in [("wet", wet_months), ("dry", dry_months)]:
        if not month_list:
            result[season] = {"n": 0, "season": season}
            continue
        mask = np.isin(months, month_list)
        if mask.sum() == 0:
            result[season] = {"n": 0, "season": season}
            continue
        y_s = y[mask]
        q_s = dynamic_quantiles[mask, :]
        mets = _group_metrics(y_s, q_s, quantile_levels)
        if return_levels is not None and return_period_labels is not None:
            rl_s = return_levels[mask, :]
            for k, lbl in enumerate(return_period_labels):
                mets[f"mean_RL_{lbl}"] = float(np.nanmean(rl_s[:, k]))
        mets["season"]      = season
        mets["month_list"]  = month_list
        result[season]      = mets

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_monthly_quantile_boxplot(
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    dynamic_quantiles: np.ndarray,
    quantile_levels: Sequence[float],
    highlight_q: float = 0.99,
    fig_path: Optional[str] = None,
    title: str = "",
) -> plt.Figure:
    """
    Monthly boxplots of the dynamic conditional 99th quantile.
    """
    months = np.array([d.month for d in dates])
    q_idx  = {q: i for i, q in enumerate(quantile_levels)}
    k      = q_idx.get(highlight_q, -1)
    if k < 0:
        k = -1   # use last (highest) quantile

    data_by_month = [dynamic_quantiles[months == m, k] for m in range(1, 13)]
    month_names   = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(10, 4))
    bp = ax.boxplot(
        data_by_month, positions=range(1, 13),
        patch_artist=True,
        boxprops=dict(facecolor="#aed6f1", color="#2980b9"),
        medianprops=dict(color="#e74c3c", lw=2),
        whiskerprops=dict(color="#2980b9"),
        flierprops=dict(marker="o", markersize=2, alpha=0.3, color="#7f8c8d"),
    )
    # Monthly mean observed rainfall (right axis)
    ax2 = ax.twinx()
    monthly_obs = [float(y[months == m].mean()) for m in range(1, 13)]
    ax2.bar(range(1, 13), monthly_obs, color="#f39c12", alpha=0.25, width=0.4, label="Mean obs.")
    ax2.set_ylabel("Mean obs. rainfall (mm)", color="#f39c12")
    ax2.tick_params(axis="y", labelcolor="#f39c12")

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Q{highlight_q} (mm)")
    ax.set_title(title or f"Monthly distribution of Q{highlight_q}")
    plt.tight_layout()

    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_enso_exceedance(
    enso_regimes: Dict,
    quantile_levels: Sequence[float],
    highlight_levels: Sequence[float] = (0.90, 0.95, 0.99),
    fig_path: Optional[str] = None,
    title: str = "",
) -> plt.Figure:
    """
    Bar chart: empirical exceedance frequency by ENSO regime for each quantile.
    """
    regimes = [ENSO_EL_NINO, ENSO_NEUTRAL, ENSO_LA_NINA]
    colors  = [ENSO_COLORS[r] for r in regimes]
    hl      = [q for q in highlight_levels if q in set(quantile_levels)]
    x       = np.arange(len(hl))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, regime in enumerate(regimes):
        mets = enso_regimes.get(regime, {})
        vals = [
            mets.get(f"exceed_emp_{int(q*10000):05d}", np.nan)
            for q in hl
        ]
        ax.bar(x + i * width, vals, width, label=regime, color=colors[i], alpha=0.85)

    # Reference lines (theoretical exceedance)
    for j, q in enumerate(hl):
        ax.hlines(1.0 - q, j * 3 * width - 0.1, j * 3 * width + 3 * width + 0.1,
                  colors="black", linestyles="--", lw=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Q{q}" for q in hl])
    ax.set_ylabel("Empirical exceedance frequency")
    ax.set_title(title or "ENSO regime: exceedance calibration")
    ax.legend(fontsize=8)
    plt.tight_layout()

    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_enso_intensity_quantile(
    intensity_result: Dict,
    quantile_levels: Sequence[float],
    highlight_q: float = 0.99,
    fig_path: Optional[str] = None,
    title: str = "",
) -> plt.Figure:
    """
    Line/scatter plot: mean dynamic quantile vs. ENSO intensity bin.
    """
    bins   = sorted(intensity_result.keys())
    values = [
        intensity_result[b].get(f"mean_Q{highlight_q}", np.nan)
        for b in bins
    ]
    nino_mids = [
        float(np.mean(intensity_result[b].get("nino34_range", (0, 0))))
        for b in bins
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(nino_mids, values, "o-", color="#e74c3c", lw=1.5, ms=6)
    ax.set_xlabel("Niño 3.4 bin centre (°C)")
    ax.set_ylabel(f"Mean Q{highlight_q} (mm)")
    ax.axvline(0, color="grey", lw=0.8, linestyle="--")
    ax.set_title(title or f"ENSO intensity vs. mean Q{highlight_q}")
    plt.tight_layout()

    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Save decomposition results to CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_decomposition_csvs(
    out_dir,
    prefix: str,
    seasonal_result: Dict,
    enso_result: Dict,
    wet_dry_result: Dict,
    wet_dry_info: Dict,
) -> None:
    """
    Save all decomposition metric dicts to individual CSV files.
    """
    from pathlib import Path as _P

    out_dir = _P(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Seasonal ───────────────────────────────────────────────────────────
    rows = []
    for m, mets in seasonal_result.items():
        row = {"month": m}
        row.update({k: v for k, v in mets.items() if not isinstance(v, (list, dict))})
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / f"{prefix}_seasonal.csv", index=False)

    # ── ENSO regimes ───────────────────────────────────────────────────────
    rows_regime = []
    for regime, mets in enso_result.get("standard_regimes", {}).items():
        row = {"regime": regime}
        row.update({k: v for k, v in mets.items() if not isinstance(v, (list, dict, tuple))})
        rows_regime.append(row)
    if rows_regime:
        pd.DataFrame(rows_regime).to_csv(
            out_dir / f"{prefix}_enso_regimes.csv", index=False
        )

    # ── ENSO intensity ─────────────────────────────────────────────────────
    rows_int = []
    for b, mets in enso_result.get("intensity_bins", {}).items():
        row = {"bin": b}
        rng = mets.get("nino34_range", (np.nan, np.nan))
        row["nino34_lo"] = float(rng[0]) if len(rng) > 0 else np.nan
        row["nino34_hi"] = float(rng[1]) if len(rng) > 1 else np.nan
        row.update({k: v for k, v in mets.items()
                    if not isinstance(v, (list, dict, tuple))})
        rows_int.append(row)
    if rows_int:
        pd.DataFrame(rows_int).to_csv(
            out_dir / f"{prefix}_enso_intensity.csv", index=False
        )

    # ── Wet/dry ────────────────────────────────────────────────────────────
    rows_wd = []
    for season, mets in wet_dry_result.items():
        row = {"season": season}
        row["months"] = str(mets.get("month_list", []))
        row.update({k: v for k, v in mets.items()
                    if not isinstance(v, (list, dict, tuple)) and k not in ("season",)})
        rows_wd.append(row)
    if rows_wd:
        pd.DataFrame(rows_wd).to_csv(
            out_dir / f"{prefix}_wet_dry.csv", index=False
        )

    # ── Wet/dry classification summary ────────────────────────────────────
    if wet_dry_info:
        classification_rows = []
        for m, cls in wet_dry_info.get("classification", {}).items():
            classification_rows.append({
                "month": m,
                "classification": cls,
                "monthly_mean_mm": wet_dry_info.get("monthly_mean", {}).get(m, np.nan),
                "grand_mean_mm":   wet_dry_info.get("grand_mean", np.nan),
            })
        if classification_rows:
            pd.DataFrame(classification_rows).to_csv(
                out_dir / f"{prefix}_season_classification.csv", index=False
            )
