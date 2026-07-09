"""
Dynamic quantile and return-level diagnostics for ZA-GAS models.

=============================================================================
THEORY
=============================================================================

For each day t in the evaluation sample, the model produces a time-varying
predictive CDF:

    F_t(y) = (1 - pi_t) + pi_t * G_t(y ; theta_t)

where pi_t is the occurrence probability and G_t is the GB2 CDF with
TV parameters evaluated at the filtered state f_t.

Dynamic Conditional Quantile
----------------------------
The τ-quantile of F_t is:

    Q_t(τ) = F_t^{-1}(τ)

For τ ≤ (1 - pi_t):  Q_t(τ) = 0  (mass at zero dominates)
For τ > (1 - pi_t):  Q_t(τ) = G_t^{-1}((τ - (1-pi_t)) / pi_t)

Daily Return Level (prompt §: Daily rarity)
--------------------------------------------
The T-day conditional return level is:

    RL_{T,t} = F_t^{-1}(1 - 1/T)

This is a quantile at the exceedance probability 1/T.

Climatological Return Level (prompt §: Climatological)
-------------------------------------------------------
For a T-year return period, the daily non-exceedance probability is:

    p(T) = (1 - 1/T)^{1/365}

and the conditional return level is:

    RL_{T,t} = F_t^{-1}(p(T))

=============================================================================
USAGE
=============================================================================

All functions operate on the OOS (or IS) paths dictionary returned by
model.simulate_oos() or loaded from paths.npz + metadata.json.

    from diagnostics.dynamics import compute_dynamic_quantiles
    quants = compute_dynamic_quantiles(model, oos_paths, QUANTILE_LEVELS)
    # quants[i, k] = Q_t(τ_k) for test day i and quantile level τ_k
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Standard quantile levels requested in prompt.md (0.98 added 2026-07-08:
# objective 4f explicitly requests q50/q75/q90/q95/q98/q99, and 0.975 alone
# does not cover q98).
QUANTILE_LEVELS = [0.50, 0.75, 0.90, 0.95, 0.975, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999]

# Daily return periods (days)
DAILY_RETURN_PERIODS = [30, 60, 90, 100, 500, 1000]

# Climatological return periods (years)
CLIM_RETURN_PERIODS_YEARS = [2, 5, 10, 25, 50, 75, 100]


def _predictive_quantile_from_paths(
    model,
    paths: dict,
    idx: int,
    q: float,
) -> float:
    """
    Compute Q_t(q) = F_t^{-1}(q) at evaluation index idx.

    Uses the distribution's ppf if available; falls back to 0.0 for
    quantiles below the zero mass.

    Parameters
    ----------
    model  : fitted ZA model with model.dist having .ppf()
    paths  : OOS (or IS) paths dict with keys pi_oos / pi, f_arr_oos / f_arr,
             tv_names, static
    idx    : observation index within the effective evaluation sample
    q      : probability level in (0, 1)

    Returns
    -------
    float quantile value (≥ 0)
    """
    # Support both OOS key format (pi_oos, f_arr_oos) and IS key format (pi, f_arr)
    if "pi_oos" in paths:
        pi_t   = float(paths["pi_oos"][idx])
        f_row  = paths["f_arr_oos"][idx]
    else:
        pi_t   = float(paths["pi"][idx])
        f_row  = paths["f_arr"][idx]

    zero_mass = max(0.0, 1.0 - pi_t)

    if q <= zero_mass:
        return 0.0

    tv_names = paths["tv_names"]
    static   = paths["static"]
    call = {name: float(f_row[j]) for j, name in enumerate(tv_names)}
    call.update(static)

    q_pos = np.clip((q - zero_mass) / max(pi_t, 1e-12), 1e-12, 1.0 - 1e-12)

    if hasattr(model.dist, "ppf"):
        try:
            val = float(model.dist.ppf(q_pos, **call))
            return max(0.0, val)
        except Exception:
            pass

    return np.nan


def compute_dynamic_quantiles(
    model,
    paths: dict,
    quantile_levels: Sequence[float] = QUANTILE_LEVELS,
) -> np.ndarray:
    """
    Compute time-varying conditional quantiles from evaluation paths.

    Parameters
    ----------
    model           : fitted ZA model
    paths           : paths dict (OOS or IS)
    quantile_levels : probability levels at which to compute quantiles

    Returns
    -------
    ndarray of shape (n_eval, n_quantiles)
        quants[i, k] = Q_{t_i}(τ_k)
    """
    # Determine sample size
    if "pi_oos" in paths:
        n = len(paths["pi_oos"])
    else:
        n = len(paths["pi"])

    q_levels = list(quantile_levels)
    result   = np.full((n, len(q_levels)), np.nan)

    for i in range(n):
        for k, q in enumerate(q_levels):
            result[i, k] = _predictive_quantile_from_paths(model, paths, i, q)

    return result


def daily_return_level_quantile(T_days: int) -> float:
    """Convert a T-day return period to the corresponding quantile level."""
    return 1.0 - 1.0 / T_days


def climatological_return_level_quantile(T_years: float) -> float:
    """
    Convert a T-year return period to the daily quantile level.

    p(T) = (1 - 1/T)^{1/365}
    """
    return (1.0 - 1.0 / T_years) ** (1.0 / 365.0)


def compute_daily_return_levels(
    model,
    paths: dict,
    return_periods_days: Sequence[int] = DAILY_RETURN_PERIODS,
) -> Tuple[np.ndarray, List[float]]:
    """
    Compute dynamic T-day conditional return levels.

    Returns
    -------
    rl_arr   : ndarray of shape (n_eval, n_periods)
    q_levels : corresponding quantile levels
    """
    q_levels = [daily_return_level_quantile(T) for T in return_periods_days]
    rl_arr   = compute_dynamic_quantiles(model, paths, q_levels)
    return rl_arr, q_levels


def compute_climatological_return_levels(
    model,
    paths: dict,
    return_periods_years: Sequence[float] = CLIM_RETURN_PERIODS_YEARS,
) -> Tuple[np.ndarray, List[float]]:
    """
    Compute dynamic climatological return levels using p(T) = (1-1/T)^{1/365}.

    Returns
    -------
    rl_arr   : ndarray of shape (n_eval, n_periods)
    q_levels : corresponding quantile levels
    """
    q_levels = [climatological_return_level_quantile(T) for T in return_periods_years]
    rl_arr   = compute_dynamic_quantiles(model, paths, q_levels)
    return rl_arr, q_levels


# ─────────────────────────────────────────────────────────────────────────────
# ENSO shading helpers
# ─────────────────────────────────────────────────────────────────────────────

def classify_enso(
    nino34: np.ndarray,
    threshold: float = 0.5,
    window: int = 90,
) -> np.ndarray:
    """
    Classify each time step into El Niño / Neutral / La Niña.

    Uses a rolling 90-day mean of the Niño 3.4 index:
        El Niño : mean ≥ +threshold
        La Niña : mean ≤ -threshold
        Neutral : otherwise

    Parameters
    ----------
    nino34    : raw daily Niño 3.4 SST anomaly array
    threshold : classification boundary (default 0.5°C per NOAA)
    window    : rolling window length in days (default 90)

    Returns
    -------
    ndarray of dtype str: "el_nino", "neutral", "la_nina"
    """
    series     = pd.Series(nino34)
    roll_mean  = series.rolling(window=window, min_periods=1).mean()
    labels     = np.full(len(nino34), "neutral", dtype=object)
    labels[roll_mean.values >=  threshold] = "el_nino"
    labels[roll_mean.values <= -threshold] = "la_nina"
    return labels


def _add_enso_shading(ax, dates, enso_labels):
    """Add El Niño (light red) and La Niña (light blue) background shading."""
    if dates is None or enso_labels is None:
        return

    date_arr = np.asarray(dates, dtype="datetime64[D]")
    n = len(date_arr)

    def _shade_spans(mask, color, alpha=0.18):
        in_span = False
        span_start = None
        for i in range(n):
            if mask[i] and not in_span:
                span_start = date_arr[i]
                in_span = True
            elif not mask[i] and in_span:
                ax.axvspan(span_start, date_arr[i - 1], color=color, alpha=alpha, lw=0)
                in_span = False
        if in_span:
            ax.axvspan(span_start, date_arr[-1], color=color, alpha=alpha, lw=0)

    _shade_spans(enso_labels == "el_nino", "red")
    _shade_spans(enso_labels == "la_nina", "blue")


# ─────────────────────────────────────────────────────────────────────────────
# Signature figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_signature_figure(
    dates: np.ndarray,
    y: np.ndarray,
    dynamic_quantiles: np.ndarray,
    quantile_levels: Sequence[float],
    nino34: Optional[np.ndarray] = None,
    tv_param: Optional[np.ndarray] = None,
    tv_param_name: str = "phi",
    long_component: Optional[np.ndarray] = None,
    short_component: Optional[np.ndarray] = None,
    fig_path: Optional[str] = None,
    title: str = "",
    highlight_quantiles: Sequence[float] = (0.95, 0.99, 0.999),
) -> plt.Figure:
    """
    Publication-quality signature figure.

    Panel 1 (top): observed rainfall + dynamic 95/99/99.9% quantiles,
                   with El Niño (red) / La Niña (blue) background shading.
    Panel 2 (bottom): dynamic tail parameter phi_t (and long/short components
                      if Harvey model).

    Parameters
    ----------
    dates              : date array (length n)
    y                  : observed rainfall (length n)
    dynamic_quantiles  : (n, n_quantiles) quantile array
    quantile_levels    : must match columns of dynamic_quantiles
    nino34             : optional Niño 3.4 values for ENSO shading
    tv_param           : time-varying parameter (phi_t) for panel 2
    tv_param_name      : label for the TV parameter
    long_component     : Harvey long component L_t
    short_component    : Harvey short component S_t
    fig_path           : if given, save figure to this path
    title              : figure title
    highlight_quantiles: quantile levels to display (must be in quantile_levels)
    """
    q_idx = {q: i for i, q in enumerate(quantile_levels)}
    hi_q  = [q for q in highlight_quantiles if q in q_idx]
    colors = ["#e74c3c", "#c0392b", "#7f0000"]  # shades of red for quantile lines

    enso_labels = classify_enso(nino34) if nino34 is not None else None

    n_panels = 2 if tv_param is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels),
                              sharex=True, gridspec_kw={"hspace": 0.08})
    if n_panels == 1:
        axes = [axes]

    # ── Panel 1: observed + quantile lines ───────────────────────────────────
    ax = axes[0]
    if enso_labels is not None:
        _add_enso_shading(ax, dates, enso_labels)

    ax.bar(dates, y, color="#3498db", alpha=0.4, width=1, label="Observed")

    for q, color in zip(hi_q, colors):
        k = q_idx[q]
        label = f"Q({q:.3%})".replace("%", r"\%") if False else f"Q={q}"
        ax.plot(dates, dynamic_quantiles[:, k], color=color, lw=0.9,
                label=f"Q{q}", alpha=0.85)

    ax.set_ylabel("Rainfall (mm)")
    ax.legend(loc="upper left", fontsize=8, ncol=min(4, len(hi_q) + 1))
    if title:
        ax.set_title(title, fontsize=11)

    # ENSO legend patches
    if enso_labels is not None:
        patches = [
            mpatches.Patch(color="red",  alpha=0.3, label="El Niño"),
            mpatches.Patch(color="blue", alpha=0.3, label="La Niña"),
        ]
        ax.legend(handles=ax.get_legend_handles_labels()[0] + patches,
                  labels=ax.get_legend_handles_labels()[1] + ["El Niño", "La Niña"],
                  loc="upper left", fontsize=7, ncol=min(5, len(hi_q) + 3))

    # ── Panel 2: tail parameter ───────────────────────────────────────────────
    if tv_param is not None:
        ax2 = axes[1]
        if enso_labels is not None:
            _add_enso_shading(ax2, dates, enso_labels)

        ax2.plot(dates, tv_param, color="black", lw=0.9,
                 label=f"$\\hat{{\\{tv_param_name}}}_t$")

        if long_component is not None:
            ax2.plot(dates, long_component, color="#27ae60", lw=1.0,
                     linestyle="--", label="Long $L_t$", alpha=0.8)
        if short_component is not None:
            ax2.plot(dates, short_component, color="#e67e22", lw=0.8,
                     linestyle=":", label="Short $S_t$", alpha=0.8)

        ax2.set_ylabel(f"${tv_param_name}_t$")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.set_xlabel("Date")

    plt.tight_layout()

    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic quantile time-series figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_dynamic_quantiles(
    dates: np.ndarray,
    y: np.ndarray,
    dynamic_quantiles: np.ndarray,
    quantile_levels: Sequence[float],
    fig_path: Optional[str] = None,
    title: str = "",
) -> plt.Figure:
    """
    Plot observed rainfall against a fan of dynamic conditional quantiles.
    """
    n_q    = dynamic_quantiles.shape[1]
    cmap   = plt.cm.get_cmap("RdYlBu_r", n_q)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(dates, y, color="#3498db", alpha=0.35, width=1, label="Observed", zorder=2)

    for k, q in enumerate(quantile_levels):
        ax.plot(dates, dynamic_quantiles[:, k], color=cmap(k / max(n_q - 1, 1)),
                lw=0.7, alpha=0.85, label=f"Q{q}")

    ax.set_ylabel("Rainfall (mm)")
    ax.set_xlabel("Date")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7, ncol=5)
    plt.tight_layout()

    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Return level figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_return_levels(
    dates: np.ndarray,
    y: np.ndarray,
    return_levels: np.ndarray,
    period_labels: List[str],
    fig_path: Optional[str] = None,
    title: str = "",
    nino34: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Plot dynamic return levels through time.
    """
    n_periods = return_levels.shape[1]
    cmap = plt.cm.get_cmap("plasma", n_periods)
    enso_labels = classify_enso(nino34) if nino34 is not None else None

    fig, ax = plt.subplots(figsize=(14, 5))
    if enso_labels is not None:
        _add_enso_shading(ax, dates, enso_labels)

    ax.bar(dates, y, color="#3498db", alpha=0.3, width=1, label="Observed", zorder=2)

    for k, label in enumerate(period_labels):
        ax.plot(dates, return_levels[:, k], color=cmap(k / max(n_periods - 1, 1)),
                lw=0.9, alpha=0.85, label=label)

    ax.set_ylabel("Rainfall (mm)")
    ax.set_xlabel("Date")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=min(4, n_periods + 1))
    plt.tight_layout()

    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tail calibration
# ─────────────────────────────────────────────────────────────────────────────

def compute_exceedance_frequencies(
    y: np.ndarray,
    dynamic_quantiles: np.ndarray,
    quantile_levels: Sequence[float],
) -> Dict[str, float]:
    """
    Compare empirical exceedance rates to theoretical exceedance probabilities.

    For each quantile level τ and the corresponding dynamic quantile Q_t(τ),
    the theoretical exceedance probability is 1 - τ.
    The empirical rate is the fraction of observations y_t > Q_t(τ).

    Returns dict mapping f"exceed_{int(q*10000)}" → empirical frequency.
    """
    n = len(y)
    result = {}
    for k, q in enumerate(quantile_levels):
        q_t     = dynamic_quantiles[:, k]
        mask    = np.isfinite(q_t)
        if mask.sum() == 0:
            empirical = np.nan
        else:
            empirical = float((y[mask] > q_t[mask]).mean())
        theoretical = 1.0 - q
        result[f"exceed_emp_{int(q*10000):05d}"] = empirical
        result[f"exceed_theo_{int(q*10000):05d}"] = theoretical
        result[f"exceed_ratio_{int(q*10000):05d}"] = (
            empirical / theoretical if theoretical > 0 else np.nan
        )
    return result


def plot_tail_calibration(
    y: np.ndarray,
    dynamic_quantiles: np.ndarray,
    quantile_levels: Sequence[float],
    fig_path: Optional[str] = None,
    title: str = "",
    subset_label: Optional[str] = None,
) -> plt.Figure:
    """
    Plot empirical vs. theoretical exceedance frequencies.

    A well-calibrated model lies on the 45° line.
    """
    exceedances = compute_exceedance_frequencies(y, dynamic_quantiles, quantile_levels)
    theoretical = [1.0 - q for q in quantile_levels]
    empirical   = [
        exceedances.get(f"exceed_emp_{int(q*10000):05d}", np.nan)
        for q in quantile_levels
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, max(theoretical) * 1.1],
            [0, max(theoretical) * 1.1], "k--", lw=1, label="Perfect calibration")
    ax.scatter(theoretical, empirical, color="#e74c3c", zorder=3, s=40)
    for t, e, q in zip(theoretical, empirical, quantile_levels):
        if np.isfinite(e):
            ax.annotate(f"{q:.4f}", (t, e), fontsize=6,
                        textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("Theoretical exceedance prob. (1 − τ)")
    ax.set_ylabel("Empirical exceedance frequency")
    label = f"Tail calibration{' — ' + subset_label if subset_label else ''}"
    ax.set_title(title or label)
    ax.legend(fontsize=8)
    plt.tight_layout()

    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Save / load dynamic arrays
# ─────────────────────────────────────────────────────────────────────────────

def save_dynamic_arrays(
    out_dir,
    model_id: str,
    dates_train, dates_test,
    quants_oos: np.ndarray,
    quants_is: Optional[np.ndarray],
    daily_rl: np.ndarray,
    clim_rl: np.ndarray,
    quantile_levels: Sequence[float],
) -> None:
    """
    Save dynamic quantile and return-level arrays as compressed NumPy + CSV.

    Saved files:
        {model_id}_quants_oos.npz   — shape (n_test, n_q)
        {model_id}_quants_oos.csv   — with date index
        {model_id}_daily_rl_oos.csv — daily return levels (test)
        {model_id}_clim_rl_oos.csv  — climatological return levels (test)
    """
    out_dir = Path(out_dir) if not hasattr(out_dir, "mkdir") else out_dir
    from pathlib import Path as _P

    out_dir = _P(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / model_id

    # OOS quantiles
    np.savez_compressed(
        str(prefix) + "_quants_oos.npz",
        quants_oos=quants_oos,
        quantile_levels=np.array(quantile_levels),
    )
    df_q = pd.DataFrame(
        quants_oos,
        index=dates_test,
        columns=[f"q{q}" for q in quantile_levels],
    )
    df_q.index.name = "date"
    df_q.to_csv(str(prefix) + "_quants_oos.csv")

    # IS quantiles (if available)
    if quants_is is not None:
        df_qi = pd.DataFrame(
            quants_is,
            index=dates_train[-len(quants_is):],
            columns=[f"q{q}" for q in quantile_levels],
        )
        df_qi.index.name = "date"
        df_qi.to_csv(str(prefix) + "_quants_is.csv")

    # Daily return levels
    rl_cols = [f"RL_{T}d" for T in DAILY_RETURN_PERIODS]
    df_drl  = pd.DataFrame(daily_rl, index=dates_test, columns=rl_cols)
    df_drl.index.name = "date"
    df_drl.to_csv(str(prefix) + "_daily_rl_oos.csv")

    # Climatological return levels
    clim_cols = [f"RL_{T}yr" for T in CLIM_RETURN_PERIODS_YEARS]
    df_crl    = pd.DataFrame(clim_rl, index=dates_test, columns=clim_cols)
    df_crl.index.name = "date"
    df_crl.to_csv(str(prefix) + "_clim_rl_oos.csv")
