"""
Diagnostic plots.

Single-dataset functions return a matplotlib Figure.
Multi-dataset functions return a mosaic Figure (one panel per location).

Plots provided:
  - quantile_residual_panel : ACF, QQ, time series, PIT histogram
  - pit_histogram            : histogram of PIT values vs Uniform(0,1)
  - qq_plot                  : normal QQ plot of quantile residuals
  - acf_plot                 : autocorrelation of quantile residuals
  - filtered_paths_plot      : phi_t, xi_t, pi_t over time
  - mosaic()                 : wraps any single-plot function into a grid
"""

from __future__ import annotations
from typing import Callable, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf


# ===========================================================================
# Single-location panels
# ===========================================================================

def quantile_residual_panel(
    residuals: np.ndarray,
    pit: np.ndarray,
    dates=None,
    location: str = "",
    n_lags: int = 40,
) -> plt.Figure:
    """
    4-panel diagnostic figure for one location:
      [ACF of r_t]  [QQ plot]  [r_t time series]  [PIT histogram]
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Quantile residual diagnostics — {location}", fontsize=13)

    r  = residuals[np.isfinite(residuals)]
    tt = dates if dates is not None else np.arange(len(r))

    # --- ACF ---
    ax = axes[0, 0]
    plot_acf(r, ax=ax, lags=n_lags, title="ACF of quantile residuals",
             alpha=0.05, zero=False)
    ax.axhline(0, color="k", lw=0.5)

    # --- QQ plot ---
    ax = axes[0, 1]
    percs = np.linspace(0.5, 99.5, 200)
    q_emp = np.percentile(r, percs)
    q_th  = norm.ppf(percs / 100.0)
    ax.scatter(q_th, q_emp, s=8, alpha=0.6, color="steelblue")
    lim = max(abs(q_emp).max(), abs(q_th).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax.set_xlabel("Theoretical N(0,1) quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.set_title("Normal QQ plot")

    # --- Time series of residuals ---
    ax = axes[1, 0]
    ax.plot(tt[:len(r)], r, lw=0.8, color="steelblue", alpha=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1.96, color="r", lw=0.8, ls="--", label="±1.96")
    ax.axhline(-1.96, color="r", lw=0.8, ls="--")
    ax.set_title("Quantile residuals over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("$r_t$")
    ax.legend(fontsize=8)

    # --- PIT histogram ---
    ax = axes[1, 1]
    u = pit[np.isfinite(pit)]
    ax.hist(u, bins=20, density=True, color="steelblue", alpha=0.7,
            edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="r", lw=1.2, ls="--", label="Uniform(0,1)")
    ax.set_xlim(0, 1)
    ax.set_title("PIT histogram")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def filtered_paths_plot(
    paths: dict,
    dates=None,
    location: str = "",
) -> plt.Figure:
    """
    Plot the filtered state paths: phi_t, xi_t (if available), pi_t.

    Parameters
    ----------
    paths : dict returned by ZAGASModel.filter()
    """
    tv_names = paths.get("tv_names", [])
    n_tv     = len(tv_names)
    n_panels = n_tv + 1   # +1 for pi_t
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(f"Filtered state paths — {location}", fontsize=13)
    n = paths["f_arr"].shape[0]
    tt = dates if dates is not None else np.arange(n)

    for j, name in enumerate(tv_names):
        axes[j].plot(tt, paths["f_arr"][:, j], lw=0.9, color="steelblue")
        axes[j].set_ylabel(f"$\\{name}_t$" if name in ("phi", "xi") else name)
        axes[j].set_title(f"Filtered {name}_t")

    ax_pi = axes[-1]
    ax_pi.plot(tt, paths["pi"], lw=0.9, color="darkorange")
    ax_pi.set_ylabel(r"$\pi_t$")
    ax_pi.set_title("Probability of non-zero observation $\\pi_t$")
    ax_pi.set_ylim(0, 1)
    ax_pi.set_xlabel("Time")

    fig.tight_layout()
    return fig


# ===========================================================================
# Mosaic helper — wraps any single-plot function into a grid of subplots
# ===========================================================================

def mosaic(
    plot_fn: Callable,
    datasets: Dict[str, dict],
    ncols: int = 2,
    fn_kwargs: dict | None = None,
) -> plt.Figure:
    """
    Arrange one plot per location in a mosaic grid.

    Parameters
    ----------
    plot_fn   : function that returns a Figure (e.g. quantile_residual_panel)
    datasets  : {location_name: kwargs_for_plot_fn}
    ncols     : number of columns in the grid
    fn_kwargs : extra keyword arguments forwarded to plot_fn

    Returns
    -------
    A single Figure with all individual figures arranged in a grid.

    Usage example::

        ds = {
            "Belo Horizonte": dict(residuals=r1, pit=u1, location="Belo Horizonte"),
            "Manaus":         dict(residuals=r2, pit=u2, location="Manaus"),
        }
        fig = mosaic(quantile_residual_panel, ds, ncols=2)
    """
    fn_kwargs = fn_kwargs or {}
    locations  = list(datasets.keys())
    n          = len(locations)
    nrows      = int(np.ceil(n / ncols))

    # Generate individual figures to measure sub-figure size
    figs = []
    for loc in locations:
        kw  = {**datasets[loc], **fn_kwargs}
        kw.setdefault("location", loc)
        f   = plot_fn(**kw)
        figs.append(f)

    # Infer individual figure dimensions
    w0, h0 = figs[0].get_size_inches()
    fig_mosaic = plt.figure(figsize=(w0 * ncols, h0 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig_mosaic,
                           hspace=0.45, wspace=0.35)

    for idx, (loc, sub_fig) in enumerate(zip(locations, figs)):
        r, c = divmod(idx, ncols)
        # Copy each axes from the sub-figure into the mosaic
        sub_axes = sub_fig.get_axes()
        n_ax     = len(sub_axes)
        if n_ax == 1:
            inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[r, c])
            ax_mosaic = fig_mosaic.add_subplot(inner[0])
            ax_mosaic.set_axis_off()
        else:
            # Determine inner grid layout from sub_fig geometry
            rows_sub = int(np.ceil(np.sqrt(n_ax)))
            cols_sub = int(np.ceil(n_ax / rows_sub))
            inner    = gridspec.GridSpecFromSubplotSpec(
                rows_sub, cols_sub, subplot_spec=gs[r, c],
                hspace=0.5, wspace=0.4
            )
            for k, sub_ax in enumerate(sub_axes):
                ri, ci = divmod(k, cols_sub)
                ax_m   = fig_mosaic.add_subplot(inner[ri, ci])
                # Copy content by re-plotting from sub_ax artists
                for line in sub_ax.get_lines():
                    ax_m.plot(line.get_xdata(), line.get_ydata(),
                              color=line.get_color(), lw=line.get_linewidth(),
                              ls=line.get_linestyle(), alpha=line.get_alpha() or 1)
                for patch in sub_ax.patches:
                    ax_m.add_patch(
                        plt.Rectangle(
                            (patch.get_x(), patch.get_y()),
                            patch.get_width(), patch.get_height(),
                            color=patch.get_facecolor(),
                            edgecolor=patch.get_edgecolor(),
                            alpha=patch.get_alpha() or 1,
                        )
                    )
                ax_m.set_xlim(sub_ax.get_xlim())
                ax_m.set_ylim(sub_ax.get_ylim())
                ax_m.set_title(sub_ax.get_title(), fontsize=8)
                ax_m.set_xlabel(sub_ax.get_xlabel(), fontsize=7)
                ax_m.set_ylabel(sub_ax.get_ylabel(), fontsize=7)
                ax_m.tick_params(labelsize=7)

        fig_mosaic.text(
            (c + 0.5) / ncols, (nrows - r - 0.02) / nrows,
            loc, ha="center", va="top", fontsize=9, fontweight="bold",
            transform=fig_mosaic.transFigure,
        )
        plt.close(sub_fig)

    return fig_mosaic


# ===========================================================================
# Convenience: ACF and QQ as standalone figures (for single-location use)
# ===========================================================================

def acf_plot(
    residuals: np.ndarray,
    n_lags: int = 40,
    location: str = "",
) -> plt.Figure:
    """Standalone ACF plot of quantile residuals."""
    fig, ax = plt.subplots(figsize=(8, 4))
    r = residuals[np.isfinite(residuals)]
    plot_acf(r, ax=ax, lags=n_lags, alpha=0.05, zero=False,
             title=f"ACF of quantile residuals — {location}")
    ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    return fig


def qq_plot(residuals: np.ndarray, location: str = "") -> plt.Figure:
    """Standalone normal QQ plot."""
    fig, ax = plt.subplots(figsize=(5, 5))
    r = np.sort(residuals[np.isfinite(residuals)])
    n = len(r)
    q_th = norm.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n))
    ax.scatter(q_th, r, s=8, alpha=0.6, color="steelblue")
    lim = max(abs(r).max(), abs(q_th).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax.set_xlabel("Theoretical N(0,1) quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.set_title(f"Normal QQ — {location}")
    fig.tight_layout()
    return fig
