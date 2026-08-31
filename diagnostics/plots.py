"""
Diagnostic plots.

Single-dataset functions return a matplotlib Figure.
Multi-dataset functions return a mosaic Figure (one panel per location).

Plots provided:
  - quantile_residual_panel : ACF, QQ, time series, PIT histogram (one location)
  - qq_plot                  : normal QQ plot of quantile residuals (one location)
  - acf_plot                 : autocorrelation of quantile residuals (one location)
  - filtered_paths_plot      : phi_t, xi_t, pi_t over time (one location)
  - mosaic()                 : generic wrapper -- re-plots any single-plot
                               function's axes into a grid, one cell per
                               location (kept for backward compatibility;
                               copies artists out of throwaway sub-figures)

Cross-location stage-diagnostic mosaics (added for the multi-location
Stage 1-4 "best alternative" comparison -- see
multilocation_diagnostic_mosaics.ipynb):
  - stage_pit_grid_mosaic    : Stages 1-3, one chart type, one branch --
                               rows = location, columns = stage
  - stage_qq_grid_mosaic     : same layout, normal QQ plots
  - stage_acf_grid_mosaic    : same layout, 400-lag ACF panels
  - stage4_composite_mosaic  : Stage 4 (no branch split) -- rows = location,
                               columns = [PIT, QQ, ACF] in one figure

Monthly pi_t seasonality (fixed 3x2 grid, one panel per location -- see
diagnostics/pi_monthly.py for how each location's monthly averages are
computed from the Stage 1 phi-only winner's in-sample filtered path):
  - monthly_pi_mosaic             : average in-sample pi_t by calendar
                                     month, one bar-chart panel per location
  - monthly_pi_occurrence_mosaic  : same, plus the empirical wet-day
                                     fraction by month as a second bar
                                     (pi_t in blue, empirical fraction in
                                     red) so the two are directly comparable

Generic location-grid QQ/ACF mosaics (fixed 2-column grid, one panel per
location, ONE model/variant per figure -- see diagnostics/pi_dynamics_alt_
mosaics.py for the occurrence-dynamics-alternatives use case that added
these, but they take a plain {location: quantile-residual array} dict so
any future single-model, all-locations diagnostic can reuse them too):
  - qq_location_mosaic   : normal QQ plot per location
  - acf_location_mosaic  : 400-lag (or n_lags) ACF per location

All four stage/branch mosaics put location on the row axis and the lower-cardinality axis
(stage, or chart type) on columns -- this bounds every figure to at most
3 columns no matter how many locations exist, so each panel keeps a
large, fixed share of the page width. These are sized for full-width or
full-page LaTeX placement, not on-screen thumbnails; a large, tall figure
is the intended trade-off for readability.

Unlike mosaic(), these draw directly onto a shared GridSpec (no throwaway
sub-figures, no artist copying), because their per-panel layout
requirements are fixed and known in advance (fixed y-limits, a shaded
confidence band, a specific lag count) rather than generic. All four share
three private single-panel drawers (_draw_pit_panel, _draw_qq_panel,
_draw_acf_panel) so the statistics plotted in a "PIT panel" are defined in
exactly one place regardless of which grid it appears in.

=============================================================================
THEORY -- shared by all three cross-location diagnostics
=============================================================================
Let r_t = Phi^{-1}(F_t(y_t)) be the quantile residual and u_t = F_t(y_t)
the PIT value at time t (see diagnostics/residuals.py for the full
zero-augmented derivation). Under a correctly specified model:

    u_t ~ iid Uniform(0, 1)          =>  PIT histogram should be flat
    r_t ~ iid N(0, 1)                =>  QQ plot should lie on the y = x line
    Corr(r_t, r_{t-k}) = 0  for k>0  =>  ACF should lie inside its null band

ACF confidence band. Under the null of no autocorrelation, the sample
autocorrelation rho_hat_k at lag k is asymptotically Normal with variance
approximately 1/n (Bartlett's approximation for a white-noise series, n =
number of residuals). A two-sided 95% band is therefore

    rho_hat_k in [-1.96/sqrt(n), +1.96/sqrt(n)]  ~=  +/- 2/sqrt(n),

which is the commonly used +/-2/sqrt(n) rule requested for these figures.
Each location generally has a different effective sample size n (models
have different warm-up lengths and covariate-driven NaN trimming), so the
band is recomputed per panel from that panel's own n, not a shared value.
"""

from __future__ import annotations
import textwrap
from typing import Callable, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf

from diagnostics.acf import acf_at_lags


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


# ===========================================================================
# Cross-location stage-diagnostic mosaics
#
# Design: a value of None anywhere in the input dicts means "this
# stage/branch was not run (or not accepted) at this location". Rather
# than silently omitting the location (which would misalign the grid),
# it renders as a labelled placeholder panel via _empty_panel, so every
# mosaic keeps its full, expected shape.
# ===========================================================================

# ---------------------------------------------------------------------------
# Print-size calibration
# =========================
# THIS IS THE SECTION TO EDIT IF FIGURE TEXT LOOKS TOO SMALL OR TOO BIG IN
# THE COMPILED PDF. (It is also exposed as notebook-level arguments -- see
# multilocation_diagnostic_mosaics.ipynb's "Figure design settings" cell --
# so most retuning never needs to touch this file at all.)
#
# The problem this solves: these mosaics are drawn many inches wide (a
# 6-location x 3-stage ACF grid is ~22in wide) so that every individual
# panel has room to be legible on screen. But LaTeX then shrinks the whole
# image down to fit wherever it's placed on the page -- e.g.
# \includegraphics[width=9.5in] on a landscape page shrinks a 22in-wide
# figure by a factor of 9.5/22 = 0.43. A matplotlib "12pt" font shrinks by
# that same factor, so it prints at ~5pt: unreadable, even though it
# looked fine in the raw PNG. Simply raising matplotlib's fontsize numbers
# (what earlier iterations of this file did) fights this blind, because
# the correct number depends on how many rows/columns are being rendered.
#
# The fix: BASE_PRINT_PT specifies the point size we actually want to see
# on the page (calibrated to this dissertation's own body text -- run
# `pdffonts` on delivery/dissertação v2 072026/*.pdf and it reports
# LMRoman12 as the main body font, i.e. 12pt Latin Modern Roman).
# TARGET_PRINT_WIDTH_IN is the width the figure will actually be placed at
# in the document -- these mosaics are wide multi-panel grids meant for a
# full landscape page (a \begin{sidewaysfigure}), not a portrait
# \textwidth column, hence the ~9.5in default (a typical A4 landscape page
# minus margins). If you place a figure at a different width, change
# TARGET_PRINT_WIDTH_IN (or pass target_print_width_in=... to the mosaic
# call in the notebook) to match -- every font in the figure rescales
# together, in real point sizes, wherever the grid is actually printed.
# ---------------------------------------------------------------------------
TARGET_PRINT_WIDTH_IN = 9.5

BASE_PRINT_PT = {
    "suptitle":   20.0,  # figure title
    "header":     16.0,  # column headers (stage names, or chart-type names)
    "row_label":  13.0,  # row labels (location names) -- deliberately smaller
                         # than "header": rotated 90 deg and constrained by
                         # row *height*, not column width, so a long name
                         # ("Cruzeiro do Sul") at "header" size can visually
                         # bleed into the neighbouring row above/below it.
    "axis_label": 13.0,  # x/y axis labels
    "tick":       11.0,  # tick labels
    "legend":     11.0,  # legend text
    "annotation": 10.0,  # in-panel "n=..." text and "not run" placeholder text
}


def _print_scale(fig_width_in: float, target_print_width_in: float) -> float:
    """Matplotlib fontsize multiplier so BASE_PRINT_PT prints at its
    literal point size once LaTeX shrinks this fig_width_in-wide figure
    down to target_print_width_in on the page. See "Print-size
    calibration" above."""
    return fig_width_in / target_print_width_in


def _pt(base_print_pt: dict, key: str, scale: float) -> float:
    return base_print_pt[key] * scale


def _empty_panel(ax: plt.Axes, note: str = "Not run at\nthis location", fontsize: float = 10.0) -> None:
    """Render a labelled placeholder for a (stage, location) or
    (chart, location) cell with no data, so the grid stays aligned instead
    of silently shrinking."""
    ax.text(0.5, 0.5, note, ha="center", va="center", fontsize=fontsize,
             color="dimgray", style="italic", transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("lightgray")


def _row_label(
    ax: plt.Axes, text: str, fontsize: float = 11.0, x_offset: float = -0.34,
    subtitle: Optional[str] = None, subtitle_fontsize: Optional[float] = None,
) -> None:
    """Bold, rotated label placed just outside the left edge of `ax`,
    identifying the row (a stage or a location) this panel belongs to.
    Kept as an annotation on the axes (not a figure-level fig.text) so it
    moves correctly with the axes under tight_layout/savefig.

    x_offset (axes-fraction units, negative = further left) may need to be
    pushed further out than the -0.34 default when the axes' own y-tick
    labels are wide relative to the axes width (e.g. multi-decimal PIT
    density ticks at a large calibrated font) -- otherwise this label can
    visually collide with the axes' native y-axis label. See
    stage4_composite_mosaic's PIT column for an example override.

    `subtitle` (e.g. "(Brazil)") renders a second, smaller, non-bold,
    italic annotation for the row (a country name) -- NOT stacked above/
    below `text` (tried first; at these mosaics' large calibrated
    print-pt font sizes, two rotation=90 lines stacked along y need a
    y-gap so large it either overlaps or forces very tall rows -- a
    rotation=90 Text's "\\n" line-break also advances *sideways*, not
    along the reading direction, so that shortcut doesn't help either).
    Instead `subtitle` is placed as a second rotated column, so the two
    read left-to-right as "name", "(country)" without needing any row-
    height increase.

    Critically, `subtitle` (not `text`) is placed at the caller-supplied
    `x_offset` -- the position already proven, at every call site in this
    module, to clear that panel's own native y-axis label/ticks for a
    *single* rotated line (that is exactly what `x_offset` was tuned for
    before `subtitle` existed). `text` is then pushed further out from
    there by however much horizontal room the `subtitle` column itself
    needs, computed from `subtitle`'s rendered font size and this axes'
    physical width (via ax.get_position() -- available pre-render, no
    need for a draw/renderer call). Putting the *new* column at the
    *proven-safe* offset, and only pushing the *existing, already-tuned*
    column further out, means this never needs its own per-mosaic
    collision tuning against the y-axis label -- only how far to push
    `text` out, which depends solely on `subtitle`'s own size."""
    x_text = x_offset
    if subtitle:
        fig = ax.figure
        ax_width_in = ax.get_position().width * fig.get_size_inches()[0]
        # Half the rotated "subtitle" column's own width (~1.3x its font
        # size, a generous single-line-height allowance) plus a small
        # fixed clearance, converted to this axes' fraction units.
        subtitle_fontsize = subtitle_fontsize or fontsize * 0.75
        col_half_width_in = (subtitle_fontsize / 72.0) * 1.3 / 2 + 0.05
        x_text = x_offset - col_half_width_in / ax_width_in
        ax.annotate(
            subtitle, xy=(x_offset, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center",
            fontsize=subtitle_fontsize, style="italic", color="dimgray",
            annotation_clip=False,
        )
    ax.annotate(
        text, xy=(x_text, 0.5), xycoords="axes fraction",
        rotation=90, ha="center", va="center",
        fontsize=fontsize, fontweight="bold", annotation_clip=False,
    )


def _panel_title(
    ax: plt.Axes, text: str, fontsize: float = 13.0,
    subtitle: Optional[str] = None, subtitle_fontsize: Optional[float] = None,
    line_gap: float = 0.11, y0: float = 1.02,
) -> None:
    """Bold panel title placed above `ax` (a location name, for the
    location-per-panel mosaics), with an optional smaller, non-bold,
    italic `subtitle` line directly underneath it (e.g. "(Brazil)") --
    unlike _row_label, this is not rotated, so "underneath" simply means a
    lower y anchor, both still above the axes (y0, y0 + line_gap in axes
    fraction). Uses ax.text with clip_on=False (matplotlib's default for
    Text) rather than ax.set_title, since set_title only supports a single
    line/style per axes."""
    y_name = y0 + line_gap if subtitle else y0
    ax.text(0.5, y_name, text, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold")
    if subtitle:
        ax.text(0.5, y0, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=subtitle_fontsize or fontsize * 0.75,
                style="italic", color="dimgray")


# ---------------------------------------------------------------------------
# Single-panel drawers -- the one place each chart type's statistics and
# reference lines are defined. Reused by every grid mosaic below (both the
# rows=stage/cols=location grids and the rows=location/cols=chart-type
# Stage-4 composite), so "what a PIT panel shows" cannot drift between
# call sites. Each returns the effective sample size n (or None if the
# input was None, i.e. "not run here" -- callers draw a placeholder then).
# `empty_fontsize` only affects the "not run" placeholder text; the
# in-panel statistics themselves (histogram, scatter, ACF bars) have no
# text of their own to scale -- their surrounding axis/tick/legend text is
# sized by the calling grid function instead.
# ---------------------------------------------------------------------------

def _draw_pit_panel(ax: plt.Axes, pit: Optional[np.ndarray], bins: int = 20,
                     empty_fontsize: float = 10.0) -> Optional[int]:
    """PIT histogram of u_t = F_t(y_t) against the Uniform(0,1) density
    expected under correct specification (docs/MODELS.md Sec 3-4;
    diagnostics/residuals.py). Left-skew = under-prediction; right-skew =
    over-prediction; U/hump shape = under-/over-dispersion."""
    if pit is None:
        _empty_panel(ax, fontsize=empty_fontsize)
        return None
    u = np.asarray(pit, dtype=float)
    u = u[np.isfinite(u)]
    ax.hist(u, bins=bins, range=(0.0, 1.0), density=True,
            color="#3498db", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="red", lw=1.4, linestyle="--", label="Uniform(0,1)")
    ax.set_xlim(0.0, 1.0)
    return len(u)


def _draw_qq_panel(ax: plt.Axes, qr: Optional[np.ndarray],
                    empty_fontsize: float = 10.0) -> Optional[int]:
    """Normal QQ plot of quantile residuals r_t = Phi^{-1}(F_t(y_t))
    against theoretical N(0,1) quantiles (Blom-type plotting positions
    Phi^{-1}((i-0.5)/n)); should lie on the 45-degree reference line under
    correct specification. Tail departures are the central scientific
    concern of this project (docs/MODELS.md Sec 27).

    Deliberately does NOT call ax.set_aspect("equal"): matplotlib's
    aspect-equal enforcement happens after fig.tight_layout()'s spacing
    is computed, so in a multi-panel grid it silently shrinks/repositions
    individual axes and breaks the shared row/column alignment (tested --
    it visibly corrupted this mosaic's layout). The reference line is
    still the mathematically correct y=x line (equal x/y data limits
    below); it just is not guaranteed to render at a literal 45-degree
    angle on screen when a panel's box is not square."""
    if qr is None:
        _empty_panel(ax, fontsize=empty_fontsize)
        return None
    r_sorted = np.sort(np.asarray(qr, dtype=float))
    r_sorted = r_sorted[np.isfinite(r_sorted)]
    n = len(r_sorted)
    q_th = norm.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n)) if n > 0 else np.array([])
    ax.scatter(q_th, r_sorted, s=6, alpha=0.5, color="#2c3e50")
    lim = max(np.abs(r_sorted).max(), np.abs(q_th).max()) * 1.05 if n > 0 else 1.0
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1.2, label="45 degrees line")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    return n


def _draw_acf_panel(ax: plt.Axes, qr: Optional[np.ndarray], n_lags: int = 400,
                     empty_fontsize: float = 10.0) -> Optional[int]:
    """Sample ACF of quantile residuals at lags 1..n_lags (lag 0 excluded,
    it is always 1 by construction -- docs/REPORTING.md Sec 13). y is
    fixed to [-1, 1]; the shaded band is the +/-2/sqrt(n) approximate 95%
    Bartlett null band for "no autocorrelation" under white noise (see
    module docstring), recomputed here from this panel's own n."""
    if qr is None:
        _empty_panel(ax, fontsize=empty_fontsize)
        return None
    r_arr = np.asarray(qr, dtype=float)
    r_arr = r_arr[np.isfinite(r_arr)]
    n = len(r_arr)
    lags_wanted = list(range(1, min(n_lags, max(n - 1, 1)) + 1))
    acf_dict = acf_at_lags(r_arr, lags_wanted)
    lags = np.array(sorted(acf_dict.keys()))
    acf_vals = np.array([acf_dict[k] for k in lags])

    ci = 1.96 / np.sqrt(n) if n > 0 else np.nan
    ax.fill_between(lags, -ci, ci, color="red", alpha=0.15,
                      label=r"$\pm 2/\sqrt{n}$ null band")
    ax.axhline(ci, color="red", lw=0.6, linestyle="--")
    ax.axhline(-ci, color="red", lw=0.6, linestyle="--")
    ax.bar(lags, acf_vals, width=1.0, color="#3498db", linewidth=0)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0, max(lags) if len(lags) else n_lags)
    return n


# ---------------------------------------------------------------------------
# Stage 1-3 grids: rows = location, columns = stage (the lower-cardinality
# axis: 3 stages vs. 6 locations), one chart type and one branch per
# figure. Putting the small axis in columns keeps the figure to 3 columns
# regardless of how many locations are added, so each panel gets a large,
# fixed share of the page width -- deliberately sized for full-width or
# full-page LaTeX placement rather than an on-screen thumbnail grid (large
# figures that occupy more of the page are expected and intentional here).
# Row headers (location names) appear once, on the leftmost column, via
# _row_label; column headers (stage names) appear once, on the top row.
# ---------------------------------------------------------------------------

def _n_annotation(ax: plt.Axes, n: Optional[int], fontsize: float = 10.0) -> None:
    if n is not None:
        ax.text(0.98, 0.96, f"n={n}", transform=ax.transAxes, ha="right", va="top",
                 fontsize=fontsize, color="dimgray")


def _resolve_print_pt(base_print_pt: Optional[dict]) -> dict:
    """Merge caller overrides on top of the BASE_PRINT_PT defaults."""
    return {**BASE_PRINT_PT, **(base_print_pt or {})}


def stage_pit_grid_mosaic(
    pit_grid: Dict[int, Dict[str, Optional[np.ndarray]]],
    stage_labels: Dict[int, str],
    stages: List[int],
    locations: List[str],
    suptitle: str = "",
    bins: int = 20,
    target_print_width_in: float = TARGET_PRINT_WIDTH_IN,
    base_print_pt: Optional[dict] = None,
    location_names: Optional[Dict[str, str]] = None,
    location_countries: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    PIT histogram grid for Stages 1-3 (one branch): rows = location,
    columns = stage.

    Parameters
    ----------
    pit_grid     : {stage: {location display name: PIT array, or None}}
    stage_labels : {stage: column header, e.g. 1: "Stage 1"}
    stages       : column order, e.g. [1, 2, 3]
    locations    : row order -- also the lookup key into pit_grid/
        location_names/location_countries
    suptitle     : figure-level title
    bins         : histogram bin count (shared across panels)
    target_print_width_in, base_print_pt : print-size calibration -- see
        "Print-size calibration" above. Override these to match wherever
        you actually place this figure in the LaTeX document; defaults
        assume a full landscape page at this dissertation's 12pt body size.
    location_names     : {locations entry: text actually drawn in the row
        label}, e.g. {"Darwin Airport": "Darwin"} -- falls back to the
        `locations` entry itself when absent, so lookup keys (and every
        other dict this figure reads) never need to change.
    location_countries  : {locations entry: subtitle drawn under the name},
        e.g. {"Darwin Airport": "(Australia)"} -- omitted entirely (no
        second line) when absent for that location.
    """
    nrows, ncols = len(locations), len(stages)
    fig_w, fig_h = 5.6 * ncols, 3.6 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    pt = _resolve_print_pt(base_print_pt)
    scale = _print_scale(fig_w, target_print_width_in)
    if suptitle:
        fig.suptitle(suptitle, fontsize=_pt(pt, "suptitle", scale), fontweight="bold")

    for i, loc in enumerate(locations):
        for j, stage in enumerate(stages):
            ax = axes[i, j]
            n = _draw_pit_panel(ax, pit_grid.get(stage, {}).get(loc), bins=bins,
                                 empty_fontsize=_pt(pt, "annotation", scale))
            _n_annotation(ax, n, fontsize=_pt(pt, "annotation", scale))
            if i == 0:
                ax.set_title(stage_labels[stage], fontsize=_pt(pt, "header", scale), fontweight="bold")
            if j == 0:
                _row_label(ax, (location_names or {}).get(loc, loc), fontsize=_pt(pt, "row_label", scale),
                           subtitle=(location_countries or {}).get(loc),
                           subtitle_fontsize=_pt(pt, "row_label", scale) * 0.75)
                ax.set_ylabel("Density", fontsize=_pt(pt, "axis_label", scale))
            if i == nrows - 1:
                ax.set_xlabel(r"PIT value $u_t=F_t(y_t)$", fontsize=_pt(pt, "axis_label", scale))
            ax.tick_params(labelsize=_pt(pt, "tick", scale))

    axes[0, 0].legend(fontsize=_pt(pt, "legend", scale), loc="upper left")
    fig.tight_layout(rect=(0.03, 0, 1, 0.97) if suptitle else (0.03, 0, 1, 1))
    return fig


def stage_qq_grid_mosaic(
    qr_grid: Dict[int, Dict[str, Optional[np.ndarray]]],
    stage_labels: Dict[int, str],
    stages: List[int],
    locations: List[str],
    suptitle: str = "",
    target_print_width_in: float = TARGET_PRINT_WIDTH_IN,
    base_print_pt: Optional[dict] = None,
    location_names: Optional[Dict[str, str]] = None,
    location_countries: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    Normal QQ plot grid for Stages 1-3 (one branch): rows = location,
    columns = stage. See stage_pit_grid_mosaic for the shared layout,
    print-size-calibration, and location_names/location_countries
    conventions, and _draw_qq_panel for what each panel plots.
    """
    nrows, ncols = len(locations), len(stages)
    fig_w, fig_h = 5.2 * ncols, 5.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    pt = _resolve_print_pt(base_print_pt)
    scale = _print_scale(fig_w, target_print_width_in)
    if suptitle:
        fig.suptitle(suptitle, fontsize=_pt(pt, "suptitle", scale), fontweight="bold")

    for i, loc in enumerate(locations):
        for j, stage in enumerate(stages):
            ax = axes[i, j]
            n = _draw_qq_panel(ax, qr_grid.get(stage, {}).get(loc),
                                empty_fontsize=_pt(pt, "annotation", scale))
            _n_annotation(ax, n, fontsize=_pt(pt, "annotation", scale))
            if i == 0:
                ax.set_title(stage_labels[stage], fontsize=_pt(pt, "header", scale), fontweight="bold")
            if j == 0:
                _row_label(ax, (location_names or {}).get(loc, loc), fontsize=_pt(pt, "row_label", scale),
                           subtitle=(location_countries or {}).get(loc),
                           subtitle_fontsize=_pt(pt, "row_label", scale) * 0.75)
                ax.set_ylabel("Sample quantile residuals $r_t$", fontsize=_pt(pt, "axis_label", scale))
            if i == nrows - 1:
                ax.set_xlabel("Theoretical N(0,1) quantiles", fontsize=_pt(pt, "axis_label", scale))
            ax.tick_params(labelsize=_pt(pt, "tick", scale))

    axes[0, 0].legend(fontsize=_pt(pt, "legend", scale), loc="upper left")
    fig.tight_layout(rect=(0.03, 0, 1, 0.97) if suptitle else (0.03, 0, 1, 1))
    return fig


def stage_acf_grid_mosaic(
    qr_grid: Dict[int, Dict[str, Optional[np.ndarray]]],
    stage_labels: Dict[int, str],
    stages: List[int],
    locations: List[str],
    suptitle: str = "",
    n_lags: int = 400,
    target_print_width_in: float = TARGET_PRINT_WIDTH_IN,
    base_print_pt: Optional[dict] = None,
    location_names: Optional[Dict[str, str]] = None,
    location_countries: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    Quantile-residual ACF grid for Stages 1-3 (one branch): rows =
    location, columns = stage. See stage_pit_grid_mosaic for the shared
    layout, print-size-calibration, and location_names/location_countries
    conventions, and _draw_acf_panel for what each panel plots (400 lags,
    y in [-1, 1], shaded +/-2/sqrt(n) null band, lag 0 excluded). Only 3
    columns (one per stage) means each ACF panel gets a wide, readable lag
    axis regardless of how many locations are stacked as rows.
    """
    nrows, ncols = len(locations), len(stages)
    # Row height (4.4in, taller than the panel content strictly needs) is
    # sized to leave room for the rotated row_label text at its calibrated
    # print-pt size without bleeding into the neighbouring row -- see
    # BASE_PRINT_PT["row_label"]'s comment. The optional country subtitle
    # (_row_label) is drawn as a second rotated column beside the name, not
    # stacked above/below it, so it needs no extra row height of its own.
    fig_w, fig_h = 7.2 * ncols, 4.4 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    pt = _resolve_print_pt(base_print_pt)
    scale = _print_scale(fig_w, target_print_width_in)
    if suptitle:
        fig.suptitle(suptitle, fontsize=_pt(pt, "suptitle", scale), fontweight="bold")

    for i, loc in enumerate(locations):
        for j, stage in enumerate(stages):
            ax = axes[i, j]
            n = _draw_acf_panel(ax, qr_grid.get(stage, {}).get(loc), n_lags=n_lags,
                                 empty_fontsize=_pt(pt, "annotation", scale))
            _n_annotation(ax, n, fontsize=_pt(pt, "annotation", scale))
            if i == 0:
                ax.set_title(stage_labels[stage], fontsize=_pt(pt, "header", scale), fontweight="bold")
            if j == 0:
                _row_label(ax, (location_names or {}).get(loc, loc), fontsize=_pt(pt, "row_label", scale),
                           subtitle=(location_countries or {}).get(loc),
                           subtitle_fontsize=_pt(pt, "row_label", scale) * 0.75)
                ax.set_ylabel("Autocorrelation", fontsize=_pt(pt, "axis_label", scale))
            if i == nrows - 1:
                ax.set_xlabel("Lag (days)", fontsize=_pt(pt, "axis_label", scale))
            ax.tick_params(labelsize=_pt(pt, "tick", scale))

    axes[0, 0].legend(fontsize=_pt(pt, "legend", scale), loc="upper right")
    fig.tight_layout(rect=(0.03, 0, 1, 0.97) if suptitle else (0.03, 0, 1, 1))
    return fig


# ---------------------------------------------------------------------------
# Stage 4 composite: rows = location, columns = [PIT, QQ, ACF]. Stage 4 has
# no phi-only/phi-xi branch split (docs/MODELS.md Sec 22-23), so a single
# figure covers all three chart types instead of three separate mosaics.
# ---------------------------------------------------------------------------

def stage4_composite_mosaic(
    pit_by_location: Dict[str, Optional[np.ndarray]],
    qr_by_location: Dict[str, Optional[np.ndarray]],
    suptitle: str = "",
    n_lags: int = 400,
    bins: int = 20,
    target_print_width_in: float = TARGET_PRINT_WIDTH_IN,
    base_print_pt: Optional[dict] = None,
    location_names: Optional[Dict[str, str]] = None,
    location_countries: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    Stage 4 (tail-sensitive xi regime) diagnostics: rows = location,
    columns = [PIT histogram, normal QQ plot, ACF (n_lags)].

    Parameters
    ----------
    pit_by_location : {location display name: PIT array, or None}
    qr_by_location  : {location display name: quantile-residual array,
                       or None} -- same keys/order as pit_by_location
    suptitle        : figure-level title
    n_lags          : ACF lag count (default 400, docs/REPORTING.md Sec 13)
    bins            : PIT histogram bin count
    target_print_width_in, base_print_pt : print-size calibration -- see
        "Print-size calibration" in the module docstring/comment above
        stage_pit_grid_mosaic.
    location_names, location_countries : see stage_pit_grid_mosaic --
        keyed by the same strings as pit_by_location/qr_by_location.
    """
    locations = list(pit_by_location.keys())
    nrows = len(locations)
    col_titles = ["PIT histogram", "Normal QQ plot", f"Quantile-residual ACF ({n_lags} lags)"]
    # Row height (5.2in) is taller than stage_*_grid_mosaic's per-row height:
    # this composite's QQ column is narrower (width_ratio 1.0 of 3.8) than
    # the dedicated Stages-1-3 QQ grid, so its rotated y-axis label needs
    # more vertical room at the same calibrated print-pt font before it
    # risks bleeding into the row above/below (the same failure mode as
    # BASE_PRINT_PT["row_label"], just for a native ax.set_ylabel here).
    fig_w, fig_h = 19.0, 5.2 * nrows
    fig, axes = plt.subplots(
        nrows, 3, squeeze=False, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.8]},
    )
    pt = _resolve_print_pt(base_print_pt)
    scale = _print_scale(fig_w, target_print_width_in)
    if suptitle:
        fig.suptitle(suptitle, fontsize=_pt(pt, "suptitle", scale), fontweight="bold")

    for i, loc in enumerate(locations):
        ax_pit, ax_qq, ax_acf = axes[i]

        n = _draw_pit_panel(ax_pit, pit_by_location[loc], bins=bins,
                             empty_fontsize=_pt(pt, "annotation", scale))
        _n_annotation(ax_pit, n, fontsize=_pt(pt, "annotation", scale))
        # Pushed further left than the -0.34 default: this PIT column's
        # y-tick labels ("0.00".."1.25") are wide at the calibrated tick
        # font, shifting the native "Density" label further out than
        # usual and risking a collision with the row label at -0.34.
        _row_label(ax_pit, (location_names or {}).get(loc, loc), fontsize=_pt(pt, "row_label", scale), x_offset=-0.5,
                   subtitle=(location_countries or {}).get(loc),
                   subtitle_fontsize=_pt(pt, "row_label", scale) * 0.75)
        ax_pit.set_ylabel("Density", fontsize=_pt(pt, "axis_label", scale))

        n = _draw_qq_panel(ax_qq, qr_by_location[loc], empty_fontsize=_pt(pt, "annotation", scale))
        _n_annotation(ax_qq, n, fontsize=_pt(pt, "annotation", scale))
        # Shorter than stage_qq_grid_mosaic's "Sample quantile residuals
        # $r_t$" -- this composite's QQ column is narrower, so the same
        # long label at the same calibrated font risked overlapping the
        # row above/below it (see fig_h comment above).
        ax_qq.set_ylabel("Quantile residuals $r_t$", fontsize=_pt(pt, "axis_label", scale))

        n = _draw_acf_panel(ax_acf, qr_by_location[loc], n_lags=n_lags,
                             empty_fontsize=_pt(pt, "annotation", scale))
        _n_annotation(ax_acf, n, fontsize=_pt(pt, "annotation", scale))
        ax_acf.set_ylabel("Autocorrelation", fontsize=_pt(pt, "axis_label", scale))

        if i == 0:
            for ax, title in zip(axes[0], col_titles):
                ax.set_title(title, fontsize=_pt(pt, "header", scale), fontweight="bold")
        if i == nrows - 1:
            ax_pit.set_xlabel(r"PIT value $u_t=F_t(y_t)$", fontsize=_pt(pt, "axis_label", scale))
            ax_qq.set_xlabel("Theoretical N(0,1) quantiles", fontsize=_pt(pt, "axis_label", scale))
            ax_acf.set_xlabel("Lag (days)", fontsize=_pt(pt, "axis_label", scale))
        for ax in (ax_pit, ax_qq, ax_acf):
            ax.tick_params(labelsize=_pt(pt, "tick", scale))

    axes[0, 0].legend(fontsize=_pt(pt, "legend", scale), loc="upper left")
    axes[0, 1].legend(fontsize=_pt(pt, "legend", scale), loc="upper left")
    axes[0, 2].legend(fontsize=_pt(pt, "legend", scale), loc="upper right")
    fig.tight_layout(rect=(0.045, 0, 1, 0.96) if suptitle else (0.045, 0, 1, 1))
    return fig


# ---------------------------------------------------------------------------
# Monthly pi_t seasonality: fixed 3-row x 2-column grid, one panel per
# location (unlike the stage/branch grids above, there is no second axis
# here -- every panel shows the same one model, the Stage 1 phi-only
# winner, at a different location). See diagnostics/pi_monthly.py for how
# each location's monthly averages are computed from the in-sample
# filtered path (model.filter(theta, y_train)["pi"]).
#
# Both panel drawers share the same x-axis convention as the grids above:
# calendar-month tick labels are shown on every panel (the x-axis means
# the same thing -- "calendar month" -- in every panel of the mosaic, so
# losing them on non-bottom rows would make those panels ambiguous on
# their own), while the "Month" axis-label *text* is added only once, on
# the bottom row, matching stage_pit_grid_mosaic etc.
# ---------------------------------------------------------------------------

_MONTH_TICKS = np.arange(12)


def _fig_legend_above_panels(fig: plt.Figure, axes: np.ndarray, fontsize: float,
                              y: float = 0.99) -> None:
    """Shared legend placed in the figure's own top margin (not inside any
    one panel's axes), so it never overlaps that panel's bars -- unlike the
    stage/branch grids above, a bar here can legitimately reach pi_t/wet-
    fraction values above 0.5 in the wet season, which an in-axes
    "upper left/right" legend would sit on top of. Uses whichever panel's
    axes actually has labelled artists (the first location with data),
    since a placeholder ("Not run") panel has none.

    `y` (figure fraction) must be placed below any suptitle -- callers pass
    a lower value when a suptitle is present so the two never overlap."""
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, fontsize=fontsize, loc="upper center",
                       bbox_to_anchor=(0.5, y), ncol=len(labels), frameon=False)
            return


# ---------------------------------------------------------------------------
# Title wrapping and title/legend vertical placement -- shared by every
# mosaic function that wants "the title's rendered width capped at (a bit
# more than) the figure's own physical width, wrapped onto extra lines
# instead of overflowing, with that width limit identical across every
# figure that uses it." The mechanism is entirely in these two functions:
# every caller passes its own suptitle/fontsize/fig size through them
# rather than hand-computing its own wrap width or y-offsets, so the same
# rule applies everywhere without needing to be re-tuned per figure.
# ---------------------------------------------------------------------------

def _wrap_title_to_width(text: str, fig_width_in: float, fontsize_pt: float,
                          overflow_factor: float = 1.08) -> str:
    """Wrap `text` onto multiple lines so each line's rendered width stays
    within roughly `fig_width_in` (the figure's own physical width, in
    inches) at `fontsize_pt` (the matplotlib fontsize actually used to
    draw it, i.e. already print-scale-calibrated -- see _print_scale).

    There is no font-metrics query here (no renderer/draw call needed):
    average character advance width for a bold sans-serif font is
    estimated as ~0.58x the font's point size, a standard typesetting
    rule of thumb precise enough for wrap decisions. `overflow_factor`
    (>1) matches the "a bit more than the picture size" allowance -- the
    title may run slightly wider than the raw figure before wrapping to a
    new line, since bbox_inches="tight" (used by every mosaic-saving
    notebook in this repo) expands the saved image to fit it anyway."""
    avg_char_width_in = fontsize_pt * 0.58 / 72.0
    max_chars = max(10, int(fig_width_in * overflow_factor / avg_char_width_in))
    return "\n".join(textwrap.wrap(text, width=max_chars, break_long_words=False))


def _place_wrapped_title_and_legend(
    fig: plt.Figure, axes: np.ndarray, suptitle: str,
    title_fontsize: float, legend_fontsize: float,
) -> float:
    """Wrap `suptitle` to the figure's own physical width
    (_wrap_title_to_width), draw it as the figure suptitle, draw the
    shared per-mosaic legend (_fig_legend_above_panels) just below it --
    however many lines the title wrapped to -- and return the top-margin
    fraction fig.tight_layout(rect=...) should use so neither the title
    nor the legend is clipped or overlaps the panels below.

    This is the ONE place that decides how much vertical headroom a
    suptitle needs: every location-grid mosaic calls this function
    instead of each hand-tuning its own y=... offsets for the suptitle
    and legend, which is what keeps a two-line title's spacing consistent
    with a one-line title's, and consistent across every mosaic function
    that calls this rather than drifting per call site.
    """
    fig_w, fig_h = fig.get_size_inches()
    if not suptitle:
        _fig_legend_above_panels(fig, axes, fontsize=legend_fontsize, y=0.99)
        return 0.94

    wrapped = _wrap_title_to_width(suptitle, fig_width_in=fig_w, fontsize_pt=title_fontsize)
    n_lines = wrapped.count("\n") + 1
    fig.suptitle(wrapped, fontsize=title_fontsize, fontweight="bold", y=0.99)

    # Vertical space one title line occupies, as a figure-height fraction
    # (point size -> inches via /72, x1.25 for line-spacing headroom).
    line_frac = (title_fontsize / 72.0 * 1.25) / fig_h
    legend_y = 0.99 - n_lines * line_frac - 0.015
    _fig_legend_above_panels(fig, axes, fontsize=legend_fontsize, y=legend_y)
    rect_top = legend_y - (legend_fontsize / 72.0 * 1.6) / fig_h - 0.01
    return rect_top


def _draw_monthly_pi_panel(
    ax: plt.Axes, month_data: Optional[dict],
    tick_fontsize: float = 11.0, empty_fontsize: float = 10.0,
) -> Optional[int]:
    """Bar chart of the Stage 1 phi-only winner's average in-sample
    occurrence probability pi_t by calendar month (diagnostics/pi_monthly.py).
    A flat set of bars indicates the AR-logistic occurrence dynamics found
    little seasonal structure in rain occurrence at this location; a
    pronounced pattern (e.g. a wet-season hump) indicates the opposite."""
    from diagnostics.pi_monthly import MONTH_LABELS

    if month_data is None:
        _empty_panel(ax, fontsize=empty_fontsize)
        return None
    pi_by_month = np.asarray(month_data["pi_by_month"], dtype=float)
    ax.bar(_MONTH_TICKS, pi_by_month, width=0.7, color="#3498db",
           edgecolor="white", linewidth=0.5, label=r"Model $\pi_t$ (in-sample average)")
    ax.set_xticks(_MONTH_TICKS)
    ax.set_xticklabels(MONTH_LABELS, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.6, 11.6)
    return int(month_data["n"])


def _draw_monthly_pi_occurrence_panel(
    ax: plt.Axes, month_data: Optional[dict],
    tick_fontsize: float = 11.0, empty_fontsize: float = 10.0,
) -> Optional[int]:
    """Grouped bar chart comparing, by calendar month, the Stage 1
    phi-only winner's average in-sample pi_t (blue) against the empirical
    fraction of wet days observed in the training sample (red) --
    diagnostics/pi_monthly.py. Bars of similar height in a given month
    indicate the fitted occurrence probability tracks the empirical
    seasonal rain frequency well; a systematic gap indicates the AR-logistic
    dynamics under- or over-estimate occurrence in that month.

    The empirical-fraction bars also get a diagonal hatch pattern on top of
    their red fill: this dissertation may be printed in black and white, at
    which point red and blue can render as similar-looking grays (a real
    risk since --e74c3c and #3498db are not equal-luminance colors, but a
    B&W print pipeline may still desaturate them close together) -- the
    hatch keeps the two series visually distinguishable by texture, not
    just by hue, independent of how the print pipeline handles color."""
    from diagnostics.pi_monthly import MONTH_LABELS

    if month_data is None:
        _empty_panel(ax, fontsize=empty_fontsize)
        return None
    pi_by_month  = np.asarray(month_data["pi_by_month"], dtype=float)
    wet_by_month = np.asarray(month_data["wet_frac_by_month"], dtype=float)
    w = 0.38
    ax.bar(_MONTH_TICKS - w / 2, pi_by_month, width=w, color="#3498db",
           edgecolor="white", linewidth=0.5, label=r"Model $\pi_t$ (in-sample average)")
    ax.bar(_MONTH_TICKS + w / 2, wet_by_month, width=w, color="#e74c3c",
           edgecolor="black", linewidth=0.6, hatch="///",
           label="Empirical wet-day fraction")
    ax.set_xticks(_MONTH_TICKS)
    ax.set_xticklabels(MONTH_LABELS, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.6, 11.6)
    return int(month_data["n"])


def monthly_pi_mosaic(
    pi_data: Dict[str, Optional[dict]],
    locations: List[str],
    suptitle: str = "",
    target_print_width_in: float = TARGET_PRINT_WIDTH_IN,
    base_print_pt: Optional[dict] = None,
    location_names: Optional[Dict[str, str]] = None,
    location_countries: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    Fixed 3-row x 2-column mosaic: one bar-chart panel per location,
    showing the Stage 1 phi-only winner's in-sample average occurrence
    probability pi_t by calendar month. Locations fill the grid row-major
    (left to right, top to bottom) in the order given by `locations`;
    exactly 6 locations are expected.

    Parameters
    ----------
    pi_data   : {location display name: dict-or-None}, from
                diagnostics.pi_monthly.compute_monthly_pi_and_occurrence
    locations : row-major panel order, length 6 -- also the lookup key
                into pi_data/location_names/location_countries
    suptitle  : figure-level title
    target_print_width_in, base_print_pt : print-size calibration -- see
        "Print-size calibration" above stage_pit_grid_mosaic. This is a
        compact 2-column figure, not a wide stage grid -- pass a smaller
        target_print_width_in (e.g. ~6.3in, a normal portrait \\textwidth
        figure) if placing it at less than a full landscape page width.
    location_names, location_countries : see stage_pit_grid_mosaic --
        e.g. {"Darwin Airport": "Darwin"} / {"Darwin Airport": "(Australia)"};
        `locations` itself never needs to change for either override.
    """
    ncols = 2
    nrows = int(np.ceil(len(locations) / ncols))
    fig_w, fig_h = 5.8 * ncols, 4.1 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    pt = _resolve_print_pt(base_print_pt)
    scale = _print_scale(fig_w, target_print_width_in)
    if suptitle:
        fig.suptitle(suptitle, fontsize=_pt(pt, "suptitle", scale), fontweight="bold", y=0.99)

    for idx, loc in enumerate(locations):
        i, j = divmod(idx, ncols)
        ax = axes[i, j]
        n = _draw_monthly_pi_panel(
            ax, pi_data.get(loc),
            tick_fontsize=_pt(pt, "tick", scale), empty_fontsize=_pt(pt, "annotation", scale),
        )
        _n_annotation(ax, n, fontsize=_pt(pt, "annotation", scale))
        _panel_title(ax, (location_names or {}).get(loc, loc), fontsize=_pt(pt, "header", scale),
                     subtitle=(location_countries or {}).get(loc),
                     subtitle_fontsize=_pt(pt, "header", scale) * 0.7)
        if j == 0:
            ax.set_ylabel(r"Average $\pi_t$", fontsize=_pt(pt, "axis_label", scale))
        if i == nrows - 1:
            ax.set_xlabel("Month", fontsize=_pt(pt, "axis_label", scale))
        ax.tick_params(axis="y", labelsize=_pt(pt, "tick", scale))

    # Unused trailing cells (e.g. an odd number of locations) are hidden
    # rather than left as empty axes with default matplotlib ticks/frame.
    for idx in range(len(locations), nrows * ncols):
        i, j = divmod(idx, ncols)
        axes[i, j].axis("off")

    _fig_legend_above_panels(fig, axes, fontsize=_pt(pt, "legend", scale),
                              y=0.94 if suptitle else 0.99)
    fig.tight_layout(rect=(0.03, 0, 1, 0.88) if suptitle else (0.03, 0, 1, 0.92))
    return fig


def monthly_pi_occurrence_mosaic(
    pi_data: Dict[str, Optional[dict]],
    locations: List[str],
    suptitle: str = "",
    target_print_width_in: float = TARGET_PRINT_WIDTH_IN,
    base_print_pt: Optional[dict] = None,
    location_names: Optional[Dict[str, str]] = None,
    location_countries: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    Fixed 3-row x 2-column mosaic: one grouped-bar-chart panel per
    location, comparing the Stage 1 phi-only winner's in-sample average
    pi_t (blue) against the empirical fraction of wet days (red, hatched
    for black-and-white-safe printing -- see _draw_monthly_pi_occurrence_panel)
    by calendar month -- see monthly_pi_mosaic for the shared grid/print-size/
    location_names/location_countries conventions and diagnostics/pi_monthly.py
    for the underlying computation.
    """
    ncols = 2
    nrows = int(np.ceil(len(locations) / ncols))
    fig_w, fig_h = 5.8 * ncols, 4.1 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    pt = _resolve_print_pt(base_print_pt)
    scale = _print_scale(fig_w, target_print_width_in)
    if suptitle:
        fig.suptitle(suptitle, fontsize=_pt(pt, "suptitle", scale), fontweight="bold", y=0.99)

    for idx, loc in enumerate(locations):
        i, j = divmod(idx, ncols)
        ax = axes[i, j]
        n = _draw_monthly_pi_occurrence_panel(
            ax, pi_data.get(loc),
            tick_fontsize=_pt(pt, "tick", scale), empty_fontsize=_pt(pt, "annotation", scale),
        )
        _n_annotation(ax, n, fontsize=_pt(pt, "annotation", scale))
        _panel_title(ax, (location_names or {}).get(loc, loc), fontsize=_pt(pt, "header", scale),
                     subtitle=(location_countries or {}).get(loc),
                     subtitle_fontsize=_pt(pt, "header", scale) * 0.7)
        if j == 0:
            ax.set_ylabel("Probability", fontsize=_pt(pt, "axis_label", scale))
        if i == nrows - 1:
            ax.set_xlabel("Month", fontsize=_pt(pt, "axis_label", scale))
        ax.tick_params(axis="y", labelsize=_pt(pt, "tick", scale))

    for idx in range(len(locations), nrows * ncols):
        i, j = divmod(idx, ncols)
        axes[i, j].axis("off")

    _fig_legend_above_panels(fig, axes, fontsize=_pt(pt, "legend", scale),
                              y=0.94 if suptitle else 0.99)
    fig.tight_layout(rect=(0.03, 0, 1, 0.88) if suptitle else (0.03, 0, 1, 0.92))
    return fig


# ---------------------------------------------------------------------------
# Generic location-grid QQ/ACF mosaics: fixed 2-column grid, one panel per
# location, ONE model/variant per figure (unlike stage_qq_grid_mosaic /
# stage_acf_grid_mosaic above, which put several stages/branches in columns
# for a single figure). Added for diagnostics/pi_dynamics_alt_mosaics.py
# (comparing occurrence-dynamics alternatives -- pi_dynamics/phi_linked.py
# vs. pi_dynamics/ar_logistic*.py), but deliberately generic (just
# {location: quantile-residual array} in, one QQ or ACF panel per location
# out) so any future single-model, all-locations diagnostic can reuse them
# instead of writing a fifth near-identical grid function. Reuses the same
# _draw_qq_panel / _draw_acf_panel drawers as the Stage 1-4 grids above, so
# "what a QQ/ACF panel shows" still cannot drift between call sites, and
# the same _panel_title (name + "(Country)" subtitle) / print-size-
# calibration conventions as monthly_pi_mosaic / monthly_pi_occurrence_mosaic.
#
# qq_location_mosaic and acf_location_mosaic both read their figure size
# from the ONE constant below (rather than each hard-coding its own
# fig_w/fig_h numbers) -- that is the entire mechanism for "these charts
# are always exactly the same size": there is nothing else to keep in
# sync. Change the two numbers here to resize both at once; they cannot
# drift apart because there is only one place either function reads them
# from. (_draw_qq_panel's own ax.set_aspect("equal") keeps its 45-degree
# reference line honestly diagonal regardless of this box's own
# width/height ratio, so the two panel *types* do not need a different
# aspect ratio to each look right.)
# ---------------------------------------------------------------------------

LOCATION_QQACF_PANEL_SIZE_IN = (6.4, 7.0)  # (width per column, height per row), inches
# Height (7.0in/row) is generous, not tightly tuned: QQ's rotated native
# "Sample quantile residuals $r_t$" y-axis label is the tallest text
# either panel type draws, and how much room it needs grows with this
# constant's own *width* (a wider figure at a fixed target_print_width_in
# means a larger calibrated font -- see _print_scale -- which then needs
# more vertical room to avoid bleeding into the row above/below, the same
# failure mode noted on BASE_PRINT_PT["row_label"] elsewhere in this
# file). Confirmed empirically at TARGET_PRINT_WIDTH_IN=6.3; if this
# constant's width is ever increased further, re-check for that overlap
# before trusting the render.


def qq_location_mosaic(
    qr_by_location: Dict[str, Optional[np.ndarray]],
    locations: List[str],
    suptitle: str = "",
    target_print_width_in: float = TARGET_PRINT_WIDTH_IN,
    base_print_pt: Optional[dict] = None,
    location_names: Optional[Dict[str, str]] = None,
    location_countries: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    Fixed 2-column mosaic (rows = ceil(len(locations)/2)): one normal QQ
    plot panel per location, for a single model/variant (see
    _draw_qq_panel for what each panel plots: sorted quantile residuals
    r_t = Phi^{-1}(F_t(y_t)) against theoretical N(0,1) quantiles, y=x
    reference line).

    Parameters
    ----------
    qr_by_location : {location display name: quantile-residual array, or
        None (renders a labelled placeholder panel)}
    locations : row-major panel order (fills left-to-right, top-to-bottom)
    suptitle  : figure-level title
    target_print_width_in, base_print_pt : print-size calibration -- see
        "Print-size calibration" above stage_pit_grid_mosaic.
    location_names, location_countries : optional per-location panel-title
        overrides -- see monthly_pi_mosaic.
    """
    ncols = 2
    nrows = int(np.ceil(len(locations) / ncols))
    fig_w, fig_h = LOCATION_QQACF_PANEL_SIZE_IN[0] * ncols, LOCATION_QQACF_PANEL_SIZE_IN[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    pt = _resolve_print_pt(base_print_pt)
    scale = _print_scale(fig_w, target_print_width_in)

    for idx, loc in enumerate(locations):
        i, j = divmod(idx, ncols)
        ax = axes[i, j]
        n = _draw_qq_panel(ax, qr_by_location.get(loc), empty_fontsize=_pt(pt, "annotation", scale))
        _n_annotation(ax, n, fontsize=_pt(pt, "annotation", scale))
        _panel_title(ax, (location_names or {}).get(loc, loc), fontsize=_pt(pt, "header", scale),
                     subtitle=(location_countries or {}).get(loc),
                     subtitle_fontsize=_pt(pt, "header", scale) * 0.7)
        if j == 0:
            # Shorter than stage_qq_grid_mosaic's "Sample quantile
            # residuals $r_t$" -- this mosaic's rows are proportionally
            # shorter relative to its calibrated font size (see
            # LOCATION_QQACF_PANEL_SIZE_IN), so the longer label bled
            # into the row above/below it (the same failure mode noted on
            # BASE_PRINT_PT["row_label"] elsewhere in this file).
            ax.set_ylabel("Quantile residuals $r_t$", fontsize=_pt(pt, "axis_label", scale))
        if i == nrows - 1:
            ax.set_xlabel("Theoretical N(0,1) quantiles", fontsize=_pt(pt, "axis_label", scale))
        ax.tick_params(labelsize=_pt(pt, "tick", scale))

    for idx in range(len(locations), nrows * ncols):
        i, j = divmod(idx, ncols)
        axes[i, j].axis("off")

    rect_top = _place_wrapped_title_and_legend(
        fig, axes, suptitle,
        title_fontsize=_pt(pt, "suptitle", scale), legend_fontsize=_pt(pt, "legend", scale),
    )
    fig.tight_layout(rect=(0.03, 0, 1, rect_top))
    return fig


def acf_location_mosaic(
    qr_by_location: Dict[str, Optional[np.ndarray]],
    locations: List[str],
    suptitle: str = "",
    n_lags: int = 400,
    target_print_width_in: float = TARGET_PRINT_WIDTH_IN,
    base_print_pt: Optional[dict] = None,
    location_names: Optional[Dict[str, str]] = None,
    location_countries: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    Fixed 2-column mosaic (rows = ceil(len(locations)/2)): one quantile-
    residual ACF panel (n_lags lags, y in [-1,1], shaded +/-2/sqrt(n) null
    band, lag 0 excluded -- see _draw_acf_panel) per location, for a single
    model/variant. See qq_location_mosaic for the shared grid/print-size/
    location_names/location_countries conventions, and
    LOCATION_QQACF_PANEL_SIZE_IN (module level, just above
    qq_location_mosaic) for why this figure is always exactly the same
    size as qq_location_mosaic's.
    """
    ncols = 2
    nrows = int(np.ceil(len(locations) / ncols))
    fig_w, fig_h = LOCATION_QQACF_PANEL_SIZE_IN[0] * ncols, LOCATION_QQACF_PANEL_SIZE_IN[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    pt = _resolve_print_pt(base_print_pt)
    scale = _print_scale(fig_w, target_print_width_in)

    for idx, loc in enumerate(locations):
        i, j = divmod(idx, ncols)
        ax = axes[i, j]
        n = _draw_acf_panel(ax, qr_by_location.get(loc), n_lags=n_lags,
                             empty_fontsize=_pt(pt, "annotation", scale))
        _n_annotation(ax, n, fontsize=_pt(pt, "annotation", scale))
        _panel_title(ax, (location_names or {}).get(loc, loc), fontsize=_pt(pt, "header", scale),
                     subtitle=(location_countries or {}).get(loc),
                     subtitle_fontsize=_pt(pt, "header", scale) * 0.7)
        if j == 0:
            ax.set_ylabel("Autocorrelation", fontsize=_pt(pt, "axis_label", scale))
        if i == nrows - 1:
            ax.set_xlabel("Lag (days)", fontsize=_pt(pt, "axis_label", scale))
        ax.tick_params(labelsize=_pt(pt, "tick", scale))

    for idx in range(len(locations), nrows * ncols):
        i, j = divmod(idx, ncols)
        axes[i, j].axis("off")

    rect_top = _place_wrapped_title_and_legend(
        fig, axes, suptitle,
        title_fontsize=_pt(pt, "suptitle", scale), legend_fontsize=_pt(pt, "legend", scale),
    )
    fig.tight_layout(rect=(0.03, 0, 1, rect_top))
    return fig
