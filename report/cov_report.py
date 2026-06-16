"""
PDF + LaTeX report generation for covariate and long-short ZA-GAS models.

Generates a multi-page PDF using matplotlib.backends.backend_pdf.PdfPages
and a companion .tex file with tables and discussion that mirrors the PDF.

The LaTeX source can be edited independently and recompiled with:
    pdflatex report.tex
without re-running any models.  See the notebook for instructions.
"""

from __future__ import annotations
import json
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm as _norm
from statsmodels.graphics.tsaplots import plot_acf

from diagnostics.tests import kupiec_test, christoffersen_test
from diagnostics.covariate_stats import (
    latex_summary_table, latex_seasonality_table, plot_covariate_page,
)


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette and helpers
# ─────────────────────────────────────────────────────────────────────────────

_C_HDR_BG  = "#1E3A6E"
_C_HDR_FG  = "white"
_C_ODD     = "#EEF2FF"
_C_EVEN    = "#FFFFFF"
_C_BEST    = "#BBF7D0"
_C_BAD     = "#FED7D7"
_C_PAGE    = "#9CA3AF"
_C_TITLE   = "#111827"
_C_BODY    = "#374151"

_LOWER_BETTER = {"crps", "rmse", "mad", "aic", "bic", "twcrps_90", "twcrps_95",
                 "qs_900", "qs_950", "qs_975", "qs_990",
                 "brier_90", "brier_95", "brier_99"}
_HIGHER_BETTER = {"loglik"}


def _fmt(x, d: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    try:
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)


def _wrap(text: str, width: int = 110) -> list:
    return textwrap.wrap(str(text), width=width) or [""]


def _save_fig(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Title page
# ─────────────────────────────────────────────────────────────────────────────

def _title_page(pdf: PdfPages, station: str, n_models: int, timestamp: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_facecolor(_C_HDR_BG)
    fig.patch.set_facecolor(_C_HDR_BG)
    ax.text(0.5, 0.70, "ZA-GAS Covariate & Long-Short Model Report",
            ha="center", va="center", fontsize=20, color="white",
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.58, f"Station: {station}",
            ha="center", va="center", fontsize=14, color="#CBD5E1",
            transform=ax.transAxes)
    ax.text(0.5, 0.50, f"Models evaluated: {n_models}",
            ha="center", va="center", fontsize=12, color="#CBD5E1",
            transform=ax.transAxes)
    ax.text(0.5, 0.42, f"Generated: {timestamp}",
            ha="center", va="center", fontsize=11, color="#94A3B8",
            transform=ax.transAxes)
    _save_fig(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Text / narrative page
# ─────────────────────────────────────────────────────────────────────────────

def _text_page(pdf: PdfPages, title: str, body: str, fontsize: int = 9) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=12, fontweight="bold",
            color=_C_TITLE, transform=ax.transAxes, va="top")
    lines = _wrap(body, width=130)
    txt   = "\n".join(lines)
    ax.text(0.0, 0.94, txt, fontsize=fontsize, color=_C_BODY,
            transform=ax.transAxes, va="top", fontfamily="monospace",
            wrap=True)
    _save_fig(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics comparison table page
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_page(
    pdf: PdfPages,
    frame: pd.DataFrame,
    title: str,
    cols: Optional[List[str]] = None,
) -> None:
    """Render a subset of the metrics DataFrame as a coloured table."""
    if cols is None:
        cols = ["model_id", "loglik", "aic", "bic", "crps",
                "rmse", "mad", "twcrps_90", "twcrps_95",
                "kup95_pvalue", "chrf95_pvalue"]
    present = [c for c in cols if c in frame.columns]
    sub = frame[present].copy().reset_index(drop=True)

    nrows, ncols = sub.shape
    fig_h = max(4.0, 0.35 * (nrows + 2))
    fig, ax = plt.subplots(figsize=(min(22, 2.0 + 1.6 * ncols), fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", color=_C_TITLE, pad=8)

    cell_text = []
    for _, row in sub.iterrows():
        cell_text.append([_fmt(row[c]) for c in present])

    col_labels = [c.replace("_", " ") for c in present]
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.auto_set_column_width(range(ncols))

    # Header styling
    for j in range(ncols):
        cell = tbl[0, j]
        cell.set_facecolor(_C_HDR_BG)
        cell.set_text_props(color="white", fontweight="bold")

    # Data rows with alternating colours and best-value highlighting
    for i in range(nrows):
        bg = _C_ODD if i % 2 == 0 else _C_EVEN
        for j in range(ncols):
            tbl[i + 1, j].set_facecolor(bg)

    # Highlight best per column
    for j, col in enumerate(present):
        try:
            vals = pd.to_numeric(sub[col], errors="coerce")
            if col in _LOWER_BETTER:
                best_idx = int(vals.idxmin())
            elif col in _HIGHER_BETTER:
                best_idx = int(vals.idxmax())
            else:
                continue
            tbl[best_idx + 1, j].set_facecolor(_C_BEST)
        except Exception:
            pass

    _save_fig(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic plots for a single model
# ─────────────────────────────────────────────────────────────────────────────

def _diag_page(
    pdf: PdfPages,
    model_id: str,
    pit: np.ndarray,
    qr: np.ndarray,
    f_arr: Optional[np.ndarray] = None,
    tv_names: Optional[list] = None,
    dates = None,
    sample: str = "IS",
) -> None:
    """4-panel diagnostic plot: PIT hist, QQ, ACF of QR, QR time series."""
    n_rows = 3 if f_arr is None else 4
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{model_id} — {sample} diagnostics", fontsize=12, fontweight="bold")

    r = qr[np.isfinite(qr)]
    u = pit[np.isfinite(pit)]

    # PIT histogram
    ax = axes[0, 0]
    ax.hist(u, bins=20, density=True, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axhline(1.0, color="red", lw=1.5, ls="--", label="Uniform(0,1)")
    ax.set_title("PIT histogram")
    ax.set_xlabel("PIT value")
    ax.legend(fontsize=8)

    # Normal QQ
    ax = axes[0, 1]
    percs = np.linspace(0.5, 99.5, 200)
    q_emp = np.percentile(r, percs)
    q_th  = _norm.ppf(percs / 100.0)
    ax.scatter(q_th, q_emp, s=7, alpha=0.5, color="steelblue")
    lim = max(abs(q_emp).max(), abs(q_th).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax.set_xlabel("Theoretical N(0,1)")
    ax.set_ylabel("Empirical")
    ax.set_title("Normal QQ plot")

    # ACF
    ax = axes[1, 0]
    try:
        plot_acf(r, ax=ax, lags=min(40, len(r) // 3), title="ACF of QR", zero=False, alpha=0.05)
    except Exception:
        ax.set_title("ACF of QR — unavailable")

    # QR time series
    ax = axes[1, 1]
    tt = dates if dates is not None else np.arange(len(r))
    ax.plot(tt[:len(r)], r, lw=0.6, color="steelblue", alpha=0.7)
    ax.axhline(0,    color="black", lw=0.5)
    ax.axhline(1.96, color="red",   lw=0.8, ls="--")
    ax.axhline(-1.96, color="red",  lw=0.8, ls="--")
    ax.set_title("QR time series")
    ax.set_xlabel("Time")

    plt.tight_layout()
    _save_fig(pdf, fig)


def _filtered_paths_page(
    pdf: PdfPages,
    model_id: str,
    paths: dict,
    dates = None,
    sample: str = "IS",
) -> None:
    """Plot filtered φ_t, ξ_t (if available), π_t."""
    tv  = paths.get("tv_names", [])
    f   = paths.get("f_arr") if "f_arr" in paths else paths.get("f_arr_oos")
    pi  = paths.get("pi")    if "pi"    in paths else paths.get("pi_oos")

    n_panels = len(tv) + 1  # +1 for pi
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(f"{model_id} — filtered states ({sample})", fontsize=11)

    tt = dates if dates is not None else np.arange(len(pi) if pi is not None else 1)

    for j, name in enumerate(tv):
        ax = axes[j]
        if f is not None and j < f.shape[1]:
            ax.plot(tt[:len(f)], f[:, j], lw=0.8, color="steelblue")
        ax.set_ylabel(rf"$\{name}_t$")
        ax.grid(axis="y", lw=0.3)

    if pi is not None:
        axes[-1].plot(tt[:len(pi)], pi, lw=0.8, color="darkorange")
        axes[-1].set_ylabel(r"$\pi_t$")
        axes[-1].set_ylim(-0.05, 1.05)
        axes[-1].grid(axis="y", lw=0.3)

    plt.tight_layout()
    _save_fig(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Automated discussion and conclusion
# ─────────────────────────────────────────────────────────────────────────────

def _auto_discussion(
    metrics_by_model: Dict[str, dict],
    inf_by_model: Optional[Dict[str, pd.DataFrame]] = None,
) -> str:
    """
    Generate an automated discussion paragraph.

    Identifies:
      - Best and worst models by OOS CRPS
      - Models where AIC/BIC improved over baseline
      - Covariate significance patterns
      - Coverage test rejections
    """
    oos = {k: v["oos"] for k, v in metrics_by_model.items() if "oos" in v}
    if not oos:
        return "No OOS metrics available."

    frame = pd.DataFrame(oos).T.apply(pd.to_numeric, errors="coerce")

    lines = []

    # Best / worst by CRPS
    if "crps" in frame.columns:
        best   = frame["crps"].idxmin()
        worst  = frame["crps"].idxmax()
        lines.append(
            f"By OOS CRPS, the best-performing model is '{best}' "
            f"(CRPS = {_fmt(frame.loc[best,'crps'])}) and the worst is "
            f"'{worst}' (CRPS = {_fmt(frame.loc[worst,'crps'])})."
        )

    # AIC / BIC
    if "aic" in frame.columns:
        best_aic = frame["aic"].idxmin()
        lines.append(
            f"The lowest IS AIC belongs to '{best_aic}' "
            f"(AIC = {_fmt(frame.loc[best_aic,'aic'])})."
        )

    # Coverage
    reject_ks = []
    if "kup95_reject" in frame.columns:
        reject_ks = list(frame.index[frame["kup95_reject"].astype(bool)])
    if reject_ks:
        lines.append(
            f"Models failing the Kupiec 95\\% unconditional coverage test (α=0.05): "
            + ", ".join(f"'{m}'" for m in reject_ks[:8]) + "."
        )
    else:
        lines.append("No model fails the Kupiec 95\\% coverage test.")

    # Tail CRPS
    for key in ["twcrps_90", "twcrps_95"]:
        if key in frame.columns:
            best_tw = frame[key].idxmin()
            lvl = key.split("_")[1] + "%"
            lines.append(
                f"Best tail-weighted CRPS ({lvl} threshold): '{best_tw}' "
                f"({_fmt(frame.loc[best_tw, key])})."
            )

    # Significance summary
    if inf_by_model:
        sig_any = []
        for mid, inf in inf_by_model.items():
            if inf is not None and "sig_5pct" in inf.columns:
                sig_cols = inf[inf["sig_5pct"] & inf["parameter"].str.startswith("gamma")]
                if not sig_cols.empty:
                    sig_any.append(
                        f"'{mid}': {', '.join(sig_cols['parameter'].tolist()[:4])}"
                    )
        if sig_any:
            lines.append(
                "Models with at least one covariate coefficient significant at 5\\%: "
                + "; ".join(sig_any[:6]) + "."
            )
        else:
            lines.append(
                "No covariate model shows covariate coefficients significant at the 5\\% level; "
                "the null H0: Γ=0 cannot be rejected in any case."
            )

    return " ".join(lines)


def _auto_conclusion(metrics_by_model: Dict[str, dict]) -> str:
    """Automated conclusion: which model to recommend and why."""
    oos = {k: v["oos"] for k, v in metrics_by_model.items() if "oos" in v}
    if not oos:
        return "No OOS metrics available for conclusion."

    frame = pd.DataFrame(oos).T.apply(pd.to_numeric, errors="coerce")

    if "crps" not in frame.columns:
        return "CRPS unavailable; no conclusion can be drawn."

    best = frame["crps"].idxmin()
    tw90 = frame["twcrps_90"].idxmin() if "twcrps_90" in frame.columns else None

    lines = [
        f"Based on OOS CRPS, the recommended model is '{best}'.",
    ]
    if tw90 and tw90 != best:
        lines.append(
            f"For upper-tail performance specifically, '{tw90}' performs best "
            "by threshold-weighted CRPS at the 90th percentile."
        )

    # Check JB
    if "jb_pvalue" in frame.columns:
        bad_jb = list(frame.index[frame["jb_pvalue"] < 0.01])
        if bad_jb:
            lines.append(
                f"Models with strong quantile-residual non-normality "
                f"(JB p<0.01): {', '.join(f'{m!r}' for m in bad_jb[:5])}; "
                "these may be misspecified."
            )

    lines.append(
        "In general, long-short models are expected to improve upper-tail "
        "performance when Niño 3.4 carries predictive signal. "
        "Standard exogenous models are preferable when the covariate effect "
        "is linear and contemporaneous."
    )
    return " ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX source generation
# ─────────────────────────────────────────────────────────────────────────────

def _latex_preamble(title: str, author: str = "") -> str:
    return rf"""
\documentclass[12pt,a4paper]{{article}}
\usepackage{{booktabs,longtable,lscape,geometry,hyperref,amsmath}}
\geometry{{margin=2.0cm}}
\title{{{title}}}
\author{{{author}}}
\date{{\today}}
\begin{{document}}
\maketitle
\tableofcontents
\clearpage
""".lstrip()


def _latex_metrics_section(
    metrics_by_model: Dict[str, dict],
    sample: str = "oos",
    caption_prefix: str = "Out-of-sample",
) -> str:
    oos = {k: v.get(sample, {}) for k, v in metrics_by_model.items()}
    if not oos:
        return ""
    frame = pd.DataFrame(oos).T

    num_cols = ["loglik", "aic", "bic", "crps", "rmse", "mad",
                "twcrps_90", "twcrps_95", "qs_900", "qs_950",
                "brier_90", "brier_95", "kup95_pvalue", "chrf95_pvalue"]
    present = [c for c in num_cols if c in frame.columns]
    sub = frame[present].apply(pd.to_numeric, errors="coerce")

    header = " & ".join(
        c.replace("_", r"\_") for c in ["model"] + present
    ) + r" \\"

    rows = []
    for mid, row in sub.iterrows():
        cells = [str(mid)] + [_fmt(row.get(c, np.nan)) for c in present]
        rows.append(" & ".join(cells) + r" \\")

    n_cols = len(present) + 1
    col_spec = "l" + "r" * len(present)

    return (
        r"\begin{landscape}" + "\n"
        + r"\begin{longtable}{" + col_spec + "}\n"
        + rf"\caption{{{caption_prefix} metrics}} \label{{tab:metrics_{sample}}} \\" + "\n"
        + r"\hline" + "\n"
        + header + "\n"
        + r"\hline" + "\n"
        + r"\endfirsthead" + "\n"
        + r"\hline" + "\n"
        + header + "\n"
        + r"\hline" + "\n"
        + r"\endhead" + "\n"
        + r"\hline \endfoot" + "\n"
        + "\n".join(rows) + "\n"
        + r"\end{longtable}" + "\n"
        + r"\end{landscape}" + "\n"
    )


def _latex_inference_section(
    inf_by_model: Dict[str, pd.DataFrame],
    max_rows_per_table: int = 40,
) -> str:
    """Emit one longtable per model for the covariate coefficients."""
    parts = []
    for mid, df in inf_by_model.items():
        if df is None or df.empty:
            continue
        # Filter to covariate / LS coefficients
        cov_df = df[
            df["parameter"].str.startswith("gamma")
            | df["parameter"].str.startswith("alpha_L")
            | df["parameter"].str.startswith("beta_L")
            | df["parameter"].str.startswith("gamma_S")
        ].copy()
        if cov_df.empty:
            cov_df = df.copy()

        from models.inference import latex_inference_table
        parts.append(
            rf"\subsection{{Model: {mid.replace('_', ' ')}}}" + "\n"
            + latex_inference_table(
                cov_df,
                caption=f"Covariate inference — {mid}",
                label=f"tab:inf_{mid}",
            )
        )
    return "\n\n".join(parts)


def _latex_postamble() -> str:
    return "\n\\end{document}\n"


# ─────────────────────────────────────────────────────────────────────────────
# Covariate descriptive stats pages
# ─────────────────────────────────────────────────────────────────────────────

def _covariate_summary_page(
    pdf: PdfPages,
    raw_summary: pd.DataFrame,
    std_summary: pd.DataFrame,
) -> None:
    """Render raw vs normalised summary stats as side-by-side tables."""
    for label, df in [("Raw", raw_summary), ("Normalised", std_summary)]:
        if df is None or df.empty:
            continue
        show_cols = ["mean", "std", "min", "q05", "q25", "q50",
                     "q75", "q95", "max", "skew", "kurt", "acf_1"]
        present = [c for c in show_cols if c in df.columns]
        sub = df[present].reset_index()

        nrows = len(sub)
        ncols = len(present) + 1
        fig_h = max(4.0, 0.3 * (nrows + 2))
        fig, ax = plt.subplots(figsize=(min(22, 1.4 * ncols + 2), fig_h))
        ax.axis("off")
        ax.set_title(f"Covariate descriptive statistics — {label}",
                     fontsize=11, fontweight="bold", color=_C_TITLE, pad=8)

        cell_text = []
        for _, row in sub.iterrows():
            cells = [str(row["variable"])]
            for c in present:
                v = row.get(c, np.nan)
                try:
                    cells.append(f"{float(v):.3f}" if np.isfinite(float(v)) else "--")
                except Exception:
                    cells.append("--")
            cell_text.append(cells)

        col_labels = ["variable"] + [c.replace("_", " ") for c in present]
        tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.5)
        tbl.auto_set_column_width(range(ncols))
        for j in range(ncols):
            cell = tbl[0, j]
            cell.set_facecolor(_C_HDR_BG)
            cell.set_text_props(color="white", fontweight="bold")
        for i in range(nrows):
            bg = _C_ODD if i % 2 == 0 else _C_EVEN
            for j in range(ncols):
                tbl[i + 1, j].set_facecolor(bg)

        _save_fig(pdf, fig)


def _covariate_seasonality_page(
    pdf: PdfPages,
    seasonality_raw: pd.DataFrame,
    seasonality_std: pd.DataFrame,
) -> None:
    """Heatmap of monthly means for each covariate."""
    for label, df in [("Raw", seasonality_raw), ("Normalised", seasonality_std)]:
        if df is None or df.empty:
            continue
        cols = list(df.columns)
        n_cols = len(cols)
        fig, ax = plt.subplots(figsize=(min(20, max(8, n_cols * 0.7 + 2)), 5))
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        mat = df.values.astype(float)
        vext = np.nanmax(np.abs(mat))
        im = ax.imshow(mat.T, cmap="RdBu_r", vmin=-vext, vmax=vext,
                       aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        ax.set_xticks(range(12))
        ax.set_xticklabels(month_names, fontsize=8)
        ax.set_yticks(range(n_cols))
        ax.set_yticklabels([c[:20] for c in cols], fontsize=7)
        ax.set_title(f"Covariate seasonality (monthly mean, {label})",
                     fontsize=10, fontweight="bold")
        plt.tight_layout()
        _save_fig(pdf, fig)


def _covariate_acf_grid_page(
    pdf: PdfPages,
    raw_cols: dict,
    std_cols: dict,
    all_dates,
    nlags: int = 60,
    max_per_page: int = 8,
) -> None:
    """Grid of ACF plots (raw and normalised) for up to max_per_page covariates."""
    col_names = list(raw_cols.keys())
    for start in range(0, len(col_names), max_per_page):
        chunk = col_names[start : start + max_per_page]
        n = len(chunk)
        fig, axes = plt.subplots(n, 2, figsize=(13, 2.5 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(f"Covariate ACF (raw left, normalised right) — cols {start+1}–{start+n}",
                     fontsize=9, fontweight="bold")
        for row_idx, col in enumerate(chunk):
            for col_idx, (series, lbl) in enumerate([
                (raw_cols.get(col), "raw"),
                (std_cols.get(col), "norm"),
            ]):
                ax = axes[row_idx, col_idx]
                if series is None:
                    ax.axis("off")
                    continue
                vals = pd.Series(series.values if hasattr(series, "values") else series).dropna()
                try:
                    plot_acf(vals, ax=ax,
                             lags=min(nlags, len(vals) // 3),
                             title=f"{col[:25]} ({lbl})",
                             zero=False, alpha=0.05)
                    ax.set_xlabel("Lag (days)", fontsize=6)
                    ax.title.set_fontsize(7)
                except Exception:
                    ax.set_title(f"{col[:20]} — ACF err")
        plt.tight_layout()
        _save_fig(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter covariance / correlation pages
# ─────────────────────────────────────────────────────────────────────────────

def _param_cov_corr_page(
    pdf: PdfPages,
    cov: np.ndarray,
    corr: np.ndarray,
    param_names: List[str],
    model_id: str,
) -> None:
    """Two heatmaps side-by-side: covariance (left) and correlation (right)."""
    from models.inference import plot_cov_corr_heatmap

    fig_cov  = plot_cov_corr_heatmap(cov,  param_names,
                                     f"{model_id} — covariance", kind="cov")
    fig_corr = plot_cov_corr_heatmap(corr, param_names,
                                     f"{model_id} — correlation", kind="corr")
    _save_fig(pdf, fig_cov)
    _save_fig(pdf, fig_corr)


def _latex_cov_corr_section(
    inf_by_model: Dict[str, pd.DataFrame],
    cache_dir: Optional[Path],
) -> str:
    """
    Emit covariance + correlation tables for every model that has a saved
    covariance_matrix.npy in cache_dir/<model_id>/.
    """
    from models.inference import correlation_from_cov, latex_cov_corr_tables

    if cache_dir is None:
        return ""

    parts = []
    for mid, inf_df in inf_by_model.items():
        cov_path = Path(cache_dir) / mid / "covariance_matrix.npy"
        if not cov_path.exists() or inf_df is None or inf_df.empty:
            continue
        try:
            cov  = np.load(str(cov_path))
            corr = correlation_from_cov(cov)
            names = list(inf_df["parameter"])
            n = min(len(names), cov.shape[0])
            cov, corr, names = cov[:n, :n], corr[:n, :n], names[:n]
            parts.append(
                rf"\subsection{{Model: {mid.replace('_', ' ')}}}" + "\n"
                + latex_cov_corr_tables(
                    cov, corr, names,
                    caption_prefix=mid.replace("_", " "),
                    label_prefix=f"tab_{mid}",
                )
            )
        except Exception as e:
            parts.append(rf"\subsection{{Model: {mid}}} Error loading matrix: {e}")

    if not parts:
        return ""
    return r"\section{Parameter Covariance and Correlation Matrices}" + "\n\n" + "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main report function
# ─────────────────────────────────────────────────────────────────────────────

def render_covariate_report(
    report_dir: Path,
    station: str,
    metrics_by_model: Dict[str, dict],
    diag_by_model: Optional[Dict[str, dict]] = None,
    inf_by_model: Optional[Dict[str, pd.DataFrame]] = None,
    is_dates = None,
    oos_dates = None,
    timestamp: str = "",
    covariate_stats: Optional[dict] = None,
    cache_dir: Optional[Path] = None,
    all_dates = None,
) -> dict:
    """
    Generate PDF + LaTeX + all individual plot files in report_dir.

    Parameters
    ----------
    report_dir       : output directory
    station          : station name for titles
    metrics_by_model : {model_id: {"is": {...}, "oos": {...}}}
    diag_by_model    : {model_id: {is_pit, oos_pit, is_qr, oos_qr, is_paths, oos_paths}}
    inf_by_model     : {model_id: inference_dataframe}
    is_dates / oos_dates / all_dates : date arrays for plots
    timestamp        : datetime string for title page
    covariate_stats  : output of compute_all_covariate_stats() — descriptive stats dict
    cache_dir        : path to cache root — used to load covariance_matrix.npy per model

    Returns
    -------
    dict with keys "pdf_path", "tex_path"
    """
    import datetime
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = report_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    ts = timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    pdf_path = report_dir / "report.pdf"
    tex_path = report_dir / "report.tex"
    n_models = len(metrics_by_model)

    # ── Build metrics DataFrames ──────────────────────────────────────────────
    oos_frame = pd.DataFrame(
        {mid: v.get("oos", {}) for mid, v in metrics_by_model.items()}
    ).T.apply(pd.to_numeric, errors="coerce").reset_index().rename(columns={"index": "model_id"})

    is_frame = pd.DataFrame(
        {mid: v.get("is", {}) for mid, v in metrics_by_model.items()}
    ).T.apply(pd.to_numeric, errors="coerce").reset_index().rename(columns={"index": "model_id"})

    # ── PDF ───────────────────────────────────────────────────────────────────
    with PdfPages(str(pdf_path)) as pdf:
        _title_page(pdf, station, n_models, ts)

        # Model inventory
        inv_lines = "\n".join(f"  [{i+1}] {mid}" for i, mid in enumerate(metrics_by_model))
        _text_page(pdf, "Model inventory", inv_lines)

        # ── Covariate descriptive statistics ─────────────────────────────────
        if covariate_stats:
            raw_summary = covariate_stats.get("raw_summary")
            std_summary = covariate_stats.get("std_summary")
            seas_raw    = covariate_stats.get("seasonality_raw")
            seas_std    = covariate_stats.get("seasonality_std")
            raw_cols_d  = covariate_stats.get("raw_cols", {})
            std_cols_d  = covariate_stats.get("std_cols", {})
            _dates_all  = all_dates if all_dates is not None else (
                is_dates.append(oos_dates) if (is_dates is not None and oos_dates is not None)
                else is_dates
            )

            if raw_summary is not None and not raw_summary.empty:
                _covariate_summary_page(pdf, raw_summary, std_summary)
            if seas_raw is not None and not seas_raw.empty:
                _covariate_seasonality_page(pdf, seas_raw, seas_std)
            if raw_cols_d:
                _covariate_acf_grid_page(pdf, raw_cols_d, std_cols_d,
                                         _dates_all, nlags=60)
            # Individual 6-panel plots for each covariate
            if raw_cols_d and _dates_all is not None:
                for col in raw_cols_d:
                    if col not in std_cols_d:
                        continue
                    try:
                        raw_s = raw_cols_d[col]
                        std_s = std_cols_d[col]
                        if hasattr(raw_s, "values"):
                            raw_s = pd.Series(raw_s.values, index=_dates_all[:len(raw_s)])
                        if hasattr(std_s, "values"):
                            std_s = pd.Series(std_s.values, index=_dates_all[:len(std_s)])
                        fig = plot_covariate_page(raw_s, std_s, _dates_all, col)
                        fig.savefig(plots_dir / f"cov_{col[:40]}.png",
                                    dpi=100, bbox_inches="tight")
                        _save_fig(pdf, fig)
                    except Exception:
                        pass

        # IS metrics table
        if not is_frame.empty:
            _metrics_page(pdf, is_frame, "In-sample metrics")

        # OOS metrics table
        if not oos_frame.empty:
            _metrics_page(pdf, oos_frame, "Out-of-sample metrics")
            _metrics_page(
                pdf, oos_frame, "OOS tail metrics",
                cols=["model_id", "crps", "twcrps_90", "twcrps_95",
                      "qs_900", "qs_950", "qs_975", "qs_990",
                      "brier_90", "brier_95", "brier_99",
                      "kup95_pvalue", "chrf95_pvalue"],
            )

        # Per-model diagnostics + covariance / correlation
        if diag_by_model:
            for mid, diag in diag_by_model.items():
                # IS diagnostics
                if diag.get("is_pit") is not None and diag.get("is_qr") is not None:
                    _diag_page(
                        pdf, mid, diag["is_pit"], diag["is_qr"],
                        f_arr=diag.get("is_paths", {}).get("f_arr"),
                        tv_names=diag.get("is_paths", {}).get("tv_names"),
                        dates=is_dates, sample="IS",
                    )
                # OOS diagnostics
                if diag.get("oos_pit") is not None and diag.get("oos_qr") is not None:
                    _diag_page(
                        pdf, mid, diag["oos_pit"], diag["oos_qr"],
                        f_arr=diag.get("oos_paths", {}).get("f_arr_oos"),
                        tv_names=diag.get("oos_paths", {}).get("tv_names"),
                        dates=oos_dates, sample="OOS",
                    )
                # Filtered paths
                if "is_paths" in diag and diag["is_paths"]:
                    _filtered_paths_page(pdf, mid, diag["is_paths"], dates=is_dates, sample="IS")
                if "oos_paths" in diag and diag["oos_paths"]:
                    _filtered_paths_page(pdf, mid, diag["oos_paths"], dates=oos_dates, sample="OOS")

                # Covariance / correlation heatmaps (loaded from cache)
                if cache_dir is not None and inf_by_model and mid in inf_by_model:
                    cov_path = Path(cache_dir) / mid / "covariance_matrix.npy"
                    inf_df   = inf_by_model.get(mid)
                    if cov_path.exists() and inf_df is not None and not inf_df.empty:
                        try:
                            from models.inference import correlation_from_cov
                            cov  = np.load(str(cov_path))
                            corr = correlation_from_cov(cov)
                            names = list(inf_df["parameter"])
                            n = min(len(names), cov.shape[0])
                            _param_cov_corr_page(
                                pdf, cov[:n, :n], corr[:n, :n], names[:n], mid
                            )
                        except Exception:
                            pass

        # Discussion
        discussion = _auto_discussion(metrics_by_model, inf_by_model)
        _text_page(pdf, "Discussion", discussion)

        # Conclusion
        conclusion = _auto_conclusion(metrics_by_model)
        _text_page(pdf, "Conclusion", conclusion)

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    tex_parts = [
        _latex_preamble(
            title=f"ZA-GAS Covariate \\& Long-Short Models — {station}",
            author="",
        ),
        r"\section{Introduction}",
        _auto_discussion(metrics_by_model, inf_by_model),
        "",
    ]

    # Covariate descriptive statistics section
    if covariate_stats:
        tex_parts.append(r"\section{Covariate Descriptive Statistics}")
        raw_sum = covariate_stats.get("raw_summary")
        std_sum = covariate_stats.get("std_summary")
        seas_r  = covariate_stats.get("seasonality_raw")
        seas_s  = covariate_stats.get("seasonality_std")
        if raw_sum is not None and not raw_sum.empty:
            tex_parts.append(r"\subsection{Summary Statistics (Before Normalisation)}")
            tex_parts.append(latex_summary_table(
                raw_sum,
                caption="Covariate summary statistics — raw series",
                label="tab:cov_summary_raw",
            ))
            if std_sum is not None and not std_sum.empty:
                tex_parts.append(r"\subsection{Summary Statistics (After Normalisation)}")
                tex_parts.append(latex_summary_table(
                    std_sum,
                    caption="Covariate summary statistics — normalised series",
                    label="tab:cov_summary_std",
                ))
        if seas_r is not None and not seas_r.empty:
            tex_parts.append(r"\subsection{Monthly Seasonality (Raw)}")
            tex_parts.append(latex_seasonality_table(
                seas_r,
                caption="Covariate monthly means — raw series",
                label="tab:cov_seas_raw",
            ))
            if seas_s is not None and not seas_s.empty:
                tex_parts.append(r"\subsection{Monthly Seasonality (Normalised)}")
                tex_parts.append(latex_seasonality_table(
                    seas_s,
                    caption="Covariate monthly means — normalised series",
                    label="tab:cov_seas_std",
                ))
        tex_parts.append("")

    tex_parts += [
        r"\section{In-Sample Metrics}",
        _latex_metrics_section(metrics_by_model, sample="is", caption_prefix="In-sample"),
        "",
        r"\section{Out-of-Sample Metrics}",
        _latex_metrics_section(metrics_by_model, sample="oos", caption_prefix="Out-of-sample"),
        "",
    ]

    if inf_by_model:
        tex_parts += [
            r"\section{Covariate Inference}",
            _latex_inference_section(inf_by_model),
            "",
        ]

    # Full inference tables (all parameters, not just covariate betas)
    if inf_by_model:
        parts_full = []
        for mid, df in inf_by_model.items():
            if df is None or df.empty:
                continue
            from models.inference import latex_inference_table
            parts_full.append(
                rf"\subsection{{Model: {mid.replace('_', ' ')}}}" + "\n"
                + latex_inference_table(
                    df,
                    caption=f"Full parameter inference — {mid}",
                    label=f"tab:full_inf_{mid}",
                )
            )
        if parts_full:
            tex_parts += [
                r"\section{Full Parameter Inference Tables}",
                "\n\n".join(parts_full),
                "",
            ]

    # Covariance and correlation matrices
    cov_corr_tex = _latex_cov_corr_section(inf_by_model or {}, cache_dir)
    if cov_corr_tex:
        tex_parts += [cov_corr_tex, ""]

    tex_parts += [
        r"\section{Discussion}",
        _auto_discussion(metrics_by_model, inf_by_model),
        "",
        r"\section{Conclusion}",
        _auto_conclusion(metrics_by_model),
        "",
        _latex_postamble(),
    ]

    tex_path.write_text("\n".join(tex_parts), encoding="utf-8")

    # Save individual metrics CSVs
    oos_frame.to_csv(report_dir / "oos_metrics.csv", index=False)
    is_frame.to_csv(report_dir / "is_metrics.csv", index=False)

    return {"pdf_path": str(pdf_path), "tex_path": str(tex_path)}


def recompile_pdf_from_latex(tex_path: str | Path) -> bool:
    """
    Recompile the PDF from the .tex file using pdflatex.

    Call this after editing report.tex manually.
    Returns True on success, False otherwise.
    """
    import subprocess
    tex_path = Path(tex_path)
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(tex_path.name)],
            capture_output=True, text=True, cwd=str(tex_path.parent),
        )
        # Run twice for TOC/references
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(tex_path.name)],
            capture_output=True, text=True, cwd=str(tex_path.parent),
        )
        return result.returncode == 0
    except FileNotFoundError:
        print(
            "pdflatex not found on PATH. To recompile, run:\n"
            f"  cd {tex_path.parent}\n"
            f"  pdflatex {tex_path.name}\n"
            "Alternatively, upload report.tex to Overleaf."
        )
        return False
