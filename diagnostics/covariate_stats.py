"""
Descriptive statistics and diagnostic plots for covariate series.

Computes summary tables, seasonality, ACF, and histograms for each
covariate block, both before and after standardisation.
All functions return plain DataFrames or matplotlib figures so they
can be embedded in the PDF report and saved as standalone files.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as _stats
from statsmodels.graphics.tsaplots import plot_acf


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def _series_stats(s: pd.Series) -> dict:
    """Scalar summary statistics for one series, ignoring NaN."""
    x = s.dropna().values
    if len(x) == 0:
        return {}
    acf1   = float(pd.Series(x).autocorr(lag=1))  if len(x) > 1   else np.nan
    acf365 = float(pd.Series(x).autocorr(lag=365)) if len(x) > 365 else np.nan
    return {
        "count":   len(x),
        "mean":    float(np.mean(x)),
        "std":     float(np.std(x, ddof=1)),
        "min":     float(np.min(x)),
        "q05":     float(np.percentile(x,  5)),
        "q25":     float(np.percentile(x, 25)),
        "q50":     float(np.percentile(x, 50)),
        "q75":     float(np.percentile(x, 75)),
        "q95":     float(np.percentile(x, 95)),
        "max":     float(np.max(x)),
        "skew":    float(_stats.skew(x)),
        "kurt":    float(_stats.kurtosis(x)),
        "acf_1":   acf1,
        "acf_365": acf365,
    }


def summary_table(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Summary statistics for every column in df.

    Returns a DataFrame with rows = columns of df,
    columns = statistics.
    """
    rows = []
    for col in df.columns:
        d = _series_stats(df[col])
        d["variable"] = col
        d["sample"]   = label
        rows.append(d)
    return pd.DataFrame(rows).set_index("variable") if rows else pd.DataFrame()


def seasonality_table(df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Monthly mean for each column in df.

    Returns a DataFrame with rows = month (1-12), columns = variables.
    """
    tmp = df.copy()
    tmp.index = dates[:len(df)]
    monthly = tmp.groupby(tmp.index.month).mean()
    monthly.index.name = "month"
    return monthly


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _hist_ax(ax, x, title, xlabel="", color="steelblue"):
    x = x[np.isfinite(x)]
    ax.hist(x, bins=40, density=True, color=color, alpha=0.75, edgecolor="w")
    xg = np.linspace(x.min(), x.max(), 200)
    ax.plot(xg, _stats.norm.pdf(xg, x.mean(), x.std()), "r--", lw=1.2, label="N fit")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.legend(fontsize=7)


def plot_covariate_page(
    raw_col: pd.Series,
    std_col: pd.Series,
    dates: pd.DatetimeIndex,
    name: str,
    nlags: int = 60,
) -> plt.Figure:
    """
    6-panel diagnostic figure for one covariate column:
      [0,0] raw histogram   [0,1] normalised histogram
      [1,0] raw ACF         [1,1] normalised ACF
      [2,0] monthly boxplot [2,1] time series (raw)
    """
    fig = plt.figure(figsize=(13, 10))
    fig.suptitle(f"Covariate diagnostics — {name}", fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    raw = raw_col.dropna().values
    std = std_col.dropna().values

    # raw histogram
    ax = fig.add_subplot(gs[0, 0])
    _hist_ax(ax, raw, "Raw histogram", color="steelblue")

    # normalised histogram
    ax = fig.add_subplot(gs[0, 1])
    _hist_ax(ax, std, "Normalised histogram", color="darkorange")

    # raw ACF
    ax = fig.add_subplot(gs[1, 0])
    raw_clean = pd.Series(raw)
    try:
        plot_acf(raw_clean, ax=ax, lags=min(nlags, len(raw_clean) // 3),
                 title="ACF (raw)", zero=False, alpha=0.05)
        ax.set_xlabel("Lag (days)", fontsize=7)
    except Exception:
        ax.set_title("ACF unavailable")

    # normalised ACF
    ax = fig.add_subplot(gs[1, 1])
    std_clean = pd.Series(std)
    try:
        plot_acf(std_clean, ax=ax, lags=min(nlags, len(std_clean) // 3),
                 title="ACF (normalised)", zero=False, alpha=0.05)
        ax.set_xlabel("Lag (days)", fontsize=7)
    except Exception:
        ax.set_title("ACF unavailable")

    # monthly boxplot
    ax = fig.add_subplot(gs[2, 0])
    try:
        raw_s = pd.Series(raw_col.values, index=dates[:len(raw_col)])
        monthly_groups = [raw_s[raw_s.index.month == m].dropna().values for m in range(1, 13)]
        month_labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]
        bp = ax.boxplot(monthly_groups, patch_artist=True, medianprops={"color":"red","lw":1.5})
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_labels, fontsize=7)
        ax.set_title("Monthly distribution (raw)", fontsize=8)
        ax.set_xlabel("Month", fontsize=7)
    except Exception:
        ax.set_title("Monthly plot unavailable")

    # time series
    ax = fig.add_subplot(gs[2, 1])
    try:
        t = dates[:len(raw_col)]
        ax.plot(t, raw_col.values, lw=0.5, color="steelblue", alpha=0.7)
        # 12-month rolling mean
        rm = raw_col.rolling(365, min_periods=180, center=True).mean()
        ax.plot(t, rm.values, lw=1.8, color="red", label="365d MA")
        ax.set_title("Time series + 365d mean", fontsize=8)
        ax.legend(fontsize=7)
    except Exception:
        ax.set_title("Time series unavailable")

    return fig


def plot_all_covariate_pages(
    raw_blocks: Dict[str, pd.DataFrame],
    std_blocks: Dict[str, pd.DataFrame],
    train_dates: pd.DatetimeIndex,
    all_dates: pd.DatetimeIndex,
    save_dir: Path,
    nlags: int = 60,
) -> Dict[str, Path]:
    """
    Save one figure per covariate column across all blocks.
    Returns {col_name: png_path}.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    seen  = set()

    for bname, raw_df in raw_blocks.items():
        std_df = std_blocks.get(bname)
        if std_df is None:
            continue
        for col in raw_df.columns:
            if col in seen:
                continue
            seen.add(col)
            fig = plot_covariate_page(
                raw_col   = raw_df[col],
                std_col   = std_df[col],
                dates     = all_dates[:len(raw_df)],
                name      = col,
                nlags     = nlags,
            )
            safe = col.replace("/", "_").replace(" ", "_")
            path = save_dir / f"cov_{safe}.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved[col] = path

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate stats across all blocks
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_covariate_stats(
    raw_blocks: Dict[str, pd.DataFrame],
    std_blocks: Dict[str, pd.DataFrame],
    train_dates: pd.DatetimeIndex,
    all_dates: pd.DatetimeIndex,
) -> dict:
    """
    Compute summary and seasonality tables for every unique covariate column,
    both before and after normalisation (training portion only for stats).

    Returns:
        {
          "raw_summary":  pd.DataFrame  (rows=columns, cols=stats),
          "std_summary":  pd.DataFrame,
          "seasonality_raw": pd.DataFrame  (rows=month, cols=variables),
          "seasonality_std": pd.DataFrame,
        }
    """
    n_train = len(train_dates)

    # Collect unique columns (first occurrence wins)
    raw_cols = {}
    std_cols = {}
    for bname, raw_df in raw_blocks.items():
        std_df = std_blocks.get(bname)
        for col in raw_df.columns:
            if col not in raw_cols:
                raw_cols[col] = raw_df[col]
                if std_df is not None and col in std_df.columns:
                    std_cols[col] = std_df[col]

    raw_train = pd.DataFrame(raw_cols).iloc[:n_train]
    std_train = pd.DataFrame(std_cols).iloc[:n_train]
    raw_all   = pd.DataFrame(raw_cols)
    std_all   = pd.DataFrame(std_cols)

    train_idx = all_dates[:n_train]

    return {
        "raw_summary":     summary_table(raw_train, label="raw"),
        "std_summary":     summary_table(std_train, label="normalised"),
        "seasonality_raw": seasonality_table(raw_train, train_idx),
        "seasonality_std": seasonality_table(std_train, train_idx),
        "raw_cols":  raw_cols,
        "std_cols":  std_cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX tables
# ─────────────────────────────────────────────────────────────────────────────

def latex_summary_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """LaTeX longtable for summary statistics."""
    stat_cols = ["count", "mean", "std", "min", "q05", "q25",
                 "q50", "q75", "q95", "max", "skew", "kurt",
                 "acf_1", "acf_365"]
    present = [c for c in stat_cols if c in df.columns]
    col_spec = "l" + "r" * len(present)
    header   = "Variable & " + " & ".join(
        c.replace("_", r"\_") for c in present
    ) + r" \\"

    def _f(x, d=3):
        try:
            v = float(x)
            if not np.isfinite(v):
                return "--"
            return f"{v:.{d}f}"
        except Exception:
            return "--"

    rows = []
    for var, row in df.iterrows():
        cells = [str(var).replace("_", r"\_")] + [_f(row.get(c, np.nan)) for c in present]
        rows.append(" & ".join(cells) + r" \\")

    return (
        r"\begin{landscape}" + "\n"
        r"\begin{longtable}{" + col_spec + "}\n"
        rf"\caption{{{caption}}} \label{{{label}}} \\" + "\n"
        r"\hline" + "\n"
        + header + "\n"
        r"\hline\endfirsthead" + "\n"
        r"\hline" + "\n"
        + header + "\n"
        r"\hline\endhead" + "\n"
        r"\hline\endfoot" + "\n"
        + "\n".join(rows) + "\n"
        r"\end{longtable}" + "\n"
        r"\end{landscape}" + "\n"
    )


def latex_seasonality_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """LaTeX longtable for monthly means (rows=months, cols=covariates)."""
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    cols = list(df.columns)
    col_spec = "l" + "r" * len(cols)
    header   = "Month & " + " & ".join(
        c.replace("_", r"\_") for c in cols
    ) + r" \\"

    def _f(x):
        try:
            v = float(x)
            return f"{v:.3f}" if np.isfinite(v) else "--"
        except Exception:
            return "--"

    rows = []
    for m_idx, row in df.iterrows():
        label_m = month_names[int(m_idx) - 1] if 1 <= int(m_idx) <= 12 else str(m_idx)
        cells = [label_m] + [_f(row[c]) for c in cols]
        rows.append(" & ".join(cells) + r" \\")

    return (
        r"\begin{landscape}" + "\n"
        r"\begin{longtable}{" + col_spec + "}\n"
        rf"\caption{{{caption}}} \label{{{label}}} \\" + "\n"
        r"\hline" + "\n"
        + header + "\n"
        r"\hline\endfirsthead" + "\n"
        r"\hline" + "\n"
        + header + "\n"
        r"\hline\endhead" + "\n"
        r"\hline\endfoot" + "\n"
        + "\n".join(rows) + "\n"
        r"\end{longtable}" + "\n"
        r"\end{landscape}" + "\n"
    )
