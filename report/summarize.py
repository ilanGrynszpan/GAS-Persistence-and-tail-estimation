"""PDF and LaTeX reporting for model comparison outputs."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

MODEL_LABELS = {
    "phi": r"$\phi$-only",
    "phi_xi": r"$\phi,\xi$",
}

_METRIC_LOWER_IS_BETTER = {"crps", "rmse", "mad", "aic", "bic"}
_METRIC_HIGHER_IS_BETTER = {"loglik"}

# Colour palette
_C_HEADER_BG = "#1E3A6E"
_C_HEADER_FG = "white"
_C_ROW_ODD = "#EEF2FF"
_C_ROW_EVEN = "#FFFFFF"
_C_BEST = "#BBF7D0"
_C_REJECT = "#FED7D7"
_C_PAGE = "#9CA3AF"
_C_TITLE = "#111827"
_C_BODY = "#374151"
_C_NOTE = "#4B5563"

# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

def display_model_label(model_id: str) -> str:
    parts = str(model_id).split("_")
    seasonal = parts[0].capitalize() if parts else str(model_id)
    if str(model_id).endswith("_phi_xi"):
        spec = MODEL_LABELS["phi_xi"]
    elif str(model_id).endswith("_phi"):
        spec = MODEL_LABELS["phi"]
    else:
        spec = str(model_id)
    return f"{seasonal} {spec}"


def _get_seasonal(model_id: str) -> str:
    return str(model_id).split("_")[0].lower()


def _series_id_from_model_id(model_id: str) -> str:
    if str(model_id).endswith("_phi_xi"):
        return str(model_id)[:-7]
    if str(model_id).endswith("_phi"):
        return str(model_id)[:-4]
    return str(model_id)


def _series_label(series_id: str) -> str:
    parts = str(series_id).split("_")
    seasonal = parts[0].capitalize() if parts else str(series_id)
    location = " ".join(p.capitalize() for p in parts[1:]) if len(parts) > 1 else ""
    return f"{seasonal} {location}".strip()


def _with_labels(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or "model_id" not in frame.columns:
        return frame
    out = frame.copy()
    out.insert(0, "Model", out["model_id"].map(display_model_label))
    return out.drop(columns=["model_id"])


def _fmt(x, decimals: int = 3) -> str:
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.{decimals}f}"
    except (TypeError, ValueError):
        return str(x)


def _wrap(text: str, width: int = 130) -> list[str]:
    return textwrap.wrap(str(text), width=width) or [""]


# ─────────────────────────────────────────────────────────────────────────────
# Page-saver (auto-numbers every page)
# ─────────────────────────────────────────────────────────────────────────────

def _make_saver(pdf: PdfPages):
    counter = [0]

    def save(fig):
        counter[0] += 1
        fig.text(
            0.5, 0.014,
            f"— {counter[0]} —",
            ha="center", fontsize=9, color=_C_PAGE,
            transform=fig.transFigure,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return save


# ─────────────────────────────────────────────────────────────────────────────
# Text pages
# ─────────────────────────────────────────────────────────────────────────────

def _text_page(save, title: str, sections: list[tuple[str, list[str]]]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.text(0.055, 0.935, title, fontsize=18, weight="bold", va="top", color=_C_TITLE)
    y = 0.875

    def new_page():
        nonlocal fig, y
        save(fig)
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("white")
        y = 0.93

    for header, lines in sections:
        if y < 0.15:
            new_page()
        fig.text(0.055, y, header, fontsize=11.5, weight="bold", va="top", color=_C_TITLE)
        y -= 0.038
        for line in lines:
            if not str(line).strip():
                y -= 0.014
                continue
            for part in _wrap(line, 132):
                if y < 0.15:
                    new_page()
                fig.text(0.065, y, part, fontsize=9.5, va="top", color=_C_BODY)
                y -= 0.026
            y -= 0.006
        y -= 0.018
    save(fig)


def _cover_page(save, title: str, observation: str) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.810, title, ha="center", fontsize=22, weight="bold", color=_C_TITLE)
    fig.text(
        0.5, 0.757,
        "ZA-GAS Model Comparison  —  Thesis Working Report",
        ha="center", fontsize=13, color="#4B5563",
    )
    fig.add_artist(
        mlines.Line2D(
            [0.08, 0.92], [0.72, 0.72],
            transform=fig.transFigure, color="#D1D5DB", linewidth=1.5,
        )
    )

    abstract = (
        "This report compares two specifications of the Zero-Augmented Generalised Autoregressive "
        "Score (ZA-GAS) model for precipitation time series. In the phi-only specification only the "
        "GB2 location parameter φ is time-varying; in the phi-xi specification both φ and "
        "the shape parameter ξ follow the GAS score-driven recursion. Models are evaluated at "
        "daily and monthly cadences on in-sample fit (log-likelihood, AIC, BIC) and out-of-sample "
        "forecast accuracy (CRPS, RMSE, MAD). Probabilistic calibration is assessed via PIT "
        "uniformity tests (Kolmogorov-Smirnov) and 95 % coverage tests (Kupiec, "
        "Christoffersen)."
    )

    fig.text(0.055, 0.700, "Abstract", fontsize=11.5, weight="bold", color=_C_TITLE)
    y = 0.668
    for part in _wrap(abstract, 128):
        fig.text(0.065, y, part, fontsize=9.5, va="top", color=_C_BODY)
        y -= 0.027
    y -= 0.018

    # Scope note box
    box_top = y - 0.004
    box_h = 0.082
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (0.055, box_top - box_h), 0.89, box_h,
            transform=fig.transFigure,
            facecolor="#FEF9C3", edgecolor="#CA8A04", linewidth=1.5,
            boxstyle="round,pad=0.008",
        )
    )
    fig.text(
        0.068, box_top - 0.012,
        "⚠️  Scope Note",
        fontsize=10.5, weight="bold", va="top", color="#92400E",
    )
    oy = box_top - 0.038
    for part in _wrap(observation, 124):
        fig.text(0.076, oy, part, fontsize=9.5, va="top", color="#78350F")
        oy -= 0.026

    save(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Model description
# ─────────────────────────────────────────────────────────────────────────────

def _model_description_page(save, results: list[dict]) -> None:
    seen: dict[tuple, dict] = {}
    for res in results:
        m = res.get("model")
        if m is None:
            continue
        key = (m.seasonal, tuple(m.gas.tv_names))
        if key not in seen:
            seen[key] = m.parameter_count_breakdown()

    fw = [
        "The observation equation is zero-augmented. Conditional on information F_{t-1},",
        r"  $p(y_t \mid \mathcal{F}_{t-1}) = (1-\pi_t)\mathbf{1}\{y_t=0\} + \pi_t g(y_t; f_{t|t-1}, \theta)\mathbf{1}\{y_t>0\}$.",
        r"The positive component $g(\cdot)$ is GB2 with log-link parameters. The state vector $f_{t|t-1}$ contains the time-varying positive-part parameters: either $f_{t|t-1}=(\phi_{t|t-1})$ or $f_{t|t-1}=(\phi_{t|t-1},\xi_{t|t-1})^\prime$.",
        "",
        "The GAS recursion follows the one-step-ahead prediction notation used in score-driven models:",
        r"  $f_{t+1|t} = \omega + \sum_{\ell \in L} A_{\ell}s_{t-\ell+1} + \sum_{\ell \in L} B_{\ell}f_{t-\ell+1|t-\ell}$.",
        r"Here $s_t = \mathcal{I}_{t|t-1}^{-1}\nabla_t$ is the Fisher-information-scaled score, $\nabla_t = \partial \log p(y_t|\mathcal{F}_{t-1})/\partial f_{t|t-1}$. The score is set to zero for exact-zero observations because those observations identify the zero probability rather than the positive GB2 density.",
        r"For the non-seasonal GAS(1,1) special case this reduces to $f_{t+1|t}=\omega + A s_t + B f_{t|t-1}$.",
    ]

    pi_lines = [
        r"The non-zero probability is $\pi_t=\Lambda(\eta_t)$, where $\Lambda(x)=1/(1+\exp(-x))$ is the logistic link.",
        r"The implemented AR-logistic recursion is $\eta_t=\omega_0+\rho\eta_{t-1}+\sum_{\ell\in L}\omega_{y,\ell}y_{t-\ell}$.",
        r"The same seasonal lag set $L$ is used for the GAS recursion and the $\pi_t$ dynamics, so the zero/non-zero occurrence process and the positive-part intensity process can both react to short-memory and calendar-seasonal precipitation information.",
    ]

    lags = [
        "  Daily cadence:   L = {1, 2, 3, 364, 365, 366, 367}   - seven lags covering +/-1 day around the annual cycle",
        "  Monthly cadence: L = {1, 2, 3, 11, 12, 13}           - six lags covering +/-1 month around the annual cycle",
    ]

    phi_lines = [
        r"Only $\phi$ is time-varying; $\xi$ is kept at its estimated constant value throughout the sample. This is the parsimonious baseline: it captures location shifts in positive-part precipitation while holding the spread/shape channel fixed.",
    ]

    phi_xi_lines = [
        r"Both $\phi$ and $\xi$ follow the GAS update. This allows both the positive-part scale/location and the GB2 shape channel to react dynamically to new observations, potentially improving calibration in regimes where rainfall intensity and dispersion change together.",
    ]

    param_lines = []
    for (seasonal, tv_names), bd in seen.items():
        param_lines.append(
            f"  {seasonal.capitalize()} {list(tv_names)}: "
            f"{bd['total_parameters']} estimated parameters "
            f"({bd['gas_dynamic_parameters']} GAS dynamics, "
            f"{bd['static_positive_parameters']} static positive-part, "
            f"{bd['pi_parameters']} pi-dynamics)"
        )

    est = [
        "Parameters are estimated by maximising the full log-likelihood using L-BFGS-B with box constraints. Standard errors are Wald-type from the optimizer's inverse Hessian approximation. Confidence intervals are at the 95% level.",
    ]

    _text_page(save, "Model Descriptions", [
        ("Framework: Zero-Augmented GAS", fw),
        ("Pi Dynamics", pi_lines),
        ("Seasonal Lag Sets", lags),
        (r"Model 1 - $\phi$-only", phi_lines),
        (r"Model 2 - $\phi,\xi$", phi_xi_lines),
        ("Parameter Counts", param_lines),
        ("Estimation", est),
    ])


def _styled_table(
    save,
    title: str,
    frame: pd.DataFrame,
    note: str = "",
    best_indices: dict | None = None,
    reject_cols: set | None = None,
    decimals: int = 3,
    max_rows: int = 30,
    font_size: float = 8.0,
) -> None:
    if frame is None or frame.empty:
        return

    shown = frame.head(max_rows).copy()
    fmt = shown.copy()
    for col in fmt.columns:
        if pd.api.types.is_numeric_dtype(fmt[col]):
            fmt[col] = fmt[col].map(lambda x, d=decimals: _fmt(x, d))
        else:
            fmt[col] = fmt[col].astype(str)

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=14, color=_C_TITLE)

    tbl = ax.table(
        cellText=fmt.values,
        colLabels=list(fmt.columns),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1.0, 1.38)

    col_names = list(fmt.columns)
    n_data = len(shown)

    for (row, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        if row == 0:
            cell.set_facecolor(_C_HEADER_BG)
            cell.set_text_props(weight="bold", color=_C_HEADER_FG)
            continue
        dr = row - 1
        if dr >= n_data:
            continue
        col_name = col_names[col_idx] if col_idx < len(col_names) else None
        orig_val = shown.iloc[dr][col_name] if col_name and col_name in shown.columns else None
        data_idx = shown.index[dr]

        is_best = (
            best_indices is not None
            and col_name in (best_indices or {})
            and best_indices[col_name] == data_idx
        )
        is_rej = (
            reject_cols is not None
            and col_name in reject_cols
            and bool(orig_val) is True
        )

        if is_best:
            cell.set_facecolor(_C_BEST)
        elif is_rej:
            cell.set_facecolor(_C_REJECT)
        elif dr % 2 == 0:
            cell.set_facecolor(_C_ROW_ODD)
        else:
            cell.set_facecolor(_C_ROW_EVEN)

    bottom = 0.10 if note else 0.04
    if note:
        fig.text(
            0.055, 0.055,
            f"Note: {note}",
            fontsize=8.5, color=_C_NOTE, ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F9FAFB", edgecolor="#E5E7EB"),
        )
    if len(frame) > max_rows:
        fig.text(
            0.5, 0.026,
            f"Showing first {max_rows} of {len(frame)} rows; full table in the LaTeX sidecar.",
            fontsize=8, color="#6B7280", ha="center",
        )

    plt.tight_layout(rect=[0, bottom, 1, 1])
    save(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model diagnostic charts
# ─────────────────────────────────────────────────────────────────────────────

def _pit_chart(save, model_id: str, pit_values: np.ndarray) -> None:
    pit = np.asarray(pit_values, dtype=float)
    pit = pit[np.isfinite(pit)]
    if len(pit) == 0:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    fig.patch.set_facecolor("white")
    ax.hist(
        pit, bins=20, density=True,
        color="#3B82F6", alpha=0.72, edgecolor="white", linewidth=0.5,
    )
    ax.axhline(1.0, color="#EF4444", linestyle="--", linewidth=1.8, label="Uniform(0,1)")
    ax.set_xlim(0, 1)
    ax.set_xlabel("PIT value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f"In-Sample PIT Histogram  —  {display_model_label(model_id)}",
        fontsize=12, weight="bold", color=_C_TITLE,
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5, 0.036,
        "Bars close to height 1 (the red dashed line) indicate good probabilistic calibration.",
        ha="center", fontsize=9, color=_C_NOTE,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    save(fig)


def _qq_chart(save, model_id: str, qr_values: np.ndarray) -> None:
    qr = np.asarray(qr_values, dtype=float)
    qr = qr[np.isfinite(qr)]
    if len(qr) < 10:
        return

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    fig.patch.set_facecolor("white")
    osm, osr = stats.probplot(qr, dist="norm", fit=False)
    ax.scatter(osm, osr, s=14, alpha=0.70, color="#2563EB", edgecolor="none")
    lo = float(min(np.min(osm), np.min(osr)))
    hi = float(max(np.max(osm), np.max(osr)))
    ax.plot([lo, hi], [lo, hi], color="#EF4444", linewidth=1.6, linestyle="--")
    ax.set_title(
        f"In-Sample Quantile Residual QQ Plot - {display_model_label(model_id)}",
        fontsize=12, weight="bold", color=_C_TITLE,
    )
    ax.set_xlabel("Theoretical N(0,1) quantiles", fontsize=10)
    ax.set_ylabel("Empirical quantile residuals", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.22)
    fig.text(
        0.5, 0.035,
        "Points close to the dashed 45-degree line indicate approximately standard-normal quantile residuals.",
        ha="center", fontsize=8.5, color=_C_NOTE,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    save(fig)


def _acf_chart(
    save,
    model_id: str,
    qr_values: np.ndarray,
    acf_frame: pd.DataFrame,
    seasonal: str,
) -> None:
    qr = np.asarray(qr_values, dtype=float)
    qr = qr[np.isfinite(qr)]
    if len(qr) < 10:
        return

    max_lag = 367 if seasonal == "daily" else 13
    max_lag = min(max_lag, len(qr) - 1)
    if max_lag < 1:
        return

    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    fig.patch.set_facecolor("white")
    plot_acf(qr, lags=max_lag, alpha=0.05, zero=False, ax=ax)
    ax.set_title(
        f"In-Sample Quantile Residual ACF - {display_model_label(model_id)}",
        fontsize=12, weight="bold", color=_C_TITLE,
    )
    ax.set_xlabel("Lag", fontsize=10)
    ax.set_ylabel("Autocorrelation", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    fig.text(
        0.5, 0.034,
        f"One model only. Lags 1-{max_lag}; shaded bands are approximate 95% confidence intervals under white-noise residuals.",
        ha="center", fontsize=8.5, color=_C_NOTE,
    )
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    save(fig)


def _series_diagnostics(
    save,
    series_summary: pd.DataFrame,
    series_values: pd.DataFrame,
) -> None:
    if series_summary.empty and series_values.empty:
        return

    _text_page(save, "Series Diagnostics", [
        ("Section Overview", [
            "This section describes the observed precipitation series before any model-specific diagnostics are shown. "
            "The statistics and ACF charts are about the data series itself, not about a fitted model.",
        ]),
    ])

    if not series_summary.empty:
        summary = series_summary.copy()
        summary["series_id"] = summary["series_id"].fillna(summary.get("model_id", ""))
        summary = summary.drop_duplicates(subset=["series_id", "sample"])
        summary.insert(0, "Series", summary["series_id"].map(_series_label))
        keep = [
            "Series", "sample", "n", "min", "max", "mean", "std", "prop_zero",
            "q01", "q05", "q25", "q50", "q75", "q90", "q95", "q99",
        ]
        summary = summary[[c for c in keep if c in summary.columns]].reset_index(drop=True)
        _styled_table(
            save,
            "Observed Series Descriptive Statistics",
            summary,
            note="Statistics are computed for the raw precipitation series. Full = IS and OOS concatenated.",
            decimals=3,
            max_rows=40,
            font_size=7.2,
        )

    if series_values.empty:
        return

    vals = series_values.copy()
    vals["series_id"] = vals["series_id"].fillna(vals.get("model_id", "").map(_series_id_from_model_id))
    for series_id in vals["series_id"].dropna().drop_duplicates():
        sub = vals[vals["series_id"] == series_id].copy()
        order = {"IS": 0, "OOS": 1, "Full": 2}
        sub["_sample_order"] = sub["sample"].map(order).fillna(9)
        sub = sub.sort_values(["_sample_order", "t_index"])
        y = sub["y"].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if len(y) < 10:
            continue
        seasonal = str(series_id).split("_")[0].lower()
        max_lag = 367 if seasonal == "daily" else 13
        max_lag = min(max_lag, len(y) - 1)
        if max_lag < 1:
            continue
        fig, ax = plt.subplots(figsize=(12.4, 5.4))
        fig.patch.set_facecolor("white")
        plot_acf(y, lags=max_lag, alpha=0.05, zero=False, ax=ax)
        ax.set_title(
            f"Observed Series ACF - {_series_label(series_id)}",
            fontsize=12, weight="bold", color=_C_TITLE,
        )
        ax.set_xlabel("Lag", fontsize=10)
        ax.set_ylabel("Autocorrelation", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.22)
        fig.text(
            0.5, 0.034,
            f"ACF of observed precipitation, lags 1-{max_lag}; shaded bands are approximate 95% confidence intervals.",
            ha="center", fontsize=8.5, color=_C_NOTE,
        )
        plt.tight_layout(rect=[0, 0.075, 1, 1])
        save(fig)


def _per_model_diagnostics(
    save,
    results: list[dict],
    series: pd.DataFrame,
    pit_df: pd.DataFrame,
    params_df: pd.DataFrame,
    acf_df: pd.DataFrame,
    jb_df: pd.DataFrame,
) -> None:
    is_series = (
        series[series["sample"] == "IS"].copy()
        if not series.empty
        else pd.DataFrame()
    )

    for res in results:
        model_id = res["model_id"]
        m = res.get("model")
        seasonal = _get_seasonal(model_id)
        label = display_model_label(model_id)

        _text_page(save, f"Diagnostics: {label}", [
            ("Scope", [
                f"This section presents in-sample diagnostic charts and tables for the {label} model. "
                "These are model-specific checks of probabilistic calibration (PIT) and residual "
                "autocorrelation (ACF). Cross-model comparisons are in the Comparison Tables section.",
            ]),
        ])

        # PIT histogram (IS only)
        if not is_series.empty and "pit" in is_series.columns:
            pit_vals = (
                is_series.loc[is_series["model_id"] == model_id, "pit"]
                .dropna()
                .to_numpy()
            )
            if len(pit_vals) > 0:
                _pit_chart(save, model_id, pit_vals)

        # PIT KS statistics table
        if not pit_df.empty:
            pit_sub = (
                pit_df[pit_df["model_id"] == model_id][
                    ["sample", "n", "ks_stat", "pvalue", "reject_5pct"]
                ].copy()
            )
            if not pit_sub.empty:
                _styled_table(
                    save,
                    f"PIT Kolmogorov-Smirnov Test  —  {label}",
                    pit_sub,
                    note=(
                        "KS test of PIT values against Uniform(0, 1). "
                        "A well-calibrated model has p-value > 0.05 and ks_stat close to zero. "
                        "IS = in-sample; OOS = out-of-sample; Full = concatenated sample."
                    ),
                )

        # Quantile residual ACF (IS only)
        if not is_series.empty and "quantile_residual" in is_series.columns:
            qr_vals = (
                is_series.loc[is_series["model_id"] == model_id, "quantile_residual"]
                .dropna()
                .to_numpy()
            )
            if len(qr_vals) > 0 and m is not None:
                _qq_chart(save, model_id, qr_vals)
                _acf_chart(save, model_id, qr_vals, acf_df, seasonal)

        # Jarque-Bera test table for in-sample quantile residuals
        if not jb_df.empty:
            jb_sub = jb_df[
                (jb_df["model_id"] == model_id) & (jb_df["sample"] == "IS")
            ].copy()
            if not jb_sub.empty:
                keep = ["sample", "n", "skewness", "kurtosis", "JB_stat", "pvalue", "reject_5pct"]
                _styled_table(
                    save,
                    f"Jarque-Bera Test for Quantile Residuals  -  {label}",
                    jb_sub[[c for c in keep if c in jb_sub.columns]].reset_index(drop=True),
                    reject_cols={"reject_5pct"},
                    note=(
                        "Jarque-Bera tests whether in-sample quantile residuals are compatible "
                        "with a Gaussian distribution. Red cells indicate rejection at 5%."
                    ),
                )

        # Parameter estimates table
        if not params_df.empty:
            param_sub = params_df[params_df["model_id"] == model_id].copy()
            if not param_sub.empty:
                keep = ["parameter", "block", "estimate", "std_error"]
                ci_lo = [c for c in param_sub.columns if c.startswith("ci_lower_")]
                ci_hi = [c for c in param_sub.columns if c.startswith("ci_upper_")]
                if ci_lo and ci_hi:
                    keep += [ci_lo[0], ci_hi[0]]
                _styled_table(
                    save,
                    f"Estimated Parameters  —  {label}",
                    param_sub[keep],
                    decimals=4,
                    max_rows=50,
                    note=(
                        "Estimates from the optimizer used for the cached model. Standard errors and 95 % "
                        "Wald CIs from the inverse Hessian; interpret with caution near optimizer bounds."
                    ),
                )


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison tables
# ─────────────────────────────────────────────────────────────────────────────

def _best_indices(frame: pd.DataFrame) -> dict[str, object]:
    best: dict[str, object] = {}
    for col in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        cname = col.lower()
        vals = pd.to_numeric(frame[col], errors="coerce")
        if vals.isna().all():
            continue
        if cname in _METRIC_LOWER_IS_BETTER:
            best[col] = vals.idxmin()
        elif cname in _METRIC_HIGHER_IS_BETTER:
            best[col] = vals.idxmax()
    return best


def _comparison_metrics(save, metrics: pd.DataFrame, seasonal: str) -> None:
    sub = metrics[metrics["model_id"].str.startswith(seasonal)].copy()
    if sub.empty:
        return

    for sample in ["IS", "OOS"]:
        ss = sub[sub["sample"] == sample].copy()
        if ss.empty:
            continue

        if sample == "OOS":
            cols = ["model_id", "rmse", "mad", "crps"]
        else:
            if sample == "OOS":
                cols = ["model_id", "rmse", "mad", "crps"]
            else:
                cols = ["model_id", "loglik", "aic", "bic", "rmse", "mad", "crps"]
        frame = ss[[c for c in cols if c in ss.columns]].copy()
        frame.insert(0, "Model", frame["model_id"].map(display_model_label))
        frame = frame.drop(columns=["model_id"])
        frame = frame.reset_index(drop=True)

        _styled_table(
            save,
            f"Performance Metrics  —  {seasonal.capitalize()} Models ({sample})",
            frame,
            best_indices=_best_indices(frame),
            note=(
                "Best value per metric highlighted in green. "
                + (
                    "Lower is better for CRPS, RMSE, and MAD."
                    if sample == "OOS"
                    else "Lower is better for CRPS, RMSE, MAD, AIC, BIC; higher is better for log-likelihood."
                )
            ),
        )


def _comparison_coverage(save, coverage: pd.DataFrame, seasonal: str) -> None:
    sub = coverage[coverage["model_id"].str.startswith(seasonal)].copy()
    if sub.empty:
        return

    for sample in ["IS", "OOS", "Full"]:
        ss = sub[sub["sample"] == sample].copy()
        if ss.empty:
            continue

        cols = [
            "model_id", "violations", "expected_violations", "violation_rate",
            "LR_uc", "pvalue_uc", "LR_cc", "pvalue_cc",
            "reject_uc_5pct", "reject_cc_5pct",
        ]
        frame = ss[[c for c in cols if c in ss.columns]].copy()
        frame.insert(0, "Model", frame["model_id"].map(display_model_label))
        frame = frame.drop(columns=["model_id"])
        frame = frame.reset_index(drop=True)

        _styled_table(
            save,
            f"95 % Coverage Tests  —  {seasonal.capitalize()} ({sample})",
            frame,
            reject_cols={"reject_uc_5pct", "reject_cc_5pct"},
            note=(
                "Kupiec LR_uc tests unconditional 95 % coverage; "
                "Christoffersen LR_cc adds independence of violations. "
                "Red cells indicate rejection at the 5 % significance level."
            ),
            font_size=7.5,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Conclusion
# ─────────────────────────────────────────────────────────────────────────────

def _conclusion_page(
    save,
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    pit: pd.DataFrame,
) -> None:
    summary = [
        "This report presents the first systematic comparison of two ZA-GAS specifications on the "
        "Belo Horizonte precipitation time series at daily and monthly cadences.",
    ]
    if not metrics.empty:
        for seasonal in ["daily", "monthly"]:
            oos = metrics[
                metrics["model_id"].str.startswith(seasonal)
                & (metrics["sample"] == "OOS")
            ]
            if oos.empty:
                continue
            for metric in ["crps", "rmse"]:
                if metric not in oos.columns:
                    continue
                vals = pd.to_numeric(oos[metric], errors="coerce")
                if vals.isna().all():
                    continue
                best = oos.loc[vals.idxmin()]
                summary.append(
                    f"  {seasonal.capitalize()} OOS — lowest {metric.upper()}: "
                    f"{display_model_label(best['model_id'])} ({float(best[metric]):.4f})."
                )

    cov_lines = []
    if not coverage.empty:
        full_cov = coverage[coverage["sample"] == "Full"]
        rejected = full_cov[
            full_cov["reject_uc_5pct"].astype(bool)
            | full_cov["reject_cc_5pct"].astype(bool)
        ]
        if rejected.empty:
            cov_lines.append(
                "Full-sample 95 % coverage tests do not reject for any model at the "
                "5 % significance level."
            )
        else:
            names = ", ".join(
                display_model_label(x) for x in rejected["model_id"].unique()
            )
            cov_lines.append(
                f"Full-sample 95 % coverage is rejected for: {names}."
            )

    pit_lines = []
    if not pit.empty:
        full_pit = pit[pit["sample"] == "Full"]
        rej = full_pit[pd.to_numeric(full_pit["pvalue"], errors="coerce") < 0.05]
        if rej.empty:
            pit_lines.append(
                "PIT uniformity is not rejected for any model at the 5 % level, "
                "suggesting adequate probabilistic calibration."
            )
        else:
            names = ", ".join(
                display_model_label(x) for x in rej["model_id"].unique()
            )
            pit_lines.append(
                f"PIT uniformity rejected at 5 % for: {names}."
            )

    next_steps = [
        "Extend the analysis to all Brazilian weather stations to assess spatial consistency.",
        "Compare ZA-GAS against benchmark models (climatological quantiles, seasonal ARIMA) "
        "on the same evaluation metrics.",
        "Investigate whether ξ dynamics consistently improve calibration in high-rainfall regimes.",
        "Explore alternative lag sets or seasonal parameterisations.",
    ]

    refs = [
        "Creal, D., Koopman, S. J., & Lucas, A. (2012). Generalized autoregressive score models "
        "with applications. Journal of Applied Econometrics, 28(5), 777–795.",
        "Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. "
        "Journal of Derivatives, 3, 73–84.",
        "Christoffersen, P. F. (1998). Evaluating interval forecasts. "
        "International Economic Review, 39(4), 841–862.",
    ]

    _text_page(save, "Conclusion", [
        ("Summary of Findings", summary),
        ("Coverage Diagnostics", cov_lines or ["No coverage data available."]),
        ("Calibration (PIT)", pit_lines or ["No PIT data available."]),
        ("Next Steps", next_steps),
        ("References", refs),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX generation
# ─────────────────────────────────────────────────────────────────────────────

def _latex_table(
    frame: pd.DataFrame,
    caption: str,
    label: str,
    best_idx: dict | None = None,
    reject_cols: set | None = None,
    decimals: int = 3,
) -> str:
    n_cols = len(frame.columns)
    col_spec = "l" + "r" * (n_cols - 1)
    rows = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(str(c).replace("_", r"\_") for c in frame.columns) + r" \\",
        r"\midrule",
    ]
    for i, (idx, row) in enumerate(frame.iterrows()):
        cells = []
        for col in frame.columns:
            v = row[col]
            s = _fmt(v, decimals) if pd.api.types.is_numeric_dtype(frame[col]) else str(v)
            s = s.replace("_", r"\_")
            if best_idx and col in best_idx and best_idx[col] == idx:
                s = r"\cellcolor{green!20}" + s
            elif reject_cols and col in reject_cols and bool(v):
                s = r"\cellcolor{red!15}" + s
            cells.append(s)
        rows.append(" & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def _generate_latex(
    tex_path: Path,
    title: str,
    results: list[dict],
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    pit: pd.DataFrame,
    params: pd.DataFrame,
) -> None:
    safe_title = title.replace("$", r"\$")

    parts = [
        r"\documentclass[a4paper,11pt]{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{geometry}",
        r"\usepackage{pdflscape}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{colortbl}",
        r"\usepackage{caption}",
        r"\usepackage{amsmath}",
        r"\usepackage{hyperref}",
        r"\geometry{margin=2cm}",
        r"\setlength{\parskip}{6pt}",
        r"\begin{document}",
        rf"\title{{{safe_title}}}",
        r"\date{}",
        r"\maketitle",
        r"\begin{abstract}",
        "This document compares two ZA-GAS specifications (phi-only and phi-xi) for daily and "
        "monthly precipitation at the Belo Horizonte station. Models are evaluated on in-sample "
        "fit and out-of-sample forecast accuracy. "
        r"\textbf{Scope note:} This is the first experiment; results for other stations are "
        "deferred to subsequent work.",
        r"\end{abstract}",
        r"\tableofcontents",
        r"\newpage",
        r"\section{Model Description}",
        r"\subsection{Framework}",
        r"The ZA-GAS model:",
        r"\begin{equation}",
        r"p(y_t \mid \mathcal{F}_{t-1}) = (1-\pi_t)\,\mathbf{1}[y_t=0]"
        r" + \pi_t\,g(y_t;\phi_t,\xi_t,\gamma,\zeta)\,\mathbf{1}[y_t>0]",
        r"\end{equation}",
        r"GAS update: $f_{t+1} = \omega + \sum_{l \in L} A_l\,s_{t-l+1}"
        r" + \sum_{l \in L} B_l\,f_{t-l+1}$, where $s_t = \mathcal{I}(f_t)^{-1}\nabla_t$.",
        r"\subsection{Variants}",
        r"\textbf{phi-only:} Only $\phi$ is time-varying; $\xi$ is static.",
        "",
        r"\textbf{phi-xi:} Both $\phi$ and $\xi$ follow the GAS update.",
        r"\section{Per-Model Diagnostics}",
    ]

    for res in results:
        model_id = res["model_id"]
        label = display_model_label(model_id)
        safe_label = label.replace("$", "").replace(",", "").replace(" ", "-")
        parts.append(rf"\subsection{{{label}}}")

        if not pit.empty:
            pit_sub = pit[pit["model_id"] == model_id][
                ["sample", "n", "ks_stat", "pvalue", "reject_5pct"]
            ].copy()
            if not pit_sub.empty:
                parts.append(r"\paragraph{PIT KS test}")
                parts.append(
                    _latex_table(
                        pit_sub,
                        caption=f"PIT KS test --- {label}",
                        label=f"tab:pit-{model_id}",
                    )
                )
        if not params.empty:
            param_sub = params[params["model_id"] == model_id].copy()
            keep = ["parameter", "block", "estimate", "std_error"]
            ci_lo = [c for c in param_sub.columns if c.startswith("ci_lower_")]
            ci_hi = [c for c in param_sub.columns if c.startswith("ci_upper_")]
            if ci_lo and ci_hi:
                keep += [ci_lo[0], ci_hi[0]]
            if not param_sub.empty:
                parts.append(r"\paragraph{Parameter estimates}")
                parts.append(r"\begin{landscape}")
                parts.append(
                    _latex_table(
                        param_sub[keep].reset_index(drop=True),
                        decimals=4,
                        caption=f"Estimated parameters --- {label}",
                        label=f"tab:params-{model_id}",
                    )
                )
                parts.append(r"\end{landscape}")

    parts.append(r"\section{Model Comparison}")
    for seasonal in ["daily", "monthly"]:
        sub = (
            metrics[metrics["model_id"].str.startswith(seasonal)].copy()
            if not metrics.empty
            else pd.DataFrame()
        )
        if sub.empty:
            continue
        parts.append(rf"\subsection{{{seasonal.capitalize()} Models}}")
        for sample in ["IS", "OOS"]:
            ss = sub[sub["sample"] == sample].copy()
            if ss.empty:
                continue
            cols = ["model_id", "loglik", "aic", "bic", "rmse", "mad", "crps"]
            frame = ss[[c for c in cols if c in ss.columns]].copy()
            frame.insert(0, "Model", frame["model_id"].map(display_model_label))
            frame = frame.drop(columns=["model_id"]).reset_index(drop=True)
            parts.append(r"\begin{landscape}")
            parts.append(
                _latex_table(
                    frame,
                    best_idx=_best_indices(frame),
                    caption=f"Metrics --- {seasonal.capitalize()} {sample}",
                    label=f"tab:metrics-{seasonal}-{sample.lower()}",
                )
            )
            parts.append(r"\end{landscape}")

        cov_sub = (
            coverage[coverage["model_id"].str.startswith(seasonal)].copy()
            if not coverage.empty
            else pd.DataFrame()
        )
        full_cov = cov_sub[cov_sub["sample"] == "Full"] if not cov_sub.empty else pd.DataFrame()
        if not full_cov.empty:
            cov_cols = [
                "model_id", "violations", "expected_violations", "violation_rate",
                "LR_uc", "pvalue_uc", "reject_uc_5pct",
                "LR_cc", "pvalue_cc", "reject_cc_5pct",
            ]
            frame = full_cov[[c for c in cov_cols if c in full_cov.columns]].copy()
            frame.insert(0, "Model", frame["model_id"].map(display_model_label))
            frame = frame.drop(columns=["model_id"]).reset_index(drop=True)
            parts.append(r"\begin{landscape}")
            parts.append(
                _latex_table(
                    frame,
                    reject_cols={"reject_uc_5pct", "reject_cc_5pct"},
                    caption=f"95\\% Coverage tests --- {seasonal.capitalize()} Full",
                    label=f"tab:coverage-{seasonal}",
                )
            )
            parts.append(r"\end{landscape}")

    parts += [
        r"\section{Conclusion}",
        "This preliminary comparison on the Belo Horizonte time series provides initial evidence "
        "on the relative merits of phi-only vs phi-xi ZA-GAS specifications. "
        "The analysis will be extended to the full set of stations in future work.",
        r"\subsection*{References}",
        r"\begin{itemize}",
        r"\item Creal, D., Koopman, S.\ J., \& Lucas, A. (2012). Generalized autoregressive "
        r"score models with applications. \textit{Journal of Applied Econometrics}, 28(5), 777--795.",
        r"\item Kupiec, P.\ H. (1995). Techniques for verifying the accuracy of risk measurement "
        r"models. \textit{Journal of Derivatives}, 3, 73--84.",
        r"\item Christoffersen, P.\ F. (1998). Evaluating interval forecasts. "
        r"\textit{International Economic Review}, 39(4), 841--862.",
        r"\end{itemize}",
        r"\end{document}",
    ]

    tex_path.write_text("\n".join(parts), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point (keep signature identical to old version)
# ─────────────────────────────────────────────────────────────────────────────

def render_model_report(
    results: list[dict],
    pdf_path: str | Path,
    tex_path: str | Path | None = None,
    title: str = "ZA-GAS Model Comparison",
) -> dict:
    """Render a structured PDF report and LaTeX sidecar from model results."""
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if tex_path is None:
        tex_path = pdf_path.with_suffix(".tex")
    tex_path = Path(tex_path)

    def _concat(key: str) -> pd.DataFrame:
        frames = [r[key] for r in results if key in r and r[key] is not None]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    metrics = _concat("metrics_frame")
    params = _concat("parameters")
    acf = _concat("acf")
    pit = _concat("pit")
    coverage = _concat("coverage")
    series = _concat("diagnostic_series")
    jb = _concat("jarque_bera")
    series_summary = _concat("series_summary")
    series_values = _concat("series_values")

    observation = (
        "This is the first experiment in the thesis research. Models were fitted and evaluated "
        "only for the Belo Horizonte (BH) time series. Results for other Brazilian weather "
        "stations will be presented in subsequent experiments once the full spatial sweep is complete."
    )

    with PdfPages(pdf_path) as pdf:
        save = _make_saver(pdf)

        # ── Cover ────────────────────────────────────────────────────────────
        _cover_page(save, title, observation)

        # ── Model descriptions ───────────────────────────────────────────────
        _model_description_page(save, results)
        _series_diagnostics(save, series_summary, series_values)

        # ── Per-model diagnostics ────────────────────────────────────────────
        _text_page(save, "Per-Model Diagnostics", [
            ("Section Overview", [
                "The following pages present individual diagnostic charts and tables for each "
                "fitted model: PIT histogram (calibration), quantile residual ACF (serial "
                "independence), KS test statistics, and estimated parameters. "
                "All diagnostic charts use only in-sample observations. "
                "Cross-model comparisons are deferred to the Comparison Tables section.",
            ]),
        ])
        _per_model_diagnostics(save, results, series, pit, params, acf, jb)

        # ── Comparison tables ────────────────────────────────────────────────
        _text_page(save, "Model Comparison Tables", [
            ("Section Overview", [
                "This section compares all models within each cadence (daily, monthly) side by side. "
                "Separate tables are provided for IS and OOS samples. "
                "The best value per metric column is highlighted in green. "
                "Red cells in coverage tables indicate rejection at the 5 % level.",
            ]),
        ])
        _text_page(save, "Daily Models", [
            ("", ["Performance metrics and coverage tests for the daily cadence models."]),
        ])
        _comparison_metrics(save, metrics, "daily")
        _comparison_coverage(save, coverage, "daily")

        _text_page(save, "Monthly Models", [
            ("", ["Performance metrics and coverage tests for the monthly cadence models."]),
        ])
        _comparison_metrics(save, metrics, "monthly")
        _comparison_coverage(save, coverage, "monthly")

        # ── Conclusion ───────────────────────────────────────────────────────
        _conclusion_page(save, metrics, coverage, pit)

    _generate_latex(tex_path, title, results, metrics, coverage, pit, params)

    return {"pdf": pdf_path, "tex": tex_path}
