"""
Pi-dynamics alternatives report generator (2026-07-14).

=============================================================================
OVERVIEW
=============================================================================

Builds the LaTeX/PDF report for the occurrence-dynamics (pi_t) comparison
carried out by run_pi_dynamics_alternatives.py. Per docs/REPORTING.md §2-3,
this script performs NO optimisation and NO model fitting: it only reads
the artifacts already saved under artifacts/pi_dynamics_experiment/ and
turns them into tables, figures, and a report.tex/report.pdf.

Four pi_t dynamics alternatives (see pi_dynamics/ar_logistic.py,
pi_dynamics/ar_logistic_custom_lags.py, pi_dynamics/phi_linked.py) are
compared at each of the six locations already used in
reports/multi_location/. Every diagnostic in this report is IN-SAMPLE ONLY
-- this experiment produces no out-of-sample forecast at any stage.

=============================================================================
CODE WALKTHROUGH
=============================================================================

    main()
      |
    for each location:
        load the 4 variants' metadata.json + paths.npz
        for each variant:
            PIT histogram figure  (paths.npz["pit"])
            400-lag ACF figure    (paths.npz["qr"], diagnostics.plots.acf_plot)
        build 3 per-location tables: occurrence (pi|dry vs pi|wet), ACF at
        lags {1,2,3,364,365,366,367}, RMSE(all/dry/wet)+CRPS(all)
      |
    assemble report.tex:
        Executive Summary -> Introduction -> Model Specification ->
        Experimental Design -> Results (Tables by location, Figures by
        location) -> Discussion (data-driven, computed from the loaded
        metrics) -> Conclusion -> References
      |
    compile report.pdf (pdflatex, two passes)

=============================================================================
USAGE
=============================================================================

    python generate_pi_dynamics_report.py [--no-compile]

Output:
    reports/pi_dynamics_alternatives/report.tex
    reports/pi_dynamics_alternatives/report.pdf
    reports/pi_dynamics_alternatives/figures/<short>/*.png
    reports/pi_dynamics_alternatives/tables/<short>/*.csv, *.tex

To recompile the PDF manually after editing report.tex (no AI needed):
    cd reports/pi_dynamics_alternatives
    pdflatex -interaction=nonstopmode report.tex
    pdflatex -interaction=nonstopmode report.tex
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from constants import GAS_SEASONAL_LAGS
from data.station_loader import STATION_REGISTRY
from diagnostics.plots import acf_plot

ARTIFACTS_DIR = ROOT / "artifacts" / "pi_dynamics_experiment"
REPORT_DIR    = ROOT / "reports" / "pi_dynamics_alternatives"
FIG_DIR       = REPORT_DIR / "figures"
TAB_DIR       = REPORT_DIR / "tables"

LOCATION_ORDER = [
    "BELO HORIZONTE",
    "CRUZEIRO DO SUL (ACRE)",
    "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)",
    "MANAUS",
    "SALVADOR",
]

# Fixed reporting order for the four alternatives, with short codes used to
# keep numeric tables narrow (REPORTING.md §9 -- avoid overflowing pages),
# and the eta_t formula shown once in a legend rather than in every table.
VARIANT_ORDER = ["existing_ar_seasonal", "short_lags_ar", "short_lags_noar", "phi_linked"]
VARIANT_CODE  = {"existing_ar_seasonal": "A", "short_lags_ar": "B", "short_lags_noar": "C", "phi_linked": "D"}
VARIANT_LABEL = {
    "existing_ar_seasonal": "A -- Existing (AR + seasonal lags)",
    "short_lags_ar":        "B -- AR + short lags (1,2,3)",
    "short_lags_noar":      "C -- No AR, short lags (1,2,3)",
    "phi_linked":           "D -- Phi-linked",
}
VARIANT_FORMULA = {
    "existing_ar_seasonal": r"\eta_t = \omega_0 + \rho\,\eta_{t-1} + \sum_{l\in\{1,365,366\}}\omega_l\,y_{t-l}",
    "short_lags_ar":        r"\eta_t = \omega_0 + \rho\,\eta_{t-1} + \sum_{l=1}^{3}\omega_l\,y_{t-l}",
    "short_lags_noar":      r"\eta_t = \omega_0 + \sum_{l=1}^{3}\omega_l\,y_{t-l}",
    "phi_linked":           r"\eta_t = \lambda_0 + \lambda_1\,\varphi_{t|t-1}",
}
ACF_TABLE_LAGS = list(GAS_SEASONAL_LAGS["daily"])  # [1,2,3,364,365,366,367]


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tex(s) -> str:
    """Escape special LaTeX characters in arbitrary strings."""
    return (
        str(s)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _fmt(v, digits: int = 4) -> str:
    """Fixed-precision numeric formatting; NaN/None -> em dash."""
    if v is None:
        return "--"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return _tex(str(v))
    if not np.isfinite(fv):
        return "--"
    return f"{fv:.{digits}f}"


def _latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Booktabs table from a DataFrame of ALREADY-FORMATTED string cells."""
    cols = "l" + "r" * len(df.columns)
    header = " & ".join([""] + [_tex(c) for c in df.columns]) + r" \\"
    rows = []
    for idx, row in df.iterrows():
        rows.append(" & ".join([_tex(idx)] + [str(v) for v in row]) + r" \\")
    body = "\n".join(rows)
    return "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        header,
        r"\midrule",
        body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


LATEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{report}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{longtable}
\usepackage{caption}
\usepackage{hyperref}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}
\usepackage{enumitem}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
\emergencystretch=3em
\sloppy
\captionsetup{font=small}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Artifact loading
# ─────────────────────────────────────────────────────────────────────────────

def load_variant(short: str, variant_key: str) -> Optional[dict]:
    """Load one (location, variant) artifact; None if missing/failed."""
    out_dir = ARTIFACTS_DIR / short / variant_key
    meta_path = out_dir / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if not meta.get("success"):
        return None
    npz_path = out_dir / "paths.npz"
    if not npz_path.exists():
        return None
    npz = np.load(npz_path)
    meta["_pit"] = npz["pit"]
    meta["_qr"]  = npz["qr"]
    return meta


def load_location(short: str) -> Dict[str, Optional[dict]]:
    return {vk: load_variant(short, vk) for vk in VARIANT_ORDER}


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def pit_histogram_figure(pit: np.ndarray, title: str) -> plt.Figure:
    """PIT histogram vs. the Uniform(0,1) reference (see diagnostics/occurrence_pit.py)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    u = pit[np.isfinite(pit)]
    ax.hist(u, bins=20, density=True, color="steelblue", alpha=0.75,
            edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="r", lw=1.2, ls="--", label="Uniform(0,1)")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Randomised PIT value")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def make_figures_for_location(short: str, display: str, loaded: Dict[str, Optional[dict]]) -> Dict[str, dict]:
    """Save PIT + 400-lag ACF figures for every available variant; return relative paths."""
    loc_fig_dir = FIG_DIR / short
    loc_fig_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, dict] = {}

    for vk in VARIANT_ORDER:
        art = loaded[vk]
        if art is None:
            continue
        label = VARIANT_LABEL[vk]

        pit_path = loc_fig_dir / f"{short}_{vk}_pit_histogram.png"
        fig = pit_histogram_figure(art["_pit"], f"{display} -- {label}: occurrence PIT")
        fig.savefig(pit_path, dpi=150)
        plt.close(fig)

        acf_path = loc_fig_dir / f"{short}_{vk}_acf400.png"
        fig = acf_plot(art["_qr"], n_lags=400, location=f"{display} -- {label}")
        fig.savefig(acf_path, dpi=150)
        plt.close(fig)

        out[vk] = {
            "pit": pit_path.relative_to(REPORT_DIR).as_posix(),
            "acf": acf_path.relative_to(REPORT_DIR).as_posix(),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

def occurrence_table(short: str, loaded: Dict[str, Optional[dict]]) -> pd.DataFrame:
    """Raw numeric table (NaN for missing variants) -- see save_table_artifacts
    for the formatted display copy used in the LaTeX/PDF version."""
    rows = {}
    for vk in VARIANT_ORDER:
        art = loaded[vk]
        rows[VARIANT_CODE[vk]] = {
            "N dry": (art["n_dry_window"] if art else np.nan),
            "N wet": (art["n_wet_window"] if art else np.nan),
            "mean pi | y=0": (art["pi_mean_dry"] if art else np.nan),
            "mean pi | y>0": (art["pi_mean_wet"] if art else np.nan),
        }
    return pd.DataFrame(rows).T


def acf_table(short: str, loaded: Dict[str, Optional[dict]]) -> pd.DataFrame:
    """Raw numeric table (NaN for missing variants/lags)."""
    rows = {}
    for vk in VARIANT_ORDER:
        art = loaded[vk]
        acf_sel = art["acf_selected_lags"] if art else {}
        rows[VARIANT_CODE[vk]] = {f"lag {l}": acf_sel.get(str(l), np.nan) for l in ACF_TABLE_LAGS}
    return pd.DataFrame(rows).T


def rmse_crps_table(short: str, loaded: Dict[str, Optional[dict]]) -> pd.DataFrame:
    """Raw numeric table (NaN for missing variants)."""
    rows = {}
    for vk in VARIANT_ORDER:
        art = loaded[vk]
        rows[VARIANT_CODE[vk]] = {
            "RMSE all": (art["rmse_all"] if art else np.nan),
            "RMSE dry": (art["rmse_dry"] if art else np.nan),
            "RMSE wet": (art["rmse_wet"] if art else np.nan),
            "CRPS all": (art["crps_all"] if art else np.nan),
        }
    return pd.DataFrame(rows).T


def save_table_artifacts(
    short: str, name: str, df: pd.DataFrame, caption: str, label: str,
    int_cols: Optional[List[str]] = None,
) -> str:
    """
    Save the RAW numeric CSV (reusable outside the report, REPORTING.md §19)
    and return the LaTeX table string built from a separately formatted
    display copy (fixed 4-decimal precision via _fmt; int_cols rendered as
    plain integers, e.g. observation counts).
    """
    loc_tab_dir = TAB_DIR / short
    loc_tab_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(loc_tab_dir / f"{short}_{name}.csv")

    int_cols = set(int_cols or [])
    display = df.copy().astype(object)
    for col in display.columns:
        if col in int_cols:
            display[col] = df[col].apply(lambda v: "--" if pd.isna(v) else str(int(v)))
        else:
            display[col] = df[col].apply(_fmt)

    tex = _latex_table(display, caption, label)
    (loc_tab_dir / f"{short}_{name}.tex").write_text(tex)
    return tex


# ─────────────────────────────────────────────────────────────────────────────
# Per-location LaTeX sections
# ─────────────────────────────────────────────────────────────────────────────

def location_tables_section(short: str, display: str, loaded: Dict[str, Optional[dict]]) -> str:
    parts = [rf"\subsection{{{_tex(display)}}}"]

    n_missing = sum(1 for vk in VARIANT_ORDER if loaded[vk] is None)
    if n_missing:
        parts.append(
            rf"\textit{{Note: {n_missing} of 4 variants did not produce a "
            rf"valid artifact at this location and are shown as `--' below "
            rf"(see the failure summary in the Discussion section).}}"
        )

    occ = occurrence_table(short, loaded)
    parts.append(save_table_artifacts(
        short, "occurrence", occ,
        caption=(f"{display}: mean fitted occurrence probability $\\pi_t$ conditional on "
                 f"$y_t=0$ (dry) vs.\\ $y_t>0$ (wet), in-sample, common window "
                 f"$t\\in[\\text{{eff\\_start}}, T)$. A model that separates dry and wet "
                 f"days well should show mean $\\pi_t\\mid y_t{{>}}0$ well above "
                 f"mean $\\pi_t\\mid y_t{{=}}0$."),
        label=f"tab:{short}-occurrence",
        int_cols=["N dry", "N wet"],
    ))

    acf = acf_table(short, loaded)
    parts.append(save_table_artifacts(
        short, "acf", acf,
        caption=(f"{display}: sample autocorrelation of the randomised occurrence "
                 f"quantile residuals at lags 1--3 (short memory) and 364--367 "
                 f"(same-calendar-day, one year prior). Under $H_0$: no residual "
                 f"autocorrelation, values should lie within "
                 f"$\\pm 1.96/\\sqrt{{n}}$ of zero."),
        label=f"tab:{short}-acf",
    ))

    rc = rmse_crps_table(short, loaded)
    parts.append(save_table_artifacts(
        short, "rmse_crps", rc,
        caption=(f"{display}: in-sample RMSE (all days, dry days only, wet days only) "
                 f"and mean CRPS (all days) of the point/probabilistic forecast of "
                 f"$y_t$ itself, combining each pi-dynamics alternative's $\\pi_t$ "
                 f"with the SAME frozen magnitude path ($\\varphi_{{t|t-1}}$, static "
                 f"$\\xi,\\gamma,\\zeta$) from this location's best-by-CRPS phi-only "
                 f"Stage 1 baseline -- so differences are attributable to the "
                 f"occurrence dynamics alone. Lower is better for all four columns."),
        label=f"tab:{short}-rmse-crps",
    ))
    return "\n\n".join(parts)


def location_figures_section(short: str, display: str, loaded: Dict[str, Optional[dict]], figs: Dict[str, dict]) -> str:
    parts = [rf"\subsection{{{_tex(display)}}}"]
    for vk in VARIANT_ORDER:
        art = loaded[vk]
        if art is None or vk not in figs:
            continue
        label = VARIANT_LABEL[vk]
        parts.append(rf"\subsubsection*{{{_tex(label)}}}")
        parts.append(rf"$${VARIANT_FORMULA[vk]}$$")

        parts.append("\n".join([
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.75\textwidth]{{{figs[vk]['pit']}}}",
            rf"\caption{{{_tex(display)} -- {_tex(label)}: histogram of the randomised "
            r"occurrence PIT values (diagnostics/occurrence\_pit.py). Under correct "
            r"specification of $\pi_t$ this should be flat at height 1 (the dashed "
            r"line); systematic departures indicate mis-calibration of the occurrence "
            r"probability.}",
            rf"\label{{fig:{short}-{vk}-pit}}",
            r"\end{figure}",
        ]))

        parts.append("\n".join([
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.85\textwidth]{{{figs[vk]['acf']}}}",
            rf"\caption{{{_tex(display)} -- {_tex(label)}: autocorrelation of the "
            r"occurrence quantile residuals at 400 lags, with 95\% confidence bands "
            r"under $H_0$: no residual autocorrelation. Persistent bars near lag "
            r"365 would indicate unmodelled annual recurrence in the occurrence "
            r"process.}",
            rf"\label{{fig:{short}-{vk}-acf}}",
            r"\end{figure}",
            r"\clearpage",
        ]))
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Discussion (data-driven)
# ─────────────────────────────────────────────────────────────────────────────

def build_discussion(all_loaded: Dict[str, Dict[str, Optional[dict]]]) -> str:
    """Tally, across locations, which variant wins on each metric -- computed
    from the loaded artifacts, never hand-typed, per docs/REPORTING.md §26.4
    ('never allow manually entered numerical results')."""
    win_separation: Dict[str, int] = {vk: 0 for vk in VARIANT_ORDER}
    win_crps:       Dict[str, int] = {vk: 0 for vk in VARIANT_ORDER}
    win_rmse:       Dict[str, int] = {vk: 0 for vk in VARIANT_ORDER}
    failures: List[str] = []

    for loc in LOCATION_ORDER:
        short = STATION_REGISTRY[loc]["short"]
        loaded = all_loaded[short]
        available = {vk: art for vk, art in loaded.items() if art is not None}
        for vk in VARIANT_ORDER:
            if loaded[vk] is None:
                failures.append(f"{STATION_REGISTRY[loc]['display']} / {VARIANT_LABEL[vk]}")
        if not available:
            continue

        sep = {vk: (art["pi_mean_wet"] - art["pi_mean_dry"]) for vk, art in available.items()
               if np.isfinite(art["pi_mean_wet"]) and np.isfinite(art["pi_mean_dry"])}
        if sep:
            win_separation[max(sep, key=sep.get)] += 1

        crps = {vk: art["crps_all"] for vk, art in available.items() if np.isfinite(art["crps_all"])}
        if crps:
            win_crps[min(crps, key=crps.get)] += 1

        rmse = {vk: art["rmse_all"] for vk, art in available.items() if np.isfinite(art["rmse_all"])}
        if rmse:
            win_rmse[min(rmse, key=rmse.get)] += 1

    # Plain-text column names: _latex_table() escapes every header via _tex(),
    # which would otherwise mangle raw LaTeX math markup (e.g. "$\pi_t$").
    tally_df = pd.DataFrame({
        "Best dry/wet pi separation": {VARIANT_CODE[vk]: win_separation[vk] for vk in VARIANT_ORDER},
        "Lowest CRPS (all days)":     {VARIANT_CODE[vk]: win_crps[vk] for vk in VARIANT_ORDER},
        "Lowest RMSE (all days)":     {VARIANT_CODE[vk]: win_rmse[vk] for vk in VARIANT_ORDER},
    })
    tex_tally = _latex_table(
        tally_df.astype(str), caption="Number of locations (out of 6) at which each variant wins each metric.",
        label="tab:cross-location-tally",
    )
    (TAB_DIR / "cross_location_tally.csv").parent.mkdir(parents=True, exist_ok=True)
    tally_df.to_csv(TAB_DIR / "cross_location_tally.csv")

    top_sep = max(win_separation, key=win_separation.get)
    top_crps = max(win_crps, key=win_crps.get)
    top_rmse = max(win_rmse, key=win_rmse.get)

    text = [
        r"\section{Discussion}",
        (r"Table~\ref{tab:cross-location-tally} tallies, across the six locations, "
         r"how often each pi-dynamics alternative achieves the best value of three "
         r"summary metrics: dry/wet occurrence-probability separation "
         r"(mean $\pi_t\mid y_t{>}0$ minus mean $\pi_t\mid y_t{=}0$), in-sample CRPS, "
         r"and in-sample RMSE (both against $y_t$ itself, using the SAME frozen "
         r"magnitude path at a given location for all four alternatives -- see "
         r"Section~\ref{sec:experimental-design}). Counts need not sum to 6 if a "
         r"variant failed to produce a valid artifact at some location (see below)."),
        tex_tally,
        (rf"Variant {_tex(VARIANT_LABEL[top_sep])} most often achieves the largest "
         rf"dry/wet separation in the fitted occurrence probability "
         rf"({win_separation[top_sep]}/6 locations); "
         rf"{_tex(VARIANT_LABEL[top_crps])} most often achieves the lowest in-sample "
         rf"CRPS ({win_crps[top_crps]}/6); and "
         rf"{_tex(VARIANT_LABEL[top_rmse])} most often achieves the lowest in-sample "
         rf"RMSE ({win_rmse[top_rmse]}/6). All reported metrics are in-sample; "
         rf"this experiment does not evaluate out-of-sample performance, so these "
         rf"counts should be read as evidence about in-sample fit and calibration of "
         rf"the occurrence process, not as an out-of-sample forecasting recommendation."),
    ]

    if failures:
        text.append(r"\subsection*{Failed or missing variants}")
        text.append(
            "The following (location, variant) combinations did not produce a "
            "valid in-sample artifact and are excluded from the tally and shown "
            "as `--' in the per-location tables above: " +
            "; ".join(_tex(f) for f in failures) + "."
        )

    return "\n\n".join(text)


# ─────────────────────────────────────────────────────────────────────────────
# Full document assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_document(all_loaded: Dict[str, Dict[str, Optional[dict]]], all_figs: Dict[str, Dict[str, dict]], generation_date: str) -> str:
    parts = [
        LATEX_PREAMBLE,
        r"\begin{document}",
        r"\title{Occurrence-Probability ($\pi_t$) Dynamics: A Stage 1 Comparison}",
        rf"\author{{Generated: {_tex(generation_date)}}}",
        r"\date{}",
        r"\maketitle",
        r"\tableofcontents",
        r"\clearpage",
    ]

    # ── Executive Summary ────────────────────────────────────────────────
    parts.append(r"\chapter{Executive Summary}")
    parts.append(
        r"This report compares four alternative specifications for the "
        r"occurrence-probability logit $\eta_t$ (with $\pi_t=\sigma(\eta_t)$, "
        r"the probability that $y_t>0$) within the basic Stage 1 setting of "
        r"the ZA-GAS framework: GAS$(p,q)$, no covariates, only $\varphi_t$ "
        r"time-varying in the positive-part distribution (docs/MODELS.md \S12, "
        r"\S26). All results are \textbf{in-sample only}; no out-of-sample "
        r"forecast is produced anywhere in this experiment. Section~"
        r"\ref{sec:model-spec} defines the four alternatives; Section~"
        r"\ref{sec:experimental-design} defines the estimation procedure and "
        r"the common evaluation window; Section~\ref{sec:results} reports, "
        r"separately for tables and figures, each split into one subsection "
        r"per location; Section~\ref{sec:discussion} synthesises the "
        r"cross-location pattern."
    )

    # ── Introduction ──────────────────────────────────────────────────────
    parts.append(r"\chapter{Introduction}")
    parts.append(
        r"Every stage of the thesis pipeline (docs/MODELS.md \S26) uses one "
        r"fixed occurrence model, \texttt{ARLogisticPiDynamics} "
        r"(pi\_dynamics/ar\_logistic.py): an AR(1) logit process augmented "
        r"with lagged rainfall at $l\in\{1,365,366\}$. This experiment asks "
        r"whether that specific choice matters, by comparing it against three "
        r"alternatives that vary the lag structure, remove the AR memory "
        r"term, or replace the rainfall-lag drivers entirely with the "
        r"magnitude model's own one-step-ahead scale prediction "
        r"$\varphi_{t|t-1}$. All four are fit independently ("
        r"\textit{standalone}) on the binary wet/dry sequence alone: the "
        r"zero-augmented log-likelihood is additively separable in $\pi_t$ "
        r"versus the positive-part parameters (see "
        r"pi\_dynamics/standalone\_fit.py and models/static\_gb2.py), so a "
        r"standalone fit reaches the same optimum as a joint fit would, at a "
        r"fraction of the computational cost."
    )

    # ── Model Specification ──────────────────────────────────────────────
    parts.append(r"\chapter{Model Specification}")
    parts.append(r"\label{sec:model-spec}")
    parts.append(
        r"Throughout, $\pi_t=\sigma(\eta_t)=1/(1+e^{-\eta_t})$. The four "
        r"alternatives compared in this report are:"
    )
    items = []
    for vk in VARIANT_ORDER:
        items.append(rf"\item \textbf{{{_tex(VARIANT_LABEL[vk])}}}: $${VARIANT_FORMULA[vk]}$$")
    parts.append(r"\begin{enumerate}" + "\n".join(items) + r"\end{enumerate}")
    parts.append(
        r"For Model D, $\varphi_{t|t-1}$ is the transformed GAS$(p,q)$ scale "
        r"state (docs/MODELS.md \S5 -- GB2LogLink already works in the "
        r"transformed/log parameterisation, so $\varphi_{t|t-1}$ is used "
        r"directly, with no additional link function), frozen from the "
        r"location's best-by-CRPS phi-only Stage 1 baseline (one of the six "
        r"cached \texttt{stage1\_phi\_\{short,seasonal\}\_\{unit,diagfi,"
        r"fullfi\}} artifacts). For $t$ before that baseline's effective "
        r"sample start, $\varphi_{t|t-1}$ is held at the filter's fitted "
        r"initial state $f_{0,\varphi}$, exactly matching the GAS filter's "
        r"own warm-up convention (models/gas\_filter.py)."
    )

    # ── Experimental Design ──────────────────────────────────────────────
    parts.append(r"\chapter{Experimental Design}")
    parts.append(r"\label{sec:experimental-design}")
    parts.append(
        r"\textbf{Locations.} The same six locations used throughout this "
        r"framework's multi-location reports: Belo Horizonte, Cruzeiro do "
        r"Sul, Darwin Airport, Garanhuns, Manaus, and Salvador."
    )
    parts.append(
        r"\textbf{Estimation.} Each of the four $\eta_t$ specifications is "
        r"fit by unbounded BFGS (with an L-BFGS-B warm start) maximising the "
        r"Bernoulli log-likelihood of $1(y_t>0)$ alone "
        r"(pi\_dynamics/standalone\_fit.py) -- no distribution parameters "
        r"for the positive part are estimated in this step."
    )
    parts.append(
        r"\textbf{Common evaluation window.} All diagnostics below use the "
        r"window $t\in[\text{eff\_start}, T)$, where eff\_start is the "
        r"frozen magnitude baseline's GAS effective-sample start (367 for "
        r"the seasonal daily lag set $\{1,2,3,364,365,366,367\}$, or 3 for "
        r"the short lag set $\{1,2,3\}$, per the selected baseline at each "
        r"location). This window is common to all four variants so that "
        r"Model D -- whose $\eta_t$ is only genuinely data-driven from "
        r"eff\_start onward, since $\varphi_{t|t-1}$ is held constant before "
        r"that -- is compared on equal footing with the other three."
    )
    parts.append(
        r"\textbf{Occurrence PIT.} $y_t>0$ is a Bernoulli$(\pi_t)$ indicator; "
        r"its CDF is a two-point step function, so both outcomes are "
        r"randomised (diagnostics/occurrence\_pit.py): "
        r"$U_t\sim\text{Uniform}(0,1-\pi_t)$ if $y_t=0$, "
        r"$U_t\sim\text{Uniform}(1-\pi_t,1)$ if $y_t>0$. Under correct "
        r"specification $U_t\sim\text{iid Uniform}(0,1)$, and quantile "
        r"residuals $r_t=\Phi^{-1}(U_t)\sim\text{iid }N(0,1)$ (Dunn \& Smyth, "
        r"1996), used for the ACF diagnostics below."
    )
    parts.append(
        r"\textbf{RMSE / CRPS against $y_t$.} To see whether the choice of "
        r"$\pi_t$ dynamics changes how well the full model predicts rainfall "
        r"itself, each variant's $\pi_t$ is combined with the SAME frozen "
        r"$(\varphi_{t|t-1},\xi,\gamma,\zeta)$ path from the location's "
        r"selected baseline (diagnostics/pi\_dynamics\_eval.py). The point "
        r"forecast is $\mu_t=\pi_t\,E[G_t]$ (GB2LogLink.mean(), an existing "
        r"analytical raw moment); CRPS uses the same Monte Carlo "
        r"energy-score estimator (diagnostics/scoring.py:crps\_mc) used "
        r"throughout this framework's OOS evaluation, applied here in-sample."
    )

    # ── Results ───────────────────────────────────────────────────────────
    parts.append(r"\chapter{Results}")
    parts.append(r"\label{sec:results}")

    parts.append(r"\section{Tables}")
    for loc in LOCATION_ORDER:
        short = STATION_REGISTRY[loc]["short"]
        display = STATION_REGISTRY[loc]["display"]
        parts.append(location_tables_section(short, display, all_loaded[short]))

    parts.append(r"\section{Figures}")
    for loc in LOCATION_ORDER:
        short = STATION_REGISTRY[loc]["short"]
        display = STATION_REGISTRY[loc]["display"]
        parts.append(location_figures_section(short, display, all_loaded[short], all_figs[short]))

    # ── Discussion ────────────────────────────────────────────────────────
    parts.append(build_discussion(all_loaded))
    parts.append(r"\label{sec:discussion}")

    # ── Conclusion ────────────────────────────────────────────────────────
    parts.append(r"\chapter{Conclusion}")
    parts.append(
        r"This report compared four occurrence-probability dynamics against "
        r"a common, frozen magnitude model, in-sample, at six locations. The "
        r"cross-location tally in Section~\ref{sec:discussion} identifies "
        r"which specification most consistently improves dry/wet separation, "
        r"CRPS, and RMSE; the per-location tables and figures in Section~"
        r"\ref{sec:results} support closer inspection of any single "
        r"location. As with every stage of this framework, adopting an "
        r"alternative occurrence model in the main thesis pipeline is a "
        r"separate decision, to be made only after also considering "
        r"out-of-sample performance (out of scope for this report)."
    )

    # ── References ────────────────────────────────────────────────────────
    parts.append(r"\chapter*{References}")
    parts.append(r"\begin{itemize}")
    parts.append(r"\item Creal, D., Koopman, S.J., \& Lucas, A. (2013). Generalized "
                 r"autoregressive score models with applications. \textit{Journal of "
                 r"Applied Econometrics}, 28(5), 777--795.")
    parts.append(r"\item Dunn, P.K., \& Smyth, G.K. (1996). Randomized quantile "
                 r"residuals. \textit{Journal of Computational and Graphical "
                 r"Statistics}, 5(3), 236--244.")
    parts.append(r"\item Czado, C., Gneiting, T., \& Held, L. (2009). Predictive "
                 r"model assessment for count data. \textit{Biometrics}, 65(4), "
                 r"1254--1261.")
    parts.append(r"\item Gneiting, T., \& Raftery, A.E. (2007). Strictly proper "
                 r"scoring rules, prediction, and estimation. \textit{JASA}, "
                 r"102(477), 359--378.")
    parts.append(r"\end{itemize}")

    parts.append(r"\end{document}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# PDF compilation
# ─────────────────────────────────────────────────────────────────────────────

def compile_latex(tex_path: Path) -> bool:
    """Run pdflatex twice to resolve cross-references (TOC, labels)."""
    tex_dir = tex_path.parent
    cmd = ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(tex_dir), str(tex_path)]
    for pass_num in range(1, 3):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=tex_dir)
            if result.returncode != 0:
                print(f"  pdflatex pass {pass_num} returned code {result.returncode}")
                log_path = tex_dir / (tex_path.stem + ".log")
                if log_path.exists():
                    for line in log_path.read_text(errors="replace").splitlines()[-40:]:
                        print(f"    {line}")
                if pass_num == 1:
                    return False
        except FileNotFoundError:
            print("  pdflatex not found -- skipping PDF compilation.")
            return False
        except subprocess.TimeoutExpired:
            print("  pdflatex timed out.")
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the pi-dynamics alternatives report.")
    parser.add_argument("--no-compile", action="store_true", help="Skip pdflatex compilation.")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    all_loaded: Dict[str, Dict[str, Optional[dict]]] = {}
    all_figs:   Dict[str, Dict[str, dict]] = {}

    for loc in LOCATION_ORDER:
        short = STATION_REGISTRY[loc]["short"]
        display = STATION_REGISTRY[loc]["display"]
        print(f"Loading artifacts: {display} ({short})")
        loaded = load_location(short)
        n_ok = sum(1 for v in loaded.values() if v is not None)
        print(f"  {n_ok}/4 variants available")
        all_loaded[short] = loaded
        all_figs[short] = make_figures_for_location(short, display, loaded)

    generation_date = time.strftime("%Y-%m-%d %H:%M:%S")
    doc = build_document(all_loaded, all_figs, generation_date)
    tex_path = REPORT_DIR / "report.tex"
    tex_path.write_text(doc, encoding="utf-8")
    print(f"Wrote {tex_path}")

    if not args.no_compile:
        ok = compile_latex(tex_path)
        if ok:
            print(f"Compiled {REPORT_DIR / 'report.pdf'}")
        else:
            print("PDF compilation failed or pdflatex unavailable -- report.tex was still written.")


if __name__ == "__main__":
    main()
