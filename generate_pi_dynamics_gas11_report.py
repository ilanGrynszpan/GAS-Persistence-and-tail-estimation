"""
Pi-dynamics alternatives report generator -- GAS(1,1) magnitude variant
(2026-09-09).

=============================================================================
OVERVIEW
=============================================================================

Builds the LaTeX/PDF report for the occurrence-dynamics (pi_t) comparison
carried out by run_pi_dynamics_gas11.py. Per docs/REPORTING.md §2-3, this
script performs NO optimisation and NO model fitting: it only reads the
artifacts under artifacts/pi_dynamics_gas11_experiment/ and turns them into
tables, figures, and report.tex/report.pdf.

This is the same report as generate_pi_dynamics_report.py, for the same four
pi_t alternatives at the same six locations, with two differences driven by
run_pi_dynamics_gas11.py:

  1. The frozen magnitude scale path varphi_{t|t-1} follows a freshly-fit
     phi-only GAS(1,1) (a single GAS lag, one score + one persistence
     coefficient) instead of a cached Stage 1 short/seasonal-lag GAS
     artifact. Its score scaling is selected per location by a gradient-tier
     then AIC rule (see run_pi_dynamics_gas11.py's docstring); this report
     adds a per-location "magnitude scaling selection" table documenting
     that choice.
  2. A new "gradient of the optimised models" table per location, plus a
     cross-location gradient-norm summary in the Discussion, reporting the
     independent numerical_gradient_checked gradient at each fitted optimum
     for the GAS(1,1) magnitude model and all four standalone pi_t fits.

Every diagnostic is IN-SAMPLE ONLY -- this experiment produces no
out-of-sample forecast at any stage.

=============================================================================
CODE WALKTHROUGH
=============================================================================

    main()
      |
    for each location:
        load magnitude_gas11 metadata + scaling_selection.csv + gradients.csv
        load the 4 variants' metadata.json + paths.npz + gradients.csv
        for each variant:
            PIT histogram figure   (paths.npz["pit"])
            400-lag ACF figure     (paths.npz["qr"], diagnostics.plots.acf_plot)
        build per-location tables:
            magnitude scaling selection (loglik / AIC / grad norms / tier / selected)
            occurrence (pi|dry vs pi|wet)
            ACF at lags {1,2,3,364,365,366,367}
            RMSE(all/dry/wet) + CRPS(all)
            gradient of the optimised models  (magnitude + A..D, per parameter)
      |
    assemble report.tex (Executive Summary -> Introduction -> Model
    Specification -> Experimental Design -> Results -> Discussion ->
    Conclusion -> References), including a cross-location gradient-norm
    summary table
      |
    compile report.pdf (pdflatex, two passes)

=============================================================================
USAGE
=============================================================================

    python generate_pi_dynamics_gas11_report.py [--no-compile]

Output:
    reports/pi_dynamics_gas11/report.tex
    reports/pi_dynamics_gas11/report.pdf
    reports/pi_dynamics_gas11/figures/<short>/*.png
    reports/pi_dynamics_gas11/tables/<short>/*.csv, *.tex

To recompile the PDF manually after editing report.tex (no AI needed):
    cd reports/pi_dynamics_gas11
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

ARTIFACTS_DIR = ROOT / "artifacts" / "pi_dynamics_gas11_experiment"
REPORT_DIR    = ROOT / "reports" / "pi_dynamics_gas11"
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
# Row label for the magnitude model in the gradient table.
MAG_LABEL = "M -- GAS(1,1) magnitude"
ACF_TABLE_LAGS = list(GAS_SEASONAL_LAGS["daily"])  # [1,2,3,364,365,366,367]


# -----------------------------------------------------------------------------
# LaTeX helpers
# -----------------------------------------------------------------------------

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


def _fmt_sci(v, digits: int = 3) -> str:
    """Scientific-notation formatting for gradients that span many orders of magnitude."""
    if v is None:
        return "--"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return _tex(str(v))
    if not np.isfinite(fv):
        return "--"
    return f"{fv:.{digits}e}"


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


# -----------------------------------------------------------------------------
# Artifact loading
# -----------------------------------------------------------------------------

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
    grad_path = out_dir / "gradients.csv"
    meta["_grad"] = pd.read_csv(grad_path) if grad_path.exists() else None
    return meta


def load_magnitude(short: str) -> Optional[dict]:
    """Load the GAS(1,1) magnitude selection + gradient artifacts; None if missing."""
    mag_dir = ARTIFACTS_DIR / short / "magnitude_gas11"
    meta_path = mag_dir / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    sel_path = mag_dir / "scaling_selection.csv"
    meta["_scaling_selection"] = pd.read_csv(sel_path) if sel_path.exists() else None
    grad_path = mag_dir / "gradients.csv"
    meta["_grad"] = pd.read_csv(grad_path) if grad_path.exists() else None
    return meta


def load_location(short: str) -> Dict[str, Optional[dict]]:
    out = {vk: load_variant(short, vk) for vk in VARIANT_ORDER}
    out["_magnitude"] = load_magnitude(short)
    return out


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------

def magnitude_selection_table(mag: Optional[dict]) -> pd.DataFrame:
    """One row per score scaling tried for the GAS(1,1) magnitude model."""
    if mag is None or mag.get("_scaling_selection") is None:
        return pd.DataFrame()
    sel = mag["_scaling_selection"].copy()
    sel = sel.set_index("scaling")
    rows = {}
    for scaling, r in sel.iterrows():
        rows[scaling] = {
            "loglik": r.get("loglik", np.nan),
            "AIC": r.get("aic", np.nan),
            "grad inf-norm": r.get("grad_norm_inf", np.nan),
            "grad 2-norm": r.get("grad_norm_2", np.nan),
            "grad conv.": "yes" if bool(r.get("grad_converged", False)) else "no",
            "tier": int(r["selection_tier"]) if not pd.isna(r.get("selection_tier", np.nan)) else np.nan,
            "selected": "yes" if bool(r.get("selected", False)) else "",
        }
    return pd.DataFrame(rows).T


def occurrence_table(loaded: Dict[str, Optional[dict]]) -> pd.DataFrame:
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


def acf_table(loaded: Dict[str, Optional[dict]]) -> pd.DataFrame:
    rows = {}
    for vk in VARIANT_ORDER:
        art = loaded[vk]
        acf_sel = art["acf_selected_lags"] if art else {}
        rows[VARIANT_CODE[vk]] = {f"lag {l}": acf_sel.get(str(l), np.nan) for l in ACF_TABLE_LAGS}
    return pd.DataFrame(rows).T


def rmse_crps_table(loaded: Dict[str, Optional[dict]]) -> pd.DataFrame:
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


def gradient_table(loaded: Dict[str, Optional[dict]]) -> pd.DataFrame:
    """
    Gradient of the exact penalised negative log-likelihood objective at the
    fitted optimum, per parameter, for the GAS(1,1) magnitude model and each
    of the four standalone pi_t fits at this location. Independent
    central-difference gradient with the eps-convergence check of
    diagnostics/numerical_diff.numerical_gradient_checked.
    """
    rows = {}
    mag = loaded.get("_magnitude")
    if mag is not None and mag.get("_grad") is not None:
        for _, r in mag["_grad"].iterrows():
            rows[f"{MAG_LABEL} / {r['parameter']}"] = {"gradient": r["gradient"]}
    for vk in VARIANT_ORDER:
        art = loaded[vk]
        if art is None or art.get("_grad") is None:
            continue
        for _, r in art["_grad"].iterrows():
            rows[f"{VARIANT_CODE[vk]} / {r['parameter']}"] = {"gradient": r["gradient"]}
    return pd.DataFrame(rows).T


def save_table_artifacts(
    short: str, name: str, df: pd.DataFrame, caption: str, label: str,
    int_cols: Optional[List[str]] = None,
    sci_cols: Optional[List[str]] = None,
    str_cols: Optional[List[str]] = None,
    digits: int = 4,
) -> str:
    """
    Save the RAW numeric CSV (reusable outside the report, REPORTING.md §19)
    and return the LaTeX table string built from a separately formatted
    display copy.
    """
    loc_tab_dir = TAB_DIR / short
    loc_tab_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(loc_tab_dir / f"{short}_{name}.csv")

    if df.empty:
        return rf"\textit{{No data available for table `{_tex(name)}' at this location.}}"

    int_cols = set(int_cols or [])
    sci_cols = set(sci_cols or [])
    str_cols = set(str_cols or [])
    display = df.copy().astype(object)
    for col in display.columns:
        if col in str_cols:
            display[col] = df[col].apply(lambda v: "--" if (v is None or (isinstance(v, float) and pd.isna(v))) else _tex(str(v)))
        elif col in int_cols:
            display[col] = df[col].apply(lambda v: "--" if pd.isna(v) else str(int(v)))
        elif col in sci_cols:
            display[col] = df[col].apply(lambda v: _fmt_sci(v))
        else:
            display[col] = df[col].apply(lambda v: _fmt(v, digits))

    tex = _latex_table(display, caption, label)
    (loc_tab_dir / f"{short}_{name}.tex").write_text(tex)
    return tex


# -----------------------------------------------------------------------------
# Per-location LaTeX sections
# -----------------------------------------------------------------------------

def location_tables_section(short: str, display: str, loaded: Dict[str, Optional[dict]]) -> str:
    parts = [rf"\subsection{{{_tex(display)}}}"]

    mag = loaded.get("_magnitude")
    if mag is not None:
        parts.append(
            rf"Magnitude model: phi-only GAS(1,1) (\texttt{{gas\_lags=[1]}}), score "
            rf"scaling \textbf{{{_tex(mag.get('selected_scaling', '?'))}}} selected via "
            rf"tier \texttt{{{_tex(mag.get('selection_tier', '?'))}}} "
            rf"(in-sample log-likelihood {_fmt(mag.get('loglik'), 2)}, "
            rf"AIC {_fmt(mag.get('aic'), 2)}, "
            rf"$\lVert\nabla\rVert_\infty = {_fmt_sci(mag.get('grad_norm_inf'))}$, "
            rf"validity \texttt{{{_tex(mag.get('validity', '?'))}}})."
        )
        parts.append(save_table_artifacts(
            short, "magnitude_selection", magnitude_selection_table(mag),
            caption=(f"{display}: phi-only GAS(1,1) magnitude model estimated under all "
                     f"three score scalings (docs/MODELS.md \\S10). Selection rule: the "
                     f"most-preferred non-empty gradient tier "
                     f"(1: $\\lVert\\nabla\\rVert_\\infty<1$; 2: $<100$; 3: not failed; "
                     f"4: any), then lowest AIC within it. The independent "
                     f"central-difference gradient (\\texttt{{numerical\\_gradient\\_checked}}) "
                     f"is used for the tier; `grad conv.' is its eps-convergence flag. "
                     f"Lower AIC is better."),
            label=f"tab:{short}-magnitude-selection",
            sci_cols=["grad inf-norm", "grad 2-norm"],
            int_cols=["tier"],
            str_cols=["grad conv.", "selected"],
            digits=2,
        ))

    n_missing = sum(1 for vk in VARIANT_ORDER if loaded[vk] is None)
    if n_missing:
        parts.append(
            rf"\textit{{Note: {n_missing} of 4 occurrence variants did not produce a "
            rf"valid artifact at this location and are shown as `--' below "
            rf"(see the failure summary in the Discussion section).}}"
        )

    parts.append(save_table_artifacts(
        short, "occurrence", occurrence_table(loaded),
        caption=(f"{display}: mean fitted occurrence probability $\\pi_t$ conditional on "
                 f"$y_t=0$ (dry) vs.\\ $y_t>0$ (wet), in-sample, common window "
                 f"$t\\in[\\text{{eff\\_start}}, T)$ with eff\\_start $=1$ (GAS(1,1), a "
                 f"single lag). A model that separates dry and wet days well should show "
                 f"mean $\\pi_t\\mid y_t{{>}}0$ well above mean $\\pi_t\\mid y_t{{=}}0$."),
        label=f"tab:{short}-occurrence",
        int_cols=["N dry", "N wet"],
    ))

    parts.append(save_table_artifacts(
        short, "acf", acf_table(loaded),
        caption=(f"{display}: sample autocorrelation of the randomised occurrence "
                 f"quantile residuals at lags 1--3 (short memory) and 364--367 "
                 f"(same-calendar-day, one year prior). Under $H_0$: no residual "
                 f"autocorrelation, values should lie within $\\pm 1.96/\\sqrt{{n}}$ of zero."),
        label=f"tab:{short}-acf",
    ))

    parts.append(save_table_artifacts(
        short, "rmse_crps", rmse_crps_table(loaded),
        caption=(f"{display}: in-sample RMSE (all/dry/wet days) and mean CRPS (all days) "
                 f"of the forecast of $y_t$ itself, combining each pi-dynamics "
                 f"alternative's $\\pi_t$ with the SAME frozen GAS(1,1) magnitude path "
                 f"($\\varphi_{{t|t-1}}$, static $\\xi,\\gamma,\\zeta$) for this location "
                 f"-- so differences are attributable to the occurrence dynamics alone. "
                 f"Lower is better for all four columns."),
        label=f"tab:{short}-rmse-crps",
    ))

    parts.append(save_table_artifacts(
        short, "gradients", gradient_table(loaded),
        caption=(f"{display}: gradient of the exact penalised negative log-likelihood "
                 f"objective at the fitted optimum, per parameter, for the GAS(1,1) "
                 f"magnitude model (M) and the four standalone occurrence-dynamics fits "
                 f"(A--D). Independent central-difference gradient with the "
                 f"eps-convergence check of \\texttt{{numerical\\_gradient\\_checked}} "
                 f"(diagnostics/numerical\\_diff.py). At a clean interior optimum every "
                 f"entry should be near zero; large entries flag a boundary-adjacent or "
                 f"unconverged direction."),
        label=f"tab:{short}-gradients",
        sci_cols=["gradient"],
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


# -----------------------------------------------------------------------------
# Discussion (data-driven)
# -----------------------------------------------------------------------------

def _gradient_norm_summary(all_loaded: Dict[str, Dict[str, Optional[dict]]]) -> str:
    """Cross-location gradient-norm summary: one row per (location, model)."""
    rows = {}
    for loc in LOCATION_ORDER:
        short = STATION_REGISTRY[loc]["short"]
        display = STATION_REGISTRY[loc]["display"]
        loaded = all_loaded[short]

        mag = loaded.get("_magnitude")
        if mag is not None:
            rows[f"{display} / M ({mag.get('selected_scaling','?')})"] = {
                "grad inf-norm": mag.get("grad_norm_inf", np.nan),
                "grad 2-norm": mag.get("grad_norm_2", np.nan),
                "eps": mag.get("grad_eps_used", np.nan),
                "converged": "yes" if bool(mag.get("grad_converged", False)) else "no",
            }
        for vk in VARIANT_ORDER:
            art = loaded[vk]
            if art is None:
                continue
            rows[f"{display} / {VARIANT_CODE[vk]}"] = {
                "grad inf-norm": art.get("grad_norm_inf", np.nan),
                "grad 2-norm": art.get("grad_norm_2", np.nan),
                "eps": art.get("grad_eps_used", np.nan),
                "converged": "yes" if bool(art.get("grad_converged", False)) else "no",
            }

    df = pd.DataFrame(rows).T
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB_DIR / "cross_location_gradient_norms.csv")
    if df.empty:
        return r"\textit{No gradient artifacts available.}"

    display_df = df.copy().astype(object)
    for col in ("grad inf-norm", "grad 2-norm", "eps"):
        display_df[col] = df[col].apply(lambda v: _fmt_sci(v))
    display_df["converged"] = df["converged"].apply(lambda v: _tex(str(v)))
    return _latex_table(
        display_df,
        caption=("Independent numerical gradient of each optimised model's own objective "
                 "at its fitted theta, across all six locations. `M' is the phi-only "
                 "GAS(1,1) magnitude model (selected score scaling in parentheses); "
                 "A--D are the standalone occurrence-dynamics fits. `eps' is the finite "
                 "-difference step chosen by the convergence check; `converged' is "
                 "whether two consecutive step sizes agreed to within 5\\%."),
        label="tab:cross-location-gradient-norms",
    )


def build_discussion(all_loaded: Dict[str, Dict[str, Optional[dict]]]) -> str:
    win_separation: Dict[str, int] = {vk: 0 for vk in VARIANT_ORDER}
    win_crps:       Dict[str, int] = {vk: 0 for vk in VARIANT_ORDER}
    win_rmse:       Dict[str, int] = {vk: 0 for vk in VARIANT_ORDER}
    failures: List[str] = []

    for loc in LOCATION_ORDER:
        short = STATION_REGISTRY[loc]["short"]
        loaded = all_loaded[short]
        available = {vk: art for vk, art in loaded.items() if vk in VARIANT_ORDER and art is not None}
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

    tally_df = pd.DataFrame({
        "Best dry/wet pi separation": {VARIANT_CODE[vk]: win_separation[vk] for vk in VARIANT_ORDER},
        "Lowest CRPS (all days)":     {VARIANT_CODE[vk]: win_crps[vk] for vk in VARIANT_ORDER},
        "Lowest RMSE (all days)":     {VARIANT_CODE[vk]: win_rmse[vk] for vk in VARIANT_ORDER},
    })
    tex_tally = _latex_table(
        tally_df.astype(str),
        caption="Number of locations (out of 6) at which each variant wins each metric.",
        label="tab:cross-location-tally",
    )
    (TAB_DIR / "cross_location_tally.csv").parent.mkdir(parents=True, exist_ok=True)
    tally_df.to_csv(TAB_DIR / "cross_location_tally.csv")

    # Magnitude scaling-selection summary across locations.
    mag_rows = {}
    for loc in LOCATION_ORDER:
        short = STATION_REGISTRY[loc]["short"]
        mag = all_loaded[short].get("_magnitude")
        if mag is None:
            continue
        mag_rows[STATION_REGISTRY[loc]["display"]] = {
            "scaling": mag.get("selected_scaling", "?"),
            "tier": mag.get("selection_tier", "?"),
            "loglik": mag.get("loglik", np.nan),
            "AIC": mag.get("aic", np.nan),
            "grad inf-norm": mag.get("grad_norm_inf", np.nan),
            "validity": mag.get("validity", "?"),
        }
    mag_df = pd.DataFrame(mag_rows).T
    if not mag_df.empty:
        mag_df.to_csv(TAB_DIR / "cross_location_magnitude_selection.csv")
        mag_disp = mag_df.copy().astype(object)
        mag_disp["loglik"] = mag_df["loglik"].apply(lambda v: _fmt(v, 2))
        mag_disp["AIC"] = mag_df["AIC"].apply(lambda v: _fmt(v, 2))
        mag_disp["grad inf-norm"] = mag_df["grad inf-norm"].apply(lambda v: _fmt_sci(v))
        for c in ("scaling", "tier", "validity"):
            mag_disp[c] = mag_df[c].apply(lambda v: _tex(str(v)))
        tex_mag = _latex_table(
            mag_disp,
            caption=("Per-location outcome of the GAS(1,1) magnitude model's score-scaling "
                     "selection (all three fit; most-preferred non-empty gradient tier, "
                     "then lowest AIC)."),
            label="tab:cross-location-magnitude-selection",
        )
    else:
        tex_mag = r"\textit{No magnitude-model artifacts available.}"

    top_sep = max(win_separation, key=win_separation.get)
    top_crps = max(win_crps, key=win_crps.get)
    top_rmse = max(win_rmse, key=win_rmse.get)

    text = [
        r"\section{Discussion}",
        r"\subsection{GAS(1,1) magnitude model}",
        (r"Table~\ref{tab:cross-location-magnitude-selection} records which score scaling "
         r"was frozen as the magnitude model at each location, together with the "
         r"gradient tier that decision fell into. Because the three scalings share the "
         r"same parameter count, the AIC tie-break within a tier is equivalent to "
         r"picking the highest in-sample log-likelihood."),
        tex_mag,
        r"\subsection{Occurrence-dynamics comparison}",
        (r"Table~\ref{tab:cross-location-tally} tallies, across the six locations, "
         r"how often each pi-dynamics alternative achieves the best value of three "
         r"summary metrics: dry/wet occurrence-probability separation "
         r"(mean $\pi_t\mid y_t{>}0$ minus mean $\pi_t\mid y_t{=}0$), in-sample CRPS, "
         r"and in-sample RMSE (both against $y_t$ itself, using the SAME frozen "
         r"GAS(1,1) magnitude path at a given location for all four alternatives). "
         r"Counts need not sum to 6 if a variant failed to produce a valid artifact "
         r"at some location (see below)."),
        tex_tally,
        (rf"Variant {_tex(VARIANT_LABEL[top_sep])} most often achieves the largest "
         rf"dry/wet separation in the fitted occurrence probability "
         rf"({win_separation[top_sep]}/6 locations); "
         rf"{_tex(VARIANT_LABEL[top_crps])} most often achieves the lowest in-sample "
         rf"CRPS ({win_crps[top_crps]}/6); and "
         rf"{_tex(VARIANT_LABEL[top_rmse])} most often achieves the lowest in-sample "
         rf"RMSE ({win_rmse[top_rmse]}/6). All reported metrics are in-sample; "
         rf"this experiment does not evaluate out-of-sample performance."),
        r"\subsection{Gradient diagnostics}",
        (r"Table~\ref{tab:cross-location-gradient-norms} summarises the independent "
         r"numerical gradient of every optimised model's own objective at its fitted "
         r"$\hat\theta$. These are the norms of the per-parameter gradients tabulated "
         r"location-by-location in Section~\ref{sec:results}. A small "
         r"$\lVert\nabla\rVert_\infty$ together with \texttt{converged=yes} is evidence "
         r"that the optimiser reached a genuine interior stationary point; a large "
         r"norm, or \texttt{converged=no}, marks a fit whose reported parameters and "
         r"standard errors should be treated with caution (docs/OPTIMIZATION.md "
         r"\S10--\S11)."),
        _gradient_norm_summary(all_loaded),
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


# -----------------------------------------------------------------------------
# Full document assembly
# -----------------------------------------------------------------------------

def build_document(all_loaded, all_figs, generation_date: str) -> str:
    parts = [
        LATEX_PREAMBLE,
        r"\begin{document}",
        r"\title{Occurrence-Probability ($\pi_t$) Dynamics with a GAS(1,1) Magnitude Model}",
        rf"\author{{Generated: {_tex(generation_date)}}}",
        r"\date{}",
        r"\maketitle",
        r"\tableofcontents",
        r"\clearpage",
    ]

    parts.append(r"\chapter{Executive Summary}")
    parts.append(
        r"This report compares four alternative specifications for the "
        r"occurrence-probability logit $\eta_t$ (with $\pi_t=\sigma(\eta_t)$) "
        r"within the zero-augmented mixture $M_t=D_t\,Y_t$, "
        r"$D_t\sim\text{Bernoulli}(\pi_{t|t-1})$, "
        r"$Y_t\sim\text{GB2}(\varphi,\gamma,\xi,\zeta)$ (docs/MODELS.md \S3--\S5). "
        r"It repeats the 2026-07-14 pi-dynamics comparison with one change: the "
        r"time-varying scale $\varphi_{t|t-1}$ follows a freshly-estimated phi-only "
        r"\textbf{GAS(1,1)} recursion "
        r"$\varphi_{t+1|t}=\omega_\varphi+A_\varphi s_t+B_\varphi\varphi_{t|t-1}$ "
        r"(the score-driven update of docs/MODELS.md \S12 restricted to a single lag), "
        r"rather than a cached Stage 1 short/seasonal-lag GAS path. All results are "
        r"\textbf{in-sample only}. Section~\ref{sec:model-spec} defines the four "
        r"alternatives and the magnitude model; Section~\ref{sec:experimental-design} "
        r"the estimation procedure, the score-scaling selection, and the gradient "
        r"diagnostics; Section~\ref{sec:results} reports per-location tables (including "
        r"the new gradient-of-the-optimised-models table) and figures; "
        r"Section~\ref{sec:discussion} synthesises the cross-location pattern."
    )

    parts.append(r"\chapter{Introduction}")
    parts.append(
        r"The 2026-07-14 experiment (reports/pi\_dynamics\_alternatives) held the "
        r"magnitude model fixed at whichever cached Stage 1 phi-only GAS artifact "
        r"(short lag set $\{1,2,3\}$ or seasonal $\{1,2,3,364,365,366,367\}$) scored "
        r"best by CRPS at each location, and compared four occurrence dynamics against "
        r"it. Here the magnitude model is instead a phi-only \textbf{GAS(1,1)}: the "
        r"same score-driven scale recursion, but with a single lag $L=\{1\}$ -- one "
        r"score coefficient $A_\varphi$ and one persistence coefficient $B_\varphi$. "
        r"This is the most parsimonious score-driven scale process, and it is "
        r"re-estimated from the data here rather than reused. The four occurrence "
        r"alternatives, their standalone estimation, the common evaluation window, and "
        r"every diagnostic are otherwise unchanged from the earlier experiment."
    )

    parts.append(r"\chapter{Model Specification}")
    parts.append(r"\label{sec:model-spec}")
    parts.append(
        r"Throughout, $\pi_t=\sigma(\eta_t)=1/(1+e^{-\eta_t})$, and the positive part "
        r"$Y_t$ is GB2 in the transformed/log parameterisation (docs/MODELS.md \S5), "
        r"so $\varphi_{t|t-1}$ is used directly with no additional link."
    )
    parts.append(r"\section{Magnitude model: phi-only GAS(1,1)}")
    parts.append(
        r"The transformed scale state evolves as "
        r"$$\varphi_{t+1|t} = \omega_\varphi + A_\varphi\,\tilde s_{\varphi,t} "
        r"+ B_\varphi\,\varphi_{t|t-1},$$ "
        r"where $\tilde s_{\varphi,t}$ is the scaled score of the GB2 log-density with "
        r"respect to $\varphi$ (docs/MODELS.md \S5, \S10). $\xi,\gamma,\zeta$ are "
        r"static. The model is estimated by unbounded BFGS with the framework's "
        r"standard soft penalties (models/za\_gas\_model.py), jointly with the "
        r"default AR-logistic occurrence term, exactly as a Stage 1 phi-only baseline "
        r"is fit; only $\varphi_{t|t-1}$ and the static $(\xi,\gamma,\zeta)$ are then "
        r"carried forward as the frozen magnitude path."
    )
    parts.append(r"\section{The four occurrence-dynamics alternatives}")
    items = []
    for vk in VARIANT_ORDER:
        items.append(rf"\item \textbf{{{_tex(VARIANT_LABEL[vk])}}}: $${VARIANT_FORMULA[vk]}$$")
    parts.append(r"\begin{enumerate}" + "\n".join(items) + r"\end{enumerate}")
    parts.append(
        r"For Model D, $\varphi_{t|t-1}$ is the frozen GAS(1,1) scale path; for $t=0$ "
        r"(before the single-lag filter's first update) it is held at the fitted "
        r"initial state $f_{0,\varphi}$, matching the GAS filter's own warm-up "
        r"convention (models/gas\_filter.py)."
    )

    parts.append(r"\chapter{Experimental Design}")
    parts.append(r"\label{sec:experimental-design}")
    parts.append(
        r"\textbf{Locations.} Belo Horizonte, Cruzeiro do Sul, Darwin Airport, "
        r"Garanhuns, Manaus, and Salvador."
    )
    parts.append(
        r"\textbf{Magnitude estimation and score-scaling selection.} The phi-only "
        r"GAS(1,1) is estimated under all three score scalings (unit, "
        r"diagonal inverse Fisher, full inverse Fisher; docs/MODELS.md \S10). The "
        r"scaling frozen as the magnitude path is chosen by: (i) a gradient screen "
        r"using the independent central-difference gradient "
        r"(\texttt{numerical\_gradient\_checked}, diagnostics/numerical\_diff.py) of "
        r"the exact penalised negative log-likelihood at the optimum -- most-preferred "
        r"non-empty tier among "
        r"$\{\lVert\nabla\rVert_\infty<1\}$, $\{\lVert\nabla\rVert_\infty<100\}$, "
        r"$\{\text{not failed}\}$, $\{\text{any}\}$; then (ii) lowest AIC within that "
        r"tier. All three candidate fits are saved."
    )
    parts.append(
        r"\textbf{Occurrence estimation.} Each of the four $\eta_t$ specifications is "
        r"fit by unbounded BFGS (with an L-BFGS-B warm start) maximising the Bernoulli "
        r"log-likelihood of $1(y_t>0)$ alone (pi\_dynamics/standalone\_fit.py). The "
        r"zero-augmented log-likelihood is additively separable in $\pi_t$ vs. the "
        r"positive-part parameters, so this reaches the same $\theta_\pi$ optimum as a "
        r"joint fit."
    )
    parts.append(
        r"\textbf{Common evaluation window.} All diagnostics use $t\in[1, T)$ "
        r"(eff\_start $=1$, the GAS(1,1) effective-sample start). This is common to "
        r"all four variants."
    )
    parts.append(
        r"\textbf{Occurrence PIT.} $y_t>0$ is a Bernoulli$(\pi_t)$ indicator with a "
        r"two-point step CDF, so PIT values are randomised "
        r"(diagnostics/occurrence\_pit.py): $U_t\sim\text{Uniform}(0,1-\pi_t)$ if "
        r"$y_t=0$, $U_t\sim\text{Uniform}(1-\pi_t,1)$ if $y_t>0$. Under correct "
        r"specification $U_t\sim\text{iid Uniform}(0,1)$ and "
        r"$r_t=\Phi^{-1}(U_t)\sim\text{iid }N(0,1)$ (Dunn \& Smyth, 1996), used for "
        r"the ACF diagnostics."
    )
    parts.append(
        r"\textbf{RMSE / CRPS against $y_t$.} Each variant's $\pi_t$ is combined with "
        r"the SAME frozen $(\varphi_{t|t-1},\xi,\gamma,\zeta)$ path from the selected "
        r"GAS(1,1) fit (diagnostics/pi\_dynamics\_eval.py). The point forecast is "
        r"$\mu_t=\pi_t\,E[G_t]$; CRPS uses the framework's Monte Carlo energy-score "
        r"estimator (diagnostics/scoring.py), applied here in-sample."
    )
    parts.append(
        r"\textbf{Gradient table.} For the GAS(1,1) magnitude model and each of the "
        r"four standalone $\pi_t$ fits, the per-parameter gradient of that model's own "
        r"exact objective at its fitted $\hat\theta$ is reported "
        r"(\texttt{numerical\_gradient\_checked}). At a clean interior optimum every "
        r"entry is near zero."
    )

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

    parts.append(build_discussion(all_loaded))
    parts.append(r"\label{sec:discussion}")

    parts.append(r"\chapter{Conclusion}")
    parts.append(
        r"This report repeated the four-way occurrence-dynamics comparison against a "
        r"common frozen magnitude model, in-sample, at six locations, with the "
        r"magnitude model now a parsimonious phi-only GAS(1,1). The cross-location "
        r"tally in Section~\ref{sec:discussion} identifies which occurrence "
        r"specification most consistently improves dry/wet separation, CRPS, and RMSE "
        r"under this magnitude model; the per-location gradient tables document the "
        r"numerical quality of every fit involved. As elsewhere in this framework, "
        r"adopting an alternative occurrence model in the main pipeline is a separate "
        r"decision, to be made only after also considering out-of-sample performance "
        r"(out of scope here)."
    )

    parts.append(r"\chapter*{References}")
    parts.append(r"\begin{itemize}")
    parts.append(r"\item Creal, D., Koopman, S.J., \& Lucas, A. (2013). Generalized "
                 r"autoregressive score models with applications. \textit{Journal of "
                 r"Applied Econometrics}, 28(5), 777--795.")
    parts.append(r"\item Dunn, P.K., \& Smyth, G.K. (1996). Randomized quantile "
                 r"residuals. \textit{Journal of Computational and Graphical "
                 r"Statistics}, 5(3), 236--244.")
    parts.append(r"\item Harvey, A.C. (2013). \textit{Dynamic Models for Volatility "
                 r"and Heavy Tails}. Cambridge University Press.")
    parts.append(r"\item Gneiting, T., \& Raftery, A.E. (2007). Strictly proper "
                 r"scoring rules, prediction, and estimation. \textit{JASA}, "
                 r"102(477), 359--378.")
    parts.append(r"\end{itemize}")

    parts.append(r"\end{document}")
    return "\n\n".join(parts)


# -----------------------------------------------------------------------------
# PDF compilation
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the pi-dynamics GAS(1,1) report.")
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
        n_ok = sum(1 for vk in VARIANT_ORDER if loaded[vk] is not None)
        has_mag = loaded.get("_magnitude") is not None
        print(f"  {n_ok}/4 variants available; magnitude model: {'yes' if has_mag else 'MISSING'}")
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
