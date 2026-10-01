"""
Seasonal-lag comparison report generator (2026-08-31).

=============================================================================
OVERVIEW
=============================================================================

Builds the LaTeX/PDF report comparing out-of-sample RMSE/CRPS between the
short-memory GAS lag set L_short={1,2,3} and the seasonal GAS lag set
L_seasonal={1,2,3,364,365,366,367} (MODELS.md §14), for every already-fitted
GAS model that offers both variants: Stage 1 (baseline, no covariates,
3 score-scaling variants) and Stage 2 (dew point weather covariate), each
separately for the 1-time-varying-parameter (phi only) and
2-time-varying-parameter (phi, xi) branches, at all six locations.

Per docs/REPORTING.md §2-3 and docs/EXECUTION.md §22, this script performs
NO optimisation and NO model fitting: every number comes from metadata.json
files already saved by run_all_locations.py's Stage 1-2 pipeline runs. This
report can be generated immediately -- it requires no new experiment run.

=============================================================================
CODE WALKTHROUGH
=============================================================================

    main()
      for each of the 6 locations:
        Stage 1: for tv in (phi, phixi), for scaling in (unit, diagfi, fullfi):
            load stage1_{tv}_{short,seasonal}_{scaling}/metadata.json
            -> RMSE, CRPS, delta (seasonal - short)
        Stage 2: for tv in (phi, phixi):
            load stage2[_phi]_dewpoint_{short,seasonal}/metadata.json
            -> RMSE, CRPS, delta
        build one Stage-1 table + one Stage-2 table per location
        build one grouped-bar figure per location (CRPS short vs seasonal)
      |
    assemble report.tex: Executive Summary -> Introduction -> Model
        Specification -> Experimental Design -> Results (per location) ->
        Discussion (data-driven tally: does seasonal help, does the answer
        depend on 1 vs 2 time-varying parameters, is it consistent across
        locations/scalings) -> Conclusion -> References
      |
    compile report.pdf (pdflatex, two passes)

=============================================================================
USAGE
=============================================================================

    python generate_seasonality_experiment_report.py [--no-compile]

Output:
    reports/multi_location/seasonality experiment/report.tex
    reports/multi_location/seasonality experiment/report.pdf
    reports/multi_location/seasonality experiment/figures/<run_id>/*.png
    reports/multi_location/seasonality experiment/tables/<run_id>/*.csv, *.tex

To recompile the PDF manually after editing report.tex (no AI needed):
    cd "reports/multi_location/seasonality experiment"
    pdflatex -interaction=nonstopmode report.tex
    pdflatex -interaction=nonstopmode report.tex
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
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

from data.station_loader import STATION_REGISTRY

ARTIFACTS_DIR = ROOT / "artifacts"
REPORT_DIR    = ROOT / "reports" / "multi_location" / "seasonality experiment"
FIG_DIR       = REPORT_DIR / "figures"
TAB_DIR       = REPORT_DIR / "tables"

LOCATIONS = [
    "BELO HORIZONTE",
    "CRUZEIRO DO SUL (ACRE)",
    "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)",
    "MANAUS",
    "SALVADOR",
]

TV_SETS   = ["phi", "phixi"]
TV_LABEL  = {"phi": "1 time-varying parameter ($\\varphi$ only)", "phixi": "2 time-varying parameters ($\\varphi,\\xi$)"}
LAG_SETS  = ["short", "seasonal"]
SCALINGS  = ["unit", "diagfi", "fullfi"]
SCALING_LABEL = {"unit": "unit", "diagfi": "diagonal inverse FI", "fullfi": "full inverse FI"}


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX helpers (self-contained, mirrors generate_pi_dynamics_report.py)
# ─────────────────────────────────────────────────────────────────────────────

def _tex(s) -> str:
    return (
        str(s)
        .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
        .replace("#", r"\#").replace("_", r"\_")
        .replace("{", r"\{").replace("}", r"\}")
        .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    )


def _fmt(v, digits: int = 4) -> str:
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
    cols = "l" + "r" * len(df.columns)
    header = " & ".join([""] + [c for c in df.columns]) + r" \\"
    rows = []
    for idx, row in df.iterrows():
        rows.append(" & ".join([str(idx)] + [str(v) for v in row]) + r" \\")
    body = "\n".join(rows)
    return "\n".join([
        r"\begin{table}[H]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}", r"\toprule", header, r"\midrule", body, r"\bottomrule",
        r"\end{tabular}", r"\end{table}",
    ])


LATEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{report}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=2.3cm]{geometry}
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


def compile_latex(tex_path: Path) -> bool:
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
# Artifact loading (metadata.json only -- no fitting, no CSV re-derivation)
# ─────────────────────────────────────────────────────────────────────────────

def _load_metric(run_dir: Path, stage_subdir: str, model_id: str) -> dict:
    meta_path = run_dir / stage_subdir / model_id / "metadata.json"
    if not meta_path.exists():
        return {"rmse": np.nan, "crps_mean": np.nan, "validity": "missing", "loglik": np.nan}
    meta = json.loads(meta_path.read_text())
    return {
        "rmse": meta.get("rmse", np.nan),
        "crps_mean": meta.get("crps_mean", np.nan),
        "validity": meta.get("validity", "--"),
        "loglik": meta.get("loglik", np.nan),
    }


def stage1_frame(run_dir: Path) -> pd.DataFrame:
    """One row per (tv, scaling); short/seasonal RMSE+CRPS and deltas."""
    rows = []
    for tv in TV_SETS:
        for scaling in SCALINGS:
            short_id    = f"stage1_{tv}_short_{scaling}"
            seasonal_id = f"stage1_{tv}_seasonal_{scaling}"
            s = _load_metric(run_dir, "stage1", short_id)
            m = _load_metric(run_dir, "stage1", seasonal_id)
            rows.append({
                "tv": tv, "scaling": scaling,
                "rmse_short": s["rmse"], "rmse_seasonal": m["rmse"],
                "rmse_delta": m["rmse"] - s["rmse"],
                "crps_short": s["crps_mean"], "crps_seasonal": m["crps_mean"],
                "crps_delta": m["crps_mean"] - s["crps_mean"],
                "validity_short": s["validity"], "validity_seasonal": m["validity"],
            })
    return pd.DataFrame(rows)


def stage2_frame(run_dir: Path) -> pd.DataFrame:
    """One row per tv; dew point short vs seasonal RMSE+CRPS and deltas."""
    rows = []
    for tv in TV_SETS:
        prefix = "stage2_phi_" if tv == "phi" else "stage2_"
        subdir = "stage2_phi" if tv == "phi" else "stage2"
        short_id    = f"{prefix}dewpoint_short"
        seasonal_id = f"{prefix}dewpoint_seasonal"
        s = _load_metric(run_dir, subdir, short_id)
        m = _load_metric(run_dir, subdir, seasonal_id)
        rows.append({
            "tv": tv,
            "rmse_short": s["rmse"], "rmse_seasonal": m["rmse"], "rmse_delta": m["rmse"] - s["rmse"],
            "crps_short": s["crps_mean"], "crps_seasonal": m["crps_mean"], "crps_delta": m["crps_mean"] - s["crps_mean"],
            "validity_short": s["validity"], "validity_seasonal": m["validity"],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

def save_table(run_id: str, name: str, df: pd.DataFrame, caption: str, label: str) -> str:
    loc_tab_dir = TAB_DIR / run_id
    loc_tab_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(loc_tab_dir / f"{name}.csv", index=False)

    display = df.copy().astype(object)
    for col in display.columns:
        display[col] = df[col].apply(_fmt) if df[col].dtype != object else df[col].apply(_tex)
    tex_df = display.drop(columns=[c for c in display.columns if c in ("tv", "scaling")], errors="ignore")
    idx = df.apply(
        lambda r: (f"{TV_LABEL[r['tv']].split('(')[0].strip()}, {SCALING_LABEL.get(r.get('scaling',''), '')}"
                   if "scaling" in df.columns else TV_LABEL[r["tv"]]),
        axis=1,
    )
    tex_df.index = idx
    tex = _latex_table(tex_df, caption, label)
    (loc_tab_dir / f"{name}.tex").write_text(tex)
    return tex


def stage1_table_tex(run_id: str, display: str, df: pd.DataFrame) -> str:
    show = df.rename(columns={
        "rmse_short": "RMSE short", "rmse_seasonal": "RMSE seasonal", "rmse_delta": "$\\Delta$RMSE",
        "crps_short": "CRPS short", "crps_seasonal": "CRPS seasonal", "crps_delta": "$\\Delta$CRPS",
    })[["tv", "scaling", "RMSE short", "RMSE seasonal", "$\\Delta$RMSE", "CRPS short", "CRPS seasonal", "$\\Delta$CRPS"]]
    return save_table(
        run_id, "stage1_seasonality", show,
        caption=(f"{_tex(display)}, Stage 1 (no covariates): out-of-sample RMSE and mean CRPS "
                 f"comparing the short GAS lag set $\\mathcal{{L}}_{{short}}=\\{{1,2,3\\}}$ against "
                 f"the seasonal lag set $\\mathcal{{L}}_{{seasonal}}=\\{{1,2,3,364,\\ldots,367\\}}$ "
                 f"(MODELS.md \\S14), for each score-scaling method and time-varying-parameter set. "
                 f"$\\Delta = $ seasonal $-$ short; negative is better for seasonal. Lower is better "
                 f"for RMSE/CRPS themselves."),
        label=f"tab:{run_id}-stage1-seasonality",
    )


def stage2_table_tex(run_id: str, display: str, df: pd.DataFrame) -> str:
    show = df.rename(columns={
        "rmse_short": "RMSE short", "rmse_seasonal": "RMSE seasonal", "rmse_delta": "$\\Delta$RMSE",
        "crps_short": "CRPS short", "crps_seasonal": "CRPS seasonal", "crps_delta": "$\\Delta$CRPS",
    })[["tv", "RMSE short", "RMSE seasonal", "$\\Delta$RMSE", "CRPS short", "CRPS seasonal", "$\\Delta$CRPS"]]
    return save_table(
        run_id, "stage2_seasonality", show,
        caption=(f"{_tex(display)}, Stage 2 (dew point weather covariate): out-of-sample RMSE and "
                 f"mean CRPS comparing $X^{{D,short}}_t$ against $X^{{D,seasonal}}_t$ "
                 f"(MODELS.md \\S15), at this location's best-by-CRPS Stage-1 score scaling. "
                 f"$\\Delta = $ seasonal $-$ short."),
        label=f"tab:{run_id}-stage2-seasonality",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def crps_bar_figure(run_id: str, display: str, s1: pd.DataFrame, s2: pd.DataFrame) -> Optional[str]:
    labels, short_vals, seasonal_vals = [], [], []
    for _, r in s1.iterrows():
        labels.append(f"S1 {r['tv']}/{r['scaling']}")
        short_vals.append(r["crps_short"]); seasonal_vals.append(r["crps_seasonal"])
    for _, r in s2.iterrows():
        labels.append(f"S2 {r['tv']}/dewpt")
        short_vals.append(r["crps_short"]); seasonal_vals.append(r["crps_seasonal"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels)), 4.5))
    w = 0.38
    ax.bar(x - w/2, short_vals, width=w, label="short lags", color="steelblue")
    ax.bar(x + w/2, seasonal_vals, width=w, label="seasonal lags", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Out-of-sample mean CRPS")
    ax.set_title(f"{display}: short vs. seasonal lags (lower is better)")
    ax.legend()
    fig.tight_layout()

    if not np.any(np.isfinite(short_vals)) and not np.any(np.isfinite(seasonal_vals)):
        plt.close(fig)
        return None

    loc_fig_dir = FIG_DIR / run_id
    loc_fig_dir.mkdir(parents=True, exist_ok=True)
    path = loc_fig_dir / "crps_short_vs_seasonal.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.relative_to(REPORT_DIR).as_posix()


# ─────────────────────────────────────────────────────────────────────────────
# Per-location section
# ─────────────────────────────────────────────────────────────────────────────

def location_section(location: str) -> str:
    cfg = STATION_REGISTRY[location]
    run_id, display = cfg["run_id"], cfg.get("display", location)
    run_dir = ARTIFACTS_DIR / run_id

    s1 = stage1_frame(run_dir)
    s2 = stage2_frame(run_dir)

    parts = [rf"\section{{{_tex(display)}}}"]
    parts.append(stage1_table_tex(run_id, display, s1))
    parts.append(stage2_table_tex(run_id, display, s2))

    fig_path = crps_bar_figure(run_id, display, s1, s2)
    if fig_path:
        parts.append("\n".join([
            r"\begin{figure}[H]", r"\centering",
            rf"\includegraphics[width=\textwidth]{{{fig_path}}}",
            rf"\caption{{{_tex(display)}: out-of-sample mean CRPS, short vs.\ seasonal lags, "
            r"for every Stage 1 (S1, no covariates, three score-scaling variants) and "
            r"Stage 2 (S2, dew point weather covariate) model at this location, split "
            r"by 1 vs.\ 2 time-varying parameters. Bars below their short-lag counterpart "
            r"indicate seasonal lags improved calibration.}",
            rf"\label{{fig:{run_id}-crps-bars}}",
            r"\end{figure}",
        ]))

    n_seasonal_better_crps = int((s1["crps_delta"] < 0).sum() + (s2["crps_delta"] < 0).sum())
    n_total = len(s1) + len(s2)
    parts.append(
        f"At {_tex(display)}, seasonal lags achieve a lower out-of-sample CRPS than the "
        f"matched short-lag model in {n_seasonal_better_crps}/{n_total} comparisons "
        f"(Stage 1 across three score-scaling methods and both time-varying-parameter "
        f"sets, plus Stage 2 for both time-varying-parameter sets)."
    )
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Discussion (data-driven)
# ─────────────────────────────────────────────────────────────────────────────

def build_discussion() -> str:
    all_s1: List[pd.DataFrame] = []
    all_s2: List[pd.DataFrame] = []
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        run_dir = ARTIFACTS_DIR / cfg["run_id"]
        s1 = stage1_frame(run_dir); s1["location"] = cfg.get("display", location)
        s2 = stage2_frame(run_dir); s2["location"] = cfg.get("display", location)
        all_s1.append(s1); all_s2.append(s2)
    s1_all = pd.concat(all_s1, ignore_index=True)
    s2_all = pd.concat(all_s2, ignore_index=True)

    def _tally(df: pd.DataFrame, tv: str) -> dict:
        sub = df[df["tv"] == tv]
        valid = sub[np.isfinite(sub["crps_delta"])]
        return {
            "n": len(valid),
            "n_seasonal_better": int((valid["crps_delta"] < 0).sum()),
            "mean_delta": float(valid["crps_delta"].mean()) if len(valid) else np.nan,
        }

    tally = {
        "Stage 1, $\\varphi$ only":  _tally(s1_all, "phi"),
        "Stage 1, $\\varphi,\\xi$":  _tally(s1_all, "phixi"),
        "Stage 2, $\\varphi$ only":  _tally(s2_all, "phi"),
        "Stage 2, $\\varphi,\\xi$":  _tally(s2_all, "phixi"),
    }
    tally_df = pd.DataFrame({
        k: {"Comparisons": v["n"], "Seasonal lower CRPS": v["n_seasonal_better"],
            "Mean $\\Delta$CRPS": _fmt(v["mean_delta"], 4)}
        for k, v in tally.items()
    }).T
    tex_tally = _latex_table(
        tally_df.astype(str),
        caption="Across all six locations: how often the seasonal lag set achieves lower out-of-sample "
                "CRPS than the matched short-lag model, and the mean CRPS difference (seasonal $-$ short; "
                "negative favours seasonal).",
        label="tab:cross-location-seasonality-tally",
    )
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    tally_df.to_csv(TAB_DIR / "cross_location_tally.csv")

    phi_frac  = (tally["Stage 1, $\\varphi$ only"]["n_seasonal_better"] / tally["Stage 1, $\\varphi$ only"]["n"]
                 if tally["Stage 1, $\\varphi$ only"]["n"] else np.nan)
    phixi_frac = (tally["Stage 1, $\\varphi,\\xi$"]["n_seasonal_better"] / tally["Stage 1, $\\varphi,\\xi$"]["n"]
                  if tally["Stage 1, $\\varphi,\\xi$"]["n"] else np.nan)

    text = [
        r"\chapter{Discussion}",
        r"\label{sec:discussion}",
        (r"Table~\ref{tab:cross-location-seasonality-tally} pools every Stage 1 and Stage 2 "
         r"short-vs-seasonal comparison across all six locations (Stage 1: 3 score-scaling "
         r"variants $\times$ 6 locations $=$ 18 comparisons per time-varying-parameter set; "
         r"Stage 2: 1 comparison $\times$ 6 locations $=$ 6 per set)."),
        tex_tally,
    ]

    if np.isfinite(phi_frac) and np.isfinite(phixi_frac):
        if abs(phi_frac - phixi_frac) < 0.15:
            comp = "a broadly similar rate to"
        elif phi_frac > phixi_frac:
            comp = "a higher rate than"
        else:
            comp = "a lower rate than"
        text.append(
            rf"In Stage 1, seasonal lags improve out-of-sample CRPS in "
            rf"{tally['Stage 1, $\\varphi$ only']['n_seasonal_better']}/{tally['Stage 1, $\\varphi$ only']['n']} "
            rf"comparisons ({_fmt(phi_frac,2)}) when only $\varphi$ is time-varying, "
            rf"{comp} the "
            rf"{tally['Stage 1, $\\varphi,\\xi$']['n_seasonal_better']}/{tally['Stage 1, $\\varphi,\\xi$']['n']} "
            rf"({_fmt(phixi_frac,2)}) seen when both $\varphi$ and $\xi$ are time-varying. "
            rf"This bears on MODELS.md \S14's scientific question -- whether annual "
            rf"recurrence improves score-driven dynamics beyond short memory -- and "
            rf"suggests the answer is {'not strongly moderated' if abs(phi_frac-phixi_frac)<0.15 else 'moderated'} "
            rf"by how many parameters are allowed to vary over time."
        )

    text.append(
        r"The per-location tables and figures in Section~\ref{sec:results} show that this "
        r"aggregate pattern is not necessarily uniform: at any single location the seasonal "
        r"lag set can still lose on RMSE while winning on CRPS (or vice versa), consistent "
        r"with MODELS.md \S27's guidance to prioritise CRPS and distributional calibration "
        r"over RMSE alone when selecting a winning specification."
    )

    return "\n\n".join(text)


# ─────────────────────────────────────────────────────────────────────────────
# Full document assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_document(generation_date: str) -> str:
    parts = [
        LATEX_PREAMBLE,
        r"\begin{document}",
        r"\title{Seasonal vs.\ Short GAS Lags: A Cross-Location Out-of-Sample Comparison}",
        rf"\author{{Generated: {_tex(generation_date)}}}",
        r"\date{}", r"\maketitle", r"\tableofcontents", r"\clearpage",
    ]

    parts.append(r"\chapter{Executive Summary}")
    parts.append(
        r"MODELS.md \S14 poses a specific scientific question for the baseline GAS "
        r"model: does extending the short GAS memory "
        r"$\mathcal{L}_{short}=\{1,2,3\}$ with same-calendar-day lags one year prior, "
        r"$\mathcal{L}_{seasonal}=\{1,2,3,364,365,366,367\}$, improve the model beyond "
        r"short memory alone? This report answers it purely from already-completed "
        r"artifacts -- no model is re-fit here -- by comparing out-of-sample RMSE and "
        r"mean CRPS between the short and seasonal variants of every Stage 1 "
        r"(no-covariate baseline, three score-scaling methods) and Stage 2 (dew point "
        r"weather covariate) model already estimated at all six locations, separately "
        r"for the 1-time-varying-parameter ($\varphi$ only) and "
        r"2-time-varying-parameter ($\varphi,\xi$) branches."
    )

    parts.append(r"\chapter{Introduction}")
    parts.append(
        r"Every location in this framework's multi-location report was fit twice at "
        r"Stage 1 for each score-scaling method and time-varying-parameter set -- once "
        r"with $\mathcal{L}_{short}$, once with $\mathcal{L}_{seasonal}$ -- and the "
        r"winning lag set carried forward into Stage 2's dew point covariate blocks, "
        r"which were themselves also fit in matched short/seasonal pairs "
        r"(MODELS.md \S15). This report collects every one of those pairs into a single "
        r"cross-location comparison that was not previously assembled."
    )

    parts.append(r"\chapter{Model Specification}")
    parts.append(r"\label{sec:model-spec}")
    parts.append(
        r"For each transformed dynamic parameter $f_{j,t}$ (MODELS.md \S12): "
        r"$$f_{j,t+1}=\omega_j+\sum_{\ell\in\mathcal{L}}A_{j,\ell}s_{j,t-\ell+1}+"
        r"\sum_{\ell\in\mathcal{L}}B_{j,\ell}f_{j,t-\ell+1},$$ "
        r"with $\mathcal{L}=\mathcal{L}_{short}$ or $\mathcal{L}_{seasonal}$ as defined "
        r"above. Stage 2 adds a covariate term $\Gamma_j'X_t$ with $X_t$ the dew point "
        r"short block $X^{D,short}_t=(D_{t-1},D_{t-2},D_{t-3})'$ or seasonal block "
        r"$X^{D,seasonal}_t$ (MODELS.md \S15)."
    )

    parts.append(r"\chapter{Experimental Design}")
    parts.append(r"\label{sec:experimental-design}")
    parts.append(
        r"\textbf{No new fitting.} Every number in this report is read from "
        r"\texttt{metadata.json} files already written by "
        r"\texttt{run\_all\_locations.py}'s Stage 1-2 pipeline "
        r"(\texttt{artifacts/<run\_id>/stage1/*}, \texttt{stage2[\_phi]/*}). "
        r"Out-of-sample RMSE and mean CRPS are the same metrics reported throughout "
        r"this framework's other reports (diagnostics/scoring.py)."
    )
    parts.append(
        r"\textbf{Locations.} The same six locations used throughout this framework's "
        r"multi-location reports: Belo Horizonte, Cruzeiro do Sul, Darwin Airport, "
        r"Garanhuns, Manaus, and Salvador."
    )
    parts.append(
        r"\textbf{Comparisons.} Stage 1: short vs.\ seasonal, crossed with 3 score-scaling "
        r"methods (unit, diagonal inverse Fisher, full inverse Fisher) and 2 "
        r"time-varying-parameter sets, giving 6 matched pairs per location. Stage 2: "
        r"short vs.\ seasonal dew point, crossed with the 2 time-varying-parameter sets "
        r"(at this location's own best-by-CRPS Stage-1 scaling), giving 2 matched pairs "
        r"per location."
    )

    parts.append(r"\chapter{Results}")
    parts.append(r"\label{sec:results}")
    for location in LOCATIONS:
        parts.append(location_section(location))

    parts.append(build_discussion())

    parts.append(r"\chapter{Conclusion}")
    parts.append(
        r"This report assembled, for the first time in one place, every short-vs-seasonal "
        r"GAS lag comparison already available across the six-location pipeline. "
        r"Section~\ref{sec:discussion}'s pooled tally answers MODELS.md \S14's scientific "
        r"question in aggregate; the per-location tables and figures in "
        r"Section~\ref{sec:results} let that answer be checked against any individual "
        r"location's climate. Because every number here is read from existing artifacts, "
        r"this report can be regenerated at any time without re-running any model."
    )

    parts.append(r"\chapter*{References}")
    parts.append(r"\begin{itemize}")
    parts.append(r"\item Creal, D., Koopman, S.J., \& Lucas, A. (2013). Generalized "
                 r"autoregressive score models with applications. \textit{Journal of "
                 r"Applied Econometrics}, 28(5), 777--795.")
    parts.append(r"\item Gneiting, T., \& Raftery, A.E. (2007). Strictly proper "
                 r"scoring rules, prediction, and estimation. \textit{JASA}, 102(477), 359--378.")
    parts.append(r"\end{itemize}")

    parts.append(r"\end{document}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the seasonality (short vs. seasonal lags) report.")
    parser.add_argument("--no-compile", action="store_true", help="Skip pdflatex compilation.")
    args = parser.parse_args()

    generation_date = datetime.date.today().isoformat()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    doc = build_document(generation_date)

    tex_path = REPORT_DIR / "report.tex"
    tex_path.write_text(doc, encoding="utf-8")
    print(f"Wrote {tex_path}")

    if args.no_compile:
        print("Skipping PDF compilation (--no-compile).")
        return

    ok = compile_latex(tex_path)
    pdf_path = REPORT_DIR / "report.pdf"
    if ok and pdf_path.exists():
        print(f"Wrote {pdf_path}")
    else:
        print("PDF compilation did not succeed -- see log above. The .tex file is still valid "
              "and can be compiled manually (see module docstring).")


if __name__ == "__main__":
    main()
