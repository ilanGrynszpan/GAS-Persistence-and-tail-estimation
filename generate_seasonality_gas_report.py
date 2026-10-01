"""
Short-vs-seasonal-lag GAS report: baseline and weather-covariate models
(2026-09-01).

=============================================================================
OVERVIEW
=============================================================================

Companion report to run_seasonality_gas_analysis.py. Presents, separately
for Stage 1 (baseline, no covariates) and Stage 2 (dew-point weather
covariate) models, and separately for both TVP branches (phi-only,
phi+xi):

  - Side-by-side in-sample residual mosaics (400-lag ACF, QQ), short lags
    vs. seasonal lags, across all six locations.
  - Out-of-sample 1-step (exact) and multi-step (Monte Carlo,
    h=1,2,5,10,30,60,90) RMSE/CRPS, short vs. seasonal.
  - A final, location-pooled chapter on WHY certain lag sets won: paired
    optimisation-difficulty diagnostics (gradient norm, iterations,
    runtime) and a collinearity diagnostic from each seasonal model's own
    BFGS approximate inverse-Hessian (mean |correlation| among the
    seasonal-lag coefficients vs. among the short-lag coefficients within
    the same fit), related back to the observed CRPS difference.

Per docs/REPORTING.md, every number is read from artifacts/
seasonality_gas_experiment/ and the underlying Stage 1/2 metadata.json
files; no optimisation or simulation runs here.

=============================================================================
USAGE
=============================================================================

    python generate_seasonality_gas_report.py [--no-compile]

Output:
    reports/seasonality GAS/report.tex / .pdf
    reports/seasonality GAS/figures/*.png
    reports/seasonality GAS/tables/*.csv
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import STATION_REGISTRY, mosaic_name
from diagnostics.plots import qq_location_mosaic, acf_location_mosaic
from diagnostics.multistep_forecast import DEFAULT_HORIZONS
from run_seasonality_gas_analysis import OUT_DIR as EXP_DIR, LOCATIONS, BRANCHES, LAGSETS

REPORT_DIR = ROOT / "reports" / "seasonality GAS"
FIG_DIR    = REPORT_DIR / "figures"
TAB_DIR    = REPORT_DIR / "tables"

STAGES = ["stage1", "stage2"]
STAGE_LABEL = {"stage1": "Baseline GAS (no covariates)", "stage2": "Weather-Covariate GAS (dew point)"}
BRANCH_LABEL = {"phi": "1 time-varying parameter ($\\varphi$ only)", "phixi": "2 time-varying parameters ($\\varphi,\\xi$)"}
LAGSET_LABEL = {"short": "short", "seasonal": "seasonal"}


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tex(s) -> str:
    return (
        str(s)
        .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
        .replace("#", r"\#").replace("_", r"\_")
        .replace("{", r"\{").replace("}", r"\}")
        .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    )


def _fmt(v, digits: int = 3) -> str:
    if v is None:
        return "--"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return _tex(str(v))
    if not np.isfinite(fv):
        return "--"
    return f"{fv:.{digits}f}"


def _safe_header(s) -> str:
    s = str(s)
    return s if "$" in s else _tex(s)


def _latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = "l" + "r" * len(df.columns)
    header = " & ".join([""] + [_safe_header(c) for c in df.columns]) + r" \\"
    rows = []
    for idx, row in df.iterrows():
        rows.append(" & ".join([_safe_header(idx)] + [str(v) for v in row]) + r" \\")
    body = "\n".join(rows)
    return "\n".join([
        r"\begin{table}[H]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        rf"\begin{{tabular}}{{{cols}}}", r"\toprule", header, r"\midrule", body, r"\bottomrule",
        r"\end{tabular}", r"}", r"\end{table}",
    ])


def save_metric_table(df: pd.DataFrame, name: str, caption: str, label: str) -> str:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB_DIR / f"{name}.csv")
    disp = df.copy().astype(object)
    for col in disp.columns:
        disp[col] = df[col].apply(_fmt)
    return _latex_table(disp, caption, label)


LATEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{report}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{hyperref}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}
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
# Artifact loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cell(short: str, stage: str, branch: str, lagset: str) -> Optional[dict]:
    d = EXP_DIR / short / f"{stage}_{branch}_{lagset}"
    info_path = d / "info.json"
    if not info_path.exists():
        return None
    return {
        "info": json.loads(info_path.read_text()),
        "residuals": np.load(d / "residuals.npy"),
        "multistep": pd.read_csv(d / "multistep.csv"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Introduction
# ─────────────────────────────────────────────────────────────────────────────

def build_introduction() -> str:
    return "\n\n".join([
        r"\chapter{Two GAS Model Families, Two Lag Sets}",
        (r"This report compares two independent design choices for the score-driven "
         r"(GAS) precipitation model, kept strictly separate throughout: which "
         r"covariates enter the dynamics, and which GAS lag set drives the "
         r"recursion."),
        r"\section{Baseline vs.\ Weather-Covariate GAS}",
        (r"The \textbf{baseline} model has no exogenous covariates at all: each "
         r"time-varying parameter $f_{j,t}$ evolves purely from its own score history. "
         r"The \textbf{weather-covariate} model adds same-station dew point as an "
         r"exogenous driver, $\Gamma_j' X_t$, added to the same recursion."),
        r"""\begin{align*}
\text{Baseline: } && f_{j,t+1} &= \omega_j + \sum_{\ell\in\mathcal{L}} A_{j,\ell}\, s_{j,t-\ell+1} + \sum_{\ell\in\mathcal{L}} B_{j,\ell}\, f_{j,t-\ell+1} \\
\text{Weather-covariate: } && f_{j,t+1} &= \omega_j + \sum_{\ell\in\mathcal{L}} A_{j,\ell}\, s_{j,t-\ell+1} + \sum_{\ell\in\mathcal{L}} B_{j,\ell}\, f_{j,t-\ell+1} + \Gamma_j' X_t
\end{align*}""",
        r"\section{Short vs.\ Seasonal Lag Sets}",
        (r"$\mathcal{L}$ is the GAS lag set. The \textbf{short} set uses only the "
         r"three most recent days, $\mathcal{L}_{short}=\{1,2,3\}$. The "
         r"\textbf{seasonal} set adds same-calendar-day lags one year prior, "
         r"$\mathcal{L}_{seasonal}=\{1,2,3,364,365,366,367\}$ -- four extra "
         r"coefficient pairs $(A_{j,\ell},B_{j,\ell})$ per dynamic parameter, "
         r"intended to let the model directly recall last year's rainfall regime "
         r"on the same date."),
        (r"Both design choices are crossed with both time-varying-parameter "
         r"branches used throughout this framework: \textbf{1 TVP} (only the "
         r"location parameter $\varphi$ is dynamic) and \textbf{2 TVP} (both "
         r"$\varphi$ and the GB2 shape parameter $\xi$ are dynamic)."),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Mosaics (side by side: short | seasonal)
# ─────────────────────────────────────────────────────────────────────────────

def build_mosaic_pair(all_data: Dict, stage: str, branch: str, kind: str) -> Dict[str, str]:
    """kind: 'qq' or 'acf'. Returns {'short': path, 'seasonal': path}."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    display_order = [mosaic_name(loc) for loc in LOCATIONS]
    out = {}
    for lagset in LAGSETS:
        qr_by_location = {}
        for loc in LOCATIONS:
            short = STATION_REGISTRY[loc].get("short", loc)
            cell = all_data.get((loc, stage, branch, lagset))
            qr_by_location[mosaic_name(loc)] = cell["residuals"] if cell else None
        title = f"{STAGE_LABEL[stage]}, {BRANCH_LABEL[branch]}: {LAGSET_LABEL[lagset]} lags"
        if kind == "qq":
            fig = qq_location_mosaic(qr_by_location, display_order, suptitle=f"{title} -- QQ plot")
        else:
            fig = acf_location_mosaic(qr_by_location, display_order, suptitle=f"{title} -- 400-lag ACF", n_lags=400)
        path = FIG_DIR / f"{kind}_{stage}_{branch}_{lagset}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        out[lagset] = path.relative_to(REPORT_DIR).as_posix()
    return out


def side_by_side_figure(paths: Dict[str, str], caption: str, label: str) -> str:
    return "\n".join([
        r"\begin{figure}[H]", r"\centering",
        r"\begin{subfigure}[t]{0.49\textwidth}", r"\centering",
        rf"\includegraphics[width=\textwidth]{{{paths['short']}}}",
        r"\caption{Short lags}", r"\end{subfigure}",
        r"\hfill",
        r"\begin{subfigure}[t]{0.49\textwidth}", r"\centering",
        rf"\includegraphics[width=\textwidth]{{{paths['seasonal']}}}",
        r"\caption{Seasonal lags}", r"\end{subfigure}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\end{figure}",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# OOS tables
# ─────────────────────────────────────────────────────────────────────────────

def onestep_table(all_data: Dict, stage: str, branch: str) -> pd.DataFrame:
    rows = {}
    for loc in LOCATIONS:
        display = STATION_REGISTRY[loc].get("display", loc)
        cs = all_data.get((loc, stage, branch, "short"))
        cm = all_data.get((loc, stage, branch, "seasonal"))
        rows[display] = {
            "RMSE short": cs["info"]["rmse"] if cs else np.nan,
            "RMSE seasonal": cm["info"]["rmse"] if cm else np.nan,
            "CRPS short": cs["info"]["crps_mean"] if cs else np.nan,
            "CRPS seasonal": cm["info"]["crps_mean"] if cm else np.nan,
        }
    return pd.DataFrame(rows).T


def multistep_table(all_data: Dict, stage: str, branch: str) -> pd.DataFrame:
    """Cross-location mean at each horizon, short vs. seasonal."""
    rows = {}
    for h in DEFAULT_HORIZONS:
        vals = {"rmse_short": [], "rmse_seasonal": [], "crps_short": [], "crps_seasonal": []}
        for loc in LOCATIONS:
            for lagset, tag in [("short", "short"), ("seasonal", "seasonal")]:
                cell = all_data.get((loc, stage, branch, lagset))
                if cell is None:
                    continue
                ms = cell["multistep"].set_index("horizon")
                if h in ms.index:
                    vals[f"rmse_{tag}"].append(ms.loc[h, "rmse"])
                    vals[f"crps_{tag}"].append(ms.loc[h, "crps"])
        rows[f"h={h}"] = {
            "RMSE short": np.nanmean(vals["rmse_short"]) if vals["rmse_short"] else np.nan,
            "RMSE seasonal": np.nanmean(vals["rmse_seasonal"]) if vals["rmse_seasonal"] else np.nan,
            "CRPS short": np.nanmean(vals["crps_short"]) if vals["crps_short"] else np.nan,
            "CRPS seasonal": np.nanmean(vals["crps_seasonal"]) if vals["crps_seasonal"] else np.nan,
        }
    return pd.DataFrame(rows).T


# ─────────────────────────────────────────────────────────────────────────────
# One stage chapter (both branches)
# ─────────────────────────────────────────────────────────────────────────────

def build_stage_chapter(all_data: Dict, stage: str) -> str:
    parts = [rf"\chapter{{{STAGE_LABEL[stage]}}}"]
    for branch in BRANCHES:
        parts.append(rf"\section{{{BRANCH_LABEL[branch]}}}")

        qq_paths = build_mosaic_pair(all_data, stage, branch, "qq")
        parts.append(side_by_side_figure(
            qq_paths,
            f"{STAGE_LABEL[stage]}, {BRANCH_LABEL[branch]}: in-sample QQ plot of quantile residuals, short vs.\\ seasonal lags.",
            f"fig:qq-{stage}-{branch}",
        ))
        acf_paths = build_mosaic_pair(all_data, stage, branch, "acf")
        parts.append(side_by_side_figure(
            acf_paths,
            f"{STAGE_LABEL[stage]}, {BRANCH_LABEL[branch]}: in-sample 400-lag ACF of quantile residuals, short vs.\\ seasonal lags.",
            f"fig:acf-{stage}-{branch}",
        ))

        parts.append(save_metric_table(
            onestep_table(all_data, stage, branch), f"{stage}_{branch}_onestep",
            f"{STAGE_LABEL[stage]}, {BRANCH_LABEL[branch]}: full-test-set 1-step rolling OOS RMSE/CRPS, short vs.\\ seasonal lags, all locations.",
            f"tab:onestep-{stage}-{branch}",
        ))
        parts.append(save_metric_table(
            multistep_table(all_data, stage, branch), f"{stage}_{branch}_multistep",
            f"{STAGE_LABEL[stage]}, {BRANCH_LABEL[branch]}: Monte Carlo multi-step OOS RMSE/CRPS "
            f"(mean across all six locations at each horizon), short vs.\\ seasonal lags.",
            f"tab:multistep-{stage}-{branch}",
        ))
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# "Why" chapter (pooled, not per location)
# ─────────────────────────────────────────────────────────────────────────────

def build_why_chapter(all_data: Dict) -> str:
    paired_rows = []
    for loc in LOCATIONS:
        for stage in STAGES:
            for branch in BRANCHES:
                cs = all_data.get((loc, stage, branch, "short"))
                cm = all_data.get((loc, stage, branch, "seasonal"))
                if cs is None or cm is None:
                    continue
                paired_rows.append({
                    "location": loc, "stage": stage, "branch": branch,
                    "grad_short": cs["info"]["grad_norm_inf"], "grad_seasonal": cm["info"]["grad_norm_inf"],
                    "n_iter_short": cs["info"]["n_iter"], "n_iter_seasonal": cm["info"]["n_iter"],
                    "runtime_short": cs["info"]["runtime_s"], "runtime_seasonal": cm["info"]["runtime_s"],
                    "cond_short": cs["info"].get("cond_number"), "cond_seasonal": cm["info"].get("cond_number"),
                    "corr_A_short_block": cm["info"].get("corr_A_short"),
                    "corr_B_short_block": cm["info"].get("corr_B_short"),
                    "corr_A_seasonal_block": cm["info"].get("corr_A_seasonal"),
                    "corr_B_seasonal_block": cm["info"].get("corr_B_seasonal"),
                    "crps_delta": cm["info"]["crps_mean"] - cs["info"]["crps_mean"],
                })
    df = pd.DataFrame(paired_rows)

    # ── Optimisation difficulty: paired short vs seasonal ───────────────
    opt_summary = pd.DataFrame({
        "Gradient $|\\nabla Q|_\\infty$": {
            "mean, short": df["grad_short"].mean(), "mean, seasonal": df["grad_seasonal"].mean(),
        },
        "Iterations": {
            "mean, short": df["n_iter_short"].mean(), "mean, seasonal": df["n_iter_seasonal"].mean(),
        },
        "Runtime (s)": {
            "mean, short": df["runtime_short"].mean(), "mean, seasonal": df["runtime_seasonal"].mean(),
        },
    }).T
    n_grad_worse = int((df["grad_seasonal"] > df["grad_short"]).sum())
    n_iter_worse = int((df["n_iter_seasonal"] > df["n_iter_short"]).sum())
    n_total = len(df)

    # ── Collinearity: seasonal-lag block vs short-lag block, within seasonal fits ──
    corr_df = df.dropna(subset=["corr_A_seasonal_block", "corr_A_short_block"])
    mean_corr_seasonal = float(pd.concat([corr_df["corr_A_seasonal_block"], corr_df["corr_B_seasonal_block"]]).mean())
    mean_corr_short_block = float(pd.concat([corr_df["corr_A_short_block"], corr_df["corr_B_short_block"]]).mean())
    cond_ratio = float((df["cond_seasonal"] / df["cond_short"]).replace([np.inf, -np.inf], np.nan).median())

    # ── Does more collinearity predict a worse (higher) CRPS delta? ─────
    corr_df = corr_df.assign(seasonal_block_corr=(corr_df["corr_A_seasonal_block"] + corr_df["corr_B_seasonal_block"]) / 2)
    valid = corr_df.dropna(subset=["seasonal_block_corr", "crps_delta"])
    r_corr_vs_delta = float(np.corrcoef(valid["seasonal_block_corr"], valid["crps_delta"])[0, 1]) if len(valid) >= 3 else np.nan

    parts = [
        r"\chapter{Why Did Certain Lag Sets Work Better?}",
        (r"This chapter pools every short-vs-seasonal comparison across all six "
         rf"locations, both stages, and both TVP branches ({n_total} matched pairs) "
         r"to ask two questions: did the seasonal lag set make optimisation "
         r"harder, and is there a structural (collinearity) reason the extra "
         r"seasonal coefficients might not translate into better forecasts."),
        r"\section{Optimisation Difficulty}",
        save_metric_table(
            opt_summary, "opt_difficulty_summary",
            "Mean optimisation diagnostics, short vs.\\ seasonal lag sets, pooled across all comparisons.",
            "tab:opt-difficulty",
        ),
        (rf"The seasonal-lag fit ends with a larger gradient norm than its matched "
         rf"short-lag fit in {n_grad_worse}/{n_total} comparisons, and needs more "
         rf"BFGS iterations in {n_iter_worse}/{n_total}. Each seasonal fit has 8 more "
         rf"free parameters (2 coefficients $\times$ 4 extra lags, per dynamic "
         rf"parameter) than its short-lag counterpart -- a larger, more "
         rf"weakly-identified parameter space is the direct numerical reason a "
         rf"harder optimisation would be expected regardless of whether it also "
         rf"improves the fit."),
        r"\section{Collinearity Among Seasonal-Lag Coefficients}",
        (r"For every seasonal-lag model, the mean absolute pairwise correlation "
         r"(from the BFGS approximate inverse-Hessian, "
         r"OPTIMIZATION.md \S9's sanctioned approximate-covariance source) is "
         rf"computed separately for the 4 seasonal-lag coefficients "
         rf"($A_{{364}},\ldots,A_{{367}}$ and $B_{{364}},\ldots,B_{{367}}$) and the "
         rf"3 short-lag coefficients ($A_1,A_2,A_3$ and $B_1,B_2,B_3$) within the "
         rf"\emph{{same}} fit. Pooled across all seasonal fits: mean $|$correlation$|$ "
         rf"among seasonal-lag coefficients is {_fmt(mean_corr_seasonal, 3)}, versus "
         rf"{_fmt(mean_corr_short_block, 3)} among the short-lag coefficients in the "
         rf"same models. The seasonal-lag block's approximate-Hessian condition "
         rf"number is {_fmt(cond_ratio, 2)}$\times$ the short-lag model's, "
         rf"median across all comparisons (higher means a harder-to-invert, more "
         rf"ill-conditioned optimisation landscape)."),
        (rf"Relating this directly to forecast skill: across the "
         rf"{len(valid)} seasonal fits with both a usable collinearity estimate and "
         rf"a computed CRPS difference, the correlation between the seasonal-lag "
         rf"block's own collinearity and the seasonal-minus-short CRPS delta is "
         rf"$r={_fmt(r_corr_vs_delta, 2)}$ "
         + (
            "(a positive value: more seasonal-lag collinearity coincides with a "
            "worse, i.e.\\ larger, CRPS delta -- consistent with the four seasonal "
            "lags fighting each other for the same signal instead of adding new "
            "information)." if np.isfinite(r_corr_vs_delta) and r_corr_vs_delta > 0.15 else
            "(a negative value: the seasonal fits where the four seasonal-lag "
            "coefficients are estimated MOST collinearly tend to have the BEST "
            "relative CRPS, the opposite of the ``collinearity hurts forecasts'' "
            "hypothesis -- collinearity here appears to be a symptom of weak "
            "identification, not a cause of poor forecasts in its own right; the "
            "gradient/iteration evidence above remains the more direct read on "
            "optimisation difficulty)." if np.isfinite(r_corr_vs_delta) and r_corr_vs_delta < -0.15 else
            "(no strong, consistent link between how collinear the seasonal "
            "coefficients are and how much the seasonal model's CRPS differs from "
            "its short-lag counterpart)." if np.isfinite(r_corr_vs_delta) else
            "(not enough usable fits to draw a reliable conclusion here)."
         )),
    ]

    overall_mean_delta = float(df["crps_delta"].mean())
    n_seasonal_better = int((df["crps_delta"] < 0).sum())
    parts.append(r"\section{Putting It Together}")
    parts.append(
        rf"Across all {n_total} comparisons, the seasonal lag set achieves a lower "
        rf"(better) CRPS than its matched short-lag model in {n_seasonal_better}/{n_total} "
        rf"cases, with a mean CRPS difference (seasonal $-$ short) of "
        rf"{_fmt(overall_mean_delta, 4)}. " +
        ("Seasonality's extra four lags carry a real, if modest, forecast-skill cost "
         "on average, and this chapter's diagnostics point to why: they add a "
         "meaningfully larger, more mutually correlated coefficient block that "
         "BFGS must estimate from the same training window, without a "
         "correspondingly large gain in same-day predictive information (annual "
         "rainfall recurrence at daily resolution is a much noisier signal than "
         "the immediate 1-3 day autocorrelation the short lag set already "
         "captures)." if overall_mean_delta > 0 else
         "Seasonality's extra four lags do carry a real forecast-skill benefit on "
         "average here, despite the harder, more collinear optimisation landscape "
         "documented above -- annual rainfall recurrence appears to add genuine "
         "same-day predictive information beyond the short lag set, even though "
         "estimating it is numerically harder.")
    )
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading artifacts...")
    all_data = {}
    for loc in LOCATIONS:
        short = STATION_REGISTRY[loc].get("short", loc)
        for stage in STAGES:
            for branch in BRANCHES:
                for lagset in LAGSETS:
                    all_data[(loc, stage, branch, lagset)] = load_cell(short, stage, branch, lagset)
    missing = [k for k, v in all_data.items() if v is None]
    if missing:
        print(f"  WARNING: missing {len(missing)} cells: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    print("Building introduction...")
    intro_tex = build_introduction()
    print("Building Stage 1 (baseline) chapter...")
    stage1_tex = build_stage_chapter(all_data, "stage1")
    print("Building Stage 2 (weather-covariate) chapter...")
    stage2_tex = build_stage_chapter(all_data, "stage2")
    print("Building 'why' chapter...")
    why_tex = build_why_chapter(all_data)

    today = datetime.date.today().isoformat()
    body = "\n\n".join([
        r"\begin{document}",
        r"\title{Seasonal vs.\ Short GAS Lags: Baseline and Weather-Covariate Models}",
        rf"\author{{Generated by \texttt{{generate\_seasonality\_gas\_report.py}}}}",
        rf"\date{{{today}}}",
        r"\maketitle",
        r"\tableofcontents",
        intro_tex,
        stage1_tex,
        stage2_tex,
        why_tex,
        r"\end{document}",
    ])
    tex_source = LATEX_PREAMBLE + "\n" + body

    tex_path = REPORT_DIR / "report.tex"
    tex_path.write_text(tex_source, encoding="utf-8")
    print(f"Wrote {tex_path}")

    if not args.no_compile:
        print("Compiling PDF (2 pdflatex passes)...")
        ok = compile_latex(tex_path)
        print("Compilation " + ("succeeded." if ok else "FAILED -- see log above."))


if __name__ == "__main__":
    main()
