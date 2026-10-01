"""
Original vs. v2 (sigmoid-bounded A/B/rho) comparison report (2026-09-02).

=============================================================================
OVERVIEW
=============================================================================

Compares the ORIGINAL Stage-2 (weather-covariate) GAS parameterisation
(joint penalty sum_l|A_l|<=2.0, sum_l|B_l|<=0.98, |rho|<=0.98, shared
across all lags of a parameter) against the v2 reparameterisation
(models/gas_filter.py's bound_ab + models/cov_gas_model.py's bound_rho:
each A_{j,l}/B_{j,l}/rho bounded INDEPENDENTLY per lag via a sigmoid,
b_max=0.98/n_lags so the worst-case persistence sum cannot exceed the
original safe budget -- see reports/seasonality GAS/
gradient_and_convergence_diagnosis.md §8 for the full derivation and the
failure mode of the earlier, unscaled attempt).

One representative Stage-2 model per location is used throughout (the one
of that location's 8 Stage-2 variants with the lowest gradient under the
ORIGINAL setting -- see the Representative Models chapter for exactly
which model this is per location, so the comparison stays apples-to-
apples). Everything here is read from already-computed artifacts
(run_original_vs_v2_comparison.py); no optimisation or simulation runs in
this script.

=============================================================================
USAGE
=============================================================================

    python generate_original_vs_v2_report.py [--no-compile]

Output:
    reports/original vs v2/report.tex / .pdf
    reports/original vs v2/figures/*.png
    reports/original vs v2/tables/*.csv
"""

from __future__ import annotations

import argparse
import datetime
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

from data.station_loader import STATION_REGISTRY
from diagnostics.plots import qq_location_mosaic, acf_location_mosaic
from diagnostics.multistep_forecast import DEFAULT_HORIZONS
from run_original_vs_v2_comparison import OUT_DIR as CMP_DIR, ARTIFACTS_DIR, V2_DIR, LOCATIONS
from report_se_utils import load_param_table_with_delta_se

REPORT_DIR = ROOT / "reports" / "original vs v2"
FIG_DIR    = REPORT_DIR / "figures"
TAB_DIR    = REPORT_DIR / "tables"

SETTINGS = ["original", "v2"]
SETTING_LABEL = {"original": "Original", "v2": "V2"}


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX helpers (self-contained, matching this repo's other report generators)
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
    if isinstance(v, str):
        return v if "$" in v else _tex(v)
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
\usepackage{longtable}
\usepackage{pdflscape}
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

def load_location(location: str) -> Optional[dict]:
    cfg = STATION_REGISTRY[location]
    short = cfg.get("short", cfg["run_id"])
    d = CMP_DIR / short
    info_path = d / "representative_model.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text())
    out = {"info": info}
    for setting in SETTINGS:
        sd = d / setting
        out[setting] = {
            "residuals": np.load(sd / "residuals.npy"),
            "onestep": json.loads((sd / "onestep_metrics.json").read_text()),
            "multistep": pd.read_csv(sd / "multistep_metrics.csv"),
        }
    return out


def load_param_table(setting: str, run_id: str, stage: str, model_id: str) -> pd.DataFrame:
    """
    Natural-scale parameter table with a genuine delta-method SE for every
    link-transformed coefficient whose link derivative is not saturated
    (report_se_utils.py), instead of a blanket "--" for every transformed
    parameter regardless of whether its own link was well-behaved.
    """
    d = (ARTIFACTS_DIR / run_id / stage / model_id) if setting == "original" else (V2_DIR / run_id / stage / model_id)
    return load_param_table_with_delta_se(d)


def save_param_table(df: pd.DataFrame, name: str, caption: str, label: str) -> str:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(TAB_DIR / f"{name}.csv", index=False)
    disp = pd.DataFrame(index=df.index.to_numpy())
    disp["Estimate"] = [_fmt(v, 4) for v in df["estimate"]]
    se_col = []
    for se, note in zip(df["se"], df["note"]):
        se_txt = _fmt(se, 4) if (se is not None and np.isfinite(se)) else "--"
        se_col.append(f"{se_txt} ({_tex(note)})" if note else se_txt)
    disp["Std. Error"] = se_col
    return _latex_table(disp, caption, label)


# ─────────────────────────────────────────────────────────────────────────────
# Introduction
# ─────────────────────────────────────────────────────────────────────────────

def build_introduction() -> str:
    return "\n\n".join([
        r"\chapter{Two Ways to Keep a GAS(p,q) Model Stationary}",
        (r"Every Stage-2 (weather-covariate) GAS model in this framework has "
         r"score-loading coefficients $A_{j,\ell}$ and autoregressive "
         r"coefficients $B_{j,\ell}$ for each lag $\ell$ in the model's lag "
         r"set $\mathcal{L}$ (3 lags for the short set, 7 for the seasonal "
         r"set), plus an occurrence-model persistence $\rho$. All three must "
         r"stay inside a bounded region for the recursion to be a stable, "
         r"stationary process rather than an explosive or non-identified "
         r"one. Two different constructions for guaranteeing this were "
         r"compared, refitting every affected model from scratch under each."),
        r"\section{Original: One Joint Penalty Per Parameter}",
        (r"The production pipeline enforces this with a single soft penalty "
         r"on the SUM of absolute coefficients across every lag of a given "
         r"parameter:"),
        r"""\begin{align*}
\textstyle\sum_{\ell\in\mathcal{L}} |A_{j,\ell}| &\le 2.0 \\
\textstyle\sum_{\ell\in\mathcal{L}} |B_{j,\ell}| &\le 0.98 \\
|\rho| &\le 0.98
\end{align*}""",
        (r"A mechanical consequence: this budget is SHARED across every lag "
         r"of a parameter, so a 7-lag seasonal model must split the same "
         r"0.98 total persistence seven ways, while a 3-lag short model "
         r"splits it three ways -- a structurally tighter per-lag "
         r"constraint precisely where there are more lags to estimate. "
         r"Diagnostic work (\texttt{reports/seasonality GAS/"
         r"gradient\_and\_convergence\_diagnosis.md}) found this associated "
         r"with widespread large gradients and, for a genuine subset, "
         r"materially under-optimised fits."),
        r"\section{V2: An Independent Sigmoid Per Lag}",
        (r"Each coefficient is instead bounded BY CONSTRUCTION, independently "
         r"of every other lag of the same parameter, via a logistic sigmoid "
         r"of an unconstrained raw value:"),
        r"""\begin{align*}
A_{j,\ell} &= 2 \cdot \mathrm{sigmoid}(z^A_{j,\ell}) &&\in (0,\,2) \\
B_{j,\ell} &= \left(\tfrac{0.98}{|\mathcal{L}|}\right) \cdot \mathrm{sigmoid}(z^B_{j,\ell}) &&\in \left(0,\,\tfrac{0.98}{|\mathcal{L}|}\right) \\
\rho &= \mathrm{sigmoid}(z^\rho) &&\in (0,\,1)
\end{align*}""",
        (r"The $B$ bound is deliberately scaled by the number of lags "
         r"$|\mathcal{L}|$ so that even in the worst case -- every lag "
         r"simultaneously at its individual ceiling -- the total persistence "
         r"$\sum_\ell B_{j,\ell}$ cannot exceed 0.98, exactly the original "
         r"safe budget. This was not the first thing tried: an earlier "
         r"attempt with a flat $B_{j,\ell}=\mathrm{sigmoid}(z)\in(0,1)$ per "
         r"lag (no rescaling) let roughly a third of all parameter blocks "
         r"push their persistence sum past 1.0 -- genuinely non-stationary "
         r"-- while their score-loading coefficients collapsed to zero, an "
         r"overfit-looking artifact rather than a better estimate of the "
         r"same model. That attempt is documented alongside this fix in "
         r"\texttt{gradient\_and\_convergence\_diagnosis.md \S8} as a "
         r"cautionary result, not repeated here."),
        (r"Both settings were fit from the SAME data, SAME covariates, and "
         r"the SAME fresh (not warm-started) starting values, so any "
         r"difference below reflects the reparameterisation itself."),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Representative models (transparency chapter)
# ─────────────────────────────────────────────────────────────────────────────

def build_representative_chapter(all_data: Dict[str, dict]) -> str:
    table_rows = {}
    for location in LOCATIONS:
        display = STATION_REGISTRY[location].get("display", location)
        d = all_data.get(location)
        if d is None:
            continue
        info = d["info"]
        table_rows[display] = {
            "Stage": info["stage"],
            "Model": info["model_id"],
        }
    df = pd.DataFrame(table_rows).T
    tex_table = save_metric_table(
        df, "representative_models",
        "The single Stage-2 model used to represent each location throughout this report -- "
        "chosen as the lowest-gradient model among that location's 8 Stage-2 variants "
        "UNDER THE ORIGINAL SETTING ONLY, so the choice cannot be biased toward whichever "
        "setting looks better. Both settings below refit this exact same model (same "
        "covariates, same lag set, same branch) so the comparison is apples-to-apples.",
        "tab:representative-models",
    )
    return "\n\n".join([
        r"\chapter{Which Model Represents Each Location}",
        r"\label{ch:representative}",
        (r"Every location has 8 Stage-2 weather-covariate model variants (2 covariate "
         r"blocks $\times$ 2 lag sets $\times$ \{1 time-varying parameter, 2 time-varying "
         r"parameters\}). Showing all 8 under both settings would triple-count the same "
         r"story six times over, so one representative model per location is used "
         r"throughout the rest of this report -- named here explicitly before any "
         r"comparison, so no number below mixes two different underlying models."),
        tex_table,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Gradient reliability, both settings, all 48 Stage-2 models
# ─────────────────────────────────────────────────────────────────────────────

def build_gradient_chapter() -> str:
    df = pd.read_csv(V2_DIR / "summary_consolidated.csv")
    name_map = {v["run_id"]: v.get("display", k) for k, v in STATION_REGISTRY.items()}
    df["location"] = df["run_id"].map(name_map)

    def _tally(col, thr=1.0):
        n = len(df)
        n_ok = int((df[col] < thr).sum())
        return n_ok, n

    n_ok_orig, n_tot = _tally("old_grad_norm_inf")
    n_ok_v2, _ = _tally("new_grad_norm_inf")

    def worst_table(col, label_col, n=10):
        d = df.sort_values(col, ascending=False)[["location", "stage", "model_id", col]].head(n)
        d = d.rename(columns={col: "grad_norm_inf"})
        d = d.set_index(d["location"] + " / " + d["stage"] + "/" + d["model_id"])[["grad_norm_inf"]]
        return d

    orig_worst_tex = save_metric_table(
        worst_table("old_grad_norm_inf", "old"), "gradient_worst_original",
        "Original setting: the 10 worst (largest) gradients among all 48 Stage-2 models.",
        "tab:grad-worst-original",
    )
    v2_worst_tex = save_metric_table(
        worst_table("new_grad_norm_inf", "new"), "gradient_worst_v2",
        "V2 setting: the 10 worst (largest) gradients among all 48 Stage-2 models "
        "(after refitting from scratch).",
        "tab:grad-worst-v2",
    )

    return "\n\n".join([
        r"\chapter{Gradient Reliability, Model by Model}",
        (r"A fitted model's post-fit gradient $|\nabla Q(\hat\theta)|_\infty$ is this "
         r"framework's diagnostic for whether an optimum was genuinely reached "
         r"(OPTIMIZATION.md \S9-11); a value below 1 is treated as reliable throughout "
         r"this project. This chapter reports that diagnostic for all 48 Stage-2 models "
         r"(6 locations $\times$ 2 covariate blocks $\times$ 2 lag sets $\times$ 2 "
         r"branches), under each setting in turn."),
        r"\section{Original Setting}",
        rf"{n_ok_orig}/{n_tot} models ({100*n_ok_orig/n_tot:.0f}\%) reach a reliable gradient. "
        r"The 10 worst are:",
        orig_worst_tex,
        r"\section{V2 Setting}",
        rf"{n_ok_v2}/{n_tot} models ({100*n_ok_v2/n_tot:.0f}\%) reach a reliable gradient "
        r"after refitting from scratch under the v2 reparameterisation. The 10 worst "
        r"(now much smaller in absolute terms than the original setting's worst cases) are:",
        v2_worst_tex,
        (r"Full model-by-model gradients for both settings are saved at "
         r"\texttt{artifacts/bound\_ab\_experiment\_v2/summary\_consolidated.csv} "
         r"(columns \texttt{old\_grad\_norm\_inf}, \texttt{new\_grad\_norm\_inf})."),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Mosaics: reuse qq_location_mosaic/acf_location_mosaic with "locations" =
# ["Original", "V2"] -- a clean, free 2-panel side-by-side comparison.
# ─────────────────────────────────────────────────────────────────────────────

def build_location_mosaics(location: str, d: dict, short: str) -> Dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    qr_by_setting = {SETTING_LABEL[s]: d[s]["residuals"] for s in SETTINGS}
    panel_order = [SETTING_LABEL[s] for s in SETTINGS]
    display = STATION_REGISTRY[location].get("display", location)

    fig_qq = qq_location_mosaic(qr_by_setting, panel_order,
                                 suptitle=f"{display}: in-sample QQ plot, Original vs. V2")
    qq_path = FIG_DIR / f"qq_{short}.png"
    fig_qq.savefig(qq_path, dpi=150)
    plt.close(fig_qq)

    fig_acf = acf_location_mosaic(qr_by_setting, panel_order, n_lags=400,
                                   suptitle=f"{display}: in-sample 400-lag ACF, Original vs. V2")
    acf_path = FIG_DIR / f"acf_{short}.png"
    fig_acf.savefig(acf_path, dpi=150)
    plt.close(fig_acf)

    return {"qq": qq_path.relative_to(REPORT_DIR).as_posix(),
            "acf": acf_path.relative_to(REPORT_DIR).as_posix()}


# ─────────────────────────────────────────────────────────────────────────────
# OOS comparison tables
# ─────────────────────────────────────────────────────────────────────────────

def onestep_table(d: dict) -> pd.DataFrame:
    o, v = d["original"]["onestep"], d["v2"]["onestep"]
    return pd.DataFrame({
        "RMSE Original": [o["rmse"]], "RMSE V2": [v["rmse"]],
        "CRPS Original": [o["crps_mean"]], "CRPS V2": [v["crps_mean"]],
    }, index=["1-step rolling OOS"])


def multistep_table(d: dict) -> pd.DataFrame:
    mo = d["original"]["multistep"].set_index("horizon")
    mv = d["v2"]["multistep"].set_index("horizon")
    rows = {}
    for h in DEFAULT_HORIZONS:
        rows[f"h={h}"] = {
            "RMSE Original": mo.loc[h, "rmse"] if h in mo.index else np.nan,
            "RMSE V2": mv.loc[h, "rmse"] if h in mv.index else np.nan,
            "CRPS Original": mo.loc[h, "crps"] if h in mo.index else np.nan,
            "CRPS V2": mv.loc[h, "crps"] if h in mv.index else np.nan,
        }
    return pd.DataFrame(rows).T


# ─────────────────────────────────────────────────────────────────────────────
# Per-location chapter
# ─────────────────────────────────────────────────────────────────────────────

def build_locations_chapter(all_data: Dict[str, dict]) -> str:
    parts = [r"\chapter{Per-Location Comparison}", r"\label{ch:results}"]
    for location in LOCATIONS:
        display = STATION_REGISTRY[location].get("display", location)
        short = STATION_REGISTRY[location].get("short", location)
        d = all_data.get(location)
        parts.append(rf"\section{{{_tex(display)}}}")
        if d is None:
            parts.append("Not available.")
            continue
        info = d["info"]
        parts.append(
            rf"Representative model: \texttt{{{_tex(info['stage'])}/{_tex(info['model_id'])}}} "
            rf"(Section~\ref{{ch:representative}})."
        )

        parts.append(r"\subsection{Residual Diagnostics}")
        mosaics = build_location_mosaics(location, d, short)
        parts += [
            r"\begin{figure}[H]", r"\centering",
            rf"\includegraphics[width=\textwidth]{{{mosaics['qq']}}}",
            rf"\caption{{{_tex(display)}: QQ plot of in-sample quantile residuals, Original vs.\ V2.}}",
            rf"\label{{fig:qq-{short}}}", r"\end{figure}",
            r"\begin{figure}[H]", r"\centering",
            rf"\includegraphics[width=\textwidth]{{{mosaics['acf']}}}",
            rf"\caption{{{_tex(display)}: 400-lag ACF of in-sample quantile residuals, Original vs.\ V2.}}",
            rf"\label{{fig:acf-{short}}}", r"\end{figure}",
        ]

        parts.append(r"\subsection{Parameter Estimates}")
        for setting in SETTINGS:
            pdf, meta = load_param_table(setting, info["run_id"], info["stage"], info["model_id"])
            grad = meta.get("grad_norm_inf", np.nan)
            parts.append(
                rf"\textbf{{{SETTING_LABEL[setting]}}} (post-fit gradient "
                rf"$|\nabla Q(\hat\theta)|_\infty={_fmt(grad, 4)}$):"
            )
            parts.append(save_param_table(
                pdf, f"{short}_{setting}_params",
                f"{_tex(display)}: {SETTING_LABEL[setting]} setting -- parameter estimates and standard errors.",
                f"tab:params-{short}-{setting}",
            ))

        parts.append(r"\subsection{Out-of-Sample Forecast Comparison}")
        parts.append(save_metric_table(
            onestep_table(d), f"{short}_onestep",
            f"{_tex(display)}: full-test-set 1-step rolling OOS RMSE/CRPS, Original vs.\\ V2.",
            f"tab:onestep-{short}",
        ))
        n_origins = info.get("original", {}).get("n_origins", "?")
        n_draws = info.get("original", {}).get("n_draws", "?")
        parts.append(save_metric_table(
            multistep_table(d), f"{short}_multistep",
            f"{_tex(display)}: Monte Carlo multi-step OOS RMSE/CRPS, Original vs.\\ V2 "
            f"({n_origins} strided forecast origins, {n_draws} draws each, identical origins "
            f"for both settings).",
            f"tab:multistep-{short}",
        ))
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Final summary chapter (before conclusion)
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_chapter(all_data: Dict[str, dict]) -> str:
    # d["original_grad"]/d["v2_grad"] are attached in main() (read from each
    # setting's own metadata.json -- onestep_metrics.json has no grad field).
    rows = {}
    for location in LOCATIONS:
        display = STATION_REGISTRY[location].get("display", location)
        d = all_data.get(location)
        if d is None:
            continue
        o, v = d["original"], d["v2"]
        rows[display] = {
            "Grad. Original": d["original_grad"], "Grad. V2": d["v2_grad"],
            "RMSE Original": o["onestep"]["rmse"], "RMSE V2": v["onestep"]["rmse"],
            "CRPS Original": o["onestep"]["crps_mean"], "CRPS V2": v["onestep"]["crps_mean"],
        }
    df = pd.DataFrame(rows).T[["Grad. Original", "Grad. V2", "RMSE Original", "RMSE V2", "CRPS Original", "CRPS V2"]]
    tex_table = save_metric_table(
        df, "summary_original_vs_v2",
        "Representative model per location: gradient, 1-step-ahead OOS RMSE, and 1-step-ahead OOS CRPS, "
        "Original vs.\\ V2.",
        "tab:summary-original-vs-v2",
    )
    return "\n\n".join([r"\chapter{Summary Table}", tex_table])


# ─────────────────────────────────────────────────────────────────────────────
# Conclusion
# ─────────────────────────────────────────────────────────────────────────────

def build_conclusion(all_data: Dict[str, dict]) -> str:
    n = sum(1 for d in all_data.values() if d is not None)
    n_grad_better = sum(1 for d in all_data.values() if d and d["v2_grad"] < d["original_grad"])
    n_crps_better = sum(1 for d in all_data.values() if d and d["v2"]["onestep"]["crps_mean"] < d["original"]["onestep"]["crps_mean"])
    return "\n\n".join([
        r"\chapter{Conclusion}",
        (rf"Across the {n} representative Stage-2 models compared here, V2's independent "
         rf"per-lag sigmoid reparameterisation reached a lower (better) post-fit gradient "
         rf"than the original joint-penalty parameterisation in {n_grad_better}/{n} cases, "
         rf"and a lower (better) 1-step-ahead out-of-sample CRPS in {n_crps_better}/{n} "
         rf"cases -- the same conclusion reached at full scale across all 48 Stage-2 "
         rf"models in Chapter~\ref{{ch:results}}'s companion diagnosis document: "
         rf"restructuring the persistence budget from one constraint shared across every "
         rf"lag into independent, per-lag bounds -- scaled so the worst case still cannot "
         rf"exceed the original safe total -- converts the large majority of previously "
         rf"unverifiable Stage-2 fits into genuinely, checkably converged ones, at little "
         rf"to no forecast-accuracy cost and in several cases a genuine forecast-accuracy "
         rf"gain."),
        (r"This report intentionally stops short of promoting V2 into the production "
         r"pipeline -- that decision affects every downstream number in the main "
         r"multi-location report and the Harvey-stage initialisation that reads Stage-2 "
         r"artifacts, and is left for explicit sign-off rather than made here."),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading artifacts...")
    all_data = {location: load_location(location) for location in LOCATIONS}
    missing = [k for k, v in all_data.items() if v is None]
    if missing:
        print(f"  WARNING: missing analysis for: {missing}")

    # Attach the actual gradient (from metadata.json, not onestep_metrics which
    # doesn't carry it) for each setting, needed by the summary/conclusion chapters.
    for location, d in all_data.items():
        if d is None:
            continue
        info = d["info"]
        orig_meta = json.loads((ARTIFACTS_DIR / info["run_id"] / info["stage"] / info["model_id"] / "metadata.json").read_text())
        v2_meta = json.loads((V2_DIR / info["run_id"] / info["stage"] / info["model_id"] / "metadata.json").read_text())
        d["original_grad"] = orig_meta.get("grad_norm_inf", np.nan)
        d["v2_grad"] = v2_meta.get("grad_norm_inf", np.nan)

    print("Building introduction...")
    intro_tex = build_introduction()
    print("Building representative-models chapter...")
    rep_tex = build_representative_chapter(all_data)
    print("Building gradient-reliability chapter...")
    grad_tex = build_gradient_chapter()
    print("Building per-location chapter (mosaics, tables)...")
    locations_tex = build_locations_chapter(all_data)
    print("Building summary chapter...")
    summary_tex = build_summary_chapter(all_data)
    conclusion_tex = build_conclusion(all_data)

    today = datetime.date.today().isoformat()
    body = "\n\n".join([
        r"\begin{document}",
        r"\title{Original vs.\ V2: Sigmoid-Bounded GAS Persistence}",
        rf"\author{{Generated by \texttt{{generate\_original\_vs\_v2\_report.py}}}}",
        rf"\date{{{today}}}",
        r"\maketitle",
        r"\tableofcontents",
        intro_tex,
        rep_tex,
        grad_tex,
        locations_tex,
        summary_tex,
        conclusion_tex,
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
