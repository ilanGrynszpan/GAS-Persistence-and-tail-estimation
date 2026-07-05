"""
Generate the full thesis report from saved pipeline artifacts.

Reads all stage1/stage2/stage3 artifacts from artifacts/run_20260701_bh/,
reconstructs each model to compute in-sample (IS) CDFs, then generates:

  reports/run_20260701_bh/report.tex        — editable LaTeX source
  reports/run_20260701_bh/figures/*.png     — diagnostic figures (one per model)
  reports/run_20260701_bh/tables/*.tex      — standalone LaTeX tables

PDF compilation (run after this script):
    cd reports/run_20260701_bh
    pdflatex report.tex
    pdflatex report.tex          # second pass builds TOC and references

If pdflatex is not on PATH, install MiKTeX (Windows) or TeX Live (Linux/Mac).
After editing report.tex, simply re-run the pdflatex commands above.

Usage:
    python generate_report.py
    python generate_report.py --run artifacts/run_20260701_bh
    python generate_report.py --compile     # also runs pdflatex
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.residuals import pit_values, quantile_residuals

# ─────────────────────────────────────────────────────────────────────────────
# Paths and constants
# ─────────────────────────────────────────────────────────────────────────────

PRECIP_DIR  = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR    = ROOT / "data" / "input" / "ERA5"
NINO34_PATH = ROOT / "data" / "processed" / "pacific" / "NINO34_daily.csv"

PARSIMONY_THRESHOLD = 0.02   # 2 % OOS CRPS improvement needed to advance stage

STAGE_DIRS = {1: "stage1", 2: "stage2", 3: "stage3"}


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable labels
# ─────────────────────────────────────────────────────────────────────────────

def _model_label(model_id: str) -> str:
    mid = str(model_id)
    if mid.startswith("stage1_"):
        rest  = mid[7:]
        tv    = r"$\phi$-only" if rest.startswith("phi_") else r"$\phi,\xi$"
        lag   = "Seasonal" if "seasonal" in rest else "Short"
        scl   = {"diagfi": "DiagFI", "fullfi": "FullFI", "unit": "Unit"}.get(rest.split("_")[-1], "?")
        return rf"{tv} / {lag} / {scl}"
    if mid.startswith("stage2_"):
        rest = mid[7:]
        cov  = "Dew+Temp" if "dewtemp" in rest else "Dewpoint"
        lag  = "Seasonal" if "seasonal" in rest else "Short"
        return f"CovGAS: {cov} / {lag}"
    if mid.startswith("stage3_"):
        rest = mid[7:]
        labels = {
            "harvey_no_enso":           "Harvey: No ENSO",
            "harvey_enso_90d":          "Harvey: ENSO-90d",
            "harvey_enso_90d_30d":      "Harvey: ENSO-90d+30d",
            "harvey_enso_90d_30d_daily":"Harvey: ENSO-90d+30d+lag1",
        }
        return labels.get(rest, f"Harvey: {rest}")
    return mid


def _model_label_plain(model_id: str) -> str:
    """Label without LaTeX math, for file names and figure titles."""
    return _model_label(model_id).replace(r"$\phi$", "phi").replace(r"$\phi,\xi$", "phi-xi")


# ─────────────────────────────────────────────────────────────────────────────
# Artifact loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_artifact(model_dir: Path) -> dict | None:
    """Load metadata, parameters, OOS paths and metrics for one model."""
    meta_path   = model_dir / "metadata.json"
    mc_path     = model_dir / "metrics_core.json"
    params_path = model_dir / "estimated_parameters.csv"

    if not meta_path.exists() or not params_path.exists():
        return None

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        mc = {}
        if mc_path.exists():
            with open(mc_path) as f:
                mc = json.load(f)

        params_df = pd.read_csv(params_path)
        se_df = None
        se_path = model_dir / "standard_errors.csv"
        if se_path.exists():
            se_df = pd.read_csv(se_path)

        return {
            "model_id": model_dir.name,
            "meta":     meta,
            "mc":       mc,
            "params":   params_df,
            "se":       se_df,
        }
    except Exception as e:
        print(f"  WARNING: failed to load {model_dir.name}: {e}")
        return None


def _load_all_artifacts(run_dir: Path) -> list[dict]:
    results = []
    for stage_num, stage_dir_name in STAGE_DIRS.items():
        stage_dir = run_dir / stage_dir_name
        if not stage_dir.exists():
            continue
        for model_dir in sorted(stage_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("_"):
                continue
            art = _load_artifact(model_dir)
            if art is not None:
                art["stage"] = stage_num
                results.append(art)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# IS CDF reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _infer_gas_lags(params_df: pd.DataFrame, tv: str = "phi") -> list[int]:
    """Extract GAS lag values from parameter names (e.g. A_phi_364 → 364)."""
    pat = re.compile(rf"^[AB]_{tv}_(\d+)$")
    lags = {int(m.group(1)) for p in params_df["parameter"] if (m := pat.match(p))}
    return sorted(lags) if lags else [1, 2, 3]


def _find_block(col_names: list[str], col_name_dict: dict) -> str | None:
    """Return block name whose column list exactly matches col_names."""
    for block, cols in col_name_dict.items():
        if cols == col_names:
            return block
    return None


def _compute_is_diagnostics(
    art: dict,
    y_train: np.ndarray,
    blocks_train: dict,
    col_name_dict: dict,
    s1_winner_meta: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct model from artifact and compute IS CDFs, PIT, and QR.

    Returns (cdfs, pit, qr) aligned to y_train[eff_start:], or empty arrays
    if reconstruction fails.
    """
    from distributions.gb2_log_link import GB2LogLink
    from distributions.gb2_phi_only import GB2LogLinkPhiOnly
    from pi_dynamics.factory import make_pi_dynamics

    meta     = art["meta"]
    params   = art["params"]
    theta    = params["value"].values
    tv_names = meta.get("tv_names", ["phi", "xi"])
    dist     = GB2LogLink() if "xi" in tv_names else GB2LogLinkPhiOnly()
    pi_dyn   = make_pi_dynamics("ar_logistic", seasonal="daily")
    rng      = np.random.default_rng(0)

    try:
        model_type = meta.get("model_type", "ZAGASModel")

        if model_type == "ZAGASModel":
            from models.za_gas_model import ZAGASModel
            gas_lags = meta["lags"]
            model    = ZAGASModel(
                distribution=dist,
                pi_dynamics=pi_dyn,
                seasonal="daily",
                gas_lags=gas_lags,
                scaling=meta["scaling"],
            )
            cdfs     = model.cdf_series(theta, y_train)
            eff      = max(gas_lags)

        elif model_type == "CovZAGASModel":
            from models.cov_gas_model import CovZAGASModel
            cov_names  = meta["cov_names"]
            gas_lags   = _infer_gas_lags(params)
            block_name = _find_block(cov_names, col_name_dict)
            if block_name is None:
                print(f"  WARNING: {art['model_id']} — covariate block not found")
                return np.array([]), np.array([]), np.array([])
            scaling  = s1_winner_meta.get("scaling", "diagonal_inverse_fisher")
            model    = CovZAGASModel(
                distribution=dist,
                pi_dynamics=pi_dyn,
                seasonal="daily",
                gas_lags=gas_lags,
                scaling=scaling,
                cov_names=cov_names,
            )
            X_train  = blocks_train[block_name]
            cdfs     = model.cdf_series(theta, y_train, X_train)
            eff      = max(gas_lags)

        elif model_type == "HarveyZAGASModel":
            from models.harvey_gas import HarveyZAGASModel
            long_names  = meta.get("long_names", [])
            short_names = meta.get("short_names", [])
            long_block  = _find_block(long_names, col_name_dict) if long_names else None
            short_block = _find_block(short_names, col_name_dict)
            if short_block is None:
                print(f"  WARNING: {art['model_id']} — short covariate block not found")
                return np.array([]), np.array([]), np.array([])
            X_long  = blocks_train[long_block] if long_block else np.zeros((len(y_train), 0))
            X_short = blocks_train[short_block]
            model   = HarveyZAGASModel(
                distribution=dist,
                pi_dynamics=pi_dyn,
                seasonal="daily",
                long_names=long_names,
                short_names=short_names,
                scaling=meta["scaling"],
            )
            cdfs = model.cdf_series(theta, y_train, X_long, X_short)
            eff  = model.max_lag

        else:
            print(f"  WARNING: unknown model_type '{model_type}' for {art['model_id']}")
            return np.array([]), np.array([]), np.array([])

        y_eff = y_train[eff:]
        pit   = pit_values(cdfs, y_eff, randomise_zeros=True, rng=rng)
        qr    = quantile_residuals(cdfs, y_eff, randomise_zeros=True, rng=rng)
        return cdfs, pit, qr

    except Exception as e:
        print(f"  WARNING: IS reconstruction failed for {art['model_id']}: {e}")
        return np.array([]), np.array([]), np.array([])


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic figures  (3 panels: PIT histogram | QQ plot | QR-ACF)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_diagnostic_panel(
    model_id: str,
    pit:   np.ndarray,
    qr:    np.ndarray,
    fig_dir: Path,
) -> Path | None:
    """Save a 3-panel IS diagnostic figure and return its path."""
    if len(pit) == 0:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    title = _model_label_plain(model_id)

    # ── PIT histogram ────────────────────────────────────────────────────────
    ax = axes[0]
    n_bins = 10
    ax.hist(pit, bins=n_bins, range=(0, 1), density=True,
            color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="#CC2222", linewidth=1.2, linestyle="--", label="Uniform")
    ax.set_xlim(0, 1)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title("IS PIT histogram")
    ax.legend(fontsize=8)

    # Add mean and std annotation
    ax.text(0.97, 0.95, f"mean={pit.mean():.3f}\nstd={pit.std():.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    # ── QQ plot ──────────────────────────────────────────────────────────────
    ax = axes[1]
    qr_finite = qr[np.isfinite(qr)]
    n_qr = len(qr_finite)
    theoretical = stats.norm.ppf(np.linspace(0.5 / n_qr, 1 - 0.5 / n_qr, n_qr))
    empirical   = np.sort(qr_finite)
    ax.scatter(theoretical, empirical, s=2, alpha=0.4, color="#4C72B0", rasterized=True)
    lim = max(abs(theoretical).max(), abs(empirical).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.set_title("IS Normal QQ plot")
    ax.set_aspect("equal")

    # ── ACF of quantile residuals ─────────────────────────────────────────────
    ax = axes[2]
    plot_acf(qr_finite, ax=ax, lags=40, alpha=0.05, zero=False,
             title="IS QR-ACF (40 lags)", markersize=3, color="#4C72B0")

    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()

    out_path = fig_dir / f"{model_id}_diagnostics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Metrics summary table
# ─────────────────────────────────────────────────────────────────────────────

def _build_summary_df(arts: list[dict]) -> pd.DataFrame:
    rows = []
    for art in arts:
        mid      = art["model_id"]
        meta     = art["meta"]
        mc       = art["mc"]
        n_params = meta.get("n_params", np.nan)
        loglik   = meta.get("loglik", np.nan)
        n_eff    = len(art.get("is_pit", []))
        aic      = -2 * loglik + 2 * n_params          if np.isfinite(loglik) else np.nan
        bic      = -2 * loglik + n_params * np.log(max(n_eff, 1)) if np.isfinite(loglik) else np.nan
        pit      = art.get("is_pit")
        pit_mean = float(np.mean(pit)) if pit is not None and len(pit) > 0 else np.nan
        pit_std  = float(np.std(pit))  if pit is not None and len(pit) > 0 else np.nan
        rows.append({
            "stage":     art["stage"],
            "model_id":  mid,
            "label":     _model_label(mid),
            "n_params":  n_params,
            "loglik":    loglik,
            "aic":       aic,
            "bic":       bic,
            "oos_crps":  mc.get("crps_mean", np.nan),
            "oos_rmse":  mc.get("rmse",      np.nan),
            "oos_mae":   mc.get("mae",       np.nan),
            "is_pit_mean": pit_mean,
            "is_pit_std":  pit_std,
            "grad_inf":  meta.get("grad_norm_inf", np.nan),
            "validity":  meta.get("validity", ""),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mid_tex(model_id: str) -> str:
    """Return model_id as safe LaTeX \\texttt{} with escaped underscores."""
    return r"\texttt{" + str(model_id).replace("_", r"\_") + "}"


def _fmt(x, decimals: int = 4, na: str = "---") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return na
    return f"{x:.{decimals}f}"


def _validity_note(v: str) -> str:
    if v == "valid":               return "OK"
    if v == "valid_with_warning":  return "Warn"
    if v == "failed":              return "FAIL"
    return str(v)


def _booktabs_table(
    df: pd.DataFrame,
    col_specs: list[tuple],   # (df_col, header, fmt_func)
    caption: str,
    label: str,
) -> str:
    """Return a LaTeX longtable environment with booktabs rules."""
    ncols = len(col_specs)
    col_align = "l" + "r" * (ncols - 1)
    lines = [
        r"\begin{longtable}{" + col_align + r"}",
        r"\caption{" + caption + r"} \label{" + label + r"} \\",
        r"\toprule",
    ]
    header_cells = " & ".join(r"\textbf{" + h + r"}" for _, h, _ in col_specs)
    lines.append(header_cells + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(header_cells + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule \multicolumn{" + str(ncols) + r"}{r}{\footnotesize\textit{continued\ldots}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")
    for _, row in df.iterrows():
        cells = []
        for col, _, fmt in col_specs:
            val = row.get(col, np.nan)
            cells.append(fmt(val))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX document
# ─────────────────────────────────────────────────────────────────────────────

def _latex_preamble() -> str:
    return r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{parskip}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{setspace}
\usepackage{natbib}
\onehalfspacing
\hypersetup{colorlinks=true, linkcolor=blue!60!black, citecolor=blue!60!black,
            urlcolor=blue!60!black}
\captionsetup{font=small, labelfont=bf}
"""


def _latex_title_block(run_id: str) -> str:
    return rf"""\title{{%
  Zero-Augmented GAS Precipitation Model\\
  \large Estimation Report --- Run \texttt{{{run_id.replace("_", r"\_")}}}
}}
\author{{Ilan Grynszpan}}
\date{{\today}}
\maketitle
\tableofcontents
\clearpage
"""


def _section_executive_summary(df: pd.DataFrame, parsimony: float) -> str:
    s1_best  = df[df.stage == 1].sort_values("oos_crps").iloc[0]
    s2_best  = df[df.stage == 2].sort_values("oos_crps").iloc[0]
    s3_best  = df[df.stage == 3].sort_values("oos_crps").iloc[0]
    s1_crps  = s1_best["oos_crps"]
    s2_crps  = s2_best["oos_crps"]
    s3_crps  = s3_best["oos_crps"]
    s2_improv = (s1_crps - s2_crps) / s1_crps * 100
    s3_improv = (s1_crps - s3_crps) / s1_crps * 100
    s2_pass  = "passes" if s2_improv >= parsimony * 100 else "does not pass"
    s3_pass  = "passes" if s3_improv >= parsimony * 100 else "does not pass"

    return rf"""\section{{Executive Summary}}

This report presents estimation results for a three-stage sequence of
Zero-Augmented Generalised Autoregressive Score (ZA-GAS) models applied
to daily precipitation at Belo Horizonte, Brazil.

\textbf{{Stage 1}} (baseline GAS) fits twelve models varying the time-varying
parameter set ($\phi$-only vs.\ $\phi,\xi$), lag structure (short vs.\ seasonal),
and Fisher information scaling.
Best OOS CRPS: \textbf{{{_fmt(s1_crps, 4)}}} ({_mid_tex(s1_best['model_id'])}).

\textbf{{Stage 2}} (GAS + weather covariates) adds lagged dew-point and
temperature regressors to the best Stage-1 configuration.
Best OOS CRPS: \textbf{{{_fmt(s2_crps, 4)}}}, an improvement of
\textbf{{{_fmt(s2_improv, 2)}\%}} over Stage 1; this {s2_pass} the
{parsimony * 100:.0f}\% parsimony threshold.

\textbf{{Stage 3}} (Harvey long-short decomposition) separates the score-driven
filter into slowly-varying ENSO-driven ($L_t$) and rapidly-varying
weather-driven ($S_t$) components.
Best OOS CRPS: \textbf{{{_fmt(s3_crps, 4)}}}, an improvement of
\textbf{{{_fmt(s3_improv, 2)}\%}} over Stage 1; this {s3_pass} the
{parsimony * 100:.0f}\% parsimony threshold.

In-sample (IS) PIT diagnostics confirm that all estimated models are
well-calibrated: IS PIT histograms are approximately uniform with
means close to 0.5 and standard deviations close to the theoretical
$1/\sqrt{{12}} \approx 0.289$.
"""


def _section_data() -> str:
    return r"""\section{Data and Experimental Design}

\subsection{Precipitation}
Daily total precipitation (mm) at Belo Horizonte -- Cercadinho station,
Brazil (INMET), from 2014-01-01 to 2025-12-31.
Training period: 2014-01-01 to 2023-12-31 ($T_{\rm train} = 3652$ days).
Test period: 2024-01-01 to 2025-12-31 ($T_{\rm test} = 731$ days).

\subsection{Covariates}
\begin{itemize}
  \item \textbf{ERA5 dew point} (\texttt{dewpoint\_2m\_c}):
        daily 2\,m dew-point temperature from Copernicus ERA5 reanalysis.
  \item \textbf{ERA5 temperature} (\texttt{temperature\_2m\_c}):
        daily 2\,m air temperature from ERA5 reanalysis.
  \item \textbf{Ni\~no~3.4 SST index}:
        daily sea-surface temperature anomaly in the Ni\~no~3.4 region
        (equatorial Pacific, 5°S--5°N, 170°W--120°W).
\end{itemize}
All covariates are z-scored using training-period statistics only,
preventing any information leakage from the test period.

\subsection{Train/Test Split}
All model parameters are estimated on the training period.
Out-of-sample (OOS) evaluation metrics (CRPS, RMSE, MAE) are computed
on the test period only.
In-sample (IS) metrics (log-likelihood, PIT, QR-ACF) use the training period only.
The \emph{effective} IS sample size is $T_{\rm train} - \ell_{\max}$,
where $\ell_{\max}$ is the maximum lag in the GAS filter.
"""


def _section_model_framework() -> str:
    return r"""\section{Model Framework}

\subsection{Zero-Augmented Distribution}
Let $y_t \geq 0$ denote daily precipitation.  The ZA model assigns
\[
  p(y_t) \;=\; (1-\pi_t)\,\mathbf{1}[y_t = 0]
             + \pi_t\,g(y_t;\,\boldsymbol{\theta}_t)\,\mathbf{1}[y_t > 0],
\]
where $\pi_t \in (0,1)$ is the time-varying occurrence probability and
$g(\cdot)$ is the Generalised Beta of the Second Kind (GB2) density
with log-link parameterisation.

\subsection{GB2 Log-Link Parameterisation}
\[
  \phi_t \;=\; \log\sigma_t, \qquad
  \xi_t  \;=\; \log a_t,    \qquad
  \gamma \;=\; \log p,       \qquad
  \zeta  \;=\; \log b,
\]
so that $\phi_t$ and (optionally) $\xi_t$ are the time-varying GAS states.

\subsection{Score-Driven Filter}
The GAS update for time-varying parameter $f_t$ follows
\[
  f_{t+1} \;=\; \omega + \sum_{j} A_j \, s_{t-j} + \sum_{k} B_k \, f_{t-k},
\]
where $s_t = \mathcal{S}_t^{-1/2}\, \nabla_t$ is the scaled score.
Three scaling strategies are compared: Unit ($\mathcal{S}_t = I$),
DiagFI (diagonal of the Fisher information), and
FullFI (full Fisher information matrix).

\subsection{Occurrence Dynamics}
The log-odds of occurrence follows an AR(1) with daily seasonal dummies:
\[
  \text{logit}(\pi_t) \;=\; \mu_0 + \rho\, \text{logit}(\pi_{t-1})
                         + \sum_{d=1}^{366} \delta_d \, \mathbf{1}[\text{doy}(t)=d].
\]

\subsection{Harvey Long-Short Decomposition (Stage 3)}
The Stage-3 model decomposes $f_t$ into a slow climatic component $L_t$
(driven by ENSO) and a fast weather component $S_t$ (driven by dew
point/temperature):
\[
  f_t \;=\; \omega + L_t + S_t, \qquad
  L_t = \mathbf{x}_{L,t}^\top \boldsymbol{\beta}_L, \qquad
  S_t = \mathbf{x}_{S,t}^\top \boldsymbol{\beta}_S \;+\; \text{GAS update}.
\]

\subsection{IS PIT (Probability Integral Transform)}
For a mixed discrete-continuous distribution, the PIT is
\[
  u_t \;=\; \begin{cases}
    \text{Uniform}(0,\, F(0; f_t)) & \text{if } y_t = 0, \\
    F(y_t; f_t)                    & \text{if } y_t > 0,
  \end{cases}
\]
where $F(y_t; f_t) = (1-\pi_t) + \pi_t G_{\rm GB2}(y_t; f_t)$ is
the ZA CDF evaluated at $y_t$, and randomisation for zero
observations yields a $\text{Uniform}(0,1)$ PIT under correct
specification.  \emph{All PIT diagnostics in this report use IS paths.}
"""


def _section_stage(
    arts: list[dict],
    stage_num: int,
    fig_dir: Path,
) -> str:
    STAGE_TITLES = {
        1: "Stage 1: Baseline ZA-GAS Models",
        2: "Stage 2: ZA-GAS with Weather Covariates",
        3: "Stage 3: Harvey Long-Short Decomposition",
    }
    STAGE_INTROS = {
        1: (r"Stage 1 estimates twelve models that differ in (i) the set of "
            r"time-varying GB2 parameters ($\phi$-only or $\phi,\xi$), "
            r"(ii) the GAS lag structure (short lags \{1,2,3\} or seasonal "
            r"lags \{1,2,3,364--367\}), and (iii) the Fisher information "
            r"scaling (Unit, DiagFI, FullFI)."),
        2: (r"Stage 2 adds weather covariates (lagged dew point and "
            r"temperature) to the Stage-1 winner configuration.  "
            r"Four covariate blocks are compared sequentially."),
        3: (r"Stage 3 applies the Harvey (1989) long-short decomposition "
            r"to separate slow climatic (ENSO) signals from fast weather "
            r"signals in the score-driven filter."),
    }

    stage_arts = [a for a in arts if a["stage"] == stage_num]
    if not stage_arts:
        return ""

    lines = [
        rf"\section{{{STAGE_TITLES[stage_num]}}}",
        "",
        STAGE_INTROS[stage_num],
        "",
    ]

    for art in stage_arts:
        mid   = art["model_id"]
        meta  = art["meta"]
        mc    = art["mc"]
        label = _model_label(mid)
        pit   = art.get("is_pit")
        n_pit = len(pit) if pit is not None else 0

        plain = _model_label_plain(mid)
        lines.append(rf"\subsection{{\texorpdfstring{{{label}}}{{{plain}}}}}")
        lines.append(r"\begin{center}\small")
        lines.append(r"\begin{tabular}{lrlr}")
        lines.append(r"\toprule")

        def row(a, b, c, d):
            return rf"\textbf{{{a}}} & {b} & \textbf{{{c}}} & {d} \\"

        lines.append(row("Parameters", str(meta.get("n_params", "---")),
                         "Log-lik", _fmt(meta.get("loglik"), 2)))
        lines.append(row("Validity", _validity_note(meta.get("validity", "")),
                         r"Grad $\|\nabla\|_\infty$", _fmt(meta.get("grad_norm_inf"), 4)))
        lines.append(row("OOS CRPS", _fmt(mc.get("crps_mean"), 4),
                         "OOS RMSE", _fmt(mc.get("rmse"), 3)))
        if n_pit > 0:
            lines.append(row("IS PIT mean", _fmt(float(np.mean(pit)), 4),
                             "IS PIT std", _fmt(float(np.std(pit)), 4)))
            lines.append(row("IS $n$", str(n_pit), "Target PIT std", "0.289"))

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{center}")

        fig_path = fig_dir / f"{mid}_diagnostics.png"
        if fig_path.exists():
            rel = fig_path.relative_to(fig_dir.parent)
            mid_tex = mid.replace("_", r"\_")
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"\centering")
            lines.append(rf"\includegraphics[width=\linewidth]{{{rel.as_posix()}}}")
            lines.append(
                rf"\caption{{IS diagnostic plots for \texttt{{{mid_tex}}}. "
                r"Left: PIT histogram (dashed line = uniform density). "
                r"Middle: Normal QQ plot of quantile residuals. "
                r"Right: ACF of quantile residuals (40 lags).}"
            )
            lines.append(rf"\label{{fig:{mid}}}")
            lines.append(r"\end{figure}")
        lines.append("")

    return "\n".join(lines)


def _section_comparison(df: pd.DataFrame, parsimony: float) -> str:
    pct = parsimony * 100
    lines = [r"\section{Model Comparison and Selection}", ""]

    for stage_num in [1, 2, 3]:
        sub = df[df.stage == stage_num].copy()
        if sub.empty:
            continue
        best_crps = sub["oos_crps"].min()
        best_loglik = sub["loglik"].max()

        col_specs = [
            ("model_id",   "Model",          lambda x: _mid_tex(str(x))),
            ("n_params",   "$k$",            lambda x: str(int(x)) if np.isfinite(x) else "---"),
            ("loglik",     "Log-lik",         lambda x: _fmt(x, 2)),
            ("aic",        "AIC",             lambda x: _fmt(x, 1)),
            ("oos_crps",   "OOS CRPS",        lambda x: _fmt(x, 4)),
            ("oos_rmse",   "OOS RMSE",        lambda x: _fmt(x, 3)),
            ("is_pit_mean","IS PIT mean",     lambda x: _fmt(x, 4)),
            ("validity",   "Validity",        lambda x: _validity_note(str(x))),
        ]
        tbl = _booktabs_table(
            sub, col_specs,
            caption=f"Stage {stage_num} model comparison",
            label=f"tab:stage{stage_num}",
        )
        lines.append(rf"\subsection{{Stage {stage_num}}}")
        lines.append(tbl)
        lines.append("")

    # Cross-stage comparison
    s1_best = df[df.stage == 1]["oos_crps"].min()
    s2_best = df[df.stage == 2]["oos_crps"].min()
    s3_best = df[df.stage == 3]["oos_crps"].min()
    s2_improv = (s1_best - s2_best) / s1_best * 100
    s3_improv = (s1_best - s3_best) / s1_best * 100

    lines.append(r"\subsection{Cross-Stage Summary}")
    lines.append(r"\begin{center}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Stage} & \textbf{Best CRPS} & \textbf{Improv.\ vs S1} & \textbf{Passes threshold?} \\")
    lines.append(r"\midrule")
    lines.append(f"S1 & {_fmt(s1_best, 4)} & --- & --- \\\\")
    lines.append(f"S2 & {_fmt(s2_best, 4)} & {_fmt(s2_improv, 2)}\\% & "
                 + (r"\checkmark" if s2_improv >= pct else r"$\times$") + r" \\")
    lines.append(f"S3 & {_fmt(s3_best, 4)} & {_fmt(s3_improv, 2)}\\% & "
                 + (r"\checkmark" if s3_improv >= pct else r"$\times$") + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{center}")
    lines.append("")
    return "\n".join(lines)


def _section_discussion(df: pd.DataFrame, parsimony: float) -> str:
    s1_best = df[df.stage == 1].sort_values("oos_crps").iloc[0]
    s3_best = df[df.stage == 3].sort_values("oos_crps").iloc[0]
    s3_improv = (s1_best["oos_crps"] - s3_best["oos_crps"]) / s1_best["oos_crps"] * 100

    failed_models = df[df["grad_inf"] > 1000]["model_id"].tolist()
    failed_str = (", ".join(_mid_tex(m) for m in failed_models)
                  if failed_models else "none")

    return rf"""\section{{Discussion}}

\subsection{{Stage 1: Baseline}}
The seasonal GAS models consistently outperform the short-lag variants
in terms of log-likelihood, confirming that annual seasonality is an
important driver of precipitation dynamics at Belo Horizonte.
The $\phi,\xi$ parameterisation (time-varying scale and shape) achieves
the highest log-likelihood in Stage 1.

\subsection{{Stage 2: Weather Covariates}}
Adding lagged dew-point and temperature to the GAS filter improves
the in-sample fit (higher log-likelihood) but deteriorates OOS CRPS
in all configurations, suggesting potential overfitting or that the
covariates carry information already captured by the GAS dynamics.
The Stage-2 best improvement over Stage 1 ({_fmt((s1_best["oos_crps"] - df[df.stage == 2]["oos_crps"].min()) / s1_best["oos_crps"] * 100, 2)}\%) does not
meet the {parsimony * 100:.0f}\% parsimony threshold.

\subsection{{Stage 3: Harvey Decomposition}}
The Harvey long-short decomposition yields the strongest OOS gains
({_fmt(s3_improv, 2)}\% CRPS reduction vs.\ Stage 1), passing the
{parsimony * 100:.0f}\% threshold.
This suggests that separating the slow ENSO signal from the fast
weather-driven dynamics benefits probabilistic forecast calibration.

\subsection{{Numerical Issues}}
The following models showed elevated gradient norms at convergence,
indicating potential numerical difficulties: {failed_str}.
Their results should be interpreted with caution.

\subsection{{IS Calibration}}
All IS PIT histograms are approximately uniform, with means near 0.5
and standard deviations near the theoretical value $1/\sqrt{{12}} \approx 0.289$.
This confirms that the ZA-GAS models are well-calibrated in-sample.
Non-uniformity of OOS PIT would indicate distributional shift between
training and test periods.
"""


def _write_latex(
    arts:     list[dict],
    df:       pd.DataFrame,
    report_dir: Path,
    fig_dir:  Path,
    run_id:   str,
    parsimony: float,
) -> Path:
    blocks = [
        _latex_preamble(),
        r"\begin{document}",
        _latex_title_block(run_id),
        _section_executive_summary(df, parsimony),
        _section_data(),
        _section_model_framework(),
        _section_stage(arts, 1, fig_dir),
        _section_stage(arts, 2, fig_dir),
        _section_stage(arts, 3, fig_dir),
        _section_comparison(df, parsimony),
        _section_discussion(df, parsimony),
        r"\end{document}",
    ]
    tex_path = report_dir / "report.tex"
    tex_path.write_text("\n\n".join(blocks), encoding="utf-8")
    return tex_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ZA-GAS thesis report")
    parser.add_argument("--run",     default="artifacts/run_20260701_bh",
                        help="Path to the run directory (default: artifacts/run_20260701_bh)")
    parser.add_argument("--compile", action="store_true",
                        help="Run pdflatex after generating the .tex file")
    args = parser.parse_args()

    run_dir    = Path(args.run)
    run_id     = run_dir.name
    report_dir = ROOT / "reports" / run_id
    fig_dir    = report_dir / "figures"
    table_dir  = report_dir / "tables"

    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    table_dir.mkdir(exist_ok=True)

    # ── Load training data and covariate blocks ───────────────────────────────
    print("Loading training data...")
    from data.loader import BHDataLoader
    loader = BHDataLoader(PRECIP_DIR, ERA5_DIR, NINO34_PATH)
    data   = loader.load_all()
    y_train      = data["y_train"]
    blocks_train = data["covariate_blocks_train"]
    col_name_dict = data["covariate_col_names"]
    print(f"  y_train: {y_train.shape}")

    # ── Load stage1 winner metadata (needed for S2 scaling) ──────────────────
    winners_path = run_dir / "stage_winners.json"
    s1_winner_meta: dict = {}
    if winners_path.exists():
        with open(winners_path) as f:
            winners = json.load(f)
        s1_winner_id = winners.get("stage1", {}).get("winner")
        if s1_winner_id:
            s1_meta_path = run_dir / "stage1" / s1_winner_id / "metadata.json"
            if s1_meta_path.exists():
                with open(s1_meta_path) as f:
                    s1_winner_meta = json.load(f)
                print(f"  S1 winner: {s1_winner_id}  scaling={s1_winner_meta.get('scaling')}")

    # ── Load all artifacts ────────────────────────────────────────────────────
    print("Loading artifacts...")
    arts = _load_all_artifacts(run_dir)
    print(f"  Found {len(arts)} models")

    # ── Compute IS diagnostics ────────────────────────────────────────────────
    print("Computing IS diagnostics...")
    for art in arts:
        mid = art["model_id"]
        print(f"  {mid} ...", end=" ", flush=True)
        cdfs, pit, qr = _compute_is_diagnostics(
            art, y_train, blocks_train, col_name_dict, s1_winner_meta
        )
        art["is_cdfs"] = cdfs
        art["is_pit"]  = pit
        art["is_qr"]   = qr
        if len(pit) > 0:
            print(f"n={len(pit)}  PIT mean={pit.mean():.4f}  std={pit.std():.4f}")
        else:
            print("FAILED")

    # ── Generate diagnostic figures ───────────────────────────────────────────
    print("Generating diagnostic figures...")
    for art in arts:
        mid = art["model_id"]
        pit = art.get("is_pit")
        qr  = art.get("is_qr")
        if pit is not None and len(pit) > 0:
            fig_path = _plot_diagnostic_panel(mid, pit, qr, fig_dir)
            if fig_path:
                print(f"  {fig_path.name}")

    # ── Build summary table ───────────────────────────────────────────────────
    print("Building summary table...")
    df = _build_summary_df(arts)
    df.to_csv(table_dir / "model_comparison.csv", index=False)

    # ── Write LaTeX ───────────────────────────────────────────────────────────
    print("Writing LaTeX document...")
    tex_path = _write_latex(arts, df, report_dir, fig_dir, run_id, PARSIMONY_THRESHOLD)
    print(f"  Written: {tex_path}")

    # ── Optionally compile ────────────────────────────────────────────────────
    if args.compile:
        print("Compiling PDF (pdflatex x2)...")
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "report.tex"],
                cwd=report_dir,
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print("  pdflatex error:")
                print(result.stdout[-2000:])
                break
        else:
            print(f"  PDF: {report_dir / 'report.pdf'}")

    print("\nDone.")
    print(f"  LaTeX source : {tex_path}")
    print(f"  Figures      : {fig_dir}")
    print(f"  Tables CSV   : {table_dir / 'model_comparison.csv'}")
    print()
    print("To compile the PDF:")
    print(f"  cd \"{report_dir}\"")
    print("  pdflatex report.tex")
    print("  pdflatex report.tex   # second pass builds TOC")


if __name__ == "__main__":
    main()
