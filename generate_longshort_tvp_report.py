"""
"What worked": one-TVP vs two-TVP Harvey long-short models, per location
(2026-09-01).

=============================================================================
OVERVIEW
=============================================================================

A focused, positive-framing companion to "long short diagnostics.pdf" and
"best models/report.pdf": for every location, both Harvey long-short
branches now have at least one reliable fit (|grad|_inf < 1) --

  * "Time-Varying Location Only"        (branch phi_only, harvey_target=[phi])
  * "Time-Varying Location + Shape"     (branch phi_xi,   harvey_target=[phi, xi])

This report presents ONLY the winning configuration at each location, for
each of the two branches: what it is, its full parameter table (estimate,
standard error, gradient), and how it compares -- in-sample residual
diagnostics and out-of-sample forecast accuracy -- against the standard
(no long-short decomposition) GAS model. Per docs/REPORTING.md, every
number is read from already-saved artifacts; no optimisation or simulation
runs here.

Data sources (all pre-computed, never re-fit):
  - artifacts/longshort_diagnostics_experiment/<run_id>/<branch>/<config>/
      estimated_parameters.csv, standard_errors.csv,
      decoded_harvey_parameters.csv, metadata.json
      (run_longshort_diagnostics.py)
  - artifacts/longshort_diagnostics_experiment/tvp_comparison/<short>/<branch>/
      best_model_info.json, {harvey,cov}_quantile_residuals.npy,
      onestep_metrics.json, multistep_metrics_{harvey,cov}.csv
      (run_longshort_tvp_comparison.py)

=============================================================================
USAGE
=============================================================================

    python generate_longshort_tvp_report.py [--no-compile]

Output:
    reports/best longshort tvps/report.tex / .pdf
    reports/best longshort tvps/figures/*.png
    reports/best longshort tvps/tables/*.csv
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import STATION_REGISTRY, mosaic_name
from diagnostics.plots import qq_location_mosaic, acf_location_mosaic
from diagnostics.multistep_forecast import DEFAULT_HORIZONS

EXP_DIR    = ROOT / "artifacts" / "longshort_diagnostics_experiment"
TVP_DIR    = EXP_DIR / "tvp_comparison"
REPORT_DIR = ROOT / "reports" / "best longshort tvps"
FIG_DIR    = REPORT_DIR / "figures"
TAB_DIR    = REPORT_DIR / "tables"

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]
BRANCHES = ["phi_only", "phi_xi"]
TVP_LABEL = {
    "phi_only": "Time-Varying Location Only",
    "phi_xi":   "Time-Varying Location + Shape",
}
MODE_DESCRIPTION = {
    "restricted":   r"persistence restricted to $0<\phi_S<\phi_L<1$ by a soft penalty",
    "free":         r"persistence restricted to $0<\phi_S<\phi_L$ (no upper bound)",
    "fixed_phiL":   r"long-run persistence fixed at $\phi_L\equiv 1$ (pure random-walk long component); only $\phi_S$ estimated",
    "sigmoid_both": r"both $\phi_L,\phi_S$ mapped through a logistic sigmoid, guaranteeing $(0,1)$ by construction",
}


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


def _fmt(v, digits: int = 3) -> str:
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
# Parameter-name prettifier
# ─────────────────────────────────────────────────────────────────────────────

TV_SYMBOL = {"phi": r"\text{loc}", "xi": r"\text{shape}"}
BLOCK_SYMBOL = {
    "omega": r"\omega", "f0": r"f_0", "kappa_L": r"\kappa_L", "phi_L": r"\phi_L",
    "L0": r"L_0", "kappa_S": r"\kappa_S", "phi_S": r"\phi_S", "S0": r"S_0",
}


def pretty_param_name(pname: str) -> str:
    if pname in ("gamma", "zeta", "xi"):
        return {"gamma": r"$\gamma$ (GB2 shape)", "zeta": r"$\zeta$ (GB2 shape)",
                "xi": r"$\xi$ (static shape)"}[pname]
    if pname == "omega0":
        return r"$\omega_0$ (occurrence intercept)"
    if pname == "rho":
        return r"$\rho$ (occurrence persistence)"
    m = re.match(r"^omega_y_(\d+)$", pname)
    if m:
        return rf"$\omega_{{y,{m.group(1)}}}$ (occurrence)"
    m = re.match(r"^(omega|f0|kappa_L|phi_L|L0|kappa_S|phi_S|S0)_(phi|xi)$", pname)
    if m:
        block, tv = m.groups()
        return f"${BLOCK_SYMBOL[block]}$ ({'location' if tv == 'phi' else 'shape'})"
    m = re.match(r"^Gamma_(L|S)_(phi|xi)_(.+)$", pname)
    if m:
        ls, tv, cov = m.groups()
        return f"$\\Gamma_{{{ls}}}$ ({'location' if tv == 'phi' else 'shape'}, {_tex(cov)})"
    return _tex(pname)


def load_param_table(run_id: str, branch: str, config: str) -> pd.DataFrame:
    """
    Natural-fit-order parameter table: pretty name, estimate, SE, quality.

    For phi_L/phi_S under a sigmoid reparameterisation, the raw fitted value
    is the pre-sigmoid coordinate, not directly interpretable -- the DECODED
    (natural-scale, 0<phi<1) value is reported instead, with the SE column
    replaced by an explicit "bounded reparam" note (the delta-method SE of a
    nonlinear sigmoid transform was not propagated here, so no number is
    fabricated in its place). For fixed_phi_L (phi_L held at a constant, not
    part of theta at all), a synthetic row is inserted showing that fixed
    value with "fixed (not estimated)" in place of an SE.
    """
    d = EXP_DIR / run_id / branch / config
    meta = json.loads((d / "metadata.json").read_text())
    est = pd.read_csv(d / "estimated_parameters.csv").set_index("parameter")["value"]
    se_path = d / "standard_errors.csv"
    se_df = pd.read_csv(se_path).set_index("parameter") if se_path.exists() else pd.DataFrame(
        columns=["std_error", "se_quality"])
    dec_path = d / "decoded_harvey_parameters.csv"
    dec = pd.read_csv(dec_path).set_index("parameter")["value"] if dec_path.exists() else pd.Series(dtype=float)

    rows = []
    for pname in est.index:
        raw_val = float(est[pname])
        se = float(se_df["std_error"][pname]) if pname in se_df.index else np.nan
        seq = str(se_df["se_quality"][pname]) if pname in se_df.index else ""
        dec_val = float(dec[pname]) if pname in dec.index else raw_val
        is_bounded = bool(re.match(r"^phi_[LS]_(phi|xi)$", pname))
        transformed = is_bounded and abs(dec_val - raw_val) > 1e-9
        rows.append({
            "parameter": pname, "display": pretty_param_name(pname),
            "estimate": dec_val if transformed else raw_val,
            "se": np.nan if transformed else se,
            "note": "bounded reparam" if transformed else seq,
        })

    fixed_phi_L = meta.get("fixed_phi_L")
    if fixed_phi_L is not None:
        # fixed_phi_L removes phi_L_{tv}'s theta slot entirely (models/harvey_gas.py
        # _block_size); reinsert it as a synthetic, non-estimated row right after
        # kappa_L_{tv} so the table still reads in the natural omega/f0/kappa_L/
        # phi_L/L0/... order for every tv this model has.
        kl_names_with_slot = {f"kappa_L_{tv}" for tv in meta.get("tv_names", [])}
        out_rows = []
        for r in rows:
            out_rows.append(r)
            if r["parameter"] in kl_names_with_slot:
                tv = r["parameter"].removeprefix("kappa_L_")
                out_rows.append({
                    "parameter": f"phi_L_{tv}", "display": pretty_param_name(f"phi_L_{tv}"),
                    "estimate": float(fixed_phi_L), "se": np.nan, "note": "fixed (not estimated)",
                })
        rows = out_rows

    df = pd.DataFrame(rows).set_index("parameter")
    return df, meta


# ─────────────────────────────────────────────────────────────────────────────
# Winning-config resolution (from tvp_comparison's best_model_info.json)
# ─────────────────────────────────────────────────────────────────────────────

def load_location_branch(location: str, branch: str) -> Optional[dict]:
    cfg = STATION_REGISTRY[location]
    short = cfg.get("short", cfg["run_id"])
    d = TVP_DIR / short / branch
    info_path = d / "best_model_info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text())
    harvey_qr = np.load(d / "harvey_quantile_residuals.npy")
    cov_qr = np.load(d / "cov_quantile_residuals.npy")
    onestep = json.loads((d / "onestep_metrics.json").read_text())
    harvey_ms = pd.read_csv(d / "multistep_metrics_harvey.csv")
    cov_ms = pd.read_csv(d / "multistep_metrics_cov.csv")
    param_df, meta = load_param_table(info["run_id"], branch, info["harvey_config"])
    return {
        "info": info, "harvey_qr": harvey_qr, "cov_qr": cov_qr, "onestep": onestep,
        "harvey_ms": harvey_ms, "cov_ms": cov_ms, "param_df": param_df, "meta": meta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Introduction (model explanation + literature)
# ─────────────────────────────────────────────────────────────────────────────

def build_introduction() -> str:
    return "\n\n".join([
        r"\chapter{Two Harvey Long-Short Models}",
        (r"Both models extend the zero-augmented GB2 score-driven (GAS) framework "
         r"of Creal, Koopman \& Lucas (2013) with a long-short (``permanent + "
         r"transitory'') decomposition of each time-varying parameter, in the "
         r"spirit of the unobserved-components / structural time series "
         r"long-run-plus-cycle decompositions described in Harvey (2013). For a "
         r"time-varying parameter $f_{j,t}$ (location or shape), the Harvey "
         r"decomposition writes"),
        r"""\begin{align*}
f_{j,t} &= \omega_j + L_{j,t} + S_{j,t} \\
L_{j,t+1} &= \phi_{L,j}\, L_{j,t} + \kappa_{L,j}\, s_{j,t} + \Gamma_{L,j}' X_{\text{long},t} \\
S_{j,t+1} &= \phi_{S,j}\, S_{j,t} + \kappa_{S,j}\, s_{j,t} + \Gamma_{S,j}' X_{\text{short},t}
\end{align*}""",
        (r"where $s_{j,t}$ is the scaled score of the predictive log-likelihood "
         r"with respect to $f_{j,t}$ (the GAS updating mechanism itself), "
         r"$L_{j,t}$ is a slow-moving, highly persistent ``long-run'' component "
         r"driven by the ENSO climate covariates ($X_{\text{long}}$: El Ni\~no / "
         r"La Ni\~na indicators), and $S_{j,t}$ is a fast-moving ``short-run'' "
         r"component driven by same-week weather ($X_{\text{short}}$: lagged dew "
         r"point). $0<\phi_{S,j}<\phi_{L,j}<1$ orders the two components by "
         r"design: the long component must decay more slowly than the short one."),
        r"\section{Time-Varying Location Only}",
        (r"Only the GB2 location parameter (log-link scale) is given the Harvey "
         r"decomposition; the shape parameters $\gamma,\zeta$ (and, for this "
         r"branch, $\xi$) are static. One dynamic parameter, $j=\text{loc}$."),
        r"\section{Time-Varying Location + Shape}",
        (r"Both the location parameter and the GB2 shape parameter $\xi$ receive "
         r"their own, independent Harvey decomposition -- two dynamic "
         r"parameters, $j\in\{\text{loc},\text{shape}\}$, each with its own "
         r"$\phi_L,\phi_S,\kappa_L,\kappa_S,\Gamma_L,\Gamma_S$. This lets the "
         r"tail heaviness of daily rainfall drift over time independently of its "
         r"typical size."),
        r"\section{Keeping the Persistence Pair Well-Behaved}",
        (r"$\phi_{L,j},\phi_{S,j}$ are persistence (autoregressive) coefficients: "
         r"mathematically, both must stay inside $(0,1)$ for either component to "
         r"be a stationary, mean-reverting process rather than a random walk. "
         r"Two constructions were used across the locations below to guarantee "
         r"this, both standard devices for keeping an autoregressive coefficient "
         r"inside its stationary region during unconstrained numerical "
         r"optimisation: a soft quadratic penalty directly on the raw "
         r"coefficient (``restricted''), and a logistic-sigmoid change of "
         r"variables $\phi = \text{sigmoid}(z)$, $z\in\mathbb{R}$, which "
         r"guarantees $\phi\in(0,1)$ by construction rather than by penalty "
         r"(``sigmoid''). One location's location-parameter model instead fixes "
         r"$\phi_L\equiv 1$ outright -- a literal random-walk long component, "
         r"as in Harvey's own local-level structural models (Harvey, 2013) -- "
         r"estimating only the short component's persistence."),
        (r"Every configuration below was selected purely by its post-fit "
         r"gradient norm $|\nabla Q(\hat\theta)|_\infty$ among all persistence "
         r"constructions and covariate initialisations tried for that "
         r"location and branch (see \texttt{reports/longshort experiment/"
         r"long short diagnostics.pdf} for the full comparison); only the "
         r"winning configuration at each location is presented here."),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Per-location parameter sections
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG_RE = re.compile(r"^(restricted|free|fixed_phiL|sigmoid_both)_gL([0-9.]+)$")


def _mode_and_gL(config: str):
    m = _CONFIG_RE.match(config)
    return (m.group(1), m.group(2)) if m else (config, "?")


def save_param_table(df: pd.DataFrame, name: str, caption: str, label: str) -> str:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(TAB_DIR / f"{name}.csv", index=False)
    # index=df["display"].to_numpy() (not the Series itself): disp's row
    # index must be the DISPLAY strings, but assigning columns from `df`
    # (indexed by the raw `parameter` names) would otherwise align on that
    # mismatched index and silently produce all-NaN columns.
    disp = pd.DataFrame(index=df["display"].to_numpy())
    disp["Estimate"] = [_fmt(v, 4) for v in df["estimate"]]
    se_col = []
    for se, note in zip(df["se"], df["note"]):
        se_txt = _fmt(se, 4) if (se is not None and np.isfinite(se)) else "--"
        se_col.append(f"{se_txt} ({_tex(note)})" if note else se_txt)
    disp["Std. Error"] = se_col
    return _latex_table(disp, caption, label)


def build_param_section(location_display: str, short: str, branch: str, data: dict) -> str:
    info, pdf = data["info"], data["param_df"]
    mode, gL = _mode_and_gL(info["harvey_config"])
    desc = MODE_DESCRIPTION.get(mode, "")
    tex = [
        rf"\subsection{{{TVP_LABEL[branch]}}}",
        (rf"Winning configuration: {desc}, $\Gamma_{{L,\text{{init}}}}={_tex(gL)}$. "
         rf"Post-fit gradient $|\nabla Q(\hat\theta)|_\infty={_fmt(info['harvey_grad_norm_inf'], 4)}$ "
         rf"(post-fit class: \texttt{{{_tex(info['harvey_validity'])}}})."),
        save_param_table(
            pdf, f"{short}_{branch}_params",
            f"{location_display}: {TVP_LABEL[branch]} -- parameter estimates and standard errors.",
            f"tab:params-{short}-{branch}",
        ),
    ]
    return "\n\n".join(tex)


def build_locations_chapter(all_data: Dict) -> str:
    parts = [r"\chapter{What Worked, Location by Location}"]
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        display, short = cfg.get("display", location), cfg.get("short", cfg["run_id"])
        parts.append(rf"\section{{{_tex(display)}}}")
        for branch in BRANCHES:
            d = all_data.get((location, branch))
            if d is None:
                parts.append(rf"\subsection{{{TVP_LABEL[branch]}}}" + "\n\nNot available.")
                continue
            parts.append(build_param_section(display, short, branch, d))
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison chapter: gradients + OOS RMSE/CRPS, 1-TVP vs 2-TVP vs baseline
# ─────────────────────────────────────────────────────────────────────────────

def onestep_comparison_table(d_only: dict, d_xi: dict) -> pd.DataFrame:
    rows = {}
    # NOTE: labels below are passed through _safe_header() -> _tex() once,
    # inside _latex_table -- do not pre-escape here (leads to double-escaped
    # underscores, "\\_" instead of "\_").
    for label, d, key in [
        ("Time-Varying Location Only", d_only, "harvey"),
        (rf"-> baseline ({d_only['info']['comparison_model_id']})", d_only, "cov"),
        ("Time-Varying Location + Shape", d_xi, "harvey"),
        (rf"-> baseline ({d_xi['info']['comparison_model_id']})", d_xi, "cov"),
    ]:
        m = d["onestep"][key]
        rows[label] = {"RMSE": m["rmse"], "CRPS": m["crps_mean"]}
    return pd.DataFrame(rows).T


def multistep_comparison_table(d: dict, branch: str) -> pd.DataFrame:
    hm = d["harvey_ms"].set_index("horizon")
    cm = d["cov_ms"].set_index("horizon")
    rows = {}
    for h in DEFAULT_HORIZONS:
        rows[f"h={h}"] = {
            f"{TVP_LABEL[branch]} RMSE": hm.loc[h, "rmse"] if h in hm.index else np.nan,
            f"{TVP_LABEL[branch]} CRPS": hm.loc[h, "crps"] if h in hm.index else np.nan,
            "Baseline RMSE": cm.loc[h, "rmse"] if h in cm.index else np.nan,
            "Baseline CRPS": cm.loc[h, "crps"] if h in cm.index else np.nan,
        }
    return pd.DataFrame(rows).T


def save_metric_table(df: pd.DataFrame, name: str, caption: str, label: str) -> str:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB_DIR / f"{name}.csv")
    disp = df.copy().astype(object)
    for col in disp.columns:
        disp[col] = df[col].apply(_fmt)
    return _latex_table(disp, caption, label)


def build_comparison_chapter(all_data: Dict) -> str:
    parts = [
        r"\chapter{Forecast Comparison: One TVP vs.\ Two TVPs vs.\ the Standard GAS Baseline}",
        (r"The 1-step table is an exact, non-simulated rolling forecast over the "
         r"full held-out test period. The multi-step table "
         rf"($h\in\{{{','.join(str(h) for h in DEFAULT_HORIZONS)}\}}$ days) is a "
         r"Monte Carlo simulation: from each forecast origin, the model's "
         r"filtered state is replayed exactly on real data up to $t_0$, then "
         r"400 independent continuations are drawn and scored by RMSE of the "
         r"simulated mean and Monte Carlo CRPS, over a stride of 50 forecast "
         r"origins per location (identical origins for every model at a given "
         r"location, so the comparison is paired). Exogenous weather/ENSO "
         r"covariates over the forecast horizon are assumed known."),
    ]
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        display, short = cfg.get("display", location), cfg.get("short", cfg["run_id"])
        d_only, d_xi = all_data.get((location, "phi_only")), all_data.get((location, "phi_xi"))
        parts.append(rf"\section{{{_tex(display)}}}")
        if d_only is None or d_xi is None:
            parts.append("Not available.")
            continue
        parts.append(save_metric_table(
            onestep_comparison_table(d_only, d_xi), f"{short}_onestep_cmp",
            f"{_tex(display)}: full-test-set 1-step rolling OOS metrics.",
            f"tab:onestep-cmp-{short}",
        ))
        parts.append(save_metric_table(
            multistep_comparison_table(d_only, "phi_only"), f"{short}_multistep_1tvp",
            f"{_tex(display)}: Time-Varying Location Only vs.\\ its baseline -- multi-step OOS.",
            f"tab:ms-1tvp-{short}",
        ))
        parts.append(save_metric_table(
            multistep_comparison_table(d_xi, "phi_xi"), f"{short}_multistep_2tvp",
            f"{_tex(display)}: Time-Varying Location + Shape vs.\\ its baseline -- multi-step OOS.",
            f"tab:ms-2tvp-{short}",
        ))
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Mosaics (4: ACF/QQ x one-TVP/two-TVP)
# ─────────────────────────────────────────────────────────────────────────────

def build_mosaics(all_data: Dict) -> Dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    display_order = [mosaic_name(loc) for loc in LOCATIONS]
    out = {}
    for branch, tag in [("phi_only", "1tvp"), ("phi_xi", "2tvp")]:
        qr_by_location = {mosaic_name(loc): (all_data[(loc, branch)]["harvey_qr"]
                                              if all_data.get((loc, branch)) else None)
                           for loc in LOCATIONS}
        fig_qq = qq_location_mosaic(
            qr_by_location, display_order,
            suptitle=f"{TVP_LABEL[branch]}: in-sample QQ plot of quantile residuals")
        qq_path = FIG_DIR / f"qq_mosaic_{tag}.png"
        fig_qq.savefig(qq_path, dpi=150)
        plt.close(fig_qq)

        fig_acf = acf_location_mosaic(
            qr_by_location, display_order, n_lags=400,
            suptitle=f"{TVP_LABEL[branch]}: in-sample 400-lag ACF of quantile residuals")
        acf_path = FIG_DIR / f"acf_mosaic_{tag}.png"
        fig_acf.savefig(acf_path, dpi=150)
        plt.close(fig_acf)

        out[f"qq_{tag}"] = qq_path.relative_to(REPORT_DIR).as_posix()
        out[f"acf_{tag}"] = acf_path.relative_to(REPORT_DIR).as_posix()
    return out


def build_figures_chapter(fig_paths: Dict[str, str]) -> str:
    parts = [r"\chapter{In-Sample Residual Diagnostics}"]
    for branch, tag in [("phi_only", "1tvp"), ("phi_xi", "2tvp")]:
        parts.append(rf"\section{{{TVP_LABEL[branch]}}}")
        parts += [
            r"\begin{figure}[H]", r"\centering",
            rf"\includegraphics[width=\textwidth]{{{fig_paths[f'qq_{tag}']}}}",
            rf"\caption{{QQ plot of in-sample quantile residuals, {TVP_LABEL[branch]}, best model per location.}}",
            rf"\label{{fig:qq-{tag}}}", r"\end{figure}",
            r"\begin{figure}[H]", r"\centering",
            rf"\includegraphics[width=\textwidth]{{{fig_paths[f'acf_{tag}']}}}",
            rf"\caption{{400-lag ACF of in-sample quantile residuals, {TVP_LABEL[branch]}, best model per location.}}",
            rf"\label{{fig:acf-{tag}}}", r"\end{figure}",
        ]
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Final summary table
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_chapter(all_data: Dict) -> str:
    rows = {}
    for location in LOCATIONS:
        display = STATION_REGISTRY[location].get("display", location)
        d_only, d_xi = all_data.get((location, "phi_only")), all_data.get((location, "phi_xi"))
        rows[display] = {
            "Time-Varying Location Only: config": d_only["info"]["harvey_config"] if d_only else "--",
            "grad": d_only["info"]["harvey_grad_norm_inf"] if d_only else np.nan,
            "Time-Varying Location + Shape: config": d_xi["info"]["harvey_config"] if d_xi else "--",
            "grad ": d_xi["info"]["harvey_grad_norm_inf"] if d_xi else np.nan,
        }
    df = pd.DataFrame(rows).T
    df.columns = ["Location-only config", r"$|\nabla Q|_\infty$",
                  "Location+shape config", r"$|\nabla Q|_\infty$ "]
    tex_table = save_metric_table(
        df, "summary_gradients",
        "Gradient achieved by the best Time-Varying Location Only and Time-Varying Location + Shape model, by location.",
        "tab:summary-gradients",
    )
    return "\n\n".join([r"\chapter{Summary}", tex_table])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading per-location, per-branch artifacts...")
    all_data = {}
    for location in LOCATIONS:
        for branch in BRANCHES:
            all_data[(location, branch)] = load_location_branch(location, branch)
    missing = [k for k, v in all_data.items() if v is None]
    if missing:
        print(f"  WARNING: missing analysis for: {missing}")

    print("Building introduction...")
    intro_tex = build_introduction()
    print("Building per-location parameter sections...")
    locations_tex = build_locations_chapter(all_data)
    print("Building comparison chapter...")
    comparison_tex = build_comparison_chapter(all_data)
    print("Building mosaics...")
    fig_paths = build_mosaics(all_data)
    figures_tex = build_figures_chapter(fig_paths)
    print("Building summary chapter...")
    summary_tex = build_summary_chapter(all_data)

    references_tex = "\n\n".join([
        r"\chapter*{References}",
        r"\begin{itemize}",
        r"\item Creal, D., Koopman, S.J., \& Lucas, A. (2013). Generalized "
        r"Autoregressive Score Models with Applications. \textit{Journal of "
        r"Applied Econometrics}, 28(5), 777--795.",
        r"\item Harvey, A.C. (2013). \textit{Dynamic Models for Volatility and "
        r"Heavy Tails: With Applications to Financial and Economic Time "
        r"Series}. Cambridge University Press.",
        r"\end{itemize}",
    ])

    today = datetime.date.today().isoformat()
    body = "\n\n".join([
        r"\begin{document}",
        r"\title{What Worked: Time-Varying Location Only vs.\ Time-Varying Location + Shape}",
        rf"\author{{Generated by \texttt{{generate\_longshort\_tvp\_report.py}}}}",
        rf"\date{{{today}}}",
        r"\maketitle",
        r"\tableofcontents",
        intro_tex,
        locations_tex,
        comparison_tex,
        figures_tex,
        summary_tex,
        references_tex,
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
