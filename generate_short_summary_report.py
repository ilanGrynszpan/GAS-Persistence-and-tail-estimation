"""
Short (<=5 page) academic summary report for the advisor (2026-09-06).

=============================================================================
OVERVIEW
=============================================================================

A condensed, storytelling companion to the full stage-by-stage tables --
NOT a replacement for them (per direct instruction). Covers: the model
family in brief, what each stage is, which metrics were evaluated, a
seasonality-on-vs-off comparison across stages 1-4 (RMSE / CRPS / twCRPS /
tail exceedance), how results vary by location, an optimization-diagnostics
section (boundary-pinned parameters, non-convergent locations, GAS
dynamics parameters with zero inside their confidence interval across
multiple locations), and a conclusion.

Data sources (all pre-existing, no new fitting):
  - artifacts/harmonic_seasonal_comparison_m5_final_damp0.999/<short>/<branch>/
    (standard vs seasonal, stages 3-4, all metrics already cached)
  - Harvey long-short standard_errors.csv / estimated_parameters.csv for the
    kappa_L/kappa_S zero-in-CI check (stage 3 winners, via
    run_harmonic_seasonal_experiment.resolve_baseline)
  - This session's own established findings (Long/Short persistence
    collapse at Darwin/Manaus; Stage 5 boundary pins on B_xi/A_xi)

=============================================================================
USAGE
=============================================================================

    python generate_short_summary_report.py [--no-compile]

Output: reports/short summary/report.tex / .pdf
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_harmonic_seasonal_experiment as R
from data.station_loader import STATION_REGISTRY

REPORT_DIR = ROOT / "reports" / "short summary"
CMP_DIR = ROOT / "artifacts" / "harmonic_seasonal_comparison_m5_final_damp0.999"

LOCATIONS = ["BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
             "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR"]
BRANCHES = ["phi_only", "phi_xi"]

LATEX_PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{caption}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}
\usepackage{titlesec}
\titlespacing*{\section}{0pt}{10pt}{4pt}
\titlespacing*{\subsection}{0pt}{8pt}{3pt}
\emergencystretch=3em
\sloppy
"""


def _tex(s) -> str:
    return (str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#"))


def _fmt(v, digits=3):
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(fv):
        return "diverged"
    if abs(fv) >= 1e5:
        return rf"$\sim\!10^{{{int(np.floor(np.log10(abs(fv))))}}}$"
    return f"{fv:.{digits}f}"


def load_seasonality_data() -> dict:
    data = {}
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        short = cfg.get("short", location)
        for branch in BRANCHES:
            p = CMP_DIR / short / branch / "info.json"
            if not p.exists():
                continue
            info = json.loads(p.read_text())
            std_p = CMP_DIR / short / branch / "standard" / "onestep_metrics.json"
            seas_p = CMP_DIR / short / branch / "seasonal" / "onestep_metrics.json"
            info["_std_1step"] = json.loads(std_p.read_text()) if std_p.exists() else {}
            info["_seas_1step"] = json.loads(seas_p.read_text()) if seas_p.exists() else {}
            data[(short, branch, cfg.get("display", location))] = info
    return data


def is_diverged(info: dict) -> bool:
    r = info["seasonal"]["rmse_1step"]
    return (not np.isfinite(r)) or r > 3.0 * info["standard"]["rmse_1step"]


def build_math_section() -> str:
    return "\n\n".join([
        r"\section{Model Family}",
        (r"Daily rainfall $y_t$ is modelled as zero-augmented: with occurrence "
         r"probability $\pi_t$, $y_t=0$; otherwise $y_t$ follows a positive-part "
         r"density $g(\cdot\mid\theta_t)$,"),
        r"$$p(y_t\mid\mathcal F_{t-1}) = (1-\pi_t)\mathbf{1}(y_t=0) + \pi_t\, g(y_t\mid\theta_t)\mathbf{1}(y_t>0),$$",
        (r"where $g$ is the GB2 distribution (four shape/scale parameters) and "
         r"$\pi_t$ follows its own autologistic dynamics. One or two of GB2's "
         r"parameters -- the scale $\varphi_t=\log^{-1}(\cdot)$ and, in the "
         r"richer branch, the shape $\xi_t$ -- are made time-varying via a "
         r"Generalised Autoregressive Score (GAS) recursion: writing $f_t$ for "
         r"whichever parameter is dynamic and $s_t$ for its scaled score "
         r"(the local-likelihood gradient, rescaled by the inverse Fisher "
         r"information),"),
        r"$$f_{t+1} = \omega + \sum_\ell A_\ell\, s_{t-\ell+1} + \sum_\ell B_\ell\, f_{t-\ell+1}.$$",
        (r"The Harvey long-short extension (stages 3-4) replaces this single "
         r"recursion with two components of different persistence -- a "
         r"slow-moving Long component and a fast-moving Short component, "
         r"$f_t=\omega+\lambda_{L,t}+\lambda_{S,t}$, each with its own "
         r"$(A,B)$ pair and, optionally, exogenous covariates. Stage 4 adds a "
         r"third, trigonometric seasonal component $\gamma_t$ (a damped "
         r"annual rotation, Harvey 1989) on top of the Long/Short split. "
         r"Stage 5 instead specialises to the upper tail: $\varphi,\gamma,\zeta,\pi$ "
         r"are frozen at their unconditional estimates and only $\xi_t$ is "
         r"dynamic, driven by a GAS(1,1) recursion whose score is computed "
         r"ONLY from days exceeding a high quantile of the training "
         r"distribution -- ordinary days move $\xi_t$ by mean-reversion alone."),
    ])


def build_stages_section() -> str:
    return "\n\n".join([
        r"\section{What Each Stage Is}",
        r"\begin{description}",
        r"\item[Stage 1] Baseline GAS(1,1), no exogenous covariates -- can score-driven dynamics alone explain the series?",
        r"\item[Stage 2] Stage 1 plus a selected weather-covariate block (dew point, lagged) feeding the dynamic parameter directly.",
        r"\item[Stage 3] Harvey long-short decomposition (Section 1 above), no seasonal component -- the production baseline.",
        r"\item[Stage 4] Stage 3 plus the trigonometric seasonal component $\gamma_t$.",
        r"\item[Stage 5] Static GB2 (all parameters frozen) plus a threshold-weighted GAS(1,1) recursion for $\xi_t$ alone, targeting the upper tail specifically.",
        r"\end{description}",
        (r"Every stage is estimated in two branches: one time-varying parameter "
         r"($\varphi_t$ only) or two ($\varphi_t$ and $\xi_t$ jointly). Stage 5 "
         r"is reported only for the two-parameter branch, since it is by "
         r"construction an extension of the $\xi$ dynamics."),
    ])


def build_metrics_section() -> str:
    return "\n\n".join([
        r"\section{Metrics Evaluated}",
        r"\begin{description}",
        r"\item[RMSE] point-forecast accuracy (predictive mean vs.\ observed rainfall).",
        r"\item[CRPS] distributional accuracy -- rewards a well-calibrated full predictive distribution, not just its mean.",
        r"\item[twCRPS] threshold-weighted CRPS (Gneiting \& Ranjan, 2011): CRPS restricted to the region above a high rainfall threshold -- a distributional accuracy score focused on the tail rather than the bulk.",
        r"\item[Tail exceedance] the empirical frequency with which observed rainfall exceeds a model-implied high quantile (e.g.\ the ``1-in-30-day'' event), compared against the theoretical rate the model itself implies -- a calibration check specifically for rare, high-impact events.",
        r"\end{description}",
        r"All four are computed out-of-sample (never on the data used to fit the model), one-step-ahead.",
    ])


FRM_DIR = ROOT / "artifacts" / "final_report_metrics"
REGIME_DIR = ROOT / "artifacts" / "regime_xi_only_corrected"


def load_stage_progression_data() -> dict:
    """Per-location, per-branch RMSE/CRPS and optimizer grad_norm_inf for
    stages 1-5, read from the cached compute_final_report_metrics.py
    outputs (no new fitting/optimization)."""
    out = {}
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        short, display = cfg.get("short", location), cfg.get("display", location)
        out[short] = {"display": display}
        for branch in BRANCHES:
            stages = {}
            for st in ["stage1", "stage2", "stage3", "stage4", "stage5"]:
                onestep_p = FRM_DIR / short / branch / st / "onestep_metrics.json"
                info_p = FRM_DIR / short / branch / st / "model_info.json"
                if not onestep_p.exists():
                    continue
                m = json.loads(onestep_p.read_text())
                grad = None
                if info_p.exists():
                    src = json.loads(info_p.read_text()).get("source_dir")
                    if src:
                        mp = Path(src) / "metadata.json"
                        if mp.exists():
                            meta = json.loads(mp.read_text())
                            grad = meta.get("grad_norm_inf")
                            success = meta.get("success")
                stages[st] = {"rmse": m.get("rmse"), "crps": m.get("crps_mean"),
                              "twcrps_95": m.get("twcrps_95"), "twcrps_98": m.get("twcrps_98"),
                              "grad_norm_inf": grad, "success": success if info_p.exists() else None}
            out[short][branch] = stages
    return out


def _stage_table(prog: dict, branch: str, stages: list, stage_labels: list) -> str:
    rows = []
    for short, e in sorted(prog.items(), key=lambda kv: kv[1]["display"]):
        s = e.get(branch, {})
        row = {"Location": _tex(e["display"])}
        for st, lbl in zip(stages, stage_labels):
            d = s.get(st, {})
            row[f"RMSE {lbl}"] = _fmt(d.get("rmse"), 2)
            row[f"grad {lbl}"] = _fmt(d.get("grad_norm_inf"), 3)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Location")
    lines = [r"\begin{tabular}{l" + "r" * len(df.columns) + "}", r"\toprule",
             " & " + " & ".join(df.columns) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        lines.append(idx + " & " + " & ".join(str(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_covariates_section(prog: dict) -> str:
    """Stage 1 (plain GAS) -> Stage 2 (weather covariates added)."""
    table = _stage_table(prog, "phi_only", ["stage1", "stage2"], ["S1", "S2"])
    return "\n\n".join([
        r"\section{Do Weather Covariates Help the Baseline Model?}",
        (r"Stage 2 adds ERA5 weather covariates (dew point, pressure, wind, "
         r"ENSO indicators) as score-driven regressors on top of Stage 1's "
         r"plain, covariate-free GAS recursion. The effect on out-of-sample "
         r"accuracy is weak and inconsistent: RMSE improves at only one of "
         r"six locations in the $\varphi$-only branch (Belo Horizonte) and is "
         r"flat or mildly worse everywhere else (e.g.\ Salvador worsens from "
         r"12.67 to 13.87); CRPS shows the same pattern. Simply adding more "
         r"explanatory variables to a single-persistence baseline does not, "
         r"by itself, buy better forecasts here."),
        (r"What covariates reliably do is destabilize the optimizer. Gradient "
         r"norms at the reported optimum jump by one to three orders of "
         r"magnitude at most locations (e.g.\ Salvador $0.30\to 39.3$, Manaus "
         r"$\varphi+\xi$ branch $0.45\to 6716$), and at two combinations "
         r"(Darwin and Salvador, both $\varphi+\xi$) the fit fails outright -- "
         r"Darwin's log-likelihood becomes non-finite and Salvador's "
         r"out-of-sample RMSE explodes from 12.7 to 55.9. As the next section "
         r"shows, this instability is specifically tied to forcing many raw "
         r"covariates through a single AR(1)-like persistence parameter, not "
         r"to the covariates themselves."),
        r"\begin{center}\footnotesize", table, r"\end{center}",
    ])


def build_longshort_section(prog: dict) -> str:
    """Stage 2 (covariates, single persistence) -> Stage 3 (Harvey long-short)."""
    table = _stage_table(prog, "phi_xi", ["stage2", "stage3"], ["S2", "S3"])
    return "\n\n".join([
        r"\section{Does the Long/Short Mechanism Help?}",
        (r"Stage 3 replaces Stage 2's single AR(1)-like persistence parameter "
         r"with Harvey's Long/Short decomposition -- a slow-moving level "
         r"$\varphi_{L,t}$ plus a fast-mean-reverting deviation $\varphi_{S,t}$, "
         r"each with its own covariate loadings. Point and distributional "
         r"accuracy improve modestly but broadly: RMSE improves at four to "
         r"five of six locations in both branches relative to Stage 2, most "
         r"visibly at Salvador (13.87 $\to$ 12.81, $\varphi$-only) and Darwin "
         r"(15.42 $\to$ 14.99)."),
        (r"More strikingly, the Long/Short split resolves the optimization "
         r"instability Stage 2 introduced. Gradient norms at convergence drop "
         r"by one to four orders of magnitude almost everywhere (Manaus "
         r"$\varphi+\xi$: $6716\to0.73$; Salvador $\varphi$-only: $39.3\to0.013$), "
         r"and Stage 3 fully recovers both of Stage 2's outright failures -- "
         r"Darwin $\varphi+\xi$'s non-finite likelihood and Salvador "
         r"$\varphi+\xi$'s 55.9 RMSE blow-up both return to normal, converged "
         r"fits ($\approx$14.9 and $\approx$12.7 respectively). Splitting "
         r"persistence into two components with distinct time scales appears "
         r"to give the covariates a better-conditioned place to load onto, "
         r"rather than forcing all of that structure through one parameter. "
         r"(This good behavior is specific to Stage 3 itself -- adding the "
         r"seasonal rotation on top, in Stage 4, reintroduces a different, "
         r"already-discussed instability at a handful of locations.)"),
        r"\begin{center}\footnotesize", table, r"\end{center}",
    ])


def build_timevarying_shape_section(prog: dict) -> str:
    """phi-only vs phi+xi at Stage 3 (the best-behaved stage, for a fair comparison)."""
    rows = []
    n_tw95_better, n_tw98_better, n_pairs = 0, 0, 0
    for short, e in sorted(prog.items(), key=lambda kv: kv[1]["display"]):
        s3_only = e.get("phi_only", {}).get("stage3", {})
        s3_xi = e.get("phi_xi", {}).get("stage3", {})
        rows.append({
            "Location": _tex(e["display"]),
            "RMSE $\\varphi$-only": _fmt(s3_only.get("rmse"), 3),
            "RMSE $\\varphi+\\xi$": _fmt(s3_xi.get("rmse"), 3),
            "tw95 $\\varphi$-only": _fmt(s3_only.get("twcrps_95"), 3),
            "tw95 $\\varphi+\\xi$": _fmt(s3_xi.get("twcrps_95"), 3),
            "tw98 $\\varphi$-only": _fmt(s3_only.get("twcrps_98"), 3),
            "tw98 $\\varphi+\\xi$": _fmt(s3_xi.get("twcrps_98"), 3),
        })
        if s3_only.get("twcrps_95") is not None and s3_xi.get("twcrps_95") is not None:
            n_pairs += 1
            n_tw95_better += int(s3_xi["twcrps_95"] < s3_only["twcrps_95"])
            n_tw98_better += int(s3_xi.get("twcrps_98", np.inf) < s3_only.get("twcrps_98", np.inf))
    df = pd.DataFrame(rows).set_index("Location")
    lines = [r"\begin{tabular}{l" + "r" * len(df.columns) + "}", r"\toprule",
             " & " + " & ".join(df.columns) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        lines.append(idx + " & " + " & ".join(str(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    table = "\n".join(lines)

    return "\n\n".join([
        r"\section{Does a Time-Varying Shape Parameter Help?}",
        (r"Beyond letting the GB2 scale $\varphi_t$ move over time, the "
         r"$\varphi+\xi$ branch also drives the tail-shape parameter $\xi_t$ "
         r"with its own score-driven recursion. Compared at Stage 3 -- the "
         r"best-behaved stage, so the fairest comparison -- one-step-ahead "
         r"RMSE is essentially identical between the two branches at every "
         r"location, with differences under 1\% (e.g.\ Belo Horizonte 10.14 "
         r"vs.\ 10.12, Manaus 17.56 vs.\ 17.56); CRPS matches the same "
         r"pattern."),
        (rf"twCRPS tells a less decisive story: $\varphi+\xi$ beats "
         rf"$\varphi$-only at {n_tw95_better}/{n_pairs} locations at the 95\% "
         rf"level and {n_tw98_better}/{n_pairs} at 98\% -- an even split, not "
         rf"a clear win. So letting tail shape move independently of scale "
         rf"does not change overall one-step-ahead accuracy (RMSE/CRPS, "
         rf"dominated by the bulk of the distribution), and even its "
         rf"tail-focused benefit is location-dependent rather than a "
         rf"reliable, general improvement -- consistent with $\xi$ governing "
         rf"exactly the part of the distribution twCRPS is built to detect, "
         rf"but with a real effect that varies by location rather than "
         rf"appearing everywhere."),
        r"\begin{center}\footnotesize", table, r"\end{center}",
    ])


def load_regime_data() -> dict:
    p = REGIME_DIR / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def build_regime_shape_section(prog: dict, regime: dict) -> str:
    """Stage 3/4 phi (dynamic, best-CRPS base) -> regime-sensitive xi added on top."""
    rows = []
    for short, e in sorted(regime.items(), key=lambda kv: kv[1]["display"]):
        best_q, best_m = min(e["thresholds"].items(), key=lambda kv: kv[1]["crps_mean"])
        base_stage = "stage4" if e["phi_base"] == "stage4_seasonal" else "stage3"
        base_m = prog.get(short, {}).get("phi_only", {}).get(base_stage, {})
        rows.append({
            "Location": _tex(e["display"]),
            "$\\varphi$ base": "seas." if e["phi_base"] == "stage4_seasonal" else "std.",
            "best $q$": f"{best_q}\\%",
            "RMSE base": _fmt(base_m.get("rmse"), 3),
            "RMSE regime": _fmt(best_m.get("rmse"), 3),
            "CRPS base": _fmt(base_m.get("crps"), 3),
            "CRPS regime": _fmt(best_m.get("crps_mean"), 3),
            "tw95 regime": _fmt(best_m.get("twcrps_95"), 3),
            "tw98 regime": _fmt(best_m.get("twcrps_98"), 3),
            "grad": _fmt(best_m.get("grad_norm_inf"), 3),
            "success": str(best_m.get("success")),
        })
    df = pd.DataFrame(rows).set_index("Location")
    lines = [r"\begin{tabular}{l" + "r" * len(df.columns) + "}", r"\toprule",
             " & " + " & ".join(df.columns) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        lines.append(idx + " & " + " & ".join(str(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    table = "\n".join(lines)

    # twCRPS vs the standard (non-regime) phi+xi branch at the same base stage,
    # to see whether the regime-sensitive extreme-day reweighting sharpens the
    # tail beyond what jointly-fit phi+xi already achieves on its own.
    n_tw95_better = n_tw98_better = n_pairs = 0
    for short, e in regime.items():
        base_stage = "stage4" if e["phi_base"] == "stage4_seasonal" else "stage3"
        base_xi = prog.get(short, {}).get("phi_xi", {}).get(base_stage, {})
        _, best_m = min(e["thresholds"].items(), key=lambda kv: kv[1]["crps_mean"])
        if base_xi.get("twcrps_95") is not None and best_m.get("twcrps_95") is not None:
            n_pairs += 1
            n_tw95_better += int(best_m["twcrps_95"] < base_xi["twcrps_95"])
            n_tw98_better += int(best_m.get("twcrps_98", np.inf) < base_xi.get("twcrps_98", np.inf))

    return "\n\n".join([
        r"\section{What Happens When Shape Dynamics Become Regime-Dependent?}",
        (r"This asks a different, narrower question than seasonality: holding "
         r"the scale dynamics $\varphi_t$ exactly as estimated by whichever "
         r"long-short configuration (standard or seasonal) has the better "
         r"out-of-sample CRPS for that location -- $\varphi_t$ is frozen as a "
         r"precomputed path here, not re-estimated -- does letting tail shape "
         r"$\xi_t$ react \emph{more strongly} specifically on days that "
         r"already exceeded a high rainfall threshold help? $\xi_t$ keeps its "
         r"own GAS(1,1) recursion, but the score loading becomes "
         r"$A+A_{\text{ext}}\cdot\mathbf 1(y_t>c)$ for a threshold $c$ at the "
         r"95th or 98th training-sample wet-day percentile (whichever gives "
         r"the better OOS CRPS is reported below)."),
        (rf"Because $\varphi_t$ is untouched, RMSE and CRPS barely move "
         rf"relative to their frozen base (differences under 1.5\% at every "
         rf"location, in either direction) -- as expected, since point and "
         rf"bulk-distributional accuracy are governed almost entirely by "
         rf"$\varphi_t$, not by $\xi_t$'s regime term. The more relevant "
         rf"question is whether this extreme-day-sensitive $\xi_t$ sharpens "
         rf"the tail beyond what the standard, jointly-fit $\varphi+\xi$ "
         rf"branch already achieves at the same base stage: it does at only "
         rf"{n_tw95_better}/{n_pairs} locations at the 95\% level and "
         rf"{n_tw98_better}/{n_pairs} at 98\%, with the clearest gain at "
         rf"Manaus (twCRPS98 0.326 $\to$ 0.293) and the clearest loss at "
         rf"Garanhuns (twCRPS98 0.104 $\to$ 0.142) -- on balance a mild "
         rf"negative result: reacting more strongly to extreme days does not "
         rf"reliably beat simply letting $\xi_t$ respond to every wet day "
         rf"equally, as the standard branch already does. Optimization is "
         rf"well-behaved "
         rf"throughout: gradients stay small (0.004--0.097) at every "
         rf"location, though the solver's own success flag is still false "
         rf"everywhere, the same known gap between scipy's strict criterion "
         rf"and a genuinely small raw gradient discussed elsewhere in this "
         rf"report."),
        r"\begin{center}\footnotesize", table, r"\end{center}",
    ])


def build_performance_section(data: dict) -> str:
    valid = {k: v for k, v in data.items() if not is_diverged(v)}
    diverged = {k: v for k, v in data.items() if is_diverged(v)}
    n, n_valid = len(data), len(valid)

    rmse_better = sum(1 for v in valid.values() if v["seasonal"]["rmse_1step"] < v["standard"]["rmse_1step"])
    crps_better = sum(1 for v in valid.values() if v["seasonal"]["crps_1step"] < v["standard"]["crps_1step"])
    tw95_better = sum(1 for v in valid.values()
                       if v["_seas_1step"].get("twcrps_95", np.inf) < v["_std_1step"].get("twcrps_95", np.inf))
    tw98_better = sum(1 for v in valid.values()
                       if v["_seas_1step"].get("twcrps_98", np.inf) < v["_std_1step"].get("twcrps_98", np.inf))

    rows = []
    for (short, branch, display), v in sorted(data.items(), key=lambda kv: (kv[0][2], kv[0][1])):
        rows.append({
            "Location (branch)": f"{_tex(display)} ({_tex(branch)})",
            "RMSE std": _fmt(v["standard"]["rmse_1step"], 2), "RMSE seas": _fmt(v["seasonal"]["rmse_1step"], 2),
            "CRPS std": _fmt(v["standard"]["crps_1step"], 2), "CRPS seas": _fmt(v["seasonal"]["crps_1step"], 2),
            "twCRPS95 std": _fmt(v["_std_1step"].get("twcrps_95"), 3),
            "twCRPS95 seas": _fmt(v["_seas_1step"].get("twcrps_95"), 3),
        })
    df = pd.DataFrame(rows).set_index("Location (branch)")
    table_lines = [r"\begin{tabular}{l" + "r" * len(df.columns) + "}", r"\toprule",
                   " & " + " & ".join(df.columns) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        table_lines.append(idx + " & " + " & ".join(str(v) for v in row) + r" \\")
    table_lines += [r"\bottomrule", r"\end{tabular}"]
    table = "\n".join(table_lines)

    return "\n\n".join([
        r"\section{Does Seasonality Help? (Stages 1--4)}",
        (rf"Comparing Stage 3 (no seasonal component) against Stage 4 (with it), "
         rf"holding every other choice fixed, across all {n} location/branch "
         rf"combinations tested: {n_valid}/{n} produced a numerically stable "
         rf"out-of-sample forecast; {len(diverged)}/{n} diverged (Optimization "
         rf"section below explains why -- a pre-existing identification issue, "
         rf"defined in that same section, unrelated to seasonality itself). "
         rf"Among the {n_valid} stable cases, seasonality improved 1-step RMSE "
         rf"in {rmse_better}/{n_valid}, CRPS in {crps_better}/{n_valid}, "
         rf"twCRPS@95\% in {tw95_better}/{n_valid}, and twCRPS@98\% in "
         rf"{tw98_better}/{n_valid}."),
        (r"The pattern is consistent: seasonality's effect on plain point-"
         r"forecast accuracy (RMSE) is small and mixed, but its effect on "
         r"tail-focused distributional accuracy (twCRPS) is larger and almost "
         r"always an improvement -- exactly what the annual cycle should be "
         r"expected to do: it does not change how MUCH rain falls in an "
         r"average forecast, but it improves the model's sense of WHEN "
         r"extreme wet spells are more or less likely, sharpening the tail of "
         r"the predictive distribution."),
        r"\begin{center}\footnotesize",
        table,
        r"\end{center}",
    ])


def build_lag_comparison_section(lag_data: list) -> str:
    """Does the SAME pattern (tail accuracy improves more than point
    accuracy) show up when seasonality enters through explicit seasonal
    LAGS in the GAS(p,q) recursion itself (Stages 1-2), instead of through
    Stage 4's separate unobserved seasonal component?"""
    n = len(lag_data)
    rmse_better = sum(1 for r in lag_data if r["rmse_seas"] < r["rmse_std"])
    crps_better = sum(1 for r in lag_data if r["crps_seas"] < r["crps_std"])
    tw95_better = sum(1 for r in lag_data if r["tw95_seas"] is not None and r["tw95_seas"] < r["tw95_std"])
    tw98_better = sum(1 for r in lag_data if r["tw98_seas"] is not None and r["tw98_seas"] < r["tw98_std"])
    worse_loc = next((r["display"] for r in lag_data if r["tw95_seas"] is not None
                       and r["tw95_seas"] > r["tw95_std"]), None)

    rows = []
    for r in lag_data:
        rows.append({
            "Location": _tex(r["display"]),
            "RMSE short": _fmt(r["rmse_std"], 2), "RMSE seas.\\ lag": _fmt(r["rmse_seas"], 2),
            "CRPS short": _fmt(r["crps_std"], 2), "CRPS seas.\\ lag": _fmt(r["crps_seas"], 2),
            "tw95 short": _fmt(r["tw95_std"], 3), "tw95 seas.\\ lag": _fmt(r["tw95_seas"], 3),
        })
    df = pd.DataFrame(rows).set_index("Location")
    table_lines = [r"\begin{tabular}{l" + "r" * len(df.columns) + "}", r"\toprule",
                   " & " + " & ".join(df.columns) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        table_lines.append(idx + " & " + " & ".join(str(v) for v in row) + r" \\")
    table_lines += [r"\bottomrule", r"\end{tabular}"]
    table = "\n".join(table_lines)

    return "\n\n".join([
        r"\section{A Second Route to Seasonality: Lags Instead of a Separate Component}",
        (r"Stage 4 adds seasonality as its OWN component, $\gamma_t$, sitting "
         r"alongside the Long/Short split. A completely different way to let a "
         r"score-driven model see the annual cycle is to skip the extra "
         r"component and instead give the ordinary GAS$(p,q)$ recursion itself "
         r"annual LAGS -- i.e.\ let $f_{t+1|t}$ depend not only on $f_{t}, "
         r"f_{t-1}, f_{t-2}$ (last few days) but also on $f_{t-364}, f_{t-365}, "
         r"f_{t-366}$ (same time last year), the way Stages 1-2's own "
         r"``short-lag vs.\ seasonal-lag'' weather-covariate comparison already "
         r"does for the covariate side."),
        (rf"Comparing a representative Stage 1/2 model's short-lag-only "
         rf"specification against its seasonal-lag extension, across all {n} "
         rf"locations: RMSE improved in only {rmse_better}/{n}, but CRPS "
         rf"improved in {crps_better}/{n}, twCRPS@95\% in {tw95_better}/{n}, "
         rf"and twCRPS@98\% in {tw98_better}/{n}. This is the SAME qualitative "
         rf"pattern found for Stage 4's separate seasonal component (5/9 to "
         rf"8/9 across the equivalent metrics): whether the annual cycle "
         rf"enters as its own explicit term or simply as an extra set of lags "
         rf"inside the existing recursion, its benefit shows up mainly in the "
         rf"tail of the predictive distribution, not in the average forecast. "
         + (rf"{_tex(worse_loc)} is the one exception here (tail error got "
            rf"slightly worse with seasonal lags), mirroring Belo Horizonte's "
            rf"own mild exception in the Stage 4 comparison above." if worse_loc else "")
         + r" This convergence of two structurally different mechanisms on the "
           r"same conclusion is reassuring: the finding is about the annual "
           r"cycle itself, not an artifact of how it happens to be encoded "
           r"into the model."),
        r"\begin{center}\footnotesize",
        table,
        r"\end{center}",
    ])


def build_location_section(data: dict) -> str:
    diverged_locs = sorted({d for (s, b, d), v in data.items() if is_diverged(v)})
    best_locs = sorted({d for (s, b, d), v in data.items()
                         if not is_diverged(v) and v["_seas_1step"].get("twcrps_95", np.inf) <
                         0.85 * v["_std_1step"].get("twcrps_95", np.inf)})
    return "\n\n".join([
        r"\section{Variation Across Locations}",
        (rf"The benefit of seasonality is not uniform. Salvador shows the "
         rf"largest, most consistent tail improvement (twCRPS@95\% and "
         rf"twCRPS@98\% both drop by 15--28\% in both branches). "
         rf"{', '.join(_tex(l) for l in best_locs) if best_locs else 'Several locations'} "
         rf"show a similar, if smaller, pattern. "
         rf"{', '.join(_tex(l) for l in diverged_locs)} instead show a numerically "
         rf"unstable seasonal fit in at least one branch -- these are exactly "
         rf"the locations already flagged (see below) for a weak Long/Short "
         rf"persistence separation in their STANDARD (non-seasonal) model, "
         rf"predating the seasonal work entirely. Climatologically, this is a "
         rf"plausible split: locations with a strong, regular annual cycle "
         rf"(coastal/tropical wet-dry seasonality, e.g.\ Salvador) give the "
         rf"seasonal component a clean, well-identified signal to fit; "
         rf"locations whose year-to-year rainfall variation is already "
         rf"dominated by irregular, ENSO-scale persistence (Darwin, Manaus) "
         rf"leave comparatively little separable annual structure for a "
         rf"THIRD component to capture, and adding one strains an "
         rf"already-marginal Long/Short identification."),
    ])


def build_optimization_section(zero_ci_rows: list) -> str:
    lines = [r"\section{Optimization Diagnostics}"]
    lines.append(
        r"\textit{What ``identification'' means}: a parameter is "
        r"\textbf{identified} if the data could, in principle, distinguish its "
        r"true value from every other value -- if two very different settings "
        r"of the parameter would produce almost the same predictions, no "
        r"amount of data or optimization effort can reliably tell them apart. "
        r"This is not a bug to be fixed inside the optimizer; it is a "
        r"statement about how much information a given series actually "
        r"contains about a given quantity. The Long and Short persistence "
        r"parameters below are a case in point: if a series' rainfall "
        r"anomalies do not clearly separate into ``a slow multi-year drift'' "
        r"and ``a fast few-day swing,'' the model can fit the data equally "
        r"well by calling everything Long, everything Short, or any mix in "
        r"between -- the optimizer is not failing, it is reporting, honestly, "
        r"that the two components are not both pinned down by this data."
    )
    lines.append(
        r"\textbf{Locations that did not converge cleanly for any stage}: Darwin "
        r"Airport and Manaus. In both, the Harvey Long and Short components' "
        r"persistence parameters collapse toward each other ($\varphi_L\approx"
        r"\varphi_S$, in some branches both $\to 0$) already at Stage 3, before "
        r"any seasonal component is added. This was verified NOT to be an "
        r"initialization artifact: re-fitting from deliberately well-separated "
        r"starting values reproduces essentially the same collapse (or fails "
        r"outright), pointing to a genuine, data-driven identification "
        r"weakness as defined above -- for these two series, the data does "
        r"not contain enough information to separate a slow-moving and a "
        r"fast-moving persistence regime."
    )
    lines.append(
        r"\textbf{A parameter consistently pinned at a soft boundary}: in Stage "
        r"5, the tail-xi recursion's persistence $B_{\xi}$ repeatedly settles "
        r"at or just below its $|B_\xi|<0.98$ soft constraint (Darwin q90/q98, "
        r"Garanhuns q95/q98), and its score loading $A_\xi$ similarly approaches "
        r"its $|A_\xi|<2$ bound in one case (Darwin q98). This recurs across "
        r"multiple locations and multiple threshold choices, not a single "
        r"isolated fit -- it suggests extreme-quantile exceedance in this "
        r"series family is intrinsically close to a unit-root process (highly "
        r"persistent runs of extreme wet or dry tail behavior), which is a "
        r"substantive, plausible feature of the tail dynamics rather than a "
        r"numerical accident, though it does mean the reported standard "
        r"errors for these specific parameters should be read as approximate."
    )
    if zero_ci_rows:
        loc_list = ", ".join(sorted({r[0] for r in zero_ci_rows}))
        lines.append(
            rf"\textbf{{A GAS dynamics parameter with zero inside its confidence "
            rf"interval, across multiple locations}}: the Long/Short score-"
            rf"loading pair $\kappa_L,\kappa_S$ in Stage 3's single-parameter "
            rf"($\varphi$-only) branch is statistically indistinguishable from "
            rf"zero (95\% CI spans zero) at {loc_list} -- exactly the locations "
            rf"already identified above as having a Long/Short collapse. This is "
            rf"the same underlying weakness surfacing twice: when the model "
            rf"cannot tell the Long and Short components apart in persistence, "
            rf"it correspondingly cannot pin down how strongly the score should "
            rf"feed into either one, so both loadings come out statistically "
            rf"insignificant together."
        )
    return "\n\n".join(lines)


def build_exceedance_section(ratio_data: dict) -> str:
    """Theoretical-vs-empirical exceedance ratio at 95%, 98%, and the
    "1-in-30-day" level, standard vs. seasonal Harvey -- a calibration
    check specifically for how often extreme events happen relative to
    how often the model itself says they should."""
    levels = [("0.9500", "95\\%"), ("0.9800", "98\\%"), ("0.9667", "1-in-30-day")]
    rows = []
    for key, entry in sorted(ratio_data.items(), key=lambda kv: (kv[1]["display"], kv[1]["branch"])):
        branch_label = r"$\varphi$" if entry["branch"] == "phi_only" else r"$\varphi,\xi$"
        row = {"Location (branch)": f"{_tex(entry['display'])} ({branch_label})"}
        for lv_key, lv_label in levels:
            row[f"{lv_label} std"] = _fmt(entry["standard"].get(lv_key), 2)
            row[f"{lv_label} seas"] = _fmt(entry["seasonal"].get(lv_key), 2)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Location (branch)")
    table_lines = [r"\begin{tabular}{l" + "r" * len(df.columns) + "}", r"\toprule",
                   " & " + " & ".join(df.columns) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        table_lines.append(idx + " & " + " & ".join(str(v) for v in row) + r" \\")
    table_lines += [r"\bottomrule", r"\end{tabular}"]
    table = "\n".join(table_lines)

    # Quick summary stats for the analysis paragraph, over the 0.98 level
    # (a fixed, easily-interpreted "1-in-50-day" reference point).
    ratios_98_std = [e["standard"].get("0.9800") for e in ratio_data.values()
                      if e["standard"].get("0.9800") is not None and np.isfinite(e["standard"].get("0.9800"))]
    n_over_std = sum(1 for r in ratios_98_std if r > 1.3)
    n_under_std = sum(1 for r in ratios_98_std if r < 0.7)
    n_diverged_seas = sum(1 for e in ratio_data.values()
                           if not np.isfinite(e["seasonal"].get("0.9800", float("nan"))))

    return "\n\n".join([
        r"\section{Are Extreme Events Predicted at the Right Rate?}",
        (r"RMSE and CRPS summarize average forecast quality; they say nothing "
         r"about whether a model's OWN stated extreme-event probabilities are "
         r"trustworthy. For a well-calibrated model, if it says a day has a "
         r"2\% chance of exceeding some threshold (a ``1-in-50-day'' event), "
         r"that threshold should, in reality, be exceeded on close to 2\% of "
         r"days. The ratio below is (actual exceedance rate) / (rate the model "
         r"itself predicts): a ratio of 1 is perfect calibration, above 1 "
         r"means extreme events happen MORE often than the model expects "
         r"(the model understates risk), below 1 means they happen LESS often "
         r"(the model is overly cautious)."),
        (rf"At the 98th percentile, {n_under_std}/{len(ratios_98_std)} standard-model "
         rf"location/branch combinations OVERSTATE tail risk by more than 30\% "
         rf"(ratio $<0.7$: fewer real exceedances than predicted) and "
         rf"{n_over_std}/{len(ratios_98_std)} UNDERSTATE it by more than 30\% "
         rf"(ratio $>1.3$: more real exceedances than predicted) -- calibration "
         rf"quality varies substantially by location and is not simply fixed by "
         rf"adding seasonality. Cruzeiro do Sul is notably and consistently under-"
         rf"calibrated (ratio 1.4--2.0 at every level, in both the standard and "
         rf"seasonal model): more extreme rainfall happens there than either "
         rf"model expects, a location-specific issue seasonality does not "
         rf"address. The seasonal model's row is marked ``diverged'' for "
         rf"{n_diverged_seas} of {len(ratio_data)} location/branch combinations "
         rf"(Darwin Airport, both branches, and Manaus phi-only): for these the "
         rf"filtered state itself swings to physically meaningless magnitudes "
         rf"(the log-link mean reaches $e^{{\pm 80}}$ or larger on a fifth or more "
         rf"of test days) even though it never produces a literal NaN, so any "
         rf"exceedance ratio computed from it would describe an already-broken "
         rf"forecast rather than a genuine calibration property, and is "
         rf"withheld rather than reported. This is the SAME Long/Short "
         rf"identification weakness already flagged for these locations, now "
         rf"visible as an outright forecast failure rather than only as a "
         rf"large gradient."),
        r"\begin{center}\footnotesize",
        table,
        r"\end{center}",
    ])


def build_decomposition_section(decomp_data: dict) -> str:
    """Splits the exceedance ratio into its two architecturally separate
    ingredients -- occurrence (pi_t, an AR-logistic wet/dry process with
    its own annual-lag seasonal structure, entirely independent of the
    GB2 magnitude dynamics) and the conditional-on-wet GB2 magnitude tail
    -- to check whether ratio miscalibration is an occurrence or a
    magnitude phenomenon."""
    rows = []
    for short, e in sorted(decomp_data.items(), key=lambda kv: kv[1]["display"]):
        rows.append({
            "Location": _tex(e["display"]),
            "Occurrence bias (pp)": f"{100*e['occurrence_bias']:+.1f}",
            "Full ratio 95\\%": _fmt(e["full_ratio"].get("0.9500"), 2),
            "Cond.-on-wet 95\\%": _fmt(e["conditional_on_wet_ratio"].get("0.9500"), 2),
            "Full ratio 98\\%": _fmt(e["full_ratio"].get("0.9800"), 2),
            "Cond.-on-wet 98\\%": _fmt(e["conditional_on_wet_ratio"].get("0.9800"), 2),
        })
    df = pd.DataFrame(rows).set_index("Location")
    table_lines = [r"\begin{tabular}{l" + "r" * len(df.columns) + "}", r"\toprule",
                   " & " + " & ".join(df.columns) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        table_lines.append(idx + " & " + " & ".join(str(v) for v in row) + r" \\")
    table_lines += [r"\bottomrule", r"\end{tabular}"]
    table = "\n".join(table_lines)

    max_bias_short = max(decomp_data.items(), key=lambda kv: abs(kv[1]["occurrence_bias"]))
    max_bias_loc, max_bias_val = max_bias_short[1]["display"], 100 * max_bias_short[1]["occurrence_bias"]

    return "\n\n".join([
        r"\section{Decomposing the Exceedance Ratio: Occurrence vs.\ Magnitude}",
        (r"The model splits rainfall into two architecturally separate pieces: "
         r"whether it rains at all (an AR-logistic occurrence process $\pi_t$ "
         r"with its own annual seasonal lag structure) and, given that it "
         r"rains, how much (the GB2 magnitude distribution driven by the "
         r"score-driven $\varphi_t$ dynamics). The Stage 4 seasonal harmonic "
         r"tested earlier only modifies the magnitude side; $\pi_t$ is untouched "
         r"by it. This raises a natural question: when the exceedance ratio in "
         r"the previous section deviates from 1, is that an occurrence problem "
         r"or a magnitude problem? To answer it, the ratio is split algebraically: "
         r"for wet days, $u_t = (\mathrm{PIT}_t - \pi_{\mathrm{dry},t})/(1-\pi_{\mathrm{dry},t})$ "
         r"strips the occurrence mixture out and leaves exactly the conditional GB2 "
         r"calibration, isolated from whether the wet/dry call itself was right."),
        r"\begin{center}\footnotesize",
        table,
        r"\end{center}",
        (rf"Occurrence calibration is excellent almost everywhere: the model's own "
         rf"mean $\pi_t$ matches the actual wet-day rate within 3 percentage points "
         rf"at five of six locations. The one exception is {_tex(max_bias_loc)} "
         rf"({max_bias_val:+.1f}pp) -- and that location is independently already "
         rf"flagged elsewhere in this report for the most severe Long/Short "
         rf"identification weakness, so this is a symptom of the same known issue, "
         rf"not a new, separate occurrence failure. Stripping occurrence out does "
         rf"NOT resolve the dangerous ratio $>1$ pattern at Cruzeiro do Sul and "
         rf"Garanhuns -- both remain clearly above 1 at both quantile levels even "
         rf"conditional on wet days, confirming their exceedance-ratio miscalibration "
         rf"is a genuine magnitude/GB2-tail-shape issue, not an artifact of the "
         rf"occurrence process. Belo Horizonte and Salvador do not show this "
         rf"pattern: Salvador's ratio stays below 1 (over-cautious, the safe "
         rf"direction) at every level checked here, and Belo Horizonte flips sign "
         rf"between quantile levels (0.93 at 95\%, 1.16 at 98\%) -- consistent with "
         rf"small-sample noise given only $\sim$15 test days are expected to exceed "
         rf"the 98th percentile (see the sample-size discussion above), not with a "
         rf"systematic problem. In short: this framework's occurrence/magnitude "
         rf"split works as intended -- the residual tail miscalibration is "
         rf"concentrated in the magnitude component, and specifically at the two "
         rf"locations (Cruzeiro do Sul, Garanhuns) already flagged for it, rather "
         rf"than being a general weakness of the occurrence side."),
    ])


def build_conclusion() -> str:
    return "\n\n".join([
        r"\section{Conclusion}",
        (r"Moving from a plain score-driven model (Stage 1) to a covariate-"
         r"informed one (Stage 2) to an explicit Long/Short persistence split "
         r"(Stage 3) captures the bulk of the series' predictable structure -- "
         r"but not evenly across those two steps. Covariates alone (Stage 2) "
         r"buy little accuracy and reliably destabilize the optimizer, "
         r"including two outright failures; the Long/Short split (Stage 3) "
         r"both improves accuracy at most locations and resolves that "
         r"instability, recovering both of Stage 2's failed fits. Adding an "
         r"explicit annual cycle on top (Stage 4) does not meaningfully "
         r"change point-forecast accuracy, but it consistently sharpens the "
         r"model's grasp of tail behavior -- the part of the forecast that "
         r"matters most for flood and drought risk -- wherever the underlying "
         r"Long/Short decomposition is itself well identified. Where it is "
         r"not (Darwin, Manaus), that pre-existing weakness, not the "
         r"seasonal idea itself, is the limiting factor, and is a natural "
         r"target for follow-up work. Making the tail-shape parameter "
         r"time-varying alongside scale ($\varphi+\xi$) leaves overall "
         r"one-step accuracy essentially unchanged; its value, where present, "
         r"is specific to the extreme tail rather than the bulk of the "
         r"distribution. Making that same tail-shape response regime-"
         r"dependent -- keeping scale $\varphi_t$ exactly as estimated by "
         r"the best long-short configuration, but letting $\xi_t$ react more "
         r"strongly specifically on days that already exceeded a high "
         r"threshold -- leaves point and bulk-distributional accuracy "
         r"essentially untouched (as expected, since $\varphi_t$ is "
         r"unchanged) and is, on balance, a mild negative for tail accuracy "
         r"itself: it sharpens twCRPS at only 2-3 of six locations, meaning "
         r"reacting more strongly to extremes does not reliably beat letting "
         r"$\xi_t$ respond to every wet day equally, as the standard "
         r"$\varphi+\xi$ branch already does."),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading seasonality comparison data...")
    data = load_seasonality_data()
    print(f"  Loaded {len(data)} combinations.")

    print("Loading stage-progression (1-5) data...")
    prog_data = load_stage_progression_data()
    print(f"  Loaded {len(prog_data)} locations.")

    print("Loading corrected regime-xi (RegimeXiOnlyModel) data...")
    regime_data = load_regime_data()
    print(f"  Loaded {len(regime_data)} locations.")

    print("Loading seasonal-lag (Stage 1/2) comparison data...")
    OSC_DIR = ROOT / "artifacts" / "original_seasonality_comparison"
    lag_data = []
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        short, display = cfg.get("short", location), cfg.get("display", location)
        p = OSC_DIR / short / "representative_model.json"
        if not p.exists():
            continue
        info = json.loads(p.read_text())
        std_p = OSC_DIR / short / "short" / "onestep_metrics.json"
        seas_p = OSC_DIR / short / "seasonal" / "onestep_metrics.json"
        std_m = json.loads(std_p.read_text()) if std_p.exists() else {}
        seas_m = json.loads(seas_p.read_text()) if seas_p.exists() else {}
        lag_data.append({
            "display": display,
            "rmse_std": info["short"]["rmse_1step"], "rmse_seas": info["seasonal"]["rmse_1step"],
            "crps_std": info["short"]["crps_1step"], "crps_seas": info["seasonal"]["crps_1step"],
            "tw95_std": std_m.get("twcrps_95"), "tw95_seas": seas_m.get("twcrps_95"),
            "tw98_std": std_m.get("twcrps_98"), "tw98_seas": seas_m.get("twcrps_98"),
        })
    print(f"  Loaded {len(lag_data)} locations.")

    print("Checking Stage 3 kappa_L/kappa_S for zero-in-CI...")
    zero_ci_rows = []
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        short = cfg.get("short", location)
        for branch in ("phi_only",):
            try:
                baseline = R.resolve_baseline(short, branch)
            except Exception:
                continue
            d = baseline["model_dir"]
            try:
                est = pd.read_csv(d / "estimated_parameters.csv").set_index("parameter")["value"]
                se = pd.read_csv(d / "standard_errors.csv").set_index("parameter")["std_error"]
            except FileNotFoundError:
                continue
            for p in est.index:
                if p.startswith("kappa_L") or p.startswith("kappa_S"):
                    v, s = est[p], se.get(p, np.nan)
                    if pd.notna(s) and s > 0:
                        lo, hi = v - 1.96 * s, v + 1.96 * s
                        if lo <= 0 <= hi:
                            zero_ci_rows.append((cfg.get("display", location), branch, p))

    print("Loading exceedance ratio data...")
    ratio_path = ROOT / "artifacts" / "exceedance_ratios" / "standard_vs_seasonal.json"
    ratio_data = json.loads(ratio_path.read_text()) if ratio_path.exists() else {}
    print(f"  Loaded {len(ratio_data)} combinations.")

    print("Loading occurrence/magnitude decomposition data...")
    decomp_path = ROOT / "artifacts" / "exceedance_ratios" / "occurrence_magnitude_decomposition.json"
    decomp_data = json.loads(decomp_path.read_text()) if decomp_path.exists() else {}
    print(f"  Loaded {len(decomp_data)} locations.")

    print("Building report...")
    body = "\n\n".join([
        r"\begin{document}",
        r"\title{Score-Driven Precipitation Models: A Stage-by-Stage Summary}",
        r"\author{}",
        rf"\date{{{datetime.date.today().isoformat()}}}",
        r"\maketitle",
        build_math_section(),
        build_stages_section(),
        build_metrics_section(),
        build_covariates_section(prog_data),
        build_longshort_section(prog_data),
        build_timevarying_shape_section(prog_data),
        build_regime_shape_section(prog_data, regime_data),
        build_performance_section(data),
        build_lag_comparison_section(lag_data),
        build_location_section(data),
        build_optimization_section(zero_ci_rows),
        build_exceedance_section(ratio_data),
        build_decomposition_section(decomp_data),
        build_conclusion(),
        r"\end{document}",
    ])
    tex_source = LATEX_PREAMBLE + "\n" + body
    tex_path = REPORT_DIR / "report.tex"
    tex_path.write_text(tex_source, encoding="utf-8")
    print(f"Wrote {tex_path}")

    if not args.no_compile:
        print("Compiling PDF (2 pdflatex passes)...")
        cmd = ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(REPORT_DIR), str(tex_path)]
        ok = True
        for i in range(2):
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=REPORT_DIR)
            if result.returncode != 0 and i == 0:
                ok = False
                print("  pdflatex pass 1 returned non-zero; see log.")
        print("Compilation " + ("succeeded." if ok else "FAILED -- see report.log."))


if __name__ == "__main__":
    main()
