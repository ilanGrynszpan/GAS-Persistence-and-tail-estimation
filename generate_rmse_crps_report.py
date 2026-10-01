"""
2026-09-11: four RMSE/CRPS tables (1-TVP / 2-TVP x RMSE/CRPS), rolling
1-step-ahead OOS, best lag-set/covariate-set option per stage (Stage 5
excluded), one column per stage.

"Best option" uses the SAME selection rule as every other 2026-09-11 report
(generate_stage_comparison_report.py / generate_ml_comparison_report.py):
gradient tier, then per-parameter convergence fraction, then OOS CRPS
(1% relative tolerance), then OOS RMSE -- applied independently per
location, per branch, per stage:
  Stage 1  : best of the 4 GAS-lag sets (no covariates)
  Stage 2  : best of the 4 GAS-lag sets x 4 covariate sets (16 combos)
  Stage 3/4/4S : best of the 5 Harvey covariate configs (no lag choice there)

Reads ONLY artifacts/stage_comparison_20260911/metrics.json -- no fitting.
Writes reports/reportRMSECRPS/report.tex (+ .pdf).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import generate_ml_comparison_report as G  # reuse select()/m()/metrics.json/labels

ROOT = Path(__file__).parent
REP = ROOT / "reports" / "reportRMSECRPS"
SHORTS = G.SHORTS
DISP = G.DISP_LONG
STAGES = ["stage1", "stage2", "stage3", "stage4", "stage4S"]
STAGE_LBL = {"stage1": "Stage 1", "stage2": "Stage 2", "stage3": "Stage 3",
             "stage4": "Stage 4", "stage4S": "Stage 4S"}


def stage1_best(short, branch):
    cands = [(("stage1", lk, branch), G.m(short, "stage1", lk, branch)) for lk in G.GAS_LAG_SETS]
    return G.select(cands)


def stage2_best(short, branch):
    cands = [(("stage2", f"{lk}_{cov}", branch), G.m(short, "stage2", f"{lk}_{cov}", branch))
             for cov in G.STAGE2_COV for lk in G.GAS_LAG_SETS]
    return G.select(cands)


def harvey_best(short, stage, branch):
    cands = [((stage, cov, branch), G.m(short, stage, cov, branch)) for cov in G.HARVEY_COV5]
    return G.select(cands)


def best_for_stage(short, stage, branch):
    if stage == "stage1":
        return stage1_best(short, branch)
    if stage == "stage2":
        return stage2_best(short, branch)
    return harvey_best(short, stage, branch)


def _fmt_cfg(ident):
    stage, cfg, branch = ident
    return cfg.replace("_", r"\_")


def _f(v, d=3):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(x):
        return "div"
    return f"{x:.{d}f}"


def metric_table(branch, metric_key, caption, label, decimals=3):
    rows = []
    for s in SHORTS:
        row = {"Location": DISP[s]}
        for stage in STAGES:
            ident, mm = best_for_stage(s, stage, branch)
            row[STAGE_LBL[stage]] = _f(mm.get(metric_key) if mm else None, decimals)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Location")
    cols = [STAGE_LBL[s] for s in STAGES]
    body = [r"\begin{table}[H]\centering\footnotesize\setlength{\tabcolsep}{5pt}",
            rf"\caption{{{caption}}}\label{{{label}}}",
            r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
            "Location & " + " & ".join(cols) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        body.append(str(idx) + " & " + " & ".join(str(v) for v in row[cols]) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(body)


def config_table(branch, caption, label):
    rows = []
    for s in SHORTS:
        row = {"Location": DISP[s]}
        for stage in STAGES:
            ident, mm = best_for_stage(s, stage, branch)
            row[STAGE_LBL[stage]] = _fmt_cfg(ident) if ident else "--"
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Location")
    cols = [STAGE_LBL[s] for s in STAGES]
    body = [r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{4pt}",
            rf"\caption{{{caption}}}\label{{{label}}}",
            r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
            "Location & " + " & ".join(cols) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        body.append(str(idx) + " & " + " & ".join(str(v) for v in row[cols]) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(body)


def build():
    REP.mkdir(parents=True, exist_ok=True)

    pre = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,amssymb,booktabs}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}\usepackage{float}
\emergencystretch=3em\sloppy
"""

    intro = (
        r"\section*{Methodology}"
        r"Each cell is the rolling 1-step-ahead out-of-sample metric of that location's BEST "
        r"lag-set/covariate-set option at that stage (Stage 5 excluded), selected independently "
        r"per location by the same rule used throughout: gradient tier ($<1$, then $<100$, then "
        r"anything else), then the fraction of individually-converged parameters, then lowest OOS "
        r"CRPS (1\% relative tolerance), then lowest OOS RMSE. Stage 1 selects among the 4 "
        r"GAS-lag sets (no covariates); Stage 2 among all 4 lag sets $\times$ 4 covariate sets "
        r"(dew, dew+temp, dew+ENSO, dew+temp+ENSO); Stage 3/4/4S among the 5 Harvey covariate "
        r"configs (none, dew, dew+temp, dew+ENSO, dew+temp+ENSO) -- no lag choice there. "
        r"Stage 3/4/5 use the sigmoid-bounded $\varphi_L$ parameterisation "
        r"(\texttt{phi\_L\_max=1.0}, 2026-09-11 fix); Stage 1/2/4S are unaffected. "
        r"\emph{div} marks a diverged (non-finite) fit."
    )

    t1 = metric_table("phi_only", "rmse_oos",
                      "Rolling 1-step-ahead OOS RMSE, best option per stage, 1 TVP ($\\varphi$).",
                      "tab:rmse_1tvp")
    t2 = metric_table("phi_xi", "rmse_oos",
                      "Rolling 1-step-ahead OOS RMSE, best option per stage, 2 TVP ($\\varphi,\\xi$).",
                      "tab:rmse_2tvp")
    t3 = metric_table("phi_only", "crps_oos",
                      "Rolling 1-step-ahead OOS CRPS, best option per stage, 1 TVP ($\\varphi$).",
                      "tab:crps_1tvp")
    t4 = metric_table("phi_xi", "crps_oos",
                      "Rolling 1-step-ahead OOS CRPS, best option per stage, 2 TVP ($\\varphi,\\xi$).",
                      "tab:crps_2tvp")

    c1 = config_table("phi_only", "Which config each cell above uses, 1 TVP ($\\varphi$).", "tab:cfg_1tvp")
    c2 = config_table("phi_xi", "Which config each cell above uses, 2 TVP ($\\varphi,\\xi$).", "tab:cfg_2tvp")

    body = "\n\n".join([
        r"\begin{document}",
        r"\title{Rolling 1-step OOS RMSE and CRPS by stage}\author{}\date{2026-09-11}\maketitle",
        intro,
        r"\section{RMSE}", t1, t2,
        r"\section{CRPS}", t3, t4,
        r"\section{Reference: which config was selected}", c1, c2,
        r"\end{document}",
    ])
    (REP / "report.tex").write_text(pre + "\n" + body, encoding="utf-8")
    print("wrote report.tex")

    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                       cwd=REP, capture_output=True, text=True)
    log = (REP / "report.log").read_text(errors="ignore") if (REP / "report.log").exists() else ""
    ok = (REP / "report.pdf").exists() and "Fatal error" not in log
    print("report.pdf:", "OK" if ok else "FAILED (see report.log)")


if __name__ == "__main__":
    build()
