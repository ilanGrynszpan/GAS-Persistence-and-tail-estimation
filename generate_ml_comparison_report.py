"""
GAS vs machine-learning-baseline comparison, updated with the current best
GAS model per location (2026-09-11).

Reproduces the two tables in "reports/oos multimethod table.tex" (RMSE
across AR / SDM(=GAS) / XGB / AR+Aux / LSTM, and CRPS AR vs GAS) but with
the GAS column replaced by whichever config Table 64 of
reports/stage_comparison_20260911/report.tex selected as the single best
model overall for that location (same selection rule: gradient tier, then
per-parameter convergence fraction, then OOS CRPS, then OOS RMSE -- see
generate_stage_comparison_report.py's module docstring). The ML/AR numbers
are UNCHANGED, read straight from the existing cached baseline runs
(reports/ml_baselines_horizon/*.csv) -- no refitting of AR/XGB/LSTM here.

Reads ONLY:
  artifacts/stage_comparison_20260911/metrics.json        (to pick the winner)
  artifacts/rerun_10092026/ , artifacts/rerun_10092026_phiL_sigmoid/  (multistep.csv of the winner)
  reports/ml_baselines_horizon/all_locations_ml_baselines_horizon.csv (AR/AR+Aux/XGB/LSTM RMSE)
  reports/ml_baselines_horizon/all_locations_ar_crps_gaussian.csv     (AR CRPS)

Writes reports/ML_comparison/report.tex (+ .pdf).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MET = json.loads((ROOT / "artifacts" / "stage_comparison_20260911" / "metrics.json").read_text())
OLD_TREE = ROOT / "artifacts" / "rerun_10092026"
SIGMOID_TREE = ROOT / "artifacts" / "rerun_10092026_phiL_sigmoid"
REP = ROOT / "reports" / "ML_comparison"

SHORTS = ["bh", "cruzeiro", "darwin", "garanhuns", "manaus", "salvador"]
DISP_LONG = {  # match the display names used in the original multimethod table
    "bh": "Belo Horizonte (Brazil)", "cruzeiro": "Cruzeiro do Sul (Acre, Brazil)",
    "darwin": "Darwin (Australia)", "garanhuns": "Garanhuns (Pernambuco, Brazil)",
    "manaus": "Manaus (Brazil)", "salvador": "Salvador (Brazil)",
}
LOC_MAP = {  # short -> full name used in the ML baseline CSVs
    "bh": "BELO HORIZONTE", "cruzeiro": "CRUZEIRO DO SUL (ACRE)",
    "darwin": "DARWIN AIRPORT", "garanhuns": "GARANHUNS (PERNAMBUCO)",
    "manaus": "MANAUS", "salvador": "SALVADOR",
}
GAS_LAG_SETS = ["L1", "L123", "L1_365_366", "L123_365_366"]
STAGE2_COV = ["dew", "dew_temp", "dew_enso", "dew_temp_enso"]
HARVEY_COV5 = ["none", "dew", "dew_temp", "dew_enso", "dew_temp_enso"]
HORIZONS = [1, 2, 5, 10, 30, 60, 90]
CRPS_REL_TOL = 0.01


def m(short, stage, cfg, branch):
    return MET.get(short, {}).get(stage, {}).get(cfg, {}).get(branch)


def tier(metrics):
    if metrics is None:
        return 3
    g = metrics.get("grad_inf")
    if g is None or not np.isfinite(g):
        return 3
    # 2026-09-12 fix: a fit can be well-converged in the M-estimation sense
    # (small penalized-objective gradient) yet still produce a diverging
    # simulate_oos forecast path (a distinct failure mode -- see this
    # session's stage34 stability diagnostics). Such a candidate must never
    # outrank a worse-gradient candidate whose OOS forecast is actually
    # usable, so a non-finite OOS CRPS forces the worst tier regardless of
    # how good grad_inf looks (previously this let a single diverged
    # candidate "win" outright whenever it happened to have the best
    # gradient tier + convergence fraction in its pool, e.g. Manaus
    # stage4/dew_enso/phi_xi in the ENSO-metrics report).
    crps = metrics.get("crps_oos")
    if crps is None or not np.isfinite(crps):
        return 3
    return 0 if g < 1 else (1 if g < 100 else 2)


def select(cands):
    """Same rule as generate_stage_comparison_report.py's select(): tier by
    grad, then per-param convergence fraction, then OOS CRPS (1% rel.
    tolerance), then OOS RMSE. cands: list of (id, metrics)."""
    cands = [(i, mm) for i, mm in cands if mm is not None]
    if not cands:
        return (None, None)
    key = lambda im: (tier(im[1]), -(im[1].get("frac_conv") or 0.0))
    best_key = min(key(im) for im in cands)
    top = [im for im in cands if key(im) == best_key]
    crps_of = lambda im: im[1]["crps_oos"] if np.isfinite(im[1].get("crps_oos", np.nan)) else np.inf
    min_crps = min(crps_of(im) for im in top)
    close = [im for im in top if crps_of(im) <= min_crps * (1 + CRPS_REL_TOL)] if np.isfinite(min_crps) else top
    if len(close) == 1:
        return close[0]
    rmse_of = lambda im: im[1]["rmse_oos"] if np.isfinite(im[1].get("rmse_oos", np.nan)) else np.inf
    return min(close, key=rmse_of)


def best_by_tw95(short):
    """2026-09-12: a DIFFERENT selection than best_overall_id -- searches the
    same full candidate universe (every stage 1-4S x both branches, Stage 5
    excluded), gates on gradient tier only ("optimized well" -- a candidate
    with non-finite OOS CRPS is still forced to the worst tier, same as
    tier()'s general rule), and picks whichever MINIMISES twCRPS95 among
    the best tier reached -- no CRPS/RMSE tie-break, since twCRPS95 IS the
    criterion here. Used by reports/kupiec/ and reports/tail_oos/."""
    cands = []
    for lk in GAS_LAG_SETS:
        for br in ("phi_only", "phi_xi"):
            cands.append((("stage1", lk, br), m(short, "stage1", lk, br)))
    for lk in GAS_LAG_SETS:
        for cov in STAGE2_COV:
            for br in ("phi_only", "phi_xi"):
                cands.append((("stage2", f"{lk}_{cov}", br), m(short, "stage2", f"{lk}_{cov}", br)))
    for stage in ("stage3", "stage4", "stage4S"):
        for cov in HARVEY_COV5:
            for br in ("phi_only", "phi_xi"):
                cands.append(((stage, cov, br), m(short, stage, cov, br)))
    cands = [(i, mm) for i, mm in cands if mm is not None]
    if not cands:
        return (None, None)
    best_tier = min(tier(mm) for _, mm in cands)
    pool = [(i, mm) for i, mm in cands if tier(mm) == best_tier]
    tw95_of = lambda im: im[1]["tw95"] if np.isfinite(im[1].get("tw95", np.nan)) else np.inf
    return min(pool, key=tw95_of)


def best_overall_id(short):
    """Re-derive Table 64's winner for `short`, returning a structured
    (stage, cfg, branch) identifier (not just a display label)."""
    cands = []

    for branch in ("phi_only", "phi_xi"):
        c = [(("stage1", lk, branch), m(short, "stage1", lk, branch)) for lk in GAS_LAG_SETS]
        cands.append(select(c))
        c = [(("stage2", f"{lk}_{cov}", branch), m(short, "stage2", f"{lk}_{cov}", branch))
             for cov in STAGE2_COV for lk in GAS_LAG_SETS]
        cands.append(select(c))

    for stage in ("stage3", "stage4", "stage4S"):
        c = [((stage, cov, branch), m(short, stage, cov, branch))
             for cov in HARVEY_COV5 for branch in ("phi_only", "phi_xi")]
        cands.append(select(c))

    return select(cands)


def _resolve_cfg_dir(short, ident):
    stage, cfg, branch = ident
    sfx = "" if branch == "phi_only" else "__xi"
    if stage in ("stage1", "stage2", "stage4S"):
        root = OLD_TREE
    else:
        root = SIGMOID_TREE
    if stage == "stage1":
        p = OLD_TREE / short / "stage1" / f"_scaling_choice{sfx}.json"
        sc = json.loads(p.read_text())["per_lag"][cfg]["scaling"]
        cfg = f"{cfg}_{sc}"
    return root / short / stage / (cfg + sfx)


def load_gas_multistep(short):
    ident, mm = best_overall_id(short)
    cdir = _resolve_cfg_dir(short, ident)
    df = pd.read_csv(cdir / "multistep.csv")
    return ident, df.set_index("horizon")


def _bold(cell_val, best_val, fmt="{:.2f}"):
    s = fmt.format(cell_val)
    return rf"\textbf{{{s}}}" if abs(cell_val - best_val) < 1e-9 else s


def build():
    REP.mkdir(parents=True, exist_ok=True)
    ml = pd.read_csv(ROOT / "reports" / "ml_baselines_horizon" / "all_locations_ml_baselines_horizon.csv")
    ar_crps = pd.read_csv(ROOT / "reports" / "ml_baselines_horizon" / "all_locations_ar_crps_gaussian.csv")

    gas_ms = {}
    gas_ident = {}
    for s in SHORTS:
        ident, df = load_gas_multistep(s)
        gas_ms[s] = df
        gas_ident[s] = ident
        print(f"  {DISP_LONG[s]}: best overall = {ident}")

    # ---- Table 1: RMSE, all 5 methods, all horizons ----
    rows_rmse = []
    for s in SHORTS:
        loc_full = LOC_MAP[s]
        sub = ml[ml.location == loc_full]
        for h in HORIZONS:
            vals = {}
            for model in ("AR", "GAS", "XGB", "AR+Aux", "LSTM"):
                if model == "GAS":
                    vals[model] = float(gas_ms[s].loc[h, "rmse"]) if h in gas_ms[s].index else np.nan
                else:
                    r = sub[(sub.model == model) & (sub.horizon == h)]
                    vals[model] = float(r["rmse_all"].iloc[0]) if len(r) else np.nan
            best = min(v for v in vals.values() if np.isfinite(v))
            rows_rmse.append((DISP_LONG[s], h, vals, best))

    body_rmse = []
    for loc, h, vals, best in rows_rmse:
        cells = " & ".join(_bold(vals[k], best) if np.isfinite(vals[k]) else "--"
                           for k in ("AR", "GAS", "XGB", "AR+Aux", "LSTM"))
        body_rmse.append(f"{loc} & {h} & {cells} \\\\")
    rmse_tex = "\n".join(body_rmse)

    # ---- Table 2: CRPS, AR vs GAS, all horizons ----
    rows_crps = []
    for s in SHORTS:
        loc_full = LOC_MAP[s]
        sub = ar_crps[ar_crps.location == loc_full]
        for h in HORIZONS:
            ar_v = float(sub[sub.horizon == h]["crps_all"].iloc[0]) if len(sub[sub.horizon == h]) else np.nan
            gas_v = float(gas_ms[s].loc[h, "crps"]) if h in gas_ms[s].index else np.nan
            rows_crps.append((DISP_LONG[s], h, ar_v, gas_v))

    body_crps = []
    for loc, h, ar_v, gas_v in rows_crps:
        if np.isfinite(ar_v) and np.isfinite(gas_v):
            best = min(ar_v, gas_v)
            body_crps.append(f"{loc} & {h} & {_bold(ar_v, best)} & {_bold(gas_v, best)} \\\\")
        else:
            body_crps.append(f"{loc} & {h} & {ar_v:.2f} & {'--' if not np.isfinite(gas_v) else f'{gas_v:.2f}'} \\\\")
    crps_tex = "\n".join(body_crps)

    ident_rows = "\n".join(
        f"{DISP_LONG[s]} & {gas_ident[s][0]} & {gas_ident[s][1].replace('_', chr(92)+chr(95))} & "
        f"{'1 TVP ($\\varphi$)' if gas_ident[s][2] == 'phi_only' else '2 TVP ($\\varphi,\\xi$)'} \\\\"
        for s in SHORTS
    )

    pre = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage[margin=1.8cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,array}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}\usepackage{float}
\emergencystretch=3em\sloppy
"""

    body = "\n\n".join([
        r"\begin{document}",
        r"\title{GAS vs.\ machine-learning baselines}\author{}\date{2026-09-11}\maketitle",
        r"\section*{Which GAS model is used}",
        r"For each location, the ``GAS'' column below is the single best model overall "
        r"(across every stage and TVP branch) per Table 64 of "
        r"\texttt{reports/stage\_comparison\_20260911/report.tex} -- same selection rule: "
        r"gradient tier, then the fraction of parameters individually converged, then OOS "
        r"CRPS, then OOS RMSE. AR/AR+Aux/XGB/LSTM figures are unchanged, read directly from "
        r"\texttt{reports/ml\_baselines\_horizon/}.",
        "\\begin{table}[H]\\centering\\footnotesize\n"
        "\\caption{Best-overall GAS model used for each location's GAS column below.}\n"
        "\\begin{tabular}{lccc}\\toprule\nLocation & stage & config & branch \\\\\\midrule\n"
        + ident_rows + "\n\\bottomrule\\end{tabular}\\end{table}",
        r"""% =========================================================
% RMSE (all observations, no wet/dry split)
% =========================================================
{
\footnotesize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{0.82}
\begin{longtable}{|p{4.3cm}|c|c|c|c|c|c|}
\caption{Out-of-sample RMSE (all observations, wet and dry days combined), rolling-origin evaluation. GAS replaces the earlier SDM column with the current best-overall GAS model (see table above). Best performance per row is highlighted in bold.}
\label{tab:daily_rmse_all} \\
\hline
Series & H & AR & GAS & XGB & AR+Aux & LSTM \\
\hline
\endfirsthead
\hline
Series & H & AR & GAS & XGB & AR+Aux & LSTM \\
\hline
\endhead
\hline
\endfoot
"""
        + rmse_tex +
        r"""
\end{longtable}
}""",
        r"""% =========================================================
% CRPS: AR (Gaussian) vs GAS (Monte Carlo)
% =========================================================
{
\footnotesize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{0.82}
\begin{longtable}{|p{4.3cm}|c|c|c|}
\caption{Out-of-sample CRPS, AR (closed-form Normal CRPS under an assumed $N(\mu,\sigma^2)$ predictive distribution, $\sigma$ estimated from in-sample residuals) vs.\ GAS (Monte Carlo energy-score estimator, current best-overall GAS model per location). Best performance per row is highlighted in bold.}
\label{tab:daily_crps_ar_gas} \\
\hline
Series & H & AR & GAS \\
\hline
\endfirsthead
\hline
Series & H & AR & GAS \\
\hline
\endhead
\hline
\endfoot
"""
        + crps_tex +
        r"""
\end{longtable}
}""",
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
