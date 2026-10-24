"""
2026-09-14: reports/best_models/report.tex.

Per location, the best config within each of 5 model classes:
  1. GAS(p,q), no external variables      -> Stage 1  (best of 4 GAS-lag sets)
  2. GAS(p,q) + external variables        -> Stage 2  (best of 4 lags x 4 covariate sets)
  3. long-short dynamics                  -> Stage 3  (best of 5 Harvey covariate configs)
  4. long-short dynamics + seasonality    -> Stage 4  (best of 5 Harvey covariate configs)
  5. DCS/UC - ST (dcs_uc_st)            -> Stage 4S (best of 5 Harvey covariate configs)

Two selection criteria, each producing a 1-TVP and a 2-TVP table:
  A. "gradient ok, then minimise RMSE and CRPS": the standard rule used
     throughout this project (gradient tier, then per-parameter convergence
     fraction, then OOS CRPS at 1% relative tolerance, then OOS RMSE).
  B. "gradient ok, then minimise twCRPS95" / "...twCRPS98": same gradient
     tier gate, but ranked directly by whichever twCRPS level is asked for.

Finally, a list of every (location, stage, config, branch) that never
reached an acceptable gradient (< 100) -- i.e. never even reached the
gate all six tables above require.

Reads ONLY artifacts/stage_comparison_20260911/metrics.json (for selection)
plus each selected candidate's own result.json (for its "scaling" field
only, to display the Fisher-scaling choice instead of the gradient value
per 2026-09-14 request). No fitting. Stage 5 is excluded throughout (as in
every other 2026-09 report), since it isn't one of the 5 requested model
classes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

import generate_ml_comparison_report as G  # m()/tier()/metrics.json/labels/GAS_LAG_SETS/STAGE2_COV/HARVEY_COV5

ROOT = Path(__file__).parent
REP = ROOT / "reports" / "best_models"
SHORTS = G.SHORTS
DISP = G.DISP_LONG

CLASSES = [
    ("stage1", "GAS(p,q), no external variables"),
    ("stage2", "GAS(p,q) + external variables"),
    ("stage3", "long-short dynamics"),
    ("stage4", "long-short dynamics + seasonality"),
    ("stage4S", "DCS/UC - ST"),
]
BRANCH_LBL = {"phi_only": "1 TVP ($\\varphi$)", "phi_xi": "2 TVP ($\\varphi,\\xi$)"}
BRANCH_SHORT = {"phi_only": "1 TVP", "phi_xi": "2 TVP"}
CLASS_SHORT = {
    "GAS(p,q), no external variables": "GAS, no ext.",
    "GAS(p,q) + external variables": "GAS + ext.",
    "long-short dynamics": "long-short",
    "long-short dynamics + seasonality": "long-short+seas.",
    "DCS/UC - ST": "dcs_uc_st",
}
DISP_SHORT = {s: DISP[s].split(" (")[0] for s in SHORTS}


def candidates(short, stage, branch):
    if stage == "stage1":
        return [(lk, G.m(short, stage, lk, branch)) for lk in G.GAS_LAG_SETS]
    if stage == "stage2":
        return [(f"{lk}_{cov}", G.m(short, stage, f"{lk}_{cov}", branch))
                for cov in G.STAGE2_COV for lk in G.GAS_LAG_SETS]
    return [(cov, G.m(short, stage, cov, branch)) for cov in G.HARVEY_COV5]


def select_by_crps_rmse(cands):
    """The standard rule: tier, then convergence fraction, then CRPS (1%
    tolerance), then RMSE."""
    cands = [(i, mm) for i, mm in cands if mm is not None]
    if not cands:
        return (None, None)
    key = lambda im: (G.tier(im[1]), -(im[1].get("frac_conv") or 0.0))
    best_key = min(key(im) for im in cands)
    top = [im for im in cands if key(im) == best_key]
    crps_of = lambda im: im[1]["crps_oos"] if np.isfinite(im[1].get("crps_oos", np.nan)) else np.inf
    min_crps = min(crps_of(im) for im in top)
    close = [im for im in top if crps_of(im) <= min_crps * (1 + G.CRPS_REL_TOL)] if np.isfinite(min_crps) else top
    if len(close) == 1:
        return close[0]
    rmse_of = lambda im: im[1]["rmse_oos"] if np.isfinite(im[1].get("rmse_oos", np.nan)) else np.inf
    return min(close, key=rmse_of)


def select_by_metric(cands, key):
    """Gradient-tier gate, then minimise `key` (tw95 or tw98) directly --
    no CRPS/RMSE tie-break, since `key` IS the criterion here."""
    cands = [(i, mm) for i, mm in cands if mm is not None]
    if not cands:
        return (None, None)
    best_tier = min(G.tier(mm) for _, mm in cands)
    pool = [(i, mm) for i, mm in cands if G.tier(mm) == best_tier]
    val = lambda im: im[1][key] if np.isfinite(im[1].get(key, np.nan)) else np.inf
    return min(pool, key=val)


def _f(v, d=3):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(x):
        return "div"
    if abs(x) >= 1e4:
        return rf"$\sim\!10^{{{int(np.floor(np.log10(abs(x))))}}}$"
    return f"{x:.{d}f}"


def _cfg_txt(cfg):
    return cfg.replace("_", "-") if cfg else "--"


SCALING_DISP = {
    "unit": "unit",
    "diagonal_inverse_fisher": r"diag.\ inv.\ Fisher",
    "inverse_fisher": r"full inv.\ Fisher",
}


def _scaling_txt(s, stage, cfg, branch):
    """Fisher-scaling choice actually used by this candidate's own fit,
    read directly from its result.json (metrics.json does not carry this
    field) -- displayed instead of the gradient value per 2026-09-14
    request. Selection above still gates on the gradient internally; only
    the DISPLAYED column changed."""
    if cfg is None:
        return "--"
    cdir = G._resolve_cfg_dir(s, (stage, cfg, branch))
    if cdir is None or not (cdir / "result.json").exists():
        return "--"
    try:
        sc = json.loads((cdir / "result.json").read_text()).get("scaling")
    except Exception:
        return "--"
    return SCALING_DISP.get(sc, sc or "--")


def build_table(branch, cols_extra, value_fn, caption, label):
    """cols_extra: list of (header, key) pairs beyond Model class/Config/
    Fisher scaling. value_fn(mm, key) -> formatted string."""
    body = [r"{\footnotesize\setlength{\tabcolsep}{4pt}",
            r"\begin{longtable}{ll" + "r" * (1 + len(cols_extra)) + "}",
            rf"\caption{{{caption}}}\label{{{label}}}\\",
            r"\toprule",
            "Model class & Config & Fisher scaling & " + " & ".join(h for h, _ in cols_extra) + r" \\",
            r"\midrule", r"\endfirsthead",
            r"\toprule",
            "Model class & Config & Fisher scaling & " + " & ".join(h for h, _ in cols_extra) + r" \\",
            r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for s in SHORTS:
        body.append(rf"\multicolumn{{{3 + len(cols_extra)}}}{{l}}{{\textbf{{{DISP[s]}}}}} \\")
        for stage, lbl in CLASSES:
            cfg, mm = value_fn(s, stage, branch)
            extras = " & ".join(_f((mm or {}).get(k)) for _, k in cols_extra)
            body.append(f"{lbl} & {_cfg_txt(cfg)} & {_scaling_txt(s, stage, cfg, branch)} & {extras} \\\\")
        body.append(r"\midrule")
    body.append(r"\end{longtable}}")
    return "\n".join(body)


def unacceptable_list():
    rows = []
    for s in SHORTS:
        for stage, lbl in CLASSES:
            for branch in ("phi_only", "phi_xi"):
                for cfg, mm in candidates(s, stage, branch):
                    if mm is None:
                        continue
                    g = mm.get("grad_inf")
                    ok = g is not None and np.isfinite(g) and g < 100
                    if not ok:
                        rows.append((DISP_SHORT[s], CLASS_SHORT[lbl], _cfg_txt(cfg),
                                    BRANCH_SHORT[branch], _scaling_txt(s, stage, cfg, branch)))
    return rows


def build():
    REP.mkdir(parents=True, exist_ok=True)
    pre = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage[margin=1.8cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}\usepackage{float}
\emergencystretch=3em\sloppy
"""
    intro = (
        r"\section*{Methodology}"
        r"For each location and each of the 5 model classes below, the best config is selected "
        r"among that class's own candidates (Stage 1: 4 GAS-lag sets; Stage 2: 4 lag sets "
        r"$\times$ 4 covariate sets; Stage 3/4/4S: 5 Harvey covariate configs), separately for "
        r"the 1-TVP ($\varphi$) and 2-TVP ($\varphi,\xi$) branches. "
        r"\textbf{Criterion A} (Sections 1-2): gradient tier ($<1$, then $<100$, then anything "
        r"else), then the fraction of individually-converged parameters, then OOS CRPS (1\% "
        r"relative tolerance), then OOS RMSE. "
        r"\textbf{Criterion B} (Sections 3-6): same gradient-tier gate, then MINIMISE OOS "
        r"twCRPS95 or twCRPS98 directly (no CRPS/RMSE tie-break). "
        r"Stage 3/4 use the sigmoid-bounded $\varphi_L$ parameterisation (2026-09-11 fix); "
        r"Stage 5 is excluded (not one of the 5 requested classes). \emph{div} marks a diverged "
        r"fit that was still the best available in its tier."
    )

    secs = [r"\begin{document}",
            r"\title{Best model per class, per location}\author{}\date{2026-09-14}\maketitle", intro]

    secs.append(r"\section{Criterion A: gradient ok, then minimise RMSE and CRPS}")
    for branch in ("phi_only", "phi_xi"):
        secs.append(rf"\subsection{{{BRANCH_LBL[branch]}}}")
        secs.append(build_table(
            branch, [("RMSE", "rmse_oos"), ("CRPS", "crps_oos")],
            lambda s, stage, br: select_by_crps_rmse(candidates(s, stage, br)),
            f"Best config per model class per location, {BRANCH_LBL[branch]} -- gradient ok, "
            "then minimise RMSE/CRPS.", f"tab:A_{branch}"))

    for lvl, key in [("95", "tw95"), ("98", "tw98")]:
        secs.append(rf"\section{{Criterion B: gradient ok, then minimise twCRPS{lvl}}}")
        for branch in ("phi_only", "phi_xi"):
            secs.append(rf"\subsection{{{BRANCH_LBL[branch]}}}")
            secs.append(build_table(
                branch, [(f"twCRPS{lvl}", key)],
                lambda s, stage, br, key=key: select_by_metric(candidates(s, stage, br), key),
                f"Best config per model class per location, {BRANCH_LBL[branch]} -- gradient ok, "
                f"then minimise twCRPS{lvl}.", f"tab:B{lvl}_{branch}"))

    secs.append(r"\section{Models that never reached an acceptable gradient ($<100$)}")
    rows = unacceptable_list()
    secs.append(rf"{len(rows)} of the candidates considered above never reached grad $<100$ "
               r"in either branch:")
    body = [r"{\footnotesize\setlength{\tabcolsep}{4pt}",
            r"\begin{longtable}{llllr}",
            r"\caption{Configs with an unacceptable ($\geq 100$ or non-finite) gradient.}"
            r"\label{tab:unacceptable}\\",
            r"\toprule", r"Location & Model class & Config & Branch & Fisher scaling \\", r"\midrule",
            r"\endfirsthead", r"\toprule", r"Location & Model class & Config & Branch & Fisher scaling \\",
            r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for loc, cls, cfg, br, g in rows:
        body.append(f"{loc} & {cls} & {cfg} & {br} & {g} \\\\")
    body.append(r"\end{longtable}}")
    secs.append("\n".join(body))

    secs.append(r"\end{document}")
    (REP / "report.tex").write_text(pre + "\n" + "\n\n".join(secs), encoding="utf-8")
    print("wrote report.tex")

    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                       cwd=REP, capture_output=True, text=True)
    log = (REP / "report.log").read_text(errors="ignore") if (REP / "report.log").exists() else ""
    ok = (REP / "report.pdf").exists() and "Fatal error" not in log
    print("report.pdf:", "OK" if ok else "FAILED (see report.log)")


if __name__ == "__main__":
    build()
