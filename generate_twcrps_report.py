"""
2026-09-11: twCRPS95/98 comparison tables, reports/twcrps/report.tex.

Group A (Stage 1 vs Stage 2):
  A1. by TVP branch: 2 tables (1TVP, 2TVP) x 2 levels (95, 98) = 4 tables,
      columns = [Stage 1, Stage 2], each cell = that stage's best lag-set
      (Stage 2: best lag-set x covariate-set) WITHIN that branch.
  A2. by seasonality instead of TVP: 2 tables (no seasonal lags, seasonal
      lags included) x 2 levels = 4 tables, same columns, each cell now
      best across BOTH TVP branches too (branch is no longer the split
      axis, so it joins lag-set/covariate-set as something `select()`
      optimises over).

Group B (Stage 3 / Stage 4 / Stage 4S):
  B1. by TVP branch: 2 tables x 2 levels = 4 tables, columns =
      [Stage 3, Stage 4, Stage 4S], each cell = that stage's best
      covariate config WITHIN that branch.
  B2. "with/without seasonality" collapse: since which of Stage 3
      (no seasonal component) / Stage 4 (seasonal component, long+short) /
      Stage 4S (seasonal component, structural) a column is ALREADY IS the
      seasonality comparison, there is no further seasonality axis to
      split additional tables by here -- instead (mirroring A2's "TVP is
      no longer the split axis, so it is absorbed into the selection") the
      TVP split is dropped and each cell becomes the best covariate config
      across BOTH branches: 1 table per level (95, 98) = 2 tables, same
      3 columns.

All "best" selections use the same rule as every other 2026-09-11 report:
gradient tier, then per-parameter convergence fraction, then OOS CRPS (1%
relative tolerance), then OOS RMSE -- see generate_ml_comparison_report.select().
Reads ONLY artifacts/stage_comparison_20260911/metrics.json. No fitting.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

import generate_ml_comparison_report as G  # select()/m()/metrics.json/labels/trees

ROOT = Path(__file__).parent
REP = ROOT / "reports" / "twcrps"
SHORTS = G.SHORTS
DISP = G.DISP_LONG
BRANCHES = ("phi_only", "phi_xi")
BRANCH_LBL = {"phi_only": "1 TVP ($\\varphi$)", "phi_xi": "2 TVP ($\\varphi,\\xi$)"}


def _f(v, d=3):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "--"
    import numpy as np
    if not np.isfinite(x):
        return "div"
    return f"{x:.{d}f}"


# ---------------------------------------------------------------------------
# resolvers
# ---------------------------------------------------------------------------

def stage1_best(short, branches, lags):
    cands = [(("stage1", lk, br), G.m(short, "stage1", lk, br)) for br in branches for lk in lags]
    return G.select(cands)


def stage2_best(short, branches, lags):
    cands = [(("stage2", f"{lk}_{cov}", br), G.m(short, "stage2", f"{lk}_{cov}", br))
             for br in branches for cov in G.STAGE2_COV for lk in lags]
    return G.select(cands)


def harvey_best(short, stage, branches):
    cands = [((stage, cov, br), G.m(short, stage, cov, br))
             for br in branches for cov in G.HARVEY_COV5]
    return G.select(cands)


ALL_LAGS = G.GAS_LAG_SETS
NO_SEAS_LAGS = ["L1", "L123"]
SEAS_LAGS = ["L1_365_366", "L123_365_366"]


# ---------------------------------------------------------------------------
# table builders
# ---------------------------------------------------------------------------

def tbl(rows, cols, caption, label):
    df = pd.DataFrame(rows).set_index("Location")
    body = [r"\begin{table}[H]\centering\footnotesize\setlength{\tabcolsep}{6pt}",
            rf"\caption{{{caption}}}\label{{{label}}}",
            r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
            "Location & " + " & ".join(cols) + r" \\", r"\midrule"]
    for idx, row in df[cols].iterrows():
        body.append(str(idx) + " & " + " & ".join(str(v) for v in row) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(body)


def group_a_by_tvp(level_key, branch, level_lbl):
    rows = []
    for s in SHORTS:
        _, m1 = stage1_best(s, [branch], ALL_LAGS)
        _, m2 = stage2_best(s, [branch], ALL_LAGS)
        rows.append({"Location": DISP[s], "Stage 1": _f(m1.get(level_key) if m1 else None),
                     "Stage 2": _f(m2.get(level_key) if m2 else None)})
    return tbl(rows, ["Stage 1", "Stage 2"],
              f"twCRPS{level_lbl}: Stage 1 vs Stage 2, best lag/covariate set, {BRANCH_LBL[branch]}.",
              f"tab:A_{level_key}_{branch}")


def group_a_by_season(level_key, seasonal, level_lbl):
    lags = SEAS_LAGS if seasonal else NO_SEAS_LAGS
    tag = "seasonal lags included" if seasonal else "no seasonal lags"
    rows = []
    for s in SHORTS:
        _, m1 = stage1_best(s, BRANCHES, lags)
        _, m2 = stage2_best(s, BRANCHES, lags)
        rows.append({"Location": DISP[s], "Stage 1": _f(m1.get(level_key) if m1 else None),
                     "Stage 2": _f(m2.get(level_key) if m2 else None)})
    return tbl(rows, ["Stage 1", "Stage 2"],
              f"twCRPS{level_lbl}: Stage 1 vs Stage 2, best lag/covariate set AND TVP branch, {tag}.",
              f"tab:A2_{level_key}_{'seas' if seasonal else 'noseas'}")


def group_b_by_tvp(level_key, branch, level_lbl):
    rows = []
    for s in SHORTS:
        vals = {}
        for stage, lbl in [("stage3", "Stage 3"), ("stage4", "Stage 4"), ("stage4S", "Stage 4S")]:
            _, mm = harvey_best(s, stage, [branch])
            vals[lbl] = _f(mm.get(level_key) if mm else None)
        rows.append({"Location": DISP[s], **vals})
    return tbl(rows, ["Stage 3", "Stage 4", "Stage 4S"],
              f"twCRPS{level_lbl}: Stage 3 vs Stage 4 vs Stage 4S, best covariate config, {BRANCH_LBL[branch]}.",
              f"tab:B_{level_key}_{branch}")


def group_b_collapsed(level_key, level_lbl):
    """No seasonal component (Stage 3, standard-rule best over covariate x
    branch) vs Seasonal component: Stage 4 and Stage 4S are each reduced to
    their own standard-rule best (covariate x branch) first, then between
    those two finalists the one with the LOWER twCRPS at this level wins --
    a direct twCRPS-minimising choice between "seasonal, long+short" and
    "seasonal, structural", per user request 2026-09-11."""
    rows = []
    for s in SHORTS:
        _, m3 = harvey_best(s, "stage3", BRANCHES)
        _, m4 = harvey_best(s, "stage4", BRANCHES)
        _, m4s = harvey_best(s, "stage4S", BRANCHES)

        def lvl(mm):
            v = mm.get(level_key) if mm else None
            try:
                x = float(v)
            except (TypeError, ValueError):
                return float("inf")
            return x if x == x else float("inf")  # NaN -> inf, so a finite rival always wins

        m_seas = m4 if lvl(m4) <= lvl(m4s) else m4s
        rows.append({"Location": DISP[s], "Stage 3": _f(m3.get(level_key) if m3 else None),
                     "Seasonal (Stage 4 or 4S)": _f(m_seas.get(level_key) if m_seas else None)})
    return tbl(rows, ["Stage 3", "Seasonal (Stage 4 or 4S)"],
              f"twCRPS{level_lbl}: Stage 3 (no seasonal component) vs whichever of Stage 4 "
              "(seasonal component, long+short) / Stage 4S (seasonal component, structural) "
              "MINIMISES twCRPS -- each of the three is itself the standard-rule best covariate "
              "config and TVP branch.", f"tab:B2_{level_key}")


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
        r"Every cell is the rolling 1-step-ahead OOS twCRPS of a location's BEST option along "
        r"whichever axis is NOT the table's own row/column identity, selected by the same rule "
        r"used throughout: gradient tier, then per-parameter convergence fraction, then lowest "
        r"OOS CRPS (1\% relative tolerance), then lowest OOS RMSE. Group A (Stage 1/2) selects "
        r"over GAS-lag sets (and, for Stage 2, covariate sets); Group B (Stage 3/4/4S) selects "
        r"over the 5 Harvey covariate configs. Section~2 and the Stage 3/4/4S collapse in "
        r"Section~4 additionally select over BOTH TVP branches, since TVP is no longer the split "
        r"axis there. Stage 3/4 use the sigmoid-bounded $\varphi_L$ parameterisation "
        r"(\texttt{phi\_L\_max=1.0}, 2026-09-11 fix). \emph{div} marks a diverged fit."
    )
    secs = [r"\begin{document}",
            r"\title{twCRPS95 / twCRPS98 comparisons}\author{}\date{2026-09-11}\maketitle", intro]

    secs.append(r"\section{Stage 1 vs Stage 2, by TVP branch}")
    for lvl, key in [("95", "tw95"), ("98", "tw98")]:
        secs.append(rf"\subsection{{twCRPS{lvl}}}")
        secs.append(group_a_by_tvp(key, "phi_only", lvl))
        secs.append(group_a_by_tvp(key, "phi_xi", lvl))

    secs.append(r"\section{Stage 3 / Stage 4 / Stage 4S, by TVP branch}")
    for lvl, key in [("95", "tw95"), ("98", "tw98")]:
        secs.append(rf"\subsection{{twCRPS{lvl}}}")
        secs.append(group_b_by_tvp(key, "phi_only", lvl))
        secs.append(group_b_by_tvp(key, "phi_xi", lvl))

    secs.append(r"\section{Stage 1 vs Stage 2, by seasonality (instead of TVP)}")
    for lvl, key in [("95", "tw95"), ("98", "tw98")]:
        secs.append(rf"\subsection{{twCRPS{lvl}}}")
        secs.append(group_a_by_season(key, False, lvl))
        secs.append(group_a_by_season(key, True, lvl))

    secs.append(r"\section{Stage 3 / Stage 4 / Stage 4S, seasonality only (TVP absorbed into selection)}")
    for lvl, key in [("95", "tw95"), ("98", "tw98")]:
        secs.append(rf"\subsection{{twCRPS{lvl}}}")
        secs.append(group_b_collapsed(key, lvl))

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
