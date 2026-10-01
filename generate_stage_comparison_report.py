"""
2026-09-11 stage-by-stage comparison report.

Reads ONLY artifacts/stage_comparison_20260911/metrics.json (produced by
collect_stage_comparison_metrics.py) -- no fitting happens here
(docs/REPORTING.md). Implements the user's exact selection rule wherever a
choice must be made among several candidate configs (lag set, covariate
set, or which branch wins):

    1. tier by checked gradient inf-norm: <1, then <100, then anything else
    2. within the best tier, prefer the candidate where the LARGEST
       fraction of individual parameters satisfy |grad_i|/n_train <= 1e-3
       ("gradient per observation of most parameters")
    3. tie-break by lowest rolling 1-step OOS CRPS
    4. if CRPS values are within 1% of each other, break the remaining tie
       by lowest rolling 1-step OOS RMSE
    Applied INDEPENDENTLY per location (winners can differ by location).

Every comparison is rendered as one table PER OPTION (rows = locations,
columns = chosen sub-config [when a reduction was needed] + the 7 metrics:
grad, RMSE-oos, RMSE-is, CRPS-oos, CRPS-is, twCRPS95-oos, twCRPS98-oos),
stacked one below the other in the document -- this is both how the user
asked to handle any table that would otherwise overflow, AND the only way
to show that the winning sub-config can differ by location.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MET = json.loads((ROOT / "artifacts" / "stage_comparison_20260911" / "metrics.json").read_text())
REP = ROOT / "reports" / "stage_comparison_20260911"

from data.station_loader import STATION_REGISTRY

LOCATIONS = ["BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
             "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR"]
SHORTS = [STATION_REGISTRY[l].get("short", l) for l in LOCATIONS]
DISP = {STATION_REGISTRY[l].get("short", l): STATION_REGISTRY[l].get("display", l) for l in LOCATIONS}

GAS_LAG_SETS = ["L1", "L123", "L1_365_366", "L123_365_366"]
LAG_LABEL = {"L1": r"$\{1\}$", "L123": r"$\{1,2,3\}$",
             "L1_365_366": r"$\{1,365,366\}$", "L123_365_366": r"$\{1,2,3,365,366\}$"}
STAGE2_COV = ["dew", "dew_temp", "dew_enso", "dew_temp_enso"]
S2_LBL = {"dew": "dew", "dew_temp": "dew+temp", "dew_enso": "dew+ENSO", "dew_temp_enso": "dew+temp+ENSO"}
HARVEY_COV5 = ["none", "dew", "dew_temp", "dew_enso", "dew_temp_enso"]
HC_LBL = {"none": "none", "dew": "dew", "dew_temp": "dew+temp", "dew_enso": "dew+ENSO",
          "dew_temp_enso": "dew+temp+ENSO"}
BRANCH_LBL = {"phi_only": "1 TVP ($\\varphi$)", "phi_xi": "2 TVP ($\\varphi,\\xi$)"}
CRPS_REL_TOL = 0.01
PER_OBS_TOL = 1e-3

PRE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage[margin=1.8cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,longtable,caption}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}\usepackage{float}
\emergencystretch=3em\sloppy
"""


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


def m(short, stage, cfg, branch):
    """Fetch the metrics dict for one (location, stage, cfg, branch) cell, or None."""
    return MET.get(short, {}).get(stage, {}).get(cfg, {}).get(branch)


def tier(metrics):
    if metrics is None:
        return 3
    g = metrics.get("grad_inf")
    if g is None or not np.isfinite(g):
        return 3
    return 0 if g < 1 else (1 if g < 100 else 2)


def select(cands):
    """cands: list of (sub_label, metrics_dict-or-None) for ONE location.
    Returns (sub_label, metrics) of the winner per the rule in the module
    docstring, or (None, None) if every candidate is missing."""
    cands = [(l, mm) for l, mm in cands if mm is not None]
    if not cands:
        return (None, None)
    key = lambda lm: (tier(lm[1]), -(lm[1].get("frac_conv") or 0.0))
    best_key = min(key(lm) for lm in cands)
    top = [lm for lm in cands if key(lm) == best_key]

    def crps_of(lm):
        c = lm[1].get("crps_oos")
        return c if (c is not None and np.isfinite(c)) else np.inf
    min_crps = min(crps_of(lm) for lm in top)
    close = [lm for lm in top if crps_of(lm) <= min_crps * (1 + CRPS_REL_TOL)] if np.isfinite(min_crps) else top
    if len(close) == 1:
        return close[0]

    def rmse_of(lm):
        r = lm[1].get("rmse_oos")
        return r if (r is not None and np.isfinite(r)) else np.inf
    return min(close, key=rmse_of)


METRIC_COLS = ["grad", "RMSE oos", "RMSE is", "CRPS oos", "CRPS is", "tw95", "tw98"]


def _metric_row(metrics):
    metrics = metrics or {}
    return {
        "grad": _f(metrics.get("grad_inf"), 2),
        "RMSE oos": _f(metrics.get("rmse_oos"), 2),
        "RMSE is": _f(metrics.get("rmse_is"), 2),
        "CRPS oos": _f(metrics.get("crps_oos"), 2),
        "CRPS is": _f(metrics.get("crps_is"), 2),
        "tw95": _f(metrics.get("tw95")),
        "tw98": _f(metrics.get("tw98")),
    }


def tbl(rows, cols, caption, label):
    df = pd.DataFrame(rows).set_index("Location")
    df = df[cols]
    body = [r"\begin{table}[H]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
            rf"\caption{{{caption}}}\label{{{label}}}",
            r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
            "Location & " + " & ".join(cols) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        body.append(str(idx) + " & " + " & ".join(str(v) for v in row) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(body)


def winner_table(cands_by_loc, sub_label_fmt, caption, label):
    """cands_by_loc[short] = list of (sub_label, metrics) to reduce via
    select(); sub_label_fmt(sub_label) -> display string for the 'chosen'
    column."""
    rows = []
    for short in SHORTS:
        sub_label, mm = select(cands_by_loc.get(short, []))
        row = {"Location": DISP[short], "chosen": sub_label_fmt(sub_label) if sub_label else "--"}
        row.update(_metric_row(mm))
        rows.append(row)
    return tbl(rows, ["chosen"] + METRIC_COLS, caption, label)


def direct_table(metrics_by_loc, caption, label):
    rows = []
    for short in SHORTS:
        row = {"Location": DISP[short]}
        row.update(_metric_row(metrics_by_loc.get(short)))
        rows.append(row)
    return tbl(rows, METRIC_COLS, caption, label)


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def stage1_section():
    secs = [r"\section{Stage 1 -- no covariates}"]
    overall = {}  # overall[branch][short] = (sub_label, metrics) best-of-4-lags, for the 1v2TVP comparison
    for branch in ("phi_only", "phi_xi"):
        bl = BRANCH_LBL[branch]
        no_seas = {s: [(lk, m(s, "stage1", lk, branch)) for lk in ("L1", "L123")] for s in SHORTS}
        seas = {s: [(lk, m(s, "stage1", lk, branch)) for lk in ("L1_365_366", "L123_365_366")] for s in SHORTS}
        t_no = winner_table(no_seas, lambda l: LAG_LABEL[l],
                            f"Stage 1 [{bl}]: best lag WITHOUT seasonality ($\\{{1\\}}$ vs $\\{{1,2,3\\}}$).",
                            f"tab:s1_noseas_{branch}")
        t_seas = winner_table(seas, lambda l: LAG_LABEL[l],
                              f"Stage 1 [{bl}]: best lag WITH seasonality ($\\{{1,365,366\\}}$ vs $\\{{1,2,3,365,366\\}}$).",
                              f"tab:s1_seas_{branch}")
        secs.append("\n\n".join([rf"\subsection{{{bl}: no-seasonality vs seasonality}}", t_no, t_seas]))

        all4 = {s: [(lk, m(s, "stage1", lk, branch)) for lk in GAS_LAG_SETS] for s in SHORTS}
        # store the ALREADY-FORMATTED (LaTeX-safe) label, not the raw lag key
        # (which contains literal underscores) -- this dict is reused
        # directly as a table label downstream (best-1-vs-2-TVP here, and
        # the cross-stage "overall best model" table in build()).
        overall[branch] = {}
        for s in SHORTS:
            lk, mm = select(all4[s])
            overall[branch][s] = (LAG_LABEL.get(lk, lk) if lk else None, mm)
    t1 = winner_table({s: [overall["phi_only"][s]] for s in SHORTS}, lambda l: l,
                      "Stage 1: best 1-TVP model overall (best of all 4 lag sets).", "tab:s1_best_1tvp")
    t2 = winner_table({s: [overall["phi_xi"][s]] for s in SHORTS}, lambda l: l,
                      "Stage 1: best 2-TVP model overall (best of all 4 lag sets).", "tab:s1_best_2tvp")
    secs.append("\n\n".join([r"\subsection{Best 1 TVP vs best 2 TVP overall}", t1, t2]))
    return secs, overall


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def stage2_section():
    secs = [r"\section{Stage 2 -- weather covariates}"]
    overall = {}
    for branch in ("phi_only", "phi_xi"):
        bl = BRANCH_LBL[branch]
        cov_secs = []
        for cov in STAGE2_COV:
            cands = {s: [(lk, m(s, "stage2", f"{lk}_{cov}", branch)) for lk in GAS_LAG_SETS] for s in SHORTS}
            t = winner_table(cands, lambda l: LAG_LABEL.get(l, l),
                             f"Stage 2 [{bl}, {S2_LBL[cov]}]: best lag set.", f"tab:s2_{cov}_{branch}")
            cov_secs.append(t)
        secs.append("\n\n".join([rf"\subsection{{{bl}: covariate-set comparison}}"] + cov_secs))

        no_seas = {s: [((cov, lk), m(s, "stage2", f"{lk}_{cov}", branch))
                       for cov in STAGE2_COV for lk in ("L1", "L123")] for s in SHORTS}
        seas = {s: [((cov, lk), m(s, "stage2", f"{lk}_{cov}", branch))
                    for cov in STAGE2_COV for lk in ("L1_365_366", "L123_365_366")] for s in SHORTS}
        fmt = lambda cl: f"{S2_LBL[cl[0]]}/{LAG_LABEL[cl[1]]}"
        t_no = winner_table(no_seas, fmt, f"Stage 2 [{bl}]: best config WITHOUT seasonality.", f"tab:s2_noseas_{branch}")
        t_seas = winner_table(seas, fmt, f"Stage 2 [{bl}]: best config WITH seasonality.", f"tab:s2_seas_{branch}")
        secs.append("\n\n".join([rf"\subsection{{{bl}: no-seasonality vs seasonality (best covariate set)}}", t_no, t_seas]))

        all16 = {s: [((cov, lk), m(s, "stage2", f"{lk}_{cov}", branch))
                     for cov in STAGE2_COV for lk in GAS_LAG_SETS] for s in SHORTS}
        # store the ALREADY-FORMATTED label (raw (cov,lag) tuples contain
        # literal underscores -- unsafe to stringify directly into LaTeX).
        overall[branch] = {}
        for s in SHORTS:
            cl, mm = select(all16[s])
            overall[branch][s] = (f"{S2_LBL[cl[0]]}/{LAG_LABEL[cl[1]]}" if cl else None, mm)

    t1 = winner_table({s: [overall["phi_only"][s]] for s in SHORTS}, lambda l: l,
                      "Stage 2: best 1-TVP model overall.", "tab:s2_best_1tvp")
    t2 = winner_table({s: [overall["phi_xi"][s]] for s in SHORTS}, lambda l: l,
                      "Stage 2: best 2-TVP model overall.", "tab:s2_best_2tvp")
    secs.append("\n\n".join([r"\subsection{Best 1 TVP vs best 2 TVP overall}", t1, t2]))
    return secs, overall


# ---------------------------------------------------------------------------
# Stage 3 / 4 / 4S (Harvey)
# ---------------------------------------------------------------------------

def harvey_stage_section(stage, name):
    secs = [rf"\section{{{name}}}"]
    if stage == "stage4S":
        secs.append(
            r"\emph{Note}: Stage 4S has no long component ($\varphi_L,\kappa_L,\Gamma_L$ do not "
            r"exist -- \texttt{include\_long=False}), and ENSO enters the Harvey recursion only "
            r"through $\Gamma_L$. Consequently \textbf{dew $\equiv$ dew+ENSO} and "
            r"\textbf{dew+temp $\equiv$ dew+temp+ENSO} for this stage specifically -- the ENSO "
            r"dummies are structurally unable to affect the fit here, so the pairs are byte-for-byte "
            r"identical fits, not a computation error. They are still both shown for direct "
            r"row-by-row comparability with Stage 3/4."
        )
    overall = {}
    for branch in ("phi_only", "phi_xi"):
        bl = BRANCH_LBL[branch]
        cov_secs = []
        for cov in HARVEY_COV5:
            metrics_by_loc = {s: m(s, stage, cov, branch) for s in SHORTS}
            t = direct_table(metrics_by_loc, f"{name} [{bl}, {HC_LBL[cov]}].", f"tab:{stage}_{cov}_{branch}")
            cov_secs.append(t)
        secs.append("\n\n".join([rf"\subsection{{{bl}: covariate comparison}}"] + cov_secs))
        cands = {s: [(cov, m(s, stage, cov, branch)) for cov in HARVEY_COV5] for s in SHORTS}
        overall[branch] = {s: select(cands[s]) for s in SHORTS}

    fmt = lambda c: HC_LBL.get(c, c) if c else "--"
    t1 = winner_table({s: [(overall["phi_only"][s][0], overall["phi_only"][s][1])] for s in SHORTS}, fmt,
                      f"{name}: best 1-TVP model overall (best covariate config).", f"tab:{stage}_best_1tvp")
    t2 = winner_table({s: [(overall["phi_xi"][s][0], overall["phi_xi"][s][1])] for s in SHORTS}, fmt,
                      f"{name}: best 2-TVP model overall (best covariate config).", f"tab:{stage}_best_2tvp")
    secs.append("\n\n".join([r"\subsection{Best 1 TVP vs best 2 TVP overall}", t1, t2]))

    # overall best of this stage (either branch, best covariate) -- used by
    # the stage3/4/4S cross-stage comparisons below. Re-tag each branch's
    # winning label with the branch itself so the final config name shows
    # BOTH which covariate set and which branch (1/2 TVP) won.
    best_overall = {}
    for s in SHORTS:
        tagged = []
        for branch in ("phi_only", "phi_xi"):
            cov, mm = overall[branch][s]
            if cov is not None:
                tagged.append((f"{HC_LBL.get(cov, cov)}/{BRANCH_LBL[branch]}", mm))
        best_overall[s] = select(tagged)
    return secs, overall, best_overall


def cross_stage_table(best_by_stage_loc, stage_label, key):
    rows = []
    for short in SHORTS:
        sub_label, mm = best_by_stage_loc.get(short, (None, None))
        row = {"Location": DISP[short], "config": sub_label or "--"}
        row.update(_metric_row(mm))
        rows.append(row)
    return tbl(rows, ["config"] + METRIC_COLS, f"Best overall model, {stage_label}.", f"tab:best_{key}")


# ---------------------------------------------------------------------------
# Stage 5
# ---------------------------------------------------------------------------

def stage5_section(overall_best_all):
    secs = [r"\section{Stage 5 -- location dynamics for $\varphi$}"]
    rows = []
    for short in SHORTS:
        r = json.loads((ROOT / "artifacts" / "rerun_10092026_phiL_sigmoid" / short / "stage5" / "best" / "result.json").read_text())
        rows.append({"Location": DISP[short],
                     "base stage": r.get("base_stage", "--"),
                     "covariates": HC_LBL.get(r.get("cov", ""), r.get("cov", "--")),
                     "base branch": BRANCH_LBL.get(r.get("base_branch", ""), r.get("base_branch", "--"))})
    t_desc = tbl(rows, ["base stage", "covariates", "base branch"],
                "Stage 5: which earlier config's $\\varphi$ dynamics were carried forward "
                "(selected by gradient tier, then OOS CRPS, then OOS RMSE), with $\\xi$ "
                "additionally made time-varying.", "tab:s5_desc")
    secs.append("\n\n".join([r"\subsection{Chosen $\varphi$ dynamics}", t_desc]))

    def narrow_row(mm):
        mm = mm or {}
        return {"grad": _f(mm.get("grad_inf"), 2), "CRPS oos": _f(mm.get("crps_oos"), 2),
                "tw95": _f(mm.get("tw95")), "tw98": _f(mm.get("tw98"))}

    rows_s5, rows_best = [], []
    for short in SHORTS:
        mm5 = m(short, "stage5", "best", "phi_xi")
        rows_s5.append({"Location": DISP[short], **narrow_row(mm5)})
        lbl, mmb = overall_best_all.get(short, (None, None))
        rows_best.append({"Location": DISP[short], "model": lbl or "--", **narrow_row(mmb)})
    t_s5 = tbl(rows_s5, ["grad", "CRPS oos", "tw95", "tw98"], "Stage 5 model.", "tab:s5_metrics")
    t_best = tbl(rows_best, ["model", "grad", "CRPS oos", "tw95", "tw98"],
                "Best model overall across every stage (same selection rule).", "tab:overall_best")
    secs.append("\n\n".join([r"\subsection{Stage 5 vs.\ the best model overall}", t_s5, t_best]))
    return secs


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def build():
    REP.mkdir(parents=True, exist_ok=True)
    all_secs = []

    intro = "\n\n".join([
        r"\section*{Methodology}",
        r"Every comparison below applies the same selection rule, independently per location, "
        r"whenever multiple candidate configs must be reduced to one: (1) tier by checked "
        r"gradient $\infty$-norm ($<1$, then $<100$, then anything else); (2) within the best "
        r"tier, prefer the candidate where the largest fraction of individual parameters satisfy "
        r"$|\partial\text{obj}/\partial\theta_i|/n_{\text{train}} \le 10^{-3}$; (3) tie-break by "
        r"lowest rolling 1-step OOS CRPS; (4) if CRPS values are within 1\% of each other, break "
        r"by lowest rolling 1-step OOS RMSE. Winners can differ by location. Every comparison is "
        r"shown as one table PER OPTION (rows = locations), stacked vertically, so a table never "
        r"overflows the page width and the location-specific winner is always visible. "
        r"Stage 1/2/4S metrics come from \texttt{artifacts/rerun\_10092026/}; Stage 3/4/5 use the "
        r"sigmoid-bounded $\varphi_L$ parameterisation (\texttt{phi\_L\_max=1.0}) from "
        r"\texttt{artifacts/rerun\_10092026\_phiL\_sigmoid/} (2026-09-11 fix). "
        r"Columns: grad = checked gradient $\infty$-norm; RMSE/CRPS oos = rolling 1-step-ahead "
        r"out-of-sample; RMSE/CRPS is = in-sample; tw95/tw98 = out-of-sample twCRPS at the "
        r"95\%/98\% threshold.",
    ])
    all_secs.append(intro)

    s1_secs, s1_overall = stage1_section()
    all_secs += s1_secs

    s2_secs, s2_overall = stage2_section()
    all_secs += s2_secs

    s3_secs, s3_overall, s3_best = harvey_stage_section("stage3", "Stage 3 -- Harvey long/short (no seasonal)")
    all_secs += s3_secs

    s4_secs, s4_overall, s4_best = harvey_stage_section("stage4", "Stage 4 -- Harvey long/short + damped seasonal")
    all_secs += s4_secs
    all_secs.append("\n\n".join([r"\subsection*{Stage 4 vs.\ Stage 3 (best overall each)}",
                                 cross_stage_table(s4_best, "Stage 4", "s4"),
                                 cross_stage_table(s3_best, "Stage 3", "s3_vs4")]))

    s4s_secs, s4s_overall, s4s_best = harvey_stage_section("stage4S", "Stage 4 structural -- one level + one seasonal")
    all_secs += s4s_secs
    all_secs.append("\n\n".join([r"\subsection*{Stage 4S vs.\ Stage 4 vs.\ Stage 3 (best overall each)}",
                                 cross_stage_table(s4s_best, "Stage 4S", "s4S_vs"),
                                 cross_stage_table(s4_best, "Stage 4", "s4_vs"),
                                 cross_stage_table(s3_best, "Stage 3", "s3_vs")]))

    # overall best model across every stage (for Stage 5's comparison)
    overall_best_all = {}
    for s in SHORTS:
        cands = [
            (f"S1-1TVP/{s1_overall['phi_only'][s][0]}", s1_overall["phi_only"][s][1]),
            (f"S1-2TVP/{s1_overall['phi_xi'][s][0]}", s1_overall["phi_xi"][s][1]),
            (f"S2-1TVP/{s2_overall['phi_only'][s][0]}", s2_overall["phi_only"][s][1]),
            (f"S2-2TVP/{s2_overall['phi_xi'][s][0]}", s2_overall["phi_xi"][s][1]),
            (f"S3-best/{s3_best[s][0]}", s3_best[s][1]),
            (f"S4-best/{s4_best[s][0]}", s4_best[s][1]),
            (f"S4S-best/{s4s_best[s][0]}", s4s_best[s][1]),
        ]
        overall_best_all[s] = select(cands)
    all_secs += stage5_section(overall_best_all)

    body = "\n\n".join([r"\begin{document}",
                        r"\title{Stage-by-stage model comparison}\author{}\date{2026-09-11}\maketitle"]
                       + all_secs + [r"\end{document}"])
    (REP / "report.tex").write_text(PRE + "\n" + body, encoding="utf-8")
    print("wrote report.tex")

    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                       cwd=REP, capture_output=True, text=True)
    log = (REP / "report.log").read_text(errors="ignore") if (REP / "report.log").exists() else ""
    ok = (REP / "report.pdf").exists() and "Fatal error" not in log
    print("report.pdf:", "OK" if ok else "FAILED (see report.log)")


if __name__ == "__main__":
    build()
