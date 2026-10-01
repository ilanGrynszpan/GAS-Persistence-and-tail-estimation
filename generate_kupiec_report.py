"""
2026-09-12: exceedance rate + Kupiec test, reports/kupiec/report.tex.

6 rows (locations) x 6 columns (violation rate / Kupiec p-value, at each of
3 thresholds: the 95% and 98% time-varying quantiles, and the "1-in-30-day"
event, i.e. the 1 - 1/30 = 29/30-th quantile), for each location's best
GAS(p,q) model overall (same selection as
generate_ml_comparison_report.best_overall_id). 95%/98% figures are read
straight from that config's cached result.json (exc_95_1s/kup95_1s_p/
exc_98_1s/kup98_1s_p, already computed at fit time); the 1-in-30 level is
computed fresh here (no refit -- same cached theta) since it was never
part of any earlier sweep.

HOW KUPIEC IS CALCULATED (diagnostics/exceedance.py, this project's own
implementation of Kupiec 1995's unconditional-coverage LR test):
  1. "Violation" at quantile level q, for the 1-step-ahead ROLLING OOS
     forecasts: a violation on day t is PIT_t > q, i.e. the observed y_t
     exceeded the model's own time-varying q-quantile forecast for that
     day (PIT = the model's own CDF evaluated at the observation, using
     randomised-uniform tie-breaking on the zero mass exactly like every
     other quantile-residual/PIT computation in this project).
  2. Let N = number of OOS forecasts, N1 = number of violations,
     N0 = N - N1, alpha = 1 - q (the NOMINAL exceedance rate under correct
     calibration).
  3. Kupiec's LR_uc statistic compares the "correctly calibrated" null
     (exceedance probability fixed at alpha) against the saturated
     alternative (exceedance probability = the empirical rate p_hat=N1/N):
         LR_uc = -2 [ N1 ln(alpha) + N0 ln(1-alpha) - N1 ln(p_hat) - N0 ln(1-p_hat) ]
     with the N1 in {0, N} boundary handled by its correct limiting
     log-likelihood (=0) rather than blowing up at log(0) -- the common
     bug in naive Kupiec implementations, and NOT the same as this
     project's diagnostics/tests.py::_kupiec_statistic, which is written
     for lower-tail violations and returns (inf, 0.0) at that same
     boundary, mis-flagging the small-sample multi-step case.
  4. LR_uc ~ chi-squared(1) under the null; the reported p-value is
     chi2.sf(LR_uc, df=1). A LOW p-value (e.g. <0.05) means the observed
     exceedance rate is significantly different from the nominal rate --
     i.e. the model's quantile forecasts at that level are miscalibrated.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import generate_ml_comparison_report as G
import collect_stage_comparison_metrics as C
import run_rerun_10092026 as D
from data.station_loader import STATION_REGISTRY
from diagnostics.scoring import compute_extended_oos_metrics
from diagnostics.exceedance import exceedance_from_pit

ROOT = Path(__file__).parent
REP = ROOT / "reports" / "kupiec"
SHORTS = G.SHORTS
DISP = G.DISP_LONG
Q_1IN30 = 1.0 - 1.0 / 30.0


def three_level_stats(short, ident, res):
    """Fresh PIT-based exceedance + Kupiec at all 3 levels (95%, 98%,
    1-in-30-day), from the same PIT array, no refitting (cached theta)."""
    stage, cfg, branch = ident
    loc_full = next(l for l in D.LOCATIONS if STATION_REGISTRY[l].get("short", l) == short)
    data = D._load_data(loc_full)
    model, _filt_kwargs = C.build_model(loc_full, stage, cfg, branch, res, data)

    if stage == "stage1":
        ok = {}
    elif stage == "stage2":
        Xtr, _ = D._cov_cols(data, res["cov"], "train")
        Xte, _ = D._cov_cols(data, res["cov"], "test")
        ok = {"X_train": Xtr, "X_test": Xte}
    else:
        Xl_tr, _, Xs_tr, _ = D._harvey_cov(data, res["cov"], "train")
        Xl_te, _, Xs_te, _ = D._harvey_cov(data, res["cov"], "test")
        ok = {"X_long_train": Xl_tr, "X_long_test": Xl_te, "X_short_train": Xs_tr, "X_short_test": Xs_te}

    theta = np.load(G._resolve_cfg_dir(short, ident) / "theta.npy")
    y_tr, y_te = data["y_train"], data["y_test"]
    oos = model.simulate_oos(theta, y_tr, y_te, **ok)
    m = compute_extended_oos_metrics(model, oos, y_te, quantile_levels=(0.95, 0.98),
                                     twcrps_levels=(0.95, 0.98), brier_levels=(),
                                     n_draws=500, seed=42, return_raw=True)
    pit = np.asarray(m.get("_pit_values", []), dtype=float)
    out = {}
    for lbl, q in (("95", 0.95), ("98", 0.98), ("1in30", Q_1IN30)):
        out[lbl] = exceedance_from_pit(pit, q)
    return out


def _f(v, d=3):
    return f"{v:.{d}f}" if (v is not None and np.isfinite(v)) else "--"


def _ratio_cell(e, q):
    """2026-09-12 fix: report the THEORETICAL/EMPIRICAL exceedance-count
    ratio -- (alpha * n) / n_viol, alpha = 1-q -- instead of the raw
    empirical rate n_viol/n. ratio = 1 -> exactly nominal coverage;
    ratio > 1 -> fewer empirical violations than expected (conservative);
    ratio < 1 -> more empirical violations than expected (too narrow).
    n_viol = 0 makes this undefined (shown as $\\infty$: the empirical
    count is zero against a nonzero theoretical expectation)."""
    alpha = 1.0 - q
    n_viol, n = e["n_viol"], e["n"]
    if n_viol == 0:
        s = r"$\infty$"
    else:
        s = f"{(alpha * n) / n_viol:.3f}"
    p = e["kupiec_p"]
    if p is not None and np.isfinite(p) and p < 0.05:
        s += "*"
    return s, _f(p)


def build():
    REP.mkdir(parents=True, exist_ok=True)
    rows = []
    idents = {}
    for s in SHORTS:
        ident, _mm = G.best_by_tw95(s)
        d = G._resolve_cfg_dir(s, ident)
        res = json.loads((d / "result.json").read_text())
        idents[s] = ident

        stats3 = three_level_stats(s, ident, res)
        r95, kup95 = _ratio_cell(stats3["95"], 0.95)
        r98, kup98 = _ratio_cell(stats3["98"], 0.98)
        r30, kup30 = _ratio_cell(stats3["1in30"], Q_1IN30)

        rows.append({
            "Location": DISP[s],
            "rate95": r95, "kup95": kup95,
            "rate98": r98, "kup98": kup98,
            "rate1in30": r30, "kup1in30": kup30,
        })
        print(f"  {s}: {ident}  ratio95={r95} (n_viol={stats3['95']['n_viol']}/{stats3['95']['n']}) "
              f"ratio98={r98} (n_viol={stats3['98']['n_viol']}/{stats3['98']['n']}) "
              f"ratio1in30={r30} (n_viol={stats3['1in30']['n_viol']}/{stats3['1in30']['n']})")

    df = pd.DataFrame(rows).set_index("Location")
    cols = ["rate95", "kup95", "rate98", "kup98", "rate1in30", "kup1in30"]

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
        r"For each location, the model used is whichever candidate (any stage 1-4S, either TVP "
        r"branch, Stage 5 excluded) both optimised well (best gradient tier reached; a "
        r"non-finite OOS CRPS forces the worst tier regardless of gradient) AND, among that tier, "
        r"MINIMISES out-of-sample twCRPS95 -- not necessarily the same model as elsewhere in this "
        r"project's ``best overall'' tables. A violation on day $t$ is $PIT_t > q$: the "
        r"observation exceeded the model's own time-varying $q$-quantile forecast for that day. "
        r"Let $N_1$ = number of violations (the EMPIRICAL exceedance count) and $N$ = number of "
        r"rolling 1-step-ahead OOS forecasts, at each of three thresholds: the 95\% and 98\% "
        r"time-varying quantiles, and the ``1-in-30-day'' event (the "
        r"$q=1-\tfrac{1}{30}=96.\overline{6}\%$ quantile, $\alpha=1-q=\tfrac{1}{30}$). The "
        r"``ratio'' column is the THEORETICAL exceedance count expected under correct calibration "
        r"divided by the EMPIRICAL count actually observed, $\alpha N / N_1$: ratio $=1$ is exact "
        r"nominal coverage, $>1$ means fewer violations occurred than expected (the model is "
        r"conservative at that quantile), $<1$ means more violations occurred than expected (the "
        r"model's tail is too narrow); $N_1=0$ is shown as $\infty$. Kupiec's (1995) LR statistic "
        r"tests this same discrepancy formally, comparing $\alpha$ against the empirical rate "
        r"$\hat p = N_1/N$: "
        r"$LR_{uc} = -2\left[N_1\ln\alpha + N_0\ln(1-\alpha) - N_1\ln\hat p - N_0\ln(1-\hat p)\right]$"
        r", $\sim\chi^2_1$ under the null of correct calibration (the $N_1\in\{0,N\}$ boundary is "
        r"handled by its correct limiting log-likelihood, not by dropping to an ad hoc "
        r"$(\infty,0)$). A low p-value ($<0.05$, marked with $^*$ on the ratio) rejects correct "
        r"calibration at that quantile."
    )

    body = [r"\begin{table}[H]\centering\small\setlength{\tabcolsep}{6pt}",
            r"\caption{Rolling 1-step-ahead OOS theoretical/empirical exceedance-count ratio and "
            r"Kupiec p-value, best-converged, twCRPS95-minimising model per location. $^*$ = "
            r"Kupiec rejects at 5\%.}",
            r"\label{tab:kupiec}",
            r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r" & \multicolumn{2}{c}{95\% quantile} & \multicolumn{2}{c}{98\% quantile} & "
            r"\multicolumn{2}{c}{1-in-30-day} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            "Location & ratio & Kupiec $p$ & ratio & Kupiec $p$ & ratio & Kupiec $p$ \\\\",
            r"\midrule"]
    for idx, row in df[cols].iterrows():
        body.append(str(idx) + " & " + " & ".join(str(v) for v in row) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table_tex = "\n".join(body)

    model_rows = []
    for s in SHORTS:
        stage, cfg, branch = idents[s]
        brl = "1 TVP" if branch == "phi_only" else "2 TVP"
        model_rows.append(f"{DISP[s]} & {stage}/{cfg.replace('_', '-')}/{brl} \\\\")
    model_table = "\n".join([
        r"\begin{table}[H]\centering\small",
        r"\caption{Which model each row above uses.}\label{tab:kupiec_models}",
        r"\begin{tabular}{ll}\toprule Location & GAS/SD model \\\midrule",
    ] + model_rows + [r"\bottomrule\end{tabular}\end{table}"])

    doc = "\n\n".join([
        r"\begin{document}",
        r"\title{Exceedance rate and Kupiec test}\author{}\date{2026-09-12}\maketitle",
        intro, table_tex, model_table,
        r"\end{document}",
    ])
    (REP / "report.tex").write_text(pre + "\n" + doc, encoding="utf-8")
    print("wrote report.tex")

    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                       cwd=REP, capture_output=True, text=True)
    log = (REP / "report.log").read_text(errors="ignore") if (REP / "report.log").exists() else ""
    ok = (REP / "report.pdf").exists() and "Fatal error" not in log
    print("report.pdf:", "OK" if ok else "FAILED (see report.log)")


if __name__ == "__main__":
    build()
