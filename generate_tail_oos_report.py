"""
2026-09-12: reports/tail_oos/report.tex -- the same exceedance/Kupiec table
as reports/kupiec/, PLUS twCRPS95/twCRPS98 (2 extra columns = 8 total),
but for a NON-ROLLING h=90-day-ahead forecast instead of the rolling
1-step-ahead forecast: from each of up to 8 origins spaced 90 days apart
in the test period, the model is simulated forward 90 days using its own
SIMULATED y beyond the origin (not real data, unlike the rolling case),
and every metric below is evaluated only at the h=90 endpoint of that
trajectory, averaged over origins (same machinery as every other
multi-step table this project uses: diagnostics/multistep_forecast.py's
multistep_metrics(), which is what stage3/4/4S's cached multistep.csv
files already come from -- this script just also asks it for the
"1-in-30-day" exceedance level, which was never part of that sweep).

Kupiec calculation is identical to reports/kupiec/ (see that report's
methodology section / diagnostics/exceedance.py) -- the only difference
here is what counts as a "violation": for the rolling case it was
PIT_t > q on every OOS day; here it is actual_{t0+90} > the q-quantile of
that origin's own 90-day-ahead Monte Carlo draws, pooled across the (few,
~8) available origins. With only ~8 origins, the Kupiec test here has very
little power -- noted explicitly in the report.
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
from diagnostics.multistep_forecast import multistep_metrics

ROOT = Path(__file__).parent
REP = ROOT / "reports" / "tail_oos"
SHORTS = G.SHORTS
DISP = G.DISP_LONG
Q_1IN30 = 1.0 - 1.0 / 30.0
H = 90


def h90_row(short, ident, res):
    stage, cfg, branch = ident
    loc_full = next(l for l in D.LOCATIONS if STATION_REGISTRY[l].get("short", l) == short)
    data = D._load_data(loc_full)
    model, _filt_kwargs = C.build_model(loc_full, stage, cfg, branch, res, data)

    if stage == "stage1":
        mk = {}
    elif stage == "stage2":
        Xtr, _ = D._cov_cols(data, res["cov"], "train")
        Xte, _ = D._cov_cols(data, res["cov"], "test")
        mk = {"X_train": Xtr, "X_test": Xte}
    else:
        Xl_tr, _, Xs_tr, _ = D._harvey_cov(data, res["cov"], "train")
        Xl_te, _, Xs_te, _ = D._harvey_cov(data, res["cov"], "test")
        mk = {"X_long_train": Xl_tr, "X_long_test": Xl_te, "X_short_train": Xs_tr, "X_short_test": Xs_te}

    theta = np.load(G._resolve_cfg_dir(short, ident) / "theta.npy")
    y_tr, y_te = data["y_train"], data["y_test"]
    origins = list(range(0, len(y_te) - H - 1, 90))[:8] or [0]
    ms = multistep_metrics(model, theta, y_tr, y_te, mk, origins=origins, horizons=[H],
                           n_draws=200, seed=0, twcrps_levels=(0.95, 0.98),
                           exceedance_levels=(0.95, 0.98, Q_1IN30))
    row = ms.iloc[0]
    k30 = int(Q_1IN30 * 100)  # 96, per multistep_metrics' int(lv*100) column-naming convention
    n = int(row["n_origins"])
    return {
        "n_origins": n,
        "rate95": row.get("exc_95"), "kup95": row.get("kup95_p"),
        "rate98": row.get("exc_98"), "kup98": row.get("kup98_p"),
        "rate1in30": row.get(f"exc_{k30}"), "kup1in30": row.get(f"kup{k30}_p"),
        "tw95": row.get("twcrps_95"), "tw98": row.get("twcrps_98"),
    }


def _f(v, d=3):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "--"
    return f"{x:.{d}f}" if np.isfinite(x) else "--"


def _ratio_cell(rate, p, q, n):
    """Same theoretical/empirical exceedance-count ratio as reports/kupiec/
    ((alpha*n)/n_viol, alpha=1-q), reconstructing the integer violation
    count n_viol = round(rate*n) from the small origin count n (exact,
    since rate is already an exact k/n fraction here)."""
    if rate is None or not np.isfinite(rate) or n == 0:
        return "--"
    alpha = 1.0 - q
    n_viol = int(round(rate * n))
    s = r"$\infty$" if n_viol == 0 else f"{(alpha * n) / n_viol:.3f}"
    if p is not None and np.isfinite(p) and p < 0.05:
        s += "*"
    return s


def build():
    REP.mkdir(parents=True, exist_ok=True)
    rows, idents, n_origins_seen = [], {}, set()
    for s in SHORTS:
        ident, _mm = G.best_by_tw95(s)
        d = G._resolve_cfg_dir(s, ident)
        res = json.loads((d / "result.json").read_text())
        idents[s] = ident
        r = h90_row(s, ident, res)
        n = r["n_origins"]
        n_origins_seen.add(n)
        rows.append({
            "Location": DISP[s],
            "rate95": _ratio_cell(r["rate95"], r["kup95"], 0.95, n), "kup95": _f(r["kup95"]),
            "rate98": _ratio_cell(r["rate98"], r["kup98"], 0.98, n), "kup98": _f(r["kup98"]),
            "rate1in30": _ratio_cell(r["rate1in30"], r["kup1in30"], Q_1IN30, n), "kup1in30": _f(r["kup1in30"]),
            "tw95": _f(r["tw95"]), "tw98": _f(r["tw98"]),
        })
        print(f"  {s}: {ident}  n_origins={r['n_origins']}  rate95={r['rate95']} rate98={r['rate98']} "
              f"rate1in30={r['rate1in30']} tw95={r['tw95']} tw98={r['tw98']}")

    df = pd.DataFrame(rows).set_index("Location")
    cols = ["rate95", "kup95", "rate98", "kup98", "rate1in30", "kup1in30", "tw95", "tw98"]

    pre = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage[margin=1.6cm]{geometry}
\usepackage{amsmath,amssymb,booktabs}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}\usepackage{float}
\emergencystretch=3em\sloppy
"""
    intro = (
        r"\section*{Methodology}"
        r"Same theoretical/empirical exceedance-count ratio and Kupiec (1995) test as "
        r"\texttt{reports/kupiec/} (see that report's methodology for the exact ratio and "
        r"$LR_{uc}$ formulas and the $N_1\in\{0,N\}$ boundary handling: ratio $=\alpha N/N_1$, "
        r"$\alpha=1-q$, $N_1=0$ shown as $\infty$), plus twCRPS95/twCRPS98, but for a NON-ROLLING "
        r"90-day-ahead forecast instead of the rolling 1-step-ahead one: from each of up to 8 "
        r"origins spaced 90 days apart in the test period, the model is simulated forward 90 days "
        r"using its OWN simulated trajectory beyond the origin (not real data), and every column "
        r"here is evaluated only at the $h=90$ endpoint, pooled over the available origins ($N$ = "
        r"number of origins here, not OOS days). A ``violation'' is therefore "
        r"$y_{t_0+90} > $ the $q$-quantile of that origin's own 90-day-ahead Monte Carlo "
        r"draws. With only a handful of origins, the Kupiec test here has much less power than "
        r"the rolling version -- treat its p-values as indicative, not definitive. Each location "
        r"uses the same model as \texttt{reports/kupiec/}: whichever candidate (any stage 1-4S, "
        r"either TVP branch, Stage 5 excluded) both optimised well (best gradient tier reached) "
        r"AND, among that tier, minimises out-of-sample twCRPS95."
    )
    n_origins = sorted(n_origins_seen)
    intro += (rf" Number of independent 90-day-ahead origins used: "
             rf"{n_origins[0] if len(n_origins) == 1 else str(n_origins)}.")

    body = [r"\begin{table}[H]\centering\footnotesize\setlength{\tabcolsep}{5pt}",
            r"\caption{Non-rolling $h=90$-day-ahead theoretical/empirical exceedance-count "
            r"ratio, Kupiec p-value, and "
            r"twCRPS95/98, best-converged, twCRPS95-minimising model per location. $^*$ = Kupiec "
            r"rejects at 5\%.}",
            r"\label{tab:tail_oos}",
            r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
            r" & \multicolumn{2}{c}{95\% quantile} & \multicolumn{2}{c}{98\% quantile} & "
            r"\multicolumn{2}{c}{1-in-30-day} & & \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            "Location & ratio & Kupiec $p$ & ratio & Kupiec $p$ & ratio & Kupiec $p$ & twCRPS95 & twCRPS98 \\\\",
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
        r"\caption{Which model each row above uses (same as reports/kupiec/).}\label{tab:tail_oos_models}",
        r"\begin{tabular}{ll}\toprule Location & GAS/SD model \\\midrule",
    ] + model_rows + [r"\bottomrule\end{tabular}\end{table}"])

    doc = "\n\n".join([
        r"\begin{document}",
        r"\title{Non-rolling $h=90$-day-ahead exceedance, Kupiec, and twCRPS}"
        r"\author{}\date{2026-09-12}\maketitle",
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
