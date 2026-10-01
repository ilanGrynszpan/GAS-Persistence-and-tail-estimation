"""
2026-09-12: occurrence-model fit report, reports/occurrence_fit/.

1. Table: out-of-sample Brier skill score per location, for that location's
   best GAS(p,q) model overall (same selection as
   generate_ml_comparison_report.best_overall_id). BS/BS_clim/BSS are read
   straight from that config's cached result.json -- computed at fit time
   exactly per the formula in this task's prompt:
       BS      = (1/T) sum_t (pi_{t|t-1} - I(y_t>0))^2                 [[test sample]]
       BS_clim = (1/T) sum_t (p_clim,m(t) - I(y_t>0))^2                [[p_clim from TRAIN]]
       BSS     = 1 - BS/BS_clim
   No fitting happens here.

2. Mosaic (3 rows x 2 cols, one panel per location): grouped bar chart of
   the OOS monthly average pi_{t|t-1} vs the OOS monthly empirical wet-day
   proportion, side by side per calendar month. Occurrence = solid full
   colour; pi = a brighter shade with a dot-hatch texture, so the two
   remain distinguishable in black-and-white print. No title, no legend
   (both drawn from this report's caption/table instead).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generate_ml_comparison_report as G
import collect_stage_comparison_metrics as C
import run_rerun_10092026 as D
from data.station_loader import STATION_REGISTRY

ROOT = Path(__file__).parent
REP = ROOT / "reports" / "occurrence_fit"
SHORTS = G.SHORTS
DISP = G.DISP_LONG
MONTH_LBL = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

COLOR_OCC = "#1a1a3d"     # occurrence: solid, dark, full colour
COLOR_PI = "#7fb2ff"      # pi: brighter shade
HATCH_PI = "..."


def winner_and_result(short):
    ident, _mm = G.best_overall_id(short)
    cdir = G._resolve_cfg_dir(short, ident)
    res = json.loads((cdir / "result.json").read_text())
    return ident, cdir, res


def bss_table():
    rows = []
    for s in SHORTS:
        ident, _cdir, res = winner_and_result(s)
        stage, cfg, branch = ident
        brl = "1 TVP" if branch == "phi_only" else "2 TVP"
        rows.append({
            "Location": DISP[s], "GAS/SD model": f"{stage}/{cfg.replace('_', '-')}/{brl}",
            "$BS_{clim}$": f"{res['BS_clim']:.4f}", "$BS_{GAS/SD}$": f"{res['BS']:.4f}",
            "BSS": f"{res['BSS']:.4f}",
        })
    df = pd.DataFrame(rows).set_index("Location")
    cols = ["GAS/SD model", "$BS_{clim}$", "$BS_{GAS/SD}$", "BSS"]
    body = [r"\begin{table}[H]\centering\small\setlength{\tabcolsep}{8pt}",
            r"\caption{Out-of-sample Brier skill score of $\pi_{t|t-1}$ vs.\ the monthly "
            r"climatological reference, best GAS(p,q) model overall per location.}",
            r"\label{tab:bss}",
            r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
            "Location & " + " & ".join(cols) + r" \\", r"\midrule"]
    for idx, row in df[cols].iterrows():
        body.append(str(idx) + " & " + " & ".join(str(v) for v in row) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(body)


def monthly_pi_and_occurrence(short, ident, res):
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
    pi_oos = np.asarray(oos["pi_oos"], dtype=float)
    n = min(len(pi_oos), len(y_te))
    pi_oos, y_te, dates = pi_oos[:n], y_te[:n], pd.DatetimeIndex(data["dates_test"])[:n]
    months = dates.month.values
    wet = (y_te > 0).astype(float)

    avg_pi = np.full(12, np.nan)
    avg_wet = np.full(12, np.nan)
    for m in range(1, 13):
        mask = months == m
        if mask.any():
            avg_pi[m - 1] = np.mean(pi_oos[mask])
            avg_wet[m - 1] = np.mean(wet[mask])
    return avg_pi, avg_wet


def build_mosaic():
    fig, axes = plt.subplots(3, 2, figsize=(9, 9.5))
    x = np.arange(12)
    w = 0.38
    for idx, s in enumerate(SHORTS):
        r, c = idx // 2, idx % 2
        ax = axes[r][c]
        ident, _cdir, res = winner_and_result(s)
        avg_pi, avg_wet = monthly_pi_and_occurrence(s, ident, res)
        ax.bar(x - w / 2, avg_wet, width=w, color=COLOR_OCC, edgecolor="none", label="occurrence")
        ax.bar(x + w / 2, avg_pi, width=w, color=COLOR_PI, edgecolor="black", linewidth=0.4,
              hatch=HATCH_PI, label="pi")
        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_LBL, fontsize=6, rotation=45)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_ylabel(DISP[s].split(" (")[0], fontsize=8)
    fig.tight_layout()
    REP.mkdir(parents=True, exist_ok=True)
    fig.savefig(REP / "pi_vs_occurrence_mosaic.pdf")
    fig.savefig(REP / "pi_vs_occurrence_mosaic.png", dpi=150)
    plt.close(fig)
    print("wrote pi_vs_occurrence_mosaic.pdf/.png")


def build():
    REP.mkdir(parents=True, exist_ok=True)
    build_mosaic()

    pre = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}\usepackage{float}
\emergencystretch=3em\sloppy
"""
    intro = (
        r"\section*{Methodology}"
        r"Predictive performance for the modeling of rainfall occurrence, $\pi_{t|t-1}$, is "
        r"evaluated through Brier skill scores (BSS) \cite{Mason2004}. Brier scores (BS) are "
        r"defined as $BS = \frac{1}{T}\sum_{t=1}^{T}(\pi_{t|t-1} - I(y_t > 0))^2$, where $T$ "
        r"represents the size of the test sample, and $I(y_t > 0)$ is one if $y_t > 0$ and zero "
        r"otherwise. Let the climatological Brier score, the reference approach, be "
        r"$BS_{clim} = \frac{1}{T}\sum_{t=1}^{T}(p_{clim, m(t)} - I(y_t > 0))^2$, where $m(t)$ is "
        r"the month relative to day $t$ in the test sample, and $p_{clim, m(t)}$ is the empirical "
        r"relative frequency of train sample days with precipitation for that month and location. "
        r"Then, the Brier skill score is"
        r"\begin{equation}BSS = 1 - \frac{BS}{BS_{clim}}.\end{equation}"
        r"The GAS/SD model used for each location is that location's best model overall across "
        r"every stage and TVP branch (gradient tier, then per-parameter convergence fraction, "
        r"then OOS CRPS with 1\% relative tolerance, then OOS RMSE)."
    )

    fig_caption = (
        r"Out-of-sample monthly average $\pi_{t|t-1}$ (brighter, dot-hatched bars) vs.\ the "
        r"out-of-sample monthly empirical proportion of wet days (solid dark bars), for each "
        r"location's best GAS(p,q) model (same model as the table above). Occurrence is drawn in "
        r"full/solid colour and $\pi$ in a brighter shade with a dotted texture so the two remain "
        r"distinguishable in black-and-white print."
    )

    body = "\n\n".join([
        r"\begin{document}",
        r"\title{Occurrence model fit}\author{}\date{2026-09-12}\maketitle",
        intro,
        r"\section{Brier skill score}",
        bss_table(),
        r"\section{Monthly $\pi_{t|t-1}$ vs.\ occurrence}",
        r"\begin{figure}[H]\centering\includegraphics[width=0.92\textwidth]{pi_vs_occurrence_mosaic.pdf}"
        rf"\caption{{{fig_caption}}}\end{{figure}}",
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
