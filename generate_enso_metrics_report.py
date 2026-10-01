"""
2026-09-12: ENSO-phase-conditional OOS metrics, reports/ENSOmetrics/report.tex.

For each location, two approaches (each its own stacked block of tables):
  A. "Best GAS(p,q)" -- the single best model overall across every stage/
     branch (same selection as generate_ml_comparison_report.best_overall_id:
     gradient tier, then per-parameter convergence fraction, then OOS CRPS
     [1% relative tolerance], then OOS RMSE).
  B. "Long-short + ENSO" -- the best Harvey model (Stage 3 or 4 only --
     Stage 4S has no long component, so it cannot be "long-short") whose
     covariate config includes ENSO (none/dew_enso/dew_temp_enso -- ENSO
     enters only via the long block), same selection rule restricted to
     that candidate set.

For each approach, three PER-CONDITION tables (El Nino / La Nina / neither
-- any condition that never occurs in the test period is dropped), 3
columns = the conditions present, 4 rows = RMSE/CRPS/twCRPS95/twCRPS98,
stacked one below the other (approach A's block, then approach B's block).
Bold = the best (lowest) value in each row across the conditions shown.

MECHANICS: this is NOT a new fit. Each candidate's cached theta is used to
re-run simulate_oos (already-fitted state paths), then the SAME per-day
CRPS/twCRPS/predictive-mean computation compute_extended_oos_metrics uses
internally is replicated day-by-day (same n_draws/seed, same GLOBAL
twCRPS thresholds computed once from the full test period -- exactly
matching every other twCRPS number already reported this project), and
only the final AGGREGATION step is restricted to the subset of test days
in each ENSO phase. ENSO phase per test day comes from the (z-scored,
binary) el_nino_t/la_nina_t dummy columns already in the data loader.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

import generate_ml_comparison_report as G  # select()/m()/metrics.json/trees/SHORTS/DISP_LONG
import collect_stage_comparison_metrics as C  # build_model() -- generic reconstruction across stages
import run_rerun_10092026 as D
from data.station_loader import STATION_REGISTRY
from diagnostics.scoring import _weighted_mean_abs_pairwise

ROOT = Path(__file__).parent
REP = ROOT / "reports" / "ENSOmetrics"
SHORTS = G.SHORTS
DISP = G.DISP_LONG
TWCRPS_LEVELS = (0.95, 0.98)
N_DRAWS, SEED = 500, 42
METRIC_ROWS = ["RMSE", "CRPS", "twCRPS95", "twCRPS98"]


def best_overall(short):
    """Re-derive generate_ml_comparison_report.best_overall_id's winner
    (best GAS(p,q) model overall)."""
    return G.best_overall_id(short)


def enso_longshort_best(short):
    """Best Harvey (Stage 3/4 only -- long-short requires include_long)
    config whose covariates include ENSO."""
    cands = []
    for stage in ("stage3", "stage4"):
        for cov in ("enso", "dew_enso", "dew_temp_enso"):
            for branch in ("phi_only", "phi_xi"):
                cands.append(((stage, cov, branch), G.m(short, stage, cov, branch)))
    return G.select(cands)


def per_day_metrics(model, oos, y_test):
    """Replicates compute_extended_oos_metrics's per-day CRPS/twCRPS/
    predictive-mean loop exactly (same seed, same draw order, same
    GLOBAL twCRPS thresholds from the full test period) so aggregating
    over ANY subset of days reproduces what the full-period number would
    have been, restricted to that subset."""
    rng = np.random.default_rng(SEED)
    n = len(y_test)
    tv_names, static = oos["tv_names"], oos["static"]
    has_ppf = hasattr(model.dist, "ppf")
    wet_y = y_test[y_test > 0]
    thr = {lv: float(np.quantile(wet_y, lv)) if len(wet_y) > 0 else 0.0 for lv in TWCRPS_LEVELS}

    crps = np.zeros(n)
    tw = {lv: np.zeros(n) for lv in TWCRPS_LEVELS}
    pred_mean = np.zeros(n)

    for i in range(n):
        pi_t = float(oos["pi_oos"][i])
        call = {nm: float(oos["f_arr_oos"][i, j]) for j, nm in enumerate(tv_names)}
        call.update(static)
        obs = float(y_test[i])

        is_pos = rng.binomial(1, pi_t, size=N_DRAWS)
        draws = np.zeros(N_DRAWS)
        n_pos = int(is_pos.sum())
        if n_pos > 0 and has_ppf:
            try:
                draws[is_pos == 1] = model.dist.ppf(rng.uniform(1e-8, 1 - 1e-8, size=n_pos), **call)
            except Exception:
                pass

        s = np.sort(draws)
        crps[i] = float(np.mean(np.abs(draws - obs)) - 0.5 * _weighted_mean_abs_pairwise(s, np.ones(len(s))))
        for lv, t in thr.items():
            w = (draws > t).astype(float)
            e1 = float(np.mean(w * np.abs(draws - obs)))
            ws = (s > t).astype(float)
            e2 = _weighted_mean_abs_pairwise(s, ws)
            tw[lv][i] = e1 - 0.5 * e2

        if hasattr(model.dist, "mean"):
            gb2_mean = model.dist.mean(**call)
            if not np.isfinite(gb2_mean):
                gb2_mean = np.exp(call.get("phi", 0.0))
        else:
            gb2_mean = np.exp(call.get("phi", 0.0))
        pred_mean[i] = pi_t * gb2_mean

    return crps, tw, pred_mean


def enso_masks(data):
    cn = data["covariate_col_names"]["enso_dummy_current"]
    bt = data["covariate_blocks_test"]["enso_dummy_current"]
    el = bt[:, cn.index("el_nino_t")] > 0
    la = bt[:, cn.index("la_nina_t")] > 0
    none_ = ~el & ~la
    return {"El Ni\\~no": el, "La Ni\\~na": la, "Neither": none_}


def approach_metrics(short, ident):
    """Returns {condition_label: {metric_row: value}} for the conditions
    that actually occur (n>0) in this location's test period."""
    if ident is None:
        return {}
    stage, cfg, branch = ident
    loc_full = next(l for l in D.LOCATIONS if STATION_REGISTRY[l].get("short", l) == short)
    data = D._load_data(loc_full)
    res_dir = G._resolve_cfg_dir(short, ident)
    import json
    res = json.loads((res_dir / "result.json").read_text())
    theta = np.load(res_dir / "theta.npy")
    model, _filt_kwargs = C.build_model(loc_full, stage, cfg, branch, res, data)
    # simulate_oos's kwarg names differ by model family; reuse the SAME
    # mapping _evaluate()/collect_stage_comparison_metrics.py use (distinct
    # from _filt_kwargs above, which is train-only, for _run_filter).
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

    y_tr, y_te = data["y_train"], data["y_test"]
    oos = model.simulate_oos(theta, y_tr, y_te, **ok)
    crps, tw, pred_mean = per_day_metrics(model, oos, y_te)
    masks = enso_masks(data)

    out = {}
    for label, mask in masks.items():
        n = int(mask.sum())
        if n == 0:
            continue
        rmse = float(np.sqrt(np.mean((pred_mean[mask] - y_te[mask]) ** 2)))
        out[label] = {
            "RMSE": rmse, "CRPS": float(np.mean(crps[mask])),
            "twCRPS95": float(np.mean(tw[0.95][mask])), "twCRPS98": float(np.mean(tw[0.98][mask])),
            "n": n,
        }
    return out


def _f(v):
    return f"{v:.3f}" if (v is not None and np.isfinite(v)) else "--"


def make_table(metrics_by_cond, caption, label):
    conds = list(metrics_by_cond.keys())
    if not conds:
        return f"% {caption}: no ENSO phase occurs in this test period -- skipped.\n"
    body = [r"\begin{table}[H]\centering\footnotesize\setlength{\tabcolsep}{8pt}",
            rf"\caption{{{caption}}}\label{{{label}}}",
            r"\begin{tabular}{l" + "r" * len(conds) + "}", r"\toprule",
            "Metric & " + " & ".join(f"{c} ($n={metrics_by_cond[c]['n']}$)" for c in conds) + r" \\",
            r"\midrule"]
    for row in METRIC_ROWS:
        vals = [metrics_by_cond[c][row] for c in conds]
        finite = [v for v in vals if np.isfinite(v)]
        best = min(finite) if finite else None
        cells = []
        for v in vals:
            s = _f(v)
            if best is not None and np.isfinite(v) and abs(v - best) < 1e-9:
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        body.append(row + " & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(body)


def _cfg_txt(ident):
    if ident is None:
        return "n/a"
    stage, cfg, branch = ident
    brl = "1 TVP" if branch == "phi_only" else "2 TVP"
    return f"{stage}/{cfg.replace('_', '-')}/{brl}"


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
        r"For each location, two approaches: (A) the single best GAS(p,q) model overall across "
        r"every stage/branch (gradient tier, then per-parameter convergence fraction, then OOS "
        r"CRPS with 1\% relative tolerance, then OOS RMSE); (B) the best Stage 3/4 (long-short) "
        r"model whose covariates include ENSO, same rule, restricted to that candidate set -- "
        r"Stage 4S is excluded here since it has no long component. For each approach, rolling "
        r"1-step-ahead OOS RMSE/CRPS/twCRPS95/twCRPS98 are recomputed from the cached fitted "
        r"$\theta$ (no refitting) and split by the realised ENSO phase of the test day (El Ni\~no "
        r"/ La Ni\~na / neither); a phase that never occurs in a location's test period is "
        r"omitted from its table. Bold marks the best (lowest) value in each row. Stage 3/4 use "
        r"the sigmoid-bounded $\varphi_L$ parameterisation (2026-09-11 fix)."
    )
    secs = [r"\begin{document}",
            r"\title{ENSO-phase-conditional out-of-sample metrics}\author{}\date{2026-09-12}\maketitle",
            intro]

    for s in SHORTS:
        secs.append(rf"\section{{{DISP[s]}}}")
        id_a, _ = best_overall(s)
        id_b, _ = enso_longshort_best(s)
        ma = approach_metrics(s, id_a)
        mb = approach_metrics(s, id_b)
        secs.append(rf"\subsection*{{A: best GAS(p,q) overall -- {_cfg_txt(id_a).replace('_', chr(92)+chr(95))}}}")
        secs.append(make_table(ma, f"{DISP[s]}: best GAS(p,q) overall, by ENSO phase.", f"tab:{s}_A"))
        secs.append(rf"\subsection*{{B: long-short + ENSO -- {_cfg_txt(id_b).replace('_', chr(92)+chr(95))}}}")
        secs.append(make_table(mb, f"{DISP[s]}: long-short model with ENSO, by ENSO phase.", f"tab:{s}_B"))

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
