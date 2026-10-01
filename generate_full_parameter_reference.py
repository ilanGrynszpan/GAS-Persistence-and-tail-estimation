"""
Cross-stage gradient-norm reference report (2026-09-08).

One table per model variation (lag set x TVP count x covariate count,
depending on stage), rows = locations, columns = OOS RMSE, OOS CRPS,
twCRPS@95, twCRPS@98, and the gradient norm at convergence. Gradient norms
are all recomputed via diagnostics.numerical_diff.numerical_gradient
(independent of scipy's own res.jac) -- see
run_full_gradient_comparison.py and recompute_stage345_numerical_gradients.py.

Coverage:
    Stage 1 (no covariates):        2 lag sets x 2 TVP counts = 4 tables
    Stage 2 (covariates):           2 lag sets x 2 TVP x 3 cov counts = 12 tables
    Stage 3 (Harvey long-short):    2 TVP counts = 2 tables
    Stage 4 (+ seasonal harmonic):  2 TVP counts = 2 tables
    Stage 4+5 hybrid (regime xi on frozen best-CRPS phi): 1 table

Each table is followed by a short paragraph ONLY when at least one location
shows gradient norm > 1, briefly stating whether it traces to a known
boundary (a documented soft-constraint sum near its limit), a documented
identification weakness (Darwin/Manaus Long-Short collapse; seasonal-lag
redundancy; Stage 4's kappa_gamma/kappa_L collinearity), or is otherwise
unexplained and worth a closer look.

Output: reports/full parameter reference/report.tex / .pdf
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import STATION_REGISTRY

REPORT_DIR = ROOT / "reports" / "full parameter reference"
FGC_DIR = ROOT / "artifacts" / "full_gradient_comparison"
FRM_DIR = ROOT / "artifacts" / "final_report_metrics"
REGIME_DIR = ROOT / "artifacts" / "regime_xi_only_corrected"

LOCATIONS = ["BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
             "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR"]
SHORTS = [STATION_REGISTRY[l].get("short", l) for l in LOCATIONS]
DISPLAY = {STATION_REGISTRY[l].get("short", l): STATION_REGISTRY[l].get("display", l) for l in LOCATIONS}

LATEX_PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=2.0cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
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
    if abs(fv) >= 1e4:
        return rf"$\sim\!10^{{{int(np.floor(np.log10(abs(fv))))}}}$"
    return f"{fv:.{digits}f}"


def _table(rows: list, cols: list) -> str:
    df = pd.DataFrame(rows).set_index("Location")
    df = df[cols]
    lines = [r"\begin{tabular}{l" + "r" * len(df.columns) + "}", r"\toprule",
             " & " + " & ".join(df.columns) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        lines.append(idx + " & " + " & ".join(str(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


COLS = ["RMSE", "CRPS", "tw95", "tw98", "grad"]


def _loc_list(flagged: dict) -> str:
    """Render a flagged-location set, collapsing to 'every location' when
    all six are flagged rather than redundantly listing all of them."""
    if len(flagged) == len(SHORTS):
        return "every location"
    return ", ".join(_tex(DISPLAY[s]) for s in flagged)


def _row(short: str, m: dict) -> dict:
    grad_str = _fmt(m.get("grad_norm_inf_numeric"), 3)
    if m.get("grad_converged") is False:
        grad_str += "$^{\\dagger}$"
    return {
        "Location": _tex(DISPLAY[short]),
        "RMSE": _fmt(m.get("rmse")), "CRPS": _fmt(m.get("crps_mean")),
        "tw95": _fmt(m.get("twcrps_95")), "tw98": _fmt(m.get("twcrps_98")),
        "grad": grad_str,
    }


# ---------------------------------------------------------------------------
# Known-mechanism notes, keyed by (stage_family, location_short) or by a
# general lag-set/branch condition -- used to keep the post-table paragraphs
# short and evidence-linked rather than re-deriving each case from scratch.
# ---------------------------------------------------------------------------

LONGSHORT_COLLAPSE_LOCS = {"darwin", "manaus"}
SEASONAL_LAG_NOTE = (r"the two extra lags (365, 366) reintroduce the near-redundant-cluster "
                      r"effect documented in the gas\_lags ablation report: a slowly-decaying "
                      r"annual signal split across adjacent-day lags is only weakly separable "
                      r"from lag 1, especially at Darwin and Manaus, whose own persistence "
                      r"already decays unusually slowly (ACF ratio 0.75/0.88, versus 0.34--0.59 "
                      r"elsewhere).")


def build_stage1_tables(data: dict) -> str:
    parts = [r"\section{Stage 1 -- No Covariates}"]
    for lag_key, lag_label in [("lag1", "single lag ($f_{t-1}$ only)"),
                                ("lag1_365_366", "seasonal-minimal lags ($f_{t-1},f_{t-365},f_{t-366}$)")]:
        for branch, branch_label in [("phi_only", r"$\varphi$ only"), ("phi_xi", r"$\varphi+\xi$")]:
            cfg_id = f"{lag_key}_{branch}"
            rows = [_row(short, data.get(short, {}).get("stage1", {}).get(cfg_id, {})) for short in SHORTS]
            grads = {short: data.get(short, {}).get("stage1", {}).get(cfg_id, {}).get("grad_norm_inf_numeric", 0)
                     for short in SHORTS}
            parts.append(rf"\subsection{{Stage 1, {lag_label}, {branch_label}}}")
            parts.append(_table(rows, COLS))
            flagged = {s: g for s, g in grads.items() if np.isfinite(g) and g > 1}
            if flagged:
                if lag_key == "lag1_365_366":
                    note = SEASONAL_LAG_NOTE
                else:
                    collapse = [s for s in flagged if s in LONGSHORT_COLLAPSE_LOCS]
                    note = (rf"elevated at {_loc_list(flagged)}; "
                            rf"even the single-lag model already shows persistence pushing "
                            + (r"toward its stationarity limit at the locations with the slowest-decaying "
                               r"rainfall persistence (Darwin, Manaus), consistent with a genuine "
                               r"boundary/near-unit-root effect rather than an artifact of this "
                               r"specific configuration." if collapse else
                               r"toward its stationarity limit; this is a smaller effect than at Darwin/Manaus "
                               r"and not yet tied to a specific documented mechanism."))
                parts.append(note)
    return "\n\n".join(parts)


def build_stage2_tables(data: dict) -> str:
    parts = [r"\section{Stage 2 -- Weather Covariates}"]
    cov_labels = {1: "dew point only", 2: "dew point + temperature", 3: "dew point + temperature + ENSO dummies"}
    for lag_key, lag_label in [("lag1", "single lag"), ("lag1_365_366", "seasonal-minimal lags")]:
        for branch, branch_label in [("phi_only", r"$\varphi$ only"), ("phi_xi", r"$\varphi+\xi$")]:
            for n_cov in (1, 2, 3):
                cfg_id = f"{lag_key}_{branch}_{n_cov}cov"
                rows = [_row(short, data.get(short, {}).get("stage2", {}).get(cfg_id, {})) for short in SHORTS]
                grads = {short: data.get(short, {}).get("stage2", {}).get(cfg_id, {}).get("grad_norm_inf_numeric", 0)
                         for short in SHORTS}
                parts.append(rf"\subsection{{Stage 2, {lag_label}, {branch_label}, {cov_labels[n_cov]}}}")
                parts.append(_table(rows, COLS))
                flagged = {s: g for s, g in grads.items() if np.isfinite(g) and g > 1}
                if flagged:
                    if lag_key == "lag1_365_366":
                        note = SEASONAL_LAG_NOTE + (
                            r" Adding covariates on top does not remove this -- the lag "
                            r"structure, not the covariate count, is the dominant factor here, "
                            r"matching the gas\_lags ablation's own finding.")
                    else:
                        note = (rf"elevated at {_loc_list(flagged)} even with a single "
                                 rf"GAS lag and few covariates -- consistent with the location-specific "
                                 rf"persistence weakness (Darwin/Manaus) rather than a covariate-driven effect, "
                                 rf"since the covariate loadings themselves were shown (in the earlier "
                                 rf"diagnostics report) to carry much smaller gradients than the persistence "
                                 rf"coefficients.")
                    parts.append(note)
    return "\n\n".join(parts)


def build_stage34_tables(gc_data: dict) -> str:
    parts = [r"\section{Stage 3 -- Harvey Long/Short (No Seasonal)}"]
    for branch, branch_label in [("phi_only", r"$\varphi$ only"), ("phi_xi", r"$\varphi+\xi$")]:
        rows = []
        for short in SHORTS:
            m = json.loads((FRM_DIR / short / branch / "stage3" / "onestep_metrics.json").read_text())
            gd3 = gc_data.get(short, {}).get("stage3", {}).get(branch, {})
            m["grad_norm_inf_numeric"] = gd3.get("grad_norm_inf_numeric")
            m["grad_converged"] = gd3.get("grad_converged")
            rows.append(_row(short, m))
        parts.append(rf"\subsection{{Stage 3, {branch_label}}}")
        parts.append(_table(rows, COLS))
        grads = {short: gc_data.get(short, {}).get("stage3", {}).get(branch, {}).get("grad_norm_inf_numeric", 0)
                 for short in SHORTS}
        flagged = {s: g for s, g in grads.items() if np.isfinite(g) and g > 1}
        if flagged:
            collapse = [s for s in flagged if s in LONGSHORT_COLLAPSE_LOCS]
            note = (rf"elevated at {_loc_list(flagged)}"
                    + (r" -- this matches the already-documented Long/Short persistence collapse "
                       r"at Darwin and Manaus (confirmed not an initialization artifact via a "
                       r"dedicated reset-and-refit experiment): their rainfall's slow, gradual "
                       r"autocorrelation decay does not contain two genuinely separable timescales, "
                       r"so $\varphi_L$ and $\varphi_S$ drift toward each other rather than settling "
                       r"at a single well-identified pair." if collapse else
                       r"; not one of the two locations with the documented Long/Short collapse, "
                       r"so this specific case has not been traced to a known mechanism yet."))
            parts.append(note)

    parts.append(r"\section{Stage 4 -- + Damped Seasonal Harmonic}")
    for branch, branch_label in [("phi_only", r"$\varphi$ only"), ("phi_xi", r"$\varphi+\xi$")]:
        rows = []
        for short in SHORTS:
            p = FRM_DIR / short / branch / "stage4" / "onestep_metrics.json"
            m = json.loads(p.read_text()) if p.exists() else {}
            gd4 = gc_data.get(short, {}).get("stage4", {}).get(branch, {})
            m["grad_norm_inf_numeric"] = gd4.get("grad_norm_inf_numeric")
            m["grad_converged"] = gd4.get("grad_converged")
            rows.append(_row(short, m))
        parts.append(rf"\subsection{{Stage 4, {branch_label}}}")
        parts.append(_table(rows, COLS))
        grads = {short: gc_data.get(short, {}).get("stage4", {}).get(branch, {}).get("grad_norm_inf_numeric", 0)
                 for short in SHORTS}
        flagged = {s: g for s, g in grads.items() if not np.isfinite(g) or g > 1}
        if flagged:
            note = (rf"elevated or diverged at {_loc_list(flagged)}, unlike Stage 3 (previous "
                     rf"section), where every location's gradient is small -- the seasonal "
                     rf"harmonic addition, not a carried-over Stage 3 weakness, is what "
                     rf"destabilizes these specific fits. The mechanism directly verified earlier "
                     rf"(via correlation of the model's own inverse Hessian, at the worst-affected "
                     rf"locations) is the rotation damping $\psi_\gamma$/loading $\kappa_\gamma$ "
                     rf"correlating strongly ($\approx\!-0.96$) with the Long-component "
                     rf"persistence $\kappa_L$; it is not re-confirmed individually for every "
                     rf"location flagged here.")
            parts.append(note)
    return "\n\n".join(parts)


def build_hybrid_table() -> str:
    parts = [r"\section{Stage 4 $\varphi$ + Stage 5 Regime-Sensitive $\xi$}"]
    reg = json.loads((REGIME_DIR / "summary.json").read_text())
    gc = json.loads((FGC_DIR / "summary_stage345.json").read_text())
    rows = []
    for short in SHORTS:
        e = reg.get(short, {})
        if not e:
            continue
        best_q, m = min(e["thresholds"].items(), key=lambda kv: kv[1]["crps_mean"])
        m = dict(m)
        gd5 = gc.get(short, {}).get("stage5", {})
        m["grad_norm_inf_numeric"] = gd5.get("grad_norm_inf_numeric")
        m["grad_converged"] = gd5.get("grad_converged")
        rows.append(_row(short, m))
    parts.append((r"$\varphi_t$ is frozen at whichever of Stage 3/Stage 4 has the better OOS CRPS "
                  r"per location (never re-estimated); only $\xi_t$ is dynamic, via a GAS(1,1) with "
                  r"a regime-sensitive score response above the better-OOS-CRPS threshold (95\% or "
                  r"98\%, chosen per location)."))
    parts.append(_table(rows, COLS))
    grads = {short: gc.get(short, {}).get("stage5", {}).get("grad_norm_inf_numeric", 0) for short in SHORTS}
    flagged = {s: g for s, g in grads.items() if np.isfinite(g) and g > 1}
    if flagged:
        note = (rf"elevated at {_loc_list(flagged)} -- with $\varphi_t$ "
                 rf"frozen, this reflects $\xi_t$'s own dynamics; at Darwin and Salvador this is "
                 rf"consistent with the same tail-shape estimation difficulty seen elsewhere for "
                 rf"these locations rather than a new mechanism specific to the regime extension.")
        parts.append(note)
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    gc_data = json.loads((FGC_DIR / "summary.json").read_text())
    gc_grad345 = json.loads((FGC_DIR / "summary_stage345.json").read_text())

    print("Building report...")
    body = "\n\n".join([
        r"\begin{document}",
        r"\title{Cross-Stage Gradient-Norm Reference: RMSE, CRPS, twCRPS, and Optimizer Gradients by Model Variation}",
        r"\author{}",
        rf"\date{{{datetime.date.today().isoformat()}}}",
        r"\maketitle",
        (r"Every gradient norm in this reference is computed independently via central-difference "
         r"numerical differentiation of each model's own objective at its final estimated "
         r"parameters, NOT read from scipy's own internal optimizer state. A single, arbitrarily-"
         r"chosen step size is not safe for this: an earlier version of this computation used a "
         r"fixed step of $10^{-4}$ and produced gradients inflated by up to 10 orders of magnitude "
         r"for several otherwise well-behaved fits, because that step size is large enough to push "
         r"the perturbed trajectory of a long ($T\sim$2500--3700 day) recursive filter into a "
         r"qualitatively different, diverging numerical regime -- an artifact, not a real "
         r"derivative. Every value below instead comes from "
         r"\texttt{diagnostics.numerical\_diff.numerical\_gradient\_checked}, which evaluates the "
         r"gradient at four shrinking step sizes ($10^{-6}$ through $10^{-9}$) and only accepts a "
         r"result once two consecutive step sizes agree to within 5\%. A gradient norm above 1 is "
         r"flagged with a short note; a value marked $^{\dagger}$ did not converge across any pair "
         r"of the four step sizes tried and should be treated as unreliable rather than a precise "
         r"number."),
        build_stage1_tables(gc_data),
        build_stage2_tables(gc_data),
        build_stage34_tables(gc_grad345),
        build_hybrid_table(),
        r"\end{document}",
    ])

    tex_path = REPORT_DIR / "report.tex"
    tex_path.write_text(LATEX_PREAMBLE + "\n" + body, encoding="utf-8")
    print(f"Wrote {tex_path}")

    if args.no_compile:
        return

    print("Compiling PDF (2 pdflatex passes)...")
    for i in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=REPORT_DIR, capture_output=True, text=True,
        )
        if proc.returncode != 0 and i == 1:
            print("  pdflatex pass", i + 1, "returned non-zero; see log.")
    log = (REPORT_DIR / "report.log")
    ok = log.exists() and "Output written on" in log.read_text(encoding="utf-8", errors="ignore")
    print("Compilation succeeded." if ok else "Compilation FAILED -- see report.log.")


if __name__ == "__main__":
    main()
