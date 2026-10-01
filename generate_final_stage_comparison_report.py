"""
Final multi-stage OOS comparison report (2026-09-06).

=============================================================================
OVERVIEW
=============================================================================

Assembles every table requested for the "final report" from already-cached
artifacts (no new fitting/simulation here):

  A. Per-location, 1-step-ahead OOS RMSE/CRPS/twCRPS@95/98, all stages --
     Table A1 (phi-only, stages 1-4) and Table A2 (phi+xi, stages 1-5).
     Source: artifacts/final_report_metrics/<short>/<branch>/<stage>/onestep_metrics.json
  B. One table per stage (1-5), multi-step (h=1,2,5,10,30,90) RMSE/CRPS/
     twCRPS@95/98, phi-only vs. phi+xi side by side.
     Source: .../multistep_metrics.csv
  C. Long-short with vs. without seasonality (Stage 3 vs Stage 4, damped
     model): Table C1 (1-step rolling) and Table C2 (multi-step).
     Source: artifacts/harmonic_seasonal_comparison_m5_final_damp0.999/
  D. Stage 1/2 with vs. without seasonal lags (one representative model
     per location -- best Stage 2 candidate under the plain/"original"
     parameterisation).
     Source: artifacts/original_seasonality_comparison/
  E. Parameter/estimate/SE/CI/constraint tables -- NOT duplicated here;
     see the separate reports/parameter constraint tables/report.pdf
     (built earlier, per the same "stage 1-5" specification).
  F. Master table: every model variation (location x branch x stage x
     seasonality) with its own convergence gradient.

Per docs/REPORTING.md: reads only already-saved artifacts, computes and
fits nothing.

=============================================================================
USAGE
=============================================================================

    python generate_final_stage_comparison_report.py [--no-compile]
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
import run_harmonic_seasonal_experiment as R

ARTIFACTS_DIR = ROOT / "artifacts"
FRM_DIR = ARTIFACTS_DIR / "final_report_metrics"
HSC_DIR = ARTIFACTS_DIR / "harmonic_seasonal_comparison_m5_final_damp0.999"
OSC_DIR = ARTIFACTS_DIR / "original_seasonality_comparison"
REPORT_DIR = ROOT / "reports" / "final stage comparison"

LOCATIONS = ["BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
             "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR"]
BRANCHES = ["phi_only", "phi_xi"]
STAGES_BY_BRANCH = {"phi_only": ["stage1", "stage2", "stage3", "stage4"],
                     "phi_xi": ["stage1", "stage2", "stage3", "stage4", "stage5"]}
STAGE_LABEL = {"stage1": "1", "stage2": "2", "stage3": "3", "stage4": "4", "stage5": "5"}
HORIZONS = (1, 2, 5, 10, 30, 90)

LATEX_PREAMBLE = r"""\documentclass[9pt,a4paper]{report}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=1.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{pdflscape}
\usepackage{caption}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}
\emergencystretch=3em
\sloppy
\captionsetup{font=small}
"""


def _tex(s) -> str:
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _fmt(v, digits=3):
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(fv):
        return "diverged"
    if abs(fv) >= 1e5:
        return rf"$\sim\!10^{{{int(np.floor(np.log10(abs(fv))))}}}$"
    return f"{fv:.{digits}f}"


def _latex_longtable(columns, rows, caption, label, col_spec=None) -> str:
    col_spec = col_spec or ("l" + "r" * (len(columns) - 1))
    header = " & ".join(columns) + r" \\"
    body = "\n".join(" & ".join(str(c) for c in row) + r" \\" for row in rows)
    return "\n".join([
        r"\begin{longtable}{" + col_spec + "}",
        rf"\caption{{{caption}}}\label{{{label}}} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\bottomrule", r"\endfoot",
        body,
        r"\end{longtable}",
    ])


def load_onestep(short: str, branch: str, stage: str):
    p = FRM_DIR / short / branch / stage / "onestep_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_multistep(short: str, branch: str, stage: str):
    p = FRM_DIR / short / branch / stage / "multistep_metrics.csv"
    return pd.read_csv(p).set_index("horizon") if p.exists() else None


def load_grad(short: str, branch: str, stage: str):
    p = FRM_DIR / short / branch / stage / "model_info.json"
    if not p.exists():
        return None
    info = json.loads(p.read_text())
    src = info.get("source_dir")
    if not src:
        return None
    meta_p = Path(src) / "metadata.json"
    if not meta_p.exists():
        return None
    return json.loads(meta_p.read_text()).get("grad_norm_inf")


# ─────────────────────────────────────────────────────────────────────────────
# Table A: per-location, 1-step, all stages
# ─────────────────────────────────────────────────────────────────────────────

def build_table_a(branch: str) -> str:
    stages = STAGES_BY_BRANCH[branch]
    rows = []
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        short, display = cfg.get("short", location), cfg.get("display", location)
        for stage in stages:
            m = load_onestep(short, branch, stage)
            if m is None:
                rows.append([_tex(display), STAGE_LABEL[stage], "--", "--", "--", "--"])
                continue
            rows.append([_tex(display), STAGE_LABEL[stage], _fmt(m.get("rmse")), _fmt(m.get("crps_mean")),
                         _fmt(m.get("twcrps_95")), _fmt(m.get("twcrps_98"))])
    branch_label = r"$\varphi$ only" if branch == "phi_only" else r"$\varphi$ and $\xi$"
    n_stages = "1--4" if branch == "phi_only" else "1--5"
    caption = (rf"Per-location, 1-step-ahead out-of-sample RMSE, CRPS, and threshold-weighted CRPS "
               rf"(95\%, 98\%) for the {branch_label} branch, Stages {n_stages}.")
    return _latex_longtable(["Location", "Stage", "RMSE", "CRPS", "twCRPS@95\\%", "twCRPS@98\\%"],
                             rows, caption, f"tab:A-{branch}")


# ─────────────────────────────────────────────────────────────────────────────
# Table B: one per stage, multi-step, phi-only vs phi+xi
# ─────────────────────────────────────────────────────────────────────────────

def build_table_b(stage: str) -> str:
    rows = []
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        short, display = cfg.get("short", location), cfg.get("display", location)
        ms1 = load_multistep(short, "phi_only", stage)
        ms2 = load_multistep(short, "phi_xi", stage) if stage in STAGES_BY_BRANCH["phi_xi"] else None
        for h in HORIZONS:
            def get(ms, col):
                if ms is None or h not in ms.index:
                    return "--"
                return _fmt(ms.loc[h, col]) if col in ms.columns else "--"
            rows.append([_tex(display) if h == HORIZONS[0] else "", h,
                         get(ms1, "rmse"), get(ms2, "rmse"),
                         get(ms1, "crps"), get(ms2, "crps"),
                         get(ms1, "twcrps_95"), get(ms2, "twcrps_95"),
                         get(ms1, "twcrps_98"), get(ms2, "twcrps_98")])
    caption = (rf"Stage {STAGE_LABEL[stage]}: multi-step Monte Carlo out-of-sample RMSE, CRPS, and "
               rf"twCRPS@95/98\%, one vs.\ two time-varying parameters ($\varphi$ only vs.\ $\varphi$ and $\xi$), "
               rf"by forecast horizon.")
    cols = ["Location", "$h$", "RMSE ($\\varphi$)", "RMSE ($\\varphi,\\xi$)",
            "CRPS ($\\varphi$)", "CRPS ($\\varphi,\\xi$)",
            "tw95 ($\\varphi$)", "tw95 ($\\varphi,\\xi$)", "tw98 ($\\varphi$)", "tw98 ($\\varphi,\\xi$)"]
    return _latex_longtable(cols, rows, caption, f"tab:B-{stage}", col_spec="ll" + "r" * 8)


# ─────────────────────────────────────────────────────────────────────────────
# Table C: seasonality on/off (Stage 3 vs Stage 4, damped model)
# ─────────────────────────────────────────────────────────────────────────────

def build_table_c1() -> str:
    rows = []
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        short, display = cfg.get("short", location), cfg.get("display", location)
        for branch in BRANCHES:
            p = HSC_DIR / short / branch / "info.json"
            if not p.exists():
                continue
            info = json.loads(p.read_text())
            std_p = HSC_DIR / short / branch / "standard" / "onestep_metrics.json"
            seas_p = HSC_DIR / short / branch / "seasonal" / "onestep_metrics.json"
            std_m = json.loads(std_p.read_text()) if std_p.exists() else {}
            seas_m = json.loads(seas_p.read_text()) if seas_p.exists() else {}
            branch_label = r"$\varphi$" if branch == "phi_only" else r"$\varphi,\xi$"
            rows.append([_tex(display), branch_label,
                         _fmt(info["standard"]["rmse_1step"]), _fmt(info["seasonal"]["rmse_1step"]),
                         _fmt(info["standard"]["crps_1step"]), _fmt(info["seasonal"]["crps_1step"]),
                         _fmt(std_m.get("twcrps_95")), _fmt(seas_m.get("twcrps_95")),
                         _fmt(std_m.get("twcrps_98")), _fmt(seas_m.get("twcrps_98"))])
    caption = (r"Long-short model WITHOUT (Stage 3) vs.\ WITH (Stage 4, damped) trigonometric seasonality: "
               r"1-step rolling out-of-sample RMSE, CRPS, and twCRPS@95/98\%.")
    cols = ["Location", "Branch", "RMSE std", "RMSE seas", "CRPS std", "CRPS seas",
            "tw95 std", "tw95 seas", "tw98 std", "tw98 seas"]
    return _latex_longtable(cols, rows, caption, "tab:C1", col_spec="ll" + "r" * 8)


def build_table_c2() -> str:
    rows = []
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        short, display = cfg.get("short", location), cfg.get("display", location)
        for branch in BRANCHES:
            std_ms_p = HSC_DIR / short / branch / "standard" / "multistep_metrics.csv"
            seas_ms_p = HSC_DIR / short / branch / "seasonal" / "multistep_metrics.csv"
            if not std_ms_p.exists():
                continue
            std_ms = pd.read_csv(std_ms_p).set_index("horizon")
            seas_ms = pd.read_csv(seas_ms_p).set_index("horizon") if seas_ms_p.exists() else None
            branch_label = r"$\varphi$" if branch == "phi_only" else r"$\varphi,\xi$"
            for h in HORIZONS:
                if h not in std_ms.index:
                    continue

                def get(df, col):
                    if df is None or h not in df.index or col not in df.columns:
                        return "--"
                    return _fmt(df.loc[h, col])
                rows.append([_tex(display) if h == HORIZONS[0] else "", branch_label if h == HORIZONS[0] else "", h,
                             get(std_ms, "rmse"), get(seas_ms, "rmse"),
                             get(std_ms, "crps"), get(seas_ms, "crps")])
    caption = (r"Long-short model WITHOUT vs.\ WITH trigonometric seasonality: multi-step Monte Carlo "
               r"out-of-sample RMSE and CRPS by forecast horizon.")
    cols = ["Location", "Branch", "$h$", "RMSE std", "RMSE seas", "CRPS std", "CRPS seas"]
    return _latex_longtable(cols, rows, caption, "tab:C2", col_spec="lll" + "r" * 4)


# ─────────────────────────────────────────────────────────────────────────────
# Table D: Stage 1/2 with vs. without seasonal lags
# ─────────────────────────────────────────────────────────────────────────────

def build_table_d() -> str:
    rows = []
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        short, display = cfg.get("short", location), cfg.get("display", location)
        p = OSC_DIR / short / "representative_model.json"
        if not p.exists():
            continue
        info = json.loads(p.read_text())
        std_p = OSC_DIR / short / "short" / "onestep_metrics.json"
        seas_p = OSC_DIR / short / "seasonal" / "onestep_metrics.json"
        std_m = json.loads(std_p.read_text()) if std_p.exists() else {}
        seas_m = json.loads(seas_p.read_text()) if seas_p.exists() else {}
        rows.append([_tex(display), _tex(info.get("stage", "")), _tex(info.get("model_id", "")),
                     _fmt(info["short"]["rmse_1step"]), _fmt(info["seasonal"]["rmse_1step"]),
                     _fmt(info["short"]["crps_1step"]), _fmt(info["seasonal"]["crps_1step"]),
                     _fmt(std_m.get("twcrps_95")), _fmt(seas_m.get("twcrps_95"))])
    caption = (r"Stage 1/2 weather-covariate lag choice: short lags only vs.\ short + annual/seasonal lags. "
               r"``Stage''/``Model'' identify the representative winning candidate compared at each location "
               r"(the plain, unrestricted parameterisation used throughout this report).")
    cols = ["Location", "Stage", "Model", "RMSE short", "RMSE seas.\\ lag", "CRPS short", "CRPS seas.\\ lag",
            "tw95 short", "tw95 seas.\\ lag"]
    return _latex_longtable(cols, rows, caption, "tab:D", col_spec="lll" + "r" * 6)


# ─────────────────────────────────────────────────────────────────────────────
# Table F: master model-variation table with gradients
# ─────────────────────────────────────────────────────────────────────────────

def build_table_f() -> str:
    rows = []
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        short, display = cfg.get("short", location), cfg.get("display", location)
        for branch in BRANCHES:
            n_tvp = 1 if branch == "phi_only" else 2
            for stage in STAGES_BY_BRANCH[branch]:
                grad = load_grad(short, branch, stage)
                seasonality = "Yes" if stage == "stage4" else ("N/A" if stage == "stage5" else "No")
                rows.append([_tex(display), n_tvp, STAGE_LABEL[stage], seasonality,
                             _fmt(grad, 4) if grad is not None else "--"])
    caption = "Master table: every model variation (location, branch, stage, seasonality) and its convergence gradient ($|\\nabla Q|_\\infty$)."
    cols = ["Location", "TVP count", "Stage", "Seasonality", "$|\\nabla Q|_\\infty$"]
    return _latex_longtable(cols, rows, caption, "tab:F", col_spec="l" + "r" * 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building Table A (per-location, 1-step, all stages)...")
    table_a1 = build_table_a("phi_only")
    table_a2 = build_table_a("phi_xi")

    print("Building Table B (per-stage multi-step, 1 vs 2 TVP)...")
    tables_b = [build_table_b(s) for s in ("stage1", "stage2", "stage3", "stage4", "stage5")]

    print("Building Table C (seasonality on/off)...")
    table_c1 = build_table_c1()
    table_c2 = build_table_c2()

    print("Building Table D (Stage 1/2 seasonal lags)...")
    table_d = build_table_d()

    print("Building Table F (master model-variation table)...")
    table_f = build_table_f()

    intro = "\n\n".join([
        r"\chapter{Introduction}",
        (r"All tables below are computed from already-saved out-of-sample artifacts "
         r"(no re-fitting). RMSE and CRPS are 1-step-ahead unless noted; twCRPS is the "
         r"threshold-weighted CRPS (Gneiting \& Ranjan, 2011) at the 95th/98th percentile "
         r"of observed wet-day rainfall. Stage 5 (static GB2 + threshold-weighted tail "
         r"$\xi$) applies only to the two-time-varying-parameter branch, since it is by "
         r"construction an extension of $\xi$'s own dynamics."),
        (r"Parameter estimate / standard error / 95\% CI / constraint tables for every "
         r"Stage 1-5 model are NOT duplicated here -- see the separate report "
         r"\texttt{reports/parameter constraint tables/report.pdf}, built to the same "
         r"specification."),
    ])

    chap_a = "\n\n".join([r"\chapter{Table A --- Per-Location OOS Comparison, All Stages (1-Step)}",
                           r"\section{$\varphi$ Only (Stages 1--4)}", table_a1,
                           r"\section{$\varphi$ and $\xi$ (Stages 1--5)}", table_a2])
    chap_b = "\n\n".join([r"\chapter{Table B --- Multi-Step Comparison, Per Stage}"] +
                          [f"\\section{{Stage {STAGE_LABEL[s]}}}\n\n{t}" for s, t in
                           zip(("stage1", "stage2", "stage3", "stage4", "stage5"), tables_b)])
    chap_c = "\n\n".join([r"\chapter{Table C --- Long-Short With vs.\ Without Seasonality}",
                           r"\section{1-Step Rolling}", table_c1,
                           r"\section{Multi-Step}", table_c2])
    chap_d = "\n\n".join([r"\chapter{Table D --- Stage 1/2 With vs.\ Without Seasonal Lags}", table_d])
    chap_f = "\n\n".join([r"\chapter{Table F --- Master Model-Variation Table}", table_f])

    body = "\n\n".join([r"\begin{document}",
                         r"\title{Final Multi-Stage Out-of-Sample Comparison}",
                         rf"\date{{{datetime.date.today().isoformat()}}}",
                         r"\maketitle", r"\tableofcontents",
                         intro, chap_a, chap_b, chap_c, chap_d, chap_f,
                         r"\end{document}"])
    tex_source = LATEX_PREAMBLE + "\n" + body
    tex_path = REPORT_DIR / "report.tex"
    tex_path.write_text(tex_source, encoding="utf-8")
    print(f"Wrote {tex_path}")

    if not args.no_compile:
        print("Compiling PDF (2 pdflatex passes)...")
        cmd = ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(REPORT_DIR), str(tex_path)]
        ok = True
        for i in range(2):
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=REPORT_DIR)
            if result.returncode != 0 and i == 0:
                ok = False
        print("Compilation " + ("succeeded." if ok else "FAILED -- see report.log."))


if __name__ == "__main__":
    main()
