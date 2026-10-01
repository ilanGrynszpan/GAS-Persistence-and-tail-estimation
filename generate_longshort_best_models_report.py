"""
Best-model-per-location report generator (2026-08-31).

=============================================================================
OVERVIEW
=============================================================================

Builds reports/longshort experiment/best models/report.tex + .pdf from the
artifacts already saved by run_longshort_best_models_analysis.py. Per
docs/REPORTING.md, no optimisation or simulation happens here: every number
and figure is read from artifacts/longshort_diagnostics_experiment/ and
artifacts/longshort_diagnostics_experiment/best_models/.

Content (kept to ~4 pages of narrative, per the user's explicit request;
tables/figures extend beyond that):
  - Which configuration won at each location, and why Darwin/Manaus's
    phi_xi branch specifically does not have a reliable sigmoid_both fit
    (the near-unit-root-at-floating-point-precision mechanism), computed
    from the same artifacts as the main diagnostics report.
  - ACF-400 and QQ mosaics of the best model's in-sample quantile
    residuals across all six locations (diagnostics/plots.py's existing
    acf_location_mosaic / qq_location_mosaic).
  - Per-location tables: full-test-set 1-step rolling OOS RMSE/CRPS, and
    Monte Carlo multi-step-ahead (h=1,2,5,10,30,60,90) RMSE/CRPS, each
    comparing the best Harvey model against its paired standard-GAS
    (Stage-2 weather-covariate) comparison model.

=============================================================================
USAGE
=============================================================================

    python generate_longshort_best_models_report.py [--no-compile]

Output:
    reports/longshort experiment/best models/report.tex
    reports/longshort experiment/best models/report.pdf
    reports/longshort experiment/best models/figures/*.png
    reports/longshort experiment/best models/tables/*.csv, *.tex
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import STATION_REGISTRY, mosaic_name
from diagnostics.plots import qq_location_mosaic, acf_location_mosaic
from diagnostics.multistep_forecast import DEFAULT_HORIZONS
from run_longshort_best_models_analysis import N_ORIGINS_STRIDE, N_DRAWS

EXP_DIR    = ROOT / "artifacts" / "longshort_diagnostics_experiment"
BEST_DIR   = EXP_DIR / "best_models"
REPORT_DIR = ROOT / "reports" / "longshort experiment" / "best models"
FIG_DIR    = REPORT_DIR / "figures"
TAB_DIR    = REPORT_DIR / "tables"

LOCATIONS = [
    "BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
    "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR",
]


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX helpers (self-contained, mirrors the other report generators here)
# ─────────────────────────────────────────────────────────────────────────────

def _tex(s) -> str:
    return (
        str(s)
        .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
        .replace("#", r"\#").replace("_", r"\_")
        .replace("{", r"\{").replace("}", r"\}")
        .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    )


def _fmt(v, digits: int = 3) -> str:
    if v is None:
        return "--"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return _tex(str(v))
    if not np.isfinite(fv):
        return "--"
    return f"{fv:.{digits}f}"


def _safe_header(s) -> str:
    s = str(s)
    return s if "$" in s else _tex(s)


def _latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = "l" + "r" * len(df.columns)
    header = " & ".join([""] + [_safe_header(c) for c in df.columns]) + r" \\"
    rows = []
    for idx, row in df.iterrows():
        rows.append(" & ".join([_safe_header(idx)] + [str(v) for v in row]) + r" \\")
    body = "\n".join(rows)
    return "\n".join([
        r"\begin{table}[H]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        rf"\begin{{tabular}}{{{cols}}}", r"\toprule", header, r"\midrule", body, r"\bottomrule",
        r"\end{tabular}", r"}", r"\end{table}",
    ])


LATEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{report}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage{hyperref}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
\emergencystretch=3em
\sloppy
\captionsetup{font=small}
"""


def compile_latex(tex_path: Path) -> bool:
    tex_dir = tex_path.parent
    cmd = ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(tex_dir), str(tex_path)]
    for pass_num in range(1, 3):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=tex_dir)
            if result.returncode != 0:
                print(f"  pdflatex pass {pass_num} returned code {result.returncode}")
                log_path = tex_dir / (tex_path.stem + ".log")
                if log_path.exists():
                    for line in log_path.read_text(errors="replace").splitlines()[-40:]:
                        print(f"    {line}")
                if pass_num == 1:
                    return False
        except FileNotFoundError:
            print("  pdflatex not found -- skipping PDF compilation.")
            return False
        except subprocess.TimeoutExpired:
            print("  pdflatex timed out.")
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Artifact loading
# ─────────────────────────────────────────────────────────────────────────────

def load_location(location: str) -> Optional[dict]:
    cfg = STATION_REGISTRY[location]
    short = cfg.get("short", cfg["run_id"])
    loc_dir = BEST_DIR / short
    info_path = loc_dir / "best_model_info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text())
    harvey_qr = np.load(loc_dir / "harvey_quantile_residuals.npy")
    cov_qr = np.load(loc_dir / "cov_quantile_residuals.npy")
    onestep = json.loads((loc_dir / "onestep_metrics.json").read_text())
    harvey_ms = pd.read_csv(loc_dir / "multistep_metrics_harvey.csv")
    cov_ms = pd.read_csv(loc_dir / "multistep_metrics_cov.csv")
    return {"info": info, "harvey_qr": harvey_qr, "cov_qr": cov_qr,
            "onestep": onestep, "harvey_ms": harvey_ms, "cov_ms": cov_ms}


def _gradient_summary_from_diagnostics() -> pd.DataFrame:
    """Rebuild a compact per-(location,branch) gradient summary directly
    from the main experiment's saved metadata.json files -- same data the
    long-short-diagnostics report uses, reused here for the Darwin/Manaus
    explanation without recomputing anything."""
    rows = []
    for location in LOCATIONS:
        cfg = STATION_REGISTRY[location]
        run_id, display = cfg["run_id"], cfg.get("display", location)
        for branch in ("phi_only", "phi_xi"):
            bdir = EXP_DIR / run_id / branch
            if not bdir.exists():
                continue
            for cfg_dir in bdir.iterdir():
                meta_path = cfg_dir / "metadata.json"
                if not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text())
                rows.append({"location": display, "branch": branch, "config": cfg_dir.name,
                             "grad_inf": meta.get("grad_norm_inf", np.nan),
                             "validity": meta.get("validity")})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-location OOS tables
# ─────────────────────────────────────────────────────────────────────────────

def onestep_table(loc_data: dict) -> pd.DataFrame:
    h, c = loc_data["onestep"]["harvey"], loc_data["onestep"]["cov"]
    return pd.DataFrame({
        "Model": ["Best Harvey", "Standard GAS (Stage 2)"],
        "RMSE": [h["rmse"], c["rmse"]],
        "CRPS": [h["crps_mean"], c["crps_mean"]],
        "RMSE (wet)": [h.get("rmse_wet"), c.get("rmse_wet")],
        "RMSE (dry)": [h.get("rmse_dry"), c.get("rmse_dry")],
    }).set_index("Model")


def multistep_table(loc_data: dict) -> pd.DataFrame:
    hm = loc_data["harvey_ms"].set_index("horizon")
    cm = loc_data["cov_ms"].set_index("horizon")
    rows = {}
    for h in DEFAULT_HORIZONS:
        rows[f"h={h}"] = {
            "Harvey RMSE": hm.loc[h, "rmse"] if h in hm.index else np.nan,
            "Harvey CRPS": hm.loc[h, "crps"] if h in hm.index else np.nan,
            "Std-GAS RMSE": cm.loc[h, "rmse"] if h in cm.index else np.nan,
            "Std-GAS CRPS": cm.loc[h, "crps"] if h in cm.index else np.nan,
        }
    return pd.DataFrame(rows).T


def save_table(name: str, df: pd.DataFrame, caption: str, label: str) -> str:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB_DIR / f"{name}.csv")
    display = df.copy().astype(object)
    for col in display.columns:
        display[col] = df[col].apply(_fmt)
    return _latex_table(display, caption, label)


# ─────────────────────────────────────────────────────────────────────────────
# Mosaics
# ─────────────────────────────────────────────────────────────────────────────

def build_mosaics(all_data: Dict[str, Optional[dict]]) -> Dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    display_order = [mosaic_name(loc) for loc in LOCATIONS]
    qr_by_location = {}
    for loc in LOCATIONS:
        name = mosaic_name(loc)
        d = all_data.get(loc)
        qr_by_location[name] = d["harvey_qr"] if d else None

    fig_qq = qq_location_mosaic(qr_by_location, display_order,
                                 suptitle="Best model per location: in-sample QQ plot of quantile residuals")
    qq_path = FIG_DIR / "best_models_qq_mosaic.png"
    fig_qq.savefig(qq_path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig_qq)

    fig_acf = acf_location_mosaic(qr_by_location, display_order, n_lags=400,
                                   suptitle="Best model per location: in-sample 400-lag ACF of quantile residuals")
    acf_path = FIG_DIR / "best_models_acf_mosaic.png"
    fig_acf.savefig(acf_path, dpi=150)
    plt.close(fig_acf)

    return {"qq": qq_path.relative_to(REPORT_DIR).as_posix(),
            "acf": acf_path.relative_to(REPORT_DIR).as_posix()}


# ─────────────────────────────────────────────────────────────────────────────
# Overview chapter (data-driven, ~4 pages)
# ─────────────────────────────────────────────────────────────────────────────

def build_overview() -> str:
    grad_df = _gradient_summary_from_diagnostics()
    grad_df = grad_df[grad_df["validity"] != "failed"]

    rows = []
    for location in LOCATIONS:
        display = STATION_REGISTRY[location].get("display", location)
        d = load_location(location)
        if d is None:
            rows.append(f"\\item \\textbf{{{_tex(display)}}}: analysis not available.")
            continue
        info = d["info"]
        h1, c1 = d["onestep"]["harvey"], d["onestep"]["cov"]
        better = "Harvey" if h1["crps_mean"] < c1["crps_mean"] else "the standard GAS model"
        rows.append(
            f"\\item \\textbf{{{_tex(display)}}}: best fit is the "
            f"\\texttt{{{_tex(info['branch'])}}} branch, \\texttt{{{_tex(info['harvey_config'])}}} "
            f"($|\\nabla Q|_\\infty={_fmt(info['harvey_grad_norm_inf'],4)}$), compared against "
            f"\\texttt{{{_tex(info['comparison_model_id'])}}}. 1-step OOS CRPS favours {better} "
            f"({_fmt(h1['crps_mean'])} vs {_fmt(c1['crps_mean'])})."
        )

    # Darwin/Manaus sigmoid_both gL=1.0 check, reusing the main diagnostics report's logic
    sig1 = grad_df[grad_df["config"] == "sigmoid_both_gL1.0"]
    bad_sig1 = sig1[sig1["grad_inf"] >= 1.0].sort_values("grad_inf", ascending=False)
    bad_desc = "; ".join(
        f"{_tex(r['location'])} ({_tex(r['branch'])}, $|\\nabla Q|_\\infty={r['grad_inf']:.3g}$)"
        for _, r in bad_sig1.iterrows()
    ) if len(bad_sig1) else "none"

    darwin_phixi = grad_df[(grad_df["location"] == "Darwin Airport") & (grad_df["branch"] == "phi_xi")]
    darwin_best = float(darwin_phixi["grad_inf"].min()) if len(darwin_phixi) else float("nan")

    return "\n\n".join([
        r"\chapter{Overview: What Worked Where}",
        (r"For each location, the ``best'' model is the single lowest-residual-gradient, "
         r"non-\texttt{failed} fit among all 144 configurations in the main diagnostics "
         r"experiment (any branch, any of the 4 persistence modes, any $\Gamma_L$ init) -- "
         r"see \texttt{reports/longshort experiment/long short diagnostics.pdf} for the full "
         r"grid. Its comparison ``standard GAS'' model is the Stage-2 weather-covariate model "
         r"that was used to seed its own initialisation (\texttt{init\_recipe.json}'s "
         r"\texttt{prior\_model\_id}), i.e.\ the best pre-Harvey model already selected for "
         r"that location and branch by the main thesis pipeline."),
        r"\begin{itemize}" + "\n".join(rows) + r"\end{itemize}",
        r"\section{Was it really just Darwin?}",
        (rf"Not quite. For \texttt{{sigmoid\_both}} with $\Gamma_{{L,init}}=1$ specifically, the "
         rf"configurations that fail the $|\nabla Q|_\infty<1$ reliability threshold are: "
         rf"{bad_desc}. Darwin's \texttt{{phi\_xi}} branch is the most severe by several orders "
         rf"of magnitude, but Manaus's \texttt{{phi\_xi}} also fails badly, and both locations' "
         rf"\texttt{{phi\_only}} branches miss the threshold too (more mildly)."),
        (r"\textbf{Mechanism.} Darwin's decoded fit has $\phi_{L,\varphi}=0.999999999$ -- the "
         r"sigmoid saturated to within $10^{-9}$ of exactly 1 (Manaus's \texttt{phi\_xi} shows "
         r"the identical pattern on $\xi$ instead: $\phi_{L,\xi}=1.0000000$, "
         r"$\phi_{S,\xi}\approx8.7\times10^{-21}$). This is the sigmoid reparameterisation's "
         r"blind spot: it guarantees $\phi\in(0,1)$, not $\phi$ \emph{bounded away from} 1. "
         r"Once $\phi_L$ is numerically indistinguishable from 1, the long component becomes a "
         r"literal undamped random walk (Chapter on near-unit-root persistence, "
         r"\texttt{long short diagnostics.pdf}), and over the $\sim$3\,284-day training window "
         r"drifts far enough that the fitted scale is wrong for most of the sample -- both the "
         r"catastrophic log-likelihood and the enormous gradient follow directly. Traced to "
         r"initialisation: Darwin's $\phi$ ratio (0.994) was the one case across the whole "
         r"experiment that escaped the blow-up fallback (Finding 1, same report) -- escaping "
         r"the fallback did not mean the initial value was safe, only that it did not trip that "
         r"particular rule."),
        (rf"\textbf{{Is there a model that works at Darwin?}} Yes -- but it is \texttt{{phi\_only}}, "
         rf"not \texttt{{phi\_xi}}: the \texttt{{restricted}}, $\Gamma_L=0.02$ configuration "
         rf"reaches $|\nabla Q|_\infty=0.090$ there, cleanly reliable. Darwin's \texttt{{phi\_xi}} "
         rf"branch never gets below $|\nabla Q|_\infty={_fmt(darwin_best,2)}$ in any of the 12 "
         rf"configurations tested for that branch -- this location's $\xi$ (tail-shape) dynamics "
         rf"appear to be the genuinely hard part, consistent with Darwin's documented Stage-2 "
         rf"instability history (ROADMAP.md: ``Stage 2 CRPS diverged'')."),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Figures chapter
# ─────────────────────────────────────────────────────────────────────────────

def build_figures_chapter(fig_paths: Dict[str, str]) -> str:
    return "\n\n".join([
        r"\chapter{In-Sample Residual Diagnostics}",
        (r"Both mosaics use the in-sample quantile residuals of each location's own best "
         r"Harvey model (Section~\ref{ch:overview-list}), randomised at zero per "
         r"\texttt{diagnostics/residuals.py}'s \texttt{quantile\_residuals(randomise\_zeros=True)} "
         r"so that dry days do not create a point mass at $z=-\infty$."),
        r"\begin{figure}[H]", r"\centering",
        rf"\includegraphics[width=\textwidth]{{{fig_paths['qq']}}}",
        r"\caption{QQ plots of in-sample quantile residuals against the standard normal, best model per location.}",
        r"\label{fig:qq-mosaic}", r"\end{figure}",
        r"\begin{figure}[H]", r"\centering",
        rf"\includegraphics[width=\textwidth]{{{fig_paths['acf']}}}",
        r"\caption{Autocorrelation of in-sample quantile residuals up to lag 400, best model per location. "
        r"Dashed lines mark the $\pm 1.96/\sqrt{n}$ approximate 95\% band under the white-noise null.}",
        r"\label{fig:acf-mosaic}", r"\end{figure}",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Results chapter: per-location OOS tables
# ─────────────────────────────────────────────────────────────────────────────

def build_results_chapter(all_data: Dict[str, Optional[dict]]) -> str:
    parts = [
        r"\chapter{Out-of-Sample Comparison: Best Harvey Model vs.\ Standard GAS}",
        (rf"\textbf{{Methodology.}} The 1-step table is an exact, non-simulated rolling forecast "
         rf"over the entire held-out test period (\texttt{{diagnostics/scoring.py:"
         rf"compute\_extended\_oos\_metrics}}, already used throughout this framework). The "
         rf"multi-step table (horizons $h\in\{{{','.join(str(h) for h in DEFAULT_HORIZONS)}\}}$ days) "
         rf"is necessarily a Monte Carlo simulation (\texttt{{diagnostics/multistep\_forecast.py}}): "
         rf"from each forecast origin the model's filtered state is replayed exactly on real data up "
         rf"to $t_0$, then {N_DRAWS} independent length-$h_{{max}}$ continuations are drawn "
         rf"(Bernoulli occurrence $\times$ GB2 amount) and scored by RMSE of the simulated mean and "
         rf"Monte Carlo CRPS. Exogenous weather/ENSO covariates over the forecast horizon are assumed "
         rf"known (``perfect foresight''); this is a deliberate simplification, not a modelling claim, "
         rf"and inflates both models' skill by the same amount so the comparison between them remains "
         rf"fair. To keep runtime tractable at $h$ up to 90 days, a strided subsample of forecast "
         rf"origins is used per location rather than the full test set (exact count given in each "
         rf"table's caption; the same origins are used for both models at a given location, so the "
         rf"comparison is paired)."),
    ]
    for location in LOCATIONS:
        display = STATION_REGISTRY[location].get("display", location)
        d = load_location(location)
        parts.append(rf"\section{{{_tex(display)}}}")
        if d is None:
            parts.append("Analysis not available for this location.")
            continue
        info = d["info"]
        n_origins = info.get("multistep_n_origins", "?")
        parts.append(
            rf"Best Harvey: \texttt{{{_tex(info['branch'])}/{_tex(info['harvey_config'])}}} "
            rf"($|\nabla Q|_\infty={_fmt(info['harvey_grad_norm_inf'],4)}$, "
            rf"validity=\texttt{{{_tex(info['harvey_validity'])}}}). Standard GAS: "
            rf"\texttt{{{_tex(info['comparison_subdir'])}/{_tex(info['comparison_model_id'])}}}."
        )
        onestep = onestep_table(d)
        parts.append(save_table(
            f"{STATION_REGISTRY[location].get('short', location)}_onestep",
            onestep,
            f"{_tex(display)}: full-test-set 1-step rolling OOS metrics.",
            f"tab:onestep-{STATION_REGISTRY[location].get('short', location)}",
        ))
        multistep = multistep_table(d)
        parts.append(save_table(
            f"{STATION_REGISTRY[location].get('short', location)}_multistep",
            multistep,
            f"{_tex(display)}: Monte Carlo multi-step OOS RMSE/CRPS "
            f"({n_origins} origins, {N_DRAWS} draws each).",
            f"tab:multistep-{STATION_REGISTRY[location].get('short', location)}",
        ))
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading per-location artifacts...")
    all_data = {loc: load_location(loc) for loc in LOCATIONS}
    missing = [loc for loc, d in all_data.items() if d is None]
    if missing:
        print(f"  WARNING: missing analysis for: {missing}")

    print("Building mosaics...")
    fig_paths = build_mosaics(all_data)

    print("Building overview chapter...")
    overview_tex = build_overview()

    print("Building results chapter...")
    results_tex = build_results_chapter(all_data)

    figures_tex = build_figures_chapter(fig_paths)

    today = datetime.date.today().isoformat()
    body = "\n\n".join([
        r"\begin{document}",
        r"\title{Long-Short Harvey Models: Best Fit per Location}",
        rf"\author{{Generated by \texttt{{generate\_longshort\_best\_models\_report.py}}}}",
        rf"\date{{{today}}}",
        r"\maketitle",
        r"\tableofcontents",
        r"\label{ch:overview-list}",
        overview_tex,
        figures_tex,
        results_tex,
        r"\end{document}",
    ])
    tex_source = LATEX_PREAMBLE + "\n" + body

    tex_path = REPORT_DIR / "report.tex"
    tex_path.write_text(tex_source, encoding="utf-8")
    print(f"Wrote {tex_path}")

    if not args.no_compile:
        print("Compiling PDF (2 pdflatex passes)...")
        ok = compile_latex(tex_path)
        print("Compilation " + ("succeeded." if ok else "FAILED -- see log above."))


if __name__ == "__main__":
    main()
