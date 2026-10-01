"""
Optimization diagnostics report: covariates, lag structure, and high-gradient
cases (2026-09-06).

=============================================================================
OVERVIEW
=============================================================================

Answers three related questions raised in the same conversation:

 1. Are Stage 2's weather covariates unnormalized, causing optimization
    instability? -- NO: data/station_loader.py already z-score standardizes
    every covariate block (fit on training data only). Verified directly
    from the loading code, not assumed.

 2. What DOES cause Stage 2's large optimizer gradients, then? A direct
    per-parameter gradient inspection (Salvador phi+xi's cached fit) showed
    the large gradients concentrated on the GAS persistence coefficients
    (B_phi_*, B_xi_* across all 7 lags of GAS_SEASONAL_LAGS) roughly
    uniformly, not on the covariate loadings -- a "generalized", not
    "concentrated", signature. run_bound_ab_experiment.py tests this by
    crossing {7-lag seasonal, 3-lag short, 3-lag minimal-seasonal, 1-lag}
    GAS lag sets against {3, 1} dew point covariate lags, at all six
    locations.

 3. Why do OTHER models show large gradients without being boundary-pinned?
    Synthesizes findings already established earlier in this project (the
    Stage 4 seasonal kappa_gamma/kappa_L collinearity, the Darwin/Manaus
    Long-Short identification collapse) using report_constraint_table.py's
    existing hess_inv-correlation diagnostic (corr_check), applied here to
    the new Stage 2 case too.

Data sources (all pre-existing artifacts, no new fitting inside this
script):
    artifacts/gas_lags_covariate_ablation/summary.json
    artifacts/run_*/stage2/stage2_dewpoint_short/{estimated_parameters,gradients}.csv
    data/station_loader.py (read directly, not executed)

=============================================================================
USAGE
=============================================================================

    python generate_optimization_diagnostics_report.py [--no-compile]

Output: reports/optimization diagnostics/report.tex / .pdf
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

REPORT_DIR = ROOT / "reports" / "optimization diagnostics"
ABLATION_PATH = ROOT / "artifacts" / "gas_lags_covariate_ablation" / "summary.json"

LOCATIONS_DISPLAY = {
    "bh": "Belo Horizonte", "cruzeiro": "Cruzeiro do Sul", "darwin": "Darwin Airport",
    "garanhuns": "Garanhuns", "manaus": "Manaus", "salvador": "Salvador",
}
CONFIG_LABELS = {
    "A_seasonal_lags_3cov": r"A: 7 lags, 3 cov",
    "B_seasonal_lags_1cov": r"B: 7 lags, 1 cov",
    "C_short_lags_3cov":    r"C: 3 lags, 3 cov",
    "D_short_lags_1cov":    r"D: 3 lags, 1 cov",
    "E_minimal_seasonal_1cov": r"E: [1,365,366], 1 cov",
    "F_single_lag_1cov":   r"F: [1] only, 1 cov",
}
CONFIG_ORDER = list(CONFIG_LABELS.keys())

LATEX_PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
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


def build_intro() -> str:
    return "\n\n".join([
        r"\section{Motivation}",
        (r"A previous report found that adding ERA5 weather covariates at "
         r"Stage 2 reliably destabilizes the optimizer -- gradient norms at "
         r"convergence jump by one to three orders of magnitude relative to "
         r"the covariate-free Stage 1 baseline, and two location/branch "
         r"combinations fail outright. The natural first hypothesis was "
         r"that the covariates (dew point, temperature) enter the model on "
         r"an unnormalized, mismatched scale. This report checks that "
         r"hypothesis directly against the code, finds it does not hold, "
         r"and instead traces the instability to the GAS lag structure "
         r"itself, via a direct ablation. It closes with a broader look at "
         r"why other models in this framework show large gradients even "
         r"when no parameter is pinned at a boundary."),
    ])


def build_normalization_section() -> str:
    return "\n\n".join([
        r"\section{Are the Covariates Actually Unnormalized?}",
        (r"No. \texttt{data/station\_loader.py} (\texttt{\_fit\_scaler} / "
         r"\texttt{\_apply\_scaler}, lines 353--467) already z-score "
         r"standardizes \emph{every} covariate block -- dew point, "
         r"temperature, and the ENSO dummy blocks alike -- fitting the mean "
         r"and standard deviation on training data only and applying that "
         r"same transform to both train and test, before any block is "
         r"handed to a model. This is done uniformly for all seven blocks "
         r"the loader builds, not selectively."),
        (r"Z-score standardization already puts every covariate on a "
         r"comparable, zero-centered, unit-variance scale -- it is at least "
         r"as well-conditioned for gradient-based optimization as min-max "
         r"scaling to $[0,1]$ would be (arguably more so, since min-max "
         r"leaves the mean wherever the data's skew places it, while "
         r"z-scoring centers everything at zero). Re-scaling from z-score "
         r"to min-max was not tested further on this basis: it would not "
         r"address a problem that standardization already solves. The "
         r"actual mechanism is investigated directly below instead."),
    ])


def build_ablation_section(ablation: dict) -> str:
    rows_grad, rows_rmse = [], []
    for short, cfgs in sorted(ablation.items(), key=lambda kv: LOCATIONS_DISPLAY.get(kv[0], kv[0])):
        disp = LOCATIONS_DISPLAY.get(short, short)
        row_g = {"Location": _tex(disp)}
        row_r = {"Location": _tex(disp)}
        for key in CONFIG_ORDER:
            lbl = CONFIG_LABELS[key].split(":")[0]
            c = cfgs.get(key, {})
            row_g[lbl] = _fmt(c.get("grad_norm_inf"), 2)
            row_r[lbl] = _fmt(c.get("rmse"), 3)
        rows_grad.append(row_g)
        rows_rmse.append(row_r)

    def _table(rows, digits_note):
        df = pd.DataFrame(rows).set_index("Location")
        lines = [r"\begin{tabular}{l" + "r" * len(df.columns) + "}", r"\toprule",
                 " & " + " & ".join(df.columns) + r" \\", r"\midrule"]
        for idx, row in df.iterrows():
            lines.append(idx + " & " + " & ".join(str(v) for v in row) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    legend = "; ".join(f"{v}" for v in CONFIG_LABELS.values())

    return "\n\n".join([
        r"\section{What Actually Drives Stage 2's Instability?}",
        (r"Direct inspection of a fitted Stage 2 model's own per-parameter "
         r"gradient vector (Salvador, $\varphi+\xi$ branch, "
         r"\texttt{stage2\_dewpoint\_short}) was the starting point: the "
         r"largest gradients (4--31 in magnitude) sit on the GAS "
         r"persistence coefficients $B_{\varphi,l}$ and $B_{\xi,l}$ at "
         r"\emph{every one of the seven lags} used "
         r"($l\in\{1,2,3,364,365,366,367\}$, GAS\_SEASONAL\_LAGS from "
         r"\texttt{constants.py}) -- roughly uniformly, not concentrated in "
         r"one or two -- while the dew point covariate loadings "
         r"($\gamma_{\xi,\text{dewpoint\_lag}\{1,2,3\}}$) were 10--20$\times$ "
         r"smaller. This pattern held at all six locations: every one uses "
         r"the same 7-lag structure at Stage 2, confirmed directly from "
         r"each model's own saved parameter names, not merely assumed from "
         r"a configuration flag."),
        (r"This points at the GAS lag structure itself, not the covariates, "
         r"as the primary suspect: lags 1,2,3 all capture nearly the same "
         r"``recent persistence'' effect, and lags 364--367 all capture "
         r"nearly the same ``same time last year'' effect, so splitting "
         r"either cluster into separate, freely-estimated coefficients is "
         r"a natural source of weak identification -- the same mechanism "
         r"already established for Stage 4's seasonal harmonic component "
         r"(the $\kappa_\gamma$/$\kappa_L$ collinearity), just via lag "
         r"count here instead of a rotation parameter."),
        (rf"To test this against the covariate-count hypothesis directly, "
         rf"six configurations were fit at every location, crossing the "
         rf"GAS lag set with the number of dew point lags used as "
         rf"covariates: {legend}. All other settings (distribution, "
         rf"occurrence dynamics, scaling, optimizer) are identical to the "
         rf"original Stage 2 fits."),
        r"\subsection{Gradient norm at convergence ($\max_i|\partial\ell/\partial\theta_i|$)}",
        r"\begin{center}\footnotesize", _table(rows_grad, None), r"\end{center}",
        r"\subsection{Out-of-sample RMSE}",
        r"\begin{center}\footnotesize", _table(rows_rmse, None), r"\end{center}",
        (r"Two things stand out. First, RMSE is essentially insensitive to "
         r"which configuration is used -- differences are within 1--3\% at "
         r"every location, so none of this is an accuracy-for-stability "
         r"trade-off. Second, the effect of reducing lags/covariates is "
         r"\emph{not} a clean, universal fix on its own: reducing "
         r"covariates alone (B) makes Darwin and Manaus dramatically "
         r"\emph{worse} (10.7$\to$1472.7 and 110.7$\to$6948.9), and "
         r"reducing GAS lags alone (C) makes Garanhuns dramatically worse "
         r"(14.2$\to$477.7), even as the same changes help Salvador and "
         r"Belo Horizonte enormously. This is the signature of a "
         r"genuinely nonlinear, multi-modal optimization landscape -- the "
         r"same lag/covariate change moves BFGS toward a different local "
         r"region depending on each location's own data, not toward a "
         r"uniformly better- or worse-conditioned version of the same "
         r"problem."),
        (r"What does generalize is the most parsimonious combination (F: "
         r"a single GAS lag, a single dew point lag) -- it gives the "
         r"smallest or near-smallest gradient at five of six locations "
         r"(all but Manaus), and its RMSE is the best of all six "
         r"configurations at four locations (Belo Horizonte, Cruzeiro, "
         r"Manaus, Salvador). At the remaining two (Darwin, Garanhuns) the "
         r"intermediate, pi-occurrence-model-inspired lag set "
         r"$\{1,365,366\}$ (E) has the best RMSE instead, with F a close "
         r"second at Darwin (15.10 vs.\ 14.92) and a mild third at "
         r"Garanhuns (10.01 vs.\ 9.89) -- E is consistently the better or "
         r"second-best choice everywhere, making it the safer general "
         r"default, with F worth considering where its stability edge "
         r"matters more."),
        (r"Manaus is the one location where even the most parsimonious "
         r"configuration leaves a real gradient (29.8, versus $<1$ "
         r"elsewhere). Its largest terms there are on the bare persistence "
         r"coefficients themselves ($B_{\varphi,1}=29.8$, "
         r"$\omega_\xi=27.1$, $B_{\xi,1}=14.5$) -- not covariates, and not "
         r"a multi-lag artifact, since this configuration has only one lag "
         r"to begin with. This is consistent with, not a new instance of, "
         r"the Long/Short persistence identification weakness already "
         r"documented for Manaus and Darwin elsewhere in this project (the "
         r"\texttt{reset\_phi\_ls} experiment confirmed it is not an "
         r"initialization artifact): Manaus's own rainfall persistence "
         r"dynamics appear to be intrinsically hard to pin down, "
         r"independent of how many lags or covariates the model carries."),
        (r"This weakness is specific to Darwin and Manaus, and a direct "
         r"check of each location's own autocorrelation shows why: the "
         r"ratio ACF(3)/ACF(1) of daily log-rainfall -- how much "
         r"persistence has faded by day 3, relative to day 1 -- is 0.75 at "
         r"Darwin and 0.88 at Manaus, distinctly higher than the other "
         r"four locations (0.34--0.59). A sharp early drop (low ratio) is "
         r"the signature a two-timescale Long/Short split needs to "
         r"identify $\varphi_L$ and $\varphi_S$ as genuinely distinct: a "
         r"fast-fading short component dominates the first few lags while "
         r"a slower long component persists as a floor after. A smooth, "
         r"gradual fade from day one (high ratio) looks instead like a "
         r"single decay rate throughout, which a two-component model "
         r"cannot uniquely decompose -- many $(\varphi_L,\varphi_S)$ pairs "
         r"reproduce it about equally well. A stronger version of this "
         r"check (whether the full decay is log-linear in lag, the "
         r"signature of a literal single exponential) was also tried and "
         r"did not cleanly separate Darwin/Manaus from Garanhuns, so only "
         r"the ACF(3)/ACF(1) finding above is asserted as a solid, "
         r"distinguishing property -- not the stronger single-exponential "
         r"claim."),
    ])


def build_broader_scan_section() -> str:
    return "\n\n".join([
        r"\section{Why Do Other Models Show Large Gradients Without a Boundary Pin?}",
        (r"A parameter's gradient can be large at the point BFGS reports as "
         r"its final answer for reasons other than genuinely poor "
         r"optimization or a mistake. Three distinct mechanisms have "
         r"appeared across this project's models, and it is worth "
         r"distinguishing them clearly:"),
        r"\begin{description}",
        (r"\item[Boundary proximity] A soft penalty (e.g.\ the $\varphi_S<\varphi_L$ "
         r"ordering constraint, or a joint $\sum_l|B_l|<0.98$ bound) can "
         r"have its own gradient nearly cancel the likelihood's gradient "
         r"exactly at the constraint, producing a small \emph{total} "
         r"gradient even though the delta-method standard error there is "
         r"invalid. This is a reporting hazard (checked independently of "
         r"gradient size in every parameter table in this project's other "
         r"reports), not the ``large gradient'' case asked about here."),
        (r"\item[Pairwise collinearity between two specific parameters] "
         r"Diagnosed directly via the fitted model's own BFGS inverse-"
         r"Hessian approximation: for a high-gradient parameter $\theta_i$, "
         r"compute its correlation with every other parameter $\theta_j$ "
         r"as $\text{hess\_inv}_{ij}/\sqrt{\text{hess\_inv}_{ii}\,\text{hess\_inv}_{jj}}$ "
         r"(\texttt{report\_constraint\_table.py}'s "
         r"\texttt{diagnose\_high\_gradient\_rows}, already used throughout "
         r"the parameter constraint tables). A correlation above 0.5 in "
         r"magnitude with one specific other parameter is this project's "
         r"working threshold for ``collinearity explains this.'' The "
         r"clearest confirmed case is Stage 4's seasonal damping "
         r"parameter $\kappa_\gamma$ against the Long-component "
         r"persistence $\kappa_L$ (correlation $\approx-0.96$): both push "
         r"the fitted mean in nearly the same direction over the sample, "
         r"so the data cannot separate how much of the trend belongs to "
         r"each."),
        (r"\item[Generalized weak identification (no single correlated "
         r"partner)] When \emph{many} parameters share a large gradient "
         r"roughly uniformly (the ``generalized'' pattern this same "
         r"function flags, versus a gradient concentrated in one or two "
         r"parameters), no single pairwise correlation check is expected "
         r"to explain it cleanly -- the weak direction in parameter space "
         r"is a \emph{combination} of several coefficients, not a single "
         r"pair. Stage 2's seven-lag GAS persistence coefficients (this "
         r"report, above) and the Long/Short collapse at Darwin and Manaus "
         r"(where $\varphi_L$ and $\varphi_S$ drift toward each other "
         r"rather than any one coefficient hitting a wall) are both this "
         r"kind of case: the fix is reducing how many near-redundant "
         r"directions the model asks the data to separate, not finding one "
         r"culprit to constrain."),
        r"\end{description}",
        (r"Practically, when a new model shows a large, non-boundary "
         r"gradient, the useful first checks are: (1) is it concentrated in "
         r"one or two parameters, or spread across many (the concentrated/"
         r"generalized split above)? (2) if concentrated, does "
         r"\texttt{corr\_check\_factory} find a single dominant correlated "
         r"partner? (3) if generalized, does the parameter set contain "
         r"clusters of near-redundant lags or components (adjacent-day GAS "
         r"lags, adjacent seasonal harmonics, two persistence components "
         r"on data that does not actually separate them)? All three "
         r"documented cases in this project so far -- Stage 2's lag "
         r"structure, Stage 4's seasonal damping, and the Darwin/Manaus "
         r"Long/Short collapse -- resolve into one of these categories "
         r"once checked directly, rather than requiring a new explanation "
         r"each time."),
    ])


def build_recommendations() -> str:
    return "\n\n".join([
        r"\section{Recommendations}",
        r"\begin{enumerate}",
        (r"\item Do not pursue min-max (or any other) re-scaling of the "
         r"weather covariates as a fix for Stage 2 -- the covariates are "
         r"already standardized, and the evidence points elsewhere."),
        (r"\item Consider the minimal lag set F ($\{1\}$, one covariate "
         r"lag) or the intermediate set E ($\{1,365,366\}$) as the default "
         r"Stage 2 (and, by extension, Stage 1) GAS lag configuration in "
         r"place of the current $\{1,2,3\}$/$\{1,2,3,364,365,366,367\}$ "
         r"sets -- both give equal-or-better RMSE with dramatically better "
         r"optimizer stability at nearly every location, at no cost found "
         r"in this experiment. This is a configuration change (per this "
         r"project's Configuration Philosophy), not a new model."),
        (r"\item Treat Manaus's residual instability as a location-specific "
         r"open problem tied to its already-documented Long/Short "
         r"identification weakness, not as evidence against the lag-count "
         r"reduction itself -- every other location resolves cleanly under "
         r"configuration F."),
        (r"\item When a future model shows a large gradient with no "
         r"boundary involved, apply the concentrated-vs-generalized check "
         r"first, then \texttt{corr\_check\_factory} if concentrated, "
         r"before assuming a new, unrelated bug."),
        r"\end{enumerate}",
    ])


def build_conclusion() -> str:
    return "\n\n".join([
        r"\section{Conclusion}",
        (r"The covariates were never the numerical problem -- they were "
         r"already standardized. The real driver of Stage 2's instability "
         r"is the seven-lag GAS persistence structure inherited from the "
         r"seasonal lag set, which asks the optimizer to separate several "
         r"near-redundant directions in parameter space at once; a direct "
         r"ablation confirms this is real but location-dependent, and that "
         r"the most parsimonious lag/covariate combination tested "
         r"resolves it almost everywhere without costing any accuracy. "
         r"The one location that does not fully resolve (Manaus) traces "
         r"to the same underlying identification weakness already "
         r"documented for its Long/Short dynamics elsewhere in this "
         r"project, not to a new mechanism. More broadly, every "
         r"high-gradient, non-boundary case examined in this project to "
         r"date falls into one of two categories -- a specific pairwise "
         r"collinearity, diagnosable from the model's own inverse Hessian, "
         r"or a generalized weak-identification effect from near-redundant "
         r"parameter clusters -- giving a repeatable procedure for "
         r"diagnosing the next one."),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading gas_lags/covariate ablation data...")
    ablation = json.loads(ABLATION_PATH.read_text())
    print(f"  Loaded {len(ablation)} locations.")

    print("Building report...")
    body = "\n\n".join([
        r"\begin{document}",
        r"\title{Optimization Diagnostics: Covariates, Lag Structure, and High-Gradient Cases}",
        r"\author{}",
        rf"\date{{{datetime.date.today().isoformat()}}}",
        r"\maketitle",
        build_intro(),
        build_normalization_section(),
        build_ablation_section(ablation),
        build_broader_scan_section(),
        build_recommendations(),
        build_conclusion(),
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
