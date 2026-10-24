"""
2026-10-01: regenerate four existing reports from the phi-linked-pi staged
rerun (artifacts/rerun_pi_01102026), each into a "<folder>_011026" copy:

    reports/mosaic11092026_011026                  generate_mosaic_report.py
    reports/std_errors_011026                      generate_std_errors_report.py
    reports/best_models_011026                     generate_best_models_report.py
    reports/marginal_effects_18092026_further_011026
                                                   run_marginal_effects_exceedance_uncertainty.py
                                                   (+ methods.tex/methods.pdf)

EXECUTION FLOW
    The generators are reused unchanged in their logic. Before calling each
    one, its inputs are pointed at the new run:
      * selection metrics  -> artifacts/rerun_pi_01102026/metrics.json
                              (built by generate_regeneratedwithpi_01102026_report)
      * artifact folders   -> artifacts/rerun_pi_01102026/<short>/<stage>/<cfg>
      * model rebuilding   -> run_rerun_pi_01102026.build_model (phi-linked pi,
                              mixed-density scalings, sigmoid phi_L)
      * output folder      -> the "_011026" copy
    The marginal-effects driver's formulas carry the occurrence channel
    d pi/d phi = lambda1 pi (1 - pi) (zero for non-phi-linked models).

No fitting happens here except the marginal-effects step's numerical
Hessian at the selected model's optimum and its 1000 re-filtered parameter
draws (Numba filter), exactly as in the original driver.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_ml_comparison_report as G
import collect_stage_comparison_metrics as C
import generate_regeneratedwithpi_01102026_report as P
import run_rerun_pi_01102026 as R
import run_rerun_10092026 as D

SUFFIX = "_011026"
REPORTS = ROOT / "reports"


def activate_new_run() -> dict:
    met = P.build_metrics()
    G.MET = met
    G._resolve_cfg_dir = P.new_resolve
    C.build_model = P.new_build
    return met


def mosaic():
    import generate_mosaic_report as M
    M.REP = REPORTS / ("mosaic11092026" + SUFFIX)
    M.main()


def stage5_mosaic():
    """Stage 5 (phi+xi long-short on each location's best Stage 3/4/4S config):
    rows = locations, columns = [QQ, ACF] of its in-sample quantile residuals."""
    import generate_mosaic_report as M
    qr = {}
    for s in G.SHORTS:
        p = R.OUT / s / "stage5" / "best" / "insample_qr.npy"
        qr[s] = np.load(p) if p.exists() else np.array([])
    M.build_qq_acf_mosaic(qr, "qq_acf_stage5", rep=REPORTS / ("mosaic11092026" + SUFFIX))


def occurrence_fit():
    """Monthly mean pi_{t|t-1} vs wet-day share mosaic + BSS table for each
    location's best model overall (requested 2026-10-02, suffix _021026)."""
    import generate_occurrence_fit_report as O
    O.REP = REPORTS / "occurrence_fit_021026"
    O.build()


def best_models():
    import generate_best_models_report as B
    B.REP = REPORTS / ("best_models" + SUFFIX)
    B.build()


def std_errors():
    import generate_std_errors_report as S
    S.REP = REPORTS / ("std_errors" + SUFFIX)
    S.build()


# ---------------------------------------------------------------------------
# marginal effects
# ---------------------------------------------------------------------------

def marginal_effects(met: dict):
    import run_marginal_effects_exceedance_uncertainty as ME
    from marginal_effects.model_loader import LoadedModel
    from marginal_effects.model_selection import LOCATION_FULL

    def load_new(sel):
        cdir = P.new_resolve(sel.short, (sel.stage, sel.cfg, sel.branch))
        res = json.loads((cdir / "result.json").read_text())
        data = D._load_data(LOCATION_FULL[sel.short])
        model, _fk, ok = R.build_model(sel.stage, sel.branch, res["scaling"], sel.cfg, data)
        theta = np.load(cdir / "theta.npy")
        return LoadedModel(sel=sel, model=model, theta=theta, data=data, ms_kwargs=ok, cfg_dir=cdir)

    ME._load_metrics = lambda: met
    ME.load = load_new
    ME.OUT_REPORT = REPORTS / ("marginal_effects_18092026_further" + SUFFIX)
    ME.OUT_ARTIFACTS = ROOT / "artifacts" / "marginal_effects" / ("exceedance_uncertainty" + SUFFIX)
    ME.main()
    write_methods(ME.OUT_REPORT, ME.OUT_ARTIFACTS, ME.N_DRAWS, ME.DELTAS)


def write_methods(rep: Path, art: Path, n_draws: int, deltas) -> None:
    info = json.loads((art / "model_info.json").read_text())
    pars = info["parameters"]
    lam = {p["name"]: p for p in pars if p["name"] in ("lambda0", "lambda1")}
    rows = "\n".join(
        f"{p['name'].replace('_', chr(92) + '_')} & {p['estimate']:.4f} & "
        f"{(p['se'] if p['se'] is not None else float('nan')):.4f} \\\\" for p in pars)
    lam_txt = (f"$\\hat\\lambda_0={lam['lambda0']['estimate']:.3f}$, "
               f"$\\hat\\lambda_1={lam['lambda1']['estimate']:.3f}$") if len(lam) == 2 else "n/a"
    deltas_txt = ", ".join(str(int(d)) for d in deltas)
    if info["family"] == "harvey":
        STATE_EQ = (r"\[\varphi_{t|t-1}=\omega+L_t+S_t,\quad S_{t+1}=\phi_S S_t+\kappa_S s_t+"
                    r"\Gamma_S^\top x_t,\quad L_{t+1}=\phi_L L_t+\kappa_L s_t+\Gamma_L^\top x^{L}_t,\]"
                    r"\[\pi_{t|t-1}=\operatorname{logistic}(\lambda_0+\lambda_1\varphi_{t|t-1}),\]"
                    r"(long--short dynamics; the dew point enters the short component, so $c_v$ is its "
                    r"element of $\Gamma_S$)")
    else:
        STATE_EQ = (r"\[\varphi_{t+1|t}=\omega+\textstyle\sum_l A_l s_{t-l+1}+\sum_l B_l"
                    r"\varphi_{t-l+1|t-l}+c_v x_t+\dots,\qquad"
                    r"\pi_{t|t-1}=\operatorname{logistic}(\lambda_0+\lambda_1\varphi_{t|t-1}).\]")
    tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable}
\usepackage{parskip}\emergencystretch=3em\sloppy
\begin{document}
\title{Marginal effects of a dew-point shock: how they are computed}\author{}\date{1 October 2026}\maketitle

\section{Model}
The selected model (Section~4) is a zero-adjusted GB2 score-driven model with one
time-varying parameter, the log-scale $\varphi_{t|t-1}$, whose recursion contains the
lagged dew-point temperature $x_{t-1}$ with coefficient $c_v$:
""" + STATE_EQ + r"""
The occurrence probability is linked to $\varphi$ (the occurrence dynamics adopted in
\texttt{reports/newpi}), so a covariate shock moves both the magnitude and the
probability of rain. With $G_t\sim\mathrm{GB2}(\varphi_t,\xi,\gamma,\zeta)$ and $q_{95}$ the
95\% quantile of wet-day training rainfall,
\[
M_t=\mathbb E[Y_t]=\pi_t\,\mathbb E[G_t],\qquad
P_t=\Pr(Y_t>q_{95})=\pi_t\,S_t(q_{95}),\quad S_t=1-F_{G_t}.
\]

\section{One-step effect of a shock at $t-1$}
A shock $\Delta$ (raw units, $^\circ$C) to $x_{t-1}$ changes only the state at $t$, by exactly
$\delta\varphi_t=c_v\Delta$ (the covariate enters the recursion additively; nothing is
propagated beyond $t$). With $c_v$ converted from the standardised scale used in estimation
($c_v=c_v^{\text{std}}/\mathrm{sd}(x)$) and the logistic link,
\[
\frac{\partial\pi_t}{\partial\varphi_t}=\lambda_1\pi_t(1-\pi_t),\qquad
\frac{\partial\mathbb E[G_t]}{\partial\varphi_t}=\mathbb E[G_t]\ \ (\varphi=\log\sigma),
\]
so the instantaneous effects per unit of dew point are
\begin{align*}
\frac{\partial M_t}{\partial x_{t-1}}&=c_v\Big[\lambda_1\pi_t(1-\pi_t)\,\mathbb E[G_t]+\pi_t\,\mathbb E[G_t]\Big],\\
\frac{\partial P_t}{\partial x_{t-1}}&=c_v\Big[\lambda_1\pi_t(1-\pi_t)\,S_t(q_{95})+\pi_t\,\frac{\partial S_t(q_{95})}{\partial\varphi_t}\Big].
\end{align*}
$\mathbb E[G_t]$ is the analytical GB2 mean and $\partial S/\partial\varphi$ the analytical
exceedance kernel (\texttt{docs/math\_proofs.tex}, Sec.~12). The absolute-effect charts
(\texttt{dpt\_shock\_delta*}) show these derivatives times $\Delta\in\{""" + deltas_txt + r"""\}$.

\paragraph{Percentage changes and levels} are computed exactly, not linearised: with
$\varphi^{a}_t=\varphi_t+c_v\Delta$ and $\pi^{a}_t=\operatorname{logistic}(\operatorname{logit}\pi_t+\lambda_1c_v\Delta)$,
\[
\frac{\pi^{a}_t\mathbb E[G_t\mid\varphi^{a}_t]-\pi_t\mathbb E[G_t]}{\pi_t\mathbb E[G_t]},\qquad
\frac{\pi^{a}_t S_t(q_{95}\mid\varphi^{a}_t)-\pi_t S_t(q_{95})}{\pi_t S_t(q_{95})}.
\]
The ``with comparison'' charts show the levels $M_t,P_t$ before and after the shock; the
``changes in time'' charts show the effects over the in-sample period with El Ni\~no / La
Ni\~na periods marked. When $\lambda_1=0$ (occurrence not linked to $\varphi$) every formula
reduces to the earlier one, in which $\pi_t$ cancels from the percentage changes.

\section{Parameter uncertainty}
\begin{enumerate}\itemsep0pt
\item Standard errors from the numerical Hessian of the penalised negative log-likelihood at
the fitted optimum (no refit).
\item """ + str(n_draws) + r""" draws $\theta^{(k)}=\hat\theta+\varepsilon^{(k)}$, $\varepsilon^{(k)}_i\sim N(0,\mathrm{SE}_i^2)$
independently per parameter (parameters without a valid SE are held at the estimate).
\item Each draw is re-filtered over the observed training series with the model's Numba filter,
giving its own $\varphi_{t|t-1}$, $\pi_{t|t-1}$, static parameters, $c_v$ and $\lambda_1$; the
formulas above are evaluated per draw and per day.
\item Reported: the mean across draws and the 2.5\%/97.5\% percentiles (shaded 95\% band).
\end{enumerate}

\section{Selected model}
Among all one-time-varying-parameter models containing dew point with checked gradient
$\|g\|_\infty<1$ (all locations, GAS($p,q$) and long--short families), the model with the
smallest maximum per-parameter gradient: """ + f"{info['location']}, {info['family']}, {info['stage']}/{info['cfg'].replace('_', chr(92) + '_')}" + r""",
""" + str(info["n_params"]) + r""" parameters, max $|g|=""" + f"{info['max_abs_grad_checked']:.3g}" + r"""$, SE quality: """ + str(info["se_quality"]).replace("_", r"\_") + r""".
Occurrence link: """ + lam_txt + r""".

{\small\begin{longtable}{lrr}\toprule Parameter & Estimate & SE \\ \midrule
""" + rows + r"""
\bottomrule\end{longtable}}
\end{document}
"""
    (rep / "methods.tex").write_text(tex, encoding="utf-8")
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "methods.tex"], cwd=rep,
                       capture_output=True, text=True)
    print("methods.pdf:", "OK" if (rep / "methods.pdf").exists() else "FAILED")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of: mosaic,stage5_mosaic,best_models,std_errors,marginal")
    only = ap.parse_args().only
    want = set(only.split(",")) if only else {"mosaic", "stage5_mosaic", "best_models", "std_errors",
                                              "marginal"}
    met = activate_new_run()
    for name, fn in (("mosaic", mosaic), ("stage5_mosaic", stage5_mosaic), ("best_models", best_models),
                     ("std_errors", std_errors), ("occurrence_fit", occurrence_fit)):
        if name in want:
            print(f"=== {name}", flush=True)
            fn()
    if "marginal" in want:
        print("=== marginal effects", flush=True)
        marginal_effects(met)


if __name__ == "__main__":
    main()
