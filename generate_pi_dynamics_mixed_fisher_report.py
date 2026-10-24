"""
Report for run_pi_dynamics_mixed_fisher.py -> reports/newpi/ (2026-10-01).

Reads ONLY artifacts/pi_dynamics_mixed_fisher/<short>/<alt>/<scaling>/
metadata.json (no fitting) and writes
    reports/newpi/tables/summary.csv   (plain snake_case columns)
    reports/newpi/tables/summary.tex   (the summary table, \\input-able)
    reports/newpi/report.tex / report.pdf

Table: per location x alternative x scaling, validation-block (final two
years, one-step-ahead, no re-fit) RMSE, CRPS, mean pi | y=0, mean pi | y>0,
N dry, N wet and the separation N_wet mean(pi|y>0) - N_dry mean(pi|y=0).
Bold = best RMSE / CRPS / separation within a location.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
ART = ROOT / "artifacts" / "pi_dynamics_mixed_fisher"
REP = ROOT / "reports" / "newpi"
SHORTS = ["bh", "cruzeiro", "darwin", "garanhuns", "manaus", "salvador"]
DISP = {"bh": "Belo Horizonte", "cruzeiro": "Cruzeiro do Sul", "darwin": "Darwin",
        "garanhuns": "Garanhuns", "manaus": "Manaus", "salvador": "Salvador"}
ALTS = ["A", "B", "C", "D", "A7", "C7"]
SCALINGS = ["unit", "mixed_inverse_fisher"]
SC_LBL = {"unit": "unit", "mixed_inverse_fisher": "inv. Fisher"}


def load() -> pd.DataFrame:
    rows = []
    for s in SHORTS:
        for a in ALTS:
            for sc in SCALINGS:
                p = ART / s / a / sc / "metadata.json"
                m = json.loads(p.read_text()) if p.exists() else {"success": False}
                rows.append({
                    "location": s, "alternative": a, "scaling": sc,
                    "success": bool(m.get("success")), "source": m.get("source", ""),
                    "rmse": m.get("val_rmse", np.nan), "crps": m.get("val_crps", np.nan),
                    "mean_pi_y0": m.get("val_pi_mean_dry", np.nan),
                    "mean_pi_ygt0": m.get("val_pi_mean_wet", np.nan),
                    "n_dry": m.get("val_n_dry", np.nan), "n_wet": m.get("val_n_wet", np.nan),
                    "separation": m.get("val_separation", np.nan),
                    "grad_norm_inf": m.get("grad_norm_inf", np.nan),
                    "loglik_fit": m.get("loglik_fit", np.nan), "aic_fit": m.get("aic_fit", np.nan),
                    "n_params": m.get("n_params", np.nan),
                })
    return pd.DataFrame(rows)


def f(v, d=3):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def summary_tex(df: pd.DataFrame) -> str:
    out = [r"{\footnotesize\setlength{\tabcolsep}{3.5pt}",
           r"\begin{longtable}{lllrrrrrrrr}",
           r"\caption{Alternatives A--D, A7, C7 under joint ZA-GAS estimation with unit and "
           r"mixed inverse-Fisher scaling of the $\varphi$ score. Validation block (final two "
           r"years), one step ahead, no re-fit. Sep.\ $= N_{wet}\,\bar\pi_{y>0} - N_{dry}\,"
           r"\bar\pi_{y=0}$. Bold: best RMSE, CRPS and Sep.\ within a location. "
           r"$^\dagger$ joint MLE taken from cache; $^*$ checked gradient $1\leq\|g\|_\infty<100$; "
           r"$^{**}$ $\|g\|_\infty\geq 100$.}"
           r"\label{tab:newpi_summary}\\",
           r"\toprule",
           r"Location & Alt. & Scaling & RMSE & CRPS & $\bar\pi_{y=0}$ & $\bar\pi_{y>0}$ & "
           r"$N$ dry & $N$ wet & Sep. & $\|g\|_\infty$ \\", r"\midrule", r"\endfirsthead",
           r"\toprule",
           r"Location & Alt. & Scaling & RMSE & CRPS & $\bar\pi_{y=0}$ & $\bar\pi_{y>0}$ & "
           r"$N$ dry & $N$ wet & Sep. & $\|g\|_\infty$ \\", r"\midrule", r"\endhead",
           r"\bottomrule", r"\endfoot"]
    for i, s in enumerate(SHORTS):
        sub = df[df.location == s]
        best = {"rmse": sub.rmse.min(), "crps": sub.crps.min(), "separation": sub.separation.max()}
        if i:
            out.append(r"\midrule")
        for _, r in sub.iterrows():
            cells = []
            for k, d in (("rmse", 3), ("crps", 3)):
                c = f(r[k], d)
                cells.append(rf"\textbf{{{c}}}" if np.isfinite(r[k]) and np.isclose(r[k], best[k]) else c)
            sep = f(r.separation, 1)
            sep = rf"\textbf{{{sep}}}" if np.isfinite(r.separation) and np.isclose(r.separation, best["separation"]) else sep
            alt = r.alternative + (r"$^\dagger$" if r.source == "cache" else "")
            g = r.grad_norm_inf
            mark = "" if not np.isfinite(g) or g < 1 else ("$^*$" if g < 100 else "$^{**}$")
            gtxt = ("--" if not np.isfinite(g) else f"{g:.1e}") + mark
            nd = "--" if not np.isfinite(r.n_dry) else str(int(r.n_dry))
            nw = "--" if not np.isfinite(r.n_wet) else str(int(r.n_wet))
            out.append(f"{DISP[s]} & {alt} & {SC_LBL[r.scaling]} & {cells[0]} & {cells[1]} & "
                       f"{f(r.mean_pi_y0)} & {f(r.mean_pi_ygt0)} & {nd} & {nw} & {sep} & {gtxt} \\\\")
    out += [r"\end{longtable}}"]
    return "\n".join(out)


def best_table(df: pd.DataFrame) -> str:
    rows = []
    for s in SHORTS:
        sub = df[(df.location == s) & df.success]
        def lab(r):
            return f"{r.alternative} ({SC_LBL[r.scaling]})"
        rows.append(f"{DISP[s]} & {lab(sub.loc[sub.rmse.idxmin()])} & {lab(sub.loc[sub.crps.idxmin()])} & "
                    f"{lab(sub.loc[sub.separation.idxmax()])} \\\\")
    return "\n".join([r"\begin{table}[H]\centering\small",
                      r"\caption{Best alternative per location and criterion (validation block).}",
                      r"\label{tab:newpi_best}", r"\begin{tabular}{llll}", r"\toprule",
                      r"Location & lowest RMSE & lowest CRPS & highest Sep. \\", r"\midrule"]
                     + rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def scaling_effect_table(df: pd.DataFrame) -> str:
    """Mixed inverse Fisher minus unit, per alternative, averaged over locations."""
    rows = []
    for a in ALTS:
        u = df[(df.alternative == a) & (df.scaling == "unit")].set_index("location")
        m = df[(df.alternative == a) & (df.scaling == "mixed_inverse_fisher")].set_index("location")
        d = (m[["rmse", "crps", "separation"]] - u[["rmse", "crps", "separation"]])
        wins = int((d.crps < 0).sum())
        rows.append(f"{a} & {f(d.rmse.mean())} & {f(d.crps.mean())} & {f(d.separation.mean(), 1)} & "
                    f"{wins}/{int(d.crps.notna().sum())} \\\\")
    return "\n".join([r"\begin{table}[H]\centering\small",
                      r"\caption{Effect of the scaling: mixed inverse Fisher minus unit, mean over the "
                      r"six locations (negative RMSE/CRPS = inverse Fisher better), and the number of "
                      r"locations where inverse Fisher has the lower CRPS.}",
                      r"\label{tab:newpi_scaling}", r"\begin{tabular}{lrrrr}", r"\toprule",
                      r"Alt. & $\Delta$RMSE & $\Delta$CRPS & $\Delta$Sep. & inv.\ Fisher better (CRPS) \\",
                      r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"])


METHOD = r"""
\section*{Set-up}
Same data, split and magnitude recursion as the earlier $\pi$-dynamics validation experiment:
each model is estimated on the first eight years of the training series and scored one step
ahead, without re-fitting, on the final two years. The magnitude model is a $\varphi$-only
GAS(1,1) with static $\xi,\gamma,\zeta$. Unlike the earlier experiment, every alternative is
now estimated \emph{jointly} as one zero-adjusted GAS model.

\paragraph{Alternatives} ($\eta_t=\operatorname{logit}\pi_{t|t-1}$)
\begin{itemize}\itemsep0pt
\item A: $\omega_0+\rho\eta_{t-1}+\sum_{l\in\{1,365,366\}}w_l y_{t-l}$
\item B: $\omega_0+\rho\eta_{t-1}+\sum_{l=1}^{3}w_l y_{t-l}$
\item C: $\omega_0+\sum_{l=1}^{3}w_l y_{t-l}$
\item D: $\lambda_0+\lambda_1\varphi_{t|t-1}$, with $\varphi_{t|t-1}$ the state of the same filter
(formed from information up to $t-1$, then $\pi_t$, then the update with $y_t$).
\item A7: as A with $l\in\{1,2,3,364,365,366,367\}$
\item C7: as C with $l\in\{1,2,3,364,365,366,367\}$
\end{itemize}

\paragraph{Score and Fisher information of the mixed density}
For $f(y)=(1-\pi_t)\mathbb 1\{y=0\}+\pi_t\,g(y;\varphi_t)\mathbb 1\{y>0\}$ and $I_t=\mathbb 1\{y_t>0\}$,
\[
s_t=\frac{\partial\log f}{\partial\varphi}=I_t\,\frac{\partial\log g}{\partial\varphi}
+\lambda_1\,(I_t-\pi_t),\qquad
\mathcal I_t=\mathbb E[s_t^2]=\pi_t\,\mathcal I_g+\lambda_1^2\,\pi_t(1-\pi_t),
\]
where $\mathcal I_g=p^2ab/(a+b+1)$ is the GB2 Fisher information for $\varphi$ and $\lambda_1=0$
except for alternative D, the only one whose $\pi_t$ depends on $\varphi$ (logistic link,
$\partial\pi/\partial\varphi=\lambda_1\pi(1-\pi)$; the cross term vanishes because
$\mathbb E[\partial\log g/\partial\varphi]=0$). ``unit'' uses $s_t$; ``inv.\ Fisher'' uses
$s_t/\mathcal I_t$. The analytical $\mathcal I_t$ was checked against a Monte Carlo estimate of
$\mathbb E[s_t^2]$ (within 1\%).

\paragraph{Estimation} BFGS on the penalised joint log-likelihood. Under unit scaling the
likelihood of A, B, C separates into an occurrence block and a magnitude block with no shared
parameters, so their joint MLE is the cached standalone $\pi$ fit plus the cached unit magnitude
fit ($^\dagger$; used as is when its joint checked gradient is below 1, otherwise re-optimised
from it). Every other cell is a fresh joint fit, keeping the best of two starting values. All
filtering runs in the Numba kernel; it matches the Python implementation (parity test
\texttt{tests/test\_mixed\_fisher\_parity.py}).

\paragraph{Convergence} Fresh fits whose checked gradient was $\geq 1$ were restarted with BFGS
(tighter tolerance) up to six times; no restart raised the log-likelihood. The remaining large
gradients are almost entirely in $B_\varphi$ (and $A_\varphi$): at Manaus (every alternative) and for
alternative D at Cruzeiro do Sul the persistence sits at $B_\varphi\approx0.980$, the threshold of the
soft persistence penalty, where the finite-difference gradient straddles the penalty kink; elsewhere
$B_\varphi$ is close to a unit root and the objective is ill-conditioned in that direction. Per
observation all gradients are below $0.05$. For D at Cruzeiro do Sul $\lambda_0$ and $\lambda_1$ are
large and of opposite sign ($\pi_t$ depends on $\varphi_{t|t-1}-\bar\varphi$ only), a weak
identification of the intercept.
"""


def build():
    (REP / "tables").mkdir(parents=True, exist_ok=True)
    df = load()
    df.to_csv(REP / "tables" / "summary.csv", index=False)
    (REP / "tables" / "summary.tex").write_text(summary_tex(df), encoding="utf-8")
    failed = df[~df.success]
    fail_txt = ("" if failed.empty else
                r"\paragraph{Failed fits} " + ", ".join(f"{DISP[r.location]} {r.alternative} "
                                                       f"({SC_LBL[r.scaling]})" for _, r in failed.iterrows()) + ".")
    doc = "\n".join([
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}",
        r"\usepackage[margin=1.8cm]{geometry}",
        r"\usepackage{amsmath,amssymb,booktabs,longtable,float}",
        r"\usepackage{parskip}\emergencystretch=3em\sloppy",
        r"\begin{document}",
        r"\title{$\pi$-dynamics alternatives: joint ZA-GAS, unit vs mixed inverse-Fisher scaling}",
        r"\author{}\date{1 October 2026}\maketitle",
        METHOD, fail_txt,
        r"\section*{Results}", r"\input{tables/summary.tex}",
        best_table(df), scaling_effect_table(df),
        r"\end{document}"])
    (REP / "report.tex").write_text(doc, encoding="utf-8")
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], cwd=REP,
                       capture_output=True, text=True)
    log = (REP / "report.log").read_text(errors="ignore")
    print("report.pdf:", "OK" if (REP / "report.pdf").exists() and "Fatal error" not in log else "FAILED")
    print(df[["location", "alternative", "scaling", "rmse", "crps", "mean_pi_y0", "mean_pi_ygt0",
              "separation", "grad_norm_inf"]].round(3).to_string(index=False))


if __name__ == "__main__":
    build()
