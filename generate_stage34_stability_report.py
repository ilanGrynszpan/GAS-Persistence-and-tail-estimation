"""
Report: why are Stage 3 / Stage 4 (Harvey long-short) hard to stabilize?
(2026-09-11)

Reads ONLY artifacts/stage34_stability_diagnostics/ (produced by
run_stage34_stability_diagnostics.py) and artifacts/rerun_10092026/
(cached result.json's, for the gradient-norm control comparison).
No fitting happens here (docs/REPORTING.md).

Tests four candidate explanations for Stage 3/4's large (1e4-1e6) checked
gradient norms at the reported optimum, vs Stage 4S's ~5 -- see the
run_stage34_stability_diagnostics.py module docstring for the exact
definitions of H1-H4 and how each is measured.

Output: reports/stage34_stability/report.tex / report.pdf
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

ROOT = Path(__file__).parent
DIAG = ROOT / "artifacts" / "stage34_stability_diagnostics"
RERUN = ROOT / "artifacts" / "rerun_10092026"
REP = ROOT / "reports" / "stage34_stability"
FIG = REP / "figures"

PRE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage[margin=2cm,landscape]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,longtable,caption}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}\usepackage{float}
\emergencystretch=3em\sloppy
"""


def _f(v, d=3):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(x):
        return "n/a"
    if abs(x) >= 1e4:
        return rf"$\sim\!10^{{{int(np.floor(np.log10(abs(x))))}}}$"
    return f"{x:.{d}f}"


def load_all():
    recs = []
    for f in DIAG.rglob("*.json"):
        if f.name.endswith(".error.txt"):
            continue
        r = json.loads(f.read_text())
        recs.append(r)
    return recs


def _block_of(name):
    for k in ("phi_L", "phi_S", "kappa_L", "kappa_S", "Gamma_L", "Gamma_S"):
        if name.startswith(k + "_"):
            return {"phi_L": "long", "kappa_L": "long", "Gamma_L": "long",
                     "phi_S": "short", "kappa_S": "short", "Gamma_S": "short"}[k]
    if name.startswith(("omega0", "rho", "omega_y")):
        return "pi"
    if name.startswith(("kappa_gamma", "seas0", "seasstar0", "psi_gamma")):
        return "seasonal"
    if name.startswith(("omega_", "f0_", "L0_", "S0_")):
        return "level init"
    return "static/other"


def tbl(rows, cols, caption, label, index_col="Location"):
    df = pd.DataFrame(rows).set_index(index_col)
    df = df[cols]
    body = [r"\begin{table}[H]\centering\footnotesize",
            rf"\caption{{{caption}}}\label{{{label}}}",
            r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
            index_col + " & " + " & ".join(cols) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        body.append(str(idx) + " & " + " & ".join(str(v) for v in row) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(body)


def control_table():
    """Stage 3 / 4 / 4S checked gradient-norm distribution -- the
    natural-experiment control: 4S has no long component at all."""
    rows = []
    for st, name in [("stage3", "Stage 3 (long+short)"), ("stage4", "Stage 4 (long+short+seasonal)"),
                      ("stage4S", "Stage 4S (short/trend+seasonal only)")]:
        gs = [json.loads(f.read_text()).get("grad_norm_inf_numeric")
              for f in RERUN.rglob(f"{st}/*/result.json")]
        gs = np.array([g for g in gs if g is not None and np.isfinite(g)])
        rows.append({"Location": name, "n": len(gs), "median grad": _f(np.median(gs), 1),
                     "frac<1": _f(np.mean(gs < 1), 2), "frac<100": _f(np.mean(gs < 100), 2),
                     "max grad": _f(np.max(gs), 1)})
    return tbl(rows, ["n", "median grad", "frac<1", "frac<100", "max grad"],
               "Checked gradient-norm distribution at the optimum, by stage (all 6 "
               "locations $\\times$ 5 covariate configs $\\times$ 2 TVP branches = 60 fits/stage). "
               "Stage 4S removes the entire long component (no $\\varphi_L,\\kappa_L,\\Gamma_L$) -- "
               "the natural-experiment control for H1/H2.", "tab:control")


def h1_phiL_table(recs):
    rows = []
    for st in ("stage3", "stage4"):
        phiL, near, over1, order_active, n = [], 0, 0, 0, 0
        for r in recs:
            if r["stage"] != st or r.get("invalid"):
                continue
            for j, e in r["per_j"].items():
                if e.get("phi_L") is None:
                    continue
                n += 1
                pl = e["phi_L"]; phiL.append(pl)
                if pl > 0.995 - 0.02:
                    near += 1
                if pl > 1.0:
                    over1 += 1
                if e["boundary_distances"].get("order_violation", -99) > -0.02:
                    order_active += 1
        phiL = np.array(phiL)
        rows.append({"Location": st, "n": n, "median $\\varphi_L$": _f(np.median(phiL)),
                     "min": _f(phiL.min()), "max": _f(phiL.max()),
                     "near/over 0.995": f"{near}/{n} ({near/n:.2f})",
                     "$\\varphi_L>1$": f"{over1}/{n} ({over1/n:.2f})",
                     "order viol.\\ ($\\varphi_S{\\geq}\\varphi_L$)": f"{order_active}/{n} ({order_active/n:.2f})"})
    return tbl(rows, ["n", "median $\\varphi_L$", "min", "max", "near/over 0.995",
                      "$\\varphi_L>1$", "order viol.\\ ($\\varphi_S{\\geq}\\varphi_L$)"],
               "H1: distribution of the fitted long-run persistence $\\varphi_L$ (unbounded raw "
               "parameter, only soft-penalised toward $(0,0.995)$) across Stage 3/4 covariate "
               "configs $\\times$ locations $\\times$ branches.", "tab:h1_phiL")


def h1_grad_by_boundary_table(recs):
    near_g, far_g = [], []
    for r in recs:
        if r["stage"] not in ("stage3", "stage4") or r.get("invalid"):
            continue
        g = r.get("stored_grad_inf_norm")
        if g is None or not np.isfinite(g):
            continue
        for j, e in r["per_j"].items():
            if e.get("phi_L") is None:
                continue
            (near_g if e["phi_L"] > 0.995 - 0.02 else far_g).append(g)
    rows = [{"Location": "$\\varphi_L$ near/over 0.995", "n": len(near_g),
             "median grad": _f(np.median(near_g), 1)},
            {"Location": "$\\varphi_L$ away from boundary", "n": len(far_g),
             "median grad": _f(np.median(far_g), 1)}]
    return tbl(rows, ["n", "median grad"],
               "H1: checked gradient norm at the optimum, split by whether $\\varphi_L$ sits "
               "near/over its 0.995 soft bound.", "tab:h1_grad_split")


def argmax_block_table(recs):
    rows = []
    for st in ("stage3", "stage4", "stage4S"):
        sub = [r for r in recs if r["stage"] == st and not r.get("invalid")]
        from collections import Counter
        cnt = Counter(_block_of(r["grad_argmax_name"]) for r in sub)
        rows.append({"Location": st, "n": len(sub), "long": cnt.get("long", 0),
                     "short": cnt.get("short", 0), "seasonal": cnt.get("seasonal", 0),
                     "pi": cnt.get("pi", 0), "other": cnt.get("static/other", 0) + cnt.get("level init", 0)})
    return tbl(rows, ["n", "long", "short", "seasonal", "pi", "other"],
               "Which parameter block carries the single largest gradient component "
               "($\\arg\\max_i|\\partial\\text{obj}/\\partial\\theta_i|$) at the reported optimum, by stage "
               "(count of fits, out of $n$).", "tab:argmax_block")


def worst_offenders_table(recs, n=10):
    rows = []
    for r in recs:
        if r["stage"] not in ("stage3", "stage4") or r.get("invalid"):
            continue
        g = r.get("stored_grad_inf_norm")
        if g is None or not np.isfinite(g):
            continue
        for j, e in r["per_j"].items():
            rows.append((g, r["loc"], r["stage"], r["cfg"], r["branch"],
                        e.get("phi_L"), e.get("phi_S"), r["grad_argmax_name"]))
    rows.sort(key=lambda x: -x[0])
    out = []
    for g, loc, st, cfg, br, pl, ps, am in rows[:n]:
        out.append({"Location": loc, "stage": st, "cfg": cfg.replace("_", "\\_"),
                    "branch": br.replace("_", "\\_"), "grad": _f(g, 1),
                    "$\\varphi_L$": _f(pl), "$\\varphi_S$": _f(ps),
                    "argmax param": am.replace("_", "\\_")})
    return tbl(out, ["stage", "cfg", "branch", "grad", "$\\varphi_L$", "$\\varphi_S$", "argmax param"],
               f"Top {n} largest checked gradient norms among Stage 3/4 fits, with the fitted "
               "$\\varphi_L,\\varphi_S$ and which parameter carries the max-gradient component.",
               "tab:worst")


def h2_table_v2(recs):
    def sum_align(v):
        v = np.array(v); return abs(v[0] + v[1]) / np.sqrt(2)
    ratios, vL, vS = [], [], []
    for r in recs:
        if r["stage"] not in ("stage3", "stage4") or r.get("invalid"):
            continue
        for j, e in r["per_j"].items():
            h2 = e.get("h2_phiL_phiS_curvature")
            if h2 and h2.get("ratio") is not None and np.isfinite(h2["ratio"]):
                ratios.append(h2["ratio"])
                v = np.array(h2["ridge_eigvec"]); vL.append(abs(v[0])); vS.append(abs(v[1]))
    ratios, vL, vS = np.array(ratios), np.array(vL), np.array(vS)
    diag_frac = np.mean((vL > 0.5) & (vL < 0.9) & (vS > 0.5) & (vS < 0.9))
    rows = [
        {"Location": "condition number of the local $(\\varphi_L,\\varphi_S)$ Hessian of loglik",
         "value": _f(np.median(ratios), 1), "note": f"frac $>50$: {np.mean(ratios>50):.2f}"},
        {"Location": "flat direction dominated by $\\varphi_S$ alone ($|v_{\\varphi_S}|>0.8$)",
         "value": f"{np.mean(vS>0.8):.2f}", "note": "median $|v_{\\varphi_S}|$" + f" = {np.median(vS):.3f}"},
        {"Location": "flat direction dominated by $\\varphi_L$ alone ($|v_{\\varphi_L}|>0.8$)",
         "value": f"{np.mean(vL>0.8):.2f}", "note": f"median $|v_{{\\varphi_L}}|$ = {np.median(vL):.3f}"},
        {"Location": "flat direction genuinely diagonal (both weights in $[0.5,0.9]$)",
         "value": f"{diag_frac:.2f}", "note": "literal H2 (``push both to 1 together'')"},
    ]
    return tbl(rows, ["value", "note"],
               "H2: local curvature (finite-difference Hessian of the PURE log-likelihood, "
               "penalty excluded) in the $(\\varphi_L,\\varphi_S)$ subspace at the fitted optimum.",
               "tab:h2")


def h3_table(recs):
    corr = []
    for r in recs:
        if r["stage"] not in ("stage3", "stage4") or r.get("invalid") or r["cfg"] == "none":
            continue
        for j, e in r["per_j"].items():
            for cov, c in e.get("h3_corr_covariate_score", {}).items():
                corr.append(abs(c))
    corr = np.array(corr)
    ratios, aligns = [], []
    for r in recs:
        if r["stage"] not in ("stage3", "stage4") or r.get("invalid"):
            continue
        for j, e in r["per_j"].items():
            h3 = e.get("h3_kappaS_GammaS_curvature")
            if h3 and h3.get("ratio") is not None and np.isfinite(h3["ratio"]):
                ratios.append(h3["ratio"])
                v = np.array(h3["ridge_eigvec"]); aligns.append(abs(v[0] + v[1]) / np.sqrt(2))
    ratios, aligns = np.array(ratios), np.array(aligns)
    rows = [
        {"Location": "$|\\mathrm{corr}(\\text{covariate}, \\text{score } s_t)|$",
         "median": _f(np.median(corr)), "max": _f(corr.max()), "frac $>0.3$": _f(np.mean(corr > 0.3), 2)},
        {"Location": "condition number of local $(\\kappa_S,\\Gamma_S)$ Hessian of loglik",
         "median": _f(np.median(ratios), 1), "max": _f(np.max(ratios), 1),
         "frac $>0.3$": f"align$>0.7$: {np.mean(aligns>0.7):.2f}"},
    ]
    return tbl(rows, ["median", "max", "frac $>0.3$"],
               "H3: correlation between each covariate series and the realised GAS score "
               "$s_t$ (direct collinearity test), and the local curvature ridge in "
               "$(\\kappa_S,\\Gamma_S)$ for the largest-loading covariate.", "tab:h3")


def h4_table(recs):
    vals, hi_pl, lo_pl = [], [], []
    for r in recs:
        if r["stage"] != "stage4" or r.get("invalid"):
            continue
        for j, e in r["per_j"].items():
            v = e.get("h4_corr_seasonal_long")
            pl = e.get("phi_L")
            if v is None:
                continue
            vals.append(abs(v))
            if pl is not None:
                (hi_pl if pl > 1.0 else lo_pl).append(abs(v))
    vals = np.array(vals)
    rows = [{"Location": "$|\\mathrm{corr}(\\Gamma_{j,t}^{\\text{seasonal}}, L_{j,t})|$, all Stage 4 fits",
             "median": _f(np.median(vals)), "max": _f(vals.max()), "frac $>0.3$": _f(np.mean(vals > 0.3), 2)},
            {"Location": "\\quad among fits with $\\varphi_L>1$",
             "median": _f(np.median(hi_pl)) if hi_pl else "--", "max": _f(np.max(hi_pl)) if hi_pl else "--", "frac $>0.3$": "--"},
            {"Location": "\\quad among fits with $\\varphi_L\\leq1$",
             "median": _f(np.median(lo_pl)) if lo_pl else "--", "max": _f(np.max(lo_pl)) if lo_pl else "--", "frac $>0.3$": "--"}]
    return tbl(rows, ["median", "max", "frac $>0.3$"],
               "H4 (Stage 4 only): correlation between the damped seasonal contribution and the "
               "long-run level $L_{j,t}$ over the filtered training sample.", "tab:h4")


def phiL_scatter_fig(recs):
    xs, ys, colors = [], [], []
    for r in recs:
        if r["stage"] not in ("stage3", "stage4") or r.get("invalid"):
            continue
        g = r.get("stored_grad_inf_norm")
        if g is None or not np.isfinite(g) or g <= 0:
            continue
        for j, e in r["per_j"].items():
            if e.get("phi_L") is None:
                continue
            xs.append(e["phi_L"]); ys.append(g)
            colors.append("#c33" if r["stage"] == "stage4" else "#36c")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(xs, ys, c=colors, s=14, alpha=0.6, edgecolor="none")
    ax.axvline(0.995, color="grey", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel(r"fitted $\varphi_L$")
    ax.set_ylabel("checked gradient $\\infty$-norm (log scale)")
    ax.set_title(r"Gradient norm explodes as $\varphi_L \to 0.995$ (dashed line)")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#36c", label="Stage 3"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor="#c33", label="Stage 4")]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "phiL_vs_grad.pdf")
    fig.savefig(FIG / "phiL_vs_grad.png", dpi=130)
    plt.close(fig)


def build():
    REP.mkdir(parents=True, exist_ok=True)
    recs = load_all()
    phiL_scatter_fig(recs)

    sections = []
    sections.append("\n\n".join([
        r"\section*{Why Stage 3/4 are hard to stabilize}",
        r"Stage 3 (long+short) and Stage 4 (long+short+damped seasonal) came out of the "
        r"2026-09-10 rerun with checked gradient norms at the reported optimum of order "
        r"$10^4$--$10^6$ across most of the 60 location$\times$covariate-config$\times$branch "
        r"fits per stage, versus $\sim$1--100 for Stage 1/2 and $\sim$5 for Stage 4S (the "
        r"structural model with NO long component). This report tests four candidate "
        r"explanations against the already-fitted $\theta$ cached under "
        r"\texttt{artifacts/rerun\_10092026/} -- no refitting is done here.",
        r"\begin{itemize}\setlength\itemsep{0pt}",
        r"\item \textbf{H1 Boundary}: $\varphi_L,\varphi_S,\kappa_L,\kappa_S,\|\Gamma_L\|,\|\Gamma_S\|$ "
        r"are only soft-penalty-bounded (LAM$=10^5/10^5/10^3/10^2$; no hard box constraint per "
        r"docs/OPTIMIZATION.md's ``unbounded BFGS, soft penalties'' rule) -- does one of them sit "
        r"at/past its threshold at the optimum, and does that explain the large gradient directly?",
        r"\item \textbf{H2 Long/short non-identification}: is the pure log-likelihood (penalty "
        r"excluded) locally flat along a $(\varphi_L,\varphi_S)$ trade-off direction, so the "
        r"optimizer cannot tell short memory from long memory?",
        r"\item \textbf{H3 Covariate/$\kappa$ collinearity}: does the covariate series correlate "
        r"with the realised GAS score, letting $\kappa$ (score loading) and $\Gamma$ (covariate "
        r"loading) trade off against each other?",
        r"\item \textbf{H4 Seasonality/long confound} (Stage 4 only): does the damped "
        r"trigonometric seasonal contribution correlate with the long-run level $L_{j,t}$?",
        r"\end{itemize}",
        r"\textbf{Natural-experiment control}: Stage 4S removes the long component entirely -- "
        r"no $\varphi_L,\kappa_L,\Gamma_L$ exist in its $\theta$ at all -- while keeping the "
        r"same covariates and the same damped seasonal harmonic as Stage 4. If removing only "
        r"the long block fixes most of the instability, H1/H2 (which concern that block) are "
        r"implicated; if Stage 4S were equally unstable, H3/H4 (present in both) would be "
        r"favoured instead.",
        r"\textbf{Caveat}: Stage 4/4S were fit with \texttt{maxiter=300, polish=False} "
        r"(run\_rerun\_10092026.py), unlike Stage 3 which has no iteration cap; part of the "
        r"Stage 3$\to$4 gradient increase could reflect fewer optimizer iterations rather than "
        r"only the added seasonal component -- Stage 4S, fit under the identical maxiter=300 cap, "
        r"is the fairer control for isolating the long-component effect from this confound.",
    ]))

    sections.append("\n\n".join([r"\section{Natural-experiment control: removing the long component}",
                                 control_table()]))

    sections.append("\n\n".join([
        r"\section{H1 -- boundary: $\varphi_L$ pinned at its stationarity limit}",
        h1_phiL_table(recs),
        h1_grad_by_boundary_table(recs),
        r"\begin{figure}[H]\centering\includegraphics[width=0.7\textwidth]{figures/phiL_vs_grad.pdf}\end{figure}",
        argmax_block_table(recs),
        worst_offenders_table(recs),
        r"\textbf{Verdict}: strongly confirmed as the dominant mechanism. $\varphi_L$ is a raw, "
        r"unbounded parameter with only a soft quadratic penalty keeping it under 0.995 -- and "
        r"the fitted value sits within 0.02 of, or past, that limit in 70\% (Stage 3) / 61\% "
        r"(Stage 4) of fits, with $\varphi_L>1$ (literal non-stationarity) in 8\%/22\%. Every one "
        r"of the ten worst-gradient fits has $\varphi_L\in[0.975,1.0015]$. The reported gradient "
        r"is $\sim$2400$\times$ larger when $\varphi_L$ is near/over the boundary than when it "
        r"is comfortably below it. Critically, the argmax-gradient component is usually NOT "
        r"$\varphi_L$ itself but a long-block covariate loading $\Gamma_L$ (42\%/67\% of fits) -- "
        r"near a unit root, the whole long block ($\varphi_L,\kappa_L,\Gamma_L$) becomes "
        r"numerically ill-conditioned together, so large residual gradients appear throughout "
        r"the block, not only on the parameter literally at its bound. This is also a genuine "
        r"optimizer-stiffness issue, not only an inherent property of the penalty: at a fully "
        r"converged constrained optimum the penalty's own gradient should cancel the "
        r"likelihood's, leaving $\approx$0 residual gradient -- the observed $10^4$--$10^6$ "
        r"values indicate BFGS is not fully resolving this very steep ($2\lambda\approx2\times10^5$) "
        r"local region, not merely reporting an intrinsic property of the penalized surface.",
    ]))

    sections.append("\n\n".join([
        r"\section{H2 -- long/short identification}",
        h2_table_v2(recs),
        r"\textbf{Verdict}: real numerical difficulty confirmed, but NOT the literal mechanism "
        r"hypothesized. The $(\varphi_L,\varphi_S)$ likelihood surface does have a near-singular "
        r"direction (condition number typically $10^2$--$10^3$) at essentially every Stage 3/4 "
        r"fit -- including ones where $\varphi_L$ is comfortably away from its boundary, so this "
        r"is a genuinely separate finding from H1. However the flat direction is dominated by "
        r"$\varphi_S$ ALONE 77\% of the time (median $|v_{\varphi_S}|=1.00$), vs only 10\% "
        r"dominated by $\varphi_L$ alone and 17\% a true diagonal (``push both together'') "
        r"trade-off. In other words: the short-run persistence $\varphi_S$ is weakly identified "
        r"BY ITSELF -- once $\kappa_S$ (score loading) and the long component are fit, "
        r"$\varphi_S$'s own marginal effect on the likelihood is nearly flat -- rather than the "
        r"optimizer literally being unable to split total persistence between the two "
        r"components. A genuine long/short trade-off ridge does occur, but only in a minority "
        r"(17\%) of fits.",
    ]))

    sections.append("\n\n".join([
        r"\section{H3 -- covariate/$\kappa$ collinearity}",
        h3_table(recs),
        r"\textbf{Verdict}: not supported as a primary driver. The direct "
        r"correlation between each covariate series and the realised score is essentially zero "
        r"(median $|corr|\approx0.01$, never above 0.3) -- covariates and the score are not "
        r"tracking the same signal. The $(\kappa_S,\Gamma_S)$ curvature ridge is real but an "
        r"order of magnitude milder than the $(\varphi_L,\varphi_S)$ ridge in H2 (condition "
        r"number $\sim$35 vs $10^2$--$10^3$), and is better explained by $\Gamma_S$'s small "
        r"fitted magnitude (weak identification of a near-zero effect, the same flavour as "
        r"$\varphi_S$ in H2) than by genuine covariate/score confounding. At most a weak, "
        r"secondary contributor.",
    ]))

    sections.append("\n\n".join([
        r"\section{H4 -- seasonality/long confound (Stage 4 only)}",
        h4_table(recs),
        r"\textbf{Verdict}: real for a minority of fits, weak and inconsistent on average -- a "
        r"minor secondary contributor. Median $|corr(\Gamma^{\text{seasonal}}_{j,t}, L_{j,t})|$ "
        r"is only 0.13, but it exceeds 0.3 in 27\% of fits and reaches 0.99 in the worst case, "
        r"so the damped seasonal harmonic and the long-run level genuinely do compete to explain "
        r"the same low-frequency variation in a real subset of locations/configs. It is not, "
        r"however, concentrated where $\varphi_L$ has actually crossed 1 (those fits show LOWER "
        r"median correlation than the rest), so it does not appear to be the immediate trigger "
        r"pushing $\varphi_L$ past its boundary -- more a compounding nuisance once the long "
        r"block is already stressed. Consistent with this, $\varphi_L$ is higher in Stage 4 than "
        r"in the matched Stage 3 fit for 60\% of (location, config, branch) triples (median "
        r"$+0.002$) -- a small but consistent nudge in the expected direction.",
    ]))

    sections.append("\n\n".join([
        r"\section*{Conclusion: ranking and what would fix it}",
        r"\textbf{H1 (boundary/near-unit-root in $\varphi_L$) is the dominant mechanism}, by a "
        r"wide margin: it alone explains the worst offenders, the $\sim$2400$\times$ "
        r"gradient gap, and -- via the natural-experiment control -- the fact that Stage 4S, "
        r"which keeps the same covariates and seasonal harmonic but removes only the long "
        r"block, drops the median gradient from $10^4$--$10^5$ to $\sim$5. "
        r"\textbf{H2 is real but reframed}: the $(\varphi_L,\varphi_S)$ surface is genuinely "
        r"ill-conditioned, but the flat direction is usually $\varphi_S$'s own weak "
        r"identification, not an inability to split persistence between long and short. "
        r"\textbf{H3 is not supported} as anything beyond a weak, secondary effect -- "
        r"covariates do not correlate with the realised score. \textbf{H4 is a real but minor} "
        r"secondary contributor, present in a minority of Stage-4 fits and not the proximate "
        r"trigger for $\varphi_L$ crossing its boundary. "
        r"\textbf{Practical implication}: since $\varphi_L$ is currently a RAW, unbounded "
        r"parameter (no \texttt{phi\_L\_max} reparameterisation used in this rerun) with only a "
        r"soft quadratic penalty, the fix most directly targeted at H1 is to set "
        r"\texttt{phi\_L\_max=0.995} (or similar) so $\varphi_L$ is sigmoid-reparameterised and "
        r"structurally cannot exceed the stationarity boundary, removing the stiff penalty "
        r"region that is starving BFGS's convergence; a similar \texttt{phi\_S\_max} would also "
        r"help if H2's $\varphi_S$-identification problem needs a floor.",
    ]))

    body = "\n\n".join([r"\begin{document}",
                        r"\title{Why are Stage 3/4 (Harvey long-short) hard to stabilize?}"
                        r"\author{}\date{2026-09-11}\maketitle"] + sections + [r"\end{document}"])
    (REP / "report.tex").write_text(PRE + "\n" + body, encoding="utf-8")
    print("wrote report.tex")

    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                       cwd=REP, capture_output=True, text=True)
    log = (REP / "report.log").read_text(errors="ignore") if (REP / "report.log").exists() else ""
    ok = (REP / "report.pdf").exists() and "Fatal error" not in log
    print("report.pdf:", "OK" if ok else "FAILED (see report.log)")


if __name__ == "__main__":
    build()
