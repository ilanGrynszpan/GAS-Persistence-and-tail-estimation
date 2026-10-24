"""
2026-10-01: Stage 5 of the phi-linked-pi staged rerun -> reports/stage5_011026/.

Stage 5 fits phi AND xi time-varying with long-short (and, where the base has
one, seasonal) dynamics on each location's best Stage 3/4/4S configuration
(gradient tier, then OOS CRPS, then OOS RMSE), with the phi-only branch's
Stage 1 scaling (run_rerun_pi_01102026.py, phase C).

Reads ONLY artifacts/rerun_pi_01102026/<short>/stage5/best/result.json (and the
2026-09 Stage 5 result for comparison). No fitting. Table built by
generate_regeneratedwithpi_01102026_report.t_stage5.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_ml_comparison_report as G
import generate_regeneratedwithpi_01102026_report as P
import generate_regenerated_01102026_report as RG

REP = ROOT / "reports" / "stage5_011026"


def details_table() -> str:
    rows = []
    for s in G.SHORTS:
        p = P.NEW_TREE / s / "stage5" / "best" / "result.json"
        if not p.exists():
            rows.append(f"{RG.DISP[s]} & \\multicolumn{{6}}{{c}}{{not available}} \\\\")
            continue
        r = json.loads(p.read_text())
        g = r.get("grad_norm_inf_numeric")
        mark = "" if g is None or g < 1 else ("$^*$" if g < 100 else "$^{**}$")
        rows.append(f"{RG.DISP[s]} & {r.get('scaling', '').replace('_', ' ')} & "
                    f"{RG.fmt(r.get('rmse'), 3)} & {RG.fmt(r.get('crps_mean'), 3)} & "
                    f"{RG.fmt(r.get('BSS'), 4)} & {g:.2e}{mark} & {r.get('loglik', float('nan')):.1f} \\\\")
    note = RG.Notes()
    note.other.append(r"$\|g\|_\infty$: checked gradient of the penalised objective at the optimum; "
                      r"$^*$ $1\leq\|g\|_\infty<100$, $^{**}$ $\geq100$ (not converged).")
    hdr = r"Location & Scaling & RMSE & CRPS & BSS & $\|g\|_\infty$ & log-lik. \\"
    return RG.table("Stage 5: rolling one-step-ahead out-of-sample RMSE, CRPS and Brier skill score, "
                    "and convergence.", "tab:stage5_details", "llrrrrr", hdr, rows, note)


def comparison_table(key: str, lvl: str) -> str:
    """Stage 5 vs the best model of each class at this twCRPS level: within each
    class (both branches pooled), the best gradient tier reached, then the lowest
    twCRPS (generate_best_models_report.select_by_metric). Bold: row minimum.
    Values from 2026-09 fallback fits keep their '*'."""
    import generate_best_models_report as B
    rows, note = [], RG.Notes()
    with P.view("new"):
        for s in G.SHORTS:
            vals = []
            p = P.NEW_TREE / s / "stage5" / "best" / "result.json"
            r5 = json.loads(p.read_text()) if p.exists() else {}
            vals.append(r5.get("twcrps_" + lvl))
            for st in RG.STAGES5:
                cands = []
                for br in ("phi_only", "phi_xi"):
                    cands += [((c, br), mm) for c, mm in B.candidates(s, st, br)]
                ident, mm = B.select_by_metric(cands, key)
                vals.append(mm.get(key) if mm else None)
                if ident:
                    note.other.append(f"{RG.DISP[s]}, {RG.CLASS_LBL[st]}: "
                                      f"{ident[0].replace('_', '-')} ({RG.BRANCH_SHORT[ident[1]]})")
            cells = RG.bold_min(vals, 3)
            if not p.exists():
                cells[0] = "pending"
            rows.append(f"{RG.DISP[s]} & " + " & ".join(cells) + r" \\")
    note.other.insert(0, r"Selected configurations --")
    SS = lambda t: r"\shortstack{" + t + "}"
    hdr = ("Location & Stage 5 & " + " & ".join([
        SS(r"GAS($p,q$)\\no ext."), SS(r"GAS($p,q$)\\+ ext."), SS(r"Long--short"),
        SS(r"Long--short\\+ seas."), SS(r"DCS/UC - ST")]) + r" \\")
    return RG.table(f"twCRPS{lvl}: Stage 5 versus the best twCRPS{lvl} model of each class "
                    "(rolling one-step-ahead, out of sample).", f"tab:stage5_vs_classes_{lvl}",
                    "lrrrrrr", hdr, rows, note, sep="4pt")


def main():
    REP.mkdir(parents=True, exist_ok=True)
    intro = (r"\section*{Stage 5}"
             r"Stage 5 lets both $\varphi_{t|t-1}$ and $\xi_{t|t-1}$ follow long--short dynamics (with the "
             r"seasonal component when the base model has one), on each location's best Stage 3/4/4S "
             r"configuration (gradient tier, then out-of-sample CRPS, then RMSE). Occurrence dynamics: "
             r"$\pi_{t|t-1}=\operatorname{logistic}(\lambda_0+\lambda_1\varphi_{t|t-1})$, estimated jointly, "
             r"with the mixed-density score; starting values from the time-invariant ZA-GB2 maximum-"
             r"likelihood estimates. ``Previous'' is the 2026-09 Stage 5 fit (autoregressive logistic "
             r"occurrence dynamics), whose base configuration may differ.")
    doc = "\n\n".join([
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}",
        r"\usepackage[margin=2cm]{geometry}",
        r"\usepackage{amsmath,amssymb,booktabs,float}",
        r"\usepackage{parskip}\emergencystretch=3em\sloppy",
        r"\begin{document}",
        r"\title{Stage 5 -- $\varphi$-linked occurrence dynamics}\author{}\date{1 October 2026}\maketitle",
        intro, P.t_stage5(), details_table(),
        comparison_table("tw95", "95"), comparison_table("tw98", "98"),
        r"\end{document}"])
    (REP / "report.tex").write_text(doc, encoding="utf-8")
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], cwd=REP,
                       capture_output=True, text=True)
    log = (REP / "report.log").read_text(errors="ignore")
    print("report.pdf:", "OK" if (REP / "report.pdf").exists() and "Fatal error" not in log else "FAILED")


if __name__ == "__main__":
    main()
