"""
Holistic diagnostic summary and actionable recommendations, by stage.

=============================================================================
OVERVIEW
=============================================================================
A reporting-only deliverable (docs/REPORTING.md: reports never trigger
optimization): synthesizes boundary_and_zero_ci_analysis.pdf's findings
(near-boundary proximity, GAS/SD zero-CI patterns, standard-error
reliability) together with the pipeline's own saved out-of-sample
comparison tables (reports/multi_location/tables/global_*.csv) into a single
set of "so what do I do about it" recommendations, one section per stage,
plus a prioritized action list. Where boundary_and_zero_ci_analysis.pdf
answers "what is true of the fitted parameters", this document answers
"which of those facts require action, and which don't".

=============================================================================
WHY THIS REUSES generate_parameter_analysis_report.py INSTEAD OF
REIMPLEMENTING
=============================================================================
"Which parameters are near their soft-penalty boundary" and "which GAS/SD
coefficients have a CI including zero, and how often across locations" are
already solved by generate_parameter_analysis_report.py's _load_all_tables/
_load_boundary_table (which themselves reuse generate_parameter_tables_
report.py's constraint classifier and generate_report_extended.py's
artifact loaders). Per CLAUDE.md ("refactor rather than duplicate"), this
script imports those functions directly rather than re-deriving them.

The one genuinely new input here is the pipeline's own out-of-sample
model-comparison tables already saved to disk by generate_report_extended.py
(reports/multi_location/tables/global_comparison.csv, global_s4.csv,
global_xicompare.csv) -- read directly, not recomputed, per docs/
REPORTING.md ("reports should be generated entirely from saved artifacts").

=============================================================================
CENTRAL FINDING THIS DOCUMENT IS BUILT AROUND
=============================================================================
Cross-referencing boundary_and_zero_ci_analysis.pdf's Stage-3 finding
(the phi+xi Harvey fit converged cleanly, in the usual BFGS sense, at only
1 of 6 locations -- Belo Horizonte) against global_comparison.csv's S1/S2/S3
OOS CRPS columns shows that Stage 3 delivers its largest CRPS improvement
over Stage 1 at that same one location (-4.5%); three of the five unstable
locations (Garanhuns, Manaus, and worse at Cruzeiro/Salvador) show a
negligible-to-negative OOS payoff (-0.3% to +7.2%), consistent with the
instability costing real predictive performance -- but Darwin Airport (the
single largest gradient norm of all six) is a partial exception, with a
CRPS improvement (-3.2%) close to Belo Horizonte's. The correspondence is
therefore directional, not perfectly clean, and is presented as a
hypothesis worth verifying after the fixes below (not a settled causal
claim), and is the basis for this document's "fix the optimization, then
rerun" recommendation (rather than either "rerun blindly" or "abandon
Harvey entirely").

=============================================================================
USAGE
=============================================================================
    python generate_pipeline_recommendations_report.py [--no-compile]

Output:
    reports/multi_location/parameter_tables/pipeline_recommendations.tex
    reports/multi_location/parameter_tables/pipeline_recommendations.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

from generate_parameter_analysis_report import (
    _load_all_tables, _load_boundary_table, LOCATIONS,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "reports" / "multi_location" / "parameter_tables"
GLOBAL_TABLES_DIR = ROOT / "reports" / "multi_location" / "tables"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-compile", action="store_true", help="Skip pdflatex compilation")
    args = parser.parse_args()
    build_document(no_compile=args.no_compile)


def build_document(no_compile: bool = False) -> Path:
    from generate_report_extended import compile_latex

    big = _load_all_tables()
    bnd = _load_boundary_table()
    n_loc = len(LOCATIONS)

    def zero_frac(stage: int, branch: str, pname: str) -> Tuple[int, int]:
        sub = big[(big["stage"] == stage) & (big["branch"] == branch) & (big["parameter"] == pname)]
        return int(sub["ci_includes_zero"].sum()), int(len(sub))

    def near_count(stage: int, branch: str, tag: str) -> int:
        sub = bnd[(bnd["stage"] == stage) & (bnd["branch"] == branch)]
        return int(sub["near_limit"].apply(lambda lst: tag in lst).sum())

    def grad(stage: int, branch: str, display: str) -> float:
        sub = bnd[(bnd["stage"] == stage) & (bnd["branch"] == branch) & (bnd["display"] == display)]
        return float(sub["grad_norm_inf"].iloc[0]) if len(sub) else float("nan")

    def niter(stage: int, branch: str, display: str) -> int:
        sub = bnd[(bnd["stage"] == stage) & (bnd["branch"] == branch) & (bnd["display"] == display)]
        return int(sub["n_iter"].iloc[0]) if len(sub) else -1

    # ---- global OOS comparison tables (already on disk) --------------------
    gcomp = pd.read_csv(GLOBAL_TABLES_DIR / "global_comparison.csv")
    gcomp["S3_vs_S1_pct"] = (gcomp["S3 CRPS"] - gcomp["S1 CRPS"]) / gcomp["S1 CRPS"] * 100
    gxi = pd.read_csv(GLOBAL_TABLES_DIR / "global_xicompare.csv")
    gs4 = pd.read_csv(GLOBAL_TABLES_DIR / "global_s4.csv")

    stable_locs = ["Belo Horizonte"]  # the only phi+xi Harvey fit that converged cleanly
    unstable_locs = [d for d in LOCATIONS_DISPLAY_ORDER(gcomp) if d not in stable_locs]

    s3_rows = "\n".join(
        rf"{_tex(r['Station'])} & {r['S1 CRPS']:.3f} & {r['S3 CRPS']:.3f} & "
        rf"{r['S3_vs_S1_pct']:+.1f}\% & {int(grad(3,'phi_xi',r['Station']))}"
        r" \\"
        for _, r in gcomp.iterrows()
        # Darwin's S2 diverged (3.3e14) and its own gradient story is mixed
        # branch-to-branch; still shown for completeness with a footnote.
    )

    z_omega_y365_1x = zero_frac(1, "phi_xi", "omega_y_365")
    z_omega_y366_1x = zero_frac(1, "phi_xi", "omega_y_366")
    z_aext = zero_frac(4, "none", "A_ext_xi")
    z_bxi1 = zero_frac(4, "none", "B_xi_1")
    manaus_niter = niter(2, "phi_only", "Manaus")
    manaus_grad = grad(2, "phi_only", "Manaus")
    bh_grad3 = grad(3, "phi_xi", "Belo Horizonte")
    bh_niter3 = niter(3, "phi_xi", "Belo Horizonte")

    body = []

    # =========================================================================
    body.append(r"""
\section*{How to read the two prior documents together}
Two facts must be read jointly, not separately, to decide whether a
parameter's estimate is trustworthy:
\begin{enumerate}
\item \textbf{Is it near a soft-penalty boundary?} (\texttt{parameter\_
tables.pdf}'s Constraint columns / \texttt{boundary\_and\_zero\_ci\_
analysis.pdf}'s boundary-proximity sections.)
\item \textbf{Did the optimizer actually converge there?} (that model's
residual gradient norm and BFGS iteration count, in \texttt{metadata.json}.)
\end{enumerate}
\begin{itemize}
\item \textbf{Near boundary + small/moderate gradient norm + a normal-
looking standard error:} no concern. This describes almost every Stage 1/2
persistence result and Stage 4's $B_{\xi,1}$ -- it is a genuine, stably-
estimated near-unit-root process, not an artifact. \textbf{Do not} try to
``fix'' this by tightening the penalty further; a highly persistent scale/
shape process is a legitimate finding for daily rainfall.
\item \textbf{Near boundary + a huge residual gradient norm ($\gg1$) and/or
a standard error sitting exactly at BFGS's identity-Hessian default (1.0):}
the optimizer did not actually resolve that parameter. Treat the point
estimate as unreliable and the CI as meaningless until refit. This is
Stage 3's problem at 5 of 6 locations -- see below.
\item \textbf{A GAS/SD coefficient's CI includes zero consistently across
most locations, and it is not one of Stage 3's dead \texttt{f0/L0/S0}
slots:} a genuine, reproducible candidate for simplifying future model
specifications -- not evidence of an optimizer problem.
\end{itemize}
""")

    # =========================================================================
    body.append(rf"""
\section{{Stage 1 (Baseline GAS) -- tips}}
\begin{{itemize}}
\item \textbf{{No action needed on persistence.}} $\sum_l|B_{{\phi,l}}|$
(and $\sum_l|B_{{\xi,l}}|$, \texttt{{rho}}) sit near their boundary at
essentially every location with small-to-moderate gradient norms and
ordinary-looking standard errors throughout -- this is Rule 1 above.
\item \textbf{{Prune the leap-day occurrence terms in future specs.}}
\texttt{{omega\_y\_365}} ({z_omega_y365_1x[0]}/{n_loc} zero) and
\texttt{{omega\_y\_366}} ({z_omega_y366_1x[0]}/{n_loc} zero) are the
cleanest, most reproducible zero-CI pattern in the whole document, and they
each represent only 1-2 calendar days/year of information -- a reasonable
default going forward is to drop them from the AR-logistic occurrence
model's seasonal lag set (\texttt{{pi\_dynamics/ar\_logistic.py}}), keeping
\texttt{{omega\_y\_1}} (robustly significant everywhere).
\item \textbf{{The $\xi$ extension earns very little CRPS on its own.}}
\texttt{{global\_xicompare.csv}} shows dynamic $\xi$ improves OOS CRPS over
$\phi$-only by only $-0.4\%$ to $+1.8\%$ across all six locations -- most of
the phi+xi branch's individually-insignificant $A_{{\xi,l}}$/$B_{{\xi,l}}$
coefficients are consistent with this: $\xi_t$'s dynamics add little
practical value even where the branch as a whole is selected as the winner
(selection is by CRPS, and small improvements still count, but don't expect
this branch to look ``cleaner'' than phi-only in a coefficient-by-
coefficient read).
\end{{itemize}}
""")

    # =========================================================================
    body.append(rf"""
\section{{Stage 2 (Weather Covariates) -- tips}}
\begin{{itemize}}
\item \textbf{{No action needed on persistence}} (same as Stage 1 -- stable,
low-gradient boundary-pinning).
\item \textbf{{Simplify the seasonal $B$-lag structure.}}
\texttt{{B\_phi\_2}}, \texttt{{B\_phi\_365}}, \texttt{{B\_phi\_366}},
\texttt{{B\_xi\_1}}, and \texttt{{B\_xi\_366}} have a CI including zero at
\textbf{{all 6/6}} locations in the $\phi+\xi$ branch. Combined with the
leap-day $\omega_y$ terms above, this is a defensible, cross-validated
argument for a leaner default seasonal-lag set in future runs (fewer $B$
coefficients to estimate, without giving up the lags that do carry signal:
lag 1, 3, 364, 367 tend to remain more often significant).
\item \textbf{{Fix an isolated data-quality problem, cheaply.}} Manaus's
accepted $\phi$-only Stage-2 model performed
\textbf{{{manaus_niter} BFGS iterations}} (residual gradient norm
$\approx${manaus_grad:.0f}) -- every standard error in that one table
defaults to 1.0 and should not be read as ``insignificant''. This needs
only a rerun of that single model (ideally from a perturbed starting point,
since the optimizer never moved at all), not a pipeline-wide fix.
\end{{itemize}}
""")

    # =========================================================================
    body.append(rf"""
\section{{Stage 3 (Harvey Long-Short) -- detailed diagnosis and rerun
recommendation}}

\subsection*{{What the evidence shows, put together}}
Three independent signals, read jointly, tell a single consistent story:
\begin{{enumerate}}
\item \textbf{{Optimizer instability.}} Every $\phi+\xi$ Harvey fit except
Belo Horizonte's terminated after 2-25 BFGS iterations with a residual
gradient norm of $10^4$-$10^6$ (Belo Horizonte: {bh_niter3} iterations,
gradient norm $\approx${bh_grad3:.3f}) -- see \texttt{{boundary\_and\_
zero\_ci\_analysis.pdf}} \S 3.
\item \textbf{{A shared root symptom: $B_L$ pinned at its boundary.}}
$B_{{L,\phi}}$ sits within 10\% of its 0.995 ceiling at all 6 locations,
both branches; $B_{{L,\xi}}$ and the $B_S$-vs-$B_L$ relative check follow
at 5/6 (all except Belo Horizonte). The persistence surface is essentially
flat near this boundary, which is exactly the geometry that produces a
large, hard-to-reduce gradient under unbounded BFGS with a quadratic
penalty wall.
\item \textbf{{The OOS payoff tracks the optimizer's success, not just the
model specification.}}
\end{{enumerate}}

\begin{{center}}
\begin{{tabular}}{{lrrrr}}
\hline
Location & S1 CRPS & S3 CRPS & S3 vs S1 & Stage-3 phi+xi grad.\ norm \\
\hline
{s3_rows}
\hline
\end{{tabular}}
\end{{center}}
\emph{{(Darwin's Stage-2 CRPS diverged separately (ROADMAP.md, pre-fix) and
is excluded from that column here; its Stage-3 gradient norm reflects the
phi+xi branch, whose overall pipeline winner ended up being Stage 1 --
see \texttt{{global\_comparison.csv}}.)}}

Belo Horizonte -- the one location with a clean optimizer run -- is also
the location where Stage 3 delivers the largest CRPS improvement over
Stage 1 ($-4.5\%$). Among the five unstable locations, three (Garanhuns,
Manaus, and most clearly Cruzeiro do Sul and Salvador) show a negligible-
to-negative OOS payoff ($-0.3\%$ to $+7.2\%$), consistent with the
instability costing real predictive performance. \textbf{{Darwin Airport is
a partial exception}}: despite the largest gradient norm of all six
locations ($\approx1.19\times10^6$), its CRPS improvement ($-3.2\%$) is
close to Belo Horizonte's -- so the correspondence between optimizer
stability and OOS payoff is directional, not a clean one-to-one
relationship, and should be described that way rather than overstated.
This is still reasonable circumstantial evidence that \textbf{{the
optimization instability is not merely a cosmetic standard-error problem
and is plausibly costing predictive performance at several (not
necessarily all) of the affected locations}}, because a fit stuck near a
flat ridge is not actually at the likelihood optimum, regardless of what
\texttt{{scipy.success}} or the post-fit validity label says -- but Darwin's
result is a reason to treat this as a hypothesis to verify after the fixes
below, not a settled conclusion.

\subsection*{{Most likely root cause}}
The $\phi+\xi$ branch jointly estimates \emph{{two}} separate near-unit-root
long-run components, $B_{{L,\phi}}$ and $B_{{L,\xi}}$, from score signals
that are correlated (both are derived from the same GB2 log-density and the
same observations $y_t$). Jointly identifying two near-integrated latent
processes from correlated innovations is a substantially harder estimation
problem than identifying one -- consistent with the phi-only branch (a
single Harvey block) being far more stable (only Darwin's phi-only fit is
unstable, vs. 5/6 for phi+xi). The weakly-identified ENSO dummy
covariates (\texttt{{gamma\_L\_*\_el\_nino\_t}}/\texttt{{la\_nina\_t}},
zero-CI at 5/5 of the locations where they appear -- \texttt{{boundary\_
and\_zero\_ci\_analysis.pdf}} \S 3) plausibly compound this by adding
further weakly-informative directions to an already hard joint problem,
rather than being the primary cause on their own (Darwin's phi-only branch
is unstable with \emph{{no}} ENSO term at all).

\subsection*{{Should you rerun Stage 3? Yes -- but fix these first, in
order}}
\begin{{enumerate}}
\item \textbf{{Fix the \texttt{{L0/S0}} dead-parameter bug in
\texttt{{models/harvey\_gas.py}}}} (\texttt{{boundary\_and\_zero\_ci\_
analysis.pdf}} \S 3): \texttt{{L\_arr[max\_lag]}}/\texttt{{S\_arr[max\_lag]}}
should be initialized from \texttt{{L0\_j}}/\texttt{{S0\_j}}, not
\texttt{{L\_arr[0]}}/\texttt{{S\_arr[0]}} (which the recursion loop never
reads, since it starts at \texttt{{t=max\_lag}}). Cheap, mechanical, and
should happen regardless of anything else below -- it currently wastes
6 free parameters per phi+xi fit for no purpose.
\item \textbf{{Reparameterize the persistence restriction.}} Replace the
penalized $0<B_S<B_L<1$ region with an unconstrained mapping, e.g.
$B_L=\mathrm{{sigmoid}}(u)$, $B_S=B_L\cdot\mathrm{{sigmoid}}(v)$, so the
restriction is satisfied \emph{{by construction}} rather than by a
quadratic penalty wall that BFGS must fight against near the boundary --
the standard fix for boundary-ridge optimization problems, and consistent
with docs/OPTIMIZATION.md \S 6 (``if a reparameterization is used, document
the mapping clearly'').
\item \textbf{{Warm-start each block's $B_L$ from that series' own,
already-well-identified Stage 1/2 persistence}} ($\sum_l|B_{{\phi,l}}|$ /
$\sum_l|B_{{\xi,l}}|$ from the selected Stage-2 winner) rather than a
generic 0.95 default shared by both blocks -- per docs/OPTIMIZATION.md
\S 7's own prescribed hierarchy (``initialize Harvey long-short models from
the selected standard GAS/weather model when possible''), which does not
currently appear to differentiate $\phi$'s history from $\xi$'s.
\item \textbf{{Add a genuine multi-start / perturbed-retry step}} for
phi+xi Harvey fits specifically (docs/OPTIMIZATION.md \S 13, point 5).
Currently only one BFGS run plus a single polish restart is used, and at
the unstable locations the polish step barely moved the objective
(polish\_improvement $\sim 10^{{-4}}$ to $10^{{-8}}$ despite a residual
gradient in the hundreds of thousands) -- that fit is stuck, not slowly
converging, and a same-point restart will not escape it.
\item \textbf{{Add a third post-fit validity class}} for
``\texttt{{scipy.success=False}} and gradient norm far above what
docs/OPTIMIZATION.md \S 11.2 calls `mildly elevated' '' -- currently these
land in the same \texttt{{valid\_with\_warning}} bucket as Belo Horizonte's
genuinely clean fit, which is why this problem was not visible from the
validity label alone.
\end{{enumerate}}
After these fixes, rerun the $\phi+\xi$ Harvey stage at Cruzeiro do Sul,
Darwin Airport, Garanhuns, Manaus, and Salvador, and re-check both the
residual gradient norm and the OOS CRPS gap vs.\ Stage 1/2 -- that gap is
the real test of whether Harvey's long-short decomposition earns its added
complexity anywhere beyond Belo Horizonte.

\subsection*{{What NOT to expect from a rerun}}
The ENSO long-component coefficients' zero-inclusive CIs are a separate,
already-credible finding, not a byproduct of the optimizer instability --
don't expect them to become significant after a clean refit. It is
plausible the monthly El Ni\~no/La Ni\~na dummy encoding simply does not
carry enough daily-resolution signal to move a persistent long-run
component at most of these locations; the honest test after refitting is
whether OOS CRPS with vs.\ without the ENSO term still diverges once the
optimizer itself is no longer the confound.
""")

    # =========================================================================
    gs4_rows = "\n".join(
        rf"{_tex(r['Station'])} & {_tex(r['q95 accepted'])} & {_tex(r['q98 accepted'])} \\"
        for _, r in gs4.iterrows()
    )
    body.append(rf"""
\section{{Stage 4 (Tail-Sensitive xi Regime) -- tips}}
\begin{{itemize}}
\item \textbf{{No rerun needed.}} Gradients are small (0.02-0.31) and
iteration counts are ordinary at all 6 locations -- Stage 4's own boundary
result ($B_{{\xi,1}}$ near 0.98 at 4/6 locations) is Rule 1: stable,
trustworthy, not an optimizer artifact.
\item \textbf{{Treat the regime extension as tested and rejected for this
framework, at these locations.}} \texttt{{A\_ext\_xi}} has a CI including
zero at \textbf{{{z_aext[0]}/{z_aext[1]} (all six)}} locations, and this
matches the out-of-sample rejection already on record:
\end{{itemize}}
\begin{{center}}
\begin{{tabular}}{{lll}}
\hline
Location & q95 accepted (vs.\ twCRPS base) & q98 accepted \\
\hline
{gs4_rows}
\hline
\end{{tabular}}
\end{{center}}
Only Cruzeiro do Sul shows a real improvement (+12-14\%); every other
location is negative, several severely so (Garanhuns: $-118\%$/$-174\%$).
Combined with $B_{{\xi,1}}$ itself never being insignificant
({z_bxi1[0]}/{z_bxi1[1]} zero) -- the baseline persistence is fine, only the
\emph{{extra}} regime term is not earning its keep -- there is no evidence
pointing to an implementation problem here, only a substantive one: this
particular threshold-triggered score-response design does not add value at
most of these locations. Keep Stage 4 implemented and disabled-by-default
(ROADMAP.md), but a rerun would not be expected to change this conclusion
without a materially different design (e.g. a different threshold
definition, or letting the regime term affect the long/short Harvey split
rather than a single GAS(1,1) block).
""")

    # =========================================================================
    body.append(r"""
\section{Pipeline-wide caveat: extreme-quantile coverage tests}
The Kupiec and Christoffersen backtests reported in the main
\texttt{report.pdf} (``Coverage tests'' subsections) reject at $p<0.001$
for the final selected model at every location checked -- including
locations whose final model is Stage 1, not Stage 3. This is therefore
\textbf{not evidence specific to Stage 3 or the Harvey decomposition}; it
is a pipeline-wide characteristic, most plausibly reflecting the low power
of exceedance-count backtests at 95th/99th-quantile levels over a
$\sim$730-day OOS window (very few exceedance events to test against). Per
your own review, the PIT histograms and quantile-residual ACFs -- which
test bulk calibration and residual autocorrelation, not extreme-quantile
exceedance counts -- look acceptable across models; this caveat concerns a
stricter, separate test, not the overall calibration story. Worth a
methodological note in the thesis (e.g., report raw exceedance counts
alongside the p-values, since $p<0.001$ reads very differently for
5 exceedances in 37 expected vs.\ 0 in 37), but not an urgent fix.
""")

    # =========================================================================
    body.append(r"""
\section{Prioritized action list}
\begin{enumerate}
\item Fix the Harvey \texttt{L0/S0} initialization bug (\texttt{models/
harvey\_gas.py}) -- cheap, mechanical, improves the honesty of every future
Stage-3 parameter table regardless of anything else here.
\item Rerun Manaus's Stage-2 $\phi$-only model from a perturbed starting
point (isolated, zero-iteration fit -- unrelated to the Stage-3 issue).
\item Reparameterize Stage 3's $0<B_S<B_L<1$ restriction, warm-start each
block's $B_L$ from its own Stage 1/2 persistence, add a genuine multi-start
retry, then rerun the $\phi+\xi$ Harvey stage at Cruzeiro do Sul, Darwin
Airport, Garanhuns, Manaus, and Salvador; re-check gradient norm and OOS
CRPS vs.\ Stage 1/2 at each.
\item Adopt a leaner default seasonal-lag set for future Stage 1/2 runs:
drop \texttt{omega\_y\_365}/\texttt{omega\_y\_366} from the occurrence
model, and consider dropping \texttt{B\_phi\_2}/\texttt{B\_phi\_365}/
\texttt{B\_phi\_366}/\texttt{B\_xi\_1}/\texttt{B\_xi\_366} from the GAS
lag set.
\item Treat Stage 4 as a concluded, substantively-negative result for this
framework -- no rerun needed unless testing a materially different regime
design.
\item Add exceedance counts (not just p-values) to the Kupiec/Christoffersen
coverage-test tables in the main report, and note their pipeline-wide,
Stage-3-independent nature if discussed in the thesis text.
\end{enumerate}
""")

    tex_path = OUT_DIR / "pipeline_recommendations.tex"
    tex_path.write_text(_PREAMBLE + "\n".join(body) + _CLOSING, encoding="utf-8")
    print(f"Wrote {tex_path}")

    if not no_compile:
        ok = compile_latex(tex_path)
        print("PDF compilation " + ("succeeded." if ok else "failed -- see log above."))

    return tex_path


def LOCATIONS_DISPLAY_ORDER(gcomp: pd.DataFrame):
    return list(gcomp["Station"])


def _tex(s: str) -> str:
    return (str(s).replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
            .replace("#", r"\#").replace("_", r"\_").replace("{", r"\{")
            .replace("}", r"\}"))


_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{float}
\usepackage{caption}
\usepackage{hyperref}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
\emergencystretch=3em
\sloppy

\begin{document}

\title{ZA-GAS Multi-Location Pipeline\\
\large Holistic Diagnostic Summary and Recommendations}
\author{Generated by generate\_pipeline\_recommendations\_report.py}
\date{}
\maketitle

\noindent This document turns \texttt{boundary\_and\_zero\_ci\_analysis.pdf}'s
findings into actionable guidance, cross-checked against the pipeline's own
out-of-sample comparison tables (\texttt{reports/multi\_location/tables/
global\_*.csv}) and the main report's coverage-test results. No model is
re-estimated to produce this document.

"""

_CLOSING = r"""
\end{document}
"""


if __name__ == "__main__":
    main()
