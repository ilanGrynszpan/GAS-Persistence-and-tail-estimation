"""
Cross-location parameter/standard-error/confidence-interval tables.

=============================================================================
OVERVIEW
=============================================================================
A reporting-only deliverable (docs/REPORTING.md: reports never trigger
optimization): for each of the six completed locations and each of that
location's 7 stage-winner models -- Stage 1 (phi+xi branch, phi-only
branch), Stage 2 (phi+xi, phi-only), Stage 3 (phi+xi, phi-only), and
Stage 4 -- builds one LaTeX table listing every optimized parameter with
its point estimate, standard error, and 95% Wald confidence interval.

These are exactly the same 7 (stage, branch) winners already shown side by
side in the Stage 1-4 diagnostic mosaics (diagnostics/multilocation_mosaics.py,
multilocation_diagnostic_mosaics.ipynb) -- "each of the stage models" here
means the same 7 models, not every specification ever fitted (42 tables
total; docs/REPORTING.md Sec 9: "avoid excessively large tables").

=============================================================================
WHY THIS REUSES generate_report_extended.py / multilocation_mosaics.py
INSTEAD OF REIMPLEMENTING
=============================================================================
"Which saved model_id is this location's Stage-X winner" is already solved
by diagnostics/multilocation_mosaics.py's resolve_stage_winner(), and
"load estimated_parameters.csv merged with standard_errors.csv" is already
solved by generate_report_extended.py's _load_params_with_se(). Per
CLAUDE.md ("determine whether functionality already exists before creating
new modules", "refactor rather than duplicate"), this script imports both
rather than re-deriving stage-winner resolution or artifact loading. The
only new logic here is the confidence-interval arithmetic and the LaTeX
table layout (value / std. error / CI lower / CI upper columns), which
does not exist elsewhere in the repository.

=============================================================================
MATHEMATICAL NOTE -- confidence interval construction
=============================================================================
Every standard error on disk is the quasi-Newton (BFGS) inverse-Hessian
approximation described in docs/OPTIMIZATION.md Sec 9 ("for BFGS, use
result.hess_inv ... compute approximate standard errors ... clearly label
these standard errors as approximate") -- see each model's saved
standard_errors.csv, column "se_quality" (always "approximate" for these
artifacts). Given a parameter estimate theta_hat and its approximate
standard error se_hat, the standard asymptotic (Wald) 95% confidence
interval is

    [theta_hat - z * se_hat,  theta_hat + z * se_hat],   z = Phi^{-1}(0.975)

i.e. the analytical normal-quantile multiplier (~1.95996), not the rounded
"1.96" shortcut, computed once via scipy.stats.norm.ppf (CLAUDE.md:
"prefer analytical methods over numerical approximations"). No alternative
interval construction (bootstrap, profile likelihood, ...) is implemented,
per CLAUDE.md ("do not implement alternative mathematical formulations
unless explicitly requested").

=============================================================================
CODE WALKTHROUGH
=============================================================================
main()
    |
    v
For each of the 6 completed locations (diagnostics.multilocation_mosaics.
LOCATIONS, same order as the main multi-location report and the
diagnostic mosaics):
    load that location's stage_winners.json
        |
        v
    For each of the 7 (stage, branch) combinations in STAGE_MODEL_SEQUENCE:
        resolve_stage_winner() -> model_id
            |
            v
        _load_params_with_se(run_dir, stage_dir, model_id)
            -> DataFrame[parameter, value, std_error, se_quality]
            |
            v
        _add_confidence_interval() -> + ci_lower, ci_upper columns
            |
            v
        _params_ci_table_tex() -> one LaTeX \\begin{table} block
    |
    v
Assemble one \\chapter per location (7 tables each) into a single
standalone LaTeX document (own preamble, own \\begin{document}) and
compile it with pdflatex (generate_report_extended.compile_latex, reused
rather than re-implemented).

Every table is also saved independently as CSV (docs/REPORTING.md Sec 19:
"every LaTeX table should also exist as CSV"), under
reports/multi_location/parameter_tables/tables/.

=============================================================================
USAGE
=============================================================================
    python generate_parameter_tables_report.py [--no-compile]

Output:
    reports/multi_location/parameter_tables/parameter_tables.tex
    reports/multi_location/parameter_tables/parameter_tables.pdf
    reports/multi_location/parameter_tables/tables/*.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from data.station_loader import STATION_REGISTRY
from diagnostics.multilocation_mosaics import (
    LOCATIONS, ARTIFACTS_DIR, STAGE_LABELS, BRANCH_LABELS,
    resolve_stage_winner,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "reports" / "multi_location" / "parameter_tables"
TABLES_DIR = OUT_DIR / "tables"

# 95% Wald multiplier, computed analytically rather than hard-coded as the
# rounded "1.96" -- see module docstring's "MATHEMATICAL NOTE".
_Z95 = float(norm.ppf(0.975))

# The 7 (stage, branch) combinations shown for every location, in the same
# order as diagnostics/multilocation_mosaics.py's STAGE_LABELS /
# BRANCH_LABELS and multilocation_diagnostic_mosaics.ipynb's mosaics, so
# this document's ordering matches what a reader already saw there.
STAGE_MODEL_SEQUENCE: List[Tuple[int, Optional[str]]] = [
    (1, "phi_xi"), (1, "phi_only"),
    (2, "phi_xi"), (2, "phi_only"),
    (3, "phi_xi"), (3, "phi_only"),
    (4, None),
]


def _section_label(stage: int, branch: Optional[str]) -> str:
    if branch is None:
        return STAGE_LABELS[stage]
    return f"{STAGE_LABELS[stage]} ({BRANCH_LABELS[branch]})"


# Branch -> "how many / which distribution parameters are time-varying",
# phrased the way the user asked for section/caption titles (2026-08-21):
# phi-only = one TV parameter ("location"); phi+xi = two ("location and
# shape"). GB2LogLink's phi enters as a log-link scale, but the user's
# naming convention calls it "location" here, so this label is deliberately
# report-only vocabulary -- it does not rename the parameter anywhere else
# (MODELS.md/parameter names are unaffected).
_TV_DESC: Dict[Optional[str], str] = {
    "phi_only": "time-varying location",
    "phi_xi":   "time-varying location and shape",
    None:       "time-varying shape (tail-sensitive regime)",  # Stage 4, no branch split
}


def _section_title(stage: int, branch: Optional[str], display: str) -> str:
    """'Stage N - time-varying <...> - <Location>', per the user's requested
    naming convention -- used for both the \\section{} heading and the table
    caption so a reader can identify a table without cross-referencing the
    enclosing chapter."""
    return f"Stage {stage} - {_TV_DESC[branch]} - {display}"


# =============================================================================
# Optimization constraints (lower/upper bound column)
# =============================================================================
# CLAUDE.md/docs/OPTIMIZATION.md Sec 2-3: every model here is fit by
# *unbounded* BFGS with smooth penalty terms, not scipy bounds -- "hard
# bounds should be avoided unless mathematically unavoidable". So there is
# no literal per-parameter box constraint object anywhere in models/ to read
# off; the "constraint" a parameter is actually subject to is the soft
# penalty threshold in that model's own _run_filter (models/gas_filter.py,
# models/za_gas_model.py, models/cov_gas_model.py, models/harvey_gas.py,
# models/regime_gas.py). This table reports those thresholds as lower/upper
# bounds -- the values a parameter cannot cross without the objective
# picking up a quadratic penalty term -- classified by parameter-name
# pattern rather than by re-reading each model's penalty code at report
# time (reports must not re-run or re-inspect live model objects per
# docs/REPORTING.md; the thresholds themselves are static constants copied
# here once, by hand, from the four _run_filter implementations).
#
# Three constraint families appear:
#   (a) individual bound        -- |A_L_j| <= 2.0, |B_xi_1| <= 0.98, etc.
#   (b) GROUP bound (lag sums)  -- Stage 1/2's standard GAS blocks penalize
#       sum_l |A_{j,l}| <= 2.0 and sum_l |B_{j,l}| <= 0.98 ACROSS the whole
#       lag set L, not per lag. The number shown for each individual A_j_l /
#       B_j_l row is therefore the loosest bound implied by the group
#       constraint (|A_j,l| <= sum_l|A_j,l| <= 2.0), not a per-lag bound --
#       see the Introduction for the precise group formula.
#   (c) joint-norm bound         -- covariate coefficient vectors Gamma_j
#       are penalized on ||Gamma_j||_2 <= 20, so each individual
#       coefficient's implied bound is +/-20 (since |Gamma_j,k| <=
#       ||Gamma_j||_2), again not a per-coefficient bound.
#   (d) relative bound            -- Harvey's B_S_j is penalized against
#       B_L_j itself (0 < B_S_j < B_L_j), not a fixed constant; the upper
#       bound shown (0.995) is the outer bound implied by B_L_j's own cap,
#       not the true (tighter, fit-specific) limit -- see the Introduction.
#
# Parameters with no penalty term at all (omega_j, f0_j, L0_j, S0_j, omega0,
# omega_y_l) are unconstrained under BFGS and rendered "NA".
_A_LIMIT      = 2.0    # LAM_SCORE penalty threshold (score-response coefficients)
_B_LIMIT      = 0.98   # LAM_PERSIST threshold, standard GAS / regime B (persistence)
_RHO_LIMIT    = 0.98   # LAM_PERSIST threshold, pi AR(1) coefficient
_STATIC_LIMIT = 20.0   # LAM_STATIC threshold, static GB2 shape params (gamma, zeta, xi)
_COV_LIMIT    = 20.0   # LAM_COV threshold, covariate-vector L2 norm (implied per-coef bound)
_BL_UPPER     = 0.995  # Harvey B_L upper threshold (LAM_LONG)

# (pattern, lower, upper) -- checked in order, first match wins. Patterns are
# checked with re.fullmatch so e.g. "A_L_phi" cannot accidentally match the
# generic "A_(phi|xi)_<lag>" rule.
_CONSTRAINT_RULES: List[Tuple[str, Optional[float], Optional[float]]] = [
    # Static GB2 shape parameters (gamma, zeta always; xi when phi-only)
    (r"gamma", -_STATIC_LIMIT, _STATIC_LIMIT),
    (r"zeta",  -_STATIC_LIMIT, _STATIC_LIMIT),
    (r"xi",    -_STATIC_LIMIT, _STATIC_LIMIT),
    # Unconstrained intercepts / initial states (no penalty term anywhere)
    (r"omega_(phi|xi)", None, None),
    (r"f0_(phi|xi)",    None, None),
    (r"L0_(phi|xi)",    None, None),
    (r"S0_(phi|xi)",    None, None),
    (r"omega0",         None, None),
    (r"omega_y_\d+",    None, None),
    # Occurrence (pi) AR(1) persistence
    (r"rho", -_RHO_LIMIT, _RHO_LIMIT),
    # Stage 4 regime extension
    (r"A_ext_(phi|xi)", -_A_LIMIT, _A_LIMIT),
    # Harvey long component
    (r"A_L_(phi|xi)",          -_A_LIMIT, _A_LIMIT),
    (r"B_L_(phi|xi)",           0.0,       _BL_UPPER),
    (r"gamma_L_(phi|xi)_.+",   -_COV_LIMIT, _COV_LIMIT),
    # Harvey short component
    (r"A_S_(phi|xi)",          -_A_LIMIT, _A_LIMIT),
    (r"B_S_(phi|xi)",           0.0,       _BL_UPPER),  # outer bound; also < that fit's B_L
    (r"gamma_S_(phi|xi)_.+",   -_COV_LIMIT, _COV_LIMIT),
    # Standard GAS score/persistence lag coefficients (group bound, see above)
    (r"A_(phi|xi)_\d+", -_A_LIMIT, _A_LIMIT),
    (r"B_(phi|xi)_\d+", -_B_LIMIT, _B_LIMIT),
    # Weather/ENSO covariate coefficients (Stage 2, Stage 4)
    (r"gamma_(phi|xi)_.+", -_COV_LIMIT, _COV_LIMIT),
]
_CONSTRAINT_RULES_COMPILED = [
    (re.compile(pat + r"$"), lo, hi) for pat, lo, hi in _CONSTRAINT_RULES
]


def _classify_constraint(pname: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (lower, upper) soft-penalty bound for a parameter name, or
    (None, None) if that parameter carries no penalty term (unconstrained
    under BFGS). See the module-level comment above for the three families
    of non-literal ("group"/"joint-norm"/"relative") bounds this collapses
    to a single implied per-parameter number."""
    for regex, lo, hi in _CONSTRAINT_RULES_COMPILED:
        if regex.fullmatch(pname):
            return lo, hi
    # Should not happen for any parameter name produced by this codebase's
    # models; surfaced rather than silently hidden (docs/REPORTING.md Sec
    # 26.9) so an unrecognised name is visible instead of mis-classified.
    return None, None


def _add_constraints(params_df: pd.DataFrame) -> pd.DataFrame:
    params_df = params_df.copy()
    bounds = params_df["parameter"].astype(str).map(_classify_constraint)
    params_df["constraint_lower"] = [b[0] for b in bounds]
    params_df["constraint_upper"] = [b[1] for b in bounds]
    return params_df


def _add_confidence_interval(params_df: pd.DataFrame) -> pd.DataFrame:
    """Append ci_lower / ci_upper columns from value +/- _Z95 * std_error.

    Rows with a missing/non-finite std_error (e.g. a parameter present in
    estimated_parameters.csv but absent from standard_errors.csv) get NaN
    CI bounds, rendered as "--" by _fmt -- never silently dropped, per
    docs/REPORTING.md Sec 26.9 ("warn if numerical inconsistencies are
    detected") -- a missing SE is visible in the table rather than hidden.
    """
    if params_df.empty or "std_error" not in params_df.columns:
        params_df = params_df.copy()
        params_df["ci_lower"] = np.nan
        params_df["ci_upper"] = np.nan
        return params_df
    params_df = params_df.copy()
    se = pd.to_numeric(params_df["std_error"], errors="coerce")
    val = pd.to_numeric(params_df["value"], errors="coerce")
    params_df["ci_lower"] = val - _Z95 * se
    params_df["ci_upper"] = val + _Z95 * se
    return params_df


def _fmt_bound(v) -> str:
    """Format a constraint bound: 'NA' when unconstrained (None/NaN), else
    a fixed-point number -- deliberately not scientific/'--' so it reads
    unambiguously as 'no penalty term' rather than 'value unavailable'
    (that latter case, a missing std_error, still uses _fmt's '--')."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "NA"
    return f"{v:.3f}"


def _params_ci_table_tex(params_df: pd.DataFrame, caption: str, label: str,
                          _tt, _fmt) -> str:
    """
    LaTeX table: parameter | estimate | std. error | 95% CI lower | 95% CI
    upper | constraint lower | constraint upper. Layout mirrors
    generate_report_extended.py's _params_table_tex (same \\resizebox-to-
    textwidth pattern for long parameter lists), with the CI columns and
    the soft-penalty constraint columns added; se_quality is dropped
    (already stated once, for every table in this document, in the
    introductory note -- see build_document()) rather than repeated as a
    redundant column 42 times. The two constraint columns come from
    _classify_constraint -- see that function's docstring and the
    Introduction for what "NA" and the group/joint-norm/relative bounds
    mean.
    """
    if params_df.empty:
        return "% No parameters\n"
    headers = (
        r"\small Parameter & \small Estimate & \small Std.\ Error & "
        r"\small 95\% CI (low) & \small 95\% CI (high) & "
        r"\small Constraint (low) & \small Constraint (high) \\"
    )
    rows = []
    for _, row in params_df.iterrows():
        cells = [
            _tt(str(row["parameter"])),
            _fmt(row["value"]),
            _fmt(row.get("std_error")),
            _fmt(row.get("ci_lower")),
            _fmt(row.get("ci_upper")),
            _fmt_bound(row.get("constraint_lower")),
            _fmt_bound(row.get("constraint_upper")),
        ]
        rows.append(" & ".join(cells) + r" \\")

    return (
        r"\begin{table}[H]" + "\n"
        r"\centering" + "\n"
        r"\footnotesize" + "\n"
        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%" + "\n"
        r"\begin{tabular}{lrrrrrr}" + "\n"
        r"\hline" + "\n"
        + headers + "\n"
        r"\hline" + "\n"
        + "\n".join(rows) + "\n"
        r"\hline" + "\n"
        r"\end{tabular}" + "\n}" + "\n"
        rf"\caption{{{caption}}}" + "\n"
        rf"\label{{{label}}}" + "\n"
        r"\end{table}" + "\n"
    )


def build_document(no_compile: bool = False) -> Path:
    # Imported lazily: pulls in matplotlib's Agg backend and the ~3000-line
    # report script only when this function actually runs, matching the
    # lazy-import convention already used in diagnostics/multilocation_mosaics.py.
    from generate_report_extended import (
        _load_winners, _load_params_with_se, _tex, _tt, _fmt, compile_latex,
        _resolve_stage_dir,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    body_parts: List[str] = []
    n_tables = 0
    n_missing = 0

    for station in LOCATIONS:
        cfg = STATION_REGISTRY[station]
        display = cfg["display"]
        short = cfg["short"]
        run_dir = ARTIFACTS_DIR / cfg["run_id"]

        body_parts.append(f"\\chapter{{{_tex(display)}}}\n")

        if not run_dir.exists():
            body_parts.append(
                f"\\textit{{Run directory not found ({_tex(str(run_dir))}).}}\n\n"
            )
            print(f"  [SKIP] {display}: run directory not found ({run_dir})")
            continue

        winners = _load_winners(run_dir)

        for stage, branch in STAGE_MODEL_SEQUENCE:
            section = _section_label(stage, branch)   # short form, log messages only
            title = _section_title(stage, branch, display)  # "Stage N - ... - Location"
            model_id = resolve_stage_winner(winners, stage, branch)

            if model_id is None:
                body_parts.append(
                    f"\\section{{{_tex(title)}}}\n"
                    "\\textit{No winner recorded for this stage/branch at "
                    "this location.}\n\n"
                )
                print(f"  [SKIP] {display} / {section}: no winner recorded")
                n_missing += 1
                continue

            stage_dir = _resolve_stage_dir(model_id)
            params_df = _load_params_with_se(run_dir, stage_dir, model_id)

            if params_df.empty:
                body_parts.append(
                    f"\\section{{{_tex(title)}}} ({_tt(model_id)})\n"
                    "\\textit{Parameter artifacts unavailable for this model.}\n\n"
                )
                print(f"  [WARN] {display} / {model_id}: no estimated_parameters.csv")
                n_missing += 1
                continue

            params_df = _add_confidence_interval(params_df)
            params_df = _add_constraints(params_df)

            # Independent CSV artifact per table (docs/REPORTING.md Sec 19).
            csv_path = TABLES_DIR / f"{short}_{stage_dir}_{model_id}.csv"
            params_df.to_csv(csv_path, index=False)

            caption = (
                f"{_tex(title)} -- "
                f"{_tt(model_id)}: estimated parameters, "
                r"approximate standard errors, 95\% Wald confidence "
                r"intervals, and optimization soft-penalty constraints."
            )
            label = f"tab:params_{short}_{stage_dir}_{model_id}".replace("__", "_")

            body_parts.append(f"\\section{{{_tex(title)}}}\n")
            body_parts.append(
                _params_ci_table_tex(params_df, caption, label, _tt, _fmt)
            )
            n_tables += 1

    print(f"Built {n_tables} parameter tables ({n_missing} missing/skipped).")

    tex_path = OUT_DIR / "parameter_tables.tex"
    tex_path.write_text(_PREAMBLE + "\n".join(body_parts) + _CLOSING, encoding="utf-8")
    print(f"Wrote {tex_path}")

    if not no_compile:
        ok = compile_latex(tex_path)
        print("PDF compilation " + ("succeeded." if ok else "failed -- see log above."))

    return tex_path


_PREAMBLE = r"""\documentclass[11pt,a4paper]{report}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{float}
\usepackage{longtable}
\usepackage{caption}
\usepackage{hyperref}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}
\usepackage{xcolor}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
\emergencystretch=3em
\sloppy
\captionsetup{font=small}

\begin{document}

\title{ZA-GAS Multi-Location Parameter Estimates\\
\large Point estimates, standard errors, confidence intervals, and
optimization constraints}
\author{Generated by generate\_parameter\_tables\_report.py}
\date{}
\maketitle
\tableofcontents
\clearpage

\chapter*{Introduction}
\addcontentsline{toc}{chapter}{Introduction}

For each of the six completed locations, this document tabulates the
estimated parameters of that location's 7 stage-winner models: Stage 1
(baseline GAS), Stage 2 (weather covariates), and Stage 3 (Harvey
long-short), each shown for both the $\phi+\xi$ branch (dynamic scale and
tail shape, selected by in-sample/out-of-sample CRPS) and the $\phi$-only
branch (dynamic scale only, selected by RMSE), plus Stage 4 (tail-sensitive
$\xi$ regime, no branch split). These are exactly the same models compared
in the Stage 1-4 diagnostic mosaics (\texttt{multilocation\_diagnostic\_mosaics.ipynb}).

Section headings follow the pattern \emph{``Stage $N$ -- time-varying
\ldots{} -- Location''}: the $\phi$-only branch is labelled \emph{time-
varying location} (one dynamic distribution parameter) and the $\phi+\xi$
branch \emph{time-varying location and shape} (two); Stage 4 has no branch
split and is labelled \emph{time-varying shape (tail-sensitive regime)}.
This is report-only vocabulary for which GB2 log-link parameters are
dynamic in each model -- it does not rename $\phi$ (scale) or $\xi$ (shape)
anywhere else in the codebase or in \texttt{docs/MODELS.md}.

Stage 3's long-component covariates (\texttt{gamma\_L\_*\_el\_nino\_t},
\texttt{gamma\_L\_*\_la\_nina\_t}, and their \texttt{lag1mo}/\texttt{lag3mo}
counterparts) are monthly El Ni\~no/La Ni\~na dummy indicators built from
\texttt{data/processed/pacific/ENSO\_clean.csv} (\texttt{data/station\_loader.py}),
not the 90-day/30-day rolling raw-SST anomaly originally described in
\texttt{docs/MODELS.md} \S 18 -- that continuous-anomaly encoding was built
from a mislabelled source and was replaced everywhere in the live pipeline
(2026-07-07 fix); the old model ids are blacklisted in
\texttt{generate\_report\_extended.py}'s \texttt{\_STALE\_MODEL\_IDS} and
never appear as an accepted winner. Every Stage-3 table in this document
already reflects the dummy-variable encoding.

Every standard error is the quasi-Newton (BFGS) inverse-Hessian
approximation described in \texttt{docs/OPTIMIZATION.md} \S 9 and is
labelled \emph{approximate} in the underlying \texttt{standard\_errors.csv}
artifact for every model shown here. The reported 95\% confidence interval
is the standard asymptotic (Wald) interval
$[\hat\theta - z\,\widehat{\mathrm{se}},\ \hat\theta + z\,\widehat{\mathrm{se}}]$
with $z = \Phi^{-1}(0.975) \approx 1.9600$, computed directly from the
saved point estimate and standard error -- no model is re-estimated to
produce this document (\texttt{docs/EXECUTION.md} \S 22).

Parameter names follow \texttt{docs/MODELS.md}: \texttt{omega\_phi},
\texttt{f0\_phi}, \texttt{A\_phi\_\{1,2,3\}}, \texttt{B\_phi\_\{1,2,3\}}
are the $\phi_t$ GAS(3,3) recursion coefficients (analogously for
\texttt{xi} when the branch is $\phi+\xi$); \texttt{gamma}, \texttt{zeta}
are the static GB2 shape parameters; \texttt{omega0}, \texttt{rho},
\texttt{omega\_y\_\{1,365,366\}} are the AR-logistic occurrence ($\pi_t$)
dynamics coefficients (\texttt{docs/MODELS.md} \S 3); Stage 2/3 add
covariate coefficients named after their source variable, and Stage 4 adds
the regime-sensitised $\xi$ recursion coefficients.

\paragraph{Constraint (low)/Constraint (high) columns.} Every model here is
estimated by \emph{unbounded} BFGS with smooth quadratic penalty terms
rather than hard optimizer bounds (\texttt{docs/OPTIMIZATION.md} \S\S 2-3:
``hard bounds should be avoided unless mathematically unavoidable''). These
two columns report the soft-penalty threshold each parameter is actually
subject to in its model's objective function -- the value beyond which the
optimizer pays a quadratic penalty, not a scipy \texttt{bounds=} argument
(those exist separately, only as \emph{initialization} bounds for a
bounded L-BFGS-B warm start, and have no effect on the reported estimates).
\texttt{NA} means that parameter carries no penalty term at all (e.g.
\texttt{omega\_phi}, \texttt{f0\_phi}, \texttt{omega0}, \texttt{omega\_y\_*}
-- unconstrained under BFGS). Three families of threshold are not literally
per-parameter and are collapsed here to the loosest implied individual
bound (the true joint constraint is stated once per family, not per row):
\begin{itemize}
  \item \textbf{Lag-sum bound} (\texttt{A\_\{phi,xi\}\_\{lag\}},
    \texttt{B\_\{phi,xi\}\_\{lag\}} in Stages 1-2): the penalty applies to
    $\sum_{l\in\mathcal L}|A_{j,l}|\le 2.0$ and
    $\sum_{l\in\mathcal L}|B_{j,l}|\le 0.98$ across the \emph{whole} lag set
    $\mathcal L$, not per lag; $\pm2.0$/$\pm0.98$ is shown for every row in
    the block as the loosest bound any single lag's coefficient could reach.
  \item \textbf{Joint-norm bound} (all \texttt{gamma\_*} covariate
    coefficients): the penalty applies to
    $\lVert\Gamma_j\rVert_2\le 20$ for the whole coefficient vector; $\pm20$
    is shown per coefficient as the loosest implied individual bound.
  \item \textbf{Relative bound} (Stage 3's \texttt{B\_S\_\{phi,xi\}}): the
    persistence restriction is $0<B_{S,j}<B_{L,j}<1$
    (\texttt{docs/MODELS.md} \S 20) -- $B_{S,j}$'s true upper bound is that
    same fit's own $B_{L,j}$ estimate (visible in the same table), not a
    fixed constant; $0.995$ is shown as the outer bound implied by
    $B_{L,j}$'s own cap.
\end{itemize}

\clearpage

"""

_CLOSING = r"""
\end{document}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-compile", action="store_true",
                         help="Skip pdflatex compilation")
    args = parser.parse_args()
    build_document(no_compile=args.no_compile)


if __name__ == "__main__":
    main()
