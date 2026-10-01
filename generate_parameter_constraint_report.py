"""
Parameter / estimate / SE / CI / constraint tables for every Stage 1-5
model, every location, every branch (2026-09-06).

=============================================================================
OVERVIEW
=============================================================================

Builds one report.tex with a chapter per location, a section per branch
(phi_only, phi_xi), and one table per stage (1-5, where applicable),
using the shared format/logic in report_constraint_table.py:

    parameter | estimate | std error | 95% CI (lower) | 95% CI (upper) |
    lower constraint | upper constraint

Per direct specification:
  - A parameter pinned against a penalty boundary shows "(boundary
    estimated)" in the SE column and "-" for both CI columns, REGARDLESS
    of how small its raw gradient happens to be at that pinned point.
  - A parameter with |gradient| >= 0.3 (and NOT at a boundary) shows
    "(high gradient)" / "-" / "-" instead.
  - A constraint shared by a COMBINATION of parameters (e.g. the joint
    sum_l|B_l|<0.98, or Harvey's persistence ordering) shows ONLY the
    marker (**, ***, ****, ...) in the constraint cell -- the actual bound
    and which parameters it spans is explained once, below the table. A
    plain single "*" is reserved for a DIFFERENT flag: it marks the
    ESTIMATE (not the constraint cell) of any GAS dynamics or
    zero-augmentation parameter (never a GB2 distribution parameter)
    whose correctly-inferred 95% CI contains zero.
  - Whenever a table contains a high-gradient case, a short diagnosis
    (concentrated vs. generalized, and any strong Hessian-implied
    correlation with another parameter) is printed below it.

Per docs/REPORTING.md: reads only already-saved artifacts, never fits
anything.

=============================================================================
USAGE
=============================================================================

    python generate_parameter_constraint_report.py [--no-compile] [--location "..."]
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_harmonic_seasonal_experiment as R
from data.station_loader import STATION_REGISTRY
import report_constraint_table as T

REPORT_DIR = ROOT / "reports" / "parameter constraint tables"
ARTIFACTS_DIR = ROOT / "artifacts"

LOCATIONS = ["BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
             "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR"]
BRANCHES = ["phi_only", "phi_xi"]
STAGE_LABELS = {"stage1": "Stage 1 (baseline GAS)", "stage2": "Stage 2 (+ weather covariates)",
                "stage3": "Stage 3 (Harvey long-short, no seasonal)",
                "stage4": "Stage 4 (Harvey long-short + trigonometric seasonal, damped)",
                "stage5": "Stage 5 (static GB2 + threshold-weighted tail $\\xi$)"}

LATEX_PREAMBLE = r"""\documentclass[10pt,a4paper]{report}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=1.8cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{caption}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}
\emergencystretch=3em
\sloppy
\captionsetup{font=small}
"""


def _tex(s) -> str:
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _stage_dir_and_winner(run_dir: Path, winners: dict, stage_key: str, branch: str):
    if branch == "phi_only":
        key = f"{stage_key}_phi_only"
        subdir = "stage1" if stage_key == "stage1" else f"{stage_key}_phi"
    else:
        key = stage_key
        subdir = stage_key
    model_id = winners[key]["winner"]
    return run_dir / subdir / model_id, model_id


def load_est_grad_hess(model_dir: Path):
    est = pd.read_csv(model_dir / "estimated_parameters.csv").set_index("parameter")["value"]
    grad_path = model_dir / "gradients.csv"
    grad = (pd.read_csv(grad_path).set_index("parameter")["gradient"].reindex(est.index)
             if grad_path.exists() else pd.Series(np.nan, index=est.index))
    hess_path = model_dir / "hess_inv.npy"
    hess_inv = np.load(hess_path) if hess_path.exists() else None
    return est, grad, hess_inv


def raw_variances_for(est: pd.Series, hess_inv: Optional[np.ndarray]):
    if hess_inv is None:
        return [None] * len(est)
    out = []
    for i in range(len(est)):
        if i >= hess_inv.shape[0]:
            out.append(None)
            continue
        v = hess_inv[i, i]
        out.append(float(v) if (np.isfinite(v) and v > 0) else None)
    return out


def corr_check_factory(names, hess_inv):
    idx = {n: i for i, n in enumerate(names)}

    def check(param_name):
        if hess_inv is None or param_name not in idx:
            return None
        i1 = idx[param_name]
        if i1 >= hess_inv.shape[0] or hess_inv[i1, i1] <= 0:
            return None
        best = None
        for other in names:
            if other == param_name:
                continue
            i2 = idx[other]
            if i2 >= hess_inv.shape[0] or hess_inv[i2, i2] <= 0:
                continue
            c = hess_inv[i1, i2] / np.sqrt(hess_inv[i1, i1] * hess_inv[i2, i2])
            if not np.isfinite(c):
                continue
            if best is None or abs(c) > abs(best[1]):
                best = (other, c)
        if best is not None and abs(best[1]) > 0.5:
            return T.pretty_name_generic(best[0]), best[1]
        return None
    return check


def build_table_for_stage(model_dir: Path, meta: dict, model_id: str, stage_key: str,
                           caption: str, label: str) -> str:
    est, grad, hess_inv = load_est_grad_hess(model_dir)
    names = list(est.index)
    raw_var = raw_variances_for(est, hess_inv)
    derivs = [1.0] * len(names)  # every parameter reported here is on its natural (untransformed) scale

    cm: T.ConstraintMap = {}
    cm.update(T.constraints_static(est))
    cm.update(T.constraints_pi(est))

    mt = meta.get("model_type", "")
    if mt in ("ZAGASModel", "CovZAGASModel"):
        tv_names = meta.get("tv_names", [])
        lags = meta.get("gas_lags") or meta.get("lags") or [1]
        cm.update(T.constraints_gas_dynamics(est, tv_names, lags))
        if mt == "CovZAGASModel":
            cm.update(T.constraints_cov(est, tv_names, meta.get("cov_names", [])))
    elif mt == "HarveyZAGASModel":
        cm.update(T.constraints_harvey(
            est, meta.get("harvey_target", []), meta.get("long_names", []), meta.get("short_names", []),
            meta.get("restrict_persistence_upper", True), meta.get("use_auxiliary_penalties", True),
            meta.get("n_harmonics", 0), meta.get("kappa_gamma_bound"), meta.get("seasonal_damping_max"),
        ))
    elif mt == "ThresholdWeightedXiModel" or stage_key == "stage5":
        cm.update(T.constraints_threshxi(est))

    rows, footnotes, zero_ci_used = T.build_rows(names, est.values, grad.values, raw_var, derivs, cm, T.pretty_name_generic)
    diag = T.diagnose_high_gradient_rows(rows, pretty_name=T.pretty_name_generic,
                                          corr_check=corr_check_factory(names, hess_inv))
    return T.save_constraint_table(rows, footnotes, caption, label, diag, zero_ci_used=zero_ci_used)


def build_stage5_table(short: str, run_id: str, caption: str, label: str) -> Optional[str]:
    run_dir = ARTIFACTS_DIR / run_id
    winners_path = run_dir / "stage_winners.json"
    if not winners_path.exists():
        return None
    winners = json.loads(winners_path.read_text())
    if "stage5" not in winners:
        return None
    suffix = "_dynpi" if winners.get("stage5_overall_best_variant") == "dynamic_pi" else ""
    winner_id = winners["stage5"]["winner"] + suffix
    model_dir = run_dir / "stage5" / winner_id
    if not model_dir.exists():
        return None
    meta = json.loads((model_dir / "metadata.json").read_text())
    return build_table_for_stage(model_dir, meta, winner_id, "stage5", caption, label)


def build_location_branch(location: str, branch: str) -> str:
    cfg = STATION_REGISTRY[location]
    short = cfg.get("short", location)
    run_id = cfg["run_id"]
    display = cfg.get("display", location)
    run_dir = ARTIFACTS_DIR / run_id
    winners_path = run_dir / "stage_winners.json"

    parts = [rf"\section{{{_tex(branch)}}}"]
    if not winners_path.exists():
        parts.append("No stage\\_winners.json found for this location.")
        return "\n\n".join(parts)
    winners = json.loads(winners_path.read_text())

    for stage_key in ("stage1", "stage2", "stage3"):
        try:
            if stage_key == "stage3":
                # Stage 3 here means the SAME "standard" Harvey long-short
                # config already used as the baseline for the Stage 4
                # seasonality comparison all session (R.resolve_baseline,
                # reading artifacts/longshort_diagnostics_experiment/...),
                # NOT pipeline/runner.py's raw stage_winners.json "stage3"
                # entry -- that is a DIFFERENT, older artifact using this
                # project's PRE-RENAME Harvey parameter names (A_L/B_L/
                # gamma_S instead of kappa_L/phi_L/Gamma_S), and using it
                # here would silently make Stage 3 and Stage 4 two
                # different underlying models rather than "with vs
                # without seasonality" on the SAME one.
                baseline = R.resolve_baseline(short, branch)
                model_dir, model_id = baseline["model_dir"], baseline["harvey_config"]
            else:
                model_dir, model_id = _stage_dir_and_winner(run_dir, winners, stage_key, branch)
            meta = json.loads((model_dir / "metadata.json").read_text())
            caption = f"{_tex(display)} ({_tex(branch)}): {STAGE_LABELS[stage_key]} -- {_tex(model_id)}."
            label = f"tab:{short}-{branch}-{stage_key}"
            parts.append(rf"\subsection{{{STAGE_LABELS[stage_key]}}}")
            parts.append(build_table_for_stage(model_dir, meta, model_id, stage_key, caption, label))
        except Exception as exc:
            parts.append(rf"\subsection{{{STAGE_LABELS[stage_key]}}}")
            parts.append(f"Could not build this table: {_tex(exc)}")

    # Stage 4: harmonic seasonal Harvey (damped), this session's own tree
    seasonal_dir = R.OUT_DIR / short / branch / "m5_final_damp0.999"
    parts.append(rf"\subsection{{{STAGE_LABELS['stage4']}}}")
    if (seasonal_dir / "metadata.json").exists():
        meta = json.loads((seasonal_dir / "metadata.json").read_text())
        caption = f"{_tex(display)} ({_tex(branch)}): {STAGE_LABELS['stage4']}."
        label = f"tab:{short}-{branch}-stage4"
        try:
            parts.append(build_table_for_stage(seasonal_dir, meta, "m5_final_damp0.999", "stage4", caption, label))
        except Exception as exc:
            parts.append(f"Could not build this table: {_tex(exc)}")
    else:
        parts.append("No seasonal (Stage 4) fit available for this location/branch.")

    # Stage 5: phi_xi branch only (frozen phi, tail-focused xi dynamics)
    if branch == "phi_xi":
        parts.append(rf"\subsection{{{STAGE_LABELS['stage5']}}}")
        caption = f"{_tex(display)}: {STAGE_LABELS['stage5']}."
        label = f"tab:{short}-stage5"
        try:
            tbl = build_stage5_table(short, run_id, caption, label)
            parts.append(tbl if tbl is not None else "No Stage 5 fit available for this location.")
        except Exception as exc:
            parts.append(f"Could not build this table: {_tex(exc)}")

    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--location", default=None)
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    locations = [args.location] if args.location else LOCATIONS

    print("Building tables...")
    chapters = []
    for location in locations:
        cfg = STATION_REGISTRY[location]
        display = cfg.get("display", location)
        print(f"  {display}")
        chapter = [rf"\chapter{{{_tex(display)}}}"]
        for branch in BRANCHES:
            chapter.append(build_location_branch(location, branch))
        chapters.append("\n\n".join(chapter))

    intro = "\n\n".join([
        r"\chapter{Introduction}",
        (r"Parameter estimates, standard errors, 95\% confidence intervals, and "
         r"mathematical/optimization constraints for every Stage 1-5 model, per "
         r"location and branch. Column format: parameter, natural-scale estimate, "
         r"delta-method standard error, 95\% CI (lower), 95\% CI (upper), lower "
         r"constraint, upper constraint."),
        (r"A parameter pinned against a penalty boundary is marked "
         r"\textbf{(boundary estimated)} in the standard-error column, with both "
         r"CI columns shown as ``--'' -- this is checked independently of the raw "
         r"gradient, since a boundary optimum can have a genuinely tiny gradient "
         r"there (the penalty's own gradient exactly cancels the likelihood's) "
         r"while the ordinary delta-method standard error remains invalid at a "
         r"constrained optimum. A parameter whose gradient could not be brought "
         r"below 0.3 (and is not at a boundary) is marked \textbf{(high gradient)}, "
         r"likewise with both CI columns as ``--''. Wherever a table contains a "
         r"high-gradient parameter, a short diagnosis appears directly below it: "
         r"whether the large gradient is concentrated in one or two parameters "
         r"(consistent with a clean boundary pin) or spread across many "
         r"(consistent with a broader identification issue), and, where the "
         r"Hessian shows a strong correlation with another specific parameter, "
         r"that correlation is reported explicitly."),
        (r"A constraint that applies to a COMBINATION of parameters rather than "
         r"one alone (e.g.\ the joint bound $\sum_\ell|B_\ell|<0.98$ across every "
         r"lag of a GAS recursion, or Harvey's persistence ordering "
         r"$0<\varphi_S<\varphi_L$) is marked with two or more asterisks in the "
         r"constraint cell -- ** for the first such combination in a table, *** "
         r"for the second, and so on -- with the actual bound and which "
         r"parameters it spans explained once, in a footnote below the table, "
         r"rather than repeated in every member's cell."),
        (r"A single asterisk on the ESTIMATE itself flags a correctly-inferred "
         r"95\% CI (i.e.\ not a boundary/high-gradient placeholder) that contains "
         r"zero, for a GAS dynamics or zero-augmentation parameter only -- never "
         r"for a GB2 distribution shape parameter ($\gamma,\zeta,\xi,\varphi$). "
         r"See the footnote below any table where it appears."),
    ])

    body = "\n\n".join([r"\begin{document}",
                         r"\title{Parameter, Standard Error, and Constraint Tables --- All Stages}",
                         rf"\date{{{datetime.date.today().isoformat()}}}",
                         r"\maketitle", r"\tableofcontents",
                         intro] + chapters + [r"\end{document}"])
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
