"""
Builds reports/marginal effects/report.tex (+ .pdf) for the marginal-
effects-of-external-variables deliverable
(docs/claude_code_marginal_effects_instructions.md).

Reads ONLY cached artifacts (docs/REPORTING.md: "Reports should never
require model re-estimation"):
    artifacts/marginal_effects/onestep/summary.csv + alternatives_summary.csv
    artifacts/marginal_effects/multistep/<short>_<branch>_<family>.csv (primary, one per available model)
    artifacts/marginal_effects/multistep/alternatives/<short>_<branch>_<family>__<stage>__<cfg>.csv
    artifacts/marginal_effects/multistep/convergence_check.csv
    artifacts/stage_comparison_20260911/metrics.json (via marginal_effects.model_selection,
        purely to redisplay the already-computed grad/RMSE/CRPS of each selected model --
        no refitting)

The alternatives files (one per non-primary grad<1, variable-bearing
candidate -- marginal_effects.model_selection.all_valid_models) feed the
"(min, max) across every other correctly-optimised model using this
variable" range shown alongside each table cell (user request 2026-09-17).

Run only after run_marginal_effects_onestep.py and
run_marginal_effects_multistep.py have both completed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marginal_effects.model_selection import select_all, LOCATION_DISPLAY, SHORTS
from marginal_effects.tables import build_onestep_table, build_multistep_table, build_baseline_table

ONESTEP_DIR = ROOT / "artifacts" / "marginal_effects" / "onestep"
ONESTEP_SUMMARY = ONESTEP_DIR / "summary.csv"
ONESTEP_ALT_SUMMARY = ONESTEP_DIR / "alternatives_summary.csv"
MULTISTEP_DIR = ROOT / "artifacts" / "marginal_effects" / "multistep"
MULTISTEP_ALT_DIR = MULTISTEP_DIR / "alternatives"
CONVERGENCE = MULTISTEP_DIR / "convergence_check.csv"
REP = ROOT / "reports" / "marginal effects"


def _load_onestep_all() -> pd.DataFrame:
    """Primary summary.csv (primary=True) concatenated with
    alternatives_summary.csv (primary=False, written only if
    run_marginal_effects_onestep.py's alternatives pass has run)."""
    frames = [pd.read_csv(ONESTEP_SUMMARY)]
    if ONESTEP_ALT_SUMMARY.exists():
        frames.append(pd.read_csv(ONESTEP_ALT_SUMMARY))
    df = pd.concat(frames, ignore_index=True)
    df["primary"] = df["primary"].astype(bool)
    return df


def _load_multistep_all() -> pd.DataFrame:
    """Concatenates every primary <short>_<branch>_<family>.csv (tagged
    primary=True) with every alternatives/<short>_<branch>_<family>__<stage>__<cfg>.csv
    (primary=False, stage/cfg parsed from the filename -- scenario_effects_for_model's
    own ScenarioEffect rows carry no stage/cfg field, since a primary file is
    already uniquely identified by its (short,branch,family) name)."""
    frames = []
    for p in sorted(MULTISTEP_DIR.glob("*_*_*.csv")):
        if p.name == "convergence_check.csv":
            continue
        df = pd.read_csv(p)
        df["primary"] = True
        frames.append(df)
    if MULTISTEP_ALT_DIR.exists():
        for p in sorted(MULTISTEP_ALT_DIR.glob("*.csv")):
            df = pd.read_csv(p)
            _, stage, cfg = p.stem.split("__")
            df["primary"] = False
            df["stage"] = stage
            df["cfg"] = cfg
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No multistep result CSVs found under {MULTISTEP_DIR}")
    return pd.concat(frames, ignore_index=True)


def _tex_escape(s: str) -> str:
    """Escapes LaTeX special characters in dynamically-inserted plain text
    (model identifiers, free-text notes) -- underscores in particular are
    common in this repo's config/stage names (e.g. "L1_365_366_dew_temp_enso")
    and would otherwise be read as a math-mode subscript, breaking compilation."""
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _selection_table() -> str:
    """Which model won each of the 24 cells, and why -- REPORTING.md Sec.6:
    every result must be traceable to how it was selected."""
    sel = select_all()
    rows = []
    for short in SHORTS:
        for branch in ("phi_only", "phi_xi"):
            for family in ("standard", "harvey"):
                s = sel[(short, branch, family)]
                fam_disp = "GAS$(p,q)$" if family == "standard" else "Long-short"
                tvp_disp = "1" if branch == "phi_only" else "2"
                if not s.available:
                    rows.append(f"{LOCATION_DISPLAY[short]} & {tvp_disp} & {fam_disp} & "
                                f"\\multicolumn{{4}}{{l}}{{\\textit{{not available -- "
                                f"{_tex_escape(s.note)}}}}} \\\\")
                    continue
                vars_disp = ", ".join(s.variables) if s.variables else "none"
                m = s.metrics
                override = " (override)" if s.overridden else ""
                model_disp = _tex_escape(f"{s.stage}/{s.cfg}") + override
                rows.append(
                    f"{LOCATION_DISPLAY[short]} & {tvp_disp} & {fam_disp} & "
                    f"{model_disp} & {vars_disp} & "
                    f"{m['grad_inf']:.3f} & {m['rmse_oos']:.2f} & {m['crps_oos']:.3f} \\\\"
                )
    body = "\n".join(rows)
    header_row = r"Location & TVP & Family & Model & Variables & Grad.\ & RMSE & CRPS \\ \midrule"
    # Model strings run up to ~34 characters (e.g. "stage2/L123_365_366_dew_temp_enso") --
    # landscape + scriptsize + unwrapped ("l") Model/Variables columns avoids the
    # ugly multi-line wrapping a narrow p{} column produces at portrait width.
    return (
        r"\begin{landscape}" + "\n"
        + r"{\scriptsize\setlength{\tabcolsep}{4pt}" + "\n"
        + r"\begin{longtable}{lllllccc}" + "\n"
        + r"\caption{Selected model per location, TVP count, and family (marginal\_effects.model\_selection). "
        r"``TVP'' = number of time-varying GB2 parameters. Grad.\ = $\|\nabla Q(\hat\theta)\|_\infty$; "
        r"RMSE/CRPS are out-of-sample. Two cells marked ``override'' use the best covariate-bearing "
        r"grad$<$1 alternative rather than the literal RMSE/CRPS-minimising (zero-covariate) winner "
        r"-- user decision, 2026-09-16 (see report text).}\\" + "\n"
        + r"\toprule " + header_row + r"\endfirsthead" + "\n"
        + r"\toprule " + header_row + r"\endhead" + "\n"
        + r"\midrule\endfoot" + "\n" + r"\bottomrule\endlastfoot" + "\n"
        + body + "\n" + r"\end{longtable}" + "\n" + "}" + "\n" + r"\end{landscape}"
    )


def _convergence_note() -> str:
    if not CONVERGENCE.exists():
        return ""
    df = pd.read_csv(CONVERGENCE)
    if len(df) == 0:
        return ""
    n_warn = int((df.get("warn_mu", False) | df.get("warn_var", False)).sum())
    return (
        r"\subsection*{Monte Carlo convergence check}" + "\n"
        rf"The single origin was re-simulated per available (location, branch, family, scenario) "
        r"with 3$\times$ the paths under an independent seed (instructions Sec.3.7). "
        rf"{n_warn} of {len(df)} (scenario, horizon) cells had a Monte Carlo standard error "
        r"exceeding 50\% of the estimated effect at the base path count -- see "
        r"\texttt{artifacts/marginal\_effects/multistep/convergence\_check.csv} for the full table."
    )


def build():
    if not ONESTEP_SUMMARY.exists():
        raise FileNotFoundError(
            "Run run_marginal_effects_onestep.py first -- this report reads only its cached CSV "
            "output, per docs/REPORTING.md."
        )
    REP.mkdir(parents=True, exist_ok=True)
    onestep = _load_onestep_all()
    multistep = _load_multistep_all()

    table1 = build_onestep_table(onestep, "phi_only")
    table2 = build_onestep_table(onestep, "phi_xi")
    # Tables 3-6, split by family (GAS(p,q) vs Long-short): with the (min,max)
    # robustness range in every cell, a single 10-column (both families
    # side by side) landscape table does not fit even at \tiny font -- see
    # build_multistep_table's own docstring.
    table3a = build_multistep_table(multistep, "phi_only", "mean", "standard")
    table3b = build_multistep_table(multistep, "phi_only", "mean", "harvey")
    table4a = build_multistep_table(multistep, "phi_only", "variance", "standard")
    table4b = build_multistep_table(multistep, "phi_only", "variance", "harvey")
    table5a = build_multistep_table(multistep, "phi_xi", "mean", "standard")
    table5b = build_multistep_table(multistep, "phi_xi", "mean", "harvey")
    table6a = build_multistep_table(multistep, "phi_xi", "variance", "standard")
    table6b = build_multistep_table(multistep, "phi_xi", "variance", "harvey")
    baseline1 = build_baseline_table(multistep, "phi_only")
    baseline2 = build_baseline_table(multistep, "phi_xi")

    pre = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage[margin=1.8cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,array}
\usepackage{pdflscape}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}\usepackage{float}
\emergencystretch=3em\sloppy
"""

    body = "\n\n".join([
        r"\begin{document}",
        r"\title{Marginal Effects of External Covariates on Rolling ZA-GB2 Forecasts}"
        r"\author{}\date{2026-09-16}\maketitle",

        r"\section*{Objective}"
        r"For the selected fitted GAS$(p,q)$ and Harvey long-short ZA-GB2 models at each of six "
        r"locations, quantify how dew point (DPT), air temperature (AT), and ENSO move the "
        r"one-step-ahead conditional mean/variance of rainfall (analytically) and the "
        r"$h$-step-ahead forecast mean/variance, $h\in\{1,2,5,10,30,60,90\}$ (paired Monte Carlo, "
        r"from the single forecast origin at the end of each model's training period).",

        r"\section*{Model selection}"
        r"Selection criterion (docs/claude\_code\_marginal\_effects\_instructions.md, "
        r"``TAKE CARE''): gradient norm "
        r"$\|\nabla Q(\hat\theta)\|_\infty<1$, minimise OOS CRPS then RMSE, preferring a "
        r"3-variable (DPT+AT+ENSO) configuration when one clears the gradient bar. "
        r"Two of the 24 (location, TVP, family) cells have no model clearing the gradient bar "
        r"at all (Cruzeiro do Sul and Manaus, 2 time-varying parameters, Harvey long-short) and "
        r"are shown as not available. Two further cells whose literal RMSE/CRPS-minimising model "
        r"had zero external covariates were, on request, replaced by the best covariate-bearing "
        r"grad$<$1 alternative instead (marked ``override'' below). Every table cell below also "
        r"reports, in parentheses when applicable, the (min, max) range of that same effect "
        r"across every OTHER grad$<$1 candidate model in the same (location, TVP, family) cell "
        r"that also uses the variable in question -- a robustness range across correctly-"
        r"optimised alternative specifications, not a confidence interval." + "\n\n" + _selection_table(),

        r"\section*{Table 1 --- One-step-ahead effects, 1 time-varying parameter}" + "\n" + table1,
        r"\section*{Table 2 --- One-step-ahead effects, 2 time-varying parameters}" + "\n" + table2,

        r"\section*{Tables 3--6 --- $h$-step-ahead effects (paired Monte Carlo)}"
        r"Five predictions per location, all from the single forecast origin at the end of each "
        r"model's training period: a baseline (no intervention) path, and four named one-time-"
        r"impulse scenarios at that origin -- DPT $+1\,^\circ$C, AT $+1\,^\circ$C, and the two "
        r"non-trivial ENSO state switches (Neutral/La Ni\~na $\to$ El Ni\~no, and Neutral/El "
        r"Ni\~no $\to$ La Ni\~na; the three ENSO states are mutually exclusive). Every scenario "
        r"shares common random numbers with the baseline per docs/claude\_code\_marginal\_"
        r"effects\_instructions.md Sec.3; fitted parameters are held fixed throughout (conditional "
        r"predictive effects, not parameter-uncertainty propagation). Monte Carlo standard errors "
        r"and 95\% intervals are in the underlying CSVs; the tables below show the point estimate "
        r"(and, where applicable, the (min, max) robustness range across alternative candidate "
        r"models). Each of Tables 3--6 is split into an (a) GAS$(p,q)$ and (b) Long-short half, "
        r"since the (min, max) range widens every cell too much for both families to fit "
        r"side by side even in landscape." + "\n\n" + _convergence_note(),

        r"\subsection*{Table 3a --- Mean effect, 1 TVP, GAS$(p,q)$}" + "\n" + table3a,
        r"\subsection*{Table 3b --- Mean effect, 1 TVP, Long-short}" + "\n" + table3b,
        r"\subsection*{Table 4a --- Variance effect, 1 TVP, GAS$(p,q)$}" + "\n" + table4a,
        r"\subsection*{Table 4b --- Variance effect, 1 TVP, Long-short}" + "\n" + table4b,
        r"\subsection*{Table 5a --- Mean effect, 2 TVP, GAS$(p,q)$}" + "\n" + table5a,
        r"\subsection*{Table 5b --- Mean effect, 2 TVP, Long-short}" + "\n" + table5b,
        r"\subsection*{Table 6a --- Variance effect, 2 TVP, GAS$(p,q)$}" + "\n" + table6a,
        r"\subsection*{Table 6b --- Variance effect, 2 TVP, Long-short}" + "\n" + table6b,

        r"\subsection*{Baseline forecasts (5th prediction: no intervention)}"
        r"The reference trajectory every $\Delta\mu_h,\Delta V_h$ above is computed against."
        + "\n\n" + baseline1 + "\n\n" + baseline2,

        r"\end{document}",
    ])

    (REP / "report.tex").write_text(pre + "\n" + body, encoding="utf-8")
    print("wrote", REP / "report.tex")

    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                       cwd=REP, capture_output=True, text=True)
    log = (REP / "report.log").read_text(errors="ignore") if (REP / "report.log").exists() else ""
    ok = (REP / "report.pdf").exists() and "Fatal error" not in log
    print("report.pdf:", "OK" if ok else "FAILED (see report.log)")


if __name__ == "__main__":
    build()
