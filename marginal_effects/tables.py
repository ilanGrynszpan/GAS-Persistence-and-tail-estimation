"""
LaTeX table builders for the marginal-effects report
(docs/claude_code_marginal_effects_instructions.md Sec.4), reading ONLY the
cached CSVs written by run_marginal_effects_onestep.py /
run_marginal_effects_multistep.py (docs/REPORTING.md: reports never
trigger computation).

Column layout (shared by every table): two top-level model-family groups,
"GAS(p,q)" (family="standard") and "Long-short" (family="harvey"), each
split into DPT / AT / ENSO sub-columns -- 6 value columns total, matching
instructions Sec.4 point 1/2's "two header columns ... each divided in 3
columns: DPT, AT, ENSO."

Cell conventions:
  "0.000"  -- the variable does not enter this model's dynamics
              (coefficients are structurally 0; instructions Sec.1:
              "Use zero for a coefficient when the variable does not
              enter").
  "---"    -- no model at all is available for this (location, branch,
              family) cell (Cruzeiro do Sul / Manaus, 2-TVP Harvey); a
              table footnote explains why.
  "val (lo, hi)" -- `val` is the SELECTED model's effect; (lo, hi) is the
              min/max of that same effect across every OTHER grad<1,
              variable-bearing candidate in the same (location, branch,
              family) cell (user request 2026-09-17: a robustness range,
              not a confidence interval). Omitted when the selected model
              is the only qualifying candidate for that variable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from marginal_effects.model_selection import SHORTS, LOCATION_DISPLAY

FAMILIES = [("standard", "GAS$(p,q)$"), ("harvey", "Long-short")]
VARS = ["DPT", "AT", "ENSO"]


def _fmt(x, unavailable: bool, sig: int = 3) -> str:
    if unavailable:
        return "---"
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "0.000"
    if float(x) == 0.0:
        return "0.000"
    return f"{x:.{sig}g}"


def _fmt_with_range(primary_val, unavailable: bool, pool_vals, sig: int = 3) -> str:
    """`pool_vals`: the same effect statistic from every grad<1, variable-
    bearing candidate in this cell (including the primary/selected one).
    Appends "(min, max)" only when more than one candidate contributes --
    a single-candidate cell has no range to report."""
    base = _fmt(primary_val, unavailable, sig)
    if unavailable or base == "0.000":
        return base
    finite = [v for v in pool_vals if v is not None and np.isfinite(v)]
    if len(finite) <= 1:
        return base
    lo, hi = min(finite), max(finite)
    return f"{base} ({_fmt(lo, False, sig)}, {_fmt(hi, False, sig)})"


# =============================================================================
# Tables 1-2: one-step aggregate mean/variance effects, 1-TVP and 2-TVP
# =============================================================================

def _onestep_row(summary: pd.DataFrame, branch: str, short: str, col: str) -> str:
    """`summary`: concatenation of the primary summary.csv (primary=True)
    and alternatives_summary.csv (primary=False) -- see
    generate_marginal_effects_report._load_onestep_all()."""
    loc_rows = summary[(summary["branch"] == branch) & (summary["location"] == short)]
    cells = []
    for family, _ in FAMILIES:
        for v in VARS:
            cell_rows = loc_rows[(loc_rows["family"] == family) & (loc_rows["variable"] == v)]
            prim = cell_rows[cell_rows["primary"].astype(bool)]
            if len(prim) == 0:
                cells.append("---")
                continue
            row = prim.iloc[0]
            # in_model False but the model itself IS available -> the effect is a
            # structural 0.000, not "unavailable"; only a genuinely missing model
            # (stage/cfg both NaN, e.g. Cruzeiro/Manaus 2-TVP Harvey) prints "---".
            model_missing = pd.isna(row.get("stage"))
            pool = cell_rows[cell_rows["in_model"].astype(bool)][col].tolist()
            cells.append(_fmt_with_range(row.get(col), model_missing, pool))
    return f"{LOCATION_DISPLAY[short]} & " + " & ".join(cells) + r" \\"


def build_onestep_table(summary: pd.DataFrame, branch: str) -> str:
    """One-step-ahead marginal effects on the ZA-GB2 conditional mean
    (E[Delta M_{1,t}]) and variance (E[Delta V_{M,t}]), origin-window-
    averaged (onestep summary.csv's dM1_dx_mean / dVM_dx_mean). ONE table
    per TVP branch, per instructions Sec.4 point 1/2: a 6-location "mean"
    block followed by a 6-location "variance" block (12 data rows total),
    under the shared GAS(p,q)/Long-short x DPT/AT/ENSO column layout. Each
    cell also carries the (min, max) range across every other grad<1
    candidate using that variable -- see _fmt_with_range."""
    header = (r"\begin{tabular}{l" + "ccc" * len(FAMILIES) + "}\n\\toprule\n"
               + " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{disp}}}" for _, disp in FAMILIES)
               + r" \\" + "\n"
               + "".join(rf"\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(FAMILIES))) + "\n"
               + "Location & " + " & ".join(VARS * len(FAMILIES)) + r" \\" + "\n\\midrule")

    mean_rows = [_onestep_row(summary, branch, short, "dM1_dx_mean") for short in SHORTS]
    var_rows = [_onestep_row(summary, branch, short, "dVM_dx_mean") for short in SHORTS]

    body = ("\\multicolumn{7}{l}{\\textit{Effect on the conditional mean, "
            "$E[\\Delta M_{1,t}]$}} \\\\\n" + "\n".join(mean_rows)
            + "\n\\midrule\n\\multicolumn{7}{l}{\\textit{Effect on the conditional variance, "
              "$E[\\Delta V_{M,t}]$}} \\\\\n" + "\n".join(var_rows))

    footer = r"\bottomrule\end{tabular}"
    tvp_label = ("1 time-varying parameter ($\\varphi$)" if branch == "phi_only"
                 else "2 time-varying parameters ($\\varphi,\\xi$)")
    caption = (
        rf"One-step-ahead rolling marginal effects of DPT/AT/ENSO on the ZA-GB2 predictive "
        rf"mean and variance, {tvp_label}, averaged over the out-of-sample rolling-forecast "
        r"window. Effect of one raw unit (1\,\textdegree C for DPT/AT; a Neutral$\to$El "
        r"Ni\~no switch for ENSO). ``0.000'' = the variable is not part of this model's "
        r"fitted covariate set; ``---'' = no gradient-converged model was available for this "
        r"cell (see report text). Parenthesised (min, max): the range of this same effect "
        r"across every other grad$<$1 candidate model (same location/TVP/family) that also "
        r"uses the variable, omitted when the selected model is the only one that does."
    )
    return (r"\begin{table}[H]\centering\footnotesize" + "\n"
            + header + "\n" + body + "\n" + footer + "\n"
            + rf"\caption{{{caption}}}" + "\n"
            + r"\end{table}")


# =============================================================================
# Tables 3-6: long tables of h-step-ahead effects (single origin, 5
# predictions per location: baseline + DPT/AT/ENSO->El Nino/ENSO->La Nina)
# =============================================================================

MS_SCENARIOS = ["DPT", "AT", "ENSO_EL_NINO", "ENSO_LA_NINA"]
MS_SCENARIO_DISP = ["DPT", "AT", "ENSO$\\to$El Ni\\~no", "ENSO$\\to$La Ni\\~na"]


def build_multistep_table(multistep: pd.DataFrame, branch: str, effect_kind: str, family: str,
                            horizons=(1, 2, 5, 10, 30, 60, 90)) -> str:
    """`multistep`: concatenation of every <short>_<branch>_<family>.csv
    written by run_marginal_effects_multistep.py (marginal_effects.multistep.
    scenario_effects_for_model's output), with location/branch/family
    columns attached. `effect_kind` in {"mean","variance"} -> reads
    delta_mu (Sec.3.4's Delta mu_h) or delta_var (Delta V_h) for the 4
    named scenarios, from the single end-of-training-sample origin.
    `family` in {"standard","harvey"} -- ONE table per family (rather than
    both side by side): with the "value (min, max)" robustness range in
    every cell, a single 10-column (2-family) landscape table does not fit
    even at \\tiny font (found via a real compile: ~76pt/2.7cm overfull);
    splitting by family halves the column count (Location, h, + 4 scenario
    columns = 6) and fits comfortably. Long table: rows grouped by
    location, 7 horizon-rows each."""
    col = "delta_mu" if effect_kind == "mean" else "delta_var"
    label = ("mean forecast, $\\Delta\\mu_h=E_1(Y_{t+h})-E_0(Y_{t+h})$" if effect_kind == "mean"
              else "forecast variance, $\\Delta V_h=\\mathrm{Var}_1(Y_{t+h})-\\mathrm{Var}_0(Y_{t+h})$")
    tvp_label = "1 TVP ($\\varphi$)" if branch == "phi_only" else "2 TVP ($\\varphi,\\xi$)"
    fam_disp = dict(FAMILIES)[family]
    n_cols = len(MS_SCENARIOS)

    header_cols = "l" + "r" * (1 + n_cols)
    top = (r"\begin{landscape}" + "\n"
           + r"{\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{0.9}" + "\n"
           + r"\begin{longtable}{" + header_cols + "}\n"
           + rf"\caption{{$h$-step-ahead paired Monte Carlo marginal effect on the {label}, "
             f"{tvp_label}, {fam_disp}, from the single forecast origin at the end of the "
             r"training period (instructions Sec.3.4; finite one-unit/one-time-impulse effects, "
             r"not exact derivatives). ENSO effects are genuine state switches (Neutral/other "
             r"$\to$ El Ni\~no or La Ni\~na), not linearised derivatives. Parenthesised "
             r"(min, max): the range of this effect across every other grad$<$1 candidate "
             r"model (same location/TVP/family) that also uses the variable. See "
             r"\texttt{artifacts/marginal\_effects/multistep/} for Monte Carlo "
             r"standard errors and 95\% intervals.}\\" + "\n"
           + r"\toprule Location & $h$ & " + " & ".join(MS_SCENARIO_DISP) + r" \\" + "\n"
           + r"\midrule\endfirsthead" + "\n"
           + r"\toprule Location & $h$ & " + " & ".join(MS_SCENARIO_DISP) + r"\\ \midrule\endhead" + "\n"
           + r"\midrule\endfoot" + "\n" + r"\bottomrule\endlastfoot")

    rows = []
    sub = multistep[(multistep["branch"] == branch) & (multistep["family"] == family)]
    for short in SHORTS:
        loc_rows = sub[sub["location"] == short]
        for hi, h in enumerate(horizons):
            h_rows = loc_rows[loc_rows["horizon"] == h]
            cells = []
            for scenario in MS_SCENARIOS:
                cell_rows = h_rows[h_rows["scenario"] == scenario]
                prim = cell_rows[cell_rows["primary"].astype(bool)]
                if len(prim) == 0:
                    cells.append("---")
                    continue
                # A primary row exists here (checked above) -> a real fitted model IS
                # available for this cell; in_model=False just means THIS scenario's
                # variable isn't part of THAT model (e.g. ENSO in a stage4S Harvey
                # config), which _fmt already renders as "0.000" via its value==0.0
                # branch. "unavailable"/"---" is reserved for no-model-at-all cells,
                # already handled above -- passing `not in_model` here (an earlier
                # bug, found 2026-09-17 against real Manaus/phi_only/harvey data)
                # wrongly printed "---" for a perfectly real, just variable-less, model.
                row = prim.iloc[0]
                pool = cell_rows[cell_rows["in_model"].astype(bool)][col].tolist()
                cells.append(_fmt_with_range(row.get(col), False, pool, sig=2))
            loc_label = LOCATION_DISPLAY[short] if hi == 0 else ""
            rows.append(f"{loc_label} & {h} & " + " & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")

    return (top + "\n" + "\n".join(rows) + "\n" + r"\end{longtable}"
            + "\n}" + "\n" + r"\end{landscape}")


def build_baseline_table(multistep: pd.DataFrame, branch: str,
                          horizons=(1, 2, 5, 10, 30, 60, 90)) -> str:
    """Supplementary table: the 5th "prediction" -- the baseline (no
    intervention) forecast mean/variance at each horizon, from the same
    single origin, for every available model. Not one of instructions
    Sec.4's 6 required tables, but explicitly one of the "5 predictions per
    location" the user asked for."""
    tvp_label = "1 TVP ($\\varphi$)" if branch == "phi_only" else "2 TVP ($\\varphi,\\xi$)"
    header = (r"\begin{tabular}{l" + "cc" * len(FAMILIES) + "}\n\\toprule\n"
              + " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{disp}}}" for _, disp in FAMILIES) + r" \\" + "\n"
              + "".join(rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(FAMILIES))) + "\n"
              + "Location, $h$ & " + " & ".join(["mean", "var."] * len(FAMILIES)) + r" \\" + "\n\\midrule")
    rows = []
    # primary-only: the baseline is a property of ONE (selected) model, not a
    # robustness range -- without this filter, an alternative candidate's own
    # BASELINE row (every candidate has one) could be picked arbitrarily.
    sub = multistep[(multistep["branch"] == branch) & (multistep["scenario"] == "BASELINE")
                     & (multistep["primary"].astype(bool))]
    for short in SHORTS:
        loc_rows = sub[sub["location"] == short]
        for hi, h in enumerate(horizons):
            h_rows = loc_rows[loc_rows["horizon"] == h]
            cells = []
            for family, _ in FAMILIES:
                r = h_rows[h_rows["family"] == family]
                if len(r) == 0:
                    cells += ["---", "---"]
                    continue
                row = r.iloc[0]
                cells += [_fmt(row.get("baseline_mean"), False), _fmt(row.get("baseline_var"), False)]
            loc_label = f"{LOCATION_DISPLAY[short]}, $h={h}$"
            rows.append(f"{loc_label} & " + " & ".join(cells) + r" \\")
    footer = r"\bottomrule\end{tabular}"
    caption = (rf"Baseline (no-intervention) $h$-step-ahead forecast mean and variance, {tvp_label}, "
               r"from the single end-of-training-period origin -- the reference every "
               r"$\Delta\mu_h,\Delta V_h$ above is computed against.")
    return (r"\begin{table}[H]\centering\scriptsize" + "\n" + header + "\n" + "\n".join(rows) + "\n"
            + footer + "\n" + rf"\caption{{{caption}}}" + "\n" + r"\end{table}")
