"""
2026-10-01: regenerate the result tables of the thesis section
"GAS/SD model empirical results" (review/GAS(p,q) staged execution.tex)
after the GB2 Fisher cross-term fix -> reports/regenerated01102026/.

EXECUTION FLOW
    1. recompute_oos_fisher_fix.py (run first) recomputed the OOS artifacts
       of every 2-TVP inverse_fisher candidate into
       artifacts/regenerated_01102026/ and wrote a corrected metrics.json.
       Every other candidate is read from its original cache.
    2. This script builds every table twice, through two "views" of the
       same selection code (generate_ml_comparison_report.G and the report
       modules that import it):
         OLD view -- artifacts/stage_comparison_20260911/metrics.json and
                     the original artifact trees (what the thesis shows);
         NEW view -- the corrected metrics.json, and the regenerated
                     artifact dir whenever one exists for a candidate.
       Selection rules are unchanged (gradient tier, per-parameter
       convergence fraction, OOS CRPS at 1% tolerance, OOS RMSE; or
       twCRPS95 minimisation where the thesis says so), so any change of
       selected model ("reselection") is caused only by corrected OOS
       metrics.
    3. Under every table, a note lists (a) each cell whose selected model
       changed (old -> new config), and (b) each cell whose value changed
       because its model's OOS metrics were recomputed.
    4. Tables that need fresh computation from a model (Kupiec 1-in-30,
       90-day-ahead, ENSO phase) are recomputed in the NEW view only; their
       note flags whether the selected model changed or was recomputed.

Naming: the class the thesis calls "DCS/UC - ST" (previously "basic
structural model") is the artifact stage "stage4S"; its class id in code is
dcs_uc_st.

No fitting happens here.
"""

from __future__ import annotations

import json
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

import generate_ml_comparison_report as G
import generate_rmse_crps_report as RC
import generate_twcrps_report as TW
import generate_kupiec_report as KU
import generate_tail_oos_report as TO
import generate_enso_metrics_report as EN

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
REGEN = ART / "regenerated_01102026"
REP = ROOT / "reports" / "regenerated01102026"

SHORTS = G.SHORTS
DISP = {"bh": "Belo Horizonte", "cruzeiro": "Cruzeiro do Sul", "darwin": "Darwin",
        "garanhuns": "Garanhuns", "manaus": "Manaus", "salvador": "Salvador"}
CLASS_ID = {"stage1": "gas_no_ext", "stage2": "gas_ext", "stage3": "long_short",
            "stage4": "long_short_seas", "stage4S": "dcs_uc_st"}
CLASS_LBL = {"stage1": "GAS(p,q) no ext.", "stage2": "GAS(p,q) + ext.", "stage3": "long-short",
             "stage4": "long-short + seas.", "stage4S": "DCS/UC - ST"}
STAGES5 = ["stage1", "stage2", "stage3", "stage4", "stage4S"]
BRANCH_SHORT = {"phi_only": "1 TVP", "phi_xi": "2 TVP"}

OLD_MET = G.MET
NEW_MET = json.loads((REGEN / "metrics.json").read_text())
_ORIG_RESOLVE = G._resolve_cfg_dir


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------

def _resolve_new(short, ident):
    d = _ORIG_RESOLVE(short, ident)
    alt = REGEN / d.relative_to(ART)
    return alt if (alt / "result.json").exists() else d


@contextmanager
def view(which):
    G.MET = NEW_MET if which == "new" else OLD_MET
    G._resolve_cfg_dir = _resolve_new if which == "new" else _ORIG_RESOLVE
    try:
        yield
    finally:
        G.MET = OLD_MET
        G._resolve_cfg_dir = _ORIG_RESOLVE


def recomputed(short, ident):
    """True if this candidate's OOS artifacts were regenerated (2-TVP, full FI)."""
    if ident is None:
        return False
    return _resolve_new(short, ident) != _ORIG_RESOLVE(short, ident)


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def rnd(x, d, plain=False):
    """Round as the benchmark does: 2-decimal tables are rounded half-up from
    the 3-decimal value (e.g. 0.535 -> 0.54), not from the raw float."""
    if d == 2 and not plain:
        return float(Decimal(f"{x:.3f}").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    return round(x, d)


class Flagged(float):
    """A metric value taken from a fallback model; printed with a trailing '*'."""
    fallback = True


def fmt(v, d, plain=False):
    x = num(v)
    star = "*" if getattr(v, "fallback", False) else ""
    return "div" if x is None else f"{rnd(x, d, plain):.{d}f}{star}"


def bold_min(vals, d, plain=False):
    """Format a list of values; bold the minimum (ties on the rounded value).
    plain=True rounds the raw float (the benchmark's ML table convention)."""
    rounded = [rnd(num(v), d, plain) if num(v) is not None else None for v in vals]
    finite = [r for r in rounded if r is not None]
    best = min(finite) if finite else None
    out = []
    for v, r in zip(vals, rounded):
        s = fmt(v, d, plain)
        out.append(rf"\textbf{{{s}}}" if (r is not None and r == best) else s)
    return out


def cfg_txt(ident):
    if ident is None:
        return "n/a"
    stage, cfg, branch = ident
    return f"{CLASS_LBL[stage]} {cfg.replace('_', '-')} ({BRANCH_SHORT[branch]})"


# Report-level settings (overridden by generate_regeneratedwithpi_01102026_report.py)
CHANGED_LABEL = "Values changed (same model, OOS recomputed after the Fisher fix)"
LIST_CHANGED = True          # list every changed cell, or only reselections
TITLE = "GAS/SD empirical results -- regenerated tables"
INTRO = None                 # None -> the Fisher-fix intro written in build()
EXTRA_TABLES = []            # callables returning extra LaTeX blocks appended at the end


class Notes:
    """Collects per-table regeneration notes."""

    def __init__(self):
        self.resel, self.changed, self.other = [], [], []

    def cell(self, where, old_ident, new_ident, old_val, new_val, d):
        if old_ident != new_ident:
            self.resel.append(f"{where}: {cfg_txt(old_ident)} $\\to$ {cfg_txt(new_ident)} "
                              f"({fmt(old_val, d)} $\\to$ {fmt(new_val, d)})")
        elif LIST_CHANGED and fmt(old_val, d) != fmt(new_val, d):
            self.changed.append(f"{where}: {fmt(old_val, d)} $\\to$ {fmt(new_val, d)}")

    def tex(self):
        parts = []
        if self.resel:
            parts.append(r"\textbf{Reselected:} " + "; ".join(self.resel) + ".")
        if self.changed:
            parts.append(r"\textbf{" + CHANGED_LABEL + r":} " + "; ".join(self.changed) + ".")
        parts += self.other
        if not parts:
            parts.append(r"No change from the benchmark: no reselection and no recomputed cell.")
        return (r"\par\smallskip\begin{minipage}{0.98\textwidth}\footnotesize\raggedright"
                r"\textit{Regeneration note.} " + " ".join(parts) + r"\end{minipage}")


def pick(fn, *args):
    with view("old"):
        oi, om = fn(*args)
    with view("new"):
        ni, nm = fn(*args)
    return oi, om, ni, nm


def table(caption, label, colspec, header, rows, notes, size=r"\footnotesize", sep="3pt"):
    return "\n".join([r"\begin{table}[H]\centering" + size + rf"\setlength{{\tabcolsep}}{{{sep}}}",
                      rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header, r"\midrule"]
                     + rows + [r"\bottomrule", r"\end{tabular}", notes.tex(), r"\end{table}"])


SS = lambda txt: r"\shortstack{" + txt + "}"
SC = lambda txt: r"\scriptsize\shortstack{" + txt + "}"
FIVE_HDR = ("Location & " + " & ".join([
    SS(r"GAS($p,q$)\\no external\\variables"), SS(r"GAS($p,q$)\\+ external\\variables"),
    SS(r"Long--short\\dynamics"), SS(r"Long--short dynamics\\+ seasonality"),
    SS(r"DCS/UC - ST")]) + r" \\")


# ---------------------------------------------------------------------------
# Tables 1-4: rolling 1-step RMSE / CRPS, best option per class
# ---------------------------------------------------------------------------

def t_metric_by_class(branch, key, caption, label):
    notes, rows = Notes(), []
    for s in SHORTS:
        vals = []
        for st in STAGES5:
            oi, om, ni, nm = pick(RC.best_for_stage, s, st, branch)
            notes.cell(f"{DISP[s]}, {CLASS_LBL[st]}", oi, ni, (om or {}).get(key), (nm or {}).get(key), 3)
            vals.append((nm or {}).get(key))
        rows.append(f"{DISP[s]} & " + " & ".join(bold_min(vals, 3)) + r" \\")
    return table(caption, label, "lrrrrr", FIVE_HDR, rows, notes)


# ---------------------------------------------------------------------------
# Table 5: multi-step GAS/SD vs ML
# ---------------------------------------------------------------------------

def t_ml():
    ml = pd.read_csv(ROOT / "reports" / "ml_baselines_horizon" / "all_locations_ml_baselines_horizon.csv")
    ar_crps = pd.read_csv(ROOT / "reports" / "ml_baselines_horizon" / "all_locations_ar_crps_gaussian.csv")
    notes, body = Notes(), []
    for s in SHORTS:
        with view("old"):
            oi, odf = G.load_gas_multistep(s)
        with view("new"):
            ni, ndf = G.load_gas_multistep(s)
        if oi != ni:
            notes.resel.append(f"{DISP[s]}: {cfg_txt(oi)} $\\to$ {cfg_txt(ni)}")
        elif recomputed(s, ni):
            notes.other.append(f"{DISP[s]}: same model ({cfg_txt(ni)}), multi-step paths recomputed.")
        sub = ml[ml.location == G.LOC_MAP[s]]
        asub = ar_crps[ar_crps.location == G.LOC_MAP[s]]
        for h in G.HORIZONS:
            rm = []
            for mdl in ("AR", "GAS", "XGB", "AR+Aux", "LSTM"):
                if mdl == "GAS":
                    rm.append(float(ndf.loc[h, "rmse"]) if h in ndf.index else np.nan)
                else:
                    r = sub[(sub.model == mdl) & (sub.horizon == h)]
                    rm.append(float(r["rmse_all"].iloc[0]) if len(r) else np.nan)
            a = asub[asub.horizon == h]
            cr = [float(a["crps_all"].iloc[0]) if len(a) else np.nan,
                  float(ndf.loc[h, "crps"]) if h in ndf.index else np.nan]
            if oi == ni and h in odf.index:
                for nm_, k in (("GAS/SD RMSE", "rmse"), ("GAS/SD CRPS", "crps")):
                    o, n = odf.loc[h, k], ndf.loc[h, k]
                    if fmt(o, 2) != fmt(n, 2):
                        notes.changed.append(f"{DISP[s]} $H={h}$ {nm_}: {fmt(o, 2)} $\\to$ {fmt(n, 2)}")
            body.append(f"{DISP[s]} & {h} & " + " & ".join(bold_min(rm, 2, plain=True)) + " & "
                        + " & ".join(bold_min(cr, 2, plain=True)) + r" \\")
        body.append(r"\hline")
    hdr = (r"Series & $H$ & \multicolumn{5}{c|}{RMSE} & "
           r"\multicolumn{2}{c|}{CRPS} \\ \cline{3-9}" "\n"
           r"& & AR & GAS/SD & XGB & AR+Aux & LSTM & AR & GAS/SD \\ \hline")
    return "\n".join([
        r"{\footnotesize\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{0.82}",
        r"\begin{longtable}{|p{2.7cm}|c|ccccc|cc|}",
        r"\caption{Out-of-sample RMSE and CRPS for GAS/SD and ML models.}",
        r"\label{tab:daily_rmse_crps_all} \\", r"\hline", hdr, r"\endfirsthead",
        r"\hline", hdr, r"\endhead", r"\endfoot"] + body[:-1] + [r"\hline", r"\end{longtable}}",
        notes.tex()])


# ---------------------------------------------------------------------------
# Tables 6-12: twCRPS
# ---------------------------------------------------------------------------

def t_tw_gas_by_branch(branch, caption, label):
    notes, rows = Notes(), []
    for s in SHORTS:
        cells = []
        for key, lvl in (("tw95", "95"), ("tw98", "98")):
            pair = []
            for st, fn in (("stage1", TW.stage1_best), ("stage2", TW.stage2_best)):
                oi, om, ni, nm = pick(fn, s, [branch], TW.ALL_LAGS)
                notes.cell(f"{DISP[s]}, {CLASS_LBL[st]}, twCRPS{lvl}", oi, ni,
                           (om or {}).get(key), (nm or {}).get(key), 3)
                pair.append((nm or {}).get(key))
            cells += bold_min(pair, 3)
        rows.append(f"{DISP[s]} & " + " & ".join(cells) + r" \\")
    g1, g2 = SC(r"GAS($p,q$)\\no external\\variables"), SC(r"GAS($p,q$)\\+ external\\variables")
    hdr = (r"& \multicolumn{2}{c}{twCRPS95} & \multicolumn{2}{c}{twCRPS98} \\"
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
           f"Location & {g1} & {g2} & {g1} & {g2} \\\\")
    return table(caption, label, "lrrrr", hdr, rows, notes, sep="5pt")


def t_tw_harvey_by_branch(key, lvl, caption, label):
    notes, rows = Notes(), []
    for s in SHORTS:
        cells = []
        for st in ("stage3", "stage4", "stage4S"):
            pair = []
            for br in ("phi_only", "phi_xi"):
                oi, om, ni, nm = pick(TW.harvey_best, s, st, [br])
                notes.cell(f"{DISP[s]}, {CLASS_LBL[st]}, {BRANCH_SHORT[br]}", oi, ni,
                           (om or {}).get(key), (nm or {}).get(key), 2)
                pair.append((nm or {}).get(key))
            cells += bold_min(pair, 2)
        rows.append(f"{DISP[s]} & " + " & ".join(cells) + r" \\")
    one, two = SC(r"One\\time-varying\\parameter"), SC(r"Two\\time-varying\\parameters")
    hdr = (r"& \multicolumn{2}{c}{Long--short dynamics} & "
           r"\multicolumn{2}{c}{\shortstack{Long--short dynamics\\+ seasonality}} & "
           r"\multicolumn{2}{c}{\shortstack{DCS/UC - ST}} \\"
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n"
           f"Location & {one} & {two} & {one} & {two} & {one} & {two} \\\\")
    return table(caption, label, "lrrrrrr", hdr, rows, notes)


def t_tw_seasonality(key, lvl, caption, label):
    notes, rows = Notes(), []
    for s in SHORTS:
        cells = []
        for st, fn in (("stage1", TW.stage1_best), ("stage2", TW.stage2_best)):
            pair = []
            for lags, tag in ((TW.NO_SEAS_LAGS, "no seasonal lags"), (TW.SEAS_LAGS, "seasonality added")):
                oi, om, ni, nm = pick(fn, s, TW.BRANCHES, lags)
                notes.cell(f"{DISP[s]}, {CLASS_LBL[st]}, {tag}", oi, ni,
                           (om or {}).get(key), (nm or {}).get(key), 2)
                pair.append((nm or {}).get(key))
            cells += bold_min(pair, 2)
        rows.append(f"{DISP[s]} & " + " & ".join(cells) + r" \\")
    a, b = SC(r"No seasonal\\lags included"), SC(r"Seasonality\\added")
    hdr = (r"& \multicolumn{2}{c}{\shortstack{GAS($p,q$)\\no external variables}} & "
           r"\multicolumn{2}{c}{\shortstack{GAS($p,q$)\\+ external variables}} \\"
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
           f"Location & {a} & {b} & {a} & {b} \\\\")
    return table(caption, label, "lrrrr", hdr, rows, notes, sep="5pt")


def _seasonal_best(s, key):
    """Stage 4 vs Stage 4S finalists (each standard-rule best over cov x branch),
    the lower twCRPS wins -- generate_twcrps_report.group_b_collapsed's rule."""
    i4, m4 = TW.harvey_best(s, "stage4", TW.BRANCHES)
    i4s, m4s = TW.harvey_best(s, "stage4S", TW.BRANCHES)
    v = lambda mm: num((mm or {}).get(key)) if num((mm or {}).get(key)) is not None else np.inf
    return (i4, m4) if v(m4) <= v(m4s) else (i4s, m4s)


def t_tw_ls_vs_seasonal():
    notes, rows = Notes(), []
    for s in SHORTS:
        cells = []
        for key, lvl in (("tw95", "95"), ("tw98", "98")):
            oi3, om3, ni3, nm3 = pick(TW.harvey_best, s, "stage3", TW.BRANCHES)
            notes.cell(f"{DISP[s]}, long-short, twCRPS{lvl}", oi3, ni3,
                       (om3 or {}).get(key), (nm3 or {}).get(key), 2)
            ois, oms, nis, nms = pick(_seasonal_best, s, key)
            notes.cell(f"{DISP[s]}, DCS/UC with seasonal component, twCRPS{lvl}", ois, nis,
                       (oms or {}).get(key), (nms or {}).get(key), 2)
            cells += bold_min([(nm3 or {}).get(key), (nms or {}).get(key)], 2)
        rows.append(f"{DISP[s]} & " + " & ".join(cells) + r" \\")
    a, b = SC(r"Long--short\\dynamics"), SC(r"DCS/UC with\\seasonal component")
    hdr = (r"& \multicolumn{2}{c}{twCRPS95} & \multicolumn{2}{c}{twCRPS98} \\"
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
           f"Location & {a} & {b} & {a} & {b} \\\\")
    return table("twCRPS95 and twCRPS98 for long--short dynamics without a seasonal component and "
                 "DCS/UC dynamics with a seasonal component.", "tab:B2_tw95_tw98_seasonality",
                 "lrrrr", hdr, rows, notes, sep="5pt")


# ---------------------------------------------------------------------------
# Table 13: ENSO phase
# ---------------------------------------------------------------------------

def t_enso():
    notes = Notes()
    blocks = {}
    for lbl, fn in (("gas", EN.best_overall), ("ls", EN.enso_longshort_best)):
        per_loc = {}
        for s in SHORTS:
            oi, _, ni, _ = pick(fn, s)
            name = "best GAS(p,q) overall" if lbl == "gas" else "best long-short with ENSO"
            if oi != ni:
                notes.resel.append(f"{DISP[s]}, {name}: {cfg_txt(oi)} $\\to$ {cfg_txt(ni)}")
            elif recomputed(s, ni):
                notes.other.append(f"{DISP[s]}, {name}: same model ({cfg_txt(ni)}), values recomputed.")
            with view("new"):
                per_loc[s] = EN.approach_metrics(s, ni)
        blocks[lbl] = per_loc
    with view("new"):
        counts = {s: {c: v["n"] for c, v in blocks["gas"][s].items()} for s in SHORTS}
    conds = ["El Ni\\~no", "La Ni\\~na", "Neither"]
    rows = [r"\multicolumn{7}{l}{\textit{Number of test days by ENSO phase}} \\"]
    for c in conds:
        rows.append(f"{c} & " + " & ".join(str(counts[s].get(c, 0)) for s in SHORTS) + r" \\")
    for lbl, title in (("gas", r"Best GAS($p,q$) model overall"),
                       ("ls", r"Best long--short model containing ENSO")):
        other = "ls" if lbl == "gas" else "gas"
        rows += [r"\hline", rf"\multicolumn{{7}}{{l}}{{\textbf{{{title}}}}} \\"]
        for c in conds:
            rows += [r"\addlinespace[2pt]", rf"\multicolumn{{7}}{{l}}{{\textit{{{c}}}}} \\"]
            for met in ("RMSE", "CRPS", "twCRPS95", "twCRPS98"):
                cells = []
                for s in SHORTS:
                    mine = blocks[lbl][s].get(c, {}).get(met)
                    theirs = blocks[other][s].get(c, {}).get(met)
                    if mine is None:
                        cells.append("--")
                        continue
                    s_ = fmt(mine, 2)
                    if theirs is None or rnd(mine, 2) <= rnd(theirs, 2):
                        s_ = rf"\textbf{{{s_}}}"
                    cells.append(s_)
                rows.append(f"{met} & " + " & ".join(cells) + r" \\")
    hdr_cells = " & ".join(SC(DISP[s].replace(" ", r"\\", 1)) if " " in DISP[s] else r"\scriptsize " + DISP[s]
                           for s in SHORTS)
    hdr = rf"\toprule & {hdr_cells} \\ \midrule"
    return "\n".join([
        r"{\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{0.90}",
        r"\begin{longtable}{lrrrrrr}",
        r"\caption{Rolling one-step-ahead out-of-sample performance by realised ENSO phase.}",
        r"\label{tab:enso_phase_performance} \\", hdr, r"\endfirsthead", hdr, r"\endhead",
        r"\bottomrule", r"\endfoot"] + rows + [r"\bottomrule", r"\end{longtable}}", notes.tex()])


# ---------------------------------------------------------------------------
# Tables 14-16: Kupiec (rolling 1-step and 90-day-ahead), 90-day twCRPS
# ---------------------------------------------------------------------------

KUPIEC_PICK_NEW = None      # optional (short) -> (ident, metrics) used for the NEW view
KUPIEC_RULE_NOTE = None     # optional sentence describing that rule (added to the notes)


def _tw95_idents():
    out, notes = {}, Notes()
    if KUPIEC_RULE_NOTE:
        notes.other.append(KUPIEC_RULE_NOTE)
    for s in SHORTS:
        if KUPIEC_PICK_NEW is None:
            oi, _, ni, _ = pick(G.best_by_tw95, s)
        else:
            with view("old"):
                oi, _ = G.best_by_tw95(s)
            with view("new"):
                ni, _ = KUPIEC_PICK_NEW(s)
        if oi != ni:
            notes.resel.append(f"{DISP[s]}: {cfg_txt(oi)} $\\to$ {cfg_txt(ni)}")
        elif recomputed(s, ni):
            notes.other.append(f"{DISP[s]}: same model ({cfg_txt(ni)}), values recomputed.")
        out[s] = ni
    return out, notes


def _kupiec_hdr():
    return (r"& \multicolumn{2}{c}{95\% quantile} & \multicolumn{2}{c}{98\% quantile} & "
            r"\multicolumn{2}{c}{1-in-30-day} \\"
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n"
            r"Location & Ratio & Kupiec $p$ & Ratio & Kupiec $p$ & Ratio & Kupiec $p$ \\")


def _ratio(n_viol, n, q, p):
    if n == 0 or n_viol == 0:
        s = "--"
    else:
        s = f"{(1.0 - q) * n / n_viol:.3f}"
    if s != "--" and p is not None and np.isfinite(p) and p < 0.05:
        s += "*"
    return s


def t_kupiec_and_tail(idents, base_notes):
    rows_1s, rows_90, rows_tw = [], [], []
    with view("new"):
        for s in SHORTS:
            ident = idents[s]
            res = json.loads((G._resolve_cfg_dir(s, ident) / "result.json").read_text())
            st = KU.three_level_stats(s, ident, res)
            cells = []
            for k, q in (("95", 0.95), ("98", 0.98), ("1in30", KU.Q_1IN30)):
                e = st[k]
                cells += [_ratio(e["n_viol"], e["n"], q, e["kupiec_p"]), fmt(e["kupiec_p"], 3)]
            rows_1s.append(f"{DISP[s]} & " + " & ".join(cells) + r" \\")

            r = TO.h90_row(s, ident, res)
            n = r["n_origins"]
            cells = []
            for k, q in (("95", 0.95), ("98", 0.98), ("1in30", TO.Q_1IN30)):
                rate, p = r[f"rate{k}"], r[f"kup{k}"]
                nv = int(round(rate * n)) if num(rate) is not None else 0
                cells += [_ratio(nv, n, q, p), fmt(p, 3)]
            rows_90.append(f"{DISP[s]} & " + " & ".join(cells) + r" \\")
            rows_tw.append(f"{DISP[s]} & {fmt(r['tw95'], 3)} & {fmt(r['tw98'], 3)} \\\\")
            print(f"  kupiec/tail {s}: {ident}")
    t14 = table("Rolling one-step-ahead out-of-sample theoretical-to-empirical exceedance-count "
                "ratios and Kupiec $p$-values.", "tab:kupiec", "lrrrrrr", _kupiec_hdr(),
                rows_1s, base_notes, size=r"\small", sep="6pt")
    t15 = table("90-day-ahead prediction theoretical-to-empirical exceedance-count ratios and "
                "Kupiec $p$-values.", "tab:tail_out-of-sample", "lrrrrrr", _kupiec_hdr(),
                rows_90, base_notes, sep="5pt")
    t16 = table("90-day-ahead prediction twCRPS95 and twCRPS98.", "tab:tail_out-of-sample_tw",
                "lrr", r"Location & twCRPS95 & twCRPS98 \\", rows_tw, base_notes, sep="5pt")
    return t14, t15, t16


# ---------------------------------------------------------------------------
# Table 17: BSS
# ---------------------------------------------------------------------------

def t_bss():
    notes, rows = Notes(), []
    for s in SHORTS:
        oi, _, ni, _ = pick(G.best_overall_id, s)
        if oi != ni:
            notes.resel.append(f"{DISP[s]}: {cfg_txt(oi)} $\\to$ {cfg_txt(ni)}")
        with view("old"):
            old = json.loads((G._resolve_cfg_dir(s, oi) / "result.json").read_text())
        with view("new"):
            res = json.loads((G._resolve_cfg_dir(s, ni) / "result.json").read_text())
        for k in ("BS_clim", "BS", "BSS"):
            if fmt(old[k], 4) != fmt(res[k], 4):
                notes.changed.append(f"{DISP[s]} {k.replace('_', ' ')}: {fmt(old[k], 4)} $\to$ {fmt(res[k], 4)}")
        rows.append(f"{DISP[s]} & {res['BS_clim']:.4f} & {res['BS']:.4f} & {res['BSS']:.4f} \\\\")
    notes.other.append(r"$\pi_{t|t-1}$ does not depend on the score scaling, so the Fisher fix can "
                       r"only change this table through reselection.")
    return table(r"Out-of-sample Brier skill score of $\pi_{t|t-1}$ vs.\ the monthly climatological "
                 "reference.", "tab:bss", "lrrr", r"Location & $BS_{clim}$ & $BS_{GAS/SD}$ & BSS \\",
                 rows, notes, size=r"\small", sep="8pt")


# ---------------------------------------------------------------------------
# model-reference table
# ---------------------------------------------------------------------------

def t_selected_models():
    rows = []
    for br in ("phi_only", "phi_xi"):
        for s in SHORTS:
            for st in STAGES5:
                oi, om, ni, nm = pick(RC.best_for_stage, s, st, br)
                flag = "reselected" if oi != ni else ("recomputed" if recomputed(s, ni) else "")
                sc = ""
                if ni is not None:
                    sc = json.loads((_resolve_new(s, ni) / "result.json").read_text()).get("scaling", "")
                rows.append(f"{DISP[s]} & {BRANCH_SHORT[br]} & {CLASS_LBL[st]} & "
                            f"{ni[1].replace('_', '-') if ni else 'n/a'} & {sc.replace('_', ' ')} & {flag} \\\\")
        rows.append(r"\midrule")
    return "\n".join([
        r"{\scriptsize\begin{longtable}{llllll}",
        r"\caption{Selected configuration per cell of Tables \ref{tab:rmse_1tvp}--\ref{tab:crps_2tvp} "
        r"(class id of DCS/UC - ST in code: \texttt{dcs\_uc\_st}).}\label{tab:selected_models}\\",
        r"\toprule Location & Branch & Class & Config & Scaling & Status \\ \midrule \endhead",
    ] + rows[:-1] + [r"\bottomrule\end{longtable}}"])


# ---------------------------------------------------------------------------

def build():
    REP.mkdir(parents=True, exist_ok=True)
    idents_tw95, tw95_notes = _tw95_idents()
    t14, t15, t16 = t_kupiec_and_tail(idents_tw95, tw95_notes)
    tables = [
        t_metric_by_class("phi_only", "rmse_oos", r"Rolling one-step-ahead out-of-sample RMSE, best option "
                          r"per model class, time-varying location ($\varphi_{t|t-1}$).", "tab:rmse_1tvp"),
        t_metric_by_class("phi_xi", "rmse_oos", r"Rolling one-step-ahead out-of-sample RMSE, best option "
                          r"per model class, time-varying location and shape ($\varphi_{t|t-1},\xi_{t|t-1}$).",
                          "tab:rmse_2tvp"),
        t_metric_by_class("phi_only", "crps_oos", r"Rolling one-step-ahead out-of-sample CRPS, best option "
                          r"per model class, time-varying location ($\varphi_{t|t-1}$).", "tab:crps_1tvp"),
        t_metric_by_class("phi_xi", "crps_oos", r"Rolling one-step-ahead out-of-sample CRPS, best option "
                          r"per model class, time-varying location and shape ($\varphi_{t|t-1},\xi_{t|t-1}$).",
                          "tab:crps_2tvp"),
        t_ml(),
        t_tw_gas_by_branch("phi_only", r"twCRPS95 and twCRPS98 for GAS($p,q$) with and without external "
                           r"variables using the best lag/external variable set, one time-varying parameter "
                           r"($\varphi_{t|t-1}$).", "tab:A_tw95_tw98_phi_only"),
        t_tw_gas_by_branch("phi_xi", r"twCRPS95 and twCRPS98 for GAS($p,q$) with and without external "
                           r"variables using the best lag/external variable set, two time-varying parameters "
                           r"($\varphi_{t|t-1},\xi_{t|t-1}$).", "tab:A_tw95_tw98_phi_xi"),
        t_tw_harvey_by_branch("tw95", "95", r"twCRPS95 for the long--short dynamics, long--short dynamics "
                              r"with seasonality, and DCS/UC - ST using the best external-variable "
                              r"configuration. One vs two time-varying parameters.", "tab:B_tw95_tvp_comparison"),
        t_tw_harvey_by_branch("tw98", "98", r"twCRPS98 for the long--short dynamics, long--short dynamics "
                              r"with seasonality, and DCS/UC - ST. One vs two time-varying parameters.",
                              "tab:B_tw98_tvp_comparison"),
        t_tw_seasonality("tw95", "95", "twCRPS95 with no seasonal lags included versus seasonality added.",
                         "tab:A2_tw95_seasonality_comparison"),
        t_tw_seasonality("tw98", "98", "twCRPS98 with no seasonal lags included versus seasonality added.",
                         "tab:A2_tw98_seasonality_comparison"),
        t_tw_ls_vs_seasonal(),
        t_enso(),
        t14, t15, t16,
        t_bss(),
    ] + [f() for f in EXTRA_TABLES]
    (REP / "tables.tex").write_text("\n\n".join(tables), encoding="utf-8")

    intro = INTRO if INTRO is not None else (
        r"\section*{What changed and why}"
        r"The Python Fisher-information cross term of the GB2 log-link distribution had the wrong sign "
        r"and denominator ($-pab/(a+b+1)$ instead of $+pab/(a+b)$; proof in "
        r"\texttt{reports/fisher/fisher\_gb2.pdf}). Fits, gradients and in-sample metrics used the "
        r"correct Numba kernel and are unchanged. The out-of-sample replay (\texttt{simulate\_oos}) and the "
        r"multi-step simulator used the Python path, so only out-of-sample metrics of models with "
        r"\emph{two} time-varying parameters and \emph{full inverse Fisher} scaling were wrong. Those 92 "
        r"candidates were recomputed from their cached parameters (no refit); every other number is read "
        r"from cache. Selection rules are those of the benchmark document; a reselection below is caused "
        r"only by corrected out-of-sample metrics. Each table ends with a note listing reselected cells "
        r"and recomputed values (old $\to$ new). ``div'' marks a non-finite metric. The benchmark had "
        r"replaced a few finite but very large values by ``div'' by hand; here every finite value is shown.")
    pre = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,float}
\usepackage{parskip}
\emergencystretch=3em\sloppy
\begin{document}
\title{TITLE_PLACEHOLDER}\author{}\date{1 October 2026}\maketitle
"""
    pre = pre.replace("TITLE_PLACEHOLDER", TITLE)
    doc = (pre + intro + "\n\n\\input{tables.tex}\n\n\\section*{Reference: selected configurations}\n"
           + t_selected_models() + "\n\\end{document}\n")
    (REP / "report.tex").write_text(doc, encoding="utf-8")
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"],
                           cwd=REP, capture_output=True, text=True)
    log = (REP / "report.log").read_text(errors="ignore")
    print("report.pdf:", "OK" if (REP / "report.pdf").exists() and "Fatal error" not in log else "FAILED")


if __name__ == "__main__":
    build()
