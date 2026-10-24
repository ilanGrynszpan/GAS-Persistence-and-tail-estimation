"""
2026-10-01: the thesis result tables (review/GAS(p,q) staged execution.tex)
regenerated from the full staged rerun with the phi-linked joint occurrence
dynamics -> reports/regeneratedwithpi01102026/.

EXECUTION FLOW
    1. run_rerun_pi_01102026.py (run first) refits every Stage 1/2/3/4/4S
       configuration and Stage 5 with pi_t = logistic(lambda0 + lambda1 phi_t)
       estimated jointly (mixed-density score and Fisher scalings).
    2. build_metrics() collects artifacts/rerun_pi_01102026/**/result.json
       into artifacts/rerun_pi_01102026/metrics.json, in the layout the
       selection code expects (metrics[short][stage][cfg][branch]); the
       per-observation convergence fraction comes from the gradient vector
       stored by the rerun.
    3. The table code of generate_regenerated_01102026_report.py is reused
       unchanged, with two views:
         old -- the Fisher-corrected 2026-09 results (AR pi; the
                reports/regenerated01102026 tables),
         new -- this rerun (models rebuilt with run_rerun_pi_01102026.build_model
                wherever a table recomputes from a model: Kupiec, 90-day, ENSO).
       Every value changes (all models are refitted), so the note under each
       table lists the reselections (old -> new configuration).
    4. Stage 5 is reported separately (generate_stage5_011026_report.py ->
       reports/stage5_011026); t_stage5() below builds its table.

No fitting happens here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_ml_comparison_report as G
import collect_stage_comparison_metrics as C
import generate_regenerated_01102026_report as RG
import run_rerun_pi_01102026 as R

NEW_TREE = R.OUT
OLD_REGEN = ROOT / "artifacts" / "regenerated_01102026"
REP = ROOT / "reports" / "regeneratedwithpi01102026"
STAGE_COV = {"stage2": [f"{lk}_{cs}" for lk in R.GAS_LAG_SETS for cs in R.STAGE2_COV],
             "stage3": R.HARVEY_COV, "stage4": R.HARVEY_COV, "stage4S": R.HARVEY_COV}


# ---------------------------------------------------------------------------
# metrics.json for the new tree
# ---------------------------------------------------------------------------

def _cell(res: dict) -> dict:
    return {"grad_inf": res.get("grad_norm_inf_numeric"), "frac_conv": res.get("frac_conv"),
            "n_params": len(res.get("grad_vec", [])), "rmse_oos": res.get("rmse"),
            "rmse_is": res.get("rmse_insample"), "crps_oos": res.get("crps_mean"),
            "crps_is": res.get("crps_insample"), "tw95": res.get("twcrps_95"),
            "tw98": res.get("twcrps_98"), "validity": res.get("validity"), "success": res.get("success")}


FALLBACK = set()   # {(short, stage, cfg, branch)} served by the 2026-09 (AR pi) fit


def _ok(cell) -> bool:
    if not cell:
        return False
    g, c = cell.get("grad_inf"), cell.get("crps_oos")
    return g is not None and np.isfinite(g) and g < 100 and c is not None and np.isfinite(c)


def apply_fallback(met: dict) -> list:
    """
    A new-run configuration that failed (fit crashed, or its out-of-sample
    forecast diverged: non-finite CRPS) is replaced by the Fisher-corrected
    2026-09 fit of the same configuration (AR occurrence dynamics), provided
    that fit is acceptable (checked gradient < 100, finite OOS metrics). Its
    metric values are wrapped as RG.Flagged so every table prints them with
    a trailing '*'.
    """
    used = []
    for short, stages in OLD_MET.items():
        for stage, cfgs in stages.items():
            for cfg, brs in cfgs.items():
                for br, old in brs.items():
                    new = met.get(short, {}).get(stage, {}).get(cfg, {}).get(br)
                    new_failed = (new is None and _expected(short, stage, cfg, br)) or (
                        new is not None and not np.isfinite(new.get("crps_oos") or np.nan))
                    if new_failed and _ok(old):
                        cell = {k: (RG.Flagged(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                                    and k in ("rmse_oos", "crps_oos", "tw95", "tw98") else v)
                                for k, v in old.items()}
                        cell["fallback_sept"] = True
                        met.setdefault(short, {}).setdefault(stage, {}).setdefault(cfg, {})[br] = cell
                        FALLBACK.add((short, stage, cfg, br))
                        used.append((short, stage, cfg, br))
    return used


def _expected(short, stage, cfg, br) -> bool:
    """True if the new run attempted this configuration (a fit_error.txt or result.json exists)."""
    if stage == "stage1":
        return False
    d = NEW_TREE / short / stage / (cfg + R._sfx(br))
    return (d / "fit_error.txt").exists() or (d / "result.json").exists()


def build_metrics() -> dict:
    met = {}
    for short in G.SHORTS:
        for br in R.BRANCHES:
            sfx = R._sfx(br)
            ch = json.loads((NEW_TREE / short / "stage1" / f"_scaling_choice{sfx}.json").read_text())
            for lk in R.GAS_LAG_SETS:
                sc = ch["per_lag"].get(lk, {}).get("scaling")
                p = NEW_TREE / short / "stage1" / f"{lk}_{sc}{sfx}" / "result.json"
                if sc and p.exists():
                    met.setdefault(short, {}).setdefault("stage1", {}).setdefault(lk, {})[br] = \
                        _cell(json.loads(p.read_text()))
            for st, cfgs in STAGE_COV.items():
                for cfg in cfgs:
                    p = NEW_TREE / short / st / f"{cfg}{sfx}" / "result.json"
                    if p.exists():
                        met.setdefault(short, {}).setdefault(st, {}).setdefault(cfg, {})[br] = \
                            _cell(json.loads(p.read_text()))
    used = apply_fallback(met)
    (NEW_TREE / "metrics.json").write_text(json.dumps(met, indent=2, default=float))
    print("fallback (2026-09 fit used):", used)
    return met


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------

_G_RESOLVE = G._resolve_cfg_dir          # original 2026-09 trees
_C_BUILD = C.build_model
OLD_MET = json.loads((OLD_REGEN / "metrics.json").read_text())


def old_resolve(short, ident):
    d = _G_RESOLVE(short, ident)
    alt = OLD_REGEN / d.relative_to(ROOT / "artifacts")
    return alt if (alt / "result.json").exists() else d


def new_resolve(short, ident):
    stage, cfg, branch = ident
    if (short, stage, cfg, branch) in FALLBACK:
        return old_resolve(short, ident)
    sfx = R._sfx(branch)
    if stage == "stage1":
        ch = json.loads((NEW_TREE / short / "stage1" / f"_scaling_choice{sfx}.json").read_text())
        cfg = f"{cfg}_{ch['per_lag'][cfg]['scaling']}"
    return NEW_TREE / short / stage / (cfg + sfx)


def new_build(loc_full, stage, cfg, branch, res, data):
    if res.get("pi_dynamics") != "phi_linked_joint":     # a 2026-09 fallback fit (AR pi)
        return _C_BUILD(loc_full, stage, cfg, branch, res, data)
    model, fk, _ = R.build_model(stage, branch, res.get("scaling", "unit"), cfg, data)
    return model, fk


@contextmanager
def view(which):
    new = which == "new"
    G.MET = NEW_MET if new else OLD_MET
    G._resolve_cfg_dir = new_resolve if new else old_resolve
    C.build_model = new_build if new else _C_BUILD
    try:
        yield
    finally:
        G.MET, G._resolve_cfg_dir, C.build_model = OLD_MET, old_resolve, _C_BUILD


# ---------------------------------------------------------------------------
# Stage 5 table
# ---------------------------------------------------------------------------

def t_stage5() -> str:
    rows = []
    for s in G.SHORTS:
        old = json.loads((ROOT / "artifacts" / "rerun_10092026_phiL_sigmoid" / s / "stage5" / "best"
                          / "result.json").read_text())
        p = NEW_TREE / s / "stage5" / "best" / "result.json"
        new = json.loads(p.read_text()) if p.exists() else {}
        base = new.get("base_stage", "--")
        base_txt = (f"{RG.CLASS_LBL.get(base, base)} {str(new.get('cov', '')).replace('_', '-')} "
                    f"({RG.BRANCH_SHORT.get(new.get('base_branch'), '?')})") if new else "--"
        tw = RG.bold_min([new.get("twcrps_95"), old.get("twcrps_95")], 3)
        tw8 = RG.bold_min([new.get("twcrps_98"), old.get("twcrps_98")], 3)
        rows.append(f"{RG.DISP[s]} & {base_txt} & {tw[0]} & {tw8[0]} & {tw[1]} & {tw8[1]} \\\\")
    note = RG.Notes()
    note.other.append(r"Stage 5 fits $\varphi$ and $\xi$ with long--short/seasonal dynamics on the best "
                      r"Stage 3/4/4S configuration (gradient tier, then CRPS, then RMSE), with the "
                      r"$\varphi$-only branch's scaling. ``Previous'' = the 2026-09 Stage 5 fit (AR "
                      r"occurrence dynamics), whose base configuration may differ. Bold: lower of the two.")
    hdr = (r"& & \multicolumn{2}{c}{$\varphi$-linked $\pi$ (new)} & \multicolumn{2}{c}{previous} \\"
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}" "\n"
           r"Location & Base configuration & twCRPS95 & twCRPS98 & twCRPS95 & twCRPS98 \\")
    return RG.table("Stage 5: rolling one-step-ahead out-of-sample twCRPS95 and twCRPS98.",
                    "tab:stage5_twcrps", "llrrrr", hdr, rows, note)


INTRO = (
    r"\section*{What changed}"
    r"Every model of every stage was refitted with the occurrence dynamics selected in "
    r"\texttt{reports/newpi} (alternative D, joint): $\pi_{t|t-1}=\operatorname{logistic}(\lambda_0+"
    r"\lambda_1\varphi_{t|t-1})$, where $\varphi_{t|t-1}$ is the model's own magnitude state, estimated "
    r"jointly with all other parameters on the training series. Because $\pi_t$ depends on $\varphi_t$, "
    r"the $\varphi$ score is that of the zero-adjusted density, $s_\varphi=\mathbb 1\{y>0\}\partial\log g/"
    r"\partial\varphi+\lambda_1(\mathbb 1\{y>0\}-\pi_t)$, and the Fisher scalings tested in Stage 1 are "
    r"those of the mixed density: unit, diagonal and full inverse of "
    r"$\begin{pmatrix}\pi\mathcal I_{\varphi\varphi}+\lambda_1^2\pi(1-\pi)&\pi\mathcal I_{\varphi\xi}"
    r"\\\pi\mathcal I_{\varphi\xi}&\pi\mathcal I_{\xi\xi}\end{pmatrix}$ (diagonal and full coincide "
    r"for one time-varying parameter). Starting values are the time-invariant ZA-GB2 maximum-likelihood "
    r"estimates. Lag sets, covariate sets, Harvey configurations, train/test split, selection rules and "
    r"metrics are those of the benchmark. All computation runs in the Numba kernels (Numba/Python "
    r"parity verified for every model family, branch and scaling). Notes under each table list the "
    r"cells whose selected configuration changed relative to the Fisher-corrected tables of "
    r"\texttt{reports/regenerated01102026}; \emph{every} value differs from those tables because every "
    r"model was refitted. ``div'' marks a non-finite metric. Values marked $^*$ come from the "
    r"Fisher-corrected 2026-09 fit of the same configuration, which uses the earlier zero "
    r"augmentation (autoregressive logistic occurrence dynamics on $y_{t-1},y_{t-2},y_{t-3}$): they "
    r"replace configurations whose new fit failed or whose out-of-sample forecast diverged, and are "
    r"used only when that earlier fit was acceptable (checked gradient $<100$, finite metrics)."
)


def reported_models(short) -> set:
    """Every (stage, cfg, branch) that appears in this report's other tables for `short`
    (RMSE/CRPS, twCRPS by branch and by seasonality, ML comparison/ENSO/BSS best overall,
    ENSO long-short). Call inside view("new")."""
    import generate_rmse_crps_report as RC
    import generate_twcrps_report as TW
    import generate_enso_metrics_report as EN
    used = set()
    for br in ("phi_only", "phi_xi"):
        for st in RG.STAGES5:
            used.add(RC.best_for_stage(short, st, br)[0])
        for fn in (TW.stage1_best, TW.stage2_best):
            used.add(fn(short, [br], TW.ALL_LAGS)[0])
        for st in ("stage3", "stage4", "stage4S"):
            used.add(TW.harvey_best(short, st, [br])[0])
    for lags in (TW.NO_SEAS_LAGS, TW.SEAS_LAGS):
        for fn in (TW.stage1_best, TW.stage2_best):
            used.add(fn(short, TW.BRANCHES, lags)[0])
    for st in ("stage3", "stage4", "stage4S"):
        used.add(TW.harvey_best(short, st, TW.BRANCHES)[0])
    used.add(G.best_overall_id(short)[0])
    used.add(EN.enso_longshort_best(short)[0])
    used.discard(None)
    return used


def kupiec_pick_reported(short):
    """Among the models already reported in the other tables for this location:
    best gradient tier, then lowest out-of-sample CRPS."""
    cands = [(i, G.m(short, *i)) for i in reported_models(short)]
    cands = [(i, mm) for i, mm in cands if mm is not None]
    crps = lambda mm: mm["crps_oos"] if np.isfinite(mm.get("crps_oos", np.nan)) else np.inf
    return min(cands, key=lambda im: (G.tier(im[1]), crps(im[1])))


def report_kupiec_alt():
    """'report kupiec': the three Kupiec tables with the benchmark rule
    (lowest twCRPS95 over every best-gradient-tier candidate)."""
    RG.KUPIEC_PICK_NEW, RG.KUPIEC_RULE_NOTE = None, (
        r"Model per location: lowest out-of-sample twCRPS95 among all candidates (every stage and "
        r"branch) in the best gradient tier reached -- the benchmark rule.")
    idents, notes = RG._tw95_idents()
    t14, t15, t16 = RG.t_kupiec_and_tail(idents, notes)
    rows = "\n".join(f"{RG.DISP[s]} & {RG.cfg_txt(i)} \\\\" for s, i in idents.items())
    models = ("\\begin{table}[H]\\centering\\small\\caption{Model used per location.}"
              "\\begin{tabular}{ll}\\toprule Location & Model \\\\ \\midrule\n" + rows +
              "\n\\bottomrule\\end{tabular}\\end{table}")
    doc = "\n\n".join([
        r"\documentclass[10pt,a4paper]{article}", r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}",
        r"\usepackage[margin=2cm]{geometry}", r"\usepackage{amsmath,amssymb,booktabs,longtable,float}",
        r"\usepackage{parskip}\emergencystretch=3em\sloppy", r"\begin{document}",
        r"\title{Exceedance ratios and Kupiec tests -- lowest-twCRPS95 models}\author{}\date{1 October 2026}\maketitle",
        models, t14, t15, t16, r"\end{document}"])
    _compile_as(doc, "report kupiec")


def _compile_as(doc: str, final_name: str) -> None:
    """Compile under a space-free temporary name, then rename (pdflatex is
    unreliable with spaces in file names)."""
    tmp = "build_tmp_" + final_name.replace(" ", "_")
    (REP / f"{tmp}.tex").write_text(doc, encoding="utf-8")
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{tmp}.tex"], cwd=REP,
                       capture_output=True, text=True)
    for ext in ("tex", "pdf"):
        if (REP / f"{tmp}.{ext}").exists():
            (REP / f"{tmp}.{ext}").replace(REP / f"{final_name}.{ext}")
    for ext in ("aux", "log", "out"):
        (REP / f"{tmp}.{ext}").unlink(missing_ok=True)
    print(f"{final_name}.pdf:", "OK" if (REP / f"{final_name}.pdf").exists() else "FAILED")


def report_christoffersen():
    """
    'report christoferssen': Christoffersen (1998) independence and
    conditional-coverage tests on the rolling one-step-ahead upper-tail
    violation sequences, for the same models as the main report's Kupiec
    tables (kupiec_pick_reported). The PIT sequence is produced by
    generate_kupiec_report.three_level_stats (model rebuild, Numba
    simulate_oos, compute_extended_oos_metrics' PIT), with the per-level
    statistic swapped for diagnostics.exceedance.christoffersen_from_pit.
    """
    import generate_kupiec_report as KU
    from diagnostics.exceedance import christoffersen_from_pit
    levels = (("95", 0.95, r"95\% quantile"), ("98", 0.98, r"98\% quantile"),
              ("1in30", KU.Q_1IN30, r"1-in-30-day ($q=1-1/30$)"))
    stats, idents = {}, {}
    orig = KU.exceedance_from_pit
    KU.exceedance_from_pit = christoffersen_from_pit
    try:
        with view("new"):
            for sh in G.SHORTS:
                ident, _ = kupiec_pick_reported(sh)
                res = json.loads((G._resolve_cfg_dir(sh, ident) / "result.json").read_text())
                idents[sh] = ident
                stats[sh] = KU.three_level_stats(sh, ident, res)
                print("  christoffersen", sh, ident, flush=True)
    finally:
        KU.exceedance_from_pit = orig

    def p_txt(p):
        if p is None or not np.isfinite(p):
            return "--"
        return f"{p:.3f}" + ("*" if p < 0.05 else "")

    tables = []
    for key, q, lbl in levels:
        rows = []
        for sh in G.SHORTS:
            c = stats[sh][key]
            rows.append(f"{RG.DISP[sh]} & {c['n_viol']} / {c['n']} & {RG.fmt(c['pi01'], 3)} & "
                        f"{RG.fmt(c['pi11'], 3) if np.isfinite(c['pi11']) else '--'} & {c['n11']} & "
                        f"{p_txt(c['p_ind'])} & {p_txt(c['p_cc'])} \\\\")
        note = RG.Notes()
        note.other.append(r"$\hat\pi_{01}$: violation rate after a day without violation; "
                          r"$\hat\pi_{11}$: after a day with one; $n_{11}$: consecutive violation pairs. "
                          r"$^*$ rejection at 5\%. ``--'': undefined (no violations, or none followed by "
                          r"another day).")
        hdr = (r"Location & Violations & $\hat\pi_{01}$ & $\hat\pi_{11}$ & $n_{11}$ & "
               r"$p_{\mathrm{ind}}$ & $p_{\mathrm{cc}}$ \\")
        tables.append(RG.table(f"Christoffersen tests, rolling one-step-ahead, {lbl}.",
                               f"tab:christoffersen_{key}", "lrrrrrr", hdr, rows, note,
                               size=r"\small", sep="6pt"))
    model_rows = "\n".join(f"{RG.DISP[sh]} & {RG.cfg_txt(i)} \\\\" for sh, i in idents.items())
    models = ("\\begin{table}[H]\\centering\\small\\caption{Model per location (as in the main "
              "report's Kupiec tables).}\\begin{tabular}{ll}\\toprule Location & Model \\\\ \\midrule\n"
              + model_rows + "\n\\bottomrule\\end{tabular}\\end{table}")
    method = (
        r"\section*{Method}"
        r"For each location's model (Table~1: among the models reported in the main tables, the best "
        r"gradient tier, then the lowest out-of-sample CRPS -- the same models as the main report's "
        r"Kupiec tables), the rolling one-step-ahead out-of-sample PIT values are computed exactly as "
        r"for the Kupiec test, and a violation is $I_t=\mathbb 1\{\mathrm{PIT}_t>q\}$: rainfall above "
        r"the model's own time-varying $q$-quantile. Kupiec tests only the \emph{number} of violations; "
        r"Christoffersen (1998) also tests whether they \emph{cluster}. With $n_{ij}$ the number of days "
        r"in state $j$ preceded by a day in state $i$,"
        r"\[\hat\pi_{01}=\frac{n_{01}}{n_{00}+n_{01}},\quad \hat\pi_{11}=\frac{n_{11}}{n_{10}+n_{11}},"
        r"\quad \hat\pi=\frac{n_{01}+n_{11}}{n_{00}+n_{01}+n_{10}+n_{11}},\]"
        r"\[LR_{\mathrm{ind}}=-2\log\frac{(1-\hat\pi)^{n_{00}+n_{10}}\hat\pi^{\,n_{01}+n_{11}}}"
        r"{(1-\hat\pi_{01})^{n_{00}}\hat\pi_{01}^{\,n_{01}}(1-\hat\pi_{11})^{n_{10}}\hat\pi_{11}^{\,n_{11}}}"
        r"\sim\chi^2_1,\qquad LR_{\mathrm{cc}}=LR_{\mathrm{uc}}+LR_{\mathrm{ind}}\sim\chi^2_2,\]"
        r"with $LR_{\mathrm{uc}}$ the Kupiec statistic (nominal rate $1-q$) and $0\log0=0$. Under "
        r"independence $\pi_{01}=\pi_{11}$; extremes that come in clusters give "
        r"$\hat\pi_{11}>\hat\pi_{01}$ and a small $p_{\mathrm{ind}}$. $p_{\mathrm{cc}}$ jointly tests the "
        r"right number of violations and their independence. With about 730 out-of-sample days and few "
        r"consecutive violations at the 98\% level, the independence test has little power there.")
    doc = "\n\n".join([
        r"\documentclass[10pt,a4paper]{article}", r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}",
        r"\usepackage[margin=2cm]{geometry}", r"\usepackage{amsmath,amssymb,booktabs,longtable,float}",
        r"\usepackage{parskip}\emergencystretch=3em\sloppy", r"\begin{document}",
        r"\title{Clustering of extremes: Christoffersen tests}\author{}\date{1 October 2026}\maketitle",
        method, models] + tables + [r"\end{document}"])
    _compile_as(doc, "report christoferssen")


def setup():
    global NEW_MET
    NEW_MET = build_metrics()
    RG.view = view
    RG.recomputed = lambda short, ident: False
    RG._resolve_new = new_resolve
    RG.REP = REP
    RG.LIST_CHANGED = False
    RG.TITLE = r"GAS/SD empirical results -- $\varphi$-linked occurrence dynamics"
    RG.INTRO = INTRO
    RG.EXTRA_TABLES = []   # Stage 5 has its own report: generate_stage5_011026_report.py


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of: main,kupiec,christoffersen")
    a = ap.parse_args()
    want = set(a.only.split(",")) if a.only else {"main", "kupiec", "christoffersen"}
    setup()
    if "main" in want:
        RG.KUPIEC_PICK_NEW = kupiec_pick_reported
        RG.KUPIEC_RULE_NOTE = (
            r"Model per location: among the models already reported in the tables above for that "
            r"location, the best gradient tier, then the lowest out-of-sample CRPS (the benchmark rule "
            r"-- lowest twCRPS95 over all candidates -- is kept in \texttt{report kupiec.pdf}).")
        RG.build()
    if "kupiec" in want:
        report_kupiec_alt()
    if "christoffersen" in want:
        report_christoffersen()


NEW_MET = {}

if __name__ == "__main__":
    main()
