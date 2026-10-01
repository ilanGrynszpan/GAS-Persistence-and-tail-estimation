"""
Report for the 2026-09-10 logistic-link staged rerun (pi = 1/(1+e^{-eta})).

Reads ONLY artifacts/rerun_10092026/ (docs/REPORTING.md: no fitting here).
Produces:
  reports/rerun10092026/report.tex / report.pdf
  reports/rerun10092026/tables.tex           (all tables, standalone)
  reports/rerun10092026/figures/*.pdf/.png   (ACF + QQ mosaics)

Per stage:
  - up-front comparison table (lag sets, or covariate sets, or -- for the
    Harvey stages -- a stage-vs-stage gradient-norm comparison)
  - 400-lag in-sample ACF mosaic of the quantile residuals, locations x
    (lag set | covariate config)
  - QQ-plot mosaic of the quantile residuals, same layout
  - 1-step OOS table  (RMSE / CRPS / twCRPS95 / twCRPS98)
  - multi-step OOS table (h in 1,2,5,10,30,60,90)
  - Brier skill score table
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data.station_loader import STATION_REGISTRY

ART = ROOT / "artifacts" / "rerun_10092026"
REP = ROOT / "reports" / "rerun10092026"
FIG = REP / "figures"

LOCS = ["BELO HORIZONTE", "CRUZEIRO DO SUL (ACRE)", "DARWIN AIRPORT",
        "GARANHUNS (PERNAMBUCO)", "MANAUS", "SALVADOR"]
SHORTS = [STATION_REGISTRY[l].get("short", l) for l in LOCS]
DISP = {STATION_REGISTRY[l].get("short", l): STATION_REGISTRY[l].get("display", l) for l in LOCS}

GAS_LAG_SETS = ["L1", "L123", "L1_365_366", "L123_365_366"]
LAG_LABEL = {"L1": r"$\{1\}$", "L123": r"$\{1,2,3\}$",
             "L1_365_366": r"$\{1,365,366\}$", "L123_365_366": r"$\{1,2,3,365,366\}$"}
S2_COV = ["dew", "dew_temp", "dew_temp_enso"]
S2_LBL = {"dew": "dew", "dew_temp": "dew+temp", "dew_temp_enso": "dew+temp+ENSO"}
S2_COV_LBL = {"dew": "dew", "dew\\_temp": "dew+temp", "dew_temp": "dew+temp",
              "dew_temp_enso": "dew+temp+ENSO"}
HARVEY_COV = ["none", "enso", "dew", "dew_enso", "dew_temp_enso"]
HC_LBL = {"none": "none", "enso": "ENSO", "dew": "dew", "dew_enso": "dew+ENSO",
          "dew_temp_enso": "dew+temp+ENSO"}
HORIZONS = [1, 2, 5, 10, 30, 60, 90]

PRE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage[margin=1.6cm,landscape]{geometry}
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
        return "div"
    if abs(x) >= 1e4:
        return rf"$\sim\!10^{{{int(np.floor(np.log10(abs(x))))}}}$"
    return f"{x:.{d}f}"


_CUR_BRANCH = "phi_only"

def _bsfx():
    return "" if _CUR_BRANCH == "phi_only" else "__xi"


_SCAL_CACHE: dict = {}

def _resolve_cfg(short, stage, cfg):
    """Stage 1 is cached per (lag x Fisher scaling); the report addresses it
    by bare lag key ('L1', 'L123_365_366', ...).  Map that to the concrete
    '<lag>_<chosen_scaling>' dir recorded in _scaling_choice{__xi}.json for
    the current branch.  Every other stage is already a real dir name."""
    if stage != "stage1" or cfg not in GAS_LAG_SETS:
        return cfg
    key = (short, _bsfx())
    if key not in _SCAL_CACHE:
        p = ART / short / "stage1" / f"_scaling_choice{_bsfx()}.json"
        _SCAL_CACHE[key] = (json.loads(p.read_text()).get("chosen_scaling")
                            if p.exists() else None)
    scal = _SCAL_CACHE[key]
    return f"{cfg}_{scal}" if scal else cfg


def load(short, stage, cfg):
    p = ART / short / stage / (_resolve_cfg(short, stage, cfg) + _bsfx()) / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_qr(short, stage, cfg):
    p = ART / short / stage / (_resolve_cfg(short, stage, cfg) + _bsfx()) / "insample_qr.npy"
    if not p.exists():
        return np.array([])
    return np.load(p)


def load_ms(short, stage, cfg):
    p = ART / short / stage / (_resolve_cfg(short, stage, cfg) + _bsfx()) / "multistep.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df if "horizon" in df.columns else None


def acf(x, nlags):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < nlags + 5:
        nlags = max(1, len(x) - 5)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return np.zeros(nlags + 1)
    out = np.empty(nlags + 1)
    for k in range(nlags + 1):
        out[k] = np.dot(x[: len(x) - k], x[k:]) / denom
    return out


# ---------------------------------------------------------------------------
# mosaics
# ---------------------------------------------------------------------------

def load_br(short, stage, cfg, branch):
    sfx = "" if branch == "phi_only" else "__xi"
    p = ART / short / stage / (cfg + sfx) / "result.json"
    import json as _j
    return _j.loads(p.read_text()) if p.exists() else None


def _s1_perlag(sh, sfx):
    """Stage-1 per-lag metrics (chosen Fisher scaling) live in the
    _scaling_choice json, not in a plain <lag> config dir."""
    p = ART / sh / "stage1" / f"_scaling_choice{sfx}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("per_lag", {}) or {}


def tvp_compare_table(stage, cfgs, cfg_labels, caption, label):
    rows = []
    cols = []
    for cl in cfg_labels:
        cols += [cl + " 1TVP RMSE", cl + " 2TVP RMSE", cl + " 1TVP grad", cl + " 2TVP grad"]
    for sh in SHORTS:
        row = {"Location": DISP[sh]}
        s1_1 = _s1_perlag(sh, "") if stage == "stage1" else {}
        s1_2 = _s1_perlag(sh, "__xi") if stage == "stage1" else {}
        for cfg, cl in zip(cfgs, cfg_labels):
            if stage == "stage1":
                r1 = s1_1.get(cfg, {}) or {}
                r2 = s1_2.get(cfg, {}) or {}
            else:
                r1 = load_br(sh, stage, cfg, "phi_only") or {}
                r2 = load_br(sh, stage, cfg, "phi_xi") or {}
            row[cl + " 1TVP RMSE"] = _f(r1.get("rmse"), 2)
            row[cl + " 2TVP RMSE"] = _f(r2.get("rmse"), 2)
            row[cl + " 1TVP grad"] = _f(r1.get("grad_norm_inf_numeric"), 2)
            row[cl + " 2TVP grad"] = _f(r2.get("grad_norm_inf_numeric"), 2)
        rows.append(row)
    groups = [(cl, 4) for cl in cfg_labels]
    leaf_labels = ["1TVP RMSE", "2TVP RMSE", "1TVP grad", "2TVP grad"] * len(cfg_labels)
    return tbl(rows, cols, caption, label, leaf_labels=leaf_labels, groups=groups)


def acf_mosaic(stage, cfgs, cfg_labels, fname, nlags=400):
    nrow, ncol = len(SHORTS), len(cfgs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 1.9 * nrow), squeeze=False)
    ci = 1.96 / np.sqrt(3000)
    for r, s in enumerate(SHORTS):
        for c, cfg in enumerate(cfgs):
            ax = axes[r][c]
            qr = load_qr(s, stage, cfg)
            if len(qr) > 20:
                a = acf(qr, nlags)
                ax.bar(range(1, len(a)), a[1:], width=1.0, color="#3b6", linewidth=0)
                ax.axhline(ci, color="grey", lw=0.5, ls="--")
                ax.axhline(-ci, color="grey", lw=0.5, ls="--")
                ax.set_ylim(-0.15, 0.25)
            else:
                ax.text(0.5, 0.5, "no fit", ha="center", va="center", transform=ax.transAxes)
            if r == 0:
                ax.set_title(cfg_labels[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(DISP[s][:12], fontsize=7)
            ax.tick_params(labelsize=5)
    fig.suptitle(f"{stage}: 400-lag ACF of in-sample quantile residuals", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{fname}.pdf"); fig.savefig(FIG / f"{fname}.png", dpi=110)
    plt.close(fig)


def qq_mosaic(stage, cfgs, cfg_labels, fname):
    nrow, ncol = len(SHORTS), len(cfgs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 1.9 * nrow), squeeze=False)
    for r, s in enumerate(SHORTS):
        for c, cfg in enumerate(cfgs):
            ax = axes[r][c]
            qr = load_qr(s, stage, cfg)
            qr = qr[np.isfinite(qr)]
            if len(qr) > 20:
                stats.probplot(qr, dist="norm", plot=ax)
                ax.get_lines()[0].set_markersize(1.5)
                ax.get_lines()[0].set_alpha(0.4)
                ax.get_lines()[1].set_color("red"); ax.get_lines()[1].set_lw(0.8)
                ax.set_title("")
            else:
                ax.text(0.5, 0.5, "no fit", ha="center", va="center", transform=ax.transAxes)
            if r == 0:
                ax.set_title(cfg_labels[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(DISP[s][:12], fontsize=7)
            ax.set_xlabel(""); ax.tick_params(labelsize=5)
    fig.suptitle(f"{stage}: QQ-plot of in-sample quantile residuals vs N(0,1)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{fname}.pdf"); fig.savefig(FIG / f"{fname}.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

# Max numeric ("leaf") columns rendered in a single table before it is split
# into several side-by-side-in-the-source (one-below-the-other-on-the-page)
# tables, so wide sweeps (many lag sets / covariate configs x several
# metrics) never overflow the landscape page width.
MAX_LEAF_COLS = 12


def tbl(rows, cols, caption, label, leaf_labels=None, groups=None):
    """Render `rows` (list of dicts, already-formatted string values) as one
    or more LaTeX tables.

    cols        : ordered list of the (unique) dict keys to show as columns.
    leaf_labels : header text for each column (defaults to `cols` itself).
    groups      : optional [(group_label_or_None, n_leaf_cols), ...] that
                  partitions `cols`/`leaf_labels` in order. When given, a
                  2-row header is rendered (group label spanning its leaf
                  columns via \\multicolumn, e.g. the covariate/lag-set name,
                  then the short per-metric labels below it) and the table is
                  split at group boundaries -- never splitting one group --
                  whenever the running leaf-column count would exceed
                  MAX_LEAF_COLS. This keeps headers short (the long lag/cov
                  label is written once per group, not once per metric) and
                  keeps every rendered table within the page width.
    """
    df = pd.DataFrame(rows).set_index("Location")
    df = df[cols]
    if leaf_labels is None:
        leaf_labels = list(cols)
    if groups is None:
        groups = [(None, 1) for _ in cols]
    assert sum(n for _, n in groups) == len(cols), "groups must partition cols"

    chunks, cur, cur_n = [], [], 0
    for g in groups:
        if cur and cur_n + g[1] > MAX_LEAF_COLS:
            chunks.append(cur); cur, cur_n = [], 0
        cur.append(g); cur_n += g[1]
    if cur:
        chunks.append(cur)

    tables, ptr = [], 0
    for ci, chunk in enumerate(chunks):
        n_leaf = sum(n for _, n in chunk)
        c_cols = cols[ptr: ptr + n_leaf]
        c_labels = leaf_labels[ptr: ptr + n_leaf]
        ptr += n_leaf
        fs = r"\footnotesize" if n_leaf <= 8 else r"\scriptsize"
        cap = caption if len(chunks) == 1 else f"{caption} (part {ci + 1}/{len(chunks)})"
        lab = label if len(chunks) == 1 else f"{label}_p{ci + 1}"
        body = [rf"\begin{{table}}[H]\centering{fs}\setlength{{\tabcolsep}}{{3pt}}",
                rf"\caption{{{cap}}}\label{{{lab}}}",
                r"\begin{tabular}{l" + "r" * n_leaf + "}", r"\toprule"]
        if any(g[0] for g in chunk):
            hdr, cline, pos = [""], [], 2
            for glabel, gn in chunk:
                hdr.append(rf"\multicolumn{{{gn}}}{{c}}{{{glabel}}}" if glabel else "")
                if glabel:
                    cline.append(rf"\cmidrule(lr){{{pos}-{pos + gn - 1}}}")
                pos += gn
            body += [" & ".join(hdr) + r" \\", " ".join(cline)]
        body.append("Location & " + " & ".join(c_labels) + r" \\")
        body.append(r"\midrule")
        for idx, row in df.iterrows():
            body.append(str(idx) + " & " + " & ".join(str(row[c]) for c in c_cols) + r" \\")
        body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        tables.append("\n".join(body))
    return "\n\n".join(tables)


def onestep_table(stage, cfgs, cfg_labels, caption, label):
    rows = []
    cols = []
    for cl in cfg_labels:
        cols += [f"{cl} RMSE", f"{cl} CRPS", f"{cl} tw95", f"{cl} tw98"]
    for s in SHORTS:
        row = {"Location": DISP[s]}
        for cfg, cl in zip(cfgs, cfg_labels):
            r = load(s, stage, cfg) or {}
            row[f"{cl} RMSE"] = _f(r.get("rmse"), 2)
            row[f"{cl} CRPS"] = _f(r.get("crps_mean"), 2)
            row[f"{cl} tw95"] = _f(r.get("twcrps_95"))
            row[f"{cl} tw98"] = _f(r.get("twcrps_98"))
        rows.append(row)
    groups = [(cl, 4) for cl in cfg_labels]
    leaf_labels = ["RMSE", "CRPS", "tw95", "tw98"] * len(cfg_labels)
    return tbl(rows, cols, caption, label, leaf_labels=leaf_labels, groups=groups)


def bss_table(stage, cfgs, cfg_labels, caption, label):
    rows = []
    for s in SHORTS:
        row = {"Location": DISP[s]}
        for cfg, cl in zip(cfgs, cfg_labels):
            r = load(s, stage, cfg) or {}
            row[cl] = _f(r.get("BSS"))
        rows.append(row)
    return tbl(rows, list(cfg_labels), caption, label)


def multistep_table(stage, cfg, cfg_label, caption, label):
    rows = []
    cols = ["RMSE", "CRPS", "tw95", "tw98", "exc95", "kup95p", "exc98", "kup98p"]
    for s in SHORTS:
        ms = load_ms(s, stage, cfg)
        for h in HORIZONS:
            row = {"Location": f"{DISP[s]} h={h}"}
            if ms is not None and (ms["horizon"] == h).any():
                m = ms[ms["horizon"] == h].iloc[0]
                row["RMSE"] = _f(m.get("rmse"), 2); row["CRPS"] = _f(m.get("crps"), 2)
                row["tw95"] = _f(m.get("twcrps_95")); row["tw98"] = _f(m.get("twcrps_98"))
                row["exc95"] = _f(m.get("exc_95")); row["kup95p"] = _f(m.get("kup95_p"))
                row["exc98"] = _f(m.get("exc_98")); row["kup98p"] = _f(m.get("kup98_p"))
            else:
                for c in cols:
                    row[c] = "--"
            rows.append(row)
    return tbl(rows, cols, caption + rf" (config: {cfg_label}); exc = exceedance rate of the "
              r"time-varying 95/98\% quantile, kupNNp = Kupiec p-value.", label)


def exceedance_1s_table(stage, cfgs, cfg_labels, caption, label):
    rows = []
    cols = []
    for cl in cfg_labels:
        cols += [cl + " exc95", cl + " kup95p", cl + " exc98", cl + " kup98p"]
    for s in SHORTS:
        row = {"Location": DISP[s]}
        for cfg, cl in zip(cfgs, cfg_labels):
            r = load(s, stage, cfg) or {}
            row[cl + " exc95"] = _f(r.get("exc_95_1s"))
            row[cl + " kup95p"] = _f(r.get("kup95_1s_p"))
            row[cl + " exc98"] = _f(r.get("exc_98_1s"))
            row[cl + " kup98p"] = _f(r.get("kup98_1s_p"))
        rows.append(row)
    groups = [(cl, 4) for cl in cfg_labels]
    leaf_labels = ["exc95", "kup95p", "exc98", "kup98p"] * len(cfg_labels)
    return tbl(rows, cols, caption, label, leaf_labels=leaf_labels, groups=groups)


def is_oos_table(stage, cfgs, cfg_labels, caption, label):
    rows = []
    cols = []
    for cl in cfg_labels:
        cols += [cl + " RMSE is", cl + " RMSE oos", cl + " CRPS is", cl + " CRPS oos"]
    for s in SHORTS:
        row = {"Location": DISP[s]}
        for cfg, cl in zip(cfgs, cfg_labels):
            r = load(s, stage, cfg) or {}
            row[cl + " RMSE is"] = _f(r.get("rmse_insample"), 2)
            row[cl + " RMSE oos"] = _f(r.get("rmse"), 2)
            row[cl + " CRPS is"] = _f(r.get("crps_insample"), 2)
            row[cl + " CRPS oos"] = _f(r.get("crps_mean"), 2)
        rows.append(row)
    groups = [(cl, 4) for cl in cfg_labels]
    leaf_labels = ["RMSE is", "RMSE oos", "CRPS is", "CRPS oos"] * len(cfg_labels)
    return tbl(rows, cols, caption, label, leaf_labels=leaf_labels, groups=groups)


def grad_compare_table(stages, cfgs, caption, label):
    """stage x config gradient-norm grid, one row per location x config."""
    rows = []
    for s in SHORTS:
        for cfg in cfgs:
            row = {"Location": f"{DISP[s]} / {HC_LBL.get(cfg, cfg).replace(chr(95), chr(92)+chr(95))}"}
            for st in stages:
                r = load(s, st, cfg) or {}
                g = r.get("grad_norm_inf_numeric")
                mark = "$^{\\dagger}$" if r.get("grad_converged") is False else ""
                row[st] = (_f(g) + mark) if g is not None else "--"
            rows.append(row)
    return tbl(rows, list(stages), caption, label)


def lag_compare_table(stage, caption, label, cov=None):
    """Stage 1/2: RMSE + gradient across the 4 gas-lag sets."""
    rows = []
    for s in SHORTS:
        row = {"Location": DISP[s]}
        for lk in GAS_LAG_SETS:
            cfg = lk if stage == "stage1" else f"{lk}_{cov}"
            r = load(s, stage, cfg) or {}
            row[f"{LAG_LABEL[lk]} RMSE"] = _f(r.get("rmse"), 2)
            row[f"{LAG_LABEL[lk]} grad"] = _f(r.get("grad_norm_inf_numeric"), 2)
        rows.append(row)
    cols = []
    for lk in GAS_LAG_SETS:
        cols += [f"{LAG_LABEL[lk]} RMSE", f"{LAG_LABEL[lk]} grad"]
    groups = [(LAG_LABEL[lk], 2) for lk in GAS_LAG_SETS]
    leaf_labels = ["RMSE", "grad"] * len(GAS_LAG_SETS)
    return tbl(rows, cols, caption, label, leaf_labels=leaf_labels, groups=groups)


def scaling_choice_table():
    rows = []
    for s in SHORTS:
        p = ART / s / "stage1" / "_scaling_choice.json"
        row = {"Location": DISP[s]}
        if p.exists():
            c = json.loads(p.read_text())
            row["chosen scaling"] = c.get("chosen_scaling", "--").replace("_", "\\_")
            row["best lag set"] = LAG_LABEL.get(c.get("best_lag_key", ""), "--")
        else:
            row["chosen scaling"] = "--"; row["best lag set"] = "--"
        rows.append(row)
    return tbl(rows, ["chosen scaling", "best lag set"],
               "Stage 1 per-location Fisher-scaling selection (gradient tier, then AIC).",
               "tab:scaling")


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------



def _set_branch(b):
    globals()["_CUR_BRANCH"] = b


def _content(tag, brname):
    """All per-stage sections+tables for one branch (tag: '1tvp' | '2tvp')."""
    tables, sections = [], []
    B = f" [{brname}]"

    # Stage 1
    acf_mosaic("stage1", GAS_LAG_SETS, [LAG_LABEL[k] for k in GAS_LAG_SETS], f"acf_stage1_{tag}")
    qq_mosaic("stage1", GAS_LAG_SETS, [LAG_LABEL[k] for k in GAS_LAG_SETS], f"qq_stage1_{tag}")
    t_lag1 = lag_compare_table("stage1", f"Stage 1{B}: RMSE and gradient by GAS-lag set.", f"tab:s1lag_{tag}")
    t_1s = onestep_table("stage1", GAS_LAG_SETS, [LAG_LABEL[k] for k in GAS_LAG_SETS],
                         f"Stage 1{B}: 1-step OOS RMSE / CRPS / twCRPS95 / twCRPS98.", f"tab:s1_1s_{tag}")
    t_bss1 = bss_table("stage1", GAS_LAG_SETS, [LAG_LABEL[k] for k in GAS_LAG_SETS],
                       f"Stage 1{B}: Brier skill score of $\\pi_{{t|t-1}}$ vs monthly climatology.", f"tab:s1bss_{tag}")
    ms1 = [multistep_table("stage1", lk, LAG_LABEL[lk], f"Stage 1{B}: multi-step OOS", f"tab:s1ms_{tag}_{lk}")
           for lk in GAS_LAG_SETS]
    tables += [t_lag1, t_1s, t_bss1] + ms1
    sections.append("\n\n".join([
        rf"\section{{Stage 1 -- no covariates{B}}}", t_lag1,
        rf"\begin{{figure}}[H]\centering\includegraphics[width=\textwidth]{{figures/acf_stage1_{tag}.pdf}}\end{{figure}}",
        rf"\begin{{figure}}[H]\centering\includegraphics[width=\textwidth]{{figures/qq_stage1_{tag}.pdf}}\end{{figure}}",
        t_1s, t_bss1] + ms1))

    # Stage 2
    acf_mosaic("stage2", [f"{lk}_dew_temp_enso" for lk in GAS_LAG_SETS],
               [LAG_LABEL[k] for k in GAS_LAG_SETS], f"acf_stage2_{tag}")
    qq_mosaic("stage2", [f"{lk}_dew_temp_enso" for lk in GAS_LAG_SETS],
              [LAG_LABEL[k] for k in GAS_LAG_SETS], f"qq_stage2_{tag}")
    s2parts = []
    for cov in S2_COV:
        lbl = S2_LBL[cov]
        cfgs = [f"{lk}_{cov}" for lk in GAS_LAG_SETS]
        clabs = [LAG_LABEL[k] for k in GAS_LAG_SETS]
        tl = lag_compare_table("stage2", f"Stage 2 ({lbl}){B}: RMSE and gradient by GAS-lag set.", f"tab:s2lag_{tag}_{cov}", cov=cov)
        t1 = onestep_table("stage2", cfgs, clabs, f"Stage 2 ({lbl}){B}: 1-step OOS metrics.", f"tab:s2_1s_{tag}_{cov}")
        tb = bss_table("stage2", cfgs, clabs, f"Stage 2 ({lbl}){B}: Brier skill score.", f"tab:s2bss_{tag}_{cov}")
        tms = [multistep_table("stage2", c, cl, f"Stage 2 ({lbl}){B}: multi-step OOS", f"tab:s2ms_{tag}_{cov}_{lk}")
               for c, cl, lk in zip(cfgs, clabs, GAS_LAG_SETS)]
        tables += [tl, t1, tb] + tms
        s2parts += [rf"\subsection{{covariates: {lbl}}}", tl, t1, tb] + tms
    sections.append("\n\n".join([
        rf"\section{{Stage 2 -- weather covariates{B}}}",
        rf"\begin{{figure}}[H]\centering\includegraphics[width=\textwidth]{{figures/acf_stage2_{tag}.pdf}}"
        rf"\caption{{Stage 2 ACF (dew+temp+ENSO){B}.}}\end{{figure}}",
        rf"\begin{{figure}}[H]\centering\includegraphics[width=\textwidth]{{figures/qq_stage2_{tag}.pdf}}"
        rf"\caption{{Stage 2 QQ (dew+temp+ENSO){B}.}}\end{{figure}}"] + s2parts))

    # Stage 3 / 4 / 4S
    for stage, name in [("stage3", "Stage 3 -- Harvey long/short"),
                        ("stage4", "Stage 4 -- long/short + damped seasonal harmonic"),
                        ("stage4S", "Stage 4 structural -- one level + one seasonal")]:
        acf_mosaic(stage, HARVEY_COV, [HC_LBL[c] for c in HARVEY_COV], f"acf_{stage}_{tag}")
        qq_mosaic(stage, HARVEY_COV, [HC_LBL[c] for c in HARVEY_COV], f"qq_{stage}_{tag}")
        t1 = onestep_table(stage, HARVEY_COV, [HC_LBL[c] for c in HARVEY_COV],
                           f"{name}{B}: 1-step OOS metrics by covariate config.", f"tab:{stage}_1s_{tag}")
        tb = bss_table(stage, HARVEY_COV, [HC_LBL[c] for c in HARVEY_COV],
                       f"{name}{B}: Brier skill score.", f"tab:{stage}_bss_{tag}")
        tms = [multistep_table(stage, c, HC_LBL[c], f"{name}{B}: multi-step OOS", f"tab:{stage}_ms_{tag}_{c}")
               for c in HARVEY_COV]
        tables += [t1, tb] + tms
        sections.append("\n\n".join([
            rf"\section{{{name}{B}}}",
            rf"\begin{{figure}}[H]\centering\includegraphics[width=\textwidth]{{figures/acf_{stage}_{tag}.pdf}}\end{{figure}}",
            rf"\begin{{figure}}[H]\centering\includegraphics[width=\textwidth]{{figures/qq_{stage}_{tag}.pdf}}\end{{figure}}",
            t1, tb] + tms))
    return sections, tables


def build():
    REP.mkdir(parents=True, exist_ok=True)
    all_sections, all_tables = [], []

    # ---- 1-vs-2 TVP comparison tables (up front) ----
    cmp_secs = [r"\section{1 vs 2 time-varying parameters -- comparison}"]
    tt = tvp_compare_table("stage1", GAS_LAG_SETS, [LAG_LABEL[k] for k in GAS_LAG_SETS],
                           "Stage 1: RMSE and gradient, 1 TVP ($\\varphi$) vs 2 TVP ($\\varphi,\\xi$), by GAS-lag set.",
                           "tab:tvp_s1")
    cmp_secs.append(tt); all_tables.append(tt)
    for stage, nm in [("stage3", "Stage 3"), ("stage4", "Stage 4"), ("stage4S", "Stage 4S")]:
        tt = tvp_compare_table(stage, HARVEY_COV, [HC_LBL[c] for c in HARVEY_COV],
                               f"{nm}: RMSE and gradient, 1 TVP vs 2 TVP, by covariate config.",
                               f"tab:tvp_{stage}")
        cmp_secs.append(tt); all_tables.append(tt)
    all_sections.append("\n\n".join(cmp_secs))

    # ---- per-branch content ----
    for tag, br, brname in [("1tvp", "phi_only", "1 TVP: $\\varphi$"),
                            ("2tvp", "phi_xi", "2 TVP: $\\varphi,\\xi$")]:
        _set_branch(br)
        secs, tabs = _content(tag, brname)
        all_sections += secs
        all_tables += tabs
    _set_branch("phi_only")

    # ---- Stage 5 (single best config, phi+xi by construction) ----
    if any((ART / s / "stage5" / "best" / "result.json").exists() for s in SHORTS):
        acf_mosaic("stage5", ["best"], ["best"], "acf_stage5")
        qq_mosaic("stage5", ["best"], ["best"], "qq_stage5")
        rows = []
        for s in SHORTS:
            r = load(s, "stage5", "best") or {}
            rows.append({"Location": DISP[s],
                         "base": f"{r.get('base_stage','--')}/{HC_LBL.get(r.get('cov',''), r.get('cov','--'))}"
                                 f"/{r.get('base_branch','--').replace(chr(95), chr(92)+chr(95))}",
                         "RMSE": _f(r.get("rmse"), 2), "CRPS": _f(r.get("crps_mean"), 2),
                         "tw95": _f(r.get("twcrps_95")), "tw98": _f(r.get("twcrps_98")),
                         "BSS": _f(r.get("BSS")), "grad": _f(r.get("grad_norm_inf_numeric"), 2)})
        t5 = tbl(rows, ["base", "RMSE", "CRPS", "tw95", "tw98", "BSS", "grad"],
                 "Stage 5: $\\varphi$ dynamics from the best Harvey config (grad tier -> CRPS -> RMSE); "
                 "$\\xi$ additionally time-varying. 1-step OOS + BSS + gradient.", "tab:s5")
        t5ms = multistep_table("stage5", "best", "best", "Stage 5: multi-step OOS", "tab:s5ms")
        all_tables += [t5, t5ms]
        all_sections.append("\n\n".join([
            r"\section{Stage 5 -- $\varphi$ dynamics from the best previous config, + time-varying $\xi$}",
            r"\begin{figure}[H]\centering\includegraphics[width=0.5\textwidth]{figures/acf_stage5.pdf}"
            r"\includegraphics[width=0.5\textwidth]{figures/qq_stage5.pdf}\end{figure}", t5, t5ms]))

    # ---- scaling choice + cross-stage gradient ----
    sc_secs = [r"\section{Stage 1 Fisher-scaling selection}"]
    for br, brt in [("phi_only", "1 TVP"), ("phi_xi", "2 TVP")]:
        rows = []
        sfx = "" if br == "phi_only" else "__xi"
        for s in SHORTS:
            pth = ART / s / "stage1" / f"_scaling_choice{sfx}.json"
            row = {"Location": DISP[s]}
            if pth.exists():
                c = json.loads(pth.read_text())
                row["chosen scaling"] = c.get("chosen_scaling", "--").replace("_", "\\_")
                row["best lag set"] = LAG_LABEL.get(c.get("best_lag_key", ""), "--")
            else:
                row["chosen scaling"] = "--"; row["best lag set"] = "--"
            rows.append(row)
        sc_secs.append(tbl(rows, ["chosen scaling", "best lag set"],
                           f"Stage 1 ({brt}) per-location Fisher-scaling selection (gradient tier, then AIC).",
                           f"tab:scaling_{br}"))
        all_tables.append(sc_secs[-1])
    all_sections.insert(0, "\n\n".join(sc_secs))

    _set_branch("phi_only")
    gt = grad_compare_table(["stage3", "stage4", "stage4S"], HARVEY_COV,
                            "Stages 3 / 4 / 4S (1 TVP): checked gradient norm at the optimum. "
                            "$^{\\dagger}$ = not converged across step sizes.", "tab:grad_harvey")
    all_tables.insert(0, gt)
    all_sections.insert(0, "\n\n".join([r"\section{Cross-stage gradient-norm comparison (Harvey stages, 1 TVP)}", gt]))

    intro = (r"\section*{Note on the occurrence model}"
             r"The occurrence probability follows the \textbf{logistic (sigmoid)} link "
             r"$\pi_{t|t-1} = 1/(1+e^{-\eta_t})$, with "
             r"$\eta_t=\omega_0+\rho\,\eta_{t-1}+\sum_{l=1}^{3}\omega_l\,y_{t-l}$ "
             r"(\texttt{ARLogisticCustomLagsPiDynamics(lags=[1,2,3], include\_ar=True, link=\"logistic\")}) "
             r"-- i.e.\ the standard link but with a short lag set $\{1,2,3\}$ plus an AR(1) term "
             r"in $\eta$. Every fit is numba-accelerated. Results are given for BOTH 1 time-varying "
             r"parameter ($\varphi$; $\xi,\gamma,\zeta$ static) and 2 ($\varphi,\xi$; $\gamma,\zeta$ static). "
             r"\emph{Note:} out-of-sample occurrence/tail metrics here incorporate the 2026-09-10 fix "
             r"to the $\eta$ recursion index in the covariate/long-short \texttt{simulate\_oos} paths "
             r"(previously off by \texttt{max\_lag} steps). "
             r"An earlier pass using the literal reciprocal link $\pi=1/(1+\eta)$ is archived under "
             r"\texttt{reports/rerun10092026\_reciprocal/}.")

    body = "\n\n".join([r"\begin{document}",
                        r"\title{Staged rerun: logistic occurrence link $\pi=1/(1+e^{-\eta})$, 1 \& 2 time-varying parameters}"
                        r"\author{}\date{2026-09-10}\maketitle", intro] + all_sections + [r"\end{document}"])
    (REP / "report.tex").write_text(PRE + "\n" + body, encoding="utf-8")
    tbody = "\n\n".join([r"\begin{document}",
                         r"\section*{All tables -- 2026-09-10 logistic-link rerun ($\pi=1/(1+e^{-\eta})$)}"] + all_tables + [r"\end{document}"])
    (REP / "tables.tex").write_text(PRE + "\n" + tbody, encoding="utf-8")
    print("wrote report.tex + tables.tex")

    for f in ("report", "tables"):
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{f}.tex"],
                           cwd=REP, capture_output=True, text=True)
        log = REP / f"{f}.log"
        ok = log.exists() and "Output written on" in log.read_text(encoding="utf-8", errors="ignore")
        print(f"{f}.pdf:", "OK" if ok else "FAILED (see log)")


if __name__ == "__main__":
    build()
