"""
Mosaic figures requested 2026-09-11: QQ-plot and 400-lag ACF grids of the
in-sample quantile residuals, for three model groupings, each column being
the per-location WINNER of a specific comparison already used in
generate_stage_comparison_report.py / generate_ml_comparison_report.py (same
selection rule -- gradient tier, then per-parameter convergence fraction,
then OOS CRPS, then OOS RMSE -- see generate_ml_comparison_report.select()).

Groupings (rows = the 6 locations in every mosaic):
  1. Stage 1: {1TVP, 2TVP} x {no seasonal lags, seasonal lags} = 4 columns.
     One image (qq/acf_stage1): readable at 4 columns.
  2. Stage 2: {1TVP, 2TVP} x {dew, dew+temp, dew+temp+ENSO}    = 6 columns
     (best GAS-lag set per location x covariate set, same rule as (1)).
  3. Stage 3/4/4S: {1TVP, 2TVP} x {Stage3 (no seasonal), Stage4 (seasonal),
     Stage4S (structural)} = 6 columns (best covariate config per location,
     same rule).

2026-09-12: groupings 2 and 3 were unreadable as single 6-column images, so
each is split into two 3-column images, one per branch (1TVP="phi_only",
2TVP="phi_xi") -- suffixed _phionly / _phixi below.

No fitting happens here -- reads insample_qr.npy from the already-fitted
caches (artifacts/rerun_10092026/ for Stage 1/2/4S, .../_phiL_sigmoid/ for
Stage 3/4).

Output: reports/mosaic11092026/{qq,acf}_stage1.{png,pdf}
        reports/mosaic11092026/{qq,acf}_stage2_{phionly,phixi}.{png,pdf}
        reports/mosaic11092026/{qq,acf}_stage34_{phionly,phixi}.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import generate_ml_comparison_report as G  # reuses select()/m()/trees/metrics.json

ROOT = Path(__file__).parent
REP = ROOT / "reports" / "mosaic11092026"
SHORTS = G.SHORTS
DISP = G.DISP_LONG
NLAGS = 400


def _winner(short, cands):
    ident, _mm = G.select(cands)
    return ident


def stage1_winner(short, branch, seasonal):
    lags = ("L1", "L123") if not seasonal else ("L1_365_366", "L123_365_366")
    cands = [(("stage1", lk, branch), G.m(short, "stage1", lk, branch)) for lk in lags]
    return _winner(short, cands)


def stage2_winner(short, branch, cov):
    cands = [(("stage2", f"{lk}_{cov}", branch), G.m(short, "stage2", f"{lk}_{cov}", branch))
             for lk in G.GAS_LAG_SETS]
    return _winner(short, cands)


def harvey_winner(short, stage, branch):
    cands = [((stage, cov, branch), G.m(short, stage, cov, branch)) for cov in G.HARVEY_COV5]
    return _winner(short, cands)


def load_qr(ident, short):
    if ident is None:
        return np.array([])
    cdir = G._resolve_cfg_dir(short, ident)
    p = cdir / "insample_qr.npy"
    if not p.exists():
        return np.array([])
    a = np.load(p)
    return a[np.isfinite(a)]


def acf(x, nlags):
    x = np.asarray(x, dtype=float)
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


def build_mosaic(columns, kind, fname):
    """columns: list of (label, winner_fn) where winner_fn(short) -> ident."""
    nrow, ncol = len(SHORTS), len(columns)
    figsize = (2.3 * ncol, 1.9 * nrow) if kind == "qq" else (2.6 * ncol, 1.9 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

    for r, short in enumerate(SHORTS):
        for c, (label, winner_fn) in enumerate(columns):
            ax = axes[r][c]
            ident = winner_fn(short)
            qr = load_qr(ident, short)
            if len(qr) < 20:
                ax.text(0.5, 0.5, "no fit", ha="center", va="center", transform=ax.transAxes, fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
            elif kind == "qq":
                stats.probplot(qr, dist="norm", plot=ax)
                ax.set_title("")
                ax.get_lines()[0].set_markersize(1.6)
                ax.get_lines()[1].set_linewidth(1.0)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(labelsize=5)
            else:  # acf
                a = acf(qr, NLAGS)
                ci = 1.96 / np.sqrt(len(qr))
                ax.bar(range(1, len(a)), a[1:], width=1.0, color="#3b6", linewidth=0)
                ax.axhline(ci, color="grey", lw=0.5, ls="--")
                ax.axhline(-ci, color="grey", lw=0.5, ls="--")
                ax.axhline(0, color="black", lw=0.4)
                ax.set_ylim(-1.0, 1.0)
                ax.tick_params(labelsize=5)
            if r == 0:
                ax.set_title(label, fontsize=7.5, linespacing=1.3)
            if c == 0:
                ax.set_ylabel(DISP[short].split(" (")[0], fontsize=7.5)

    fig.tight_layout()
    REP.mkdir(parents=True, exist_ok=True)
    fig.savefig(REP / f"{fname}.pdf")
    fig.savefig(REP / f"{fname}.png", dpi=150)
    plt.close(fig)
    print(f"wrote {fname}.pdf/.png ({nrow}x{ncol})")


def main():
    # ---- 1. Stage 1: 4 columns ----
    stage1_cols = [
        ("Time varying\nlocation - no\nseasonal lags",
         lambda s: stage1_winner(s, "phi_only", False)),
        ("Time varying\nlocation -\nseasonal lags\nincluded",
         lambda s: stage1_winner(s, "phi_only", True)),
        ("Time varying\nlocation and\nshape - no\nseasonal lags",
         lambda s: stage1_winner(s, "phi_xi", False)),
        ("Time varying\nlocation and\nshape - seasonal\nlags included",
         lambda s: stage1_winner(s, "phi_xi", True)),
    ]
    build_mosaic(stage1_cols, "qq", "qq_stage1")
    build_mosaic(stage1_cols, "acf", "acf_stage1")

    # ---- 2. Stage 2: 6 columns (1/2 TVP x dew / dew+temp / dew+temp+ENSO) ----
    # 2026-09-12: a single 6-column image was unreadable -- split into two
    # 3-column images, one per branch (phi_only / phi_xi), rather than an
    # arbitrary first-3/last-3 cut.
    s2_cov_labels = [("dew", "added DPT"), ("dew_temp", "added DPT + AT"),
                     ("dew_temp_enso", "added DPT + AT + ENSO")]
    for branch, tag, suffix in [
        ("phi_only", "Time varying\nlocation -", "phionly"),
        ("phi_xi",   "Time varying\nlocation and\nshape -", "phixi"),
    ]:
        cols = [(f"{tag}\n{lbl}", (lambda s, b=branch, c=cov: stage2_winner(s, b, c)))
                for cov, lbl in s2_cov_labels]
        build_mosaic(cols, "qq", f"qq_stage2_{suffix}")
        build_mosaic(cols, "acf", f"acf_stage2_{suffix}")

    # ---- 3. Stage 3/4/4S: 6 columns (best covariate set per location) ----
    # 2026-09-12: same split, one 3-column image per branch.
    stage34_phionly = [
        ("Time varying\nlocation - no\nseasonal component",
         lambda s: harvey_winner(s, "stage3", "phi_only")),
        ("Time varying\nlocation -\nseasonal component\nincluded",
         lambda s: harvey_winner(s, "stage4", "phi_only")),
        ("Time varying\nlocation - basic\nstructural model",
         lambda s: harvey_winner(s, "stage4S", "phi_only")),
    ]
    stage34_phixi = [
        ("Time varying\nlocation and\nshape - no\nseasonal component",
         lambda s: harvey_winner(s, "stage3", "phi_xi")),
        ("Time varying\nlocation and\nshape - seasonal\ncomponent included",
         lambda s: harvey_winner(s, "stage4", "phi_xi")),
        ("Time varying\nlocation and shape\n- basic structural\nmodel",
         lambda s: harvey_winner(s, "stage4S", "phi_xi")),
    ]
    build_mosaic(stage34_phionly, "qq", "qq_stage34_phionly")
    build_mosaic(stage34_phionly, "acf", "acf_stage34_phionly")
    build_mosaic(stage34_phixi, "qq", "qq_stage34_phixi")
    build_mosaic(stage34_phixi, "acf", "acf_stage34_phixi")


if __name__ == "__main__":
    main()
