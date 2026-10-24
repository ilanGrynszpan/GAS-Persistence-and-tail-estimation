"""
2026-10-02: percentage marginal effect of a dew-point shock, from the log
derivative, for the marginal-effects model of the phi-linked-pi run ->
reports/marginal_effects_021026/.

MATHEMATICS
    For a property Q_t of the predictive distribution (Q = M_t = E[Y_t], or
    Q = P_t = Pr(Y_t > q95)),
        d log Q / dx = (dQ / dx) / Q   =>   100 * dQ/Q ~= 100 * (d log Q / dx) * dx,
    the percentage change in Q for a shock dx at t-1 (dx = 1 and 5 degC). With
    pi_t = logistic(lambda0 + lambda1 phi_t), M = pi E[G], P = pi S(q95),
    phi = log scale (d log E[G]/d phi = 1) and d phi_t / d x_{t-1} = c_v:
        d log M / dx = c_v [ 1 + lambda1 (1 - pi_t) ]
        d log P / dx = c_v [ lambda1 (1 - pi_t) + (dS/d phi) / S ]
    (d log pi / d phi = lambda1 (1 - pi)). Linear in dx: the dx = 5 series
    (and band) is exactly 5 times the dx = 1 one.

UNCERTAINTY
    The same 1000 parameter draws as run_marginal_effects_exceedance_
    uncertainty.py (theta_k = theta_hat + N(0, SE^2) per parameter, SEs
    from the numerical Hessian, seed 123, parameters without SE held fixed),
    each re-filtered with the Numba filter; solid line = mean over draws,
    shaded band = 2.5/97.5 percentiles.

INPUTS
    artifacts/marginal_effects/exceedance_uncertainty_011026/model_info.json
        (selected model and its SEs, written by generate_reports_011026.py)
OUTPUTS
    reports/marginal_effects_021026/logpct_{mean,prob}_delta{1,5}.{png,pdf,csv}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_marginal_effects_exceedance_uncertainty as ME
import run_rerun_10092026 as D
import run_rerun_pi_01102026 as R
from marginal_effects.coefficients import column_std
from marginal_effects.model_selection import LOCATION_FULL

INFO = ROOT / "artifacts" / "marginal_effects" / "exceedance_uncertainty_011026" / "model_info.json"
OUT = ROOT / "reports" / "marginal_effects_021026"
DELTAS = (1.0, 5.0)
YMAX_PCT = 30.0      # common y-axis 0-30 % for the 1 and 5 degC charts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = json.loads(INFO.read_text())
    short, stage, cfg = info["location"], info["stage"], info["cfg"]
    names = [p["name"] for p in info["parameters"]]
    theta_hat = np.array([p["estimate"] for p in info["parameters"]], float)
    se = np.array([p["se"] if p["se"] is not None else np.nan for p in info["parameters"]], float)
    res = json.loads((R.OUT / short / stage / f"{cfg}{R._sfx('phi_only')}" / "result.json").read_text())
    data = D._load_data(LOCATION_FULL[short])
    model, fk, _ = R.build_model(stage, "phi_only", res["scaling"], cfg, data)
    assert model.parameter_names() == names
    y = data["y_train"]
    q95 = float(np.quantile(y[y > 0], 0.95))
    dpt_std = column_std(data, ME.DPT_COLUMN)
    is_harvey = hasattr(model, "harvey_target")
    cv_name = f"Gamma_S_phi_{ME.DPT_COLUMN}" if is_harvey else None
    dpt_idx = None if is_harvey else list(model.cov_names).index(ME.DPT_COLUMN)

    p0 = model.filter(theta_hat, y, **fk)
    dates = pd.DatetimeIndex(data["dates_train"])[int(p0["eff_start"]):]
    T = len(dates)
    rng = np.random.default_rng(ME.SEED)                    # identical draws to the original driver
    se_safe = np.where(np.isfinite(se), se, 0.0)
    dist = model.dist
    lm = np.empty((ME.N_DRAWS, T))
    lp = np.empty((ME.N_DRAWS, T))
    for k in range(ME.N_DRAWS):
        th = theta_hat + rng.standard_normal(len(theta_hat)) * se_safe
        p = model.filter(th, y, **fk)
        phi, pi, st = p["f_arr"][:, 0], np.asarray(p["pi"]), p["static"]
        c_std = float(dict(zip(names, th))[cv_name]) if is_harvey else float(p["gamma"]["phi"][dpt_idx])
        cv = c_std / dpt_std
        lam1 = float(dict(zip(names, th)).get("lambda1", 0.0))
        occ = lam1 * (1.0 - pi)
        s = 1.0 - dist.cdf(q95, phi=phi, xi=st["xi"], gamma=st["gamma"], zeta=st["zeta"])
        ker = dist.exceedance_effect_kernel(q95, phi=phi, xi=st["xi"], gamma=st["gamma"], zeta=st["zeta"])
        lm[k] = 100.0 * cv * (1.0 + occ)            # per 1 degC
        lp[k] = 100.0 * cv * (occ + ker / s)

    periods = ME._enso_periods(np.asarray(data["enso_label_train"])[int(p0["eff_start"]):], dates)
    for (tag, unit), delta in [(tp, d) for tp in (("mean", lm), ("prob", lp)) for d in DELTAS]:
        arr = unit * delta
        tag = f"{tag}_delta{int(delta)}"
        df = pd.DataFrame({"date": dates, "pct": arr.mean(0), "lo": np.quantile(arr, 0.025, 0),
                           "hi": np.quantile(arr, 0.975, 0)})
        df.to_csv(OUT / f"logpct_{tag}.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(dates, df["lo"], df["hi"], color="red", alpha=0.20, linewidth=0)
        ax.plot(dates, df["pct"], color=ME.EFFECT_COLOR, linewidth=1.0, linestyle="-")
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_xlim(dates[0], dates[-1])
        ax.set_ylim(0.0, YMAX_PCT)                         # same scale on every chart
        ax.yaxis.set_major_locator(MultipleLocator(5.0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ME._add_enso_annotations(ax, periods, dates[0], dates[-1])
        fig.tight_layout()
        ME._savefig(fig, OUT / f"logpct_{tag}.png", dpi=200)
        ME._savefig(fig, OUT / f"logpct_{tag}.pdf")
        plt.close(fig)
        print(f"wrote logpct_{tag}.png/.pdf/.csv")


if __name__ == "__main__":
    main()
