"""
Parameter-uncertainty-propagated, in-sample, one-step-ahead marginal
effects of a dew-point-temperature (DPT) shock at t-1 on (a) the ZA-GB2
conditional mean and (b) the exceedance probability P(Y_t > q95)
(user request, 2026-09-18).

=============================================================================
CODE WALKTHROUGH
=============================================================================

select_model()
    Among every 1-TVP (phi-only), DPT-bearing, aggregate-grad<1 candidate
    across all 6 locations and both model families (marginal_effects.
    model_selection), pick the one with the smallest MAX per-parameter
    checked numerical gradient (every candidate already has all params<1,
    since that's implied by the aggregate ||grad||_inf<1 filter -- so the
    real differentiator is "smallest possible for ... most parameters").
    Winner: Garanhuns, standard GAS(p,q), stage2/L1_dew_temp, 14 params,
    max|grad|=0.0126 (next best: 0.043-0.045; most candidates: 0.1-0.9).
    |
compute_standard_errors()
    Independent numerical Hessian at the already-fitted theta (NOT a
    refit -- docs/OPTIMIZATION.md explicitly allows a numerical Hessian
    "for final selected models ... when explicitly requested", which this
    is), via diagnostics.numerical_diff.numerical_std_errors. 14 params ->
    well under a second.
    |
draw_and_filter()
    1000 independent theta draws, theta_k = theta_hat + N(0,SE_i^2) per
    parameter i (the user's own instruction: "use the obtained standard
    deviations"). Each draw is re-filtered over the REAL in-sample
    y_train via model.filter() -- the model's own already-numba-compiled
    kernel (models/_nb_gb2_filter.py, dispatched via _use_nb_filter) -- no
    new numba code needed; looping the existing kernel 1000x already
    satisfies "make everything numba". This is genuinely fast: 1000 calls
    measured at 0.36s.
    |
effect_arrays()
    Per draw, per in-sample t: the one-step marginal effect of a
    +1-raw-unit DPT shock at t-1 on (a) the ZA-GB2 mean (dM1/dx, the same
    formula marginal_effects/onestep.py already uses) and (b) the ZA-GB2
    exceedance probability at u=q95 (distributions.gb2_log_link.
    GB2LogLink.exceedance_effect_kernel, new method, proved in
    docs/math_proofs.tex Sec.12). Both formulas are evaluated ONLY from
    the model's own REAL filtered state phi_{t|t-1} (built from y up to
    t-1) -- there is no forward propagation into t+1, t+2, ...: "the
    effect does not persist in time ... only evaluate the effect of a
    change in t-1 on the things at t" (user, 2026-09-18). +5-unit effects
    are obtained by scaling the SAME +1-unit draws by 5 (both formulas are
    first-order/linear in the covariate perturbation -- no new draws or
    filtering needed).
    |
aggregate_and_plot()
    Mean and 95% interval (2.5/97.5 percentile) across the 1000 draws, at
    every t. Two ABSOLUTE-effect charts (Delta=1, Delta=5): green = mean
    effect + shaded 95% band, red = exceedance-probability effect +
    shaded 95% band, twin y-axes (the two differ by ~2 orders of
    magnitude), no title/legend.

    Plus four PERCENTAGE-CHANGE charts (user request, 2026-09-18):
    (things_after - things_without)/things_without, for {mean, P(Y>q95)} x
    {Delta=1, Delta=5}, one series each, always red with a shaded 95% CI,
    no title/legend. Computed EXACTLY per draw (not a linear rescaling of
    the Delta=1 draws, unlike the two absolute-effect charts): the
    covariate enters the log-link recursion additively, so a finite raw
    shock Delta at t-1 shifts the log-state at t by EXACTLY
    c_v_raw*Delta (the recursion's other terms do not depend on x_{t-1}
    at all) -- "after" and "without" are then each mapped through the
    TRUE nonlinear moment/CDF functions and differenced/ratioed. pi_t is
    unchanged (c_v^pi=0 throughout this repository) and cancels out of
    both ratios, so the ZA percentage change equals the positive-part
    percentage change in each case.
    Saved to reports/marginal_effects_18092026_further/.

=============================================================================
OUTPUTS
=============================================================================

  artifacts/marginal_effects/exceedance_uncertainty/model_info.json
      selected model identity, per-parameter checked gradient, standard errors.
  artifacts/marginal_effects/exceedance_uncertainty/effects_delta{1,5}.csv
      date, green_mean/lo/hi (ZA mean effect), red_mean/lo/hi (P(Y>q95) effect).
  artifacts/marginal_effects/exceedance_uncertainty/{mean,prob}_pct_delta{1,5}.csv
      date, pct_mean/lo/hi -- the exact percentage-change quantities.
  reports/marginal_effects_18092026_further/dpt_shock_delta{1,5}.{png,pdf}
      absolute effects (green=mean, red=P(Y>q95), twin axes).
  reports/marginal_effects_18092026_further/dpt_shock_{mean,prob}_pct_delta{1,5}.{png,pdf}
      percentage-change charts (single red series + shaded CI each).
  reports/marginal_effects_18092026_further/changes in time/mean_effect_degC_delta{1,5}.{png,pdf,csv}
      mean effect in mm/degC-shock units (same values as the green series
      above, replotted alone in red per the user's later spec), with
      El Nino/La Nina period annotations (user request, 2026-09-18):
      solid vertical line + "EN"/"LN" text at each period's start, dashed
      vertical line (no text) at its end, from data/station_loader.py's
      own enso_label_train (ONI-based, already used elsewhere in this
      repository -- not a new classification).
  reports/marginal_effects_18092026_further/changes in time/prob_exceedance_pct_delta{1,5}.{png,pdf,csv}
      100 * (after-before)/before for P(Y>q95) -- the same exact quantity
      as dpt_shock_prob_pct_delta{1,5} above, in percentage-point units
      instead of a fraction, with the same El Nino/La Nina annotations.
  reports/marginal_effects_18092026_further/with comparison/mean_comparison_delta{1,5}.{png,pdf,csv}
      LEVELS, not effects (user request, 2026-09-18): M1_before (mean
      WITHOUT the DPT shock, mm, solid, ORIGINAL_COLOR) vs. M1_after (mean
      WITH the shock, mm, dashed, EFFECT_COLOR -- darker, the CI-shaded
      series), same El Nino/La Nina annotations. Colours revised
      2026-09-18 ("the original values are not visible"): three-tier
      darkness, pale shaded CI < solid original < dashed effect.
  reports/marginal_effects_18092026_further/with comparison/prob_comparison_delta{1,5}.{png,pdf,csv}
      same idea for the exceedance probability: 100*P_before (solid) vs.
      100*P_after (dashed, darker, shaded CI), both in percentage units,
      same annotations and colour convention.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marginal_effects.model_selection import (
    _load_metrics, _candidates_standard, _candidates_harvey, GRAD_THRESHOLD, SHORTS, ModelSelection,
)
from marginal_effects.model_loader import load
from marginal_effects.coefficients import column_std
from diagnostics.numerical_diff import numerical_gradient_checked, numerical_std_errors

OUT_ARTIFACTS = ROOT / "artifacts" / "marginal_effects" / "exceedance_uncertainty"
OUT_REPORT = ROOT / "reports" / "marginal_effects_18092026_further"
N_DRAWS = 1000
SEED = 123
DPT_COLUMN = "dewpoint_lag1"
DELTAS = (1.0, 5.0)
_SAVEFIG_RETRIES = 5
_SAVEFIG_BACKOFF_S = 0.5

# "with comparison" chart colours (user fix, 2026-09-18: "the original
# values are not visible" -- both lines were the same plain red as the
# shaded CI band, so the thin dashed "before" line was nearly invisible
# against it). Three-tier darkness, palest to darkest:
#   CI band (shaded)      -> pale red
#   original/before line  -> solid, a clear mid-red (darker than the
#                             shading, not too dark)
#   effect/after line     -> dashed, darker still (the most prominent line)
CI_SHADE_COLOR = "red"
ORIGINAL_COLOR = "#e03131"   # solid "before" line
EFFECT_COLOR = "#7a0000"     # dashed "after" line


def _savefig(fig, path: Path, **kwargs) -> None:
    """fig.savefig with a short retry/backoff: this project directory is
    OneDrive-synced, and OneDrive's background sync client transiently
    locks a just-written file for indexing/upload, which intermittently
    raises `OSError: [Errno 22] Invalid argument` from PIL's own
    `open(path, "w+b")` a moment later in the SAME run -- not a real path
    or encoding problem (confirmed: the identical call succeeds on a
    retry a fraction of a second later, and which specific file fails is
    different each run)."""
    last_exc = None
    for attempt in range(_SAVEFIG_RETRIES):
        try:
            fig.savefig(path, **kwargs)
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(_SAVEFIG_BACKOFF_S * (attempt + 1))
    raise last_exc


def _enso_periods(enso_label: np.ndarray, dates: pd.DatetimeIndex):
    """Contiguous (start_date, end_date, kind) runs of El Nino / La Nina
    within `dates`, from data/station_loader.py's own `enso_label_train`
    (ONI-based monthly classification broadcast to daily and 1-month-lagged
    -- see docs/MODELS.md Sec.18 / that module's own docstring; the SAME
    label array every other ENSO-aware part of this repo uses, not a new
    classification). Matched by the label's first two ASCII letters ("El"
    / "La") to sidestep the accented "Nino"/"Nina" spelling entirely
    (console/file encoding of "niño"/"niña" is unreliable on this
    Windows setup, per this session's own repeated mangled-console-output
    experience; the raw string data itself is fine, only display is not)."""
    s = pd.Series([str(x) for x in enso_label], index=pd.DatetimeIndex(dates))
    kind = np.where(s.str.startswith("El"), "EN", np.where(s.str.startswith("La"), "LN", ""))
    periods = []
    start_i = None
    for i in range(len(kind)):
        changed = (i == 0) or (kind[i] != kind[i - 1])
        if changed:
            if start_i is not None and kind[start_i]:
                periods.append((s.index[start_i], s.index[i - 1], kind[start_i]))
            start_i = i
    if start_i is not None and kind[start_i]:
        periods.append((s.index[start_i], s.index[-1], kind[start_i]))
    return periods


def _add_enso_annotations(ax, periods, x_lo, x_hi) -> None:
    """Solid vertical line + 'EN'/'LN' text near the top at each period's
    start (clipped to the visible x-range); dashed vertical line (no text)
    at its end -- user's exact spec, 2026-09-18."""
    y_lo, y_hi = ax.get_ylim()
    y_text = y_lo + 0.94 * (y_hi - y_lo)
    for start, end, kind in periods:
        if x_lo <= start <= x_hi:
            ax.axvline(start, color="black", linewidth=0.7, linestyle="-", alpha=0.6)
            ax.text(start, y_text, kind, fontsize=8, ha="left", va="top")
        if x_lo <= end <= x_hi:
            ax.axvline(end, color="black", linewidth=0.7, linestyle="--", alpha=0.6)


def _objective(model, family, data, ms_kwargs, theta):
    if family == "standard":
        return model._run_filter(theta, data["y_train"], ms_kwargs["X_train"], return_paths=False)
    return model._run_filter(theta, data["y_train"], ms_kwargs["X_long_train"],
                              ms_kwargs["X_short_train"], return_paths=False)


def select_model():
    """1-TVP, DPT-bearing, aggregate grad<1 candidates, all locations and
    both families; winner = smallest max per-parameter checked gradient."""
    met = _load_metrics()
    pool = []
    for short in SHORTS:
        for family, fn in (("standard", _candidates_standard), ("harvey", _candidates_harvey)):
            for stage, cfg, mm, variables in fn(met, short, "phi_only"):
                g = mm.get("grad_inf")
                if g is None or not np.isfinite(g) or g >= GRAD_THRESHOLD:
                    continue
                if not np.isfinite(mm.get("crps_oos", np.nan)) or "DPT" not in variables:
                    continue
                pool.append((short, family, stage, cfg, mm, variables))

    results = []
    for short, family, stage, cfg, mm, variables in pool:
        sel = ModelSelection(short, "phi_only", family, stage, cfg, mm, variables, "candidate")
        lm = load(sel)
        obj = lambda th, m=lm.model, f=family, d=lm.data, k=lm.ms_kwargs: _objective(m, f, d, k, th)
        g = numerical_gradient_checked(obj, lm.theta)
        results.append({"short": short, "family": family, "stage": stage, "cfg": cfg,
                         "variables": variables,
                         "n_params": len(lm.theta), "max_abs_grad": float(np.max(np.abs(g["grad"]))),
                         "n_under_1": int(np.sum(np.abs(g["grad"]) < 1.0)), "converged": g["converged"]})

    all_under1 = [r for r in results if r["n_under_1"] == r["n_params"]]
    pool_for_winner = all_under1 if all_under1 else results
    key = (lambda r: r["max_abs_grad"]) if all_under1 else (lambda r: (-r["n_under_1"], r["max_abs_grad"]))
    winner = min(pool_for_winner, key=key)
    return winner, results


def main():
    t_start = time.time()
    OUT_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.mkdir(parents=True, exist_ok=True)

    print("Selecting model (1-TVP, DPT-bearing, best per-parameter convergence)...")
    winner, all_results = select_model()
    print(f"  winner: {winner['short']}/{winner['family']}/{winner['stage']}/{winner['cfg']} "
          f"n_params={winner['n_params']} max|grad|={winner['max_abs_grad']:.4g}")

    met = _load_metrics()
    mm = met[winner["short"]][winner["stage"]][winner["cfg"]]["phi_only"]
    sel = ModelSelection(winner["short"], "phi_only", winner["family"], winner["stage"], winner["cfg"],
                          mm, winner["variables"], "selected")
    lm = load(sel)
    model, theta_hat, data = lm.model, lm.theta, lm.data
    dist = model.dist
    param_names = model.parameter_names()

    print("Computing standard errors (numerical Hessian at the fitted optimum)...")
    t0 = time.time()
    obj = lambda th: _objective(model, winner["family"], data, lm.ms_kwargs, th)
    se_result = numerical_std_errors(obj, theta_hat)
    se = se_result["std_errors"]
    if se_result["quality"] == "failed" and se_result.get("hessian") is not None:
        # Singular Hessian: drop the directions with (numerically) no curvature
        # -- the rule generate_std_errors_report.py uses -- and invert the rest.
        # Dropped parameters get no SE and are held at the estimate below.
        H = np.asarray(se_result["hessian"], float)
        d = np.abs(np.diag(H))
        keep = np.where(np.isfinite(d) & (d > 1e-8 * np.nanmax(d)))[0]
        try:
            cov = np.linalg.inv(H[np.ix_(keep, keep)])
            v = np.diag(cov)
            se = np.full(len(theta_hat), np.nan)
            se[keep] = np.where(v > 0, np.sqrt(np.abs(v)), np.nan)
            se_result["quality"] = "partial (flat directions dropped)"
        except np.linalg.LinAlgError:
            pass
    print(f"  quality={se_result['quality']}  ({time.time()-t0:.2f}s)")

    n_nan_se = int(np.sum(~np.isfinite(se)))
    if n_nan_se:
        print(f"  WARNING: {n_nan_se} parameter(s) have no valid SE; held fixed at the point "
              f"estimate in every draw: {[param_names[i] for i in range(len(se)) if not np.isfinite(se[i])]}")

    model_info = {
        "location": winner["short"], "family": winner["family"], "stage": winner["stage"], "cfg": winner["cfg"],
        "n_params": winner["n_params"], "max_abs_grad_checked": winner["max_abs_grad"],
        "grad_converged": winner["converged"], "se_quality": se_result["quality"],
        "parameters": [{"name": n, "estimate": float(v), "se": (float(s) if np.isfinite(s) else None)}
                        for n, v, s in zip(param_names, theta_hat, se)],
        "all_candidates_considered": all_results,
    }
    (OUT_ARTIFACTS / "model_info.json").write_text(json.dumps(model_info, indent=2, default=float))

    # ---- q95: wet-day (y>0) 95th percentile of TRAINING (in-sample) rainfall ----
    y_train = data["y_train"]
    wet = y_train[y_train > 0]
    q95 = float(np.quantile(wet, 0.95))
    print(f"  q95 (wet-day, training) = {q95:.3f}")

    dpt_std = column_std(data, DPT_COLUMN)
    is_harvey = hasattr(model, "harvey_target")
    if is_harvey:
        # Long-short family: DPT enters the short component, S_{t+1} = ... + Gamma_S x_t,
        # and phi_t = omega + L_t + S_t, so a shock at t-1 moves phi_t by Gamma_S * Delta.
        cv_name = f"Gamma_S_phi_{DPT_COLUMN}"
        filt_args = (lm.ms_kwargs["X_long_train"], lm.ms_kwargs["X_short_train"])
    else:
        dpt_idx = list(model.cov_names).index(DPT_COLUMN)
        filt_args = (lm.ms_kwargs["X_train"],)

    def _cv_std(paths, theta_k):
        if is_harvey:
            return float(dict(zip(param_names, theta_k))[cv_name])
        return float(paths["gamma"]["phi"][dpt_idx])

    # ---- point-estimate in-sample filtered path (for the date index only) ----
    paths0 = model.filter(theta_hat, y_train, *filt_args)
    eff_start = paths0["eff_start"]
    dates = pd.DatetimeIndex(data["dates_train"])[eff_start:]
    T_eff = len(dates)
    assert paths0["f_arr"].shape[0] == T_eff

    # ---- 1000 parameter draws, each re-filtered (numba-backed .filter()) ----
    print(f"Drawing {N_DRAWS} parameter vectors and re-filtering (numba)...")
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    se_safe = np.where(np.isfinite(se), se, 0.0)  # NaN-SE params held at point estimate
    green = np.empty((N_DRAWS, T_eff))  # d(ZA mean)/d(DPT), +1 raw unit
    red = np.empty((N_DRAWS, T_eff))    # d P(Y>q95)/d(DPT), +1 raw unit
    # Percentage change (things_after - things_without)/things_without, user
    # request 2026-09-18: computed EXACTLY, not via the linear kernels above.
    # This is legitimate (not merely a first-order approximation) because
    # the covariate enters the log-link recursion additively -- shocking
    # x_{t-1} by a FINITE raw amount Delta shifts the log-state at t by
    # EXACTLY c_v_raw*Delta (the recursion's other terms -- omega, the
    # score/AR parts -- do not depend on x_{t-1} at all), before any
    # nonlinear moment/CDF map is applied. Since 2026-10-01 pi_t is shifted
    # too when it is phi-linked (eta_t moves by exactly lambda1 c_v Delta), so
    # the ratios use pi_after; for any other pi dynamics lambda1 = 0, pi
    # cancels, and the ZA percentage change equals the positive-part one.
    pct_mean = {delta: np.empty((N_DRAWS, T_eff)) for delta in DELTAS}
    pct_prob = {delta: np.empty((N_DRAWS, T_eff)) for delta in DELTAS}
    # LEVELS (not effects/percentages), for the "with comparison" charts
    # (user request, 2026-09-18): M1_before/P_before don't depend on delta
    # (pi_t is unchanged by the shock, and neither is the un-shocked state).
    M1_before_lv = np.empty((N_DRAWS, T_eff))
    P_before_lv = np.empty((N_DRAWS, T_eff))
    M1_after_lv = {delta: np.empty((N_DRAWS, T_eff)) for delta in DELTAS}
    P_after_lv = {delta: np.empty((N_DRAWS, T_eff)) for delta in DELTAS}

    for k in range(N_DRAWS):
        theta_k = theta_hat + rng.standard_normal(len(theta_hat)) * se_safe
        paths = model.filter(theta_k, y_train, *filt_args)
        phi_path = paths["f_arr"][:, 0]
        pi_path = paths["pi"]
        stat = paths["static"]
        xi_k, gamma_k, zeta_k = stat["xi"], stat["gamma"], stat["zeta"]
        c_v_std = _cv_std(paths, theta_k)
        c_v_raw = c_v_std / dpt_std
        # Occurrence channel (2026-10-01): with phi-linked occurrence dynamics
        # pi_t = logistic(lambda0 + lambda1 phi_t), the shock also moves pi_t:
        # d pi/d phi = lambda1 pi (1 - pi). lambda1 = 0 for every other pi
        # dynamics, which reproduces the earlier (pi-invariant) formulas exactly.
        lam1 = float(dict(zip(param_names, theta_k)).get("lambda1", 0.0))
        dpi_dphi = lam1 * pi_path * (1.0 - pi_path)
        logit_pi = np.log(pi_path) - np.log1p(-pi_path)

        m1_before = dist.raw_moment(1, phi=phi_path, xi=xi_k, gamma=gamma_k, zeta=zeta_k)
        # d(pi E[G])/dx = c_v [dpi/dphi E[G] + pi dE[G]/dphi], dE[G]/dphi = E[G] (phi = log scale)
        green[k, :] = c_v_raw * (dpi_dphi * m1_before + pi_path * m1_before)
        M1_before_lv[k, :] = pi_path * m1_before

        kernel = dist.exceedance_effect_kernel(q95, phi=phi_path, xi=xi_k, gamma=gamma_k, zeta=zeta_k)
        s_before = 1.0 - dist.cdf(q95, phi=phi_path, xi=xi_k, gamma=gamma_k, zeta=zeta_k)
        # d(pi S(u))/dx = c_v [dpi/dphi S(u) + pi dS(u)/dphi]
        red[k, :] = c_v_raw * (dpi_dphi * s_before + pi_path * kernel)
        P_before_lv[k, :] = pi_path * s_before
        for delta in DELTAS:
            phi_after = phi_path + c_v_raw * delta
            pi_after = 1.0 / (1.0 + np.exp(-(logit_pi + lam1 * c_v_raw * delta)))
            m1_after = dist.raw_moment(1, phi=phi_after, xi=xi_k, gamma=gamma_k, zeta=zeta_k)
            pct_mean[delta][k, :] = (pi_after * m1_after - pi_path * m1_before) / (pi_path * m1_before)
            M1_after_lv[delta][k, :] = pi_after * m1_after

            s_after = 1.0 - dist.cdf(q95, phi=phi_after, xi=xi_k, gamma=gamma_k, zeta=zeta_k)
            pct_prob[delta][k, :] = (pi_after * s_after - pi_path * s_before) / (pi_path * s_before)
            P_after_lv[delta][k, :] = pi_after * s_after

    print(f"  done in {time.time()-t0:.2f}s")

    # ---- aggregate + save + plot, for each Delta (linear scaling, no new draws) ----
    for delta in DELTAS:
        g = green * delta
        r = red * delta
        out = pd.DataFrame({
            "date": dates,
            "green_mean": g.mean(axis=0), "green_lo": np.quantile(g, 0.025, axis=0), "green_hi": np.quantile(g, 0.975, axis=0),
            "red_mean": r.mean(axis=0), "red_lo": np.quantile(r, 0.025, axis=0), "red_hi": np.quantile(r, 0.975, axis=0),
        })
        tag = f"delta{int(delta)}"
        out.to_csv(OUT_ARTIFACTS / f"effects_{tag}.csv", index=False)

        # Twin y-axes: the mean-rainfall effect (mm) and the exceedance-
        # probability effect (unitless, in [0,1]) differ by ~2 orders of
        # magnitude -- on one shared axis the red series is visually flat
        # at zero. Colour-coding each axis's ticks/spine to match its
        # series substitutes for a legend (still none, per spec).
        fig, ax_g = plt.subplots(figsize=(10, 4))
        ax_g.fill_between(dates, out["green_lo"], out["green_hi"], color="green", alpha=0.25, linewidth=0)
        ax_g.plot(dates, out["green_mean"], color="green", linewidth=1.0)
        ax_g.set_xlim(dates[0], dates[-1])
        ax_g.tick_params(axis="y", colors="green")
        ax_g.spines["left"].set_color("green")
        ax_g.spines["top"].set_visible(False)

        ax_r = ax_g.twinx()
        ax_r.fill_between(dates, out["red_lo"], out["red_hi"], color="red", alpha=0.25, linewidth=0)
        ax_r.plot(dates, out["red_mean"], color="red", linewidth=1.0)
        ax_r.set_ylim(bottom=0)
        ax_r.tick_params(axis="y", colors="red")
        ax_r.spines["right"].set_color("red")
        ax_r.spines["top"].set_visible(False)
        ax_r.spines["left"].set_visible(False)
        fig.tight_layout()
        _savefig(fig, OUT_REPORT / f"dpt_shock_{tag}.png", dpi=200)
        _savefig(fig, OUT_REPORT / f"dpt_shock_{tag}.pdf")
        plt.close(fig)
        print(f"  wrote {OUT_REPORT / f'dpt_shock_{tag}.png'} / .pdf and "
              f"{OUT_ARTIFACTS / f'effects_{tag}.csv'}")

    # ---- percentage-change charts: one series each (red, shaded 95% CI) ----
    for delta in DELTAS:
        tag = f"delta{int(delta)}"
        for kind, arr in (("mean_pct", pct_mean[delta]), ("prob_pct", pct_prob[delta])):
            out = pd.DataFrame({
                "date": dates,
                "pct_mean": arr.mean(axis=0),
                "pct_lo": np.quantile(arr, 0.025, axis=0),
                "pct_hi": np.quantile(arr, 0.975, axis=0),
            })
            out.to_csv(OUT_ARTIFACTS / f"{kind}_{tag}.csv", index=False)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.fill_between(dates, out["pct_lo"], out["pct_hi"], color="red", alpha=0.25, linewidth=0)
            ax.plot(dates, out["pct_mean"], color="red", linewidth=1.0)
            ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
            ax.set_xlim(dates[0], dates[-1])
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            fig.tight_layout()
            _savefig(fig, OUT_REPORT / f"dpt_shock_{kind}_{tag}.png", dpi=200)
            _savefig(fig, OUT_REPORT / f"dpt_shock_{kind}_{tag}.pdf")
            plt.close(fig)
            print(f"  wrote {OUT_REPORT / f'dpt_shock_{kind}_{tag}.png'} / .pdf and "
                  f"{OUT_ARTIFACTS / f'{kind}_{tag}.csv'}")

    # ---- "changes in time" charts: mean effect (degC units) and exceedance-
    # probability percentage change, each annotated with El Nino/La Nina
    # periods (user request, 2026-09-18) ----
    out_ts = OUT_REPORT / "changes in time"
    out_ts.mkdir(parents=True, exist_ok=True)
    enso_periods = _enso_periods(data["enso_label_train"], pd.DatetimeIndex(data["dates_train"]))
    print(f"\n{len(enso_periods)} El Nino/La Nina periods detected in the training window "
          f"({sum(1 for *_, k in enso_periods if k=='EN')} EN, "
          f"{sum(1 for *_, k in enso_periods if k=='LN')} LN)")

    for delta in DELTAS:
        tag = f"delta{int(delta)}"
        g = green * delta
        mean_out = pd.DataFrame({
            "date": dates, "mean_mm": g.mean(axis=0),
            "mean_lo": np.quantile(g, 0.025, axis=0), "mean_hi": np.quantile(g, 0.975, axis=0),
        })
        mean_out.to_csv(out_ts / f"mean_effect_degC_{tag}.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(dates, mean_out["mean_lo"], mean_out["mean_hi"], color="red", alpha=0.25, linewidth=0)
        ax.plot(dates, mean_out["mean_mm"], color="red", linewidth=1.0)
        ax.set_xlim(dates[0], dates[-1])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        _add_enso_annotations(ax, enso_periods, dates[0], dates[-1])
        fig.tight_layout()
        _savefig(fig, out_ts / f"mean_effect_degC_{tag}.png", dpi=200)
        _savefig(fig, out_ts / f"mean_effect_degC_{tag}.pdf")
        plt.close(fig)
        print(f"  wrote {out_ts / f'mean_effect_degC_{tag}.png'} / .pdf")

        p = 100.0 * pct_prob[delta]
        prob_out = pd.DataFrame({
            "date": dates, "prob_pct_mean": p.mean(axis=0),
            "prob_pct_lo": np.quantile(p, 0.025, axis=0), "prob_pct_hi": np.quantile(p, 0.975, axis=0),
        })
        prob_out.to_csv(out_ts / f"prob_exceedance_pct_{tag}.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(dates, prob_out["prob_pct_lo"], prob_out["prob_pct_hi"], color="red", alpha=0.25, linewidth=0)
        ax.plot(dates, prob_out["prob_pct_mean"], color="red", linewidth=1.0)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_xlim(dates[0], dates[-1])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        _add_enso_annotations(ax, enso_periods, dates[0], dates[-1])
        fig.tight_layout()
        _savefig(fig, out_ts / f"prob_exceedance_pct_{tag}.png", dpi=200)
        _savefig(fig, out_ts / f"prob_exceedance_pct_{tag}.pdf")
        plt.close(fig)
        print(f"  wrote {out_ts / f'prob_exceedance_pct_{tag}.png'} / .pdf")

    # ---- "with comparison" charts: LEVELS, after (solid, shaded CI) vs.
    # before (dashed, thinner, no CI shading), same ENSO annotations (user
    # request, 2026-09-18) ----
    out_cmp = OUT_REPORT / "with comparison"
    out_cmp.mkdir(parents=True, exist_ok=True)

    m1b_out = pd.DataFrame({
        "date": dates, "M1_before_mean": M1_before_lv.mean(axis=0),
        "M1_before_lo": np.quantile(M1_before_lv, 0.025, axis=0),
        "M1_before_hi": np.quantile(M1_before_lv, 0.975, axis=0),
    })
    p_before_pct = 100.0 * P_before_lv
    pb_out = pd.DataFrame({
        "date": dates, "P_before_pct_mean": p_before_pct.mean(axis=0),
        "P_before_pct_lo": np.quantile(p_before_pct, 0.025, axis=0),
        "P_before_pct_hi": np.quantile(p_before_pct, 0.975, axis=0),
    })

    for delta in DELTAS:
        tag = f"delta{int(delta)}"

        m1a_out = pd.DataFrame({
            "date": dates, "M1_after_mean": M1_after_lv[delta].mean(axis=0),
            "M1_after_lo": np.quantile(M1_after_lv[delta], 0.025, axis=0),
            "M1_after_hi": np.quantile(M1_after_lv[delta], 0.975, axis=0),
        })
        out = pd.concat([m1a_out, m1b_out.drop(columns="date")], axis=1)
        out.to_csv(out_cmp / f"mean_comparison_{tag}.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(dates, out["M1_after_lo"], out["M1_after_hi"], color=CI_SHADE_COLOR, alpha=0.22, linewidth=0)
        ax.plot(dates, out["M1_before_mean"], color=ORIGINAL_COLOR, linewidth=1.1, linestyle="-")
        ax.plot(dates, out["M1_after_mean"], color=EFFECT_COLOR, linewidth=1.1, linestyle="--")
        ax.set_xlim(dates[0], dates[-1])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        _add_enso_annotations(ax, enso_periods, dates[0], dates[-1])
        fig.tight_layout()
        _savefig(fig, out_cmp / f"mean_comparison_{tag}.png", dpi=200)
        _savefig(fig, out_cmp / f"mean_comparison_{tag}.pdf")
        plt.close(fig)
        print(f"  wrote {out_cmp / f'mean_comparison_{tag}.png'} / .pdf")

        p_after_pct = 100.0 * P_after_lv[delta]
        pa_out = pd.DataFrame({
            "date": dates, "P_after_pct_mean": p_after_pct.mean(axis=0),
            "P_after_pct_lo": np.quantile(p_after_pct, 0.025, axis=0),
            "P_after_pct_hi": np.quantile(p_after_pct, 0.975, axis=0),
        })
        out2 = pd.concat([pa_out, pb_out.drop(columns="date")], axis=1)
        out2.to_csv(out_cmp / f"prob_comparison_{tag}.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(dates, out2["P_after_pct_lo"], out2["P_after_pct_hi"], color=CI_SHADE_COLOR, alpha=0.22, linewidth=0)
        ax.plot(dates, out2["P_before_pct_mean"], color=ORIGINAL_COLOR, linewidth=1.1, linestyle="-")
        ax.plot(dates, out2["P_after_pct_mean"], color=EFFECT_COLOR, linewidth=1.1, linestyle="--")
        ax.set_xlim(dates[0], dates[-1])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        _add_enso_annotations(ax, enso_periods, dates[0], dates[-1])
        fig.tight_layout()
        _savefig(fig, out_cmp / f"prob_comparison_{tag}.png", dpi=200)
        _savefig(fig, out_cmp / f"prob_comparison_{tag}.pdf")
        plt.close(fig)
        print(f"  wrote {out_cmp / f'prob_comparison_{tag}.png'} / .pdf")

    print(f"\nTotal time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
