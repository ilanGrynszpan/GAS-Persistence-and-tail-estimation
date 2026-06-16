"""
Extended probabilistic scoring rules for ZA-GAS model evaluation.

Implements:
  - CRPS (Continuous Ranked Probability Score)
  - Threshold-weighted CRPS (twCRPS) at quantile levels
  - Quantile Score (QS) at multiple quantile levels
  - Exceedance Brier Score (EBS) at multiple probability levels
  - Upper-tail PIT diagnostics
  - Comprehensive metrics function (IS + OOS)

References
----------
Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules, prediction,
    and estimation. JASA, 102, 359–378.
Gneiting, T. & Ranjan, R. (2011). Comparing density forecasts using threshold-
    and quantile-weighted scoring rules. JBES, 29, 411–422.
Kupiec, P.H. (1995). Techniques for verifying the accuracy of risk measurement
    models. J. Derivatives, 3, 73–84.
Christoffersen, P.F. (1998). Evaluating interval forecasts. Int. Econ. Rev.,
    39, 841–862.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm, chi2 as _chi2

from diagnostics.residuals import pit_values, quantile_residuals
from diagnostics.tests import jarque_bera, kupiec_test, christoffersen_test


# ──────────────────────────────────────────────────────────────────────────────
# Monte-Carlo helpers shared across scoring rules
# ──────────────────────────────────────────────────────────────────────────────

def _draw_predictive(model, paths: dict, idx: int, n_draws: int, rng) -> np.ndarray:
    """Draw from the ZA-GB2 predictive distribution at time index idx."""
    pi_t   = float(paths["pi_oos"][idx])
    static = paths["static"]
    call   = {
        name: float(paths["f_arr_oos"][idx, j])
        for j, name in enumerate(paths["tv_names"])
    }
    call.update(static)

    is_pos = rng.binomial(1, pi_t, size=n_draws)
    draws  = np.zeros(n_draws)
    n_pos  = int(is_pos.sum())
    if n_pos > 0:
        try:
            if hasattr(model.dist, "ppf"):
                draws[is_pos == 1] = model.dist.ppf(
                    rng.uniform(1e-8, 1 - 1e-8, size=n_pos), **call
                )
            else:
                draws[is_pos == 1] = model.dist.rvs(n_pos, **call)
        except Exception:
            pass
    return draws


def _predictive_quantile(model, paths: dict, idx: int, q: float) -> float:
    """Quantile of the ZA predictive distribution (uses ppf if available)."""
    pi_t      = float(paths["pi_oos"][idx])
    zero_mass = 1.0 - pi_t
    if q <= zero_mass:
        return 0.0
    static = paths["static"]
    call   = {
        name: float(paths["f_arr_oos"][idx, j])
        for j, name in enumerate(paths["tv_names"])
    }
    call.update(static)
    q_pos = np.clip((q - zero_mass) / max(pi_t, 1e-12), 1e-12, 1 - 1e-12)
    if hasattr(model.dist, "ppf"):
        return float(model.dist.ppf(q_pos, **call))
    return np.nan


# ──────────────────────────────────────────────────────────────────────────────
# CRPS (Monte Carlo)
# ──────────────────────────────────────────────────────────────────────────────

def crps_mc(draws: np.ndarray, y_obs: float) -> float:
    """Energy-score CRPS: E|Y - y| - 0.5 * E|Y - Y'|."""
    s = np.sort(draws)
    e1 = float(np.mean(np.abs(draws - y_obs)))
    e2 = float(np.mean(np.abs(s - s[::-1])))
    return e1 - 0.5 * e2


# ──────────────────────────────────────────────────────────────────────────────
# Threshold-weighted CRPS (twCRPS)  — Gneiting & Ranjan (2011)
# ──────────────────────────────────────────────────────────────────────────────

def twcrps_mc(
    draws: np.ndarray,
    y_obs: float,
    threshold: float,
) -> float:
    """
    Threshold-weighted CRPS with weight function w(y) = I(y > threshold).

    twCRPS_t = E[|F_t(y_obs) - I(y_obs > threshold)|^2 * w(...)] (energy form)
    Approximated via:
        twCRPS = E[w(Y)|Y - y| ] - 0.5 * E[w(Y)|Y - Y'|]
    where w(Y) = I(Y > threshold).
    """
    w  = (draws > threshold).astype(float)
    e1 = float(np.mean(w * np.abs(draws - y_obs)))
    s  = np.sort(draws)
    ws = (s > threshold).astype(float)
    e2 = float(np.mean(ws * np.abs(s - s[::-1])))
    return e1 - 0.5 * e2


# ──────────────────────────────────────────────────────────────────────────────
# Quantile Score  (Gneiting & Raftery, 2007; Koenker & Bassett, 1978)
# ──────────────────────────────────────────────────────────────────────────────

def quantile_score(y_obs: float, q_hat: float, tau: float) -> float:
    """
    Quantile score: QS(τ) = (y_obs - q_hat) * (τ - I(y_obs < q_hat)).

    Lower is better.
    """
    return float((y_obs - q_hat) * (tau - float(y_obs < q_hat)))


# ──────────────────────────────────────────────────────────────────────────────
# Exceedance Brier Score
# ──────────────────────────────────────────────────────────────────────────────

def brier_score(y_obs: float, prob_exceed: float, threshold: float) -> float:
    """
    Binary Brier score for exceeding a fixed threshold.

    BS = (prob_exceed - I(y_obs > threshold))^2
    """
    return float((prob_exceed - float(y_obs > threshold)) ** 2)


def exceedance_prob(
    model,
    paths: dict,
    idx: int,
    threshold: float,
) -> float:
    """P(Y_t > threshold) = pi_t * (1 - G_t(threshold))."""
    pi_t   = float(paths["pi_oos"][idx])
    static = paths["static"]
    call   = {
        name: float(paths["f_arr_oos"][idx, j])
        for j, name in enumerate(paths["tv_names"])
    }
    call.update(static)
    try:
        G_threshold = model.dist.cdf(threshold, **call)
        p_exceed = pi_t * (1.0 - G_threshold)
    except Exception:
        p_exceed = np.nan
    return float(p_exceed)


# ──────────────────────────────────────────────────────────────────────────────
# Full evaluation (IS + OOS) — main entry point
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    model,
    theta: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    paths_is: dict,
    paths_oos: dict,
    cdfs_is: np.ndarray,
    cdfs_oos: np.ndarray,
    quantile_levels: Sequence[float] = (0.90, 0.95, 0.975, 0.99),
    twcrps_levels:   Sequence[float] = (0.90, 0.95),
    brier_levels:    Sequence[float] = (0.90, 0.95, 0.99),
    n_draws: int = 500,
    seed:    int = 42,
) -> dict:
    """
    Comprehensive IS + OOS metrics table.

    Computes for each sample:
      loglik, AIC, BIC, CRPS, twCRPS@90/95, QS@90/95/97.5/99,
      Brier@90/95/99, upper-tail PIT, Kupiec, Christoffersen,
      RMSE (all/dry/wet), MAD (all/dry/wet), JB on quantile residuals.

    IS paths must have keys "pi", "f_arr", "tv_names", "static", "y_eff".
    OOS paths must have keys "pi_oos", "f_arr_oos", "tv_names", "static".
    """
    from scipy.special import beta as beta_fn

    rng = np.random.default_rng(seed)

    def _gb2_mean(call: dict) -> float:
        phi   = call.get("phi", 0.0)
        xi    = call.get("xi", np.log(1.2))
        gamma = call.get("gamma", 1.0)
        zeta  = call.get("zeta", 3.0)
        a     = np.exp(xi)
        b     = np.exp(zeta)
        p     = np.exp(-gamma)
        sigma = np.exp(phi)
        try:
            return float(sigma * beta_fn(a + p, b - p) / beta_fn(a, b))
        except Exception:
            return float(sigma)

    # ── IS preparation ────────────────────────────────────────────────────────
    # Adapt IS paths to OOS-style keys for reuse
    is_oos_style = {
        "f_arr_oos": paths_is["f_arr"],
        "pi_oos":    paths_is["pi"],
        "static":    paths_is["static"],
        "tv_names":  paths_is["tv_names"],
        "y_test":    paths_is["y_eff"],
    }
    y_is = paths_is["y_eff"]

    n_params = model.n_params if hasattr(model, "n_params") else len(theta)
    is_loglik  = float(paths_is.get("loglik", 0.0))
    oos_loglik = float(sum(
        np.log(np.clip(
            float(cdfs_oos[i]) if y_test[i] > 0 else (1.0 - float(paths_oos["pi_oos"][i])),
            1e-300, np.inf
        ))
        for i in range(len(y_test))
    )) if False else 0.0  # placeholder – computed in block below

    # ── Helper: compute one sample's metrics ─────────────────────────────────

    def _sample_metrics(
        y: np.ndarray,
        paths_oos_style: dict,
        cdfs: np.ndarray,
        loglik_val: float,
    ) -> dict:
        n = len(y)
        T = max(n, 1)

        # Predictive means
        means = np.array([
            paths_oos_style["pi_oos"][i] * _gb2_mean(
                {nm: paths_oos_style["f_arr_oos"][i, j]
                 for j, nm in enumerate(paths_oos_style["tv_names"])}
                | paths_oos_style["static"]
            )
            for i in range(n)
        ])

        rmse     = float(np.sqrt(np.mean((means - y) ** 2)))
        mad      = float(np.mean(np.abs(means - y)))
        dry_mask = y == 0
        wet_mask = y > 0
        rmse_dry = float(np.sqrt(np.mean((means[dry_mask] - y[dry_mask]) ** 2))) if dry_mask.any() else np.nan
        rmse_wet = float(np.sqrt(np.mean((means[wet_mask] - y[wet_mask]) ** 2))) if wet_mask.any() else np.nan
        mad_dry  = float(np.mean(np.abs(means[dry_mask] - y[dry_mask])))         if dry_mask.any() else np.nan
        mad_wet  = float(np.mean(np.abs(means[wet_mask] - y[wet_mask])))         if wet_mask.any() else np.nan

        # PIT / QR
        pit = pit_values(cdfs, y, randomise_zeros=True, rng=rng)
        qr  = quantile_residuals(cdfs, y, randomise_zeros=True, rng=rng)
        jb  = jarque_bera(qr)

        # CRPS + twCRPS (Monte Carlo)
        crps_vals    = []
        twcrps_vals  = {lv: [] for lv in twcrps_levels}
        qs_vals      = {tau: [] for tau in quantile_levels}
        brier_vals   = {lv: [] for lv in brier_levels}

        for i in range(n):
            draws = _draw_predictive(model, paths_oos_style, i, n_draws, rng)
            crps_vals.append(crps_mc(draws, y[i]))
            for lv in twcrps_levels:
                thr = float(np.quantile(y[y > 0], lv)) if np.any(y > 0) else 0.0
                twcrps_vals[lv].append(twcrps_mc(draws, y[i], thr))
            for tau in quantile_levels:
                q_hat = _predictive_quantile(model, paths_oos_style, i, tau)
                qs_vals[tau].append(quantile_score(y[i], q_hat, tau))
            for lv in brier_levels:
                thr = float(np.quantile(y[y > 0], lv)) if np.any(y > 0) else 0.0
                p_exc = exceedance_prob(model, paths_oos_style, i, thr)
                brier_vals[lv].append(brier_score(y[i], p_exc, thr))

        crps  = float(np.mean(crps_vals))
        aic   = float(-2.0 * loglik_val + 2.0 * n_params)
        bic   = float(-2.0 * loglik_val + n_params * np.log(max(T, 1)))

        # Coverage at alpha = 0.05 (95% upper tail)
        kup_95  = kupiec_test(pit, alpha=0.05)
        chrf_95 = christoffersen_test(pit, alpha=0.05)
        # Upper-tail PIT: pit_t for y_t in top 10%
        upper_mask = y > np.quantile(y[y > 0], 0.90) if np.any(y > 0) else np.zeros(n, bool)

        row = {
            "loglik":     loglik_val,
            "aic":        aic,
            "bic":        bic,
            "crps":       crps,
            "rmse":       rmse,
            "mad":        mad,
            "rmse_dry":   rmse_dry,
            "rmse_wet":   rmse_wet,
            "mad_dry":    mad_dry,
            "mad_wet":    mad_wet,
            "jb_stat":    jb["stat"],
            "jb_pvalue":  jb["pvalue"],
            "pit_mean":   float(np.mean(pit)),
            "pit_std":    float(np.std(pit)),
            "n":          n,
            # Kupiec 95%
            "kup95_stat":    kup_95["stat"],
            "kup95_pvalue":  kup_95["pvalue"],
            "kup95_reject":  kup_95["reject_5pct"],
            # Christoffersen 95%
            "chrf95_LRcc":   chrf_95["LR_cc"],
            "chrf95_pvalue": chrf_95["pvalue_cc"],
            "chrf95_reject": chrf_95["reject_5pct_cc"],
        }

        for lv in twcrps_levels:
            row[f"twcrps_{int(lv*100)}"] = float(np.mean(twcrps_vals[lv]))
        for tau in quantile_levels:
            row[f"qs_{int(tau*1000)}"] = float(np.mean(qs_vals[tau]))
        for lv in brier_levels:
            row[f"brier_{int(lv*100)}"] = float(np.mean(brier_vals[lv]))

        return row, pit, qr

    # ── IS ───────────────────────────────────────────────────────────────────
    # Compute OOS loglik properly
    oos_loglik_val = 0.0
    for i in range(len(y_test)):
        pi_t = float(paths_oos["pi_oos"][i])
        if y_test[i] <= 0:
            ll = np.log(max(1.0 - pi_t, 1e-300))
        else:
            call = {
                nm: float(paths_oos["f_arr_oos"][i, j])
                for j, nm in enumerate(paths_oos["tv_names"])
            }
            call.update(paths_oos["static"])
            ll = np.log(max(pi_t, 1e-300)) + model.dist.logpdf(y_test[i], **call)
        if np.isfinite(ll):
            oos_loglik_val += ll

    is_metrics, is_pit, is_qr = _sample_metrics(
        y_is, is_oos_style, cdfs_is, is_loglik
    )
    oos_metrics, oos_pit, oos_qr = _sample_metrics(
        y_test, paths_oos, cdfs_oos, oos_loglik_val
    )

    return {
        "is":      is_metrics,
        "oos":     oos_metrics,
        "is_pit":  is_pit,
        "oos_pit": oos_pit,
        "is_qr":   is_qr,
        "oos_qr":  oos_qr,
    }


def metrics_to_frame(
    metrics_by_model: Dict[str, dict],
    sample: str = "oos",
) -> pd.DataFrame:
    """Flatten per-model metrics into a comparison DataFrame."""
    rows = []
    for model_id, m in metrics_by_model.items():
        row = {"model_id": model_id}
        row.update(m[sample])
        rows.append(row)
    return pd.DataFrame(rows)


def latex_metrics_table(
    frame: pd.DataFrame,
    cols: Optional[List[str]] = None,
    caption: str = "Model comparison",
    label: str = "tab:metrics",
    landscape: bool = True,
) -> str:
    """LaTeX table of model comparison metrics (landscape-friendly)."""
    if cols is None:
        cols = [
            "model_id", "loglik", "aic", "bic", "crps",
            "rmse", "mad", "twcrps_90", "twcrps_95",
            "qs_900", "qs_950", "brier_90", "brier_95",
            "kup95_pvalue", "chrf95_pvalue",
        ]
    present = [c for c in cols if c in frame.columns]
    sub = frame[present].copy()

    def _fmtval(v):
        if pd.isna(v):
            return "--"
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    header = " & ".join(c.replace("_", r"\_") for c in present) + r" \\"
    body_lines = []
    for _, row in sub.iterrows():
        body_lines.append(" & ".join(_fmtval(row[c]) for c in present) + r" \\")
    body = "\n".join(body_lines)

    env = "landscape" if landscape else ""
    wrap_open  = r"\begin{landscape}" + "\n" if landscape else ""
    wrap_close = r"\end{landscape}" + "\n" if landscape else ""

    return (
        wrap_open
        + r"\begin{table}[ht]" + "\n"
        + r"\centering" + "\n"
        + r"\tiny" + "\n"
        + rf"\caption{{{caption}}}" + "\n"
        + rf"\label{{{label}}}" + "\n"
        + rf"\begin{{tabular}}{{{'l' + 'r' * (len(present)-1)}}}" + "\n"
        + r"\hline" + "\n"
        + header + "\n"
        + r"\hline" + "\n"
        + body + "\n"
        + r"\hline" + "\n"
        + r"\end{tabular}" + "\n"
        + r"\end{table}" + "\n"
        + wrap_close
    )
