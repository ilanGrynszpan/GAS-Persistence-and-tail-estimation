"""
Analytical rolling one-step-ahead marginal effects
(docs/claude_code_marginal_effects_instructions.md Sec.1-2).

=============================================================================
MATHEMATICAL SUMMARY  (see docs/math_proofs.tex for the full derivation)
=============================================================================

For the positive-GB2 component, with m1_t = phi_t*R_{1,t}, m2_t =
phi_t^2*R_{2,t}, sigma2_t = m2_t - m1_t^2, O_t^k = psi(xi_t+k*gamma) -
psi(xi_t) (xi_t, gamma here in the natural/exponentiated scale -- see
distributions.gb2_log_link.GB2LogLink.psi_diff's own docstring):

  phi-only models (xi static):
      d m1/dx    = c_v^phi * m1_t
      d sigma2/dx = 2 * c_v^phi * sigma2_t

  phi+xi models (both dynamic):
      d m1/dx     = m1_t * (c_v^phi + xi_t * c_v^xi * O_t^1)
      d sigma2/dx = 2*c_v^phi*sigma2_t + xi_t*c_v^xi*(m2_t*O_t^2 - 2*m1_t^2*O_t^1)

For the zero-augmented mixture, M1_t = pi_t*m1_t, VM_t = pi_t*sigma2_t +
pi_t*(1-pi_t)*m1_t^2, and (full product rule, d pi_t/dx = c_v^pi *
pi_t*(1-pi_t) for a logit occurrence predictor -- 0 today since no
pi_dynamics/ class in this repo accepts covariates, see
coefficients.occurrence_coefficient):

  d M1/dx = pi_t * d m1/dx + (d pi_t/dx) * m1_t
  d VM/dx = pi_t * d sigma2/dx + 2*pi_t*(1-pi_t)*m1_t*(d m1/dx)
            + (d pi_t/dx) * (sigma2_t + (1-2*pi_t)*m1_t^2)

c_v^phi, c_v^xi, c_v^pi are RAW-physical-unit coefficients from
marginal_effects.coefficients (already divided by the covariate's training
std). All four derivatives are 0 at any origin where the variable does not
enter the model (coefficients.CovariateCoefficients.in_model is False).

=============================================================================
WHERE phi_t, xi_t, pi_t COME FROM
=============================================================================

Exactly the model's own one-step-ahead rolling filter over the OOS window
-- CovZAGASModel.simulate_oos / HarveyZAGASModel.simulate_oos (no
re-implementation of the filter). "The marginal effect happens at time t,
not anytime in the projected sample or the future" (instructions doc Sec.2
step 2) is satisfied because phi_{t|t-1}/xi_{t|t-1}/pi_{t|t-1} are the
REAL rolling filter's states (built from realised y up to t-1, holding the
realised score fixed), never a simulated/forecast continuation.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from marginal_effects.model_loader import LoadedModel
from marginal_effects.coefficients import VARIABLES, covariate_coefficients

QUANTILE_LEVELS = (0.025, 0.25, 0.75, 0.975)


def _oos_states(loaded: LoadedModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, pd.DatetimeIndex]:
    """Returns (phi_oos, xi_oos_or_None, pi_oos, static_params, dates_oos)
    from the model's own simulate_oos, dispatching on model type."""
    model, theta, data = loaded.model, loaded.theta, loaded.data
    y_train, y_test = data["y_train"], data["y_test"]
    oos = model.simulate_oos(theta, y_train, y_test, **loaded.ms_kwargs)

    tv_names = list(oos["tv_names"])
    static = oos["static"]
    dates_oos = pd.DatetimeIndex(data["dates_test"])[: len(y_test)]

    if "f_arr_oos" in oos and oos["f_arr_oos"] is not None and oos["f_arr_oos"].ndim == 2 \
            and "phi_oos" not in oos:
        # Harvey-style return: f_arr_oos columns ordered by tv_names.
        f_arr = oos["f_arr_oos"]
        phi_oos = f_arr[:, tv_names.index("phi")]
        xi_oos = f_arr[:, tv_names.index("xi")] if "xi" in tv_names else None
    else:
        # Cov/base-style return: phi_oos / xi_oos given directly.
        phi_oos = oos["phi_oos"]
        xi_oos = oos.get("xi_oos")

    pi_oos = oos["pi_oos"]
    return phi_oos, xi_oos, pi_oos, static, dates_oos


def rolling_one_step_effects(loaded: LoadedModel, variable: str) -> pd.DataFrame:
    """One row per OOS rolling-forecast date: analytical marginal effects of
    `variable` (one of coefficients.VARIABLES) on the positive-GB2 mean
    (dm1_dx) and variance (dsigma2_dx), and on the full ZA-GB2 mean
    (dM1_dx) and variance (dVM_dx). All four columns are identically 0.0
    (not NaN) if the model does not use `variable`."""
    dist = loaded.model.dist
    coef = covariate_coefficients(loaded, variable)
    phi_oos, xi_oos, pi_oos, static, dates_oos = _oos_states(loaded)
    n = len(phi_oos)
    xi_is_dynamic = xi_oos is not None

    gamma_static = static["gamma"]
    zeta_static = static["zeta"]
    xi_static = static.get("xi")  # only present when xi is NOT dynamic

    dm1 = np.zeros(n)
    dsigma2 = np.zeros(n)
    dM1 = np.zeros(n)
    dVM = np.zeros(n)
    m1_arr = np.full(n, np.nan)
    sigma2_arr = np.full(n, np.nan)

    for t in range(n):
        phi_t = phi_oos[t]
        xi_state_t = xi_oos[t] if xi_is_dynamic else xi_static
        params = dict(phi=phi_t, xi=xi_state_t, gamma=gamma_static, zeta=zeta_static)
        m1_t = dist.raw_moment(1, **params)
        m2_t = dist.raw_moment(2, **params)
        if not (np.isfinite(m1_t) and np.isfinite(m2_t)):
            dm1[t] = dsigma2[t] = dM1[t] = dVM[t] = np.nan
            continue
        sigma2_t = m2_t - m1_t ** 2
        m1_arr[t] = m1_t
        sigma2_arr[t] = sigma2_t

        if coef.in_model:
            if xi_is_dynamic:
                xi_nat_t = float(np.exp(xi_state_t))  # docs' xi_t is natural-scale (a = exp(xi_state))
                O1_t = dist.psi_diff(1, **params)
                O2_t = dist.psi_diff(2, **params)
                dm1_t = m1_t * (coef.c_phi + xi_nat_t * coef.c_xi * O1_t)
                dsigma2_t = (2 * coef.c_phi * sigma2_t
                             + xi_nat_t * coef.c_xi * (m2_t * O2_t - 2 * m1_t ** 2 * O1_t))
            else:
                dm1_t = coef.c_phi * m1_t
                dsigma2_t = 2 * coef.c_phi * sigma2_t
        else:
            dm1_t = dsigma2_t = 0.0

        pi_t = float(pi_oos[t])
        dpi_t = coef.c_pi * pi_t * (1.0 - pi_t)  # 0.0 today (see coefficients.occurrence_coefficient)
        dM1_t = pi_t * dm1_t + dpi_t * m1_t
        dVM_t = (pi_t * dsigma2_t + 2.0 * pi_t * (1.0 - pi_t) * m1_t * dm1_t
                 + dpi_t * (sigma2_t + (1.0 - 2.0 * pi_t) * m1_t ** 2))

        dm1[t], dsigma2[t], dM1[t], dVM[t] = dm1_t, dsigma2_t, dM1_t, dVM_t

    return pd.DataFrame({
        "date": dates_oos[:n],
        "variable": variable,
        "in_model": coef.in_model,
        "phi_t": phi_oos[:n],
        "xi_t_natural": (np.exp(xi_oos[:n]) if xi_is_dynamic else np.full(n, np.exp(xi_static))),
        "pi_t": pi_oos[:n],
        "m1_t": m1_arr,
        "sigma2_t": sigma2_arr,
        "dm1_dx": dm1,
        "dsigma2_dx": dsigma2,
        "dM1_dx": dM1,
        "dVM_dx": dVM,
    })


def all_variables_one_step(loaded: LoadedModel) -> pd.DataFrame:
    """Concatenates rolling_one_step_effects over every named variable."""
    return pd.concat([rolling_one_step_effects(loaded, v) for v in VARIABLES], ignore_index=True)


def summarize(df_one_variable: pd.DataFrame) -> Dict[str, float]:
    """Mean/median/std + the instructions doc's 2.5/25/75/97.5% empirical
    quantiles for each of the 4 effect columns, over the rolling OOS window
    (finite values only)."""
    out: Dict[str, float] = {}
    for col in ("dm1_dx", "dsigma2_dx", "dM1_dx", "dVM_dx"):
        x = df_one_variable[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            out.update({f"{col}_mean": np.nan, f"{col}_median": np.nan, f"{col}_std": np.nan})
            for q in QUANTILE_LEVELS:
                out[f"{col}_q{q}"] = np.nan
            continue
        out[f"{col}_mean"] = float(np.mean(x))
        out[f"{col}_median"] = float(np.median(x))
        out[f"{col}_std"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        for q in QUANTILE_LEVELS:
            out[f"{col}_q{q}"] = float(np.quantile(x, q))
    out["n_obs"] = int(len(df_one_variable))
    out["n_finite"] = int(np.isfinite(df_one_variable["dM1_dx"].to_numpy(dtype=float)).sum())
    return out
