"""
Standalone (magnitude-independent) fit of a PiDynamics occurrence model.

=============================================================================
THEORY
=============================================================================

The zero-augmented log-likelihood is additively separable in pi (occurrence)
versus the positive-part magnitude distribution -- they share no parameters:

    l(theta) = sum_{y_t=0} log(1-pi_t) + sum_{y_t>0} [log(pi_t) + log g_y(y_t;...)]

Everywhere else in this codebase, pi_dynamics is fit *jointly* with the GAS
magnitude recursion inside one combined optimisation (ZAGASModel._run_filter),
purely for implementation convenience -- not because joint estimation is
statistically necessary. This module fits pi_dynamics on its own, using only
the binary occurrence indicator 1(y_t>0), by maximising exactly the pi-only
term of the log-likelihood above:

    l_pi(theta_pi) = sum_t [ y_t>0 ] * log(sigma(eta_t)) + [ y_t<=0 ] * log(1-sigma(eta_t))

where eta_t is produced by the PiDynamics subclass's own compute_eta()
recursion (e.g. ARLogisticPiDynamics: eta_t = omega0 + rho*eta_{t-1} +
sum_l omega_y_l * y_{t-l}) -- reused unchanged from pi_dynamics/ar_logistic.py,
not reimplemented.

=============================================================================
IMPLEMENTATION NOTES
=============================================================================

* Bounded L-BFGS-B warm start (PiDynamics.default_bounds) followed by an
  unbounded BFGS polish with a soft eta-magnitude penalty, matching this
  codebase's standard optimisation convention (models/static_gb2.py uses the
  identical two-step pattern for the GB2 side of the same separable
  likelihood) -- the bounded step alone is not treated as a final answer.
* No "effective sample" truncation: ARLogisticPiDynamics.compute_eta already
  guards its own seasonal lags (y_full[t-l] if t-l>=0 else 0.0), so the
  recursion is well-defined for every t from 0, unlike a GAS filter's
  positive-part recursion which needs a max_lag warm-up window.
"""

from __future__ import annotations
from typing import Optional

import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from pi_dynamics.base import PiDynamics

LAM_ETA   = 1e2    # matches models/za_gas_model.py's own eta penalty weight
ETA_LIMIT = 30.0    # matches models/za_gas_model.py's own eta soft bound


def _eta_path(pi_dyn: PiDynamics, theta: np.ndarray, y: np.ndarray, seasonal: str) -> np.ndarray:
    param_names = pi_dyn.param_names(seasonal)
    params = dict(zip(param_names, theta))
    T = len(y)
    eta = np.zeros(T)
    for t in range(T):
        eta[t] = pi_dyn.compute_eta(
            i=t, eta_hist=eta, y_full=y, t=t, params=params, seasonal=seasonal,
        )
    return eta


def _neg_loglik(theta: np.ndarray, pi_dyn: PiDynamics, y: np.ndarray, seasonal: str) -> float:
    eta = _eta_path(pi_dyn, theta, y, seasonal)
    if not np.all(np.isfinite(eta)):
        return 1e12
    penalty = float(np.sum(LAM_ETA * np.maximum(0.0, np.abs(eta) - ETA_LIMIT) ** 2))
    wet = y > 0
    ll = float(np.sum(log_expit(eta[wet]))) + float(np.sum(log_expit(-eta[~wet])))
    if not np.isfinite(ll):
        return 1e12 + penalty
    return -ll + penalty


def fit_pi_dynamics_standalone(
    y_train: np.ndarray,
    pi_dyn: Optional[PiDynamics] = None,
    seasonal: str = "daily",
    verbose: bool = False,
) -> dict:
    """
    Fit a PiDynamics occurrence model on y_train alone (no magnitude model
    involved). Defaults to ARLogisticPiDynamics -- the same occurrence model
    used everywhere else in this framework.

    Returns a dict with theta, param_names, loglik, eta_train (full in-sample
    eta_t path, t-indexed, length len(y_train)), validity, and diagnostics.
    """
    if pi_dyn is None:
        from pi_dynamics.ar_logistic import ARLogisticPiDynamics
        pi_dyn = ARLogisticPiDynamics()

    y_train = np.asarray(y_train, dtype=float)
    param_names = pi_dyn.param_names(seasonal)
    bounds = pi_dyn.default_bounds(seasonal)
    x0 = pi_dyn.initial_params(y_train, seasonal)

    t0 = time.time()
    res = minimize(
        _neg_loglik, x0, args=(pi_dyn, y_train, seasonal),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-6},
    )
    # Unbounded polish -- the bounded step above exists only to keep the
    # optimiser away from degenerate starting regions, not as a final answer
    # (same convention as models/static_gb2.py's GB2 fit).
    res2 = minimize(
        _neg_loglik, res.x, args=(pi_dyn, y_train, seasonal),
        method="BFGS", options={"maxiter": 500, "gtol": 1e-6},
    )
    if res2.fun < res.fun:
        res = res2
    runtime_s = time.time() - t0

    finite = bool(np.isfinite(res.fun) and np.all(np.isfinite(res.x)))
    eta_train = _eta_path(pi_dyn, res.x, y_train, seasonal) if finite else np.full(len(y_train), np.nan)

    if verbose:
        print(f"  pi_dynamics standalone fit: loglik={-res.fun:.2f} "
              f"runtime={runtime_s:.1f}s params={dict(zip(param_names, res.x))}")

    return {
        "theta": res.x, "param_names": param_names,
        "loglik": float(-res.fun) if finite else float("nan"),
        "eta_train": eta_train,
        "validity": "valid_converged" if finite else "failed",
        "success": finite, "runtime_s": runtime_s,
        "seasonal": seasonal,
        "pi_dyn_type": type(pi_dyn).__name__,
    }


def eta_path_full(pi_dyn: PiDynamics, theta: np.ndarray, y_full: np.ndarray, seasonal: str) -> np.ndarray:
    """Public wrapper: compute the full eta_t path for an already-fitted theta
    over an arbitrary y array (e.g. y_train concatenated with y_test, for OOS
    evaluation) -- 1-step-ahead in spirit, since compute_eta at t only reads
    y_full up to t and eta_hist up to t-1, never future values."""
    return _eta_path(pi_dyn, theta, y_full, seasonal)
