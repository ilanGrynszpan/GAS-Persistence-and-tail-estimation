"""
Configurable-lag AR-logistic pi dynamics -- Stage 1 occurrence-dynamics
comparison (2026-07-14).

=============================================================================
PURPOSE
=============================================================================

pi_dynamics/ar_logistic.py already implements

    eta_t = omega0 + rho * eta_{t-1} + sum_{l in K} omega_l * y_{t-l}

with K = SEASONAL_LAGS[seasonal] (short lags {1,365,366} for "daily",
short calendar lags for "monthly") -- this is the "existing" occurrence
model used everywhere else in the framework (Stages 1-4).

This module answers a different scientific question: does the AR(1) term
on eta_{t-1} matter, and does using purely short-memory rainfall lags
{1,2,3} (instead of the existing K) change the fitted occurrence
dynamics? Per docs/MODELS.md §6/§30 ("configuration over duplication"),
this is implemented as ONE class parameterised by the lag set and an
"include_ar" flag, rather than two near-duplicate files:

    ARLogisticCustomLagsPiDynamics(lags=[1, 2, 3], include_ar=True)
        eta_t = omega0 + rho * eta_{t-1} + sum_{l=1,2,3} omega_l * y_{t-l}

    ARLogisticCustomLagsPiDynamics(lags=[1, 2, 3], include_ar=False)
        eta_t = omega0 + sum_{l=1,2,3} omega_l * y_{t-l}

pi_dynamics/ar_logistic.py itself is NOT modified -- this is an additive
sibling class implementing the same PiDynamics interface (pi_dynamics/base.py),
so it plugs into fit_pi_dynamics_standalone() and ZAGASModel unchanged.

=============================================================================
IMPLEMENTATION NOTES
=============================================================================

* `seasonal` is accepted by every PiDynamics abstract method for interface
  compatibility (see pi_dynamics/base.py) but is NOT used to select the lag
  set here -- the lag set is fixed at construction time via `lags`. This is
  the deliberate difference from ARLogisticPiDynamics, whose lag set is
  chosen by `seasonal` via constants.SEASONAL_LAGS.
* Like ARLogisticPiDynamics.compute_eta, out-of-range lookups y_full[t-l]
  for (t-l) < 0 are treated as 0 rather than raising -- this is well-defined
  from t=0 onward, no warm-up window is required (see
  pi_dynamics/standalone_fit.py's module docstring for why this matters:
  no "effective sample" truncation is needed for pi-only standalone fits).
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from pi_dynamics.base import PiDynamics


class ARLogisticCustomLagsPiDynamics(PiDynamics):
    """
    AR-logistic occurrence dynamics with an explicit, caller-supplied lag
    set and an optional AR(1) term on the latent logit itself.

    Parameters
    ----------
    lags        : list of positive integer lags l used in
                  sum_l omega_l * y_{t-l}.  For the short-memory comparison
                  in this experiment, lags=[1, 2, 3].
    include_ar  : if True, include the rho * eta_{t-1} autoregressive term
                  (matches the functional form of ARLogisticPiDynamics,
                  just with a different lag set). If False, eta_t depends
                  only on the contemporaneous y-lags (a pure "moving
                  average in y" occurrence model, no memory in eta itself).

    Parameters (flat vector order):
        omega0          -- intercept
        rho             -- AR(1) coefficient on eta_{t-1}   [only if include_ar]
        omega_y[0..K-1] -- coefficients on lagged y values, in `lags` order
    """

    def __init__(self, lags: List[int], include_ar: bool = True):
        if not lags or any(l <= 0 for l in lags):
            raise ValueError(
                f"ARLogisticCustomLagsPiDynamics requires a non-empty list "
                f"of strictly positive lags, got {lags!r}."
            )
        self.lags = list(lags)
        self.include_ar = bool(include_ar)

    # ------------------------------------------------------------------
    def param_names(self, seasonal: str) -> List[str]:
        names = ["omega0"]
        if self.include_ar:
            names.append("rho")
        names += [f"omega_y_{l}" for l in self.lags]
        return names

    def default_bounds(self, seasonal: str):
        bounds = [(-5.0, 2.0)]  # omega0, same range as ARLogisticPiDynamics
        if self.include_ar:
            bounds.append((-0.95, 0.95))  # rho
        # Short rainfall lags (l in {1,2,3}) act directly on eta in mm units;
        # keep the same per-lag bound used by ARLogisticPiDynamics for its
        # daily short lags (l=1) so the warm-start L-BFGS-B step is not
        # artificially tight.
        bounds += [(-0.10, 0.10)] * len(self.lags)
        return bounds

    def initial_params(self, y: np.ndarray, seasonal: str) -> np.ndarray:
        pi_hat = np.mean(y > 0)
        pi_hat = np.clip(pi_hat, 0.01, 0.99)
        omega0 = float(np.log(pi_hat / (1.0 - pi_hat)))
        theta0 = [omega0]
        if self.include_ar:
            theta0.append(0.5)
        theta0 += [0.0] * len(self.lags)
        return np.array(theta0)

    def compute_eta(
        self,
        i: int,
        eta_hist: np.ndarray,
        y_full: np.ndarray,
        t: int,
        params: dict,
        seasonal: str,
    ) -> float:
        omega0 = params["omega0"]
        y_lags = np.array([y_full[t - l] if (t - l) >= 0 else 0.0 for l in self.lags])
        omega_y = np.array([params[f"omega_y_{l}"] for l in self.lags])
        lag_term = float(np.dot(omega_y, y_lags))

        if not self.include_ar:
            return float(omega0 + lag_term)

        rho = params["rho"]
        eta_prev = eta_hist[i - 1] if i > 0 else 0.0
        return float(omega0 + rho * eta_prev + lag_term)
