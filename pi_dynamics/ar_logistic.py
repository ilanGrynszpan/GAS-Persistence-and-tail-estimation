"""
AR-Logistic π dynamics.

The latent logit η_t follows an AR(1) process augmented with contemporaneous
lags of y:

    η_t = ω₀ + ρ · η_{t-1} + Σ_{l ∈ K} ωˡ · y_{t-l}

where K is the seasonal lag set:
  - daily:   K = [1, 2, 3, 364, 365, 366, 367]
  - monthly: K = [1, 2, 3, 11, 12, 13]

Lags y_{t-l} are taken from the full original series so that the model can
access both short-memory lags (1-3 days) and the same-period-last-year lags.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from pi_dynamics.base import PiDynamics
from constants import SEASONAL_LAGS


class ARLogisticPiDynamics(PiDynamics):
    """
    AR(1) logistic dynamics with seasonal y-lags for η_t.

    Parameters (in flat vector order):
        omega0         -- intercept
        rho            -- AR(1) coefficient on η_{t-1}
        omega_y[0..K-1] -- coefficients on lagged y values
    """

    def param_names(self, seasonal: str) -> List[str]:
        lags = SEASONAL_LAGS[seasonal]
        return ["omega0", "rho"] + [f"omega_y_{l}" for l in lags]

    def default_bounds(self, seasonal: str):
        lags = SEASONAL_LAGS[seasonal]

        if seasonal == "daily":
            return [(-5.0, 2.0), (-0.80, 0.80)] + [(-0.10, 0.10)] * len(  # omega0, rho
                lags
            )  # omega_y

        return [(-5.0, 2.0), (-0.90, 0.90)] + [(-1.0, 1.0)] * len(lags)

    def initial_params(self, y: np.ndarray, seasonal: str) -> np.ndarray:
        lags = SEASONAL_LAGS[seasonal]
        pi_hat = np.mean(y > 0)
        pi_hat = np.clip(pi_hat, 0.01, 0.99)
        omega0 = float(np.log(pi_hat / (1.0 - pi_hat)))
        return np.array([omega0, 0.5] + [0.0] * len(lags))

    def compute_eta(
        self,
        i: int,
        eta_hist: np.ndarray,
        y_full: np.ndarray,
        t: int,
        params: dict,
        seasonal: str,
    ) -> float:
        lags = SEASONAL_LAGS[seasonal]
        omega0 = params["omega0"]
        rho = params["rho"]
        omega_y = np.array([params[f"omega_y_{l}"] for l in lags])

        eta_prev = eta_hist[i - 1] if i > 0 else 0.0

        y_lags = np.array([y_full[t - l] if (t - l) >= 0 else 0.0 for l in lags])

        return float(omega0 + rho * eta_prev + np.dot(omega_y, y_lags))
