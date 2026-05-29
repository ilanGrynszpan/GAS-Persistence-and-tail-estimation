"""
GB2 distribution with log-link parameterisation.

Parameters (all on the unconstrained/log scale used in GAS dynamics):
  phi   -- log of the scale parameter σ  (time-varying)
  xi    -- log of shape parameter a       (time-varying)
  gamma -- log of the power parameter p   (static; p = exp(-gamma) in the
           standard GB2 notation so gamma = -log p > 0 for p < 1)
  zeta  -- log of shape parameter b       (static)

The density of Y > 0 is (McDonald 1984 / Creal et al. 2012):

  f(y) = p * y^{ap-1} / [σ^{ap} * B(a,b) * (1 + (y/σ)^p)^{a+b}]

with  a = exp(xi), b = exp(zeta), p = exp(-gamma), σ = exp(phi).
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
from scipy.special import digamma, polygamma, betainc
from scipy.special import beta as beta_fn

from distributions.base import Distribution


class GB2LogLink(Distribution):
    """GB2 distribution for strictly positive observations."""

    @property
    def tv_param_names(self) -> List[str]:
        return ["phi", "xi"]

    def _unpack(self, params: dict):
        phi = params["phi"]
        xi = params["xi"]
        gamma = params["gamma"]
        zeta = params["zeta"]
        return phi, xi, gamma, zeta

    def logpdf(self, y: float, **params) -> float:
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)
        z = (y / sigma) ** p
        return (
            np.log(p)
            - np.log(sigma)
            + (a * p - 1) * np.log(y / sigma)
            - np.log(beta_fn(a, b))
            - (a + b) * np.log(1.0 + z)
        )

    def score(self, y: float, **params) -> Dict[str, float]:
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)
        z = (y / sigma) ** p
        logy = np.log(y / sigma)

        # ∂logf/∂phi
        dphi = p * ((a + b) * z / (1.0 + z) - a)

        # ∂logf/∂xi  (xi = log a)
        dxi = a * (p * logy - digamma(a) + digamma(a + b) - np.log(1.0 + z))

        return {"phi": float(dphi), "xi": float(dxi)}

    def fisher_info_diag(self, **params) -> Dict[str, float]:
        """Diagonal of the expected Fisher information for (phi, xi)."""
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)

        # I_phi = p^2 * a*b / (a+b+1)  (exact, from McDonald 1984)
        I_phi = p ** 2 * a * b / (a + b + 1.0)

        # I_xi = a^2 * [ψ1(a) - ψ1(a+b)]  (exact, Beta Fisher information)
        I_xi = a ** 2 * (polygamma(1, a) - polygamma(1, a + b))

        return {"phi": float(max(I_phi, 1e-8)), "xi": float(max(I_xi, 1e-8))}

    def cdf(self, y: float, **params) -> float:
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)
        z = (y / sigma) ** p
        u = np.clip(z / (1.0 + z), 1e-12, 1.0 - 1e-12)
        return float(betainc(a, b, u))

    def rvs(self, n: int = 1, **params) -> np.ndarray:
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)
        U = np.random.beta(a, b, size=n)
        U = np.clip(U, 1e-12, 1.0 - 1e-12)
        return sigma * (U / (1.0 - U)) ** (1.0 / p)
