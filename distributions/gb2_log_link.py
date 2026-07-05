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
from scipy.special import digamma, polygamma, betainc, betaincinv, betaln
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

    def logpdf_sum(self, y_arr: np.ndarray, **params) -> float:
        """Vectorized sum of log-densities over an array of positive observations.

        Identical mathematics to logpdf() but operates on the full array at
        once using numpy, avoiding the Python-loop overhead in _unconditional_mle.
        """
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)
        z = (y_arr / sigma) ** p
        ln_beta = betaln(a, b)
        ll = (
            np.log(p)
            - np.log(sigma)
            + (a * p - 1) * np.log(y_arr / sigma)
            - ln_beta
            - (a + b) * np.log(1.0 + z)
        )
        if not np.all(np.isfinite(ll)):
            return -np.inf
        return float(np.sum(ll))

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
        """Diagonal of the expected Fisher information for (phi, xi).

        With a=exp(xi), b=exp(zeta), p=exp(-gamma), sigma=exp(phi):

            I_{phi,phi} = p^2 * a * b / (a+b+1)

            I_{xi,xi}   = a^2 * [psi_1(a) - psi_1(a+b)]

        These are the log-space FI elements obtained by the chain rule
        I_{log-param, log-param} = (natural-param)^2 * I_{natural}.
        Full derivation in docs/math_proofs.tex §5.

        Returns raw values without clamping.  Callers must handle the case
        where a returned value is non-positive (indicates degenerate params).
        """
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)

        I_phi = p ** 2 * a * b / (a + b + 1.0)
        I_xi  = a ** 2 * (polygamma(1, a) - polygamma(1, a + b))

        return {"phi": float(I_phi), "xi": float(I_xi)}

    def fisher_info_full(self, **params) -> np.ndarray:
        """Full 2x2 Fisher information matrix for (phi, xi) in log-space.

        The matrix is:
            I = [[I_phi_phi,  I_phi_xi ],
                 [I_xi_phi,   I_xi_xi  ]]

        Cross-term derivation (chain rule from natural-parameter FI):
            I_{phi,xi} = I_{sigma,a} * sigma * a
                       = [-p/(sigma) * b/(a+b+1)] * sigma * a
                       = -p * a * b / (a+b+1)

        where I_{sigma,a} in natural space comes from MODELS.md:
            I_{phi_nat, xi_nat} = -1/(gamma_nat * phi_nat) * zeta_nat/(xi_nat+zeta_nat+1)
        with phi_nat = sigma, gamma_nat = 1/p, xi_nat = a, zeta_nat = b.

        Returns
        -------
        2x2 ndarray  [[I_pp, I_px], [I_xp, I_xx]] where p=phi, x=xi.
        """
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        denom = a + b + 1.0

        I_pp = p ** 2 * a * b / denom
        I_xi_xi = a ** 2 * (polygamma(1, a) - polygamma(1, a + b))
        I_px = -p * a * b / denom   # cross-term in log-space

        mat = np.array([[I_pp, I_px],
                        [I_px, I_xi_xi]])
        return mat

    def fisher_info_submatrix(self, tv_names: List[str], **params) -> np.ndarray:
        """Return the FI submatrix restricted to the requested tv_names.

        Parameters
        ----------
        tv_names : subset of ["phi", "xi"] in the order they appear in the
                   GAS parameter vector.

        Returns
        -------
        ndarray of shape (n, n) where n = len(tv_names).
        """
        all_names = ["phi", "xi"]
        full = self.fisher_info_full(**params)
        idx = [all_names.index(n) for n in tv_names]
        return full[np.ix_(idx, idx)]

    def cdf(self, y: float, **params) -> float:
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)
        z = (y / sigma) ** p
        u = np.clip(z / (1.0 + z), 1e-12, 1.0 - 1e-12)
        return float(betainc(a, b, u))

    def ppf(self, q: float | np.ndarray, **params) -> float | np.ndarray:
        """Quantile function for the positive GB2 component."""
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)
        q_arr = np.clip(np.asarray(q, dtype=float), 1e-12, 1.0 - 1e-12)
        u = betaincinv(a, b, q_arr)
        y = sigma * (u / (1.0 - u)) ** (1.0 / p)
        return float(y) if np.ndim(q) == 0 else y

    def rvs(self, n: int = 1, **params) -> np.ndarray:
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)
        U = np.random.beta(a, b, size=n)
        U = np.clip(U, 1e-12, 1.0 - 1e-12)
        return sigma * (U / (1.0 - U)) ** (1.0 / p)
