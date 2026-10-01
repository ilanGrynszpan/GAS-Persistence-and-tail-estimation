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
        """`phi` (and only `phi`) may be a numpy array; returns an array of
        the same shape in that case, a Python float otherwise -- matching
        raw_moment/exceedance_effect_kernel's identical convention."""
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)
        z = (y / sigma) ** p
        u = np.clip(z / (1.0 + z), 1e-12, 1.0 - 1e-12)
        result = betainc(a, b, u)
        return float(result) if np.ndim(phi) == 0 else result

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

    def raw_moment(self, k: int, **params) -> float:
        """
        k-th raw moment of the positive GB2 component:

            E[Y^k] = sigma^k * B(a + k/p, b - k/p) / B(a, b)

        which only exists (is finite) for b > k/p, i.e. k < b*p. Returns
        NaN rather than a spurious huge/negative value when this condition
        fails, so that a genuinely non-existent moment (heavy enough tail)
        is distinguishable from a normal finite one -- this matters because
        b - k/p appears as a Beta-function argument and scipy's beta_fn
        silently returns a large-magnitude or non-finite value once that
        argument crosses zero, rather than raising.

        `phi` (and only `phi`) may be a numpy array; xi/gamma/zeta must stay
        scalar (arg2 = b - k/p depends only on them, so the existence check
        above is well-defined either way). Returns an array of the same
        shape when phi is an array, a Python float otherwise -- matching
        exceedance_effect_kernel's identical convention.
        """
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)
        arg2 = b - k / p
        if not (arg2 > 0):
            return np.full_like(sigma, np.nan) if np.ndim(phi) > 0 else float("nan")
        result = sigma ** k * beta_fn(a + k / p, arg2) / beta_fn(a, b)
        return float(result) if np.ndim(phi) == 0 else result

    def mean(self, **params) -> float:
        """E[Y] for the positive GB2 component (see raw_moment)."""
        return self.raw_moment(1, **params)

    def R_moment_ratio(self, k: int, **params) -> float:
        """
        R_{k,t} = E[Y^k] / sigma^k = B(a + k/p, b - k/p) / B(a, b), i.e. the
        moment ratio used by the marginal-effects formulas in
        docs/claude_code_marginal_effects_instructions.md (there written
        R_{k,t} = B(xi_t+k*gamma, zeta-k*gamma)/B(xi_t,zeta) in the natural
        -scale notation of docs/MODELS.md Sec.5, where a=exp(xi)=xi_t,
        b=exp(zeta)=zeta, and k/p = k*exp(gamma_code) = k*gamma_t in that
        same natural-scale notation). Equivalent to raw_moment(k,...) /
        sigma**k, exposed separately because the marginal-effects formulas
        need R_{k,t} directly (not scaled by sigma^k). Returns NaN under the
        same non-existence condition as raw_moment (b <= k/p).
        """
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        arg2 = b - k / p
        if not (arg2 > 0):
            return float("nan")
        return float(beta_fn(a + k / p, arg2) / beta_fn(a, b))

    def psi_diff(self, k: float, **params) -> float:
        """
        O_t^k = psi(xi_t + k*gamma) - psi(xi_t) in the natural-scale
        notation of docs/claude_code_marginal_effects_instructions.md
        Sec.1 (xi_t = a = exp(xi), gamma = 1/p = exp(gamma_code) here).
        Used by the ZA-GB2 marginal-effects formulas for models where both
        phi and xi are time-varying (the O_t^1, O_t^2 terms). k need not be
        an integer (k=1, k=2 are the only values the formulas use).
        """
        _, xi, gamma, _ = self._unpack(params)
        a = np.exp(xi)
        gamma_nat = np.exp(gamma)  # = 1/p, MODELS.md Sec.5's "gamma"
        return float(digamma(a + k * gamma_nat) - digamma(a))

    def exceedance_effect_kernel(self, u, **params) -> float:
        """
        d S(u)/d phi_t, expressed as a pure function of (u, phi, xi, gamma,
        zeta) -- the kernel of the exceedance-probability marginal-effect
        formula (docs/math_proofs.tex Sec.12; user-supplied formula,
        2026-09-18):

            d P(Y_t > u) / d x_{v,t-1}
                = c_v^phi * w^xi_t * (1-w)^zeta / (gamma_t * B(xi_t, zeta)),
            w = z/(1+z),  z = (u/phi_t)^{1/gamma_t},

        all in the natural-scale notation of R_moment_ratio/psi_diff
        (xi_t=a=exp(xi), zeta=b=exp(zeta), gamma_t=1/p=exp(gamma_code),
        phi_t=sigma=exp(phi)). This method returns the kernel alone (i.e.
        the RHS above divided by c_v^phi); multiply by c_v^phi externally
        exactly as the other one-step marginal-effect formulas do
        (marginal_effects.onestep multiplies raw_moment()/psi_diff() by
        c_v the same way). Computed in log-space (log w, log(1-w) via
        log1p, and betaln) to avoid under/overflow when w is close to 0
        or 1, or xi/zeta are large -- direct w**xi*(1-w)**zeta would not
        be safe there.

        Returns 0.0 (not NaN) when u<=0, matching the formula's own
        "y_t>0" domain restriction (the positive-GB2 survival function is
        undefined/degenerate at u<=0 in this zero-augmented setting).

        `phi` (and only `phi`) may be a numpy array (e.g. a whole filtered
        f_arr[:,0] path); xi/gamma/zeta must stay scalar (the formula holds
        them static -- this is the phi-only marginal-effect case). Returns
        an array in that case, a Python float when phi is scalar too,
        matching distributions.gb2_log_link's existing ppf() convention.
        """
        if u <= 0:
            return 0.0
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        gamma_nat = np.exp(gamma)  # = 1/p
        sigma = np.exp(phi)
        z = (u / sigma) ** p
        log_w = np.log(z) - np.log1p(z)          # log(z/(1+z))
        log_1mw = -np.log1p(z)                    # log(1/(1+z))
        log_kernel = a * log_w + b * log_1mw - np.log(gamma_nat) - betaln(a, b)
        result = np.exp(log_kernel)
        return float(result) if np.ndim(phi) == 0 else result

    def rvs(self, n: int = 1, **params) -> np.ndarray:
        phi, xi, gamma, zeta = self._unpack(params)
        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)
        U = np.random.beta(a, b, size=n)
        U = np.clip(U, 1e-12, 1.0 - 1e-12)
        return sigma * (U / (1.0 - U)) ** (1.0 / p)
