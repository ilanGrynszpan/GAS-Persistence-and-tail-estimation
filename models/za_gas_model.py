"""
Zero-Augmented GAS (ZA-GAS) model.

=============================================================================
STRUCTURE
=============================================================================

The full model is:

    p(y_t | F_{t-1})  =  I(y_t = 0) * (1 - pi_t)
                       +  I(y_t > 0) *  pi_t * g(y_t ; f_t, theta_static)

where:
    pi_t  =  sigma(eta_t)            [probability of a non-zero observation]
    eta_t =  <pi dynamics>           [logit process, see PiDynamics]
    f_t   =  <GAS update>            [time-varying distribution params]
    g(.)  =  conditional density     [e.g. GB2 log-link]

The model has three completely separated components:

    1.  Distribution  (distributions/)
        Computes logpdf, score, Fisher info, CDF, and random samples
        for y_t > 0.

    2.  Pi Dynamics   (pi_dynamics/)
        Computes eta_t → pi_t for each t.
        Pluggable via factory; the default is ARLogisticPiDynamics.

    3.  GAS Filter    (models/gas_filter.py)
        Updates the time-varying distribution parameters (phi, xi)
        using the GAS(L,L) update equation from Creal et al. (2012).

=============================================================================
PARAMETER VECTOR  (flat numpy array passed to scipy.minimize)
=============================================================================

The flat theta vector is the concatenation of:

    [  gas_theta  |  pi_theta  ]

where:
    gas_theta  :  GAS hyperparameters  (see GASFilter._ParamCodec)
    pi_theta   :  Pi dynamics parameters  (see PiDynamics.param_names)

=============================================================================
EFFECTIVE SAMPLE
=============================================================================

Both the GAS filter and the pi dynamics require a warm-up period.
The effective sample starts at  t_start = max(gas_filter.max_lag, pi_max_lag)
so that all lagged values are available.

For daily data  (lags = [1,2,3,364,365,366,367])  t_start = 367.
For monthly data (lags = [1,2,3,11,12,13])         t_start = 13.
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit   # logistic σ(x) = 1/(1+e^{-x})

from distributions.base import Distribution
from pi_dynamics.base import PiDynamics
from models.gas_filter import GASFilter
from models.lags import SEASONAL_LAGS


class ZAGASModel:
    """
    Zero-Augmented GAS model.

    Parameters
    ----------
    distribution  : Distribution  — positive-part density (e.g. GB2LogLink)
    pi_dynamics   : PiDynamics    — model for the logit probability eta_t
    seasonal      : 'daily' or 'monthly'
    scale_score   : passed through to GASFilter
    static_params : passed through to GASFilter  (default ['gamma', 'zeta'])
    """

    def __init__(
        self,
        distribution: Distribution,
        pi_dynamics: PiDynamics,
        seasonal: str = "daily",
        scale_score: bool = True,
        static_params: List[str] | None = None,
    ):
        self.dist        = distribution
        self.pi_dyn      = pi_dynamics
        self.seasonal    = seasonal
        self.lags        = SEASONAL_LAGS[seasonal]
        self.max_lag     = max(self.lags)

        # Build the GAS filter for the distribution's TV parameters
        self.gas         = GASFilter(
            distribution=distribution,
            seasonal=seasonal,
            scale_score=scale_score,
            static_params=static_params,
        )

        # Cache the pi parameter names (depends on the seasonal lag set)
        self._pi_names   = pi_dynamics.param_names(seasonal)

    # ==================================================================
    # Parameter vector helpers
    # ==================================================================

    def _split_theta(self, theta: np.ndarray):
        """
        Split the combined theta into (gas_theta, pi_theta).

        Layout:  [ gas_theta (n_gas) | pi_theta (n_pi) ]
        """
        n_gas = self.gas.codec.n_params
        return theta[:n_gas], theta[n_gas:]

    def _decode_pi(self, pi_theta: np.ndarray) -> dict:
        """Return dict mapping pi parameter names → values."""
        return {name: float(pi_theta[i])
                for i, name in enumerate(self._pi_names)}

    def parameter_names(self) -> List[str]:
        """Flat theta parameter names in optimizer order."""
        gas_names = [
            name for name, _ in sorted(
                self.gas.codec._idx.items(), key=lambda item: item[1]
            )
        ]
        return gas_names + list(self._pi_names)

    def parameter_blocks(self) -> dict[str, str]:
        """Map each flat theta parameter to its model block."""
        blocks = {}
        for name in self.parameter_names():
            if name in self._pi_names:
                blocks[name] = "pi"
            elif name in self.gas.static_names:
                blocks[name] = "positive_static"
            else:
                blocks[name] = "gas"
        return blocks

    def parameter_count_breakdown(self) -> dict:
        """Explain why the optimizer has this many parameters."""
        n_lags = len(self.lags)
        n_tv = len(self.gas.tv_names)
        per_tv = 2 + 2 * n_lags
        n_gas_dynamic = n_tv * per_tv
        n_static = len(self.gas.static_names)
        n_pi = len(self._pi_names)
        return {
            "seasonal": self.seasonal,
            "lags": list(self.lags),
            "n_lags": n_lags,
            "tv_parameters": list(self.gas.tv_names),
            "per_tv_parameter": per_tv,
            "gas_dynamic_parameters": n_gas_dynamic,
            "static_positive_parameters": n_static,
            "pi_parameters": n_pi,
            "total_parameters": n_gas_dynamic + n_static + n_pi,
            "formula": (
                "n_tv * (omega + f0 + A_lags + B_lags) + "
                "n_static_positive + n_pi"
            ),
        }

    # ==================================================================
    # Core recursion
    # ==================================================================

    def _run_filter(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        return_paths: bool = False,
    ):
        """
        Run the combined GAS + pi recursion over the series y.

        For each t in the effective sample:

          (a) GAS step:
              f_t  = current state of distribution params (phi_t, xi_t)
              s_t  = scaled score from distribution  (only when y_t > 0)
              f_{t+1} updated via GAS(L,L) eq.

          (b) Pi step:
              eta_t computed from pi_dynamics
              pi_t = sigma(eta_t)

          (c) Likelihood:
              log p(y_t | F_{t-1}) =
                  log(1 - pi_t)                          if y_t = 0
                  log(pi_t) + log g(y_t; f_t, static)   if y_t > 0

        Parameters
        ----------
        theta        : combined [gas_theta | pi_theta]
        y            : observed series (length T)
        return_paths : if True return filtered paths, else return neg-loglik

        Returns
        -------
        float (neg-loglik) when return_paths=False, or dict when True.
        """

        T = len(y)
        if T <= self.max_lag:
            return 1e12 if not return_paths else {}

        gas_theta, pi_theta = self._split_theta(theta)
        pi_params = self._decode_pi(pi_theta)

        # --- Decode GAS state arrays ---
        # We reuse GASFilter._run_filter to advance the state.
        # On return_paths=True it gives f_arr and s_arr indexed from max_lag.
        # Here we need a slightly different approach: we need the GAS states
        # at each t in [max_lag, T-1] and the ability to update them, while
        # simultaneously running pi dynamics.
        #
        # Strategy: run the GAS recursion manually (mirrors gas_filter._run_filter)
        # so we can interleave the pi update and the joint likelihood.

        gp     = self.gas.codec.decode(gas_theta)   # dict of GAS params
        L      = self.lags
        n_tv   = len(self.gas.tv_names)

        # Validity guard for static distribution parameters
        gamma_v = gp.get("gamma", 1.0)
        zeta_v  = gp.get("zeta",  3.0)
        if np.exp(gamma_v) <= 0 or np.exp(zeta_v) <= np.exp(gamma_v):
            return 1e12 if not return_paths else {}

        # Validity guard for pi AR coefficient
        rho = pi_params.get("rho", 0.0)
        if abs(rho) >= 1.0:
            return 1e12 if not return_paths else {}

        # -------------------------------------------------------------------
        # Allocate state and score arrays (full length for easy lag indexing)
        # -------------------------------------------------------------------
        f_arr  = np.zeros((T + 1, n_tv))   # f_arr[t, j] = f^j_t
        s_arr  = np.zeros((T,     n_tv))   # s_arr[t, j] = scaled score at t
        eta_arr = np.zeros(T)              # logit of pi
        pi_arr  = np.zeros(T)             # probability of non-zero

        # Initialise burn-in with f0 for each TV parameter
        for j, name in enumerate(self.gas.tv_names):
            f_arr[:self.max_lag + 1, j] = gp[f"f0_{name}"]

        loglik = 0.0

        # -------------------------------------------------------------------
        # Main recursion
        # -------------------------------------------------------------------
        for t in range(self.max_lag, T):

            # ---- (a) Assemble current parameter dict ----
            call_params = {
                name: f_arr[t, j]
                for j, name in enumerate(self.gas.tv_names)
            }
            call_params.update({s: gp[s] for s in self.gas.static_names})

            # ---- (b) Pi step ----
            # eta_t  =  pi_dynamics.compute_eta(...)
            # pi_t   =  sigma(eta_t)
            eta_arr[t] = self.pi_dyn.compute_eta(
                i        = t - self.max_lag,   # effective index
                eta_hist = eta_arr,             # full array; valid up to t-1
                y_full   = y,
                t        = t,
                params   = pi_params,
                seasonal = self.seasonal,
            )
            pi_arr[t] = float(expit(eta_arr[t]))

            # Guard against degenerate probabilities
            pi_t   = np.clip(pi_arr[t], 1e-10, 1.0 - 1e-10)

            # ---- (c) Likelihood contribution ----
            if y[t] == 0:
                # Zero observation: only pi contributes
                ll_t = np.log(1.0 - pi_t)
            else:
                # Positive observation: pi * g(y_t)
                ll_dist = self.dist.logpdf(y[t], **call_params)
                if not np.isfinite(ll_dist):
                    if not return_paths:
                        return 1e12
                    ll_dist = 0.0
                ll_t = np.log(pi_t) + ll_dist

            if not np.isfinite(ll_t):
                if not return_paths:
                    return 1e12
                ll_t = 0.0
            loglik += ll_t

            # ---- (d) Score for distribution parameters ----
            # Score is zero for y_t = 0 (no info about positive-part params).
            # See: score contribution to GAS update only from positive obs.
            if y[t] > 0:
                raw_score = self.dist.score(y[t], **call_params)
                if self.gas.scale_score:
                    fi = self.dist.fisher_info_diag(**call_params)
                    for j, name in enumerate(self.gas.tv_names):
                        s_arr[t, j] = raw_score[name] / max(fi[name], 1e-8)
                else:
                    for j, name in enumerate(self.gas.tv_names):
                        s_arr[t, j] = raw_score[name]
            # else: s_arr[t, :] stays 0

            # ---- (e) GAS update: f_{t+1}  =  omega  +  A(L) s_t  +  B(L) f_t ----
            for j, name in enumerate(self.gas.tv_names):
                omega_j    = gp[f"omega_{name}"]
                ar_part    = 0.0
                score_part = 0.0

                for l in L:
                    # l=1 → current (index t); l=2 → one step back (t-1)
                    idx    = t - l + 1
                    s_past = s_arr[idx, j] if idx >= 0 else 0.0
                    f_past = f_arr[idx, j] if idx >= 0 else gp[f"f0_{name}"]

                    score_part += gp[f"A_{name}_{l}"] * s_past
                    ar_part    += gp[f"B_{name}_{l}"] * f_past

                f_arr[t + 1, j] = np.clip(
                    omega_j + score_part + ar_part, -15.0, 15.0
                )

        # -------------------------------------------------------------------
        # Return
        # -------------------------------------------------------------------
        if not return_paths:
            return -loglik

        eff = self.max_lag
        return {
            # Sliced to effective sample  t = max_lag, ..., T-1
            "phi":        f_arr[eff:T, 0],              # TV param 1 (phi)
            "xi":         f_arr[eff:T, 1] if n_tv > 1 else None,  # TV param 2 (xi)
            "f_arr":      f_arr[eff:T, :],              # all TV params
            "s_arr":      s_arr[eff:T, :],              # all scaled scores
            "eta":        eta_arr[eff:T],               # logit pi
            "pi":         pi_arr[eff:T],                # prob of non-zero
            "tv_names":   self.gas.tv_names,
            "static":     {s: gp[s] for s in self.gas.static_names},
            "eff_start":  eff,
            "loglik":     loglik,
            "y_eff":      y[eff:T],                     # observed y in effective window
        }

    # ==================================================================
    # Public interface
    # ==================================================================

    def loglik(self, theta: np.ndarray, y: np.ndarray) -> float:
        """Log-likelihood (positive = better)."""
        return -self._run_filter(theta, y, return_paths=False)

    def filter(self, theta: np.ndarray, y: np.ndarray) -> dict:
        """Return the full set of filtered paths."""
        return self._run_filter(theta, y, return_paths=True)

    # ------------------------------------------------------------------
    def cdf_series(self, theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the predictive CDF  F_t(y_t)  at each t in the effective sample.

        For the ZA model:
            F_t(y_t)  =  (1 - pi_t)  +  pi_t * G_t(y_t)    if y_t >= 0
        where  G_t  is the GB2 CDF.
        Returns an array of length (T - max_lag).
        """
        paths = self.filter(theta, y)
        if not paths:
            return np.array([])

        T    = len(y)
        eff  = paths["eff_start"]
        n    = T - eff
        cdfs = np.zeros(n)

        static = paths["static"]

        for i in range(n):
            t = eff + i
            pi_t = paths["pi"][i]
            call = {
                name: paths["f_arr"][i, j]
                for j, name in enumerate(paths["tv_names"])
            }
            call.update(static)

            if y[t] <= 0:
                # P(Y <= 0) = P(Y = 0) = 1 - pi_t
                cdfs[i] = 1.0 - pi_t
            else:
                # P(Y <= y_t) = (1 - pi_t) + pi_t * G_t(y_t)
                G_t = self.dist.cdf(y[t], **call)
                cdfs[i] = (1.0 - pi_t) + pi_t * G_t

        return cdfs

    # ------------------------------------------------------------------
    def initial_theta(self, y: np.ndarray) -> np.ndarray:
        """Combined starting vector [gas_theta | pi_theta]."""
        gas_theta = self.gas.initial_theta(y)
        pi_theta  = self.pi_dyn.initial_params(y, self.seasonal)
        return np.concatenate([gas_theta, pi_theta])

    def default_bounds(self) -> List[Tuple[float, float]]:
        """Combined bounds [gas_bounds | pi_bounds]."""
        return (
            self.gas.default_bounds()
            + self.pi_dyn.default_bounds(self.seasonal)
        )

    def fit(
        self,
        y: np.ndarray,
        verbose: bool = False,
        method: str = "L-BFGS-B",
        use_bounds: bool = True,
        theta0: np.ndarray | None = None,
        options: dict | None = None,
    ) -> dict:
        """
        Estimate all parameters by maximum likelihood.

        Returns
        -------
        dict with keys:
            'theta'    : optimal flat parameter vector
            'loglik'   : log-likelihood at optimum
            'success'  : bool
            'result'   : scipy OptimizeResult
            'gas_theta': slice for GAS hyperparameters
            'pi_theta' : slice for pi dynamics parameters
        """
        theta0 = self.initial_theta(y) if theta0 is None else np.asarray(theta0, dtype=float)
        bounds = self.default_bounds() if use_bounds else None
        opt_options = {"maxiter": 5000, "disp": verbose}
        if method.upper() == "L-BFGS-B":
            opt_options.update({"ftol": 1e-10, "gtol": 1e-6})
        elif method.upper() == "BFGS":
            opt_options.update({"gtol": 1e-5})
        if options:
            opt_options.update(options)

        result = minimize(
            fun=self._run_filter,
            x0=theta0,
            args=(y, False),
            method=method,
            bounds=bounds,
            options=opt_options,
        )

        n_gas = self.gas.codec.n_params
        return {
            "theta":     result.x,
            "loglik":   -result.fun,
            "success":   result.success,
            "result":    result,
            "gas_theta": result.x[:n_gas],
            "pi_theta":  result.x[n_gas:],
        }
