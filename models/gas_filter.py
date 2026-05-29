"""
GAS(p,q) filter — Creal, Koopman & Lucas (2012, J. Appl. Econometrics).

=============================================================================
THEORY  (equations reference the JAE paper)
=============================================================================

Observation equation  [eq. 1]:
    y_t | F_{t-1}  ~  p(y_t ; f_t, theta_static)

Score and scaled score  [eqs. 2-4]:
    nabla_t  =  d/df_t  log p(y_t ; f_t, theta_static)
    S_t      =  I(f_t)^{-1}           (Scaling type 1: inverse Fisher info)
    s_t      =  S_t * nabla_t

Update equation  [eq. 5]:
    f_{t+1}  =  omega  +  A(L) s_t  +  B(L) f_t

For the standard GAS(1,1) the lag polynomial is simply:
    f_{t+1}  =  omega  +  A * s_t  +  B * f_t

=============================================================================
SEASONAL GENERALIZATION  (non-consecutive lags)
=============================================================================

Instead of the consecutive lags 1, 2, ..., p  we allow any ordered set L.
This captures both short-memory dependence (l = 1, 2, 3) and the annual cycle
(l = 364, 365, 366, 367 for daily; l = 11, 12, 13 for monthly) with a single,
parsimonious GAS update:

    f_{t+1}  =  omega  +  sum_{l in L}  A_l * s_{t-l+1}
                       +  sum_{l in L}  B_l * f_{t-l+1}

Here  l = 1  refers to the CURRENT time step (i.e. A_1 * s_t + B_1 * f_t),
      l = 2  refers to one step back, and so on.

=============================================================================
IMPLEMENTATION NOTES
=============================================================================

* Each time-varying distribution parameter (e.g. phi, xi for GB2) gets its
  own independent set of  (omega_j, f0_j, {A_j_l, B_j_l for l in L}).
  Updates are decoupled because S_t is taken to be diagonal.

* Static distribution parameters (gamma, zeta for GB2) are estimated
  jointly with the GAS hyperparameters via maximum likelihood.

* Score contributions are set to zero when y_t = 0 (no information about
  the positive-part distribution from a zero observation).  This is by
  design: the ZA-GAS model keeps the pi dynamics separate.

* Pre-effective-start history (t < eff_start) is set to the initial state
  f0_j for states and 0 for scores.  This is the standard burn-in approach.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

from distributions.base import Distribution
from models.lags import SEASONAL_LAGS


# ===========================================================================
# Parameter codec: maps a flat numpy vector <-> named parameter dict
# ===========================================================================

class _ParamCodec:
    """
    Builds a two-way mapping between a flat 1-D theta array and named
    parameters.

    Parameter ordering in theta:
        For each time-varying parameter j  (in distribution.tv_param_names):
            omega_j          -- GAS intercept  (eq. 5: omega)
            f0_j             -- initial state (burn-in value for t < eff_start)
            A_j_{l}          -- score coefficient for lag l, for l in L
            B_j_{l}          -- AR   coefficient for lag l, for l in L
        Static parameters (one per entry in static_param_names):
            gamma, zeta, ...
    """

    def __init__(
        self,
        tv_param_names: List[str],      # e.g. ['phi', 'xi']
        static_param_names: List[str],  # e.g. ['gamma', 'zeta']
        lags: List[int],                # seasonal lag set L
    ):
        self.tv_names     = tv_param_names
        self.static_names = static_param_names
        self.lags         = lags

        # Build index: parameter name -> position in flat vector
        self._idx: Dict[str, int] = {}
        pos = 0

        for name in tv_param_names:
            self._idx[f"omega_{name}"] = pos; pos += 1   # intercept
            self._idx[f"f0_{name}"]    = pos; pos += 1   # initial state
            for l in lags:
                self._idx[f"A_{name}_{l}"] = pos; pos += 1  # score coefficients
            for l in lags:
                self._idx[f"B_{name}_{l}"] = pos; pos += 1  # AR coefficients

        for name in static_param_names:
            self._idx[name] = pos; pos += 1

        self.n_params = pos   # total length of theta

    # ------------------------------------------------------------------
    def decode(self, theta: np.ndarray) -> Dict[str, float]:
        """Return a plain dict {param_name: value}."""
        return {k: float(theta[v]) for k, v in self._idx.items()}

    def get(self, theta: np.ndarray, key: str) -> float:
        return float(theta[self._idx[key]])


# ===========================================================================
# GASFilter
# ===========================================================================

class GASFilter:
    """
    GAS(L,L) filter with a seasonal (non-consecutive) lag set.

    Parameters
    ----------
    distribution   : any Distribution subclass  (defines logpdf, score,
                     fisher_info_diag, tv_param_names)
    seasonal       : 'daily' or 'monthly' — determines the lag set L
    scale_score    : if True, multiply nabla_t by diag(I(f_t))^{-1}  before
                     the GAS update (Scaling type 1, recommended by Creal et al.)
    static_params  : names of static distribution parameters to estimate.
                     Defaults to ['gamma', 'zeta'] for the GB2 model.
    """

    # Default static parameters for the GB2LogLink distribution.
    # Override via the constructor if using a different distribution.
    DEFAULT_STATIC = ["gamma", "zeta"]

    def __init__(
        self,
        distribution: Distribution,
        seasonal: str = "daily",
        scale_score: bool = True,
        static_params: List[str] | None = None,
    ):
        if seasonal not in SEASONAL_LAGS:
            raise ValueError(
                f"seasonal must be one of {list(SEASONAL_LAGS)}, got '{seasonal}'"
            )

        self.dist          = distribution
        self.seasonal      = seasonal
        self.lags          = SEASONAL_LAGS[seasonal]          # e.g. [1,2,3,364,365,366,367]
        self.max_lag       = max(self.lags)                   # burn-in length
        self.scale_score   = scale_score
        self.tv_names      = distribution.tv_param_names      # e.g. ['phi', 'xi']
        self.static_names  = static_params or self.DEFAULT_STATIC

        # Build the flat-vector codec
        self.codec = _ParamCodec(self.tv_names, self.static_names, self.lags)

    # ==================================================================
    # Core GAS recursion
    # ==================================================================

    def _run_filter(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        return_paths: bool = False,
    ):
        """
        Execute the GAS recursion.

        Parameters
        ----------
        theta        : flat parameter vector (see _ParamCodec for layout)
        y            : observed time series (length T)
        return_paths : if True, return filtered state paths; else return
                       only the total log-likelihood.

        Returns
        -------
        float (neg-loglik) when return_paths=False, or dict when True.
        """

        T        = len(y)
        n_tv     = len(self.tv_names)
        L        = self.lags
        max_lag  = self.max_lag

        # --- Minimum data length check ---
        # We need at least max_lag observations for the burn-in plus one
        # effective observation.
        if T <= max_lag:
            return 1e12 if not return_paths else {}

        # --- Decode theta into named values ---
        p = self.codec.decode(theta)

        # --- Basic validity guards for static params ---
        gamma_v = p.get("gamma", 1.0)   # GB2 shape (log scale)
        zeta_v  = p.get("zeta",  3.0)   # GB2 shape (log scale)
        if np.exp(gamma_v) <= 0 or np.exp(zeta_v) <= np.exp(gamma_v):
            # GB2 requires b > a*p and both positive; return large penalty
            return 1e12 if not return_paths else {}

        # -------------------------------------------------------------------
        # Allocate full-length state and score arrays.
        #
        # f_arr[t, j]  =  f^j_t  :  state of parameter j going INTO time t
        # s_arr[t, j]  =  s^j_t  :  scaled score at time t (computed AFTER y_t)
        #
        # Pre-effective-start (t < max_lag):
        #   f_arr[t, j]  =  f0_j   (initial state, estimated parameter)
        #   s_arr[t, j]  =  0      (no score before effective window)
        # -------------------------------------------------------------------
        f_arr = np.zeros((T + 1, n_tv))  # +1 so f_arr[T] can be computed
        s_arr = np.zeros((T,     n_tv))

        for j, name in enumerate(self.tv_names):
            f0_j = p[f"f0_{name}"]
            # Fill burn-in period with the initial state value
            f_arr[:max_lag + 1, j] = f0_j   # includes index max_lag itself

        loglik = 0.0  # accumulated log-likelihood over effective sample

        # -------------------------------------------------------------------
        # Main GAS recursion  (effective sample: t = max_lag, ..., T-1)
        # -------------------------------------------------------------------
        for t in range(max_lag, T):

            # --- 1.  Assemble current call_params dict for distribution ---
            call_params = {
                name: f_arr[t, j]
                for j, name in enumerate(self.tv_names)
            }
            call_params.update({s: p[s] for s in self.static_names})

            # --- 2.  Evaluate log-likelihood contribution ---
            # For the ZA model, the distribution logpdf is only evaluated on
            # positive observations; zeros contribute through the pi term
            # (handled by ZAGASModel).  Here we evaluate unconditionally
            # so this filter can also be used standalone.
            if y[t] > 0:
                ll_t = self.dist.logpdf(y[t], **call_params)
                if not np.isfinite(ll_t):
                    if not return_paths:
                        return 1e12
                    ll_t = 0.0
                loglik += ll_t

            # --- 3.  Compute the (optionally scaled) score  s_t ---
            #
            # From Creal et al. eq. 4:
            #   nabla_t = d/df_t  log p(y_t; f_t, theta)
            #   s_t     = S_t * nabla_t
            #
            # We set s_t = 0 when y_t = 0 because zeros carry no information
            # about the positive-part distribution parameters.
            if y[t] > 0:
                raw_score = self.dist.score(y[t], **call_params)  # dict: name -> float

                if self.scale_score:
                    # Scaling type 1: S_t = I(f_t)^{-1}  (diagonal approximation)
                    fi = self.dist.fisher_info_diag(**call_params)  # dict: name -> float
                    for j, name in enumerate(self.tv_names):
                        s_arr[t, j] = raw_score[name] / max(fi[name], 1e-8)
                else:
                    # No scaling (unit matrix S_t = I)
                    for j, name in enumerate(self.tv_names):
                        s_arr[t, j] = raw_score[name]
            # else: s_arr[t, j] remains 0 (already initialized)

            # --- 4.  Update  f_{t+1}  via the GAS(L,L) equation  [eq. 5] ---
            #
            # f_{t+1}  =  omega  +  sum_{l in L} A_l * s_{t-l+1}
            #                    +  sum_{l in L} B_l * f_{t-l+1}
            #
            # Note: l = 1 accesses s[t] and f[t] (current step),
            #       l = 2 accesses s[t-1] and f[t-1] (one step back), etc.
            for j, name in enumerate(self.tv_names):
                omega_j = p[f"omega_{name}"]
                ar_part    = 0.0
                score_part = 0.0

                for l in L:
                    idx = t - l + 1   # index into full arrays at lag l
                    # idx = t   when l = 1  (current)
                    # idx = t-1 when l = 2  (one step back)
                    # idx = t-363 when l = 364
                    s_past = s_arr[idx, j] if idx >= 0 else 0.0
                    f_past = f_arr[idx, j] if idx >= 0 else p[f"f0_{name}"]

                    score_part += p[f"A_{name}_{l}"] * s_past
                    ar_part    += p[f"B_{name}_{l}"] * f_past

                f_arr[t + 1, j] = omega_j + score_part + ar_part

                # Soft clip to prevent runaway states in early optimisation
                f_arr[t + 1, j] = np.clip(f_arr[t + 1, j], -15.0, 15.0)

        # -------------------------------------------------------------------
        # Return
        # -------------------------------------------------------------------
        if not return_paths:
            return -loglik   # negate because scipy.minimize minimises

        return {
            # f_arr[t] is the state ENTERING time t; eff_start = max_lag
            "f_arr":      f_arr[max_lag:T, :],   # shape (T - max_lag, n_tv)
            "s_arr":      s_arr[max_lag:T, :],   # shape (T - max_lag, n_tv)
            "tv_names":   self.tv_names,
            "static":     {s: p[s] for s in self.static_names},
            "eff_start":  max_lag,
            "loglik":     loglik,
        }

    # ==================================================================
    # Public interface
    # ==================================================================

    def loglik(self, theta: np.ndarray, y: np.ndarray) -> float:
        """Return the log-likelihood (positive = better)."""
        return -self._run_filter(theta, y, return_paths=False)

    def filter(self, theta: np.ndarray, y: np.ndarray) -> dict:
        """Return filtered state paths and scores."""
        return self._run_filter(theta, y, return_paths=True)

    # ------------------------------------------------------------------
    def initial_theta(self, y: np.ndarray) -> np.ndarray:
        """
        Data-driven starting point for the optimiser.

        Heuristics:
          - omega  : small fraction of the unconditional mean state
          - f0     : unconditional log-mean of positive observations (for phi)
                     or log(1.2) for shape parameters (for xi)
          - A_l    : small positive values, divided equally across lags
          - B_l    : small positive values summing to ~0.9 (near unit-root)
          - static : gamma=1.0, zeta=3.0  (moderate GB2 tail)
        """
        y_pos = y[y > 0]
        if len(y_pos) == 0:
            y_pos = np.array([1.0])

        theta0 = np.zeros(self.codec.n_params)
        idx    = self.codec._idx
        n_l    = len(self.lags)

        for j, name in enumerate(self.tv_names):
            if name == "phi":
                f0_guess = float(np.log(np.mean(y_pos) + 1e-3))
            else:
                # xi = log(a); a ~ 1.2 is a reasonable shape starting point
                f0_guess = float(np.log(1.2))

            theta0[idx[f"omega_{name}"]] = f0_guess * 0.05   # small intercept
            theta0[idx[f"f0_{name}"]]    = f0_guess

            # Score coefficients A_l: small uniform positive values
            for l in self.lags:
                theta0[idx[f"A_{name}_{l}"]] = 0.01 / n_l

            # AR coefficients B_l: sum to ~0.9 so the process is persistent
            for l in self.lags:
                theta0[idx[f"B_{name}_{l}"]] = 0.9 / n_l

        # Static GB2 parameters (on log scale for gamma / natural scale for zeta)
        if "gamma" in idx:
            theta0[idx["gamma"]] = 1.0   # p = exp(-gamma) ~ 0.37
        if "zeta" in idx:
            theta0[idx["zeta"]]  = 3.0   # b = exp(zeta) ~ 20 > a

        return theta0

    def default_bounds(self) -> List[Tuple[float, float]]:
        """
        L-BFGS-B box constraints.

        omega  : unconstrained in a generous range
        f0     : generous range (log scale for phi and xi)
        A_l    : small-to-moderate; score coefficients should stay bounded
        B_l    : each in (-1, 1); sum < 1 enforced softly via initialisation
        gamma  : (0.1, 5)  -- log scale, so exp(gamma) in (1.1, 148)
        zeta   : (0.2, 10) -- log scale, must exceed gamma for finite variance
        """
        bounds = []
        for name in self.tv_names:
            bounds.append((-5.0, 8.0))              # omega
            bounds.append((-8.0, 8.0))              # f0
            for _ in self.lags:
                bounds.append((-0.5, 1.0))          # A_l  (score)
            for _ in self.lags:
                bounds.append((-0.99, 0.99))        # B_l  (AR)
        bounds.append((0.1, 5.0))                   # gamma (static)
        bounds.append((0.2, 10.0))                  # zeta  (static)
        return bounds

    def fit(self, y: np.ndarray, verbose: bool = False) -> dict:
        """
        Estimate GAS hyperparameters by maximum likelihood.

        Uses L-BFGS-B with the data-driven starting point from
        `initial_theta`.  Returns a dict with keys:
            'theta'   : optimal parameter vector
            'loglik'  : log-likelihood at optimum
            'success' : bool
            'result'  : raw scipy OptimizeResult
        """
        theta0 = self.initial_theta(y)
        bounds = self.default_bounds()

        result = minimize(
            fun=self._run_filter,
            x0=theta0,
            args=(y, False),          # return_paths=False → negate loglik
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 5000,
                "ftol":    1e-10,
                "gtol":    1e-6,
                "disp":    verbose,
            },
        )

        return {
            "theta":   result.x,
            "loglik":  -result.fun,
            "success": result.success,
            "result":  result,
        }
