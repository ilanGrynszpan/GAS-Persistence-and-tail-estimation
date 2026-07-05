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
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit  # logistic σ(x) = 1/(1+e^{-x})

from distributions.base import Distribution
from pi_dynamics.base import PiDynamics
from models.gas_filter import GASFilter
from models.lags import SEASONAL_LAGS

try:
    from models._nb_gb2_filter import nb_filter_phi_only, nb_filter_phi_xi
    _NB_AVAIL = True
except Exception:
    _NB_AVAIL = False

_NB_DIST_NAMES = {"GB2LogLink", "GB2LogLinkPhiOnly"}
_NB_PI_NAMES   = {"ARLogisticPiDynamics"}


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
        gas_lags: List[int] | None = None,
        scaling: str = "diagonal_inverse_fisher",
        # Legacy alias kept for backward compat
        scale_score: bool | None = None,
        static_params: List[str] | None = None,
    ):
        self.dist = distribution
        self.pi_dyn = pi_dynamics
        self.seasonal = seasonal

        # Build the GAS filter — gas_lags and scaling are explicit per MODELS.md §10
        self.gas = GASFilter(
            distribution=distribution,
            seasonal=seasonal,
            gas_lags=gas_lags,
            scaling=scaling,
            scale_score=scale_score,
            static_params=static_params,
        )

        self.lags    = self.gas.lags
        self.max_lag = self.gas.max_lag

        # Cache the pi parameter names (depends on seasonal lag set, not gas_lags)
        self._pi_names = pi_dynamics.param_names(seasonal)

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
        return {name: float(pi_theta[i]) for i, name in enumerate(self._pi_names)}

    def parameter_names(self) -> List[str]:
        """Flat theta parameter names in optimizer order."""
        gas_names = [
            name
            for name, _ in sorted(self.gas.codec._idx.items(), key=lambda item: item[1])
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

    @property
    def n_params(self) -> int:
        """Total number of estimated parameters (GAS + pi)."""
        return self.gas.codec.n_params + len(self._pi_names)

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
                "n_tv * (omega + f0 + A_lags + B_lags) + " "n_static_positive + n_pi"
            ),
        }

    # ==================================================================
    # Core recursion
    # ==================================================================

    def _use_nb_filter(self) -> bool:
        """Return True when the compiled numba filter covers this model config."""
        return (
            _NB_AVAIL
            and type(self.dist).__name__ in _NB_DIST_NAMES
            and type(self.pi_dyn).__name__ in _NB_PI_NAMES
        )

    # Scaling name → int code expected by the numba functions
    _SCALING_INT = {"unit": 0, "diagonal_inverse_fisher": 1, "inverse_fisher": 2}

    def _nb_dispatch(
        self,
        y:           np.ndarray,
        gp:          dict,
        pi_params:   dict,
        pre_penalty: float,
        T:           int,
        return_paths: bool,
    ):
        """
        Call the appropriate numba-compiled filter and convert its output to
        the same format as the Python loop path.

        pre_penalty  — parameter-level penalties already computed in _run_filter
                       (persistence, score bounds, static bounds, moment cond.)
        """
        L          = self.lags
        eff        = self.max_lag
        n_tv       = len(self.gas.tv_names)
        scaling_i  = self._SCALING_INT.get(self.gas.scaling, 0)

        lags_gas   = np.array(list(L), dtype=np.int64)
        pi_lags_py = SEASONAL_LAGS[self.seasonal]
        lags_pi    = np.array(pi_lags_py, dtype=np.int64)
        pi_wy      = np.array([pi_params[f"omega_y_{l}"] for l in pi_lags_py])
        pi_omega0  = float(pi_params["omega0"])
        pi_rho     = float(pi_params["rho"])

        LAM_STATE   = 1e3
        LAM_ETA     = 1e2
        STATE_LIMIT = 15.0
        ETA_LIMIT   = 30.0

        y_f64 = np.asarray(y, dtype=np.float64)

        if n_tv == 1:
            # ── phi-only (GB2LogLinkPhiOnly) ──────────────────────────────────
            xi_s    = float(gp["xi"])
            gamma_s = float(gp["gamma"])
            zeta_s  = float(gp["zeta"])

            out = nb_filter_phi_only(
                y_f64,
                f0=float(gp["f0_phi"]),
                omega=float(gp["omega_phi"]),
                A=np.array([gp[f"A_phi_{l}"] for l in L]),
                B=np.array([gp[f"B_phi_{l}"] for l in L]),
                lags_gas=lags_gas,
                xi_s=xi_s, gamma_s=gamma_s, zeta_s=zeta_s,
                pi_omega0=pi_omega0, pi_rho=pi_rho,
                pi_wy=pi_wy, lags_pi=lags_pi,
                max_lag=eff, scaling=scaling_i,
                lam_state=LAM_STATE, state_lim=STATE_LIMIT,
                lam_eta=LAM_ETA,    eta_lim=ETA_LIMIT,
            )
            loglik_nb, loop_pen, f1d, s1d, eta1d, pi1d = out

            if loglik_nb == -1e12:
                return (1e12 + pre_penalty) if not return_paths else {}

            total_pen = pre_penalty + loop_pen
            obj       = -loglik_nb + total_pen

            if not return_paths:
                return obj

            n_eff = T - eff
            return {
                "phi":       f1d[eff:T],
                "xi":        None,
                "f_arr":     f1d[eff:T].reshape(n_eff, 1),
                "s_arr":     s1d[eff:T].reshape(n_eff, 1),
                "eta":       eta1d[:n_eff],
                "pi":        pi1d[:n_eff],
                "tv_names":  self.gas.tv_names,
                "static":    {s: gp[s] for s in self.gas.static_names},
                "eff_start": eff,
                "loglik":    float(loglik_nb),
                "penalty":   float(total_pen),
                "objective": float(obj),
                "y_eff":     y[eff:T],
            }

        else:
            # ── phi + xi (GB2LogLink) ─────────────────────────────────────────
            gamma_s = float(gp["gamma"])
            zeta_s  = float(gp["zeta"])

            out = nb_filter_phi_xi(
                y_f64,
                f0_phi=float(gp["f0_phi"]),
                f0_xi=float(gp["f0_xi"]),
                omega_phi=float(gp["omega_phi"]),
                omega_xi=float(gp["omega_xi"]),
                A_phi=np.array([gp[f"A_phi_{l}"] for l in L]),
                A_xi=np.array([gp[ f"A_xi_{l}"]  for l in L]),
                B_phi=np.array([gp[f"B_phi_{l}"] for l in L]),
                B_xi=np.array([gp[ f"B_xi_{l}"]  for l in L]),
                lags_gas=lags_gas,
                gamma_s=gamma_s, zeta_s=zeta_s,
                pi_omega0=pi_omega0, pi_rho=pi_rho,
                pi_wy=pi_wy, lags_pi=lags_pi,
                max_lag=eff, scaling=scaling_i,
                lam_state=LAM_STATE, state_lim=STATE_LIMIT,
                lam_eta=LAM_ETA,    eta_lim=ETA_LIMIT,
            )
            loglik_nb, loop_pen, f_phi, f_xi, s_phi, s_xi, eta1d, pi1d = out

            if loglik_nb == -1e12:
                return (1e12 + pre_penalty) if not return_paths else {}

            total_pen = pre_penalty + loop_pen
            obj       = -loglik_nb + total_pen

            if not return_paths:
                return obj

            n_eff = T - eff
            f_arr_2d = np.column_stack([f_phi[eff:T], f_xi[eff:T]])
            s_arr_2d = np.column_stack([s_phi[eff:T], s_xi[eff:T]])
            return {
                "phi":       f_phi[eff:T],
                "xi":        f_xi[eff:T],
                "f_arr":     f_arr_2d,
                "s_arr":     s_arr_2d,
                "eta":       eta1d[:n_eff],
                "pi":        pi1d[:n_eff],
                "tv_names":  self.gas.tv_names,
                "static":    {s: gp[s] for s in self.gas.static_names},
                "eff_start": eff,
                "loglik":    float(loglik_nb),
                "penalty":   float(total_pen),
                "objective": float(obj),
                "y_eff":     y[eff:T],
            }

    def _soft_stationarity_penalty(self, gp: dict, pi_params: dict) -> float:
        penalty = 0.0
        lam = 1e5

        # Penalize GAS AR persistence: sum |B_l| < 0.98 for each TV state
        for name in self.gas.tv_names:
            b_vals = np.array([gp[f"B_{name}_{l}"] for l in self.lags])
            b_sum = np.sum(np.abs(b_vals))

            excess = max(0.0, b_sum - 0.98)
            penalty += lam * excess**2

        # Penalize very large score response, not stationarity, just stability
        for name in self.gas.tv_names:
            a_vals = np.array([gp[f"A_{name}_{l}"] for l in self.lags])
            a_sum = np.sum(np.abs(a_vals))

            excess = max(0.0, a_sum - 2.0)
            penalty += 1e3 * excess**2

        # Penalize pi AR coefficient
        rho = pi_params.get("rho", 0.0)
        excess = max(0.0, abs(rho) - 0.98)
        penalty += lam * excess**2

        return penalty

    def _run_filter(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        return_paths: bool = False,
    ):
        """
        Run the combined GAS + pi recursion over the series y.

        This version avoids clipping the filtered GAS states. Instead, it adds
        smooth penalties for unstable persistence, excessive state magnitudes,
        and extreme pi logits. This is better suited to unconstrained BFGS.
        """

        T = len(y)
        if T <= self.max_lag:
            return 1e12 if not return_paths else {}

        gas_theta, pi_theta = self._split_theta(theta)
        pi_params = self._decode_pi(pi_theta)
        gp = self.gas.codec.decode(gas_theta)

        L = self.lags
        n_tv = len(self.gas.tv_names)

        # -----------------------------
        # Penalty hyperparameters
        # -----------------------------
        penalty = 0.0

        LAM_PERSIST = 1e5
        LAM_SCORE = 1e3
        LAM_STATE = 1e3
        LAM_ETA = 1e2
        LAM_STATIC = 1e5

        B_LIMIT = 0.98
        RHO_LIMIT = 0.98
        A_LIMIT = 2.0
        STATE_LIMIT = 15.0
        ETA_LIMIT = 30.0

        # -----------------------------
        # Static distribution checks
        # -----------------------------
        gamma_v = gp.get("gamma")
        zeta_v  = gp.get("zeta")

        if gamma_v is not None and zeta_v is not None:
            # Moment condition: E[Y] < ∞ requires b·p > 1, i.e. exp(zeta-gamma) > 1,
            # i.e. zeta > gamma.  When violated the GB2 has an infinite mean and
            # the log-likelihood is undefined.  Return a hard barrier.
            if gamma_v >= zeta_v:
                return (1e12 + penalty) if not return_paths else {}

            # Soft push away from the boundary: penalize log(b·p) < 0.5
            bp_log = zeta_v - gamma_v
            if bp_log < 0.5:
                penalty += LAM_STATIC * (0.5 - bp_log) ** 2

        # Prevent absurd static values under unconstrained BFGS
        for s in self.gas.static_names:
            val = gp[s]
            excess = max(0.0, abs(val) - 20.0)
            penalty += LAM_STATIC * excess**2

        # -----------------------------
        # GAS persistence penalties
        # -----------------------------
        for name in self.gas.tv_names:
            b_vals = np.array([gp[f"B_{name}_{l}"] for l in L])
            b_sum = np.sum(np.abs(b_vals))
            excess_b = max(0.0, b_sum - B_LIMIT)
            penalty += LAM_PERSIST * excess_b**2

            a_vals = np.array([gp[f"A_{name}_{l}"] for l in L])
            a_sum = np.sum(np.abs(a_vals))
            excess_a = max(0.0, a_sum - A_LIMIT)
            penalty += LAM_SCORE * excess_a**2

        # -----------------------------
        # Pi persistence penalty
        # -----------------------------
        rho = pi_params.get("rho", 0.0)
        excess_rho = max(0.0, abs(rho) - RHO_LIMIT)
        penalty += LAM_PERSIST * excess_rho**2

        # ---- Compiled fast path ------------------------------------------------
        # For GB2 + ARLogistic pi dynamics the numba filter handles the loop
        # ~50-100x faster than the equivalent Python.  Penalties computed above
        # (persistence / score / static / moment) are passed as pre_penalty;
        # the numba function adds its own per-step state/eta penalties.
        if self._use_nb_filter():
            return self._nb_dispatch(y, gp, pi_params, penalty, T, return_paths)

        # -----------------------------
        # Allocate arrays
        # -----------------------------
        f_arr = np.zeros((T + 1, n_tv))
        s_arr = np.zeros((T, n_tv))
        eta_arr  = np.zeros(T)   # t-indexed: eta_arr[t] for output / CDF use
        eta_eff  = np.zeros(T)   # i-indexed: eta_eff[i] for compute_eta AR history
        pi_arr   = np.zeros(T)

        for j, name in enumerate(self.gas.tv_names):
            f_arr[: self.max_lag + 1, j] = gp[f"f0_{name}"]

        loglik = 0.0

        # -----------------------------
        # Main recursion
        # -----------------------------
        for t in range(self.max_lag, T):

            call_params = {
                name: f_arr[t, j] for j, name in enumerate(self.gas.tv_names)
            }
            call_params.update({s: gp[s] for s in self.gas.static_names})

            # Penalize extreme states instead of clipping them
            for j in range(n_tv):
                state_excess = max(0.0, abs(f_arr[t, j]) - STATE_LIMIT)
                penalty += LAM_STATE * state_excess**2

            # Pi recursion
            # eta_eff is i-indexed so compute_eta's AR access eta_eff[i-1]
            # gives the immediately-preceding step (not max_lag steps back).
            i = t - self.max_lag
            eta_t = self.pi_dyn.compute_eta(
                i=i,
                eta_hist=eta_eff,
                y_full=y,
                t=t,
                params=pi_params,
                seasonal=self.seasonal,
            )
            eta_eff[i]  = eta_t
            eta_arr[t]  = eta_t

            eta_excess = max(0.0, abs(eta_t) - ETA_LIMIT)
            penalty += LAM_ETA * eta_excess**2

            pi_arr[t] = float(expit(eta_t))

            # Stable log probabilities; no clipping needed
            log_pi_t = log_expit(eta_arr[t])
            log_one_minus_pi_t = log_expit(-eta_arr[t])

            # Likelihood
            if y[t] == 0:
                ll_t = log_one_minus_pi_t
            else:
                ll_dist = self.dist.logpdf(y[t], **call_params)

                if not np.isfinite(ll_dist):
                    return 1e12 + penalty if not return_paths else {}

                ll_t = log_pi_t + ll_dist

            if not np.isfinite(ll_t):
                return 1e12 + penalty if not return_paths else {}

            loglik += ll_t

            # Score — use gas._scaled_score to support all 3 scaling modes
            if y[t] > 0:
                raw_score = self.dist.score(y[t], **call_params)
                scaled = self.gas._scaled_score(raw_score, call_params)
                for j, name in enumerate(self.gas.tv_names):
                    s_arr[t, j] = scaled[name]
                if not np.all(np.isfinite(s_arr[t, :])):
                    return 1e12 + penalty if not return_paths else {}

            # GAS update, no clipping
            for j, name in enumerate(self.gas.tv_names):
                omega_j = gp[f"omega_{name}"]
                ar_part = 0.0
                score_part = 0.0

                for l in L:
                    idx = t - l + 1

                    s_past = s_arr[idx, j] if idx >= 0 else 0.0
                    f_past = f_arr[idx, j] if idx >= 0 else gp[f"f0_{name}"]

                    score_part += gp[f"A_{name}_{l}"] * s_past
                    ar_part += gp[f"B_{name}_{l}"] * f_past

                f_next = omega_j + score_part + ar_part

                if not np.isfinite(f_next):
                    return 1e12 + penalty if not return_paths else {}

                f_arr[t + 1, j] = f_next

        objective = -loglik + penalty

        if not return_paths:
            return objective

        eff = self.max_lag

        return {
            "phi": f_arr[eff:T, 0],
            "xi": f_arr[eff:T, 1] if n_tv > 1 else None,
            "f_arr": f_arr[eff:T, :],
            "s_arr": s_arr[eff:T, :],
            "eta": eta_arr[eff:T],
            "pi": pi_arr[eff:T],
            "tv_names": self.gas.tv_names,
            "static": {s: gp[s] for s in self.gas.static_names},
            "eff_start": eff,
            "loglik": loglik,
            "penalty": penalty,
            "objective": objective,
            "y_eff": y[eff:T],
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

        T = len(y)
        eff = paths["eff_start"]
        n = T - eff
        cdfs = np.zeros(n)

        static = paths["static"]

        for i in range(n):
            t = eff + i
            pi_t = paths["pi"][i]
            call = {
                name: paths["f_arr"][i, j] for j, name in enumerate(paths["tv_names"])
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
        pi_theta = self.pi_dyn.initial_params(y, self.seasonal)
        return np.concatenate([gas_theta, pi_theta])

    def default_bounds(self) -> List[Tuple[float, float]]:
        """Combined bounds [gas_bounds | pi_bounds]."""
        return self.gas.default_bounds() + self.pi_dyn.default_bounds(self.seasonal)

    def fit(
        self,
        y: np.ndarray,
        verbose: bool = False,
        theta0: np.ndarray | None = None,
        options: dict | None = None,
        polish: bool = True,
    ) -> dict:
        """
        Estimate all parameters by unbounded BFGS maximum likelihood.

        Includes a polish step (second BFGS restart) and convergence
        classification per OPTIMIZATION.md §11.

        Returns
        -------
        dict with keys:
            theta, loglik, validity, success, message,
            n_iter, n_fev, grad, grad_norm_inf, grad_norm_2,
            hess_inv, std_errors, se_quality,
            runtime_s, peak_mem_mb, polish_improvement,
            param_names,
            gas_theta, pi_theta  (backward-compat slices)
        """
        import tracemalloc

        theta0 = (
            self.initial_theta(y) if theta0 is None
            else np.asarray(theta0, dtype=float)
        )
        opt_options = {"maxiter": 1000, "gtol": 1e-3, "disp": verbose}
        if options:
            opt_options.update(options)

        t_start = time.time()
        tracemalloc.start()

        def obj(theta):
            return self._run_filter(theta, y, return_paths=False)

        res = minimize(fun=obj, x0=theta0, method="BFGS", options=opt_options)

        # Polish step: short restart to catch any remaining gradient, capped at
        # 200 iterations so it doesn't double the runtime.
        polish_improvement = 0.0
        if polish:
            polish_opts = {**opt_options, "maxiter": 200}
            res2 = minimize(fun=obj, x0=res.x, method="BFGS", options=polish_opts)
            polish_improvement = abs(res.fun - res2.fun) / (1.0 + abs(res.fun))
            if res2.fun < res.fun:
                res = res2

        runtime_s = time.time() - t_start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Gradient norms
        grad         = getattr(res, "jac", None)
        grad_norm_inf = float(np.max(np.abs(grad))) if grad is not None else float("nan")
        grad_norm_2   = float(np.linalg.norm(grad)) if grad is not None else float("nan")

        # Convergence classification
        paths      = self._run_filter(res.x, y, return_paths=True)
        ll_final   = float(paths["loglik"]) if paths else float(-res.fun)
        finite_p   = bool(np.all(np.isfinite(res.x)))
        finite_ll  = bool(np.isfinite(ll_final))
        states_ok  = bool(paths) and bool(np.all(np.isfinite(paths.get("f_arr", [[0]]))))
        grad_ok    = (grad_norm_inf < 1e-2) if not np.isnan(grad_norm_inf) else False

        if res.success and finite_p and finite_ll and states_ok and grad_ok:
            validity = "valid_converged"
        elif finite_p and finite_ll and states_ok:
            validity = "valid_with_warning"
        else:
            validity = "failed"

        # Standard errors from BFGS inverse Hessian
        hess_inv   = np.array(res.hess_inv) if hasattr(res, "hess_inv") else None
        std_errors = None
        se_quality = "unavailable"
        if hess_inv is not None and hess_inv.ndim == 2:
            diag_h = np.diag(hess_inv)
            if np.all(diag_h > 0) and np.all(np.isfinite(diag_h)):
                std_errors = np.sqrt(diag_h)
                se_quality = "approximate"
            else:
                se_quality = "unreliable"

        # ── Bound diagnostics ─────────────────────────────────────────────────
        # Check whether any optimized parameter is near a soft-penalty boundary.
        # "near limit" = within 10 % of the threshold.  Saved in metadata.json.
        gp_fin  = self.gas.codec.decode(res.x[:self.gas.codec.n_params])
        pi_fin  = self._decode_pi(res.x[self.gas.codec.n_params:])
        B_LIMIT      = 0.98
        A_LIMIT      = 2.0
        RHO_LIMIT    = 0.98
        STATE_LIMIT  = 15.0
        ETA_LIMIT    = 30.0
        STATIC_LIMIT = 20.0

        bound_diags: dict = {}
        for name in self.gas.tv_names:
            b_sum = float(np.sum(np.abs([gp_fin[f"B_{name}_{l}"] for l in self.lags])))
            a_sum = float(np.sum(np.abs([gp_fin[f"A_{name}_{l}"] for l in self.lags])))
            bound_diags[f"{name}_B_sum"]          = b_sum
            bound_diags[f"{name}_B_near_limit"]   = bool(b_sum > 0.9 * B_LIMIT)
            bound_diags[f"{name}_A_sum"]          = a_sum
            bound_diags[f"{name}_A_near_limit"]   = bool(a_sum > 0.9 * A_LIMIT)
        rho = float(pi_fin.get("rho", 0.0))
        bound_diags["pi_rho"]             = rho
        bound_diags["pi_rho_near_limit"]  = bool(abs(rho) > 0.9 * RHO_LIMIT)
        for s in self.gas.static_names:
            v = float(gp_fin.get(s, 0.0))
            bound_diags[f"static_{s}"]              = v
            bound_diags[f"static_{s}_near_limit"]   = bool(abs(v) > 0.9 * STATIC_LIMIT)
        if paths:
            f_max = float(np.max(np.abs(paths.get("f_arr", np.zeros((1, 1))))))
            eta_max = float(np.max(np.abs(paths.get("eta", np.zeros(1)))))
            bound_diags["max_state_magnitude"]   = f_max
            bound_diags["state_near_limit"]      = bool(f_max > 0.9 * STATE_LIMIT)
            bound_diags["max_eta_magnitude"]     = eta_max
            bound_diags["eta_near_limit"]        = bool(eta_max > 0.9 * ETA_LIMIT)
        bound_diags["scaling_fallback_count"] = int(self.gas._fallback_count)

        n_gas = self.gas.codec.n_params
        return {
            "theta":              res.x,
            "loglik":             ll_final,
            "validity":           validity,
            "success":            bool(res.success),
            "message":            str(res.message),
            "n_iter":             int(res.nit),
            "n_fev":              int(res.nfev),
            "grad":               grad,
            "grad_norm_inf":      grad_norm_inf,
            "grad_norm_2":        grad_norm_2,
            "hess_inv":           hess_inv,
            "std_errors":         std_errors,
            "se_quality":         se_quality,
            "runtime_s":          runtime_s,
            "peak_mem_mb":        peak_mem / 1e6,
            "polish_improvement": polish_improvement,
            "param_names":        self.parameter_names(),
            "bound_diagnostics":  bound_diags,
            # backward-compat slices
            "gas_theta":          res.x[:n_gas],
            "pi_theta":           res.x[n_gas:],
        }

    def save_result(self, result: dict, cache_dir: "Path") -> None:
        """Save optimization result and metadata to cache_dir."""
        import json as _json
        from pathlib import Path as P
        cache_dir = P(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        names = result.get("param_names", self.parameter_names())
        theta = result["theta"]
        pd.DataFrame({"parameter": names, "value": theta}).to_csv(
            cache_dir / "estimated_parameters.csv", index=False
        )

        if result.get("std_errors") is not None:
            pd.DataFrame({
                "parameter": names,
                "std_error":  result["std_errors"],
                "se_quality": result["se_quality"],
            }).to_csv(cache_dir / "standard_errors.csv", index=False)

        meta = {
            "model_type":         "ZAGASModel",
            "n_params":           self.n_params,
            "tv_names":           list(self.gas.tv_names),
            "lags":               list(self.lags),
            "scaling":            self.gas.scaling,
            "seasonal":           self.seasonal,
            "loglik":             float(result["loglik"]),
            "validity":           result.get("validity", "unknown"),
            "success":            bool(result.get("success", False)),
            "message":            str(result.get("message", "")),
            "n_iter":             int(result.get("n_iter", 0)),
            "n_fev":              int(result.get("n_fev", 0)),
            "grad_norm_inf":      float(result.get("grad_norm_inf", float("nan"))),
            "grad_norm_2":        float(result.get("grad_norm_2",   float("nan"))),
            "runtime_s":          float(result.get("runtime_s",     float("nan"))),
            "peak_mem_mb":        float(result.get("peak_mem_mb",   float("nan"))),
            "se_quality":         result.get("se_quality", "unavailable"),
            "polish_improvement": float(result.get("polish_improvement", 0.0)),
            "bound_diagnostics":  result.get("bound_diagnostics", {}),
        }
        (cache_dir / "metadata.json").write_text(_json.dumps(meta, indent=2))

        if result.get("grad") is not None:
            pd.DataFrame({"parameter": names, "gradient": result["grad"]}).to_csv(
                cache_dir / "gradients.csv", index=False
            )
        if result.get("hess_inv") is not None:
            hi = np.array(result["hess_inv"])
            if hi.ndim == 2:
                np.save(cache_dir / "hess_inv.npy", hi)

    def simulate_oos(
        self,
        theta: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """
        1-step-ahead rolling OOS evaluation.

        Runs the GAS recursion over the concatenated train+test series,
        then returns the test-period slice of filtered states.

        Returns dict with keys expected by pipeline.runner._compute_oos_metrics:
            f_arr_oos, pi_oos, eta_oos, static, tv_names, y_test
        """
        gas_theta, pi_theta = self._split_theta(theta)
        gp        = self.gas.codec.decode(gas_theta)
        pi_params = self._decode_pi(pi_theta)

        y_full  = np.concatenate([y_train, y_test])
        T_train = len(y_train)
        T_full  = len(y_full)
        n_tv    = len(self.gas.tv_names)
        L       = self.lags
        max_lag = self.max_lag

        f_arr    = np.zeros((T_full + 1, n_tv))
        s_arr    = np.zeros((T_full,     n_tv))
        eta_arr  = np.zeros(T_full)   # t-indexed (for output)
        eta_eff  = np.zeros(T_full)   # i-indexed (for AR history in compute_eta)
        pi_arr   = np.zeros(T_full)

        for j, name in enumerate(self.gas.tv_names):
            f_arr[:max_lag + 1, j] = gp[f"f0_{name}"]

        for t in range(max_lag, T_full):
            call_params = {name: f_arr[t, j]
                           for j, name in enumerate(self.gas.tv_names)}
            call_params.update({s: gp[s] for s in self.gas.static_names})

            i = t - max_lag
            eta_t = self.pi_dyn.compute_eta(
                i=i, eta_hist=eta_eff,
                y_full=y_full, t=t,
                params=pi_params, seasonal=self.seasonal,
            )
            eta_eff[i] = eta_t
            eta_arr[t] = eta_t
            pi_arr[t] = float(expit(eta_t))

            if y_full[t] > 0:
                raw_score = self.dist.score(y_full[t], **call_params)
                # Use gas._scaled_score to support all 3 scaling modes
                scaled    = self.gas._scaled_score(raw_score, call_params)
                for j, name in enumerate(self.gas.tv_names):
                    s_arr[t, j] = scaled[name]

            for j, name in enumerate(self.gas.tv_names):
                omega_j = gp[f"omega_{name}"]
                ar_part = score_part = 0.0
                for l in L:
                    idx    = t - l + 1
                    s_past = s_arr[idx, j] if idx >= 0 else 0.0
                    f_past = f_arr[idx, j] if idx >= 0 else gp[f"f0_{name}"]
                    score_part += gp[f"A_{name}_{l}"] * s_past
                    ar_part    += gp[f"B_{name}_{l}"] * f_past
                f_arr[t + 1, j] = omega_j + score_part + ar_part

        oos = slice(T_train, T_full)
        return {
            "f_arr_oos": f_arr[oos, :],
            "pi_oos":    pi_arr[oos],
            "eta_oos":   eta_arr[oos],
            "static":    {s: gp[s] for s in self.gas.static_names},
            "tv_names":  list(self.gas.tv_names),
            "y_test":    y_test,
        }
