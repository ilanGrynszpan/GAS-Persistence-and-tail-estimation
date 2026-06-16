"""
Exogenous (covariate-augmented) ZA-GAS model.

For each time-varying parameter j ∈ {phi, xi} (or a subset):

    f_{j,t+1} = ω_j
              + Σ_{l ∈ L} A_{j,l} s_{j,t-l+1}    (score terms)
              + Σ_{l ∈ L} B_{j,l} f_{j,t-l+1}     (AR terms)
              + Γ_j' X_t                             (covariate contribution)

where X_t is a standardised covariate vector.

The covariate coefficients Γ_j are appended to the parameter vector AFTER the
standard [gas_theta | pi_theta] block:

    theta = [ gas_theta | pi_theta | cov_theta ]

cov_theta layout:
    For j in cov_target (ordered by tv_param_names):
        Γ_{j,0}, Γ_{j,1}, …, Γ_{j,n_cov-1}

This design ensures the base ZAGASModel parameters are untouched and cached
results remain backward-compatible.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from distributions.base import Distribution
from pi_dynamics.base import PiDynamics
from models.gas_filter import GASFilter
from models.lags import SEASONAL_LAGS
from models.za_gas_model import ZAGASModel


# ─────────────────────────────────────────────────────────────────────────────

class CovZAGASModel:
    """
    ZA-GAS model extended with exogenous covariates.

    Parameters
    ----------
    distribution   : Distribution
    pi_dynamics    : PiDynamics
    seasonal       : 'daily' or 'monthly'
    cov_names      : list of covariate column names (for reporting)
    cov_target     : TV params that receive covariate terms; None = all TV params
    scale_score    : passed to GASFilter
    static_params  : static distribution params (passed to GASFilter)
    """

    def __init__(
        self,
        distribution: Distribution,
        pi_dynamics: PiDynamics,
        seasonal: str = "daily",
        cov_names: Optional[List[str]] = None,
        cov_target: Optional[List[str]] = None,
        scale_score: bool = True,
        static_params: Optional[List[str]] = None,
    ):
        self.dist = distribution
        self.pi_dyn = pi_dynamics
        self.seasonal = seasonal
        self.lags = SEASONAL_LAGS[seasonal]
        self.max_lag = max(self.lags)

        self.gas = GASFilter(
            distribution=distribution,
            seasonal=seasonal,
            scale_score=scale_score,
            static_params=static_params,
        )

        self._pi_names = pi_dynamics.param_names(seasonal)
        self.tv_names = distribution.tv_param_names

        # Covariates
        self.cov_names = list(cov_names) if cov_names else []
        self.n_cov = len(self.cov_names)
        self.cov_target = list(cov_target) if cov_target is not None else list(self.tv_names)
        # Restrict cov_target to actual TV params
        self.cov_target = [j for j in self.cov_target if j in self.tv_names]

    # ─────────────────────────────────────────────────────────────────────────
    # Parameter vector layout
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def n_gas(self) -> int:
        return self.gas.codec.n_params

    @property
    def n_pi(self) -> int:
        return len(self._pi_names)

    @property
    def n_cov_params(self) -> int:
        return len(self.cov_target) * self.n_cov

    @property
    def n_params(self) -> int:
        return self.n_gas + self.n_pi + self.n_cov_params

    def _split_theta(
        self, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split flat theta into (gas_theta, pi_theta, cov_theta)."""
        gas = theta[: self.n_gas]
        pi = theta[self.n_gas : self.n_gas + self.n_pi]
        cov = theta[self.n_gas + self.n_pi :]
        return gas, pi, cov

    def _decode_cov(self, cov_theta: np.ndarray) -> Dict[str, np.ndarray]:
        """Return dict {tv_param_j: Gamma_j array of length n_cov}."""
        result = {}
        for k, j in enumerate(self.cov_target):
            start = k * self.n_cov
            result[j] = cov_theta[start : start + self.n_cov]
        return result

    def parameter_names(self) -> List[str]:
        gas_names = [
            name
            for name, _ in sorted(
                self.gas.codec._idx.items(), key=lambda item: item[1]
            )
        ]
        pi_names = list(self._pi_names)
        cov_names_flat = []
        for j in self.cov_target:
            for c in self.cov_names:
                cov_names_flat.append(f"gamma_{j}_{c}")
        return gas_names + pi_names + cov_names_flat

    # ─────────────────────────────────────────────────────────────────────────
    # Core filter
    # ─────────────────────────────────────────────────────────────────────────

    def _run_filter(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        X: np.ndarray,  # shape (T, n_cov)
        return_paths: bool = False,
    ):
        """
        GAS recursion with exogenous covariates.

        X must be aligned with y (row t corresponds to time t).
        NaN in X is treated as 0 (mean-zero after standardisation).
        """
        T = len(y)
        if T <= self.max_lag:
            return 1e12 if not return_paths else {}

        gas_theta, pi_theta, cov_theta = self._split_theta(theta)
        pi_params = {name: float(pi_theta[i]) for i, name in enumerate(self._pi_names)}
        gp = self.gas.codec.decode(gas_theta)
        gamma = self._decode_cov(cov_theta)  # {j: array}

        L = self.lags
        n_tv = len(self.tv_names)
        X_safe = np.nan_to_num(X, nan=0.0)  # no leakage from missing

        # ── Penalties ────────────────────────────────────────────────────────
        penalty = 0.0
        LAM_PERSIST = 1e5
        LAM_SCORE   = 1e3
        LAM_STATE   = 1e3
        LAM_ETA     = 1e2
        LAM_STATIC  = 1e5
        LAM_COV     = 1e2

        gamma_v = gp.get("gamma", 1.0)
        zeta_v  = gp.get("zeta", 3.0)
        static_excess = max(0.0, gamma_v - zeta_v + 1e-6)
        penalty += LAM_STATIC * static_excess**2
        for s in self.gas.static_names:
            excess = max(0.0, abs(gp[s]) - 20.0)
            penalty += LAM_STATIC * excess**2

        for name in self.tv_names:
            b_sum = np.sum(np.abs([gp[f"B_{name}_{l}"] for l in L]))
            penalty += LAM_PERSIST * max(0.0, b_sum - 0.98) ** 2
            a_sum = np.sum(np.abs([gp[f"A_{name}_{l}"] for l in L]))
            penalty += LAM_SCORE * max(0.0, a_sum - 2.0) ** 2

        rho = pi_params.get("rho", 0.0)
        penalty += LAM_PERSIST * max(0.0, abs(rho) - 0.98) ** 2

        # Penalise very large covariate coefficients
        for j in self.cov_target:
            g_arr = gamma.get(j, np.zeros(self.n_cov))
            excess = max(0.0, np.sum(g_arr**2)**0.5 - 20.0)
            penalty += LAM_COV * excess**2

        # ── Allocate arrays ───────────────────────────────────────────────────
        f_arr   = np.zeros((T + 1, n_tv))
        s_arr   = np.zeros((T,     n_tv))
        eta_arr = np.zeros(T)
        pi_arr  = np.zeros(T)

        for j, name in enumerate(self.tv_names):
            f_arr[: self.max_lag + 1, j] = gp[f"f0_{name}"]

        loglik = 0.0

        # ── Main recursion ────────────────────────────────────────────────────
        for t in range(self.max_lag, T):

            call_params = {
                name: f_arr[t, j] for j, name in enumerate(self.tv_names)
            }
            call_params.update({s: gp[s] for s in self.gas.static_names})

            for j in range(n_tv):
                state_excess = max(0.0, abs(f_arr[t, j]) - 15.0)
                penalty += LAM_STATE * state_excess**2

            # Pi recursion
            eta_arr[t] = self.pi_dyn.compute_eta(
                i=t - self.max_lag,
                eta_hist=eta_arr,
                y_full=y,
                t=t,
                params=pi_params,
                seasonal=self.seasonal,
            )
            penalty += LAM_ETA * max(0.0, abs(eta_arr[t]) - 30.0) ** 2
            pi_arr[t] = float(expit(eta_arr[t]))

            log_pi   = log_expit(eta_arr[t])
            log_1mpi = log_expit(-eta_arr[t])

            if y[t] == 0:
                ll_t = log_1mpi
            else:
                ll_dist = self.dist.logpdf(y[t], **call_params)
                if not np.isfinite(ll_dist):
                    return 1e12 + penalty if not return_paths else {}
                ll_t = log_pi + ll_dist

            if not np.isfinite(ll_t):
                return 1e12 + penalty if not return_paths else {}
            loglik += ll_t

            # Score
            if y[t] > 0:
                raw_score = self.dist.score(y[t], **call_params)
                if self.gas.scale_score:
                    fi = self.dist.fisher_info_diag(**call_params)
                    for j, name in enumerate(self.tv_names):
                        s_arr[t, j] = raw_score[name] / max(fi[name], 1e-8)
                else:
                    for j, name in enumerate(self.tv_names):
                        s_arr[t, j] = raw_score[name]
                if not np.all(np.isfinite(s_arr[t, :])):
                    return 1e12 + penalty if not return_paths else {}

            # GAS update + covariate contribution
            x_t = X_safe[t]  # shape (n_cov,)
            for j, name in enumerate(self.tv_names):
                omega_j = gp[f"omega_{name}"]
                ar_part = score_part = 0.0
                for l in L:
                    idx = t - l + 1
                    s_past = s_arr[idx, j] if idx >= 0 else 0.0
                    f_past = f_arr[idx, j] if idx >= 0 else gp[f"f0_{name}"]
                    score_part += gp[f"A_{name}_{l}"] * s_past
                    ar_part    += gp[f"B_{name}_{l}"] * f_past

                # Covariate term (only for params in cov_target)
                cov_term = 0.0
                if name in self.cov_target:
                    g_arr = gamma.get(name, np.zeros(self.n_cov))
                    cov_term = float(np.dot(g_arr, x_t))

                f_next = omega_j + score_part + ar_part + cov_term
                if not np.isfinite(f_next):
                    return 1e12 + penalty if not return_paths else {}
                f_arr[t + 1, j] = f_next

        objective = -loglik + penalty

        if not return_paths:
            return objective

        eff = self.max_lag
        return {
            "f_arr":    f_arr[eff:T, :],
            "s_arr":    s_arr[eff:T, :],
            "eta":      eta_arr[eff:T],
            "pi":       pi_arr[eff:T],
            "tv_names": self.tv_names,
            "static":   {s: gp[s] for s in self.gas.static_names},
            "eff_start": eff,
            "loglik":   loglik,
            "penalty":  penalty,
            "objective": objective,
            "y_eff":    y[eff:T],
            "gamma":    {j: gamma.get(j, np.zeros(self.n_cov)) for j in self.tv_names},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def loglik(self, theta: np.ndarray, y: np.ndarray, X: np.ndarray) -> float:
        return -self._run_filter(theta, y, X, return_paths=False)

    def filter(self, theta: np.ndarray, y: np.ndarray, X: np.ndarray) -> dict:
        return self._run_filter(theta, y, X, return_paths=True)

    def cdf_series(
        self, theta: np.ndarray, y: np.ndarray, X: np.ndarray
    ) -> np.ndarray:
        paths = self.filter(theta, y, X)
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
                name: paths["f_arr"][i, j]
                for j, name in enumerate(paths["tv_names"])
            }
            call.update(static)
            if y[t] <= 0:
                cdfs[i] = 1.0 - pi_t
            else:
                G_t = self.dist.cdf(y[t], **call)
                cdfs[i] = (1.0 - pi_t) + pi_t * G_t
        return cdfs

    def initial_theta(self, y: np.ndarray) -> np.ndarray:
        """Combined starting vector [gas_theta | pi_theta | zeros_for_covariates]."""
        gas0 = self.gas.initial_theta(y)
        pi0  = self.pi_dyn.initial_params(y, self.seasonal)
        cov0 = np.zeros(self.n_cov_params)
        return np.concatenate([gas0, pi0, cov0])

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        verbose: bool = False,
        theta0: Optional[np.ndarray] = None,
        options: Optional[dict] = None,
    ) -> dict:
        """
        Estimate all parameters by maximum likelihood (unbounded BFGS).

        Parameters
        ----------
        y      : training observations (T,)
        X      : covariate matrix (T, n_cov), already standardised
        theta0 : optional starting values
        """
        if theta0 is None:
            theta0 = self.initial_theta(y)
        else:
            theta0 = np.asarray(theta0, dtype=float)

        opt_options = {"maxiter": 5000, "gtol": 1e-5, "disp": verbose}
        if options:
            opt_options.update(options)

        def obj(theta):
            return self._run_filter(theta, y, X, return_paths=False)

        result = minimize(
            fun=obj,
            x0=theta0,
            method="BFGS",
            options=opt_options,
        )

        return {
            "theta":    result.x,
            "loglik":   -result.fun,
            "success":  result.success,
            "message":  result.message,
            "n_iter":   result.nit,
            "grad":     result.jac if hasattr(result, "jac") else None,
            "hess_inv": result.hess_inv if hasattr(result, "hess_inv") else None,
            "result":   result,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # OOS simulation
    # ─────────────────────────────────────────────────────────────────────────

    def simulate_oos(
        self,
        theta: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> dict:
        """
        Advance the fitted model into the test period (1-step-ahead rolling).

        Returns dict compatible with simulator.py predictive helpers.
        """
        gas_theta, pi_theta, cov_theta = self._split_theta(theta)
        gp       = self.gas.codec.decode(gas_theta)
        pi_params = {name: float(pi_theta[i]) for i, name in enumerate(self._pi_names)}
        gamma    = self._decode_cov(cov_theta)

        L       = self.lags
        max_lag = self.max_lag
        n_tv    = len(self.tv_names)

        y_full  = np.concatenate([y_train, y_test])
        X_full  = np.vstack([X_train, X_test])
        X_full  = np.nan_to_num(X_full, nan=0.0)
        T_train = len(y_train)
        T_full  = len(y_full)

        f_arr   = np.zeros((T_full + 1, n_tv))
        s_arr   = np.zeros((T_full,     n_tv))
        eta_arr = np.zeros(T_full)
        pi_arr  = np.zeros(T_full)

        for j, name in enumerate(self.tv_names):
            f_arr[: max_lag + 1, j] = gp[f"f0_{name}"]

        def _update(t, y_full, f_arr, s_arr, eta_arr, pi_arr):
            call_params = {
                name: f_arr[t, j] for j, name in enumerate(self.tv_names)
            }
            call_params.update({s: gp[s] for s in self.gas.static_names})

            eta_arr[t] = self.pi_dyn.compute_eta(
                i=t - max_lag, eta_hist=eta_arr,
                y_full=y_full, t=t, params=pi_params, seasonal=self.seasonal,
            )
            pi_arr[t] = float(expit(eta_arr[t]))

            if y_full[t] > 0:
                raw = self.dist.score(y_full[t], **call_params)
                if self.gas.scale_score:
                    fi = self.dist.fisher_info_diag(**call_params)
                    for j, name in enumerate(self.tv_names):
                        s_arr[t, j] = raw[name] / max(fi[name], 1e-8)
                else:
                    for j, name in enumerate(self.tv_names):
                        s_arr[t, j] = raw[name]

            x_t = X_full[t]
            for j, name in enumerate(self.tv_names):
                omega_j  = gp[f"omega_{name}"]
                ar_part = score_part = 0.0
                for l in L:
                    idx    = t - l + 1
                    s_past = s_arr[idx, j] if idx >= 0 else 0.0
                    f_past = f_arr[idx, j] if idx >= 0 else gp[f"f0_{name}"]
                    score_part += gp[f"A_{name}_{l}"] * s_past
                    ar_part    += gp[f"B_{name}_{l}"] * f_past
                cov_term = 0.0
                if name in self.cov_target:
                    g_arr = gamma.get(name, np.zeros(self.n_cov))
                    cov_term = float(np.dot(g_arr, x_t))
                f_arr[t + 1, j] = omega_j + score_part + ar_part + cov_term

        for t in range(max_lag, T_full):
            _update(t, y_full, f_arr, s_arr, eta_arr, pi_arr)

        oos_slice = slice(T_train, T_full)
        return {
            "f_arr_oos": f_arr[oos_slice, :],
            "phi_oos":   f_arr[oos_slice, 0],
            "xi_oos":    f_arr[oos_slice, 1] if n_tv > 1 else None,
            "eta_oos":   eta_arr[oos_slice],
            "pi_oos":    pi_arr[oos_slice],
            "static":    {s: gp[s] for s in self.gas.static_names},
            "tv_names":  self.tv_names,
            "y_test":    y_test,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Cache helpers
    # ─────────────────────────────────────────────────────────────────────────

    def save_result(self, result: dict, cache_dir: Path) -> None:
        """Save optimization result and model metadata to cache_dir."""
        import json
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        names = self.parameter_names()
        theta = result["theta"]
        pd.DataFrame({"parameter": names, "value": theta}).to_csv(
            cache_dir / "estimated_parameters.csv", index=False
        )

        meta = {
            "n_params":   self.n_params,
            "cov_names":  self.cov_names,
            "cov_target": self.cov_target,
            "tv_names":   list(self.tv_names),
            "seasonal":   self.seasonal,
            "loglik":     float(result["loglik"]),
            "success":    bool(result["success"]),
            "message":    str(result.get("message", "")),
            "n_iter":     int(result.get("n_iter", 0)),
        }
        (cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        if result.get("grad") is not None:
            pd.DataFrame({"parameter": names, "gradient": result["grad"]}).to_csv(
                cache_dir / "gradients.csv", index=False
            )

        # Save BFGS inverse Hessian approximation if available
        if result.get("hess_inv") is not None:
            hi = np.array(result["hess_inv"])
            if hi.ndim == 2:
                np.save(cache_dir / "hess_inv.npy", hi)

    @staticmethod
    def load_theta(cache_dir: Path) -> np.ndarray:
        """Load theta from cached estimated_parameters.csv."""
        df = pd.read_csv(Path(cache_dir) / "estimated_parameters.csv")
        return df["value"].to_numpy(dtype=float)
