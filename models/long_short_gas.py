"""
Long-Short ZA-GAS model.

Decomposes each time-varying parameter θ ∈ {φ, ξ} into:

    h(θ_t) = ω_θ  +  GAS^θ_t  +  L^θ_t  +  S^θ_t

where

    GAS^θ_t  =  Σ_{l ∈ L} A_{θ,l} s_{θ,t-l+1}   (score terms)
             +  Σ_{l ∈ L} B_{θ,l} f_{θ,t-l+1}    (AR terms on total state)

    L^θ_t    =  α^θ_L · L^θ_{t-1}  +  β^θ_L' · X^long_t
                                         (long-run: N34 variables)

    S^θ_t    =  Γ^θ_S' · X^short_t
                                         (short-run: D, T, MJO, contemporaneous)

X^long  (N34 variables)  and  X^short  (D, T, MJO)  are passed as separate
standardised matrices.

Parameter vector layout:
    [ gas_theta | pi_theta | ls_theta ]

ls_theta contains, for each j in ls_target (in tv_param_names order):
    alpha_L_j   (AR coefficient of L)
    L0_j        (initial state L^j_0)
    beta_L_j_0, …, beta_L_j_{n_long-1}   (X^long coefficients)
    Gamma_S_j_0, …, Gamma_S_j_{n_short-1} (X^short coefficients)

If a parameter is not in ls_target, L and S are zero for it.
"""

from __future__ import annotations
import json
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


class LongShortZAGASModel:
    """
    ZA-GAS model with long-short (L_t / S_t) covariate decomposition.

    Parameters
    ----------
    distribution   : Distribution
    pi_dynamics    : PiDynamics
    seasonal       : 'daily' or 'monthly'
    long_names     : covariate names for X_long (N34 variables)
    short_names    : covariate names for X_short (D, T, MJO)
    ls_target      : TV params that get the L/S decomposition; None = all TV
    scale_score    : passed to GASFilter
    static_params  : passed to GASFilter
    """

    def __init__(
        self,
        distribution: Distribution,
        pi_dynamics: PiDynamics,
        seasonal: str = "daily",
        long_names: Optional[List[str]] = None,
        short_names: Optional[List[str]] = None,
        ls_target: Optional[List[str]] = None,
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

        self.long_names  = list(long_names)  if long_names  else []
        self.short_names = list(short_names) if short_names else []
        self.n_long  = len(self.long_names)
        self.n_short = len(self.short_names)

        # Restrict ls_target to actual TV params
        self.ls_target = list(ls_target) if ls_target is not None else list(self.tv_names)
        self.ls_target = [j for j in self.ls_target if j in self.tv_names]

    # ─────────────────────────────────────────────────────────────────────────
    # Parameter counts
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def n_ls_per_param(self) -> int:
        """Number of long-short parameters per TV param in ls_target."""
        # alpha_L + L0 + beta_L (n_long) + Gamma_S (n_short)
        return 2 + self.n_long + self.n_short

    @property
    def n_gas(self) -> int:
        return self.gas.codec.n_params

    @property
    def n_pi(self) -> int:
        return len(self._pi_names)

    @property
    def n_ls_params(self) -> int:
        return len(self.ls_target) * self.n_ls_per_param

    @property
    def n_params(self) -> int:
        return self.n_gas + self.n_pi + self.n_ls_params

    # ─────────────────────────────────────────────────────────────────────────
    # Parameter vector helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _split_theta(
        self, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gas = theta[: self.n_gas]
        pi  = theta[self.n_gas : self.n_gas + self.n_pi]
        ls  = theta[self.n_gas + self.n_pi :]
        return gas, pi, ls

    def _decode_ls(self, ls_theta: np.ndarray) -> Dict[str, dict]:
        """Return {j: {alpha_L, L0, beta_L, Gamma_S}} for each j in ls_target."""
        result = {}
        per = self.n_ls_per_param
        for k, j in enumerate(self.ls_target):
            block = ls_theta[k * per : (k + 1) * per]
            result[j] = {
                "alpha_L":  float(block[0]),
                "L0":       float(block[1]),
                "beta_L":   block[2 : 2 + self.n_long],
                "Gamma_S":  block[2 + self.n_long : 2 + self.n_long + self.n_short],
            }
        return result

    def parameter_names(self) -> List[str]:
        gas_names = [
            name
            for name, _ in sorted(
                self.gas.codec._idx.items(), key=lambda x: x[1]
            )
        ]
        pi_names = list(self._pi_names)
        ls_names_flat = []
        for j in self.ls_target:
            ls_names_flat.append(f"alpha_L_{j}")
            ls_names_flat.append(f"L0_{j}")
            for c in self.long_names:
                ls_names_flat.append(f"beta_L_{j}_{c}")
            for c in self.short_names:
                ls_names_flat.append(f"gamma_S_{j}_{c}")
        return gas_names + pi_names + ls_names_flat

    # ─────────────────────────────────────────────────────────────────────────
    # Core filter
    # ─────────────────────────────────────────────────────────────────────────

    def _run_filter(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        X_long: np.ndarray,   # shape (T, n_long)
        X_short: np.ndarray,  # shape (T, n_short)
        return_paths: bool = False,
    ):
        T = len(y)
        if T <= self.max_lag:
            return 1e12 if not return_paths else {}

        gas_theta, pi_theta, ls_theta = self._split_theta(theta)
        pi_params = {name: float(pi_theta[i]) for i, name in enumerate(self._pi_names)}
        gp  = self.gas.codec.decode(gas_theta)
        lsd = self._decode_ls(ls_theta)  # {j: {alpha_L, L0, beta_L, Gamma_S}}

        L    = self.lags
        n_tv = len(self.tv_names)
        Xl   = np.nan_to_num(X_long,  nan=0.0)
        Xs   = np.nan_to_num(X_short, nan=0.0)

        # ── Penalties ────────────────────────────────────────────────────────
        penalty = 0.0
        LAM_PERSIST = 1e5
        LAM_SCORE   = 1e3
        LAM_STATE   = 1e3
        LAM_ETA     = 1e2
        LAM_STATIC  = 1e5
        LAM_COV     = 1e2
        LAM_LONG    = 1e5

        gamma_v = gp.get("gamma", 1.0)
        zeta_v  = gp.get("zeta",  3.0)
        penalty += LAM_STATIC * max(0.0, gamma_v - zeta_v + 1e-6) ** 2
        for s in self.gas.static_names:
            penalty += LAM_STATIC * max(0.0, abs(gp[s]) - 20.0) ** 2

        for name in self.tv_names:
            b_sum = np.sum(np.abs([gp[f"B_{name}_{l}"] for l in L]))
            penalty += LAM_PERSIST * max(0.0, b_sum - 0.98) ** 2
            a_sum = np.sum(np.abs([gp[f"A_{name}_{l}"] for l in L]))
            penalty += LAM_SCORE * max(0.0, a_sum - 2.0) ** 2

        rho = pi_params.get("rho", 0.0)
        penalty += LAM_PERSIST * max(0.0, abs(rho) - 0.98) ** 2

        for j in self.ls_target:
            params_j = lsd[j]
            aL = params_j["alpha_L"]
            # Stationarity of L_t
            penalty += LAM_LONG * max(0.0, abs(aL) - 0.98) ** 2
            # Penalise extreme coefficients
            bL = params_j["beta_L"]
            gS = params_j["Gamma_S"]
            penalty += LAM_COV * max(0.0, np.linalg.norm(bL) - 20.0) ** 2
            penalty += LAM_COV * max(0.0, np.linalg.norm(gS) - 20.0) ** 2

        # ── Allocate state arrays ─────────────────────────────────────────────
        f_arr   = np.zeros((T + 1, n_tv))
        s_arr   = np.zeros((T,     n_tv))
        eta_arr = np.zeros(T)
        pi_arr  = np.zeros(T)

        # Long component state: one per TV param (even if not in ls_target -> 0)
        L_arr = np.zeros((T + 1, n_tv))

        # Initialise from f0 and L0
        for j, name in enumerate(self.tv_names):
            f_arr[: self.max_lag + 1, j] = gp[f"f0_{name}"]
            if name in lsd:
                L_arr[:, j] = lsd[name]["L0"]

        loglik = 0.0

        # ── Main recursion ────────────────────────────────────────────────────
        for t in range(self.max_lag, T):

            call_params = {
                name: f_arr[t, j] for j, name in enumerate(self.tv_names)
            }
            call_params.update({s: gp[s] for s in self.gas.static_names})

            for j in range(n_tv):
                penalty += LAM_STATE * max(0.0, abs(f_arr[t, j]) - 15.0) ** 2

            # Pi
            eta_arr[t] = self.pi_dyn.compute_eta(
                i=t - self.max_lag, eta_hist=eta_arr,
                y_full=y, t=t, params=pi_params, seasonal=self.seasonal,
            )
            penalty += LAM_ETA * max(0.0, abs(eta_arr[t]) - 30.0) ** 2
            pi_arr[t] = float(expit(eta_arr[t]))

            # Likelihood
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
                raw = self.dist.score(y[t], **call_params)
                if self.gas.scale_score:
                    fi = self.dist.fisher_info_diag(**call_params)
                    for j, name in enumerate(self.tv_names):
                        s_arr[t, j] = raw[name] / max(fi[name], 1e-8)
                else:
                    for j, name in enumerate(self.tv_names):
                        s_arr[t, j] = raw[name]
                if not np.all(np.isfinite(s_arr[t, :])):
                    return 1e12 + penalty if not return_paths else {}

            # Long-component update:  L_t = alpha_L * L_{t-1} + beta_L' X_long_t
            x_long_t  = Xl[t]   # shape (n_long,)
            x_short_t = Xs[t]   # shape (n_short,)

            for j, name in enumerate(self.tv_names):
                if name in lsd:
                    params_j = lsd[name]
                    aL = params_j["alpha_L"]
                    bL = params_j["beta_L"]
                    L_arr[t, j] = aL * L_arr[t - 1, j] + float(np.dot(bL, x_long_t))

            # GAS update + L_t + S_t
            for j, name in enumerate(self.tv_names):
                omega_j = gp[f"omega_{name}"]
                ar_part = score_part = 0.0
                for l in L:
                    idx    = t - l + 1
                    s_past = s_arr[idx, j] if idx >= 0 else 0.0
                    f_past = f_arr[idx, j] if idx >= 0 else gp[f"f0_{name}"]
                    score_part += gp[f"A_{name}_{l}"] * s_past
                    ar_part    += gp[f"B_{name}_{l}"] * f_past

                long_term  = L_arr[t, j]  # already updated above

                short_term = 0.0
                if name in lsd:
                    gS = lsd[name]["Gamma_S"]
                    short_term = float(np.dot(gS, x_short_t))

                f_next = omega_j + score_part + ar_part + long_term + short_term
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
            "L_arr":    L_arr[eff:T, :],   # long components (shape: T-eff, n_tv)
            "eta":      eta_arr[eff:T],
            "pi":       pi_arr[eff:T],
            "tv_names": self.tv_names,
            "static":   {s: gp[s] for s in self.gas.static_names},
            "eff_start": eff,
            "loglik":   loglik,
            "penalty":  penalty,
            "objective": objective,
            "y_eff":    y[eff:T],
            "ls_decoded": lsd,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def loglik(
        self, theta: np.ndarray, y: np.ndarray,
        X_long: np.ndarray, X_short: np.ndarray
    ) -> float:
        return -self._run_filter(theta, y, X_long, X_short, return_paths=False)

    def filter(
        self, theta: np.ndarray, y: np.ndarray,
        X_long: np.ndarray, X_short: np.ndarray
    ) -> dict:
        return self._run_filter(theta, y, X_long, X_short, return_paths=True)

    def cdf_series(
        self, theta: np.ndarray, y: np.ndarray,
        X_long: np.ndarray, X_short: np.ndarray
    ) -> np.ndarray:
        paths = self.filter(theta, y, X_long, X_short)
        if not paths:
            return np.array([])
        T   = len(y)
        eff = paths["eff_start"]
        n   = T - eff
        cdfs = np.zeros(n)
        static = paths["static"]
        for i in range(n):
            t    = eff + i
            pi_t = paths["pi"][i]
            call = {
                name: paths["f_arr"][i, j]
                for j, name in enumerate(paths["tv_names"])
            }
            call.update(static)
            if y[t] <= 0:
                cdfs[i] = 1.0 - pi_t
            else:
                cdfs[i] = (1.0 - pi_t) + pi_t * self.dist.cdf(y[t], **call)
        return cdfs

    def initial_theta(self, y: np.ndarray) -> np.ndarray:
        gas0 = self.gas.initial_theta(y)
        pi0  = self.pi_dyn.initial_params(y, self.seasonal)
        # Long-short initial: alpha_L=0.5, L0=0, betas/gammas=0
        ls0  = np.zeros(self.n_ls_params)
        per  = self.n_ls_per_param
        for k in range(len(self.ls_target)):
            ls0[k * per] = 0.5  # alpha_L
        return np.concatenate([gas0, pi0, ls0])

    def fit(
        self,
        y: np.ndarray,
        X_long: np.ndarray,
        X_short: np.ndarray,
        verbose: bool = False,
        theta0: Optional[np.ndarray] = None,
        options: Optional[dict] = None,
    ) -> dict:
        """Estimate all parameters by maximum likelihood (unbounded BFGS)."""
        if theta0 is None:
            theta0 = self.initial_theta(y)
        else:
            theta0 = np.asarray(theta0, dtype=float)

        opt_options = {"maxiter": 5000, "gtol": 1e-5, "disp": verbose}
        if options:
            opt_options.update(options)

        def obj(theta):
            return self._run_filter(theta, y, X_long, X_short, return_paths=False)

        result = minimize(
            fun=obj, x0=theta0, method="BFGS", options=opt_options,
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
        X_long_train: np.ndarray,
        X_long_test: np.ndarray,
        X_short_train: np.ndarray,
        X_short_test: np.ndarray,
    ) -> dict:
        """1-step-ahead rolling OOS evaluation."""
        gas_theta, pi_theta, ls_theta = self._split_theta(theta)
        gp        = self.gas.codec.decode(gas_theta)
        pi_params = {name: float(pi_theta[i]) for i, name in enumerate(self._pi_names)}
        lsd       = self._decode_ls(ls_theta)

        L       = self.lags
        max_lag = self.max_lag
        n_tv    = len(self.tv_names)

        y_full  = np.concatenate([y_train, y_test])
        Xl_full = np.vstack([X_long_train,  X_long_test])
        Xs_full = np.vstack([X_short_train, X_short_test])
        Xl_full = np.nan_to_num(Xl_full, nan=0.0)
        Xs_full = np.nan_to_num(Xs_full, nan=0.0)

        T_train = len(y_train)
        T_full  = len(y_full)

        f_arr   = np.zeros((T_full + 1, n_tv))
        s_arr   = np.zeros((T_full,     n_tv))
        L_arr   = np.zeros((T_full + 1, n_tv))
        eta_arr = np.zeros(T_full)
        pi_arr  = np.zeros(T_full)

        for j, name in enumerate(self.tv_names):
            f_arr[: max_lag + 1, j] = gp[f"f0_{name}"]
            if name in lsd:
                L_arr[:, j] = lsd[name]["L0"]

        def _update(t):
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

            x_long_t  = Xl_full[t]
            x_short_t = Xs_full[t]

            for j, name in enumerate(self.tv_names):
                if name in lsd:
                    aL = lsd[name]["alpha_L"]
                    bL = lsd[name]["beta_L"]
                    L_arr[t, j] = aL * L_arr[t - 1, j] + float(np.dot(bL, x_long_t))

            for j, name in enumerate(self.tv_names):
                omega_j = gp[f"omega_{name}"]
                ar_part = score_part = 0.0
                for l in L:
                    idx    = t - l + 1
                    s_past = s_arr[idx, j] if idx >= 0 else 0.0
                    f_past = f_arr[idx, j] if idx >= 0 else gp[f"f0_{name}"]
                    score_part += gp[f"A_{name}_{l}"] * s_past
                    ar_part    += gp[f"B_{name}_{l}"] * f_past
                long_term  = L_arr[t, j]
                short_term = 0.0
                if name in lsd:
                    short_term = float(np.dot(lsd[name]["Gamma_S"], x_short_t))
                f_arr[t + 1, j] = omega_j + score_part + ar_part + long_term + short_term

        for t in range(max_lag, T_full):
            _update(t)

        oos = slice(T_train, T_full)
        return {
            "f_arr_oos": f_arr[oos, :],
            "L_arr_oos": L_arr[oos, :],
            "phi_oos":   f_arr[oos, 0],
            "xi_oos":    f_arr[oos, 1] if n_tv > 1 else None,
            "eta_oos":   eta_arr[oos],
            "pi_oos":    pi_arr[oos],
            "static":    {s: gp[s] for s in self.gas.static_names},
            "tv_names":  self.tv_names,
            "y_test":    y_test,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Cache helpers
    # ─────────────────────────────────────────────────────────────────────────

    def save_result(self, result: dict, cache_dir: Path) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        names = self.parameter_names()
        theta = result["theta"]
        pd.DataFrame({"parameter": names, "value": theta}).to_csv(
            cache_dir / "estimated_parameters.csv", index=False
        )

        meta = {
            "n_params":    self.n_params,
            "long_names":  self.long_names,
            "short_names": self.short_names,
            "ls_target":   self.ls_target,
            "tv_names":    list(self.tv_names),
            "seasonal":    self.seasonal,
            "loglik":      float(result["loglik"]),
            "success":     bool(result["success"]),
            "message":     str(result.get("message", "")),
            "n_iter":      int(result.get("n_iter", 0)),
        }
        (cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        if result.get("grad") is not None:
            pd.DataFrame({"parameter": names, "gradient": result["grad"]}).to_csv(
                cache_dir / "gradients.csv", index=False
            )

        if result.get("hess_inv") is not None:
            hi = np.array(result["hess_inv"])
            if hi.ndim == 2:
                np.save(cache_dir / "hess_inv.npy", hi)

    @staticmethod
    def load_theta(cache_dir: Path) -> np.ndarray:
        df = pd.read_csv(Path(cache_dir) / "estimated_parameters.csv")
        return df["value"].to_numpy(dtype=float)
