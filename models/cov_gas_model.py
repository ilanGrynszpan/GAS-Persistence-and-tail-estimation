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
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from distributions.base import Distribution
from pi_dynamics.base import PiDynamics
from models.gas_filter import GASFilter, _sigmoid_scalar, _logit_scalar
from models.lags import SEASONAL_LAGS
from models.za_gas_model import ZAGASModel

# Numba fast path — compiled covariate filter functions
_NB_AVAIL_COV = False
try:
    from models._nb_gb2_filter import nb_filter_phi_only_cov, nb_filter_phi_xi_cov
    _NB_AVAIL_COV = True
except ImportError:
    pass


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
        gas_lags: Optional[List[int]] = None,
        scaling: str = "diagonal_inverse_fisher",
        cov_names: Optional[List[str]] = None,
        cov_target: Optional[List[str]] = None,
        # Legacy alias kept for backward compat
        scale_score: Optional[bool] = None,
        static_params: Optional[List[str]] = None,
        # Experimental: per-lag sigmoid-bounded A/B (default off) and
        # sigmoid-bounded occurrence persistence rho (default off) -- see
        # reports/seasonality GAS/gradient_and_convergence_diagnosis.md §8.
        # Both default to reproducing the exact previous behaviour.
        bound_ab: bool = False,
        a_max: float = 2.0,
        b_max: float = 1.0,
        bound_rho: bool = False,
        rho_max: float = 0.999,
        # Link-only ("v3") extensions, all default off -- see
        # ZAGASModel's constructor docstring for the full explanation of
        # each (identical semantics here).
        ab_signed: bool = False,
        rho_signed: bool = False,
        free_zeta_gamma: bool = False,
        kurtosis_margin: float = 4.0,
        unconstrained_penalties: bool = False,
        # Joint-softmax B (2026-09-02, requires bound_ab=True): replaces
        # the independent per-lag B box with a single joint bound on
        # sum_l B_{j,l} via a softmax-with-phantom-zero link -- see
        # models/gas_filter.py's _ParamCodec docstring. Restores the TRUE
        # stationarity requirement (a bound on the SUM only) instead of
        # the per-lag box's implicit even split, which was strictly
        # tighter than necessary. Default off; b_max above is ignored for
        # B (not A) when this is set -- use b_total_max instead.
        bound_b_joint: bool = False,
        b_total_max: float = 0.98,
        # Tanh-rescale B (2026-09-02, alternative to bound_b_joint): a
        # SIGNED joint sum link, sum_l B_{j,l} in (-b_sum_max, b_sum_max),
        # via B_{j,l} = raw_l * b_sum_max*tanh(S)/S, S=sum_l raw_l -- see
        # models/gas_filter.py's _ParamCodec docstring. Restores the exact
        # symmetric stationarity boundary (b_sum_max=1.0 = the true
        # |sum B_l|<1 requirement) while preserving whatever relative
        # split across lags the data wants, with NO restriction to
        # non-negative B (needed alongside ab_signed).
        bound_b_tanh_sum: bool = False,
        b_sum_max: float = 1.0,
        # State-magnitude safety net (default on, matching all prior
        # behaviour): a soft quadratic penalty on the raw recursive GAS
        # state |f_{j,t}| exceeding STATE_LIMIT=15, ADDED ON TOP OF the
        # unconditional isfinite() crash-guard in models/_nb_gb2_filter.py
        # (which is independent of this flag and always stays active --
        # this flag only removes the *statistical-looking* soft penalty
        # that was found to be the sole source of essentially every
        # "(boundary optimum)" SE label under V2, never the A/B/rho boxes
        # -- see report_se_utils.check_penalty_contamination). Setting
        # this True is an experiment, not (yet) a validated default.
        unconstrained_state: bool = False,
        # Static box (2026-09-02, mutually exclusive with free_zeta_gamma):
        # boxes gamma/zeta/xi/f0_* into the SAME (lo,hi) via a smooth
        # affine-sigmoid -- see models/gas_filter.py's _ParamCodec
        # docstring. Unlike free_zeta_gamma, does NOT guarantee the GB2
        # moment-existence relationship by construction -- see the
        # "static box" report for what that means for validity.
        bound_static_box: bool = False,
        static_box: Tuple[float, float] = (-2.0, 15.0),
    ):
        self.dist = distribution
        self.pi_dyn = pi_dynamics
        self.seasonal = seasonal
        self.bound_rho = bound_rho
        self.rho_max = rho_max
        self.rho_signed = rho_signed
        self.unconstrained_penalties = unconstrained_penalties
        self.unconstrained_state = unconstrained_state

        self.gas = GASFilter(
            distribution=distribution,
            seasonal=seasonal,
            gas_lags=gas_lags,
            scaling=scaling,
            scale_score=scale_score,
            static_params=static_params,
            bound_ab=bound_ab, a_max=a_max, b_max=b_max, ab_signed=ab_signed,
            bound_b_joint=bound_b_joint, b_total_max=b_total_max,
            bound_b_tanh_sum=bound_b_tanh_sum, b_sum_max=b_sum_max,
            free_zeta_gamma=free_zeta_gamma, kurtosis_margin=kurtosis_margin,
            bound_static_box=bound_static_box, static_box=static_box,
        )

        self.lags    = self.gas.lags
        self.max_lag = self.gas.max_lag

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

    def _decode_pi(self, pi_theta: np.ndarray) -> dict:
        """
        Return dict mapping pi parameter names -> values.

        When bound_rho is set, "rho" is decoded through rho_max*sigmoid(raw)
        (or the SIGNED rho_max*(2*sigmoid(raw)-1) when rho_signed too)
        instead of being read raw -- mirrors ZAGASModel._decode_pi; see the
        constructor docstring's bound_ab/bound_rho/rho_signed note.
        """
        out = {name: float(pi_theta[i]) for i, name in enumerate(self._pi_names)}
        if self.bound_rho and "rho" in out:
            if self.rho_signed:
                out["rho"] = self.rho_max * (2.0 * _sigmoid_scalar(out["rho"]) - 1.0)
            else:
                out["rho"] = self.rho_max * _sigmoid_scalar(out["rho"])
        return out

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
    # Numba fast path
    # ─────────────────────────────────────────────────────────────────────────

    # Maps scaling name to the integer code expected by the Numba functions.
    _SCALING_INT = {"unit": 0, "diagonal_inverse_fisher": 1, "inverse_fisher": 2}

    @property
    def _use_nb_filter(self) -> bool:
        """True when the Numba covariate filter covers this model configuration."""
        if not _NB_AVAIL_COV:
            return False
        # Numba filter only supports ARLogistic pi dynamics (hardcoded recursion)
        from pi_dynamics.ar_logistic import ARLogisticPiDynamics
        from pi_dynamics.ar_logistic_custom_lags import ARLogisticCustomLagsPiDynamics
        if not isinstance(self.pi_dyn, (ARLogisticPiDynamics, ARLogisticCustomLagsPiDynamics)):
            return False
        # Numba filter only supports GB2LogLink / GB2LogLinkPhiOnly distributions
        from distributions.gb2_log_link import GB2LogLink
        from distributions.gb2_phi_only import GB2LogLinkPhiOnly
        return isinstance(self.dist, (GB2LogLink, GB2LogLinkPhiOnly))

    def _nb_cov_dispatch(
        self,
        y:            np.ndarray,
        X_safe:       np.ndarray,
        gp:           dict,
        pi_params:    dict,
        gamma:        dict,
        pre_penalty:  float,
        T:            int,
        return_paths: bool,
    ):
        """
        Dispatch to the Numba covariate filter.

        pre_penalty: sum of all parameter-level penalties already computed in
                     _run_filter (persistence, score, static, covariate bounds).
        The Numba function adds per-step state/eta penalties internally and
        returns them as loop_pen.  Total = -loglik + pre_penalty + loop_pen.

        C_phi[t] = X_t @ gamma_phi  and  C_xi[t] = X_t @ gamma_xi are computed
        here via NumPy before entering nopython mode.
        """
        L          = self.lags
        eff        = self.max_lag
        scaling_i  = self._SCALING_INT.get(self.gas.scaling, 0)

        lags_gas  = np.array(list(L), dtype=np.int64)
        pi_lags_py = list(getattr(self.pi_dyn, "lags", None) or SEASONAL_LAGS[self.seasonal])
        lags_pi   = np.array(pi_lags_py, dtype=np.int64)
        pi_wy     = np.array([pi_params[f"omega_y_{l}"] for l in pi_lags_py],
                              dtype=np.float64)
        pi_omega0 = float(pi_params["omega0"])
        pi_rho    = float(pi_params.get("rho", 0.0))
        pi_link   = 1 if getattr(self.pi_dyn, "link", "logistic") == "reciprocal" else 0

        # Zeroed entirely under unconstrained_state -- the independent
        # isfinite() crash-guard in models/_nb_gb2_filter.py stays active
        # regardless, so this only removes the soft statistical-looking
        # penalty, not numerical safety.
        LAM_STATE   = 0.0 if self.unconstrained_state else 1e3
        LAM_ETA     = 1e2
        STATE_LIMIT = 15.0
        ETA_LIMIT   = 30.0

        y_f64     = np.asarray(y, dtype=np.float64)
        gamma_s   = float(gp["gamma"])
        zeta_s    = float(gp["zeta"])

        # Precompute covariate contributions: C_j[t] = X_t @ gamma_j
        zeros_T = np.zeros(T, dtype=np.float64)
        C_phi = (X_safe @ gamma["phi"].astype(np.float64)
                 if "phi" in self.cov_target else zeros_T)
        C_xi  = (X_safe @ gamma["xi"].astype(np.float64)
                 if "xi" in self.cov_target else zeros_T)

        n_tv = len(self.tv_names)

        if n_tv == 1:
            # phi-only distribution
            out = nb_filter_phi_only_cov(
                y_f64,
                f0=float(gp["f0_phi"]),
                omega=float(gp["omega_phi"]),
                A=np.array([gp[f"A_phi_{l}"] for l in L], dtype=np.float64),
                B=np.array([gp[f"B_phi_{l}"] for l in L], dtype=np.float64),
                lags_gas=lags_gas,
                xi_s=float(gp["xi"]), gamma_s=gamma_s, zeta_s=zeta_s,
                pi_omega0=pi_omega0, pi_rho=pi_rho,
                pi_wy=pi_wy, lags_pi=lags_pi, pi_link=pi_link,
                max_lag=eff, scaling=scaling_i,
                lam_state=LAM_STATE, state_lim=STATE_LIMIT,
                lam_eta=LAM_ETA,    eta_lim=ETA_LIMIT,
                C_phi=C_phi,
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
                "f_arr":     f1d[eff:T].reshape(n_eff, 1),
                "s_arr":     s1d[eff:T].reshape(n_eff, 1),
                "eta":       eta1d[:n_eff],
                "pi":        pi1d[:n_eff],
                "tv_names":  self.tv_names,
                "static":    {s: gp[s] for s in self.gas.static_names},
                "eff_start": eff,
                "loglik":    float(loglik_nb),
                "penalty":   float(total_pen),
                "objective": float(obj),
                "y_eff":     y[eff:T],
                "gamma":     {j: gamma.get(j, np.zeros(self.n_cov))
                              for j in self.tv_names},
            }

        else:
            # phi + xi distribution
            out = nb_filter_phi_xi_cov(
                y_f64,
                f0_phi=float(gp["f0_phi"]),
                f0_xi=float(gp["f0_xi"]),
                omega_phi=float(gp["omega_phi"]),
                omega_xi=float(gp["omega_xi"]),
                A_phi=np.array([gp[f"A_phi_{l}"] for l in L], dtype=np.float64),
                A_xi =np.array([gp[f"A_xi_{l}"]  for l in L], dtype=np.float64),
                B_phi=np.array([gp[f"B_phi_{l}"] for l in L], dtype=np.float64),
                B_xi =np.array([gp[f"B_xi_{l}"]  for l in L], dtype=np.float64),
                lags_gas=lags_gas,
                gamma_s=gamma_s, zeta_s=zeta_s,
                pi_omega0=pi_omega0, pi_rho=pi_rho,
                pi_wy=pi_wy, lags_pi=lags_pi, pi_link=pi_link,
                max_lag=eff, scaling=scaling_i,
                lam_state=LAM_STATE, state_lim=STATE_LIMIT,
                lam_eta=LAM_ETA,    eta_lim=ETA_LIMIT,
                C_phi=C_phi,
                C_xi=C_xi,
            )
            loglik_nb, loop_pen, f_phi, f_xi, s_phi, s_xi, eta1d, pi1d = out

            if loglik_nb == -1e12:
                return (1e12 + pre_penalty) if not return_paths else {}

            total_pen = pre_penalty + loop_pen
            obj       = -loglik_nb + total_pen

            if not return_paths:
                return obj

            n_eff    = T - eff
            f_arr_2d = np.column_stack([f_phi[eff:T], f_xi[eff:T]])
            s_arr_2d = np.column_stack([s_phi[eff:T], s_xi[eff:T]])
            return {
                "f_arr":     f_arr_2d,
                "s_arr":     s_arr_2d,
                "eta":       eta1d[:n_eff],
                "pi":        pi1d[:n_eff],
                "tv_names":  self.tv_names,
                "static":    {s: gp[s] for s in self.gas.static_names},
                "eff_start": eff,
                "loglik":    float(loglik_nb),
                "penalty":   float(total_pen),
                "objective": float(obj),
                "y_eff":     y[eff:T],
                "gamma":     {j: gamma.get(j, np.zeros(self.n_cov))
                              for j in self.tv_names},
            }

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
        pi_params = self._decode_pi(pi_theta)
        gp = self.gas.codec.decode(gas_theta)
        gamma = self._decode_cov(cov_theta)  # {j: array}

        L = self.lags
        n_tv = len(self.tv_names)
        X_safe = np.nan_to_num(X, nan=0.0)  # no leakage from missing

        # ── Penalties ────────────────────────────────────────────────────────
        penalty = 0.0
        LAM_PERSIST = 1e5
        LAM_SCORE   = 1e3
        LAM_STATE   = 0.0 if self.unconstrained_state else 1e3
        LAM_ETA     = 1e2
        LAM_STATIC  = 1e5
        LAM_COV     = 1e2

        gamma_v = gp.get("gamma")
        zeta_v  = gp.get("zeta")
        # See ZAGASModel's identical note: under free_zeta_gamma, gp["zeta"]
        # already satisfies exp(zeta) > kurtosis_margin*exp(gamma) for every
        # real delta by construction (models/gas_filter.py's _ParamCodec.
        # decode()), so this check would never fire -- skipped explicitly.
        if not self.gas.codec.free_zeta_gamma and gamma_v is not None and zeta_v is not None:
            # gamma_v >= zeta_v is a bare NUMERICAL-VALIDITY guard (E[Y]
            # infinite -- the GB2 density is not even proper there), kept
            # regardless of unconstrained_penalties -- this is the same
            # class of safeguard as the isfinite() checks in
            # models/_nb_gb2_filter.py, not a statistical "restriction" in
            # the sense unconstrained_penalties/bound_static_box remove.
            if gamma_v >= zeta_v:   # bp ≤ 1: E[Y] infinite
                return (1e12 + penalty) if not return_paths else {}
            # The soft "push away from the boundary" term below IS a
            # genuine statistical restriction (keeps zeta-gamma comfortably
            # inside the valid region, not just barely inside it) -- unlike
            # the hard reject above, this one IS dropped under
            # unconstrained_penalties, matching the |static|<20 guard just
            # below and the "static box" report's "no restrictions beyond
            # bare validity" design.
            bp_log = zeta_v - gamma_v
            if bp_log < 0.5 and not self.unconstrained_penalties:
                penalty += LAM_STATIC * (0.5 - bp_log) ** 2
        # Auxiliary numerical guard, not a real domain constraint -- dropped
        # entirely under unconstrained_penalties.
        if not self.unconstrained_penalties:
            for s in self.gas.static_names:
                excess = max(0.0, abs(gp[s]) - 20.0)
                penalty += LAM_STATIC * excess**2

        # Skipped when self.gas.codec.bound_ab / self.bound_rho is set: those
        # already bound A/B/rho by construction (independent per-lag
        # sigmoids), a deliberately looser alternative to these joint sum
        # penalties -- see reports/seasonality GAS/
        # gradient_and_convergence_diagnosis.md §8.
        if not self.gas.codec.bound_ab:
            for name in self.tv_names:
                b_sum = np.sum(np.abs([gp[f"B_{name}_{l}"] for l in L]))
                penalty += LAM_PERSIST * max(0.0, b_sum - 0.98) ** 2
                a_sum = np.sum(np.abs([gp[f"A_{name}_{l}"] for l in L]))
                penalty += LAM_SCORE * max(0.0, a_sum - 2.0) ** 2

        if not self.bound_rho:
            rho = pi_params.get("rho", 0.0)
            penalty += LAM_PERSIST * max(0.0, abs(rho) - 0.98) ** 2

        # Penalise very large covariate coefficients -- auxiliary numerical
        # guard, not a real domain constraint; dropped under
        # unconstrained_penalties.
        if not self.unconstrained_penalties:
            for j in self.cov_target:
                g_arr = gamma.get(j, np.zeros(self.n_cov))
                excess = max(0.0, np.sum(g_arr**2)**0.5 - 20.0)
                penalty += LAM_COV * excess**2

        # ── Numba fast path ──────────────────────────────────────────────────
        # All pre-loop penalties are now accumulated in `penalty`.  The Numba
        # function adds per-step state/eta penalties and returns them separately.
        if self._use_nb_filter:
            return self._nb_cov_dispatch(
                y, X_safe, gp, pi_params, gamma, penalty, T, return_paths
            )

        # ── Allocate arrays ───────────────────────────────────────────────────
        f_arr   = np.zeros((T + 1, n_tv))
        s_arr   = np.zeros((T,     n_tv))
        eta_arr = np.zeros(T)   # t-indexed: for return paths and log-lik
        eta_eff = np.zeros(T)   # i-indexed: AR history for compute_eta (i = t−max_lag)
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

            # Pi recursion — eta_eff is i-indexed so compute_eta's AR access
            # eta_eff[i-1] gives the immediately-preceding step (matches Numba).
            i = t - self.max_lag
            eta_t = self.pi_dyn.compute_eta(
                i=i,
                eta_hist=eta_eff,
                y_full=y,
                t=t,
                params=pi_params,
                seasonal=self.seasonal,
            )
            eta_eff[i]  = eta_t   # i-indexed history for next step's AR term
            eta_arr[t]  = eta_t   # t-indexed for return paths
            # Numerical-safety guard (see models/harvey_gas.py's identical
            # fix, 2026-09-11): a transient rho excursion during a BFGS
            # line-search probe can make the AR-in-eta recursion compound
            # geometrically over thousands of steps to a magnitude where
            # squaring it below overflows a Python float instead of merely
            # returning a large-but-finite penalty.
            if not np.isfinite(eta_t) or abs(eta_t) > 1e100:
                return (1e12 + penalty) if not return_paths else {}
            penalty += LAM_ETA * max(0.0, abs(eta_t) - 30.0) ** 2
            pi_arr[t] = float(self.pi_dyn.pi_from_eta(eta_t))

            log_pi, log_1mpi = self.pi_dyn.log_pi_terms(eta_t)

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

            # Score — use gas._scaled_score to support all 3 scaling modes
            if y[t] > 0:
                raw_score = self.dist.score(y[t], **call_params)
                scaled = self.gas._scaled_score(raw_score, call_params)
                for j, name in enumerate(self.tv_names):
                    s_arr[t, j] = scaled[name]
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
        if self.bound_rho and "rho" in self._pi_names:
            # pi_dyn's own natural-scale rho target (typically 0.5) must be
            # converted to the raw pre-sigmoid value _decode_pi will map
            # back to it -- otherwise the starting rho would silently land
            # somewhere else entirely.
            i = self._pi_names.index("rho")
            pi0 = pi0.copy()
            target = pi0[i] / self.rho_max
            pi0[i] = _logit_scalar((target + 1.0) / 2.0) if self.rho_signed else _logit_scalar(target)
        cov0 = np.zeros(self.n_cov_params)
        return np.concatenate([gas0, pi0, cov0])

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        verbose: bool = False,
        theta0: Optional[np.ndarray] = None,
        options: Optional[dict] = None,
        polish: bool = True,
    ) -> dict:
        """
        Estimate all parameters by unbounded BFGS maximum likelihood.

        Includes a polish step and convergence classification per
        OPTIMIZATION.md §11.

        Returns
        -------
        dict with keys:
            theta, loglik, validity, success, message,
            n_iter, n_fev, grad, grad_norm_inf, grad_norm_2,
            hess_inv, std_errors, se_quality,
            runtime_s, peak_mem_mb, polish_improvement, param_names
        """
        import tracemalloc

        if theta0 is None:
            theta0 = self.initial_theta(y)
        else:
            theta0 = np.asarray(theta0, dtype=float)

        opt_options = {"maxiter": 1000, "gtol": 1e-3, "disp": verbose}
        if options:
            opt_options.update(options)

        t_start = time.time()
        tracemalloc.start()

        def obj(theta):
            return self._run_filter(theta, y, X, return_paths=False)

        res = minimize(fun=obj, x0=theta0, method="BFGS", options=opt_options)

        # Polish step capped at 200 iterations
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
        grad          = getattr(res, "jac", None)
        grad_norm_inf = float(np.max(np.abs(grad))) if grad is not None else float("nan")
        grad_norm_2   = float(np.linalg.norm(grad)) if grad is not None else float("nan")

        # Convergence classification
        paths     = self._run_filter(res.x, y, X, return_paths=True)
        ll_final  = float(paths["loglik"]) if paths else float(-res.fun)
        finite_p  = bool(np.all(np.isfinite(res.x)))
        finite_ll = bool(np.isfinite(ll_final))
        states_ok = bool(paths) and bool(np.all(np.isfinite(paths.get("f_arr", [[0]]))))
        grad_ok   = (grad_norm_inf < 1e-2) if not np.isnan(grad_norm_inf) else False

        if res.success and finite_p and finite_ll and states_ok and grad_ok:
            validity = "valid_converged"
        elif finite_p and finite_ll and states_ok:
            validity = "valid_with_warning"
        else:
            validity = "failed"

        # Standard errors from inverse Hessian
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

        # Bound diagnostics
        B_LIMIT     = 0.98
        A_LIMIT     = 2.0
        RHO_LIMIT   = 0.98
        STATE_LIMIT = 15.0
        ETA_LIMIT   = 30.0
        STATIC_LIMIT = 20.0

        gas_th_fin, pi_th_fin, _ = self._split_theta(res.x)
        gp_fin    = self.gas.codec.decode(gas_th_fin)
        pi_fin    = self._decode_pi(pi_th_fin)

        bound_diags: dict = {}
        for name in self.tv_names:
            b_sum = float(np.sum(np.abs([gp_fin[f"B_{name}_{l}"] for l in self.lags])))
            a_sum = float(np.sum(np.abs([gp_fin[f"A_{name}_{l}"] for l in self.lags])))
            bound_diags[f"{name}_B_sum"]        = b_sum
            bound_diags[f"{name}_B_near_limit"] = bool(b_sum > 0.9 * B_LIMIT)
            bound_diags[f"{name}_A_sum"]        = a_sum
            bound_diags[f"{name}_A_near_limit"] = bool(a_sum > 0.9 * A_LIMIT)
        rho = float(pi_fin.get("rho", 0.0))
        bound_diags["pi_rho"]           = rho
        bound_diags["pi_rho_near_limit"] = bool(abs(rho) > 0.9 * RHO_LIMIT)
        for s in self.gas.static_names:
            v = float(gp_fin.get(s, 0.0))
            bound_diags[f"static_{s}"]            = v
            bound_diags[f"static_{s}_near_limit"] = bool(abs(v) > 0.9 * STATIC_LIMIT)
        if paths:
            f_max   = float(np.max(np.abs(paths.get("f_arr", np.zeros((1, 1))))))
            eta_max = float(np.max(np.abs(paths.get("eta",   np.zeros(1)))))
            bound_diags["max_state_magnitude"] = f_max
            bound_diags["state_near_limit"]    = bool(f_max   > 0.9 * STATE_LIMIT)
            bound_diags["max_eta_magnitude"]   = eta_max
            bound_diags["eta_near_limit"]      = bool(eta_max > 0.9 * ETA_LIMIT)
        bound_diags["scaling_fallback_count"] = int(self.gas._fallback_count)

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
        pi_params = self._decode_pi(pi_theta)
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
        eta_arr = np.zeros(T_full)          # t-indexed, for output / CDF use
        eta_eff = np.zeros(T_full)          # i-indexed, for compute_eta's AR history
        pi_arr  = np.zeros(T_full)

        for j, name in enumerate(self.tv_names):
            f_arr[: max_lag + 1, j] = gp[f"f0_{name}"]

        def _update(t, y_full, f_arr, s_arr, eta_arr, pi_arr):
            call_params = {
                name: f_arr[t, j] for j, name in enumerate(self.tv_names)
            }
            call_params.update({s: gp[s] for s in self.gas.static_names})

            # eta_eff must be i-indexed here, exactly as in _run_filter: compute_eta
            # reads eta_hist[i-1] for the AR term, so passing the t-indexed eta_arr
            # would fetch a value max_lag steps too early and break the recursion.
            i = t - max_lag
            eta_arr[t] = self.pi_dyn.compute_eta(
                i=i, eta_hist=eta_eff,
                y_full=y_full, t=t, params=pi_params, seasonal=self.seasonal,
            )
            eta_eff[i] = eta_arr[t]
            pi_arr[t] = float(self.pi_dyn.pi_from_eta(eta_arr[t]))

            if y_full[t] > 0:
                raw    = self.dist.score(y_full[t], **call_params)
                scaled = self.gas._scaled_score(raw, call_params)
                for j, name in enumerate(self.tv_names):
                    s_arr[t, j] = scaled[name]

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
            "model_type":         "CovZAGASModel",
            "n_params":           self.n_params,
            "cov_names":          self.cov_names,
            "cov_target":         self.cov_target,
            "tv_names":           list(self.tv_names),
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
            "bound_ab":           self.gas.codec.bound_ab,
            "a_max":              self.gas.codec.a_max,
            "b_max":              self.gas.codec.b_max,
            "bound_rho":          self.bound_rho,
            "rho_max":            self.rho_max,
            "ab_signed":          self.gas.codec.ab_signed,
            "rho_signed":         self.rho_signed,
            "free_zeta_gamma":    self.gas.codec.free_zeta_gamma,
            "kurtosis_margin":    self.gas.codec.kurtosis_margin,
            "unconstrained_penalties": self.unconstrained_penalties,
            "unconstrained_state": self.unconstrained_state,
            "bound_b_joint":      self.gas.codec.bound_b_joint,
            "b_total_max":        self.gas.codec.b_total_max,
            "bound_b_tanh_sum":   self.gas.codec.bound_b_tanh_sum,
            "b_sum_max":          self.gas.codec.b_sum_max,
            "bound_static_box":   self.gas.codec.bound_static_box,
            "static_box":         list(self.gas.codec.static_box),
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
        """Load theta from cached estimated_parameters.csv."""
        df = pd.read_csv(Path(cache_dir) / "estimated_parameters.csv")
        return df["value"].to_numpy(dtype=float)
