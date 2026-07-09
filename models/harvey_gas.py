"""
Harvey Long-Short ZA-GAS model.

=============================================================================
MATHEMATICAL SPECIFICATION  (MODELS.md §19–21)
=============================================================================

For each time-varying parameter j ∈ {φ, ξ} (or any configured subset):

    f_{j,t}   =  ω_j  +  L_{j,t}  +  S_{j,t}

The combined state is the sum of two score-driven components.

Long component (persistent climate-scale dynamics):

    L_{j,t+1}  =  B_{L,j} · L_{j,t}
               +  A_{L,j} · s_{j,t}
               +  Γ_{L,j}' · X^{long}_t

Short component (local atmospheric dynamics):

    S_{j,t+1}  =  B_{S,j} · S_{j,t}
               +  A_{S,j} · s_{j,t}
               +  Γ_{S,j}' · X^{short}_t

Persistence restriction (ensures L is more persistent than S):

    0  <  B_{S,j}  <  B_{L,j}  <  1

Both components receive the SAME scaled score s_{j,t} at each time step.
The score is computed from the current predictive distribution defined by f_{j,t}.

=============================================================================
THIS IS THE ONLY HARVEY IMPLEMENTATION
=============================================================================

f = ω + L + S with no separate GAS update on f itself; L and S are BOTH
score-driven recursions with their own B and A; the persistence restriction
is enforced by a penalty, not a hard bound (OPTIMIZATION.md §3).

This is the sole Harvey long-short model in the framework and the one
`pipeline.runner.build_stage3_specs` calls for every Stage-3 fit reported
in reports/multi_location/report.tex.

=============================================================================
EXECUTION FLOW
=============================================================================

fit()
↓
Initialize parameters (from prior model if available)
↓
For each t in effective sample:
    ↓
    Assemble f_{j,t} = ω_j + L_{j,t} + S_{j,t}
    ↓
    Evaluate predictive log-likelihood
    ↓
    Compute analytical score s_{j,t} from distribution
    ↓
    Apply score scaling (diagonal inverse Fisher, full inverse Fisher, or unit)
    ↓
    Update L_{j,t+1} = B_L L_{j,t} + A_L s_{j,t} + Γ_L' X_long_t
    ↓
    Update S_{j,t+1} = B_S S_{j,t} + A_S s_{j,t} + Γ_S' X_short_t
↓
Return -loglik + penalties
↓
Optimizer (BFGS, unbounded with penalties)
↓
Post-fit: save filtered states, metrics, artifacts

=============================================================================
PARAMETER VECTOR LAYOUT
=============================================================================

    theta = [ gas_static_theta | pi_theta | harvey_theta ]

where:
    gas_static_theta  :  static distribution params (gamma, zeta) + ω_j, f0_j
                         per GASFilter._ParamCodec (but WITHOUT A/B lags —
                         those are replaced by A_L, B_L, A_S, B_S below)
    pi_theta          :  occurrence model parameters
    harvey_theta      :  for each j in harvey_target (in tv_param_names order):
                             omega_j    — GAS intercept
                             f0_j       — initial state f_{j,0} (= L0 + S0 at t=0)
                             A_L_j      — long score response
                             B_L_j      — long persistence (< 1)
                             L0_j       — initial long state
                             gamma_L_j_0 … gamma_L_j_{n_long-1}  (X_long coefficients)
                             A_S_j      — short score response
                             B_S_j      — short persistence (< B_L_j)
                             S0_j       — initial short state
                             gamma_S_j_0 … gamma_S_j_{n_short-1} (X_short coefficients)

Parameters NOT in harvey_target get a standard constant (no score-driven dynamics).
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
from constants import SEASONAL_LAGS

# ─────────────────────────────────────────────────────────────────────────────
# Numba Harvey filter (optional; falls back to Python loop if unavailable)
# ─────────────────────────────────────────────────────────────────────────────

_NB_AVAIL_HARVEY = False
try:
    from models._nb_gb2_filter import (
        nb_harvey_phi_only_filter,
        nb_harvey_phi_xi_filter,
    )
    _NB_AVAIL_HARVEY = True
except ImportError:
    pass

_HARVEY_SCALING_INT = {"unit": 0, "diagonal_inverse_fisher": 1, "inverse_fisher": 2}

# ─────────────────────────────────────────────────────────────────────────────
# Per-parameter Harvey block size
#
#   omega_j (1) + f0_j (1) + A_L_j (1) + B_L_j (1) + L0_j (1)
#   + gamma_L (n_long) + A_S_j (1) + B_S_j (1) + S0_j (1) + gamma_S (n_short)
# ─────────────────────────────────────────────────────────────────────────────

def _block_size(n_long: int, n_short: int) -> int:
    """Number of Harvey parameters per dynamic parameter j."""
    return 2 + 3 + n_long + 3 + n_short  # omega + f0 + (A_L,B_L,L0,gL) + (A_S,B_S,S0,gS)


class HarveyZAGASModel:
    """
    Zero-Augmented GAS model with Harvey long-short decomposition.

    Parameters
    ----------
    distribution   : Distribution  (e.g. GB2LogLink or GB2LogLinkPhiOnly)
    pi_dynamics    : PiDynamics    (occurrence model, unchanged from base ZAGASModel)
    seasonal       : 'daily' — controls pi_dynamics lag set
    long_names     : covariate column labels for X_long  (ENSO features)
    short_names    : covariate column labels for X_short (dew point, temperature)
    harvey_target  : TV params that receive the Harvey decomposition.
                     None → all TV params.
    scaling        : score scaling mode ("unit", "diagonal_inverse_fisher",
                     "inverse_fisher")
    static_params  : static distribution params passed to GASFilter codec
    """

    def __init__(
        self,
        distribution: Distribution,
        pi_dynamics: PiDynamics,
        seasonal: str = "daily",
        long_names: Optional[List[str]] = None,
        short_names: Optional[List[str]] = None,
        harvey_target: Optional[List[str]] = None,
        scaling: str = "diagonal_inverse_fisher",
        static_params: Optional[List[str]] = None,
    ):
        from models.gas_filter import GASFilter, _VALID_SCALING

        if scaling not in _VALID_SCALING:
            raise ValueError(
                f"scaling must be one of {sorted(_VALID_SCALING)}, got '{scaling!r}'"
            )

        self.dist      = distribution
        self.pi_dyn    = pi_dynamics
        self.seasonal  = seasonal
        self.scaling   = scaling

        # pi_dynamics determines effective sample start
        self._pi_names = pi_dynamics.param_names(seasonal)
        self._pi_lags  = SEASONAL_LAGS[seasonal]
        self.max_lag   = max(self._pi_lags)

        self.tv_names     = list(distribution.tv_param_names)
        if static_params is not None:
            self.static_names = list(static_params)
        elif hasattr(distribution, "default_static_params"):
            self.static_names = list(distribution.default_static_params)
        else:
            self.static_names = ["gamma", "zeta"]

        self.long_names  = list(long_names)  if long_names  else []
        self.short_names = list(short_names) if short_names else []
        self.n_long  = len(self.long_names)
        self.n_short = len(self.short_names)

        # Restrict harvey_target to actual TV params
        if harvey_target is None:
            self.harvey_target = list(self.tv_names)
        else:
            self.harvey_target = [j for j in harvey_target if j in self.tv_names]

        self._blk = _block_size(self.n_long, self.n_short)

        # Scaling fallback counter (mirrors GASFilter._fallback_count)
        self._fallback_count: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Parameter layout helpers
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def n_static(self) -> int:
        return len(self.static_names)

    @property
    def n_pi(self) -> int:
        return len(self._pi_names)

    @property
    def n_harvey(self) -> int:
        return len(self.harvey_target) * self._blk

    @property
    def n_params(self) -> int:
        return self.n_static + self.n_pi + self.n_harvey

    def _split_theta(
        self, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split theta into (static_theta, pi_theta, harvey_theta)."""
        a = self.n_static
        b = a + self.n_pi
        return theta[:a], theta[a:b], theta[b:]

    def _decode_static(self, static_theta: np.ndarray) -> Dict[str, float]:
        return {name: float(static_theta[i]) for i, name in enumerate(self.static_names)}

    def _decode_pi(self, pi_theta: np.ndarray) -> Dict[str, float]:
        return {name: float(pi_theta[i]) for i, name in enumerate(self._pi_names)}

    def _decode_harvey(self, harvey_theta: np.ndarray) -> Dict[str, dict]:
        """
        Return {j: {omega, f0, A_L, B_L, L0, gamma_L, A_S, B_S, S0, gamma_S}}.

        Layout per j (in harvey_target order):
            [0]  omega_j
            [1]  f0_j
            [2]  A_L_j
            [3]  B_L_j
            [4]  L0_j
            [5..5+n_long-1]              gamma_L_j
            [5+n_long]                   A_S_j
            [5+n_long+1]                 B_S_j
            [5+n_long+2]                 S0_j
            [5+n_long+3..5+n_long+3+n_short-1]  gamma_S_j
        """
        result = {}
        blk = self._blk
        for k, j in enumerate(self.harvey_target):
            b = harvey_theta[k * blk : (k + 1) * blk]
            gl_end = 5 + self.n_long
            result[j] = {
                "omega":   float(b[0]),
                "f0":      float(b[1]),
                "A_L":     float(b[2]),
                "B_L":     float(b[3]),
                "L0":      float(b[4]),
                "gamma_L": b[5 : gl_end],
                "A_S":     float(b[gl_end]),
                "B_S":     float(b[gl_end + 1]),
                "S0":      float(b[gl_end + 2]),
                "gamma_S": b[gl_end + 3 : gl_end + 3 + self.n_short],
            }
        return result

    def parameter_names(self) -> List[str]:
        """Flat theta parameter names in optimizer order."""
        names = list(self.static_names) + list(self._pi_names)
        for j in self.harvey_target:
            names += [
                f"omega_{j}",
                f"f0_{j}",
                f"A_L_{j}",
                f"B_L_{j}",
                f"L0_{j}",
            ]
            names += [f"gamma_L_{j}_{c}" for c in self.long_names]
            names += [
                f"A_S_{j}",
                f"B_S_{j}",
                f"S0_{j}",
            ]
            names += [f"gamma_S_{j}_{c}" for c in self.short_names]
        return names

    # ─────────────────────────────────────────────────────────────────────────
    # Score scaling
    # ─────────────────────────────────────────────────────────────────────────

    def _scaled_score(
        self,
        raw_score: Dict[str, float],
        call_params: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply score scaling with automatic fallback.
        Mirrors gas_filter.GASFilter._scaled_score (same fallback logic):
            inverse_fisher → diagonal_inverse_fisher → unit
        Fallback events tracked in self._fallback_count.
        """
        unit_score = {n: raw_score[n] for n in self.tv_names}

        if self.scaling == "unit":
            return unit_score

        def _try_diagonal() -> Optional[Dict[str, float]]:
            try:
                fi = self.dist.fisher_info_diag(**call_params)
                result: Dict[str, float] = {}
                for name in self.tv_names:
                    fi_val = fi.get(name, 0.0)
                    if fi_val <= 0.0 or not np.isfinite(fi_val):
                        return None
                    s = raw_score[name] / fi_val
                    if not np.isfinite(s):
                        return None
                    result[name] = s
                return result
            except Exception:
                return None

        if self.scaling == "diagonal_inverse_fisher":
            d = _try_diagonal()
            if d is not None:
                return d
            self._fallback_count += 1
            return unit_score

        # inverse_fisher
        n_tv = len(self.tv_names)

        if n_tv == 1:
            d = _try_diagonal()
            if d is not None:
                return d
            self._fallback_count += 1
            return unit_score

        try:
            if hasattr(self.dist, "fisher_info_submatrix"):
                I_mat = self.dist.fisher_info_submatrix(self.tv_names, **call_params)
            else:
                fi = self.dist.fisher_info_diag(**call_params)
                I_mat = np.diag([fi.get(n, 0.0) for n in self.tv_names])

            if not np.all(np.isfinite(I_mat)):
                raise np.linalg.LinAlgError("Non-finite FI matrix")
            if np.any(np.diag(I_mat) <= 0.0):
                raise np.linalg.LinAlgError("Non-positive FI diagonal")

            grad_vec = np.array([raw_score[n] for n in self.tv_names])
            s_vec = np.linalg.solve(I_mat, grad_vec)

            if not np.all(np.isfinite(s_vec)):
                raise np.linalg.LinAlgError("Solve returned non-finite")

            return {n: float(s_vec[i]) for i, n in enumerate(self.tv_names)}

        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError, KeyError):
            d = _try_diagonal()
            if d is not None:
                self._fallback_count += 1
                return d
            self._fallback_count += 1
            return unit_score

    # ─────────────────────────────────────────────────────────────────────────
    # Numba dispatch
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def _use_nb_filter(self) -> bool:
        if not _NB_AVAIL_HARVEY:
            return False
        from pi_dynamics.ar_logistic import ARLogisticPiDynamics
        if not isinstance(self.pi_dyn, ARLogisticPiDynamics):
            return False
        from distributions.gb2_log_link import GB2LogLink
        from distributions.gb2_phi_only import GB2LogLinkPhiOnly
        if not isinstance(self.dist, (GB2LogLink, GB2LogLinkPhiOnly)):
            return False
        # Only support 1 or 2 Harvey targets matching tv_param_names order
        return len(self.harvey_target) in (1, 2)

    def _nb_harvey_dispatch(
        self,
        y: np.ndarray,
        X_long: np.ndarray,
        X_short: np.ndarray,
        sp: dict,
        pip: dict,
        hvd: dict,
        pre_penalty: float,
        T: int,
        return_paths: bool,
    ):
        pi_lags_py = SEASONAL_LAGS[self.seasonal]
        lags_pi    = np.array(pi_lags_py, dtype=np.int64)
        pi_wy      = np.array([pip[f"omega_y_{l}"] for l in pi_lags_py], dtype=np.float64)
        pi_omega0  = float(pip["omega0"])
        pi_rho     = float(pip["rho"])
        scaling_i  = _HARVEY_SCALING_INT.get(self.scaling, 0)

        Xl = np.nan_to_num(X_long,  nan=0.0).astype(np.float64)
        Xs = np.nan_to_num(X_short, nan=0.0).astype(np.float64)
        zeros_T = np.zeros(T, dtype=np.float64)

        eff   = self.max_lag
        n_tv  = len(self.tv_names)

        if len(self.harvey_target) == 1:
            # ── phi-only Harvey ──────────────────────────────────────────────
            j  = self.harvey_target[0]
            hj = hvd[j]
            CL = (Xl @ hj["gamma_L"].astype(np.float64)) if self.n_long  > 0 else zeros_T
            CS = (Xs @ hj["gamma_S"].astype(np.float64)) if self.n_short > 0 else zeros_T

            xi_s    = float(sp.get("xi",    0.0))
            gamma_s = float(sp["gamma"])
            zeta_s  = float(sp["zeta"])

            loglik_nb, loop_pen, L_arr, S_arr, s_arr, eta_nb, pi_nb = (
                nb_harvey_phi_only_filter(
                    y.astype(np.float64),
                    omega=float(hj["omega"]),
                    L0=float(hj["L0"]), A_L=float(hj["A_L"]), B_L=float(hj["B_L"]),
                    S0=float(hj["S0"]), A_S=float(hj["A_S"]), B_S=float(hj["B_S"]),
                    xi_s=xi_s, gamma_s=gamma_s, zeta_s=zeta_s,
                    pi_omega0=pi_omega0, pi_rho=pi_rho,
                    pi_wy=pi_wy, lags_pi=lags_pi,
                    max_lag=self.max_lag, scaling=scaling_i,
                    lam_state=1e3, state_lim=15.0,
                    lam_eta=1e2,   eta_lim=30.0,
                    CL=CL, CS=CS,
                )
            )
            total_pen = pre_penalty + loop_pen
            obj       = -loglik_nb + total_pen
            if loglik_nb == -1e12:
                return obj if not return_paths else {}
            if not return_paths:
                return obj

            j_idx    = self.tv_names.index(j)
            omega_j  = float(hj["omega"])
            n_eff    = T - eff
            f_out    = np.zeros((n_eff, n_tv))
            L_out    = np.zeros((n_eff, n_tv))
            S_out    = np.zeros((n_eff, n_tv))
            s_out    = np.zeros((n_eff, n_tv))
            for ii in range(n_eff):
                t = eff + ii
                f_out[ii, j_idx] = omega_j + L_arr[t] + S_arr[t]
                L_out[ii, j_idx] = L_arr[t]
                S_out[ii, j_idx] = S_arr[t]
                s_out[ii, j_idx] = s_arr[t]
            return {
                "f_arr": f_out, "L_arr": L_out, "S_arr": S_out, "s_arr": s_out,
                "eta": eta_nb, "pi": pi_nb,
                "tv_names": self.tv_names, "static": dict(sp),
                "eff_start": eff, "loglik": float(loglik_nb),
                "penalty": total_pen, "objective": obj,
                "y_eff": y[eff:T], "harvey": hvd,
            }

        else:
            # ── phi+xi Harvey ────────────────────────────────────────────────
            hphi = hvd["phi"]
            hxi  = hvd["xi"]
            CL_phi = (Xl @ hphi["gamma_L"].astype(np.float64)) if self.n_long  > 0 else zeros_T
            CS_phi = (Xs @ hphi["gamma_S"].astype(np.float64)) if self.n_short > 0 else zeros_T
            CL_xi  = (Xl @ hxi["gamma_L"].astype(np.float64))  if self.n_long  > 0 else zeros_T
            CS_xi  = (Xs @ hxi["gamma_S"].astype(np.float64))  if self.n_short > 0 else zeros_T
            gamma_s = float(sp["gamma"])
            zeta_s  = float(sp["zeta"])

            loglik_nb, loop_pen, L_phi, S_phi, L_xi, S_xi, s_phi, s_xi, eta_nb, pi_nb = (
                nb_harvey_phi_xi_filter(
                    y.astype(np.float64),
                    omega_phi=float(hphi["omega"]), omega_xi=float(hxi["omega"]),
                    L0_phi=float(hphi["L0"]), S0_phi=float(hphi["S0"]),
                    A_L_phi=float(hphi["A_L"]), B_L_phi=float(hphi["B_L"]),
                    A_S_phi=float(hphi["A_S"]), B_S_phi=float(hphi["B_S"]),
                    L0_xi=float(hxi["L0"]),  S0_xi=float(hxi["S0"]),
                    A_L_xi=float(hxi["A_L"]),  B_L_xi=float(hxi["B_L"]),
                    A_S_xi=float(hxi["A_S"]),  B_S_xi=float(hxi["B_S"]),
                    gamma_s=gamma_s, zeta_s=zeta_s,
                    pi_omega0=pi_omega0, pi_rho=pi_rho,
                    pi_wy=pi_wy, lags_pi=lags_pi,
                    max_lag=self.max_lag, scaling=scaling_i,
                    lam_state=1e3, state_lim=15.0,
                    lam_eta=1e2,   eta_lim=30.0,
                    CL_phi=CL_phi, CS_phi=CS_phi,
                    CL_xi=CL_xi,   CS_xi=CS_xi,
                )
            )
            total_pen = pre_penalty + loop_pen
            obj       = -loglik_nb + total_pen
            if loglik_nb == -1e12:
                return obj if not return_paths else {}
            if not return_paths:
                return obj

            phi_idx = self.tv_names.index("phi")
            xi_idx  = self.tv_names.index("xi")
            n_eff   = T - eff
            o_phi   = float(hphi["omega"])
            o_xi    = float(hxi["omega"])
            f_out   = np.zeros((n_eff, n_tv))
            L_out   = np.zeros((n_eff, n_tv))
            S_out   = np.zeros((n_eff, n_tv))
            s_out   = np.zeros((n_eff, n_tv))
            for ii in range(n_eff):
                t = eff + ii
                f_out[ii, phi_idx] = o_phi + L_phi[t] + S_phi[t]
                f_out[ii, xi_idx]  = o_xi  + L_xi[t]  + S_xi[t]
                L_out[ii, phi_idx] = L_phi[t];  S_out[ii, phi_idx] = S_phi[t]
                L_out[ii, xi_idx]  = L_xi[t];   S_out[ii, xi_idx]  = S_xi[t]
                s_out[ii, phi_idx] = s_phi[t]
                s_out[ii, xi_idx]  = s_xi[t]
            return {
                "f_arr": f_out, "L_arr": L_out, "S_arr": S_out, "s_arr": s_out,
                "eta": eta_nb, "pi": pi_nb,
                "tv_names": self.tv_names, "static": dict(sp),
                "eff_start": eff, "loglik": float(loglik_nb),
                "penalty": total_pen, "objective": obj,
                "y_eff": y[eff:T], "harvey": hvd,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Core filter
    # ─────────────────────────────────────────────────────────────────────────

    def _run_filter(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        X_long: np.ndarray,   # (T, n_long)
        X_short: np.ndarray,  # (T, n_short)
        return_paths: bool = False,
    ):
        """
        Harvey two-component recursion.

        At each t:
            f_{j,t}  =  omega_j + L_{j,t} + S_{j,t}
            ... evaluate likelihood, compute score s_{j,t} ...
            L_{j,t+1} = B_L_j * L_{j,t} + A_L_j * s_{j,t} + gamma_L_j' X_long_t
            S_{j,t+1} = B_S_j * S_{j,t} + A_S_j * s_{j,t} + gamma_S_j' X_short_t

        Stationarity penalties:
            - 0 < B_S < B_L < 1  for each j in harvey_target
            - |gamma_L|, |gamma_S| soft-bounded
        """
        T = len(y)
        if T <= self.max_lag:
            return 1e12 if not return_paths else {}

        st_theta, pi_theta, hv_theta = self._split_theta(theta)
        sp   = self._decode_static(st_theta)
        pip  = self._decode_pi(pi_theta)
        hvd  = self._decode_harvey(hv_theta)  # {j: {omega, A_L, B_L, ...}}

        n_tv  = len(self.tv_names)
        Xl    = np.nan_to_num(X_long,  nan=0.0)
        Xs    = np.nan_to_num(X_short, nan=0.0)

        # ── Penalties ────────────────────────────────────────────────────────
        penalty = 0.0
        LAM_LONG    = 1e5   # stationarity of L
        LAM_SHORT   = 1e5   # stationarity of S, and B_S < B_L
        LAM_COV     = 1e2   # covariate magnitude
        LAM_STATE   = 1e3   # state explosion
        LAM_ETA     = 1e2   # pi logit explosion
        LAM_STATIC  = 1e5   # static param validity
        LAM_SCORE   = 1e3   # score response amplitude

        # Static distribution constraints
        gamma_v = sp.get("gamma", 1.0)
        zeta_v  = sp.get("zeta",  3.0)
        penalty += LAM_STATIC * max(0.0, gamma_v - zeta_v + 1e-6) ** 2
        for s in self.static_names:
            penalty += LAM_STATIC * max(0.0, abs(sp[s]) - 20.0) ** 2

        # Pi persistence
        rho = pip.get("rho", 0.0)
        penalty += LAM_LONG * max(0.0, abs(rho) - 0.98) ** 2

        for j in self.harvey_target:
            hj = hvd[j]
            B_L, B_S = hj["B_L"], hj["B_S"]
            A_L, A_S = hj["A_L"], hj["A_S"]

            # B_L must be in (0, 1)
            penalty += LAM_LONG  * max(0.0, B_L - 0.995) ** 2
            penalty += LAM_LONG  * max(0.0, -B_L)        ** 2
            # B_S must be in (0, B_L)
            penalty += LAM_SHORT * max(0.0, B_S - B_L + 1e-4) ** 2
            penalty += LAM_SHORT * max(0.0, -B_S)              ** 2
            # Score amplitude soft bounds
            penalty += LAM_SCORE * max(0.0, abs(A_L) - 2.0) ** 2
            penalty += LAM_SCORE * max(0.0, abs(A_S) - 2.0) ** 2
            # Covariate magnitude
            penalty += LAM_COV * max(0.0, np.linalg.norm(hj["gamma_L"]) - 20.0) ** 2
            penalty += LAM_COV * max(0.0, np.linalg.norm(hj["gamma_S"]) - 20.0) ** 2

        # ── Numba dispatch (pre_penalty fully computed above) ─────────────────
        if self._use_nb_filter:
            return self._nb_harvey_dispatch(
                y, X_long, X_short, sp, pip, hvd, penalty, T, return_paths
            )

        # ── Allocate state arrays ─────────────────────────────────────────────
        L_arr   = np.zeros((T + 1, n_tv))   # long component
        S_arr   = np.zeros((T + 1, n_tv))   # short component
        f_arr   = np.zeros((T,     n_tv))   # combined state entering time t
        s_arr   = np.zeros((T,     n_tv))   # scaled score at t
        eta_arr = np.zeros(T)               # t-indexed  (for return paths)
        eta_eff = np.zeros(T)               # i-indexed  (AR history for pi recursion)
        pi_arr  = np.zeros(T)

        # Initialize L_0 and S_0
        tv_idx = {name: j for j, name in enumerate(self.tv_names)}
        for j, name in enumerate(self.tv_names):
            if name in hvd:
                L_arr[0, j] = hvd[name]["L0"]
                S_arr[0, j] = hvd[name]["S0"]

        loglik = 0.0

        # ── Main recursion ─────────────────────────────────────────────────────
        for t in range(self.max_lag, T):

            # Combine long + short into current state
            for j, name in enumerate(self.tv_names):
                if name in hvd:
                    f_arr[t, j] = hvd[name]["omega"] + L_arr[t, j] + S_arr[t, j]
                else:
                    # Param not in harvey_target: use static f0 (no dynamics)
                    f_arr[t, j] = sp.get(name, 0.0)

            call_params = {name: f_arr[t, j] for j, name in enumerate(self.tv_names)}
            call_params.update(sp)

            # State magnitude soft bound
            for j in range(n_tv):
                penalty += LAM_STATE * max(0.0, abs(f_arr[t, j]) - 15.0) ** 2

            # Pi recursion (eta_eff is i-indexed to avoid t-vs-i indexing bug)
            i_eff = t - self.max_lag
            eta_t = self.pi_dyn.compute_eta(
                i=i_eff,
                eta_hist=eta_eff,
                y_full=y,
                t=t,
                params=pip,
                seasonal=self.seasonal,
            )
            eta_eff[i_eff] = eta_t
            eta_arr[t]     = eta_t
            penalty += LAM_ETA * max(0.0, abs(eta_t) - 30.0) ** 2
            pi_arr[t] = float(expit(eta_t))

            # Log-likelihood
            if y[t] == 0:
                ll_t = log_expit(-eta_t)
            else:
                ll_dist = self.dist.logpdf(y[t], **call_params)
                if not np.isfinite(ll_dist):
                    return (1e12 + penalty) if not return_paths else {}
                ll_t = log_expit(eta_t) + ll_dist
            if not np.isfinite(ll_t):
                return (1e12 + penalty) if not return_paths else {}
            loglik += ll_t

            # Score
            if y[t] > 0:
                raw_score = self.dist.score(y[t], **call_params)
                scaled    = self._scaled_score(raw_score, call_params)
                for j, name in enumerate(self.tv_names):
                    s_arr[t, j] = scaled[name]
                if not np.all(np.isfinite(s_arr[t, :])):
                    return (1e12 + penalty) if not return_paths else {}

            # Component updates
            xl_t = Xl[t]
            xs_t = Xs[t]
            for j, name in enumerate(self.tv_names):
                s_jt = s_arr[t, j]
                if name in hvd:
                    hj = hvd[name]
                    L_next = hj["B_L"] * L_arr[t, j] + hj["A_L"] * s_jt
                    if self.n_long > 0:
                        L_next += float(np.dot(hj["gamma_L"], xl_t))
                    S_next = hj["B_S"] * S_arr[t, j] + hj["A_S"] * s_jt
                    if self.n_short > 0:
                        S_next += float(np.dot(hj["gamma_S"], xs_t))
                    if not (np.isfinite(L_next) and np.isfinite(S_next)):
                        return (1e12 + penalty) if not return_paths else {}
                    L_arr[t + 1, j] = L_next
                    S_arr[t + 1, j] = S_next
                else:
                    L_arr[t + 1, j] = L_arr[t, j]
                    S_arr[t + 1, j] = S_arr[t, j]

        objective = -loglik + penalty

        if not return_paths:
            return objective

        eff = self.max_lag
        return {
            "f_arr":      f_arr[eff:T, :],
            "L_arr":      L_arr[eff:T, :],
            "S_arr":      S_arr[eff:T, :],
            "s_arr":      s_arr[eff:T, :],
            "eta":        eta_arr[eff:T],
            "pi":         pi_arr[eff:T],
            "tv_names":   self.tv_names,
            "static":     dict(sp),
            "eff_start":  eff,
            "loglik":     loglik,
            "penalty":    penalty,
            "objective":  objective,
            "y_eff":      y[eff:T],
            "harvey":     hvd,
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
        sp   = paths["static"]
        for i in range(n):
            t    = eff + i
            pi_t = paths["pi"][i]
            call = {
                name: paths["f_arr"][i, j]
                for j, name in enumerate(paths["tv_names"])
            }
            call.update(sp)
            if y[t] <= 0:
                cdfs[i] = 1.0 - pi_t
            else:
                cdfs[i] = (1.0 - pi_t) + pi_t * self.dist.cdf(y[t], **call)
        return cdfs

    def initial_theta(
        self,
        y: np.ndarray,
        warm_start: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Construct starting values for the Harvey model.

        warm_start (optional) should be the filter paths dict from the best
        Stage-2 covariate model, used to initialize the short component.
        """
        y_pos   = y[y > 0]
        f0_phi  = float(np.log(np.mean(y_pos) + 1e-3)) if len(y_pos) > 0 else 1.0
        f0_xi   = float(np.log(1.2))
        f0_guess = {"phi": f0_phi, "xi": f0_xi}

        # Static params
        st0 = np.zeros(self.n_static)
        for i, name in enumerate(self.static_names):
            if name == "gamma": st0[i] = 1.0
            if name == "zeta":  st0[i] = 3.0

        # Pi params
        pi0 = self.pi_dyn.initial_params(y, self.seasonal)

        # Harvey params: start B_L=0.95, B_S=0.5, A_L=0.01, A_S=0.05
        hv0 = np.zeros(self.n_harvey)
        blk = self._blk
        for k, j in enumerate(self.harvey_target):
            base = k * blk
            f0   = f0_guess.get(j, 0.0)
            hv0[base + 0] = f0 * 0.05   # omega
            hv0[base + 1] = f0           # f0 (initial combined state)
            hv0[base + 2] = 0.01         # A_L
            hv0[base + 3] = 0.95         # B_L
            hv0[base + 4] = 0.0          # L0
            # gamma_L: zeros (n_long entries, positions 5..5+n_long-1)
            gl_end = 5 + self.n_long
            hv0[base + gl_end]     = 0.05  # A_S
            hv0[base + gl_end + 1] = 0.50  # B_S  (< B_L = 0.95)
            hv0[base + gl_end + 2] = 0.0   # S0
            # gamma_S: zeros (n_short entries)

        return np.concatenate([st0, pi0, hv0])

    def fit(
        self,
        y: np.ndarray,
        X_long: np.ndarray,
        X_short: np.ndarray,
        verbose: bool = False,
        theta0: Optional[np.ndarray] = None,
        options: Optional[dict] = None,
        polish: bool = True,
    ) -> dict:
        """
        Estimate all parameters by BFGS maximum likelihood.

        Parameters
        ----------
        y        : training observations (T,)
        X_long   : long-component covariate matrix (T, n_long), standardised
        X_short  : short-component covariate matrix (T, n_short), standardised
        theta0   : optional starting values (use initial_theta() if None)
        options  : overrides for scipy minimize options
        polish   : if True, restart BFGS from final params as a stability check
        """
        import time, tracemalloc
        t_start = time.time()
        tracemalloc.start()

        if theta0 is None:
            theta0 = self.initial_theta(y)
        theta0 = np.asarray(theta0, dtype=float)

        opt_options = {"maxiter": 1000, "gtol": 1e-3, "disp": verbose}
        if options:
            opt_options.update(options)

        def obj(theta):
            return self._run_filter(theta, y, X_long, X_short, return_paths=False)

        result = minimize(fun=obj, x0=theta0, method="BFGS", options=opt_options)

        # Polish step capped at 200 iterations
        polish_result = None
        polish_improvement = 0.0
        if polish:
            polish_opts = {**opt_options, "maxiter": 200}
            result2 = minimize(fun=obj, x0=result.x, method="BFGS", options=polish_opts)
            polish_improvement = abs(result.fun - result2.fun) / (1.0 + abs(result.fun))
            if result2.fun < result.fun:
                polish_result = result2
                result = result2

        # Runtime and memory
        runtime_s = time.time() - t_start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Gradient norm
        grad = getattr(result, "jac", None)
        grad_norm_inf = float(np.max(np.abs(grad))) if grad is not None else np.nan
        grad_norm_2   = float(np.linalg.norm(grad)) if grad is not None else np.nan

        # Convergence classification per OPTIMIZATION.md §11
        paths = self.filter(result.x, y, X_long, X_short)
        ll_final = float(paths.get("loglik", -result.fun)) if paths else -result.fun
        is_finite_params   = bool(np.all(np.isfinite(result.x)))
        is_finite_ll       = bool(np.isfinite(ll_final))
        is_states_ok       = bool(paths) and bool(np.all(np.isfinite(paths.get("f_arr", [[0]]))))
        grad_ok            = (grad_norm_inf < 1e-2) if not np.isnan(grad_norm_inf) else False

        if result.success and is_finite_params and is_finite_ll and is_states_ok and grad_ok:
            validity = "valid_converged"
        elif is_finite_params and is_finite_ll and is_states_ok:
            validity = "valid_with_warning"
        else:
            validity = "failed"

        # BFGS inverse Hessian → approximate standard errors
        hess_inv = np.array(result.hess_inv) if hasattr(result, "hess_inv") else None
        std_errors = None
        se_quality = "unavailable"
        if hess_inv is not None and hess_inv.ndim == 2:
            diag_h = np.diag(hess_inv)
            if np.all(diag_h > 0) and np.all(np.isfinite(diag_h)):
                std_errors = np.sqrt(diag_h)
                se_quality = "approximate"
            else:
                se_quality = "unreliable"

        return {
            "theta":              result.x,
            "loglik":             ll_final,
            "objective":          float(result.fun),
            "penalty":            float(paths.get("penalty", 0.0)) if paths else np.nan,
            "success":            bool(result.success),
            "message":            str(result.message),
            "n_iter":             int(result.nit),
            "n_fev":              int(result.nfev),
            "grad":               grad,
            "grad_norm_inf":      grad_norm_inf,
            "grad_norm_2":        grad_norm_2,
            "hess_inv":           hess_inv,
            "std_errors":         std_errors,
            "se_quality":         se_quality,
            "validity":           validity,
            "runtime_s":          runtime_s,
            "peak_mem_mb":        peak_mem / 1e6,
            "polish_improvement": polish_improvement,
            "param_names":        self.parameter_names(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # OOS rolling forecast
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
        """1-step-ahead rolling OOS evaluation by concatenating train + test."""
        st_theta, pi_theta, hv_theta = self._split_theta(theta)
        sp  = self._decode_static(st_theta)
        pip = self._decode_pi(pi_theta)
        hvd = self._decode_harvey(hv_theta)

        y_full   = np.concatenate([y_train, y_test])
        Xl_full  = np.vstack([X_long_train,  X_long_test])
        Xs_full  = np.vstack([X_short_train, X_short_test])
        Xl_full  = np.nan_to_num(Xl_full, nan=0.0)
        Xs_full  = np.nan_to_num(Xs_full, nan=0.0)

        T_train = len(y_train)
        T_full  = len(y_full)
        n_tv    = len(self.tv_names)
        max_lag = self.max_lag

        L_arr   = np.zeros((T_full + 1, n_tv))
        S_arr   = np.zeros((T_full + 1, n_tv))
        f_arr   = np.zeros((T_full,     n_tv))
        s_arr   = np.zeros((T_full,     n_tv))
        eta_arr = np.zeros(T_full)
        pi_arr  = np.zeros(T_full)

        for j, name in enumerate(self.tv_names):
            if name in hvd:
                L_arr[0, j] = hvd[name]["L0"]
                S_arr[0, j] = hvd[name]["S0"]

        for t in range(max_lag, T_full):
            for j, name in enumerate(self.tv_names):
                if name in hvd:
                    f_arr[t, j] = hvd[name]["omega"] + L_arr[t, j] + S_arr[t, j]
                else:
                    f_arr[t, j] = sp.get(name, 0.0)

            call_params = {name: f_arr[t, j] for j, name in enumerate(self.tv_names)}
            call_params.update(sp)

            eta_arr[t] = self.pi_dyn.compute_eta(
                i=t - max_lag, eta_hist=eta_arr,
                y_full=y_full, t=t, params=pip, seasonal=self.seasonal,
            )
            pi_arr[t] = float(expit(eta_arr[t]))

            if y_full[t] > 0:
                raw_score = self.dist.score(y_full[t], **call_params)
                scaled    = self._scaled_score(raw_score, call_params)
                for j, name in enumerate(self.tv_names):
                    s_arr[t, j] = scaled[name]

            xl_t = Xl_full[t]
            xs_t = Xs_full[t]
            for j, name in enumerate(self.tv_names):
                s_jt = s_arr[t, j]
                if name in hvd:
                    hj = hvd[name]
                    L_arr[t + 1, j] = (
                        hj["B_L"] * L_arr[t, j] + hj["A_L"] * s_jt
                        + (float(np.dot(hj["gamma_L"], xl_t)) if self.n_long  > 0 else 0.0)
                    )
                    S_arr[t + 1, j] = (
                        hj["B_S"] * S_arr[t, j] + hj["A_S"] * s_jt
                        + (float(np.dot(hj["gamma_S"], xs_t)) if self.n_short > 0 else 0.0)
                    )

        oos = slice(T_train, T_full)
        return {
            "f_arr_oos":  f_arr[oos, :],
            "L_arr_oos":  L_arr[oos, :],
            "S_arr_oos":  S_arr[oos, :],
            "pi_oos":     pi_arr[oos],
            "eta_oos":    eta_arr[oos],
            "static":     dict(sp),
            "tv_names":   self.tv_names,
            "y_test":     y_test,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Artifact helpers
    # ─────────────────────────────────────────────────────────────────────────

    def save_result(self, result: dict, cache_dir: Path) -> None:
        """Save full optimization result and model metadata."""
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
            "model_type":       "HarveyZAGASModel",
            "n_params":         self.n_params,
            "harvey_target":    self.harvey_target,
            "tv_names":         self.tv_names,
            "long_names":       self.long_names,
            "short_names":      self.short_names,
            "scaling":          self.scaling,
            "seasonal":         self.seasonal,
            "loglik":           float(result["loglik"]),
            "objective":        float(result["objective"]),
            "penalty":          float(result.get("penalty", 0.0)),
            "success":          bool(result["success"]),
            "validity":         result["validity"],
            "message":          str(result["message"]),
            "n_iter":           int(result["n_iter"]),
            "n_fev":            int(result["n_fev"]),
            "grad_norm_inf":    float(result["grad_norm_inf"]),
            "grad_norm_2":      float(result["grad_norm_2"]),
            "runtime_s":        float(result["runtime_s"]),
            "peak_mem_mb":      float(result["peak_mem_mb"]),
            "se_quality":       result["se_quality"],
            "polish_improvement": float(result["polish_improvement"]),
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
