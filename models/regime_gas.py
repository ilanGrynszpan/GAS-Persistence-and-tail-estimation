"""
Regime-Sensitive GAS Model — MODELS.md §22–23.

=============================================================================
THEORY
=============================================================================

GAS(1,1) update for TV parameter j with regime extension:

    f_{j,t+1|t} = ω_j + B_j · f_{j,t|t-1}
                + (A_j + A_ext_j · R_t(c)) · s_{j,t}   [if j ∈ regime_tv_names]
                or
                + A_j · s_{j,t}                          [if j ∉ regime_tv_names]

where

    R_t(c) = 𝟙(y_t > c)

and c is a fixed high threshold computed from training wet-day observations.

The regime_tv_names parameter controls which TV parameters receive the
regime extension A_ext_j · R_t(c).  Three variants are estimated:
    - phi_only   : regime_tv_names=["phi"]  — only scale gets regime term
    - xi_only    : regime_tv_names=["xi"]   — only shape gets regime term
    - phi_and_xi : regime_tv_names=["phi","xi"] — both get regime term

The non-regime parameter still has full GAS(1,1) dynamics (it is not frozen).

The model is always estimated as GAS(1,1) (gas_lags=[1]) with the phi+xi
distribution (both parameters time-varying) regardless of the base winner.
Three threshold quantiles are tested: 0.90, 0.95, 0.98.  Total: 9 models.

=============================================================================
PARAMETER VECTOR
=============================================================================

The flat theta vector extends the base ZAGASModel:

    [  gas_theta  |  pi_theta  |  A_ext_phi  (|  A_ext_xi if phi+xi)  ]

A_ext_j initialises to 0 (no regime effect) and is penalised the same as
A_j_l (soft bound |A_ext_j| ≤ 2.0).

=============================================================================
NUMBA
=============================================================================

The numba-compiled filter does NOT implement the regime term.
RegimeZAGASModel always runs the pure-Python loop — this is ~50× slower
but correct for the small regime-model stage (2–4 models per location).

=============================================================================
INHERITANCE
=============================================================================

RegimeZAGASModel subclasses ZAGASModel.  All save/load, cdf_series,
initial_theta (GAS+pi block), simulate_oos, and save_result methods are
reused.  Only _run_filter and the parameter codec are extended.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from models.za_gas_model import ZAGASModel
from distributions.base import Distribution
from pi_dynamics.base import PiDynamics


class RegimeZAGASModel(ZAGASModel):
    """
    Zero-Augmented GAS(1,1) model with regime-sensitive score response.

    Extends ZAGASModel.  For each TV parameter j listed in regime_tv_names,
    the lag-1 score coefficient becomes (A_j + A_ext_j * R_t(c)) when
    y_t > c.  TV parameters NOT in regime_tv_names retain a plain GAS(1,1)
    update (no regime term, still time-varying).

    Parameters
    ----------
    distribution       : GB2 distribution (use GB2LogLink for phi+xi)
    pi_dynamics        : occurrence probability dynamics
    seasonal           : "daily" (365-day harmonics for pi)
    gas_lags           : use [1] for the standard GAS(1,1) structure
    scaling            : score scaling ("diagonal_inverse_fisher" recommended)
    threshold_quantile : wet-day quantile for the regime threshold c
    threshold_c        : if provided, skip quantile computation (testing only)
    regime_tv_names    : TV parameters that get the A_ext regime term;
                         defaults to all TV parameters if None
    """

    def __init__(
        self,
        distribution: Distribution,
        pi_dynamics:  PiDynamics,
        seasonal:     str = "daily",
        gas_lags:     List[int] | None = None,
        scaling:      str = "diagonal_inverse_fisher",
        threshold_quantile: float = 0.95,
        threshold_c:  Optional[float] = None,
        regime_tv_names: Optional[List[str]] = None,
    ):
        super().__init__(
            distribution=distribution,
            pi_dynamics=pi_dynamics,
            seasonal=seasonal,
            gas_lags=gas_lags,
            scaling=scaling,
        )
        self.threshold_quantile = threshold_quantile
        self._threshold_c = threshold_c   # None until fit() sets it
        # Must be set AFTER super().__init__() because gas.tv_names is populated there.
        # If None, defaults to all TV parameters (regime applied to all).
        self.regime_tv_names: List[str] = (
            regime_tv_names if regime_tv_names is not None else list(self.gas.tv_names)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Regime threshold
    # ──────────────────────────────────────────────────────────────────────────

    def compute_threshold(self, y: np.ndarray) -> float:
        """Compute the 95th quantile of training wet-day observations."""
        wet = y[y > 0]
        if len(wet) == 0:
            return 1.0
        return float(np.quantile(wet, self.threshold_quantile))

    @property
    def threshold_c(self) -> float:
        if self._threshold_c is None:
            raise RuntimeError("threshold_c not set — call fit() first.")
        return self._threshold_c

    # ──────────────────────────────────────────────────────────────────────────
    # Parameter vector extension
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def n_ext(self) -> int:
        """Number of regime-extension parameters (one per regime TV parameter)."""
        return len(self.regime_tv_names)

    @property
    def n_params(self) -> int:  # type: ignore[override]
        return super().n_params + self.n_ext

    @property
    def n_gas(self) -> int:
        return self.gas.codec.n_params

    @property
    def n_pi(self) -> int:
        return len(self._pi_names)

    def parameter_names(self) -> List[str]:  # type: ignore[override]
        base = super().parameter_names()
        ext  = [f"A_ext_{name}" for name in self.regime_tv_names]
        return base + ext

    def _split_theta_regime(self, theta: np.ndarray):
        """Return (gas_theta, pi_theta, A_ext_arr)."""
        n_gas = self.n_gas
        n_pi  = self.n_pi
        gas_theta = theta[:n_gas]
        pi_theta  = theta[n_gas: n_gas + n_pi]
        A_ext     = theta[n_gas + n_pi:]   # length n_ext
        return gas_theta, pi_theta, A_ext

    def initial_theta(self, y: np.ndarray) -> np.ndarray:  # type: ignore[override]
        """Base initial theta + zeros for A_ext parameters."""
        base = super().initial_theta(y)
        return np.concatenate([base, np.zeros(self.n_ext)])

    # ──────────────────────────────────────────────────────────────────────────
    # Regime-modified filter  (Python only — numba path not used)
    # ──────────────────────────────────────────────────────────────────────────

    def _use_nb_filter(self) -> bool:  # type: ignore[override]
        """Always use the Python loop — numba does not support the regime term."""
        return False

    def _run_filter(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        return_paths: bool = False,
    ):
        """
        Run the regime-sensitive GAS + pi recursion.

        Identical to ZAGASModel._run_filter except the lag-1 score update
        is modified by the regime indicator R_{t-1}(c).
        """
        T = len(y)
        if T <= self.max_lag:
            return 1e12 if not return_paths else {}

        gas_theta, pi_theta, A_ext = self._split_theta_regime(theta)
        pi_params = self._decode_pi(pi_theta)
        gp        = self.gas.codec.decode(gas_theta)

        L    = self.lags
        n_tv = len(self.gas.tv_names)
        c    = self._threshold_c if self._threshold_c is not None else 0.0

        # ── Penalty hypers (same as ZAGASModel) ───────────────────────────
        penalty     = 0.0
        LAM_PERSIST = 1e5
        LAM_SCORE   = 1e3
        LAM_STATE   = 1e3
        LAM_ETA     = 1e2
        LAM_STATIC  = 1e5
        B_LIMIT     = 0.98
        RHO_LIMIT   = 0.98
        A_LIMIT     = 2.0
        STATE_LIMIT = 15.0
        ETA_LIMIT   = 30.0

        # ── Static distribution checks ─────────────────────────────────────
        gamma_v = gp.get("gamma")
        zeta_v  = gp.get("zeta")
        if gamma_v is not None and zeta_v is not None:
            if gamma_v >= zeta_v:
                return (1e12 + penalty) if not return_paths else {}
            bp_log = zeta_v - gamma_v
            if bp_log < 0.5:
                penalty += LAM_STATIC * (0.5 - bp_log) ** 2

        for s in self.gas.static_names:
            excess = max(0.0, abs(gp[s]) - 20.0)
            penalty += LAM_STATIC * excess**2

        # ── GAS persistence penalties ─────────────────────────────────────
        for name in self.gas.tv_names:
            b_vals  = np.array([gp[f"B_{name}_{l}"] for l in L])
            excess_b = max(0.0, np.sum(np.abs(b_vals)) - B_LIMIT)
            penalty += LAM_PERSIST * excess_b**2

            a_vals  = np.array([gp[f"A_{name}_{l}"] for l in L])
            excess_a = max(0.0, np.sum(np.abs(a_vals)) - A_LIMIT)
            penalty += LAM_SCORE * excess_a**2

        # Penalise A_ext_j same as A_j_l
        for a_e in A_ext:
            excess = max(0.0, abs(a_e) - A_LIMIT)
            penalty += LAM_SCORE * excess**2

        # ── Pi persistence penalty ─────────────────────────────────────────
        rho = pi_params.get("rho", 0.0)
        excess_rho = max(0.0, abs(rho) - RHO_LIMIT)
        penalty += LAM_PERSIST * excess_rho**2

        # ── Allocate arrays ────────────────────────────────────────────────
        f_arr  = np.zeros((T + 1, n_tv))
        s_arr  = np.zeros((T, n_tv))
        eta_arr = np.zeros(T)
        eta_eff = np.zeros(T)
        pi_arr  = np.zeros(T)

        for j, name in enumerate(self.gas.tv_names):
            f_arr[: self.max_lag + 1, j] = gp[f"f0_{name}"]

        loglik = 0.0

        # Pre-compute regime index map once (not per timestep)
        regime_ext_idx = {nm: idx for idx, nm in enumerate(self.regime_tv_names)}

        # ── Main recursion ─────────────────────────────────────────────────
        for t in range(self.max_lag, T):

            call_params = {
                name: f_arr[t, j] for j, name in enumerate(self.gas.tv_names)
            }
            call_params.update({s: gp[s] for s in self.gas.static_names})

            # State penalties
            for j in range(n_tv):
                excess = max(0.0, abs(f_arr[t, j]) - STATE_LIMIT)
                penalty += LAM_STATE * excess**2

            # Pi recursion
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

            eta_excess = min(max(0.0, abs(eta_t) - ETA_LIMIT), 1e8)  # cap avoids overflow
            penalty += LAM_ETA * eta_excess**2

            pi_arr[t] = float(expit(eta_t))
            log_pi_t          = log_expit(eta_arr[t])
            log_one_minus_pi_t = log_expit(-eta_arr[t])

            # Likelihood
            if y[t] == 0:
                ll_t = log_one_minus_pi_t
            else:
                ll_dist = self.dist.logpdf(y[t], **call_params)
                if not np.isfinite(ll_dist):
                    return (1e12 + penalty) if not return_paths else {}
                ll_t = log_pi_t + ll_dist

            if not np.isfinite(ll_t):
                return (1e12 + penalty) if not return_paths else {}

            loglik += ll_t

            # Score
            if y[t] > 0:
                raw_score = self.dist.score(y[t], **call_params)
                scaled    = self.gas._scaled_score(raw_score, call_params)
                for j, name in enumerate(self.gas.tv_names):
                    s_arr[t, j] = scaled[name]
                if not np.all(np.isfinite(s_arr[t, :])):
                    return (1e12 + penalty) if not return_paths else {}

            # Regime indicator for previous observation
            # At step t we update f_{t+1}; R_t(c) = 1(y_t > c)
            regime_t = float(y[t] > c) if t >= self.max_lag else 0.0

            # Regime-sensitive GAS update — applies regime only to regime_tv_names
            for j, name in enumerate(self.gas.tv_names):
                omega_j = gp[f"omega_{name}"]
                ar_part    = 0.0
                score_part = 0.0

                first_lag = L[0]   # the smallest lag, i.e. lag-1 in L

                for l in L:
                    idx    = t - l + 1
                    s_past = s_arr[idx, j] if idx >= 0 else 0.0
                    f_past = f_arr[idx, j] if idx >= 0 else gp[f"f0_{name}"]

                    A_l = gp[f"A_{name}_{l}"]
                    if l == first_lag and name in regime_ext_idx:
                        # Apply regime-sensitive extra score response
                        A_l = A_l + A_ext[regime_ext_idx[name]] * regime_t

                    score_part += A_l * s_past
                    ar_part    += gp[f"B_{name}_{l}"] * f_past

                f_next = omega_j + score_part + ar_part

                if not np.isfinite(f_next):
                    return (1e12 + penalty) if not return_paths else {}

                f_arr[t + 1, j] = f_next

        objective = -loglik + penalty

        if not return_paths:
            return objective

        eff = self.max_lag
        n_eff = T - eff

        return {
            "phi":        f_arr[eff:T, 0],
            "xi":         f_arr[eff:T, 1] if n_tv > 1 else None,
            "f_arr":      f_arr[eff:T, :],
            "s_arr":      s_arr[eff:T, :],
            "eta":        eta_arr[eff:T],
            "pi":         pi_arr[eff:T],
            "tv_names":   self.gas.tv_names,
            "static":     {s: gp[s] for s in self.gas.static_names},
            "eff_start":  eff,
            "loglik":     loglik,
            "penalty":    penalty,
            "objective":  objective,
            "y_eff":      y[eff:T],
            "threshold_c": c,
            "A_ext":      A_ext.tolist(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # fit() override — sets threshold before optimisation
    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        y: np.ndarray,
        verbose: bool = False,
        theta0: np.ndarray | None = None,
        options: dict | None = None,
        polish: bool = True,
    ) -> dict:
        """
        Compute threshold from training data, then estimate all parameters
        by unbounded BFGS (identical protocol to ZAGASModel.fit).

        Parameters
        ----------
        y      : training observations
        theta0 : optional starting vector (length n_params);
                 if provided from a base ZAGASModel, the caller should
                 append zeros for A_ext before passing.
        """
        # Set threshold before filter calls
        self._threshold_c = self.compute_threshold(y)
        if verbose:
            print(f"  Regime threshold c = {self._threshold_c:.3f} mm "
                  f"(q{self.threshold_quantile*100:.0f} wet-day, n_wet={int((y>0).sum())})")

        # Delegate to parent fit() — it calls self.initial_theta() and
        # self._run_filter() which now use the regime-extended theta.
        result = super().fit(y, verbose=verbose, theta0=theta0,
                             options=options, polish=polish)

        # Persist threshold in metadata
        result["threshold_c"] = self._threshold_c
        result["A_ext"] = [
            float(result["theta"][self.n_gas + self.n_pi + j])
            for j in range(self.n_ext)
        ]
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # save_result override — persist regime-specific metadata
    # ──────────────────────────────────────────────────────────────────────────

    def save_result(self, result: dict, out_dir: Path) -> None:
        """Delegate to parent and append regime metadata."""
        super().save_result(result, out_dir)
        import json
        meta_path = out_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["threshold_c"]    = result.get("threshold_c")
            meta["A_ext"]          = result.get("A_ext")
            meta["regime_tv_names"]= self.regime_tv_names
            meta["model_class"]    = "RegimeZAGASModel"
            meta_path.write_text(json.dumps(meta, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Builder: create RegimeZAGASModel from a Stage-1/2 base winner
# ─────────────────────────────────────────────────────────────────────────────

def build_regime_from_winner(
    dist_phixi,
    pi_dyn,
    regime_tv_names: List[str],
    threshold_quantile: float = 0.95,
    base_params_df: Optional["pd.DataFrame"] = None,
    y_train: Optional[np.ndarray] = None,
    scaling: str = "diagonal_inverse_fisher",
) -> Tuple[RegimeZAGASModel, Optional[np.ndarray]]:
    """
    Build a GAS(1,1) regime-sensitive model with phi+xi distribution.

    Always uses gas_lags=[1] (GAS(1,1)) regardless of the base winner's lag
    structure.  Always uses the phi+xi distribution (GB2LogLink).

    Warm-starts by mapping base winner parameters by name:
      - Parameters that exist in base_params_df are copied by name.
      - Parameters missing from base (e.g. xi params when base was phi-only,
        or A_ext params) are initialised from initial_theta().

    Parameters
    ----------
    dist_phixi         : GB2LogLink instance (phi+xi distribution, always used)
    pi_dyn             : PiDynamics instance
    regime_tv_names    : which TV parameters get the A_ext regime term
    threshold_quantile : quantile for threshold_c
    base_params_df     : estimated_parameters.csv of the Stage-1/2 base winner
    y_train            : training observations (for initial_theta warm-start)
    scaling            : score scaling method

    Returns
    -------
    model   : RegimeZAGASModel ready to call fit()
    theta0  : warm-start vector or None
    """
    import pandas as pd

    model = RegimeZAGASModel(
        distribution=dist_phixi,
        pi_dynamics=pi_dyn,
        seasonal="daily",
        gas_lags=[1],          # always GAS(1,1)
        scaling=scaling,
        threshold_quantile=threshold_quantile,
        regime_tv_names=regime_tv_names,
    )

    if y_train is None:
        return model, None

    # Start from the model's own initial_theta
    theta0 = model.initial_theta(y_train)

    # Map base winner parameters by name (best-effort)
    if base_params_df is not None and not base_params_df.empty:
        if "parameter" in base_params_df.columns and "value" in base_params_df.columns:
            base_dict = dict(zip(base_params_df["parameter"], base_params_df["value"]))
            param_names = model.parameter_names()
            for i, pname in enumerate(param_names):
                if pname in base_dict:
                    theta0[i] = float(base_dict[pname])
                # A_ext_* params default to 0 (initial_theta() already provides 0)

    return model, theta0
