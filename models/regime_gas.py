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


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4 (2026-07-07 redesign): xi-only tail-sensitive regime, frozen phi
# ═════════════════════════════════════════════════════════════════════════════
#
# prompt.md objective 3: "Stage 4 ... only makes sense as tail sensitive
# dynamics. Therefore, it will only be used with xi, not with the scale
# parameter phi. The dynamics of phi ... will follow the best calibration
# for phi [...] from earlier stages ... Phi dynamics here should not be
# reestimated."
#
# RegimeZAGASModel above (round 1) always re-optimizes BOTH phi and xi
# jointly as a GAS(1,1) pair, which is exactly what objective 3 now
# prohibits. RegimeXiOnlyModel replaces it for Stage 4: phi (and the
# occurrence probability pi, which is architecturally independent of phi/xi
# per MODELS.md §3/§24) are supplied as precomputed, fixed trajectories from
# whichever phi-only model won Stages 1-3 (pipeline.artifact_utils);
# they never appear in this model's theta and are never touched by the
# optimizer. Only xi -- its own GAS(1,1) block, short-term weather
# covariates (dew point + temperature; see run_all_locations.py for why
# those specific covariates), and the regime extension A_ext_xi -- is
# estimated (objective item 5: "xi ... requires its own optimization").


class RegimeXiOnlyModel:
    """
    Tail-sensitive regime model for xi only, with phi and pi frozen.

    f_xi,t = omega_xi + L(xi) via a plain GAS(1,1) recursion:
        xi_{t+1} = omega_xi + (A_xi + A_ext_xi * R_t(c)) * s_{xi,t}
                             + B_xi * xi_t + Gamma_xi' * X_t
        R_t(c) = 1(y_t > c)

    phi_t and pi_t are NOT part of theta -- they are read from precomputed
    arrays (`phi_full`, `pi_full`) covering the whole y_train+y_test range,
    built by pipeline.artifact_utils.build_full_path() from whichever
    phi-only model won Stages 1-3. `warmup` is that base model's own
    max_lag: phi_full/pi_full are only defined from index `warmup` onward
    (matching the base model's own effective-sample convention), so this
    model's recursion never starts before max(warmup, its own lag=1).

    gamma, zeta (static GB2 shape parameters) are likewise frozen at the
    base model's estimated values -- they describe the whole distribution,
    not either dynamic parameter individually, so re-estimating them here
    while phi is frozen would be statistically incoherent.
    """

    GAS_LAG = 1  # always GAS(1,1) for xi, matching RegimeZAGASModel's precedent

    def __init__(
        self,
        distribution: Distribution,
        scaling: str = "diagonal_inverse_fisher",
        threshold_quantile: float = 0.95,
        cov_names: Optional[List[str]] = None,
    ):
        self.dist = distribution
        self.scaling = scaling
        self.threshold_quantile = threshold_quantile
        self.cov_names = list(cov_names) if cov_names else []
        self.n_cov = len(self.cov_names)
        self._threshold_c: Optional[float] = None

    # ──────────────────────────────────────────────────────────────────────
    # Parameter vector: [omega_xi, f0_xi, A_xi, B_xi, A_ext_xi, gamma_xi(n_cov)]
    # ──────────────────────────────────────────────────────────────────────

    @property
    def n_params(self) -> int:
        return 5 + self.n_cov

    def parameter_names(self) -> List[str]:
        names = ["omega_xi", "f0_xi", "A_xi_1", "B_xi_1", "A_ext_xi"]
        names += [f"gamma_xi_{c}" for c in self.cov_names]
        return names

    def _decode(self, theta: np.ndarray) -> dict:
        d = {
            "omega": float(theta[0]), "f0": float(theta[1]),
            "A": float(theta[2]), "B": float(theta[3]), "A_ext": float(theta[4]),
        }
        d["gamma_cov"] = theta[5:5 + self.n_cov] if self.n_cov else np.zeros(0)
        return d

    def initial_theta(self) -> np.ndarray:
        theta = np.zeros(self.n_params)
        theta[3] = 0.90  # B_xi: start close to persistent-but-stable
        return theta

    def compute_threshold(self, y: np.ndarray) -> float:
        """95th/98th (or configured) percentile of training wet-day observations."""
        wet = y[y > 0]
        return float(np.quantile(wet, self.threshold_quantile)) if len(wet) else 1.0

    # ──────────────────────────────────────────────────────────────────────
    # Score scaling for a single dynamic parameter: with only xi dynamic,
    # "diagonal_inverse_fisher" and "inverse_fisher" coincide (a 1x1 Fisher
    # sub-matrix has no cross-parameter term to invert) -- both reduce to
    # dividing the raw score by I_xixi. "unit" leaves the raw score as is.
    # ──────────────────────────────────────────────────────────────────────

    def _scaled_score(self, raw_xi_score: float, phi_t: float, xi_t: float,
                       gamma_v: float, zeta_v: float) -> float:
        if self.scaling == "unit":
            return raw_xi_score
        fi = self.dist.fisher_info_diag(phi=phi_t, xi=xi_t, gamma=gamma_v, zeta=zeta_v)
        i_xixi = fi.get("xi", np.nan)
        if not np.isfinite(i_xixi) or i_xixi <= 1e-8:
            return raw_xi_score  # fall back to unit scaling if FI is degenerate
        return raw_xi_score / i_xixi

    # ──────────────────────────────────────────────────────────────────────
    # Core recursion
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _align_frozen(raw: np.ndarray, warmup: int, T: int) -> np.ndarray:
        """
        `raw` is a build_full_path()-style array: raw[0] corresponds to
        global index `warmup` (the base model's own effective-sample
        start), covering indices [warmup, warmup + len(raw)). Return a
        length-T array index-aligned with `y` (index t <-> global time t),
        with NaN before `warmup` (never read -- the recursion always starts
        at eff = max(GAS_LAG, warmup)).
        """
        out = np.full(T, np.nan)
        n_avail = min(len(raw), T - warmup)
        if n_avail > 0:
            out[warmup: warmup + n_avail] = raw[:n_avail]
        return out

    def _run_filter(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        phi_full: np.ndarray,
        pi_full: np.ndarray,
        X: np.ndarray,
        gamma_v: float,
        zeta_v: float,
        warmup: int,
        return_paths: bool = False,
    ):
        """
        y and X have length T (the full y_train, or y_train+y_test for OOS
        simulation). phi_full/pi_full must ALREADY be length-T, index-
        aligned with y (see `_align_frozen`) -- callers (fit/simulate_oos)
        perform that alignment before calling this. phi_full/pi_full are
        only valid from index `warmup` onward (see class docstring); the
        recursion starts at eff = max(GAS_LAG, warmup).
        """
        T = len(y)
        eff = max(self.GAS_LAG, warmup)
        if T <= eff:
            return 1e12 if not return_paths else {}

        d = self._decode(theta)
        X_safe = np.nan_to_num(X, nan=0.0) if self.n_cov else np.zeros((T, 0))

        penalty = 0.0
        LAM_PERSIST, LAM_SCORE, LAM_STATE, LAM_COV = 1e5, 1e3, 1e3, 1e2
        penalty += LAM_PERSIST * max(0.0, abs(d["B"]) - 0.98) ** 2
        penalty += LAM_SCORE   * max(0.0, abs(d["A"]) - 2.0) ** 2
        penalty += LAM_SCORE   * max(0.0, abs(d["A_ext"]) - 2.0) ** 2
        if self.n_cov:
            penalty += LAM_COV * max(0.0, np.linalg.norm(d["gamma_cov"]) - 20.0) ** 2

        xi_arr = np.zeros(T + 1)
        s_arr  = np.zeros(T)
        xi_arr[: eff + 1] = d["f0"]
        loglik = 0.0

        for t in range(eff, T):
            phi_t = float(phi_full[t])
            xi_t  = xi_arr[t]
            if not np.isfinite(phi_t):
                return (1e12 + penalty) if not return_paths else {}

            state_excess = max(0.0, abs(xi_t) - 15.0)
            penalty += LAM_STATE * state_excess ** 2

            pi_t = float(pi_full[t])
            pi_t = min(max(pi_t, 1e-12), 1.0 - 1e-12)

            if y[t] == 0:
                ll_t = np.log(1.0 - pi_t)
            else:
                ll_dist = self.dist.logpdf(y[t], phi=phi_t, xi=xi_t, gamma=gamma_v, zeta=zeta_v)
                if not np.isfinite(ll_dist):
                    return (1e12 + penalty) if not return_paths else {}
                ll_t = np.log(pi_t) + ll_dist
            if not np.isfinite(ll_t):
                return (1e12 + penalty) if not return_paths else {}
            loglik += ll_t

            if y[t] > 0:
                raw = self.dist.score(y[t], phi=phi_t, xi=xi_t, gamma=gamma_v, zeta=zeta_v)
                s_arr[t] = self._scaled_score(raw["xi"], phi_t, xi_t, gamma_v, zeta_v)
                if not np.isfinite(s_arr[t]):
                    return (1e12 + penalty) if not return_paths else {}

            regime_t = float(y[t] > self.threshold_c)
            A_eff = d["A"] + d["A_ext"] * regime_t
            cov_term = float(np.dot(d["gamma_cov"], X_safe[t])) if self.n_cov else 0.0
            xi_next = d["omega"] + A_eff * s_arr[t] + d["B"] * xi_t + cov_term
            if not np.isfinite(xi_next):
                return (1e12 + penalty) if not return_paths else {}
            xi_arr[t + 1] = xi_next

        objective = -loglik + penalty
        if not return_paths:
            return objective

        return {
            "xi": xi_arr[eff:T], "s_arr": s_arr[eff:T],
            "pi": pi_full[eff:T], "phi": phi_full[eff:T],
            "eff_start": eff, "loglik": loglik, "penalty": penalty,
            "objective": objective, "y_eff": y[eff:T],
            "threshold_c": self.threshold_c,
        }

    @property
    def threshold_c(self) -> float:
        if self._threshold_c is None:
            raise RuntimeError("threshold_c not set -- call fit() first.")
        return self._threshold_c

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    # phi_full / pi_full accepted here in build_full_path()'s native
    # (shortened, index-0-is-global-index-`warmup`) form -- aligned to
    # length len(y) internally via _align_frozen before filtering.

    def loglik(self, theta, y, phi_full, pi_full, X, gamma_v, zeta_v, warmup) -> float:
        phi_a = self._align_frozen(phi_full, warmup, len(y))
        pi_a  = self._align_frozen(pi_full,  warmup, len(y))
        return -self._run_filter(theta, y, phi_a, pi_a, X, gamma_v, zeta_v,
                                  warmup, return_paths=False)

    def filter(self, theta, y, phi_full, pi_full, X, gamma_v, zeta_v, warmup) -> dict:
        phi_a = self._align_frozen(phi_full, warmup, len(y))
        pi_a  = self._align_frozen(pi_full,  warmup, len(y))
        return self._run_filter(theta, y, phi_a, pi_a, X, gamma_v, zeta_v,
                                 warmup, return_paths=True)

    def fit(
        self,
        y: np.ndarray,
        phi_full: np.ndarray,
        pi_full: np.ndarray,
        gamma_v: float,
        zeta_v: float,
        warmup: int,
        X: Optional[np.ndarray] = None,
        verbose: bool = False,
        theta0: Optional[np.ndarray] = None,
        options: Optional[dict] = None,
        polish: bool = True,
    ) -> dict:
        """
        Estimate xi's GAS(1,1) block + covariates + regime term by
        unbounded BFGS. phi_full/pi_full/gamma_v/zeta_v are frozen inputs,
        never optimized (objective 3/item 4).
        """
        import tracemalloc

        X = X if X is not None else np.zeros((len(y), self.n_cov))
        phi_a = self._align_frozen(phi_full, warmup, len(y))
        pi_a  = self._align_frozen(pi_full,  warmup, len(y))
        self._threshold_c = self.compute_threshold(y)
        if verbose:
            print(f"  Stage-4 xi regime threshold c = {self._threshold_c:.3f} mm "
                  f"(q{self.threshold_quantile*100:.0f} wet-day)")

        theta0 = self.initial_theta() if theta0 is None else np.asarray(theta0, dtype=float)
        opt_options = {"maxiter": 1000, "gtol": 1e-3, "disp": verbose}
        if options:
            opt_options.update(options)

        t_start = time.time()
        tracemalloc.start()

        def obj(theta):
            return self._run_filter(theta, y, phi_a, pi_a, X, gamma_v, zeta_v,
                                     warmup, return_paths=False)

        res = minimize(fun=obj, x0=theta0, method="BFGS", options=opt_options)

        polish_improvement = 0.0
        if polish:
            res2 = minimize(fun=obj, x0=res.x, method="BFGS",
                             options={**opt_options, "maxiter": 200})
            polish_improvement = abs(res.fun - res2.fun) / (1.0 + abs(res.fun))
            if res2.fun < res.fun:
                res = res2

        runtime_s = time.time() - t_start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        grad = getattr(res, "jac", None)
        grad_norm_inf = float(np.max(np.abs(grad))) if grad is not None else float("nan")
        grad_norm_2   = float(np.linalg.norm(grad)) if grad is not None else float("nan")

        paths     = self._run_filter(res.x, y, phi_a, pi_a, X, gamma_v, zeta_v,
                                      warmup, return_paths=True)
        ll_final  = float(paths["loglik"]) if paths else float(-res.fun)
        finite_p  = bool(np.all(np.isfinite(res.x)))
        finite_ll = bool(np.isfinite(ll_final))
        states_ok = bool(paths) and bool(np.all(np.isfinite(paths.get("xi", [0]))))
        grad_ok   = (grad_norm_inf < 1e-2) if not np.isnan(grad_norm_inf) else False

        if res.success and finite_p and finite_ll and states_ok and grad_ok:
            validity = "valid_converged"
        elif finite_p and finite_ll and states_ok:
            validity = "valid_with_warning"
        else:
            validity = "failed"

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

        return {
            "theta": res.x, "loglik": ll_final, "validity": validity,
            "success": bool(res.success), "message": str(res.message),
            "n_iter": int(res.nit), "n_fev": int(res.nfev),
            "grad": grad, "grad_norm_inf": grad_norm_inf, "grad_norm_2": grad_norm_2,
            "hess_inv": hess_inv, "std_errors": std_errors, "se_quality": se_quality,
            "runtime_s": runtime_s, "peak_mem_mb": peak_mem / 1e6,
            "polish_improvement": polish_improvement,
            "param_names": self.parameter_names(),
            "threshold_c": self._threshold_c,
            # Frozen (not optimized) inputs, persisted so downstream metric
            # recomputation (e.g. extended twCRPS) can rebuild the static
            # distribution parameters without needing the base model artifact.
            "gamma_frozen": gamma_v,
            "zeta_frozen":  zeta_v,
        }

    def simulate_oos(
        self,
        theta: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        phi_full: np.ndarray,
        pi_full: np.ndarray,
        gamma_v: float,
        zeta_v: float,
        warmup: int,
        X_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
    ) -> dict:
        """1-step-ahead rolling OOS evaluation over the concatenated series."""
        y_full = np.concatenate([y_train, y_test])
        X_train = X_train if X_train is not None else np.zeros((len(y_train), self.n_cov))
        X_test  = X_test  if X_test  is not None else np.zeros((len(y_test),  self.n_cov))
        X_full  = np.concatenate([X_train, X_test], axis=0)

        phi_aligned = self._align_frozen(phi_full, warmup, len(y_full))
        pi_aligned  = self._align_frozen(pi_full,  warmup, len(y_full))

        paths = self._run_filter(theta, y_full, phi_aligned, pi_aligned, X_full,
                                  gamma_v, zeta_v, warmup, return_paths=True)
        if not paths:
            return {"f_arr_oos": np.zeros((len(y_test), 1)),
                    "pi_oos": np.zeros(len(y_test)),
                    "static": {"gamma": gamma_v, "zeta": zeta_v},
                    "tv_names": ["xi"], "y_test": y_test}

        T_train = len(y_train)
        eff = paths["eff_start"]
        # Slice the OOS portion out of the effective-sample-truncated arrays.
        oos_start_in_eff = max(T_train - eff, 0)
        xi_oos  = paths["xi"][oos_start_in_eff:]
        pi_oos  = paths["pi"][oos_start_in_eff:]
        phi_oos = paths["phi"][oos_start_in_eff:]

        return {
            "f_arr_oos": np.column_stack([phi_oos, xi_oos]),
            "pi_oos":    pi_oos,
            "static":    {"gamma": gamma_v, "zeta": zeta_v},
            "tv_names":  ["phi", "xi"],
            "y_test":    y_test,
        }

    def save_result(self, result: dict, out_dir: "Path") -> None:
        import json
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        names = result.get("param_names", self.parameter_names())
        pd.DataFrame({"parameter": names, "value": result["theta"]}).to_csv(
            out_dir / "estimated_parameters.csv", index=False
        )
        if result.get("std_errors") is not None:
            pd.DataFrame({
                "parameter": names, "std_error": result["std_errors"],
                "se_quality": result["se_quality"],
            }).to_csv(out_dir / "standard_errors.csv", index=False)

        meta = {
            "model_type": "RegimeXiOnlyModel",
            "model_class": "RegimeXiOnlyModel",
            "n_params": self.n_params,
            # phi is frozen (exogenous), so the OOS predictive distribution
            # is still phi+xi -- tv_param_names describes what simulate_oos()
            # returns, not what this model's own theta optimizes (that is
            # xi alone; see parameter_names()).
            "tv_param_names": ["phi", "xi"],
            "cov_names": self.cov_names,
            "scaling": self.scaling,
            "threshold_quantile": self.threshold_quantile,
            "threshold_c": result.get("threshold_c"),
            "gamma": result.get("gamma_frozen"),
            "zeta":  result.get("zeta_frozen"),
            "loglik": float(result["loglik"]),
            "validity": result.get("validity", "unknown"),
            "success": bool(result.get("success", False)),
            "message": str(result.get("message", "")),
            "n_iter": int(result.get("n_iter", 0)),
            "n_fev": int(result.get("n_fev", 0)),
            "grad_norm_inf": float(result.get("grad_norm_inf", float("nan"))),
            "grad_norm_2": float(result.get("grad_norm_2", float("nan"))),
            "runtime_s": float(result.get("runtime_s", float("nan"))),
            "peak_mem_mb": float(result.get("peak_mem_mb", float("nan"))),
            "se_quality": result.get("se_quality", "unavailable"),
            "polish_improvement": float(result.get("polish_improvement", 0.0)),
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        if result.get("grad") is not None:
            pd.DataFrame({"parameter": names, "gradient": result["grad"]}).to_csv(
                out_dir / "gradients.csv", index=False
            )
        if result.get("hess_inv") is not None:
            hi = np.array(result["hess_inv"])
            if hi.ndim == 2:
                np.save(out_dir / "hess_inv.npy", hi)


def build_regime_xi_only_from_frozen_phi(
    base_model_dir: "Path",
    y_train: np.ndarray,
    y_test: np.ndarray,
    xi_cov_train: np.ndarray,
    xi_cov_test: np.ndarray,
    xi_cov_names: List[str],
    threshold_quantile: float,
    base_extra_fit_kwargs: Optional[dict] = None,
    base_extra_oos_kwargs: Optional[dict] = None,
) -> Tuple["RegimeXiOnlyModel", dict]:
    """
    Build a Stage-4 RegimeXiOnlyModel from a saved phi-only base artifact
    (Stage 1, 2, or 3 winner -- ZAGASModel, CovZAGASModel, or
    HarveyZAGASModel, whichever won by OOS RMSE per objective 1a).

    Returns (model, frozen) where `frozen` carries everything the caller
    needs to call model.fit()/simulate_oos(): phi_full, pi_full, gamma_v,
    zeta_v, warmup.
    """
    from pipeline.artifact_utils import load_model_and_theta, build_full_path
    from distributions.gb2_log_link import GB2LogLink

    base_model, base_theta, base_meta, _ = load_model_and_theta(base_model_dir)
    if base_model is None or base_theta is None:
        raise RuntimeError(f"Could not reconstruct base model from {base_model_dir}")

    full = build_full_path(
        base_model, base_theta, y_train, y_test,
        extra_fit_kwargs=base_extra_fit_kwargs, extra_oos_kwargs=base_extra_oos_kwargs,
    )
    phi_full = full["phi"]
    pi_full  = full["pi"]
    gamma_v  = float(full["static"].get("gamma"))
    zeta_v   = float(full["static"].get("zeta"))
    warmup   = full["offset"]

    model = RegimeXiOnlyModel(
        distribution=GB2LogLink(),
        scaling=base_meta.get("scaling", "diagonal_inverse_fisher"),
        threshold_quantile=threshold_quantile,
        cov_names=xi_cov_names,
    )

    frozen = {
        "phi_full": phi_full, "pi_full": pi_full,
        "gamma_v": gamma_v, "zeta_v": zeta_v, "warmup": warmup,
        "X_train": xi_cov_train, "X_test": xi_cov_test,
        "base_model_id": base_meta.get("model_id", str(base_model_dir.name)),
    }
    return model, frozen
