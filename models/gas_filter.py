"""
GAS(p,q) filter — Creal, Koopman & Lucas (2012, J. Appl. Econometrics).

=============================================================================
THEORY  (equations reference the JAE paper)
=============================================================================

Observation equation  [eq. 1]:
    y_t | F_{t-1}  ~  p(y_t ; f_t, theta_static)

Score and scaled score  [eqs. 2-4]:
    nabla_t  =  d/df_t  log p(y_t ; f_t, theta_static)
    S_t      =  scaling_matrix(f_t)
    s_t      =  S_t * nabla_t

Update equation  [eq. 5]:
    f_{t+1}  =  omega  +  A(L) s_t  +  B(L) f_t

=============================================================================
SCORE SCALING MODES  (per MODELS.md §10)
=============================================================================

"unit"
    S_t = I  (identity matrix; no scaling)
    s_t = nabla_t

"diagonal_inverse_fisher"
    S_t = diag(I(f_t))^{-1}
    s_t[j] = nabla_t[j] / I_{jj}(f_t)
    Scales each parameter by its own expected curvature.
    Removes cross-parameter influence while preserving individual scaling.

"inverse_fisher"
    S_t = I(f_t)^{-1}   (full matrix inverse)
    s_t = I(f_t)^{-1} nabla_t
    Statistically natural but mixes score components across parameters.
    For single-parameter models this is identical to diagonal_inverse_fisher.

=============================================================================
SEASONAL GENERALIZATION  (non-consecutive lags)
=============================================================================

Instead of consecutive lags 1…p we allow any ordered set L (gas_lags).
Two standard sets are defined in constants.py:

    GAS_SHORT_LAGS["daily"]    = [1, 2, 3]
    GAS_SEASONAL_LAGS["daily"] = [1, 2, 3, 364, 365, 366, 367]

The GAS filter accepts any custom lag set via the gas_lags argument.
The seasonal parameter controls pi_dynamics only, not the GAS lags.

=============================================================================
IMPLEMENTATION NOTES
=============================================================================

* Each time-varying distribution parameter (e.g. phi, xi for GB2) gets its
  own independent set of  (omega_j, f0_j, {A_j_l, B_j_l for l in L}).
  Diagonal scaling decouples updates. Full inverse Fisher introduces coupling.

* Static distribution parameters (gamma, zeta for GB2) are estimated
  jointly with the GAS hyperparameters via maximum likelihood.

* Score contributions are set to zero when y_t = 0 (no information about
  the positive-part distribution from a zero observation).

* Pre-effective-start history (t < eff_start) is set to the initial state
  f0_j for states and 0 for scores.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from distributions.base import Distribution
from constants import SEASONAL_LAGS

# Supported scaling modes
_VALID_SCALING = {"unit", "diagonal_inverse_fisher", "inverse_fisher"}


# ===========================================================================
# Parameter codec: maps a flat numpy vector <-> named parameter dict
# ===========================================================================


class _ParamCodec:
    """
    Two-way mapping between a flat 1-D theta array and named parameters.

    Parameter ordering in theta:
        For each time-varying parameter j  (in distribution.tv_param_names):
            omega_j          -- GAS intercept
            f0_j             -- initial state (burn-in value for t < eff_start)
            A_j_{l}          -- score coefficient for lag l, for l in L
            B_j_{l}          -- AR   coefficient for lag l, for l in L
        Static parameters (one per entry in static_param_names):
            gamma, zeta, ...
    """

    def __init__(
        self,
        tv_param_names: List[str],
        static_param_names: List[str],
        lags: List[int],
    ):
        self.tv_names = tv_param_names
        self.static_names = static_param_names
        self.lags = lags

        self._idx: Dict[str, int] = {}
        pos = 0

        for name in tv_param_names:
            self._idx[f"omega_{name}"] = pos
            pos += 1
            self._idx[f"f0_{name}"] = pos
            pos += 1
            for l in lags:
                self._idx[f"A_{name}_{l}"] = pos
                pos += 1
            for l in lags:
                self._idx[f"B_{name}_{l}"] = pos
                pos += 1

        for name in static_param_names:
            self._idx[name] = pos
            pos += 1

        self.n_params = pos

    def decode(self, theta: np.ndarray) -> Dict[str, float]:
        return {k: float(theta[v]) for k, v in self._idx.items()}

    def get(self, theta: np.ndarray, key: str) -> float:
        return float(theta[self._idx[key]])


# ===========================================================================
# GASFilter
# ===========================================================================


class GASFilter:
    """
    GAS(L,L) filter with configurable lag set and score scaling.

    Parameters
    ----------
    distribution   : Distribution subclass  (defines logpdf, score,
                     fisher_info_diag, fisher_info_full, tv_param_names)
    seasonal       : 'daily' or 'monthly' — used only for pi_dynamics
                     compatibility; does NOT determine gas_lags here.
    gas_lags       : explicit lag set L for the GAS update.  If None,
                     defaults to SEASONAL_LAGS[seasonal] for backward compat.
    scaling        : score scaling mode — one of:
                         "unit"
                         "diagonal_inverse_fisher"   (default)
                         "inverse_fisher"
    static_params  : names of static distribution parameters to estimate.
                     Defaults to ['gamma', 'zeta'] for the GB2 model.
    """

    DEFAULT_STATIC = ["gamma", "zeta"]

    def __init__(
        self,
        distribution: Distribution,
        seasonal: str = "daily",
        gas_lags: Optional[List[int]] = None,
        scaling: str = "diagonal_inverse_fisher",
        static_params: Optional[List[str]] = None,
        # Legacy alias kept for backward compatibility
        scale_score: Optional[bool] = None,
    ):
        if seasonal not in SEASONAL_LAGS:
            raise ValueError(
                f"seasonal must be one of {list(SEASONAL_LAGS)}, got '{seasonal}'"
            )
        if scaling not in _VALID_SCALING:
            raise ValueError(
                f"scaling must be one of {sorted(_VALID_SCALING)}, got '{scaling!r}'"
            )

        # Backward-compatibility shim: scale_score=True → "diagonal_inverse_fisher"
        # scale_score=False → "unit".  Explicit `scaling` takes precedence.
        if scale_score is not None and scaling == "diagonal_inverse_fisher":
            scaling = "diagonal_inverse_fisher" if scale_score else "unit"

        self.dist = distribution
        self.seasonal = seasonal
        self.scaling = scaling
        # scale_score property for code that checks it directly
        self.scale_score = scaling != "unit"

        # Determine lag set
        if gas_lags is not None:
            self.lags = list(gas_lags)
        else:
            self.lags = list(SEASONAL_LAGS[seasonal])

        self.max_lag = max(self.lags)
        self.tv_names = distribution.tv_param_names
        if static_params is not None:
            self.static_names = list(static_params)
        elif hasattr(distribution, "default_static_params"):
            self.static_names = list(distribution.default_static_params)
        else:
            self.static_names = self.DEFAULT_STATIC

        self.codec = _ParamCodec(self.tv_names, self.static_names, self.lags)
        # Counts how often score scaling fell back to a less aggressive mode.
        # Saved in metadata.json after fitting as a diagnostic indicator.
        self._fallback_count: int = 0

    # ==================================================================
    # Score scaling helper
    # ==================================================================

    def _scaled_score(
        self,
        raw_score: Dict[str, float],
        call_params: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply score scaling with automatic fallback.

        Fallback order when the requested mode produces invalid results
        (non-positive FI diagonal, singular matrix, or non-finite output):

            inverse_fisher → diagonal_inverse_fisher → unit

        Each fallback event increments self._fallback_count, which is saved
        in metadata.json after fitting.  A high count warns that the FI matrix
        is near-singular for the current parameter trajectory.

        No silent clamping (no max(fi, 1e-8)) — invalid FI values trigger
        the fallback explicitly.
        """
        unit_score = {name: raw_score[name] for name in self.tv_names}

        if self.scaling == "unit":
            return unit_score

        # ── diagonal_inverse_fisher ──────────────────────────────────────────
        if self.scaling == "diagonal_inverse_fisher":
            try:
                fi = self.dist.fisher_info_diag(**call_params)
                result: Dict[str, float] = {}
                for name in self.tv_names:
                    fi_val = fi.get(name, 0.0)
                    if fi_val <= 0.0 or not np.isfinite(fi_val):
                        raise ValueError(f"Non-positive FI diagonal for {name}: {fi_val}")
                    s = raw_score[name] / fi_val
                    if not np.isfinite(s):
                        raise ValueError(f"Non-finite scaled score for {name}")
                    result[name] = s
                return result
            except Exception:
                self._fallback_count += 1
                return unit_score

        # ── inverse_fisher (full matrix) ─────────────────────────────────────
        def _try_diagonal() -> Optional[Dict[str, float]]:
            """Attempt diagonal scaling; return None on failure."""
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

        n = len(self.tv_names)

        if n == 1:
            d = _try_diagonal()
            if d is not None:
                return d
            self._fallback_count += 1
            return unit_score

        # Multi-parameter: solve I @ s = nabla
        try:
            if hasattr(self.dist, "fisher_info_submatrix"):
                I_mat = self.dist.fisher_info_submatrix(self.tv_names, **call_params)
            else:
                fi = self.dist.fisher_info_diag(**call_params)
                I_mat = np.diag([fi.get(name, 0.0) for name in self.tv_names])

            if not np.all(np.isfinite(I_mat)):
                raise np.linalg.LinAlgError("Non-finite FI matrix")
            if np.any(np.diag(I_mat) <= 0.0):
                raise np.linalg.LinAlgError("Non-positive FI diagonal")

            grad_vec = np.array([raw_score[name] for name in self.tv_names])
            s_vec = np.linalg.solve(I_mat, grad_vec)

            if not np.all(np.isfinite(s_vec)):
                raise np.linalg.LinAlgError("Solve returned non-finite result")

            return {name: float(s_vec[j]) for j, name in enumerate(self.tv_names)}

        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError, KeyError):
            # Fallback 1: diagonal
            d = _try_diagonal()
            if d is not None:
                self._fallback_count += 1
                return d
            # Fallback 2: unit
            self._fallback_count += 1
            return unit_score

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

        Execution flow:
            decode theta → validate static params → allocate arrays
            → for t in effective sample:
                assemble call_params → eval log-likelihood
                → compute raw score → apply scaling → update f_{t+1}
            → return objective or full path dict

        Parameters
        ----------
        theta        : flat parameter vector (see _ParamCodec for layout)
        y            : observed time series (length T)
        return_paths : if True return filtered state paths; else scalar.

        Returns
        -------
        float (neg-loglik + penalty) when return_paths=False, or dict.
        """
        T = len(y)
        n_tv = len(self.tv_names)
        L = self.lags
        max_lag = self.max_lag

        if T <= max_lag:
            return 1e12 if not return_paths else {}

        p = self.codec.decode(theta)

        # Static param validity guard
        gamma_v = p.get("gamma", 1.0)
        zeta_v = p.get("zeta", 3.0)
        if gamma_v >= zeta_v:
            return 1e12 if not return_paths else {}

        # Penalty accumulators
        penalty = 0.0
        LAM_PERSIST = 1e5
        LAM_SCORE = 1e3
        LAM_STATE = 1e3
        LAM_STATIC = 1e5

        # Guide the optimizer away from the bp=1 boundary (moment condition).
        # The hard check above already blocks bp ≤ 1; this adds a smooth
        # gradient pushing toward log(bp) = zeta-gamma > 0.5 (i.e. bp > 1.65).
        bp_log = zeta_v - gamma_v
        if bp_log < 0.5:
            penalty += LAM_STATIC * (0.5 - bp_log) ** 2

        for s in self.static_names:
            penalty += LAM_STATIC * max(0.0, abs(p[s]) - 20.0) ** 2

        # GAS persistence + score amplitude penalties
        for name in self.tv_names:
            b_sum = np.sum(np.abs([p[f"B_{name}_{l}"] for l in L]))
            penalty += LAM_PERSIST * max(0.0, b_sum - 0.98) ** 2
            a_sum = np.sum(np.abs([p[f"A_{name}_{l}"] for l in L]))
            penalty += LAM_SCORE * max(0.0, a_sum - 2.0) ** 2

        # State arrays
        f_arr = np.zeros((T + 1, n_tv))
        s_arr = np.zeros((T, n_tv))

        for j, name in enumerate(self.tv_names):
            f_arr[: max_lag + 1, j] = p[f"f0_{name}"]

        loglik = 0.0

        for t in range(max_lag, T):

            call_params = {name: f_arr[t, j] for j, name in enumerate(self.tv_names)}
            call_params.update({s: p[s] for s in self.static_names})

            # State soft bound
            for j in range(n_tv):
                penalty += LAM_STATE * max(0.0, abs(f_arr[t, j]) - 15.0) ** 2

            if y[t] > 0:
                ll_t = self.dist.logpdf(y[t], **call_params)
                if not np.isfinite(ll_t):
                    return (1e12 + penalty) if not return_paths else {}
                loglik += ll_t

            # Score + scaling
            if y[t] > 0:
                raw_score = self.dist.score(y[t], **call_params)
                scaled = self._scaled_score(raw_score, call_params)
                for j, name in enumerate(self.tv_names):
                    s_arr[t, j] = scaled[name]
                if not np.all(np.isfinite(s_arr[t, :])):
                    return (1e12 + penalty) if not return_paths else {}

            # GAS update
            for j, name in enumerate(self.tv_names):
                omega_j = p[f"omega_{name}"]
                ar_part = 0.0
                score_part = 0.0
                for l in L:
                    idx = t - l + 1
                    s_past = s_arr[idx, j] if idx >= 0 else 0.0
                    f_past = f_arr[idx, j] if idx >= 0 else p[f"f0_{name}"]
                    score_part += p[f"A_{name}_{l}"] * s_past
                    ar_part += p[f"B_{name}_{l}"] * f_past
                f_arr[t + 1, j] = omega_j + score_part + ar_part

        if not return_paths:
            return -loglik + penalty

        return {
            "f_arr": f_arr[max_lag:T, :],
            "s_arr": s_arr[max_lag:T, :],
            "tv_names": self.tv_names,
            "static": {s: p[s] for s in self.static_names},
            "eff_start": max_lag,
            "loglik": loglik,
        }

    # ==================================================================
    # Public interface
    # ==================================================================

    def loglik(self, theta: np.ndarray, y: np.ndarray) -> float:
        return -self._run_filter(theta, y, return_paths=False)

    def filter(self, theta: np.ndarray, y: np.ndarray) -> dict:
        return self._run_filter(theta, y, return_paths=True)

    def _unconditional_mle(self, y_pos: np.ndarray) -> dict:
        """
        Fit the distribution with constant (time-invariant) parameters to y_pos.

        This is the "static model" MLE used to initialize the GAS filter:
        - TV params → initial state f0 (the unconditional mean of the process)
        - Static params (gamma, zeta, possibly xi) → MLE starting values

        Rationale: Creal, Koopman & Lucas (2012, JAE) suggest initializing the
        GAS filter at the unconditional parameter values.  Fitting the static
        distribution first is the most principled way to obtain these values;
        it avoids the arbitrary guesses that cause the optimizer to waste many
        iterations correcting the static params before learning the dynamics.

        Uses L-BFGS-B with hard parameter bounds (not BFGS) because:
        - Standard errors are NOT needed from this optimization.
        - Hard bounds prevent the initialiser from exploring the degenerate
          region where logpdf overflows (e.g. gamma ≥ zeta).
        - L-BFGS-B converges in fewer iterations than unbounded BFGS when the
          search space is naturally bounded.

        If the distribution provides logpdf_sum(y_arr, **params) the objective
        is evaluated as a single vectorised numpy expression; otherwise the
        per-observation loop is used as fallback.

        Returns an empty dict if optimization fails; callers use fallback
        heuristics in that case.
        """
        if len(y_pos) < 5:
            return {}

        all_params = list(self.tv_names) + list(self.static_names)

        # Hard bounds for each parameter — only for initialisation; no impact
        # on the main GAS optimisation which is unbounded.
        _BOUNDS: Dict[str, tuple] = {
            "phi":   (-5.0,  8.0),   # sigma = exp(phi) ∈ [0.007, 3000] mm
            "xi":    (-3.0,  4.0),   # a     = exp(xi)  ∈ [0.05,  55]
            "gamma": (-2.0,  4.0),   # p     = exp(-gamma) ∈ [0.018, 7.4]
            "zeta":  ( 0.5,  7.0),   # b     = exp(zeta)   ∈ [1.65, 1097]
        }
        bounds = [_BOUNDS.get(name, (-10.0, 10.0)) for name in all_params]

        _fallback_guess: Dict[str, float] = {
            "phi":   float(np.log(np.mean(y_pos) + 1e-3)),
            "xi":    float(np.log(1.2)),
            "gamma": 0.0,
            "zeta":  1.5,
        }
        x0 = np.clip(
            np.array([_fallback_guess.get(name, 0.0) for name in all_params]),
            [lo for lo, _ in bounds],
            [hi for _, hi in bounds],
        )

        use_vec = hasattr(self.dist, "logpdf_sum")

        def _obj(x: np.ndarray) -> float:
            call = {name: float(x[i]) for i, name in enumerate(all_params)}
            # Moment condition enforced by bounds but guard against edge cases
            gv = call.get("gamma"); zv = call.get("zeta")
            if gv is not None and zv is not None and gv >= zv:
                return 1e8
            if use_vec:
                ll = self.dist.logpdf_sum(y_pos, **call)
                return -ll if np.isfinite(ll) else 1e8
            # Fallback: scalar loop
            total = 0.0
            for yi in y_pos:
                ll = self.dist.logpdf(yi, **call)
                if not np.isfinite(ll):
                    return 1e8
                total += ll
            return -total

        try:
            res = minimize(
                _obj, x0, method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-9, "gtol": 1e-5, "disp": False},
            )
            if np.isfinite(res.fun) and np.all(np.isfinite(res.x)):
                return {name: float(res.x[i]) for i, name in enumerate(all_params)}
        except Exception:
            pass

        return {}

    def initial_theta(self, y: np.ndarray) -> np.ndarray:
        """
        Data-driven starting values for the parameter vector.

        Strategy:
        1.  Fit the static (constant-parameter) distribution to y[y>0] by
            MLE (_unconditional_mle).  This provides:
              - f0_{name}: initial GAS state ← static MLE value
              - Static params (gamma, zeta, xi if static): MLE values
        2.  Initialize GAS dynamics:
              - B_{j,l} = 0.90 / n_lags per lag (total persistence ≈ 0.9)
              - A_{j,l} = 0.01 / n_lags per lag (small score loading)
              - omega_j  = f0_j × 0.10
                  With B_total=0.9, the unconditional mean of the GAS process
                  is E[f] = omega / (1 - B_total) = 0.10·f0 / 0.10 = f0  ✓

        References:
          Creal, Koopman & Lucas (2012, JAE), Remark 1 — init at unconditional mean.
          Harvey (2013), Ch. 2 — B ≈ 0.9 heuristic for EGARCH-family models.

        See docs/design_choices_02072026.pdf §6 for the full justification.
        """
        y_pos = y[y > 0]
        if len(y_pos) == 0:
            y_pos = np.array([1.0])

        # Step 1: unconditional MLE
        ucmle = self._unconditional_mle(y_pos)

        # Heuristic fallbacks when MLE fails or a param is missing
        _heuristic: Dict[str, float] = {
            "phi":   float(np.log(np.mean(y_pos) + 1e-3)),
            "xi":    float(np.log(1.2)),
            "gamma": 0.0,
            "zeta":  1.5,
        }

        theta0 = np.zeros(self.codec.n_params)
        idx    = self.codec._idx
        n_l    = len(self.lags)

        for name in self.tv_names:
            f0_guess = float(ucmle.get(name, _heuristic.get(name, 0.0)))
            theta0[idx[f"omega_{name}"]] = f0_guess * 0.10
            theta0[idx[f"f0_{name}"]]    = f0_guess
            for l in self.lags:
                theta0[idx[f"A_{name}_{l}"]] = 0.01 / n_l
            for l in self.lags:
                theta0[idx[f"B_{name}_{l}"]] = 0.90 / n_l

        for s in self.static_names:
            theta0[idx[s]] = float(ucmle.get(s, _heuristic.get(s, 0.0)))

        return theta0

    def default_bounds(self):
        bounds = []
        for name in self.tv_names:
            bounds.append((-3.0, 5.0))
            bounds.append((-5.0, 6.0))
            for _ in self.lags:
                bounds.append((-0.25, 0.50))
            for _ in self.lags:
                bounds.append((-0.80, 0.80))
        for name in self.static_names:
            if name in ("xi", "gamma"):
                bounds.append((-2.0, 2.0))
            elif name == "zeta":
                bounds.append((0.1, 5.0))
            else:
                bounds.append((-5.0, 5.0))
        return bounds

    def fit(self, y: np.ndarray, verbose: bool = False) -> dict:
        """Estimate GAS hyperparameters by BFGS maximum likelihood."""
        theta0 = self.initial_theta(y)
        result = minimize(
            fun=self._run_filter,
            x0=theta0,
            args=(y, False),
            method="BFGS",
            options={"maxiter": 5000, "gtol": 1e-3, "disp": verbose},
        )
        return {
            "theta": result.x,
            "loglik": -result.fun,
            "success": result.success,
            "result": result,
        }
