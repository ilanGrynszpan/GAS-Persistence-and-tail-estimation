"""
Threshold-weighted xi GAS model — Stage 5 step 2.

=============================================================================
THEORY
=============================================================================

phi, gamma, zeta are frozen at their Stage 5 step 1 unconditional-MLE
values (phi_MV, gamma_MV, zeta_MV -- see models/static_gb2.py); pi is
likewise frozen at pi_MV. None of the four are re-estimated here. Only xi
is dynamic, via a plain GAS(1,1) recursion:

    xi_{t+1} = omega_xi + A_xi * s_t + B_xi * xi_t

but the score s_t driving that recursion, and the log-likelihood used to
fit (omega_xi, f0_xi, A_xi, B_xi), are both computed from a
threshold-WEIGHTED (not merely threshold-modified) objective:

    l(theta) = sum_{t=1}^{T} log g_y(y_t; phi_MV, xi_t, gamma_MV, zeta_MV) * I_t

    I_t = 1{y_t >= q_c},   q_c = the c-th percentile of training wet-day y

for c in {90, 95, 98, 99} (four separate fits; see run_all_locations.py's
STAGE5_THRESHOLDS). Days with I_t = 0 (including all zero-rainfall days,
by construction, since q_c > 0) contribute nothing to the log-likelihood
AND nothing to the score at t -- xi's state still evolves on those days
purely via the autoregressive term B_xi * xi_t (mean-reversion), exactly
as every GAS filter in this codebase already treats y_t = 0 (no
information from an uninformative observation) -- Stage 5 simply extends
"uninformative" from "zero rainfall" to "below the c-th percentile".

This is a different mechanism from Stage 4 (models/regime_gas.py
RegimeXiOnlyModel), which computes the score/likelihood from every wet day
and instead multiplies the score-response coefficient A by
(1 + A_ext/A * R_t(c)) above the threshold -- i.e. Stage 4 reweights how
strongly a wet day's information feeds into the recursion, while Stage 5
discards below-threshold information entirely.

=============================================================================
REPORTING LOG-LIKELIHOOD  (why two loglik values are saved)
=============================================================================

The optimiser only ever sees the threshold-weighted objective above --
call this `loglik_weighted_training` (saved for transparency). It is NOT
comparable across different c (each c maximises a different, non-nested
subset-restricted objective, and is not comparable to any other model's
full-series log-likelihood either). For AIC/BIC and for comparison against
every other stage's models, `loglik` instead reports the FULL, standard,
un-weighted zero-augmented log-likelihood evaluated over every observation
using the states (xi_t) that the threshold-weighted training produced:

    loglik_full = sum_t [ log(1-pi_MV)                                  if y_t=0
                           log(pi_MV) + log g_y(y_t; phi_MV,xi_t,...)   if y_t>0 ]

This is the number that appears in every AIC/BIC/model-comparison table
alongside all other stages, exactly as for any other model in this report.

=============================================================================
IMPLEMENTATION NOTES
=============================================================================

* GAS(1,1), no covariates -- matches Stage 4's own simplifying precedent
  (RegimeXiOnlyModel.GAS_LAG = 1) of using a single-lag recursion for a
  frozen-phi, xi-only tail model regardless of what lag structure the
  station's own best phi+xi model used.
* phi_v, gamma_v, zeta_v are plain Python floats (constants for the whole
  series) -- unlike Stage 4, which freezes phi/pi as whole time-varying
  arrays from a dynamic base model; Stage 5 step 1's GB2 side is itself
  always static, so there is no GB2 time series to align. pi_v MAY be
  either a scalar (static-pi step-1 variant, models.static_gb2.StaticGB2Model)
  or a full-length array index-aligned to whichever y array the calling
  method itself receives (dynamic-pi variant,
  models.static_gb2.StaticGB2DynamicPiModel; 2026-07-09) -- xi's own
  GAS(1,1) fit is completely unaffected either way, since pi never enters
  the threshold-weighted objective (see THEORY above); only the OOS/IS
  evaluation and the reported full-series loglik change.
* This is not a subclass of RegimeXiOnlyModel: the recursion loop differs
  at exactly the two lines that matter (log-likelihood accumulation and
  score computation are both gated by I_t here, vs. gated by y_t>0 with a
  regime-modified A there) -- correctness is clearer written out directly
  than parameterising a shared loop with a strategy-pattern hook.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from distributions.base import Distribution
from distributions.gb2_log_link import GB2LogLink


def _to_pi_array(pi_v, n: int) -> np.ndarray:
    """
    pi_v is either a scalar (broadcast to length n) or an array already
    index-aligned to length n (the caller's own y array) -- see module
    docstring. Every ThresholdWeightedXiModel method that receives pi_v
    passes its own full, unsliced length here and lets this helper do the
    (trivial) broadcast or (identity) pass-through.
    """
    if np.isscalar(pi_v):
        return np.full(n, float(pi_v))
    arr = np.asarray(pi_v, dtype=float)
    if len(arr) != n:
        raise ValueError(f"pi_v array length {len(arr)} != expected {n}")
    return arr


class ThresholdWeightedXiModel:
    """Frozen phi/gamma/zeta/pi, GAS(1,1)-dynamic xi driven by a
    threshold-weighted (not merely threshold-modified) log-likelihood."""

    GAS_LAG = 1

    def __init__(
        self,
        distribution: Distribution,
        scaling: str = "diagonal_inverse_fisher",
        threshold_quantile: float = 0.95,
    ):
        self.dist = distribution
        self.scaling = scaling
        self.threshold_quantile = threshold_quantile
        self._threshold_c: Optional[float] = None

    # ------------------------------------------------------------------
    # theta = [omega_xi, f0_xi, A_xi, B_xi]
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        return 4

    def parameter_names(self) -> List[str]:
        return ["omega_xi", "f0_xi", "A_xi_1", "B_xi_1"]

    def _decode(self, theta: np.ndarray) -> dict:
        return {
            "omega": float(theta[0]), "f0": float(theta[1]),
            "A": float(theta[2]), "B": float(theta[3]),
        }

    def initial_theta(self, xi_mv: float = 0.0) -> np.ndarray:
        theta = np.zeros(4)
        theta[1] = xi_mv   # f0_xi warm-started at step 1's static xi_MV
        theta[3] = 0.90    # B_xi: persistent-but-stable start
        return theta

    def compute_threshold(self, y: np.ndarray, c: float) -> float:
        """c-th percentile of training wet-day observations (c in (0,1))."""
        wet = y[y > 0]
        return float(np.quantile(wet, c)) if len(wet) else 1.0

    @property
    def threshold_c(self) -> float:
        if self._threshold_c is None:
            raise RuntimeError("threshold_c not set -- call fit() first.")
        return self._threshold_c

    # ------------------------------------------------------------------
    def _scaled_score(self, raw_xi_score: float, phi_v: float, xi_t: float,
                       gamma_v: float, zeta_v: float) -> float:
        if self.scaling == "unit":
            return raw_xi_score
        fi = self.dist.fisher_info_diag(phi=phi_v, xi=xi_t, gamma=gamma_v, zeta=zeta_v)
        i_xixi = fi.get("xi", np.nan)
        if not np.isfinite(i_xixi) or i_xixi <= 1e-8:
            return raw_xi_score
        return raw_xi_score / i_xixi

    # ------------------------------------------------------------------
    # Core recursion. y has length T; phi_v/gamma_v/zeta_v/pi_v are frozen
    # scalars (constant over the whole series -- see module docstring).
    # ------------------------------------------------------------------

    def _run_filter(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        phi_v: float,
        gamma_v: float,
        zeta_v: float,
        threshold_c: float,
        return_paths: bool = False,
    ):
        T = len(y)
        eff = self.GAS_LAG
        if T <= eff:
            return 1e12 if not return_paths else {}

        d = self._decode(theta)

        penalty = 0.0
        LAM_PERSIST, LAM_SCORE, LAM_STATE = 1e5, 1e3, 1e3
        penalty += LAM_PERSIST * max(0.0, abs(d["B"]) - 0.98) ** 2
        penalty += LAM_SCORE   * max(0.0, abs(d["A"]) - 2.0) ** 2

        xi_arr = np.zeros(T + 1)
        s_arr  = np.zeros(T)
        xi_arr[: eff + 1] = d["f0"]
        loglik_weighted = 0.0

        for t in range(eff, T):
            xi_t = xi_arr[t]
            state_excess = max(0.0, abs(xi_t) - 15.0)
            penalty += LAM_STATE * state_excess ** 2

            exceed = y[t] >= threshold_c   # I_t: threshold indicator (y_t>0 implied)
            if exceed:
                ll_dist = self.dist.logpdf(y[t], phi=phi_v, xi=xi_t, gamma=gamma_v, zeta=zeta_v)
                if not np.isfinite(ll_dist):
                    return (1e12 + penalty) if not return_paths else {}
                loglik_weighted += ll_dist

                raw = self.dist.score(y[t], phi=phi_v, xi=xi_t, gamma=gamma_v, zeta=zeta_v)
                s_arr[t] = self._scaled_score(raw["xi"], phi_v, xi_t, gamma_v, zeta_v)
                if not np.isfinite(s_arr[t]):
                    return (1e12 + penalty) if not return_paths else {}
            # else: I_t=0 -- no likelihood contribution, no score (s_arr[t] stays 0),
            # xi still evolves via the AR term below (mean-reversion only).

            xi_next = d["omega"] + d["A"] * s_arr[t] + d["B"] * xi_t
            if not np.isfinite(xi_next):
                return (1e12 + penalty) if not return_paths else {}
            xi_arr[t + 1] = xi_next

        objective = -loglik_weighted + penalty
        if not return_paths:
            return objective

        return {
            "xi": xi_arr[eff:T], "s_arr": s_arr[eff:T],
            "eff_start": eff, "loglik_weighted": loglik_weighted,
            "penalty": penalty, "objective": objective,
            "y_eff": y[eff:T], "threshold_c": threshold_c,
        }

    def _full_za_loglik(self, xi_path: np.ndarray, y_eff: np.ndarray,
                         phi_v: float, gamma_v: float, zeta_v: float, pi_v) -> float:
        """Standard (unweighted) ZA log-likelihood over every observation in
        y_eff, using the fitted xi_t path -- see module docstring on why
        this (not loglik_weighted) is used for AIC/BIC/model comparison.
        pi_v: scalar or array already the same length as y_eff (caller does
        any eff-offset slicing -- see _to_pi_array)."""
        pi_arr = _to_pi_array(pi_v, len(y_eff))
        total = 0.0
        for i, yt in enumerate(y_eff):
            if yt <= 0:
                total += np.log(1.0 - pi_arr[i])
            else:
                ll = self.dist.logpdf(yt, phi=phi_v, xi=xi_path[i], gamma=gamma_v, zeta=zeta_v)
                total += np.log(pi_arr[i]) + ll
        return float(total)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        y: np.ndarray,
        phi_v: float,
        gamma_v: float,
        zeta_v: float,
        pi_v: float,
        threshold_quantile_c: float,
        xi_mv: float = 0.0,
        verbose: bool = False,
        theta0: Optional[np.ndarray] = None,
        options: Optional[dict] = None,
        polish: bool = True,
        pi_dynamic_source_model_id: Optional[str] = None,
    ) -> dict:
        """Estimate xi's GAS(1,1) block by unbounded BFGS on the threshold-
        weighted objective. phi_v/gamma_v/zeta_v/pi_v are frozen Stage 5
        step 1 constants, never optimised here."""
        import tracemalloc

        self._threshold_c = self.compute_threshold(y, threshold_quantile_c)
        if verbose:
            print(f"  Stage-5 threshold c=q{threshold_quantile_c*100:.0f} "
                  f"-> {self._threshold_c:.3f} mm (wet-day)")

        theta0 = self.initial_theta(xi_mv) if theta0 is None else np.asarray(theta0, dtype=float)
        opt_options = {"maxiter": 1000, "gtol": 1e-3, "disp": verbose}
        if options:
            opt_options.update(options)

        t_start = time.time()
        tracemalloc.start()

        def obj(theta):
            return self._run_filter(theta, y, phi_v, gamma_v, zeta_v,
                                     self._threshold_c, return_paths=False)

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

        paths = self._run_filter(res.x, y, phi_v, gamma_v, zeta_v,
                                  self._threshold_c, return_paths=True)
        finite_p  = bool(np.all(np.isfinite(res.x)))
        states_ok = bool(paths) and bool(np.all(np.isfinite(paths.get("xi", [0]))))
        grad_ok   = (grad_norm_inf < 1e-2) if not np.isnan(grad_norm_inf) else False

        loglik_weighted = float(paths["loglik_weighted"]) if paths else float("nan")
        if paths:
            # pi_v is aligned to the full `y` passed to fit(); _full_za_loglik
            # needs it aligned to y_eff = y[eff:] instead (see _to_pi_array).
            pi_eff = pi_v if np.isscalar(pi_v) else np.asarray(pi_v, dtype=float)[paths["eff_start"]:]
            loglik_full = self._full_za_loglik(paths["xi"], paths["y_eff"], phi_v, gamma_v, zeta_v, pi_eff)
        else:
            loglik_full = float("nan")
        finite_ll = bool(np.isfinite(loglik_full))

        if res.success and finite_p and finite_ll and states_ok and grad_ok:
            validity = "valid_converged"
        elif finite_p and finite_ll and states_ok:
            validity = "valid_with_warning"
        else:
            validity = "failed"

        hess_inv = np.array(res.hess_inv) if hasattr(res, "hess_inv") else None
        std_errors, se_quality = None, "unavailable"
        if hess_inv is not None and hess_inv.ndim == 2:
            diag_h = np.diag(hess_inv)
            if np.all(diag_h > 0) and np.all(np.isfinite(diag_h)):
                std_errors = np.sqrt(diag_h)
                se_quality = "approximate"
            else:
                se_quality = "unreliable"

        return {
            "theta": res.x, "loglik": loglik_full,
            "loglik_weighted_training": loglik_weighted,
            "validity": validity, "success": bool(res.success), "message": str(res.message),
            "n_iter": int(res.nit), "n_fev": int(res.nfev),
            "grad": grad, "grad_norm_inf": grad_norm_inf, "grad_norm_2": grad_norm_2,
            "hess_inv": hess_inv, "std_errors": std_errors, "se_quality": se_quality,
            "runtime_s": runtime_s, "peak_mem_mb": peak_mem / 1e6,
            "polish_improvement": polish_improvement,
            "param_names": self.parameter_names(),
            "threshold_c": self._threshold_c,
            "threshold_quantile": threshold_quantile_c,
            "phi_frozen": phi_v, "gamma_frozen": gamma_v,
            "zeta_frozen": zeta_v, "pi_frozen": pi_v,
            "pi_dynamic_source_model_id": pi_dynamic_source_model_id,
        }

    def simulate_oos(
        self,
        theta: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        phi_v: float,
        gamma_v: float,
        zeta_v: float,
        pi_v: float,
        threshold_c: float,
    ) -> dict:
        """1-step-ahead rolling OOS evaluation over the concatenated series.
        pi_v: scalar, or array aligned to y_full=concat(y_train,y_test)
        (i.e. the same length/order StaticGB2DynamicPiModel's own eta path
        over y_train+y_test would produce -- see _to_pi_array)."""
        y_full = np.concatenate([y_train, y_test])
        pi_full = _to_pi_array(pi_v, len(y_full))
        paths = self._run_filter(theta, y_full, phi_v, gamma_v, zeta_v,
                                  threshold_c, return_paths=True)
        if not paths:
            return {"f_arr_oos": np.zeros((len(y_test), 1)),
                    "pi_oos": pi_full[-len(y_test):],
                    "static": {"phi": phi_v, "gamma": gamma_v, "zeta": zeta_v},
                    "tv_names": ["xi"], "y_test": y_test}

        T_train = len(y_train)
        eff = paths["eff_start"]
        oos_start_in_eff = max(T_train - eff, 0)
        xi_oos = paths["xi"][oos_start_in_eff:]
        pi_oos = pi_full[eff + oos_start_in_eff:]

        return {
            "f_arr_oos": xi_oos.reshape(-1, 1),
            "pi_oos": pi_oos,
            "static": {"phi": phi_v, "gamma": gamma_v, "zeta": zeta_v},
            "tv_names": ["xi"],
            "y_test": y_test[-len(xi_oos):] if len(xi_oos) <= len(y_test) else y_test,
        }

    def filter(self, theta: np.ndarray, y: np.ndarray, phi_v: float,
               gamma_v: float, zeta_v: float, pi_v, threshold_c: float) -> dict:
        """In-sample filter, for IS PIT/ACF diagnostics. pi_v: scalar, or
        array aligned to y (see _to_pi_array)."""
        paths = self._run_filter(theta, y, phi_v, gamma_v, zeta_v,
                                  threshold_c, return_paths=True)
        if not paths:
            return {}
        eff = paths["eff_start"]
        n = len(paths["xi"])
        pi_arr = _to_pi_array(pi_v, len(y))[eff:]
        return {
            "f_arr": paths["xi"].reshape(-1, 1),
            "pi": pi_arr,
            "static": {"phi": phi_v, "gamma": gamma_v, "zeta": zeta_v},
            "tv_names": ["xi"],
            "eff_start": eff,
            "y_eff": paths["y_eff"],
        }

    def cdf_series(self, theta: np.ndarray, y: np.ndarray, phi_v: float,
                   gamma_v: float, zeta_v: float, pi_v: float, threshold_c: float) -> np.ndarray:
        paths = self.filter(theta, y, phi_v, gamma_v, zeta_v, pi_v, threshold_c)
        if not paths:
            return np.array([])
        n = len(paths["pi"])
        cdfs = np.zeros(n)
        y_eff = paths["y_eff"]
        for i in range(n):
            pi_t = paths["pi"][i]
            xi_t = paths["f_arr"][i, 0]
            if y_eff[i] <= 0:
                cdfs[i] = 1.0 - pi_t
            else:
                G_t = self.dist.cdf(y_eff[i], phi=phi_v, xi=xi_t, gamma=gamma_v, zeta=zeta_v)
                cdfs[i] = (1.0 - pi_t) + pi_t * G_t
        return cdfs

    def save_result(self, result: dict, out_dir: "Path") -> None:
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

        # pi_frozen may be a full time-varying array (dynamic-pi variant,
        # 2026-07-09) -- too large/not meaningful to inline into metadata.json
        # as a scalar; instead record where to rebuild it from (the sibling
        # StaticGB2DynamicPiModel artifact) via pi_dynamic_source_model_id.
        pi_frozen = result.get("pi_frozen")
        pi_is_dynamic = pi_frozen is not None and not np.isscalar(pi_frozen)
        meta = {
            "model_type": "ThresholdWeightedXiModel",
            "n_params": self.n_params,
            "tv_param_names": ["xi"],
            "scaling": self.scaling,
            "threshold_quantile": result.get("threshold_quantile"),
            "threshold_c": result.get("threshold_c"),
            "phi":   result.get("phi_frozen"),
            "gamma": result.get("gamma_frozen"),
            "zeta":  result.get("zeta_frozen"),
            "pi":    None if pi_is_dynamic else result.get("pi_frozen"),
            "pi_dynamic": pi_is_dynamic,
            "pi_dynamic_source_model_id": result.get("pi_dynamic_source_model_id") if pi_is_dynamic else None,
            "loglik": float(result["loglik"]) if np.isfinite(result["loglik"]) else None,
            "loglik_weighted_training": result.get("loglik_weighted_training"),
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
            "estimation_note": (
                "phi/gamma/zeta/pi frozen at Stage 5 step 1's unconditional "
                "MLE (models/static_gb2.py); only xi is dynamic (GAS(1,1)). "
                "The optimiser objective (loglik_weighted_training) sums "
                "log g_y(y_t;...) * 1(y_t >= q_c) -- below-threshold days "
                "contribute neither likelihood nor score. `loglik` (used "
                "for AIC/BIC here and in every comparison table) is instead "
                "the standard full-series ZA log-likelihood evaluated at "
                "every observation using the resulting fitted xi_t path."
            ),
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


def build_threshold_weighted_xi_from_static(
    static_theta: np.ndarray,
    scaling: str = "diagonal_inverse_fisher",
) -> "ThresholdWeightedXiModel":
    """
    Build a Stage 5 step 2 model given step 1's fitted theta
    [phi_MV, xi_MV, gamma_MV, zeta_MV, pi_MV] (see models.static_gb2).
    """
    return ThresholdWeightedXiModel(distribution=GB2LogLink(), scaling=scaling)
