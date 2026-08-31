"""
Static (unconditional) Zero-Augmented GB2 model — Stage 5 step 1.

=============================================================================
THEORY
=============================================================================

No GAS recursion at all: phi, xi, gamma, zeta are constants for the whole
series (not driven by any state equation), and pi (the rain/no-rain
probability) is likewise a single constant. This is the "completely
static" benchmark against which Stage 5's threshold-weighted dynamic-xi
model (models/threshold_weighted_xi_gas.py) is built on top of.

The zero-augmented log-likelihood factorises additively:

    l(theta) = sum_{y_t=0} log(1 - pi)
             + sum_{y_t>0} [ log(pi) + log g_y(y_t; phi,xi,gamma,zeta) ]

pi and (phi,xi,gamma,zeta) do not share any parameters, so their MLEs are
separable and estimated independently, not via one joint optimisation:

    pi_MV   = (# wet days) / (# total days)                    [closed form]
    theta_MV = argmax  sum_{y_t>0} log g_y(y_t; phi,xi,gamma,zeta)  [MLE]

theta_MV is obtained via GASFilter.fit_unconditional(y_pos), the same
static-distribution MLE routine the GAS filter itself uses to warm-start
its own optimisation (models/gas_filter.py::_unconditional_mle) -- reused
here directly as a standalone "fit a static GB2" result rather than merely
an internal warm-start step.

=============================================================================
EXECUTION-FLOW / CODE WALKTHROUGH
=============================================================================

StaticGB2Model.fit(y_train):
  1. y_pos = y_train[y_train > 0]
  2. theta_MV = GASFilter(GB2LogLink()).fit_unconditional(y_pos)
       -> {phi, xi, gamma, zeta}, via L-BFGS-B maximising logpdf_sum(y_pos)
  3. pi_MV = mean(y_train > 0)                          [closed form]
  4. loglik_gb2   = sum_{y_t>0} log g_y(y_t; theta_MV)  [reported separately]
     loglik_pi    = n_wet*log(pi_MV) + n_dry*log(1-pi_MV)
     loglik_total = loglik_gb2 + loglik_pi              [used for AIC/BIC]
  5. theta = [phi_MV, xi_MV, gamma_MV, zeta_MV, pi_MV]  (5 free parameters)

simulate_oos / filter return constant paths (tv_names=[], f_arr shape
(T, 0), pi array filled with the single pi_MV value) -- this makes the
model interoperate with every diagnostics/* function exactly like any
time-varying model, just with a degenerate (flat) predictive trajectory.

=============================================================================
IMPLEMENTATION NOTES
=============================================================================

* Not a subclass of ZAGASModel: ZAGASModel unconditionally routes every
  observation through a GAS recursion and a PiDynamics object: there is
  no configuration that turns it into a genuinely constant-parameter
  model. Reimplementing the five-method interface directly here (fit,
  save_result, filter, simulate_oos, cdf_series) is simpler and clearer
  than forcing a degenerate GAS(1,1) with A=B=0 through machinery that
  was not designed for it.
* pi is estimated in closed form, never by numerical optimisation --
  see the additive log-likelihood decomposition above.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from distributions.gb2_log_link import GB2LogLink
from models.gas_filter import GASFilter


def _fit_static_gb2(dist, y_pos: np.ndarray) -> dict:
    """
    Shared GB2-only static fit: bounded L-BFGS-B warm start
    (GASFilter.fit_unconditional) followed by an unbounded BFGS polish (see
    module docstring for why the bounded step alone is not the final
    answer). Used by both StaticGB2Model and StaticGB2DynamicPiModel --
    the GB2 side of the fit is identical either way, since pi never enters
    this objective (additive separability).

    Returns {} on failure, else {"phi","xi","gamma","zeta","loglik_gb2","runtime_s"}.
    """
    t0 = time.time()
    warm = GASFilter(distribution=dist).fit_unconditional(y_pos)
    if not warm:
        return {}
    x0 = np.array([warm["phi"], warm["xi"], warm["gamma"], warm["zeta"]])

    def _obj(x: np.ndarray) -> float:
        phi_x, xi_x, gamma_x, zeta_x = x
        if gamma_x >= zeta_x:
            return 1e8
        penalty = 0.0
        for val in (phi_x, xi_x, gamma_x, zeta_x):
            excess = max(0.0, abs(val) - 20.0)
            penalty += 1e3 * excess ** 2
        ll = dist.logpdf_sum(y_pos, phi=phi_x, xi=xi_x, gamma=gamma_x, zeta=zeta_x)
        if not np.isfinite(ll):
            return 1e8 + penalty
        return -ll + penalty

    res = minimize(_obj, x0, method="BFGS", options={"maxiter": 1000, "gtol": 1e-6})
    res2 = minimize(_obj, res.x, method="BFGS", options={"maxiter": 300, "gtol": 1e-6})
    if res2.fun < res.fun:
        res = res2
    runtime_s = time.time() - t0

    phi_mv, xi_mv, gamma_mv, zeta_mv = [float(v) for v in res.x]
    loglik_gb2 = float(dist.logpdf_sum(y_pos, phi=phi_mv, xi=xi_mv, gamma=gamma_mv, zeta=zeta_mv))
    return {
        "phi": phi_mv, "xi": xi_mv, "gamma": gamma_mv, "zeta": zeta_mv,
        "loglik_gb2": loglik_gb2, "runtime_s": runtime_s,
    }


class StaticGB2Model:
    """Fully static (no GAS dynamics) zero-augmented GB2 model."""

    def __init__(self, distribution: Optional[GB2LogLink] = None):
        self.dist = distribution if distribution is not None else GB2LogLink()
        self.tv_names = []  # nothing time-varying -- see module docstring
        self.static_names = ["phi", "xi", "gamma", "zeta"]

    @property
    def n_params(self) -> int:
        return 5  # phi_MV, xi_MV, gamma_MV, zeta_MV, pi_MV

    def parameter_names(self):
        return ["phi_MV", "xi_MV", "gamma_MV", "zeta_MV", "pi_MV"]

    # ------------------------------------------------------------------
    def fit(self, y: np.ndarray, verbose: bool = False, **_ignored) -> dict:
        y = np.asarray(y, dtype=float)
        n = len(y)
        y_pos = y[y > 0]
        n_wet = len(y_pos)

        # Bounded L-BFGS-B warm start followed by an unbounded BFGS polish --
        # see _fit_static_gb2 and module docstring for why the bounded step
        # alone is not the final answer (verified empirically: phi hit its
        # warm-start upper bound exactly at Darwin before this fix).
        fit_result = _fit_static_gb2(self.dist, y_pos)
        runtime_s = fit_result.get("runtime_s", 0.0)

        if not fit_result:
            theta = np.full(5, np.nan)
            return {
                "theta": theta, "loglik": float("nan"),
                "phi_MV": np.nan, "xi_MV": np.nan, "gamma_MV": np.nan,
                "zeta_MV": np.nan, "pi_MV": np.nan,
                "loglik_gb2": float("nan"), "loglik_pi": float("nan"),
                "validity": "failed", "success": False,
                "message": "unconditional MLE failed to converge",
                "n_iter": 0, "n_fev": 0,
                "grad_norm_inf": float("nan"), "grad_norm_2": float("nan"),
                "runtime_s": runtime_s, "peak_mem_mb": float("nan"),
                "polish_improvement": 0.0, "param_names": self.parameter_names(),
                "n_obs": n, "n_wet": n_wet,
            }

        phi_mv, xi_mv, gamma_mv, zeta_mv = (
            fit_result["phi"], fit_result["xi"], fit_result["gamma"], fit_result["zeta"]
        )
        loglik_gb2 = fit_result["loglik_gb2"]

        # pi: closed-form Bernoulli MLE, independent of the GB2 fit above
        # (see module docstring -- the ZA log-likelihood is additively
        # separable in pi vs. (phi,xi,gamma,zeta), so this is not a warm
        # start for a joint optimisation, it IS the MLE).
        pi_mv = n_wet / n if n > 0 else 0.0
        pi_mv = float(min(max(pi_mv, 1e-6), 1.0 - 1e-6))
        loglik_pi = float(n_wet * np.log(pi_mv) + (n - n_wet) * np.log(1.0 - pi_mv))
        loglik_total = loglik_gb2 + loglik_pi

        theta = np.array([phi_mv, xi_mv, gamma_mv, zeta_mv, pi_mv])
        finite = bool(np.isfinite(loglik_total) and np.all(np.isfinite(theta)))

        return {
            "theta": theta, "loglik": loglik_total,
            "phi_MV": phi_mv, "xi_MV": xi_mv, "gamma_MV": gamma_mv,
            "zeta_MV": zeta_mv, "pi_MV": pi_mv,
            "loglik_gb2": loglik_gb2, "loglik_pi": loglik_pi,
            "validity": "valid_converged" if finite else "failed",
            "success": finite,
            "message": "unconditional L-BFGS-B MLE (GB2 shape/scale) + closed-form pi",
            "n_iter": 0, "n_fev": 0,
            "grad_norm_inf": float("nan"), "grad_norm_2": float("nan"),
            "runtime_s": runtime_s, "peak_mem_mb": float("nan"),
            "polish_improvement": 0.0, "param_names": self.parameter_names(),
            "n_obs": n, "n_wet": n_wet,
        }

    # ------------------------------------------------------------------
    def filter(self, theta: np.ndarray, y: np.ndarray, **_ignored) -> dict:
        phi_mv, xi_mv, gamma_mv, zeta_mv, pi_mv = [float(v) for v in theta]
        T = len(y)
        return {
            "f_arr": np.zeros((T, 0)),
            "pi": np.full(T, pi_mv),
            "static": {"phi": phi_mv, "xi": xi_mv, "gamma": gamma_mv, "zeta": zeta_mv},
            "tv_names": [],
            "eff_start": 0,
            "y_eff": y,
        }

    def cdf_series(self, theta: np.ndarray, y: np.ndarray, **_ignored) -> np.ndarray:
        """F_t(y_t) = (1-pi) + pi*G(y_t) for y_t>0, else (1-pi). Constant pi/G params."""
        paths = self.filter(theta, y)
        pi_v = paths["pi"][0]
        static = paths["static"]
        T = len(y)
        cdfs = np.zeros(T)
        for t in range(T):
            if y[t] <= 0:
                cdfs[t] = 1.0 - pi_v
            else:
                cdfs[t] = (1.0 - pi_v) + pi_v * self.dist.cdf(y[t], **static)
        return cdfs

    def simulate_oos(self, theta: np.ndarray, y_train: np.ndarray,
                      y_test: np.ndarray, **_ignored) -> dict:
        phi_mv, xi_mv, gamma_mv, zeta_mv, pi_mv = [float(v) for v in theta]
        T = len(y_test)
        return {
            "f_arr_oos": np.zeros((T, 0)),
            "pi_oos": np.full(T, pi_mv),
            "static": {"phi": phi_mv, "xi": xi_mv, "gamma": gamma_mv, "zeta": zeta_mv},
            "tv_names": [],
            "y_test": y_test,
        }

    # ------------------------------------------------------------------
    def save_result(self, result: dict, out_dir: "Path") -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        names = result.get("param_names", self.parameter_names())
        pd.DataFrame({"parameter": names, "value": result["theta"]}).to_csv(
            out_dir / "estimated_parameters.csv", index=False
        )

        meta = {
            "model_type": "StaticGB2Model",
            "n_params": self.n_params,
            "tv_param_names": [],
            "static_param_names": self.parameter_names(),
            "phi": result.get("phi_MV"), "xi": result.get("xi_MV"),
            "gamma": result.get("gamma_MV"), "zeta": result.get("zeta_MV"),
            "pi": result.get("pi_MV"),
            "loglik": float(result["loglik"]) if np.isfinite(result["loglik"]) else None,
            "loglik_gb2_only": result.get("loglik_gb2"),
            "loglik_pi_only": result.get("loglik_pi"),
            "n_obs": result.get("n_obs"), "n_wet": result.get("n_wet"),
            "validity": result.get("validity", "unknown"),
            "success": bool(result.get("success", False)),
            "message": str(result.get("message", "")),
            "n_iter": int(result.get("n_iter", 0)),
            "n_fev": int(result.get("n_fev", 0)),
            "runtime_s": float(result.get("runtime_s", float("nan"))),
            "polish_improvement": 0.0,
            "estimation_note": (
                "Fully static ZA-GB2: phi/xi/gamma/zeta fit by unconditional "
                "MLE (L-BFGS-B) on wet-day observations only "
                "(GASFilter.fit_unconditional); pi fit in closed form as the "
                "empirical wet-day fraction, independent of the GB2 fit "
                "(the ZA log-likelihood is additively separable in pi vs. "
                "the GB2 shape/scale parameters). No time variation."
            ),
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Static GB2 + dynamic (time-varying) pi -- 2026-07-09 variant
# ─────────────────────────────────────────────────────────────────────────────

class StaticGB2DynamicPiModel:
    """
    Same static (phi,xi,gamma,zeta) GB2 as StaticGB2Model, but pi_t is now
    time-varying, fit by pi_dynamics.standalone_fit (the same AR-logistic
    occurrence model used everywhere else in this framework, e.g. Stages
    1-4), independently of the GB2 fit -- the ZA log-likelihood is
    additively separable in pi vs. (phi,xi,gamma,zeta), so this is not a
    joint optimisation, each half is estimated on its own (see
    models/static_gb2.py's module docstring and
    pi_dynamics/standalone_fit.py for the separability argument).

    Kept as a distinct class alongside StaticGB2Model (not a flag on it) so
    both the static-pi and dynamic-pi variants remain independently
    reproducible artifacts, per the 2026-07-09 request to keep both.
    """

    def __init__(self, distribution: Optional[GB2LogLink] = None,
                 pi_dyn=None, seasonal: str = "daily"):
        self.dist = distribution if distribution is not None else GB2LogLink()
        if pi_dyn is None:
            from pi_dynamics.ar_logistic import ARLogisticPiDynamics
            pi_dyn = ARLogisticPiDynamics()
        self.pi_dyn = pi_dyn
        self.seasonal = seasonal
        self.tv_names = []  # GB2 side still static; occurrence is a separate concept
        self.static_names = ["phi", "xi", "gamma", "zeta"]

    @property
    def _pi_param_names(self):
        return self.pi_dyn.param_names(self.seasonal)

    @property
    def n_params(self) -> int:
        return 4 + len(self._pi_param_names)  # phi,xi,gamma,zeta + pi_dynamics theta

    def parameter_names(self):
        return ["phi_MV", "xi_MV", "gamma_MV", "zeta_MV"] + self._pi_param_names

    def _split_theta(self, theta: np.ndarray):
        theta = np.asarray(theta, dtype=float)
        phi_mv, xi_mv, gamma_mv, zeta_mv = theta[:4]
        pi_theta = theta[4:]
        return float(phi_mv), float(xi_mv), float(gamma_mv), float(zeta_mv), pi_theta

    # ------------------------------------------------------------------
    def fit(self, y: np.ndarray, verbose: bool = False, **_ignored) -> dict:
        from pi_dynamics.standalone_fit import fit_pi_dynamics_standalone

        y = np.asarray(y, dtype=float)
        n = len(y)
        y_pos = y[y > 0]
        n_wet = len(y_pos)

        gb2_fit = _fit_static_gb2(self.dist, y_pos)
        pi_fit = fit_pi_dynamics_standalone(y, pi_dyn=self.pi_dyn, seasonal=self.seasonal, verbose=verbose)

        if not gb2_fit or not pi_fit.get("success"):
            return {
                "theta": np.full(self.n_params, np.nan), "loglik": float("nan"),
                "validity": "failed", "success": False,
                "message": "static GB2 or pi_dynamics standalone fit failed",
                "runtime_s": gb2_fit.get("runtime_s", 0.0) + pi_fit.get("runtime_s", 0.0),
                "param_names": self.parameter_names(), "n_obs": n, "n_wet": n_wet,
            }

        theta = np.concatenate([
            [gb2_fit["phi"], gb2_fit["xi"], gb2_fit["gamma"], gb2_fit["zeta"]],
            pi_fit["theta"],
        ])
        loglik_gb2 = gb2_fit["loglik_gb2"]
        loglik_pi  = pi_fit["loglik"]
        loglik_total = loglik_gb2 + loglik_pi
        finite = bool(np.isfinite(loglik_total) and np.all(np.isfinite(theta)))

        return {
            "theta": theta, "loglik": loglik_total,
            "phi_MV": gb2_fit["phi"], "xi_MV": gb2_fit["xi"],
            "gamma_MV": gb2_fit["gamma"], "zeta_MV": gb2_fit["zeta"],
            "loglik_gb2": loglik_gb2, "loglik_pi": loglik_pi,
            "pi_theta": pi_fit["theta"], "pi_param_names": pi_fit["param_names"],
            "validity": "valid_converged" if finite else "failed",
            "success": finite,
            "message": (
                "GB2: unconditional L-BFGS-B+BFGS MLE on wet days; "
                "pi: standalone AR-logistic MLE on the full binary occurrence "
                "sequence -- both independent, per additive ZA separability."
            ),
            "n_iter": 0, "n_fev": 0,
            "grad_norm_inf": float("nan"), "grad_norm_2": float("nan"),
            "runtime_s": gb2_fit["runtime_s"] + pi_fit["runtime_s"],
            "peak_mem_mb": float("nan"), "polish_improvement": 0.0,
            "param_names": self.parameter_names(), "n_obs": n, "n_wet": n_wet,
        }

    # ------------------------------------------------------------------
    def _eta_over(self, theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        from pi_dynamics.standalone_fit import eta_path_full
        _, _, _, _, pi_theta = self._split_theta(theta)
        return eta_path_full(self.pi_dyn, pi_theta, y, self.seasonal)

    def filter(self, theta: np.ndarray, y: np.ndarray, **_ignored) -> dict:
        phi_mv, xi_mv, gamma_mv, zeta_mv, _ = self._split_theta(theta)
        eta = self._eta_over(theta, y)
        pi_t = 1.0 / (1.0 + np.exp(-eta))
        return {
            "f_arr": np.zeros((len(y), 0)),
            "pi": pi_t,
            "static": {"phi": phi_mv, "xi": xi_mv, "gamma": gamma_mv, "zeta": zeta_mv},
            "tv_names": [],
            "eff_start": 0,
            "y_eff": y,
        }

    def cdf_series(self, theta: np.ndarray, y: np.ndarray, **_ignored) -> np.ndarray:
        paths = self.filter(theta, y)
        static = paths["static"]
        T = len(y)
        cdfs = np.zeros(T)
        for t in range(T):
            pi_t = paths["pi"][t]
            if y[t] <= 0:
                cdfs[t] = 1.0 - pi_t
            else:
                cdfs[t] = (1.0 - pi_t) + pi_t * self.dist.cdf(y[t], **static)
        return cdfs

    def simulate_oos(self, theta: np.ndarray, y_train: np.ndarray,
                      y_test: np.ndarray, **_ignored) -> dict:
        phi_mv, xi_mv, gamma_mv, zeta_mv, _ = self._split_theta(theta)
        y_full = np.concatenate([y_train, y_test])
        eta_full = self._eta_over(theta, y_full)
        pi_oos = 1.0 / (1.0 + np.exp(-eta_full[len(y_train):]))
        return {
            "f_arr_oos": np.zeros((len(y_test), 0)),
            "pi_oos": pi_oos,
            "static": {"phi": phi_mv, "xi": xi_mv, "gamma": gamma_mv, "zeta": zeta_mv},
            "tv_names": [],
            "y_test": y_test,
        }

    # ------------------------------------------------------------------
    def save_result(self, result: dict, out_dir: "Path") -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        names = result.get("param_names", self.parameter_names())
        pd.DataFrame({"parameter": names, "value": result["theta"]}).to_csv(
            out_dir / "estimated_parameters.csv", index=False
        )

        meta = {
            "model_type": "StaticGB2DynamicPiModel",
            "n_params": self.n_params,
            "tv_param_names": [],
            "static_param_names": ["phi", "xi", "gamma", "zeta"],
            "pi_param_names": self._pi_param_names,
            "pi_dyn_type": type(self.pi_dyn).__name__,
            "seasonal": self.seasonal,
            "phi": result.get("phi_MV"), "xi": result.get("xi_MV"),
            "gamma": result.get("gamma_MV"), "zeta": result.get("zeta_MV"),
            "loglik": float(result["loglik"]) if np.isfinite(result["loglik"]) else None,
            "loglik_gb2_only": result.get("loglik_gb2"),
            "loglik_pi_only": result.get("loglik_pi"),
            "n_obs": result.get("n_obs"), "n_wet": result.get("n_wet"),
            "validity": result.get("validity", "unknown"),
            "success": bool(result.get("success", False)),
            "message": str(result.get("message", "")),
            "runtime_s": float(result.get("runtime_s", float("nan"))),
            "polish_improvement": 0.0,
            "estimation_note": (
                "phi/xi/gamma/zeta: same static unconditional MLE as "
                "StaticGB2Model. pi_t: time-varying, fit standalone via the "
                "AR-logistic occurrence model (pi_dynamics/ar_logistic.py, "
                "the same one used in Stages 1-4) on the binary wet/dry "
                "sequence alone, independent of the GB2 fit -- see "
                "pi_dynamics/standalone_fit.py for the separability argument."
            ),
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
