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

# Inverse Fisher information of the zero-adjusted (mixed) density
#     f(y) = (1 - pi_t) 1{y=0} + pi_t g(y; phi_t) 1{y>0}
# w.r.t. phi:  I_mixed = pi_t * I_g(phi) [+ lambda_1^2 pi_t (1 - pi_t) when
# pi_t depends on phi_t]. It needs pi_t, which only the zero-adjusted model
# knows, so it is accepted only when the owning model opts in
# (GASFilter(..., allow_mixed_scaling=True), see ZAGASModel) and is applied
# by that model, never by GASFilter._scaled_score.
MIXED_SCALING = "mixed_inverse_fisher"
MIXED_SCALINGS = {"mixed_inverse_fisher", "mixed_diagonal_inverse_fisher"}


# ===========================================================================
# Parameter codec: maps a flat numpy vector <-> named parameter dict
# ===========================================================================


def _sigmoid_scalar(z: float) -> float:
    """Numerically stable logistic sigmoid for a Python/numpy scalar."""
    if z >= 0.0:
        return 1.0 / (1.0 + np.exp(-z))
    ez = np.exp(z)
    return ez / (1.0 + ez)


def _logit_scalar(p: float, eps: float = 1e-9) -> float:
    """Inverse sigmoid, clipped away from 0/1 to stay finite."""
    p = min(max(float(p), eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


def _tanh_over_x(S: float) -> float:
    """
    tanh(S)/S, with the S=0 removable singularity handled via its Taylor
    series (tanh(S)/S = 1 - S^2/3 + 2*S^4/15 - ...) rather than division --
    both branches agree to machine precision at the |S|=1e-4 crossover
    (naive tanh(S)/S there differs from the Taylor value by < 1e-9).
    """
    if abs(S) < 1e-4:
        S2 = S * S
        return 1.0 - S2 / 3.0 + 2.0 * S2 * S2 / 15.0
    return float(np.tanh(S) / S)


def _tanh_over_x_prime(S: float) -> float:
    """
    d/dS [tanh(S)/S], needed for the delta-method Jacobian of
    bound_b_tanh_sum. Exact: (S*sech^2(S) - tanh(S)) / S^2; Taylor near 0
    (from d/dS[1 - S^2/3 + 2S^4/15] = -2S/3 + 8S^3/15) to avoid 0/0.
    """
    if abs(S) < 1e-4:
        return -2.0 * S / 3.0 + 8.0 * S ** 3 / 15.0
    t = np.tanh(S)
    sech2 = 1.0 - t * t
    return float((S * sech2 - t) / (S * S))


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

    bound_ab (experimental, default off -- see reports/seasonality GAS/
    gradient_and_convergence_diagnosis.md §8): when True, each A_{j,l} and
    B_{j,l} is reparameterised through an independent, PER-LAG logistic
    sigmoid, decoupled from every other lag's coefficient:

        A_{j,l} = a_max * sigmoid(raw_A_{j,l})   in (0, a_max)
        B_{j,l} = b_max * sigmoid(raw_B_{j,l})   in (0, b_max)

    theta itself still stores the RAW (unbounded) values under the same
    A_{j,l}/B_{j,l} names; decode() is where the bounded, natural-scale
    value used by the recursion is produced. This REPLACES the default
    model's joint penalty sum_l|A_{j,l}| <= A_LIMIT (resp. B) with
    independent per-lag box constraints guaranteed by construction --
    deliberately a MUCH looser restriction on the total budget across lags
    (e.g. a 7-lag seasonal model can now reach a total |B| budget up to
    7*b_max instead of a shared 0.98), at the cost of no longer enforcing
    that joint sum at all. Codec callers (ZAGASModel/CovZAGASModel) must
    skip their own a_sum/b_sum penalty terms when bound_ab is set -- see
    those classes' _run_filter docstrings. bound_ab=False reproduces the
    exact previous (raw, unconstrained-by-construction) behaviour.

    bound_b_joint (experimental, default off, requires bound_ab=True;
    2026-09-02 follow-up): reparameterises B_{j,l} (only B -- A has no
    stationarity-motivated joint bound, see the constructor docstrings on
    ZAGASModel/CovZAGASModel) through a JOINT link that bounds
    sum_l B_{j,l} directly -- restoring the TRUE stationarity requirement
    (a bound on the SUM only) instead of the independent-per-lag box's
    implicit even split (b_max=b_total_max/n_lags), which was found to be
    a strictly TIGHTER restriction than stationarity actually requires
    (no single lag could exceed its even share, even when the data wanted
    one lag to carry nearly the whole budget and the others almost none).
    Uses a multinomial-logit / "softmax with an implicit zero-logit
    reference category" map, applied to the SAME L raw values B_{j,l}
    already in theta (no extra parameters):

        e_l    = exp(raw_B_{j,l})                     for l in lags
        denom  = 1 + sum_l e_l
        B_{j,l} = b_total_max * e_l / denom

    so that sum_l B_{j,l} = b_total_max * (denom-1)/denom < b_total_max
    STRICTLY, for ANY real raw_B_{j,1..L} -- itself literally a sigmoid
    (e_l/denom = sigmoid(log e_l - log(denom - e_l))) of a monotone
    function of the sum, hence "another wall on the sigmoid of the sum".
    Each individual B_{j,l} may still range up to nearly the full
    b_total_max if the others are near zero -- no per-lag ceiling at all,
    only the joint one. Incompatible with ab_signed for B (a signed
    simplex is not constructed here; B stays one-sided, matching this
    project's B_{j,l}>=0 convention throughout) -- raising in __init__ if
    both are requested together rather than silently picking one.
    """

    def __init__(
        self,
        tv_param_names: List[str],
        static_param_names: List[str],
        lags: List[int],
        bound_ab: bool = False,
        a_max: float = 2.0,
        b_max: float = 1.0,
        ab_signed: bool = False,
        bound_b_joint: bool = False,
        b_total_max: float = 0.98,
        bound_b_tanh_sum: bool = False,
        b_sum_max: float = 1.0,
        free_zeta_gamma: bool = False,
        kurtosis_margin: float = 4.0,
        bound_static_box: bool = False,
        static_box: Tuple[float, float] = (-2.0, 15.0),
    ):
        self.tv_names = tv_param_names
        self.static_names = static_param_names
        self.lags = lags
        self.bound_ab = bound_ab
        self.a_max = a_max
        self.b_max = b_max
        self.bound_b_joint = bound_b_joint
        self.b_total_max = b_total_max
        # bound_b_tanh_sum (2026-09-02, alternative to bound_b_joint):
        # applies a signed sigmoid (tanh) DIRECTLY to the SUM of the raw
        # (otherwise-unconstrained) B_{j,l}'s, then rescales every B_{j,l}
        # by the SAME factor so their sum matches exactly -- see decode()
        # for the full derivation. Unlike bound_b_joint's softmax (which
        # gives an ASYMMETRIC (0, b_total_max) sum, no negative persistence
        # at all), this gives a SYMMETRIC (-b_sum_max, b_sum_max) sum,
        # matching a genuinely signed B (as needed when ab_signed lets
        # individual A/B be negative) while still guaranteeing the true
        # stationarity boundary |sum_l B_{j,l}| < b_sum_max is respected BY
        # CONSTRUCTION for ANY real raw values, however large.
        self.bound_b_tanh_sum = bound_b_tanh_sum
        self.b_sum_max = b_sum_max
        if bound_b_joint and bound_b_tanh_sum:
            raise ValueError("bound_b_joint and bound_b_tanh_sum are alternative B-sum links -- pick one.")
        if bound_b_joint and ab_signed:
            raise ValueError(
                "bound_b_joint's softmax-with-phantom-zero map produces only "
                "non-negative B_{j,l} (by construction, matching this "
                "project's B>=0 convention) -- combining it with ab_signed "
                "(which asks for a SIGNED B) is not supported; pick one."
            )
        # ab_signed (default off -- see reports/original vs v2/ link-only
        # follow-up): when True, A_{j,l}/B_{j,l} decode through a SIGNED
        # sigmoid, max*(2*sigmoid(z)-1) in (-max, max), instead of the
        # one-sided max*sigmoid(z) in (0, max). The original model's own
        # penalty is on |A_l|/|B_l| (sum of absolute values), so its true
        # domain is symmetric -- several already-fitted models have
        # genuinely negative A/B coefficients. The one-sided map silently
        # excludes that half of the domain; ab_signed=True matches the
        # original model's actual domain exactly. Only meaningful when
        # bound_ab=True.
        self.ab_signed = ab_signed
        # free_zeta_gamma (default off): when True, "zeta"'s theta slot is
        # reinterpreted as a free real delta, and the NATURAL-scale zeta is
        # computed as gamma + log(kurtosis_margin) + exp(delta) -- this
        # guarantees exp(zeta) > kurtosis_margin*exp(gamma) for ANY real
        # (gamma, delta), with no penalty or rejection needed at all (see
        # the constructor docstrings on ZAGASModel/CovZAGASModel for the
        # exact moment-existence derivation this replaces).
        # kurtosis_margin=4.0 is the 4th-moment/kurtosis-existence margin;
        # kurtosis_margin=1.0 would reproduce the ORIGINAL model's mean-only
        # existence condition (zeta>gamma) via the same mechanism.
        self.free_zeta_gamma = free_zeta_gamma
        self.kurtosis_margin = kurtosis_margin
        # bound_static_box (default off, 2026-09-02): boxes EVERY static
        # parameter (gamma, zeta, xi, ...) AND every f0_{name} (initial GAS
        # state) into the SAME (lo, hi) interval via a smooth, monotone,
        # always-defined affine-sigmoid link:
        #     param = lo + (hi - lo) * sigmoid(z)      z in (-inf, +inf)
        # (NOT the piecewise "-2*sigmoid(z) for z<0, 15*sigmoid(z) for z>0"
        # form, which has a jump discontinuity at z=0 -- e.g. for (lo,hi)=
        # (-2,15): left limit -2*sigmoid(0)=-1, right limit 15*sigmoid(0)=
        # 7.5. The affine form above is the standard, continuous
        # generalisation of a sigmoid box to an ASYMMETRIC interval,
        # reducing to the familiar hi*sigmoid(z) when lo=0.) Incompatible
        # with free_zeta_gamma (both target zeta's link; free_zeta_gamma's
        # analytic construction GUARANTEES the GB2 moment-existence
        # relationship exp(zeta)>margin*exp(gamma), which an independent
        # box on zeta and gamma does NOT -- that trade-off is deliberate
        # here, not an oversight; see the constructor docstrings on
        # ZAGASModel/CovZAGASModel and the "static_box" report for the
        # discussion of what this means for validity).
        self.bound_static_box = bound_static_box
        self.static_box = static_box
        if bound_static_box and free_zeta_gamma:
            raise ValueError(
                "bound_static_box and free_zeta_gamma both reparameterise "
                "zeta -- pick one (bound_static_box does NOT guarantee the "
                "GB2 moment-existence relationship; free_zeta_gamma does)."
            )

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
        out = {k: float(theta[v]) for k, v in self._idx.items()}
        if self.bound_ab:
            for name in self.tv_names:
                # A: always the independent per-lag sigmoid, regardless of
                # bound_b_joint -- no stationarity-motivated joint bound on A.
                for l in self.lags:
                    ak = f"A_{name}_{l}"
                    if self.ab_signed:
                        out[ak] = self.a_max * (2.0 * _sigmoid_scalar(out[ak]) - 1.0)
                    else:
                        out[ak] = self.a_max * _sigmoid_scalar(out[ak])
                if self.bound_b_joint:
                    # Joint softmax-with-phantom-zero-logit: see class
                    # docstring. Computed with the raw B's clipped to a wide
                    # but overflow-safe range (matching _sigmoid_scalar's own
                    # np.clip(z, -700, 700) philosophy elsewhere; +-30 is
                    # already far beyond any raw value this project's fits
                    # have produced -- e[±30] cannot overflow float64).
                    raw_b = np.clip(
                        [out[f"B_{name}_{l}"] for l in self.lags], -30.0, 30.0
                    )
                    e = np.exp(raw_b)
                    denom = 1.0 + np.sum(e)
                    for l, el in zip(self.lags, e):
                        out[f"B_{name}_{l}"] = self.b_total_max * float(el) / denom
                elif self.bound_b_tanh_sum:
                    # Tanh-rescale: B_{j,l} = raw_l * (b_sum_max*tanh(S)/S),
                    # S = sum_l raw_l -- see class docstring. Guarantees
                    # |sum_l B_{j,l}| < b_sum_max for ANY real raw values
                    # (however large), while preserving whatever relative
                    # split across lags the raw values already express
                    # (unlike bound_b_joint, no even-split assumption and
                    # no restriction to non-negative B).
                    raw_b = np.array([out[f"B_{name}_{l}"] for l in self.lags])
                    S = float(np.sum(raw_b))
                    scale = self.b_sum_max * _tanh_over_x(S)
                    for l, r in zip(self.lags, raw_b):
                        out[f"B_{name}_{l}"] = float(r) * scale
                else:
                    for l in self.lags:
                        bk = f"B_{name}_{l}"
                        if self.ab_signed:
                            out[bk] = self.b_max * (2.0 * _sigmoid_scalar(out[bk]) - 1.0)
                        else:
                            out[bk] = self.b_max * _sigmoid_scalar(out[bk])
        if self.free_zeta_gamma and "gamma" in out and "zeta" in out:
            out["zeta"] = out["gamma"] + np.log(self.kurtosis_margin) + np.exp(out["zeta"])
        if self.bound_static_box:
            lo, hi = self.static_box
            for name in self.static_names:
                out[name] = lo + (hi - lo) * _sigmoid_scalar(out[name])
            for name in self.tv_names:
                fk = f"f0_{name}"
                out[fk] = lo + (hi - lo) * _sigmoid_scalar(out[fk])
        return out

    def encode_static_box_target(self, target: float) -> float:
        """
        Convert a NATURAL-scale target (for any static param OR f0_{name})
        into the raw theta value decode() maps back to it under
        bound_static_box -- used by initial_theta(). Inverting
        param = lo + (hi-lo)*sigmoid(z):  z = logit((target-lo)/(hi-lo)).
        Raises if the target does not already lie in (lo, hi) -- a genuine
        domain requirement (callers should choose a target inside the box,
        e.g. clip the unconditional-MLE value into range rather than pass
        one outside it).
        """
        if not self.bound_static_box:
            raise RuntimeError("encode_static_box_target only applies when bound_static_box=True")
        lo, hi = self.static_box
        if not (lo < target < hi):
            raise ValueError(f"target={target} is not inside the box ({lo}, {hi})")
        return _logit_scalar((target - lo) / (hi - lo))

    def encode_ab_target(self, key: str, target: float) -> float:
        """
        Convert a NATURAL-scale target A_{j,l}/B_{j,l} value into the raw
        theta value that decode() will map back to it -- used by
        initial_theta() so a data-driven starting value still lands in the
        right place under bound_ab. Only meaningful when bound_ab is set;
        raises if called otherwise (callers should branch on self.bound_ab).
        B under bound_b_joint is NOT single-lag-encodable (the joint link
        couples every lag of the same tv-parameter) -- use
        encode_b_joint_targets for that case instead.
        """
        if not self.bound_ab:
            raise RuntimeError("encode_ab_target only applies when bound_ab=True")
        if key.startswith("B_") and self.bound_b_joint:
            raise RuntimeError(
                "encode_ab_target cannot encode a single B_{j,l} under "
                "bound_b_joint=True -- use encode_b_joint_targets(name, "
                "targets) with ALL lags of this tv-parameter at once."
            )
        m = self.a_max if key.startswith("A_") else self.b_max
        if self.ab_signed:
            return _logit_scalar((target / m + 1.0) / 2.0)
        return _logit_scalar(target / m)

    def encode_b_joint_targets(self, name: str, targets: Dict[int, float]) -> Dict[int, float]:
        """
        Convert NATURAL-scale B_{name,l} targets -- ALL lags of the SAME
        tv-parameter `name`, passed together since bound_b_joint's link
        couples them -- into the raw theta values decode() maps back to
        them. Used by initial_theta() under bound_b_joint.

        Inverting B_l = b_total_max * e_l/denom with denom = 1+sum(e):
            w_l = B_l / b_total_max            (so sum_l w_l < 1 required)
            e_l = w_l / (1 - sum_l w_l)
            raw_l = log(e_l)

        Raises if the targets' sum does not already satisfy the joint
        bound (a genuine domain requirement, not silently rescaled).
        """
        if not self.bound_b_joint:
            raise RuntimeError("encode_b_joint_targets only applies when bound_b_joint=True")
        w = {l: targets[l] / self.b_total_max for l in self.lags}
        total = sum(w.values())
        if total >= 1.0:
            raise ValueError(
                f"B targets for '{name}' sum to {sum(targets.values())}, which "
                f">= b_total_max={self.b_total_max}; cannot encode exactly "
                f"(choose targets whose sum already satisfies the joint bound)."
            )
        remainder = 1.0 - total
        return {l: float(np.log(max(wl, 1e-12) / remainder)) for l, wl in w.items()}

    def encode_zeta_target(self, gamma_target: float, zeta_target: float) -> float:
        """
        Convert NATURAL-scale (gamma, zeta) targets into the raw delta value
        that decode() will map back to zeta_target given gamma_target --
        used by initial_theta() under free_zeta_gamma. Raises if the target
        pair does not already satisfy the margin (callers should choose a
        sensible target, e.g. widen zeta_target rather than pass an invalid
        pair -- this is a genuine domain requirement, not a numerical
        artifact, so it is not silently clipped).
        """
        if not self.free_zeta_gamma:
            raise RuntimeError("encode_zeta_target only applies when free_zeta_gamma=True")
        residual = zeta_target - gamma_target - np.log(self.kurtosis_margin)
        if residual <= 0.0:
            raise ValueError(
                f"zeta_target={zeta_target} does not satisfy "
                f"exp(zeta) > {self.kurtosis_margin}*exp(gamma) for gamma_target={gamma_target}"
            )
        return float(np.log(residual))

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
        # Experimental per-lag sigmoid-bounded A/B, default off -- see
        # _ParamCodec's docstring and reports/seasonality GAS/
        # gradient_and_convergence_diagnosis.md §8.
        bound_ab: bool = False,
        a_max: float = 2.0,
        b_max: float = 1.0,
        ab_signed: bool = False,
        bound_b_joint: bool = False,
        b_total_max: float = 0.98,
        bound_b_tanh_sum: bool = False,
        b_sum_max: float = 1.0,
        free_zeta_gamma: bool = False,
        kurtosis_margin: float = 4.0,
        bound_static_box: bool = False,
        static_box: Tuple[float, float] = (-2.0, 15.0),
        allow_mixed_scaling: bool = False,
    ):
        if seasonal not in SEASONAL_LAGS:
            raise ValueError(
                f"seasonal must be one of {list(SEASONAL_LAGS)}, got '{seasonal}'"
            )
        if scaling in MIXED_SCALINGS and not allow_mixed_scaling:
            raise ValueError(
                f"scaling '{scaling}' needs pi_t and is only supported by the zero-adjusted models"
            )
        if scaling not in _VALID_SCALING and scaling not in MIXED_SCALINGS:
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

        self.codec = _ParamCodec(
            self.tv_names, self.static_names, self.lags,
            bound_ab=bound_ab, a_max=a_max, b_max=b_max, ab_signed=ab_signed,
            bound_b_joint=bound_b_joint, b_total_max=b_total_max,
            bound_b_tanh_sum=bound_b_tanh_sum, b_sum_max=b_sum_max,
            free_zeta_gamma=free_zeta_gamma, kurtosis_margin=kurtosis_margin,
            bound_static_box=bound_static_box, static_box=static_box,
        )
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
        if self.scaling in MIXED_SCALINGS:
            raise RuntimeError(
                f"'{self.scaling}' depends on pi_t and is applied by "
                "models/mixed_score.py::scaled_score, not GASFilter._scaled_score"
            )

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

    def fit_unconditional(self, y_pos: np.ndarray) -> dict:
        """
        Public entry point for the static (constant-parameter, no GAS
        recursion) MLE fit -- see `_unconditional_mle` for full docs. Exposed
        separately so callers that want a genuine "static model" result
        (e.g. Stage 5 step 1: an unconditional GB2 fit, not merely a GAS
        warm-start) do not need to reach into a private method.
        """
        return self._unconditional_mle(y_pos)

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

        static_box_lo, static_box_hi = self.codec.static_box if self.codec.bound_static_box else (None, None)

        def _clip_into_box(value: float) -> float:
            # Keep a small margin off the exact edges so encode_static_box_
            # target's (lo, target, hi) strict inequality never borders on
            # floating-point equality.
            margin = 1e-3 * (static_box_hi - static_box_lo)
            return float(np.clip(value, static_box_lo + margin, static_box_hi - margin))

        for name in self.tv_names:
            f0_guess = float(ucmle.get(name, _heuristic.get(name, 0.0)))
            theta0[idx[f"omega_{name}"]] = f0_guess * 0.10
            if self.codec.bound_static_box:
                theta0[idx[f"f0_{name}"]] = self.codec.encode_static_box_target(_clip_into_box(f0_guess))
            else:
                theta0[idx[f"f0_{name}"]] = f0_guess
            if self.codec.bound_ab and self.codec.bound_b_joint:
                # B: all lags of this tv-parameter encoded TOGETHER (the
                # joint link couples them) -- same even-split 0.90 total
                # target as the per-lag case below, just expressed jointly.
                b_targets = {l: 0.90 / n_l for l in self.lags}
                raw_b = self.codec.encode_b_joint_targets(name, b_targets)
                for l in self.lags:
                    a_key, b_key = f"A_{name}_{l}", f"B_{name}_{l}"
                    a_target = 0.01 / n_l
                    theta0[idx[a_key]] = self.codec.encode_ab_target(a_key, a_target)
                    theta0[idx[b_key]] = raw_b[l]
            elif self.codec.bound_ab and self.codec.bound_b_tanh_sum:
                # B: raw_l = target_l directly is an APPROXIMATE (not exact)
                # inverse here -- decode()'s tanh(S)/S rescaling shrinks the
                # sum slightly (e.g. targets summing to 0.90 decode to
                # tanh(0.90)=0.716), but this only affects the STARTING
                # point BFGS explores from, not what it can reach, and the
                # result is guaranteed comfortably inside the valid
                # (-b_sum_max, b_sum_max) region either way -- no encode/
                # decode exactness is needed for an initial condition.
                for l in self.lags:
                    a_key, b_key = f"A_{name}_{l}", f"B_{name}_{l}"
                    a_target, b_target = 0.01 / n_l, 0.90 / n_l
                    theta0[idx[a_key]] = self.codec.encode_ab_target(a_key, a_target)
                    theta0[idx[b_key]] = b_target
            else:
                for l in self.lags:
                    a_key, b_key = f"A_{name}_{l}", f"B_{name}_{l}"
                    a_target, b_target = 0.01 / n_l, 0.90 / n_l
                    if self.codec.bound_ab:
                        theta0[idx[a_key]] = self.codec.encode_ab_target(a_key, a_target)
                        theta0[idx[b_key]] = self.codec.encode_ab_target(b_key, b_target)
                    else:
                        theta0[idx[a_key]] = a_target
                        theta0[idx[b_key]] = b_target

        for s in self.static_names:
            s_guess = float(ucmle.get(s, _heuristic.get(s, 0.0)))
            if self.codec.bound_static_box:
                theta0[idx[s]] = self.codec.encode_static_box_target(_clip_into_box(s_guess))
            else:
                theta0[idx[s]] = s_guess

        if self.codec.free_zeta_gamma and "gamma" in self.static_names and "zeta" in self.static_names:
            gamma_target = theta0[idx["gamma"]]
            zeta_target = theta0[idx["zeta"]]
            margin = self.codec.kurtosis_margin
            # The unconditional MLE's own (gamma, zeta) may not satisfy the
            # margin (that is exactly the failure mode this reparameterisation
            # exists to remove) -- widen zeta_target just enough to be valid
            # rather than raising out of initial_theta(), which every caller
            # in this framework expects to always succeed.
            min_valid_zeta = gamma_target + np.log(margin) + 1e-3
            if zeta_target <= min_valid_zeta:
                zeta_target = min_valid_zeta
            theta0[idx["zeta"]] = self.codec.encode_zeta_target(gamma_target, zeta_target)

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
