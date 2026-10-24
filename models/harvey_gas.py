"""
Harvey Long-Short ZA-GAS model.

=============================================================================
MATHEMATICAL SPECIFICATION  (MODELS.md §19–21)
=============================================================================

For each time-varying parameter j ∈ {φ, ξ} (or any configured subset):

    f_{j,t}   =  ω_j  +  L_{j,t}  +  S_{j,t}

The combined state is the sum of two score-driven components.

Long component (persistent climate-scale dynamics):

    L_{j,t+1}  =  φ_{L,j} · L_{j,t}
               +  κ_{L,j} · s_{j,t}
               +  Γ_{L,j}' · X^{long}_t

Short component (local atmospheric dynamics):

    S_{j,t+1}  =  φ_{S,j} · S_{j,t}
               +  κ_{S,j} · s_{j,t}
               +  Γ_{S,j}' · X^{short}_t

Persistence restriction (ensures L is more persistent than S):

    0  <  φ_{S,j}  <  φ_{L,j}  <  1

Both components receive the SAME scaled score s_{j,t} at each time step.
The score is computed from the current predictive distribution defined by f_{j,t}.

NOTE ON VARIABLE NAMES: earlier versions of this module used A_L/B_L/gamma_L
(long) and A_S/B_S/gamma_S (short) instead of the kappa_L/phi_L/Gamma_L and
kappa_S/phi_S/Gamma_S names used throughout the code below. Every Python
identifier, dict key, and generated parameter-name string in this file now
matches MODELS.md §20 directly -- there is no separate "display" relabeling
layer. The ONE place the old A_L/B_L names still appear is the keyword
arguments passed to the external Numba kernels in models/_nb_gb2_filter.py
(nb_harvey_phi_only_filter, nb_harvey_phi_xi_filter), because those are
separately compiled functions with their own fixed signatures; see
_nb_harvey_dispatch for the translation at that boundary.

=============================================================================
THIS IS THE ONLY HARVEY IMPLEMENTATION
=============================================================================

f = ω + L + S with no separate GAS update on f itself; L and S are BOTH
score-driven recursions with their own φ and κ; the persistence restriction
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
    Update L_{j,t+1} = φ_L L_{j,t} + κ_L s_{j,t} + Γ_L' X_long_t
    ↓
    Update S_{j,t+1} = φ_S S_{j,t} + κ_S s_{j,t} + Γ_S' X_short_t
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
                         those are replaced by kappa_L, phi_L, kappa_S, phi_S below)
    pi_theta          :  occurrence model parameters
    harvey_theta      :  for each j in harvey_target (in tv_param_names order):
                             omega_j    — GAS intercept
                             f0_j       — initial state f_{j,0} (= L0 + S0 at t=0)
                             kappa_L_j  — long score response      (doc: κ_{L,j})
                             phi_L_j    — long persistence (< 1)   (doc: φ_{L,j})
                             L0_j       — initial long state
                             Gamma_L_j_0 … Gamma_L_j_{n_long-1}  (X_long coefficients, doc: Γ_{L,j})
                             kappa_S_j  — short score response     (doc: κ_{S,j})
                             phi_S_j    — short persistence (< phi_L_j)  (doc: φ_{S,j})
                             S0_j       — initial short state
                             Gamma_S_j_0 … Gamma_S_j_{n_short-1} (X_short coefficients, doc: Γ_{S,j})

Parameters NOT in harvey_target get a standard constant (no score-driven dynamics).

=============================================================================
CONFIGURABLE PENALTY FLAGS (constructor)
=============================================================================

restrict_persistence_upper (default True)
    Gates only the phi_L < 1 upper-bound penalty term. The ordering
    (phi_S < phi_L) and positivity (phi_S, phi_L > 0) penalties are always
    active regardless of this flag -- MODELS.md §20's persistence
    restriction is 0 < phi_S < phi_L < 1, and dropping only the "< 1" part
    is a deliberate experimental configuration, not a relaxation of the
    ordering requirement.

use_auxiliary_penalties (default True)
    Gates the score-amplitude (|kappa_L|,|kappa_S| < 2) and covariate-norm
    (||Gamma_L||,||Gamma_S|| < 20) and static-magnitude (|gamma|,|zeta| < 20)
    penalty terms. These are not required by MODELS.md/OPTIMIZATION.md; they
    were added as numerical guard rails for the production Stage-3 pipeline.
    Numerical safeguards that ARE mathematically required stay on
    unconditionally: the gamma < zeta GB2-moment-existence penalty
    (OPTIMIZATION.md §6), and the state/eta explosion guards
    (OPTIMIZATION.md §4-5 -- these prevent invalid likelihood evaluations,
    they are not statistical bounds on the model).

Both flags default to True so every existing call site (in particular
pipeline.runner.build_stage3_specs, which feeds reports/multi_location/)
reproduces its exact previous behaviour unchanged.

=============================================================================
OPTIONAL TRIGONOMETRIC SEASONAL COMPONENT  (n_harmonics > 0, 2026-09-03;
per-harmonic free initial conditions, 2026-09-04)
=============================================================================

Adds a THIRD additive score-driven component to the combined state, on top
of the existing Long/Short decomposition. Using lambda for the (log-link)
dynamic parameter being filtered (lambda = log(phi) or log(xi)):

    lambda_{j,t|t-1}  =  omega_j  +  lambda_{L,j,t|t-1}  +  lambda_{S,j,t|t-1}  +  gamma_{j,t|t-1}

    lambda_{L,j,t+1|t}  =  phi_{L,j} lambda_{L,j,t|t-1}  +  kappa_{L,j} s_{j,t}  +  c_{L,j}' x_{L,t}
    lambda_{S,j,t+1|t}  =  phi_{S,j} lambda_{S,j,t|t-1}  +  kappa_{S,j} s_{j,t}  +  c_{S,j}' x_{S,t}

    gamma_{j,t+1|t}     =  sum_{h=1}^{m}  gamma_{j,t+1|t,h}

    gamma_{j,t+1|t,h}   =  gamma_{j,t|t-1,h}   cos(2*pi*h/S)
                        +  gamma*_{j,t|t-1,h}  sin(2*pi*h/S)
                        +  kappa_{gamma,j} * s_{j,t}

    gamma*_{j,t+1|t,h}  = -gamma_{j,t|t-1,h}   sin(2*pi*h/S)
                        +  gamma*_{j,t|t-1,h}  cos(2*pi*h/S)
                        +  kappa_{gamma,j} * s_{j,t}

for h = 1..m, m <= floor(S/2). S=365.25 (annual cycle on daily data) is
the caller's responsibility to pass; this class only implements the
recursion for whatever (S, m) it is given. lambda_L/lambda_S here are what
the code calls L_{j,t}/S_{j,t}; c_{L,j}/c_{S,j} (the covariate-coefficient
vectors) are what the code and MODELS.md §20 call Gamma_L_j/Gamma_S_j --
that pre-existing "Gamma" name is UNCHANGED in code/CSV columns (it
predates this section and other reports/artifacts depend on it), but is
never used for the SEASONAL gamma in written mathematics or report tables,
to avoid exactly the collision it would otherwise create with the new
gamma_{j,t}/gamma*_{j,t,h} notation above.

Each (gamma_h, gamma*_h) pair is a PURE ROTATION (eigenvalues exactly on
the unit circle, |cos|+|sin| terms forming an orthogonal 2x2 matrix) driven
by the SAME shared scaled score s_{j,t} already computed for L and S at
every step -- architecturally a third score-driven component alongside the
existing two, not a new mechanism. Per direct specification, there is
deliberately NO damping/persistence factor on this rotation (unlike
phi_L/phi_S on L/S): a pure trigonometric seasonal is, by construction, an
INTEGRATED (unit-root-type) component, exactly analogous to Harvey's own
classical local-level trend and to this project's own standing discussion
of phi_L->1 not being inherently invalid, just requiring the existing
state-magnitude safety net (already applied to the COMBINED f_{j,t}, see
_run_filter) rather than a statistical bound of its own.

theta layout gains, per j in harvey_target, appended after Gamma_S_j (see
_block_size/parameter_names/_decode_harvey):
    kappa_gamma_j                                (1 scalar)
    seas0_j_h    for h=1..m   (gamma_{j,1|0,h}, FREE)
    seasstar0_j_h for h=1..m  (gamma*_{j,1|0,h}, FREE)

CORRECTION (2026-09-04): an earlier version of this code fixed
seas0_j_h/seasstar0_j_h at (0, 0) for every harmonic rather than making
them free parameters, incorrectly modelled as a single shared cold start
rather than m independent initial conditions (per direct correction: "each
m=1...5 has an initial parameter gamma_{1|0,m}, and a parameter
gamma_{1|0,m}*"). That version ALSO seeded them via the same array-index-0
write that L0_j/S0_j use -- since this class's recursion loop only ever
reads state from index self.max_lag onward (never index 0; see L0_j/S0_j's
own discussion below), a value written to index 0 has NO effect on the
recursion regardless of what it is, so even a hypothetical earlier attempt
to make them free would have silently optimised a parameter with zero
gradient almost everywhere. This is now fixed FOR THE SEASONAL COMPONENT
ONLY (see _run_filter/simulate_oos): seas0_j/seasstar0_j are seeded at
array index self.max_lag -- the position the loop's first executed
iteration (t=self.max_lag) actually reads -- so gamma_{j,1|0,h} truly is
"the seasonal state feeding the very first period the recursion computes,"
matching the equations above, and has a real, non-zero effect on the
likelihood and its gradient.

L0_j/S0_j are DELIBERATELY LEFT UNFIXED (still written to index 0, still
dead): both are ALSO written to index 0 of their state array while the
recursion loop only ever reads from index max_lag onward -- confirmed
identical in both this class's Python fallback and models/_nb_gb2_filter.py's
compiled kernel -- so L0_j/S0_j have had NO effect on any Harvey fit run in
this project to date. This predates the seasonal component entirely and is
flagged here, not fixed, since fixing it would silently change every past
Harvey result (production Stage-3 fits under reports/multi_location/
included). The seasonal component has no such history to preserve -- it
was introduced in this same 2026-09-03/04 work and no artifact outside
artifacts/harmonic_seasonal_experiment/ depends on its old (equally dead)
behaviour -- so it alone is corrected.

n_harmonics=0 (default) omits kappa_gamma_j/seas0_j_h/seasstar0_j_h
entirely and reproduces the exact previous theta layout and recursion --
every existing caller is unaffected.

UPDATE (2026-09-04): the harmonic recursion is now ALSO compiled into
models/_nb_gb2_filter.py's numba kernels (nb_harvey_phi_only_filter /
nb_harvey_phi_xi_filter both gained n_harmonics/cos_h/sin_h/kappa_gamma[_phi|_xi]
parameters), so _use_nb_filter no longer special-cases n_harmonics>0 --
the same eligibility rule as every other Harvey fit applies (pi_dynamics
type, distribution type, harvey_target count). This was cross-verified
against the pure-Python path (which remains the reference implementation
and is still reachable by forcing _use_nb_filter to return False) to
within the SAME floating-point noise floor already present in the
(unmodified) phi+xi branch's inverse-Fisher cross-scaling computation at
n_harmonics=0 -- i.e. this floor pre-dates the harmonic work entirely and
is not something the harmonic kernels introduce; phi-only matches to
machine precision (~1e-15) at every m tested. A single likelihood
evaluation that previously cost ~175ms in pure Python now costs
sub-millisecond, removing the need for the reduced search-phase
optimizer budget described in run_harmonic_seasonal_experiment.py's
PERFORMANCE note (that note is left as-is for the historical record of
why the search phase used a smaller budget; new fits should simply use
numba's normal, unreduced budget). The 2026-09-04 seas0/seasstar0 fix
above changed both nb_harvey_phi_only_filter's and nb_harvey_phi_xi_filter's
signatures (new seas0/seasstar0[_phi/_xi] arguments, seeding the local Gm/
GmStar loop state directly instead of always starting at zero) -- re-run
with genuinely nonzero seas0/seasstar0 to reconfirm parity, since the
original verification predates this change.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

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
        nb_gas1phi_harveyxi_filter,
    )
    _NB_AVAIL_HARVEY = True
except ImportError:
    pass

from models.mixed_score import SCALING_CODE as _HARVEY_SCALING_INT  # 0..4, see models/mixed_score.py
from models import mixed_score as MS

# ─────────────────────────────────────────────────────────────────────────────
# Per-parameter Harvey block size
#
#   omega_j (1) + f0_j (1) + kappa_L_j (1) + phi_L_j (1) + L0_j (1)
#   + Gamma_L (n_long) + kappa_S_j (1) + phi_S_j (1) + S0_j (1) + Gamma_S (n_short)
# ─────────────────────────────────────────────────────────────────────────────

def _block_size(n_long: int, n_short: int, fixed_phi_L: bool = False, fixed_phi_S: bool = False,
                 n_harmonics: int = 0, seasonal_damping: bool = False, include_long: bool = True) -> int:
    """
    Number of Harvey parameters per dynamic parameter j.

    fixed_phi_L / fixed_phi_S each remove their own slot from theta entirely
    (that persistence becomes a constant instead of an optimised value).
    Sigmoid-reparameterising phi_L or phi_S (via phi_L_max / phi_S_max on
    HarveyZAGASModel) does NOT change the slot count -- it reinterprets the
    same slot through a bounded transform, so no layout change is needed
    for that case.

    n_harmonics > 0 (see module docstring's "OPTIONAL TRIGONOMETRIC
    SEASONAL COMPONENT" section) appends kappa_gamma_j (1 scalar) plus TWO
    free initial-condition parameters per harmonic, seas0_j_h and
    seasstar0_j_h for h=1..n_harmonics (the equation's gamma_{1|0,h} and
    gamma*_{1|0,h}) -- 1 + 2*n_harmonics slots total. Unlike L0_j/S0_j
    (which remain a known dead write -- see module docstring -- left
    untouched to avoid silently changing any historical Harvey result),
    these ARE live: _run_filter/_nb_harvey_dispatch seed the seasonal
    recursion's first executed step directly from them (not through the
    same array-index-0 pattern that makes L0_j/S0_j inert). n_harmonics=0
    (default) adds nothing -- exact previous layout.
    """
    # omega + f0 + (kappa_L,phi_L,L0,Gamma_L) + (kappa_S,phi_S,S0,Gamma_S) --
    # the long block (3+n_long) is dropped entirely when include_long=False
    # (models.harvey_gas's structural-model ablation: single trend (S,
    # renamed) + seasonal, no long/climate-scale component at all -- see
    # HarveyZAGASModel's constructor docstring).
    base = 2 + (3 + n_long if include_long else 0) + 3 + n_short
    base -= int(bool(fixed_phi_L)) + int(bool(fixed_phi_S))
    if n_harmonics > 0:
        base += 1 + 2 * n_harmonics  # kappa_gamma + (seas0_h, seasstar0_h) for h=1..n_harmonics
        if seasonal_damping:
            base += 1  # psi_gamma_j -- see seasonal_damping_max's constructor docstring
    return base


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


def persistence_init_from_filtered_path(f_arr_j: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    """
    Data-driven persistence-coefficient initialisation from a prior model's
    filtered dynamic-parameter path.

    Computes r_t = f_{t+1} / f_t over an already-fitted (no re-optimisation)
    model's in-sample filtered path for one dynamic parameter j, and returns
    mean(r_t) as a candidate initial value for the Harvey short-component
    persistence phi_{S,j} (MODELS.md §20 notation; code identifier phi_S).
    Terms with |f_t| < eps are masked (the ratio is numerically meaningless
    there) rather than silently producing inf/nan.

    This is an experimental initialisation heuristic for
    run_longshort_diagnostics.py, not a generally-applicable estimator: a
    single-lag ratio of a GAS filtered path is not a persistence estimate in
    any formal sense, it is only a plausible order-of-magnitude starting
    point. Callers must inspect ratio_mean/ratio_std for sanity before use
    (see the blow-up fallback in run_longshort_diagnostics.py).

    Returns
    -------
    dict with keys:
        ratio_mean, ratio_std : float  (nan if no valid terms)
        n_terms   : number of t with both f_t and f_{t+1} finite
        n_masked  : number of those terms dropped for |f_t| < eps
    """
    f = np.asarray(f_arr_j, dtype=float).ravel()
    if f.size < 2:
        return {"ratio_mean": np.nan, "ratio_std": np.nan, "n_terms": 0, "n_masked": 0}

    f_t   = f[:-1]
    f_tp1 = f[1:]
    finite = np.isfinite(f_t) & np.isfinite(f_tp1)
    n_terms = int(np.sum(finite))

    small = finite & (np.abs(f_t) < eps)
    n_masked = int(np.sum(small))

    valid = finite & ~small
    if not np.any(valid):
        return {"ratio_mean": np.nan, "ratio_std": np.nan, "n_terms": n_terms, "n_masked": n_masked}

    ratios = f_tp1[valid] / f_t[valid]
    return {
        "ratio_mean": float(np.mean(ratios)),
        "ratio_std":  float(np.std(ratios)),
        "n_terms":    n_terms,
        "n_masked":   n_masked,
    }


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
        gas1_target: Optional[List[str]] = None,
        scaling: str = "diagonal_inverse_fisher",
        static_params: Optional[List[str]] = None,
        restrict_persistence_upper: bool = True,
        use_auxiliary_penalties: bool = True,
        fixed_phi_L: Optional[float] = None,
        phi_L_max: Optional[float] = None,
        phi_S_max: Optional[float] = None,
        fixed_phi_S: Optional[float] = None,
        raw_z_ridge: float = 0.0,
        # Trigonometric seasonal component (default off -- see module
        # docstring's "OPTIONAL TRIGONOMETRIC SEASONAL COMPONENT" section).
        n_harmonics: int = 0,
        seasonal_period: float = 365.25,
        kappa_gamma_bound: Optional[float] = None,
        seasonal_damping_max: Optional[float] = None,
        # include_long=False (default True, reproduces exact previous
        # behaviour): drops the Long component (kappa_L, phi_L, L0, Gamma_L)
        # entirely -- no theta slots, no recursion contribution (all treated
        # as the constant 0) -- leaving f_{j,t} = omega_j + S_{j,t} [+
        # seasonal_t]. This is Harvey's own classical structural
        # (local-level/trend + seasonal) model rather than the two-component
        # long/short decomposition: a single persistence component (S,
        # renamed "trend" in this configuration) plus, when n_harmonics>0,
        # the same trigonometric seasonal component used elsewhere in this
        # class. Added 2026-09-08 to test whether removing the long
        # component (a documented source of near-unit-root identification
        # weakness at some locations) improves fit/stability when combined
        # with the seasonal harmonic.
        include_long: bool = True,
    ):
        from models.gas_filter import GASFilter, _VALID_SCALING, MIXED_SCALINGS
        from models import mixed_score as MS

        if scaling not in _VALID_SCALING and scaling not in MIXED_SCALINGS:
            raise ValueError(
                f"scaling must be one of {sorted(_VALID_SCALING | MIXED_SCALINGS)}, got '{scaling!r}'"
            )
        MS.check_scaling(pi_dynamics, scaling)
        if MS.pi_uses_phi(pi_dynamics) and gas1_target:
            raise ValueError("phi-linked pi dynamics are not implemented together with gas1_target")
        if fixed_phi_L is not None and phi_L_max is not None:
            raise ValueError(
                "fixed_phi_L and phi_L_max are mutually exclusive parameterisations of "
                "phi_L (constant vs. sigmoid-bounded) -- set at most one."
            )
        if fixed_phi_S is not None and phi_S_max is not None:
            raise ValueError(
                "fixed_phi_S and phi_S_max are mutually exclusive parameterisations of "
                "phi_S (constant vs. sigmoid-bounded) -- set at most one."
            )

        self.dist      = distribution
        self.pi_dyn    = pi_dynamics
        self.seasonal  = seasonal
        self.scaling   = scaling

        # See module docstring "CONFIGURABLE PENALTY FLAGS" for exactly what
        # each flag gates. Both default True: unchanged behaviour for every
        # existing caller (pipeline.runner.build_stage3_specs).
        self.restrict_persistence_upper = restrict_persistence_upper
        self.use_auxiliary_penalties    = use_auxiliary_penalties

        # Three optional reparameterisations of phi_L/phi_S, all motivated by
        # the near-unit-root gradient/standard-error breakdown documented in
        # reports/longshort experiment/ -- MODELS.md §20's raw persistence
        # penalty lets phi_L drift to (or past) a unit root, which makes the
        # recursion's sensitivity to kappa_L/Gamma_L/phi_L itself explode
        # (near-integrated-process pathology). All three defaults (None)
        # reproduce the exact previous raw/penalty-based behaviour.
        #
        # fixed_phi_L: phi_L held at this CONSTANT for every harvey_target
        #   parameter and removed from theta entirely (one fewer slot per
        #   parameter -- see _block_size). A literal integrated/random-walk
        #   long component, as in Harvey's own local-level structural time
        #   series models: sidesteps the identifiability question for the
        #   long component entirely, leaving only the short component's
        #   dynamics to be estimated. Mutually exclusive with phi_L_max.
        #
        # phi_L_max / phi_S_max: phi_L (resp. phi_S) reparameterised as
        #   phi = max_value * sigmoid(z), z the actual optimised quantity in
        #   theta, guaranteeing 0 < phi < max_value BY CONSTRUCTION rather
        #   than by a penalty (OPTIMIZATION.md §6: "If a reparameterisation
        #   is used, document the mapping clearly" -- this is that
        #   documentation). z is the raw value that appears in
        #   estimated_parameters.csv under the existing phi_L_{j}/phi_S_{j}
        #   names; the decoded (natural-scale) value is what
        #   filter()/_run_filter actually use in the recursion, and is also
        #   written to decoded_harvey_parameters.csv by save_result(). The
        #   phi_S < phi_L ordering restriction (MODELS.md §20) still applies
        #   as a soft penalty on the DECODED values -- two independent
        #   sigmoids do not order themselves.
        self.fixed_phi_L = fixed_phi_L
        self.phi_L_max   = phi_L_max
        self.phi_S_max   = phi_S_max
        # fixed_phi_S: mirrors fixed_phi_L on the short component -- used by
        # the profile-likelihood stability experiment
        # (run_stability_experiment.py) to grid-search the persistence pair
        # (phi_L, phi_S) directly rather than relying on gradient-based
        # optimisation through the sigmoid, which can develop a vanishing
        # derivative near saturation (see
        # reports/longshort experiment/stability_investigation_log.md).
        self.fixed_phi_S = fixed_phi_S

        # kappa_gamma_bound: kappa_gamma_j reparameterised as
        #   kappa_gamma = bound * (2*sigmoid(z) - 1), z the raw optimised
        #   quantity in theta, guaranteeing -bound < kappa_gamma < bound BY
        #   CONSTRUCTION -- the SAME phi_L_max/phi_S_max pattern above, just
        #   symmetric about 0 instead of one-sided (kappa_gamma has no sign
        #   restriction, unlike a persistence parameter). None (default)
        #   reproduces the exact previous raw/unbounded theta layout and
        #   recursion. Introduced 2026-09-05 to test whether bounding
        #   kappa_gamma's magnitude prevents the optimiser from landing on
        #   the large values (found, via direct investigation, to be
        #   strongly correlated with phi_L/the ENSO Long covariates --
        #   hess_inv condition numbers of 1e11-1e12) that make the
        #   undamped seasonal recursion diverge out-of-sample -- see
        #   reports/harmonic seasonal harvey/report.tex's Conclusion.
        self.kappa_gamma_bound = kappa_gamma_bound
        self.include_long = include_long

        # seasonal_damping_max: DEVIATES from the pure-rotation equations
        #   (module docstring / MODELS.md Section 20.1) on direct instruction.
        #   Adds one more free parameter per j, psi_gamma_j = damping_max *
        #   sigmoid(z), 0 < psi_gamma_j < damping_max <= 1, multiplying the
        #   ENTIRE rotation each step:
        #     gamma_{t+1,h}  = psi_gamma_j * (gamma_{t,h}*cos + gamma*_{t,h}*sin) + kappa_gamma_j*s_t
        #     gamma*_{t+1,h} = psi_gamma_j * (-gamma_{t,h}*sin + gamma*_{t,h}*cos) + kappa_gamma_j*s_t
        #   This moves the rotation's eigenvalues from EXACTLY on the unit
        #   circle (modulus 1, integrated -- the specified model) to modulus
        #   psi_gamma_j < 1 (a damped, mean-reverting rotation, exactly the
        #   phi_L/phi_S<1 pattern already used for L/S). Introduced 2026-09-05
        #   specifically to test whether an undamped seasonal component is
        #   what makes simulate_oos's real-data-conditioned one-step-ahead
        #   recursion numerically explode over a ~2-year test window (see
        #   reports/harmonic seasonal harvey/report.tex's Conclusion: genuine
        #   Monte Carlo multi-step forecasts from the SAME fitted models stay
        #   perfectly sane, so the instability is specific to feeding real
        #   y_test into an undamped state -- damping removes the "no restoring
        #   force" property that makes that specific failure mode possible).
        #   None (default) reproduces the exact previous pure-rotation
        #   recursion and theta layout -- every existing caller unaffected.
        self.seasonal_damping_max = seasonal_damping_max

        # raw_z_ridge: an L2 penalty lambda*z**2 applied to the RAW
        # pre-sigmoid coordinate z itself (only meaningful when
        # phi_L_max/phi_S_max is active; ignored -- and harmless -- for
        # fixed_phi_L/fixed_phi_S and for the raw/unreparameterised case).
        #
        # Why this has to live in z-space, not phi-space: chain rule gives
        #   d(penalty)/dz = d(penalty)/dphi_L * dphi_L/dz.
        # Once the optimiser has driven z far into the sigmoid's saturation
        # tail (|z| large), dphi_L/dz = phi_L_max*sigmoid'(z) -> 0, so ANY
        # penalty defined as a function of the DECODED phi_L -- however
        # strong, including a divergent log-barrier -- still has a
        # vanishing gradient in the coordinate BFGS actually steps in.
        # Empirically this is exactly what was observed for Darwin Airport's
        # phi_{L,phi} parameter (reports/longshort experiment/
        # stability_investigation_log.md): phi_L saturates to 1.000000 in
        # every sigmoid_both fit at that location regardless of Gamma_L
        # init or penalty strength on the decoded scale.
        #
        # A penalty on z directly sidesteps this: d(lambda*z**2)/dz = 2*lambda*z
        # never vanishes, so it keeps providing a restoring force that pulls
        # the optimiser back out of the saturation tail before phi_L(z) can
        # round to exactly 1.0 in floating point -- while still leaving z
        # free to settle anywhere the (much better-behaved, non-vanishing)
        # likelihood gradient and this ridge jointly prefer. lambda=0.0
        # (default) reproduces the exact previous sigmoid_both behaviour.
        self.raw_z_ridge = float(raw_z_ridge)

        # Trigonometric seasonal (n_harmonics=0 reproduces exact previous
        # behaviour -- see module docstring). cos_h/sin_h (length m) are
        # precomputed ONCE here since they depend only on (h, S), never on
        # theta or t -- see the recursion in _run_filter/simulate_oos.
        self.n_harmonics = int(n_harmonics)
        self.seasonal_period = float(seasonal_period)
        if self.n_harmonics > 0:
            h = np.arange(1, self.n_harmonics + 1, dtype=float)
            angle = 2.0 * np.pi * h / self.seasonal_period
            self._cos_h = np.cos(angle)
            self._sin_h = np.sin(angle)
        else:
            self._cos_h = self._sin_h = np.array([])

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

        # gas1_target: TV params that follow a PLAIN, single-lag GAS(1,1)
        # recursion -- f_{t+1} = omega + A*s_t + B*f_t, no covariates, no
        # long/short decomposition, no seasonal component -- instead of
        # either the Harvey treatment or a frozen constant. Added 2026-09-11
        # for the "phi plain GAS(1,1), xi Harvey-driven" hybrid Stage-5
        # variant requested directly. Mutually exclusive with harvey_target
        # (a parameter cannot be both); silently restricted to actual TV
        # params, matching harvey_target's own convention. Empty by default
        # -- every existing caller (which never passes gas1_target) is
        # completely unaffected: self.gas1_target == [] short-circuits every
        # new branch this flag adds in _run_filter/simulate_oos/
        # initial_theta/parameter_names below.
        overlap = set(gas1_target or []) & set(self.harvey_target)
        if overlap:
            raise ValueError(f"gas1_target and harvey_target overlap: {sorted(overlap)}")
        self.gas1_target = [j for j in (gas1_target or []) if j in self.tv_names]

        self._blk = _block_size(
            self.n_long, self.n_short,
            fixed_phi_L=self.fixed_phi_L is not None,
            fixed_phi_S=self.fixed_phi_S is not None,
            n_harmonics=self.n_harmonics,
            seasonal_damping=self.seasonal_damping_max is not None,
            include_long=self.include_long,
        )

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

    _BLK_GAS1 = 4  # omega, f0, A, B -- see gas1_target in __init__

    @property
    def n_gas1(self) -> int:
        return len(self.gas1_target) * self._BLK_GAS1

    @property
    def n_params(self) -> int:
        return self.n_static + self.n_pi + self.n_harvey + self.n_gas1

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
        Return {j: {omega, f0, kappa_L, phi_L, L0, Gamma_L, kappa_S, phi_S, S0, Gamma_S}}.

        Names match MODELS.md §20 directly: kappa = score response,
        phi = persistence, Gamma = covariate effects. phi_L/phi_S here are
        always the DECODED (natural-scale, 0<phi<1 when reparameterised)
        values used by the recursion -- see the constructor docstring for
        fixed_phi_L/phi_L_max/phi_S_max. raw_phi_L/raw_phi_S are the
        pre-sigmoid theta values (None when fixed or not sigmoid-
        reparameterised) -- exposed so _run_filter can apply raw_z_ridge
        directly in the coordinate the optimiser actually sees; see that
        flag's docstring for why the penalty must live there and not on the
        decoded phi_L/phi_S value.

        Layout per j (in harvey_target order; walked with a running index
        rather than fixed offsets because fixed_phi_L removes one slot):
            omega_j, f0_j, kappa_L_j, [phi_L_j unless fixed_phi_L], L0_j,
            Gamma_L_j (n_long), kappa_S_j, phi_S_j, S0_j, Gamma_S_j (n_short)
        """
        result = {}
        blk = self._blk
        for k, j in enumerate(self.harvey_target):
            b = harvey_theta[k * blk : (k + 1) * blk]
            i = 0
            omega = float(b[i]); i += 1
            f0    = float(b[i]); i += 1
            raw_L = None
            if not self.include_long:
                # No theta slots at all for the long block -- treated as the
                # constant 0 throughout (see constructor docstring).
                kappa_L, phi_L, L0 = 0.0, 0.0, 0.0
                Gamma_L = np.zeros(self.n_long)
            else:
                kappa_L = float(b[i]); i += 1
                if self.fixed_phi_L is not None:
                    phi_L = float(self.fixed_phi_L)
                else:
                    raw_L = float(b[i]); i += 1
                    phi_L = self.phi_L_max * _sigmoid_scalar(raw_L) if self.phi_L_max is not None else raw_L
                L0 = float(b[i]); i += 1
                Gamma_L = b[i : i + self.n_long]; i += self.n_long
            kappa_S = float(b[i]); i += 1
            raw_S = None
            if self.fixed_phi_S is not None:
                phi_S = float(self.fixed_phi_S)
            else:
                raw_S = float(b[i]); i += 1
                phi_S = self.phi_S_max * _sigmoid_scalar(raw_S) if self.phi_S_max is not None else raw_S
            S0 = float(b[i]); i += 1
            Gamma_S = b[i : i + self.n_short]; i += self.n_short
            kappa_gamma = None
            raw_kappa_gamma = None
            seas0 = seasstar0 = None
            if self.n_harmonics > 0:
                raw_kg = float(b[i]); i += 1
                if self.kappa_gamma_bound is not None:
                    raw_kappa_gamma = raw_kg
                    kappa_gamma = self.kappa_gamma_bound * (2.0 * _sigmoid_scalar(raw_kg) - 1.0)
                else:
                    kappa_gamma = raw_kg
                # Free theta entries (gamma_{1|0,h} / gamma*_{1|0,h} in the
                # equations) -- see module docstring's "OPTIONAL
                # TRIGONOMETRIC SEASONAL COMPONENT" section for why these
                # are live, unlike L0_j/S0_j.
                seas0     = np.asarray(b[i : i + self.n_harmonics], dtype=float); i += self.n_harmonics
                seasstar0 = np.asarray(b[i : i + self.n_harmonics], dtype=float); i += self.n_harmonics
            psi_gamma = None
            raw_psi_gamma = None
            if self.n_harmonics > 0 and self.seasonal_damping_max is not None:
                raw_psi = float(b[i]); i += 1
                raw_psi_gamma = raw_psi
                psi_gamma = self.seasonal_damping_max * _sigmoid_scalar(raw_psi)
            result[j] = {
                "raw_phi_L": raw_L, "raw_phi_S": raw_S,
                "omega": omega, "f0": f0, "kappa_L": kappa_L, "phi_L": phi_L, "L0": L0,
                "Gamma_L": Gamma_L, "kappa_S": kappa_S, "phi_S": phi_S, "S0": S0, "Gamma_S": Gamma_S,
                "kappa_gamma": kappa_gamma, "raw_kappa_gamma": raw_kappa_gamma,
                "psi_gamma": psi_gamma, "raw_psi_gamma": raw_psi_gamma,
                "seas0": seas0, "seasstar0": seasstar0,
            }
        return result

    def _decode_gas1(self, gas1_theta: np.ndarray) -> Dict[str, dict]:
        """{j: {omega, f0, A, B}} for j in self.gas1_target -- plain GAS(1,1),
        see gas1_target in __init__. gas1_theta is the TAIL of the harvey
        theta segment (theta[b:]) left over after _decode_harvey consumes
        the first len(harvey_target)*self._blk entries -- callers pass
        harvey_theta[len(self.harvey_target)*self._blk:]."""
        result = {}
        for k, j in enumerate(self.gas1_target):
            b = gas1_theta[k * self._BLK_GAS1: (k + 1) * self._BLK_GAS1]
            result[j] = {"omega": float(b[0]), "f0": float(b[1]), "A": float(b[2]), "B": float(b[3])}
        return result

    def parameter_names(self) -> List[str]:
        """
        Flat theta parameter names in optimizer order, matching MODELS.md
        §20 notation directly (kappa = score response, phi = persistence,
        Gamma = covariate effects) -- these are the exact column names
        written to estimated_parameters.csv / standard_errors.csv.

        When fixed_phi_L is set, phi_L_{j} is omitted (it is not part of
        theta). When phi_L_max/phi_S_max is set, phi_L_{j}/phi_S_{j} still
        name the theta slot, but its VALUE is the pre-sigmoid raw quantity,
        not the natural-scale persistence -- see save_result()'s
        decoded_harvey_parameters.csv for the decoded values. Same for
        kappa_gamma_{j} when kappa_gamma_bound is set: the theta slot holds
        the pre-sigmoid raw value, and decoded_harvey_parameters.csv holds
        kappa_gamma = kappa_gamma_bound*(2*sigmoid(raw)-1).
        """
        names = list(self.static_names) + list(self._pi_names)
        for j in self.harvey_target:
            names += [f"omega_{j}", f"f0_{j}"]
            if self.include_long:
                names.append(f"kappa_L_{j}")
                if self.fixed_phi_L is None:
                    names.append(f"phi_L_{j}")
                names.append(f"L0_{j}")
                names += [f"Gamma_L_{j}_{c}" for c in self.long_names]
            names.append(f"kappa_S_{j}")
            if self.fixed_phi_S is None:
                names.append(f"phi_S_{j}")
            names.append(f"S0_{j}")
            names += [f"Gamma_S_{j}_{c}" for c in self.short_names]
            if self.n_harmonics > 0:
                names.append(f"kappa_gamma_{j}")
                names += [f"seas0_{j}_{h}" for h in range(1, self.n_harmonics + 1)]
                names += [f"seasstar0_{j}_{h}" for h in range(1, self.n_harmonics + 1)]
                if self.seasonal_damping_max is not None:
                    names.append(f"psi_gamma_{j}")
        for j in self.gas1_target:
            names += [f"omega_{j}", f"f0_{j}", f"A_{j}_1", f"B_{j}_1"]
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
        # 2026-09-04: the compiled kernels in models/_nb_gb2_filter.py now
        # DO support the trigonometric seasonal component (n_harmonics>0),
        # numba-accelerated in place -- verified bit-for-bit against the
        # pure-Python fallback below (test_nb_harmonic_parity.py) before
        # being trusted. Originally this returned False unconditionally for
        # n_harmonics>0 while only the Python path existed; that path
        # remains as the (now redundant, but still correct and still used
        # if _NB_AVAIL_HARVEY is False) fallback.
        if not _NB_AVAIL_HARVEY:
            return False
        if self.gas1_target:
            # gas1_target (2026-09-11): a compiled kernel exists ONLY for the
            # specific "phi plain GAS(1,1), xi Harvey-driven" combination
            # (nb_gas1phi_harveyxi_filter) -- everything else with
            # gas1_target set (any other pairing, or >1 gas1/harvey target)
            # falls back to the pure-Python path.
            return self.gas1_target == ["phi"] and self.harvey_target == ["xi"]
        from pi_dynamics.ar_logistic import ARLogisticPiDynamics
        from pi_dynamics.ar_logistic_custom_lags import ARLogisticCustomLagsPiDynamics
        from pi_dynamics.phi_linked import PhiLinkedJointPiDynamics
        if not isinstance(self.pi_dyn, (ARLogisticPiDynamics, ARLogisticCustomLagsPiDynamics,
                                        PhiLinkedJointPiDynamics)):
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
        gas1d: Optional[dict] = None,
    ):
        lags_pi, pi_wy, pi_omega0, pi_rho, pi_lam_phi, pi_link = MS.nb_pi_args(
            self.pi_dyn, pip, SEASONAL_LAGS[self.seasonal])
        scaling_i  = _HARVEY_SCALING_INT.get(self.scaling, 0)

        Xl = np.nan_to_num(X_long,  nan=0.0).astype(np.float64)
        Xs = np.nan_to_num(X_short, nan=0.0).astype(np.float64)
        zeros_T = np.zeros(T, dtype=np.float64)
        n_tv  = len(self.tv_names)
        eff   = self.max_lag

        if self.gas1_target:
            # ── gas1(phi) + Harvey(xi) hybrid (2026-09-11) ──────────────────
            gd  = gas1d["phi"]
            hxi = hvd["xi"]
            CL_xi = (Xl @ hxi["Gamma_L"].astype(np.float64)) if self.n_long  > 0 else zeros_T
            CS_xi = (Xs @ hxi["Gamma_S"].astype(np.float64)) if self.n_short > 0 else zeros_T
            gamma_s = float(sp["gamma"])
            zeta_s  = float(sp["zeta"])

            loglik_nb, loop_pen, G_phi, L_xi, S_xi, s_phi, s_xi, eta_nb, pi_nb, Gamma_xi_nb = (
                nb_gas1phi_harveyxi_filter(
                    y.astype(np.float64),
                    omega_phi=float(gd["omega"]), G0_phi=float(gd["f0"]),
                    A_phi=float(gd["A"]), B_phi=float(gd["B"]),
                    omega_xi=float(hxi["omega"]),
                    L0_xi=float(hxi["L0"]), S0_xi=float(hxi["S0"]),
                    A_L_xi=float(hxi["kappa_L"]), B_L_xi=float(hxi["phi_L"]),
                    A_S_xi=float(hxi["kappa_S"]), B_S_xi=float(hxi["phi_S"]),
                    gamma_s=gamma_s, zeta_s=zeta_s,
                    pi_omega0=pi_omega0, pi_rho=pi_rho,
                    pi_wy=pi_wy, lags_pi=lags_pi, pi_link=pi_link,
                    max_lag=self.max_lag, scaling=scaling_i,
                    lam_state=1e3, state_lim=15.0,
                    lam_eta=1e2,   eta_lim=30.0,
                    CL_xi=CL_xi, CS_xi=CS_xi,
                    n_harmonics=self.n_harmonics, cos_h=self._cos_h, sin_h=self._sin_h,
                    kappa_gamma_xi=float(hxi["kappa_gamma"]) if self.n_harmonics > 0 else 0.0,
                    seas0_xi=hxi["seas0"] if self.n_harmonics > 0 else np.zeros(0),
                    seasstar0_xi=hxi["seasstar0"] if self.n_harmonics > 0 else np.zeros(0),
                    psi_gamma_xi=float(hxi["psi_gamma"]) if hxi.get("psi_gamma") is not None else 1.0,
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
            o_xi    = float(hxi["omega"])
            f_out   = np.zeros((n_eff, n_tv))
            L_out   = np.zeros((n_eff, n_tv))
            S_out   = np.zeros((n_eff, n_tv))
            s_out   = np.zeros((n_eff, n_tv))
            Gamma_out = np.zeros((n_eff, n_tv))
            for ii in range(n_eff):
                t = eff + ii
                f_out[ii, phi_idx] = G_phi[t]
                f_out[ii, xi_idx]  = o_xi + L_xi[t] + S_xi[t] + Gamma_xi_nb[t]
                L_out[ii, xi_idx]  = L_xi[t]; S_out[ii, xi_idx] = S_xi[t]
                s_out[ii, phi_idx] = s_phi[t]
                s_out[ii, xi_idx]  = s_xi[t]
                Gamma_out[ii, xi_idx] = Gamma_xi_nb[t]
            out = {
                "f_arr": f_out, "L_arr": L_out, "S_arr": S_out, "s_arr": s_out,
                "eta": eta_nb, "pi": pi_nb,
                "tv_names": self.tv_names, "static": dict(sp),
                "eff_start": eff, "loglik": float(loglik_nb),
                "penalty": total_pen, "objective": obj,
                "y_eff": y[eff:T], "harvey": hvd,
            }
            if self.n_harmonics > 0:
                out["Gamma_arr"] = Gamma_out
            return out

        eff   = self.max_lag
        n_tv  = len(self.tv_names)

        if len(self.harvey_target) == 1:
            # ── phi-only Harvey ──────────────────────────────────────────────
            j  = self.harvey_target[0]
            hj = hvd[j]
            CL = (Xl @ hj["Gamma_L"].astype(np.float64)) if self.n_long  > 0 else zeros_T
            CS = (Xs @ hj["Gamma_S"].astype(np.float64)) if self.n_short > 0 else zeros_T

            xi_s    = float(sp.get("xi",    0.0))
            gamma_s = float(sp["gamma"])
            zeta_s  = float(sp["zeta"])

            # models/_nb_gb2_filter.py is a separately-compiled Numba kernel
            # with its own fixed A_L/B_L/A_S/B_S parameter names; translate
            # our kappa_L/phi_L/kappa_S/phi_S dict keys to those names only
            # at this call boundary.
            loglik_nb, loop_pen, L_arr, S_arr, s_arr, eta_nb, pi_nb, Gamma_nb = (
                nb_harvey_phi_only_filter(
                    y.astype(np.float64),
                    omega=float(hj["omega"]),
                    L0=float(hj["L0"]), A_L=float(hj["kappa_L"]), B_L=float(hj["phi_L"]),
                    S0=float(hj["S0"]), A_S=float(hj["kappa_S"]), B_S=float(hj["phi_S"]),
                    xi_s=xi_s, gamma_s=gamma_s, zeta_s=zeta_s,
                    pi_omega0=pi_omega0, pi_rho=pi_rho,
                    pi_wy=pi_wy, lags_pi=lags_pi, pi_link=pi_link,
                    max_lag=self.max_lag, scaling=scaling_i,
                    lam_state=1e3, state_lim=15.0,
                    lam_eta=1e2,   eta_lim=30.0,
                    CL=CL, CS=CS,
                    n_harmonics=self.n_harmonics, cos_h=self._cos_h, sin_h=self._sin_h,
                    kappa_gamma=float(hj["kappa_gamma"]) if self.n_harmonics > 0 else 0.0,
                    seas0=hj["seas0"] if self.n_harmonics > 0 else np.zeros(0),
                    seasstar0=hj["seasstar0"] if self.n_harmonics > 0 else np.zeros(0),
                    psi_gamma=float(hj["psi_gamma"]) if hj.get("psi_gamma") is not None else 1.0,
                    pi_lam_phi=pi_lam_phi,
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
            Gamma_out = np.zeros((n_eff, n_tv))
            for ii in range(n_eff):
                t = eff + ii
                f_out[ii, j_idx] = omega_j + L_arr[t] + S_arr[t] + Gamma_nb[t]
                L_out[ii, j_idx] = L_arr[t]
                S_out[ii, j_idx] = S_arr[t]
                s_out[ii, j_idx] = s_arr[t]
                Gamma_out[ii, j_idx] = Gamma_nb[t]
            out = {
                "f_arr": f_out, "L_arr": L_out, "S_arr": S_out, "s_arr": s_out,
                "eta": eta_nb, "pi": pi_nb,
                "tv_names": self.tv_names, "static": dict(sp),
                "eff_start": eff, "loglik": float(loglik_nb),
                "penalty": total_pen, "objective": obj,
                "y_eff": y[eff:T], "harvey": hvd,
            }
            if self.n_harmonics > 0:
                out["Gamma_arr"] = Gamma_out
            return out

        else:
            # ── phi+xi Harvey ────────────────────────────────────────────────
            hphi = hvd["phi"]
            hxi  = hvd["xi"]
            CL_phi = (Xl @ hphi["Gamma_L"].astype(np.float64)) if self.n_long  > 0 else zeros_T
            CS_phi = (Xs @ hphi["Gamma_S"].astype(np.float64)) if self.n_short > 0 else zeros_T
            CL_xi  = (Xl @ hxi["Gamma_L"].astype(np.float64))  if self.n_long  > 0 else zeros_T
            CS_xi  = (Xs @ hxi["Gamma_S"].astype(np.float64))  if self.n_short > 0 else zeros_T
            gamma_s = float(sp["gamma"])
            zeta_s  = float(sp["zeta"])

            # Same external-kernel boundary translation as the phi-only branch above.
            loglik_nb, loop_pen, L_phi, S_phi, L_xi, S_xi, s_phi, s_xi, eta_nb, pi_nb, Gamma_phi_nb, Gamma_xi_nb = (
                nb_harvey_phi_xi_filter(
                    y.astype(np.float64),
                    omega_phi=float(hphi["omega"]), omega_xi=float(hxi["omega"]),
                    L0_phi=float(hphi["L0"]), S0_phi=float(hphi["S0"]),
                    A_L_phi=float(hphi["kappa_L"]), B_L_phi=float(hphi["phi_L"]),
                    A_S_phi=float(hphi["kappa_S"]), B_S_phi=float(hphi["phi_S"]),
                    L0_xi=float(hxi["L0"]),  S0_xi=float(hxi["S0"]),
                    A_L_xi=float(hxi["kappa_L"]),  B_L_xi=float(hxi["phi_L"]),
                    A_S_xi=float(hxi["kappa_S"]),  B_S_xi=float(hxi["phi_S"]),
                    gamma_s=gamma_s, zeta_s=zeta_s,
                    pi_omega0=pi_omega0, pi_rho=pi_rho,
                    pi_wy=pi_wy, lags_pi=lags_pi, pi_link=pi_link,
                    max_lag=self.max_lag, scaling=scaling_i,
                    lam_state=1e3, state_lim=15.0,
                    lam_eta=1e2,   eta_lim=30.0,
                    CL_phi=CL_phi, CS_phi=CS_phi,
                    CL_xi=CL_xi,   CS_xi=CS_xi,
                    n_harmonics=self.n_harmonics, cos_h=self._cos_h, sin_h=self._sin_h,
                    kappa_gamma_phi=float(hphi["kappa_gamma"]) if self.n_harmonics > 0 else 0.0,
                    kappa_gamma_xi=float(hxi["kappa_gamma"]) if self.n_harmonics > 0 else 0.0,
                    seas0_phi=hphi["seas0"] if self.n_harmonics > 0 else np.zeros(0),
                    seasstar0_phi=hphi["seasstar0"] if self.n_harmonics > 0 else np.zeros(0),
                    seas0_xi=hxi["seas0"] if self.n_harmonics > 0 else np.zeros(0),
                    seasstar0_xi=hxi["seasstar0"] if self.n_harmonics > 0 else np.zeros(0),
                    psi_gamma_phi=float(hphi["psi_gamma"]) if hphi.get("psi_gamma") is not None else 1.0,
                    psi_gamma_xi=float(hxi["psi_gamma"]) if hxi.get("psi_gamma") is not None else 1.0,
                    pi_lam_phi=pi_lam_phi,
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
            Gamma_out = np.zeros((n_eff, n_tv))
            for ii in range(n_eff):
                t = eff + ii
                f_out[ii, phi_idx] = o_phi + L_phi[t] + S_phi[t] + Gamma_phi_nb[t]
                f_out[ii, xi_idx]  = o_xi  + L_xi[t]  + S_xi[t]  + Gamma_xi_nb[t]
                L_out[ii, phi_idx] = L_phi[t];  S_out[ii, phi_idx] = S_phi[t]
                L_out[ii, xi_idx]  = L_xi[t];   S_out[ii, xi_idx]  = S_xi[t]
                s_out[ii, phi_idx] = s_phi[t]
                s_out[ii, xi_idx]  = s_xi[t]
                Gamma_out[ii, phi_idx] = Gamma_phi_nb[t]
                Gamma_out[ii, xi_idx]  = Gamma_xi_nb[t]
            out = {
                "f_arr": f_out, "L_arr": L_out, "S_arr": S_out, "s_arr": s_out,
                "eta": eta_nb, "pi": pi_nb,
                "tv_names": self.tv_names, "static": dict(sp),
                "eff_start": eff, "loglik": float(loglik_nb),
                "penalty": total_pen, "objective": obj,
                "y_eff": y[eff:T], "harvey": hvd,
            }
            if self.n_harmonics > 0:
                out["Gamma_arr"] = Gamma_out
            return out

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
            L_{j,t+1} = phi_L_j * L_{j,t} + kappa_L_j * s_{j,t} + Gamma_L_j' X_long_t
            S_{j,t+1} = phi_S_j * S_{j,t} + kappa_S_j * s_{j,t} + Gamma_S_j' X_short_t

        Stationarity penalties:
            - 0 < phi_S < phi_L < 1  for each j in harvey_target (the "< 1"
              part gated by restrict_persistence_upper; see module docstring)
            - |Gamma_L|, |Gamma_S| soft-bounded when use_auxiliary_penalties
        """
        T = len(y)
        if T <= self.max_lag:
            return 1e12 if not return_paths else {}

        st_theta, pi_theta, hv_theta = self._split_theta(theta)
        sp   = self._decode_static(st_theta)
        pip  = self._decode_pi(pi_theta)
        hvd  = self._decode_harvey(hv_theta)  # {j: {omega, kappa_L, phi_L, ...}}

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

        # Static distribution constraints. gamma < zeta is a genuine GB2
        # moment-existence requirement (OPTIMIZATION.md §6), not an
        # arbitrary bound, so it is always enforced. The |param| < 20
        # magnitude guard below it is an auxiliary numerical guard rail
        # (not required by MODELS.md/OPTIMIZATION.md) and is gated.
        gamma_v = sp.get("gamma", 1.0)
        zeta_v  = sp.get("zeta",  3.0)
        penalty += LAM_STATIC * max(0.0, gamma_v - zeta_v + 1e-6) ** 2
        if self.use_auxiliary_penalties:
            for s in self.static_names:
                penalty += LAM_STATIC * max(0.0, abs(sp[s]) - 20.0) ** 2

        # Pi persistence
        rho = pip.get("rho", 0.0)
        penalty += LAM_LONG * max(0.0, abs(rho) - 0.98) ** 2

        for j in self.harvey_target:
            hj = hvd[j]
            phi_L, phi_S = hj["phi_L"], hj["phi_S"]
            kappa_L, kappa_S = hj["kappa_L"], hj["kappa_S"]

            # Persistence restriction, MODELS.md §20: 0 < phi_S < phi_L < 1.
            # The ordering penalty always applies (two independent sigmoids,
            # or a fixed constant, do not order themselves against phi_S).
            # The "< 1" upper bound and positivity penalties on phi_L, and
            # the positivity penalty on phi_S, are only needed when that
            # parameter is actually free and raw -- fixed_phi_L/phi_L_max/
            # phi_S_max already guarantee the relevant domain by
            # construction, so applying the penalty there would just add a
            # pointless (harmless, since non-optimised) constant offset.
            if self.fixed_phi_L is None and self.phi_L_max is None:
                if self.restrict_persistence_upper:
                    penalty += LAM_LONG * max(0.0, phi_L - 0.995) ** 2
                penalty += LAM_LONG * max(0.0, -phi_L) ** 2
            # phi_S must be in (0, phi_L)
            penalty += LAM_SHORT * max(0.0, phi_S - phi_L + 1e-4) ** 2
            if self.fixed_phi_S is None and self.phi_S_max is None:
                penalty += LAM_SHORT * max(0.0, -phi_S) ** 2

            # Raw-coordinate ridge (see constructor docstring): a
            # non-vanishing-derivative alternative to any decoded-scale
            # penalty, specifically to prevent the sigmoid reparameterisation
            # from saturating phi_L/phi_S to exactly 0 or 1.
            if self.raw_z_ridge > 0.0:
                if hj["raw_phi_L"] is not None:
                    penalty += self.raw_z_ridge * hj["raw_phi_L"] ** 2
                if hj["raw_phi_S"] is not None:
                    penalty += self.raw_z_ridge * hj["raw_phi_S"] ** 2
            if self.use_auxiliary_penalties:
                # Score amplitude soft bounds
                penalty += LAM_SCORE * max(0.0, abs(kappa_L) - 2.0) ** 2
                penalty += LAM_SCORE * max(0.0, abs(kappa_S) - 2.0) ** 2
                # Covariate magnitude
                penalty += LAM_COV * max(0.0, np.linalg.norm(hj["Gamma_L"]) - 20.0) ** 2
                penalty += LAM_COV * max(0.0, np.linalg.norm(hj["Gamma_S"]) - 20.0) ** 2

        # gas1_target: plain GAS(1,1) params -- decode + the SAME soft
        # persistence limit ZAGASModel's own GAS(1,1) uses (|B|<0.98),
        # mirrored here for consistency with "phi behaves like Stage 1's
        # GAS(1,1)" rather than leaving B fully unconstrained.
        gas1d = self._decode_gas1(hv_theta[len(self.harvey_target) * self._blk:]) if self.gas1_target else {}
        for j, gd in gas1d.items():
            penalty += LAM_LONG * max(0.0, abs(gd["B"]) - 0.98) ** 2
            if self.use_auxiliary_penalties:
                penalty += LAM_SCORE * max(0.0, abs(gd["A"]) - 2.0) ** 2

        # ── Numba dispatch (pre_penalty fully computed above) ─────────────────
        if self._use_nb_filter:
            return self._nb_harvey_dispatch(
                y, X_long, X_short, sp, pip, hvd, penalty, T, return_paths, gas1d
            )

        # ── Allocate state arrays ─────────────────────────────────────────────
        L_arr   = np.zeros((T + 1, n_tv))   # long component
        S_arr   = np.zeros((T + 1, n_tv))   # short component
        G_arr   = np.zeros((T + 1, n_tv))   # gas1_target state (plain GAS(1,1))
        f_arr   = np.zeros((T,     n_tv))   # combined state entering time t
        s_arr   = np.zeros((T,     n_tv))   # scaled score at t
        eta_arr = np.zeros(T)               # t-indexed  (for return paths)
        eta_eff = np.zeros(T)               # i-indexed  (AR history for pi recursion)
        pi_arr  = np.zeros(T)
        m = self.n_harmonics
        if m > 0:
            Gm_arr     = np.zeros((T + 1, n_tv, m))  # seas_h        (harmonic states)
            GmStar_arr = np.zeros((T + 1, n_tv, m))  # seas*_h
            gamma_arr  = np.zeros((T, n_tv))         # Gamma_{j,t} = sum_h seas_h -- for return_paths

        # Initialize L_0 and S_0 -- written to index 0, which the recursion
        # below never reads (loop starts at self.max_lag): a known,
        # long-standing dead write, left untouched here (fixing it would
        # silently change every historical Harvey result -- see module
        # docstring). seas0/seasstar0 are DELIBERATELY seeded at index
        # self.max_lag instead, not index 0 -- that is the array position
        # the loop's first iteration actually reads (t=self.max_lag), so
        # this is where gamma_{1|0,h}/gamma*_{1|0,h} must live to have any
        # effect at all. This asymmetry with L0/S0 is intentional: the
        # seasonal component is new (no historical artifact depends on its
        # old, equally-dead behaviour), so it is fixed here rather than
        # preserved.
        tv_idx = {name: j for j, name in enumerate(self.tv_names)}
        for j, name in enumerate(self.tv_names):
            if name in hvd:
                L_arr[0, j] = hvd[name]["L0"]
                S_arr[0, j] = hvd[name]["S0"]
                if m > 0:
                    Gm_arr[self.max_lag, j, :]     = hvd[name]["seas0"]
                    GmStar_arr[self.max_lag, j, :] = hvd[name]["seasstar0"]
            elif name in gas1d:
                G_arr[0, j] = gas1d[name]["f0"]  # same dead-write convention as L0/S0 above

        loglik = 0.0

        # ── Main recursion ─────────────────────────────────────────────────────
        for t in range(self.max_lag, T):

            # Combine long + short (+ trigonometric seasonal, if any) into
            # the current state.
            for j, name in enumerate(self.tv_names):
                if name in hvd:
                    f_arr[t, j] = hvd[name]["omega"] + L_arr[t, j] + S_arr[t, j]
                    if m > 0:
                        gamma_jt = float(np.sum(Gm_arr[t, j, :]))
                        gamma_arr[t, j] = gamma_jt
                        f_arr[t, j] += gamma_jt
                elif name in gas1d:
                    # Plain GAS(1,1): omega is baked into G_arr's own
                    # recursion (see Component updates below), so the state
                    # is read directly here, unlike the Harvey branch which
                    # adds omega at read time on top of L+S.
                    f_arr[t, j] = G_arr[t, j]
                else:
                    # Param not in harvey_target/gas1_target: use static f0 (no dynamics)
                    f_arr[t, j] = sp.get(name, 0.0)

            call_params = {name: f_arr[t, j] for j, name in enumerate(self.tv_names)}
            call_params.update(sp)

            # State magnitude soft bound
            for j in range(n_tv):
                penalty += LAM_STATE * max(0.0, abs(f_arr[t, j]) - 15.0) ** 2

            # Pi recursion (eta_eff is i-indexed to avoid t-vs-i indexing bug)
            i_eff = t - self.max_lag
            eta_t = MS.eta_step(self.pi_dyn, i_eff, eta_eff, y, t, pip, self.seasonal,
                                f_arr[t, self.tv_names.index("phi")])
            eta_eff[i_eff] = eta_t
            eta_arr[t]     = eta_t
            # Numerical-safety guard (not a statistical restriction): during
            # a BFGS line-search probe, a transient rho slightly above its
            # own soft 0.98 limit can make the AR-in-eta recursion compound
            # geometrically over thousands of steps to a magnitude where
            # squaring it below overflows a Python float
            # (OverflowError: 'Result too large') rather than merely
            # returning a large-but-finite penalty. Treat "eta already
            # absurd" as an invalid draw, same as the isfinite() checks
            # elsewhere in this loop, instead of letting the square blow up.
            if not np.isfinite(eta_t) or abs(eta_t) > 1e100:
                return (1e12 + penalty) if not return_paths else {}
            penalty += LAM_ETA * max(0.0, abs(eta_t) - 30.0) ** 2
            pi_arr[t] = float(self.pi_dyn.pi_from_eta(eta_t))
            _log_pi_t, _log_1mpi_t = self.pi_dyn.log_pi_terms(eta_t)

            # Log-likelihood
            if y[t] == 0:
                ll_t = _log_1mpi_t
            else:
                ll_dist = self.dist.logpdf(y[t], **call_params)
                if not np.isfinite(ll_dist):
                    return (1e12 + penalty) if not return_paths else {}
                ll_t = _log_pi_t + ll_dist
            if not np.isfinite(ll_t):
                return (1e12 + penalty) if not return_paths else {}
            loglik += ll_t

            # Score of the zero-adjusted density (models/mixed_score.py)
            scaled = MS.scaled_score(self.dist, self.tv_names, self.scaling, y[t], call_params,
                                     pi_arr[t], MS.phi_link_lambda(self.pi_dyn, pip),
                                     historical_fn=self._scaled_score)
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
                    L_next = hj["phi_L"] * L_arr[t, j] + hj["kappa_L"] * s_jt
                    if self.n_long > 0:
                        L_next += float(np.dot(hj["Gamma_L"], xl_t))
                    S_next = hj["phi_S"] * S_arr[t, j] + hj["kappa_S"] * s_jt
                    if self.n_short > 0:
                        S_next += float(np.dot(hj["Gamma_S"], xs_t))
                    if not (np.isfinite(L_next) and np.isfinite(S_next)):
                        return (1e12 + penalty) if not return_paths else {}
                    L_arr[t + 1, j] = L_next
                    S_arr[t + 1, j] = S_next
                    if m > 0:
                        # Trigonometric seasonal (module docstring): the SAME
                        # scalar score s_jt drives every harmonic identically,
                        # only the rotation angle (cos_h, sin_h) differs
                        # across h. psi_gamma (only present when
                        # seasonal_damping_max is set) damps the ENTIRE
                        # rotation each step -- see that flag's docstring;
                        # psi_gamma=1.0 (the default when damping is off)
                        # reproduces the exact pure-rotation recursion.
                        kg = hj["kappa_gamma"]
                        psi = hj["psi_gamma"] if hj["psi_gamma"] is not None else 1.0
                        gm, gms = Gm_arr[t, j, :], GmStar_arr[t, j, :]
                        gm_next  = psi * ( gm * self._cos_h + gms * self._sin_h) + kg * s_jt
                        gms_next = psi * (-gm * self._sin_h + gms * self._cos_h) + kg * s_jt
                        if not (np.all(np.isfinite(gm_next)) and np.all(np.isfinite(gms_next))):
                            return (1e12 + penalty) if not return_paths else {}
                        Gm_arr[t + 1, j, :]     = gm_next
                        GmStar_arr[t + 1, j, :] = gms_next
                elif name in gas1d:
                    gd = gas1d[name]
                    G_next = gd["omega"] + gd["A"] * s_jt + gd["B"] * G_arr[t, j]
                    if not np.isfinite(G_next):
                        return (1e12 + penalty) if not return_paths else {}
                    G_arr[t + 1, j] = G_next
                else:
                    L_arr[t + 1, j] = L_arr[t, j]
                    S_arr[t + 1, j] = S_arr[t, j]
                    if m > 0:
                        Gm_arr[t + 1, j, :]     = Gm_arr[t, j, :]
                        GmStar_arr[t + 1, j, :] = GmStar_arr[t, j, :]

        objective = -loglik + penalty

        if not return_paths:
            return objective

        eff = self.max_lag
        out = {
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
        if m > 0:
            out["Gamma_arr"] = gamma_arr[eff:T, :]  # Gamma_{j,t} = sum_h seas_h, the seasonal contribution to f_arr
        return out

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
        overrides: Optional[Dict[str, dict]] = None,
        static_overrides: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Construct starting values for the Harvey model.

        warm_start (optional) should be the filter paths dict from the best
        Stage-2 covariate model, used to initialize the short component.

        overrides (optional) : {j: {key: value}} for j in self.harvey_target,
            where key is any subset of
            {"omega","f0","kappa_L","phi_L","L0","Gamma_L","kappa_S","phi_S","S0","Gamma_S"}
            (same names _decode_harvey uses; MODELS.md §20 notation). Scalar
            keys overwrite the corresponding default; "Gamma_L"/"Gamma_S"
            accept an array matching n_long/n_short (or a scalar broadcast
            to all entries). overrides=None reproduces the exact previous
            default behaviour.

            "phi_L"/"phi_S" are always given and interpreted on the NATURAL
            (decoded, 0<phi<1) scale regardless of parameterisation: when
            phi_L_max/phi_S_max is set, the natural-scale target is
            converted to the raw pre-sigmoid theta value via the inverse
            sigmoid (logit) before being stored; when fixed_phi_L is set,
            any "phi_L" override is ignored (there is no theta slot for it).

        static_overrides (optional) : {static_param_name: value}, e.g.
            {"gamma": ..., "zeta": ...} to seed from a bounded warm-start fit
            (models.gas_filter.GASFilter.fit_unconditional) instead of the
            hardcoded gamma=1.0/zeta=3.0 defaults.
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
        if static_overrides:
            for i, name in enumerate(self.static_names):
                if name in static_overrides:
                    st0[i] = float(static_overrides[name])

        # Pi params
        pi0 = self.pi_dyn.initial_params(y, self.seasonal)

        # Harvey params: start phi_L=0.95, phi_S=0.5, kappa_L=0.01, kappa_S=0.05
        # (overridden per-j below when `overrides` is supplied). Walked with
        # a running index, matching _decode_harvey, since fixed_phi_L
        # removes a slot.
        hv0 = np.zeros(self.n_harvey)
        blk = self._blk
        overrides = overrides or {}
        for k, j in enumerate(self.harvey_target):
            base = k * blk
            f0   = f0_guess.get(j, 0.0)
            ov   = overrides.get(j, {})
            i = 0

            hv0[base + i] = ov.get("omega", f0 * 0.05); i += 1
            hv0[base + i] = ov.get("f0", f0); i += 1
            if self.include_long:
                hv0[base + i] = ov.get("kappa_L", 0.01); i += 1
                if self.fixed_phi_L is None:
                    phi_L_target = ov.get("phi_L", 0.95)
                    hv0[base + i] = (
                        _logit_scalar(phi_L_target / self.phi_L_max) if self.phi_L_max is not None
                        else phi_L_target
                    )
                    i += 1
                hv0[base + i] = ov.get("L0", 0.0); i += 1

                if self.n_long > 0:
                    hv0[base + i : base + i + self.n_long] = np.broadcast_to(
                        ov.get("Gamma_L", 0.0), (self.n_long,)
                    )
                i += self.n_long

            hv0[base + i] = ov.get("kappa_S", 0.05); i += 1
            if self.fixed_phi_S is None:
                phi_S_target = ov.get("phi_S", 0.50)
                hv0[base + i] = (
                    _logit_scalar(phi_S_target / self.phi_S_max) if self.phi_S_max is not None
                    else phi_S_target
                )
                i += 1
            hv0[base + i] = ov.get("S0", 0.0); i += 1

            if self.n_short > 0:
                hv0[base + i : base + i + self.n_short] = np.broadcast_to(
                    ov.get("Gamma_S", 0.0), (self.n_short,)
                )
            i += self.n_short

            if self.n_harmonics > 0:
                # Per direct instruction (2026-09-04): kappa_gamma and every
                # seas0_h/seasstar0_h (gamma_{1|0,h}/gamma*_{1|0,h}) start at
                # a small common value, 0.02 -- a modest, symmetric,
                # nonzero seasonal presence from the very first optimizer
                # evaluation, rather than the earlier all-zero cold start
                # (which, for kappa_gamma, made the seasonal component's
                # OWN score-driven update d(Gamma)/d(kappa_gamma) start
                # from a null direction, and for seas0/seasstar0 gave every
                # harmonic an identical, uninformative starting phase).
                kg_target = ov.get("kappa_gamma", 0.02)
                if self.kappa_gamma_bound is not None:
                    z = (kg_target + self.kappa_gamma_bound) / (2.0 * self.kappa_gamma_bound)
                    z = min(max(z, 1e-6), 1.0 - 1e-6)  # keep logit finite
                    hv0[base + i] = _logit_scalar(z)
                else:
                    hv0[base + i] = kg_target
                i += 1
                seas0_ov = ov.get("seas0", 0.02)
                hv0[base + i : base + i + self.n_harmonics] = np.broadcast_to(
                    seas0_ov, (self.n_harmonics,)
                )
                i += self.n_harmonics
                seasstar0_ov = ov.get("seasstar0", 0.02)
                hv0[base + i : base + i + self.n_harmonics] = np.broadcast_to(
                    seasstar0_ov, (self.n_harmonics,)
                )
                i += self.n_harmonics

                if self.seasonal_damping_max is not None:
                    # Start close to the undamped/pure-rotation case (psi
                    # near its own max, i.e. near 1) -- damping is meant to
                    # be a SAFETY NET the optimiser can engage if the data
                    # calls for it, not an aggressive prior imposed from the
                    # start.
                    psi_target = ov.get("psi_gamma", min(0.995, self.seasonal_damping_max))
                    zp = psi_target / self.seasonal_damping_max
                    zp = min(max(zp, 1e-6), 1.0 - 1e-6)
                    hv0[base + i] = _logit_scalar(zp)
                    i += 1

        # gas1_target: plain GAS(1,1) start -- omega/f0 from the same
        # distribution-specific guess as the Harvey block above (so phi/xi
        # start at a sane natural-scale value regardless of which mechanism
        # drives them), A small and positive, B close to but under the
        # 0.98 soft persistence limit enforced in _run_filter.
        g0 = np.zeros(self.n_gas1)
        for k, j in enumerate(self.gas1_target):
            base = k * self._BLK_GAS1
            f0 = f0_guess.get(j, 0.0)
            ov = overrides.get(j, {})
            g0[base + 0] = ov.get("omega", f0 * 0.05)
            g0[base + 1] = ov.get("f0", f0)
            g0[base + 2] = ov.get("A", 0.05)
            g0[base + 3] = ov.get("B", 0.90)

        return np.concatenate([st0, pi0, hv0, g0])

    def fit(
        self,
        y: np.ndarray,
        X_long: np.ndarray,
        X_short: np.ndarray,
        verbose: bool = False,
        theta0: Optional[np.ndarray] = None,
        options: Optional[dict] = None,
        polish: bool = True,
        jac: Optional[Union[str, Callable]] = None,
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
        jac      : forwarded to scipy.optimize.minimize's `jac` argument.
            None (default) reproduces the previous behaviour exactly --
            scipy's default forward-difference ('2-point') gradient. Passing
            '3-point' (central difference, same real-valued objective, no
            code changes elsewhere) or 'cs' (complex-step; requires every
            operation in _run_filter's call graph to be complex-safe, which
            is NOT guaranteed here -- expect it to raise or silently return
            wrong values through any max()/abs()-based penalty term) is used
            by run_stability_experiment.py to check whether large reported
            gradients are finite-difference noise rather than genuine
            ill-conditioning. See
            reports/longshort experiment/stability_investigation_log.md.
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

        jac_kwargs = {} if jac is None else {"jac": jac}
        result = minimize(fun=obj, x0=theta0, method="BFGS", options=opt_options, **jac_kwargs)

        # Polish step capped at 200 iterations
        polish_result = None
        polish_improvement = 0.0
        if polish:
            polish_opts = {**opt_options, "maxiter": 200}
            result2 = minimize(fun=obj, x0=result.x, method="BFGS", options=polish_opts, **jac_kwargs)
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
        use_numba: bool = True,
    ) -> dict:
        """
        1-step-ahead rolling OOS evaluation (2026-10-01: Numba route).

        The one-step-ahead OOS path is the filter run over train+test with
        the fitted theta, so the compiled filter (the same kernel used for
        estimation) produces it directly; the pure-Python recursion
        (_simulate_oos_python) is the fallback and the parity reference
        (tests/test_pi_linked_all_models.py).
        """
        if use_numba and self._use_nb_filter:
            T_train = len(y_train)
            p = self._run_filter(theta, np.concatenate([y_train, y_test]),
                                 np.vstack([X_long_train, X_long_test]),
                                 np.vstack([X_short_train, X_short_test]), return_paths=True)
            if p:
                k = T_train - int(p["eff_start"])
                return {"f_arr_oos": np.asarray(p["f_arr"])[k:, :],
                        "L_arr_oos": np.asarray(p["L_arr"])[k:, :],
                        "S_arr_oos": np.asarray(p["S_arr"])[k:, :],
                        "pi_oos": np.asarray(p["pi"])[k:], "eta_oos": np.asarray(p["eta"])[k:],
                        "static": dict(p["static"]), "tv_names": self.tv_names, "y_test": y_test}
        return self._simulate_oos_python(theta, y_train, y_test, X_long_train, X_long_test,
                                         X_short_train, X_short_test)

    def _simulate_oos_python(
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
        gas1d = self._decode_gas1(hv_theta[len(self.harvey_target) * self._blk:]) if self.gas1_target else {}

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
        G_arr   = np.zeros((T_full + 1, n_tv))   # gas1_target state (plain GAS(1,1))
        f_arr   = np.zeros((T_full,     n_tv))
        s_arr   = np.zeros((T_full,     n_tv))
        eta_arr = np.zeros(T_full)          # t-indexed  (for return paths)
        eta_eff = np.zeros(T_full)          # i-indexed  (AR history for pi recursion)
        pi_arr  = np.zeros(T_full)
        m = self.n_harmonics
        if m > 0:
            Gm_arr     = np.zeros((T_full + 1, n_tv, m))
            GmStar_arr = np.zeros((T_full + 1, n_tv, m))

        for j, name in enumerate(self.tv_names):
            if name in hvd:
                L_arr[0, j] = hvd[name]["L0"]
                S_arr[0, j] = hvd[name]["S0"]
                if m > 0:
                    # See _run_filter's identical comment: seeded at
                    # max_lag (the loop's first read), not index 0.
                    Gm_arr[max_lag, j, :]     = hvd[name]["seas0"]
                    GmStar_arr[max_lag, j, :] = hvd[name]["seasstar0"]
            elif name in gas1d:
                G_arr[0, j] = gas1d[name]["f0"]

        for t in range(max_lag, T_full):
            for j, name in enumerate(self.tv_names):
                if name in hvd:
                    f_arr[t, j] = hvd[name]["omega"] + L_arr[t, j] + S_arr[t, j]
                    if m > 0:
                        f_arr[t, j] += float(np.sum(Gm_arr[t, j, :]))
                elif name in gas1d:
                    f_arr[t, j] = G_arr[t, j]
                else:
                    f_arr[t, j] = sp.get(name, 0.0)

            call_params = {name: f_arr[t, j] for j, name in enumerate(self.tv_names)}
            call_params.update(sp)

            # eta_eff is i-indexed, matching _run_filter: compute_eta reads
            # eta_hist[i-1] for the AR term, so the t-indexed eta_arr would
            # fetch a value max_lag steps too early and break the recursion.
            i_eff = t - max_lag
            eta_arr[t] = MS.eta_step(self.pi_dyn, i_eff, eta_eff, y_full, t, pip, self.seasonal,
                                     f_arr[t, self.tv_names.index("phi")])
            eta_eff[i_eff] = eta_arr[t]
            pi_arr[t] = float(self.pi_dyn.pi_from_eta(eta_arr[t]))

            scaled = MS.scaled_score(self.dist, self.tv_names, self.scaling, y_full[t], call_params,
                                     pi_arr[t], MS.phi_link_lambda(self.pi_dyn, pip),
                                     historical_fn=self._scaled_score)
            for j, name in enumerate(self.tv_names):
                s_arr[t, j] = scaled[name]

            xl_t = Xl_full[t]
            xs_t = Xs_full[t]
            for j, name in enumerate(self.tv_names):
                s_jt = s_arr[t, j]
                if name in hvd:
                    hj = hvd[name]
                    L_arr[t + 1, j] = (
                        hj["phi_L"] * L_arr[t, j] + hj["kappa_L"] * s_jt
                        + (float(np.dot(hj["Gamma_L"], xl_t)) if self.n_long  > 0 else 0.0)
                    )
                    S_arr[t + 1, j] = (
                        hj["phi_S"] * S_arr[t, j] + hj["kappa_S"] * s_jt
                        + (float(np.dot(hj["Gamma_S"], xs_t)) if self.n_short > 0 else 0.0)
                    )
                    if m > 0:
                        kg = hj["kappa_gamma"]
                        psi = hj["psi_gamma"] if hj["psi_gamma"] is not None else 1.0
                        gm, gms = Gm_arr[t, j, :], GmStar_arr[t, j, :]
                        Gm_arr[t + 1, j, :]     = psi * ( gm * self._cos_h + gms * self._sin_h) + kg * s_jt
                        GmStar_arr[t + 1, j, :] = psi * (-gm * self._sin_h + gms * self._cos_h) + kg * s_jt
                elif name in gas1d:
                    gd = gas1d[name]
                    G_arr[t + 1, j] = gd["omega"] + gd["A"] * s_jt + gd["B"] * G_arr[t, j]
                elif m > 0:
                    Gm_arr[t + 1, j, :]     = Gm_arr[t, j, :]
                    GmStar_arr[t + 1, j, :] = GmStar_arr[t, j, :]

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

        # Decoded (natural-scale) Harvey parameters -- identical to
        # estimated_parameters.csv's phi_L/phi_S rows when neither
        # fixed_phi_L nor phi_L_max/phi_S_max is set, but the only place to
        # read the ACTUAL 0<phi<1 persistence when a sigmoid
        # reparameterisation is active (estimated_parameters.csv then holds
        # the pre-sigmoid raw value under that same name -- see
        # parameter_names()'s docstring).
        hvd = self._decode_harvey(np.asarray(theta)[self.n_static + self.n_pi :])
        decoded_rows = []
        for j, hj in hvd.items():
            for key in ("omega", "f0", "kappa_L", "phi_L", "L0", "kappa_S", "phi_S", "S0"):
                decoded_rows.append({"parameter": f"{key}_{j}", "value": hj[key]})
            for c, v in zip(self.long_names, hj["Gamma_L"]):
                decoded_rows.append({"parameter": f"Gamma_L_{j}_{c}", "value": float(v)})
            for c, v in zip(self.short_names, hj["Gamma_S"]):
                decoded_rows.append({"parameter": f"Gamma_S_{j}_{c}", "value": float(v)})
            if self.n_harmonics > 0:
                decoded_rows.append({"parameter": f"kappa_gamma_{j}", "value": hj["kappa_gamma"]})
                for h, v in enumerate(hj["seas0"], start=1):
                    decoded_rows.append({"parameter": f"seas0_{j}_{h}", "value": float(v)})
                for h, v in enumerate(hj["seasstar0"], start=1):
                    decoded_rows.append({"parameter": f"seasstar0_{j}_{h}", "value": float(v)})
                if hj.get("psi_gamma") is not None:
                    decoded_rows.append({"parameter": f"psi_gamma_{j}", "value": hj["psi_gamma"]})
        pd.DataFrame(decoded_rows).to_csv(cache_dir / "decoded_harvey_parameters.csv", index=False)

        meta = {
            "model_type":       "HarveyZAGASModel",
            "n_params":         self.n_params,
            "harvey_target":    self.harvey_target,
            "tv_names":         self.tv_names,
            "long_names":       self.long_names,
            "short_names":      self.short_names,
            "scaling":          self.scaling,
            "seasonal":         self.seasonal,
            "restrict_persistence_upper": self.restrict_persistence_upper,
            "use_auxiliary_penalties":    self.use_auxiliary_penalties,
            "fixed_phi_L":      self.fixed_phi_L,
            "phi_L_max":        self.phi_L_max,
            "phi_S_max":        self.phi_S_max,
            "fixed_phi_S":      self.fixed_phi_S,
            "raw_z_ridge":      self.raw_z_ridge,
            "n_harmonics":      self.n_harmonics,
            "seasonal_period":  self.seasonal_period,
            "kappa_gamma_bound": self.kappa_gamma_bound,
            "seasonal_damping_max": self.seasonal_damping_max,
            "aic":              2.0 * self.n_params - 2.0 * float(result["loglik"]),
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
