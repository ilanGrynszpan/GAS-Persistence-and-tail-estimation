"""
Phi-linked pi dynamics -- Stage 1 occurrence-dynamics comparison (2026-07-14).

=============================================================================
THEORY
=============================================================================

All other PiDynamics implementations in this package (ARLogisticPiDynamics,
ARLogisticCustomLagsPiDynamics) drive the occurrence logit eta_t purely from
the rainfall series y itself (lagged values and/or its own AR memory).  This
module asks a different question: does the *magnitude* model's own
one-step-ahead scale prediction carry information about whether tomorrow is
wet at all?

    eta_t = lambda_0 + lambda_1 * phi_{t|t-1}

where phi_{t|t-1} is the transformed GAS(p,q) scale state (see
docs/MODELS.md §5 -- GB2LogLink already works in the transformed/log
parameterisation, so phi_{t|t-1} is used directly here, with no additional
link function) filtered from a phi-only, no-covariate Stage 1 GAS(p,q)
model (models/gas_filter.py via models/za_gas_model.py), i.e. exactly the
"basic Stage 1 model, phi the only time-varying parameter" baseline this
comparison is scoped to. phi_{t|t-1} is fixed (frozen) before this class is
constructed -- fitting lambda_0, lambda_1 here never re-estimates the
magnitude GAS recursion, consistent with the additive separability of the
zero-augmented log-likelihood in pi vs. the positive-part parameters (see
pi_dynamics/standalone_fit.py and models/static_gb2.py for the same
argument applied elsewhere in this codebase).

=============================================================================
PADDING CONVENTION FOR t < eff_start
=============================================================================

A GAS(p,q) filter only produces a genuine, data-driven phi_{t|t-1} for
t >= eff_start (eff_start = max(gas_lags), e.g. 367 for the seasonal daily
lag set {1,2,3,364,365,366,367}); before that, the filter itself holds the
state at its fitted initial value f0_phi (see models/gas_filter.py,
GASFilter._run_filter: `f_arr[:max_lag+1, j] = p[f"f0_{name}"]`). This
module requires the caller to supply a FULL-length, t-indexed phi_path
array (length == len(y_full) it will be evaluated against) that already
follows this exact convention -- see `build_full_length_phi_path` below,
used by run_pi_dynamics_alternatives.py. Padding with the constant f0_phi
for the warm-up region is not a workaround; it is the literal definition
of the GAS recursion's pre-effective-sample history used everywhere else
in this framework (models/gas_filter.py, models/za_gas_model.py).

=============================================================================
IMPLEMENTATION NOTES
=============================================================================

* compute_eta ignores `eta_hist`, `y_full`, `i`, and `seasonal` entirely --
  the only state this dynamics needs at time t is phi_path[t], which was
  already computed once, upfront, by the frozen magnitude GAS filter. This
  makes eta_t here memoryless in the pi_dynamics recursion itself (no AR
  term, no y-lags), while still carrying dynamic information indirectly
  through phi_path.
* `default_bounds` for lambda_1 is deliberately wider than the y-lag
  coefficients used by the other PiDynamics classes: phi_{t|t-1} is a
  transformed GAS state whose soft bound is +/-15 (models/za_gas_model.py
  STATE_LIMIT), so a unit change in phi can plausibly need a small
  lambda_1 to stay within the eta soft bound of +/-30 (ETA_LIMIT) -- see
  bound choice below.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from pi_dynamics.base import PiDynamics


def build_full_length_phi_path(
    phi_is: np.ndarray,
    eff_start: int,
    f0_phi: float,
    total_length: int,
) -> np.ndarray:
    """
    Build a t-indexed, full-length phi_{t|t-1} array from a filter's
    in-sample output.

    Parameters
    ----------
    phi_is        : filtered phi path for t in [eff_start, total_length),
                     i.e. `ZAGASModel.filter(theta, y)["phi"]` -- length
                     (total_length - eff_start).
    eff_start      : the GAS filter's effective-sample start (max_lag).
    f0_phi         : the fitted initial state, used to pad t < eff_start
                     (matches the filter's own warm-up convention).
    total_length   : len(y_full) this phi_path will be evaluated against.

    Returns
    -------
    phi_path : array of shape (total_length,), phi_path[t] = phi_{t|t-1}.
    """
    if len(phi_is) != total_length - eff_start:
        raise ValueError(
            f"phi_is has length {len(phi_is)}, expected "
            f"{total_length - eff_start} = total_length - eff_start."
        )
    phi_path = np.full(total_length, float(f0_phi), dtype=float)
    phi_path[eff_start:] = phi_is
    return phi_path


class PhiLinkedPiDynamics(PiDynamics):
    """
    Occurrence logit driven linearly by the frozen magnitude GAS state:

        eta_t = lambda_0 + lambda_1 * phi_{t|t-1}

    Parameters
    ----------
    phi_path : full-length, t-indexed array of phi_{t|t-1} values (see
               `build_full_length_phi_path`). Must have the same length as
               whatever y array this instance's compute_eta will be called
               against (checked lazily, on first access, since y is not
               known at construction time).

    Parameters (flat vector order):
        lambda0  -- intercept
        lambda1  -- slope on phi_{t|t-1}
    """

    def __init__(self, phi_path: np.ndarray):
        self.phi_path = np.asarray(phi_path, dtype=float)

    # ------------------------------------------------------------------
    def param_names(self, seasonal: str) -> List[str]:
        return ["lambda0", "lambda1"]

    def default_bounds(self, seasonal: str):
        return [(-5.0, 2.0), (-3.0, 3.0)]

    def initial_params(self, y: np.ndarray, seasonal: str) -> np.ndarray:
        pi_hat = np.mean(y > 0)
        pi_hat = np.clip(pi_hat, 0.01, 0.99)
        lambda0 = float(np.log(pi_hat / (1.0 - pi_hat)))
        return np.array([lambda0, 0.0])

    def compute_eta(
        self,
        i: int,
        eta_hist: np.ndarray,
        y_full: np.ndarray,
        t: int,
        params: dict,
        seasonal: str,
    ) -> float:
        if t >= len(self.phi_path):
            raise IndexError(
                f"PhiLinkedPiDynamics.compute_eta: t={t} is out of range "
                f"for phi_path of length {len(self.phi_path)}. phi_path "
                f"must span the full y array this instance is fit against."
            )
        return float(params["lambda0"] + params["lambda1"] * self.phi_path[t])
