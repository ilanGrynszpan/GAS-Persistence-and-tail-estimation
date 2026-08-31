"""
Cross-location loader for the occurrence-dynamics (pi_t) alternatives
experiment's saved in-sample quantile residuals.

=============================================================================
OVERVIEW
=============================================================================
run_pi_dynamics_alternatives.py already fit and saved four pi_t dynamics
alternatives at each of the six completed locations (see
pi_dynamics/ar_logistic_custom_lags.py, pi_dynamics/phi_linked.py):

    "existing_ar_seasonal"  A -- eta_t = omega0 + rho*eta_{t-1}
                                          + sum_{l in {1,365,366}} omega_l*y_{t-l}
                                 (the pre-existing AR-logistic occurrence
                                 dynamics used everywhere else in the Stage
                                 1-4 pipeline; "autoregressive" below)
    "short_lags_ar"          B -- same AR feedback, short lags {1,2,3} only
    "short_lags_noar"        C -- no AR feedback (no rho term), short lags
    "phi_linked"             D -- eta_t = lambda0 + lambda1*varphi_{t|t-1}
                                 ("phi-linked" below)

This module answers one question: "for one of these variant_key's, what
is the saved in-sample quantile-residual array at every location?" -- so
that pi_dynamics_alternatives_mosaics.ipynb can hand the results straight
to diagnostics.plots' qq_location_mosaic / acf_location_mosaic. It does
not fit or re-fit anything (docs/EXECUTION.md Sec 22: "Reports should
never call optimization") -- every quantile residual was already computed
and saved by run_pi_dynamics_alternatives.py at fit time.

=============================================================================
WHY THIS WRAPS generate_pi_dynamics_report.py INSTEAD OF REIMPLEMENTING
=============================================================================
"Load one (location, variant) artifact -- metadata.json + paths.npz,
None if missing/failed" is already solved by
generate_pi_dynamics_report.py's load_variant(). Per CLAUDE.md ("refactor
rather than duplicate", "determine whether functionality already exists
before creating new modules"), this module imports and reuses that
function rather than re-deriving the artifact schema (paths.npz's "qr"
array is exactly the windowed quantile residual Phi^{-1}(F_t(y_t)) for
this Bernoulli occurrence model; see diagnostics/occurrence_pit.py for
the derivation and run_pi_dynamics_alternatives.py for where it is saved).

=============================================================================
CODE WALKTHROUGH
=============================================================================
load_qr_by_location(variant_key)
    |
    v
For each of the 6 locations (diagnostics.multilocation_mosaics.LOCATIONS,
same order as every other mosaic notebook in this repo):
    STATION_REGISTRY[station]["short"]  -> artifact directory key
        |
        v
    generate_pi_dynamics_report.load_variant(short, variant_key)
        -> {..., "_qr": np.ndarray}  or  None (missing/failed artifact)
        |
        v
    collect qr array, keyed by location *display* name
    |
    v
Return {location display name: quantile-residual array, or None} -- None
where the variant was not fit or failed at that location, so
diagnostics.plots' qq_location_mosaic / acf_location_mosaic render a
labelled placeholder panel instead of silently dropping a location.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from data.station_loader import STATION_REGISTRY
from diagnostics.multilocation_mosaics import LOCATIONS

# The four fitted alternatives, in the same order/labels/formulas as
# generate_pi_dynamics_report.py -- imported lazily inside
# load_qr_by_location (see that function) rather than at module level, to
# avoid a module-import-time dependency on matplotlib's Agg backend for
# callers that only need LOCATIONS/VARIANT_* from this module.
VARIANT_ORDER = ["existing_ar_seasonal", "short_lags_ar", "short_lags_noar", "phi_linked"]

# Two variants singled out as "the" autoregressive vs. phi-linked
# alternatives for the dissertation figures (pi_dynamics_alternatives_
# mosaics.ipynb) -- "existing_ar_seasonal" or is the *only* variant with
# genuine AR feedback (rho term) *and* seasonal lags, i.e. exactly the
# occurrence dynamics used everywhere else in this framework
# (docs/MODELS.md Sec 3), making it the natural "autoregressive" baseline
# to contrast against the new "phi-linked" proposal. "short_lags_ar" is
# also technically autoregressive but is an ablation of "existing_ar_
# seasonal" (short lags instead of seasonal), not a separate headline
# alternative -- change AUTOREGRESSIVE_VARIANT_KEY below if a different
# variant should be shown as "the" autoregressive alternative instead.
PHI_LINKED_VARIANT_KEY    = "phi_linked"
AUTOREGRESSIVE_VARIANT_KEY = "existing_ar_seasonal"


def load_qr_by_location(variant_key: str) -> Dict[str, Optional[np.ndarray]]:
    """
    Return {location display name: in-sample quantile-residual array, or
    None} for one pi_t-dynamics variant, across every location in
    diagnostics.multilocation_mosaics.LOCATIONS.

    Parameters
    ----------
    variant_key : one of VARIANT_ORDER, e.g. "phi_linked" or
        "existing_ar_seasonal" (see PHI_LINKED_VARIANT_KEY /
        AUTOREGRESSIVE_VARIANT_KEY above).
    """
    # Imported lazily (not at module import time): pulls in matplotlib's
    # Agg backend and the full report script only when a caller actually
    # asks for artifact loading, matching the lazy-import convention used
    # throughout diagnostics/multilocation_mosaics.py.
    from generate_pi_dynamics_report import load_variant

    out: Dict[str, Optional[np.ndarray]] = {}
    for station in LOCATIONS:
        cfg = STATION_REGISTRY[station]
        display = cfg["display"]
        variant = load_variant(cfg["short"], variant_key)
        out[display] = variant["_qr"] if variant is not None else None
        if variant is None:
            print(f"  [SKIP] {display}: no successful '{variant_key}' artifact")
    return out
