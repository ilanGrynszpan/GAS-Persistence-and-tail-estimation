"""
Score and Fisher information of the zero-adjusted (mixed) GB2 density, and
the occurrence-logit step, shared by every ZA-GAS model family (2026-10-01).

WALKTHROUGH
    Every ZA-GAS filter (ZAGASModel, CovZAGASModel, HarveyZAGASModel, and the
    multi-step simulator in diagnostics/multistep_forecast.py) does, at each t:
        1. form the magnitude state(s) phi_{t|t-1} (and xi_{t|t-1});
        2. eta_t  <- occurrence dynamics       (eta_step)
        3. pi_t   = logistic(eta_t)
        4. scaled score of y_t                  (scaled_score)
        5. update the states with the scaled score.
    This module holds steps 2 and 4 for the Python paths, and the argument
    packing (nb_pi_args, SCALING_CODE) that the Numba kernels need. The Numba
    twins of step 4 are models/_nb_gb2_filter.py::_scaled_score_1/_2; the two
    implementations are kept identical (tests/test_mixed_fisher_parity.py,
    tests/test_pi_linked_all_models.py).

MATHEMATICS
    f(y) = (1 - pi_t) 1{y=0} + pi_t g(y; phi_t, xi_t) 1{y>0},
    eta_t = logit pi_t = [y-lag / AR terms] + lambda_1 phi_{t|t-1}
    (lambda_1 != 0 only for PhiLinkedJointPiDynamics). With w = 1{y_t>0}:
        s_phi = w dlog g/dphi + lambda_1 (w - pi_t),    s_xi = w dlog g/dxi
        M     = [[pi I_pp + lambda_1^2 pi (1-pi),  pi I_px],
                 [pi I_px,                         pi I_xx]]
    where I_.. is the GB2 Fisher information on the log scale
    (distributions/gb2_log_link.py, proof in reports/fisher/fisher_gb2.pdf).
    The occurrence/magnitude cross terms vanish because E[dlog g] = 0.

    Scalings:
        unit                            s
        diagonal_inverse_fisher         w * dlog g / diag(I_g)   (historical, lambda_1 = 0)
        inverse_fisher                  w * I_g^{-1} dlog g      (historical, lambda_1 = 0)
        mixed_diagonal_inverse_fisher   s / diag(M)
        mixed_inverse_fisher            M^{-1} s
    For one time-varying parameter the diagonal and full versions coincide.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

SCALING_CODE = {"unit": 0, "diagonal_inverse_fisher": 1, "inverse_fisher": 2,
                "mixed_inverse_fisher": 3, "mixed_diagonal_inverse_fisher": 4}
MIXED_SCALINGS = {"mixed_inverse_fisher", "mixed_diagonal_inverse_fisher"}


def pi_uses_phi(pi_dyn) -> bool:
    return bool(getattr(pi_dyn, "uses_filtered_phi", False))


def phi_link_lambda(pi_dyn, pi_params: dict) -> float:
    return float(pi_params["lambda1"]) if pi_uses_phi(pi_dyn) else 0.0


def check_scaling(pi_dyn, scaling: str) -> None:
    """Historical GB2-only scalings are undefined when pi depends on phi."""
    if pi_uses_phi(pi_dyn) and scaling not in ({"unit"} | MIXED_SCALINGS):
        raise ValueError(
            f"pi dynamics {type(pi_dyn).__name__} make pi depend on phi: use scaling "
            f"'unit' or one of {sorted(MIXED_SCALINGS)} (got '{scaling}')"
        )


def eta_step(pi_dyn, i: int, eta_hist: np.ndarray, y: np.ndarray, t: int,
             pi_params: dict, seasonal: str, phi_t: float) -> float:
    """eta_t for any pi dynamics; phi-linked dynamics read phi_{t|t-1}."""
    if pi_uses_phi(pi_dyn):
        return pi_dyn.eta_from_phi(phi_t, pi_params)
    return pi_dyn.compute_eta(i=i, eta_hist=eta_hist, y_full=y, t=t,
                              params=pi_params, seasonal=seasonal)


def nb_pi_args(pi_dyn, pi_params: dict, seasonal_lags: List[int]):
    """(lags_pi, pi_wy, pi_omega0, pi_rho, pi_lam_phi, pi_link) for the kernels."""
    pi_link = 1 if getattr(pi_dyn, "link", "logistic") == "reciprocal" else 0
    if pi_uses_phi(pi_dyn):
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64),
                float(pi_params["lambda0"]), 0.0, float(pi_params["lambda1"]), pi_link)
    lags = list(getattr(pi_dyn, "lags", None) or seasonal_lags)
    return (np.array(lags, dtype=np.int64),
            np.array([pi_params[f"omega_y_{l}"] for l in lags], dtype=np.float64),
            float(pi_params["omega0"]), float(pi_params.get("rho", 0.0)), 0.0, pi_link)


def scaled_score(dist, tv_names: List[str], scaling: str, y_t: float, call_params: dict,
                 pi_t: float, lam: float, historical_fn=None) -> Dict[str, float]:
    """
    Scaled score of the mixed density (Python twin of _scaled_score_1/_2).

    historical_fn(raw_score, call_params): the model's own GB2-only scaling
    routine (with its fallback bookkeeping), used for unit / historical
    scalings when lam == 0 so existing behaviour is reproduced exactly.
    """
    if scaling not in MIXED_SCALINGS and lam == 0.0:
        if y_t > 0:
            raw = dist.score(y_t, **call_params)
            if historical_fn is None:
                return {n: float(raw[n]) for n in tv_names}
            return historical_fn(raw, call_params)
        return {n: 0.0 for n in tv_names}

    wet = 1.0 if y_t > 0 else 0.0
    raw = dist.score(y_t, **call_params) if y_t > 0 else {n: 0.0 for n in tv_names}
    s = {n: float(raw[n]) for n in tv_names}
    s["phi"] += lam * (wet - pi_t)
    if scaling == "unit":
        return s

    if len(tv_names) == 1:
        i_g = float(dist.fisher_info_diag(**call_params)["phi"])
        m = pi_t * i_g + lam * lam * pi_t * (1.0 - pi_t)
        if m > 0.0 and math.isfinite(m) and math.isfinite(s["phi"] / m):
            return {"phi": s["phi"] / m}
        return s

    full = dist.fisher_info_full(**call_params)
    m_pp = pi_t * full[0, 0] + lam * lam * pi_t * (1.0 - pi_t)
    m_xx = pi_t * full[1, 1]
    m_px = pi_t * full[0, 1]
    sp, sx = s["phi"], s["xi"]

    def _diag():
        a = sp / m_pp if (m_pp > 0.0 and math.isfinite(m_pp)) else sp
        b = sx / m_xx if (m_xx > 0.0 and math.isfinite(m_xx)) else sx
        return {"phi": a if math.isfinite(a) else sp, "xi": b if math.isfinite(b) else sx}

    if scaling == "mixed_diagonal_inverse_fisher":
        return _diag()
    det = m_pp * m_xx - m_px * m_px
    if det > 1e-12 and math.isfinite(det):
        a = (m_xx * sp - m_px * sx) / det
        b = (-m_px * sp + m_pp * sx) / det
        if math.isfinite(a) and math.isfinite(b):
            return {"phi": a, "xi": b}
        return {"phi": sp / m_pp if m_pp > 0 else sp, "xi": sx / m_xx if m_xx > 0 else sx}
    return s
