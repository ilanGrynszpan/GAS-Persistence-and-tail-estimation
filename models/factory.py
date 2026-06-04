"""Convenience constructors for comparable ZA-GAS model specifications."""

from __future__ import annotations

from distributions import GB2LogLink, GB2LogLinkPhiOnly
from models.za_gas_model import ZAGASModel
from pi_dynamics.ar_logistic import ARLogisticPiDynamics


def build_zagas_model(
    model_type: str = "phi_xi",
    seasonal: str = "daily",
    scale_score: bool = True,
) -> ZAGASModel:
    """
    Build one of the comparison specifications.

    model_type:
        "phi" or "phi_only"   -> phi is time-varying; xi is static.
        "phi_xi" or "full"    -> phi and xi are both time-varying.
    """
    key = model_type.lower().replace("-", "_")
    if key in {"phi", "phi_only"}:
        return ZAGASModel(
            distribution=GB2LogLinkPhiOnly(),
            pi_dynamics=ARLogisticPiDynamics(),
            seasonal=seasonal,
            scale_score=scale_score,
            static_params=["xi", "gamma", "zeta"],
        )
    if key in {"phi_xi", "full", "two_tv", "two_time_varying"}:
        return ZAGASModel(
            distribution=GB2LogLink(),
            pi_dynamics=ARLogisticPiDynamics(),
            seasonal=seasonal,
            scale_score=scale_score,
            static_params=["gamma", "zeta"],
        )
    raise ValueError(f"Unknown model_type: {model_type!r}")
