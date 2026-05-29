"""
Factory for π dynamics.  Extend by registering new PiDynamics subclasses.
"""

from pi_dynamics.base import PiDynamics
from pi_dynamics.ar_logistic import ARLogisticPiDynamics

_REGISTRY: dict[str, type[PiDynamics]] = {
    "ar_logistic": ARLogisticPiDynamics,
}


class PiDynamicsFactory:
    @staticmethod
    def get(name: str) -> PiDynamics:
        """
        Return an instance of the named PiDynamics.

        Available names: 'ar_logistic'
        """
        if name not in _REGISTRY:
            raise ValueError(
                f"Unknown PiDynamics '{name}'. Available: {list(_REGISTRY)}"
            )
        return _REGISTRY[name]()

    @staticmethod
    def register(name: str, cls: type[PiDynamics]) -> None:
        """Register a custom PiDynamics under `name`."""
        _REGISTRY[name] = cls

    @staticmethod
    def available() -> list[str]:
        return list(_REGISTRY)
