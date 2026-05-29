"""
Abstract base class for π_t dynamics.

π_t = P(y_t > 0 | F_{t-1}) is the probability of a non-zero observation.
It is modelled via a latent variable η_t:

    π_t = σ(η_t) = 1 / (1 + exp(-η_t))

Subclasses define how η_t evolves over time.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class PiDynamics(ABC):
    """
    Abstract interface for the dynamics of the logit probability η_t.

    Subclasses must implement:
      - `param_names(seasonal)`: list of parameter names in optimisation order
      - `default_bounds(seasonal)`: L-BFGS-B bounds for each parameter
      - `initial_params(y, seasonal)`: data-driven starting point
      - `compute_eta(i, eta_hist, y_full, t, params, seasonal)`: update rule
    """

    @abstractmethod
    def param_names(self, seasonal: str) -> List[str]:
        """
        Parameter names in the order they appear in the flat parameter vector.
        `seasonal` is 'daily' or 'monthly'.
        """

    @abstractmethod
    def default_bounds(self, seasonal: str) -> List[Tuple[float, float]]:
        """Optimisation bounds for each parameter."""

    @abstractmethod
    def initial_params(self, y: np.ndarray, seasonal: str) -> np.ndarray:
        """Data-driven initial parameter vector."""

    @abstractmethod
    def compute_eta(
        self,
        i: int,
        eta_hist: np.ndarray,
        y_full: np.ndarray,
        t: int,
        params: dict,
        seasonal: str,
    ) -> float:
        """
        Compute η_t given history.

        Parameters
        ----------
        i        : index in the effective sample (0-based)
        eta_hist : array of past η values (length = effective sample up to i)
        y_full   : full original y array
        t        : index in y_full corresponding to position i in effective sample
        params   : dict of parameter values (keys match param_names)
        seasonal : 'daily' or 'monthly'
        """
