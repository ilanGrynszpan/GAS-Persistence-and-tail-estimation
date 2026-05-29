from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np


class Distribution(ABC):
    """
    Abstract base class for conditional distributions used in the GAS model.

    A subclass must declare which of its parameters are time-varying
    (i.e. driven by GAS dynamics) via the `tv_param_names` property.
    All other parameters are considered static and estimated jointly.
    """

    @property
    @abstractmethod
    def tv_param_names(self) -> List[str]:
        """Names of time-varying parameters, in the order returned by `score`."""

    @abstractmethod
    def logpdf(self, y: float, **params) -> float:
        """Log-density of y given parameters."""

    @abstractmethod
    def score(self, y: float, **params) -> Dict[str, float]:
        """
        Partial derivatives of log-density w.r.t. each time-varying parameter.
        Returns a dict keyed by tv_param_names.
        """

    @abstractmethod
    def fisher_info_diag(self, **params) -> Dict[str, float]:
        """
        Diagonal of the expected Fisher information matrix for each time-varying
        parameter.  Used to scale the score (Creal et al. 2012, equation (7)).
        Returns a dict keyed by tv_param_names.
        """

    @abstractmethod
    def cdf(self, y: float, **params) -> float:
        """Cumulative distribution function at y."""

    @abstractmethod
    def rvs(self, n: int = 1, **params) -> np.ndarray:
        """Draw n independent samples from the distribution."""
