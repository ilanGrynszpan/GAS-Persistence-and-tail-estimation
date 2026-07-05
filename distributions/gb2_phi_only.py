"""
GB2 log-link variant with only phi time-varying.

This keeps the positive-part scale state dynamic while estimating xi as a
static distribution parameter together with gamma and zeta.
"""

from __future__ import annotations
from typing import List

from distributions.gb2_log_link import GB2LogLink


class GB2LogLinkPhiOnly(GB2LogLink):
    """GB2 distribution where phi is the only GAS state.

    xi is static (estimated but not score-driven).
    GASFilter must include "xi" in static_params when using this distribution.
    The `default_static_params` property signals this requirement.
    """

    @property
    def tv_param_names(self) -> List[str]:
        return ["phi"]

    @property
    def default_static_params(self) -> List[str]:
        return ["xi", "gamma", "zeta"]
