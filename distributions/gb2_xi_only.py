"""
GB2 log-link variant with only xi time-varying.

phi (log scale) is estimated as a static parameter alongside gamma and zeta;
xi (log shape a) is the sole GAS state.
"""

from __future__ import annotations
from typing import List

from distributions.gb2_log_link import GB2LogLink


class GB2LogLinkXiOnly(GB2LogLink):
    """GB2 distribution where xi is the only GAS state."""

    @property
    def tv_param_names(self) -> List[str]:
        return ["xi"]
