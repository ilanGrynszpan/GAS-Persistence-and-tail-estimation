"""
Shared constants for the ZA-GAS framework.

Kept in a top-level module so both `pi_dynamics` and `models` can import
from here without creating circular dependencies.
"""

# Seasonal lag sets used by the GAS filter and pi dynamics.
# l=1 refers to the immediately preceding time step;
# l=364 to approximately the same calendar day one year ago.
SEASONAL_LAGS: dict[str, list[int]] = {
    "daily":   [1, 2, 3, 364, 365, 366, 367],
    "monthly": [1, 2, 3, 11, 12, 13],
}
