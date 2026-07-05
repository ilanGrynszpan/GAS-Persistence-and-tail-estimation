"""
Shared constants for the ZA-GAS framework.

Kept in a top-level module so both `pi_dynamics` and `models` can import
from here without creating circular dependencies.

Lag set conventions
-------------------
SEASONAL_LAGS
    Used exclusively by pi_dynamics (occurrence model). Do NOT change without
    also updating the pi_dynamics parameter layout, as the lag set determines
    how many omega_y parameters are estimated.

GAS_SHORT_LAGS
    Short-memory GAS lag set per MODELS.md §14: L_short = {1, 2, 3}.
    Captures recent rainfall dynamics without seasonal structure.

GAS_SEASONAL_LAGS
    Seasonal GAS lag set per MODELS.md §14: L_seasonal = {1,2,3,364,365,366,367}.
    Extends the short set with same-calendar-day lags from the previous year.

Usage
-----
In GASFilter (and derived models), pass the chosen lag set explicitly:

    from constants import GAS_SHORT_LAGS, GAS_SEASONAL_LAGS
    gas = GASFilter(distribution, seasonal="daily",
                    gas_lags=GAS_SHORT_LAGS["daily"])

The GASFilter.seasonal parameter continues to control pi_dynamics only.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Pi-dynamics lag sets (occurrence model — do not change)
# ─────────────────────────────────────────────────────────────────────────────

# l=1 refers to the immediately preceding time step;
# l=365/366 to approximately the same calendar day one year ago.
SEASONAL_LAGS: dict[str, list[int]] = {
    "daily":   [1, 365, 366],
    "monthly": [1, 2, 3, 11, 12, 13],
}

# ─────────────────────────────────────────────────────────────────────────────
# GAS filter lag sets (positive-part intensity model — per MODELS.md §14)
# ─────────────────────────────────────────────────────────────────────────────

# Short memory: captures recent atmospheric dynamics only.
GAS_SHORT_LAGS: dict[str, list[int]] = {
    "daily":   [1, 2, 3],
    "monthly": [1, 2, 3],
}

# Seasonal memory: extends short set with same-period-last-year lags.
# Tests whether annual recurrence improves score-driven dynamics.
GAS_SEASONAL_LAGS: dict[str, list[int]] = {
    "daily":   [1, 2, 3, 364, 365, 366, 367],
    "monthly": [1, 2, 3, 11, 12, 13],
}
