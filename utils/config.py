"""
Global configuration parameters shared across all ALMT models.
"""

# Interest rate shocks in basis points
SHOCK_SIZES = {
    "parallel_up_200":    200,
    "parallel_down_200": -200,
    "parallel_up_100":    100,
    "parallel_down_100": -100,
    "steepener": {"short": -100, "long": 100},
    "flattener": {"short":  100, "long": -100},
}

# Six standard IRRBB scenarios required by BCBS 368
IRRBB_SCENARIOS = [
    "parallel_up",
    "parallel_down",
    "steepener",
    "flattener",
    "short_up",
    "short_down",
]

# Regulatory minimum thresholds
REGULATORY_FLOORS = {
    "LCR_minimum":  1.00,  # 100% minimum
    "NSFR_minimum": 1.00,  # 100% minimum
    "EVE_limit":    0.15,  # 15% of Tier 1 capital
    "NII_limit":    0.05,  # 5% of NII
}

# General conventions
BASE_CURRENCY = "EUR"
DAY_COUNT_CONVENTION = "ACT/360"
COMPOUNDING = "continuous"

# Standard ALM time buckets in years
ALM_BUCKETS = [0, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]

# All amounts expressed in EUR millions
BALANCE_SHEET_SCALE = 1_000_000
