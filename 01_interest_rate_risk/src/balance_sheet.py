"""
balance_sheet.py
================
Core balance sheet model for the IRRBB module.

Defines the building blocks of the bank's balance sheet:
    - Position       : a single financial instrument (asset or liability)
    - BalanceSheet   : collection of positions with aggregation helpers

Each position carries enough information to:
    - Feed the repricing gap (bucket, rate type, repricing date)
    - Feed the EVE engine   (full cash flow schedule)
    - Feed the NII engine   (current rate, notional, remaining term)

Supported instrument types
---------------------------
Assets    : fixed_rate_loan, floating_rate_loan, bond, mortgage
Liabilities: demand_deposit, term_deposit, wholesale_funding, nmd
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Regulatory time buckets (BCBS 368 / EBA IRRBB guidelines)
BUCKETS: list[str] = [
    "O/N", "1W", "1M", "3M", "6M", "9M",
    "1Y", "2Y", "3Y", "5Y", "10Y", "15Y", "20Y", ">20Y",
]

# Upper bound in years for each bucket (used to assign positions)
BUCKET_UPPER_YEARS: list[float] = [
    1/365, 7/365, 1/12, 3/12, 6/12, 9/12,
    1.0,   2.0,   3.0,  5.0,  10.0, 15.0, 20.0, np.inf,
]

RateType = Literal["fixed", "floating", "nmd"]
Side     = Literal["asset", "liability"]


# ---------------------------------------------------------------------------
# POSITION DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """
    A single balance sheet instrument.

    Attributes
    ----------
    position_id   : unique identifier (e.g. "LOAN_0001")
    label         : human-readable description
    side          : 'asset' or 'liability'
    instrument    : instrument type (fixed_rate_loan, demand_deposit, ...)
    notional      : outstanding amount in €M
    rate          : current contractual rate (decimal, e.g. 0.035 = 3.5%)
    rate_type     : 'fixed', 'floating', or 'nmd'
    start_date    : origination / value date
    maturity_date : contractual maturity
    repricing_date: next repricing date (= maturity for fixed, next reset for floating)
    spread        : spread over index for floating instruments (decimal)
    currency      : ISO currency code (default EUR)
    """

    position_id   : str
    label         : str
    side          : Side
    instrument    : str
    notional      : float                  # €M
    rate          : float                  # decimal
    rate_type     : RateType
    start_date    : date
    maturity_date : date
    repricing_date: date
    spread        : float = 0.0
    currency      : str   = "EUR"

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def remaining_years(self) -> float:
        """Years from today to maturity."""
        today = date.today()
        return max((self.maturity_date - today).days / 365.25, 0.0)

    @property
    def time_to_reprice(self) -> float:
        """Years from today to the next repricing date."""
        today = date.today()
        return max((self.repricing_date - today).days / 365.25, 0.0)

    @property
    def repricing_bucket(self) -> str:
        """
        Assign the position to the shortest bucket that contains
        its time-to-reprice.
        """
        ttr = self.time_to_reprice
        for bucket, upper in zip(BUCKETS, BUCKET_UPPER_YEARS):
            if ttr <= upper:
                return bucket
        return ">20Y"

    @property
    def signed_notional(self) -> float:
        """
        Convention: assets are positive, liabilities are negative.
        Used when computing the repricing gap.
        """
        return self.notional if self.side == "asset" else -self.notional

    def __repr__(self) -> str:
        return (
            f"Position({self.position_id} | {self.side} | {self.instrument} | "
            f"€{self.notional:.1f}M | {self.rate*100:.2f}% {self.rate_type} | "
            f"reprice: {self.repricing_bucket})"
        )


# ---------------------------------------------------------------------------
# BALANCE SHEET
# ---------------------------------------------------------------------------

class BalanceSheet:
    """
    Collection of Position objects representing the bank's banking book.

    Provides helpers to:
        - Add / remove positions
        - Filter by side, instrument, rate type or bucket
        - Export to DataFrame for downstream engines
        - Compute basic summary statistics (total assets, NII proxy, duration)

    Usage
    -----
    >>> bs = BalanceSheet()
    >>> bs.add(Position(...))
    >>> df = bs.to_dataframe()
    >>> assets = bs.assets
    """

    def __init__(self, name: str = "Banking Book") -> None:
        self.name       : str                    = name
        self._positions : dict[str, Position]    = {}

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def add(self, position: Position) -> None:
        """Add a position. Raises ValueError on duplicate ID."""
        if position.position_id in self._positions:
            raise ValueError(f"Duplicate position_id: {position.position_id}")
        self._positions[position.position_id] = position

    def remove(self, position_id: str) -> None:
        """Remove a position by ID."""
        self._positions.pop(position_id, None)

    def get(self, position_id: str) -> Position:
        """Retrieve a position by ID."""
        return self._positions[position_id]

    def __len__(self) -> int:
        return len(self._positions)

    def __iter__(self):
        return iter(self._positions.values())

    # ------------------------------------------------------------------
    # Filtered views
    # ------------------------------------------------------------------

    @property
    def positions(self) -> list[Position]:
        """All positions."""
        return list(self._positions.values())

    @property
    def assets(self) -> list[Position]:
        """Asset-side positions only."""
        return [p for p in self if p.side == "asset"]

    @property
    def liabilities(self) -> list[Position]:
        """Liability-side positions only."""
        return [p for p in self if p.side == "liability"]

    def by_rate_type(self, rate_type: RateType) -> list[Position]:
        """Filter positions by rate type ('fixed', 'floating', 'nmd')."""
        return [p for p in self if p.rate_type == rate_type]

    def by_bucket(self, bucket: str) -> list[Position]:
        """Return all positions whose repricing falls in the given bucket."""
        return [p for p in self if p.repricing_bucket == bucket]

    def by_instrument(self, instrument: str) -> list[Position]:
        """Filter by instrument type string."""
        return [p for p in self if p.instrument == instrument]

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @property
    def total_assets(self) -> float:
        """Sum of asset notionals (€M)."""
        return sum(p.notional for p in self.assets)

    @property
    def total_liabilities(self) -> float:
        """Sum of liability notionals (€M)."""
        return sum(p.notional for p in self.liabilities)

    @property
    def leverage(self) -> float:
        """Total assets / total liabilities ratio."""
        tl = self.total_liabilities
        return self.total_assets / tl if tl else np.inf

    def nii_proxy(self) -> float:
        """
        Simple NII proxy: sum of (notional × rate) for assets
        minus sum of (notional × rate) for liabilities.

        This is an approximation — the NII engine provides a full simulation.
        """
        asset_income   = sum(p.notional * p.rate for p in self.assets)
        liability_cost = sum(p.notional * p.rate for p in self.liabilities)
        return asset_income - liability_cost

    def weighted_average_rate(self, side: Side) -> float:
        """
        Notional-weighted average rate for a given side.

        Args:
            side : 'asset' or 'liability'
        """
        positions = self.assets if side == "asset" else self.liabilities
        total_notional = sum(p.notional for p in positions)
        if total_notional == 0:
            return 0.0
        return sum(p.notional * p.rate for p in positions) / total_notional

    def gap_by_bucket(self) -> dict[str, float]:
        """
        Repricing gap per bucket = sum of signed notionals per bucket.

        Positive gap → asset-sensitive (NII rises when rates go up).
        Negative gap → liability-sensitive (NII falls when rates go up).
        """
        gap: dict[str, float] = {b: 0.0 for b in BUCKETS}
        for p in self:
            gap[p.repricing_bucket] += p.signed_notional
        return gap

    def cumulative_gap(self) -> dict[str, float]:
        """Cumulative repricing gap across buckets."""
        gap = self.gap_by_bucket()
        cumulative: dict[str, float] = {}
        running = 0.0
        for b in BUCKETS:
            running += gap[b]
            cumulative[b] = running
        return cumulative

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all positions to a tidy DataFrame.

        Columns: position_id, label, side, instrument, notional, rate,
                 rate_type, start_date, maturity_date, repricing_date,
                 spread, currency, remaining_years, time_to_reprice,
                 repricing_bucket, signed_notional
        """
        records = []
        for p in self:
            records.append({
                "position_id"    : p.position_id,
                "label"          : p.label,
                "side"           : p.side,
                "instrument"     : p.instrument,
                "notional"       : p.notional,
                "rate"           : p.rate,
                "rate_type"      : p.rate_type,
                "start_date"     : p.start_date,
                "maturity_date"  : p.maturity_date,
                "repricing_date" : p.repricing_date,
                "spread"         : p.spread,
                "currency"       : p.currency,
                "remaining_years": round(p.remaining_years, 4),
                "time_to_reprice": round(p.time_to_reprice, 4),
                "repricing_bucket": p.repricing_bucket,
                "signed_notional": p.signed_notional,
            })
        return pd.DataFrame(records)

    def summary(self) -> pd.DataFrame:
        """
        One-row-per-bucket summary table with:
            - asset notional
            - liability notional
            - gap
            - cumulative gap
        """
        gap  = self.gap_by_bucket()
        cgap = self.cumulative_gap()

        asset_by_bucket = {b: 0.0 for b in BUCKETS}
        liab_by_bucket  = {b: 0.0 for b in BUCKETS}

        for p in self.assets:
            asset_by_bucket[p.repricing_bucket] += p.notional
        for p in self.liabilities:
            liab_by_bucket[p.repricing_bucket]  += p.notional

        rows = []
        for b in BUCKETS:
            rows.append({
                "bucket"          : b,
                "assets_€M"       : round(asset_by_bucket[b], 2),
                "liabilities_€M"  : round(liab_by_bucket[b],  2),
                "gap_€M"          : round(gap[b],  2),
                "cumulative_gap_€M": round(cgap[b], 2),
            })
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"BalanceSheet('{self.name}' | "
            f"{len(self.assets)} assets €{self.total_assets:.0f}M | "
            f"{len(self.liabilities)} liabilities €{self.total_liabilities:.0f}M)"
        )


# ---------------------------------------------------------------------------
# SYNTHETIC BALANCE SHEET FACTORY
# ---------------------------------------------------------------------------

def make_synthetic_balance_sheet(as_of: date | None = None) -> BalanceSheet:
    """
    Build a representative synthetic banking book for testing and demos.

    The portfolio includes:
        Assets     : fixed-rate mortgages, floating-rate corporate loans, government bonds
        Liabilities: demand deposits (NMD), term deposits, wholesale funding

    Args:
        as_of : reference date (defaults to today)

    Returns:
        A populated BalanceSheet instance.
    """
    if as_of is None:
        as_of = date.today()

    bs = BalanceSheet(name="Synthetic Banking Book")

    # -- ASSETS ------------------------------------------------------------

    # Fixed-rate mortgages (long duration → reprices at maturity)
    for i in range(1, 6):
        years = 10 + i * 2          # 12y to 20y
        bs.add(Position(
            position_id   = f"MTG_{i:03d}",
            label         = f"Fixed mortgage {years}y",
            side          = "asset",
            instrument    = "mortgage",
            notional      = 200.0 + i * 50,
            rate          = 0.030 + i * 0.002,
            rate_type     = "fixed",
            start_date    = as_of.replace(year=as_of.year - 2),
            maturity_date = as_of.replace(year=as_of.year + years),
            repricing_date= as_of.replace(year=as_of.year + years),  # reprices at maturity
        ))

    # Floating-rate corporate loans (reprices every 3 months)
    for i in range(1, 5):
        bs.add(Position(
            position_id   = f"CORP_{i:03d}",
            label         = f"Floating corp loan {i}",
            side          = "asset",
            instrument    = "floating_rate_loan",
            notional      = 150.0 + i * 30,
            rate          = 0.045 + i * 0.005,
            rate_type     = "floating",
            start_date    = as_of.replace(year=as_of.year - 1),
            maturity_date = as_of.replace(year=as_of.year + 3),
            repricing_date= as_of.replace(month=as_of.month + 3
                                          if as_of.month <= 9
                                          else as_of.month - 9,
                                          year=as_of.year
                                          if as_of.month <= 9
                                          else as_of.year + 1),
            spread        = 0.010 + i * 0.002,
        ))

    # Government bonds (fixed rate, 5y)
    for i in range(1, 4):
        bs.add(Position(
            position_id   = f"BOND_{i:03d}",
            label         = f"Government bond 5y #{i}",
            side          = "asset",
            instrument    = "bond",
            notional      = 300.0,
            rate          = 0.025 + i * 0.001,
            rate_type     = "fixed",
            start_date    = as_of.replace(year=as_of.year - 1),
            maturity_date = as_of.replace(year=as_of.year + 5),
            repricing_date= as_of.replace(year=as_of.year + 5),
        ))

    # -- LIABILITIES -------------------------------------------------------

    # Demand deposits — NMD (non-maturity deposits, behaviorally modelled)
    bs.add(Position(
        position_id   = "NMD_001",
        label         = "Retail demand deposits",
        side          = "liability",
        instrument    = "demand_deposit",
        notional      = 1_500.0,
        rate          = 0.005,
        rate_type     = "nmd",
        start_date    = as_of.replace(year=as_of.year - 5),
        maturity_date = as_of.replace(year=as_of.year + 5),  # behavioural maturity
        repricing_date= as_of.replace(year=as_of.year + 1),  # assumed repricing horizon
    ))
    bs.add(Position(
        position_id   = "NMD_002",
        label         = "Corporate current accounts",
        side          = "liability",
        instrument    = "demand_deposit",
        notional      = 800.0,
        rate          = 0.010,
        rate_type     = "nmd",
        start_date    = as_of.replace(year=as_of.year - 3),
        maturity_date = as_of.replace(year=as_of.year + 2),
        repricing_date= as_of.replace(year=as_of.year + 1),
    ))

    # Term deposits (fixed rate, short term)
    for i in range(1, 4):
        months = i * 6           # 6M, 12M, 18M
        rep_date = date(
            as_of.year + (as_of.month + months - 1) // 12,
            (as_of.month + months - 1) % 12 + 1,
            1,
        )
        bs.add(Position(
            position_id   = f"TD_{i:03d}",
            label         = f"Term deposit {months}M",
            side          = "liability",
            instrument    = "term_deposit",
            notional      = 400.0 - i * 50,
            rate          = 0.020 + i * 0.005,
            rate_type     = "fixed",
            start_date    = as_of,
            maturity_date = rep_date,
            repricing_date= rep_date,
        ))

    # Wholesale funding (medium term, fixed)
    bs.add(Position(
        position_id   = "WF_001",
        label         = "Senior unsecured 3y",
        side          = "liability",
        instrument    = "wholesale_funding",
        notional      = 500.0,
        rate          = 0.032,
        rate_type     = "fixed",
        start_date    = as_of.replace(year=as_of.year - 1),
        maturity_date = as_of.replace(year=as_of.year + 3),
        repricing_date= as_of.replace(year=as_of.year + 3),
    ))
    bs.add(Position(
        position_id   = "WF_002",
        label         = "Covered bond 5y",
        side          = "liability",
        instrument    = "wholesale_funding",
        notional      = 300.0,
        rate          = 0.028,
        rate_type     = "fixed",
        start_date    = as_of.replace(year=as_of.year - 2),
        maturity_date = as_of.replace(year=as_of.year + 5),
        repricing_date= as_of.replace(year=as_of.year + 5),
    ))

    return bs


# ---------------------------------------------------------------------------
# QUICK SANITY CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bs = make_synthetic_balance_sheet()
    print(bs)
    print(f"\nTotal assets      : €{bs.total_assets:,.0f}M")
    print(f"Total liabilities : €{bs.total_liabilities:,.0f}M")
    print(f"NII proxy         : €{bs.nii_proxy():,.1f}M")
    print(f"Avg asset rate    : {bs.weighted_average_rate('asset')*100:.2f}%")
    print(f"Avg liability rate: {bs.weighted_average_rate('liability')*100:.2f}%")
    print("\nRepricing gap summary:")
    print(bs.summary().to_string(index=False))