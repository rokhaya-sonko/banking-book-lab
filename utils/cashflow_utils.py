"""
Cash flow generation utilities used across IRR, liquidity and FTP projects.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from utils.config import ALM_BUCKETS


def generate_fixed_cashflows(
    notional: float,
    coupon_rate: float,
    maturity_years: float,
    freq: int = 2,
    start_year: float = 0.0,
) -> pd.Series:
    """
    Generate cash flows for a fixed-rate instrument.

    Parameters
    ----------
    notional : float
        Face value of the instrument.
    coupon_rate : float
        Annual coupon rate in decimal (0.04 = 4%).
    maturity_years : float
        Maturity in years.
    freq : int
        Coupon payments per year. 2 = semi-annual.
    start_year : float
        Start date offset in years. 0 = today.
    """
    dt = 1 / freq
    periods = np.arange(start_year + dt, start_year + maturity_years + dt / 2, dt)
    coupon = notional * coupon_rate / freq

    cfs = {round(t, 6): coupon for t in periods}
    # Add principal repayment at maturity
    cfs[round(periods[-1], 6)] += notional

    return pd.Series(cfs, name="cashflow")


def generate_floating_cashflows(
    notional: float,
    spread: float,
    forward_rates: pd.Series,
    maturity_years: float,
    freq: int = 4,
    start_year: float = 0.0,
) -> pd.Series:
    """
    Generate cash flows for a floating-rate instrument using forward rates.

    Parameters
    ----------
    notional : float
        Face value of the instrument.
    spread : float
        Spread over the index rate in decimal (0.012 = 120bps).
    forward_rates : pd.Series
        Forward rates indexed by time in years.
    maturity_years : float
        Maturity in years.
    freq : int
        Reset frequency per year. 4 = quarterly.
    start_year : float
        Start date offset in years.
    """
    interp = interp1d(
        forward_rates.index, forward_rates.values,
        kind="linear", fill_value="extrapolate"
    )

    dt = 1 / freq
    periods = np.arange(start_year + dt, start_year + maturity_years + dt / 2, dt)

    cfs = {}
    for t in periods:
        # Coupon = (forward rate at reset date + spread) × notional × period
        rate = float(interp(t)) + spread
        cfs[round(t, 6)] = notional * rate / freq

    # Add principal repayment at maturity
    cfs[round(periods[-1], 6)] += notional

    return pd.Series(cfs, name="cashflow")


def bucket_cashflows(
    cashflows: pd.Series,
    buckets: list = None,
) -> pd.Series:
    """
    Aggregate cash flows into standard ALM time buckets.

    Parameters
    ----------
    cashflows : pd.Series
        Index = time in years, values = cash flow amounts.
    buckets : list
        Bucket boundaries in years. Defaults to ALM_BUCKETS from config.
    """
    if buckets is None:
        buckets = ALM_BUCKETS

    labels = [f"{buckets[i]}Y-{buckets[i+1]}Y" for i in range(len(buckets) - 1)]
    bins = pd.cut(cashflows.index, bins=buckets, labels=labels, right=True)

    return cashflows.groupby(bins, observed=True).sum()
