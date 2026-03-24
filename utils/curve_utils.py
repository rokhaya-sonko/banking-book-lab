"""
Yield curve utilities used across all projects.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def discount_factor(rate: float, maturity: float, compounding: str = "continuous") -> float:
    """
    Compute a discount factor given a rate and maturity.

    Parameters
    ----------
    rate : float
        Annual rate in decimal (0.04 = 4%).
    maturity : float
        Time in years.
    compounding : str
        'continuous', 'annual', or 'semi-annual'.
    """
    if compounding == "continuous":
        return np.exp(-rate * maturity)
    elif compounding == "annual":
        return 1 / (1 + rate) ** maturity
    elif compounding == "semi-annual":
        return 1 / (1 + rate / 2) ** (2 * maturity)
    else:
        raise ValueError(f"Unknown compounding: {compounding}")


def forward_rate(zero_rates: pd.Series, t1: float, t2: float) -> float:
    """
    Compute the continuously compounded forward rate between t1 and t2.

    Parameters
    ----------
    zero_rates : pd.Series
        Index = maturities in years, values = zero rates in decimal.
    t1, t2 : float
        Start and end of the forward period (t2 > t1).
    """
    interp = interp1d(
        zero_rates.index, zero_rates.values,
        kind="linear", fill_value="extrapolate"
    )
    r1 = float(interp(t1))
    r2 = float(interp(t2))
    return (r2 * t2 - r1 * t1) / (t2 - t1)


def bootstrap_zero_curve(
    maturities: list,
    par_rates: list,
    freq: int = 2,
) -> pd.Series:
    """
    Bootstrap a zero-coupon curve from par swap or bond rates.

    Parameters
    ----------
    maturities : list
        Instrument maturities in years e.g. [0.5, 1, 2, 5, 10].
    par_rates : list
        Par rates in decimal e.g. 0.04 for 4%.
    freq : int
        Coupon frequency per year. 2 = semi-annual.
    """
    dt = 1 / freq
    zero_rates = {}
    discount_factors = {}

    for T, c in zip(maturities, par_rates):
        coupon = c / freq
        # Sum discounted coupons for all prior periods
        pv_coupons = sum(
            coupon * discount_factors[t]
            for t in np.arange(dt, T, dt)
            if t in discount_factors
        )
        # Solve for the terminal discount factor
        df_T = (1 - pv_coupons) / (1 + coupon)
        discount_factors[T] = df_T
        # Convert to continuously compounded zero rate
        zero_rates[T] = -np.log(df_T) / T

    return pd.Series(zero_rates, name="zero_rate")


def apply_shock(zero_curve: pd.Series, shock_bps) -> pd.Series:
    """
    Apply a rate shock to a zero curve.

    Parameters
    ----------
    zero_curve : pd.Series
        Index = maturities, values = zero rates.
    shock_bps : int or dict
        Parallel shock in bps if int.
        Non-parallel shock if dict with 'short' and 'long' keys.
        Pivot point is fixed at 2 years.
    """
    if isinstance(shock_bps, (int, float)):
        return zero_curve + shock_bps / 10_000

    shocked = zero_curve.copy()
    pivot = 2.0
    for mat in shocked.index:
        if mat <= pivot:
            shocked[mat] += shock_bps.get("short", 0) / 10_000
        else:
            shocked[mat] += shock_bps.get("long", 0) / 10_000
    return shocked
