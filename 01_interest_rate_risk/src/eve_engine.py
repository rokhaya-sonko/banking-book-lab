"""
eve_engine.py
=============
Economic Value of Equity (EVE) engine for the IRRBB module.

EVE measures the present value of all future cash flows of the banking book:

    EVE = PV(all asset cash flows) - PV(all liability cash flows)

Under a rate shock, the discount curve shifts and all present values change:

    ΔEVE = EVE(shocked curve) - EVE(base curve)

A negative ΔEVE under a rate increase indicates that liabilities reprice
faster than assets (typical for banks with long-duration fixed-rate assets
funded by short-term deposits).

Key differences vs NII engine
------------------------------
    NII engine  : 12-month horizon, income statement impact
    EVE engine  : full maturity horizon, balance sheet / capital impact
                  → more sensitive to long-duration positions
                  → regulatory metric under BCBS 368 / EBA IRRBB

Cash flow generation
--------------------
    Fixed rate     : regular coupon payments + bullet principal at maturity
    Floating rate  : projected coupons using forward rates + bullet principal
    NMD            : modelled as a bullet payment at the behavioural maturity
                     (simplified — full behavioural model in behavioral_models.py)

Curve interface
---------------
This engine expects a simple curve object with a discount_factor(t) method.
If utils/curve_utils.py is available, pass a Curve instance.
Otherwise, a FlatCurve helper is provided for testing without curve_utils.

Downstream consumers
--------------------
    irrbb_scenarios.py : calls run() for each of the 6 BCBS 368 scenarios
    notebook.ipynb     : uses position_df for waterfall / contribution charts
    plot_utils.py      : plot_eve_sensitivity() consumes scenario ΔEVE values
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Protocol

import numpy as np
import pandas as pd

from balance_sheet import BalanceSheet, Position


# ---------------------------------------------------------------------------
# CURVE PROTOCOL
# ---------------------------------------------------------------------------

class CurveProtocol(Protocol):
    """
    Minimal interface expected from a yield curve object.
    Compatible with utils/curve_utils.Curve.
    """
    def discount_factor(self, t: float) -> float:
        """Return P(0, t) — the discount factor for maturity t years."""
        ...

    def zero_rate(self, t: float) -> float:
        """Return the continuously compounded zero rate for maturity t."""
        ...


# ---------------------------------------------------------------------------
# FLAT CURVE — used for testing without curve_utils
# ---------------------------------------------------------------------------

class FlatCurve:
    """
    Flat yield curve at a constant rate.
    Useful for unit tests and quick sanity checks.

    Args:
        rate      : flat rate (decimal, e.g. 0.03 = 3%)
        shock_bps : parallel shift to apply on top of rate (bps)

    Usage:
        curve = FlatCurve(rate=0.03)
        shocked_curve = FlatCurve(rate=0.03, shock_bps=200)
    """

    def __init__(self, rate: float = 0.03, shock_bps: float = 0.0) -> None:
        self.rate = rate + shock_bps / 10_000

    def discount_factor(self, t: float) -> float:
        """P(0,t) = exp(-r * t) under continuous compounding."""
        if t <= 0:
            return 1.0
        return np.exp(-self.rate * t)

    def zero_rate(self, t: float) -> float:
        return self.rate

    def shift(self, shock_bps: float) -> "FlatCurve":
        """Return a new FlatCurve shifted by shock_bps."""
        return FlatCurve(rate=self.rate, shock_bps=shock_bps)


# ---------------------------------------------------------------------------
# EVE RESULT CONTAINERS
# ---------------------------------------------------------------------------

@dataclass
class EVEResult:
    """
    Output of a single EVEEngine.run() call.

    Attributes
    ----------
    shock_bps    : rate shock applied (bps)
    eve          : Economic Value of Equity (€M)
    pv_assets    : present value of all asset cash flows (€M)
    pv_liabs     : present value of all liability cash flows (€M)
    position_df  : per-position PV and contribution to EVE (€M)
    """
    shock_bps   : float
    eve         : float
    pv_assets   : float
    pv_liabs    : float
    position_df : pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"EVEResult(shock={self.shock_bps:+.0f}bps | "
            f"EVE=€{self.eve:.1f}M | "
            f"PV assets=€{self.pv_assets:.1f}M | "
            f"PV liabs=€{self.pv_liabs:.1f}M)"
        )


@dataclass
class EVEScenarioResult:
    """
    Comparison between a base EVE and a shocked EVE.

    Attributes
    ----------
    base         : EVEResult for zero shock
    shocked      : EVEResult for the applied shock
    delta_eve    : shocked.eve − base.eve (€M)
    delta_pct    : delta_eve as % of base EVE
    position_delta: per-position ΔPVE contribution (€M)
    """
    base          : EVEResult
    shocked       : EVEResult
    delta_eve     : float
    delta_pct     : float
    position_delta: pd.DataFrame

    def __repr__(self) -> str:
        sign = "+" if self.delta_eve >= 0 else ""
        return (
            f"EVEScenarioResult(shock={self.shocked.shock_bps:+.0f}bps | "
            f"ΔEVE={sign}{self.delta_eve:.2f}€M | {sign}{self.delta_pct:.1f}%)"
        )


# ---------------------------------------------------------------------------
# CASH FLOW GENERATOR
# ---------------------------------------------------------------------------

def generate_cash_flows(
    position : Position,
    as_of    : date,
    curve    : CurveProtocol,
    shock_bps: float = 0.0,
) -> list[tuple[float, float]]:
    """
    Generate the future cash flow schedule for a single position.

    Returns a list of (time_in_years, cash_flow_€M) tuples,
    where time_in_years is measured from as_of.

    Cash flow conventions
    ---------------------
    Fixed rate instrument:
        - Regular coupon payments every 6 months (simplified semi-annual)
        - Bullet principal repayment at maturity

    Floating rate instrument:
        - Pre-repricing coupons at current rate (semi-annual)
        - Post-repricing coupons at current rate + shock
        - Bullet principal at maturity

    NMD (non-maturity deposit):
        - Modelled as a single bullet payment at behavioural maturity
        - No intermediate coupon cash flows (simplification)

    Args:
        position  : Position object
        as_of     : valuation date
        curve     : discount curve (used for forward rate projection)
        shock_bps : parallel rate shock applied after repricing date

    Returns:
        List of (t_years, cash_flow_€M) — positive values only,
        sign applied in the EVE engine based on position.side.
    """
    cash_flows: list[tuple[float, float]] = []
    shock_decimal = shock_bps / 10_000

    remaining_years = max((position.maturity_date - as_of).days / 365.25, 0.0)

    if remaining_years <= 0:
        return []  # position has already matured

    # NMD: single bullet at behavioural maturity
    if position.rate_type == "nmd":
        cash_flows.append((remaining_years, position.notional))
        return cash_flows

    # Semi-annual coupon schedule
    coupon_times: list[float] = []
    t = 0.5
    while t < remaining_years:
        coupon_times.append(round(t, 4))
        t += 0.5
    # Final period may be shorter than 6 months
    coupon_times.append(round(remaining_years, 4))

    repricing_years = max((position.repricing_date - as_of).days / 365.25, 0.0)

    prev_t = 0.0
    for t in coupon_times:
        period_length = t - prev_t  # actual length of this coupon period

        if position.rate_type == "fixed":
            # Fixed: same rate throughout
            coupon = position.notional * position.rate * period_length

        elif position.rate_type == "floating":
            # Floating: current rate before repricing, shocked rate after
            if t <= repricing_years:
                effective_rate = position.rate
            else:
                effective_rate = position.rate + shock_decimal
            coupon = position.notional * effective_rate * period_length

        else:
            coupon = 0.0

        cash_flows.append((t, coupon))
        prev_t = t

    # Principal repayment at maturity (bullet)
    cash_flows.append((remaining_years, position.notional))

    return cash_flows


# ---------------------------------------------------------------------------
# EVE ENGINE
# ---------------------------------------------------------------------------

class EVEEngine:
    """
    Computes EVE and ΔEVE from a BalanceSheet and a yield curve.

    Args:
        balance_sheet : populated BalanceSheet instance
        base_curve    : base yield curve (zero shock)
                        Must implement discount_factor(t) and zero_rate(t).
                        Pass a FlatCurve for testing without curve_utils.
        as_of         : valuation date (default today)

    Usage
    -----
    >>> from balance_sheet import make_synthetic_balance_sheet
    >>> bs     = make_synthetic_balance_sheet()
    >>> curve  = FlatCurve(rate=0.03)
    >>> engine = EVEEngine(bs, curve)
    >>> base   = engine.run(shock_bps=0)
    >>> up200  = engine.run(shock_bps=200)
    >>> result = engine.compare(shock_bps=200)
    >>> print(result)
    """

    def __init__(
        self,
        balance_sheet: BalanceSheet,
        base_curve   : CurveProtocol,
        as_of        : Optional[date] = None,
    ) -> None:
        self.bs         = balance_sheet
        self.base_curve = base_curve
        self.as_of      = as_of or date.today()

    # ------------------------------------------------------------------
    # Shocked curve builder
    # ------------------------------------------------------------------

    def _build_shocked_curve(self, shock_bps: float) -> CurveProtocol:
        """
        Apply a parallel shock to the base curve.

        If the base curve is a FlatCurve, use its shift() method.
        If the base curve is a utils.curve_utils.Curve, use its shift() method.
        Falls back to wrapping the base curve in a ShiftedCurve wrapper.
        """
        # FlatCurve
        if isinstance(self.base_curve, FlatCurve):
            return FlatCurve(
                rate      = self.base_curve.rate,
                shock_bps = shock_bps,
            )

        # utils.curve_utils.Curve — attempt native shift
        if hasattr(self.base_curve, "shift"):
            return self.base_curve.shift(shock_bps / 10_000)

        # Generic fallback: wrap in a parallel-shifted adapter
        return _ParallelShiftedCurve(self.base_curve, shock_bps)

    # ------------------------------------------------------------------
    # Present value of a single position
    # ------------------------------------------------------------------

    def _pv_position(
        self,
        position : Position,
        curve    : CurveProtocol,
        shock_bps: float,
    ) -> float:
        """
        Compute the present value of all cash flows for a single position.

        Args:
            position  : Position object
            curve     : (shocked) discount curve
            shock_bps : passed to generate_cash_flows for coupon projection

        Returns:
            PV in €M (always positive — sign applied at aggregation level)
        """
        cash_flows = generate_cash_flows(position, self.as_of, curve, shock_bps)
        pv = sum(
            cf * curve.discount_factor(t)
            for t, cf in cash_flows
            if t > 0
        )
        return round(pv, 4)

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------

    def run(self, shock_bps: float = 0.0) -> EVEResult:
        """
        Compute EVE under a parallel rate shock.

        Args:
            shock_bps : parallel rate shift in basis points

        Returns:
            EVEResult with eve, pv_assets, pv_liabs, position_df
        """
        curve = self._build_shocked_curve(shock_bps)

        records   : list[dict] = []
        pv_assets : float = 0.0
        pv_liabs  : float = 0.0

        for p in self.bs:
            pv = self._pv_position(p, curve, shock_bps)

            if p.side == "asset":
                pv_assets += pv
                eve_contribution = pv        # assets add to EVE
            else:
                pv_liabs  += pv
                eve_contribution = -pv       # liabilities subtract from EVE

            records.append({
                "position_id"     : p.position_id,
                "label"           : p.label,
                "side"            : p.side,
                "instrument"      : p.instrument,
                "notional_€M"     : p.notional,
                "pv_€M"           : pv,
                "eve_contribution": round(eve_contribution, 4),
            })

        eve = round(pv_assets - pv_liabs, 4)

        position_df = (
            pd.DataFrame(records)
            .sort_values("eve_contribution", ascending=False)
            .reset_index(drop=True)
        )

        return EVEResult(
            shock_bps   = shock_bps,
            eve         = eve,
            pv_assets   = round(pv_assets, 4),
            pv_liabs    = round(pv_liabs, 4),
            position_df = position_df,
        )

    # ------------------------------------------------------------------
    # Scenario comparison
    # ------------------------------------------------------------------

    def compare(self, shock_bps: float) -> EVEScenarioResult:
        """
        Compare EVE in the base scenario (0bp) vs a shocked scenario.

        Args:
            shock_bps : shock to apply (bps)

        Returns:
            EVEScenarioResult with delta_eve, delta_pct, position_delta
        """
        base    = self.run(shock_bps=0.0)
        shocked = self.run(shock_bps=shock_bps)

        delta_eve = shocked.eve - base.eve
        delta_pct = (delta_eve / abs(base.eve) * 100) if base.eve else np.nan

        # Per-position ΔEVE contribution
        base_pos    = base.shocked.position_df    if hasattr(base, "shocked") else base.position_df
        shocked_pos = shocked.position_df

        position_delta = base.position_df[["position_id", "label", "side", "notional_€M"]].copy()
        position_delta["pv_base_€M"]    = base.position_df["pv_€M"].values
        position_delta["pv_shocked_€M"] = shocked.position_df["pv_€M"].values
        position_delta["delta_pv_€M"]   = (
            shocked.position_df["pv_€M"].values - base.position_df["pv_€M"].values
        ).round(4)
        position_delta["eve_delta_contribution"] = (
            shocked.position_df["eve_contribution"].values
            - base.position_df["eve_contribution"].values
        ).round(4)

        return EVEScenarioResult(
            base          = base,
            shocked       = shocked,
            delta_eve     = round(delta_eve, 4),
            delta_pct     = round(delta_pct, 2),
            position_delta= position_delta,
        )

    # ------------------------------------------------------------------
    # Multi-scenario sweep
    # ------------------------------------------------------------------

    def scenario_sweep(
        self,
        shocks_bps: Optional[list[float]] = None,
    ) -> pd.DataFrame:
        """
        Compute ΔEVE for a list of parallel shocks.

        Args:
            shocks_bps : list of shocks in bps
                         (default: [-300, -200, -100, 0, +100, +200, +300])

        Returns:
            DataFrame with columns:
                shock_bps | eve_€M | delta_eve_€M | delta_pct

        Usage in irrbb_scenarios.py:
            sweep = eve_engine.scenario_sweep()
            scenarios_dict = dict(zip(
                [f"{s:+d}bp" for s in sweep["shock_bps"]],
                sweep["delta_eve_€M"]
            ))
            plot_eve_sensitivity(scenarios_dict)
        """
        if shocks_bps is None:
            shocks_bps = [-300, -200, -100, -50, 0, 50, 100, 200, 300]

        base_eve = self.run(shock_bps=0.0).eve
        rows     = []

        for shock in shocks_bps:
            eve   = self.run(shock_bps=shock).eve
            delta = eve - base_eve
            pct   = (delta / abs(base_eve) * 100) if base_eve else np.nan

            rows.append({
                "shock_bps"   : shock,
                "eve_€M"      : round(eve, 4),
                "delta_eve_€M": round(delta, 4),
                "delta_pct"   : round(pct, 2),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def print_summary(self, shock_bps: float = 200.0) -> None:
        """Print a formatted EVE scenario comparison to stdout."""
        result = self.compare(shock_bps=shock_bps)
        sign   = "+" if result.delta_eve >= 0 else ""

        print(f"\n{'='*60}")
        print(f"  EVE ENGINE — {self.bs.name}")
        print(f"  Shock : {shock_bps:+.0f}bps")
        print(f"{'='*60}")
        print(f"  PV Assets  (base)    : €{result.base.pv_assets:>10,.2f}M")
        print(f"  PV Assets  (shocked) : €{result.shocked.pv_assets:>10,.2f}M")
        print(f"  PV Liabs   (base)    : €{result.base.pv_liabs:>10,.2f}M")
        print(f"  PV Liabs   (shocked) : €{result.shocked.pv_liabs:>10,.2f}M")
        print(f"  EVE        (base)    : €{result.base.eve:>10,.2f}M")
        print(f"  EVE        (shocked) : €{result.shocked.eve:>10,.2f}M")
        print(f"  ΔEVE                 : {sign}€{result.delta_eve:>9,.2f}M ({sign}{result.delta_pct:.1f}%)")

        print(f"\n  Top 5 ΔEVE contributors:")
        top5 = result.position_delta.nlargest(5, "eve_delta_contribution")
        print(top5[["position_id", "label", "side", "notional_€M",
                     "delta_pv_€M", "eve_delta_contribution"]].to_string(index=False))


# ---------------------------------------------------------------------------
# PARALLEL SHIFTED CURVE ADAPTER
# ---------------------------------------------------------------------------

class _ParallelShiftedCurve:
    """
    Wraps any curve object and applies a parallel shift.
    Used as a fallback when the curve does not expose a shift() method.

    Args:
        base_curve : original curve implementing CurveProtocol
        shock_bps  : parallel shift in basis points
    """

    def __init__(self, base_curve: CurveProtocol, shock_bps: float) -> None:
        self._base  = base_curve
        self._shift = shock_bps / 10_000

    def discount_factor(self, t: float) -> float:
        """Apply parallel shift: P_shocked(t) = exp(-(r(t) + shift) * t)."""
        if t <= 0:
            return 1.0
        r_base = self._base.zero_rate(t)
        return np.exp(-(r_base + self._shift) * t)

    def zero_rate(self, t: float) -> float:
        return self._base.zero_rate(t) + self._shift


# ---------------------------------------------------------------------------
# QUICK SANITY CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from balance_sheet import make_synthetic_balance_sheet

    bs     = make_synthetic_balance_sheet()
    curve  = FlatCurve(rate=0.03)
    engine = EVEEngine(bs, curve)

    print("=== EVE scenario sweep ===")
    sweep = engine.scenario_sweep()
    print(sweep.to_string(index=False))

    print("\n=== +200bp comparison ===")
    engine.print_summary(shock_bps=200)

    print("\n=== -200bp comparison ===")
    engine.print_summary(shock_bps=-200)