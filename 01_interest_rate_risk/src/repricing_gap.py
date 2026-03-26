"""
repricing_gap.py
================
Repricing gap analysis for the IRRBB module.

The repricing gap measures the mismatch between assets and liabilities
that reprice (i.e. reset to market rates) within each time bucket.

    Gap(bucket) = RSA(bucket) - RSL(bucket)

Where:
    RSA = Rate-Sensitive Assets   repricing in that bucket
    RSL = Rate-Sensitive Liabilities repricing in that bucket

A positive gap is asset-sensitive  → NII rises when rates go up.
A negative gap is liability-sensitive → NII falls when rates go up.

The NII sensitivity approximation used here is:
    ΔNII ≈ Gap(bucket) × Δr × time_factor(bucket)

Where time_factor is the fraction of the year remaining in the bucket
(how long the repriced position earns the new rate within the 12M horizon).

Downstream consumers
--------------------
    nii_engine.py      : uses gap_df and nii_sensitivity()
    irrbb_scenarios.py : calls nii_sensitivity() for each scenario shock
    plot_utils.py      : plot_repricing_gap() consumes gap_df columns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from balance_sheet import BalanceSheet, BUCKETS, BUCKET_UPPER_YEARS


# ---------------------------------------------------------------------------
# TIME FACTORS
# ---------------------------------------------------------------------------
# For each bucket, the time factor represents the fraction of a 12-month
# NII horizon during which the repriced position earns the new rate.
# Convention: midpoint of the bucket expressed as a fraction of 1 year.
# Positions repricing in O/N earn the new rate for ~1 year; positions
# repricing in the 10Y bucket only contribute within the 1Y horizon if
# they fall inside it — beyond 1Y we cap at 0 for the 12M NII metric.

BUCKET_MIDPOINTS_YEARS: dict[str, float] = {
    "O/N" : 0.5 / 365,
    "1W"  : 0.5 * 7 / 365,
    "1M"  : 0.5 / 12,
    "3M"  : 2 / 12,
    "6M"  : 4.5 / 12,
    "9M"  : 7.5 / 12,
    "1Y"  : 10.5 / 12,
    # Buckets beyond 1Y: repricing occurs outside the 12M NII horizon
    # → time factor = 0 for the standard 12M NII calculation
    "2Y"  : 0.0,
    "3Y"  : 0.0,
    "5Y"  : 0.0,
    "10Y" : 0.0,
    "15Y" : 0.0,
    "20Y" : 0.0,
    ">20Y": 0.0,
}


# ---------------------------------------------------------------------------
# REPRICING GAP RESULT
# ---------------------------------------------------------------------------

@dataclass
class RepricingGapResult:
    """
    Container for repricing gap outputs.

    Attributes
    ----------
    gap_df        : tidy DataFrame with one row per bucket (see columns below)
    total_rsa     : total rate-sensitive assets (€M)
    total_rsl     : total rate-sensitive liabilities (€M)
    net_gap       : total gap across all buckets (€M)

    gap_df columns
    --------------
    bucket            : time bucket label
    assets_€M         : RSA in bucket
    liabilities_€M    : RSL in bucket  (positive, shown as positive)
    gap_€M            : RSA − RSL
    cumulative_gap_€M : running sum of gap
    time_factor       : fraction of 12M horizon (for NII sensitivity)
    nii_sensitivity_1bp: ΔNII for a 1bp parallel shock (€M)
    """
    gap_df    : pd.DataFrame
    total_rsa : float
    total_rsl : float
    net_gap   : float


# ---------------------------------------------------------------------------
# REPRICING GAP ENGINE
# ---------------------------------------------------------------------------

class RepricingGap:
    """
    Computes the repricing gap from a BalanceSheet instance.

    Args:
        balance_sheet : populated BalanceSheet object
        nmd_bucket    : bucket to assign NMD positions to (default '1Y')
                        Override to reflect your behavioural model assumption.

    Usage
    -----
    >>> from balance_sheet import make_synthetic_balance_sheet
    >>> bs  = make_synthetic_balance_sheet()
    >>> rg  = RepricingGap(bs)
    >>> result = rg.compute()
    >>> print(result.gap_df)
    >>> delta_nii = rg.nii_sensitivity(shock_bps=200)
    """

    def __init__(
        self,
        balance_sheet: BalanceSheet,
        nmd_bucket: str = "1Y",
    ) -> None:
        self.bs         = balance_sheet
        self.nmd_bucket = nmd_bucket
        self._result: Optional[RepricingGapResult] = None

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self, force: bool = False) -> RepricingGapResult:
        """
        Build the repricing gap table.

        Results are cached after the first call. Pass force=True to
        recompute (e.g. after modifying the balance sheet).

        Returns:
            RepricingGapResult with gap_df and summary statistics.
        """
        if self._result is not None and not force:
            return self._result

        # Accumulators per bucket
        rsa: dict[str, float] = {b: 0.0 for b in BUCKETS}
        rsl: dict[str, float] = {b: 0.0 for b in BUCKETS}

        for p in self.bs:
            # Override NMD bucket assignment with the behavioral assumption
            bucket = self.nmd_bucket if p.rate_type == "nmd" else p.repricing_bucket

            if p.side == "asset":
                rsa[bucket] += p.notional
            else:
                rsl[bucket] += p.notional

        # Build gap DataFrame
        rows        = []
        cumulative  = 0.0

        for b in BUCKETS:
            gap        = rsa[b] - rsl[b]
            cumulative += gap
            tf         = BUCKET_MIDPOINTS_YEARS[b]
            # NII sensitivity for 1bp shock: gap × 0.0001 × time_factor
            nii_1bp    = gap * 0.0001 * tf

            rows.append({
                "bucket"              : b,
                "assets_€M"           : round(rsa[b], 2),
                "liabilities_€M"      : round(rsl[b], 2),
                "gap_€M"              : round(gap, 2),
                "cumulative_gap_€M"   : round(cumulative, 2),
                "time_factor"         : round(tf, 4),
                "nii_sensitivity_1bp" : round(nii_1bp, 4),
            })

        gap_df = pd.DataFrame(rows)

        self._result = RepricingGapResult(
            gap_df    = gap_df,
            total_rsa = round(sum(rsa.values()), 2),
            total_rsl = round(sum(rsl.values()), 2),
            net_gap   = round(sum(rsa[b] - rsl[b] for b in BUCKETS), 2),
        )
        return self._result

    # ------------------------------------------------------------------
    # NII sensitivity
    # ------------------------------------------------------------------

    def nii_sensitivity(
        self,
        shock_bps: float,
        buckets: Optional[list[str]] = None,
    ) -> float:
        """
        Approximate ΔNII for a parallel rate shock over a 12-month horizon.

            ΔNII ≈ Σ_bucket [ Gap(b) × shock × time_factor(b) ]

        Args:
            shock_bps : rate shock in basis points (e.g. 200 for +200bp)
            buckets   : restrict calculation to specific buckets (default: all)

        Returns:
            ΔNII in €M (positive = NII increases, negative = NII decreases)

        Note:
            This is a static gap sensitivity — it does not account for:
            - New business / balance sheet dynamics
            - Non-linear effects (floors, caps, options)
            - Behavioural changes in NMD volumes
            Use nii_engine.py for a full dynamic simulation.
        """
        result = self.compute()
        df     = result.gap_df

        if buckets:
            df = df[df["bucket"].isin(buckets)]

        shock_decimal = shock_bps / 10_000
        delta_nii     = (df["gap_€M"] * shock_decimal * df["time_factor"]).sum()
        return round(delta_nii, 4)

    def nii_sensitivity_by_bucket(self, shock_bps: float) -> pd.Series:
        """
        ΔNII contribution per bucket for a given parallel shock.

        Useful for identifying which buckets drive the NII sensitivity.

        Args:
            shock_bps : rate shock in basis points

        Returns:
            pd.Series indexed by bucket with ΔNII contribution (€M)
        """
        result        = self.compute()
        shock_decimal = shock_bps / 10_000
        contributions = (
            result.gap_df.set_index("bucket")["gap_€M"]
            * shock_decimal
            * result.gap_df.set_index("bucket")["time_factor"]
        )
        return contributions.round(4)

    # ------------------------------------------------------------------
    # Scenario sweep
    # ------------------------------------------------------------------

    def scenario_sweep(
        self,
        shocks_bps: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Compute NII sensitivity for a range of parallel shocks.

        Args:
            shocks_bps : list of shocks in bps (default: -300 to +300 in 100bp steps)

        Returns:
            DataFrame with columns ['shock_bps', 'delta_nii_€M', 'pct_of_nii_baseline']

        Usage (consumed by nii_engine and irrbb_scenarios):
            sweep = rg.scenario_sweep()
            sweep.plot(x='shock_bps', y='delta_nii_€M')
        """
        if shocks_bps is None:
            shocks_bps = [-300, -200, -100, -50, 0, 50, 100, 200, 300]

        nii_baseline = self.bs.nii_proxy()

        rows = []
        for shock in shocks_bps:
            delta = self.nii_sensitivity(shock_bps=shock)
            pct   = (delta / nii_baseline * 100) if nii_baseline else np.nan
            rows.append({
                "shock_bps"          : shock,
                "delta_nii_€M"       : delta,
                "pct_of_nii_baseline": round(pct, 2),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a formatted repricing gap summary to stdout."""
        result = self.compute()
        print(f"\n{'='*60}")
        print(f"  REPRICING GAP — {self.bs.name}")
        print(f"{'='*60}")
        print(f"  Total RSA   : €{result.total_rsa:>10,.1f}M")
        print(f"  Total RSL   : €{result.total_rsl:>10,.1f}M")
        print(f"  Net gap     : €{result.net_gap:>10,.1f}M")
        sensitivity = (
            "asset-sensitive (NII ↑ when rates ↑)"
            if result.net_gap > 0
            else "liability-sensitive (NII ↓ when rates ↑)"
        )
        print(f"  Position    : {sensitivity}")
        print(f"\n{result.gap_df.to_string(index=False)}")

        print(f"\n  NII sensitivity (parallel shocks):")
        sweep = self.scenario_sweep()
        for _, row in sweep.iterrows():
            sign = "+" if row["delta_nii_€M"] >= 0 else ""
            print(
                f"    {row['shock_bps']:>+5.0f}bp : "
                f"ΔNII = {sign}{row['delta_nii_€M']:.2f} €M "
                f"({sign}{row['pct_of_nii_baseline']:.1f}% of baseline NII)"
            )


# ---------------------------------------------------------------------------
# QUICK SANITY CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from balance_sheet import make_synthetic_balance_sheet

    bs = make_synthetic_balance_sheet()
    rg = RepricingGap(bs, nmd_bucket="1Y")
    rg.print_summary()

    print("\nNII sensitivity by bucket (+200bp):")
    print(rg.nii_sensitivity_by_bucket(shock_bps=200).to_string())