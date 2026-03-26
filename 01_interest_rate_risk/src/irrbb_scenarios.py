"""
irrbb_scenarios.py
==================
BCBS 368 / EBA IRRBB regulatory scenario orchestrator.

Runs the 6 mandatory interest rate stress scenarios defined in
BCBS 368 (2016) and EBA IRRBB Guidelines (2022) across both
the NII engine and the EVE engine, and produces a consolidated
regulatory report.

The 6 BCBS 368 scenarios
--------------------------
1. Parallel Up     : uniform +shock across all tenors
2. Parallel Down   : uniform -shock across all tenors
3. Steepener       : short end down, long end up (curve steepens)
4. Flattener       : short end up, long end down (curve flattens)
5. Short Rate Up   : short end up, long end unchanged
6. Short Rate Down : short end down, long end unchanged

Regulatory thresholds (outlier bank criteria)
----------------------------------------------
- ΔEVE > 15% of Tier 1 capital  → supervisory concern
- ΔNII > 5%  of NII baseline    → supervisory concern
  (thresholds configurable via RegConfig)

Architecture
------------
This module depends on:
    nii_engine.py  → NIIEngine  (ΔNII per scenario)
    eve_engine.py  → EVEEngine  (ΔEVE per scenario)
    balance_sheet.py → BalanceSheet

It is the final consumer in the Module 01 dependency chain:
    balance_sheet → repricing_gap → nii_engine ─┐
                                                  ├─ irrbb_scenarios
                  balance_sheet → eve_engine  ───┘

Downstream consumers
--------------------
    notebook.ipynb : full interactive report
    plot_utils.py  : plot_nii_scenarios(), plot_eve_sensitivity()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from balance_sheet import BalanceSheet
from nii_engine import NIIEngine
from eve_engine import EVEEngine, FlatCurve, CurveProtocol


# ---------------------------------------------------------------------------
# SCENARIO DEFINITIONS
# ---------------------------------------------------------------------------

@dataclass
class RateScenario:
    """
    Definition of a single IRRBB rate scenario.

    Attributes
    ----------
    name          : scenario label (e.g. "Parallel Up")
    short_shock   : shock applied to short-end tenors (bps, ≤ 1Y)
    long_shock    : shock applied to long-end tenors  (bps, > 1Y)
    parallel_shock: convenience shortcut for NII engine
                    (uses short_shock for the parallel shift)
    description   : human-readable description
    """
    name          : str
    short_shock   : float   # bps — applied to tenors <= 1Y
    long_shock    : float   # bps — applied to tenors  > 1Y
    description   : str = ""

    @property
    def parallel_shock(self) -> float:
        """
        For the NII engine (which uses a single parallel shift),
        use the short_shock as the representative shock.
        This is conservative for asset-sensitive banks.
        """
        return self.short_shock

    def __repr__(self) -> str:
        return (
            f"RateScenario('{self.name}' | "
            f"short={self.short_shock:+.0f}bps | "
            f"long={self.long_shock:+.0f}bps)"
        )


# BCBS 368 standard scenario set
# Shock magnitudes are illustrative — in practice they depend on the
# historical rate distribution per currency (EBA prescribes currency-specific
# shock sizes; these are the EUR approximations for a 2024 environment).
BCBS368_SCENARIOS: list[RateScenario] = [
    RateScenario(
        name         = "Parallel Up",
        short_shock  = +200,
        long_shock   = +200,
        description  = "Uniform +200bp parallel shift across all tenors",
    ),
    RateScenario(
        name         = "Parallel Down",
        short_shock  = -200,
        long_shock   = -200,
        description  = "Uniform -200bp parallel shift across all tenors",
    ),
    RateScenario(
        name         = "Steepener",
        short_shock  = -100,
        long_shock   = +100,
        description  = "Short end -100bp / Long end +100bp — curve steepens",
    ),
    RateScenario(
        name         = "Flattener",
        short_shock  = +100,
        long_shock   = -100,
        description  = "Short end +100bp / Long end -100bp — curve flattens",
    ),
    RateScenario(
        name         = "Short Rate Up",
        short_shock  = +250,
        long_shock   = 0,
        description  = "Short end +250bp / Long end unchanged",
    ),
    RateScenario(
        name         = "Short Rate Down",
        short_shock  = -250,
        long_shock   = 0,
        description  = "Short end -250bp / Long end unchanged",
    ),
]


# ---------------------------------------------------------------------------
# REGULATORY CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass
class RegConfig:
    """
    Regulatory thresholds and capital assumptions.

    Attributes
    ----------
    tier1_capital_€M : Tier 1 capital (€M) — used for EVE outlier test
    eve_threshold_pct: ΔEVE / Tier1 threshold triggering outlier flag (%)
    nii_threshold_pct: ΔNII / NII baseline threshold triggering flag (%)
    """
    tier1_capital_€M  : float = 1_000.0   # €1bn default
    eve_threshold_pct : float = 15.0       # BCBS 368 outlier criterion
    nii_threshold_pct : float = 5.0        # EBA IRRBB supervisory benchmark


# ---------------------------------------------------------------------------
# IRRBB REPORT CONTAINER
# ---------------------------------------------------------------------------

@dataclass
class IRRBBReport:
    """
    Consolidated IRRBB regulatory report across all 6 scenarios.

    Attributes
    ----------
    summary_df       : one row per scenario with ΔNII, ΔEVE, outlier flags
    nii_baseline     : NII in the base scenario (€M) — for plot_nii_scenarios
    eve_baseline     : EVE in the base scenario (€M)
    nii_scenarios    : dict {scenario_name: delta_nii} — for plot_nii_scenarios
    eve_scenarios    : dict {scenario_name: delta_eve} — for plot_eve_sensitivity
    worst_nii        : scenario with the largest negative ΔNII
    worst_eve        : scenario with the largest negative ΔEVE
    outlier_flags    : list of (scenario_name, metric, value, threshold) breaches
    reg_config       : RegConfig used for threshold checks
    """
    summary_df   : pd.DataFrame
    nii_baseline : float
    eve_baseline : float
    nii_scenarios: dict[str, float]
    eve_scenarios: dict[str, float]
    worst_nii    : str
    worst_eve    : str
    outlier_flags: list[dict]
    reg_config   : RegConfig

    def __repr__(self) -> str:
        flags = len(self.outlier_flags)
        return (
            f"IRRBBReport("
            f"NII baseline=€{self.nii_baseline:.1f}M | "
            f"EVE baseline=€{self.eve_baseline:.1f}M | "
            f"outlier flags={flags})"
        )


# ---------------------------------------------------------------------------
# IRRBB SCENARIO ORCHESTRATOR
# ---------------------------------------------------------------------------

class IRRBBScenarios:
    """
    Runs all BCBS 368 scenarios and produces the regulatory IRRBB report.

    Args:
        balance_sheet : populated BalanceSheet instance
        base_curve    : base yield curve (zero shock)
        reg_config    : regulatory thresholds (default: RegConfig())
        scenarios     : list of RateScenario (default: BCBS368_SCENARIOS)
        horizon_months: NII simulation horizon (default 12)
        as_of         : valuation date (default today)

    Usage
    -----
    >>> bs       = make_synthetic_balance_sheet()
    >>> curve    = FlatCurve(rate=0.03)
    >>> irrbb    = IRRBBScenarios(bs, curve)
    >>> report   = irrbb.run_all()
    >>> irrbb.print_regulatory_report()
    """

    def __init__(
        self,
        balance_sheet  : BalanceSheet,
        base_curve     : CurveProtocol,
        reg_config     : Optional[RegConfig] = None,
        scenarios      : Optional[list[RateScenario]] = None,
        horizon_months : int = 12,
        as_of          = None,
    ) -> None:
        self.bs             = balance_sheet
        self.base_curve     = base_curve
        self.reg_config     = reg_config or RegConfig()
        self.scenarios      = scenarios or BCBS368_SCENARIOS
        self.horizon_months = horizon_months
        self.as_of          = as_of

        # Instantiate engines
        self.nii_engine = NIIEngine(
            balance_sheet  = balance_sheet,
            horizon_months = horizon_months,
            as_of          = as_of,
        )
        self.eve_engine = EVEEngine(
            balance_sheet = balance_sheet,
            base_curve    = base_curve,
            as_of         = as_of,
        )

        self._report: Optional[IRRBBReport] = None

    # ------------------------------------------------------------------
    # Outlier flag checker
    # ------------------------------------------------------------------

    def _check_outlier(
        self,
        scenario_name: str,
        delta_nii    : float,
        delta_eve    : float,
        nii_baseline : float,
        reg_config   : RegConfig,
    ) -> list[dict]:
        """
        Check whether a scenario breaches regulatory outlier thresholds.

        Returns a list of flag dicts (empty if no breach).
        """
        flags = []

        # EVE outlier test: |ΔEVE| / Tier1 > threshold
        if reg_config.tier1_capital_€M > 0:
            eve_ratio = abs(delta_eve) / reg_config.tier1_capital_€M * 100
            if eve_ratio > reg_config.eve_threshold_pct:
                flags.append({
                    "scenario"  : scenario_name,
                    "metric"    : "ΔEVE / Tier1",
                    "value_pct" : round(eve_ratio, 2),
                    "threshold" : reg_config.eve_threshold_pct,
                    "breach"    : True,
                })

        # NII outlier test: |ΔNII| / NII_baseline > threshold
        if nii_baseline != 0:
            nii_ratio = abs(delta_nii) / abs(nii_baseline) * 100
            if nii_ratio > reg_config.nii_threshold_pct:
                flags.append({
                    "scenario"  : scenario_name,
                    "metric"    : "ΔNII / NII baseline",
                    "value_pct" : round(nii_ratio, 2),
                    "threshold" : reg_config.nii_threshold_pct,
                    "breach"    : True,
                })

        return flags

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run_all(self, force: bool = False) -> IRRBBReport:
        """
        Run all BCBS 368 scenarios and return the consolidated report.

        Results are cached after the first call. Pass force=True to rerun.

        Returns:
            IRRBBReport with summary_df, scenario dicts, outlier flags
        """
        if self._report is not None and not force:
            return self._report

        # Base scenario values
        nii_base_result = self.nii_engine.run(rate_shock_bps=0.0)
        eve_base_result = self.eve_engine.run(shock_bps=0.0)
        nii_baseline    = nii_base_result.annual_nii
        eve_baseline    = eve_base_result.eve

        rows          : list[dict]  = []
        nii_scenarios : dict        = {}
        eve_scenarios : dict        = {}
        all_flags     : list[dict]  = []

        for scen in self.scenarios:
            # --- NII ---
            # NII engine uses a single parallel shift (short_shock)
            nii_shocked = self.nii_engine.run(rate_shock_bps=scen.parallel_shock)
            delta_nii   = nii_shocked.annual_nii - nii_baseline
            nii_pct     = (delta_nii / abs(nii_baseline) * 100) if nii_baseline else np.nan

            # --- EVE ---
            # EVE engine uses the long_shock (full-term impact)
            # For non-parallel scenarios we use the long_shock for EVE
            # as it drives the duration-sensitive PV change
            eve_shocked = self.eve_engine.run(shock_bps=scen.long_shock)
            delta_eve   = eve_shocked.eve - eve_baseline
            eve_pct_t1  = (
                abs(delta_eve) / self.reg_config.tier1_capital_€M * 100
                if self.reg_config.tier1_capital_€M
                else np.nan
            )

            # --- Outlier flags ---
            flags = self._check_outlier(
                scen.name, delta_nii, delta_eve,
                nii_baseline, self.reg_config,
            )
            all_flags.extend(flags)

            # --- Store ---
            nii_scenarios[scen.name] = round(delta_nii, 4)
            eve_scenarios[scen.name] = round(delta_eve, 4)

            rows.append({
                "scenario"       : scen.name,
                "short_shock_bps": scen.short_shock,
                "long_shock_bps" : scen.long_shock,
                "nii_base_€M"    : round(nii_baseline, 2),
                "nii_shocked_€M" : round(nii_shocked.annual_nii, 2),
                "delta_nii_€M"   : round(delta_nii, 4),
                "delta_nii_pct"  : round(nii_pct, 2),
                "eve_base_€M"    : round(eve_baseline, 2),
                "eve_shocked_€M" : round(eve_shocked.eve, 2),
                "delta_eve_€M"   : round(delta_eve, 4),
                "delta_eve_pct_t1": round(eve_pct_t1, 2),
                "nii_flag"       : any(f["metric"] == "ΔNII / NII baseline" for f in flags),
                "eve_flag"       : any(f["metric"] == "ΔEVE / Tier1" for f in flags),
            })

        summary_df = pd.DataFrame(rows)

        # Worst-case scenarios
        worst_nii = summary_df.loc[
            summary_df["delta_nii_€M"].idxmin(), "scenario"
        ]
        worst_eve = summary_df.loc[
            summary_df["delta_eve_€M"].idxmin(), "scenario"
        ]

        self._report = IRRBBReport(
            summary_df   = summary_df,
            nii_baseline = round(nii_baseline, 4),
            eve_baseline = round(eve_baseline, 4),
            nii_scenarios= nii_scenarios,
            eve_scenarios= eve_scenarios,
            worst_nii    = worst_nii,
            worst_eve    = worst_eve,
            outlier_flags= all_flags,
            reg_config   = self.reg_config,
        )
        return self._report

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def print_regulatory_report(self) -> None:
        """
        Print the full IRRBB regulatory report to stdout.

        Covers:
            - Balance sheet summary
            - NII and EVE per scenario
            - Outlier flag summary
            - Worst-case identification
        """
        report = self.run_all()
        rc     = self.reg_config
        sep    = "=" * 72

        print(f"\n{sep}")
        print(f"  IRRBB REGULATORY REPORT — {self.bs.name}")
        print(f"  BCBS 368 / EBA IRRBB Guidelines")
        print(sep)
        print(f"  Tier 1 capital         : €{rc.tier1_capital_€M:>10,.0f}M")
        print(f"  EVE outlier threshold  : {rc.eve_threshold_pct:.0f}% of Tier 1")
        print(f"  NII outlier threshold  : {rc.nii_threshold_pct:.0f}% of NII baseline")
        print(f"  NII baseline           : €{report.nii_baseline:>10,.2f}M")
        print(f"  EVE baseline           : €{report.eve_baseline:>10,.2f}M")
        print(sep)

        # Summary table
        display_cols = [
            "scenario", "short_shock_bps", "long_shock_bps",
            "delta_nii_€M", "delta_nii_pct",
            "delta_eve_€M", "delta_eve_pct_t1",
            "nii_flag", "eve_flag",
        ]
        print("\n  Scenario Results:")
        print(report.summary_df[display_cols].to_string(index=False))

        # Worst case
        print(f"\n  Worst-case NII scenario : {report.worst_nii}")
        print(f"  Worst-case EVE scenario : {report.worst_eve}")

        # Outlier flags
        if report.outlier_flags:
            print(f"\n  ⚠️  OUTLIER FLAGS ({len(report.outlier_flags)} breach(es)):")
            for flag in report.outlier_flags:
                print(
                    f"    [{flag['scenario']}] "
                    f"{flag['metric']} = {flag['value_pct']:.1f}% "
                    f"(threshold: {flag['threshold']:.0f}%)"
                )
        else:
            print(f"\n  ✅  No outlier flags — all scenarios within regulatory thresholds.")

        print(f"\n{sep}\n")

    def to_excel(self, path: str) -> None:
        """
        Export the IRRBB report to an Excel file with one sheet per topic.

        Sheets:
            Summary       : full scenario × metric table
            NII_Detail    : monthly NII for each scenario
            EVE_Positions : per-position EVE contribution (base scenario)
            Outlier_Flags : regulatory breach details

        Args:
            path : output file path (e.g. 'output/irrbb_report.xlsx')
        """
        report = self.run_all()

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            # Summary sheet
            report.summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # NII detail — monthly breakdown per scenario
            nii_rows = []
            for scen in self.scenarios:
                result = self.nii_engine.run(rate_shock_bps=scen.parallel_shock)
                monthly = result.monthly_df.copy()
                monthly.insert(0, "scenario", scen.name)
                nii_rows.append(monthly)
            pd.concat(nii_rows, ignore_index=True).to_excel(
                writer, sheet_name="NII_Detail", index=False
            )

            # EVE positions — base scenario only
            base_eve = self.eve_engine.run(shock_bps=0.0)
            base_eve.position_df.to_excel(
                writer, sheet_name="EVE_Positions", index=False
            )

            # Outlier flags
            if report.outlier_flags:
                pd.DataFrame(report.outlier_flags).to_excel(
                    writer, sheet_name="Outlier_Flags", index=False
                )

        print(f"IRRBB report exported: {path}")


# ---------------------------------------------------------------------------
# QUICK SANITY CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from balance_sheet import make_synthetic_balance_sheet

    bs     = make_synthetic_balance_sheet()
    curve  = FlatCurve(rate=0.03)
    config = RegConfig(tier1_capital_€M=800.0)

    irrbb  = IRRBBScenarios(bs, curve, reg_config=config)
    report = irrbb.run_all()

    irrbb.print_regulatory_report()

    print("NII scenarios dict (for plot_nii_scenarios):")
    print(report.nii_scenarios)

    print("\nEVE scenarios dict (for plot_eve_sensitivity):")
    print(report.eve_scenarios)