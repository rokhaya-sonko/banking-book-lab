"""
plot_utils.py
=============
Standardized visualization functions for the Banking Book Lab.
Built on Plotly for interactivity in notebooks.

Modules covered:
    - Module 01 : IRRBB (repricing gap, NII, EVE)
    - Module 02 : Liquidity (cash flow ladder, LCR/NSFR)
    - Module 03 : Yield Curve (curves, Monte Carlo fan chart)
    - Module 04 : Prepayment (CPR, OAD)
    - Module 05 : FTP (NIM waterfall, decomposition)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

# ---------------------------------------------------------------------------
# COMMON DESIGN SYSTEM
# ---------------------------------------------------------------------------

COLORS = {
    "primary":    "#1f77b4",
    "secondary":  "#ff7f0e",
    "positive":   "#2ca02c",
    "negative":   "#d62728",
    "neutral":    "#7f7f7f",
    "accent":     "#9467bd",
    "background": "#ffffff",
    "grid":       "#e5e5e5",
}

PALETTE = [
    COLORS["primary"], COLORS["secondary"], COLORS["positive"],
    COLORS["negative"], COLORS["accent"], COLORS["neutral"],
]

LAYOUT_DEFAULTS = dict(
    font=dict(family="Arial, sans-serif", size=12, color="#333333"),
    plot_bgcolor=COLORS["background"],
    paper_bgcolor=COLORS["background"],
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=30, t=60, b=60),
    xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False),
    yaxis=dict(
        showgrid=True, gridcolor=COLORS["grid"],
        zeroline=True, zerolinecolor=COLORS["neutral"], zerolinewidth=1,
    ),
)


def _apply_defaults(fig: go.Figure, title: str, **kwargs) -> go.Figure:
    """Merge LAYOUT_DEFAULTS with a centered title and any extra overrides."""
    layout = {**LAYOUT_DEFAULTS, "title": dict(text=title, x=0.5, xanchor="center")}
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# MODULE 03 — YIELD CURVE
# ---------------------------------------------------------------------------

def plot_yield_curve(tenors, rates, label="Curve", title="Yield Curve", rate_format="percent"):
    """Plot a single yield curve (zero, par, or forward)."""
    y = [r * 100 for r in rates] if rate_format == "percent" else rates
    ylabel = "Rate (%)" if rate_format == "percent" else "Rate (bps)"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tenors, y=y, mode="lines+markers", name=label,
        line=dict(color=COLORS["primary"], width=2), marker=dict(size=6),
        hovertemplate=f"Maturity: %{{x}}y<br>{ylabel}: %{{y:.4f}}<extra></extra>",
    ))
    _apply_defaults(fig, title, xaxis_title="Maturity (years)", yaxis_title=ylabel)
    return fig


def plot_multiple_curves(curves, title="Yield Curve Comparison"):
    """Overlay multiple yield curves on a single chart."""
    fig = go.Figure()
    for i, (label, (tenors, rates)) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=tenors, y=[r * 100 for r in rates],
            mode="lines+markers", name=label,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2), marker=dict(size=5),
        ))
    _apply_defaults(fig, title, xaxis_title="Maturity (years)", yaxis_title="Rate (%)")
    return fig


def plot_fan_chart(tenors, base_rates, simulated_rates, title="Fan Chart — Monte Carlo", percentiles=(5, 25, 75, 95)):
    """Fan chart showing the distribution of Monte Carlo rate simulations."""
    fig = go.Figure()
    p = [np.percentile(simulated_rates, pct, axis=0) * 100 for pct in percentiles]
    fig.add_trace(go.Scatter(
        x=tenors + tenors[::-1], y=list(p[3]) + list(p[0][::-1]),
        fill="toself", fillcolor="rgba(31,119,180,0.10)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"{percentiles[0]}%–{percentiles[3]}%", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=tenors + tenors[::-1], y=list(p[2]) + list(p[1][::-1]),
        fill="toself", fillcolor="rgba(31,119,180,0.25)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"{percentiles[1]}%–{percentiles[2]}%", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=tenors, y=[r * 100 for r in base_rates],
        mode="lines", name="Base scenario",
        line=dict(color=COLORS["primary"], width=2.5),
    ))
    _apply_defaults(fig, title, xaxis_title="Maturity (years)", yaxis_title="Rate (%)")
    return fig


# ---------------------------------------------------------------------------
# MODULE 01 — IRRBB
# ---------------------------------------------------------------------------

def plot_repricing_gap(buckets, assets, liabilities, gap=None, title="Repricing Gap"):
    """Two-panel bar chart: assets vs liabilities (top) and net repricing gap (bottom)."""
    if gap is None:
        gap = [a - l for a, l in zip(assets, liabilities)]
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Assets vs Liabilities", "Repricing Gap"),
        vertical_spacing=0.15, row_heights=[0.55, 0.45],
    )
    fig.add_trace(go.Bar(x=buckets, y=assets, name="Assets",
                         marker_color=COLORS["primary"]), row=1, col=1)
    fig.add_trace(go.Bar(x=buckets, y=[-l for l in liabilities], name="Liabilities",
                         marker_color=COLORS["secondary"]), row=1, col=1)
    gap_colors = [COLORS["positive"] if g >= 0 else COLORS["negative"] for g in gap]
    fig.add_trace(go.Bar(x=buckets, y=gap, name="Gap",
                         marker_color=gap_colors), row=2, col=1)
    fig.update_layout(
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k not in ("xaxis", "yaxis")},
        title=dict(text=title, x=0.5, xanchor="center"), barmode="relative",
    )
    fig.update_yaxes(title_text="Amount (€bn)", gridcolor=COLORS["grid"])
    return fig


def plot_nii_scenarios(scenarios, baseline, title="NII Sensitivity by Rate Scenario"):
    """Horizontal bar chart of NII impact under IRRBB stress scenarios."""
    labels, deltas = list(scenarios.keys()), list(scenarios.values())
    colors = [COLORS["positive"] if d >= 0 else COLORS["negative"] for d in deltas]
    fig = go.Figure(go.Bar(
        x=deltas, y=labels, orientation="h", marker_color=colors,
        text=[f"{d:+.1f}" for d in deltas], textposition="outside",
        hovertemplate="%{y}: %{x:+.1f} €bn<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["neutral"])
    _apply_defaults(fig, f"{title}<br><sup>NII baseline: {baseline:.1f} €bn</sup>",
                    xaxis_title="NII Change (€bn)", height=400)
    return fig


def plot_eve_sensitivity(scenarios, title="EVE Sensitivity — BCBS 368 Scenarios"):
    """Bar chart of EVE change under the 6 regulatory rate scenarios."""
    labels, values = list(scenarios.keys()), list(scenarios.values())
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in values]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:+.1f}" for v in values], textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"])
    _apply_defaults(fig, title, xaxis_title="Scenario", yaxis_title="ΔEVE (€bn)")
    return fig


# ---------------------------------------------------------------------------
# MODULE 02 — LIQUIDITY RISK
# ---------------------------------------------------------------------------

def plot_cashflow_ladder(buckets, inflows, outflows, title="Cash Flow Ladder"):
    """Stacked bar chart of cash inflows/outflows with net position overlay."""
    net = [i - o for i, o in zip(inflows, outflows)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=buckets, y=inflows, name="Inflows", marker_color=COLORS["positive"]))
    fig.add_trace(go.Bar(x=buckets, y=[-o for o in outflows], name="Outflows",
                         marker_color=COLORS["negative"]))
    fig.add_trace(go.Scatter(
        x=buckets, y=net, mode="lines+markers", name="Net position",
        line=dict(color=COLORS["accent"], width=2, dash="dot"), marker=dict(size=7),
    ))
    fig.update_layout(
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k not in ("xaxis", "yaxis")},
        title=dict(text=title, x=0.5, xanchor="center"), barmode="relative",
        yaxis=dict(showgrid=True, gridcolor=COLORS["grid"], title="Amount (€bn)"),
    )
    return fig


def plot_lcr_nsfr(dates, lcr_values, nsfr_values, title="Regulatory Liquidity Ratios"):
    """Time series of LCR and NSFR with the 100% regulatory floor highlighted."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=lcr_values, mode="lines+markers", name="LCR",
                             line=dict(color=COLORS["primary"], width=2)))
    fig.add_trace(go.Scatter(x=dates, y=nsfr_values, mode="lines+markers", name="NSFR",
                             line=dict(color=COLORS["secondary"], width=2)))
    fig.add_hline(y=100, line_dash="dash", line_color=COLORS["negative"],
                  annotation_text="Regulatory minimum: 100%", annotation_position="bottom right")
    _apply_defaults(fig, title, xaxis_title="Date", yaxis_title="Ratio (%)")
    return fig


# ---------------------------------------------------------------------------
# MODULE 04 — PREPAYMENT
# ---------------------------------------------------------------------------

def plot_cpr_curve(months, cpr_model, cpr_psa=None, title="Conditional Prepayment Rate (CPR)"):
    """CPR curve from the logistic model, optionally compared to the PSA benchmark."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=[r * 100 for r in cpr_model], mode="lines",
                             name="Logistic model", line=dict(color=COLORS["primary"], width=2)))
    if cpr_psa:
        fig.add_trace(go.Scatter(x=months, y=[r * 100 for r in cpr_psa], mode="lines",
                                 name="PSA benchmark",
                                 line=dict(color=COLORS["secondary"], width=2, dash="dash")))
    _apply_defaults(fig, title, xaxis_title="Month", yaxis_title="CPR (%)")
    return fig


def plot_oad_profile(rate_shocks, oad_values, duration_no_prepay=None, title="Option-Adjusted Duration"):
    """OAD profile across a range of rate shocks."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rate_shocks, y=oad_values, mode="lines+markers", name="OAD",
        line=dict(color=COLORS["primary"], width=2), marker=dict(size=7),
        hovertemplate="Shock: %{x:+d}bps<br>OAD: %{y:.2f}y<extra></extra>",
    ))
    if duration_no_prepay is not None:
        fig.add_hline(y=duration_no_prepay, line_dash="dash", line_color=COLORS["neutral"],
                      annotation_text="Duration without prepayment")
    _apply_defaults(fig, title, xaxis_title="Rate shock (bps)", yaxis_title="Duration (years)")
    return fig


# ---------------------------------------------------------------------------
# MODULE 05 — FTP / NIM WATERFALL
# ---------------------------------------------------------------------------

def plot_nim_waterfall(labels, values, title="Net Interest Margin (NIM) Decomposition"):
    """Waterfall chart decomposing NIM into its components (in bps)."""
    measure = ["relative"] * (len(labels) - 1) + ["total"]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measure, x=labels, y=values,
        connector=dict(line=dict(color=COLORS["neutral"], width=1, dash="dot")),
        increasing=dict(marker_color=COLORS["positive"]),
        decreasing=dict(marker_color=COLORS["negative"]),
        totals=dict(marker_color=COLORS["primary"]),
        text=[f"{v:+d}bps" for v in values], textposition="outside",
        hovertemplate="%{x}: %{y:+d}bps<extra></extra>",
    ))
    _apply_defaults(fig, title, yaxis_title="Basis points (bps)", showlegend=False)
    return fig


def plot_ftp_curve(tenors, ftp_rates, ois_rates=None, title="FTP Curve"):
    """FTP curve with optional OIS base and liquidity premium shading."""
    fig = go.Figure()
    if ois_rates:
        fig.add_trace(go.Scatter(x=tenors, y=[r * 100 for r in ois_rates], mode="lines",
                                 name="OIS (risk-free)", line=dict(color=COLORS["primary"], width=2)))
        # Shaded area between OIS and FTP represents the liquidity premium
        fig.add_trace(go.Scatter(
            x=tenors + tenors[::-1],
            y=[r * 100 for r in ftp_rates] + [r * 100 for r in ois_rates[::-1]],
            fill="toself", fillcolor="rgba(255,127,14,0.20)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Liquidity premium", hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(x=tenors, y=[r * 100 for r in ftp_rates], mode="lines+markers",
                             name="FTP curve", line=dict(color=COLORS["secondary"], width=2.5),
                             marker=dict(size=6)))
    _apply_defaults(fig, title, xaxis_title="Maturity (years)", yaxis_title="Rate (%)")
    return fig


# ---------------------------------------------------------------------------
# GENERIC UTILITIES
# ---------------------------------------------------------------------------

def plot_time_series(df, x_col, y_cols, title="Time Series", yaxis_title="Value"):
    """Generic multi-series time series chart."""
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[col], mode="lines", name=col,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        ))
    _apply_defaults(fig, title, xaxis_title=x_col, yaxis_title=yaxis_title)
    return fig


def plot_heatmap(data, x_labels, y_labels, title="Heatmap", colorscale="RdYlGn", zmid=0):
    """Generic heatmap — useful for cross-bucket sensitivities or correlation matrices."""
    fig = go.Figure(go.Heatmap(
        z=data, x=x_labels, y=y_labels,
        colorscale=colorscale, zmid=zmid,
        hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z:.2f}<extra></extra>",
        texttemplate="%{z:.1f}",
    ))
    _apply_defaults(fig, title)
    return fig


def save_fig(fig, path, scale=2.0):
    """
    Export a Plotly figure to high-resolution PNG.
    Falls back to HTML if kaleido is not installed.
    """
    try:
        fig.write_image(path, scale=scale)
        print(f"Figure saved: {path}")
    except Exception as e:
        html_path = path.replace(".png", ".html")
        fig.write_html(html_path)
        print(f"PNG export failed ({e}). Saved as HTML: {html_path}")
        print("To enable PNG export: pip install kaleido")
