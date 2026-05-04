"""
Plotly chart factory. Each function returns a fully configured Figure
that the Streamlit app and the PDF generator can consume.

Charts are deliberately minimal — equity research notes don't need to
look like Bloomberg Terminal screenshots.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.integrated_valuation import ValuationResult
from src.sensitivity import TornadoBar


_PRIMARY = "#1f4e79"
_BUY = "#2e7d32"
_SELL = "#c62828"
_HOLD = "#f9a825"
_GREY = "#9e9e9e"


# ---------------------------------------------------------------------------
def valuation_bar_chart(result: ValuationResult) -> go.Figure:
    """Side-by-side bars: current price, DDM, Relative, Blended, MC band."""
    labels = ["Current Price", "DDM", "Relative", "Blended"]
    values = [
        result.target.price,
        result.ddm.value_per_share if result.ddm.valid else None,
        result.relative.weighted_value,
        result.blended_value,
    ]
    colors = [_GREY, _PRIMARY, _PRIMARY, _BUY if result.recommendation == "BUY" else
              (_SELL if result.recommendation == "SELL" else _HOLD)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"₹{v:,.0f}" if v else "—" for v in values],
        textposition="outside",
    ))

    # Monte Carlo P10–P90 as a shaded band overlay on Blended
    mc = result.monte_carlo
    if np.isfinite(mc.p10):
        fig.add_shape(
            type="line", x0=3, x1=3,
            y0=mc.p10, y1=mc.p90,
            line=dict(color="black", width=2),
        )
        fig.add_annotation(
            x=3, y=mc.p90, text=f"P90 ₹{mc.p90:,.0f}",
            showarrow=False, yshift=10, font=dict(size=10),
        )
        fig.add_annotation(
            x=3, y=mc.p10, text=f"P10 ₹{mc.p10:,.0f}",
            showarrow=False, yshift=-15, font=dict(size=10),
        )

    fig.update_layout(
        title=f"{result.target.name} — Valuation Snapshot",
        yaxis_title="₹ per share",
        plot_bgcolor="white",
        showlegend=False,
        height=460,
    )
    return fig


# ---------------------------------------------------------------------------
def tornado_chart(bars: List[TornadoBar]) -> go.Figure:
    """Classic horizontal tornado, sorted by swing magnitude."""
    bars = sorted(bars, key=lambda b: b.swing)
    fig = go.Figure()
    for b in bars:
        fig.add_trace(go.Bar(
            y=[b.lever], x=[b.high_value - b.base_value],
            base=b.base_value, orientation="h",
            marker_color=_BUY, name="Up",
            hovertemplate=f"High: ₹{b.high_value:,.2f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            y=[b.lever], x=[b.low_value - b.base_value],
            base=b.base_value, orientation="h",
            marker_color=_SELL, name="Down",
            hovertemplate=f"Low: ₹{b.low_value:,.2f}<extra></extra>",
        ))
    fig.add_vline(x=bars[-1].base_value if bars else 0,
                  line=dict(color="black", width=1, dash="dot"))
    fig.update_layout(
        title="Tornado — DDM Sensitivity to Key Inputs",
        xaxis_title="Implied Intrinsic Value (₹)",
        showlegend=False, barmode="overlay",
        plot_bgcolor="white", height=420,
    )
    return fig


# ---------------------------------------------------------------------------
def monte_carlo_histogram(result: ValuationResult) -> go.Figure:
    samples = result.monte_carlo.samples
    if samples.size == 0:
        return go.Figure().update_layout(title="Monte Carlo — no samples")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=samples, nbinsx=60, marker_color=_PRIMARY, opacity=0.85,
    ))
    for q, label, color in [
        (result.monte_carlo.p10, "P10", _SELL),
        (result.monte_carlo.p50, "P50", "black"),
        (result.monte_carlo.p90, "P90", _BUY),
    ]:
        fig.add_vline(x=q, line=dict(color=color, dash="dash"),
                      annotation_text=f"{label} ₹{q:,.0f}",
                      annotation_position="top right")
    fig.add_vline(x=result.target.price, line=dict(color="orange", width=3),
                  annotation_text=f"CMP ₹{result.target.price:,.0f}",
                  annotation_position="top left")

    fig.update_layout(
        title="Monte Carlo — Distribution of Blended Intrinsic Value (10,000 paths)",
        xaxis_title="Intrinsic Value per Share (₹)",
        yaxis_title="Frequency",
        plot_bgcolor="white", height=460, bargap=0.02,
    )
    return fig


# ---------------------------------------------------------------------------
def peer_multiples_table(result: ValuationResult) -> go.Figure:
    rows = []
    for k, mr in result.relative.multiples.items():
        rows.append([
            k,
            f"{mr.aggregated_multiple:,.2f}" if np.isfinite(mr.aggregated_multiple) else "—",
            f"₹{mr.implied_price:,.2f}" if mr.valid else "—",
            f"{result.relative.weights.get(k, 0):.0%}",
        ])
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Multiple", "Peer Aggregate", "Implied Price", "Weight"],
                    fill_color=_PRIMARY, font=dict(color="white", size=12)),
        cells=dict(values=list(zip(*rows)), align="center"),
    )])
    fig.update_layout(title="Peer Multiples → Implied Prices", height=320)
    return fig


# ---------------------------------------------------------------------------
def price_history_chart(result: ValuationResult) -> go.Figure:
    h = result.target.price_history
    if h.empty:
        return go.Figure().update_layout(title="Price history unavailable")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index, y=h.values, mode="lines",
                             line=dict(color=_PRIMARY, width=2),
                             name="Adj Close"))
    fig.add_hline(y=result.blended_value, line=dict(color=_BUY, dash="dash"),
                  annotation_text=f"Blended IV ₹{result.blended_value:,.0f}",
                  annotation_position="top right")
    fig.update_layout(
        title=f"{result.target.name} — 5Y Price History vs Intrinsic Value",
        yaxis_title="₹", plot_bgcolor="white", height=420,
    )
    return fig
