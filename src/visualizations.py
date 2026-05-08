"""
Plotly chart factory. Each function returns a fully configured Figure
that the Streamlit app and the PDF generator can consume.

Dark, high-contrast theme tuned for the Streamlit dark canvas — every
chart shares the same palette, typography and grid treatment so the
report reads like a single product, not a stack of notebooks.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.integrated_valuation import ValuationResult
from src.sensitivity import TornadoBar


# --- Palette ---------------------------------------------------------------
# Tuned to match the Streamlit dark theme set in .streamlit/config.toml.
_BG          = "#0b0f17"                # explicit page background
_PLOT_BG     = "#0b0f17"                # plot area background
_PANEL       = "#11151c"                # subtle panel tint when needed
_GRID        = "rgba(255,255,255,0.08)"
_AXIS        = "rgba(255,255,255,0.35)"
_TEXT        = "#e6edf3"
_MUTED       = "#9da7b3"

_PRIMARY     = "#4c8bf5"   # electric blue
_PRIMARY_DK  = "#2563eb"
_ACCENT      = "#22d3ee"   # cyan
_BUY         = "#10b981"   # emerald
_SELL        = "#ef4444"   # red
_HOLD        = "#f59e0b"   # amber
_GREY        = "#64748b"   # slate

_FONT = dict(family="Inter, -apple-system, Segoe UI, sans-serif",
             color=_TEXT, size=13)


def _apply_dark_theme(fig: go.Figure, *, height: int = 460,
                      title: str | None = None) -> go.Figure:
    """Apply the shared dark-mode styling to any figure."""
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_PLOT_BG,
        font=_FONT,
        height=height,
        margin=dict(l=60, r=40, t=70, b=60),
        title=dict(
            text=title or fig.layout.title.text or "",
            font=dict(family=_FONT["family"], color=_TEXT, size=18),
            x=0.01, xanchor="left", y=0.96,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=_TEXT, size=12),
            bordercolor=_GRID, borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="#1f2937", bordercolor=_PRIMARY,
            font=dict(color=_TEXT, family=_FONT["family"]),
        ),
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        linecolor=_AXIS, tickcolor=_AXIS,
        tickfont=dict(color=_TEXT, size=12),
        title_font=dict(color=_MUTED, size=12),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=_GRID, zeroline=False,
        linecolor=_AXIS, tickcolor=_AXIS,
        tickfont=dict(color=_TEXT, size=12),
        title_font=dict(color=_MUTED, size=12),
    )
    return fig


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
    rec_color = (_BUY if result.recommendation == "BUY" else
                 _SELL if result.recommendation == "SELL" else _HOLD)
    colors = [_GREY, _PRIMARY_DK, _PRIMARY, rec_color]

    # Glow line on top of every bar
    line_colors = ["#94a3b8", "#60a5fa", "#7dd3fc", rec_color]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values, marker=dict(
            color=colors,
            line=dict(color=line_colors, width=0),
        ),
        text=[f"₹{v:,.0f}" if v else "—" for v in values],
        textposition="outside",
        textfont=dict(color=_TEXT, size=13, family=_FONT["family"]),
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>",
    ))

    # Headroom so the P90 annotation doesn't get clipped at the top
    finite_vals = [v for v in values if v]
    mc = result.monte_carlo
    top_candidates = finite_vals + ([mc.p90] if np.isfinite(mc.p90) else [])
    y_max = max(top_candidates) * 1.18 if top_candidates else 1

    # Monte Carlo P10–P90 band overlay on Blended
    if np.isfinite(mc.p10):
        fig.add_shape(
            type="line", x0=3, x1=3,
            y0=mc.p10, y1=mc.p90,
            line=dict(color=_TEXT, width=2),
        )
        # Caps at top and bottom of the whisker
        for y, dy in [(mc.p90, 0), (mc.p10, 0)]:
            fig.add_shape(
                type="line", x0=2.92, x1=3.08, y0=y, y1=y,
                line=dict(color=_TEXT, width=2),
            )
        fig.add_annotation(
            x=3, y=mc.p90, text=f"P90  ₹{mc.p90:,.0f}",
            showarrow=False, yshift=14,
            font=dict(size=11, color=_MUTED, family=_FONT["family"]),
        )
        fig.add_annotation(
            x=3, y=mc.p10, text=f"P10  ₹{mc.p10:,.0f}",
            showarrow=False, yshift=-14,
            font=dict(size=11, color=_MUTED, family=_FONT["family"]),
        )

    fig.update_layout(
        title=f"{result.target.name} — Valuation Snapshot",
        yaxis_title="₹ per share",
        showlegend=False,
        bargap=0.45,
    )
    fig.update_yaxes(range=[0, y_max])
    return _apply_dark_theme(fig, height=480)


# ---------------------------------------------------------------------------
def tornado_chart(bars: List[TornadoBar]) -> go.Figure:
    """Classic horizontal tornado, sorted by swing magnitude."""
    bars = sorted(bars, key=lambda b: b.swing)
    fig = go.Figure()
    for i, b in enumerate(bars):
        show_legend = (i == 0)
        fig.add_trace(go.Bar(
            y=[b.lever], x=[b.high_value - b.base_value],
            base=b.base_value, orientation="h",
            marker=dict(color=_BUY, line=dict(color="#34d399", width=0)),
            name="Upside", showlegend=show_legend,
            hovertemplate=f"<b>{b.lever}</b><br>High: ₹{b.high_value:,.2f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            y=[b.lever], x=[b.low_value - b.base_value],
            base=b.base_value, orientation="h",
            marker=dict(color=_SELL, line=dict(color="#f87171", width=0)),
            name="Downside", showlegend=show_legend,
            hovertemplate=f"<b>{b.lever}</b><br>Low: ₹{b.low_value:,.2f}<extra></extra>",
        ))
    fig.add_vline(x=bars[-1].base_value if bars else 0,
                  line=dict(color=_TEXT, width=1, dash="dot"))
    fig.update_layout(
        title="Tornado — DDM Sensitivity to Key Inputs",
        xaxis_title="Implied Intrinsic Value (₹)",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return _apply_dark_theme(fig, height=440)


# ---------------------------------------------------------------------------
def monte_carlo_histogram(result: ValuationResult) -> go.Figure:
    samples = result.monte_carlo.samples
    if samples.size == 0:
        fig = go.Figure()
        fig.update_layout(title="Monte Carlo — no samples")
        return _apply_dark_theme(fig, height=460)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=samples, nbinsx=60,
        marker=dict(color=_PRIMARY, line=dict(color=_ACCENT, width=0.5)),
        opacity=0.85, name="Samples",
        hovertemplate="₹%{x:,.0f}<br>%{y} paths<extra></extra>",
    ))
    for q, label, color, ypos in [
        (result.monte_carlo.p10, "P10", _SELL, 0.95),
        (result.monte_carlo.p50, "P50", _TEXT, 1.02),
        (result.monte_carlo.p90, "P90", _BUY, 0.95),
    ]:
        fig.add_vline(
            x=q, line=dict(color=color, dash="dash", width=2),
        )
        fig.add_annotation(
            x=q, y=ypos, yref="paper",
            text=f"<b>{label}</b>  ₹{q:,.0f}",
            showarrow=False,
            font=dict(color=color, size=11, family=_FONT["family"]),
            bgcolor="rgba(11,15,23,0.85)",
            borderpad=3,
        )
    fig.add_vline(
        x=result.target.price,
        line=dict(color=_HOLD, width=3),
    )
    fig.add_annotation(
        x=result.target.price, y=0.88, yref="paper",
        text=f"<b>CMP</b>  ₹{result.target.price:,.0f}",
        showarrow=False,
        font=dict(color=_HOLD, size=12, family=_FONT["family"]),
        bgcolor="rgba(11,15,23,0.85)",
        borderpad=3,
    )

    fig.update_layout(
        title="Monte Carlo — Distribution of Blended Intrinsic Value (10,000 paths)",
        xaxis_title="Intrinsic Value per Share (₹)",
        yaxis_title="Frequency",
        bargap=0.02, showlegend=False,
    )
    return _apply_dark_theme(fig, height=480)


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

    # Alternating row stripes for readability against the dark canvas.
    n_rows = len(rows)
    stripe = ["#11151c", "#161c26"]
    fill_cols = [[stripe[i % 2] for i in range(n_rows)] for _ in range(4)]

    fig = go.Figure(data=[go.Table(
        columnwidth=[1.1, 1.4, 1.3, 0.9],
        header=dict(
            values=[f"<b>{h}</b>" for h in
                    ["Multiple", "Peer Aggregate", "Implied Price", "Weight"]],
            fill_color=_PRIMARY_DK,
            font=dict(color="white", size=13, family=_FONT["family"]),
            align="center", height=36,
            line=dict(color=_PRIMARY_DK, width=0),
        ),
        cells=dict(
            values=list(zip(*rows)) if rows else [[], [], [], []],
            align=["left", "center", "center", "center"],
            fill_color=fill_cols,
            font=dict(color=_TEXT, size=12, family=_FONT["family"]),
            height=32,
            line=dict(color=_GRID, width=0.5),
        ),
    )])
    fig.update_layout(
        title="Peer Multiples → Implied Prices",
        height=max(320, 100 + 32 * n_rows),
    )
    return _apply_dark_theme(fig, height=max(320, 100 + 32 * n_rows))


# ---------------------------------------------------------------------------
def price_history_chart(result: ValuationResult) -> go.Figure:
    h = result.target.price_history
    if h.empty:
        fig = go.Figure()
        fig.update_layout(title="Price history unavailable")
        return _apply_dark_theme(fig, height=420)

    fig = go.Figure()
    # Soft area fill under the line for depth.
    fig.add_trace(go.Scatter(
        x=h.index, y=h.values, mode="lines",
        line=dict(color=_PRIMARY, width=2.2),
        fill="tozeroy", fillcolor="rgba(76, 139, 245, 0.12)",
        name="Adj Close",
        hovertemplate="%{x|%b %Y}<br>₹%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(
        y=result.blended_value,
        line=dict(color=_BUY, dash="dash", width=2),
        annotation_text=f"Blended IV ₹{result.blended_value:,.0f}",
        annotation_position="top right",
        annotation_font=dict(color=_BUY, size=12, family=_FONT["family"]),
    )
    fig.update_layout(
        title=f"{result.target.name} — 5Y Price History vs Intrinsic Value",
        yaxis_title="₹",
        showlegend=False,
    )
    # Y-axis must start above 0 so the area fill doesn't squash the line
    y_min = float(h.min()) * 0.85
    y_max = max(float(h.max()), result.blended_value) * 1.08
    fig.update_yaxes(range=[y_min, y_max])
    return _apply_dark_theme(fig, height=440)
