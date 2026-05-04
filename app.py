"""
Streamlit dashboard — the interactive face of the valuation engine.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sibling imports work when run via `streamlit run app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from config import (
    DEFAULT_UNIVERSE, EQUITY_RISK_PREMIUM_IN, RISK_FREE_RATE_IN,
    PROJECT_TITLE, PROJECT_SUBTITLE, AUTHOR_NAME, SUPERVISOR_NAME, INSTITUTION,
)
from src.integrated_valuation import value_stock
from src.report_generator import generate_pdf
from src.visualizations import (
    valuation_bar_chart, tornado_chart, monte_carlo_histogram,
    peer_multiples_table, price_history_chart,
)


st.set_page_config(
    page_title=PROJECT_TITLE,
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### {PROJECT_TITLE}")
    st.caption(PROJECT_SUBTITLE)
    st.markdown("---")
    st.markdown(f"**Author:** {AUTHOR_NAME}")
    st.markdown(f"**Supervisor:** {SUPERVISOR_NAME}")
    st.markdown(f"**{INSTITUTION}**")
    st.markdown("---")

    universe = sorted(DEFAULT_UNIVERSE.keys())
    ticker = st.selectbox(
        "Select an Indian stock", universe,
        index=universe.index("TCS.NS") if "TCS.NS" in universe else 0,
    )
    custom = st.text_input("...or enter a custom NSE ticker (e.g. TCS.NS)")
    if custom:
        ticker = custom.strip().upper()

    st.markdown("#### Capital market inputs")
    rf = st.number_input("Risk-free rate (10Y G-Sec)", value=RISK_FREE_RATE_IN,
                         step=0.0025, format="%.4f")
    erp = st.number_input("Equity risk premium (India)", value=EQUITY_RISK_PREMIUM_IN,
                          step=0.0025, format="%.4f")
    g_term = st.number_input("Terminal growth (perpetuity)", value=0.045,
                             step=0.0025, format="%.4f")

    st.markdown("#### Run options")
    use_cache = st.checkbox("Use cached data if fresh", value=True)
    offline = st.checkbox("Offline mode (cache-only)", value=False)
    run_btn = st.button("🚀 Run Valuation", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(PROJECT_TITLE)
st.caption(PROJECT_SUBTITLE)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if run_btn:
    with st.spinner(f"Running full valuation pipeline for {ticker} ..."):
        try:
            result = value_stock(
                ticker,
                risk_free=rf,
                erp=erp,
                g_terminal=g_term,
                offline=offline,
                force_refresh=not use_cache,
                verbose=False,
            )
        except Exception as e:
            st.error(f"Failed to value {ticker}: {e}")
            st.stop()

    # ---- Top metrics row ----
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Current Price", f"₹{result.target.price:,.2f}")
    m2.metric("DDM Value",
              f"₹{result.ddm.value_per_share:,.2f}" if result.ddm.valid else "—",
              help=result.ddm.model)
    m3.metric("Relative Value", f"₹{result.relative.weighted_value:,.2f}")
    m4.metric("Blended IV", f"₹{result.blended_value:,.2f}")
    delta = f"{result.margin_of_safety:+.1%} MoS"
    m5.metric("Recommendation", result.recommendation, delta=delta,
              delta_color=("normal" if result.recommendation == "BUY"
                           else "inverse" if result.recommendation == "SELL"
                           else "off"))

    st.markdown("---")

    # ---- Tabbed body ----
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", "🧮 DDM detail", "🤝 Peers & Multiples",
        "🎲 Monte Carlo", "🌪️ Sensitivity", "📄 Report"
    ])

    with tab1:
        st.plotly_chart(valuation_bar_chart(result), use_container_width=True)
        st.plotly_chart(price_history_chart(result), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cost of equity**")
            st.code(result.coe.explain())
        with c2:
            st.markdown("**Quality scoring**")
            st.json({
                "Composite (0-100)": round(result.quality.composite, 1),
                "Piotroski F-Score": result.quality.piotroski_f,
                "Altman Z''": round(result.quality.altman_z, 2),
                "Dividend Quality": result.quality.dividend_quality,
            })

    with tab2:
        st.subheader(f"DDM Variant Used: {result.ddm.model}")
        if result.ddm.valid:
            st.write("**Inputs**")
            st.json({k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in result.ddm.inputs.items()})
            if result.ddm.note:
                st.warning(result.ddm.note)
        else:
            st.error(result.ddm.note)

        if not result.target.dividends_annual.empty:
            st.write("**Annual Dividends Paid (₹/share)**")
            df = result.target.dividends_annual.copy()
            df.index = df.index.year
            st.bar_chart(df)

    with tab3:
        st.subheader("Peer set")
        st.caption(f"Selection method: {result.peer_set.method}")
        if result.peer_set.debug_features is not None:
            st.dataframe(result.peer_set.debug_features.style.format("{:.3f}"))
        st.plotly_chart(peer_multiples_table(result), use_container_width=True)
        # Per-multiple breakdown
        for k, mr in result.relative.multiples.items():
            with st.expander(f"{k} — peer values"):
                st.dataframe(mr.peer_values.dropna().to_frame("multiple")
                             .style.format("{:.2f}"))

    with tab4:
        st.plotly_chart(monte_carlo_histogram(result), use_container_width=True)
        st.markdown(
            f"**Distribution stats** — mean ₹{result.monte_carlo.mean:,.0f}, "
            f"std ₹{result.monte_carlo.std:,.0f}, "
            f"P10/P50/P90: ₹{result.monte_carlo.p10:,.0f} / "
            f"₹{result.monte_carlo.p50:,.0f} / ₹{result.monte_carlo.p90:,.0f}"
        )

    with tab5:
        if result.tornado:
            st.plotly_chart(tornado_chart(result.tornado), use_container_width=True)
            df = pd.DataFrame([
                {"Lever": b.lever,
                 "Low": round(b.low_value, 2),
                 "High": round(b.high_value, 2),
                 "Swing (₹)": round(b.swing, 2)}
                for b in result.tornado
            ])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Tornado not available (DDM did not produce a valid value).")

    with tab6:
        st.write("Generate a polished PDF research note for this stock.")
        if st.button("Generate PDF report", type="primary"):
            with st.spinner("Composing report..."):
                path = generate_pdf(result)
            st.success(f"Report saved to {path}")
            with open(path, "rb") as f:
                st.download_button(
                    "⬇️ Download report",
                    f.read(),
                    file_name=path.name,
                    mime="application/pdf",
                )

else:
    st.info(
        "Pick a ticker from the sidebar and hit **Run Valuation** to begin.\n\n"
        "Each run computes DDM (auto-selected variant), peer-based relative "
        "valuation, Monte Carlo confidence bands, sensitivity tornado, and a "
        "BUY / HOLD / SELL recommendation."
    )
    st.markdown(
        "##### Methodology, in one paragraph\n"
        "We compute two independent intrinsic values — a Dividend Discount "
        "Model and a peer-multiples relative valuation — and blend them. The "
        "blend defaults to 50/50 per the project brief, but tilts toward "
        "whichever model is more credible based on the firm's dividend track "
        "record and earnings history. Confidence bands come from a 10,000-path "
        "Monte Carlo over cost of equity, growth, and peer-multiple noise."
    )
