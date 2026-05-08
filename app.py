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
    SECTOR_UNLEVERED_BETA,
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
# Custom theme — keeps the dashboard visually consistent with the Plotly
# charts in src/visualizations.py. Without this, Streamlit's default
# white widget chrome bleeds through the dark canvas.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      html, body, [class*="css"] {
        font-family: "Inter", -apple-system, "Segoe UI", Roboto, sans-serif;
        color: #e6edf3;
      }
      h1, h2, h3, h4 { letter-spacing: -0.01em; }
      h1 { font-weight: 700; }

      .stApp {
        background: radial-gradient(1200px 700px at 10% -10%,
                    rgba(76,139,245,0.10) 0%, rgba(11,15,23,0) 60%),
                    radial-gradient(900px 600px at 100% 0%,
                    rgba(34,211,238,0.07) 0%, rgba(11,15,23,0) 55%),
                    #0b0f17;
      }

      section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1320 0%, #0b0f17 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
      }
      section[data-testid="stSidebar"] .stMarkdown { color: #cbd5e1; }

      div[data-testid="stMetric"] {
        background: linear-gradient(135deg,
                    rgba(76,139,245,0.10) 0%,
                    rgba(34,211,238,0.04) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset,
                    0 8px 24px rgba(0,0,0,0.35);
      }
      div[data-testid="stMetricLabel"] p {
        color: #9da7b3 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }
      div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 600;
      }

      .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
      }
      .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #9da7b3;
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
      }
      .stTabs [aria-selected="true"] {
        background: rgba(76,139,245,0.12);
        color: #e6edf3 !important;
        border-bottom: 2px solid #4c8bf5 !important;
      }

      .stButton > button {
        background: linear-gradient(135deg, #4c8bf5 0%, #2563eb 100%);
        color: #fff;
        border: 0;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
        box-shadow: 0 6px 18px rgba(37,99,235,0.35);
        transition: transform 0.12s ease, box-shadow 0.2s ease;
      }
      .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(37,99,235,0.45);
      }

      [data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        overflow: hidden;
      }

      [data-testid="stCodeBlock"], pre, code {
        background: #0f141d !important;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        color: #e6edf3 !important;
      }

      div[data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
      }

      div[data-baseweb="select"] > div,
      .stTextInput > div > div,
      .stNumberInput > div > div {
        background: #0f141d !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 10px !important;
        color: #e6edf3 !important;
      }

      hr {
        border: 0 !important;
        height: 1px !important;
        background: linear-gradient(90deg,
                    rgba(255,255,255,0) 0%,
                    rgba(255,255,255,0.12) 50%,
                    rgba(255,255,255,0) 100%) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# Human-readable labels for the peer-set debug feature matrix.
_PEER_FEATURE_LABELS = {
    "role":              "Role",
    "log_market_cap":    "Log Market Cap",
    "roe":               "ROE",
    "debt_to_equity":    "Debt / Equity",
    "payout_ratio":      "Payout Ratio",
    "revenue_growth_5y": "Revenue Growth (5Y)",
    "ebitda_margin":     "EBITDA Margin",
}


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

    # Sector override — yfinance occasionally misclassifies a custom
    # ticker (e.g. SWIGGY → "Consumer Cyclical → Consumer Durables")
    # which produces nonsense peers. Let the user correct it without
    # editing config.
    sector_options = ["(auto-detect)"] + sorted(SECTOR_UNLEVERED_BETA.keys())
    auto_default = DEFAULT_UNIVERSE.get(ticker, "(auto-detect)")
    if auto_default not in sector_options:
        auto_default = "(auto-detect)"
    sector_choice = st.selectbox(
        "Sector (override if peers look wrong)",
        sector_options,
        index=sector_options.index(auto_default),
        help=(
            "Leave on auto-detect for tickers in the curated universe. "
            "For custom tickers where yfinance's industry tag is wrong, "
            "pick the right sector here so the peer pool is sensible."
        ),
    )
    sector_override = None if sector_choice == "(auto-detect)" else sector_choice

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
# Run button is a one-shot — it returns True only on the click that
# triggered it. We persist the result in session_state so subsequent
# reruns (caused by the PDF button, tab interactions, etc.) don't blow
# the dashboard away.
if run_btn:
    with st.spinner(f"Running full valuation pipeline for {ticker} ..."):
        try:
            result = value_stock(
                ticker,
                sector=sector_override,
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
        st.session_state["result"] = result
        st.session_state["result_ticker"] = ticker
        # New valuation invalidates any cached PDF from the previous run.
        st.session_state.pop("pdf_bytes", None)
        st.session_state.pop("pdf_name", None)

result = st.session_state.get("result")
if result is not None:

    # ---- Top metrics row ----
    import math as _math
    is_na = result.recommendation == "N/A"

    def _fmt_money(x):
        if x is None or not _math.isfinite(x):
            return "—"
        return f"₹{x:,.2f}"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Current Price", _fmt_money(result.target.price))
    m2.metric(
        "DDM Value",
        _fmt_money(result.ddm.value_per_share) if result.ddm.valid else "—",
        delta=None if result.ddm.valid else "Not applicable (non-payer)",
        delta_color="off",
        help=(
            result.ddm.note if not result.ddm.valid
            else f"Variant used: {result.ddm.model}"
        ),
    )
    m3.metric("Relative Value", _fmt_money(result.relative.weighted_value))

    # Show how the blend was actually weighted, not just the headline
    # 50/50 from the brief — for non-payers it collapses to 100% Relative
    # and the viewer needs to see that on the card.
    if is_na:
        blend_label = "—"
    elif result.ddm.valid:
        blend_label = (
            f"DDM {result.weight_ddm:.0%} · Rel {result.weight_relative:.0%}"
        )
    else:
        blend_label = "Relative-only (DDM not applicable)"
    m4.metric(
        "Blended IV",
        "—" if is_na else _fmt_money(result.blended_value),
        delta=None if blend_label == "—" else blend_label,
        delta_color="off",
    )

    delta = ("—" if is_na or not _math.isfinite(result.margin_of_safety)
             else f"{result.margin_of_safety:+.1%} MoS")
    m5.metric("Recommendation", result.recommendation,
              delta=None if delta == "—" else delta,
              delta_color=("normal" if result.recommendation == "BUY"
                           else "inverse" if result.recommendation == "SELL"
                           else "off"))
    if is_na:
        st.warning(
            "Insufficient data for a defensible recommendation. "
            f"Reason: {result.peer_set.method}. Try selecting a different "
            "sector from the **Sector** dropdown in the sidebar, or pick a "
            "stock with a deeper peer set."
        )
    elif not result.ddm.valid:
        st.info(
            f"**{result.target.name}** does not pay regular dividends, so "
            "the Dividend Discount Model is not applicable for this stock. "
            "The valuation falls back to the relative-valuation track "
            "(peer-multiples), which is the standard approach for "
            "non-dividend-paying companies."
        )

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
        if result.ddm.valid:
            st.subheader(f"DDM Variant Used: {result.ddm.model}")
            st.write("**Inputs**")
            st.json({k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in result.ddm.inputs.items()})
            if result.ddm.note:
                st.warning(result.ddm.note)
        else:
            st.subheader("DDM not applicable for this stock")
            st.info(
                f"**Reason:** {result.ddm.note}\n\n"
                "The Dividend Discount Model values a stock as the present "
                "value of its expected future dividends. When a company pays "
                "no dividend (or a token amount that doesn't reflect the "
                "earnings being reinvested), the model has no cash flow to "
                "discount — so we deliberately skip it rather than produce "
                "a misleading number. The valuation falls back to the "
                "relative-valuation (peer-multiples) track, which is the "
                "standard approach for non-payers."
            )

        if not result.target.dividends_annual.empty:
            st.write("**Annual Dividends Paid (₹/share)**")
            df = result.target.dividends_annual.copy()
            df.index = df.index.year
            st.bar_chart(df)

    with tab3:
        st.subheader("Peer set")
        st.caption(f"Selection method: {result.peer_set.method}")
        if result.peer_set.debug_features is not None:
            df_peers = (result.peer_set.debug_features
                        .rename(columns=_PEER_FEATURE_LABELS)
                        .rename_axis("Ticker")
                        .reset_index())
            numeric_cols = df_peers.select_dtypes(include="number").columns.tolist()
            pct_cols = [c for c in ("ROE", "Payout Ratio",
                                    "Revenue Growth (5Y)", "EBITDA Margin")
                        if c in df_peers.columns]
            col_cfg = {c: st.column_config.NumberColumn(c, format="%.2f")
                       for c in numeric_cols}
            for c in pct_cols:
                col_cfg[c] = st.column_config.NumberColumn(c, format="%.1f%%")
            # NumberColumn formats raw values, so scale percent fields ×100
            for c in pct_cols:
                df_peers[c] = df_peers[c] * 100
            if "Log Market Cap" in df_peers.columns:
                col_cfg["Log Market Cap"] = st.column_config.NumberColumn(
                    "Log Market Cap", format="%.2f",
                    help="Natural log of market cap (in INR). Used for clustering peers of comparable size.",
                )
            if "Role" in df_peers.columns:
                col_cfg["Role"] = st.column_config.TextColumn(
                    "Role", help="Whether this row is the target firm or a peer."
                )
            st.dataframe(
                df_peers, use_container_width=True, hide_index=True,
                column_config=col_cfg,
            )
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
            st.info(
                "**Sensitivity tornado is not shown for this stock.** "
                "The tornado visualises how DDM inputs (cost of equity, "
                "growth rate, payout ratio) move the DDM output — but the "
                "DDM track was skipped here because the company does not "
                "pay regular dividends. For uncertainty around the "
                "relative-valuation track, see the **Monte Carlo** tab, "
                "which shows the full distribution of fair-value outcomes."
            )

    with tab6:
        st.write("Generate a polished PDF research note for this stock.")

        # Two-step flow:
        #   1. "Generate" runs the (slow) PDF build and caches the bytes in
        #      session_state so they survive the rerun triggered by step 2.
        #   2. "Download" reads from session_state. It MUST live outside the
        #      generate button's `if`-branch — otherwise the download button
        #      vanishes on the rerun before the file is served.
        if st.button("Generate PDF report", type="primary"):
            with st.spinner("Composing report..."):
                try:
                    path = generate_pdf(result)
                    with open(path, "rb") as f:
                        st.session_state["pdf_bytes"] = f.read()
                    st.session_state["pdf_name"] = path.name
                    st.session_state["pdf_path"] = str(path)
                except Exception as e:
                    st.session_state.pop("pdf_bytes", None)
                    st.error(f"Failed to generate report: {e}")

        if "pdf_bytes" in st.session_state:
            st.success(
                f"✅ Report ready — {st.session_state['pdf_name']}  "
                f"({len(st.session_state['pdf_bytes']) / 1024:,.0f} KB)"
            )
            st.download_button(
                "⬇️ Download PDF report",
                data=st.session_state["pdf_bytes"],
                file_name=st.session_state["pdf_name"],
                mime="application/pdf",
                type="primary",
            )
            st.caption(f"Saved to: `{st.session_state['pdf_path']}`")

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
