"""
Research demo — runs the full valuation pipeline across a small basket of
Indian stocks and prints a comparative table.

This script is the equivalent of the "presentation deck" for the project —
it shows the engine working across diverse sectors (IT, Banking, FMCG,
Auto, Pharma, Oil & Gas) and gives a side-by-side BUY/HOLD/SELL view.

Run with:
    python notebooks/research_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integrated_valuation import value_stock


BASKET = [
    "TCS.NS",         # IT
    "INFY.NS",        # IT
    "HDFCBANK.NS",    # Banking
    "ICICIBANK.NS",   # Banking
    "ITC.NS",         # FMCG (conglomerate)
    "HINDUNILVR.NS",  # FMCG (pure-play)
    "MARUTI.NS",      # Auto
    "SUNPHARMA.NS",   # Pharma
    "RELIANCE.NS",    # Oil & Gas (mixed)
    "BHARTIARTL.NS",  # Telecom
]


def main() -> None:
    rows = []
    for t in BASKET:
        try:
            r = value_stock(t, verbose=False)
            rows.append({
                "Ticker": t,
                "Sector": r.target.sector,
                "CMP (₹)": round(r.target.price, 2),
                "DDM (₹)": round(r.ddm.value_per_share, 2) if r.ddm.valid else None,
                "Relative (₹)": round(r.relative.weighted_value, 2),
                "Blended (₹)": round(r.blended_value, 2),
                "MoS": f"{r.margin_of_safety:+.1%}" if r.margin_of_safety == r.margin_of_safety else "—",
                "Rating": r.recommendation,
                "Quality": round(r.quality.composite, 1),
                "DDM Variant": r.ddm.model,
            })
        except Exception as e:
            print(f"  ! {t}: {e}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("  COMPARATIVE VALUATION SUMMARY  —  Indian Equity Basket")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
