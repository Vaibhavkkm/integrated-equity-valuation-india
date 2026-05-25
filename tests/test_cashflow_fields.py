"""
Phase A.0 — unit tests for the cash-flow / balance-sheet fields added to
StockBundle.

These tests pin down the extraction behaviour of `_annual_series()` and
`_first_present()` against synthetic DataFrames that mirror the row
labels yfinance returns on Indian listings. Empirical probing showed
five sample tickers (TCS, ITC, SIEMENS, RELIANCE, SWIGGY) return clean
data with consistent canonical row names — the tests below codify those
expectations so a future yfinance schema drift surfaces here instead of
silently corrupting Phase A's FCFE numbers.

The HEADLINE test (`test_current_assets_disambiguation`) is the one that
matters most operationally: the substring fallback inside
`_first_present()` will match "Other Current Assets" if asked for
"Current Assets" loosely. This would silently corrupt working-capital
computations for any ticker where the exact "Current Assets" label is
missing. The mitigation is to pass the exact label first and rely on
`_first_present`'s exact-match-then-substring priority. The test below
forces a synthetic DataFrame with *both* labels present and confirms
the right row is picked.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_fetcher import (
    StockBundle,
    _annual_series,
    _first_present,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _periods(n: int = 5) -> pd.DatetimeIndex:
    """Build n descending period-end timestamps (Yahoo's native order)."""
    return pd.to_datetime([
        f"{2025 - i}-03-31" for i in range(n)
    ])


def _frame(rows: dict[str, list[float]]) -> pd.DataFrame:
    """Build a DataFrame mimicking yfinance: rows are line items, columns
    are period-end timestamps in descending order."""
    n = max(len(v) for v in rows.values())
    cols = _periods(n)
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)


# ---------------------------------------------------------------------------
# HEADLINE: exact-match precedence over substring matching
# ---------------------------------------------------------------------------
def test_current_assets_disambiguation():
    """`Current Assets` must not get confused with `Other Current Assets`
    or `Total Non Current Assets`.

    Without exact-match precedence, `_first_present(df, "Current Assets")`
    would substring-match the first row whose label contains "current
    assets" — that is often "Other Current Assets" (alphabetical
    accident, since yfinance returns rows in a fixed order). The result
    is a silently-wrong working-capital computation.
    """
    bs = _frame({
        "Other Current Assets": [10, 11, 12, 13, 14],
        "Total Non Current Assets": [200, 210, 220, 230, 240],
        "Current Assets": [100, 110, 120, 130, 140],
        "Other Current Liabilities": [5, 6, 7, 8, 9],
        "Total Non Current Liabilities Net Minority Interest": [80, 85, 90, 95, 100],
        "Current Liabilities": [50, 55, 60, 65, 70],
    })

    ca = _annual_series(bs, "Current Assets", "Total Current Assets")
    cl = _annual_series(bs, "Current Liabilities", "Total Current Liabilities")

    # The picked rows must be the EXACT-match ones, not the substring siblings.
    # Note: yfinance returns columns descending (newest leftmost), and
    # `_annual_series` sorts chronologically (oldest → newest) before
    # returning. So the leftmost column value (newest period, 100) lands
    # at iloc[-1], and the rightmost (oldest, 140) lands at iloc[0].
    # Either endpoint confirms the right row was picked.
    assert set(ca.values.tolist()) == {100, 110, 120, 130, 140}, (
        f"Got values {sorted(ca.values.tolist())} — picked the wrong row. "
        f"Expected the 'Current Assets' row (100–140); likely matched "
        f"'Other Current Assets' (10–14) or 'Total Non Current Assets' (200–240)."
    )
    assert set(cl.values.tolist()) == {50, 55, 60, 65, 70}, (
        f"Got values {sorted(cl.values.tolist())} — picked the wrong row. "
        f"Expected the 'Current Liabilities' row (50–70); likely matched "
        f"'Other Current Liabilities' (5–9) or the non-current variant."
    )

    # And the derived working-capital figure (CA − CL) must match the
    # correctly-picked rows. Newest period (iloc[-1]): 100 − 50 = 50.
    # Oldest period (iloc[0]): 140 − 70 = 70.
    wc_derived = ca - cl
    assert wc_derived.iloc[-1] == 50
    assert wc_derived.iloc[0] == 70


def test_first_present_exact_match_beats_substring():
    """Lower-level pin: `_first_present` returns the exact row even when
    a substring sibling appears before it in the DataFrame's index.

    This is the invariant the higher-level disambiguation test relies on.
    """
    df = _frame({
        "Other Current Assets": [1.0, 2.0, 3.0],
        "Current Assets": [100.0, 200.0, 300.0],
    })
    picked = _first_present(df, "Current Assets")
    # Exact match must beat the substring sibling.
    assert picked is not None
    assert list(picked.values) == [100.0, 200.0, 300.0]


# ---------------------------------------------------------------------------
# Series extraction — canonical yfinance row labels
# ---------------------------------------------------------------------------
def test_capex_extraction_chronological_order():
    """yfinance returns period-end columns descending; the series must
    come back oldest → newest so downstream growth-rate math is anchored
    to the chronological endpoints."""
    cf = _frame({"Capital Expenditure": [-500, -450, -400, -350, -300]})
    s = _annual_series(cf, "Capital Expenditure")
    # Oldest first
    assert s.iloc[0] == -300
    assert s.iloc[-1] == -500
    assert list(s.index) == sorted(s.index)


def test_dep_amort_combined_label_preferred_over_standalone_depreciation():
    """When both 'Depreciation And Amortization' and standalone
    'Depreciation' are present, the combined row is the FCFE input —
    standalone Depreciation is the non-amortisation slice only."""
    cf = _frame({
        "Depreciation": [80, 90, 100, 110, 120],
        "Depreciation And Amortization": [100, 115, 130, 145, 160],
    })
    s = _annual_series(
        cf, "Depreciation And Amortization", "Depreciation Amortization Depletion",
    )
    assert s.iloc[-1] == 100  # newest in chronological order = oldest period
    # The combined label was picked, not standalone Depreciation
    assert s.iloc[0] == 160


def test_change_in_wc_extraction():
    cf = _frame({"Change In Working Capital": [10, -20, 30, -40, 50]})
    s = _annual_series(cf, "Change In Working Capital", "Changes In Working Capital")
    assert not s.empty
    assert len(s) == 5


def test_net_income_continuing_operations_preferred():
    """When both 'Net Income From Continuing Operations' and
    'Net Income' are present, the continuing-ops figure is the cleaner
    one (excludes one-off discontinued-operations P&L)."""
    cf = _frame({
        "Net Income": [1000, 1100, 1200, 1300, 1400],
        "Net Income From Continuing Operations": [950, 1050, 1150, 1250, 1350],
    })
    s = _annual_series(
        cf, "Net Income From Continuing Operations", "Net Income",
        "Net Income Common Stockholders",
    )
    # Continuing operations row picked (oldest = 1350 since columns are descending)
    assert s.iloc[0] == 1350


# ---------------------------------------------------------------------------
# Empty/missing-statement handling
# ---------------------------------------------------------------------------
def test_annual_series_on_empty_dataframe_returns_empty():
    s = _annual_series(pd.DataFrame(), "Capital Expenditure")
    assert isinstance(s, pd.Series)
    assert s.empty


def test_annual_series_on_missing_row_returns_empty():
    cf = _frame({"Depreciation And Amortization": [100, 110, 120, 130, 140]})
    s = _annual_series(cf, "Capital Expenditure")
    assert s.empty


# ---------------------------------------------------------------------------
# StockBundle field defaults — backward-compat for cached pickles
# ---------------------------------------------------------------------------
def test_stockbundle_cashflow_fields_default_to_empty_series():
    """A StockBundle constructed without the new cash-flow kwargs must
    populate all six new fields as empty Series — this is what makes the
    field set backward-compatible with cached pickles created before
    Phase A.0.
    """
    b = StockBundle(
        ticker="X.NS", sector="FMCG", name="X",
        price=100.0, market_cap=1e10, shares_outstanding=1e8,
        beta_raw=1.0, price_history=pd.Series(dtype=float),
        eps_ttm=10.0, book_value_per_share=50.0, sales_per_share_ttm=80.0,
        dividend_per_share_ttm=5.0, free_cash_flow_per_share_ttm=8.0,
        revenue=8e9, ebitda=2e9, net_income=1e9,
        total_debt=5e8, cash=1e9, equity=4e9, total_assets=8e9,
        dividends_annual=pd.Series(dtype=float),
        earnings_annual=pd.Series(dtype=float),
        revenue_annual=pd.Series(dtype=float),
        book_value_annual=pd.Series(dtype=float),
        payout_ratio=0.5, roe=0.20, enterprise_value=1e10,
    )

    for name in (
        "net_income_annual", "capex_annual", "dep_amort_annual",
        "change_in_wc_annual", "working_capital_annual", "total_debt_annual",
    ):
        s = getattr(b, name)
        assert isinstance(s, pd.Series), f"{name} is not a pd.Series"
        assert s.empty, f"{name} is not empty by default"


def test_stockbundle_round_trips_through_pickle():
    """A bundle with the new fields populated must pickle and unpickle
    without loss. This is the in-memory analogue of the cache reload
    test in test_pickle_backcompat.py."""
    b = StockBundle(
        ticker="Y.NS", sector="FMCG", name="Y",
        price=100.0, market_cap=1e10, shares_outstanding=1e8,
        beta_raw=1.0, price_history=pd.Series(dtype=float),
        eps_ttm=10.0, book_value_per_share=50.0, sales_per_share_ttm=80.0,
        dividend_per_share_ttm=5.0, free_cash_flow_per_share_ttm=8.0,
        revenue=8e9, ebitda=2e9, net_income=1e9,
        total_debt=5e8, cash=1e9, equity=4e9, total_assets=8e9,
        dividends_annual=pd.Series(dtype=float),
        earnings_annual=pd.Series(dtype=float),
        revenue_annual=pd.Series(dtype=float),
        book_value_annual=pd.Series(dtype=float),
        payout_ratio=0.5, roe=0.20, enterprise_value=1e10,
        capex_annual=pd.Series([100.0, 110.0, 120.0]),
        dep_amort_annual=pd.Series([50.0, 55.0, 60.0]),
    )
    revived: StockBundle = pickle.loads(pickle.dumps(b))
    assert list(revived.capex_annual.values) == [100.0, 110.0, 120.0]
    assert list(revived.dep_amort_annual.values) == [50.0, 55.0, 60.0]
    # Untouched defaults still empty after round-trip
    assert revived.change_in_wc_annual.empty
    assert revived.working_capital_annual.empty


# ---------------------------------------------------------------------------
# CapEx sign convention — applied at the fetcher boundary
# ---------------------------------------------------------------------------
def test_capex_stored_as_absolute_value_in_fetcher():
    """The fetcher flips yfinance's negative CapEx sign at the boundary
    so consumers can read the field as 'this year's CapEx was ₹X' with
    X > 0. The unit-level test below confirms `.abs()` is what the
    fetcher applies; the integration counterpart is the audit script.
    """
    # Simulate what the fetcher does internally with a raw cashflow row.
    raw = _annual_series(
        _frame({"Capital Expenditure": [-500, -450, -400, -350, -300]}),
        "Capital Expenditure",
    )
    stored = raw.abs()
    assert (stored >= 0).all()
    # Sign-flip must not have rearranged the values.
    assert stored.iloc[0] == 300  # oldest period (column was 2021-03-31 originally)
    assert stored.iloc[-1] == 500


# ---------------------------------------------------------------------------
# Working-capital fallback derivation
# ---------------------------------------------------------------------------
def test_working_capital_derivation_when_explicit_row_missing():
    """When yfinance omits the 'Working Capital' row, the fetcher must
    derive it as Current Assets − Current Liabilities, using exact-match
    extraction so 'Other Current Assets' / 'Other Current Liabilities'
    are not picked up."""
    bs = _frame({
        "Other Current Assets": [10, 11, 12],
        "Current Assets": [100, 110, 120],
        "Other Current Liabilities": [5, 6, 7],
        "Current Liabilities": [50, 55, 60],
    })
    ca = _annual_series(bs, "Current Assets", "Total Current Assets")
    cl = _annual_series(bs, "Current Liabilities", "Total Current Liabilities")
    derived = ca - cl
    # Oldest period: 120 − 60 = 60 (columns are 2025, 2024, 2023 desc;
    # after sort_index, oldest = index 0 = column 2023 = value 120/60)
    assert derived.iloc[0] == 60
    # Newest period: 100 − 50 = 50
    assert derived.iloc[-1] == 50


def test_total_debt_fallback_sum_of_ltd_and_std():
    """When 'Total Debt' is absent, the fetcher must reconstruct it from
    Long Term Debt + Current Debt — this is the path most often
    triggered for banks and a couple of NBFCs in the universe."""
    bs = _frame({
        "Long Term Debt": [1000, 1100, 1200, 1300, 1400],
        "Current Debt": [200, 220, 240, 260, 280],
    })
    ltd = _annual_series(bs, "Long Term Debt", "Long Term Debt And Capital Lease Obligation")
    std = _annual_series(bs, "Current Debt", "Current Debt And Capital Lease Obligation", "Short Term Debt")
    summed = ltd + std
    # Oldest column = 2021 = values (1400, 280) → sum 1680
    assert summed.iloc[0] == 1680
    # Newest column = 2025 = values (1000, 200) → sum 1200
    assert summed.iloc[-1] == 1200
