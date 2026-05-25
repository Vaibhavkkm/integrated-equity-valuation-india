"""
Synthetic StockBundle factory shared between conftest fixtures and
direct test imports.

Lives outside conftest.py because pytest doesn't expose conftest as a
regular import target; tests that need to *call* the factory directly
(rather than receive it as a fixture) import from here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_fetcher import StockBundle


def _annual_index(n: int, end_year: int = 2025) -> pd.DatetimeIndex:
    return pd.date_range(end=f"{end_year}-12-31", periods=n, freq="YE")


def make_bundle(**overrides) -> StockBundle:
    """Build a plausible StockBundle for an Indian large-cap.

    Defaults model a stable, mature dividend payer (~ITC-ish profile).
    Override any field to model loss-makers, fast growers, etc.
    """
    n_years = overrides.pop("n_years", 7)
    eps_series = pd.Series(
        np.linspace(15.0, 25.0, n_years), index=_annual_index(n_years)
    )
    rev_series = pd.Series(
        np.linspace(40_000.0, 70_000.0, n_years) * 1e7, index=_annual_index(n_years)
    )
    bv_series = pd.Series(
        np.linspace(80.0, 120.0, n_years), index=_annual_index(n_years)
    )
    dps_series = pd.Series(
        np.linspace(8.0, 14.0, n_years), index=_annual_index(n_years)
    )

    defaults = dict(
        ticker="TEST.NS",
        sector="FMCG",
        name="Test FMCG Ltd",
        price=400.0,
        market_cap=5e12,
        shares_outstanding=1.25e10,
        beta_raw=0.85,
        price_history=pd.Series(
            np.linspace(350.0, 400.0, 250),
            index=pd.date_range(end="2026-01-01", periods=250, freq="B"),
        ),
        eps_ttm=25.0,
        book_value_per_share=120.0,
        sales_per_share_ttm=56.0,
        dividend_per_share_ttm=14.0,
        free_cash_flow_per_share_ttm=20.0,
        revenue=70_000e7,
        ebitda=22_000e7,
        net_income=18_000e7,
        total_debt=2_000e7,
        cash=8_000e7,
        equity=150_000e7,
        total_assets=200_000e7,
        dividends_annual=dps_series,
        earnings_annual=eps_series,
        revenue_annual=rev_series,
        book_value_annual=bv_series,
        payout_ratio=0.56,
        roe=0.21,
        enterprise_value=5e12 + 2_000e7 - 8_000e7,
    )
    defaults.update(overrides)
    return StockBundle(**defaults)


def make_sparse_history_bundle(**overrides) -> StockBundle:
    """Build a StockBundle with only 1 year of cash-flow history.

    Used to test the boundary behaviour of FCFE applicability and
    related checks that require ≥3 years of cash-flow data. Empirically
    no real ticker in the curated NSE universe is genuinely sparse —
    yfinance backfills pre-IPO data from the RHP, so even Nov-2024
    listings like SWIGGY.NS return a full 5-year history. The
    boundary cases this fixture covers therefore have to be tested with
    a synthetic bundle; there is no naturally-occurring example to lean
    on.
    """
    idx_one = _annual_index(1)
    sparse_overrides = dict(
        ticker="SPARSE.NS",
        name="Recent IPO Co",
        n_years=1,
        earnings_annual=pd.Series([18.0], index=idx_one),
        revenue_annual=pd.Series([20_000e7], index=idx_one),
        dividends_annual=pd.Series(dtype=float),
        book_value_annual=pd.Series([60.0], index=idx_one),
        # Single year of every cash-flow field — below the 3-year
        # applicability floor downstream consumers will enforce.
        net_income_annual=pd.Series([1_800e7], index=idx_one),
        capex_annual=pd.Series([600e7], index=idx_one),
        dep_amort_annual=pd.Series([400e7], index=idx_one),
        change_in_wc_annual=pd.Series([200e7], index=idx_one),
        working_capital_annual=pd.Series([1_500e7], index=idx_one),
        total_debt_annual=pd.Series([2_000e7], index=idx_one),
        payout_ratio=0.0,
        dividend_per_share_ttm=0.0,
    )
    sparse_overrides.update(overrides)
    return make_bundle(**sparse_overrides)
