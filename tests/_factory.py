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
