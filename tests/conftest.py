"""
Shared pytest fixtures.

The factories below build deterministic ``StockBundle`` instances so that
tests don't have to hit the network. Tweak any field via kwargs.

The actual factory lives in ``tests/_factory.py`` so test modules that
need to call it directly (not via fixture) can import it normally —
pytest does not expose conftest.py as a regular import target.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make `src.*` importable from the tests dir, and `_factory` from this dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _factory import _annual_index, make_bundle


@pytest.fixture
def mature_payer():
    """Stable mature dividend payer."""
    return make_bundle()


@pytest.fixture
def non_payer():
    """Profitable but pays no dividend (think: a young IT services firm)."""
    return make_bundle(
        ticker="GROWTH.NS", name="Growth Co",
        sector="Information Technology",
        dividend_per_share_ttm=0.0,
        dividends_annual=pd.Series(dtype=float),
        payout_ratio=0.0,
        roe=0.28,
    )


@pytest.fixture
def loss_maker():
    """Net loss in the latest year — DDM and most multiples should bow out."""
    eps_loss = pd.Series(
        [5.0, 3.0, 1.0, -2.0, -8.0],
        index=_annual_index(5),
    )
    return make_bundle(
        ticker="LOSS.NS", name="Loss Inc",
        eps_ttm=-8.0,
        net_income=-5e9,
        earnings_annual=eps_loss,
        dividend_per_share_ttm=0.0,
        dividends_annual=pd.Series(dtype=float),
        payout_ratio=0.0,
        roe=-0.10,
    )


@pytest.fixture
def thin_history():
    """Only 2 years of fundamentals — should trigger LOW data-quality."""
    return make_bundle(
        ticker="THIN.NS",
        n_years=2,
        earnings_annual=pd.Series([10.0, 12.0], index=_annual_index(2)),
        revenue_annual=pd.Series([5_000e7, 6_000e7], index=_annual_index(2)),
        dividends_annual=pd.Series([2.0, 2.5], index=_annual_index(2)),
        book_value_annual=pd.Series([40.0, 50.0], index=_annual_index(2)),
        price_history=pd.Series(dtype=float),
    )


@pytest.fixture
def negative_book():
    """Negative book value — Altman Z gets weird. Engine should not crash."""
    return make_bundle(
        ticker="NEGBV.NS", name="Underwater Co",
        book_value_per_share=-15.0,
        equity=-2e9,
    )
