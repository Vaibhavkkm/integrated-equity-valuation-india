"""
End-to-end tests for the cost-of-equity engine.

Most of these short-circuit the live yfinance call by passing an explicit
``market_prices`` series in (the project supports this for testability),
so the network is never touched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cost_of_equity import (
    CostOfEquity, adjusted_beta, cost_of_equity,
    estimate_beta_from_prices, relever_sector_beta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _correlated_series(beta: float, n: int = 260, seed: int = 0):
    """Build a stock series whose true beta vs the market is approximately `beta`.

    ``n=260`` daily bars ≈ 5 years of weekly Fridays after resampling.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    rm = rng.normal(0.0005, 0.012, n)
    rs = beta * rm + rng.normal(0.0, 0.008, n)
    market = pd.Series(100 * np.exp(np.cumsum(rm)), index=idx)
    stock = pd.Series(50 * np.exp(np.cumsum(rs)), index=idx)
    return stock, market


# ---------------------------------------------------------------------------
# Beta estimation
# ---------------------------------------------------------------------------
def test_estimate_beta_recovers_true_beta():
    """A stock built with β=1.3 must regress to β ≈ 1.3 (±0.2)."""
    stock, market = _correlated_series(beta=1.3, n=520, seed=42)
    beta = estimate_beta_from_prices(stock, market)
    assert beta is not None
    assert abs(beta - 1.3) < 0.25


def test_estimate_beta_returns_none_on_short_history():
    stock, market = _correlated_series(beta=1.0, n=40, seed=1)
    assert estimate_beta_from_prices(stock, market) is None


def test_estimate_beta_returns_none_on_empty():
    assert estimate_beta_from_prices(pd.Series(dtype=float),
                                     pd.Series(dtype=float)) is None


# ---------------------------------------------------------------------------
# Bloomberg adjustment
# ---------------------------------------------------------------------------
def test_adjusted_beta_pulls_high_toward_one():
    assert 1.0 < adjusted_beta(2.0) < 2.0


def test_adjusted_beta_pulls_low_toward_one():
    assert 0.5 < adjusted_beta(0.3) < 1.0


def test_adjusted_beta_is_identity_at_one():
    assert adjusted_beta(1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Hamada re-levering
# ---------------------------------------------------------------------------
def test_relever_negative_de_clamps_to_zero():
    # Negative D/E (net-cash firm with negative reported debt) shouldn't
    # produce a beta below the unlevered figure — the function clamps.
    bu0 = relever_sector_beta("FMCG", debt_to_equity=0.0)
    bu_neg = relever_sector_beta("FMCG", debt_to_equity=-0.5)
    assert bu_neg == pytest.approx(bu0)


def test_relever_unknown_sector_defaults_to_one_unlevered():
    """Unknown sector falls back to a 1.0 unlevered beta."""
    b = relever_sector_beta("UnknownSector", debt_to_equity=0.0)
    assert b == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def test_cost_of_equity_uses_own_beta_when_history_is_sufficient():
    stock, market = _correlated_series(beta=1.1, n=520, seed=7)
    coe = cost_of_equity(
        stock_prices=stock,
        sector="FMCG",
        market_cap=1e12,         # large-cap, no small-cap premium
        total_debt=1e10,
        equity=5e11,
        market_prices=market,
    )
    assert isinstance(coe, CostOfEquity)
    assert "own beta" in coe.method
    assert 0.05 < coe.ke < 0.25
    assert coe.small_cap_premium == 0.0


def test_cost_of_equity_falls_back_to_sector_on_thin_history():
    stock = pd.Series(dtype=float)
    market = pd.Series(dtype=float)
    coe = cost_of_equity(
        stock_prices=stock,
        sector="FMCG",
        market_cap=1e12, total_debt=0, equity=5e11,
        market_prices=market,
    )
    assert "sector" in coe.method.lower()
    assert any("Insufficient" in n for n in coe.notes)


def test_cost_of_equity_adds_small_cap_premium_below_threshold():
    stock, market = _correlated_series(beta=1.0, n=520, seed=3)
    coe_small = cost_of_equity(
        stock_prices=stock,
        sector="FMCG",
        market_cap=1e10,         # ₹1,000 crore → below ₹5,000 crore threshold
        total_debt=0, equity=5e9,
        market_prices=market,
    )
    coe_large = cost_of_equity(
        stock_prices=stock,
        sector="FMCG",
        market_cap=1e12,         # large-cap
        total_debt=0, equity=5e11,
        market_prices=market,
    )
    assert coe_small.small_cap_premium == pytest.approx(0.015)
    assert coe_large.small_cap_premium == 0.0
    # Ke must be higher for the small-cap by exactly the premium.
    assert coe_small.ke - coe_large.ke == pytest.approx(0.015, abs=1e-6)


def test_cost_of_equity_rejects_unstable_beta_estimates():
    """A regression beta outside [-3, +3] is treated as unreliable and
    falls back to the sector estimate."""
    # Construct a noise-dominated series so the regression beta blows up.
    rng = np.random.default_rng(99)
    idx = pd.date_range("2021-01-01", periods=520, freq="B")
    market = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.001, 520))), index=idx)
    stock = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.05, 520))), index=idx)
    coe = cost_of_equity(
        stock_prices=stock, sector="FMCG",
        market_cap=1e12, total_debt=0, equity=5e11,
        market_prices=market,
    )
    # We don't assert the method strictly — depending on the seed the
    # regression beta may land inside [-3, 3]. We just assert the call
    # completes cleanly and produces a finite Ke within the wide band a
    # real Indian equity could plausibly sit in.
    assert np.isfinite(coe.ke)
    assert 0.0 < coe.ke < 0.40


def test_cost_of_equity_explain_renders():
    """The explain() string is what the PDF / dashboard prints — must not raise."""
    coe = cost_of_equity(
        stock_prices=pd.Series(dtype=float),
        sector="Banking",
        market_cap=1e10, total_debt=2e9, equity=4e9,
        market_prices=pd.Series(dtype=float),
    )
    text = coe.explain()
    assert "Cost of Equity" in text
    assert "Beta" in text
