"""
Unit tests covering the core valuation maths. These are deterministic
and do NOT require network access — they exercise the model formulas
against textbook values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ddm_models import (
    gordon_growth, two_stage_ddm, three_stage_ddm, h_model,
    select_and_value, historical_dividend_cagr, sustainable_growth_rate,
)
from src.cost_of_equity import adjusted_beta, relever_sector_beta
from src.quality_score import dividend_quality
from src.relative_valuation import _trimmed_harmonic_mean


# ---------------------------------------------------------------------------
# DDM closed forms — known values
# ---------------------------------------------------------------------------
def test_gordon_growth_basic():
    """Damodaran textbook example: D1=2, Ke=10%, g=4% → V = 2/0.06 = 33.33."""
    iv = gordon_growth(d1=2.0, ke=0.10, g=0.04)
    assert iv.valid
    assert iv.value_per_share == pytest.approx(33.333333, rel=1e-4)


def test_gordon_caps_growth_above_ke():
    """g >= Ke is invalid — model should clip and warn, not crash."""
    iv = gordon_growth(d1=2.0, ke=0.08, g=0.10)
    assert iv.valid
    assert iv.inputs["g"] < iv.inputs["Ke"]
    assert "Gordon" in iv.note


def test_gordon_zero_dividend_invalid():
    iv = gordon_growth(d1=0.0, ke=0.10, g=0.04)
    assert not iv.valid


def test_two_stage_ddm_basic():
    """Hand-checked: D0=1, g_high=15%, g_term=4%, Ke=10%, n=5."""
    iv = two_stage_ddm(d0=1.0, g_high=0.15, g_terminal=0.04, ke=0.10, n_years=5)
    assert iv.valid
    # Sum of explicit + terminal must be a positive number
    assert iv.value_per_share > 10
    assert iv.value_per_share < 100


def test_h_model_collapses_to_gordon_when_g_high_equals_g_terminal():
    """When g_high == g_terminal, H-Model should equal Gordon."""
    g = 0.04; ke = 0.10; d0 = 2.0
    iv_h = h_model(d0=d0, g_high=g, g_terminal=g, ke=ke, transition_years=10)
    iv_g = gordon_growth(d1=d0 * (1 + g), ke=ke, g=g)
    assert iv_h.value_per_share == pytest.approx(iv_g.value_per_share, rel=1e-3)


def test_three_stage_ddm_runs():
    iv = three_stage_ddm(d0=1.0, g_high=0.18, g_terminal=0.04, ke=0.11,
                         high_years=5, fade_years=5)
    assert iv.valid
    assert iv.value_per_share > 0


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------
def test_selector_skips_non_payer():
    iv = select_and_value(
        eps_ttm=50, dps_ttm=0, payout_ratio=0, roe=0.15, ke=0.12,
        historical_dps=pd.Series([0, 0, 0]),
        historical_eps=pd.Series([30, 40, 50]),
    )
    assert not iv.valid
    assert "non-payer" in iv.note.lower() or "skipped" in iv.model.lower()


def test_selector_picks_gordon_for_mature_payer():
    historical = pd.Series(
        [10, 11, 12, 13, 14, 15, 16],
        index=pd.date_range("2018", periods=7, freq="YE"),
    )
    iv = select_and_value(
        eps_ttm=20, dps_ttm=16, payout_ratio=0.80, roe=0.12, ke=0.10,
        historical_dps=historical,
        historical_eps=pd.Series([15, 16, 18, 19, 20]),
        sector_g_terminal=0.04,
    )
    assert iv.valid
    assert iv.model == "Gordon Growth"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def test_historical_dividend_cagr_with_special_div_outlier():
    # Special dividend in middle should be trimmed.
    s = pd.Series([10, 11, 100, 13, 14],
                  index=pd.date_range("2020", periods=5, freq="YE"))
    g = historical_dividend_cagr(s)
    assert -0.05 < g < 0.30
    # Without trim it would be massive; with trim it's roughly 9-10%.
    assert g < 0.50


def test_sustainable_growth_rate():
    g = sustainable_growth_rate(roe=0.20, payout_ratio=0.40)
    # g = ROE * b = 0.20 * 0.60 = 0.12
    assert g == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# Beta + relevering
# ---------------------------------------------------------------------------
def test_adjusted_beta_pulls_toward_one():
    """Adjusted beta must lie between raw and 1."""
    raw = 1.6
    adj = adjusted_beta(raw)
    assert 1.0 < adj < raw


def test_relever_sector_beta_increases_with_leverage():
    bu_low = relever_sector_beta("FMCG", debt_to_equity=0.0)
    bu_hi  = relever_sector_beta("FMCG", debt_to_equity=2.0)
    assert bu_hi > bu_low


# ---------------------------------------------------------------------------
# Trimmed harmonic mean
# ---------------------------------------------------------------------------
def test_trimmed_harmonic_mean_handles_outlier():
    # 9 values around 20, one extreme 200 → trimmed harmonic should ≈ 20
    s = pd.Series([18, 19, 20, 21, 22, 23, 24, 25, 200, 26])
    hm = _trimmed_harmonic_mean(s, trim_pct=0.10)
    assert 19 < hm < 25


def test_trimmed_harmonic_mean_drops_negatives():
    s = pd.Series([-5, 10, 12, 14, 16])
    hm = _trimmed_harmonic_mean(s, trim_pct=0.10)
    assert hm > 0
    assert np.isfinite(hm)


# ---------------------------------------------------------------------------
# Quality score on a synthetic bundle
# ---------------------------------------------------------------------------
def test_dividend_quality_zero_for_non_payer():
    from src.data_fetcher import StockBundle
    b = StockBundle(
        ticker="X.NS", sector="FMCG", name="X",
        price=100, market_cap=1e9, shares_outstanding=1e6,
        beta_raw=1.0, price_history=pd.Series(dtype=float),
        eps_ttm=10, book_value_per_share=50, sales_per_share_ttm=30,
        dividend_per_share_ttm=0,
        free_cash_flow_per_share_ttm=8,
        revenue=3e7, ebitda=8e6, net_income=5e6,
        total_debt=0, cash=2e6, equity=2e7, total_assets=3e7,
        dividends_annual=pd.Series(dtype=float),
        earnings_annual=pd.Series([3, 4, 5]),
        revenue_annual=pd.Series([20, 25, 30]),
        book_value_annual=pd.Series([30, 40, 50]),
        payout_ratio=0, roe=0.20, enterprise_value=1e9,
    )
    score, _ = dividend_quality(b)
    assert score == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
