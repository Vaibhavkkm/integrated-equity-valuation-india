"""
Tests for the reverse-DCF / implied-expectations solver.

Key invariants we test:
  * round-trip consistency: feed the model's own fair value back in,
    recover the input growth rate.
  * monotonicity: higher market price → higher implied growth.
  * graceful failure for non-payers and degenerate inputs.
"""
from __future__ import annotations

import pytest

from src.ddm_models import two_stage_ddm
from src.reverse_dcf import implied_growth


# ---------------------------------------------------------------------------
# Round-trip — the most important test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("g_truth", [0.04, 0.08, 0.12, 0.20])
def test_round_trip_recovers_input_growth(g_truth):
    """If we price the stock at its own fair value, reverse DCF must
    return the growth rate we used to compute it."""
    d0, ke, g_term, n = 10.0, 0.12, 0.04, 5
    iv = two_stage_ddm(d0=d0, g_high=g_truth, g_terminal=g_term, ke=ke, n_years=n)
    assert iv.valid

    out = implied_growth(
        market_price=iv.value_per_share,
        d0=d0, ke=ke, g_terminal=g_term,
        high_growth_years=n,
    )
    assert out.valid
    assert out.implied_growth == pytest.approx(g_truth, abs=1e-3)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------
def test_higher_price_implies_higher_growth():
    base_args = dict(d0=5.0, ke=0.11, g_terminal=0.04, high_growth_years=5)
    g_low = implied_growth(market_price=80, **base_args)
    g_high = implied_growth(market_price=200, **base_args)
    assert g_low.valid and g_high.valid
    assert g_high.implied_growth > g_low.implied_growth


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------
def test_non_payer_returns_no_solution():
    out = implied_growth(market_price=500, d0=0.0, ke=0.12, g_terminal=0.04)
    assert not out.valid
    assert "dividend" in out.verdict.lower() or "fcfe" in out.verdict.lower()


def test_zero_price_returns_no_solution():
    out = implied_growth(market_price=0.0, d0=5.0, ke=0.12, g_terminal=0.04)
    assert not out.valid


def test_g_terminal_above_ke_returns_no_solution():
    out = implied_growth(
        market_price=100, d0=5.0, ke=0.05, g_terminal=0.07,
    )
    assert not out.valid
    assert "terminal" in out.verdict.lower() or "degenerate" in out.verdict.lower()


def test_extremely_overpriced_stock_flagged():
    """If g=30% can't justify the price, the verdict should say so."""
    out = implied_growth(
        market_price=100_000, d0=1.0, ke=0.12, g_terminal=0.04,
        g_low=-0.05, g_high=0.30,
    )
    assert not out.valid
    assert "ceiling" in out.verdict.lower() or "extraordinary" in out.verdict.lower()


# ---------------------------------------------------------------------------
# Verdict text — useful for the report; regression-test it lightly
# ---------------------------------------------------------------------------
def test_expectation_gap_populated_when_history_provided():
    out = implied_growth(
        market_price=200, d0=5.0, ke=0.11, g_terminal=0.04,
        historical_growth=0.10,
    )
    assert out.valid
    assert out.expectation_gap is not None
    assert out.historical_growth == pytest.approx(0.10)


def test_summary_string_mentions_solution_or_reason():
    ok = implied_growth(market_price=200, d0=5.0, ke=0.11, g_terminal=0.04)
    bad = implied_growth(market_price=200, d0=0, ke=0.11, g_terminal=0.04)
    assert "g =" in ok.summary() or "growth" in ok.summary().lower()
    assert "no solution" in bad.summary().lower()
