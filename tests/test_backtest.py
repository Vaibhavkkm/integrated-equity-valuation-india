"""
Tests for the backtest grading pipeline.

We avoid hitting the network in tests — instead we monkeypatch
``_forward_return`` to return synthetic returns. This lets us verify the
binning, spread calculation, and survivorship-bias accounting in
deterministic isolation.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import backtest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stub_forward_returns(returns_by_ticker, benchmark_return=0.10):
    """Return a stub that maps tickers → fixed returns."""
    def stub(ticker, start, end):
        if ticker == "^NSEI":
            return benchmark_return
        return returns_by_ticker.get(ticker, None)
    return stub


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
def test_evaluate_signals_bins_by_recommendation(monkeypatch):
    panel = pd.DataFrame([
        dict(signal_date="2024-01-01", ticker="A.NS", recommendation="BUY",  price_at_signal=100),
        dict(signal_date="2024-01-01", ticker="B.NS", recommendation="BUY",  price_at_signal=100),
        dict(signal_date="2024-01-01", ticker="C.NS", recommendation="SELL", price_at_signal=100),
        dict(signal_date="2024-01-01", ticker="D.NS", recommendation="HOLD", price_at_signal=100),
    ])
    monkeypatch.setattr(
        backtest, "_forward_return",
        _stub_forward_returns({
            "A.NS": 0.30, "B.NS": 0.20,    # BUYs win
            "C.NS": -0.10,                 # SELL loses
            "D.NS": 0.05,                  # HOLD flat-ish
        }),
    )
    res = backtest.evaluate_signals(panel, horizon_months=12)

    assert res.n_signals == 4
    assert res.n_evaluated == 4
    assert res.by_bin["BUY"].n == 2
    assert res.by_bin["SELL"].n == 1
    assert res.by_bin["HOLD"].n == 1
    assert res.by_bin["BUY"].mean_return == pytest.approx(0.25)
    assert res.spread_buy_minus_sell == pytest.approx(0.35)


def test_evaluate_handles_dropped_tickers(monkeypatch):
    panel = pd.DataFrame([
        dict(signal_date="2024-01-01", ticker="OK.NS",       recommendation="BUY", price_at_signal=100),
        dict(signal_date="2024-01-01", ticker="DELISTED.NS", recommendation="BUY", price_at_signal=100),
    ])
    monkeypatch.setattr(
        backtest, "_forward_return",
        _stub_forward_returns({"OK.NS": 0.15}),  # delisted returns None
    )
    res = backtest.evaluate_signals(panel, horizon_months=12)
    assert res.n_signals == 2
    assert res.n_evaluated == 1
    assert any("dropped" in n.lower() for n in res.notes)


def test_hit_rate_vs_benchmark(monkeypatch):
    panel = pd.DataFrame([
        dict(signal_date="2024-01-01", ticker="WIN.NS",  recommendation="BUY", price_at_signal=100),
        dict(signal_date="2024-01-01", ticker="LOSE.NS", recommendation="BUY", price_at_signal=100),
    ])
    monkeypatch.setattr(
        backtest, "_forward_return",
        _stub_forward_returns(
            {"WIN.NS": 0.20, "LOSE.NS": 0.05},
            benchmark_return=0.10,
        ),
    )
    res = backtest.evaluate_signals(panel, horizon_months=12)
    # Only WIN.NS beat the 10% benchmark → 50%
    assert res.hit_rate_buy_vs_index == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Required-column validation
# ---------------------------------------------------------------------------
def test_missing_columns_raises():
    bad = pd.DataFrame([dict(signal_date="2024-01-01", ticker="X.NS")])
    with pytest.raises(ValueError):
        backtest.evaluate_signals(bad)


# ---------------------------------------------------------------------------
# Append signal round-trip
# ---------------------------------------------------------------------------
def test_append_and_read_round_trip(tmp_path, monkeypatch):
    log_path = tmp_path / "blog.csv"
    monkeypatch.setattr(backtest, "_LOG_PATH", log_path)

    backtest.append_signal(
        ticker="X.NS", recommendation="BUY",
        blended_value=120.0, price=100.0,
        margin_of_safety=0.20, quality_score=70.0,
    )
    backtest.append_signal(
        ticker="Y.NS", recommendation="SELL",
        blended_value=80.0, price=100.0,
        margin_of_safety=-0.20, quality_score=40.0,
    )

    df = pd.read_csv(log_path)
    assert len(df) == 2
    assert set(df["ticker"]) == {"X.NS", "Y.NS"}
    assert "BUY" in set(df["recommendation"])
