"""
Tests for relative valuation aggregation and the multi-multiple blend.

We build small synthetic peer sets so we can hand-check the math.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.peer_identification import PeerSet
from src.relative_valuation import (
    DEFAULT_MULTIPLE_WEIGHTS, SECTOR_OVERRIDES, _trimmed_harmonic_mean,
    value_by_multiples,
)


# Local import of the shared bundle factory — pytest auto-discovers conftest.
from conftest import make_bundle  # noqa: E402


# ---------------------------------------------------------------------------
# Trimmed harmonic mean
# ---------------------------------------------------------------------------
def test_harmonic_mean_under_arithmetic_for_skewed_input():
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 100.0])
    hm = _trimmed_harmonic_mean(s, trim_pct=0.0)
    am = s.mean()
    # Harmonic mean is always ≤ arithmetic for positive values.
    assert hm < am


def test_trim_kills_single_outlier():
    s = pd.Series([18, 19, 20, 21, 22, 23, 24, 25, 26, 1_000])
    hm = _trimmed_harmonic_mean(s, trim_pct=0.10)
    # With the 1000 trimmed, harmonic should be roughly 22.
    assert 18 <= hm <= 28


def test_handles_all_negatives_returns_nan():
    s = pd.Series([-5.0, -10.0, -3.0])
    hm = _trimmed_harmonic_mean(s, trim_pct=0.10)
    assert not np.isfinite(hm)


def test_handles_empty_series_returns_nan():
    hm = _trimmed_harmonic_mean(pd.Series(dtype=float), trim_pct=0.10)
    assert not np.isfinite(hm)


# ---------------------------------------------------------------------------
# Sector weights
# ---------------------------------------------------------------------------
def test_default_weights_sum_to_one():
    assert sum(DEFAULT_MULTIPLE_WEIGHTS.values()) == pytest.approx(1.0)


def test_sector_override_weights_sum_to_one():
    for sector, w in SECTOR_OVERRIDES.items():
        assert sum(w.values()) == pytest.approx(1.0), sector


def test_banking_weights_emphasise_pb():
    bank = SECTOR_OVERRIDES["Banking"]
    assert bank["PB"] > bank["PE"]
    assert bank["EV_EBITDA"] == 0.0   # EV/EBITDA is meaningless for banks


# ---------------------------------------------------------------------------
# End-to-end: synthetic peer set
# ---------------------------------------------------------------------------
def _make_peers(n: int) -> list:
    """A row of nearly identical FMCG peers, all priced at 20× EPS."""
    return [
        make_bundle(ticker=f"P{i}.NS", price=400, eps_ttm=20)
        for i in range(n)
    ]


def test_value_by_multiples_returns_nan_when_all_invalid():
    target = make_bundle(eps_ttm=float("nan"), book_value_per_share=float("nan"),
                         sales_per_share_ttm=float("nan"), ebitda=float("nan"),
                         enterprise_value=float("nan"))
    peer_set = PeerSet(target=target, peers=_make_peers(5), method="test")
    result = value_by_multiples(peer_set)
    # With no valid target metric, the implied prices cascade to NaN.
    assert not np.isfinite(result.weighted_value) or result.weighted_value == 0


def test_value_by_multiples_runs_on_clean_set():
    target = make_bundle(price=400)
    peer_set = PeerSet(target=target, peers=_make_peers(8), method="test")
    result = value_by_multiples(peer_set)
    assert np.isfinite(result.weighted_value)
    assert result.weighted_value > 0
    assert sum(result.weights.values()) == pytest.approx(1.0)
