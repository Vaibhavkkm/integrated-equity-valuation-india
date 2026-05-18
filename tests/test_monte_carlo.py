"""
Tests for the vectorised Monte Carlo blender + tornado.

Key invariants:
  * The same seed reproduces the same distribution (reproducibility).
  * The blended mean sits between the two track values when both are valid.
  * NaN propagation: when DDM is invalid, the result reflects the relative
    track alone; when both are invalid, samples is empty.
  * P10 ≤ P50 ≤ P90.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cost_of_equity import CostOfEquity
from src.relative_valuation import MultipleResult, RelativeValuation
from src.sensitivity import (
    MonteCarloResult, monte_carlo_blended, tornado_ddm,
)

from _factory import make_bundle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def coe():
    return CostOfEquity(
        ke=0.12, beta_raw=0.9, beta_adjusted=0.93,
        risk_free=0.071, erp=0.07, method="test",
    )


def _rel_at(value: float) -> RelativeValuation:
    empty = pd.Series(dtype=float)
    if not np.isfinite(value):
        mr = MultipleResult(
            multiple_name="PE", peer_values=empty,
            aggregated_multiple=float("nan"),
            target_per_share_metric=float("nan"),
            implied_price=float("nan"), valid=False,
        )
        return RelativeValuation(
            multiples={"PE": mr}, weighted_value=float("nan"),
            weights={}, median_value=float("nan"),
        )
    mr = MultipleResult(
        multiple_name="PE", peer_values=empty,
        aggregated_multiple=20.0, target_per_share_metric=value / 20.0,
        implied_price=value, valid=True,
    )
    return RelativeValuation(
        multiples={"PE": mr}, weighted_value=value,
        weights={"PE": 1.0}, median_value=value,
    )


# ---------------------------------------------------------------------------
# Reproducibility + shape
# ---------------------------------------------------------------------------
def test_same_seed_reproduces_distribution(coe):
    target = make_bundle()
    rel = _rel_at(500.0)
    mc1 = monte_carlo_blended(target, coe, rel, 0.045, 0.5,
                              n_paths=2000, seed=123)
    mc2 = monte_carlo_blended(target, coe, rel, 0.045, 0.5,
                              n_paths=2000, seed=123)
    assert np.allclose(mc1.samples, mc2.samples)
    assert mc1.p50 == pytest.approx(mc2.p50)


def test_percentiles_are_ordered(coe):
    target = make_bundle()
    rel = _rel_at(500.0)
    mc = monte_carlo_blended(target, coe, rel, 0.045, 0.5,
                             n_paths=5000, seed=7)
    assert mc.p10 <= mc.p50 <= mc.p90
    assert mc.std >= 0
    assert isinstance(mc, MonteCarloResult)


# ---------------------------------------------------------------------------
# NaN propagation
# ---------------------------------------------------------------------------
def test_invalid_relative_falls_back_to_ddm(coe):
    target = make_bundle()
    rel = _rel_at(float("nan"))
    mc = monte_carlo_blended(target, coe, rel, 0.045, 0.5,
                             n_paths=2000, seed=1)
    # With the relative track NaN, every kept sample is a DDM-only value.
    assert mc.samples.size > 0
    assert np.all(np.isfinite(mc.samples))


def test_non_payer_falls_back_to_relative(coe):
    """When the target doesn't pay dividends, the DDM track is NaN and
    samples must come from the lognormal-perturbed relative track."""
    target = make_bundle(
        dividend_per_share_ttm=0.0,
        dividends_annual=pd.Series(dtype=float),
        payout_ratio=0.0,
    )
    rel = _rel_at(400.0)
    mc = monte_carlo_blended(target, coe, rel, 0.045, 0.5,
                             n_paths=2000, seed=2)
    assert mc.samples.size > 0
    # Lognormal centred on 400 with σ_log=0.20: median is exactly 400 in
    # the limit; with 2000 paths, ±5% is the right tolerance.
    assert 380 < mc.p50 < 420


def test_both_invalid_produces_empty_samples(coe):
    target = make_bundle(
        dividend_per_share_ttm=0.0,
        dividends_annual=pd.Series(dtype=float),
        payout_ratio=0.0,
    )
    rel = _rel_at(float("nan"))
    mc = monte_carlo_blended(target, coe, rel, 0.045, 0.5,
                             n_paths=2000, seed=3)
    assert mc.samples.size == 0
    assert np.isnan(mc.p50)


# ---------------------------------------------------------------------------
# Blend behaviour
# ---------------------------------------------------------------------------
def test_blended_median_lies_between_track_values(coe):
    """With DDM ~₹350 (central case for the synthetic bundle) and
    Relative pinned at ₹500, a 50/50 blend's median should land between
    them — neither below the lower nor above the upper."""
    target = make_bundle()
    rel = _rel_at(500.0)
    mc = monte_carlo_blended(target, coe, rel, 0.045, 0.5,
                             n_paths=10_000, seed=4)
    # Rough but robust band: blend median should sit between the two
    # central values, ±some MC noise.
    assert 250 < mc.p50 < 500


# ---------------------------------------------------------------------------
# Tornado
# ---------------------------------------------------------------------------
def test_tornado_returns_bars_in_descending_swing_order():
    target = make_bundle()
    bars = tornado_ddm(target, base_ke=0.12, base_g_terminal=0.045)
    assert len(bars) == 3
    for i in range(len(bars) - 1):
        assert bars[i].swing >= bars[i + 1].swing


def test_tornado_ke_swing_is_meaningful():
    """A ±1% Ke bump should produce a non-trivial swing on a mature payer."""
    target = make_bundle()
    bars = tornado_ddm(target, base_ke=0.12, base_g_terminal=0.045)
    ke_bar = [b for b in bars if "Cost of Equity" in b.lever][0]
    assert ke_bar.swing > 0
