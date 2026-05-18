"""
Tests for the composite quality score and Piotroski sub-scorer.

`dividend_quality` and `altman_z_em` already have coverage in
test_valuation.py / test_edge_cases.py; this file fills the gap on the
0-100 composite and the F-Score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.earnings_momentum import EarningsMomentum
from src.quality_score import (
    QualityScore, altman_z_em, piotroski_score, quality_score,
)

from _factory import make_bundle


# ---------------------------------------------------------------------------
# Piotroski
# ---------------------------------------------------------------------------
def test_piotroski_high_for_healthy_firm():
    b = make_bundle()                          # default = healthy FMCG
    score, breakdown = piotroski_score(b)
    assert score >= 4                          # F ≥ 4 given the available signals
    assert breakdown["positive_net_income"] == 1
    assert breakdown["positive_fcf"] == 1


def test_piotroski_handles_loss_maker():
    b = make_bundle(net_income=-5e9,
                    free_cash_flow_per_share_ttm=-2.0)
    score, breakdown = piotroski_score(b)
    assert breakdown["positive_net_income"] == 0
    assert breakdown["positive_fcf"] == 0
    assert 0 <= score <= 9


def test_piotroski_abstains_on_missing_signals():
    """Where Yahoo data can't compute a signal, the score must neither
    crash nor inflate — the breakdown carries `—` and the point isn't
    awarded."""
    b = make_bundle(total_assets=float("nan"))
    score, breakdown = piotroski_score(b)
    assert breakdown["improving_roa"] == "—"
    assert score <= 9


# ---------------------------------------------------------------------------
# Altman Z'' — basic sanity (deeper coverage lives in test_edge_cases)
# ---------------------------------------------------------------------------
def test_altman_z_finite_for_healthy_firm():
    z = altman_z_em(make_bundle())
    assert np.isfinite(z)


def test_altman_z_nan_when_total_assets_missing():
    z = altman_z_em(make_bundle(total_assets=float("nan")))
    assert np.isnan(z)


# ---------------------------------------------------------------------------
# Composite quality_score
# ---------------------------------------------------------------------------
def test_composite_in_zero_hundred_range():
    q = quality_score(make_bundle())
    assert 0.0 <= q.composite <= 100.0
    assert isinstance(q, QualityScore)


def test_composite_higher_with_strong_momentum():
    """The momentum overlay must move the composite — that's its whole job."""
    b = make_bundle()
    strong = EarningsMomentum(score=9, direction="IMPROVING")
    weak = EarningsMomentum(score=1, direction="DETERIORATING")
    q_strong = quality_score(b, momentum=strong)
    q_weak = quality_score(b, momentum=weak)
    assert q_strong.composite > q_weak.composite


def test_composite_breakdown_carries_all_pillars():
    q = quality_score(make_bundle())
    assert set(q.breakdown.keys()) >= {
        "piotroski_components", "altman_z_value",
        "dividend_components", "momentum_components",
    }


def test_composite_missing_momentum_neutral():
    """Omitting momentum must contribute its mid-point — not penalise
    the firm with a zero momentum slice."""
    b = make_bundle()
    q_none = quality_score(b)
    q_mid = quality_score(b, momentum=EarningsMomentum(score=5, direction="FLAT"))
    # Within rounding, they should match.
    assert abs(q_none.composite - q_mid.composite) < 1.0


def test_composite_non_payer_loses_dividend_pillar():
    """A non-payer must score lower on the dividend pillar than a payer,
    holding everything else equal."""
    payer = make_bundle()
    non_payer = make_bundle(
        dividend_per_share_ttm=0.0,
        dividends_annual=pd.Series(dtype=float),
        payout_ratio=0.0,
    )
    q_p = quality_score(payer)
    q_np = quality_score(non_payer)
    assert q_np.dividend_quality == 0
    assert q_np.composite < q_p.composite
