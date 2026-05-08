"""
Tests for the earnings-momentum overlay.

Covers:
  * Strong YoY profit growth → high score, IMPROVING direction.
  * Insufficient quarterly history → neutral score, FLAT direction.
  * Peer comparison flips when target outperforms vs trails the cluster.
  * The score plumbs through quality_score and changes the composite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import pytest

from _factory import make_bundle
from src.earnings_momentum import earnings_momentum
from src.quality_score import quality_score


def _quarter_index(n: int, end="2026-03-31") -> pd.DatetimeIndex:
    return pd.date_range(end=end, periods=n, freq="QE")


@dataclass
class _FakePeerSet:
    peers: List


# ---------------------------------------------------------------------------
# Basic computations
# ---------------------------------------------------------------------------
def test_strong_yoy_growth_scores_high():
    """8 quarters trending up → score should be near max with consistent peer outperformance."""
    qe = pd.Series(
        [100, 110, 120, 130, 140, 155, 170, 190],   # ~30-50% YoY
        index=_quarter_index(8),
    )
    qr = pd.Series(
        [1000, 1050, 1100, 1150, 1200, 1280, 1350, 1450],
        index=_quarter_index(8),
    )
    target = make_bundle(quarterly_earnings=qe, quarterly_revenue=qr)

    # Peers: flat earnings, so target wildly outperforms.
    peer = make_bundle(
        ticker="PEER.NS",
        quarterly_earnings=pd.Series([100] * 8, index=_quarter_index(8)),
        quarterly_revenue=pd.Series([1000] * 8, index=_quarter_index(8)),
    )

    m = earnings_momentum(target, _FakePeerSet([peer]))

    assert m.score >= 8, f"expected high score, got {m.score} ({m.breakdown})"
    assert m.direction == "IMPROVING"
    assert m.yoy_profit_growth_q > 0.30
    assert m.consecutive_profit_growth_quarters >= 4


def test_insufficient_history_returns_neutral():
    """Fewer than 5 quarters → can't compute YoY; returns score=0 + reason."""
    qe = pd.Series([100, 105, 110], index=_quarter_index(3))
    target = make_bundle(quarterly_earnings=qe, quarterly_revenue=qe)

    m = earnings_momentum(target, _FakePeerSet([]))

    assert m.score == 0
    assert m.direction == "FLAT"
    assert "insufficient" in m.breakdown["reason"].lower()
    assert not np.isfinite(m.yoy_profit_growth_q)


def test_peer_outperformance_flips_signal():
    """Same target growth, but peer median above vs below it changes the beats_peer flag."""
    qe = pd.Series(
        [100, 105, 110, 115, 120, 125, 130, 135],   # modest ~20% YoY
        index=_quarter_index(8),
    )
    qr = pd.Series([1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350],
                   index=_quarter_index(8))
    target = make_bundle(quarterly_earnings=qe, quarterly_revenue=qr)

    # Slow peer (target wins): peer flat.
    slow_peer = make_bundle(
        ticker="SLOW.NS",
        quarterly_earnings=pd.Series([100] * 8, index=_quarter_index(8)),
        quarterly_revenue=pd.Series([1000] * 8, index=_quarter_index(8)),
    )
    # Fast peer (target loses): peer doubling.
    fast_peer = make_bundle(
        ticker="FAST.NS",
        quarterly_earnings=pd.Series(np.linspace(100, 250, 8), index=_quarter_index(8)),
        quarterly_revenue=pd.Series(np.linspace(1000, 2500, 8), index=_quarter_index(8)),
    )

    m_win = earnings_momentum(target, _FakePeerSet([slow_peer]))
    m_lose = earnings_momentum(target, _FakePeerSet([fast_peer]))

    assert m_win.breakdown["beats_peer_profit"] == 2
    assert m_lose.breakdown["beats_peer_profit"] == 0
    assert m_win.score > m_lose.score


def test_no_peers_still_computes_target_metrics():
    """When peer set is empty, target metrics still fill in; peer flags abstain."""
    qe = pd.Series(np.linspace(100, 140, 8), index=_quarter_index(8))
    qr = pd.Series(np.linspace(1000, 1300, 8), index=_quarter_index(8))
    target = make_bundle(quarterly_earnings=qe, quarterly_revenue=qr)

    m = earnings_momentum(target, _FakePeerSet([]))

    assert np.isfinite(m.yoy_profit_growth_q)
    assert m.breakdown["beats_peer_profit"] == "—"
    assert m.breakdown["beats_peer_revenue"] == "—"
    assert m.n_peers_with_data == 0


# ---------------------------------------------------------------------------
# Integration with quality_score
# ---------------------------------------------------------------------------
def test_quality_score_lifts_with_strong_momentum(mature_payer):
    """Composite should rise when momentum is strong vs when it's absent.

    Both calls compute Piotroski/Altman/dividend identically, so the only
    moving piece is the momentum pillar. The 'no momentum' baseline gets
    the abstain mid-point (10 pp); a strong (10/10) momentum gets the
    full 20 pp — so the lift is bounded by ~10 pp.
    """
    qe = pd.Series(np.linspace(100, 200, 8), index=_quarter_index(8))
    qr = pd.Series(np.linspace(1000, 2000, 8), index=_quarter_index(8))
    target_with_q = make_bundle(quarterly_earnings=qe, quarterly_revenue=qr)

    strong_m = earnings_momentum(target_with_q, _FakePeerSet([]))

    q_baseline = quality_score(mature_payer)              # no momentum
    q_strong = quality_score(target_with_q, momentum=strong_m)

    assert q_strong.composite > q_baseline.composite
    assert q_strong.earnings_momentum > 0
    assert q_strong.momentum_detail is strong_m


def test_quality_score_handles_missing_momentum(mature_payer):
    """Backward-compat: callers that don't pass momentum still work."""
    q = quality_score(mature_payer)
    assert q.earnings_momentum == 0
    assert q.momentum_detail is None
    assert 0 <= q.composite <= 100
