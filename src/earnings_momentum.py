"""
Earnings-momentum overlay — does the firm's recent quarterly trend
support the long-run intrinsic-value thesis, and how does it stack up
against the K-Means peer set?

Why this isn't a third valuation track
--------------------------------------
The DDM and Relative tracks both estimate *intrinsic value* (long-horizon).
Earnings momentum is a *short-horizon* signal — useful, well-documented
(post-earnings-announcement drift, Bernard & Thomas 1989), but mixing
horizons inside the price target itself would muddle the headline 50/50
DDM/Relative methodology this project commits to.

So we expose momentum as a 0-10 score that flows into the existing
quality composite (which already modulates blend weight and confidence)
rather than the price target. A firm posting four quarters of profit
growth above its peer median nudges DDM weight up and confidence up; one
missing peer growth nudges confidence down. The price *target* moves
only through the Bayes-shrunk growth input that's already in DDM.

Signals
-------
We score five sub-signals (each 0-2 points, capped at 10):

  1. Latest quarter YoY net-profit growth         (positive → +2; >peer median → +2)
  2. Latest quarter YoY revenue growth            (positive → +1)
  3. QoQ profit growth                            (positive → +1)
  4. Consecutive quarters of positive YoY profit  (≥4 → +2; ≥2 → +1)
  5. Profit beats peer median                     (folded into #1)

Peers are reused from the K-Means cluster the relative-valuation track
already builds — no extra clustering work, and the comparison set is
guaranteed to match what the multiples were taken from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data_fetcher import StockBundle
from src.peer_identification import PeerSet


@dataclass
class EarningsMomentum:
    # Target — latest quarter
    yoy_profit_growth_q: float = float("nan")        # latest quarter NI vs same quarter prior year
    yoy_revenue_growth_q: float = float("nan")
    qoq_profit_growth: float = float("nan")          # latest quarter vs immediately prior quarter
    qoq_revenue_growth: float = float("nan")
    consecutive_profit_growth_quarters: int = 0      # # of trailing quarters with YoY NI growth > 0

    # Peer comparison
    peer_median_yoy_profit_growth: float = float("nan")
    peer_median_yoy_revenue_growth: float = float("nan")
    n_peers_with_data: int = 0

    # Verdict
    score: int = 0                                   # 0-10 composite
    direction: str = "FLAT"                          # IMPROVING / FLAT / DETERIORATING
    breakdown: dict = field(default_factory=dict)

    def summary(self) -> str:
        if not np.isfinite(self.yoy_profit_growth_q):
            return "Earnings momentum: insufficient quarterly data"

        peer_str = (
            f"peer median {self.peer_median_yoy_profit_growth:+.1%}"
            if np.isfinite(self.peer_median_yoy_profit_growth)
            else "no peer data"
        )
        verdict = (
            "outperforming peers"
            if (
                np.isfinite(self.peer_median_yoy_profit_growth)
                and self.yoy_profit_growth_q > self.peer_median_yoy_profit_growth
            )
            else "trailing peers"
            if np.isfinite(self.peer_median_yoy_profit_growth)
            else self.direction.lower()
        )
        return (
            f"Earnings momentum: {self.yoy_profit_growth_q:+.1%} YoY profit "
            f"({peer_str}) — {verdict} | score {self.score}/10"
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _yoy_growth(q: pd.Series, lag: int = 4) -> float:
    """Year-on-year growth between the latest value and the value `lag` rows back.

    Uses ratio-form so a base of zero or negative returns NaN rather than
    producing a meaningless infinite or sign-flipped percentage.
    """
    s = q.dropna()
    if len(s) <= lag:
        return float("nan")
    prev = float(s.iloc[-lag - 1])
    curr = float(s.iloc[-1])
    if prev <= 0:
        return float("nan")
    return curr / prev - 1.0


def _consecutive_yoy_positive(q: pd.Series, lag: int = 4) -> int:
    """Count trailing quarters where YoY (vs `lag` quarters back) is positive."""
    s = q.dropna()
    if len(s) <= lag:
        return 0
    count = 0
    for i in range(len(s) - 1, lag - 1, -1):
        prev = float(s.iloc[i - lag])
        curr = float(s.iloc[i])
        if prev > 0 and curr > prev:
            count += 1
        else:
            break
    return count


def _peer_median_yoy(peers: list[StockBundle], attr: str) -> tuple[float, int]:
    """Median YoY growth across peers for the named quarterly series.

    Returns ``(median, n_with_data)``; median is NaN when no peer has
    enough quarterly history to compute YoY.
    """
    growths: list[float] = []
    for p in peers:
        s = getattr(p, attr, None)
        if s is None or s.empty:
            continue
        g = _yoy_growth(s)
        if np.isfinite(g):
            growths.append(g)
    if not growths:
        return float("nan"), 0
    return float(np.median(growths)), len(growths)


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------
def earnings_momentum(
    target: StockBundle,
    peers: Optional[PeerSet] = None,
) -> EarningsMomentum:
    """Compute the earnings-momentum overlay for ``target``.

    Parameters
    ----------
    target : StockBundle
        Firm being valued. Reads ``quarterly_earnings`` and
        ``quarterly_revenue`` (oldest → newest).
    peers : PeerSet, optional
        K-Means peer set from ``peer_identification.find_peers``. When
        omitted (or empty), the score still computes but peer-comparison
        signals abstain — the score caps lower as a result.

    Returns
    -------
    EarningsMomentum
        Container with growth metrics, peer comparison, and a 0-10 score
        that downstream code folds into the quality composite.
    """
    qe = target.quarterly_earnings if target.quarterly_earnings is not None else pd.Series(dtype=float)
    qr = target.quarterly_revenue if target.quarterly_revenue is not None else pd.Series(dtype=float)

    # If we don't have at least one full year of quarterly history,
    # return a neutral result rather than fake confidence.
    if len(qe.dropna()) < 5 and len(qr.dropna()) < 5:
        return EarningsMomentum(
            score=0,
            direction="FLAT",
            breakdown={"reason": "insufficient quarterly history (need ≥5 quarters)"},
        )

    yoy_p = _yoy_growth(qe)
    yoy_r = _yoy_growth(qr)
    qoq_p = _yoy_growth(qe, lag=1)
    qoq_r = _yoy_growth(qr, lag=1)
    streak = _consecutive_yoy_positive(qe)

    # Peer medians
    peer_list = peers.peers if peers is not None else []
    peer_yoy_p, n_peers_p = _peer_median_yoy(peer_list, "quarterly_earnings")
    peer_yoy_r, _ = _peer_median_yoy(peer_list, "quarterly_revenue")

    # ---- Scoring (each capped, sum capped at 10) ----
    breakdown: dict = {}
    score = 0

    # 1. YoY profit growth — magnitude
    if np.isfinite(yoy_p):
        if yoy_p > 0.10:
            score += 2; breakdown["yoy_profit_growth"] = 2
        elif yoy_p > 0:
            score += 1; breakdown["yoy_profit_growth"] = 1
        else:
            breakdown["yoy_profit_growth"] = 0
    else:
        breakdown["yoy_profit_growth"] = "—"

    # 2. YoY revenue growth (lower weight; revenue is easier to grow than profit)
    if np.isfinite(yoy_r):
        if yoy_r > 0.05:
            score += 1; breakdown["yoy_revenue_growth"] = 1
        else:
            breakdown["yoy_revenue_growth"] = 0
    else:
        breakdown["yoy_revenue_growth"] = "—"

    # 3. Beats peer median on profit
    if np.isfinite(yoy_p) and np.isfinite(peer_yoy_p):
        if yoy_p > peer_yoy_p:
            score += 2; breakdown["beats_peer_profit"] = 2
        else:
            breakdown["beats_peer_profit"] = 0
    else:
        breakdown["beats_peer_profit"] = "—"

    # 4. Beats peer median on revenue
    if np.isfinite(yoy_r) and np.isfinite(peer_yoy_r):
        if yoy_r > peer_yoy_r:
            score += 1; breakdown["beats_peer_revenue"] = 1
        else:
            breakdown["beats_peer_revenue"] = 0
    else:
        breakdown["beats_peer_revenue"] = "—"

    # 5. Consecutive growth quarters — momentum durability
    if streak >= 4:
        score += 2; breakdown["streak"] = 2
    elif streak >= 2:
        score += 1; breakdown["streak"] = 1
    else:
        breakdown["streak"] = 0

    # 6. QoQ profit positive — sequential trend
    if np.isfinite(qoq_p) and qoq_p > 0:
        score += 1; breakdown["qoq_profit"] = 1
    else:
        breakdown["qoq_profit"] = 0 if np.isfinite(qoq_p) else "—"

    score = int(min(10, score))

    # Direction label — orthogonal to the numeric score, used in summary text.
    if score >= 7:
        direction = "IMPROVING"
    elif score <= 3:
        direction = "DETERIORATING"
    else:
        direction = "FLAT"

    return EarningsMomentum(
        yoy_profit_growth_q=yoy_p,
        yoy_revenue_growth_q=yoy_r,
        qoq_profit_growth=qoq_p,
        qoq_revenue_growth=qoq_r,
        consecutive_profit_growth_quarters=streak,
        peer_median_yoy_profit_growth=peer_yoy_p,
        peer_median_yoy_revenue_growth=peer_yoy_r,
        n_peers_with_data=n_peers_p,
        score=score,
        direction=direction,
        breakdown=breakdown,
    )
