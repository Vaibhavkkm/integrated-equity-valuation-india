"""
Reverse DCF — solve for the growth rate the market is *already* pricing in.

Why this matters
----------------
Standard valuation asks: "given my assumptions, what is this stock worth?"
Reverse DCF flips the question: "given the price the market has set, what
growth rate would I need to believe in to justify it?"

This is the framing championed by Mauboussin (Credit Suisse, *Expectations
Investing*) and Damodaran. It is a sharper way to challenge a buy thesis:
instead of arguing about whose fair-value estimate is right, you ask
whether the *implied* growth rate is plausible given history, sector
ceilings, and competitive dynamics.

For a SEM-4 project, including reverse DCF is a meaningful step beyond
the usual textbook DDM/relative blend. It is also defensively useful in
the report — every BUY recommendation now comes with: "and the market is
pricing in only X% growth, while the firm has compounded at Y%."

What this module computes
-------------------------
Given current price P, dividend D₀ (or earnings × payout proxy), cost of
equity Ke, and a fixed terminal growth g_T, we solve for g_high (the
high-stage growth rate) that makes the two-stage DDM equal to P.

If a real solution exists in [-5%, 30%], we return it. Otherwise we
return a verdict explaining why the market price can't be reconciled
with the chosen model — usually because P >> Gordon ceiling at g_T,
which itself is informative ("the price implies growth above the
terminal cap, suggesting the market expects a sustained high-growth
regime").

Method
------
We use Brent's method (`scipy.optimize.brentq`) on the function
    f(g) = two_stage_ddm_value(g) − P
which is monotonically increasing in g over the relevant domain, so
bracketing → root is guaranteed when one exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.optimize import brentq
except ImportError:  # pragma: no cover
    brentq = None  # we'll fall back to bisection

from src.ddm_models import two_stage_ddm
from src.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class ImpliedExpectations:
    """Output of the reverse-DCF solve."""

    implied_growth: Optional[float]   # g_high that prices ≡ market; None if no solution
    market_price: float
    historical_growth: Optional[float]  # for comparison
    expectation_gap: Optional[float]    # historical − implied; positive = implied is undemanding
    verdict: str                         # human-readable interpretation
    valid: bool

    def summary(self) -> str:
        if not self.valid or self.implied_growth is None:
            return f"Reverse DCF: NO SOLUTION — {self.verdict}"
        ig = self.implied_growth
        line = f"Reverse DCF: market is pricing in g = {ig:.2%}"
        if self.historical_growth is not None and np.isfinite(self.historical_growth):
            line += f" (vs historical {self.historical_growth:.2%})"
        line += f" — {self.verdict}"
        return line


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------
def implied_growth(
    market_price: float,
    *,
    d0: float,
    ke: float,
    g_terminal: float,
    high_growth_years: int = 5,
    historical_growth: Optional[float] = None,
    g_low: float = -0.05,
    g_high: float = 0.30,
) -> ImpliedExpectations:
    """Solve the two-stage DDM for the high-stage growth rate that yields `market_price`.

    Parameters
    ----------
    market_price : float
        Current traded price.
    d0 : float
        Last paid dividend per share. Must be positive — non-payers
        cannot be valued by reverse DDM (use FCFE-DCF instead).
    ke : float
        Cost of equity (decimal).
    g_terminal : float
        Terminal-stage growth rate (decimal). Must be < ke.
    high_growth_years : int
        Length of the explicit high-stage period.
    historical_growth : float, optional
        Prior dividend or earnings CAGR — for the gap field.
    g_low, g_high : float
        Search bracket. Defaults span -5% → 30%.
    """
    if market_price <= 0 or not np.isfinite(market_price):
        return _no_solution(market_price, "Market price is non-positive.", historical_growth)
    if d0 <= 0 or not np.isfinite(d0):
        return _no_solution(
            market_price,
            "Firm pays no dividend — reverse DDM is not applicable. "
            "Use the FCFE form for non-payers.",
            historical_growth,
        )
    if g_terminal >= ke:
        return _no_solution(
            market_price,
            "Terminal growth ≥ cost of equity — model degenerate.",
            historical_growth,
        )

    def f(g: float) -> float:
        iv = two_stage_ddm(d0=d0, g_high=g, g_terminal=g_terminal,
                           ke=ke, n_years=high_growth_years)
        if not iv.valid:
            return float("nan")
        return iv.value_per_share - market_price

    f_lo, f_hi = f(g_low), f(g_high)

    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return _no_solution(
            market_price,
            "DDM produced NaN at the search bounds — likely a Ke/g_terminal collision.",
            historical_growth,
        )

    # The DDM is monotonically increasing in g, so f changes sign
    # iff a real solution lies in the bracket.
    if f_lo > 0:
        return _no_solution(
            market_price,
            f"Even at g={g_low:.0%}, model fair value exceeds market price. "
            f"Stock looks deeply oversold — implied g is below {g_low:.0%}.",
            historical_growth,
        )
    if f_hi < 0:
        return _no_solution(
            market_price,
            f"Even at g={g_high:.0%}, model fair value is below market price. "
            f"Market is pricing growth above the search ceiling — extraordinary expectations.",
            historical_growth,
        )

    # Solve.
    if brentq is not None:
        try:
            root = float(brentq(f, g_low, g_high, xtol=1e-5))
        except Exception as exc:  # pragma: no cover
            log.warning("brentq failed (%s); using bisection fallback", exc)
            root = _bisect(f, g_low, g_high)
    else:
        root = _bisect(f, g_low, g_high)

    gap = (historical_growth - root) if historical_growth is not None else None
    verdict = _verdict(root, historical_growth, g_terminal)

    return ImpliedExpectations(
        implied_growth=root,
        market_price=market_price,
        historical_growth=historical_growth,
        expectation_gap=gap,
        verdict=verdict,
        valid=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bisect(f, lo: float, hi: float, *, n: int = 80) -> float:
    """Plain bisection — used only if scipy is unavailable."""
    f_lo = f(lo)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def _verdict(g_implied: float, g_hist: Optional[float], g_terminal: float) -> str:
    """Human-readable interpretation of the implied vs historical gap."""
    if g_hist is None or not np.isfinite(g_hist):
        if g_implied < g_terminal:
            return "implied growth is below the terminal cap — modest expectations baked in."
        if g_implied < 0.10:
            return "modest growth assumption embedded — limited downside if execution is steady."
        if g_implied < 0.20:
            return "double-digit growth expected — needs continued operational momentum."
        return "very aggressive growth priced in — execution risk is elevated."

    gap = g_hist - g_implied
    if gap > 0.05:
        return (
            f"implied growth is {gap*100:.1f} pp BELOW historical — "
            f"market expectations look conservative; possible mispricing on the upside."
        )
    if gap > 0.0:
        return (
            f"implied growth is mildly below historical; market is roughly aligned with the past."
        )
    if gap > -0.05:
        return (
            f"implied growth modestly above historical — market expects mean-reversion-up."
        )
    return (
        f"implied growth is {abs(gap)*100:.1f} pp ABOVE historical — "
        f"market is paying for an acceleration the firm has not yet delivered."
    )


def _no_solution(price: float, reason: str, hist: Optional[float]) -> ImpliedExpectations:
    return ImpliedExpectations(
        implied_growth=None,
        market_price=price,
        historical_growth=hist,
        expectation_gap=None,
        verdict=reason,
        valid=False,
    )
