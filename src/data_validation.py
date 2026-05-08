"""
Data validation and quality scoring for the fetched bundle.

The original data layer trusted whatever yfinance returned. That's
dangerous: a single bad ratio (e.g. ROE pulled from a year of one-off
write-offs) can cascade through DDM growth, peer clustering, and Monte
Carlo, producing a confident-looking number that's wrong.

This module sits *between* the raw fetch and the rest of the pipeline.
It does two things:

  1. Hard validation — raises typed exceptions for things that make
     valuation impossible (zero shares, missing price, etc.).

  2. Soft scoring — emits a `DataQualityReport` (0-100) that downstream
     code uses to widen Monte Carlo bands, flag the recommendation as
     LOW confidence, or annotate the PDF report with caveats.

The split matters: a stock with a 4-year EPS history isn't broken — it's
just less certain. We want to value it, but tell the user we're guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from src.exceptions import (
    ImplausibleFinancialsError,
    InsufficientHistoryError,
)
from src.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------
@dataclass
class DataQualityReport:
    """0-100 score plus the reasons for any deductions."""

    score: float                     # 0 = unusable, 100 = pristine
    completeness: float              # % of required fields present
    history_years: int               # earliest financial year available
    has_price_history: bool
    has_dividends: bool
    warnings: List[str] = field(default_factory=list)
    blocking: List[str] = field(default_factory=list)   # any => bundle unusable

    @property
    def is_usable(self) -> bool:
        return not self.blocking and self.score >= 30

    @property
    def confidence_tier(self) -> str:
        if self.score >= 75:
            return "HIGH"
        if self.score >= 50:
            return "MEDIUM"
        return "LOW"


# ---------------------------------------------------------------------------
# Hard validation — raise if these fail
# ---------------------------------------------------------------------------
_REQUIRED_SCALAR_FIELDS = (
    "price", "market_cap", "shares_outstanding",
    "eps_ttm", "book_value_per_share",
    "revenue", "equity",
)


def validate_bundle(bundle, *, strict: bool = True) -> DataQualityReport:
    """Inspect a StockBundle and return a quality report.

    Parameters
    ----------
    bundle : StockBundle
        The freshly fetched bundle.
    strict : bool
        If True, raise on blocking issues. If False, return the report
        with `blocking` populated and let the caller decide.
    """
    blocking: List[str] = []
    warnings: List[str] = []

    # 1. The non-negotiables ------------------------------------------------
    if not _is_pos_finite(getattr(bundle, "price", None)):
        blocking.append("Price is missing or non-positive.")
    if not _is_pos_finite(getattr(bundle, "shares_outstanding", None)):
        blocking.append("Shares outstanding missing or non-positive.")
    if not _is_pos_finite(getattr(bundle, "market_cap", None)):
        blocking.append("Market cap missing or non-positive.")

    # 2. Implausibility checks ---------------------------------------------
    eps = getattr(bundle, "eps_ttm", float("nan"))
    bvps = getattr(bundle, "book_value_per_share", float("nan"))
    if np.isfinite(eps) and abs(eps) > 100_000:
        warnings.append(f"EPS of {eps:,.0f} is implausibly large; check for unit error.")
    if np.isfinite(bvps) and bvps <= 0:
        warnings.append("Non-positive book value — accumulated losses or covenant breach; P/B will be skipped.")

    roe = getattr(bundle, "roe", float("nan"))
    if np.isfinite(roe) and abs(roe) > 1.5:
        warnings.append(f"ROE of {roe:.0%} is extreme; one-off items likely.")

    # Inspect the *raw* (uncapped) payout where available — the capped
    # value tops out at 1.0 and cannot reveal Vedanta-style 190% extraction.
    payout_raw = getattr(bundle, "payout_ratio_raw", float("nan"))
    payout_capped = getattr(bundle, "payout_ratio", 0.0) or 0.0
    payout = payout_raw if np.isfinite(payout_raw) else payout_capped
    if payout > 1.5:
        warnings.append(
            f"Payout ratio of {payout:.0%} indicates dividends financed "
            "from debt or reserves rather than earnings; DDM will be skipped."
        )

    # 3. History depth ------------------------------------------------------
    eps_series = getattr(bundle, "earnings_annual", pd.Series(dtype=float))
    eps_history = int(eps_series.dropna().shape[0])
    rev_series = getattr(bundle, "revenue_annual", pd.Series(dtype=float))
    rev_history = int(rev_series.dropna().shape[0])
    history_years = min(eps_history, rev_history) if rev_history else eps_history

    if history_years < 3:
        warnings.append(
            f"Only {history_years} years of fundamentals available; growth estimates are weak."
        )

    # 4. Price history ------------------------------------------------------
    ph = getattr(bundle, "price_history", pd.Series(dtype=float))
    has_price_history = ph is not None and not ph.empty and len(ph) >= 60
    if not has_price_history:
        warnings.append("Price history < 60 days — beta will fall back to sector estimate.")

    # 5. Dividend history ---------------------------------------------------
    # Require at least 2 years of *positive* DPS — a single one-off return
    # of capital (or a one-year special dividend) shouldn't qualify a name
    # as a "dividend payer" for DDM purposes.
    div_series = getattr(bundle, "dividends_annual", pd.Series(dtype=float))
    if div_series is not None and not div_series.empty:
        positive_div_years = int((div_series > 0).sum())
        has_dividends = positive_div_years >= 2
    else:
        has_dividends = False

    # 6. Completeness --------------------------------------------------------
    present = sum(
        1 for f in _REQUIRED_SCALAR_FIELDS
        if _is_finite(getattr(bundle, f, float("nan")))
    )
    completeness = present / len(_REQUIRED_SCALAR_FIELDS)

    # 7. Score --------------------------------------------------------------
    score = 100.0
    score -= (1 - completeness) * 30
    score -= max(0, 5 - history_years) * 4         # up to -20 for thin history
    score -= len(warnings) * 5
    if not has_price_history:
        score -= 10
    if not has_dividends:
        score -= 5  # not a defect for a non-payer, but a smaller scoreable signal

    score = float(max(0.0, min(100.0, score)))

    report = DataQualityReport(
        score=score,
        completeness=completeness,
        history_years=history_years,
        has_price_history=has_price_history,
        has_dividends=has_dividends,
        warnings=warnings,
        blocking=blocking,
    )

    log.debug(
        "validation: ticker=%s score=%.1f completeness=%.0f%% history=%dy warnings=%d blocking=%d",
        getattr(bundle, "ticker", "?"), score, completeness * 100,
        history_years, len(warnings), len(blocking),
    )

    if strict and blocking:
        raise ImplausibleFinancialsError(
            f"{getattr(bundle, 'ticker', '?')}: {'; '.join(blocking)}"
        )

    return report


# ---------------------------------------------------------------------------
# Public guards used at the model boundary
# ---------------------------------------------------------------------------
def require_history(series: pd.Series, *, needed: int, ticker: str, what: str) -> None:
    """Raise InsufficientHistoryError if the series is too short."""
    available = int(series.dropna().shape[0]) if series is not None else 0
    if available < needed:
        raise InsufficientHistoryError(
            ticker=ticker, needed=needed, available=available, what=what,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_finite(x) -> bool:
    try:
        return bool(np.isfinite(x))
    except (TypeError, ValueError):
        return False


def _is_pos_finite(x) -> bool:
    return _is_finite(x) and x > 0
