"""
Typed exception hierarchy for the valuation engine.

Why typed exceptions instead of bare `Exception`?

  * Callers can catch only what they can handle. A failing price fetch is
    recoverable (use cache); a malformed ticker is not.
  * The Streamlit UI and the CLI render different error messages depending
    on the exception class — without typing, every failure looks the same.
  * Tests can assert that the *right* error is raised, not just that
    *some* error is raised.

Hierarchy
---------
    ValuationEngineError                   (root — never raised directly)
        │
        ├── DataFetchError                 (anything wrong with sourcing)
        │       ├── TickerNotFoundError    (yfinance returned nothing)
        │       ├── DataSourceUnavailable  (network / rate-limit)
        │       └── StaleCacheError        (offline + cache expired)
        │
        ├── DataQualityError               (data fetched but unusable)
        │       ├── InsufficientHistoryError
        │       └── ImplausibleFinancialsError
        │
        ├── ValuationModelError            (math / model logic)
        │       ├── DDMNotApplicableError
        │       └── NoValidPeersError
        │
        └── InvalidInputError              (caller's fault)
"""
from __future__ import annotations


class ValuationEngineError(Exception):
    """Root exception for the engine. Never raise this directly."""


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------
class DataFetchError(ValuationEngineError):
    """Generic failure to source data from an upstream provider."""


class TickerNotFoundError(DataFetchError):
    """Provider returned no rows for the requested ticker."""

    def __init__(self, ticker: str, source: str = "yfinance"):
        self.ticker = ticker
        self.source = source
        super().__init__(
            f"Ticker '{ticker}' not found on {source}. "
            f"Check the suffix (.NS for NSE, .BO for BSE)."
        )


class DataSourceUnavailable(DataFetchError):
    """Network failure, rate-limit, or upstream 5xx."""


class StaleCacheError(DataFetchError):
    """`offline=True` requested but the cached copy has expired."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        super().__init__(
            f"No fresh cache for '{ticker}' and offline mode is on. "
            f"Re-run without --offline to refresh."
        )


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
class DataQualityError(ValuationEngineError):
    """Data was fetched but is too thin / implausible to value the firm."""


class InsufficientHistoryError(DataQualityError):
    """Not enough years of financial history to run the chosen model."""

    def __init__(self, ticker: str, needed: int, available: int, what: str):
        self.ticker = ticker
        super().__init__(
            f"{ticker}: need at least {needed} years of {what}, "
            f"only {available} available."
        )


class ImplausibleFinancialsError(DataQualityError):
    """Numbers fail a sanity check (negative shares, market_cap of zero, etc.)."""


# ---------------------------------------------------------------------------
# Model layer
# ---------------------------------------------------------------------------
class ValuationModelError(ValuationEngineError):
    """Math or model logic failure."""


class DDMNotApplicableError(ValuationModelError):
    """Firm is a non-payer or otherwise outside DDM's domain."""


class NoValidPeersError(ValuationModelError):
    """Couldn't assemble a peer set with at least `min_peers` members."""

    def __init__(self, ticker: str, sector: str):
        self.ticker = ticker
        self.sector = sector
        super().__init__(
            f"No usable peer set for {ticker} (sector={sector}). "
            f"Relative valuation is unavailable; DDM may still produce a number."
        )


# ---------------------------------------------------------------------------
# Caller errors
# ---------------------------------------------------------------------------
class InvalidInputError(ValuationEngineError):
    """The caller passed something nonsensical."""
