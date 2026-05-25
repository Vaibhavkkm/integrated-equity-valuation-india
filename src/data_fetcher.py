"""
Data layer for the Indian valuation engine.

The job of this module is unglamorous but critical: get clean financials
and price history out of upstream providers and persist them so we don't
re-hit the wire on every run.

Design choices worth flagging:

* **Multi-source with fallback.** yfinance is the primary source because
  it is the only free, programmatic feed for NSE/BSE without KYC. When
  it returns garbage (intermittent JSON schema drift, rate-limits, ticker
  delisting glitches), we fall back through:
      1. yfinance → 2. on-disk last-known-good cache (any age) →
      3. raise a typed DataFetchError.
  The user sees a coherent error rather than a half-populated bundle.

* **Validation runs after every fetch.** A `DataQualityReport` is
  attached to the bundle so the rest of the pipeline can widen MC bands
  or downgrade confidence based on data sketchiness.

* **Cache files are pickled** keyed by ticker. TTL is 24h by default but
  the fallback path will read *any* cached copy regardless of age — a
  stale price is more useful than a NaN.

* **The `StockBundle` dataclass is the single contract** the rest of the
  code consumes. Swapping data providers later (e.g. moving to a paid
  Bloomberg/Refinitiv feed) is a single-file change.

* **Cache schema policy (shim-and-backfill).** When `StockBundle` gains
  new fields, existing on-disk pickles must keep loading. The policy is:

      1. Bump ``CACHE_SCHEMA_VERSION`` by 1 and document v_old vs v_new
         in the constant's docstring.
      2. Add the new field(s) with a sensible default (typically empty
         ``pd.Series`` for histories, ``np.nan`` for scalars).
      3. Extend ``StockBundle.__setstate__`` to backfill the new
         field(s) and stamp ``self._schema_version = CACHE_SCHEMA_VERSION``
         so the migrated bundle self-reports as up-to-date.
      4. Add a regression test in ``tests/test_pickle_backcompat.py``
         covering a synthesised pre-migration pickle.

  This policy *supersedes* the c58b4a7 "wipe caches on schema change"
  approach, which was reverted in Phase A.0 because it forced every
  user with accumulated local cache to lose history on upgrade — and
  there is no migration path back. The shim is ~10 lines per
  migration; the user cost of the alternative is much higher.
"""
from __future__ import annotations

import pickle
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "yfinance is required. Install it with `pip install yfinance`."
    ) from exc

from config import CACHE_DIR, DEFAULT_UNIVERSE
from src.exceptions import (
    DataSourceUnavailable,
    StaleCacheError,
    TickerNotFoundError,
)
from src.logging_setup import get_logger


# yfinance reports its own sector/industry taxonomy. Map it to ours so
# custom tickers (anything not in DEFAULT_UNIVERSE) still get a sensible
# peer pool instead of falling through to "Diversified" with no peers.
_YF_SECTOR_TO_OURS = {
    "Technology": "Information Technology",
    "Healthcare": "Pharma",
    "Consumer Defensive": "FMCG",
    "Communication Services": "Telecom",
    "Energy": "Oil & Gas",
    "Real Estate": "Real Estate",
    "Utilities": "Power",
    "Industrials": "Capital Goods",
    "Consumer Cyclical": "Consumer Durables",
    "Basic Materials": "Metals",
    "Financial Services": "Banking",
}

# Industry-level refinements take precedence over the broad sector map.
_YF_INDUSTRY_TO_OURS = {
    "auto manufacturers": "Auto",
    "auto parts": "Auto",
    "specialty chemicals": "Chemicals",
    "chemicals": "Chemicals",
    "agricultural inputs": "Chemicals",
    "building materials": "Cement",
    "cement": "Cement",
    "asset management": "NBFC",
    "credit services": "NBFC",
    "insurance - life": "NBFC",
    "insurance - diversified": "NBFC",
    "insurance - property & casualty": "NBFC",
    "capital markets": "NBFC",
    "financial conglomerates": "NBFC",
    "steel": "Metals",
    "aluminum": "Metals",
    "copper": "Metals",
    "other industrial metals & mining": "Metals",
    "coking coal": "Metals",
    "thermal coal": "Metals",
    "oil & gas integrated": "Oil & Gas",
    "oil & gas refining & marketing": "Oil & Gas",
    "oil & gas e&p": "Oil & Gas",
    "drug manufacturers - general": "Pharma",
    "drug manufacturers - specialty & generic": "Pharma",
    "pharmaceutical retailers": "Pharma",
    "biotechnology": "Pharma",
    "engineering & construction": "Capital Goods",
    "infrastructure operations": "Capital Goods",
    "specialty industrial machinery": "Capital Goods",
    "electrical equipment & parts": "Capital Goods",
    "real estate services": "Real Estate",
    "real estate - development": "Real Estate",
    "telecom services": "Telecom",
    "utilities - regulated electric": "Power",
    "utilities - independent power producers": "Power",
    "utilities - renewable": "Power",
    "household & personal products": "FMCG",
    "packaged foods": "FMCG",
    "tobacco": "FMCG",
    "beverages - non-alcoholic": "FMCG",
    "beverages - brewers": "FMCG",
    "consumer electronics": "Consumer Durables",
    "furnishings, fixtures & appliances": "Consumer Durables",
    "luxury goods": "Consumer Durables",
    # New-age listed internet platforms. yfinance lumps Zomato/Eternal,
    # Swiggy and Nykaa under "Consumer Cyclical / Internet Retail" — the
    # default sector route would put them with TITAN / HAVELLS / VOLTAS,
    # which has no economic basis. Override at the industry level so any
    # uncurated Internet Retail ticker still gets the right peer pool.
    "internet retail": "Internet & Platform",
    "internet content & information": "Internet & Platform",
}


def _infer_sector(info: dict) -> str:
    """Map yfinance's sector/industry to the project's sector taxonomy."""
    industry = (info.get("industry") or "").strip().lower()
    if industry and industry in _YF_INDUSTRY_TO_OURS:
        return _YF_INDUSTRY_TO_OURS[industry]
    yf_sector = (info.get("sector") or "").strip()
    return _YF_SECTOR_TO_OURS.get(yf_sector, "Diversified")

log = get_logger(__name__)


# Yahoo throttles aggressive callers; keep this conservative.
_REQUEST_GAP_SECONDS = 0.4
_DEFAULT_TTL = timedelta(hours=24)


# ---------------------------------------------------------------------------
# Cache schema versioning
# ---------------------------------------------------------------------------
# Increment this when a backward-incompatible change is made to
# ``StockBundle``. Every bump must be accompanied by a corresponding
# branch in ``StockBundle.__setstate__`` that backfills the new
# field(s). See the "Cache schema policy" section of the module
# docstring for the full procedure.
#
# Version history
# ---------------
#   v1 — Pre-Phase-A.0 schema. Lacks the six cash-flow series
#        (net_income_annual, capex_annual, dep_amort_annual,
#        change_in_wc_annual, working_capital_annual, total_debt_annual).
#        Most pickles on disk before this commit are v1.
#   v2 — Phase A.0 schema. Adds the six cash-flow series listed above.
#        New pickles produced by this module are v2.
CACHE_SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# The single object the rest of the codebase consumes
# ---------------------------------------------------------------------------
@dataclass
class StockBundle:
    """Everything the valuation models need, in one place."""

    ticker: str
    sector: str
    name: str

    # Market data
    price: float
    market_cap: float                           # in INR
    shares_outstanding: float
    beta_raw: Optional[float]                   # 5-yr monthly regression beta
    price_history: pd.Series                    # daily adj close, ~5 years

    # Per-share fundamentals (TTM)
    eps_ttm: float
    book_value_per_share: float
    sales_per_share_ttm: float
    dividend_per_share_ttm: float
    free_cash_flow_per_share_ttm: float

    # Whole-firm fundamentals (latest annual)
    revenue: float
    ebitda: float
    net_income: float
    total_debt: float
    cash: float
    equity: float
    total_assets: float

    # Multi-year series (annual, oldest → newest)
    dividends_annual: pd.Series                 # 5-10y of DPS
    earnings_annual: pd.Series                  # 5-10y of EPS
    revenue_annual: pd.Series                   # 5-10y of revenue
    book_value_annual: pd.Series

    # Derived & meta
    payout_ratio: float                         # DPS / EPS, TTM (capped at 1.0)
    roe: float
    enterprise_value: float
    # Quarterly statement series (oldest → newest, ~4-8 quarters). Default
    # to empty so older cached pickles unpickle cleanly and synthetic test
    # bundles don't need to construct them.
    quarterly_earnings: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    quarterly_revenue: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # ------------------------------------------------------------------
    # Cash-flow & balance-sheet series for FCFE (Phase A.0)
    # ------------------------------------------------------------------
    # All series are oldest → newest, indexed by fiscal period end.
    # Default to empty pd.Series so cached pickles from before this
    # field-set was added unpickle cleanly (same pattern as the
    # quarterly_* fields above).
    #
    # NB on sign conventions and units: any transform that differs
    # from the raw upstream provider is applied AT THE FETCHER BOUNDARY
    # (inside _fetch_from_yahoo) so downstream consumers can read these
    # fields naturally without worrying about Yahoo's quirks. See the
    # per-field notes below.

    # Whole-firm net income, annual (₹). yfinance row "Net Income From
    # Continuing Operations" preferred; falls back to "Net Income".
    # NOTE: earnings_annual above is per-share EPS — this is the *unit-
    # of-firm* counterpart used for cash-flow attribution.
    net_income_annual: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Capital expenditure, annual (₹). CONVENTION: stored as the
    # *absolute value* (positive). yfinance reports CapEx as a negative
    # cash outflow; we flip the sign at the fetcher boundary so the
    # natural-language reading ("FY24 CapEx was ₹X") holds. The FCFE
    # algebra in fcfe_valuation.py reads `NI - CapEx + D&A`, which only
    # works if CapEx is positive here.
    capex_annual: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Depreciation & amortisation, annual (₹). Positive values (raw
    # yfinance sign). yfinance row "Depreciation And Amortization" is
    # the combined D&A series we want; standalone "Depreciation" is the
    # non-amortisation slice and is not used.
    dep_amort_annual: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Change in working capital, annual (₹). CONVENTION: stored with
    # the raw yfinance sign — the cash-flow-statement view, where
    # *positive* = cash was released by a reduction in working-capital
    # tie-up (favourable to FCFE) and *negative* = cash was absorbed by
    # a WC build-up.
    #
    # Empirically verified (TCS FY24/FY26 and ITC FY24/FY26): adding
    # this row to NI + D&A + other-non-cash items recovers reported
    # Operating Cash Flow; subtracting it does not. See
    # ``scripts/verify_wc_sign.py`` if a future yfinance schema change
    # warrants re-validation.
    #
    # FCFE consequence for Phase A: the Damodaran textbook FCFE formula
    # reads ``FCFE = NI − Net CapEx(1−DR) − ΔWC(1−DR) + Net Borrowing``
    # using ΔWC = *increase* in WC (asset-side sign). yfinance's value
    # is the negative of that, so the formula that consumes THIS field
    # must read:
    #
    #     FCFE = NI − Net CapEx(1−DR) + change_in_wc_annual(1−DR)
    #            + Net Borrowing
    #
    # i.e. ADD the field with its raw sign — do not flip it.
    change_in_wc_annual: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Working-capital snapshot from the balance sheet, annual (₹).
    # Prefers yfinance's pre-computed "Working Capital" row; falls back
    # to `Current Assets − Current Liabilities` if that row is missing.
    # Useful as a sanity check against change_in_wc_annual and for the
    # 5y average debt-ratio computation in FCFE (Phase A).
    working_capital_annual: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Total debt snapshot from the balance sheet, annual (₹). Positive
    # values. yfinance "Total Debt" preferred; falls back to the sum of
    # "Long Term Debt" + "Current Debt" when "Total Debt" is missing.
    total_debt_annual: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "yfinance"                    # which provider supplied this
    quality_score: Optional[float] = None       # populated by validate_bundle()
    # Uncapped payout — set in __post_init__ from the original raw value.
    # Used downstream to detect over-distribution (>100% payout) where the
    # company is funding dividends from debt rather than earnings.
    payout_ratio_raw: float = field(default=float("nan"))

    # Schema version this bundle was built against. Default is 1 (the
    # pre-Phase-A.0 schema) so pickles whose __dict__ predates this
    # field self-identify as v1 after unpickle; __post_init__ and
    # __setstate__ both stamp this to CACHE_SCHEMA_VERSION so any
    # bundle that has gone through either constructor or migration
    # ends up reporting the current version. See the module-level
    # "Cache schema policy" docstring section.
    _schema_version: int = field(default=1)

    # ------------------------------------------------------------------
    # Cache-forward compatibility (shim-and-backfill policy)
    # ------------------------------------------------------------------
    # Pickle restores objects by populating ``__dict__`` directly — it
    # does NOT call ``__init__`` and does NOT apply dataclass field
    # defaults. Any field added to this class after older cache files
    # were written will be missing on unpickle, raising AttributeError
    # at runtime even though the dataclass declaration has a default.
    #
    # The project-wide policy is to migrate stale pickles forward here
    # rather than wipe them on schema change — see the "Cache schema
    # policy" section of the module docstring for the full procedure.
    # This supersedes the c58b4a7 "wipe-on-change" decision, which was
    # reverted in Phase A.0 because forcing every user to lose their
    # local cache on upgrade is a worse user experience than carrying
    # ~10 lines of migration code per schema bump.
    #
    # When CACHE_SCHEMA_VERSION is bumped, add a new branch below that
    # backfills the new field(s). The final ``self._schema_version``
    # assignment stamps the migrated bundle as current.
    def __setstate__(self, state):
        self.__dict__.update(state)

        # --- v1 → v2 backfill ------------------------------------------
        # Older fields (quarterly_* added in the earnings-momentum
        # migration, payout_ratio_raw added in the
        # unsustainable-payout-guard migration; both predate v1 but
        # the formal version field, so we backfill them here too).
        for name in ("quarterly_earnings", "quarterly_revenue"):
            cur = getattr(self, name, None)
            if cur is None or not isinstance(cur, pd.Series):
                setattr(self, name, pd.Series(dtype=float))
        if not hasattr(self, "payout_ratio_raw"):
            self.payout_ratio_raw = float("nan")
        # Phase A.0 cash-flow / balance-sheet series — the actual v1→v2
        # delta.
        for name in (
            "net_income_annual", "capex_annual", "dep_amort_annual",
            "change_in_wc_annual", "working_capital_annual", "total_debt_annual",
        ):
            cur = getattr(self, name, None)
            if cur is None or not isinstance(cur, pd.Series):
                setattr(self, name, pd.Series(dtype=float))

        # Stamp the migrated bundle with the current schema version.
        # New (freshly-constructed via __init__) bundles also pass
        # through this assignment via __post_init__ below; both code
        # paths converge on CACHE_SCHEMA_VERSION.
        self._schema_version = CACHE_SCHEMA_VERSION

    # Convenience
    def __post_init__(self):
        # Preserve the *raw* payout for downstream sustainability checks
        # (Vedanta-style firms pay 190% of EPS by financing dividends from
        # debt or asset sales — DDM must see this is unsustainable rather
        # than treat the inflated DPS as a forever-cashflow).
        raw_payout = self.payout_ratio
        if raw_payout is not None and np.isfinite(raw_payout):
            self.payout_ratio_raw = float(raw_payout)
        else:
            self.payout_ratio_raw = float("nan")

        # Cap modelling payout at 1.0 (anything higher means the company is
        # paying from reserves — treat as 100% for the sustainable-growth
        # algebra). Floor at 0 so retention math doesn't go negative.
        if self.payout_ratio is not None and self.payout_ratio > 1.0:
            self.payout_ratio = 1.0
        if self.payout_ratio is not None and self.payout_ratio < 0:
            self.payout_ratio = 0.0

        # Stamp every freshly-constructed bundle with the current schema
        # version. Pickles whose __dict__ already contains a stale
        # _schema_version are upgraded by __setstate__ at unpickle time;
        # this assignment handles the live-fetch path. Both routes
        # converge on CACHE_SCHEMA_VERSION.
        self._schema_version = CACHE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("&", "AND")
    return CACHE_DIR / f"{safe}.pkl"


def _load_from_cache(
    ticker: str,
    ttl: Optional[timedelta] = None,
) -> Optional[StockBundle]:
    """Load cached bundle; if `ttl` is None, ignore expiry (last-known-good)."""
    p = _cache_path(ticker)
    if not p.exists():
        return None
    try:
        bundle: StockBundle = pickle.loads(p.read_bytes())
        # Old caches were pickled with naive UTC timestamps; upgrade in-place
        # so age math works against the tz-aware "now".
        if bundle.fetched_at.tzinfo is None:
            bundle.fetched_at = bundle.fetched_at.replace(tzinfo=timezone.utc)
        if ttl is not None and datetime.now(timezone.utc) - bundle.fetched_at > ttl:
            return None
        return bundle
    except Exception as exc:
        log.warning("cache: corrupt file for %s, deleting (%s)", ticker, exc)
        p.unlink(missing_ok=True)
        return None


def _save_to_cache(bundle: StockBundle) -> None:
    try:
        _cache_path(bundle.ticker).write_bytes(pickle.dumps(bundle))
    except OSError as exc:
        log.warning("cache: failed to persist %s (%s)", bundle.ticker, exc)


def _reapply_curated_sector(bundle: StockBundle, override_sector: Optional[str]) -> None:
    """If the curated universe (or an explicit caller override) disagrees
    with the sector baked into a cached bundle, prefer the curated value.

    Without this, edits to ``DEFAULT_UNIVERSE`` (e.g. moving SWIGGY from
    "Consumer Durables" to "Internet & Platform") wouldn't take effect
    until every cached pickle was manually invalidated.
    """
    desired = override_sector or DEFAULT_UNIVERSE.get(bundle.ticker)
    if desired and desired != bundle.sector:
        log.info("sector: reclassifying %s in cache (%s → %s)",
                 bundle.ticker, bundle.sector, desired)
        bundle.sector = desired


# ---------------------------------------------------------------------------
# Field-extraction helpers — Yahoo's statements are inconsistent in naming
# ---------------------------------------------------------------------------
def _first_present(df: pd.DataFrame, *candidates: str) -> Optional[pd.Series]:
    """Return the first row whose label matches any candidate (case-insensitive)."""
    if df is None or df.empty:
        return None
    lowered = {str(idx).lower(): idx for idx in df.index}
    for c in candidates:
        key = c.lower()
        if key in lowered:
            return df.loc[lowered[key]]
    # Looser match: partial / contains
    for c in candidates:
        for low, orig in lowered.items():
            if c.lower() in low:
                return df.loc[orig]
    return None


def _safe_float(x, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _annual_series(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """Pull a multi-year series from a Yahoo statements DataFrame.

    Yahoo returns columns as period-end timestamps in *descending* order;
    we flip them so that the resulting Series is chronological — that's the
    direction every growth-rate function in this project expects.
    """
    row = _first_present(df, *candidates)
    if row is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(row, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index, errors="coerce")
    return s.sort_index()


# ---------------------------------------------------------------------------
# The actual fetcher (multi-source, fallback-aware)
# ---------------------------------------------------------------------------
def fetch_stock(
    ticker: str,
    sector: Optional[str] = None,
    *,
    force_refresh: bool = False,
    ttl: timedelta = _DEFAULT_TTL,
    offline: bool = False,
    validate: bool = True,
) -> StockBundle:
    """Build a StockBundle for `ticker`.

    Resolution order:
      1. Fresh cache (if not `force_refresh` and not stale) — fast path.
      2. Live fetch from yfinance.
      3. Stale cache (any age) as graceful degradation if yfinance fails.
      4. Raise `TickerNotFoundError` / `DataSourceUnavailable`.

    Parameters
    ----------
    ticker : str
        NSE ticker with `.NS` suffix (or `.BO` for BSE).
    sector : str, optional
        Override the sector mapping in config.DEFAULT_UNIVERSE.
    force_refresh : bool
        Ignore any cached copy and re-hit Yahoo.
    ttl : timedelta
        Maximum cache age before we re-fetch.
    offline : bool
        Hard-fail (StaleCacheError) if no cache is present at all.
    validate : bool
        Run `validate_bundle` on the result and attach the score.
    """
    if not ticker or not isinstance(ticker, str):
        from src.exceptions import InvalidInputError
        raise InvalidInputError("ticker must be a non-empty string")

    # Resolve sector. Order: explicit override → curated universe → yfinance
    # auto-inference (handled inside _fetch_from_yahoo, which has info).
    sector = sector or DEFAULT_UNIVERSE.get(ticker)

    # --- 1. Fresh cache ----------------------------------------------------
    if not force_refresh:
        cached = _load_from_cache(ticker, ttl=ttl)
        if cached is not None:
            log.debug("cache: hit (fresh) for %s", ticker)
            _reapply_curated_sector(cached, sector)
            if validate and cached.quality_score is None:
                _attach_quality(cached)
            return cached

    # --- 2/3. Offline branch ----------------------------------------------
    if offline:
        stale = _load_from_cache(ticker, ttl=None)
        if stale is not None:
            log.info("offline: serving stale cache for %s (age=%s)",
                     ticker, datetime.now(timezone.utc) - stale.fetched_at)
            _reapply_curated_sector(stale, sector)
            if validate and stale.quality_score is None:
                _attach_quality(stale)
            return stale
        raise StaleCacheError(ticker)

    # --- 2. Live fetch -----------------------------------------------------
    try:
        bundle = _fetch_from_yahoo(ticker, sector)
    except TickerNotFoundError:
        raise
    except Exception as exc:
        # Network blip, rate-limit, schema drift — try stale cache.
        log.warning("yahoo: live fetch failed for %s (%s); falling back to stale cache",
                    ticker, exc)
        stale = _load_from_cache(ticker, ttl=None)
        if stale is not None:
            log.info("yahoo: served stale cache (age=%s) after live failure",
                     datetime.now(timezone.utc) - stale.fetched_at)
            if validate and stale.quality_score is None:
                _attach_quality(stale)
            return stale
        raise DataSourceUnavailable(
            f"yfinance failed for {ticker} and no cache available: {exc}"
        ) from exc

    _save_to_cache(bundle)
    time.sleep(_REQUEST_GAP_SECONDS)
    if validate:
        _attach_quality(bundle)
    return bundle


def _attach_quality(bundle: StockBundle) -> None:
    """Run validation and attach a quality score; never raise here."""
    try:
        # Local import to avoid a circular dependency at module load.
        from src.data_validation import validate_bundle
        report = validate_bundle(bundle, strict=False)
        bundle.quality_score = report.score
    except Exception as exc:
        log.warning("validation: failed to score %s (%s)", bundle.ticker, exc)


def _looks_empty(info: dict, hist: pd.DataFrame) -> bool:
    """yfinance returned nothing usable for this ticker (transient or dead)."""
    no_info = (not info) or (
        info.get("regularMarketPrice") is None
        and info.get("currentPrice") is None
        and info.get("previousClose") is None
    )
    return no_info and (hist is None or hist.empty)


def _fetch_from_yahoo(ticker: str, sector: Optional[str]) -> StockBundle:
    """The actual network call. Wrapped because it's the bit that fails.

    yfinance is famously flaky — empty `info` dicts, missing schema fields,
    and 429 rate-limits are routine. We do ONE silent retry before giving
    up; if the second attempt is also empty we raise `DataSourceUnavailable`
    so the outer handler can fall through to the stale-cache path. We do
    NOT raise `TickerNotFoundError` here, because empty data from yfinance
    is rarely strong evidence of a non-existent ticker — it's almost
    always a transient API issue, and treating it as fatal makes valid
    large-caps like TATAMOTORS.NS look "not found" during yfinance
    hiccups.
    """
    yticker = yf.Ticker(ticker)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        info = _safe_info(yticker)
        hist = yticker.history(period="5y", auto_adjust=True)

        if _looks_empty(info, hist):
            log.info("yahoo: empty response for %s; retrying once", ticker)
            time.sleep(0.8)
            info = _safe_info(yticker)
            hist = yticker.history(period="5y", auto_adjust=True)

        income_stmt = _safe_statement(yticker, "income_stmt")
        balance_sheet = _safe_statement(yticker, "balance_sheet")
        cashflow_stmt = _safe_statement(yticker, "cashflow")
        quarterly_income_stmt = _safe_statement(yticker, "quarterly_income_stmt")
        dividends = yticker.dividends if hasattr(yticker, "dividends") else pd.Series(dtype=float)

    if _looks_empty(info, hist):
        # Could be a dead ticker, could be yfinance throttling.
        # Either way, surface as DataSourceUnavailable so the outer
        # `fetch_stock` handler tries the stale cache before giving up.
        raise DataSourceUnavailable(
            f"yfinance returned no usable data for '{ticker}' after retry. "
            f"This is most often a transient rate-limit or schema-drift "
            f"issue; the engine will fall back to cached data if available."
        )

    name = info.get("longName") or info.get("shortName") or ticker

    # Sector resolution: caller wins, then yfinance auto-inference, then
    # "Diversified" as last resort. This is what lets a custom ticker
    # like HGINFRA.NS land in "Capital Goods" (its real peer pool)
    # rather than the empty "Diversified" bucket.
    if not sector:
        sector = _infer_sector(info)
        if sector != "Diversified":
            log.info("sector: inferred '%s' for %s from yfinance "
                     "(sector=%s, industry=%s)",
                     sector, ticker, info.get("sector"), info.get("industry"))

    price = _safe_float(info.get("currentPrice")) or _safe_float(info.get("regularMarketPrice"))
    if not np.isfinite(price) and not hist.empty:
        price = float(hist["Close"].iloc[-1])

    shares = _safe_float(info.get("sharesOutstanding"))
    market_cap = _safe_float(info.get("marketCap"))
    if not np.isfinite(market_cap) and np.isfinite(price) and np.isfinite(shares):
        market_cap = price * shares

    # ----- Multi-year fundamentals -----
    earnings_annual = _annual_series(income_stmt, "Net Income", "Net Income Common Stockholders")
    revenue_annual = _annual_series(income_stmt, "Total Revenue", "Operating Revenue")
    # Quarterly series for earnings-momentum overlay. yfinance's quarterly
    # statements typically expose 4-8 trailing quarters; we use the same
    # _annual_series helper because the row labels are identical — only
    # the column cadence differs.
    quarterly_earnings = _annual_series(quarterly_income_stmt, "Net Income", "Net Income Common Stockholders")
    quarterly_revenue = _annual_series(quarterly_income_stmt, "Total Revenue", "Operating Revenue")
    book_equity_annual = _annual_series(balance_sheet, "Stockholders Equity", "Total Equity Gross Minority Interest")
    total_debt_series = _annual_series(balance_sheet, "Total Debt", "Long Term Debt")
    cash_series = _annual_series(balance_sheet, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")

    # ------------------------------------------------------------------
    # Cash-flow & working-capital series for FCFE (Phase A.0).
    # All transforms that diverge from the raw provider are applied
    # here so consumers see naturally-signed values. See field-level
    # docstrings on StockBundle for the conventions.
    # ------------------------------------------------------------------
    capex_annual_raw = _annual_series(
        cashflow_stmt,
        "Capital Expenditure",
        "Capital Expenditure Reported",
        "Purchase Of PPE",
    )
    # yfinance reports CapEx as a negative cash outflow; flip the sign
    # at the boundary so downstream FCFE math reads naturally.
    capex_annual = capex_annual_raw.abs() if not capex_annual_raw.empty else capex_annual_raw

    dep_amort_annual = _annual_series(
        cashflow_stmt,
        "Depreciation And Amortization",
        "Depreciation Amortization Depletion",
    )

    change_in_wc_annual = _annual_series(
        cashflow_stmt,
        "Change In Working Capital",
        "Changes In Working Capital",
    )

    net_income_annual_full = _annual_series(
        cashflow_stmt,
        "Net Income From Continuing Operations",
        "Net Income",
        "Net Income Common Stockholders",
    )
    # Fallback: if the cashflow statement lacks an NI row, derive from
    # the income statement (which we already parsed earlier into a
    # *per-share* series for EPS; here we go back to the raw whole-firm
    # statement so we have a directly comparable ₹-denominated series).
    if net_income_annual_full.empty:
        net_income_annual_full = _annual_series(
            income_stmt, "Net Income", "Net Income Common Stockholders",
        )

    # Working-capital snapshot. Pass exact labels first so the
    # _first_present() exact-match pass picks "Current Assets" before
    # the substring fallback would erroneously match "Other Current
    # Assets" / "Total Non Current Assets".
    working_capital_annual = _annual_series(balance_sheet, "Working Capital")
    if working_capital_annual.empty:
        ca = _annual_series(balance_sheet, "Current Assets", "Total Current Assets")
        cl = _annual_series(balance_sheet, "Current Liabilities", "Total Current Liabilities")
        if not ca.empty and not cl.empty:
            # Align on intersection so a single missing year doesn't
            # silently drop the whole series.
            common = ca.index.intersection(cl.index)
            if len(common) > 0:
                working_capital_annual = ca.loc[common] - cl.loc[common]

    # Total debt history. yfinance "Total Debt" preferred; reconstruct
    # from long-term + current debt rows when that row is unavailable
    # (banks and a few NBFCs in the universe sometimes lack the
    # combined row).
    total_debt_annual = _annual_series(balance_sheet, "Total Debt")
    if total_debt_annual.empty:
        ltd = _annual_series(balance_sheet, "Long Term Debt", "Long Term Debt And Capital Lease Obligation")
        std = _annual_series(balance_sheet, "Current Debt", "Current Debt And Capital Lease Obligation", "Short Term Debt")
        if not ltd.empty or not std.empty:
            # Sum on union of indices, treating missing as 0 — better
            # than dropping a year that only has one of the two.
            idx = ltd.index.union(std.index)
            total_debt_annual = (
                ltd.reindex(idx, fill_value=0.0)
                + std.reindex(idx, fill_value=0.0)
            )

    if log.isEnabledFor(10):  # DEBUG
        log.debug(
            "cashflow fields for %s: capex=%d, dep_amort=%d, ΔWC=%d, "
            "NI=%d, WC=%d, total_debt=%d",
            ticker, len(capex_annual), len(dep_amort_annual),
            len(change_in_wc_annual), len(net_income_annual_full),
            len(working_capital_annual), len(total_debt_annual),
        )

    # Dividends as annual sums (Yahoo gives ex-date payments)
    if not dividends.empty:
        # Make timezone naive so groupby works cleanly across pandas versions.
        dividends.index = pd.to_datetime(dividends.index).tz_localize(None)
        dps_annual = dividends.groupby(dividends.index.year).sum()
        dps_annual.index = pd.to_datetime(dps_annual.index.astype(str) + "-12-31")
    else:
        dps_annual = pd.Series(dtype=float)

    # Per-share annuals
    eps_annual = _per_share(earnings_annual, shares)
    bvps_annual = _per_share(book_equity_annual, shares)

    # TTM proxies (Yahoo's `info` already has these, but we sanity-check).
    eps_ttm = _safe_float(info.get("trailingEps")) if info.get("trailingEps") is not None else np.nan
    if not np.isfinite(eps_ttm) and not eps_annual.empty:
        eps_ttm = float(eps_annual.iloc[-1])

    book_value_per_share = _safe_float(info.get("bookValue"))
    if not np.isfinite(book_value_per_share) and not bvps_annual.empty:
        book_value_per_share = float(bvps_annual.iloc[-1])
    # Negative BVPS (accumulated losses, equity-erosion firms) makes P/B
    # multiples meaningless — surface as NaN so relative valuation drops
    # the multiple instead of producing a negative implied price.
    if np.isfinite(book_value_per_share) and book_value_per_share <= 0:
        book_value_per_share = np.nan

    revenue = _safe_float(info.get("totalRevenue"))
    if not np.isfinite(revenue) and not revenue_annual.empty:
        revenue = float(revenue_annual.iloc[-1])
    # Yahoo occasionally returns negative revenue from a bad statement
    # parse (return/credit-memo edge case). Treat as missing.
    if np.isfinite(revenue) and revenue < 0:
        revenue = np.nan

    fcf = _safe_float(info.get("freeCashflow"))
    fcf_per_share = fcf / shares if (np.isfinite(fcf) and np.isfinite(shares) and shares > 0) else np.nan

    dps_ttm = _safe_float(info.get("dividendRate"))
    if not np.isfinite(dps_ttm):
        if not dps_annual.empty:
            dps_ttm = float(dps_annual.iloc[-1])
        else:
            dps_ttm = 0.0

    payout_ratio = _safe_float(info.get("payoutRatio"))
    if not np.isfinite(payout_ratio):
        payout_ratio = (dps_ttm / eps_ttm) if (np.isfinite(eps_ttm) and eps_ttm > 0) else 0.0

    ebitda = _safe_float(info.get("ebitda"))
    net_income = _safe_float(info.get("netIncomeToCommon"))
    if not np.isfinite(net_income) and not earnings_annual.empty:
        net_income = float(earnings_annual.iloc[-1])

    # Sanity: yfinance occasionally returns IT-exporter revenue/EBITDA in
    # USD (HCL, Infosys) even though price/EPS/market-cap are in INR — this
    # makes net_income exceed revenue, which is mathematically impossible.
    # When that happens, fall back to the local annual statement, which is
    # consistently currency-aligned with EPS.
    if (
        np.isfinite(revenue) and np.isfinite(net_income)
        and revenue > 0 and net_income > 0
        and revenue < net_income
        and not revenue_annual.empty
    ):
        local_rev = float(revenue_annual.iloc[-1])
        if local_rev >= net_income:
            log.info("revenue: yfinance.totalRevenue (%.2e) below net income; "
                     "falling back to annual statement (%.2e) for %s",
                     revenue, local_rev, ticker)
            revenue = local_rev
        else:
            revenue = float("nan")
    if (
        np.isfinite(ebitda) and np.isfinite(net_income)
        and ebitda > 0 and net_income > 0
        and ebitda < net_income
    ):
        # EBITDA must be ≥ net income (it's pre-tax, pre-interest, pre-D&A).
        log.info("ebitda: yfinance value (%.2e) below net income; treating as unavailable for %s",
                 ebitda, ticker)
        ebitda = float("nan")

    sales_per_share = revenue / shares if (np.isfinite(revenue) and np.isfinite(shares) and shares > 0) else np.nan

    total_debt = _safe_float(info.get("totalDebt"))
    if not np.isfinite(total_debt) and not total_debt_series.empty:
        total_debt = float(total_debt_series.iloc[-1])
    if not np.isfinite(total_debt):
        total_debt = 0.0

    cash = _safe_float(info.get("totalCash"))
    if not np.isfinite(cash) and not cash_series.empty:
        cash = float(cash_series.iloc[-1])
    if not np.isfinite(cash):
        cash = 0.0

    equity = _safe_float(info.get("totalStockholderEquity"))
    if not np.isfinite(equity) and not book_equity_annual.empty:
        equity = float(book_equity_annual.iloc[-1])

    total_assets = _safe_float(info.get("totalAssets"))

    roe = _safe_float(info.get("returnOnEquity"))
    if not np.isfinite(roe) and np.isfinite(net_income) and np.isfinite(equity) and equity > 0:
        roe = net_income / equity
    # ROE computed against negative equity is mathematically defined but
    # economically nonsense — a distressed firm with -₹50cr equity and
    # ₹20cr income has ROE = -40%, which would skew clustering and
    # quality scoring. Mark as missing instead.
    if np.isfinite(roe) and np.isfinite(equity) and equity <= 0:
        roe = np.nan

    enterprise_value = _safe_float(info.get("enterpriseValue"))
    if not np.isfinite(enterprise_value) and np.isfinite(market_cap):
        enterprise_value = market_cap + total_debt - cash

    beta_raw = _safe_float(info.get("beta"))

    return StockBundle(
        ticker=ticker,
        sector=sector,
        name=name,
        price=price,
        market_cap=market_cap,
        shares_outstanding=shares,
        beta_raw=beta_raw if np.isfinite(beta_raw) else None,
        price_history=hist["Close"] if not hist.empty else pd.Series(dtype=float),
        eps_ttm=eps_ttm,
        book_value_per_share=book_value_per_share,
        sales_per_share_ttm=sales_per_share,
        dividend_per_share_ttm=dps_ttm,
        free_cash_flow_per_share_ttm=fcf_per_share,
        revenue=revenue,
        ebitda=ebitda,
        net_income=net_income,
        total_debt=total_debt,
        cash=cash,
        equity=equity,
        total_assets=total_assets,
        dividends_annual=dps_annual,
        earnings_annual=eps_annual,
        revenue_annual=revenue_annual,
        book_value_annual=bvps_annual,
        payout_ratio=payout_ratio,
        roe=roe if np.isfinite(roe) else np.nan,
        enterprise_value=enterprise_value,
        quarterly_earnings=quarterly_earnings,
        quarterly_revenue=quarterly_revenue,
        net_income_annual=net_income_annual_full,
        capex_annual=capex_annual,
        dep_amort_annual=dep_amort_annual,
        change_in_wc_annual=change_in_wc_annual,
        working_capital_annual=working_capital_annual,
        total_debt_annual=total_debt_annual,
        source="yfinance",
    )


def _per_share(series: pd.Series, shares: float) -> pd.Series:
    if series is None or series.empty or not np.isfinite(shares) or shares <= 0:
        return pd.Series(dtype=float)
    return series / shares


def _safe_info(yticker) -> dict:
    try:
        return yticker.info or {}
    except Exception as exc:
        log.debug("yahoo info() raised: %s", exc)
        return {}


def _safe_statement(yticker, attr: str) -> pd.DataFrame:
    try:
        df = getattr(yticker, attr)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception as exc:
        log.debug("yahoo %s raised: %s", attr, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Convenience: bulk-fetch a peer list (used by relative valuation)
# ---------------------------------------------------------------------------
def fetch_peers(
    tickers: list[str],
    *,
    force_refresh: bool = False,
    offline: bool = False,
) -> list[StockBundle]:
    bundles: list[StockBundle] = []
    for t in tickers:
        try:
            bundles.append(
                fetch_stock(t, force_refresh=force_refresh, offline=offline)
            )
        except Exception as exc:
            # One bad peer shouldn't sink the whole valuation.
            log.warning("peer fetch skipped: %s (%s: %s)",
                        t, exc.__class__.__name__, exc)
    return bundles
