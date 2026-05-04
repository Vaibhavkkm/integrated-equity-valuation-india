"""
Tests for the typed exception hierarchy.

The point of typed exceptions is that callers can catch *exactly* what
they can recover from. These tests pin the inheritance tree so
refactors don't silently break that contract.
"""
from __future__ import annotations

import pytest

from src.exceptions import (
    DataFetchError, DataQualityError, DDMNotApplicableError,
    ImplausibleFinancialsError, InsufficientHistoryError,
    InvalidInputError, NoValidPeersError, StaleCacheError,
    TickerNotFoundError, ValuationEngineError, ValuationModelError,
)


# ---------------------------------------------------------------------------
# Inheritance tree
# ---------------------------------------------------------------------------
def test_all_descend_from_root():
    for cls in (
        DataFetchError, DataQualityError, ValuationModelError,
        InvalidInputError, TickerNotFoundError, StaleCacheError,
        ImplausibleFinancialsError, InsufficientHistoryError,
        DDMNotApplicableError, NoValidPeersError,
    ):
        assert issubclass(cls, ValuationEngineError), cls


def test_data_fetch_subtypes_inherit_correctly():
    assert issubclass(TickerNotFoundError, DataFetchError)
    assert issubclass(StaleCacheError, DataFetchError)


def test_data_quality_subtypes_inherit_correctly():
    assert issubclass(InsufficientHistoryError, DataQualityError)
    assert issubclass(ImplausibleFinancialsError, DataQualityError)


# ---------------------------------------------------------------------------
# Constructor messages
# ---------------------------------------------------------------------------
def test_ticker_not_found_message_includes_ticker():
    err = TickerNotFoundError("XYZ.NS")
    assert "XYZ.NS" in str(err)
    assert err.ticker == "XYZ.NS"


def test_stale_cache_message_includes_ticker():
    err = StaleCacheError("ABC.NS")
    assert "ABC.NS" in str(err)
    assert "offline" in str(err).lower()


def test_insufficient_history_message_is_informative():
    err = InsufficientHistoryError(
        ticker="X.NS", needed=5, available=2, what="EPS",
    )
    msg = str(err)
    assert "X.NS" in msg
    assert "5" in msg
    assert "2" in msg
    assert "EPS" in msg


def test_no_valid_peers_message_includes_sector():
    err = NoValidPeersError("X.NS", "Banking")
    assert "X.NS" in str(err)
    assert "Banking" in str(err)


# ---------------------------------------------------------------------------
# Catch-by-base behaviour
# ---------------------------------------------------------------------------
def test_can_catch_specific_error_via_base():
    """A caller catching DataFetchError should also see TickerNotFoundError."""
    try:
        raise TickerNotFoundError("X.NS")
    except DataFetchError as exc:
        assert isinstance(exc, TickerNotFoundError)
    else:
        pytest.fail("TickerNotFoundError did not propagate via DataFetchError")


def test_root_catches_everything():
    for ctor in (
        lambda: TickerNotFoundError("X.NS"),
        lambda: StaleCacheError("X.NS"),
        lambda: InvalidInputError("nope"),
        lambda: NoValidPeersError("X.NS", "Banking"),
    ):
        try:
            raise ctor()
        except ValuationEngineError:
            pass
        else:
            pytest.fail(f"{ctor.__name__} did not descend from ValuationEngineError")
