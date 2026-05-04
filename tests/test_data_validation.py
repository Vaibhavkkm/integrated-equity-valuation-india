"""
Tests for the data-validation gate.

These test the hard guarantees we promise to downstream models:
  * blocking issues raise (or are flagged with strict=False)
  * the score collapses for thin-history bundles
  * the confidence tier maps cleanly off the score
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_validation import (
    DataQualityReport, require_history, validate_bundle,
)
from src.exceptions import (
    ImplausibleFinancialsError, InsufficientHistoryError,
)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_validate_returns_report(mature_payer):
    report = validate_bundle(mature_payer, strict=False)
    assert isinstance(report, DataQualityReport)
    assert report.is_usable
    assert report.score >= 70
    assert report.confidence_tier in ("HIGH", "MEDIUM")
    assert not report.blocking


def test_confidence_tier_buckets_correctly():
    high = DataQualityReport(score=80, completeness=1.0, history_years=10,
                             has_price_history=True, has_dividends=True)
    med = DataQualityReport(score=60, completeness=0.8, history_years=5,
                            has_price_history=True, has_dividends=True)
    low = DataQualityReport(score=30, completeness=0.5, history_years=2,
                            has_price_history=False, has_dividends=False)
    assert high.confidence_tier == "HIGH"
    assert med.confidence_tier == "MEDIUM"
    assert low.confidence_tier == "LOW"


# ---------------------------------------------------------------------------
# Blocking conditions
# ---------------------------------------------------------------------------
def test_blocking_raises_in_strict_mode(mature_payer):
    mature_payer.price = 0.0
    with pytest.raises(ImplausibleFinancialsError):
        validate_bundle(mature_payer, strict=True)


def test_blocking_returned_softly_in_non_strict_mode(mature_payer):
    mature_payer.price = 0.0
    report = validate_bundle(mature_payer, strict=False)
    assert not report.is_usable
    assert any("Price" in b for b in report.blocking)


def test_zero_shares_blocks(mature_payer):
    mature_payer.shares_outstanding = 0
    with pytest.raises(ImplausibleFinancialsError):
        validate_bundle(mature_payer, strict=True)


# ---------------------------------------------------------------------------
# Thin history → low score
# ---------------------------------------------------------------------------
def test_thin_history_drops_score(thin_history):
    report = validate_bundle(thin_history, strict=False)
    assert report.history_years <= 2
    assert report.score < 70
    assert any("years" in w.lower() or "history" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Implausible numbers
# ---------------------------------------------------------------------------
def test_negative_book_value_warns(negative_book):
    report = validate_bundle(negative_book, strict=False)
    assert any("book" in w.lower() for w in report.warnings)


def test_implausible_eps_warns(mature_payer):
    mature_payer.eps_ttm = 1_000_000  # unit-error scale
    report = validate_bundle(mature_payer, strict=False)
    assert any("eps" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# require_history
# ---------------------------------------------------------------------------
def test_require_history_passes_when_sufficient():
    s = pd.Series([1, 2, 3, 4, 5])
    require_history(s, needed=3, ticker="X.NS", what="EPS")  # no raise


def test_require_history_raises_when_insufficient():
    s = pd.Series([1, 2])
    with pytest.raises(InsufficientHistoryError) as exc_info:
        require_history(s, needed=5, ticker="X.NS", what="EPS")
    assert "X.NS" in str(exc_info.value)
    assert "5" in str(exc_info.value)
