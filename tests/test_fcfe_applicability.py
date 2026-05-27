"""
Phase A — fcfe_applicable() boundary tests.

Covers each False branch in ``fcfe_applicable`` with a synthetic
fixture so the rejection reasons stay locked even as the engine
gains complexity. Each case asserts on the exact reason string —
the UI surfaces this verbatim to the user, so a typo or rewording
is a user-visible regression.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fcfe_valuation import fcfe_applicable, fcfe_value
from tests._factory import _annual_index, make_bundle, make_sparse_history_bundle


def _make_fcfe_ready_bundle(**overrides):
    """A bundle that passes every applicability check by default.
    Tests override the field they want to break."""
    n = 5
    idx = _annual_index(n)
    flat = lambda v: pd.Series([float(v)] * n, index=idx)
    growing_eps = pd.Series(
        [10.0 * (1.08) ** k for k in range(n)], index=idx,
    )
    base = dict(
        sector="Capital Goods",
        net_income_annual=flat(1_000.0),
        capex_annual=flat(200.0),
        dep_amort_annual=flat(150.0),
        change_in_wc_annual=flat(-50.0),
        total_debt_annual=flat(500.0),
        working_capital_annual=flat(800.0),
        earnings_annual=growing_eps,
        eps_ttm=float(growing_eps.iloc[-1]),
        net_income=1_000.0,
        total_debt=500.0,
        equity=4_000.0,
    )
    base.update(overrides)
    return make_bundle(**base)


# ---------------------------------------------------------------------------
# Happy path — confirm the fixture itself is FCFE-applicable
# ---------------------------------------------------------------------------
def test_baseline_bundle_is_applicable():
    bundle = _make_fcfe_ready_bundle()
    ok, reason = fcfe_applicable(bundle)
    assert ok and reason is None


# ---------------------------------------------------------------------------
# Directive #2: financials excluded regardless of data populating
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sector", ["Banking", "NBFC", "Insurance"])
def test_financials_excluded_even_with_full_data(sector):
    """The most important applicability case: a bank or insurer whose
    cash-flow fields happen to populate must STILL be rejected. The
    exclusion belongs at the model layer."""
    bundle = _make_fcfe_ready_bundle(sector=sector)
    ok, reason = fcfe_applicable(bundle)
    assert not ok
    assert reason == "FCFE not applicable to financial firms"


# ---------------------------------------------------------------------------
# Directive #1: <3y history floors
# ---------------------------------------------------------------------------
def test_sparse_capex_history_rejected():
    bundle = _make_fcfe_ready_bundle(
        capex_annual=pd.Series([100.0, 110.0], index=_annual_index(2)),
    )
    ok, reason = fcfe_applicable(bundle)
    assert not ok
    assert "Insufficient capex history" in reason


def test_sparse_dep_amort_history_rejected():
    bundle = _make_fcfe_ready_bundle(
        dep_amort_annual=pd.Series([90.0, 95.0], index=_annual_index(2)),
    )
    ok, reason = fcfe_applicable(bundle)
    assert not ok
    assert "D&A history" in reason


def test_sparse_ni_history_rejected():
    bundle = _make_fcfe_ready_bundle(
        net_income_annual=pd.Series([1_000.0, 1_100.0], index=_annual_index(2)),
    )
    ok, reason = fcfe_applicable(bundle)
    assert not ok
    assert "NI history" in reason


# ---------------------------------------------------------------------------
# Negative-NI and EPS-CAGR guards
# ---------------------------------------------------------------------------
def test_negative_average_ni_rejected():
    n = 5
    idx = _annual_index(n)
    bundle = _make_fcfe_ready_bundle(
        net_income_annual=pd.Series([-100.0] * n, index=idx),
    )
    ok, reason = fcfe_applicable(bundle)
    assert not ok
    assert "Average net income non-positive" in reason


def test_negative_eps_cagr_rejected():
    n = 5
    idx = _annual_index(n)
    # EPS declines monotonically from 20 to 5 — negative CAGR
    bundle = _make_fcfe_ready_bundle(
        earnings_annual=pd.Series([20.0, 17.0, 14.0, 11.0, 5.0], index=idx),
        eps_ttm=5.0,
    )
    ok, reason = fcfe_applicable(bundle)
    assert not ok
    assert "EPS CAGR non-positive" in reason


def test_eps_starting_at_zero_undefined_cagr():
    n = 5
    idx = _annual_index(n)
    bundle = _make_fcfe_ready_bundle(
        earnings_annual=pd.Series([0.0, 5.0, 10.0, 15.0, 20.0], index=idx),
        eps_ttm=20.0,
    )
    ok, reason = fcfe_applicable(bundle)
    assert not ok
    assert "undefined" in reason.lower()


# ---------------------------------------------------------------------------
# All-zero / missing series
# ---------------------------------------------------------------------------
def test_all_zero_capex_series_rejected():
    n = 5
    idx = _annual_index(n)
    bundle = _make_fcfe_ready_bundle(
        capex_annual=pd.Series([0.0] * n, index=idx),
    )
    ok, reason = fcfe_applicable(bundle)
    assert not ok
    assert reason == "Required cash-flow series missing"


# ---------------------------------------------------------------------------
# Phase A.0 sparse-history fixture integration
# ---------------------------------------------------------------------------
def test_sparse_history_bundle_rejected():
    """The synthetic 1-year-history fixture from Phase A.0 must be
    rejected by the 3y floor."""
    bundle = make_sparse_history_bundle(sector="Capital Goods")
    ok, reason = fcfe_applicable(bundle)
    assert not ok
    # The first failing check (in order) is capex history.
    assert "Insufficient" in reason


# ---------------------------------------------------------------------------
# fcfe_value short-circuits to FCFEResult(valid=False) when not applicable
# ---------------------------------------------------------------------------
def test_fcfe_value_propagates_invalid_reason():
    bundle = _make_fcfe_ready_bundle(sector="Banking")
    result = fcfe_value(bundle, ke=0.12, terminal_g=0.045)
    assert not result.valid
    assert result.reason_invalid == "FCFE not applicable to financial firms"
    assert result.value_per_share != result.value_per_share  # NaN
