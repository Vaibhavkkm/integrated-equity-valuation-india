"""
Phase A — textbook FCFE math regression.

Build a synthetic firm with constant cash-flow components and a
no-taper configuration (g1 == g_terminal) so the two-stage formula
collapses to a closed-form Gordon perpetuity. Verify the engine's
output matches the hand-computed value within rounding noise.

The collapse condition is: when ``g1 == g_terminal``, the linear
taper degenerates (taper_progress × 0), every year of stage 1 grows
at the same constant ``g_terminal``, and stage 1 plus stage 2 reduce
to a Gordon valuation of the base FCFE:

    V = FCFE_0 × (1 + g) / (Ke − g) / shares_outstanding

This is the same simplification used in DDM testing — see
``test_valuation.py::test_h_model_degenerates_to_gordon``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fcfe_valuation import fcfe_value
from tests._factory import _annual_index, make_bundle


def _flat_bundle(*, ni: float, capex: float, da: float, wc: float,
                 shares: float, debt: float, equity: float,
                 g_eps: float, n_years: int = 5,
                 roe: float = float("nan")) -> "StockBundle":
    """Build a synthetic bundle whose cash-flow series are CONSTANT
    year-on-year. EPS grows at ``g_eps`` so the historical CAGR is
    deterministic.

    ``roe`` defaults to NaN so the FCFE engine's
    "missing-ROE → skip SGR" fallback fires and g1 reduces to a
    Bayes-shrunk EPS CAGR alone. Pass an explicit ROE to test the
    SGR-blend branch.
    """
    idx = _annual_index(n_years)
    # EPS series rigged so that CAGR = g_eps exactly across n_years − 1 steps
    eps_start = 10.0
    eps_series = pd.Series(
        [eps_start * (1 + g_eps) ** k for k in range(n_years)],
        index=idx,
    )
    flat = lambda v: pd.Series([float(v)] * n_years, index=idx)

    return make_bundle(
        ticker="FLAT.NS",
        n_years=n_years,
        eps_ttm=float(eps_series.iloc[-1]),
        net_income=ni,
        total_debt=debt,
        equity=equity,
        shares_outstanding=shares,
        earnings_annual=eps_series,
        revenue_annual=flat(ni * 5),
        dividends_annual=pd.Series(dtype=float),  # non-payer
        book_value_annual=flat(equity / shares),
        net_income_annual=flat(ni),
        capex_annual=flat(capex),
        dep_amort_annual=flat(da),
        change_in_wc_annual=flat(wc),
        total_debt_annual=flat(debt),
        working_capital_annual=flat(0.0),  # not used by fcfe_value
        payout_ratio=0.0,
        dividend_per_share_ttm=0.0,
        sector="Capital Goods",  # non-financial
        roe=roe,
    )


def test_fcfe_collapses_to_gordon_when_growth_uniform():
    """When the Bayes-shrunk g1 equals g_terminal, the taper has no
    effect and the two-stage formula reduces to a Gordon valuation
    of the base FCFE."""
    # Pick a small constant growth rate so g1 (Bayes-shrunk EPS CAGR)
    # lands exactly at the prior (5%) — which we'll also use as
    # terminal_g. With n_eps_obs=5 and prior_strength=5, the posterior
    # = (5×0.05 + 5×0.05) / 10 = 0.05 — perfectly flat.
    g = 0.05
    ke = 0.12
    ni, capex, da, wc = 10_000.0, 800.0, 600.0, 100.0
    debt, equity = 2_000.0, 8_000.0
    shares = 1_000.0

    bundle = _flat_bundle(
        ni=ni, capex=capex, da=da, wc=wc,
        shares=shares, debt=debt, equity=equity,
        g_eps=g,
    )

    result = fcfe_value(bundle, ke=ke, terminal_g=g, stage1_years=7)
    assert result.valid, result.reason_invalid

    # Hand computation: DR = 2000/(2000+8000) = 0.20
    dr = debt / (debt + equity)
    base_fcfe = ni - (capex - da) * (1 - dr) + wc * (1 - dr)
    expected_total = base_fcfe * (1 + g) / (ke - g)
    expected_per_share = expected_total / shares

    rel_err = abs(result.value_per_share - expected_per_share) / abs(expected_per_share)
    assert rel_err < 0.0001, (
        f"FCFE collapses-to-Gordon mismatch: engine={result.value_per_share:.6f}, "
        f"hand={expected_per_share:.6f}, rel_err={rel_err:.6%}"
    )


def test_fcfe_inputs_record_base_components():
    """The ``inputs`` dict on FCFEResult should record every component
    the year-by-year sheet depends on, so the UI can render the
    workings without recomputing anything."""
    bundle = _flat_bundle(
        ni=10_000.0, capex=800.0, da=600.0, wc=100.0,
        shares=1_000.0, debt=2_000.0, equity=8_000.0,
        g_eps=0.05,
    )
    r = fcfe_value(bundle, ke=0.12, terminal_g=0.05)

    expected_keys = {
        "ni0", "capex0", "da0", "wc0_yf", "base_fcfe",
        "g1", "g_terminal", "ke", "dr", "stage1_years",
        "pv_explicit", "pv_terminal", "shares_outstanding",
    }
    assert expected_keys <= set(r.inputs.keys())
    assert r.inputs["ni0"] == 10_000.0
    assert r.inputs["capex0"] == 800.0
    assert r.inputs["da0"] == 600.0
    assert r.inputs["wc0_yf"] == 100.0
    assert r.inputs["dr"] == pytest.approx(0.20)


def test_fcfe_year_by_year_has_correct_row_count():
    """``year_by_year`` should have exactly ``stage1_years`` rows."""
    bundle = _flat_bundle(
        ni=10_000.0, capex=800.0, da=600.0, wc=100.0,
        shares=1_000.0, debt=2_000.0, equity=8_000.0, g_eps=0.05,
    )
    r = fcfe_value(bundle, ke=0.12, terminal_g=0.05, stage1_years=7)
    assert len(r.year_by_year) == 7
    assert list(r.year_by_year.columns) == [
        "year", "growth", "fcfe", "pv", "pv_cumulative",
    ]


def test_fcfe_pv_explicit_plus_pv_terminal_equals_total_value():
    """Sanity: PV(stage 1) + PV(terminal) → equity value → per share."""
    shares = 1_000.0
    bundle = _flat_bundle(
        ni=10_000.0, capex=800.0, da=600.0, wc=100.0,
        shares=shares, debt=2_000.0, equity=8_000.0, g_eps=0.05,
    )
    r = fcfe_value(bundle, ke=0.12, terminal_g=0.05)
    reconstructed = (r.inputs["pv_explicit"] + r.pv_terminal) / shares
    assert r.value_per_share == pytest.approx(reconstructed, rel=1e-9)


def test_fcfe_growth_taper_in_final_two_years():
    """When g1 > g_terminal, the final 2 years of stage 1 fade
    linearly. Inspect the year_by_year growth column to confirm."""
    # Force g1 high (so EPS CAGR > terminal_g materially)
    bundle = _flat_bundle(
        ni=10_000.0, capex=800.0, da=600.0, wc=100.0,
        shares=1_000.0, debt=2_000.0, equity=8_000.0,
        g_eps=0.12,  # >> 0.05 terminal
    )
    r = fcfe_value(bundle, ke=0.14, terminal_g=0.05, stage1_years=7)
    assert r.valid
    growth = r.year_by_year["growth"]
    g1 = r.inputs["g1"]
    g_term = r.inputs["g_terminal"]

    # Years 1..5 are pre-taper: growth == g1
    for i in range(5):
        assert growth.iloc[i] == pytest.approx(g1)
    # Year 6 = halfway between g1 and g_terminal
    assert growth.iloc[5] == pytest.approx(g1 + 0.5 * (g_term - g1))
    # Year 7 = full terminal
    assert growth.iloc[6] == pytest.approx(g_term)


def test_g1_blends_eps_cagr_with_sustainable_growth():
    """Phase A fix: FCFE g1 must blend historical EPS CAGR with SGR
    the same way DDM blends historical DPS CAGR with SGR. Without
    this, the two legs disagree on what stage-1 growth rate is being
    applied to the same firm — a methodologically incoherent blend
    regardless of which is "more accurate".

    Construct a firm where EPS CAGR (5%) and SGR (15%) differ
    materially, and verify the inputs dict records both ingredients
    AND that g1 sits BETWEEN them (Bayes-shrunk toward the 5% prior).
    """
    # ROE 30%, payout 50% → SGR = 30% × 0.5 = 15%
    # EPS CAGR rigged to 5% via constant-growth EPS series
    bundle = _flat_bundle(
        ni=10_000.0, capex=800.0, da=600.0, wc=100.0,
        shares=1_000.0, debt=2_000.0, equity=8_000.0,
        g_eps=0.05,
    )
    # Override payout/ROE explicitly (the flat-bundle helper doesn't set them)
    bundle.payout_ratio = 0.50
    bundle.roe = 0.30

    r = fcfe_value(bundle, ke=0.14, terminal_g=0.045)
    assert r.valid

    assert r.inputs["g_eps_cagr"] == pytest.approx(0.05, abs=1e-6)
    assert r.inputs["g_sgr"] == pytest.approx(0.15, abs=1e-6)
    # g1 is Bayes-shrunk blend of (0.05 + 0.15)/2 = 0.10 toward 5% prior
    # with n_eps_obs = 5 (the flat-bundle default). Posterior:
    #   (5 × 0.10 + 5 × 0.05) / 10 = 0.075
    assert r.inputs["g1"] == pytest.approx(0.075, abs=1e-6)


def test_g1_falls_back_to_eps_cagr_when_roe_missing():
    """When ROE is missing or non-positive, the blend collapses to
    EPS CAGR alone — mirroring the DDM's identical fallback so the
    two legs stay consistent even on data-thin firms."""
    bundle = _flat_bundle(
        ni=10_000.0, capex=800.0, da=600.0, wc=100.0,
        shares=1_000.0, debt=2_000.0, equity=8_000.0,
        g_eps=0.05,
    )
    bundle.roe = float("nan")  # missing
    bundle.payout_ratio = 0.50

    r = fcfe_value(bundle, ke=0.14, terminal_g=0.045)
    assert r.valid
    assert r.inputs["g_sgr"] is None
    # g1 is Bayes-shrunk EPS CAGR alone: (5×0.05 + 5×0.05)/10 = 0.05
    assert r.inputs["g1"] == pytest.approx(0.05, abs=1e-6)
