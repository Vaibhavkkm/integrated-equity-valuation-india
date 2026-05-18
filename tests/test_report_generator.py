"""
Smoke test for the PDF report generator.

We build a synthetic ValuationResult end-to-end and assert the PDF
writes to disk without raising. Validating the PDF content byte-by-byte
is out of scope — what we want is a regression net against accidental
template breakage when sections, fields, or ReportLab styles drift.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.cost_of_equity import CostOfEquity
from src.data_validation import DataQualityReport
from src.ddm_models import IntrinsicValue
from src.earnings_momentum import EarningsMomentum
from src.integrated_valuation import ValuationResult
from src.peer_identification import PeerSet
from src.quality_score import QualityScore
from src.relative_valuation import MultipleResult, RelativeValuation
from src.report_generator import generate_pdf
from src.reverse_dcf import ImpliedExpectations
from src.sensitivity import MonteCarloResult, TornadoBar

from _factory import make_bundle


def _stub_relative(value: float) -> RelativeValuation:
    empty = pd.Series(dtype=float)
    mr = MultipleResult(
        multiple_name="PE", peer_values=empty,
        aggregated_multiple=20.0, target_per_share_metric=25.0,
        implied_price=value, valid=True,
    )
    return RelativeValuation(
        multiples={"PE": mr}, weighted_value=value,
        weights={"PE": 1.0}, median_value=value,
    )


def _stub_result(recommendation="BUY") -> ValuationResult:
    target = make_bundle()
    coe = CostOfEquity(
        ke=0.123, beta_raw=0.85, beta_adjusted=0.90,
        risk_free=0.071, erp=0.07, method="CAPM (own beta)",
    )
    ddm = IntrinsicValue(
        value_per_share=450.0, model="Gordon Growth",
        inputs={"D1": 14.6, "Ke": 0.123, "g": 0.045}, valid=True,
    )
    rel = _stub_relative(500.0)
    peer_set = PeerSet(
        target=target, peers=[target] * 5,
        method="k-means cluster + Mahalanobis",
        debug_features=None,
    )
    quality = QualityScore(
        piotroski_f=6, altman_z=4.2, dividend_quality=7,
        composite=72.3, breakdown={"piotroski_components": {}},
        earnings_momentum=5,
    )
    momentum = EarningsMomentum(
        yoy_profit_growth_q=0.18, yoy_revenue_growth_q=0.10,
        score=6, direction="IMPROVING",
    )
    data_q = DataQualityReport(
        score=85.0, completeness=1.0, history_years=7,
        has_price_history=True, has_dividends=True,
    )
    mc_samples = np.array([400.0, 450.0, 500.0, 480.0, 470.0])
    mc = MonteCarloResult(
        samples=mc_samples, p10=410.0, p50=470.0, p90=495.0,
        mean=460.0, std=35.0,
    )
    tornado = [
        TornadoBar("Cost of Equity (±1%)", 430.0, 490.0, 460.0),
        TornadoBar("Terminal Growth (±1%)", 440.0, 480.0, 460.0),
    ]
    reverse = ImpliedExpectations(
        implied_growth=0.07, market_price=target.price,
        historical_growth=0.08, expectation_gap=0.01,
        verdict="modest expectations", valid=True,
    )
    return ValuationResult(
        target=target, coe=coe, ddm=ddm, relative=rel,
        peer_set=peer_set, quality=quality, data_quality=data_q,
        momentum=momentum,
        weight_ddm=0.5, weight_relative=0.5,
        blended_value=475.0, margin_of_safety=0.16,
        recommendation=recommendation, confidence="HIGH",
        monte_carlo=mc, tornado=tornado, reverse_dcf=reverse,
        base_g_terminal=0.045,
        timestamp=datetime(2026, 5, 18, tzinfo=timezone.utc),
        notes=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_generate_pdf_writes_a_buy_report(tmp_path, monkeypatch):
    monkeypatch.setattr("src.report_generator.REPORT_DIR", tmp_path)
    result = _stub_result(recommendation="BUY")
    path = generate_pdf(result)
    assert path.exists()
    assert path.stat().st_size > 1_000              # non-trivial PDF
    head = path.read_bytes()[:4]
    assert head == b"%PDF"                           # valid PDF magic bytes


def test_generate_pdf_handles_sell_recommendation(tmp_path, monkeypatch):
    monkeypatch.setattr("src.report_generator.REPORT_DIR", tmp_path)
    result = _stub_result(recommendation="SELL")
    path = generate_pdf(result)
    assert path.exists()


def test_generate_pdf_handles_invalid_ddm(tmp_path, monkeypatch):
    """Non-payers route the DDM section through the 'not applicable'
    branch — must still render without raising."""
    monkeypatch.setattr("src.report_generator.REPORT_DIR", tmp_path)
    result = _stub_result(recommendation="HOLD")
    result.ddm = IntrinsicValue(
        value_per_share=float("nan"), model="DDM (skipped)",
        inputs={"reason": "Non-payer"}, valid=False,
        note="Company does not currently pay dividends.",
    )
    path = generate_pdf(result)
    assert path.exists()
