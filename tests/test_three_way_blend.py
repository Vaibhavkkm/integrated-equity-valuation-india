"""
Phase A.3 — three-way blend wrapper tests.

The wrapper has three branches:
  1. FCFE not applicable → two-way fallback (bit-identical to legacy)
  2. DDM invalid AND FCFE applicable → force w_ddm = 0, split per payout
  3. Both valid → wrap legacy _blend_weights, split remainder per payout

Plus the universal invariant: w_ddm + w_fcfe + w_rel == 1.0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SETTINGS
from src.ddm_models import IntrinsicValue
from src.fcfe_valuation import FCFEResult
from src.integrated_valuation import ThreeWayBlend, blend_three_way
from src.quality_score import QualityScore
from src.relative_valuation import RelativeValuation
from tests._factory import make_bundle


# ---------------------------------------------------------------------------
# Fixtures: lightweight stubs for ddm/rel/fcfe/quality
# ---------------------------------------------------------------------------
def _ddm(valid=True, value=1000.0):
    return IntrinsicValue(
        value_per_share=value if valid else float("nan"),
        model="Three-Stage DDM" if valid else "DDM (skipped)",
        valid=valid,
    )


def _rel(value=1200.0):
    return RelativeValuation(
        multiples={},
        weighted_value=value,
        weights={},
        median_value=value,
    )


def _fcfe(valid=True, value=900.0, reason=None):
    return FCFEResult(
        value_per_share=value if valid else float("nan"),
        valid=valid,
        reason_invalid=reason,
    )


def _quality(piotroski=6, dq=6):
    return QualityScore(
        piotroski_f=piotroski,
        altman_z=2.5,
        dividend_quality=dq,
        composite=70.0,
        breakdown={},
        earnings_momentum=5,
    )


# ---------------------------------------------------------------------------
# Universal invariant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("payout", [0.05, 0.15, 0.30, 0.55, 0.80])
@pytest.mark.parametrize("ddm_valid", [True, False])
@pytest.mark.parametrize("fcfe_valid", [True, False])
def test_weights_always_sum_to_one(payout, ddm_valid, fcfe_valid):
    """The single most important invariant: across every combination
    of payout bucket, DDM validity, and FCFE applicability, the three
    weights must sum to exactly 1.0."""
    bundle = make_bundle(payout_ratio=payout, sector="Capital Goods")
    blend = blend_three_way(
        ddm=_ddm(valid=ddm_valid),
        rel=_rel(),
        fcfe=_fcfe(valid=fcfe_valid),
        quality=_quality(),
        target=bundle,
    )
    total = blend.w_ddm + blend.w_fcfe + blend.w_rel
    assert abs(total - 1.0) < 1e-9, (
        f"payout={payout} ddm_valid={ddm_valid} fcfe_valid={fcfe_valid}: "
        f"weights sum to {total}, not 1.0"
    )


# ---------------------------------------------------------------------------
# Branch 1 — FCFE not applicable → two-way fallback
# ---------------------------------------------------------------------------
def test_branch1_fcfe_not_applicable_collapses_to_two_way():
    """When FCFE is not applicable, w_fcfe = 0 and the blend value is
    identical to the legacy two-way pipeline."""
    bundle = make_bundle(payout_ratio=0.45, sector="Capital Goods")
    blend = blend_three_way(
        ddm=_ddm(valid=True, value=1000.0),
        rel=_rel(value=1200.0),
        fcfe=_fcfe(valid=False, reason="not applicable for test"),
        quality=_quality(),
        target=bundle,
    )
    assert blend.branch_taken == "two_way_fallback"
    assert blend.w_fcfe == 0.0
    assert blend.w_ddm + blend.w_rel == pytest.approx(1.0)
    # Value matches two-way: w_ddm * 1000 + w_rel * 1200
    assert blend.value == pytest.approx(
        blend.w_ddm * 1000.0 + blend.w_rel * 1200.0
    )


def test_branch1_ddm_invalid_fcfe_invalid_falls_back_to_relative_only():
    """Neither intrinsic-value leg is usable → 100% relative."""
    bundle = make_bundle(payout_ratio=0.0, sector="Capital Goods")
    blend = blend_three_way(
        ddm=_ddm(valid=False),
        rel=_rel(value=500.0),
        fcfe=_fcfe(valid=False, reason="no data"),
        quality=_quality(),
        target=bundle,
    )
    assert blend.branch_taken == "two_way_fallback"
    assert blend.w_fcfe == 0.0
    assert blend.w_ddm == 0.0
    assert blend.w_rel == pytest.approx(1.0)
    assert blend.value == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Branch 2 — DDM invalid, FCFE applicable (directive #4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("payout,expected_share,bucket_tag", [
    (0.02, SETTINGS.fcfe_split_low_payout, "low_payout"),
    (0.15, SETTINGS.fcfe_split_low_payout, "low_payout"),
    (0.30, SETTINGS.fcfe_split_mid_payout, "mid_payout"),
    (0.60, SETTINGS.fcfe_split_high_payout, "high_payout"),
])
def test_branch2_ddm_invalid_forces_w_ddm_zero(payout, expected_share, bucket_tag):
    """Directive #4: when DDM returns invalid (e.g. payout < 5% guard
    fires), force w_ddm = 0 in the blend regardless of what
    _blend_weights returns. Split 100% between FCFE and Rel per the
    payout-bucket rule."""
    bundle = make_bundle(payout_ratio=payout, sector="Capital Goods")
    blend = blend_three_way(
        ddm=_ddm(valid=False),
        rel=_rel(value=1200.0),
        fcfe=_fcfe(valid=True, value=2000.0),
        quality=_quality(),
        target=bundle,
    )
    assert blend.branch_taken == f"fcfe_ddm_invalid_{bucket_tag}"
    assert blend.w_ddm == 0.0
    assert blend.w_fcfe == pytest.approx(expected_share[0])
    assert blend.w_rel == pytest.approx(expected_share[1])
    assert blend.value == pytest.approx(
        expected_share[0] * 2000.0 + expected_share[1] * 1200.0
    )


def test_branch2_ddm_invalid_rel_invalid_routes_to_pure_fcfe():
    """Edge case: DDM and Rel both invalid → 100% FCFE."""
    bundle = make_bundle(payout_ratio=0.02, sector="Capital Goods")
    blend = blend_three_way(
        ddm=_ddm(valid=False),
        rel=RelativeValuation(multiples={}, weighted_value=float("nan"), weights={}, median_value=float("nan")),
        fcfe=_fcfe(valid=True, value=1500.0),
        quality=_quality(),
        target=bundle,
    )
    assert blend.w_ddm == 0.0
    assert blend.w_fcfe == pytest.approx(1.0)
    assert blend.w_rel == 0.0
    assert blend.value == pytest.approx(1500.0)


# ---------------------------------------------------------------------------
# Branch 3 — both valid, payout splits the remainder
# ---------------------------------------------------------------------------
def test_branch3_high_payout_itc_style():
    """High-payout firm (ITC-like, payout ≥ 50%): legacy w_ddm slice
    is preserved; the remainder splits 30/70 FCFE/Rel.

    This is the specific case the user called out in the spec:
    'w_ddm stays high and FCFE gets a small slice'.
    """
    bundle = make_bundle(payout_ratio=0.85, sector="FMCG")
    blend = blend_three_way(
        ddm=_ddm(valid=True, value=1000.0),
        rel=_rel(value=1200.0),
        fcfe=_fcfe(valid=True, value=900.0),
        quality=_quality(dq=8),
        target=bundle,
    )
    assert blend.branch_taken == "fcfe_high_payout"
    # FCFE share should be small: 30% × (1 − w_ddm_legacy)
    assert blend.w_fcfe < 0.30
    # Rel gets the bigger slice of the non-DDM remainder
    assert blend.w_rel > blend.w_fcfe


def test_branch3_low_payout_siemens_style():
    """Low-payout firm (SIEMENS-like, payout < 20%): the legacy
    _blend_weights would already tilt strongly toward Relative
    (w_ddm clipped to 0.10), and FCFE gets the larger of the two
    remainder slices (70%)."""
    bundle = make_bundle(payout_ratio=0.10, sector="Capital Goods")
    blend = blend_three_way(
        ddm=_ddm(valid=True, value=200.0),
        rel=_rel(value=2300.0),
        fcfe=_fcfe(valid=True, value=3500.0),
        quality=_quality(),
        target=bundle,
    )
    assert blend.branch_taken == "fcfe_low_payout"
    # Remainder = 1 − w_ddm_legacy; FCFE share is 70% of that.
    assert blend.w_fcfe > blend.w_rel
    # Sanity: FCFE share > 0.5 × remainder, < remainder
    remainder = blend.w_fcfe + blend.w_rel
    assert blend.w_fcfe == pytest.approx(0.70 * remainder, rel=1e-9)
    assert blend.w_rel == pytest.approx(0.30 * remainder, rel=1e-9)


def test_branch3_mid_payout_splits_evenly():
    """Mid-payout (20% ≤ p < 50%): remainder splits 50/50."""
    bundle = make_bundle(payout_ratio=0.30, sector="Capital Goods")
    blend = blend_three_way(
        ddm=_ddm(valid=True, value=1000.0),
        rel=_rel(value=1100.0),
        fcfe=_fcfe(valid=True, value=950.0),
        quality=_quality(),
        target=bundle,
    )
    assert blend.branch_taken == "fcfe_mid_payout"
    assert blend.w_fcfe == pytest.approx(blend.w_rel, rel=1e-9)


def test_branch3_blended_value_matches_weighted_sum():
    """For branch 3 with all three legs valid, blend.value must equal
    the weighted sum of the three intrinsic values."""
    bundle = make_bundle(payout_ratio=0.30, sector="Capital Goods")
    blend = blend_three_way(
        ddm=_ddm(valid=True, value=1000.0),
        rel=_rel(value=1100.0),
        fcfe=_fcfe(valid=True, value=950.0),
        quality=_quality(),
        target=bundle,
    )
    expected = (
        blend.w_ddm * 1000.0
        + blend.w_fcfe * 950.0
        + blend.w_rel * 1100.0
    )
    assert blend.value == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Backward compatibility — pre-Phase-A behaviour unchanged when FCFE off
# ---------------------------------------------------------------------------
def test_two_way_fallback_matches_legacy_blend_weights_arithmetic():
    """When FCFE is not applicable, the blend value must be exactly
    what the legacy `_blend_weights` + manual weighted-sum would have
    produced. Regression guard for non-FCFE-applicable tickers."""
    from src.integrated_valuation import _blend_weights

    bundle = make_bundle(payout_ratio=0.50, sector="Capital Goods")
    ddm = _ddm(valid=True, value=1000.0)
    rel = _rel(value=1200.0)
    q = _quality()

    legacy_w_ddm, legacy_w_rel = _blend_weights(ddm, rel, q, bundle)
    legacy_value = legacy_w_ddm * 1000.0 + legacy_w_rel * 1200.0

    blend = blend_three_way(
        ddm=ddm, rel=rel,
        fcfe=_fcfe(valid=False, reason="not applicable"),
        quality=q, target=bundle,
    )
    assert blend.w_ddm == pytest.approx(legacy_w_ddm)
    assert blend.w_rel == pytest.approx(legacy_w_rel)
    assert blend.value == pytest.approx(legacy_value)
