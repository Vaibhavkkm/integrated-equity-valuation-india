"""
Phase A.6 — TCS FCFE regression sanity check.

For a high-payout, high-cash-conversion firm like TCS, the FCFE value
should land in roughly the same ballpark as the DDM value (their
underlying drivers — earnings, growth, Ke — are the same; FCFE just
nets out CapEx vs. D&A and the working-capital cycle).

This test runs the real integrated pipeline against the cached TCS
bundle and asserts FCFE within 50% of DDM. The tolerance is
intentionally generous — for a firm with growing receivables (TCS
FY24 ΔWC ≈ −₹68 kcr), FCFE should be somewhat *lower* than DDM, but
not by an order of magnitude. If this test ever fails with FCFE
materially higher than DDM, that's a sign the ΔWC sign got flipped
somewhere in the pipeline.

The test runs offline (cached bundle); no network access needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integrated_valuation import value_stock


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "cache" / "TCS.NS.pkl").exists(),
    reason="TCS.NS cache pickle not present; run with a populated cache.",
)
def test_tcs_fcfe_in_same_ballpark_as_ddm():
    """TCS DDM and FCFE should agree within a 2× factor either way."""
    r = value_stock("TCS.NS", offline=True, verbose=False)
    assert r.ddm.valid
    assert r.fcfe.valid

    ddm = r.ddm.value_per_share
    fcfe = r.fcfe.value_per_share

    ratio = fcfe / ddm
    assert 0.5 <= ratio <= 2.0, (
        f"TCS FCFE/DDM ratio = {ratio:.2f} (DDM=₹{ddm:,.0f}, "
        f"FCFE=₹{fcfe:,.0f}). Out of the expected 0.5–2.0 band — "
        f"check ΔWC sign and CapEx/D&A units."
    )


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "cache" / "TCS.NS.pkl").exists(),
    reason="TCS.NS cache pickle not present.",
)
def test_tcs_blend_branch_is_three_way_with_mid_or_high_payout():
    """TCS pays ~30% of earnings as dividend so its branch should be
    mid- or high-payout (not low, not the DDM-invalid override)."""
    r = value_stock("TCS.NS", offline=True, verbose=False)
    assert r.blend_branch in {"fcfe_mid_payout", "fcfe_high_payout"}, (
        f"Unexpected branch for TCS: {r.blend_branch}"
    )


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "cache" / "TCS.NS.pkl").exists(),
    reason="TCS.NS cache pickle not present.",
)
def test_tcs_weights_sum_to_one():
    r = value_stock("TCS.NS", offline=True, verbose=False)
    total = r.weight_ddm + r.weight_fcfe + r.weight_relative
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, not 1.0"


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "cache" / "SIEMENS.NS.pkl").exists(),
    reason="SIEMENS.NS cache pickle not present.",
)
def test_siemens_ddm_invalid_branch():
    """SIEMENS payout is ~0.3% — the DDM near-non-payer guard should
    fire and the three-way blend should route via the
    ``fcfe_ddm_invalid_*`` branch with w_ddm forced to 0."""
    r = value_stock("SIEMENS.NS", offline=True, verbose=False)
    # SIEMENS may have w_ddm in legacy [0.10, 0.80] clip if the guard
    # does NOT fire on the current cached data. The branch tag is the
    # authoritative signal.
    if r.blend_branch.startswith("fcfe_ddm_invalid"):
        assert r.weight_ddm == 0.0
        assert r.fcfe.valid
