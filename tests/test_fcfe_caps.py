"""
Phase A — FCFE caps and floors.

Pin the boundary behaviour of the engine's caps:
  * Terminal growth ``g_terminal`` is capped at
    ``min(input, LONG_RUN_NOMINAL_GROWTH_IN, ke − fcfe_ke_margin)``.
  * Stage-1 growth ``g1`` is capped at ``fcfe_g1_cap`` (15%) before
    Bayesian shrinkage; the post-shrinkage value may be lower.
  * Debt ratio ``DR`` is capped at ``fcfe_debt_ratio_cap`` (60%).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LONG_RUN_NOMINAL_GROWTH_IN, SETTINGS
from src.fcfe_valuation import _debt_ratio, fcfe_value
from tests._factory import _annual_index, make_bundle


def _baseline(**overrides):
    n = 5
    idx = _annual_index(n)
    flat = lambda v: pd.Series([float(v)] * n, index=idx)
    base = dict(
        sector="Capital Goods",
        net_income_annual=flat(1_000.0),
        capex_annual=flat(200.0),
        dep_amort_annual=flat(150.0),
        change_in_wc_annual=flat(-50.0),
        total_debt_annual=flat(500.0),
        working_capital_annual=flat(800.0),
        earnings_annual=pd.Series([10.0 * (1.08) ** k for k in range(n)], index=idx),
        eps_ttm=10.0 * (1.08) ** (n - 1),
        equity=4_000.0,
        total_debt=500.0,
    )
    base.update(overrides)
    return make_bundle(**base)


# ---------------------------------------------------------------------------
# Terminal growth caps
# ---------------------------------------------------------------------------
def test_terminal_g_capped_below_ke():
    """When the caller passes terminal_g > ke − fcfe_ke_margin, the
    engine quietly caps it to keep the Gordon denominator well-
    conditioned."""
    bundle = _baseline()
    ke = 0.07
    # 9% terminal > 6.5% ke ceiling → should be capped
    r = fcfe_value(bundle, ke=ke, terminal_g=0.09)
    assert r.valid
    assert r.inputs["g_terminal"] <= ke - SETTINGS.fcfe_ke_margin + 1e-12


def test_terminal_g_capped_at_long_run_nominal():
    """Even when the caller passes a terminal_g below ke, the engine
    still caps at LONG_RUN_NOMINAL_GROWTH_IN (currently 6%)."""
    bundle = _baseline()
    # 8% terminal, ke = 12% — would otherwise pass the ke-margin gate.
    r = fcfe_value(bundle, ke=0.12, terminal_g=0.08)
    assert r.valid
    assert r.inputs["g_terminal"] == pytest.approx(LONG_RUN_NOMINAL_GROWTH_IN)


def test_terminal_g_floored_at_zero():
    """Negative terminal growth is mathematically defined but
    economically nonsense for a non-distressed firm. Engine floors
    at zero."""
    bundle = _baseline()
    r = fcfe_value(bundle, ke=0.12, terminal_g=-0.02)
    assert r.valid
    assert r.inputs["g_terminal"] == 0.0


# ---------------------------------------------------------------------------
# Stage-1 growth cap
# ---------------------------------------------------------------------------
def test_g1_capped_at_fcfe_g1_cap():
    """Even when EPS CAGR is sky-high, the engine caps the raw input
    to fcfe_g1_cap (15%) BEFORE Bayesian shrinkage. The post-shrinkage
    value is necessarily ≤ the cap (mixing 15% with the 5% prior
    pulls the posterior down)."""
    n = 5
    idx = _annual_index(n)
    # EPS grows at 40%/yr — far above the cap
    bundle = _baseline(
        earnings_annual=pd.Series([5.0 * (1.40) ** k for k in range(n)], index=idx),
        eps_ttm=5.0 * (1.40) ** (n - 1),
    )
    r = fcfe_value(bundle, ke=0.14, terminal_g=0.045)
    assert r.valid
    assert r.inputs["g1"] <= SETTINGS.fcfe_g1_cap


# ---------------------------------------------------------------------------
# Debt-ratio cap
# ---------------------------------------------------------------------------
def test_debt_ratio_capped_for_overlevered_firm():
    """A firm with D/(D+E) > 60% should land at the 60% cap."""
    bundle = _baseline(
        total_debt_annual=pd.Series([9_000.0] * 5, index=_annual_index(5)),
        equity=1_000.0,
    )
    dr = _debt_ratio(bundle)
    assert dr == pytest.approx(SETTINGS.fcfe_debt_ratio_cap)


def test_debt_ratio_floored_at_zero():
    """Negative debt is not a real-world case but the function must
    not return a negative ratio for any input."""
    bundle = _baseline(
        total_debt_annual=pd.Series([0.0] * 5, index=_annual_index(5)),
        total_debt=0.0,
        equity=10_000.0,
    )
    assert _debt_ratio(bundle) == 0.0


def test_debt_ratio_negative_equity_returns_cap():
    """Distressed firm with negative equity — return the cap rather
    than a mathematically-defined-but-meaningless ratio."""
    bundle = _baseline(
        total_debt_annual=pd.Series([5_000.0] * 5, index=_annual_index(5)),
        equity=-1_000.0,
    )
    assert _debt_ratio(bundle) == SETTINGS.fcfe_debt_ratio_cap


# ---------------------------------------------------------------------------
# stage1_years bounds
# ---------------------------------------------------------------------------
def test_stage1_years_below_min_history_raises():
    bundle = _baseline()
    with pytest.raises(ValueError, match="stage1_years"):
        fcfe_value(bundle, ke=0.12, terminal_g=0.045, stage1_years=2)
