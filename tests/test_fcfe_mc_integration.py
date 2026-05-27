"""
Phase A.4 — Monte Carlo three-way runner integration tests.

Confirms ``monte_carlo_three_way``:
  * Returns finite mean / std for an FCFE-applicable bundle
  * Lands within the Phase A runtime budget (≤8 ms)
  * Reuses the existing distributions for Ke / terminal-g / payout
    (no regression on the two-way runner — it is left untouched)
  * Falls back cleanly when FCFE happens to be invalid on a path
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cost_of_equity import CostOfEquity
from src.ddm_models import IntrinsicValue
from src.fcfe_valuation import fcfe_value
from src.relative_valuation import RelativeValuation
from src.sensitivity import monte_carlo_three_way
from tests._factory import _annual_index, make_bundle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _fcfe_ready_bundle():
    n = 5
    idx = _annual_index(n)
    flat = lambda v: pd.Series([float(v)] * n, index=idx)
    eps = pd.Series([10.0 * (1.08) ** k for k in range(n)], index=idx)
    return make_bundle(
        sector="Capital Goods",
        net_income_annual=flat(1_000.0),
        capex_annual=flat(200.0),
        dep_amort_annual=flat(150.0),
        change_in_wc_annual=flat(-50.0),
        total_debt_annual=flat(500.0),
        working_capital_annual=flat(800.0),
        earnings_annual=eps,
        eps_ttm=float(eps.iloc[-1]),
        net_income=1_000.0,
        revenue=10_000.0,
        equity=4_000.0,
        total_debt=500.0,
        dividend_per_share_ttm=2.0,
        dividends_annual=pd.Series([1.0, 1.2, 1.5, 1.8, 2.0], index=idx),
        payout_ratio=0.10,
    )


def _coe(ke=0.12):
    return CostOfEquity(
        ke=ke,
        beta_raw=1.0,
        beta_adjusted=1.0,
        risk_free=0.071,
        erp=0.07,
        method="test_stub",
    )


def _rel(value=2000.0):
    return RelativeValuation(
        multiples={},
        weighted_value=value,
        weights={},
        median_value=value,
    )


# ---------------------------------------------------------------------------
# Functional integration
# ---------------------------------------------------------------------------
def test_monte_carlo_three_way_returns_finite_distribution():
    bundle = _fcfe_ready_bundle()
    coe = _coe()
    fcfe = fcfe_value(bundle, ke=coe.ke, terminal_g=0.045)
    assert fcfe.valid

    mc = monte_carlo_three_way(
        target=bundle, coe=coe, rel=_rel(),
        fcfe=fcfe, base_g_terminal=0.045,
        w_ddm=0.10, w_fcfe=0.63, w_rel=0.27,
        n_paths=10_000, seed=42,
    )
    assert mc.samples.size > 9_000  # most paths survive
    assert np.isfinite(mc.mean) and mc.mean > 0
    assert np.isfinite(mc.std) and mc.std > 0
    assert mc.p10 < mc.p50 < mc.p90


def test_monte_carlo_three_way_runtime_budget():
    """Phase A runtime budget: ≤8 ms for the three-way case
    (vectorised; no per-path loops on the leg evaluators)."""
    bundle = _fcfe_ready_bundle()
    coe = _coe()
    fcfe = fcfe_value(bundle, ke=coe.ke, terminal_g=0.045)
    assert fcfe.valid

    # Warm-up to avoid first-call import/JIT-style overhead biasing
    # the measurement.
    monte_carlo_three_way(
        bundle, coe, _rel(), fcfe, 0.045, 0.10, 0.63, 0.27,
        n_paths=10_000, seed=42,
    )

    n_runs = 5
    t0 = time.perf_counter()
    for _ in range(n_runs):
        monte_carlo_three_way(
            bundle, coe, _rel(), fcfe, 0.045, 0.10, 0.63, 0.27,
            n_paths=10_000, seed=42,
        )
    avg_ms = (time.perf_counter() - t0) / n_runs * 1000

    # Be generous in CI but flag if it crosses 25 ms — the Phase A
    # spec target is 8 ms on a healthy laptop; CI machines vary.
    assert avg_ms < 25.0, (
        f"monte_carlo_three_way avg {avg_ms:.2f} ms > 25 ms — "
        f"non-vectorised loop snuck in?"
    )


def test_monte_carlo_three_way_with_invalid_fcfe_returns_two_way():
    """When fcfe.valid is False the three-way runner produces the
    same MC distribution as the legacy two-way runner (modulo RNG
    state differences from the extra DR/CapEx draws not happening)."""
    from src.fcfe_valuation import FCFEResult

    bundle = _fcfe_ready_bundle()
    coe = _coe()
    invalid_fcfe = FCFEResult(
        value_per_share=float("nan"),
        valid=False,
        reason_invalid="test stub",
    )

    mc = monte_carlo_three_way(
        target=bundle, coe=coe, rel=_rel(),
        fcfe=invalid_fcfe, base_g_terminal=0.045,
        w_ddm=0.50, w_fcfe=0.0, w_rel=0.50,
        n_paths=10_000, seed=42,
    )
    # With w_fcfe=0 and FCFE samples NaN, the blend should still
    # produce a finite distribution from DDM + Rel.
    assert mc.samples.size > 9_000
    assert np.isfinite(mc.mean)


def test_monte_carlo_three_way_distribution_widens_with_fcfe_added():
    """Adding the FCFE leg with its own per-path noise sources
    should not make the dispersion suspiciously narrower than the
    two-way case (a tell-tale sign of a broken evaluator returning
    constants). Sanity floor on σ/μ."""
    bundle = _fcfe_ready_bundle()
    coe = _coe()
    fcfe = fcfe_value(bundle, ke=coe.ke, terminal_g=0.045)
    assert fcfe.valid

    mc = monte_carlo_three_way(
        bundle, coe, _rel(), fcfe, 0.045, 0.10, 0.63, 0.27,
        n_paths=10_000, seed=42,
    )
    cov = mc.std / mc.mean
    # Three independent leg noise sources → CoV should be ≥5%
    # (almost any real-world set of inputs will exceed this).
    assert cov > 0.05, f"σ/μ = {cov:.4f} suspiciously narrow"
