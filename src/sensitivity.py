"""
Sensitivity analysis: tornado chart inputs + Monte Carlo on intrinsic value.

Two outputs:

  1. **Tornado** — one-at-a-time bumps to each big-lever input
     (Ke, terminal g, high-stage g, peer P/E multiple). Returns the
     low/high intrinsic value for each lever, sorted by the size of
     swing it produces. This is the chart that goes on slide 2 of every
     equity-research deck for a reason.

  2. **Monte Carlo** — a 10,000-path simulation that draws inputs from
     truncated normals (Ke, growth) and lognormals (multiples) and
     reports the distribution of intrinsic value. Two MC configurations:
       * *DDM-only*   → varies Ke, g_high, g_terminal.
       * *Relative*   → varies the aggregated peer multiples.
     We then mix them at the same blend weight as the headline number.

The randomness uses a fixed seed (config.SETTINGS.mc_seed) so the
report is reproducible run-to-run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from config import SETTINGS
from src.cost_of_equity import CostOfEquity
from src.data_fetcher import StockBundle
from src.ddm_models import IntrinsicValue, select_and_value
from src.relative_valuation import RelativeValuation


# ---------------------------------------------------------------------------
# Tornado
# ---------------------------------------------------------------------------
@dataclass
class TornadoBar:
    lever: str
    low_value: float
    high_value: float
    base_value: float

    @property
    def swing(self) -> float:
        return abs(self.high_value - self.low_value)


def tornado_ddm(
    target: StockBundle,
    base_ke: float,
    base_g_terminal: float,
    *,
    ke_band: float = 0.01,
    g_band: float = 0.01,
    payout_band: float = 0.10,
) -> List[TornadoBar]:
    """One-at-a-time DDM sensitivity."""
    base_iv = _ddm_value(target, base_ke, base_g_terminal, target.payout_ratio)

    bars: List[TornadoBar] = []

    # Ke
    low = _ddm_value(target, base_ke + ke_band, base_g_terminal, target.payout_ratio)
    high = _ddm_value(target, base_ke - ke_band, base_g_terminal, target.payout_ratio)
    bars.append(TornadoBar("Cost of Equity (±1%)", low, high, base_iv))

    # Terminal growth
    low = _ddm_value(target, base_ke, base_g_terminal - g_band, target.payout_ratio)
    high = _ddm_value(target, base_ke, base_g_terminal + g_band, target.payout_ratio)
    bars.append(TornadoBar("Terminal Growth (±1%)", low, high, base_iv))

    # Payout ratio
    p_lo = max(0.05, target.payout_ratio - payout_band)
    p_hi = min(0.95, target.payout_ratio + payout_band)
    low = _ddm_value(target, base_ke, base_g_terminal, p_lo)
    high = _ddm_value(target, base_ke, base_g_terminal, p_hi)
    bars.append(TornadoBar("Payout Ratio (±10%)", low, high, base_iv))

    bars.sort(key=lambda b: b.swing, reverse=True)
    return bars


def _ddm_value(target: StockBundle, ke: float, g_term: float, payout: float) -> float:
    """Helper that re-runs the DDM with perturbed inputs."""
    iv = select_and_value(
        eps_ttm=target.eps_ttm,
        dps_ttm=target.dividend_per_share_ttm,
        payout_ratio=payout,
        roe=target.roe,
        ke=ke,
        historical_dps=target.dividends_annual,
        historical_eps=target.earnings_annual,
        sector_g_terminal=g_term,
    )
    return float(iv.value_per_share) if iv.valid else float("nan")


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
@dataclass
class MonteCarloResult:
    samples: np.ndarray
    p10: float
    p50: float
    p90: float
    mean: float
    std: float


def monte_carlo_blended(
    target: StockBundle,
    coe: CostOfEquity,
    rel: RelativeValuation,
    base_g_terminal: float,
    blend_w_ddm: float,
    *,
    n_paths: int = SETTINGS.mc_paths,
    seed: int = SETTINGS.mc_seed,
) -> MonteCarloResult:
    """Run a joint MC over DDM and Relative tracks, then blend each path."""
    rng = np.random.default_rng(seed)

    # Ke ~ truncated normal around base, σ=80 bps; clip to [4%, 25%]
    ke_draws = np.clip(rng.normal(coe.ke, 0.008, n_paths), 0.04, 0.25)

    # Terminal growth ~ truncated normal, σ=40 bps; clip to [0, 6%]
    g_term_draws = np.clip(rng.normal(base_g_terminal, 0.004, n_paths), 0.0, 0.06)

    # Payout ratio ~ Beta(α, β) centred on observed; reasonable for 0-1 range
    p = max(0.05, min(0.95, target.payout_ratio if target.payout_ratio else 0.30))
    a = p * 30
    b = (1 - p) * 30
    payout_draws = rng.beta(a, b, n_paths)

    # DDM samples (this is the slow loop — vectorised forms break for the
    # auto-selector logic, but 10k DDM evals is still <0.5s).
    ddm_samples = np.empty(n_paths)
    for i in range(n_paths):
        ddm_samples[i] = _ddm_value(target, ke_draws[i], g_term_draws[i], payout_draws[i])

    # Relative track: assume aggregated peer multiples ~ lognormal around the
    # measured value with σ_log = 0.20 (≈ 20% multiplicative noise).
    rel_base = rel.weighted_value
    if np.isfinite(rel_base) and rel_base > 0:
        log_mu = np.log(rel_base)
        rel_samples = rng.lognormal(log_mu - 0.5 * 0.20**2, 0.20, n_paths)
    else:
        rel_samples = np.full(n_paths, np.nan)

    # Blend per-path. If one track is NaN on a path, fall back to the other.
    blended = np.empty(n_paths)
    for i in range(n_paths):
        d, r = ddm_samples[i], rel_samples[i]
        if np.isfinite(d) and np.isfinite(r):
            blended[i] = blend_w_ddm * d + (1 - blend_w_ddm) * r
        elif np.isfinite(d):
            blended[i] = d
        elif np.isfinite(r):
            blended[i] = r
        else:
            blended[i] = np.nan

    blended = blended[np.isfinite(blended)]
    if blended.size == 0:
        return MonteCarloResult(np.array([]), np.nan, np.nan, np.nan, np.nan, np.nan)

    return MonteCarloResult(
        samples=blended,
        p10=float(np.percentile(blended, 10)),
        p50=float(np.percentile(blended, 50)),
        p90=float(np.percentile(blended, 90)),
        mean=float(np.mean(blended)),
        std=float(np.std(blended)),
    )
