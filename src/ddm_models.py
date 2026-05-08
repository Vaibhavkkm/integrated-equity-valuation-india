"""
Dividend Discount Models — the four canonical variants.

The hierarchy below mirrors the academic taxonomy:

  Gordon Growth  →  for stable, low-growth dividend payers (utilities,
                    legacy FMCG names like ITC, Hindustan Unilever).

  Two-Stage      →  for firms in clear high-growth phase that will fade
                    to perpetuity growth at a known horizon.

  H-Model        →  Fuller & Hsia's elegant approximation that lets
                    growth decline *linearly* over the transition window,
                    avoiding the discontinuity of the two-stage cliff.

  Three-Stage    →  the kitchen-sink: explicit high-growth, smooth
                    fade, and stable terminal — used when the business
                    is mid-cycle and a single fade slope is too coarse.

Auto-selection (`select_and_value`) picks the most appropriate variant
from a stock's payout history and growth profile, so callers don't need
to know the difference. But every individual model is exposed for the
research notebook and the sensitivity engine.

All models return an `IntrinsicValue` object with:
  - per-share value
  - the model name actually used
  - the inputs that drove it (so the report can show the workings)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import LONG_RUN_NOMINAL_GROWTH_IN, SETTINGS


# ---------------------------------------------------------------------------
# Result container — keeps callers honest about the assumptions used
# ---------------------------------------------------------------------------
@dataclass
class IntrinsicValue:
    value_per_share: float
    model: str
    inputs: dict = field(default_factory=dict)
    valid: bool = True
    note: str = ""

    def __repr__(self) -> str:
        v = self.value_per_share
        if not self.valid or not np.isfinite(v):
            return f"<IntrinsicValue model={self.model} INVALID note='{self.note}'>"
        return f"<IntrinsicValue model={self.model} ₹{v:,.2f}>"


# ---------------------------------------------------------------------------
# Sanity helpers
# ---------------------------------------------------------------------------
def _check_growth_below_discount(g: float, ke: float, label: str) -> Optional[str]:
    """The single most common DDM error is g >= Ke; trap it explicitly."""
    if g >= ke:
        return (
            f"{label}: growth ({g:.2%}) must be strictly less than cost of "
            f"equity ({ke:.2%}). Capping g at Ke − 50bps."
        )
    return None


def _cap_terminal_growth(g_terminal: float, ke: float) -> float:
    """Terminal growth must be less than Ke and the long-run economy growth."""
    ceiling = min(LONG_RUN_NOMINAL_GROWTH_IN, ke - 0.005)
    return float(min(g_terminal, ceiling))


# ---------------------------------------------------------------------------
# 1. Gordon Growth Model — the textbook starting point
# ---------------------------------------------------------------------------
def gordon_growth(d1: float, ke: float, g: float) -> IntrinsicValue:
    """V = D1 / (Ke − g).

    Parameters
    ----------
    d1 : next year's expected dividend per share (₹).
    ke : cost of equity (decimal).
    g  : perpetual growth rate (decimal).
    """
    note = _check_growth_below_discount(g, ke, "Gordon")
    g_eff = min(g, ke - 0.005) if note else g

    if d1 <= 0 or ke <= 0:
        return IntrinsicValue(
            value_per_share=float("nan"),
            model="Gordon Growth",
            inputs=dict(D1=d1, Ke=ke, g=g),
            valid=False,
            note="Non-positive dividend or discount rate.",
        )

    v = d1 / (ke - g_eff)
    return IntrinsicValue(
        value_per_share=v,
        model="Gordon Growth",
        inputs=dict(D1=d1, Ke=ke, g=g_eff),
        valid=True,
        note=note or "",
    )


# ---------------------------------------------------------------------------
# 2. Two-Stage DDM — explicit high-growth then perpetuity
# ---------------------------------------------------------------------------
def two_stage_ddm(
    d0: float,
    g_high: float,
    g_terminal: float,
    ke: float,
    n_years: int = SETTINGS.high_growth_years,
) -> IntrinsicValue:
    """Two-stage DDM with a sharp transition at year ``n_years``.

    Discounts an explicit forecast of dividends growing at ``g_high`` for
    ``n_years`` and tacks on a Gordon perpetuity at ``g_terminal``.

    Parameters
    ----------
    d0 : float
        Last actual dividend per share (TTM).
    g_high : float
        Growth rate during the explicit forecast horizon (decimal, e.g. 0.12).
    g_terminal : float
        Perpetual growth rate after year ``n_years``. Must satisfy
        ``g_terminal < ke``; the helper ``_cap_terminal_growth`` enforces this.
    ke : float
        Cost of equity (decimal).
    n_years : int, optional
        Length of the explicit high-growth phase. Defaults to
        ``SETTINGS.high_growth_years``.

    Returns
    -------
    IntrinsicValue
        ``valid=False`` (with NaN value) if the firm pays no dividend;
        otherwise the per-share intrinsic value plus an ``inputs`` dict that
        records the PV split between explicit dividends and terminal value.
    """
    g_terminal = _cap_terminal_growth(g_terminal, ke)
    note = _check_growth_below_discount(g_terminal, ke, "Two-Stage terminal")

    if d0 <= 0:
        return IntrinsicValue(
            value_per_share=float("nan"),
            model="Two-Stage DDM",
            inputs=dict(D0=d0, g_high=g_high, g_terminal=g_terminal, Ke=ke, n=n_years),
            valid=False,
            note="Company does not currently pay dividends.",
        )

    # PV of explicit dividends
    pv_explicit = 0.0
    d_t = d0
    for t in range(1, n_years + 1):
        d_t = d_t * (1 + g_high)
        pv_explicit += d_t / ((1 + ke) ** t)

    # Terminal value at end of year n
    d_terminal_plus_one = d_t * (1 + g_terminal)
    terminal_value = d_terminal_plus_one / (ke - g_terminal)
    pv_terminal = terminal_value / ((1 + ke) ** n_years)

    return IntrinsicValue(
        value_per_share=pv_explicit + pv_terminal,
        model="Two-Stage DDM",
        inputs=dict(
            D0=d0, g_high=g_high, g_terminal=g_terminal, Ke=ke,
            n_years=n_years, pv_explicit=pv_explicit, pv_terminal=pv_terminal,
        ),
        valid=True,
        note=note or "",
    )


# ---------------------------------------------------------------------------
# 3. H-Model — Fuller & Hsia (1984)
# ---------------------------------------------------------------------------
def h_model(
    d0: float,
    g_high: float,
    g_terminal: float,
    ke: float,
    half_life_years: float = SETTINGS.transition_years,
) -> IntrinsicValue:
    """H-Model closed form (Fuller & Hsia, 1984).

        V = [D0 * (1 + g_L) + D0 * H * (g_S − g_L)] / (Ke − g_L)

    where H = half_life_years / 2, g_S is the initial high growth rate, and
    g_L is the terminal growth rate. Growth declines linearly from g_S to
    g_L over ``2H`` years, eliminating the cliff in the two-stage model.

    Parameters
    ----------
    d0 : float
        Last actual dividend per share (TTM).
    g_high : float
        Initial (high) growth rate ``g_S`` at year 0.
    g_terminal : float
        Long-run growth rate ``g_L`` reached after the linear fade. Capped
        below ``ke`` upstream.
    ke : float
        Cost of equity (decimal).
    half_life_years : float, optional
        Number of years over which growth decays from ``g_S`` to ``g_L``.
        ``H`` is half this value. Defaults to ``SETTINGS.transition_years``.

    Returns
    -------
    IntrinsicValue
        ``valid=False`` for non-payers; otherwise the closed-form per-share
        intrinsic value with the implied ``H`` reported in ``inputs``.
    """
    g_terminal = _cap_terminal_growth(g_terminal, ke)
    note = _check_growth_below_discount(g_terminal, ke, "H-Model terminal")

    if d0 <= 0:
        return IntrinsicValue(
            value_per_share=float("nan"),
            model="H-Model",
            inputs=dict(D0=d0, g_high=g_high, g_terminal=g_terminal, Ke=ke),
            valid=False,
            note="Company does not currently pay dividends.",
        )

    H = half_life_years / 2.0
    numerator = d0 * (1 + g_terminal) + d0 * H * (g_high - g_terminal)
    v = numerator / (ke - g_terminal)
    return IntrinsicValue(
        value_per_share=v,
        model="H-Model",
        inputs=dict(
            D0=d0, g_S=g_high, g_L=g_terminal, Ke=ke, H=H,
            half_life_years=half_life_years,
        ),
        valid=True,
        note=note or "",
    )


# ---------------------------------------------------------------------------
# 4. Three-Stage DDM — high / fade / stable
# ---------------------------------------------------------------------------
def three_stage_ddm(
    d0: float,
    g_high: float,
    g_terminal: float,
    ke: float,
    high_years: int = SETTINGS.high_growth_years,
    fade_years: int = SETTINGS.transition_years,
) -> IntrinsicValue:
    """Three-stage DDM with a linear fade from ``g_high`` to ``g_terminal``.

    Stage 1 grows dividends at ``g_high`` for ``high_years``; stage 2 fades
    growth linearly across ``fade_years`` to the terminal rate; stage 3 is a
    Gordon perpetuity discounted back from the end of stage 2.

    Parameters
    ----------
    d0 : float
        Last actual dividend per share (TTM).
    g_high : float
        Stage-1 growth rate (decimal).
    g_terminal : float
        Stage-3 perpetual growth rate. Capped below ``ke`` upstream.
    ke : float
        Cost of equity (decimal).
    high_years : int, optional
        Length of stage 1. Defaults to ``SETTINGS.high_growth_years``.
    fade_years : int, optional
        Length of the linear-fade stage 2. Defaults to
        ``SETTINGS.transition_years``.

    Returns
    -------
    IntrinsicValue
        ``valid=False`` for non-payers; otherwise the per-share intrinsic
        value, with ``inputs`` recording each stage's PV contribution.
    """
    g_terminal = _cap_terminal_growth(g_terminal, ke)
    note = _check_growth_below_discount(g_terminal, ke, "Three-Stage terminal")

    if d0 <= 0:
        return IntrinsicValue(
            value_per_share=float("nan"),
            model="Three-Stage DDM",
            inputs=dict(D0=d0, g_high=g_high, g_terminal=g_terminal, Ke=ke),
            valid=False,
            note="Company does not currently pay dividends.",
        )

    pv_total = 0.0
    d_t = d0

    # Stage 1: high growth
    for t in range(1, high_years + 1):
        d_t *= (1 + g_high)
        pv_total += d_t / ((1 + ke) ** t)

    # Stage 2: linear fade
    fade_step = (g_high - g_terminal) / (fade_years + 1)
    g_t = g_high
    for k in range(1, fade_years + 1):
        g_t = g_high - fade_step * k
        d_t *= (1 + g_t)
        t_global = high_years + k
        pv_total += d_t / ((1 + ke) ** t_global)

    # Stage 3: stable perpetuity
    d_terminal_plus_one = d_t * (1 + g_terminal)
    terminal_value = d_terminal_plus_one / (ke - g_terminal)
    pv_terminal = terminal_value / ((1 + ke) ** (high_years + fade_years))
    pv_total += pv_terminal

    return IntrinsicValue(
        value_per_share=pv_total,
        model="Three-Stage DDM",
        inputs=dict(
            D0=d0, g_high=g_high, g_terminal=g_terminal, Ke=ke,
            high_years=high_years, fade_years=fade_years,
            pv_terminal=pv_terminal,
        ),
        valid=True,
        note=note or "",
    )


# ---------------------------------------------------------------------------
# Helpers used by the auto-selector
# ---------------------------------------------------------------------------
def historical_dividend_cagr(dps_series: pd.Series, fallback: float = 0.05) -> float:
    """CAGR of the trimmed dividend series, capped at sensible bounds.

    Why trim? Indian companies sometimes pay one-off special dividends
    (think: Coal India 2021), which make a naive CAGR explode. We drop the
    top and bottom value before computing growth, then re-sort by date
    so the geometric-mean math is anchored to the chronological endpoints.

    Parameters
    ----------
    dps_series : pd.Series
        Annual dividends per share, indexed by year. Order is restored
        chronologically internally — the caller may pass either direction.
    fallback : float, optional
        Returned when the series is too thin (< 3 paying years) or starts
        from zero. Defaults to 5% — the project's long-run prior.

    Returns
    -------
    float
        Annualised dividend growth rate, clipped to ``[-5%, 30%]``.
    """
    s = dps_series.dropna()
    s = s[s > 0]
    if len(s) < 3:
        return fallback
    if len(s) >= 5:
        # Trim min and max to defang special-dividend years.
        s = s.sort_values().iloc[1:-1]
        s = s.sort_index()

    n_years = max(1, len(s) - 1)
    start, end = float(s.iloc[0]), float(s.iloc[-1])
    if start <= 0:
        return fallback
    cagr = (end / start) ** (1 / n_years) - 1
    # Hard cap: nobody grows dividends at 40%/yr forever.
    return float(np.clip(cagr, -0.05, 0.30))


def sustainable_growth_rate(roe: float, payout_ratio: float) -> float:
    """``g = ROE * retention``. Classic dividend-irrelevance algebra.

    Parameters
    ----------
    roe : float
        Return on equity (decimal).
    payout_ratio : float
        Dividend / earnings (decimal). Retention ``= 1 - payout``,
        floored at 0 so over-distribution does not feed a negative
        sustainable growth.

    Returns
    -------
    float
        Sustainable growth rate, clipped to ``[-2%, 30%]``.
    """
    retention = max(0.0, 1.0 - payout_ratio)
    return float(np.clip(roe * retention, -0.02, 0.30))


def bayesian_growth_shrinkage(
    g_observed: float,
    *,
    n_observations: int,
    g_prior: float = 0.05,
    prior_strength: float = 5.0,
) -> float:
    """Shrink an observed growth estimate toward a sector prior.

    Why
    ---
    Historical CAGR computed off 3-4 noisy data points has high variance.
    Naively trusting it leads to one of the most common DDM failures —
    extrapolating a flukey 25% past growth into perpetuity. The standard
    statistical fix is to Bayes-shrink the estimate toward a prior:

        g_posterior = (n × g_observed + k × g_prior) / (n + k)

    where `k` is the prior strength expressed in pseudo-observations.

    With `n=3, k=5, g_observed=0.25, g_prior=0.05`, the posterior is
    0.125 — half the noisy observation, half the prior. With `n=10`,
    the posterior moves to 0.183, much closer to the data. This is the
    "James-Stein" idea borrowed for dividend forecasting and used in
    practice by buy-side equity desks.

    Parameters
    ----------
    g_observed : float
        Growth rate computed from a short series.
    n_observations : int
        How many years of data the observed rate is built on.
    g_prior : float
        Sector or market-wide expected growth (default 5% — long-run
        Indian nominal growth less inflation buffer).
    prior_strength : float
        Pseudo-observation count for the prior. Larger = more skeptical
        of the data.
    """
    n = max(0, int(n_observations))
    if n == 0:
        return float(g_prior)
    k = float(prior_strength)
    return float((n * g_observed + k * g_prior) / (n + k))


# ---------------------------------------------------------------------------
# The auto-selector — the brains of this module
# ---------------------------------------------------------------------------
def select_and_value(
    *,
    eps_ttm: float,
    dps_ttm: float,
    payout_ratio: float,
    roe: float,
    ke: float,
    historical_dps: pd.Series,
    historical_eps: pd.Series,
    sector_g_terminal: Optional[float] = None,
) -> IntrinsicValue:
    """Pick the most appropriate DDM variant and return its valuation.

    Decision rules (in plain English):

      * No history of dividends → return an invalid result. The relative
        valuation track will carry the load for these names.
      * Stable, mature payer (>5y of dividends, low growth, payout > 40%)
        → Gordon Growth.
      * Young / fast-growing payer (high g, payout < 40%) → H-Model so
        the growth fade is smooth.
      * Everything in between → Three-Stage DDM.

    Parameters
    ----------
    eps_ttm : float
        Trailing twelve-month earnings per share. Used only to detect
        token-dividend payers that should be treated as non-payers.
    dps_ttm : float
        Trailing twelve-month dividend per share. Zero or negative routes to
        the non-payer branch.
    payout_ratio : float
        Dividend / earnings, capped at 1.0 upstream. Drives both the
        sustainable-growth estimate and the variant-selection rules.
    roe : float
        Return on equity (decimal). Combined with retention to derive the
        sustainable growth rate ``g = ROE * (1 - payout)``.
    ke : float
        Cost of equity (decimal) — the discount rate for every variant.
    historical_dps : pd.Series
        Annual dividends per share, oldest → newest. Length governs the
        Bayesian shrinkage strength applied to the historical CAGR.
    historical_eps : pd.Series
        Annual earnings per share. Currently passed through for downstream
        consistency checks; not used in the closed-form valuation.
    sector_g_terminal : float, optional
        Sector-specific terminal growth override. Falls back to 4.5% — the
        long-run Indian nominal-growth-minus-inflation default.

    Returns
    -------
    IntrinsicValue
        For payers, the variant chosen plus the per-share intrinsic value.
        For non-payers (or token payers below 5% payout), an
        ``valid=False`` placeholder so the integrated pipeline can defer to
        relative valuation cleanly.
    """
    if dps_ttm <= 0:
        return IntrinsicValue(
            value_per_share=float("nan"),
            model="DDM (skipped)",
            inputs=dict(reason="Non-payer"),
            valid=False,
            note=(
                "Stock pays no dividend; DDM is not applicable. "
                "Valuation will rely on the relative-valuation track."
            ),
        )

    # Near-non-payer guard: a firm earning ₹60/share but paying ₹2/share
    # (HGINFRA-style infra/EPC reinvestors) has a 3% payout. DDM only
    # prices the dividend stream, so it values the ₹2 and ignores the
    # ₹58 being reinvested at ROE — yielding an intrinsic value an
    # order of magnitude below any defensible figure. Treat as a
    # non-payer and let relative valuation carry the call.
    if (
        np.isfinite(eps_ttm) and eps_ttm > 0
        and np.isfinite(payout_ratio) and 0 < payout_ratio < 0.05
    ):
        return IntrinsicValue(
            value_per_share=float("nan"),
            model="DDM (skipped)",
            inputs=dict(reason="Near-non-payer", payout_ratio=payout_ratio),
            valid=False,
            note=(
                f"Token dividend (payout={payout_ratio:.1%}) — firm retains "
                "~all earnings, so DDM cannot capture the reinvested cash "
                "flow. Valuation defers to the relative-valuation track."
            ),
        )

    # Build the growth signal from two sources: historical CAGR and the
    # sustainable-growth identity (g = ROE * b). Average them for stability,
    # then Bayes-shrink toward a 5% prior — short series get pulled in
    # harder, long series barely move. This is the project's defence
    # against extrapolating noisy 3-year CAGRs into perpetuity.
    g_hist_raw = historical_dividend_cagr(historical_dps)
    g_sgr = sustainable_growth_rate(roe, payout_ratio)
    g_blend = 0.5 * g_hist_raw + 0.5 * g_sgr
    n_obs = int((historical_dps.dropna() > 0).sum())
    g_high = float(np.clip(
        bayesian_growth_shrinkage(g_blend, n_observations=n_obs),
        -0.02, 0.25,
    ))

    g_terminal = sector_g_terminal if sector_g_terminal is not None else 0.045
    g_terminal = _cap_terminal_growth(g_terminal, ke)

    n_years_div_history = (historical_dps > 0).sum()

    # --- Gordon: mature, slow grower ---
    is_mature = (
        n_years_div_history >= 5
        and g_high <= g_terminal + 0.015
        and payout_ratio >= 0.40
    )
    if is_mature:
        d1 = dps_ttm * (1 + g_terminal)
        return gordon_growth(d1=d1, ke=ke, g=g_terminal)

    # --- H-Model: fast grower with low payout ---
    is_high_growth = g_high >= 0.12 or payout_ratio < 0.30
    if is_high_growth:
        return h_model(
            d0=dps_ttm,
            g_high=g_high,
            g_terminal=g_terminal,
            ke=ke,
            half_life_years=SETTINGS.transition_years * 2,
        )

    # --- Default: Three-Stage ---
    return three_stage_ddm(
        d0=dps_ttm,
        g_high=g_high,
        g_terminal=g_terminal,
        ke=ke,
        high_years=SETTINGS.high_growth_years,
        fade_years=SETTINGS.transition_years,
    )
