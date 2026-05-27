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
     reports the distribution of intrinsic value. The DDM track is
     evaluated **vectorised** with NumPy: the auto-selector picks the
     variant once from the central inputs (Gordon / H-Model / Three-
     Stage), then the variant's closed form is applied across all
     ``n_paths`` perturbations in a single broadcast. This is also more
     methodologically honest than the previous per-path approach — the
     confidence bands answer "how sensitive is THIS chosen model to
     inputs?" rather than mixing model-selection uncertainty into the
     same number. The relative track is sampled lognormally around the
     measured peer-aggregated value; the per-path blend uses ``np.where``
     to fall back to whichever track is valid on each draw.

The randomness uses a fixed seed (config.SETTINGS.mc_seed) so the
report is reproducible run-to-run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from config import LONG_RUN_NOMINAL_GROWTH_IN, SETTINGS
from src.cost_of_equity import CostOfEquity
from src.data_fetcher import StockBundle
from src.ddm_models import (
    IntrinsicValue,
    historical_dividend_cagr,
    select_and_value,
)
from src.fcfe_valuation import FCFEResult
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
    """One-at-a-time DDM sensitivity.

    Bumps each lever individually around the base case, holding the
    others fixed, and ranks the resulting bars by absolute swing. The
    output drives the tornado chart in the dashboard and the PDF report.

    Parameters
    ----------
    target : StockBundle
        The stock being valued. Provides DPS, EPS, ROE, payout, and
        history needed by ``select_and_value`` under each perturbation.
    base_ke : float
        Base cost of equity (decimal) — the centre of the Ke band.
    base_g_terminal : float
        Base terminal growth rate (decimal) — the centre of the g band.
    ke_band : float, optional
        Half-width of the cost-of-equity perturbation. Defaults to 1%.
    g_band : float, optional
        Half-width of the terminal-growth perturbation. Defaults to 1%.
    payout_band : float, optional
        Half-width of the payout-ratio perturbation, clamped to
        ``[0.05, 0.95]`` after applying. Defaults to 10 percentage points.

    Returns
    -------
    list[TornadoBar]
        Sorted by descending swing (largest mover first).
    """
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
        payout_ratio_raw=getattr(target, "payout_ratio_raw", float("nan")),
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
    """Run a joint MC over DDM and Relative tracks, then blend each path.

    Draws ``n_paths`` correlated samples for cost of equity, terminal
    growth, payout, and a per-multiple noise term, recomputes the DDM
    and relative implied prices on each draw, and returns the
    distribution of blended intrinsic values that drives the project's
    confidence bands.

    The DDM track is **vectorised**: the variant (Gordon / H-Model /
    Three-Stage) is locked in by a single auto-selector call on the
    central inputs, then the variant's closed form is broadcast across
    all path arrays in NumPy. This is both ~20× faster than the previous
    per-path loop and methodologically cleaner — bands measure
    input-sensitivity for *one* model, not a Frankenstein mix of model
    selections fired on perturbed draws. See module docstring.

    Parameters
    ----------
    target : StockBundle
        The stock being valued — supplies DPS, EPS, ROE, history.
    coe : CostOfEquity
        Base cost-of-equity object. Its ``ke`` is the centre of the
        Ke distribution.
    rel : RelativeValuation
        Output of :func:`relative_valuation.value_by_multiples`. Provides
        the per-multiple peer distributions sampled inside the MC loop.
    base_g_terminal : float
        Base terminal growth rate (decimal).
    blend_w_ddm : float
        Weight applied to the DDM track on each path; the relative track
        receives ``1 - blend_w_ddm``.
    n_paths : int, optional
        Number of Monte Carlo paths. Defaults to ``SETTINGS.mc_paths``
        (10,000 in the project default).
    seed : int, optional
        RNG seed for reproducibility. Defaults to ``SETTINGS.mc_seed``.

    Returns
    -------
    MonteCarloResult
        ``samples`` (np.ndarray of finite blended values), plus the
        P10 / P50 / P90 / mean / std summary statistics.
    """
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

    # --- DDM track (vectorised) ----------------------------------------
    # Pin the variant at the central case, then broadcast its closed form
    # across the path arrays.
    base_iv = select_and_value(
        eps_ttm=target.eps_ttm,
        dps_ttm=target.dividend_per_share_ttm,
        payout_ratio=target.payout_ratio,
        roe=target.roe,
        ke=coe.ke,
        historical_dps=target.dividends_annual,
        historical_eps=target.earnings_annual,
        sector_g_terminal=base_g_terminal,
        payout_ratio_raw=getattr(target, "payout_ratio_raw", float("nan")),
    )
    ddm_samples = _vectorised_ddm_samples(
        target, base_iv, ke_draws, g_term_draws, payout_draws,
    )

    # --- Relative track ------------------------------------------------
    # Aggregated peer multiples ~ lognormal around the measured value with
    # σ_log = 0.20 (≈ 20% multiplicative noise).
    rel_base = rel.weighted_value
    if np.isfinite(rel_base) and rel_base > 0:
        log_mu = np.log(rel_base)
        rel_samples = rng.lognormal(log_mu - 0.5 * 0.20**2, 0.20, n_paths)
    else:
        rel_samples = np.full(n_paths, np.nan)

    # --- Blend (vectorised) --------------------------------------------
    # If one track is NaN on a path, fall back to the other.
    d_ok = np.isfinite(ddm_samples)
    r_ok = np.isfinite(rel_samples)
    blended = np.full(n_paths, np.nan)
    both = d_ok & r_ok
    blended[both] = blend_w_ddm * ddm_samples[both] + (1 - blend_w_ddm) * rel_samples[both]
    blended[d_ok & ~r_ok] = ddm_samples[d_ok & ~r_ok]
    blended[~d_ok & r_ok] = rel_samples[~d_ok & r_ok]

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


# ---------------------------------------------------------------------------
# Vectorised DDM evaluator (used by `monte_carlo_blended`)
# ---------------------------------------------------------------------------
def _vectorised_ddm_samples(
    target: StockBundle,
    base_iv: IntrinsicValue,
    ke_arr: np.ndarray,
    g_term_arr: np.ndarray,
    payout_arr: np.ndarray,
) -> np.ndarray:
    """Closed-form DDM evaluated across all MC paths simultaneously.

    The variant is fixed by ``base_iv.model`` (Gordon / H-Model / Three-
    Stage); per-path inputs perturb Ke, terminal growth, and payout
    around the centre. ``g_high`` is recomputed per path using the same
    blend-of-historical-and-SGR rule as :func:`select_and_value`, then
    Bayes-shrunk against a 5% prior.

    Returns an array of length ``n_paths``; paths where the variant's
    domain assumptions are violated (e.g. ``payout < 0.05``, ``g_terminal
    ≥ ke``) are returned as NaN so the downstream blender drops them.
    """
    n_paths = ke_arr.shape[0]
    if not base_iv.valid:
        return np.full(n_paths, np.nan)

    d0 = target.dividend_per_share_ttm
    if d0 <= 0:
        return np.full(n_paths, np.nan)

    # --- Per-path g_high (Bayes-shrunk blend of g_hist and SGR) --------
    g_hist_raw = historical_dividend_cagr(target.dividends_annual)
    n_obs = int((target.dividends_annual.dropna() > 0).sum())
    roe = target.roe
    if np.isfinite(roe) and roe > 0:
        g_sgr_arr = np.clip(roe * (1 - payout_arr), -0.02, 0.30)
        g_blend_arr = 0.5 * g_hist_raw + 0.5 * g_sgr_arr
    else:
        g_blend_arr = np.full(n_paths, g_hist_raw)
    # Bayesian shrinkage: (n × g_obs + k × g_prior) / (n + k)
    g_prior, prior_strength = 0.05, 5.0
    g_high_arr = (n_obs * g_blend_arr + prior_strength * g_prior) / (n_obs + prior_strength)
    g_high_arr = np.clip(g_high_arr, -0.02, 0.25)

    # --- Per-path terminal growth cap (min of GDP ceiling and Ke − 50 bps) ---
    ceiling = np.minimum(LONG_RUN_NOMINAL_GROWTH_IN, ke_arr - 0.005)
    g_term_capped = np.minimum(g_term_arr, ceiling)

    # --- Path-level invalid mask ---------------------------------------
    # Mirrors the guard clauses in `select_and_value` and the individual
    # variant constructors.
    invalid = (
        (ke_arr <= 0)
        | (payout_arr < 0.05)            # near-non-payer guard
        | (g_term_capped >= ke_arr)      # Gordon degenerate
    )

    model = base_iv.model

    if model == "Gordon Growth":
        d1 = d0 * (1 + g_term_capped)
        v = d1 / (ke_arr - g_term_capped)

    elif model == "H-Model":
        H = SETTINGS.transition_years     # = (transition_years * 2) / 2
        v = (
            d0 * (1 + g_term_capped)
            + d0 * H * (g_high_arr - g_term_capped)
        ) / (ke_arr - g_term_capped)

    elif model == "Three-Stage DDM":
        high_years = SETTINGS.high_growth_years
        fade_years = SETTINGS.transition_years

        # Stage 1 — explicit high growth, t = 1..high_years
        t1 = np.arange(1, high_years + 1)                          # (H,)
        growth1 = (1.0 + g_high_arr[:, None]) ** t1[None, :]       # (N, H)
        disc1 = (1.0 + ke_arr[:, None]) ** t1[None, :]             # (N, H)
        pv_stage1 = (d0 * growth1 / disc1).sum(axis=1)             # (N,)
        d_end_stage1 = d0 * (1.0 + g_high_arr) ** high_years       # (N,)

        # Stage 2 — linear fade, k = 1..fade_years
        fade_step_arr = (g_high_arr - g_term_capped) / (fade_years + 1)  # (N,)
        k_idx = np.arange(1, fade_years + 1)                       # (F,)
        # g_t at step k:  g_high − fade_step × k
        g_t = g_high_arr[:, None] - fade_step_arr[:, None] * k_idx[None, :]  # (N, F)
        # D_t evolves multiplicatively from d_end_stage1
        d_factors = np.cumprod(1.0 + g_t, axis=1)                  # (N, F)
        d_stage2 = d_end_stage1[:, None] * d_factors               # (N, F)
        t2 = high_years + k_idx                                    # (F,)
        disc2 = (1.0 + ke_arr[:, None]) ** t2[None, :]             # (N, F)
        pv_stage2 = (d_stage2 / disc2).sum(axis=1)                 # (N,)

        # Stage 3 — Gordon perpetuity discounted from end of fade
        d_end_fade = d_stage2[:, -1]                               # (N,)
        d_term_plus_one = d_end_fade * (1.0 + g_term_capped)       # (N,)
        terminal_v = d_term_plus_one / (ke_arr - g_term_capped)    # (N,)
        pv_terminal = terminal_v / (1.0 + ke_arr) ** (high_years + fade_years)

        v = pv_stage1 + pv_stage2 + pv_terminal

    else:
        # Two-Stage isn't picked by the selector, and "DDM (skipped)"
        # implies base_iv.valid==False (handled above). Defensive NaN.
        return np.full(n_paths, np.nan)

    return np.where(invalid, np.nan, v)


# ---------------------------------------------------------------------------
# Three-way Monte Carlo (Phase A) — parallel to monte_carlo_blended
# ---------------------------------------------------------------------------
# A separate function (rather than generalising monte_carlo_blended) so the
# two-way path stays bit-identical for non-FCFE-applicable tickers. The
# existing pipeline is the regression baseline; if I edit
# monte_carlo_blended directly, every numeric assertion in
# test_monte_carlo.py becomes a moving target. Cleaner to leave it alone
# and let the integrated valuation pick which MC runner to call based on
# whether FCFE is applicable.
def monte_carlo_three_way(
    target: StockBundle,
    coe: CostOfEquity,
    rel: RelativeValuation,
    fcfe: FCFEResult,
    base_g_terminal: float,
    w_ddm: float,
    w_fcfe: float,
    w_rel: float,
    *,
    n_paths: int = SETTINGS.mc_paths,
    seed: int = SETTINGS.mc_seed,
) -> MonteCarloResult:
    """Vectorised three-way Monte Carlo over DDM + FCFE + Relative legs.

    Reuses the existing four distributions for Ke, terminal g, payout,
    and the per-multiple relative sample (do NOT introduce new
    distributions for these — Phase A spec). Adds three FCFE-specific
    per-path distributions:

      * CapEx / Revenue ratio: truncated normal, σ = 15% of mean,
        lower-bounded at 0.
      * D&A / Revenue ratio: truncated normal, σ = 10% of mean,
        lower-bounded at 0.
      * Debt ratio: tight Beta around the historical mean with
        α + β = 50 (sharper than the payout Beta — DR moves slowly
        for established firms).

    Stage-1 growth is recomputed per path from the same Bayes-shrunk
    blend the deterministic FCFE engine uses; it shares the existing
    growth distribution rather than introducing a new one.

    The blended sample on each path is
        w_ddm * V_DDM_path + w_fcfe * V_FCFE_path + w_rel * V_Rel_path
    with ``np.where`` fallback to whichever legs are finite on a
    given path, so a path where (say) FCFE collapses (g → ke) still
    contributes a sensible DDM+Rel blended value.
    """
    rng = np.random.default_rng(seed)

    # ---- Re-use existing distributions for Ke, g_term, payout, rel ----
    ke_draws = np.clip(rng.normal(coe.ke, 0.008, n_paths), 0.04, 0.25)
    g_term_draws = np.clip(rng.normal(base_g_terminal, 0.004, n_paths), 0.0, 0.06)
    p = max(0.05, min(0.95, target.payout_ratio if target.payout_ratio else 0.30))
    a = p * 30
    b = (1 - p) * 30
    payout_draws = rng.beta(a, b, n_paths)

    # ---- DDM track (reuse existing vectorised closed-form) -----------
    if not fcfe.valid:
        # Branch 1 (two-way fallback) — the caller normally would have
        # routed to monte_carlo_blended. Defensive: forward to the same
        # math by treating FCFE samples as NaN throughout.
        base_iv = select_and_value(
            eps_ttm=target.eps_ttm,
            dps_ttm=target.dividend_per_share_ttm,
            payout_ratio=target.payout_ratio,
            roe=target.roe,
            ke=coe.ke,
            historical_dps=target.dividends_annual,
            historical_eps=target.earnings_annual,
            sector_g_terminal=base_g_terminal,
            payout_ratio_raw=getattr(target, "payout_ratio_raw", float("nan")),
        )
        ddm_samples = _vectorised_ddm_samples(
            target, base_iv, ke_draws, g_term_draws, payout_draws,
        )
        fcfe_samples = np.full(n_paths, np.nan)
    else:
        base_iv = select_and_value(
            eps_ttm=target.eps_ttm,
            dps_ttm=target.dividend_per_share_ttm,
            payout_ratio=target.payout_ratio,
            roe=target.roe,
            ke=coe.ke,
            historical_dps=target.dividends_annual,
            historical_eps=target.earnings_annual,
            sector_g_terminal=base_g_terminal,
            payout_ratio_raw=getattr(target, "payout_ratio_raw", float("nan")),
        )
        ddm_samples = _vectorised_ddm_samples(
            target, base_iv, ke_draws, g_term_draws, payout_draws,
        )
        fcfe_samples = _vectorised_fcfe_samples(
            target, fcfe, ke_draws, g_term_draws, rng,
        )

    # ---- Relative track (reuse) --------------------------------------
    rel_base = rel.weighted_value
    if np.isfinite(rel_base) and rel_base > 0:
        log_mu = np.log(rel_base)
        rel_samples = rng.lognormal(log_mu - 0.5 * 0.20**2, 0.20, n_paths)
    else:
        rel_samples = np.full(n_paths, np.nan)

    # ---- Blend per path with NaN-aware fallback ----------------------
    # When a leg is NaN on a particular path, redistribute its weight
    # to the surviving legs proportionally so the total still sums to
    # the same effective intrinsic value.
    blended = _blend_three_way_vectorised(
        ddm_samples, fcfe_samples, rel_samples,
        w_ddm, w_fcfe, w_rel,
    )

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


def _vectorised_fcfe_samples(
    target: StockBundle,
    base_fcfe: FCFEResult,
    ke_arr: np.ndarray,
    g_term_arr: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Closed-form two-stage FCFE evaluated across all MC paths.

    Mirrors the structure of ``_vectorised_ddm_samples``: pins the
    central inputs from ``base_fcfe.inputs`` and broadcasts the
    closed-form math across path arrays. Per-path noise on CapEx/Rev,
    D&A/Rev, and DR is sampled here; growth and Ke come from the
    shared draws so all three legs are jointly correlated to the
    same scenario.
    """
    n_paths = ke_arr.shape[0]
    inputs = base_fcfe.inputs
    stage1_years = int(inputs["stage1_years"])
    taper_years = SETTINGS.fcfe_taper_years
    taper_start = stage1_years - taper_years
    g1_base = float(inputs["g1"])
    dr_base = float(inputs["dr"])
    ni0 = float(inputs["ni0"])
    capex0 = float(inputs["capex0"])
    da0 = float(inputs["da0"])
    wc0 = float(inputs["wc0_yf"])
    shares = float(inputs["shares_outstanding"])
    revenue = float(target.revenue) if (target.revenue and np.isfinite(target.revenue)) else float("nan")

    # ---- Per-path noise on CapEx/Rev, D&A/Rev, DR --------------------
    if np.isfinite(revenue) and revenue > 0:
        capex_ratio_mean = capex0 / revenue
        da_ratio_mean = da0 / revenue
        capex_ratio = np.clip(
            rng.normal(capex_ratio_mean, 0.15 * abs(capex_ratio_mean), n_paths),
            0.0, None,
        )
        da_ratio = np.clip(
            rng.normal(da_ratio_mean, 0.10 * abs(da_ratio_mean), n_paths),
            0.0, None,
        )
        capex_draws = capex_ratio * revenue
        da_draws = da_ratio * revenue
    else:
        # Revenue missing — fall back to direct ±15%/±10% noise on the
        # latest-year values themselves. Avoids dropping the path.
        capex_draws = np.clip(
            rng.normal(capex0, 0.15 * abs(capex0), n_paths), 0.0, None,
        )
        da_draws = np.clip(
            rng.normal(da0, 0.10 * abs(da0), n_paths), 0.0, None,
        )

    # DR: tight Beta around the historical mean (α + β = 50).
    if 0 < dr_base < 1:
        a = dr_base * 50
        b = (1 - dr_base) * 50
        dr_draws = rng.beta(a, b, n_paths)
    else:
        dr_draws = np.full(n_paths, dr_base)
    # Re-apply the engine cap on each path so the (1−DR) multiplier
    # stays bounded.
    dr_draws = np.clip(dr_draws, 0.0, SETTINGS.fcfe_debt_ratio_cap)

    # ---- Terminal growth caps -----------------------------------------
    ceiling = np.minimum(LONG_RUN_NOMINAL_GROWTH_IN, ke_arr - SETTINGS.fcfe_ke_margin)
    g_term_capped = np.clip(np.minimum(g_term_arr, ceiling), 0.0, None)

    # Path-level invalid mask (mirrors the engine's domain checks)
    invalid = (ke_arr <= 0) | (g_term_capped >= ke_arr - 1e-9)

    # ---- Base-year FCFE per path -------------------------------------
    # Match the engine's formula: FCFE_0 = NI − (CapEx − D&A)(1−DR)
    #                                       + ΔWC_yf (1−DR)
    one_minus_dr = 1.0 - dr_draws
    base_fcfe_arr = ni0 - (capex_draws - da_draws) * one_minus_dr + wc0 * one_minus_dr

    # ---- Stage 1: per-year compounding with taper --------------------
    fcfe_t = base_fcfe_arr.copy()
    pv_explicit = np.zeros(n_paths)
    for t in range(1, stage1_years + 1):
        if t <= taper_start:
            g_t = g1_base  # scalar
        else:
            taper_progress = (t - taper_start) / taper_years
            g_t = g1_base + (g_term_capped - g1_base) * taper_progress  # (n_paths,)
        fcfe_t = fcfe_t * (1 + g_t)
        disc = (1 + ke_arr) ** t
        pv_explicit += fcfe_t / disc

    # ---- Stage 2: Gordon terminal ------------------------------------
    fcfe_terminal_plus_one = fcfe_t * (1 + g_term_capped)
    terminal_v = fcfe_terminal_plus_one / (ke_arr - g_term_capped)
    pv_tv = terminal_v / ((1 + ke_arr) ** stage1_years)

    total_equity = pv_explicit + pv_tv
    per_share = total_equity / shares if shares > 0 else np.full(n_paths, np.nan)

    return np.where(invalid, np.nan, per_share)


def _blend_three_way_vectorised(
    ddm_s: np.ndarray, fcfe_s: np.ndarray, rel_s: np.ndarray,
    w_ddm: float, w_fcfe: float, w_rel: float,
) -> np.ndarray:
    """Per-path three-way blend with NaN-aware weight redistribution.

    On paths where one or more legs are NaN, the surviving legs'
    weights are renormalised so the path still contributes a
    sensible blended value rather than dropping out.
    """
    n_paths = ddm_s.shape[0]
    d_ok = np.isfinite(ddm_s)
    f_ok = np.isfinite(fcfe_s)
    r_ok = np.isfinite(rel_s)

    # Build per-path effective weights with NaN-aware renormalisation.
    w_d = np.where(d_ok, w_ddm, 0.0)
    w_f = np.where(f_ok, w_fcfe, 0.0)
    w_r = np.where(r_ok, w_rel, 0.0)
    total = w_d + w_f + w_r
    safe_total = np.where(total > 0, total, 1.0)  # avoid /0; paths with total=0 → NaN below
    w_d = w_d / safe_total
    w_f = w_f / safe_total
    w_r = w_r / safe_total

    # Replace NaNs with 0 before the weighted sum (the corresponding
    # weights are already 0 by construction).
    ddm_clean = np.where(d_ok, ddm_s, 0.0)
    fcfe_clean = np.where(f_ok, fcfe_s, 0.0)
    rel_clean = np.where(r_ok, rel_s, 0.0)

    blended = w_d * ddm_clean + w_f * fcfe_clean + w_r * rel_clean
    blended = np.where(total > 0, blended, np.nan)
    return blended
