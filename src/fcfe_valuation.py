"""
Free-Cash-Flow-to-Equity (FCFE) valuation — Phase A.

Two-stage closed-form FCFE intrinsic value, used as the third leg in
the blended valuation pipeline. Motivated by the fact that the four
DDM variants discount dividends; for ultra-low-payout firms (e.g.
SIEMENS at ~0.3%) the dividend stream is tiny in absolute terms
regardless of growth fade, and the existing engine forces the
relative leg to carry 100% of the load. FCFE values the cash flow
actually available to equity holders — earnings reinvested at the
firm's marginal returns — and therefore prices retained-and-deployed
cash that pure-DDM cannot see.

Cash-flow attribution
---------------------
Per the Damodaran two-stage formulation, with the firm maintaining a
constant debt ratio so net borrowing tracks net capex + ΔWC:

    FCFE_t = NI_t − (CapEx_t − D&A_t) × (1 − DR)
                  + change_in_wc_annual_t × (1 − DR)

The ``+ change_in_wc_annual`` term is the consequence of yfinance's
cash-flow-statement sign convention (positive = cash freed by WC
reduction). The Damodaran textbook writes the asset-side counterpart
as ``− ΔWC × (1 − DR)`` using a positive-means-WC-increase reading;
yfinance has the opposite sign, so the formula that consumes
``StockBundle.change_in_wc_annual`` adds the raw value rather than
subtracting it.

ΔWC sign verification (carried forward from Phase A.0)
------------------------------------------------------
The convention was pinned down empirically by checking the operating-
cash-flow identity ``OCF ≈ NI + D&A + ΔWC + Other`` across four firm-
years. yfinance ΔWC must be ADDED (with its raw sign) to recover the
reported OCF; subtracting it does not. TCS FY24 cross-checks
specifically: reported net income ₹46,099 cr (matches the published
TCS FY24 annual report exactly), reported OCF ~₹44,338 cr, and
``ni + da + wc_yf + other`` lands within typical D&A/disposal noise
of OCF whereas ``ni + da − wc_yf + other`` is off by ~60%. See
``scripts/verify_wc_sign.py`` for the reproducible probe; future
maintainers should re-run it if yfinance's row schema ever drifts.

Reuses (do not re-implement)
----------------------------
  * Ke from ``src.cost_of_equity`` (CAPM with 5y weekly β regression,
    Bloomberg β adjustment, sector unlevered fallback, small-cap
    premium). The integrated pipeline already computes Ke once and
    passes it in.
  * ``bayesian_growth_shrinkage`` from ``src.ddm_models`` for stage-1
    growth shrinkage toward the 5% long-run prior. Short EPS series
    get pulled harder toward the prior; long series barely move.
  * Terminal growth cap from ``config.LONG_RUN_NOMINAL_GROWTH_IN``
    (currently 6%) and the project-wide ``ke − fcfe_ke_margin`` floor.

Applicability (directive #2 from Phase A.0 sign-off)
----------------------------------------------------
Financials (Banking, NBFC, Insurance) are excluded at the model layer
regardless of whether the data populates. FCFE is not the right model
for banks/insurers — they need FCFF or regulated-capital DDM. The
data layer does not enforce this exclusion; the model layer does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import LONG_RUN_NOMINAL_GROWTH_IN, SETTINGS
from src.data_fetcher import StockBundle
from src.ddm_models import bayesian_growth_shrinkage, sustainable_growth_rate


# Financial-sector tags excluded from FCFE.
# "Insurance" is reserved for future taxonomy extension; today's
# listed insurers (SBILIFE.NS, HDFCLIFE.NS) carry the "NBFC" tag in
# DEFAULT_UNIVERSE, so they're already covered by the NBFC entry.
_FINANCIAL_SECTORS = frozenset({"Banking", "NBFC", "Insurance"})


@dataclass
class FCFEResult:
    """Mirror of ``IntrinsicValue`` for the FCFE leg.

    ``valid=False`` carries the human-readable reason in
    ``reason_invalid``; the UI surfaces this on the FCFE detail tab so
    users see *why* FCFE was skipped rather than a silent NaN.
    """

    value_per_share: float
    valid: bool
    reason_invalid: Optional[str] = None
    inputs: dict = field(default_factory=dict)
    year_by_year: pd.DataFrame = field(default_factory=pd.DataFrame)
    terminal_value: float = float("nan")
    pv_terminal: float = float("nan")

    def __repr__(self) -> str:
        if not self.valid:
            return f"<FCFEResult invalid: {self.reason_invalid}>"
        return f"<FCFEResult ₹{self.value_per_share:,.2f}>"


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------
def fcfe_applicable(bundle: StockBundle) -> tuple[bool, Optional[str]]:
    """Decide whether FCFE valuation is well-defined for this bundle.

    Returns ``(applicable, reason_invalid)``; ``reason_invalid`` is
    ``None`` when applicable is True.

    Implements Phase A directives:
      * #2 — sector-level financials exclusion. Checked BEFORE any
        data-shape gates so a populated bundle in the Banking sector
        still gets correctly skipped.
      * #1 — ≥3y history threshold on CapEx, D&A, and NI series. The
        Phase A.0 audit found a median 4y coverage; a 5y floor would
        over-exclude.

    Negative-NI guard
    -----------------
    A 3y average NI ≤ 0 means the firm has been loss-making on
    average; the FCFE base in such cases would be negative even
    before WC/CapEx adjustments, and stage-1 growth is undefined.
    These names belong on the relative-valuation track exclusively.

    EPS CAGR guard
    --------------
    Stage-1 growth seeds off the historical EPS CAGR. If the
    chronological endpoints disagree on sign (or the start/end are
    non-positive) the CAGR is mathematically undefined; we return
    early rather than propagate NaN through the perpetuity math.
    """
    # Directive #2: financials excluded at the model layer.
    if bundle.sector in _FINANCIAL_SECTORS:
        return False, "FCFE not applicable to financial firms"

    # Directive #1: ≥3y history on each required series.
    min_years = SETTINGS.fcfe_min_history_years
    for name, msg in (
        ("capex_annual", f"Insufficient capex history (<{min_years}y)"),
        ("dep_amort_annual", f"Insufficient D&A history (<{min_years}y)"),
        ("net_income_annual", f"Insufficient NI history (<{min_years}y)"),
    ):
        series = getattr(bundle, name, pd.Series(dtype=float))
        if len(series.dropna()) < min_years:
            return False, msg

    # All-zero series is a different failure mode than empty — the
    # data layer populated rows of zeros (typical when a fallback
    # branch fired with 0-fill defaults). Treat as missing.
    for name in ("capex_annual", "dep_amort_annual", "net_income_annual"):
        series = getattr(bundle, name).dropna()
        if not series.empty and (series == 0).all():
            return False, "Required cash-flow series missing"

    # 3y average NI must be positive.
    ni = bundle.net_income_annual.dropna()
    if ni.tail(3).mean() <= 0:
        return False, "Average net income non-positive"

    # EPS CAGR must be positive and defined.
    eps = bundle.earnings_annual.dropna()
    if len(eps) < min_years:
        return False, f"Insufficient EPS history for growth signal (<{min_years}y)"
    if eps.iloc[0] <= 0 or eps.iloc[-1] <= 0:
        return False, "EPS CAGR undefined (non-positive endpoint)"
    cagr = (eps.iloc[-1] / eps.iloc[0]) ** (1.0 / (len(eps) - 1)) - 1
    if cagr <= 0:
        return False, "EPS CAGR non-positive"

    return True, None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _historical_eps_cagr(eps_series: pd.Series, cap: float) -> float:
    """5y EPS CAGR, capped. NaN when not computable.

    Caller (``fcfe_value``) has already gated on the same conditions
    via ``fcfe_applicable``; this helper is defensive in case it is
    called from a different code path (e.g. the Monte Carlo runner
    perturbing inputs).
    """
    s = eps_series.dropna()
    if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return float("nan")
    n = len(s) - 1
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1.0 / n) - 1
    return float(min(cagr, cap))


def _debt_ratio(bundle: StockBundle) -> float:
    """5y average D/(D+E), capped at SETTINGS.fcfe_debt_ratio_cap.

    Uses ``total_debt_annual`` when available; falls back to the
    scalar ``total_debt`` if the series is empty (older cached
    bundles may have the scalar but not the series).
    """
    cap = SETTINGS.fcfe_debt_ratio_cap
    debt_series = bundle.total_debt_annual.dropna()
    if debt_series.empty:
        debt_avg = float(bundle.total_debt) if bundle.total_debt else 0.0
    else:
        debt_avg = float(debt_series.tail(5).mean())

    equity = float(bundle.equity) if bundle.equity else 0.0
    if equity <= 0:
        # Negative-equity firms can't sustain a meaningful debt ratio;
        # use the cap so net-borrowing math doesn't go off the rails.
        return cap
    dr = debt_avg / (debt_avg + equity)
    return float(min(max(dr, 0.0), cap))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fcfe_value(
    bundle: StockBundle,
    ke: float,
    terminal_g: float,
    stage1_years: int = SETTINGS.fcfe_stage1_years,
) -> FCFEResult:
    """Two-stage FCFE valuation, per share.

    Stage 1 (explicit forecast)
    ---------------------------
    ``stage1_years`` (default 7) years of explicit FCFE. The first
    ``stage1_years − fcfe_taper_years`` years grow at ``g1``; the
    final ``fcfe_taper_years`` years fade linearly from ``g1`` to
    the terminal growth rate. This keeps a smooth glide path into
    perpetuity (no two-stage cliff) without the closed-form
    awkwardness of an H-Model on a cash-flow series.

    ``g1`` is the Bayes-shrunk historical EPS CAGR, capped at
    ``SETTINGS.fcfe_g1_cap`` (15%) before shrinkage. The Bayesian
    shrinkage uses the same 5% prior and prior_strength=5 as the DDM
    module so the two legs agree on growth.

    Stage 2 (Gordon perpetuity)
    ---------------------------
    Discount FCFE_{n+1} / (Ke − g) at the end of year ``stage1_years``,
    pulled back to t=0.

    Caps and floors
    ---------------
      * ``g_terminal`` capped at ``min(LONG_RUN_NOMINAL_GROWTH_IN,
        ke − fcfe_ke_margin)`` and floored at 0 (no perpetual
        shrinkage).
      * ``g1`` capped at ``fcfe_g1_cap``.
      * ``DR`` capped at ``fcfe_debt_ratio_cap``.

    Parameters
    ----------
    bundle : StockBundle
        The target firm. Must satisfy ``fcfe_applicable``; otherwise
        an invalid FCFEResult is returned with the reason populated.
    ke : float
        Cost of equity (decimal). Pass the value computed by the
        existing ``cost_of_equity`` helper — do not recompute β.
    terminal_g : float
        Terminal growth rate (decimal). Capped internally.
    stage1_years : int, optional
        Length of the explicit forecast. Defaults to
        ``SETTINGS.fcfe_stage1_years`` (7).

    Returns
    -------
    FCFEResult
        ``valid=False`` with a human-readable ``reason_invalid`` when
        applicability fails; otherwise per-share intrinsic value plus
        the full year-by-year worksheet for the UI.
    """
    if stage1_years < SETTINGS.fcfe_min_history_years:
        raise ValueError(
            f"stage1_years must be >= {SETTINGS.fcfe_min_history_years}; "
            f"got {stage1_years}"
        )

    ok, reason = fcfe_applicable(bundle)
    if not ok:
        return FCFEResult(
            value_per_share=float("nan"),
            valid=False,
            reason_invalid=reason,
        )

    # ----- Latest-year FCFE components -----
    ni0 = float(bundle.net_income_annual.iloc[-1])
    capex0 = float(bundle.capex_annual.iloc[-1])
    da0 = float(bundle.dep_amort_annual.iloc[-1])
    wc_series = bundle.change_in_wc_annual.dropna()
    # change_in_wc_annual may be empty when yfinance lacks that row
    # for a sparse-history ticker. The other applicability checks have
    # already passed, so absent ΔWC just means we model the firm as if
    # WC was flat YoY — a conservative assumption that under-states
    # FCFE for growth firms (their WC builds drain cash) and over-
    # states it for harvest firms.
    wc0 = float(wc_series.iloc[-1]) if not wc_series.empty else 0.0

    # ----- Debt ratio -----
    dr = _debt_ratio(bundle)

    # ----- Stage-1 growth (Bayes-shrunk blend of EPS CAGR and SGR) ----
    # Mirrors the DDM module's blend recipe (``select_and_value`` in
    # src/ddm_models.py): combine historical CAGR with the sustainable
    # growth identity ``g = ROE × (1 − payout)`` before Bayesian
    # shrinkage. Without this, FCFE shrinks only from the historical
    # observation while DDM shrinks from a CAGR + SGR blend — for a
    # firm like TCS (long DPS history at 21% CAGR, short EPS history
    # at 5% CAGR, ROE 48%) the two legs end up using stage-1 growth
    # rates ~15 pp apart, which produces a methodologically incoherent
    # blend regardless of which is "more accurate". Aligning the
    # recipe leaves the conservatism that genuinely belongs to FCFE
    # (the EPS-CAGR signal is shorter than DPS history and typically
    # lower) intact, but removes the silent disagreement on what
    # growth assumption is being used.
    g_eps_cagr = _historical_eps_cagr(bundle.earnings_annual, cap=SETTINGS.fcfe_g1_cap)
    if not np.isfinite(g_eps_cagr):
        # Defensive — fcfe_applicable already gates this, but the
        # check is cheap and the alternative is NaN propagation.
        return FCFEResult(
            value_per_share=float("nan"),
            valid=False,
            reason_invalid="EPS CAGR undefined",
        )

    if (np.isfinite(bundle.roe) and bundle.roe > 0
            and np.isfinite(bundle.payout_ratio)):
        g_sgr = sustainable_growth_rate(bundle.roe, bundle.payout_ratio)
        g_observed = 0.5 * g_eps_cagr + 0.5 * g_sgr
    else:
        # Mirror the DDM fallback: when ROE is missing or non-positive
        # (distressed / loss-making firm), fall back to the historical
        # observation alone rather than contaminating the blend with
        # a negative-ROE × positive-retention SGR.
        g_sgr = None
        g_observed = g_eps_cagr

    n_eps_obs = int(bundle.earnings_annual.dropna().shape[0])
    g1 = bayesian_growth_shrinkage(g_observed, n_observations=n_eps_obs)
    g1 = float(min(g1, SETTINGS.fcfe_g1_cap))

    # ----- Terminal growth caps -----
    g_terminal = min(
        terminal_g,
        LONG_RUN_NOMINAL_GROWTH_IN,
        ke - SETTINGS.fcfe_ke_margin,
    )
    g_terminal = max(g_terminal, 0.0)

    # ----- Base-year FCFE -----
    # FCFE_0 = NI − (CapEx − D&A)(1−DR) + ΔWC_yf (1−DR)
    # See module docstring for the sign-convention derivation.
    base_fcfe = ni0 - (capex0 - da0) * (1 - dr) + wc0 * (1 - dr)

    # ----- Explicit forecast (stage 1) -----
    taper_start = stage1_years - SETTINGS.fcfe_taper_years
    pv_explicit = 0.0
    fcfe_t = base_fcfe
    yby_rows = []
    for t in range(1, stage1_years + 1):
        if t <= taper_start:
            g_t = g1
        else:
            # Linear taper from g1 (at taper_start) to g_terminal
            # (at stage1_years).
            taper_progress = (t - taper_start) / SETTINGS.fcfe_taper_years
            g_t = g1 + (g_terminal - g1) * taper_progress
        fcfe_t = fcfe_t * (1 + g_t)
        pv = fcfe_t / ((1 + ke) ** t)
        pv_explicit += pv
        yby_rows.append({
            "year": t,
            "growth": g_t,
            "fcfe": fcfe_t,
            "pv": pv,
            "pv_cumulative": pv_explicit,
        })

    # ----- Stage 2 (Gordon perpetuity) -----
    fcfe_terminal_plus_one = fcfe_t * (1 + g_terminal)
    terminal_v = fcfe_terminal_plus_one / (ke - g_terminal)
    pv_tv = terminal_v / ((1 + ke) ** stage1_years)

    # ----- Per-share -----
    shares = float(bundle.shares_outstanding)
    if shares <= 0:
        return FCFEResult(
            value_per_share=float("nan"),
            valid=False,
            reason_invalid="Shares outstanding non-positive",
        )
    total_value = pv_explicit + pv_tv
    per_share = total_value / shares

    return FCFEResult(
        value_per_share=per_share,
        valid=True,
        inputs={
            "ni0": ni0,
            "capex0": capex0,
            "da0": da0,
            "wc0_yf": wc0,
            "base_fcfe": base_fcfe,
            "g_eps_cagr": g_eps_cagr,
            "g_sgr": g_sgr,
            "g1": g1,
            "g_terminal": g_terminal,
            "ke": ke,
            "dr": dr,
            "stage1_years": stage1_years,
            "pv_explicit": pv_explicit,
            "pv_terminal": pv_tv,
            "shares_outstanding": shares,
        },
        year_by_year=pd.DataFrame(yby_rows),
        terminal_value=terminal_v,
        pv_terminal=pv_tv,
    )
