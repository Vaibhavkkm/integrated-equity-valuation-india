"""
Quality scoring — feeds the credibility-weighted blender.

We compute four independent scores and fold them into a single 0-100
quality number:

  * Piotroski F-Score (0-9)        — fundamental health.
  * Altman Z'' (modified for EM)   — distress likelihood.
  * Dividend Quality Score (0-10)  — track record + payout sanity.
  * Earnings Momentum (0-10)       — recent quarterly trend vs peers.

Why bother? Because the 50/50 DDM-vs-Relative weight in the original
project brief is brittle. If a stock has paid dividends for two years
straight after a long drought, blindly trusting the DDM is silly. By
nudging the blend toward whichever model is more defensible per stock,
we get a more robust intrinsic value — and the supervisor gets a clean
audit trail of *why* the weights moved.

Earnings momentum was added as the fourth pillar because the long-run
DDM/Relative tracks alone can't see whether the firm's *recent* quarters
support the thesis. It's a confidence-modulating overlay (it does not
move the price target itself) — the long-horizon valuation is still
driven by Bayes-shrunk historical growth in the DDM and peer multiples
in the relative track.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.data_fetcher import StockBundle
from src.earnings_momentum import EarningsMomentum


@dataclass
class QualityScore:
    piotroski_f: int
    altman_z: float
    dividend_quality: int               # 0-10
    composite: float                    # 0-100
    breakdown: dict
    earnings_momentum: int = 0          # 0-10; defaults to 0 when momentum
                                        # data is unavailable (callers can
                                        # construct without passing it).
    momentum_detail: Optional[EarningsMomentum] = field(default=None)


# ---------------------------------------------------------------------------
# Piotroski F-Score (9 binary signals; we approximate with available data)
# ---------------------------------------------------------------------------
def piotroski_score(b: StockBundle) -> tuple[int, dict]:
    """Approximate Piotroski with the data Yahoo gives us.

    The strict implementation needs Operating CFO and accruals which Yahoo
    sometimes drops. Where a check is impossible, we abstain (the firm
    neither gains nor loses the point), which is conservative.

    Parameters
    ----------
    b : StockBundle
        Target firm. The function reads ``net_income``,
        ``free_cash_flow_per_share_ttm``, ``earnings_annual``,
        ``revenue_annual``, ``total_assets``, and ``shares_outstanding``.

    Returns
    -------
    tuple[int, dict]
        Capped 0-9 score plus a breakdown dict mapping each signal name
        to its individual contribution (or ``"—"`` when abstained).
    """
    breakdown: dict = {}
    score = 0

    # 1. Positive ROA (net income > 0 is the easy proxy)
    pos_roa = b.net_income > 0
    breakdown["positive_net_income"] = int(pos_roa)
    score += int(pos_roa)

    # 2. Positive operating CFO (proxied by positive FCF/share)
    pos_cfo = np.isfinite(b.free_cash_flow_per_share_ttm) and b.free_cash_flow_per_share_ttm > 0
    breakdown["positive_fcf"] = int(pos_cfo)
    score += int(pos_cfo)

    # 3. Improving ROA
    e = b.earnings_annual.dropna()
    if len(e) >= 2 and np.isfinite(b.total_assets) and b.total_assets > 0:
        roa_now = e.iloc[-1] / b.total_assets
        roa_prev = e.iloc[-2] / b.total_assets
        improving_roa = roa_now > roa_prev
        breakdown["improving_roa"] = int(improving_roa)
        score += int(improving_roa)
    else:
        breakdown["improving_roa"] = "—"

    # 4. CFO > Net Income (quality of earnings)
    if pos_cfo and pos_roa and np.isfinite(b.shares_outstanding):
        cfo_total = b.free_cash_flow_per_share_ttm * b.shares_outstanding
        cfo_gt_ni = cfo_total > b.net_income
        breakdown["cfo_gt_ni"] = int(cfo_gt_ni)
        score += int(cfo_gt_ni)
    else:
        breakdown["cfo_gt_ni"] = "—"

    # 5. Debt declining (lower D/E than prior, proxy)
    # Without prior-year debt we abstain.
    breakdown["declining_debt"] = "—"

    # 6. Current ratio improving — abstain (no current ratio in bundle)
    breakdown["current_ratio_improving"] = "—"

    # 7. No new shares issued — abstain
    breakdown["no_dilution"] = "—"

    # 8. Improving gross margin — proxy via revenue growth > debt growth?
    rev = b.revenue_annual.dropna()
    if len(rev) >= 2 and rev.iloc[-2] > 0:
        improving_rev = rev.iloc[-1] > rev.iloc[-2]
        breakdown["growing_revenue"] = int(improving_rev)
        score += int(improving_rev)
    else:
        breakdown["growing_revenue"] = "—"

    # 9. Improving asset turnover — abstain
    breakdown["asset_turnover_improving"] = "—"

    return int(min(9, score)), breakdown


# ---------------------------------------------------------------------------
# Altman Z'' (emerging-market variant) — works for non-manufacturing firms
# ---------------------------------------------------------------------------
def altman_z_em(b: StockBundle) -> float:
    """Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4 + 3.25.

    Where:
      X1 = Working Capital / Total Assets   (we proxy WC ≈ Cash − ST debt;
           a very rough approximation given Yahoo's data.)
      X2 = Retained Earnings / Total Assets (proxied by Equity / TA)
      X3 = EBIT / Total Assets              (proxied by EBITDA / TA)
      X4 = Book Equity / Total Liabilities

    Z'' > 2.6 → safe; 1.1 < Z'' < 2.6 → grey; Z'' < 1.1 → distress.

    Parameters
    ----------
    b : StockBundle
        Target firm. Reads ``cash``, ``total_debt``, ``equity``,
        ``ebitda``, and ``total_assets``.

    Returns
    -------
    float
        Z'' score, or ``NaN`` when ``total_assets`` is missing or
        non-positive (the score has no meaningful interpretation in
        either case).
    """
    if not (np.isfinite(b.total_assets) and b.total_assets > 0):
        return float("nan")

    x1 = (b.cash - b.total_debt * 0.3) / b.total_assets   # crude WC proxy
    x2 = (b.equity / b.total_assets) if b.equity else 0.0
    x3 = (b.ebitda / b.total_assets) if np.isfinite(b.ebitda) else 0.0
    total_liab = b.total_assets - b.equity if b.equity else b.total_debt
    x4 = (b.equity / total_liab) if (total_liab and total_liab > 0) else 0.0

    return float(6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4 + 3.25)


# ---------------------------------------------------------------------------
# Dividend quality (0-10)
# ---------------------------------------------------------------------------
def dividend_quality(b: StockBundle) -> tuple[int, dict]:
    """Score a payer's dividend track record (0-10).

    Combines five sub-signals: pays currently, multi-year history,
    absence of major cuts, payout sustainability, and dividend growth.

    Parameters
    ----------
    b : StockBundle
        Reads ``dividend_per_share_ttm``, ``dividends_annual``, and
        ``payout_ratio``.

    Returns
    -------
    tuple[int, dict]
        Capped 0-10 score plus a breakdown dict naming each sub-signal.
        Non-payers short-circuit to ``(0, {"pays_currently": 0})``.
    """
    breakdown: dict = {}
    score = 0

    # 1. Has paid dividends in the most recent year
    if b.dividend_per_share_ttm > 0:
        score += 2
        breakdown["pays_currently"] = 2
    else:
        return 0, {"pays_currently": 0}

    dps = b.dividends_annual.dropna()
    n_years = (dps > 0).sum()

    # 2. Multi-year track record
    if n_years >= 10:
        score += 3
        breakdown["track_record"] = 3
    elif n_years >= 5:
        score += 2
        breakdown["track_record"] = 2
    elif n_years >= 3:
        score += 1
        breakdown["track_record"] = 1
    else:
        breakdown["track_record"] = 0

    # 3. No major cuts in last 5 years.
    # The relative-cut test (`diff < -prev * 0.20`) misses outright
    # suspensions because the diff goes from prev → 0 and the threshold
    # `-prev * 0.20` is typically larger in magnitude. Count year-on-year
    # transitions that drop ≥20% — including suspensions to zero — by
    # working with the ratio `curr / prev` directly.
    recent = dps[-5:] if len(dps) >= 5 else dps
    if len(recent) >= 2:
        prev = recent.shift()
        ratio = recent / prev
        # Drop the first row (NaN from shift) and any row where prev<=0
        cuts_mask = (prev > 0) & (ratio < 0.80)
        cuts = int(cuts_mask.sum())
    else:
        cuts = 0
    if cuts == 0:
        score += 2
        breakdown["no_major_cuts"] = 2
    elif cuts == 1:
        score += 1
        breakdown["no_major_cuts"] = 1
    else:
        breakdown["no_major_cuts"] = 0

    # 4. Sustainable payout (< 80%). Use the *raw* (uncapped) payout —
    # the modelling field tops out at 1.0 and would silently treat a
    # debt-funded 190% extraction the same as a healthy 100% payer.
    payout_for_quality = getattr(b, "payout_ratio_raw", float("nan"))
    if not np.isfinite(payout_for_quality):
        payout_for_quality = b.payout_ratio
    if 0 < payout_for_quality < 0.80:
        score += 2
        breakdown["sustainable_payout"] = 2
    elif payout_for_quality < 1.0:
        score += 1
        breakdown["sustainable_payout"] = 1
    else:
        breakdown["sustainable_payout"] = 0

    # 5. Growing dividend stream
    if len(dps) >= 5:
        if dps.iloc[-1] > dps.iloc[-5]:
            score += 1
            breakdown["growing"] = 1
        else:
            breakdown["growing"] = 0

    return int(min(10, score)), breakdown


# ---------------------------------------------------------------------------
# Public composite
# ---------------------------------------------------------------------------
def quality_score(
    b: StockBundle,
    *,
    momentum: Optional[EarningsMomentum] = None,
) -> QualityScore:
    """Composite 0-100 quality score for ``b``.

    Blends Piotroski (30%), Altman Z'' (25%), dividend quality (25%), and
    earnings momentum (20%) on a calibration that still maps an "average
    firm" to roughly 50. Earlier versions used 35/30/35 without a
    momentum pillar; the rebalance shaves ~5 pp from each existing
    pillar to make room without inflating the composite.

    When ``momentum`` is omitted (the integrated pipeline always passes
    one; standalone callers may not), the momentum pillar contributes
    its mid-point to keep the composite calibration symmetric — that is,
    not knowing momentum should not look like *bad* momentum.

    Parameters
    ----------
    b : StockBundle
        Target firm — passed through to each sub-scorer.
    momentum : EarningsMomentum, optional
        Pre-computed momentum overlay (see ``earnings_momentum`` module).
        Pass when peer set is available so peer-comparison signals fire.

    Returns
    -------
    QualityScore
        Composite plus the underlying Piotroski / Altman / dividend /
        momentum components and their per-signal breakdowns.
    """
    f, fbreak = piotroski_score(b)
    z = altman_z_em(b)
    dq, dqbreak = dividend_quality(b)

    # Normalise to 0-100. Rough calibration that maps "average firm" → ~50.
    f_norm = (f / 9.0) * 30
    if not np.isfinite(z):
        z_norm = 12.5  # mid-point of the 25 pp band
    else:
        z_norm = float(np.clip((z - 1.1) / (2.6 - 1.1), 0, 1) * 25)
    dq_norm = (dq / 10.0) * 25

    if momentum is not None:
        m_score = momentum.score
        m_norm = (m_score / 10.0) * 20
    else:
        m_score = 0
        m_norm = 10.0  # mid-point — abstain rather than penalise

    composite = float(np.clip(f_norm + z_norm + dq_norm + m_norm, 0, 100))

    return QualityScore(
        piotroski_f=f,
        altman_z=z,
        dividend_quality=dq,
        earnings_momentum=m_score,
        momentum_detail=momentum,
        composite=composite,
        breakdown={
            "piotroski_components": fbreak,
            "altman_z_value": z,
            "dividend_components": dqbreak,
            "momentum_components": momentum.breakdown if momentum is not None else "—",
        },
    )
