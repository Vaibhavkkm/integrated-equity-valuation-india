"""
The integrated valuation engine — this is the entry point most callers
will use. It wires every other module into a single pipeline:

    fetch_stock(ticker)
        ↓
    validate_bundle  → DataQualityReport
        ↓
    cost_of_equity(prices) → Ke
        ↓
    select_and_value(...) → DDM intrinsic value (Bayes-shrunk growth)
        ↓
    find_peers(target) → PeerSet
        ↓
    value_by_multiples(peer_set) → Relative intrinsic value
        ↓
    quality_score(target) → composite 0-100
        ↓
    blend with credibility-adjusted weights
        ↓
    Reverse DCF → implied growth rate priced in by the market
        ↓
    Monte Carlo + tornado for confidence bands
        ↓
    Recommendation (BUY / HOLD / SELL) + confidence (HIGH / MED / LOW)
        ↓
    Optionally: persist signal for backtesting

The blender tilts the DDM weight away from the brief's 50% baseline
based on dividend quality and earnings track length — capped at ±15 pp,
so the project's headline 50/50 methodology is preserved while the
numbers stay honest.

The confidence label (HIGH / MEDIUM / LOW) is a separate axis from the
recommendation. It folds in:

  * Monte Carlo dispersion (wide bands → low confidence)
  * Data quality score
  * Number of valid peers contributing to the relative track
  * Whether DDM was applicable
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from config import SETTINGS, RISK_FREE_RATE_IN, EQUITY_RISK_PREMIUM_IN
from src.cost_of_equity import CostOfEquity, cost_of_equity
from src.data_fetcher import StockBundle, fetch_stock
from src.data_validation import DataQualityReport, validate_bundle
from src.ddm_models import IntrinsicValue, select_and_value
from src.earnings_momentum import EarningsMomentum, earnings_momentum
from src.exceptions import InvalidInputError
from src.logging_setup import get_logger
from src.peer_identification import PeerSet, find_peers
from src.quality_score import QualityScore, quality_score
from src.relative_valuation import RelativeValuation, value_by_multiples
from src.reverse_dcf import ImpliedExpectations, implied_growth
from src.sensitivity import (
    MonteCarloResult,
    TornadoBar,
    monte_carlo_blended,
    tornado_ddm,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ValuationResult:
    target: StockBundle
    coe: CostOfEquity
    ddm: IntrinsicValue
    relative: RelativeValuation
    peer_set: PeerSet
    quality: QualityScore
    data_quality: DataQualityReport
    momentum: EarningsMomentum

    weight_ddm: float
    weight_relative: float
    blended_value: float
    margin_of_safety: float
    recommendation: str
    confidence: str                                  # HIGH / MEDIUM / LOW

    monte_carlo: MonteCarloResult
    tornado: list[TornadoBar]
    reverse_dcf: ImpliedExpectations

    base_g_terminal: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        rec_color = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(self.recommendation, "")
        ddm_v = self.ddm.value_per_share if self.ddm.valid else float("nan")
        rel_v = self.relative.weighted_value
        lines = [
            "=" * 78,
            f"  {self.target.name}  ({self.target.ticker})  |  Sector: {self.target.sector}",
            "=" * 78,
            f"  Current Market Price       : ₹{self.target.price:>10,.2f}",
            f"  DDM Intrinsic Value        : ₹{ddm_v:>10,.2f}   ({self.ddm.model})",
            f"  Relative Intrinsic Value   : ₹{rel_v:>10,.2f}   (multi-multiple weighted)",
            f"  Blended Intrinsic Value    : ₹{self.blended_value:>10,.2f}   "
            f"(weights: DDM {self.weight_ddm:.0%}, Rel {self.weight_relative:.0%})",
            f"  Monte Carlo P10/P50/P90    : ₹{self.monte_carlo.p10:>7,.2f}  /"
            f"  ₹{self.monte_carlo.p50:>7,.2f}  /  ₹{self.monte_carlo.p90:>7,.2f}",
            f"  Cost of Equity (CAPM)      : {self.coe.ke:>10.2%}",
            f"  Quality Composite (0-100)  : {self.quality.composite:>10.1f}",
            f"  Data Quality (0-100)       : {self.data_quality.score:>10.1f}",
            f"  Margin of Safety           : {self.margin_of_safety:>10.2%}",
            f"  Recommendation             : {self.recommendation}  {rec_color}  "
            f"(confidence: {self.confidence})",
            f"  {self.reverse_dcf.summary()}",
            f"  {self.momentum.summary()}",
            "=" * 78,
        ]
        if self.notes:
            lines.append("  Notes:")
            for n in self.notes:
                lines.append(f"    • {n}")
            lines.append("=" * 78)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Credibility-weighted blender
# ---------------------------------------------------------------------------
def _blend_weights(
    ddm: IntrinsicValue,
    rel: RelativeValuation,
    quality: QualityScore,
    target: StockBundle,
    *,
    base_w_ddm: float = SETTINGS.ddm_weight,
) -> tuple[float, float]:
    """Return (w_ddm, w_relative) summing to 1.0.

    Payout ratio is the dominant signal. The DDM only prices cash
    returned to shareholders; firms that reinvest most earnings (low
    payout) hold the bulk of their value in growth the dividend stream
    alone cannot capture, so the blend has to lean on the relative
    track for them. Track-record quality is a tie-breaker on top.
    """
    if not ddm.valid:
        return 0.0, 1.0
    if (
        rel.weighted_value is None
        or not np.isfinite(rel.weighted_value)
        or rel.weighted_value <= 0
    ):
        return 1.0, 0.0

    tilt = 0.0

    payout = (
        float(target.payout_ratio)
        if (target.payout_ratio is not None and np.isfinite(target.payout_ratio))
        else 0.0
    )
    if payout < 0.30:
        tilt -= 0.30
    elif payout < 0.50:
        tilt -= 0.10
    elif payout >= 0.70:
        tilt += 0.10

    dq = quality.dividend_quality
    if dq >= 8 and payout >= 0.40:
        tilt += 0.05
    elif dq <= 2:
        tilt -= 0.10

    n_eps_years = (target.earnings_annual.dropna() > 0).sum()
    if n_eps_years >= 8:
        tilt += 0.05
    elif n_eps_years <= 3:
        tilt -= 0.05

    # DDM-vs-Relative divergence. When the two tracks disagree sharply,
    # one of them is mis-specified for this firm; in practice it's
    # almost always the DDM, because the dividend stream alone can't
    # capture brand premia / growth optionality the market is paying
    # for. Lean toward the relative track in that case.
    if (
        np.isfinite(rel.weighted_value)
        and rel.weighted_value > 0
        and np.isfinite(ddm.value_per_share)
    ):
        ratio = ddm.value_per_share / rel.weighted_value
        if ratio > 0:
            if ratio < 0.40:
                tilt -= 0.20
            elif ratio < 0.65:
                tilt -= 0.10
            elif ratio > 2.50:
                tilt -= 0.10  # the rare opposite case — Rel is suspect, but DDM
                              # alone is also not safe; small de-emphasis.

    tilt = float(np.clip(tilt, -0.45, 0.15))
    w_ddm = float(np.clip(base_w_ddm + tilt, 0.10, 0.80))
    return w_ddm, 1 - w_ddm


def _recommendation(
    margin_of_safety: float,
    *,
    ddm_valid: bool,
    rel_valid: bool,
    payout: float,
) -> str:
    """BUY / HOLD / SELL / N/A.

    N/A is returned when the model can't defensibly take a position:

      * MoS is non-finite (both tracks failed).
      * Only the DDM track is valid AND the firm has low payout (<30%) —
        DDM mechanically under-prices retainers, so calling SELL on the
        back of an empty peer set produces nonsense like "−2000% MoS".
      * MoS magnitude exceeds 250% — the model is well outside its
        calibrated range, almost always a data issue rather than a
        legitimate valuation disagreement.
    """
    if not np.isfinite(margin_of_safety):
        return "N/A"
    if not rel_valid and (not ddm_valid or payout < 0.30):
        return "N/A"
    if abs(margin_of_safety) > 2.5:
        return "N/A"
    if margin_of_safety >= SETTINGS.buy_threshold:
        return "BUY"
    if margin_of_safety <= SETTINGS.sell_threshold:
        return "SELL"
    return "HOLD"


def _confidence(
    *, mc: MonteCarloResult, blended: float,
    data_q: DataQualityReport, peers: PeerSet, ddm_valid: bool,
) -> tuple[str, list[str]]:
    """HIGH / MEDIUM / LOW based on dispersion + data + peer count.

    The label is what the user actually wants — "BUY (HIGH confidence)"
    is far more actionable than just "BUY". Without this, every
    recommendation looks equally credible, which it isn't.

    Scoring (0-100, higher = more confident):
      * Monte Carlo width: σ/μ < 15% → +35; <30% → +20; else 0.
      * Data quality:     score / 100 × 25.
      * Peer count:       ≥7 → +25; ≥4 → +15; else +5.
      * DDM applicable:   +15 if valid (only one track is fragile).

    Then bucket: ≥75 HIGH, ≥50 MEDIUM, else LOW.
    """
    notes: list[str] = []
    score = 0.0

    # Dispersion
    if mc.samples is not None and mc.samples.size > 0 and np.isfinite(blended) and blended > 0:
        cov = mc.std / blended if blended else float("inf")
        if cov < 0.15:
            score += 35
        elif cov < 0.30:
            score += 20
        else:
            notes.append(f"Wide Monte Carlo bands (σ/μ={cov:.0%}) — fair value is sensitive to inputs.")
    else:
        notes.append("Monte Carlo did not converge — using point-estimate only.")

    # Data quality
    score += (data_q.score / 100.0) * 25
    if data_q.score < 60:
        notes.append(f"Underlying data is thin (score={data_q.score:.0f}/100); see warnings.")

    # Peer count
    n_peers = len(peers.peers)
    if n_peers >= 7:
        score += 25
    elif n_peers >= 4:
        score += 15
    else:
        score += 5
        notes.append(f"Only {n_peers} peers — relative valuation is fragile.")

    # DDM availability
    if ddm_valid:
        score += 15
    else:
        notes.append("DDM not applicable (non-payer); relying on relative valuation alone.")

    if score >= 75:
        tier = "HIGH"
    elif score >= 50:
        tier = "MEDIUM"
    else:
        tier = "LOW"
    return tier, notes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def value_stock(
    ticker: str,
    *,
    sector: Optional[str] = None,
    risk_free: float = RISK_FREE_RATE_IN,
    erp: float = EQUITY_RISK_PREMIUM_IN,
    g_terminal: Optional[float] = None,
    offline: bool = False,
    force_refresh: bool = False,
    log_for_backtest: bool = False,
    verbose: bool = True,
) -> ValuationResult:
    """End-to-end valuation for a single Indian-listed equity.

    Parameters
    ----------
    ticker : str
        NSE/BSE ticker (e.g. ``"TCS.NS"``).
    sector : str, optional
        Override the sector mapping in ``config.DEFAULT_UNIVERSE``.
    risk_free, erp : float
        CAPM inputs; defaults are India-calibrated.
    g_terminal : float, optional
        Terminal growth rate; defaults to 4.5% (below India's nominal
        long-run growth ceiling).
    offline : bool
        Force cache-only mode; raises if no cache is present.
    force_refresh : bool
        Ignore cache; re-hit upstream provider.
    log_for_backtest : bool
        If True, append the resulting recommendation to the backtest log
        (see ``src/backtest.py``). Use this to accumulate a panel that
        can later be graded against forward returns.
    verbose : bool
        Emit progress lines via the engine logger.
    """
    # ------------------------------------------------------------------
    # 0. Input validation
    # ------------------------------------------------------------------
    if not ticker or not isinstance(ticker, str):
        raise InvalidInputError("ticker must be a non-empty string")
    if not 0.0 < risk_free < 0.30:
        raise InvalidInputError(f"risk_free out of plausible range: {risk_free}")
    if not 0.0 < erp < 0.20:
        raise InvalidInputError(f"erp out of plausible range: {erp}")
    if g_terminal is not None and not -0.02 < g_terminal < 0.10:
        raise InvalidInputError(f"g_terminal out of plausible range: {g_terminal}")

    if verbose:
        log.info("→ Valuing %s", ticker)

    # ------------------------------------------------------------------
    # 1. Fetch + validate
    # ------------------------------------------------------------------
    target = fetch_stock(
        ticker, sector=sector, offline=offline, force_refresh=force_refresh,
    )
    data_q = validate_bundle(target, strict=False)

    if verbose:
        log.info("  loaded %s | sector=%s | price=₹%,.2f | data-quality=%.0f/100",
                 target.name, target.sector, target.price, data_q.score)

    if not data_q.is_usable:
        log.warning("  data quality below usable threshold — proceeding but flagging LOW confidence")

    # ------------------------------------------------------------------
    # 2. Cost of equity
    # ------------------------------------------------------------------
    coe = cost_of_equity(
        target.price_history,
        sector=target.sector,
        market_cap=target.market_cap,
        total_debt=target.total_debt,
        equity=target.equity,
        risk_free=risk_free,
        erp=erp,
    )
    if verbose:
        log.info("  Ke=%.2f%% (β_adj=%.2f, method=%s)",
                 coe.ke * 100, coe.beta_adjusted, coe.method)

    # ------------------------------------------------------------------
    # 3. DDM (with Bayes-shrunk growth, baked in inside ddm_models)
    # ------------------------------------------------------------------
    g_term = g_terminal if g_terminal is not None else 0.045
    ddm = select_and_value(
        eps_ttm=target.eps_ttm,
        dps_ttm=target.dividend_per_share_ttm,
        payout_ratio=target.payout_ratio,
        roe=target.roe,
        ke=coe.ke,
        historical_dps=target.dividends_annual,
        historical_eps=target.earnings_annual,
        sector_g_terminal=g_term,
        payout_ratio_raw=getattr(target, "payout_ratio_raw", float("nan")),
    )
    if verbose:
        log.info("  DDM: %s", ddm)

    # ------------------------------------------------------------------
    # 4. Peers + Relative
    # ------------------------------------------------------------------
    peers = find_peers(target, offline=offline)
    rel = value_by_multiples(peers)
    if verbose:
        log.info("  peers=%d via %s | relative=₹%,.2f",
                 len(peers.peers), peers.method, rel.weighted_value)

    # ------------------------------------------------------------------
    # 5. Earnings momentum overlay (peers needed for the comparison signals)
    # ------------------------------------------------------------------
    momentum = earnings_momentum(target, peers)
    if verbose:
        log.info("  %s", momentum.summary())

    # ------------------------------------------------------------------
    # 6. Quality score (now includes momentum as the 4th pillar)
    # ------------------------------------------------------------------
    q = quality_score(target, momentum=momentum)
    if verbose:
        log.info("  quality=%.1f/100 (F=%d/9, DQ=%d/10, M=%d/10)",
                 q.composite, q.piotroski_f, q.dividend_quality, q.earnings_momentum)

    # ------------------------------------------------------------------
    # 6. Blend
    # ------------------------------------------------------------------
    w_ddm, w_rel = _blend_weights(ddm, rel, q, target)

    if ddm.valid and np.isfinite(rel.weighted_value):
        blended = w_ddm * ddm.value_per_share + w_rel * rel.weighted_value
    elif ddm.valid:
        blended = ddm.value_per_share
    else:
        blended = rel.weighted_value

    mos = (blended - target.price) / blended if (np.isfinite(blended) and blended > 0) else float("nan")
    rec = _recommendation(
        mos,
        ddm_valid=ddm.valid,
        rel_valid=bool(np.isfinite(rel.weighted_value)) if rel.weighted_value is not None else False,
        payout=float(target.payout_ratio) if (target.payout_ratio is not None and np.isfinite(target.payout_ratio)) else 0.0,
    )

    # ------------------------------------------------------------------
    # 7. Reverse DCF — what growth is the market pricing in?
    # ------------------------------------------------------------------
    g_hist_for_compare: Optional[float] = None
    dps = target.dividends_annual.dropna()
    if len(dps) >= 4 and dps.iloc[0] > 0:
        n = len(dps) - 1
        g_hist_for_compare = float((dps.iloc[-1] / dps.iloc[0]) ** (1 / n) - 1)

    rev = implied_growth(
        market_price=target.price,
        d0=target.dividend_per_share_ttm,
        ke=coe.ke,
        g_terminal=g_term,
        historical_growth=g_hist_for_compare,
    )
    if verbose:
        log.info("  %s", rev.summary())

    # ------------------------------------------------------------------
    # 8. Monte Carlo + Tornado
    # ------------------------------------------------------------------
    mc = monte_carlo_blended(target, coe, rel, g_term, w_ddm)
    tornado = tornado_ddm(target, coe.ke, g_term) if ddm.valid else []

    # ------------------------------------------------------------------
    # 9. Confidence label
    # ------------------------------------------------------------------
    confidence, conf_notes = _confidence(
        mc=mc, blended=blended, data_q=data_q, peers=peers, ddm_valid=ddm.valid,
    )
    notes = list(data_q.warnings) + conf_notes

    if verbose:
        log.info("  recommendation=%s (confidence=%s)", rec, confidence)

    result = ValuationResult(
        target=target,
        coe=coe,
        ddm=ddm,
        relative=rel,
        peer_set=peers,
        quality=q,
        data_quality=data_q,
        momentum=momentum,
        weight_ddm=w_ddm,
        weight_relative=w_rel,
        blended_value=float(blended),
        margin_of_safety=float(mos),
        recommendation=rec,
        confidence=confidence,
        monte_carlo=mc,
        tornado=tornado,
        reverse_dcf=rev,
        base_g_terminal=g_term,
        notes=notes,
    )

    # ------------------------------------------------------------------
    # 10. Optional: persist for later backtesting
    # ------------------------------------------------------------------
    if log_for_backtest and rec in ("BUY", "HOLD", "SELL"):
        try:
            from src.backtest import append_signal
            append_signal(
                ticker=target.ticker, recommendation=rec,
                blended_value=blended, price=target.price,
                margin_of_safety=mos, quality_score=q.composite,
            )
        except Exception as exc:  # pragma: no cover
            log.warning("backtest log failed: %s", exc)

    return result
