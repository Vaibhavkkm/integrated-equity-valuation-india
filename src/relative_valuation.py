"""
Relative valuation engine.

Five multiples are computed; each yields its own implied price:

  * P/E         — price per rupee of trailing earnings.
  * P/B         — price per rupee of book value (the only sensible
                  multiple for banks; we therefore over-weight it for
                  the Banking sector).
  * P/S         — price per rupee of revenue. Useful when earnings are
                  noisy or negative (loss-making but growing names).
  * EV/EBITDA   — capital-structure neutral; preferred for capital
                  intensive sectors (Cement, Metals, Telecom).
  * PEG         — P/E adjusted for growth. Rough but useful.

The peer multiples are aggregated using the **harmonic mean**, not the
arithmetic mean, because multiples are ratios and arithmetic averaging
biases them upward. We also trim the top and bottom decile of peer
multiples to neutralise the effect of any single outlier.

The final per-multiple implied prices are then combined using
**sector-aware weights** — e.g. for a bank, P/B carries 50% of the weight,
EV/EBITDA carries 0%.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import SETTINGS
from src.data_fetcher import StockBundle
from src.peer_identification import PeerSet


# ---------------------------------------------------------------------------
# Sector-specific multiple weights — sums to 1.0 within each sector
# ---------------------------------------------------------------------------
DEFAULT_MULTIPLE_WEIGHTS: Dict[str, float] = {
    "PE": 0.30,
    "PB": 0.20,
    "PS": 0.10,
    "EV_EBITDA": 0.30,
    "PEG": 0.10,
}

SECTOR_OVERRIDES: Dict[str, Dict[str, float]] = {
    "Banking": {"PE": 0.30, "PB": 0.50, "PS": 0.10, "EV_EBITDA": 0.00, "PEG": 0.10},
    "NBFC":    {"PE": 0.30, "PB": 0.50, "PS": 0.10, "EV_EBITDA": 0.00, "PEG": 0.10},
    "Metals":  {"PE": 0.20, "PB": 0.20, "PS": 0.10, "EV_EBITDA": 0.40, "PEG": 0.10},
    "Cement":  {"PE": 0.20, "PB": 0.10, "PS": 0.10, "EV_EBITDA": 0.50, "PEG": 0.10},
    "Telecom": {"PE": 0.15, "PB": 0.10, "PS": 0.20, "EV_EBITDA": 0.45, "PEG": 0.10},
    "Real Estate": {"PE": 0.20, "PB": 0.40, "PS": 0.10, "EV_EBITDA": 0.20, "PEG": 0.10},
    "FMCG":    {"PE": 0.40, "PB": 0.10, "PS": 0.10, "EV_EBITDA": 0.30, "PEG": 0.10},
    "Information Technology": {"PE": 0.45, "PB": 0.05, "PS": 0.15, "EV_EBITDA": 0.25, "PEG": 0.10},
}


@dataclass
class MultipleResult:
    multiple_name: str
    peer_values: pd.Series                  # individual peer multiples
    aggregated_multiple: float              # harmonic mean after trim
    target_per_share_metric: float          # EPS / BVPS / SPS / EBITDA-PS / PEG g
    implied_price: float
    valid: bool = True
    note: str = ""


@dataclass
class RelativeValuation:
    multiples: Dict[str, MultipleResult]
    weighted_value: float
    weights: Dict[str, float]
    median_value: float
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Multiple computation per peer
# ---------------------------------------------------------------------------
def _peer_multiple(b: StockBundle, kind: str) -> Optional[float]:
    if not np.isfinite(b.price) or b.price <= 0:
        return None

    if kind == "PE":
        return b.price / b.eps_ttm if (np.isfinite(b.eps_ttm) and b.eps_ttm > 0) else None

    if kind == "PB":
        return b.price / b.book_value_per_share if (np.isfinite(b.book_value_per_share) and b.book_value_per_share > 0) else None

    if kind == "PS":
        return b.price / b.sales_per_share_ttm if (np.isfinite(b.sales_per_share_ttm) and b.sales_per_share_ttm > 0) else None

    if kind == "EV_EBITDA":
        if not (np.isfinite(b.enterprise_value) and np.isfinite(b.ebitda) and b.ebitda > 0):
            return None
        return b.enterprise_value / b.ebitda

    if kind == "PEG":
        # PEG = (P/E) / (earnings growth %)
        pe = _peer_multiple(b, "PE")
        if pe is None:
            return None
        # 5-yr earnings CAGR proxy
        e = b.earnings_annual.dropna()
        if len(e) >= 4 and e.iloc[0] > 0:
            n = len(e) - 1
            g = (e.iloc[-1] / e.iloc[0]) ** (1 / n) - 1
        else:
            g = None
        if g is None or g <= 0:
            return None
        return pe / (g * 100)

    raise ValueError(f"Unknown multiple kind: {kind}")


def _trimmed_harmonic_mean(values: pd.Series, trim_pct: float) -> float:
    s = values.dropna()
    s = s[s > 0]
    if s.empty:
        return float("nan")

    if len(s) >= 5:
        lo = s.quantile(trim_pct)
        hi = s.quantile(1 - trim_pct)
        s = s[(s >= lo) & (s <= hi)]
    if s.empty:
        return float("nan")

    return float(len(s) / np.sum(1.0 / s))


# ---------------------------------------------------------------------------
# Per-multiple → implied price for the target
# ---------------------------------------------------------------------------
def _implied_price(target: StockBundle, multiple_name: str, agg_multiple: float):
    """Returns (implied_price, target_metric_used)."""
    if not np.isfinite(agg_multiple):
        return float("nan"), float("nan")

    if multiple_name == "PE":
        m = target.eps_ttm
        return agg_multiple * m, m
    if multiple_name == "PB":
        m = target.book_value_per_share
        return agg_multiple * m, m
    if multiple_name == "PS":
        m = target.sales_per_share_ttm
        return agg_multiple * m, m
    if multiple_name == "EV_EBITDA":
        # implied EV → equity value → per share
        if not (np.isfinite(target.ebitda) and target.ebitda > 0):
            return float("nan"), float("nan")
        implied_ev = agg_multiple * target.ebitda
        equity_value = implied_ev - target.total_debt + target.cash
        if not np.isfinite(target.shares_outstanding) or target.shares_outstanding <= 0:
            return float("nan"), target.ebitda
        return equity_value / target.shares_outstanding, target.ebitda
    if multiple_name == "PEG":
        # Implied P/E = PEG * g; then implied price = PE * EPS
        e = target.earnings_annual.dropna()
        if len(e) >= 4 and e.iloc[0] > 0:
            n = len(e) - 1
            g = (e.iloc[-1] / e.iloc[0]) ** (1 / n) - 1
        else:
            g = None
        if g is None or g <= 0 or not (np.isfinite(target.eps_ttm) and target.eps_ttm > 0):
            return float("nan"), float("nan")
        implied_pe = agg_multiple * (g * 100)
        return implied_pe * target.eps_ttm, g

    raise ValueError(multiple_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def value_by_multiples(
    peer_set: PeerSet,
    *,
    trim_pct: float = SETTINGS.multiple_trim_pct,
) -> RelativeValuation:
    target = peer_set.target
    peers = peer_set.peers

    multiples_to_run = ["PE", "PB", "PS", "EV_EBITDA", "PEG"]
    results: Dict[str, MultipleResult] = {}

    for kind in multiples_to_run:
        peer_vals = pd.Series(
            {p.ticker: _peer_multiple(p, kind) for p in peers},
            dtype=float,
        )
        agg = _trimmed_harmonic_mean(peer_vals, trim_pct)
        implied, target_metric = _implied_price(target, kind, agg)

        results[kind] = MultipleResult(
            multiple_name=kind,
            peer_values=peer_vals,
            aggregated_multiple=agg,
            target_per_share_metric=target_metric,
            implied_price=implied,
            valid=np.isfinite(implied) and implied > 0,
            note="" if np.isfinite(implied) else "Insufficient data",
        )

    # Sector-aware weights, dropping invalid multiples and renormalising.
    base_weights = SECTOR_OVERRIDES.get(target.sector, DEFAULT_MULTIPLE_WEIGHTS)
    valid_weights = {k: w for k, w in base_weights.items() if results[k].valid}
    if not valid_weights:
        return RelativeValuation(
            multiples=results,
            weighted_value=float("nan"),
            weights=base_weights,
            median_value=float("nan"),
            notes=["No valid multiples — relative valuation unavailable."],
        )
    s = sum(valid_weights.values())
    valid_weights = {k: w / s for k, w in valid_weights.items()}

    weighted = sum(results[k].implied_price * w for k, w in valid_weights.items())
    median_v = float(np.median([
        results[k].implied_price for k in valid_weights
    ]))

    return RelativeValuation(
        multiples=results,
        weighted_value=float(weighted),
        weights=valid_weights,
        median_value=median_v,
    )
