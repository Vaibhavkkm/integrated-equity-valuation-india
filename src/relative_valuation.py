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
from typing import Dict, Optional

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
    # Loss-making or thin-margin platforms; PE/PEG are unreliable, so
    # the weight sits on revenue scale (P/S) and operating cash run-rate
    # (EV/EBITDA). P/B carries a small slice for the asset-lighter names.
    "Internet & Platform": {"PE": 0.10, "PB": 0.10, "PS": 0.45, "EV_EBITDA": 0.30, "PEG": 0.05},
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
_PEG_MAX_GROWTH = 0.25  # cap historical earnings g; 25%/yr forever is noise


def _earnings_cagr(earnings_annual: pd.Series) -> Optional[float]:
    """5-yr earnings CAGR, with sanity caps. ``None`` if not computable."""
    e = earnings_annual.dropna()
    if len(e) < 4 or e.iloc[0] <= 0 or e.iloc[-1] <= 0:
        return None
    n = len(e) - 1
    g = (e.iloc[-1] / e.iloc[0]) ** (1 / n) - 1
    if not np.isfinite(g) or g <= 0:
        return None
    return float(min(g, _PEG_MAX_GROWTH))


def _has_consistent_units(b: StockBundle) -> tuple[bool, bool]:
    """Returns (revenue_ok, ebitda_ok).

    yfinance occasionally returns IT-exporter revenue / EBITDA in USD
    while price and EPS are in INR — making peer ratios meaningless. The
    cheap detection: net income (EPS × shares, all in INR) must not
    exceed revenue or EBITDA. If it does, the unit is inconsistent and
    the multiple has to be dropped.
    """
    if not (np.isfinite(b.eps_ttm) and np.isfinite(b.shares_outstanding)
            and b.shares_outstanding > 0):
        # Can't verify; trust the data.
        return True, True
    ni = b.eps_ttm * b.shares_outstanding
    if ni <= 0:
        return True, True
    rev_ok = not (np.isfinite(b.revenue) and b.revenue > 0 and b.revenue < ni)
    ebitda_ok = not (np.isfinite(b.ebitda) and b.ebitda > 0 and b.ebitda < ni)
    return rev_ok, ebitda_ok


def _peer_multiple(b: StockBundle, kind: str) -> Optional[float]:
    if not np.isfinite(b.price) or b.price <= 0:
        return None

    if kind == "PE":
        return b.price / b.eps_ttm if (np.isfinite(b.eps_ttm) and b.eps_ttm > 0) else None

    if kind == "PB":
        return b.price / b.book_value_per_share if (np.isfinite(b.book_value_per_share) and b.book_value_per_share > 0) else None

    if kind == "PS":
        rev_ok, _ = _has_consistent_units(b)
        if not rev_ok:
            return None
        return b.price / b.sales_per_share_ttm if (np.isfinite(b.sales_per_share_ttm) and b.sales_per_share_ttm > 0) else None

    if kind == "EV_EBITDA":
        _, ebitda_ok = _has_consistent_units(b)
        if not ebitda_ok:
            return None
        if not (np.isfinite(b.enterprise_value) and np.isfinite(b.ebitda) and b.ebitda > 0):
            return None
        return b.enterprise_value / b.ebitda

    if kind == "PEG":
        pe = _peer_multiple(b, "PE")
        if pe is None:
            return None
        g = _earnings_cagr(b.earnings_annual)
        if g is None:
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
        rev_ok, _ = _has_consistent_units(target)
        if not rev_ok:
            return float("nan"), float("nan")
        m = target.sales_per_share_ttm
        return agg_multiple * m, m
    if multiple_name == "EV_EBITDA":
        _, ebitda_ok = _has_consistent_units(target)
        if not ebitda_ok:
            return float("nan"), float("nan")
        if not (np.isfinite(target.ebitda) and target.ebitda > 0):
            return float("nan"), float("nan")
        implied_ev = agg_multiple * target.ebitda
        equity_value = implied_ev - target.total_debt + target.cash
        if not np.isfinite(target.shares_outstanding) or target.shares_outstanding <= 0:
            return float("nan"), target.ebitda
        return equity_value / target.shares_outstanding, target.ebitda
    if multiple_name == "PEG":
        g = _earnings_cagr(target.earnings_annual)
        if g is None or not (np.isfinite(target.eps_ttm) and target.eps_ttm > 0):
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
    """Value the target using sector-aware peer multiples.

    Computes P/E, P/B, P/S, EV/EBITDA, and PEG implied prices, aggregates
    each via a trimmed harmonic mean (the standard Aswath Damodaran
    correction for the upward bias of arithmetic means on ratios), then
    combines them with sector-specific weights into a single weighted
    intrinsic value.

    Parameters
    ----------
    peer_set : PeerSet
        Output of :func:`peer_identification.find_peers`. Below
        ``SETTINGS.min_peers`` the function returns NaN multiples and a
        ``weighted_value`` of NaN so the integrated engine flags the
        relative track unavailable.
    trim_pct : float, optional
        Fraction trimmed from each tail of the per-multiple peer
        distribution before harmonic-mean aggregation. Defaults to
        ``SETTINGS.multiple_trim_pct``.

    Returns
    -------
    RelativeValuation
        Per-multiple results, sector-weighted blend, median fallback, and
        any caveats (insufficient peers, negative-earnings exclusions, …).
    """
    target = peer_set.target
    peers = peer_set.peers

    # Hard-fail when the peer pool is below the minimum. Three random
    # same-sector names produce a number, but it isn't a defensible
    # relative valuation (this was how SWIGGY against TITAN/HAVELLS/VOLTAS
    # got past the engine). Force the blender to fall back to N/A.
    if len(peers) < SETTINGS.min_peers:
        empty = pd.Series(dtype=float)
        return RelativeValuation(
            multiples={
                k: MultipleResult(
                    multiple_name=k,
                    peer_values=empty,
                    aggregated_multiple=float("nan"),
                    target_per_share_metric=float("nan"),
                    implied_price=float("nan"),
                    valid=False,
                    note=f"Peer pool too small ({len(peers)} < {SETTINGS.min_peers}).",
                )
                for k in ("PE", "PB", "PS", "EV_EBITDA", "PEG")
            },
            weighted_value=float("nan"),
            weights={},
            median_value=float("nan"),
            notes=[
                f"Relative valuation unavailable: only {len(peers)} peers "
                f"(need ≥{SETTINGS.min_peers}). {peer_set.method}"
            ],
        )

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

    # Outlier filter: if any multiple's implied price is more than 5x off
    # the median of the other valid implieds, that multiple is almost
    # certainly built on bad units / corrupted statements (we have seen
    # yfinance return USD revenue for IT exporters). Drop it before the
    # weighted average so it can't drag the answer.
    valid_implied = [
        (k, results[k].implied_price)
        for k in results if results[k].valid
    ]
    if len(valid_implied) >= 3:
        med = float(np.median([v for _, v in valid_implied]))
        if med > 0:
            for k, v in valid_implied:
                if v <= 0 or v < 0.20 * med or v > 5.0 * med:
                    results[k].valid = False
                    results[k].note = (
                        f"Implied price ₹{v:,.0f} is {v/med:.1f}× peer median "
                        f"(₹{med:,.0f}); excluded as outlier."
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
