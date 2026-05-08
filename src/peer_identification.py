"""
Peer identification using clustering, not just sector membership.

The textbook approach to relative valuation is "find companies in the
same sector and average their multiples." That falls apart in India for
two big reasons:

  1. Sectors are heterogeneous. "FMCG" includes both ITC (cigarettes,
     30%+ EBITDA margin, low growth) and Nestle (packaged foods, 25%
     margin, premium multiple). Averaging them yields nonsense.
  2. Conglomerates bend the categories — Reliance is "Oil & Gas" by
     legacy classification but ~40% of EBITDA is now telecom and retail.

So we do something more defensible:

  Step 1: Take the same-sector universe as the candidate pool.
  Step 2: Standardize the firms on a small ratio vector that captures
          *business shape* — size, profitability, growth, leverage,
          payout. Z-score within sector.
  Step 3: K-Means cluster. Pick the cluster the target firm sits in.
  Step 4: Within that cluster, rank by Mahalanobis distance to the
          target on the same ratio vector and keep the closest N.

This gives a peer set whose members actually *look like* the target,
not just share a GICS bucket.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_UNIVERSE, SETTINGS
from src.data_fetcher import StockBundle, fetch_stock
from src.logging_setup import get_logger

log = get_logger(__name__)


# Ratio vector used both for clustering and for Mahalanobis ranking.
_FEATURE_NAMES = [
    "log_market_cap",
    "roe",
    "debt_to_equity",
    "payout_ratio",
    "revenue_growth_5y",
    "ebitda_margin",
]


@dataclass
class PeerSet:
    target: StockBundle
    peers: List[StockBundle]
    method: str
    debug_features: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _extract_features(b: StockBundle) -> dict:
    # Floor at ₹10 crore to keep log() out of the long left tail; below
    # that, "log of micro-cap" is more noise than signal and would yank
    # the firm into a cluster of its own.
    if b.market_cap and b.market_cap >= 100_000_000:
        log_mcap = float(np.log(b.market_cap))
    else:
        log_mcap = np.nan

    # D/E only meaningful when equity is positive *and* debt is non-
    # negative. Negative-equity firms (accumulated losses) and any
    # debt-data glitch get flagged as missing rather than producing
    # negative leverage figures that distort clustering.
    if (b.equity and b.equity > 0
            and b.total_debt is not None
            and np.isfinite(b.total_debt)
            and b.total_debt >= 0):
        de = b.total_debt / b.equity
    else:
        de = np.nan

    # ROE: NaN propagates correctly through median imputation downstream.
    roe = b.roe if np.isfinite(b.roe) else np.nan

    # 5-year revenue CAGR
    rev = b.revenue_annual.dropna()
    if len(rev) >= 4 and rev.iloc[0] > 0:
        n = len(rev) - 1
        rev_g = (rev.iloc[-1] / rev.iloc[0]) ** (1 / n) - 1
    else:
        rev_g = np.nan

    ebitda_margin = (b.ebitda / b.revenue) if (b.ebitda and b.revenue and b.revenue > 0) else np.nan

    return {
        "log_market_cap": log_mcap,
        "roe": roe,
        "debt_to_equity": de,
        "payout_ratio": b.payout_ratio,
        "revenue_growth_5y": rev_g,
        "ebitda_margin": ebitda_margin,
    }


def _build_feature_matrix(bundles: List[StockBundle]) -> pd.DataFrame:
    rows = {b.ticker: _extract_features(b) for b in bundles}
    df = pd.DataFrame.from_dict(rows, orient="index", columns=_FEATURE_NAMES)
    # Median-impute missing values so clustering doesn't choke; firms with
    # truly bad data will be filtered out later. If a whole column is NaN
    # (e.g. EBITDA margin / D-to-E for banks), median is NaN too — fall
    # back to 0 so KMeans can still run. The matrix is z-scored downstream,
    # so a constant column contributes nothing to the cluster geometry.
    df = df.fillna(df.median(numeric_only=True)).fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Mahalanobis distance — sensitive to inter-feature correlation
# ---------------------------------------------------------------------------
def _mahalanobis(X: np.ndarray, target_row: np.ndarray) -> np.ndarray:
    cov = np.cov(X, rowvar=False)
    # Numerical stability: ridge on the covariance.
    cov += np.eye(cov.shape[0]) * 1e-6
    inv_cov = np.linalg.pinv(cov)
    diffs = X - target_row
    return np.sqrt(np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _within_size_band(target: StockBundle, peer: StockBundle, band: float) -> bool:
    """Reject peers whose market cap is outside [target / band, target × band].

    Large-cap multiples don't apply to small-caps and vice versa: the
    market discounts small-caps for liquidity, governance, and scale.
    Without this filter, an orphan small-cap (HGINFRA against LT/ABB)
    inherits big-cap multiples and the relative valuation explodes.
    """
    if not (np.isfinite(target.market_cap) and target.market_cap > 0):
        return True  # Can't enforce; let it through.
    if not (np.isfinite(peer.market_cap) and peer.market_cap > 0):
        return False
    return target.market_cap / band <= peer.market_cap <= target.market_cap * band


def find_peers(
    target: StockBundle,
    universe: Optional[dict] = None,
    *,
    n_clusters: int = SETTINGS.n_peer_clusters,
    min_peers: int = SETTINGS.min_peers,
    max_peers: int = SETTINGS.max_peers,
    offline: bool = False,
    size_band: float = 5.0,
) -> PeerSet:
    """Build a clustering-based peer set for ``target``.

    Same-sector candidates are filtered to a market-cap band, fed into
    K-Means on a normalised feature matrix, and the target's cluster-mates
    are ranked by Mahalanobis distance to pick the closest peers. The size
    band auto-widens (5x → 10x → 20x) when a sector has few in-band names.

    Parameters
    ----------
    target : StockBundle
        The stock being valued. Sector and market cap drive the candidate
        filter; the full feature vector (ROE, D/E, payout, growth, margin)
        anchors the Mahalanobis ranking.
    universe : dict, optional
        Ticker → sector mapping. Defaults to ``config.DEFAULT_UNIVERSE``.
    n_clusters : int, optional
        K for the K-Means partition over the same-sector pool.
    min_peers, max_peers : int, optional
        Acceptable peer-count band for the final result. Below ``min_peers``
        the function returns an empty peer set so the relative-valuation
        track is flagged unavailable rather than reporting noise.
    offline : bool, optional
        If True, do not hit yfinance — read peer bundles from disk only.
    size_band : float, optional
        Initial market-cap multiple bracketing the target (default 5x).
        Auto-widens to 10x and 20x when fewer than ``min_peers`` survive.

    Returns
    -------
    PeerSet
        Container carrying the resolved peers, the selection method
        (cluster id, band used, fallback note), and the debug feature
        matrix that the dashboard renders for transparency.
    """
    universe = universe or DEFAULT_UNIVERSE
    same_sector_tickers = [
        t for t, s in universe.items()
        if s == target.sector and t != target.ticker
    ]

    # Fetch the same-sector candidate pool, then drop peers outside the
    # size band so a small-cap target doesn't get valued at large-cap
    # multiples (or vice versa). Widen progressively; if we still can't
    # hit min_peers we return an empty peer set, which forces the engine
    # to fall back to N/A rather than print an inflated relative value.
    pool_raw = _safe_fetch_many(same_sector_tickers, offline=offline)
    pool: List[StockBundle] = []
    band_used = size_band
    for band in (size_band, size_band * 2, size_band * 4):
        pool = [p for p in pool_raw if _within_size_band(target, p, band)]
        band_used = band
        if len(pool) >= min_peers:
            if band > size_band:
                log.info("peers: widened size band to %.0fx for %s "
                         "(only %d in-band peers at %.0fx)",
                         band, target.ticker, len(pool), size_band)
            break

    if len(pool) < min_peers:
        return PeerSet(
            target=target,
            peers=pool,
            method=(
                f"insufficient size-matched peers — "
                f"{len(pool)} of {len(pool_raw)} candidates remain after "
                f"{band_used:.0f}x size filter; "
                "relative valuation will be flagged unavailable."
            ),
        )

    # Build feature matrix including target
    all_bundles = [target] + pool
    feats = _build_feature_matrix(all_bundles)

    scaler = StandardScaler()
    X = scaler.fit_transform(feats.values)

    # Cluster — clamp k so it's never larger than the data allows.
    k = max(2, min(n_clusters, len(X) - 1))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    target_label = labels[0]
    in_cluster_mask = (labels == target_label)
    in_cluster_idx = np.where(in_cluster_mask)[0]
    in_cluster_idx = in_cluster_idx[in_cluster_idx != 0]  # drop the target

    if len(in_cluster_idx) < min_peers:
        # Cluster too small — relax to the whole sector and just use
        # Mahalanobis ranking.
        candidate_idx = np.arange(1, len(X))
    else:
        candidate_idx = in_cluster_idx

    # Rank candidates by Mahalanobis distance from target
    target_row = X[0]
    distances = _mahalanobis(X[candidate_idx], target_row)
    order = np.argsort(distances)
    chosen = candidate_idx[order][:max_peers]

    if len(chosen) < min_peers:
        # Final fallback: just take the top-N closest in the entire sector.
        all_idx = np.arange(1, len(X))
        all_distances = _mahalanobis(X[all_idx], target_row)
        chosen = all_idx[np.argsort(all_distances)][:min_peers]
        method = "sector + Mahalanobis (cluster too small)"
    else:
        method = f"k-means cluster {target_label} + Mahalanobis"

    peers = [all_bundles[i] for i in chosen]

    debug = feats.iloc[[0] + list(chosen)].copy()
    debug.insert(0, "role", ["target"] + ["peer"] * len(chosen))

    return PeerSet(target=target, peers=peers, method=method, debug_features=debug)


def _safe_fetch_many(tickers: List[str], *, offline: bool) -> List[StockBundle]:
    out: List[StockBundle] = []
    for t in tickers:
        try:
            out.append(fetch_stock(t, offline=offline))
        except Exception as e:
            log.warning("peer skipped: %s (%s)", t, e.__class__.__name__)
    return out
