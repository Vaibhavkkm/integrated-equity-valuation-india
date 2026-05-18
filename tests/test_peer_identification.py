"""
Tests for the K-Means + Mahalanobis peer selector.

The fetch path is monkey-patched so the network is never touched — we
inject a synthetic universe of bundles and assert against the chosen
peer set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import peer_identification
from src.peer_identification import (
    PeerSet, _within_size_band, _mahalanobis, find_peers,
)

from _factory import make_bundle


# ---------------------------------------------------------------------------
# Size-band filter
# ---------------------------------------------------------------------------
def test_size_band_accepts_within_window():
    t = make_bundle(market_cap=1e11)
    p = make_bundle(market_cap=2e11)
    assert _within_size_band(t, p, band=5.0)


def test_size_band_rejects_outside_window():
    t = make_bundle(market_cap=1e9)        # ₹100 cr
    p = make_bundle(market_cap=1e12)       # ₹100,000 cr — 1000× larger
    assert not _within_size_band(t, p, band=5.0)


def test_size_band_passes_when_target_has_no_mcap():
    t = make_bundle(market_cap=float("nan"))
    p = make_bundle(market_cap=1e11)
    # Can't enforce filter → let it through.
    assert _within_size_band(t, p, band=5.0)


# ---------------------------------------------------------------------------
# Mahalanobis distance
# ---------------------------------------------------------------------------
def test_mahalanobis_zero_at_target_row():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    d = _mahalanobis(X, X[0])
    assert d[0] == pytest.approx(0.0, abs=1e-6)
    assert d[1] > 0 and d[2] > 0


def test_mahalanobis_handles_singular_covariance():
    """Constant columns produce singular covariance; the ridge term must
    keep the calculation finite."""
    X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
    d = _mahalanobis(X, X[0])
    assert np.all(np.isfinite(d))


# ---------------------------------------------------------------------------
# End-to-end find_peers with a stubbed fetch
# ---------------------------------------------------------------------------
def _stub_pool(target, n_peers: int = 6):
    """Build N synthetic FMCG peers around the target's profile."""
    peers = []
    for i in range(n_peers):
        peers.append(make_bundle(
            ticker=f"PEER{i}.NS",
            name=f"Peer {i}",
            sector="FMCG",
            market_cap=target.market_cap * (1 + 0.1 * (i - n_peers / 2)),
            roe=0.20 + 0.01 * i,
        ))
    return peers


def test_find_peers_returns_min_peers_when_pool_is_adequate(monkeypatch):
    target = make_bundle()
    pool = _stub_pool(target, n_peers=8)
    monkeypatch.setattr(
        peer_identification, "_safe_fetch_many",
        lambda tickers, *, offline: pool,
    )
    ps = find_peers(target, universe={"X.NS": "FMCG"} | {p.ticker: "FMCG" for p in pool})
    assert isinstance(ps, PeerSet)
    assert len(ps.peers) >= 4
    assert len(ps.peers) <= 10
    assert "Mahalanobis" in ps.method


def test_find_peers_empty_set_when_universe_has_no_matching_sector(monkeypatch):
    target = make_bundle(sector="Information Technology")
    monkeypatch.setattr(
        peer_identification, "_safe_fetch_many",
        lambda tickers, *, offline: [],
    )
    ps = find_peers(target, universe={"X.NS": "FMCG"})
    assert ps.peers == []
    assert "insufficient" in ps.method.lower()


def test_find_peers_widens_size_band_when_needed(monkeypatch):
    """If the first 5× band has too few survivors, the function should
    auto-widen to 10× / 20× before giving up."""
    target = make_bundle(market_cap=1e11)
    # Build peers at ~15× target — they fail the 5× band but pass at 20×.
    pool = [
        make_bundle(ticker=f"BIG{i}.NS", sector="FMCG",
                    market_cap=1.5e12 * (1 + 0.05 * i))
        for i in range(6)
    ]
    monkeypatch.setattr(
        peer_identification, "_safe_fetch_many",
        lambda tickers, *, offline: pool,
    )
    ps = find_peers(target, universe={p.ticker: "FMCG" for p in pool})
    # Should successfully widen and keep enough peers.
    assert len(ps.peers) >= 4


def test_find_peers_debug_features_has_target_row(monkeypatch):
    target = make_bundle()
    pool = _stub_pool(target, n_peers=6)
    monkeypatch.setattr(
        peer_identification, "_safe_fetch_many",
        lambda tickers, *, offline: pool,
    )
    ps = find_peers(target, universe={p.ticker: "FMCG" for p in pool})
    assert ps.debug_features is not None
    assert "role" in ps.debug_features.columns
    assert (ps.debug_features["role"] == "target").sum() == 1
