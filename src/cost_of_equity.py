"""
Cost-of-equity engine.

The DDM is mathematically very sensitive to Ke — a 50 bps mis-estimate
shifts intrinsic value by 10-15%. So this module is paranoid about
getting Ke right:

  1. We compute beta from a 5-yr regression of weekly stock returns vs
     Nifty 50 (proxied by ^NSEI). We don't trust Yahoo's `beta` field
     because it's often stale.
  2. We Bloomberg-adjust the raw beta toward 1.0 (alpha=2/3 + 1/3*beta_raw).
  3. If we don't have enough price history (< 2 years), we fall back to
     the sector unlevered beta from Damodaran's India tables and re-lever
     using the firm's own D/E.
  4. CAPM is the base. We then add a country default spread modifier
     for distressed/illiquid names and a small-cap premium when the
     market cap is below ₹5,000 crore.

The output is a `CostOfEquity` object whose `ke` field is what every
DDM call expects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # tests / offline use will swap in a fake.

from config import (
    EQUITY_RISK_PREMIUM_IN,
    INDIA_DEFAULT_SPREAD,
    RISK_FREE_RATE_IN,
    SECTOR_UNLEVERED_BETA,
    SETTINGS,
    EFFECTIVE_TAX_RATE_IN,
)


_NIFTY_TICKER = "^NSEI"
_SMALL_CAP_THRESHOLD_INR = 50_000_000_000  # ₹5,000 crore
_SMALL_CAP_PREMIUM = 0.015                 # 150 bps add-on


@dataclass
class CostOfEquity:
    ke: float
    beta_raw: Optional[float]
    beta_adjusted: float
    risk_free: float
    erp: float
    method: str
    small_cap_premium: float = 0.0
    notes: list[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = [
            f"Cost of Equity = {self.ke:.2%}",
            f"  Method:                {self.method}",
            f"  Risk-free rate:        {self.risk_free:.2%}",
            f"  Equity risk premium:   {self.erp:.2%}",
            f"  Beta (raw):            {self.beta_raw if self.beta_raw is not None else '—'}",
            f"  Beta (adjusted):       {self.beta_adjusted:.3f}",
        ]
        if self.small_cap_premium:
            lines.append(f"  Small-cap add-on:      {self.small_cap_premium:.2%}")
        for n in self.notes:
            lines.append(f"  Note: {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Beta estimation
# ---------------------------------------------------------------------------
def estimate_beta_from_prices(
    stock_prices: pd.Series,
    market_prices: pd.Series,
) -> Optional[float]:
    """OLS slope of weekly log returns: stock vs market."""
    if stock_prices is None or market_prices is None:
        return None
    if stock_prices.empty or market_prices.empty:
        return None

    # Resample to weekly to reduce microstructure noise.
    sp = stock_prices.resample("W-FRI").last().dropna()
    mp = market_prices.resample("W-FRI").last().dropna()

    aligned = pd.concat([sp, mp], axis=1, join="inner").dropna()
    if len(aligned) < 60:  # ~14 months of weekly observations
        return None

    aligned.columns = ["s", "m"]
    rs = np.log(aligned["s"] / aligned["s"].shift(1)).dropna()
    rm = np.log(aligned["m"] / aligned["m"].shift(1)).dropna()
    rs, rm = rs.align(rm, join="inner")

    cov = np.cov(rs, rm, ddof=1)
    var_m = cov[1, 1]
    if var_m <= 0:
        return None
    return float(cov[0, 1] / var_m)


def adjusted_beta(beta_raw: float) -> float:
    """Bloomberg-style adjustment: 0.67 * 1 + 0.33 * raw."""
    return (
        SETTINGS.beta_adjustment_alpha * 1.0
        + SETTINGS.beta_adjustment_beta * beta_raw
    )


def relever_sector_beta(
    sector: str,
    debt_to_equity: float,
    tax_rate: float = EFFECTIVE_TAX_RATE_IN,
) -> float:
    """Hamada equation: β_levered = β_unlevered * [1 + (1 − t) * D/E]."""
    bu = SECTOR_UNLEVERED_BETA.get(sector, 1.0)
    return float(bu * (1 + (1 - tax_rate) * max(0.0, debt_to_equity)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def cost_of_equity(
    stock_prices: pd.Series,
    *,
    sector: str,
    market_cap: float,
    total_debt: float,
    equity: float,
    risk_free: float = RISK_FREE_RATE_IN,
    erp: float = EQUITY_RISK_PREMIUM_IN,
    add_default_spread: bool = False,
    market_prices: Optional[pd.Series] = None,
) -> CostOfEquity:
    """Compute Ke for a single Indian equity.

    The function will try, in order:

      1. CAPM with own beta from price regression.
      2. CAPM with sector unlevered beta re-levered via Hamada.

    Either way, the raw beta is Bloomberg-adjusted before CAPM is applied.
    """
    notes: list[str] = []

    # ----- 1. Own beta from prices -----
    if market_prices is None:
        market_prices = _download_market_prices()

    beta_raw = estimate_beta_from_prices(stock_prices, market_prices)
    method = "CAPM (own beta)"

    # ----- 2. Sector fallback -----
    if beta_raw is None or not np.isfinite(beta_raw) or abs(beta_raw) > 3.0:
        d_e = (total_debt / equity) if (equity and equity > 0) else 0.0
        beta_raw = relever_sector_beta(sector, d_e)
        method = "CAPM (sector-relevered beta)"
        notes.append(
            "Insufficient or unstable price history — falling back to "
            f"sector-relevered beta for '{sector}'."
        )

    beta_adj = adjusted_beta(beta_raw)

    # ----- CAPM core -----
    ke = risk_free + beta_adj * erp

    # ----- Modifiers -----
    small_cap_prem = 0.0
    if np.isfinite(market_cap) and market_cap < _SMALL_CAP_THRESHOLD_INR:
        small_cap_prem = _SMALL_CAP_PREMIUM
        ke += small_cap_prem
        notes.append("Small-cap premium of 150 bps applied.")

    if add_default_spread:
        ke += INDIA_DEFAULT_SPREAD
        notes.append(
            "India default spread added — appropriate only for distressed names."
        )

    return CostOfEquity(
        ke=float(ke),
        beta_raw=float(beta_raw),
        beta_adjusted=float(beta_adj),
        risk_free=float(risk_free),
        erp=float(erp),
        method=method,
        small_cap_premium=small_cap_prem,
        notes=notes,
    )


def _download_market_prices() -> pd.Series:
    """5-yr Nifty 50 close. Cached in module memory after first call."""
    if not hasattr(_download_market_prices, "_cache"):
        _download_market_prices._cache = None  # type: ignore
    if _download_market_prices._cache is not None:  # type: ignore
        return _download_market_prices._cache  # type: ignore

    if yf is None:
        return pd.Series(dtype=float)

    try:
        hist = yf.Ticker(_NIFTY_TICKER).history(period="5y", auto_adjust=True)
        prices = hist["Close"] if not hist.empty else pd.Series(dtype=float)
    except Exception:
        prices = pd.Series(dtype=float)

    _download_market_prices._cache = prices  # type: ignore
    return prices


# ---------------------------------------------------------------------------
# WACC — used by the EV-based relative valuation paths
# ---------------------------------------------------------------------------
def wacc(
    *,
    ke: float,
    market_cap: float,
    total_debt: float,
    pre_tax_kd: float = 0.085,
    tax_rate: float = EFFECTIVE_TAX_RATE_IN,
) -> float:
    """Standard WACC with Indian default tax rate."""
    v = market_cap + total_debt
    if v <= 0:
        return ke
    we = market_cap / v
    wd = total_debt / v
    return float(we * ke + wd * pre_tax_kd * (1 - tax_rate))
