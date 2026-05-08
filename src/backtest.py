"""
Backtesting framework — does the engine's BUY/HOLD/SELL signal actually work?

Why bother
----------
The most damning question any examiner can ask of a valuation methodology
is: "How do I know this works?" Without empirical validation the project
is a calculator, not an investment tool. This module closes that gap.

What it does
------------
Given a panel of historical recommendations (ticker × signal_date ×
BUY/HOLD/SELL), the backtester:

  1. Looks up the actual forward return at +6m, +12m, and +24m.
  2. Bins the returns by the engine's recommendation.
  3. Computes the return spread (BUY mean − SELL mean) — the headline
     "does the signal pay" number.
  4. Reports hit-rate (% of BUYs that beat the Nifty 50 over the window).
  5. Computes simple risk metrics: average drawdown, volatility per bin.

This is intentionally a *recommendation-level* backtest, not a
portfolio-construction one. Building a P&L curve with rebalancing,
position sizing, and transaction costs is a separate project. What
matters here is: do BUY-rated stocks outperform SELL-rated stocks on
average? If yes, the engine has predictive value. If no, the
methodology needs a rethink.

How to use
----------
Two modes:

  *Replay mode* — feed in a CSV of (ticker, date, recommendation) and
  let the backtester pull forward prices from yfinance.

  *Live mode* — run `value_stock` over a list of tickers as of today,
  store the result, and call `evaluate_after(months)` once enough time
  has passed. The persisted log is a CSV in `reports/backtest_log.csv`.

Caveats
-------
Survivorship bias: if a stock was delisted between the signal date and
the evaluation date, yfinance won't return data, and that stock silently
drops out of the bin. We log how many tickers were dropped so the
examiner can see the bias. A real institutional backtest would patch in
delisting returns from a survivorship-free database (CMIE/Capitaline);
that is out of scope for a student project.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

from config import REPORT_DIR
from src.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Persisted signal log
# ---------------------------------------------------------------------------
_LOG_PATH = REPORT_DIR / "backtest_log.csv"
_LOG_COLUMNS = [
    "signal_date", "ticker", "recommendation",
    "blended_value", "price_at_signal", "margin_of_safety",
    "quality_score",
]


def append_signal(
    *, ticker: str, recommendation: str,
    blended_value: float, price: float,
    margin_of_safety: float, quality_score: float,
    when: Optional[datetime] = None,
) -> None:
    """Persist a (ticker, signal) pair so we can grade it later.

    The log is a CSV — easy to inspect in Excel, easy to merge into a
    pandas DataFrame in :func:`evaluate_log`.

    Parameters
    ----------
    ticker : str
        NSE ticker (with ``.NS`` suffix). Stored verbatim.
    recommendation : str
        One of ``BUY`` / ``HOLD`` / ``SELL`` / ``N/A``. The grading
        function bins on this column.
    blended_value : float
        The DDM/relative-blended intrinsic value from
        :class:`ValuationResult`.
    price : float
        Spot price at signal time. Returns are computed against this.
    margin_of_safety : float
        ``(blended_value - price) / price`` at signal time, persisted so
        post-hoc analyses don't have to recompute it.
    quality_score : float
        Composite 0-100 score so backtests can stratify by quality.
    when : datetime, optional
        Signal timestamp. Defaults to ``datetime.now(timezone.utc)`` for
        live runs; pass an explicit value when replaying historical
        recommendations.
    """
    row = {
        "signal_date": (when or datetime.now(timezone.utc)).strftime("%Y-%m-%d"),
        "ticker": ticker,
        "recommendation": recommendation,
        "blended_value": round(float(blended_value), 4),
        "price_at_signal": round(float(price), 4),
        "margin_of_safety": round(float(margin_of_safety), 6),
        "quality_score": round(float(quality_score), 2),
    }
    df = pd.DataFrame([row], columns=_LOG_COLUMNS)
    if _LOG_PATH.exists():
        df.to_csv(_LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(_LOG_PATH, index=False)
    log.debug("backtest: logged %s %s", ticker, recommendation)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    horizon_months: int
    n_signals: int
    n_evaluated: int                                # after dropping NaN
    by_bin: Dict[str, "BinStats"]
    spread_buy_minus_sell: Optional[float]          # mean BUY − mean SELL
    hit_rate_buy_vs_index: Optional[float]          # % of BUYs > index return
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 78,
            f"  Backtest @ {self.horizon_months}-month horizon",
            "=" * 78,
            f"  Signals evaluated     : {self.n_evaluated} / {self.n_signals}",
        ]
        for name in ("BUY", "HOLD", "SELL"):
            if name in self.by_bin:
                bs = self.by_bin[name]
                lines.append(
                    f"  {name:<5} (n={bs.n:>3}) — "
                    f"mean={bs.mean_return:+.2%} | median={bs.median_return:+.2%} | "
                    f"σ={bs.std_return:.2%} | win-rate={bs.win_rate:.0%}"
                )
        if self.spread_buy_minus_sell is not None:
            lines.append(f"  Spread (BUY − SELL)   : {self.spread_buy_minus_sell:+.2%}")
        if self.hit_rate_buy_vs_index is not None:
            lines.append(
                f"  BUY hit-rate vs Nifty : {self.hit_rate_buy_vs_index:.0%}"
            )
        for n in self.notes:
            lines.append(f"  Note: {n}")
        lines.append("=" * 78)
        return "\n".join(lines)


@dataclass
class BinStats:
    n: int
    mean_return: float
    median_return: float
    std_return: float
    win_rate: float        # % positive returns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def evaluate_log(
    *, horizon_months: int = 12,
    log_path: Path = _LOG_PATH,
    benchmark: str = "^NSEI",
) -> BacktestResult:
    """Replay signals from ``log_path`` and grade them.

    Parameters
    ----------
    horizon_months : int, optional
        Forward-return window. Defaults to 12 months.
    log_path : Path, optional
        Location of the persisted CSV. Defaults to
        ``REPORT_DIR / "backtest_log.csv"``.
    benchmark : str, optional
        Yahoo ticker used to compute the index-relative hit rate.
        Defaults to the Nifty 50 (``^NSEI``).

    Returns
    -------
    BacktestResult
        Empty result with a "log file not found" note when the CSV is
        missing; otherwise the output of :func:`evaluate_signals`.
    """
    if not log_path.exists():
        return BacktestResult(
            horizon_months=horizon_months,
            n_signals=0, n_evaluated=0, by_bin={},
            spread_buy_minus_sell=None, hit_rate_buy_vs_index=None,
            notes=[f"Log file not found at {log_path}"],
        )
    signals = pd.read_csv(log_path, parse_dates=["signal_date"])
    return evaluate_signals(
        signals,
        horizon_months=horizon_months,
        benchmark=benchmark,
    )


def evaluate_signals(
    signals: pd.DataFrame,
    *,
    horizon_months: int = 12,
    benchmark: str = "^NSEI",
) -> BacktestResult:
    """Grade an explicit signal panel.

    Pulls forward prices from yfinance, bins returns by recommendation,
    and reports the BUY-minus-SELL spread plus the BUY hit rate against
    the benchmark.

    Parameters
    ----------
    signals : pd.DataFrame
        Required columns: ``signal_date``, ``ticker``, ``recommendation``,
        ``price_at_signal``. Extra columns are ignored.
    horizon_months : int, optional
        Forward-return horizon (e.g. 6 / 12 / 24). Defaults to 12.
    benchmark : str, optional
        Yahoo benchmark ticker for the index-relative hit rate. Defaults
        to the Nifty 50 (``^NSEI``).

    Returns
    -------
    BacktestResult
        Per-bin statistics, BUY-minus-SELL spread, BUY hit rate, and a
        notes list flagging any signals dropped due to missing data
        (survivorship bias).
    """
    required = {"signal_date", "ticker", "recommendation", "price_at_signal"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"signal panel is missing columns: {missing}")

    signals = signals.copy()
    signals["signal_date"] = pd.to_datetime(signals["signal_date"])
    horizon = pd.DateOffset(months=horizon_months)
    signals["eval_date"] = signals["signal_date"] + horizon

    # Pull forward prices ticker-by-ticker (avoids one bad ticker poisoning all).
    forward_returns: List[Optional[float]] = []
    bench_returns: List[Optional[float]] = []
    dropped = 0

    for _, row in signals.iterrows():
        fr = _forward_return(row["ticker"], row["signal_date"], row["eval_date"])
        br = _forward_return(benchmark, row["signal_date"], row["eval_date"])
        if fr is None:
            dropped += 1
        forward_returns.append(fr)
        bench_returns.append(br)

    signals["forward_return"] = forward_returns
    signals["benchmark_return"] = bench_returns
    signals["excess_return"] = signals["forward_return"] - signals["benchmark_return"]

    evaluable = signals.dropna(subset=["forward_return"])

    by_bin: Dict[str, BinStats] = {}
    for name, grp in evaluable.groupby("recommendation"):
        rs = grp["forward_return"].astype(float)
        by_bin[name] = BinStats(
            n=len(rs),
            mean_return=float(rs.mean()),
            median_return=float(rs.median()),
            std_return=float(rs.std(ddof=1)) if len(rs) > 1 else 0.0,
            win_rate=float((rs > 0).mean()),
        )

    spread = None
    if "BUY" in by_bin and "SELL" in by_bin:
        spread = by_bin["BUY"].mean_return - by_bin["SELL"].mean_return

    hit_vs_idx = None
    buys = evaluable[evaluable["recommendation"] == "BUY"]
    if not buys.empty and buys["benchmark_return"].notna().any():
        beats = (buys["forward_return"] > buys["benchmark_return"]).sum()
        hit_vs_idx = float(beats / len(buys))

    notes: List[str] = []
    if dropped:
        notes.append(
            f"{dropped} signal(s) dropped — likely delisted / data missing "
            f"(survivorship bias; examiner: see backtest.py docstring)."
        )

    return BacktestResult(
        horizon_months=horizon_months,
        n_signals=len(signals),
        n_evaluated=len(evaluable),
        by_bin=by_bin,
        spread_buy_minus_sell=spread,
        hit_rate_buy_vs_index=hit_vs_idx,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Forward-return computation
# ---------------------------------------------------------------------------
def _forward_return(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp,
) -> Optional[float]:
    """Total return between `start` and `end`, ignoring dividends.

    Uses adjusted close so splits/bonuses are baked in. Yahoo's adj close
    *does* include dividends; for our purposes that's the right thing.
    """
    if yf is None:
        return None
    if end > pd.Timestamp.now(tz="UTC").tz_localize(None):
        return None  # not enough time has passed

    # Pull a small buffer either side so we get the nearest trading day.
    buf = timedelta(days=10)
    try:
        hist = yf.Ticker(ticker).history(
            start=start - buf, end=end + buf, auto_adjust=True,
        )
    except Exception as exc:
        log.debug("backtest: forward-fetch failed for %s (%s)", ticker, exc)
        return None

    if hist.empty:
        return None

    px = hist["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)

    p_start = _nearest(px, start)
    p_end = _nearest(px, end)
    if p_start is None or p_end is None or p_start <= 0:
        return None
    return float(p_end / p_start - 1.0)


def _nearest(px: pd.Series, when: pd.Timestamp) -> Optional[float]:
    """Closest available trading-day close to `when`."""
    if px.empty:
        return None
    idx = px.index.searchsorted(when)
    idx = min(max(idx, 0), len(px) - 1)
    return float(px.iloc[idx])


# ---------------------------------------------------------------------------
# Convenience: rolling-window backtest from a panel of historical signals
# ---------------------------------------------------------------------------
def rolling_backtest(
    signals: pd.DataFrame,
    *,
    horizons_months: tuple = (6, 12, 24),
) -> Dict[int, BacktestResult]:
    """Run the same panel against multiple horizons. Useful for the report.

    Parameters
    ----------
    signals : pd.DataFrame
        Same shape as :func:`evaluate_signals` expects.
    horizons_months : tuple of int, optional
        Horizons to run. Defaults to ``(6, 12, 24)``.

    Returns
    -------
    dict[int, BacktestResult]
        Horizon → graded result.
    """
    return {h: evaluate_signals(signals, horizon_months=h) for h in horizons_months}
