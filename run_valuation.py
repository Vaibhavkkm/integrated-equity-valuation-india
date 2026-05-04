"""
Command-line entry point.

Examples
--------
    python run_valuation.py --ticker TCS.NS
    python run_valuation.py --ticker HDFCBANK.NS --report
    python run_valuation.py --ticker ITC.NS --rf 0.0710 --erp 0.075
    python run_valuation.py --batch TCS.NS INFY.NS WIPRO.NS --report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python run_valuation.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EQUITY_RISK_PREMIUM_IN, RISK_FREE_RATE_IN
from src.integrated_valuation import value_stock
from src.report_generator import generate_pdf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Integrated Equity Valuation of Indian Stocks (DDM + Relative)",
    )
    p.add_argument("--ticker", help="NSE ticker, e.g. TCS.NS")
    p.add_argument("--batch", nargs="+", help="Multiple tickers to value in one go")
    p.add_argument("--rf", type=float, default=RISK_FREE_RATE_IN,
                   help=f"Risk-free rate (default {RISK_FREE_RATE_IN:.4f})")
    p.add_argument("--erp", type=float, default=EQUITY_RISK_PREMIUM_IN,
                   help=f"Equity risk premium (default {EQUITY_RISK_PREMIUM_IN:.4f})")
    p.add_argument("--g-terminal", type=float, default=0.045,
                   help="Terminal growth rate (default 0.045)")
    p.add_argument("--offline", action="store_true",
                   help="Use cached data only — fail if not present")
    p.add_argument("--force-refresh", action="store_true",
                   help="Bust the cache and re-fetch from source")
    p.add_argument("--report", action="store_true",
                   help="Generate a PDF research note")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-step progress output")
    return p.parse_args()


def _run_one(ticker: str, args) -> None:
    result = value_stock(
        ticker,
        risk_free=args.rf,
        erp=args.erp,
        g_terminal=args.g_terminal,
        offline=args.offline,
        force_refresh=args.force_refresh,
        verbose=not args.quiet,
    )
    print(result.summary())

    if args.report:
        path = generate_pdf(result)
        print(f"  📄 Report written to: {path}")


def main() -> int:
    args = parse_args()
    if not args.ticker and not args.batch:
        print("ERROR: pass --ticker or --batch.", file=sys.stderr)
        return 2

    tickers = args.batch if args.batch else [args.ticker]
    for t in tickers:
        try:
            _run_one(t, args)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            return 130
        except Exception as e:
            print(f"\n!! {t}: {e.__class__.__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
