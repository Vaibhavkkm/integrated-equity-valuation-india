"""
Empirical sign-convention check for yfinance 'Change In Working Capital'.

Method: in a standard cash-flow statement, the operating section is
  OCF ≈ NI + D&A + Other-Non-Cash + ΔWC_cashflow_view
where "Change in working capital" is the NET cash effect of the period
movement in operating assets/liabilities. Positive sign = cash freed
by WC reduction. We check which sign on ΔWC recovers reported OCF.

If yfinance follows the cash-flow-statement convention (the IFRS/Ind-AS
norm), substituting the ΔWC row *with its raw sign* recovers OCF —
i.e., the row is added, not subtracted.

Phase A FCFE math: the Damodaran textbook reads
  FCFE = NI − Net CapEx (1−DR) − ΔWC_textbook (1−DR) + Net Borrowing
where ΔWC_textbook is the *asset-side* increase. yfinance's ΔWC has the
opposite sign, so the formula becomes
  FCFE = NI − Net CapEx (1−DR) + ΔWC_yfinance (1−DR) + Net Borrowing
"""
from __future__ import annotations

import sys
import warnings

import yfinance as yf

warnings.simplefilter("ignore")


def check_ticker_year(ticker: str, period_idx: int):
    yt = yf.Ticker(ticker)
    cf = yt.cashflow
    if cf is None or cf.empty:
        print(f"{ticker}: cashflow empty")
        return None
    period = cf.columns[period_idx]
    print(f"\n--- {ticker} fiscal period ending {period.date()} ---")

    def get(label):
        return float(cf.at[label, period]) if label in cf.index else 0.0

    ni = get("Net Income From Continuing Operations")
    da = get("Depreciation And Amortization")
    wc = get("Change In Working Capital")
    oth = get("Other Non Cash Items") + get("Stock Based Compensation") + get("Deferred Tax")
    ocf = get("Operating Cash Flow")

    print(f"  NI (continuing ops)   : ₹{ni:>20,.0f}")
    print(f"  D&A                   : ₹{da:>20,.0f}")
    print(f"  ΔWC (yfinance)        : ₹{wc:>20,.0f}")
    print(f"  Other non-cash sum    : ₹{oth:>20,.0f}")
    print(f"  Reported OCF          : ₹{ocf:>20,.0f}")

    plus_branch = ni + da + wc + oth
    minus_branch = ni + da - wc + oth
    err_plus = abs(plus_branch - ocf) / abs(ocf) * 100 if ocf else float("inf")
    err_minus = abs(minus_branch - ocf) / abs(ocf) * 100 if ocf else float("inf")
    print(f"  NI + D&A + ΔWC + other: ₹{plus_branch:>20,.0f}  (off by {err_plus:.1f}%)")
    print(f"  NI + D&A − ΔWC + other: ₹{minus_branch:>20,.0f}  (off by {err_minus:.1f}%)")
    return ("add" if err_plus < err_minus else "subtract")


def main():
    verdicts = []
    # TCS FY26 (most recent) and FY24 (the year referenced in the spec)
    verdicts.append(("TCS FY26", check_ticker_year("TCS.NS", 0)))
    verdicts.append(("TCS FY24", check_ticker_year("TCS.NS", 2)))
    # ITC FY26 (cross-sector + cross-firm sanity)
    verdicts.append(("ITC FY26", check_ticker_year("ITC.NS", 0)))
    verdicts.append(("ITC FY24", check_ticker_year("ITC.NS", 2)))

    print("\n\n=== VERDICT TABLE ===")
    for label, v in verdicts:
        print(f"  {label:<12s}  → ΔWC must be {v} to recover OCF")
    if all(v == "add" for _, v in verdicts if v is not None):
        print("\nUNANIMOUS: yfinance ΔWC follows cash-flow-statement convention.")
        print("           Positive = cash freed by WC reduction.")
        print("           Phase A FCFE formula: + ΔWC_yfinance × (1−DR)")
    else:
        print("\nINCONSISTENT verdicts — manual review required.")


if __name__ == "__main__":
    main()
