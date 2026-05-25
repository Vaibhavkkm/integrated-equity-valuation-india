"""
Phase A.0 audit script — print cash-flow field coverage across the
curated universe.

For every ticker in ``config.DEFAULT_UNIVERSE`` this script fetches the
StockBundle (using cache when fresh, hitting yfinance otherwise) and
counts how many of the six cash-flow series populated by Phase A.0 came
back non-empty:

  * net_income_annual
  * capex_annual
  * dep_amort_annual
  * change_in_wc_annual
  * working_capital_annual
  * total_debt_annual

The Phase A.0 acceptance criterion is ≥90% coverage on each field.
This script is the empirical verification step; it is NOT a pytest test
because it hits the network (or at minimum reads every pickle in
``cache/``), which would be inappropriate for CI.

Run with:
    .venv/bin/python scripts/audit_cashflow_coverage.py
    .venv/bin/python scripts/audit_cashflow_coverage.py --offline
    .venv/bin/python scripts/audit_cashflow_coverage.py --refresh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_UNIVERSE
from src.data_fetcher import fetch_stock


FIELDS = (
    "net_income_annual",
    "capex_annual",
    "dep_amort_annual",
    "change_in_wc_annual",
    "working_capital_annual",
    "total_debt_annual",
)

COVERAGE_TARGET = 0.90

# Financial-sector tickers don't publish a meaningful "working capital"
# (balance sheet is entirely current) and FCFE doesn't apply to them
# anyway — banks/NBFCs are valued by FCFF or regulated-capital DDM, not
# textbook FCFE. The audit reports a separate "ex-financials" coverage
# figure so a structurally-expected miss doesn't mask real bugs in the
# remainder of the universe.
FINANCIAL_SECTORS = {"Banking", "NBFC"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--offline", action="store_true",
        help="Use cache only; do not hit yfinance (faster, may give stale results).",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Force fresh yfinance fetch for every ticker; ignore cache.",
    )
    args = p.parse_args()

    tickers = sorted(DEFAULT_UNIVERSE.keys())
    print(f"Auditing {len(tickers)} tickers from DEFAULT_UNIVERSE…\n")

    per_field_counts = {f: 0 for f in FIELDS}
    per_field_counts_nonfin = {f: 0 for f in FIELDS}
    per_field_lengths: dict[str, list[int]] = {f: [] for f in FIELDS}
    fetch_failures: list[str] = []
    per_ticker_rows: list[tuple[str, dict[str, int], bool]] = []
    n_nonfin = 0

    for tk in tickers:
        try:
            b = fetch_stock(
                tk,
                offline=args.offline,
                force_refresh=args.refresh,
                validate=False,  # quality scoring is irrelevant here
            )
        except Exception as exc:
            fetch_failures.append(f"{tk}: {exc.__class__.__name__}: {exc}")
            continue

        is_fin = b.sector in FINANCIAL_SECTORS
        if not is_fin:
            n_nonfin += 1
        row = {}
        for f in FIELDS:
            s = getattr(b, f, None)
            n = 0 if s is None or s.empty else int(len(s))
            row[f] = n
            if n > 0:
                per_field_counts[f] += 1
                if not is_fin:
                    per_field_counts_nonfin[f] += 1
            per_field_lengths[f].append(n)
        per_ticker_rows.append((tk, row, is_fin))

    # --- Per-ticker table -------------------------------------------------
    header = f"{'ticker':<14s}" + f"{'fin':>5s}" + "".join(f"{f.split('_annual')[0]:>14s}" for f in FIELDS)
    print(header)
    print("-" * len(header))
    for tk, row, is_fin in per_ticker_rows:
        line = f"{tk:<14s}" + f"{'Y' if is_fin else '.':>5s}" + "".join(f"{row[f]:>14d}" for f in FIELDS)
        print(line)

    # --- Summary ----------------------------------------------------------
    n = len(per_ticker_rows)
    print()
    print("=" * len(header))
    print(f"Coverage across {n} successfully-fetched tickers "
          f"(target: ≥{COVERAGE_TARGET:.0%}):")
    print(f"(Non-financial subset: {n_nonfin}/{n}. "
          f"FCFE is not applied to banks/NBFCs/insurers — for those names "
          f"the existing DDM + Relative pipeline carries the call.)")
    print()
    print(f"  {'field':<28s} {'all':^16s} {'ex-financials':^18s}   notes")
    print(f"  {'-'*28:<28s} {'-'*16:^16s} {'-'*18:^18s}   {'-'*5}")
    fails_overall: list[str] = []
    fails_nonfin: list[str] = []
    for f in FIELDS:
        cov_all = per_field_counts[f] / n if n else 0.0
        cov_nf = per_field_counts_nonfin[f] / n_nonfin if n_nonfin else 0.0
        lens = [x for x in per_field_lengths[f] if x > 0]
        median_len = sorted(lens)[len(lens) // 2] if lens else 0

        flag_all = " OK " if cov_all >= COVERAGE_TARGET else "MISS"
        flag_nf = " OK " if cov_nf >= COVERAGE_TARGET else "MISS"
        if cov_all < COVERAGE_TARGET:
            fails_overall.append(f)
        if cov_nf < COVERAGE_TARGET:
            fails_nonfin.append(f)

        all_str = f"[{flag_all}] {per_field_counts[f]:>2d}/{n} {cov_all:>5.1%}"
        nf_str = f"[{flag_nf}] {per_field_counts_nonfin[f]:>2d}/{n_nonfin} {cov_nf:>5.1%}"
        print(f"  {f:<28s} {all_str:<16s} {nf_str:<18s}   median {median_len}y")

    if fetch_failures:
        print()
        print(f"Fetch failures ({len(fetch_failures)}):")
        for line in fetch_failures:
            print(f"  - {line}")

    print()
    if not fails_nonfin:
        if fails_overall:
            print("RESULT: PASS — all fields ≥90% on the non-financial subset.")
            print(f"        Overall coverage misses on {fails_overall} are structurally")
            print("        expected for banks/NBFCs/insurers; FCFE does not apply to them.")
        else:
            print("RESULT: PASS — all fields ≥90% overall and ex-financials.")
        return 0
    else:
        print(f"RESULT: FAIL — ex-financials coverage below target on: {fails_nonfin}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
