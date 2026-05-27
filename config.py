"""
Central config for the Indian equity valuation engine.

Everything that looks like a "magic number" lives here so that the
supervisor (or a future me) can reset the country-level assumptions in one
place without combing through model code. All rates are decimals — 0.071
means 7.1%, not 71%.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
REPORT_DIR = ROOT / "reports"

for _p in (DATA_DIR, CACHE_DIR, REPORT_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Country-level capital market inputs (India, calibrated to early 2026)
# ---------------------------------------------------------------------------
# 10-yr GoI G-Sec yield is the canonical risk-free rate for Indian DCFs.
# Source: RBI weekly statistical supplement; refresh quarterly.
RISK_FREE_RATE_IN: float = 0.0710

# Damodaran's India ERP (mature-market premium + country risk premium).
EQUITY_RISK_PREMIUM_IN: float = 0.0700

# Long-run nominal GDP growth — used as the ceiling for terminal growth.
# A perpetuity growth assumption above this is mathematically nonsense.
LONG_RUN_NOMINAL_GROWTH_IN: float = 0.060

# Effective marginal corporate tax rate (post 22% concessional regime).
EFFECTIVE_TAX_RATE_IN: float = 0.2517


# ---------------------------------------------------------------------------
# Model knobs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSettings:
    # Blend weights — DDM vs Relative. Defaults are 50/50 per the project
    # brief, but credibility-weighted blending in integrated_valuation.py
    # may shift these on a per-stock basis.
    ddm_weight: float = 0.50
    relative_weight: float = 0.50

    # Beta adjustment toward 1.0 a la Bloomberg. Empirically improves
    # forward-looking accuracy.
    beta_adjustment_alpha: float = 2.0 / 3.0
    beta_adjustment_beta: float = 1.0 / 3.0

    # DDM stage lengths (years) for the 2-stage and 3-stage variants.
    high_growth_years: int = 5
    transition_years: int = 5

    # Margin of safety bands for the recommendation logic.
    buy_threshold: float = 0.20
    sell_threshold: float = -0.15

    # Monte Carlo
    mc_paths: int = 10_000
    mc_seed: int = 42

    # Peer selection
    n_peer_clusters: int = 5
    min_peers: int = 4
    max_peers: int = 10

    # Outlier trimming for relative valuation (each tail).
    multiple_trim_pct: float = 0.10

    # ------------------------------------------------------------------
    # FCFE (Phase A)
    # ------------------------------------------------------------------
    # Two-stage explicit forecast horizon. The final FCFE_TAPER_YEARS
    # years fade growth linearly from g1 to terminal_g.
    fcfe_stage1_years: int = 7
    fcfe_taper_years: int = 2
    # Stage-1 growth cap before Bayesian shrinkage.
    fcfe_g1_cap: float = 0.15
    # Debt ratio for net-borrowing assumption: 5y average D/(D+E),
    # hard-capped here to prevent over-levered firms from amplifying
    # the (1−DR) multiplier on net CapEx and ΔWC.
    fcfe_debt_ratio_cap: float = 0.60
    # Stage-1 growth must sit at least this far below Ke so the
    # division in the Gordon perpetuity stays well-conditioned.
    fcfe_ke_margin: float = 0.005
    # Minimum years of cash-flow history before fcfe_applicable()
    # returns True. The Phase A.0 audit found a median 4y coverage
    # for non-financials; a 5y floor would over-exclude. See
    # `project_phase_a_directives.md` directive #1.
    fcfe_min_history_years: int = 3

    # Three-way blend payout-rule splits (only used when FCFE
    # applicable). Each tuple is (fcfe_share, rel_share); fcfe_share +
    # rel_share = 1.0. The slot the DDM doesn't claim is split per
    # these on the firm's TTM payout ratio.
    fcfe_split_low_payout: tuple = (0.70, 0.30)     # payout < 20%
    fcfe_split_mid_payout: tuple = (0.50, 0.50)     # 20% ≤ payout < 50%
    fcfe_split_high_payout: tuple = (0.30, 0.70)    # payout ≥ 50%


SETTINGS = ModelSettings()


# ---------------------------------------------------------------------------
# Sector → indicative unlevered beta (Damodaran India industry tables, 2026).
# Used as a Bayesian prior when a stock has insufficient price history for
# its own beta to be reliable.
# ---------------------------------------------------------------------------
SECTOR_UNLEVERED_BETA: Dict[str, float] = {
    "Information Technology":   0.95,
    "Banking":                  1.10,
    "NBFC":                     1.20,
    "FMCG":                     0.65,
    "Auto":                     1.05,
    "Pharma":                   0.80,
    "Metals":                   1.30,
    "Oil & Gas":                0.95,
    "Telecom":                  0.85,
    "Cement":                   1.00,
    "Capital Goods":            1.10,
    "Power":                    0.75,
    "Real Estate":              1.25,
    "Chemicals":                1.00,
    "Consumer Durables":        0.95,
    "Media":                    1.15,
    "Textiles":                 1.05,
    "Internet & Platform":      1.40,
    "Diversified":              1.00,
}


# A trimmed Nifty-500 sector mapping. Extend in data/nifty500_universe.csv.
DEFAULT_UNIVERSE: Dict[str, str] = {
    # IT
    "TCS.NS": "Information Technology",
    "INFY.NS": "Information Technology",
    "WIPRO.NS": "Information Technology",
    "HCLTECH.NS": "Information Technology",
    "TECHM.NS": "Information Technology",
    # Banking
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "SBIN.NS": "Banking",
    "AXISBANK.NS": "Banking",
    "KOTAKBANK.NS": "Banking",
    "INDUSINDBK.NS": "Banking",
    # FMCG
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG",
    "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG",
    "DABUR.NS": "FMCG",
    "MARICO.NS": "FMCG",
    "GODREJCP.NS": "FMCG",
    # Auto
    "MARUTI.NS": "Auto",
    "M&M.NS": "Auto",
    "TVSMOTOR.NS": "Auto",
    "BAJAJ-AUTO.NS": "Auto",
    "EICHERMOT.NS": "Auto",
    "HEROMOTOCO.NS": "Auto",
    # Pharma
    "SUNPHARMA.NS": "Pharma",
    "DRREDDY.NS": "Pharma",
    "CIPLA.NS": "Pharma",
    "DIVISLAB.NS": "Pharma",
    "LUPIN.NS": "Pharma",
    "TORNTPHARM.NS": "Pharma",
    # Metals
    "TATASTEEL.NS": "Metals",
    "JSWSTEEL.NS": "Metals",
    "HINDALCO.NS": "Metals",
    "VEDL.NS": "Metals",
    "COALINDIA.NS": "Metals",
    # Oil & Gas
    "RELIANCE.NS": "Oil & Gas",
    "ONGC.NS": "Oil & Gas",
    "BPCL.NS": "Oil & Gas",
    "IOC.NS": "Oil & Gas",
    "GAIL.NS": "Oil & Gas",
    # Telecom
    "BHARTIARTL.NS": "Telecom",
    "IDEA.NS": "Telecom",
    # Cement
    "ULTRACEMCO.NS": "Cement",
    "SHREECEM.NS": "Cement",
    "AMBUJACEM.NS": "Cement",
    "ACC.NS": "Cement",
    # Capital Goods (incl. mid-cap infra / road EPC so small-cap targets
    # like HGINFRA, KNRCON have size-matched peers within the 20x band).
    "LT.NS": "Capital Goods",
    "SIEMENS.NS": "Capital Goods",
    "ABB.NS": "Capital Goods",
    "BHEL.NS": "Capital Goods",
    "HGINFRA.NS": "Capital Goods",
    "KNRCON.NS": "Capital Goods",
    "PNCINFRA.NS": "Capital Goods",
    "GRINFRA.NS": "Capital Goods",
    "ASHOKA.NS": "Capital Goods",
    "NCC.NS": "Capital Goods",
    "KEC.NS": "Capital Goods",
    "IRB.NS": "Capital Goods",
    "RVNL.NS": "Capital Goods",
    "IRCON.NS": "Capital Goods",
    # Power
    "NTPC.NS": "Power",
    "POWERGRID.NS": "Power",
    "TATAPOWER.NS": "Power",
    "ADANIPOWER.NS": "Power",
    # NBFC
    "BAJFINANCE.NS": "NBFC",
    "BAJAJFINSV.NS": "NBFC",
    "HDFCLIFE.NS": "NBFC",
    "SBILIFE.NS": "NBFC",
    # Consumer Durables
    "TITAN.NS": "Consumer Durables",
    "HAVELLS.NS": "Consumer Durables",
    "VOLTAS.NS": "Consumer Durables",
    # Internet & Platform — new-age listings (Zomato/Eternal, Swiggy,
    # Nykaa, Paytm, Policybazaar, IRCTC, Delhivery). Most are loss-making
    # or thin-margin, so PE/PEG are unreliable; the relative-valuation
    # weights (see SECTOR_OVERRIDES) lean on P/S and EV/EBITDA instead.
    "ETERNAL.NS": "Internet & Platform",
    "SWIGGY.NS": "Internet & Platform",
    "NYKAA.NS": "Internet & Platform",
    "PAYTM.NS": "Internet & Platform",
    "POLICYBZR.NS": "Internet & Platform",
    "IRCTC.NS": "Internet & Platform",
    "DELHIVERY.NS": "Internet & Platform",
}


# Reporting metadata
PROJECT_TITLE = "Integrated Equity Valuation of Indian Stocks"
PROJECT_SUBTITLE = (
    "DDM + FCFE + Relative Valuation, blended per-stock by credibility "
    "(payout-aware weighting; FCFE auto-skipped when not applicable)"
)
AUTHOR_NAME = "Vaibhav Mangroliya"
SUPERVISOR_NAME = "Dr. Senthil Murugan NAGARAJAN"
INSTITUTION = "Student Project — Semester IV"
