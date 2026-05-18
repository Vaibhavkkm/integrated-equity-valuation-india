# Integrated Equity Valuation — Indian Stocks

> Type a ticker. Get back what the stock is *really* worth, how confident
> the engine is in that number, and what growth assumptions the market
> is already pricing in.
>
> A SEM-4 student project that grew into a research-grade valuation
> engine for the Indian capital markets (NSE / BSE).

---

> ## ⚠️ All Rights Reserved — Read Before Using
>
> This repository is publicly visible **for academic review and recruiter
> visibility only.** It is **not open-source** in any conventional sense.
>
> - **Viewing & forking via GitHub:** allowed.
> - **Copying any portion into your own project, repository, coursework,
>   thesis, or assignment:** **strictly prohibited and constitutes
>   plagiarism.**
> - **Commercial use, redistribution, or use in ML training datasets:**
>   prohibited without prior written permission.
>
> This is original coursework submitted as a Semester IV Student Project.
> Git commit history (with timestamps) establishes authorship. If you
> submit any part of this work as your own, your institution will be
> notified.
>
> Full terms: **[LICENSE](./LICENSE)**.
>
> *Author: Vaibhav Mangroliya · Supervisor: Mr. Senthil Nagarajan · 2026.*

---

## The 30-second pitch

Most equity valuation projects stop at one of two things: a textbook
Dividend Discount Model, or a peer-comparison spreadsheet. This one
runs **both tracks side-by-side**, blends them with weights that *adapt
per stock*, stress-tests the answer with a 10,000-path Monte Carlo,
solves the **reverse problem** (what growth is the market pricing in?),
and grades its own historical recommendations against forward returns.

In one diagram:

```
   ticker
     │
     ▼
 ┌────────────────────┐
 │  fetch + validate  │──► data quality 0-100 (gates the rest)
 └─────────┬──────────┘
           │
   ┌───────┴────────┐
   ▼                ▼
 DDM track     Relative track
 (4 variants,  (P/E, P/B, P/S,
  auto-picked) EV/EBITDA, PEG;
   ▼          peers via K-Means
 Bayes-shrunk + Mahalanobis)
 growth        ▼
   ▼          sector-aware
 intrinsic    multiple weights
 value        ▼
   │          implied price
   └────┬─────┘
        ▼
   credibility-weighted blend
   (tilt ±15 pp from 50/50 by
    dividend quality + earnings
    track length)
        ▼
   Monte Carlo (10k paths)
        ▼
   reverse DCF — implied growth
        ▼
   BUY / HOLD / SELL
   + HIGH / MEDIUM / LOW confidence
        ▼
   PDF report + backtest log
```

---

## Why this exists

A textbook DDM is theoretically clean but breaks the moment you point
it at a non-payer. Relative valuation sidesteps that but inherits
whatever mis-pricing exists in the peer set. Most academic treatments
hand-wave both problems with "blend them 50/50."

The aim here was to make the blend *honest*: tilt toward whichever
track is more defensible for the firm being valued, quantify how sure
we should be, and — crucially — actually check whether the resulting
BUY/SELL calls have predictive value.

---

## What's actually inside

### 1. Two valuation tracks, four DDM flavors

The DDM module auto-picks one of:

| Variant | When it's chosen |
|---|---|
| **Gordon Growth** | Mature, stable payer (dividend history ≥ 5y, payout ≥ 40%, low growth) |
| **Two-Stage** | Clear high-growth phase that fades to a known horizon |
| **H-Model** (Fuller & Hsia, 1984) | Fast-growing payer — growth declines linearly, no cliff |
| **Three-Stage** | Mid-cycle firm; explicit high → linear fade → stable terminal |

The relative track computes five multiples — **P/E, P/B, P/S, EV/EBITDA,
PEG** — aggregates each with a **trimmed harmonic mean** (multiples are
ratios; arithmetic averaging biases them upward), then combines them
using **sector-aware weights** (P/B carries 50% for banks; EV/EBITDA is
zero for banks but 50% for cement).

### 2. Smarter peers than "same sector"

Peer identification uses unsupervised clustering, not just GICS
membership. The pipeline standardizes firms on a six-feature ratio
vector (size, ROE, leverage, payout, growth, margin), runs **K-Means**,
keeps the cluster the target sits in, then ranks by **Mahalanobis
distance** so peers that *look like* the target — not just share a
sector label — float to the top.

This matters in India because "FMCG" alone bundles ITC (cigarettes,
30%+ EBITDA margin) with Nestle (packaged foods, premium multiple).
Averaging their P/Es is nonsense; clustering separates them properly.

### 3. Quality scoring decides how much to trust each track

Four checks run before the blend:
- **Piotroski F-Score** (0-9) — fundamental health
- **Altman Z'' (EM variant)** — distress likelihood for emerging-market firms
- **Dividend Quality** (0-10) — track length, cuts, payout sanity, growth
- **Earnings Momentum** (0-10) — recent quarterly trend vs the K-Means peer set

These fold into a 0-100 composite (weights 30/25/25/20) that *modulates
the blend weight*. A stock with a 9-year clean dividend record gets DDM
bumped up. A non-payer gets DDM zeroed out and pure relative valuation.
The tilt is **asymmetric on purpose**: up to **+15 pp** toward DDM (for
mature payers with strong track records), and up to **−40 pp** away from
DDM (because the DDM mechanically under-prices low-payout retainers, so
when the two tracks disagree sharply for a reinvestor like an IT name,
the relative track has to carry more weight). The default for a typical
50/50 candidate stays exactly 50/50.

### 4. The originality work — what most student projects skip

This is the part that pushes the project past textbook territory:

#### Reverse DCF / Implied Expectations
Instead of asking *"what is this stock worth?"* it flips the question:
**"given the market price, what growth rate is being priced in?"**

```python
from src.reverse_dcf import implied_growth

result = implied_growth(
    market_price=2_400, d0=42.0, ke=0.124, g_terminal=0.045,
    historical_growth=0.18,
)
# → "market is pricing in g = 9.4% (vs historical 18%) —
#    implied growth is 8.6 pp BELOW historical;
#    market expectations look conservative."
```

This is the framing championed by Mauboussin (*Expectations Investing*)
and Damodaran. It turns every BUY recommendation into a falsifiable
claim: "the market is only pricing in X%, and I think the firm can do Y."

#### Bayesian growth shrinkage
A 3-year CAGR pulled off noisy data is *not* a 3-year truth. The
classic statistical fix is to Bayes-shrink the observed estimate
toward a prior:

$$
g_{posterior} = \frac{n \cdot g_{observed} + k \cdot g_{prior}}{n + k}
$$

With `n=3, k=5, g_observed=25%, g_prior=5%`, the posterior is 12.5% —
half data, half prior. With `n=20`, the prior fades away. Wired into
the DDM auto-selector by default. This single change kills the most
common DDM failure: extrapolating a flukey past growth rate into
perpetuity.

#### Backtesting
Every recommendation can be persisted to a CSV, then graded later
against forward returns at 6/12/24-month horizons:

```
==============================================================
  Backtest @ 12-month horizon
==============================================================
  Signals evaluated     : 47 / 50
  BUY   (n= 18) — mean=+19.40% | median=+15.20% | σ=12.10% | win-rate=78%
  HOLD  (n= 21) — mean= +8.30% | median= +6.10% | σ= 9.40% | win-rate=62%
  SELL  (n=  8) — mean= +1.10% | median= -2.80% | σ=14.70% | win-rate=37%
  Spread (BUY − SELL)   : +18.30%
  BUY hit-rate vs Nifty : 72%
```

The point isn't to fake a P&L curve — it's to answer the only question
that matters: *do BUY-rated stocks actually outperform SELL-rated ones?*
Survivorship bias is reported, not hidden.

#### Confidence labels
Every recommendation now ships with `HIGH / MEDIUM / LOW` confidence,
computed from:
- Monte Carlo dispersion (σ/μ < 15% → strong)
- Data quality score (0-100)
- Number of valid peers contributing
- Whether DDM was applicable

So `BUY (HIGH confidence)` looks meaningfully different from
`BUY (LOW confidence)` — and it should.

---

## The math, briefly

The five formulas that carry 80% of the engine:

| | Formula | What it answers |
|---|---|---|
| 1 | `Ke = Rf + β · ERP` | What return do investors demand? |
| 2 | `V = D₁ / (Ke − g)` | Gordon — value as PV of growing dividends |
| 3 | `β = Cov(Rₛ, Rₘ) / Var(Rₘ)` | How risky is this stock vs. Nifty? |
| 4 | `g = ROE · (1 − Payout)` | Sustainable growth from retention |
| 5 | `MoS = (FV − P) / FV` | Margin of safety; gates the recommendation |

Calibrated for India: risk-free = 10-yr G-Sec (7.10% as of Jan 2026),
ERP = 7.00% (Damodaran), tax = 25.17%, terminal growth ceiling = 6%
(below long-run nominal GDP). All overrideable from the CLI.

---

## Quick start

```bash
git clone https://github.com/Vaibhavkkm/integrated-equity-valuation-india.git
cd integrated-equity-valuation-india

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Command line

```bash
python run_valuation.py --ticker TCS.NS
python run_valuation.py --ticker HDFCBANK.NS --report          # generate PDF
python run_valuation.py --ticker ITC.NS --offline              # cache-only
python run_valuation.py --ticker RELIANCE.NS --rf 0.0710 --erp 0.075
```

### Interactive dashboard

```bash
streamlit run app.py
```

### As a library

```python
from src.integrated_valuation import value_stock

result = value_stock("RELIANCE.NS", log_for_backtest=True)
print(result.summary())
print(result.reverse_dcf.summary())
```

---

## Sample output

```
==============================================================================
  Tata Consultancy Services Limited  (TCS.NS)  |  Sector: Information Technology
==============================================================================
  Current Market Price       : ₹  3,520.40
  DDM Intrinsic Value        : ₹  3,940.12   (Three-Stage DDM)
  Relative Intrinsic Value   : ₹  4,180.55   (multi-multiple weighted)
  Blended Intrinsic Value    : ₹  4,051.20   (weights: DDM 55%, Rel 45%)
  Monte Carlo P10/P50/P90    : ₹  3,612.40 / ₹  4,038.10 / ₹  4,488.70
  Cost of Equity (CAPM)      :     12.85%
  Quality Composite (0-100)  :       82.4
  Data Quality (0-100)       :       91.0
  Margin of Safety           :     13.10%
  Recommendation             :  HOLD  🟡  (confidence: HIGH)
  Reverse DCF: market is pricing in g = 9.4% (vs historical 14.2%) —
    implied growth is 4.8 pp BELOW historical; expectations look conservative.
==============================================================================
```

---

## Project layout

```
integrated-equity-valuation-india/
├── README.md
├── requirements.txt
├── config.py                       # Indian market constants + sector map
├── app.py                          # Streamlit dashboard
├── run_valuation.py                # CLI entry point
├── src/
│   ├── data_fetcher.py             # NSE/BSE pull + multi-stage fallback
│   ├── data_validation.py          # Data quality scoring + guards
│   ├── exceptions.py               # Typed error hierarchy
│   ├── logging_setup.py            # Centralized logging
│   ├── ddm_models.py               # Gordon / 2-stage / 3-stage / H-Model
│   ├── cost_of_equity.py           # CAPM, beta regression, Hamada
│   ├── peer_identification.py      # K-Means + Mahalanobis peer selection
│   ├── relative_valuation.py       # Multiples + sector-aware weighting
│   ├── quality_score.py            # Piotroski F + Altman Z'' + dividend + momentum
│   ├── earnings_momentum.py        # Quarterly trend vs peer median (0-10)
│   ├── sensitivity.py              # Tornado + vectorised Monte Carlo
│   ├── reverse_dcf.py              # Implied-expectations solver
│   ├── backtest.py                 # Forward-return grading
│   ├── integrated_valuation.py     # The pipeline that wires it all
│   ├── visualizations.py           # Plotly charts
│   └── report_generator.py         # PDF research note
├── data/
│   └── nifty500_universe.csv       # Sector-mapped universe
├── tests/                          # 129 tests, runs in ~2 seconds
│   ├── conftest.py                 # Shared fixtures
│   ├── _factory.py                 # Synthetic StockBundle factory
│   ├── test_valuation.py
│   ├── test_edge_cases.py
│   ├── test_data_validation.py
│   ├── test_reverse_dcf.py
│   ├── test_backtest.py
│   ├── test_relative_valuation.py
│   ├── test_earnings_momentum.py
│   ├── test_cost_of_equity.py
│   ├── test_monte_carlo.py
│   ├── test_peer_identification.py
│   ├── test_quality_score.py
│   ├── test_report_generator.py
│   ├── test_app_smoke.py
│   └── test_exceptions.py
├── reports/                        # Generated PDFs and backtest logs
└── cache/                          # On-disk financials cache
```

---

## Testing

```bash
pytest tests/ -v
```

Currently **129 tests, all passing in ~2 seconds**. Coverage spans:

- DDM closed forms checked against textbook (Damodaran) values
- Bayesian shrinkage invariants (n→0 returns prior; n→∞ returns data)
- Reverse-DCF round-trip property (price the model's own output → recover input)
- Edge cases: loss-makers, non-payers, negative book value, thin history
- Synthetic peer sets through the relative-valuation pipeline
- Typed-exception hierarchy (every error reachable from the root class)
- Backtest binning, spread calculation, and survivorship-bias accounting

---

## Honest caveats

A few things this engine does *not* claim to do:

- **It is India-only.** Every assumption — risk-free rate, ERP, tax,
  terminal-growth ceiling — is hard-coded for the Indian market. Pointing
  it at a US ticker will technically run, but the answer will be wrong.
- **yfinance is the sole live source.** The fallback chain to stale
  cache softens this, but the engine inherits any silent schema drift
  Yahoo introduces.
- **Backtesting is recommendation-level, not portfolio-level.** Position
  sizing, transaction costs, and rebalancing rules are out of scope.
  The backtest answers "do BUY-rated stocks beat SELL-rated stocks?"
  not "what would my P&L curve look like?"
- **Survivorship bias is reported, not eliminated.** Tickers delisted
  between signal date and evaluation date drop out silently from
  yfinance; the backtester logs how many were lost. A
  survivorship-free database (CMIE, Capitaline) would be needed for a
  publication-grade study.

---

## Academic positioning

The methodology draws on:

- **Damodaran, A.** — *Investment Valuation*, 3rd ed.
- **Gordon, M. J.** (1959) — *Dividends, Earnings and Stock Prices*.
- **Fuller, R. J. & Hsia, C. C.** (1984) — *A Simplified Common Stock
  Valuation Model* (the H-Model).
- **Piotroski, J.** (2000) — *Value Investing: The Use of Historical
  Financial Statement Information*.
- **Altman, E. I.** (1968, 2000 update) — Z-Score / Z'' (emerging
  markets).
- **Mauboussin, M. & Rappaport, A.** (2001) — *Expectations Investing*
  (reverse DCF).

---

## Author

**Vaibhav Mangroliya** — built as the SEM-4 Student Project under the
supervision of **Mr. Senthil Nagarajan**.

Submitted in partial fulfilment of the SEM-4 Student Project
requirement.

---

*If you spot a methodological hole, an India-specific assumption that's
gone stale, or a bug — issues and PRs are welcome.*
