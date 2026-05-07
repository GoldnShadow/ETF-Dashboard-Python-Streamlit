# ETF Analysis Dashboard

A Streamlit dashboard applying 10 financial models to 34 ETFs id my choice using live data from Yahoo Finance.

---

## Setup

```bash
# 1. Clone / copy this folder
cd etf_dashboard

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Mac/Linux
.venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py

# If there is a package issue and Streamlit is not installed to your PATH, run with this
python -m streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## Models

| # | Model | Chart Value | Green = |
|---|-------|-------------|---------|
| 1 | CAPM | Annualised Alpha | High positive alpha |
| 2 | Fama-French 3-Factor | FF Alpha | High positive alpha |
| 3 | MPT Efficient Frontier | Risk/Return scatter | Max-Sharpe portfolio |
| 4 | Total Cost of Ownership | Basis points/yr | Low cost |
| 5 | Tracking Error | Ann. TE vs SPY | Low divergence |
| 6 | Risk-Adjusted (Sharpe/Sortino/Treynor) | Sharpe ratio | High ratio |
| 7 | Liquidity & Market Impact | Log Dollar Volume | High volume |
| 8 | Premium/Discount to NAV | % deviation from NAV | Near zero |
| 9 | Look-Through Fundamentals | Trailing P/E | Low P/E (value) |
| 10 | Discounted Dividend Model | Implied upside % | Undervalued |

---

## Data Notes

- **History:** Since inception for each ETF (max available from Yahoo Finance).
- **Benchmark:** SPY for all models.
- **Risk-free rate:** 13-week T-bill (^IRX), falling back to 5.25%.
- **Cache TTL:** 1 hour — data refreshes automatically on a new session.
- **Fama-French factors:** Fetched from Kenneth French's public data library via `pandas-datareader`. Requires internet access. If unavailable, model 2 shows a graceful fallback message.
- **NAV data:** Only available for ETFs where Yahoo Finance exposes `navPrice`. Many ETFs will show N/A.

### Insufficient Data
ETFs with fewer than 252 trading days of history display:
> *"ETF does not have enough history to complete analysis."*

This is expected for newer ETFs such as CEPI, CHPY, SYPI, DTCR, TSPY, XQQI, IAUI.

---

## Pages

- **Overview:** All 10 model charts side-by-side with plain-English interpretations.
- **ETF Detail:** Select any ETF from the sidebar to see a dedicated page with key metrics, price history vs SPY, and all model results for that ETF.

---

## File Structure

```
etf_dashboard/
├── app.py           Main Streamlit application
├── data_fetcher.py  yFinance data layer (cached)
├── models.py        All 10 financial models
├── charts.py        Plotly figure builders
└── requirements.txt Python dependencies
```
