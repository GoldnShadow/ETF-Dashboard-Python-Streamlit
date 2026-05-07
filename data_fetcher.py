# data_fetcher.py
# Handles all yFinance calls. Everything is cached with a 1-hour TTL so
# the dashboard doesn't hammer Yahoo on every interaction.

import yfinance as yf
import pandas as pd
import streamlit as st

ETFS = [
    "IDVO", "DIVO", "OVL",  "OVF",  "OVS",  "GPIQ", "GPIX", "JEPQ",
    "GRID", "MLPI", "MCHI", "CEPI", "CHPY", "YMAX", "FEPI", "AIPI",
    "QQQI", "SYPI", "USOY", "SCHD", "EWJ",  "INDA", "VIGI", "VEA",
    "VOO",  "IAUI", "BTCI", "DTCR", "TSPY", "SMH",  "VUG",  "XQQI",
    "VGT",  "VDE",
]

# Minimum trading days required for quantitative models (≈ 1 year).
MIN_DAYS = 252


@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data() -> pd.DataFrame:
    """
    Download adjusted close prices for every ETF + SPY + ^IRX (T-bill proxy)
    going back to the maximum available history on Yahoo Finance.
    Returns a DataFrame indexed by date with one column per ticker.
    """
    tickers = ETFS + ["SPY", "^IRX"]
    raw = yf.download(tickers, period="max", auto_adjust=True, progress=False)

    # yfinance returns a MultiIndex when multiple tickers are requested.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw  # single-ticker fallback (shouldn't happen here)

    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def load_etf_info(ticker: str) -> dict:
    """Fetch the yFinance .info dict for a single ticker. Returns {} on failure."""
    try:
        info = yf.Ticker(ticker).info
        return info if info else {}
    except Exception:
        return {}


@st.cache_data(ttl=7200, show_spinner=False)
def load_dividends(ticker: str) -> pd.Series:
    """Fetch dividend payment history for a ticker. Returns empty Series on failure."""
    try:
        divs = yf.Ticker(ticker).dividends
        return divs if divs is not None and len(divs) > 0 else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def get_risk_free_rate(prices: pd.DataFrame) -> float:
    """
    Derive an annualised risk-free rate from ^IRX (13-week T-bill).
    Falls back to 5.25 % if the series is unavailable.
    """
    if "^IRX" in prices.columns:
        irx = prices["^IRX"].dropna()
        if len(irx) > 0:
            return float(irx.iloc[-1]) / 100  # ^IRX is already annualised %
    return 0.0525


def get_ticker_returns(prices: pd.DataFrame, ticker: str) -> tuple[pd.Series, bool]:
    """
    Return (daily_returns, has_enough_data).
    has_enough_data is True when ≥ MIN_DAYS of price history exist.
    """
    if ticker not in prices.columns:
        return pd.Series(dtype=float), False
    series = prices[ticker].dropna()
    returns = series.pct_change().dropna()
    return returns, len(returns) >= MIN_DAYS


def get_inception_date(prices: pd.DataFrame, ticker: str) -> str:
    """Return the first date of available price data for a ticker."""
    if ticker not in prices.columns:
        return "N/A"
    series = prices[ticker].dropna()
    if series.empty:
        return "N/A"
    return series.index[0].strftime("%Y-%m-%d")
