# app.py
# Run with:  streamlit run app.py

import streamlit as st  
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_fetcher import (
    ETFS, load_price_data, load_etf_info,
    get_risk_free_rate, get_inception_date,
)
from models import run_all_models, run_mpt
from charts import (
    chart_capm, chart_mpt,
    chart_tracking, chart_risk_adjusted, chart_liquidity,
    chart_nav_premium, chart_look_through, chart_ddm,
    chart_price_history,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ETF Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>

    /* Make table text smaller and more compact */
    [data-testid="stTable"] td, [data-testid="stTable"] th {
        font-size: 10px !important;
        padding: 4px 8px !important;
        line-height: 1.2 !important;
    }

    /* Optional: Make the ticker column bold but keep it small */
    [data-testid="stTable"] td:first-child {
        font-weight: bold;
    }
    
    /* Tighten sidebar */
    section[data-testid="stSidebar"] { min-width: 270px; max-width: 270px; }
    /* Metric card style */
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px;
    }
    /* Section headers */
    h3 { border-bottom: 2px solid #e9ecef; padding-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 ETF Dashboard")
    st.caption("Data sourced from Yahoo Finance via yFinance.")
    st.divider()

    view = st.radio(
        "View",
        options=["🗺 Overview (All Models)", "🔍 ETF Detail"],
        index=0,
    )

    if view == "🔍 ETF Detail":
        selected_etf = st.selectbox("Select ETF", ETFS)
    else:
        selected_etf = None

    st.divider()
    st.caption(
        "**Benchmark:** SPY  \n"
        "**History:** Since inception per ETF  \n"
        "**Min data:** 252 trading days  \n"
        "Models refresh every hour."
    )

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOAD  (cached — runs once per session then reuses)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_info() -> dict:
    return {t: load_etf_info(t) for t in ETFS}


with st.spinner("📡 Fetching price history from Yahoo Finance…"):
    prices   = load_price_data()

with st.spinner("📋 Loading ETF metadata…"):
    info_all = load_all_info()

rf = get_risk_free_rate(prices)

with st.spinner("⚙️ Running financial models — this may take a minute on first load…"):
    all_results = run_all_models(prices, info_all, rf)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_META = [
    ("capm",         "1. CAPM",
     "Measures whether each ETF is earning more or less than its market beta predicts."),
    ("tracking",     "5. Tracking Error & Difference",
     "Divergence from SPY. Low TE = market-like behaviour; high TE = active/thematic."),
    ("risk_adjusted","6. Risk-Adjusted Performance",
     "Sharpe, Sortino, and Treynor ratios — reward per unit of risk."),
    ("liquidity",    "7. Liquidity & Market Impact",
     "Avg daily dollar volume and estimated cost to execute a $1M trade."),
    ("nav_premium",  "8. Premium / Discount to NAV",
     "How far the market price deviates from the fund's underlying asset value."),
    ("look_through", "9. Look-Through Fundamentals",
     "Portfolio-level P/E, P/B, and dividend yield as reported by yFinance."),
    ("ddm",          "10. Discounted Dividend Model",
     "Gordon Growth DDM fair value vs current market price. For dividend-paying ETFs only."),
]

CHART_FNS = {
    "capm":          chart_capm,
    "tracking":      chart_tracking,
    "risk_adjusted": chart_risk_adjusted,
    "liquidity":     chart_liquidity,
    "nav_premium":   chart_nav_premium,
    "look_through":  chart_look_through,
    "ddm":           chart_ddm,
}


def _fmt(val, fmt=".3f"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:{fmt}}"


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW PAGE
# ─────────────────────────────────────────────────────────────────────────────

def show_overview():
    st.title("📊 ETF Analysis Dashboard — All Models")
    st.markdown(
        f"Analysing **{len(ETFS)} ETFs** against SPY as benchmark. "
        f"Risk-free rate: **{rf:.2%}** (from ^IRX).  "
        f"Minimum data threshold: **252 trading days**.  "
        f"🟢 Green = stronger outlook &nbsp;&nbsp; ⬜ Grey = neutral &nbsp;&nbsp; 🔴 Red = weaker outlook"
    )

    # ── MODEL 3: MPT — Interactive (needs ETF selection first) ──────────────
    st.divider()
    st.subheader("3. Modern Portfolio Theory (MPT) — Efficient Frontier")
    st.caption("Select ETFs to include in the frontier simulation.")
    mpt_selected = st.multiselect(
        "ETFs for MPT", ETFS,
        default=[t for t in ["VOO", "SMH", "VGT", "SCHD", "VEA", "VDE", "INDA"]
                 if t in ETFS],
        key="mpt_multiselect",
    )
    mpt_data = run_mpt(prices, mpt_selected)
    st.plotly_chart(chart_mpt(mpt_data), use_container_width=True)

    if "max_sharpe_weights" in mpt_data:
        st.caption("**Max-Sharpe Portfolio Weights:**  " +
                   "  |  ".join(f"{t}: {w:.1%}"
                                for t, w in mpt_data["max_sharpe_weights"].items()
                                if w > 0.01))

    # ── MODELS 1, 2, 4-10 ───────────────────────────────────────────────────
    for key, title, desc in MODEL_META:
        st.divider()
        st.subheader(title)
        st.caption(desc)

        model_res = all_results[key]
        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            fig = CHART_FNS[key](model_res)
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            # Plain-English interpretation table
            rows = []
            for t in ETFS:
                r = model_res.get(t, {})
                rows.append({"ETF": t, "Interpretation": r.get("label", "N/A")})
            df = pd.DataFrame(rows)
            st.table(df)

# ─────────────────────────────────────────────────────────────────────────────
# ETF DETAIL PAGE
# ─────────────────────────────────────────────────────────────────────────────

def show_etf_detail(ticker: str):
    info     = info_all.get(ticker, {})
    inc_date = get_inception_date(prices, ticker)

    # ── Header ──────────────────────────────────────────────────────────────
    long_name = info.get("longName") or info.get("shortName") or ticker
    st.title(f"🔍 {ticker} — {long_name}")
    desc = info.get("longBusinessSummary") or info.get("description")
    if desc:
        with st.expander("Fund Description"):
            st.write(desc)

    # ── Key Metrics Cards ────────────────────────────────────────────────────
    st.subheader("Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)

    pe = info.get("trailingPE") or info.get("forwardPE")
    dy = info.get("dividendYield") or info.get("yield")
    nav= info.get("navPrice") or info.get("nav")
    mp = info.get("regularMarketPrice") or info.get("previousClose")
    sharpe = all_results["risk_adjusted"].get(ticker, {}).get("sharpe")
    exp = (info.get("annualReportExpenseRatio")
           or info.get("totalExpenseRatio")
           or info.get("expenseRatio"))

    c1.metric("Price",          f"${mp:.2f}"         if mp      else "N/A")
    c2.metric("P/E Ratio",      f"{pe:.1f}"          if pe      else "N/A")
    c3.metric("Dividend Yield", f"{dy:.2%}"          if dy      else "N/A")
    c4.metric("NAV",            f"${nav:.2f}"        if nav     else "N/A")
    c5.metric("Expense Ratio",  f"{exp:.2%}"         if exp     else "N/A")

    c1b, c2b, c3b, c4b, c5b = st.columns(5)
    c1b.metric("Sharpe Ratio",  f"{sharpe:.2f}"      if sharpe  else "N/A")
    c2b.metric("Inception Date", inc_date)
    beta = all_results["capm"].get(ticker, {}).get("beta")
    c3b.metric("Beta (vs SPY)", f"{beta:.2f}"        if beta    else "N/A")
    alpha = all_results["capm"].get(ticker, {}).get("alpha")
    c4b.metric("CAPM Alpha",    f"{alpha:.2%}"       if alpha is not None else "N/A")
    te = all_results["tracking"].get(ticker, {}).get("tracking_error")
    c5b.metric("Tracking Error",f"{te:.2%}"          if te      else "N/A")

    # ── Price History Chart ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Price History vs SPY (Indexed to Inception)")
    st.plotly_chart(chart_price_history(prices, ticker), use_container_width=True)

    # ── All 10 Model Results ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Model Results")
    st.caption("All 10 financial models applied to this ETF.")

    # MPT result for this ticker
    mpt_note = ""
    if ticker in prices.columns:
        p_series = prices[ticker].dropna()
        if len(p_series) >= 252:
            ret  = float(p_series.pct_change().dropna().mean() * 252)
            vol  = float(p_series.pct_change().dropna().std() * (252 ** 0.5))
            mpt_note = f"Annualised Return: {ret:.2%} | Annualised Volatility: {vol:.2%}"

    mpt_row = {
        "Model": "3. MPT",
        "Result": mpt_note or "Requires multi-ETF selection on the Overview page.",
    }

    rows = []
    for key, title, _ in MODEL_META:
        r = all_results[key].get(ticker, {})
        rows.append({"Model": title, "Result": r.get("label", "N/A")})

    rows.insert(2, mpt_row)  # insert MPT at position 3
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        column_config={"Result": st.column_config.TextColumn(width="large")},
        hide_index=True,
        use_container_width=True,
    )

    # ── Detailed numeric breakdown ───────────────────────────────────────────
    st.divider()
    st.subheader("Numeric Details")
    with st.expander("Expand for all model numeric outputs"):
        detail_rows = []
        for key, title, _ in MODEL_META:
            r = all_results[key].get(ticker, {})
            if not r.get("enough_data"):
                continue
            for k, v in r.items():
                if k in ("label", "enough_data", "value"):
                    continue
                detail_rows.append({
                    "Model": title,
                    "Metric": k,
                    "Value": str(round(v, 6)) if isinstance(v, float) else str(v),
                })
        if detail_rows:
            st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)
        else:
            st.info("No numeric detail available — insufficient data for this ETF.")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────

if view == "🗺 Overview (All Models)":
    show_overview()
else:
    show_etf_detail(selected_etf)
