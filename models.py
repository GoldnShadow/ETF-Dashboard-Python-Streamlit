# models.py
# Ten financial models applied to every ETF.
# Each public run_* function returns a dict keyed by ticker:
#   {
#     "value":       float | None,   # primary numeric result for charts
#     "label":       str,            # plain-English 1-line interpretation
#     "enough_data": bool,           # False → show INSUFFICIENT message
#     **extra_keys                   # model-specific details for detail page
#   }

import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

try:
    import pandas_datareader as pdr
    _PDR_OK = True
except ImportError:
    _PDR_OK = False

from data_fetcher import ETFS, MIN_DAYS, load_dividends

INSUFFICIENT = "ETF does not have enough history to complete analysis."
_BENCH = "SPY"
_NON_ETF = {"SPY", "^IRX"}


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _align(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Inner-join two return series on common dates."""
    df = pd.concat([a, b], axis=1).dropna()
    df.columns = ["etf", "bench"]
    return df


def _annualise(daily: float) -> float:
    return daily * 252


def _ann_vol(daily_returns: pd.Series) -> float:
    return float(daily_returns.std() * np.sqrt(252))


def _sharpe(returns: pd.Series, rf_daily: float) -> float:
    excess = returns - rf_daily
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def _sortino(returns: pd.Series, rf_daily: float) -> float:
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float((excess.mean() / downside.std()) * np.sqrt(252))


def _beta_alpha(etf_excess: pd.Series, bench_excess: pd.Series):
    """OLS β and daily α."""
    slope, intercept, *_ = stats.linregress(bench_excess, etf_excess)
    return float(slope), float(intercept)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CAPITAL ASSET PRICING MODEL (CAPM)
# ─────────────────────────────────────────────────────────────────────────────

def run_capm(prices: pd.DataFrame, rf: float) -> dict:
    """
    Computes annualised alpha and beta vs SPY.
    Chart value: annualised alpha (higher = better = green).
    """
    results = {}
    spy_ret = prices[_BENCH].pct_change().dropna()
    rf_d = rf / 252

    for ticker in ETFS:
        etf_ret, ok = _etf_ret_check(prices, ticker)
        if not ok:
            results[ticker] = _no_data(ticker); continue

        df = _align(etf_ret, spy_ret)
        beta, alpha_d = _beta_alpha(df["etf"] - rf_d, df["bench"] - rf_d)
        alpha_a = _annualise(alpha_d)

        expected = rf + beta * (_annualise(spy_ret.mean()) - rf)
        actual   = _annualise(etf_ret.mean())
        diff     = actual - expected

        if diff > 0.02:
            label = (f"Following CAPM: {ticker} is outperforming its risk-adjusted "
                     f"expected return — suggesting it may be undervalued or has "
                     f"genuine positive alpha ({alpha_a:+.1%} annualised).")
        elif diff < -0.02:
            label = (f"Following CAPM: {ticker} is underperforming its risk-adjusted "
                     f"expected return — suggesting it may be overvalued or has "
                     f"persistent negative alpha ({alpha_a:+.1%} annualised).")
        else:
            label = (f"Following CAPM: {ticker} is fairly valued — returns align "
                     f"with what its market beta ({beta:.2f}) would predict.")

        results[ticker] = {
            "value": alpha_a, "label": label, "enough_data": True,
            "beta": round(beta, 3), "alpha": round(alpha_a, 4),
            "expected_return": round(expected, 4), "actual_return": round(actual, 4),
        }
    return results




# ─────────────────────────────────────────────────────────────────────────────
# 3. MODERN PORTFOLIO THEORY (MPT) — EFFICIENT FRONTIER
# ─────────────────────────────────────────────────────────────────────────────

def run_mpt(prices: pd.DataFrame, selected_tickers: list[str]) -> dict:
    """
    Monte Carlo efficient frontier for the user-selected ETF subset.
    Returns portfolio scatter data + per-ETF individual risk/return points.
    This function returns raw frontier data, not per-ticker labels —
    the chart function handles the MPT visualisation directly.
    """
    valid = []
    for t in selected_tickers:
        if t in prices.columns:
            s = prices[t].dropna()
            if len(s) >= MIN_DAYS:
                valid.append(t)

    if len(valid) < 2:
        return {"error": "Select at least 2 ETFs with sufficient history for MPT."}

    # Use overlapping history only
    ret_df = prices[valid].pct_change().dropna().dropna(axis=0)
    if len(ret_df) < MIN_DAYS:
        return {"error": "Insufficient overlapping history across selected ETFs."}

    mu   = ret_df.mean() * 252
    cov  = ret_df.cov()  * 252
    n    = len(valid)

    # Monte Carlo: 6000 random portfolios
    port_ret, port_vol, port_sharpe, port_weights = [], [], [], []
    rng = np.random.default_rng(42)
    for _ in range(6000):
        w = rng.dirichlet(np.ones(n))
        r = float(w @ mu.values)
        v = float(np.sqrt(w @ cov.values @ w))
        s = r / v if v > 0 else 0
        port_ret.append(r); port_vol.append(v)
        port_sharpe.append(s); port_weights.append(w)

    # Individual ETF points
    etf_points = {t: {"ret": float(mu[t]), "vol": float(np.sqrt(cov.loc[t, t]))}
                  for t in valid}

    # Max-Sharpe portfolio
    best_idx   = int(np.argmax(port_sharpe))
    best_w     = dict(zip(valid, port_weights[best_idx]))

    return {
        "tickers": valid,
        "port_ret": port_ret, "port_vol": port_vol, "port_sharpe": port_sharpe,
        "etf_points": etf_points,
        "max_sharpe_weights": best_w,
        "max_sharpe_ret": port_ret[best_idx],
        "max_sharpe_vol": port_vol[best_idx],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOTAL COST OF OWNERSHIP (TCO)
# ─────────────────────────────────────────────────────────────────────────────

def run_tco(info_dict: dict[str, dict]) -> dict:
    """
    Expense ratio + estimated implicit trading cost (0.5 × bid-ask spread proxy).
    Chart value: total annual cost bps (lower = better = green).
    """
    results = {}
    for ticker in ETFS:
        info = info_dict.get(ticker, {})

        # yFinance may expose the expense ratio under different keys
        exp_ratio = (
            info.get("annualReportExpenseRatio")
            or info.get("totalExpenseRatio")
            or info.get("expenseRatio")
        )

        if exp_ratio is None:
            results[ticker] = {
                "value": None, "enough_data": False,
                "label": f"TCO: Expense ratio data unavailable for {ticker} from yFinance.",
            }
            continue

        exp_bps = exp_ratio * 10000  # convert to basis points

        # Estimate implicit cost from bid-ask spread proxy.
        # Yahoo provides spread data for some instruments; otherwise approximate
        # using average daily volume: higher volume → tighter spread.
        avg_vol   = info.get("averageVolume", 0) or 0
        avg_price = info.get("regularMarketPrice") or info.get("previousClose") or 0
        dollar_vol = avg_vol * avg_price

        # Very rough spread estimate: 1 bp for highly liquid (>$100M ADV),
        # up to 10 bps for illiquid (<$1M ADV).
        if dollar_vol > 100_000_000:
            implicit_bps = 1.0
        elif dollar_vol > 10_000_000:
            implicit_bps = 3.0
        elif dollar_vol > 1_000_000:
            implicit_bps = 6.0
        else:
            implicit_bps = 10.0

        total_bps = exp_bps + implicit_bps

        if total_bps < 20:
            label = (f"TCO: {ticker} is a low-cost ETF with an estimated total "
                     f"annual ownership cost of {total_bps:.1f} bps "
                     f"(expense ratio: {exp_bps:.1f} bps).")
        elif total_bps < 60:
            label = (f"TCO: {ticker} carries a moderate cost of {total_bps:.1f} bps/yr "
                     f"— reasonable but worth comparing to lower-cost alternatives.")
        else:
            label = (f"TCO: {ticker} has a relatively high estimated annual cost "
                     f"of {total_bps:.1f} bps — factor this into long-term return projections.")

        results[ticker] = {
            "value": total_bps, "label": label, "enough_data": True,
            "expense_ratio_bps": round(exp_bps, 2),
            "implicit_cost_bps": round(implicit_bps, 2),
            "total_cost_bps":    round(total_bps, 2),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRACKING ERROR & TRACKING DIFFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_tracking(prices: pd.DataFrame) -> dict:
    """
    Tracking Error (TE): annualised std of (ETF_ret - SPY_ret).
    Tracking Difference (TD): cumulative ETF return minus SPY return.
    Note: SPY is used as the market proxy, not each ETF's own benchmark index.
    Chart value: tracking error (lower = better = green for passive ETFs,
                 but high TE is expected for active/thematic ETFs).
    """
    results = {}
    spy_ret = prices[_BENCH].pct_change().dropna()

    for ticker in ETFS:
        etf_ret, ok = _etf_ret_check(prices, ticker)
        if not ok:
            results[ticker] = _no_data(ticker); continue

        df = _align(etf_ret, spy_ret)
        diff = df["etf"] - df["bench"]
        te   = float(diff.std() * np.sqrt(252))

        # Tracking Difference: cumulative return gap over the full history
        cum_etf   = (1 + df["etf"]).prod() - 1
        cum_bench = (1 + df["bench"]).prod() - 1
        td        = float(cum_etf - cum_bench)

        if te < 0.05:
            label = (f"Tracking: {ticker} closely follows the broad market "
                     f"(tracking error {te:.1%} vs SPY). Behaves like a passive index fund.")
        elif te < 0.15:
            label = (f"Tracking: {ticker} shows moderate divergence from the market "
                     f"(TE {te:.1%}). Expected for thematic or factor ETFs.")
        else:
            label = (f"Tracking: {ticker} diverges significantly from SPY "
                     f"(TE {te:.1%}) — indicates an active, sector-specific, "
                     f"or alternative strategy.")

        if td > 0:
            label += f" Cumulative outperformance vs SPY: {td:+.1%}."
        else:
            label += f" Cumulative underperformance vs SPY: {td:+.1%}."

        results[ticker] = {
            "value": te, "label": label, "enough_data": True,
            "tracking_error": round(te, 4),
            "tracking_difference": round(td, 4),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. RISK-ADJUSTED PERFORMANCE (SHARPE, SORTINO, TREYNOR)
# ─────────────────────────────────────────────────────────────────────────────

def run_risk_adjusted(prices: pd.DataFrame, rf: float) -> dict:
    """
    Computes Sharpe, Sortino, and Treynor ratios.
    Chart value: Sharpe ratio (higher = better = green).
    """
    results = {}
    spy_ret = prices[_BENCH].pct_change().dropna()
    rf_d    = rf / 252

    for ticker in ETFS:
        etf_ret, ok = _etf_ret_check(prices, ticker)
        if not ok:
            results[ticker] = _no_data(ticker); continue

        sharpe  = _sharpe(etf_ret, rf_d)
        sortino = _sortino(etf_ret, rf_d)

        df   = _align(etf_ret, spy_ret)
        beta, _ = _beta_alpha(df["etf"] - rf_d, df["bench"] - rf_d)
        ann_exc = _annualise(etf_ret.mean()) - rf
        treynor = ann_exc / beta if beta != 0 else 0.0

        if sharpe > 1.0:
            qual = "excellent risk-adjusted returns"
        elif sharpe > 0.5:
            qual = "solid risk-adjusted returns"
        elif sharpe > 0.0:
            qual = "modest risk-adjusted returns"
        else:
            qual = "negative risk-adjusted returns"

        label = (f"Risk-Adjusted: {ticker} delivers {qual} "
                 f"(Sharpe {sharpe:.2f} | Sortino {sortino:.2f} | Treynor {treynor:.2f}).")

        results[ticker] = {
            "value": sharpe, "label": label, "enough_data": True,
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "treynor": round(treynor, 3),
            "beta": round(beta, 3),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. LIQUIDITY & MARKET IMPACT
# ─────────────────────────────────────────────────────────────────────────────

def run_liquidity(info_dict: dict[str, dict], prices: pd.DataFrame) -> dict:
    """
    Average daily dollar volume as the primary liquidity metric.
    Market impact estimate: approximate cost to trade a $1M position.
    Chart value: log10(average daily dollar volume) (higher = better = green).
    """
    results = {}
    for ticker in ETFS:
        info  = info_dict.get(ticker, {})
        price = info.get("regularMarketPrice") or info.get("previousClose")
        
        if not price and ticker in prices.columns:
            series = prices[ticker].dropna()
            if not series.empty:
                price = series.iloc[-1]
        vol   = info.get("averageVolume") or info.get("averageDailyVolume10Day")

        if not price or not vol:
            results[ticker] = {
                "value": None, "enough_data": False,
                "label": f"Liquidity: Volume or price data unavailable for {ticker}.",
            }
            continue

        dollar_vol = float(price) * float(vol)
        log_dv     = np.log10(max(dollar_vol, 1))

        # Amihud-style impact: estimated slippage for a $1M trade
        # Impact ≈ $1M / daily_dollar_vol  (as a % of price)
        impact_pct = min(1_000_000 / dollar_vol * 100, 100) if dollar_vol > 0 else 100

        if dollar_vol > 500_000_000:
            tier = "extremely liquid — minimal market impact"
        elif dollar_vol > 50_000_000:
            tier = "very liquid — low market impact"
        elif dollar_vol > 5_000_000:
            tier = "moderately liquid"
        else:
            tier = "relatively illiquid — larger trades may move the price"

        label = (f"Liquidity: {ticker} is {tier}. "
                 f"Avg daily dollar volume: ${dollar_vol:,.0f}. "
                 f"Estimated market impact on a $1M trade: ~{impact_pct:.2f}%.")

        results[ticker] = {
            "value": log_dv, "label": label, "enough_data": True,
            "avg_dollar_volume": round(dollar_vol, 0),
            "market_impact_pct": round(impact_pct, 3),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 8. PREMIUM / DISCOUNT TO NAV
# ─────────────────────────────────────────────────────────────────────────────

def run_nav_premium(info_dict: dict[str, dict]) -> dict:
    """
    Premium or discount to Net Asset Value: (Market Price - NAV) / NAV × 100.
    Chart value: premium % (0 = neutral, positive = premium, negative = discount).
    Colour coding flips here: near 0 = green, large premium or discount = yellow/red.
    """
    results = {}
    for ticker in ETFS:
        info     = info_dict.get(ticker, {})
        nav      = info.get("navPrice") or info.get("nav")
        mkt_price = info.get("regularMarketPrice") or info.get("previousClose")

        if not nav or not mkt_price:
            results[ticker] = {
                "value": None, "enough_data": False,
                "label": (f"NAV: Premium/Discount data unavailable for {ticker} — "
                          "yFinance does not expose NAV for all ETFs."),
            }
            continue

        premium_pct = ((float(mkt_price) - float(nav)) / float(nav)) * 100

        if abs(premium_pct) < 0.10:
            label = (f"NAV: {ticker} is trading essentially at par with its NAV "
                     f"({premium_pct:+.3f}%) — no meaningful premium or discount.")
        elif premium_pct > 0:
            label = (f"NAV: {ticker} is trading at a {premium_pct:.2f}% premium to NAV "
                     f"— you are paying above the value of the underlying holdings.")
        else:
            label = (f"NAV: {ticker} is trading at a {abs(premium_pct):.2f}% discount to NAV "
                     f"— potentially a bargain relative to underlying asset value.")

        results[ticker] = {
            "value": premium_pct, "label": label, "enough_data": True,
            "nav": float(nav),
            "market_price": float(mkt_price),
            "premium_pct": round(premium_pct, 4),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 9. LOOK-THROUGH / FUNDAMENTAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_look_through(info_dict: dict[str, dict]) -> dict:
    """
    Surfaces yFinance portfolio-level fundamentals: P/E, P/B, dividend yield.
    Chart value: trailing P/E (lower = cheaper = green from a value perspective).
    For ETFs with no earnings (e.g. commodity/crypto ETFs), notes N/A.
    """
    results = {}
    for ticker in ETFS:
        info = info_dict.get(ticker, {})
        pe   = info.get("trailingPE")   or info.get("forwardPE")
        pb   = info.get("priceToBook")
        dy   = info.get("dividendYield") or info.get("yield")

        if pe is None and pb is None:
            results[ticker] = {
                "value": None, "enough_data": False,
                "label": (f"Fundamentals: {ticker} has no P/E or P/B data available — "
                          "typical for commodity, crypto, or very new ETFs."),
            }
            continue

        pe_str = f"P/E {pe:.1f}" if pe else "P/E N/A"
        pb_str = f"P/B {pb:.2f}" if pb else "P/B N/A"
        dy_str = f"Yield {dy:.1%}" if dy else "Yield N/A"

        if pe and pe < 15:
            val_label = "appears value-priced (low P/E)"
        elif pe and pe > 35:
            val_label = "appears growth-priced (elevated P/E)"
        elif pe:
            val_label = "trades at a market-average valuation"
        else:
            val_label = "valuation assessed on P/B basis"

        label = (f"Fundamentals: {ticker} {val_label}. "
                 f"{pe_str} | {pb_str} | {dy_str}.")

        results[ticker] = {
            "value": pe or pb,  # use P/E, fall back to P/B for chart
            "label": label, "enough_data": True,
            "pe": pe, "pb": pb, "dividend_yield": dy,
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 10. DISCOUNTED DIVIDEND MODEL (DDM)
# ─────────────────────────────────────────────────────────────────────────────

def run_ddm(prices: pd.DataFrame, capm_results: dict, rf: float) -> dict:
    """
    Gordon Growth DDM: Fair Value = D₁ / (r - g)
      D₁   = projected next annual dividend (trailing 12m × (1+g))
      r    = CAPM discount rate for the ETF
      g    = trailing dividend CAGR (capped at r - 1% to ensure convergence)
    Chart value: (Fair Value - Market Price) / Market Price
                 positive = undervalued (green), negative = overvalued (red).
    """
    results = {}

    for ticker in ETFS:
        divs = load_dividends(ticker)
        price_series = prices[ticker].dropna() if ticker in prices.columns else pd.Series()

        if price_series.empty:
            results[ticker] = _no_data(ticker); continue

        current_price = float(price_series.iloc[-1])

        # Minimum: at least 4 quarterly dividend payments
        if len(divs) < 4:
            results[ticker] = {
                "value": None, "enough_data": False,
                "label": (f"DDM: {ticker} does not have sufficient dividend history "
                          "to apply the Discounted Dividend Model."),
            }
            continue

        # Trailing 12-month dividend using modern slicing
        one_year_ago = divs.index.max() - pd.Timedelta(days=365)
        trailing_divs = divs[divs.index >= one_year_ago]
        
        if trailing_divs.empty:
            results[ticker] = {
                "value": None, "enough_data": False,
                "label": f"DDM: {ticker} has no dividends in the last 365 days."
            }
            continue
            
        trailing_12m = float(trailing_divs.sum())

        if trailing_12m <= 0:
            results[ticker] = {
                "value": None, "enough_data": False,
                "label": (f"DDM: {ticker} paid no dividends in the past 12 months — "
                          "DDM is not applicable."),
            }
            continue

        # Dividend growth rate: CAGR from first to last annual dividend bucket
        annual_divs = divs.resample("YE").sum()
        annual_divs = annual_divs[annual_divs > 0]

        if len(annual_divs) >= 3:
            years = (annual_divs.index[-1] - annual_divs.index[0]).days / 365.25
            g_raw = (float(annual_divs.iloc[-1]) / float(annual_divs.iloc[0])) ** (1 / max(years, 1)) - 1
            g = float(np.clip(g_raw, -0.10, 0.20))  # bound growth rate
        else:
            g = 0.03  # default 3 % if insufficient history

        # Discount rate from CAPM result; fall back to rf + 5%
        capm = capm_results.get(ticker, {})
        r = capm.get("expected_return", rf + 0.05)

        # Ensure r > g for model convergence
        if r <= g:
            g = r - 0.01

        D1 = trailing_12m * (1 + g)
        fair_value = D1 / (r - g)
        upside = (fair_value - current_price) / current_price

        if upside > 0.10:
            label = (f"DDM: {ticker} appears undervalued — fair value estimate "
                     f"${fair_value:.2f} vs current price ${current_price:.2f} "
                     f"({upside:+.1%} implied upside). "
                     f"Based on trailing yield and {g:.1%} dividend growth.")
        elif upside < -0.10:
            label = (f"DDM: {ticker} appears overvalued — fair value estimate "
                     f"${fair_value:.2f} vs current price ${current_price:.2f} "
                     f"({upside:+.1%} implied downside). "
                     f"Based on trailing yield and {g:.1%} dividend growth.")
        else:
            label = (f"DDM: {ticker} appears fairly valued — fair value estimate "
                     f"${fair_value:.2f} vs current price ${current_price:.2f} "
                     f"({upside:+.1%}). Based on {g:.1%} projected dividend growth.")

        results[ticker] = {
            "value": upside, "label": label, "enough_data": True,
            "fair_value": round(fair_value, 2),
            "current_price": round(current_price, 2),
            "d1": round(D1, 4),
            "growth_rate": round(g, 4),
            "discount_rate": round(r, 4),
            "implied_upside": round(upside, 4),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_all_models(prices: pd.DataFrame, info_dict: dict, rf: float) -> dict:
    """Run all 10 models and return a nested dict: results[model_name][ticker]."""
    capm = run_capm(prices, rf)
    return {
        "capm":         capm, 
        "tracking":     run_tracking(prices),
        "risk_adjusted": run_risk_adjusted(prices, rf),
        "liquidity":    run_liquidity(info_dict, prices),
        "nav_premium":  run_nav_premium(info_dict),
        "look_through": run_look_through(info_dict),
        "ddm":          run_ddm(prices, capm, rf),
        # MPT is handled separately (requires interactive ETF selection)
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _etf_ret_check(prices: pd.DataFrame, ticker: str) -> tuple[pd.Series, bool]:
    if ticker not in prices.columns:
        return pd.Series(dtype=float), False
    s = prices[ticker].dropna()
    r = s.pct_change().dropna()
    return r, len(r) >= MIN_DAYS


def _no_data(ticker: str) -> dict:
    return {"value": None, "enough_data": False, "label": INSUFFICIENT}
