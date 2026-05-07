# charts.py
# Plotly figure factories for every model.
# All figures use the green → grey → red colour scale described in the spec:
#   high percentile = green (#2ecc71)
#   middle          = grey  (#aaaaaa)
#   low percentile  = red   (#e74c3c)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _metric_colors(values: list, higher_is_better: bool = True) -> list[str]:
    """
    Map a list of floats to hex colours using a green-grey-red scale.
    None / NaN values get a transparent grey.
    """
    arr   = np.array([float(v) if v is not None else np.nan for v in values])
    valid = arr[~np.isnan(arr)]

    if len(valid) < 2:
        return ["rgba(150,150,150,0.7)"] * len(arr)

    vmin, vmax = valid.min(), valid.max()
    span = vmax - vmin if vmax != vmin else 1.0

    colors = []
    for v in arr:
        if np.isnan(v):
            colors.append("rgba(150,150,150,0.3)")
            continue
        norm = (v - vmin) / span            # 0 → worst, 1 → best (raw)
        if not higher_is_better:
            norm = 1.0 - norm               # flip so low values are green

        # Interpolate: 0 = red (#e74c3c), 0.5 = grey (#aaaaaa), 1 = green (#2ecc71)
        if norm >= 0.5:
            t = (norm - 0.5) / 0.5
            r = int(170 + (46  - 170) * t)
            g = int(170 + (204 - 170) * t)
            b = int(170 + (113 - 170) * t)
        else:
            t = norm / 0.5
            r = int(231 + (170 - 231) * t)
            g = int(76  + (170 - 76)  * t)
            b = int(60  + (170 - 60)  * t)
        colors.append(f"rgb({r},{g},{b})")

    return colors


def _nav_colors(premiums: list) -> list[str]:
    """
    Special colour for NAV chart: near-zero is green, large deviation is red.
    """
    arr = np.array([float(v) if v is not None else np.nan for v in premiums])
    colors = []
    for v in arr:
        if np.isnan(v):
            colors.append("rgba(150,150,150,0.3)"); continue
        d = abs(v)
        if d < 0.1:
            colors.append("rgb(46,204,113)")   # green — at NAV
        elif d < 0.5:
            colors.append("rgb(170,170,170)")  # grey — small deviation
        else:
            colors.append("rgb(231,76,60)")    # red — large premium/discount
    return colors


_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Arial, sans-serif", size=12),
    margin=dict(l=60, r=20, t=50, b=80),
    legend=dict(orientation="h", y=-0.25),
)


def _scatter(tickers, x_vals, y_vals, colors, x_label, y_label, title,
             hover_extra: dict | None = None) -> go.Figure:
    """Generic scatter/bar with green-grey-red markers."""
    hover = [f"<b>{t}</b><br>{x_label}: {x:.4f}<br>{y_label}: {y:.4f}"
             for t, x, y in zip(tickers, x_vals, y_vals)]
    if hover_extra:
        for i, t in enumerate(tickers):
            for k, v in hover_extra.items():
                hover[i] += f"<br>{k}: {v[i]}"

    fig = go.Figure(go.Scatter(
        x=list(x_vals), y=list(y_vals),
        mode="markers+text",
        text=tickers,
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(color=colors, size=12, line=dict(color="white", width=1)),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
    ))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, **_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. CAPM — Beta vs Alpha scatter
# ─────────────────────────────────────────────────────────────────────────────

def chart_capm(results: dict) -> go.Figure:
    tickers, betas, alphas, colors_raw = [], [], [], []
    for t, r in results.items():
        if r["enough_data"] and r["value"] is not None:
            tickers.append(t)
            betas.append(r.get("beta", 0))
            alphas.append(r.get("alpha", 0))
            colors_raw.append(r.get("alpha", 0))

    colors = _metric_colors(colors_raw, higher_is_better=True)
    fig = _scatter(tickers, betas, alphas, colors,
                   "Beta (vs SPY)", "Annualised Alpha", "CAPM — Beta vs Alpha")
    # Reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    fig.add_vline(x=1, line_dash="dash", line_color="grey", opacity=0.5)
    return fig



# ─────────────────────────────────────────────────────────────────────────────
# 3. MPT — Efficient Frontier
# ─────────────────────────────────────────────────────────────────────────────

def chart_mpt(mpt_data: dict) -> go.Figure:
    if "error" in mpt_data:
        fig = go.Figure()
        fig.add_annotation(text=mpt_data["error"], xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        fig.update_layout(title="MPT — Efficient Frontier", **_LAYOUT)
        return fig

    # Random portfolio cloud, coloured by Sharpe ratio
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mpt_data["port_vol"], y=mpt_data["port_ret"],
        mode="markers",
        marker=dict(
            color=mpt_data["port_sharpe"],
            colorscale=[[0, "rgb(231,76,60)"], [0.5, "rgb(170,170,170)"],
                        [1, "rgb(46,204,113)"]],
            size=4, opacity=0.4,
            colorbar=dict(title="Sharpe"),
        ),
        name="Random Portfolios",
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    ))

    # Individual ETF points
    for t, pt in mpt_data["etf_points"].items():
        fig.add_trace(go.Scatter(
            x=[pt["vol"]], y=[pt["ret"]],
            mode="markers+text", text=[t],
            textposition="top center", textfont=dict(size=9),
            marker=dict(size=12, color="white",
                        line=dict(color="rgb(52,73,94)", width=2)),
            name=t,
            hovertemplate=f"<b>{t}</b><br>Vol: {pt['vol']:.2%}<br>Ret: {pt['ret']:.2%}<extra></extra>",
        ))

    # Max-Sharpe portfolio star
    fig.add_trace(go.Scatter(
        x=[mpt_data["max_sharpe_vol"]], y=[mpt_data["max_sharpe_ret"]],
        mode="markers+text", text=["★ Max Sharpe"],
        textposition="top right",
        marker=dict(size=18, color="gold", symbol="star",
                    line=dict(color="black", width=1)),
        name="Max Sharpe Portfolio",
        hovertemplate=(f"Max Sharpe Portfolio<br>"
                       f"Vol: {mpt_data['max_sharpe_vol']:.2%}<br>"
                       f"Ret: {mpt_data['max_sharpe_ret']:.2%}<extra></extra>"),
    ))

    fig.update_layout(
        title="MPT — Efficient Frontier (colour = Sharpe ratio)",
        xaxis_title="Annualised Volatility", yaxis_title="Annualised Return",
        xaxis=dict(tickformat=".0%"), yaxis=dict(tickformat=".0%"),
        showlegend=False, **_LAYOUT,
    )
    return fig




# ─────────────────────────────────────────────────────────────────────────────
# 5. TRACKING ERROR — Horizontal bar
# ─────────────────────────────────────────────────────────────────────────────

def chart_tracking(results: dict) -> go.Figure:
    rows = [(t, r) for t, r in results.items()
            if r["enough_data"] and r["value"] is not None]
    rows.sort(key=lambda x: x[1]["tracking_error"])

    tickers = [r[0] for r in rows]
    te_vals = [r[1]["tracking_error"] for r in rows]
    # Lower TE → greener (for passive ETFs)
    colors = _metric_colors(te_vals, higher_is_better=False)

    fig = go.Figure(go.Bar(
        y=tickers, x=te_vals, orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Tracking Error: %{x:.2%}<extra></extra>",
    ))
    fig.update_layout(
        title="Tracking Error vs SPY (annualised)",
        xaxis_title="Tracking Error", xaxis=dict(tickformat=".0%"),
        **_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. RISK-ADJUSTED — Grouped bar (Sharpe / Sortino / Treynor)
# ─────────────────────────────────────────────────────────────────────────────

def chart_risk_adjusted(results: dict) -> go.Figure:
    rows = [(t, r) for t, r in results.items() if r["enough_data"]]
    rows.sort(key=lambda x: x[1].get("sharpe", 0), reverse=True)

    tickers  = [r[0] for r in rows]
    sharpes  = [r[1].get("sharpe",  0) for r in rows]
    sortinos = [r[1].get("sortino", 0) for r in rows]
    treynors = [r[1].get("treynor", 0) for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Sharpe",  x=tickers, y=sharpes,
                         marker_color="rgb(46,204,113)"))
    fig.add_trace(go.Bar(name="Sortino", x=tickers, y=sortinos,
                         marker_color="rgb(52,152,219)"))
    fig.add_trace(go.Bar(name="Treynor", x=tickers, y=treynors,
                         marker_color="rgb(155,89,182)"))
    fig.update_layout(
        barmode="group", title="Risk-Adjusted Ratios (Sharpe | Sortino | Treynor)",
        yaxis_title="Ratio", **_LAYOUT,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. LIQUIDITY — Scatter: Avg Dollar Volume vs Market Impact
# ─────────────────────────────────────────────────────────────────────────────

def chart_liquidity(results: dict) -> go.Figure:
    tickers, log_dvols, impacts, colors_raw = [], [], [], []
    for t, r in results.items():
        if r["enough_data"] and r["value"] is not None:
            tickers.append(t)
            log_dvols.append(r["value"])
            impacts.append(r["market_impact_pct"])
            colors_raw.append(r["value"])

    colors = _metric_colors(colors_raw, higher_is_better=True)
    fig = _scatter(tickers, log_dvols, impacts, colors,
                   "Log₁₀(Avg Daily Dollar Volume)",
                   "Market Impact on $1M Trade (%)",
                   "Liquidity — Volume vs Market Impact")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. NAV PREMIUM/DISCOUNT — Horizontal bar
# ─────────────────────────────────────────────────────────────────────────────

def chart_nav_premium(results: dict) -> go.Figure:
    rows = [(t, r) for t, r in results.items()
            if r["enough_data"] and r["value"] is not None]
    rows.sort(key=lambda x: x[1]["premium_pct"])

    tickers  = [r[0] for r in rows]
    premiums = [r[1]["premium_pct"] for r in rows]
    colors   = _nav_colors(premiums)

    fig = go.Figure(go.Bar(
        y=tickers, x=premiums, orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Premium/Discount: %{x:.3f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="solid", line_color="black", opacity=0.3)
    fig.update_layout(
        title="Premium / Discount to NAV (%)",
        xaxis_title="Premium (+) / Discount (−) %",
        **_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. LOOK-THROUGH FUNDAMENTALS — P/E scatter
# ─────────────────────────────────────────────────────────────────────────────

def chart_look_through(results: dict) -> go.Figure:
    tickers, pe_vals, dy_vals, colors_raw = [], [], [], []
    for t, r in results.items():
        if r["enough_data"] and r.get("pe") is not None:
            tickers.append(t)
            pe_vals.append(r["pe"])
            dy_vals.append((r.get("dividend_yield") or 0) * 100)
            colors_raw.append(r["pe"])

    # Lower P/E = cheaper = greener
    colors = _metric_colors(colors_raw, higher_is_better=False)
    fig = _scatter(tickers, pe_vals, dy_vals, colors,
                   "Trailing P/E", "Dividend Yield (%)",
                   "Fundamentals — P/E vs Dividend Yield")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. DDM — Implied Upside/Downside bar
# ─────────────────────────────────────────────────────────────────────────────

def chart_ddm(results: dict) -> go.Figure:
    rows = [(t, r) for t, r in results.items()
            if r["enough_data"] and r["value"] is not None]
    rows.sort(key=lambda x: x[1]["implied_upside"], reverse=True)

    tickers = [r[0] for r in rows]
    upsides = [r[1]["implied_upside"] * 100 for r in rows]
    colors  = _metric_colors(upsides, higher_is_better=True)

    fig = go.Figure(go.Bar(
        x=tickers, y=upsides, marker_color=colors,
        hovertemplate="<b>%{x}</b><br>DDM Implied Upside: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    fig.update_layout(
        title="DDM — Implied Upside (+) / Downside (−) vs Current Price",
        yaxis_title="Implied Return (%)", **_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ETF DETAIL PAGE — Price History Chart
# ─────────────────────────────────────────────────────────────────────────────

def chart_price_history(prices: pd.DataFrame, ticker: str) -> go.Figure:
    if ticker not in prices.columns:
        return go.Figure()
    series = prices[ticker].dropna()
    spy    = prices["SPY"].reindex(series.index).dropna()

    # Normalise both to 100 at the ETF's inception for fair comparison
    start  = series.index[0]
    spy_al = spy.reindex(series.index).ffill()

    norm_etf = (series / series.iloc[0]) * 100
    norm_spy = (spy_al / spy_al.iloc[0]) * 100 if len(spy_al) > 0 else None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=norm_etf.index, y=norm_etf.values,
        name=ticker, line=dict(color="rgb(46,204,113)", width=2),
    ))
    if norm_spy is not None:
        fig.add_trace(go.Scatter(
            x=norm_spy.index, y=norm_spy.values,
            name="SPY (benchmark)", line=dict(color="rgb(149,165,166)", width=1.5, dash="dash"),
        ))
    fig.update_layout(
        title=f"{ticker} — Indexed Price History vs SPY (base 100 at ETF inception)",
        xaxis_title="Date", yaxis_title="Indexed Value (100 = inception)",
        **_LAYOUT,
    )
    return fig
