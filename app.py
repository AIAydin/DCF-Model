"""
DCF (Discounted Cash Flow) Valuation App
A beautiful Streamlit application for stock valuation using DCF methodology.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, Tuple

# Page configuration
st.set_page_config(
    page_title="DCF Valuation Model",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a beautiful light UI
st.markdown(
    """
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #E2E8F0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E293B;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748B;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Valuation result cards */
    .valuation-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 2px solid #22C55E;
        text-align: center;
    }
    
    .valuation-card.undervalued {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border-color: #22C55E;
    }
    
    .valuation-card.overvalued {
        background: linear-gradient(135deg, #FEF2F2 0%, #FECACA 100%);
        border-color: #EF4444;
    }
    
    .valuation-card.fair {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border-color: #F59E0B;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E293B;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E2E8F0;
    }
    
    /* Info boxes */
    .info-box {
        background: #F8FAFC;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #4F46E5;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #F8FAFC;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #E2E8F0;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4F46E5;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: #4F46E5;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #F8FAFC;
        border-radius: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)


class StockDataFetcher:
    """Fetches and processes stock data from Yahoo Finance."""

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_stock_data(ticker: str) -> Dict[str, Any]:
        """Fetch comprehensive stock data."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Validate that we got actual data (yfinance returns empty dict on soft failures)
            if not info or not info.get("symbol") and not info.get("shortName"):
                return {
                    "success": False,
                    "error": f"No data found for ticker '{ticker}'. The symbol may be invalid or data is temporarily unavailable.",
                }

            # Get financial statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow

            # Get historical prices
            hist = stock.history(period="5y")

            return {
                "info": info,
                "income_stmt": income_stmt,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow,
                "history": hist,
                "success": True,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def clear_cache():
        StockDataFetcher.get_stock_data.clear()

    @staticmethod
    def extract_financials(data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key financial metrics from stock data."""
        info = data.get("info", {})
        income_stmt = data.get("income_stmt", pd.DataFrame())
        cash_flow = data.get("cash_flow", pd.DataFrame())
        balance_sheet = data.get("balance_sheet", pd.DataFrame())

        # Extract key metrics with fallbacks
        financials = {
            "company_name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "current_price": info.get(
                "currentPrice", info.get("regularMarketPrice", 0)
            ),
            "market_cap": info.get("marketCap", 0),
            "shares_outstanding": info.get("sharesOutstanding", 0),
            "beta": info.get("beta", 1.0),
            "trailing_pe": info.get("trailingPE", 0),
            "forward_pe": info.get("forwardPE", 0),
            "price_to_book": info.get("priceToBook", 0),
            "dividend_yield": info.get("dividendYield", 0),
            "profit_margin": info.get("profitMargins", 0),
            "revenue_growth": info.get("revenueGrowth", 0),
            "earnings_growth": info.get("earningsGrowth", 0),
        }

        # Extract Free Cash Flow
        if not cash_flow.empty:
            try:
                operating_cf = (
                    cash_flow.loc["Operating Cash Flow"].iloc[0]
                    if "Operating Cash Flow" in cash_flow.index
                    else 0
                )
                capex = (
                    abs(cash_flow.loc["Capital Expenditure"].iloc[0])
                    if "Capital Expenditure" in cash_flow.index
                    else 0
                )
                financials["free_cash_flow"] = operating_cf - capex

                # Get historical FCF for growth calculation
                fcf_history = []
                for i in range(min(5, len(cash_flow.columns))):
                    try:
                        ocf = (
                            cash_flow.loc["Operating Cash Flow"].iloc[i]
                            if "Operating Cash Flow" in cash_flow.index
                            else 0
                        )
                        cx = (
                            abs(cash_flow.loc["Capital Expenditure"].iloc[i])
                            if "Capital Expenditure" in cash_flow.index
                            else 0
                        )
                        fcf_history.append(ocf - cx)
                    except:
                        pass
                financials["fcf_history"] = fcf_history
            except:
                financials["free_cash_flow"] = 0
                financials["fcf_history"] = []
        else:
            financials["free_cash_flow"] = 0
            financials["fcf_history"] = []

        # Extract Total Debt and Cash
        if not balance_sheet.empty:
            try:
                financials["total_debt"] = (
                    balance_sheet.loc["Total Debt"].iloc[0]
                    if "Total Debt" in balance_sheet.index
                    else 0
                )
                financials["cash"] = (
                    balance_sheet.loc["Cash And Cash Equivalents"].iloc[0]
                    if "Cash And Cash Equivalents" in balance_sheet.index
                    else 0
                )
            except:
                financials["total_debt"] = 0
                financials["cash"] = 0
        else:
            financials["total_debt"] = 0
            financials["cash"] = 0

        # Extract Revenue
        if not income_stmt.empty:
            try:
                financials["revenue"] = (
                    income_stmt.loc["Total Revenue"].iloc[0]
                    if "Total Revenue" in income_stmt.index
                    else 0
                )
                financials["net_income"] = (
                    income_stmt.loc["Net Income"].iloc[0]
                    if "Net Income" in income_stmt.index
                    else 0
                )
            except:
                financials["revenue"] = 0
                financials["net_income"] = 0
        else:
            financials["revenue"] = 0
            financials["net_income"] = 0

        return financials


class DCFCalculator:
    """Calculates intrinsic value using DCF methodology."""

    def __init__(
        self,
        free_cash_flow: float,
        growth_rate_y1_5: float,
        growth_rate_y6_10: float,
        terminal_growth_rate: float,
        discount_rate: float,
        shares_outstanding: float,
        total_debt: float,
        cash: float,
    ):
        self.fcf = free_cash_flow
        self.growth_y1_5 = growth_rate_y1_5
        self.growth_y6_10 = growth_rate_y6_10
        self.terminal_growth = terminal_growth_rate
        self.discount_rate = discount_rate
        self.shares = shares_outstanding
        self.debt = total_debt
        self.cash = cash

    def calculate(self) -> Dict[str, Any]:
        """Perform DCF calculation and return detailed results."""

        # Project cash flows for years 1-10
        projected_fcf = []
        pv_fcf = []

        # Years 1-5: Higher growth
        current_fcf = self.fcf
        for year in range(1, 6):
            current_fcf = current_fcf * (1 + self.growth_y1_5)
            pv = current_fcf / ((1 + self.discount_rate) ** year)
            projected_fcf.append(
                {
                    "year": year,
                    "fcf": current_fcf,
                    "pv_fcf": pv,
                    "growth_rate": self.growth_y1_5,
                }
            )
            pv_fcf.append(pv)

        # Years 6-10: Lower growth
        for year in range(6, 11):
            current_fcf = current_fcf * (1 + self.growth_y6_10)
            pv = current_fcf / ((1 + self.discount_rate) ** year)
            projected_fcf.append(
                {
                    "year": year,
                    "fcf": current_fcf,
                    "pv_fcf": pv,
                    "growth_rate": self.growth_y6_10,
                }
            )
            pv_fcf.append(pv)

        # Calculate Terminal Value (Gordon Growth Model)
        terminal_fcf = current_fcf * (1 + self.terminal_growth)
        terminal_value = terminal_fcf / (self.discount_rate - self.terminal_growth)
        pv_terminal_value = terminal_value / ((1 + self.discount_rate) ** 10)

        # Sum of all present values
        sum_pv_fcf = sum(pv_fcf)
        enterprise_value = sum_pv_fcf + pv_terminal_value

        # Equity Value = Enterprise Value - Debt + Cash
        equity_value = enterprise_value - self.debt + self.cash

        # Intrinsic Value per Share
        intrinsic_value = equity_value / self.shares if self.shares > 0 else 0

        return {
            "projected_fcf": projected_fcf,
            "sum_pv_fcf": sum_pv_fcf,
            "terminal_value": terminal_value,
            "pv_terminal_value": pv_terminal_value,
            "enterprise_value": enterprise_value,
            "equity_value": equity_value,
            "intrinsic_value": intrinsic_value,
            "terminal_fcf": terminal_fcf,
        }


def calculate_wacc(
    risk_free_rate: float,
    market_return: float,
    beta: float,
    cost_of_debt: float,
    tax_rate: float,
    debt: float,
    equity: float,
) -> float:
    """Calculate Weighted Average Cost of Capital (WACC)."""
    total_capital = debt + equity
    if total_capital == 0:
        return 0.10  # Default 10%

    weight_debt = debt / total_capital
    weight_equity = equity / total_capital

    # Cost of Equity using CAPM
    cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)

    # After-tax cost of debt
    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)

    # WACC
    wacc = (weight_equity * cost_of_equity) + (weight_debt * after_tax_cost_of_debt)

    return wacc


def format_currency(value: float) -> str:
    """Format number as currency."""
    if abs(value) >= 1e12:
        return f"${value / 1e12:.2f}T"
    elif abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"${value / 1e3:.2f}K"
    else:
        return f"${value:.2f}"


def format_percentage(value: float) -> str:
    """Format number as percentage."""
    return f"{value * 100:.2f}%"


def create_fcf_projection_chart(results: Dict[str, Any]) -> go.Figure:
    """Create FCF projection chart."""
    df = pd.DataFrame(results["projected_fcf"])

    fig = go.Figure()

    # FCF bars
    fig.add_trace(
        go.Bar(
            x=df["year"],
            y=df["fcf"],
            name="Projected FCF",
            marker_color="#4F46E5",
            opacity=0.8,
        )
    )

    # PV FCF line
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["pv_fcf"],
            name="Present Value of FCF",
            mode="lines+markers",
            line=dict(color="#22C55E", width=3),
            marker=dict(size=8),
        )
    )

    fig.update_layout(
        title="Projected Free Cash Flow & Present Value",
        xaxis_title="Year",
        yaxis_title="Amount ($)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="sans-serif"),
        margin=dict(l=60, r=40, t=80, b=60),
    )

    return fig


def create_value_breakdown_chart(results: Dict[str, Any]) -> go.Figure:
    """Create enterprise value breakdown pie chart."""
    labels = ["PV of Projected FCF", "PV of Terminal Value"]
    values = [results["sum_pv_fcf"], results["pv_terminal_value"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker_colors=["#4F46E5", "#22C55E"],
                textinfo="label+percent",
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="Enterprise Value Breakdown",
        template="plotly_white",
        font=dict(family="sans-serif"),
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40),
        annotations=[
            dict(
                text=f"{format_currency(results['enterprise_value'])}",
                x=0.5,
                y=0.5,
                font_size=16,
                showarrow=False,
            )
        ],
    )

    return fig


def create_sensitivity_analysis(
    base_intrinsic: float, dcf_params: Dict[str, Any], current_price: float
) -> Tuple[go.Figure, pd.DataFrame]:
    """Create sensitivity analysis heatmap."""
    discount_rates = np.arange(0.06, 0.16, 0.02)
    terminal_growths = np.arange(0.01, 0.05, 0.01)

    matrix = []
    for dr in discount_rates:
        row = []
        for tg in terminal_growths:
            calc = DCFCalculator(
                free_cash_flow=dcf_params["fcf"],
                growth_rate_y1_5=dcf_params["growth_y1_5"],
                growth_rate_y6_10=dcf_params["growth_y6_10"],
                terminal_growth_rate=tg,
                discount_rate=dr,
                shares_outstanding=dcf_params["shares"],
                total_debt=dcf_params["debt"],
                cash=dcf_params["cash"],
            )
            result = calc.calculate()
            row.append(result["intrinsic_value"])
        matrix.append(row)

    df_matrix = pd.DataFrame(
        matrix,
        index=[f"{r * 100:.0f}%" for r in discount_rates],
        columns=[f"{g * 100:.0f}%" for g in terminal_growths],
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"{g * 100:.0f}%" for g in terminal_growths],
            y=[f"{r * 100:.0f}%" for r in discount_rates],
            colorscale=[[0, "#EF4444"], [0.5, "#FBBF24"], [1, "#22C55E"]],
            text=[[f"${v:.2f}" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Discount Rate: %{y}<br>Terminal Growth: %{x}<br>Intrinsic Value: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Sensitivity Analysis: Intrinsic Value",
        xaxis_title="Terminal Growth Rate",
        yaxis_title="Discount Rate (WACC)",
        template="plotly_white",
        font=dict(family="sans-serif"),
        margin=dict(l=80, r=40, t=80, b=60),
    )

    return fig, df_matrix


def main():
    """Main application function."""

    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>DCF Valuation Model</h1>
        <p>Professional Discounted Cash Flow Analysis for Stock Valuation</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### Stock Selection")
        ticker = (
            st.text_input(
                "Enter Stock Ticker",
                value="AAPL",
                placeholder="e.g., AAPL, MSFT, GOOGL",
                help="Enter the stock ticker symbol",
            )
            .upper()
            .strip()
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            analyze_button = st.button("Analyze Stock", use_container_width=True)
        with col2:
            if st.button("🔄", help="Clear cached data and refresh"):
                StockDataFetcher.clear_cache()
                st.rerun()

        st.markdown("---")

        st.markdown("### DCF Parameters")

        st.markdown("##### Growth Rates")
        growth_y1_5 = (
            st.slider(
                "Years 1-5 Growth Rate",
                min_value=-20.0,
                max_value=50.0,
                value=15.0,
                step=0.5,
                format="%.1f%%",
                help="Expected annual FCF growth rate for years 1-5",
            )
            / 100
        )

        growth_y6_10 = (
            st.slider(
                "Years 6-10 Growth Rate",
                min_value=-10.0,
                max_value=30.0,
                value=8.0,
                step=0.5,
                format="%.1f%%",
                help="Expected annual FCF growth rate for years 6-10",
            )
            / 100
        )

        terminal_growth = (
            st.slider(
                "Terminal Growth Rate",
                min_value=0.0,
                max_value=5.0,
                value=2.5,
                step=0.25,
                format="%.2f%%",
                help="Perpetual growth rate (typically 2-3%)",
            )
            / 100
        )

        st.markdown("##### Discount Rate")
        discount_rate = (
            st.slider(
                "Discount Rate (WACC)",
                min_value=4.0,
                max_value=20.0,
                value=10.0,
                step=0.5,
                format="%.1f%%",
                help="Weighted Average Cost of Capital",
            )
            / 100
        )

        st.markdown("---")

        with st.expander("About DCF"):
            st.markdown("""
            **Discounted Cash Flow (DCF)** is a valuation method that estimates 
            the value of an investment based on its expected future cash flows.
            
            **Key Components:**
            - **Free Cash Flow (FCF)**: Cash available after operations and capex
            - **Growth Rate**: Expected FCF growth over time
            - **Discount Rate**: Rate used to discount future cash flows (WACC)
            - **Terminal Value**: Value of cash flows beyond forecast period
            
            **Formula:**
            ```
            Intrinsic Value = Σ(PV of FCF) + PV of Terminal Value
            ```
            """)

    # Main content
    if analyze_button and ticker:
        with st.spinner(f"Fetching data for {ticker}..."):
            data = StockDataFetcher.get_stock_data(ticker)

        if not data.get("success", False):
            error_msg = data.get("error", "Unknown error")
            st.error(f"Could not fetch data for {ticker}: {error_msg}")
            st.info("💡 Try clicking the 🔄 button to clear the cache and retry.")
            return

        financials = StockDataFetcher.extract_financials(data)

        if financials["free_cash_flow"] == 0 or financials["shares_outstanding"] == 0:
            st.warning(
                f"Insufficient financial data available for {ticker}. Some metrics may not be accurate."
            )

        # Company Overview
        st.markdown(f"## {financials['company_name']} ({ticker})")
        st.markdown(f"**{financials['sector']}** | {financials['industry']}")

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Price", f"${financials['current_price']:.2f}")
        with col2:
            st.metric("Market Cap", format_currency(financials["market_cap"]))
        with col3:
            st.metric(
                "P/E Ratio",
                f"{financials['trailing_pe']:.2f}"
                if financials["trailing_pe"]
                else "N/A",
            )
        with col4:
            st.metric(
                "Beta", f"{financials['beta']:.2f}" if financials["beta"] else "N/A"
            )

        st.markdown("---")

        # Financial Metrics
        st.markdown(
            '<p class="section-header">Financial Overview</p>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Revenue (TTM)", format_currency(financials["revenue"]))
        with col2:
            st.metric("Net Income (TTM)", format_currency(financials["net_income"]))
        with col3:
            st.metric("Free Cash Flow", format_currency(financials["free_cash_flow"]))
        with col4:
            revenue_growth = financials.get("revenue_growth", 0)
            st.metric(
                "Revenue Growth",
                f"{revenue_growth * 100:.1f}%" if revenue_growth else "N/A",
            )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Debt", format_currency(financials["total_debt"]))
        with col2:
            st.metric("Cash & Equivalents", format_currency(financials["cash"]))
        with col3:
            profit_margin = financials.get("profit_margin", 0)
            st.metric(
                "Profit Margin",
                f"{profit_margin * 100:.1f}%" if profit_margin else "N/A",
            )
        with col4:
            st.metric(
                "Shares Outstanding",
                format_currency(financials["shares_outstanding"]).replace("$", ""),
            )

        st.markdown("---")

        # DCF Calculation
        st.markdown(
            '<p class="section-header">DCF Valuation</p>', unsafe_allow_html=True
        )

        dcf = DCFCalculator(
            free_cash_flow=financials["free_cash_flow"],
            growth_rate_y1_5=growth_y1_5,
            growth_rate_y6_10=growth_y6_10,
            terminal_growth_rate=terminal_growth,
            discount_rate=discount_rate,
            shares_outstanding=financials["shares_outstanding"],
            total_debt=financials["total_debt"],
            cash=financials["cash"],
        )

        results = dcf.calculate()

        # Valuation Result
        intrinsic_value = results["intrinsic_value"]
        current_price = financials["current_price"]
        upside = (
            ((intrinsic_value - current_price) / current_price * 100)
            if current_price > 0
            else 0
        )

        # Determine valuation status
        if upside > 15:
            status = "undervalued"
            status_text = "UNDERVALUED"
            status_color = "#22C55E"
        elif upside < -15:
            status = "overvalued"
            status_text = "OVERVALUED"
            status_color = "#EF4444"
        else:
            status = "fair"
            status_text = "FAIRLY VALUED"
            status_color = "#F59E0B"

        # Valuation Cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
            <div class="metric-card">
                <p class="metric-label">Current Price</p>
                <p class="metric-value">${current_price:.2f}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="metric-card" style="border: 2px solid {status_color};">
                <p class="metric-label">Intrinsic Value</p>
                <p class="metric-value" style="color: {status_color};">${intrinsic_value:.2f}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            upside_color = "#22C55E" if upside > 0 else "#EF4444"
            st.markdown(
                f"""
            <div class="metric-card">
                <p class="metric-label">Upside Potential</p>
                <p class="metric-value" style="color: {upside_color};">{upside:+.1f}%</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Status Banner
        st.markdown(
            f"""
        <div class="valuation-card {status}" style="margin-top: 1rem;">
            <h2 style="color: {status_color}; margin: 0;">{status_text}</h2>
            <p style="color: #64748B; margin: 0.5rem 0 0 0;">Based on DCF analysis with {discount_rate * 100:.1f}% discount rate</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            fcf_chart = create_fcf_projection_chart(results)
            st.plotly_chart(fcf_chart, use_container_width=True)

        with col2:
            value_chart = create_value_breakdown_chart(results)
            st.plotly_chart(value_chart, use_container_width=True)

        # DCF Details
        st.markdown(
            '<p class="section-header">DCF Calculation Details</p>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Value Components**")
            details_df = pd.DataFrame(
                {
                    "Component": [
                        "Sum of PV (Years 1-10)",
                        "PV of Terminal Value",
                        "Enterprise Value",
                        "Less: Total Debt",
                        "Plus: Cash",
                        "Equity Value",
                        "Shares Outstanding",
                        "Intrinsic Value per Share",
                    ],
                    "Value": [
                        format_currency(results["sum_pv_fcf"]),
                        format_currency(results["pv_terminal_value"]),
                        format_currency(results["enterprise_value"]),
                        f"({format_currency(financials['total_debt'])})",
                        format_currency(financials["cash"]),
                        format_currency(results["equity_value"]),
                        f"{financials['shares_outstanding'] / 1e9:.2f}B",
                        f"${results['intrinsic_value']:.2f}",
                    ],
                }
            )
            st.dataframe(details_df, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("**Projected Cash Flows**")
            fcf_df = pd.DataFrame(results["projected_fcf"])
            fcf_df["fcf"] = fcf_df["fcf"].apply(format_currency)
            fcf_df["pv_fcf"] = fcf_df["pv_fcf"].apply(format_currency)
            fcf_df["growth_rate"] = fcf_df["growth_rate"].apply(
                lambda x: f"{x * 100:.1f}%"
            )
            fcf_df.columns = ["Year", "Projected FCF", "Present Value", "Growth Rate"]
            st.dataframe(fcf_df, hide_index=True, use_container_width=True)

        st.markdown("---")

        # Sensitivity Analysis
        st.markdown(
            '<p class="section-header">Sensitivity Analysis</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="info-box">
            <strong>How to read this:</strong> The heatmap shows how intrinsic value changes with different discount rates and terminal growth rates.
            <strong style="color: #22C55E;">Green</strong> indicates higher valuations, <strong style="color: #EF4444;">red</strong> indicates lower valuations.
        </div>
        """,
            unsafe_allow_html=True,
        )

        dcf_params = {
            "fcf": financials["free_cash_flow"],
            "growth_y1_5": growth_y1_5,
            "growth_y6_10": growth_y6_10,
            "shares": financials["shares_outstanding"],
            "debt": financials["total_debt"],
            "cash": financials["cash"],
        }

        sensitivity_chart, sensitivity_df = create_sensitivity_analysis(
            intrinsic_value, dcf_params, current_price
        )

        st.plotly_chart(sensitivity_chart, use_container_width=True)

        with st.expander("View Sensitivity Table"):
            st.dataframe(
                sensitivity_df.style.format("${:.2f}"), use_container_width=True
            )

        # Disclaimer
        st.markdown("---")
        st.markdown(
            """
        <div class="info-box" style="border-left-color: #F59E0B;">
            <strong>Disclaimer:</strong> This DCF model is for educational and informational purposes only. 
            It should not be considered as financial advice. The intrinsic value calculated depends heavily 
            on the assumptions used (growth rates, discount rate, etc.). Always conduct thorough research 
            and consult with financial professionals before making investment decisions.
        </div>
        """,
            unsafe_allow_html=True,
        )

    else:
        # Default view when no stock is analyzed
        st.markdown(
            """
        <div style="text-align: center; padding: 4rem 2rem;">
            <h2 style="color: #64748B;">Enter a stock ticker to begin</h2>
            <p style="color: #94A3B8;">Use the sidebar to enter a ticker symbol and customize DCF parameters</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Example stocks
        st.markdown("### Popular Stocks to Analyze")

        col1, col2, col3, col4 = st.columns(4)

        examples = [
            ("AAPL", "Apple Inc.", "Technology"),
            ("MSFT", "Microsoft Corp.", "Technology"),
            ("GOOGL", "Alphabet Inc.", "Technology"),
            ("AMZN", "Amazon.com Inc.", "Consumer"),
        ]

        for col, (ticker, name, sector) in zip([col1, col2, col3, col4], examples):
            with col:
                st.markdown(
                    f"""
                <div class="metric-card" style="text-align: center;">
                    <p style="font-size: 1.5rem; font-weight: 700; margin: 0;">{ticker}</p>
                    <p style="font-size: 0.875rem; color: #64748B; margin: 0.25rem 0;">{name}</p>
                    <p style="font-size: 0.75rem; color: #94A3B8; margin: 0;">{sector}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
