import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yaml
import sys
from pathlib import Path
from datetime import datetime

# --- PATH SETUP ---
base_path = Path(__file__).parent.parent
sentinel_path = base_path / "src" / "sentinel"
if str(sentinel_path) not in sys.path:
    sys.path.insert(0, str(sentinel_path))

from principle_paydown.principle_payment_calculator import RealEstateSentinel
from data_loader import FinancialDataLoader

# --- PAGE CONFIG ---
st.set_page_config(page_title="Real Estate Sentinel", layout="wide", page_icon="🛡️")
st.title("🛡️ Real Estate Sentinel: Strategic Paydown Simulator")

# --- 1. CONFIGURATION LOADING ---
def load_defaults():
    try:
        base = Path(__file__).parent.parent
        with open(base / "config" / "privacy" / "user_profile.yaml") as f:
            prof = yaml.safe_load(f)['personal_finance']
        with open(base / "config" / "constants.yaml") as f:
            const = yaml.safe_load(f)['financial_assumptions']
        return prof, const
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return {}, {}

default_prof, default_const = load_defaults()

# --- INPUT SECTION ---
with st.expander("⚙️ Simulation Inputs", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Asset Data**")
        ui_home_value = st.number_input("Home Value ($)", value=float(default_prof.get('home_value', 0.0)))
        ui_liquid_cash = st.number_input("Current Liquid Cash ($)", value=float(default_prof.get('liquid_cash', 0.0)))
        ui_cash_growth = st.number_input("Monthly Cash Growth ($)", value=float(default_prof.get('liquid_cash_growth', 0.0)), step=500.0)
        
    with col2:
        st.markdown("**Proposed Paydown**")
        def_date_str = default_prof.get('principle_bulk_paydown_date', None)
        if def_date_str:
            def_date = datetime.strptime(str(def_date_str), "%Y-%m-%d").date()
        else:
            def_date = datetime.today()
            
        ui_bulk_to_pay = st.number_input("Principal To-Pay ($)", value=float(default_prof.get('principle_bulk_to_pay', 0.0)), step=1000.0)
        ui_pay_date = st.date_input("Paydown Date", value=def_date)
        
    with col3:
        st.markdown("**Monthly Burn**")
        ui_monthly_pi = st.number_input("Monthly P&I ($)", value=float(default_prof.get('monthly_p&i', 0.0)))
        ui_tax_ins = st.number_input("Tax & Ins ($)", value=float(default_prof.get('monthly_house_insurance&tax', 0.0)))

    with col4:
        st.markdown("**Tax & Share**")
        ui_share = st.slider("Ownership Share", 0.0, 1.0, float(default_prof.get('share_of_home', 0.5)))
        
        # Load Tax Rates from CONSTANTS.YAML
        loaded_tax_rate = float(default_const.get('default_tax_rate', 0.37))
        loaded_cap_gains = float(default_const.get('capital_gains_tax', 0.238))
        
        ui_tax_rate = st.slider("Marginal Tax Rate", 0.1, 0.5, loaded_tax_rate, help="Fed + State. Pulled from constants.yaml")
        ui_cap_gains = st.slider("Capital Gains Tax", 0.0, 0.4, loaded_cap_gains, help="Cap Gains + NIIT. Pulled from constants.yaml")

    scenario_inputs = {
        'home_value': ui_home_value,
        'liquid_cash': ui_liquid_cash, 
        'liquid_cash_growth': ui_cash_growth, 
        'principle_bulk_to_pay': ui_bulk_to_pay,
        'principle_bulk_paydown_date': ui_pay_date,
        'monthly_p&i': ui_monthly_pi,
        'monthly_house_insurance&tax': ui_tax_ins,
        'share_of_home': ui_share,
        'marginal_tax_rate': ui_tax_rate,
        'capital_gains_tax': ui_cap_gains
    }

# --- RUN ENGINE ---
sentinel = RealEstateSentinel(
    user_config_path=base_path / "config" / "privacy" / "user_profile.yaml",
    constants_config_path=base_path / "config" / "constants.yaml",
    override_inputs=scenario_inputs
)
results = sentinel.run_analysis()

# ==============================================================================
# SECTION 1: MORTGAGE SCHEDULE & REALITY
# ==============================================================================
st.header("1. Mortgage Schedule & Reality")

# A. Metrics
loan_bal = results['loan']['balance']
personal_equity_ratio = results['loan']['personal_equity_ratio']
my_net_equity = (ui_home_value * ui_share) - loan_bal
savings = results['loan']['savings']

c_m1, c_m2, c_m3, c_m4 = st.columns(4)
with c_m1:
    st.metric("My Net Equity", f"${my_net_equity:,.0f}", help=f"Value ({ui_home_value:,.0f} * {ui_share}) - Debt ({loan_bal:,.0f})")
with c_m2:
    st.metric("Personal Equity Stake", f"{personal_equity_ratio*100:.1f}%", help="100% - Personal LTV")
with c_m3:
    st.metric("Total Interest Saved", f"${savings['interest_saved']:,.0f}", delta=f"{savings['interest_saved']:,.0f}" if savings['interest_saved'] > 0 else None)
with c_m4:
    years_saved = savings['months_saved'] / 12.0
    st.metric("Time Saved", f"{years_saved:.1f} Years", help=f"New Payoff: {savings['new_payoff'].strftime('%b %Y')}")

# B. Amortization Chart
st.subheader("Amortization Visualization (Includes Historical $38k)")
df_sched = results['loan']['schedule_df']

fig_amort = go.Figure()

# 1. Principal Bar (Dark Blue) - Shows spikes for Bulk Payments
fig_amort.add_trace(go.Bar(
    x=df_sched['date'], 
    y=df_sched['principal_payment'], 
    name='Principal Paid', 
    marker_color='#003366',
    hovertemplate="Date: %{x}<br>Principal: $%{y:,.0f}<extra></extra>"
))

# 2. Interest Bar (Light Blue)
fig_amort.add_trace(go.Bar(
    x=df_sched['date'], 
    y=df_sched['interest_payment'], 
    name='Interest Paid', 
    marker_color='#66CCFF',
    hovertemplate="Date: %{x}<br>Interest: $%{y:,.0f}<extra></extra>"
))

# 3. Balance Line (Red)
fig_amort.add_trace(go.Scatter(
    x=df_sched['date'], 
    y=df_sched['balance'], 
    name='Remaining Balance', 
    line=dict(color='red', width=3), 
    yaxis='y2'
))

fig_amort.update_layout(
    barmode='stack', 
    title="Monthly Payments & Loan Balance (Notice the $38k Bump in Feb 2025)", 
    height=450,
    yaxis=dict(title="Monthly Payment ($)"),
    yaxis2=dict(title="Loan Balance ($)", overlaying='y', side='right', showgrid=False),
    legend=dict(orientation="h", y=1.1, x=0), 
    hovermode="x unified"
)
st.plotly_chart(fig_amort, use_container_width=True)

# ==============================================================================
# SECTION 2: MARKET DYNAMICS & STRATEGY
# ==============================================================================
st.header("2. Strategic Outlook: Invest vs Pay Down")

# A. Combo Chart (Context)
with st.expander("Show Market Context (S&P 500 vs Mortgage Rates)", expanded=True):
    loader = FinancialDataLoader()
    try:
        macro_df = loader.get_macro_data(years=10)
        sp500 = loader.get_sp500_returns_series(years=10)
        port = loader.get_portfolio_returns_series(years=10)
        sp500_raw = sp500.pct_change().fillna(0)
        
        # Volatility Calculation
        rolling_vol = sp500_raw.rolling(window=20).std() * 100 
        
        fig = go.Figure()
        # Left Axis: Investment Returns
        fig.add_trace(go.Scatter(x=sp500.index, y=sp500.values, name='S&P 500', line=dict(color='blue', width=1.5)))
        fig.add_trace(go.Scatter(x=port.index, y=port.values, name='My Portfolio', line=dict(color='orange', width=2)))
        
        # Right Axis: Mortgage Rates
        fig.add_trace(go.Scatter(x=macro_df.index, y=macro_df['15y Mortgage'], name='15Y Fixed', line=dict(color='#90EE90', width=1.5), yaxis='y2'))
        fig.add_trace(go.Scatter(x=macro_df.index, y=macro_df['30y Mortgage'], name='30Y Fixed', line=dict(color='darkgreen', width=2), yaxis='y2'))
        
        # RESTORED: 20-Day Volatility Bar
        fig.add_trace(go.Bar(x=rolling_vol.index, y=rolling_vol.values, name='20d Volatility', marker_color='lightgray', opacity=0.5, yaxis='y2'))
        
        fig.update_layout(
            height=350, 
            yaxis=dict(title="Cum. Return (1.0 = Start)"), 
            yaxis2=dict(title="Rate/Vol %", overlaying='y', side='right', showgrid=False), 
            legend=dict(orientation="h", y=1.1, x=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Market data currently unavailable.")

# B. The Problem & Solution (Monte Carlo)
mc = results['monte_carlo']

st.markdown("""
**The Problem:** Should you pay down your 5.5% mortgage or invest in the market?
* **The Time Horizon:** These numbers are projected **Total Wealth after 15 Years** (End of Loan Term).
* **The Formula:** * **Invest Strategy:** Takes your cash, grows it at variable market rates (minus 23.8% tax on gains).
    * **Paydown Savings:** Takes your cash, "saves" it at the effective mortgage rate (guaranteed ~4% return).
""")

col_strat_1, col_strat_2 = st.columns([1, 2])

with col_strat_1:
    st.subheader("Summary")
    prob_regret = mc['prob_regret'] * 100
    st.write(f"**Probability Paydown > Investing:**")
    st.progress(int(prob_regret))
    
    if prob_regret > 50:
        st.caption(f"**{prob_regret:.1f}%** chance Paydown wins.")
        st.success("Recommendation: **PAY DEBT**")
    else:
        st.caption(f"**{prob_regret:.1f}%** chance Paydown wins.")
        st.info("Recommendation: **INVEST**")

with col_strat_2:
    st.subheader("Outcome Distribution (Projected 15-Year Totals)")
    
    # Calculate Spread for table rows
    spread_p5 = mc['invest_p5'] - mc['mortgage_end_val']
    spread_p50 = mc['invest_p50'] - mc['mortgage_end_val']
    spread_p95 = mc['invest_p95'] - mc['mortgage_end_val']
    
    df_mc = pd.DataFrame({
        "Scenario": ["Bear Market (5%)", "Median Market (50%)", "Bull Market (95%)"],
        "Investment Value": [mc['invest_p5'], mc['invest_p50'], mc['invest_p95']],
        "Guaranteed Debt Savings": [mc['mortgage_end_val'], mc['mortgage_end_val'], mc['mortgage_end_val']],
        "Net Spread (Invest - Paydown)": [spread_p5, spread_p50, spread_p95]
    })
    
    # Formatting helper for color
    def color_spread(val):
        color = 'green' if val > 0 else 'red'
        return f'color: {color}'

    st.dataframe(
        df_mc.style.format({
            "Investment Value": "${:,.0f}", 
            "Guaranteed Debt Savings": "${:,.0f}",
            "Net Spread (Invest - Paydown)": "${:+,.0f}"
        }).applymap(color_spread, subset=['Net Spread (Invest - Paydown)']), 
        hide_index=True, 
        use_container_width=True
    )
    st.caption("Positive Green Spread = Investing Wins. Negative Red Spread = Paydown Wins.")

# ==============================================================================
# SECTION 3: SOLVENCY IMPACT
# ==============================================================================
st.header("3. Solvency Impact")
col_solvency, col_sens = st.columns(2)

with col_solvency:
    post_pay_cash = results['solvency']['post_pay_cash']
    runway = results['solvency']['runway']
    burn = results['solvency']['burn']
    liq_pos = results['solvency']['liquidity_position']
    
    sc1, sc2 = st.columns(2)
    with sc1: st.metric("Projected Cash", f"${post_pay_cash:,.0f}")
    with sc2: st.metric("New Runway", f"{int(runway)} Months", help=f"Cash / Burn (${burn:,.0f})")
    
    if runway < default_const.get('min_runway_months', 6):
        st.error(f"⚠️ **DANGER**: Runway is below safety threshold!")
    else:
        st.success("✅ Runway Safe")
        
    if liq_pos >= 0:
        st.metric("Net Cash Surplus (If Sold)", f"${liq_pos:,.0f}", delta="Safe to Sell")
    else:
        st.metric("Net Cash Shortfall (If Sold)", f"-${abs(liq_pos):,.0f}", delta="Underwater", delta_color="inverse")

with col_sens:
    st.markdown("**Price Sensitivity Table**")
    df_sens = results['solvency']['sensitivity_table']
    
    # Check if 'Scenario' column exists (backward compatibility/safety)
    if 'Scenario' in df_sens.columns:
        tab1, tab2 = st.tabs(["Sell Now", "Sell Dec 2026"])
        
        with tab1:
            st.caption("ℹ️ **Tax Policy**: Assumes standard sale. You owe Federal Income Tax on 50% of the Gain (at Default Tax Rate).")
            df_now = df_sens[df_sens['Scenario'] == "Sell Now"].drop(columns=['Scenario'])
            st.dataframe(df_now.style.format({
                "Home Value": "${:,.0f}", 
                "My Share Value": "${:,.0f}",
                "My Mortgage Balance": "${:,.0f}",
                "Net Cash Before Tax": "${:+,.0f}",
                "Net Cash After Tax": "${:+,.0f}"
            }).applymap(lambda v: 'color: red;' if v < 0 else 'color: green;', subset=['Net Cash After Tax']), use_container_width=True)
            
        with tab2:
            st.caption("ℹ️ **Tax Policy**: Assumes sale after Dec 1 2026. First $500k of Gain is Tax-Free (Exclusion). Remaining Gain taxed at Capital Gains Rate.")
            df_later = df_sens[df_sens['Scenario'] == "Sell Dec 2026"].drop(columns=['Scenario'])
            st.dataframe(df_later.style.format({
                "Home Value": "${:,.0f}", 
                "My Share Value": "${:,.0f}",
                "My Mortgage Balance": "${:,.0f}",
                "Net Cash Before Tax": "${:+,.0f}",
                "Net Cash After Tax": "${:+,.0f}"
            }).applymap(lambda v: 'color: red;' if v < 0 else 'color: green;', subset=['Net Cash After Tax']), use_container_width=True)
    else:
        # Fallback for old dataframe structure (just in case)
        st.dataframe(df_sens.style.format({"Home Value": "${:,.0f}", "My Share Value": "${:,.0f}", "Net Cash Position": "${:+,.0f}"}).applymap(lambda v: 'color: red;' if v < 0 else 'color: green;', subset=['Net Cash Position']), use_container_width=True)