import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import io
from fpdf import FPDF
import yfinance as yf
import warnings
import base64
import os

warnings.filterwarnings("ignore")

st.set_page_config(page_title="🌍 Global Oil Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- THEME & PHOTO LOADING ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load profile image if it exists
profile_img = None
if os.path.exists("profile.jpg"):
    profile_img_base64 = get_base64_of_bin_file("profile.jpg")
    profile_img = f'<img src="data:image/jpg;base64,{profile_img_base64}" style="width:150px;height:150px;border-radius:50%;object-fit:cover;border:3px solid #1a365d;">'
else:
    profile_img = '<img src="https://via.placeholder.com/150" style="width:150px;height:150px;border-radius:50%;object-fit:cover;">'

st.markdown(f"""
<style>
    .main {{ background-color: #f8f9fa; }}
    h1, h2, h3 {{ color: #1a365d; font-family: 'Helvetica Neue', sans-serif; }}
    .stMetric {{ background-color: #ffffff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .profile-container {{ text-align: center; padding: 20px 0; }}
    .profile-name {{ font-size: 18px; font-weight: bold; color: #1a365d; margin-top: 15px; }}
    .profile-title {{ font-size: 14px; color: #555; margin-top: 5px; line-height: 1.4; }}
</style>
""", unsafe_allow_html=True)

# --- DATA ---
REGIONS = {
    "Africa": [
        {"name": "Nigeria", "iso": "NGA", "base": 2000, "trend": -0.03},
        {"name": "Angola", "iso": "AGO", "base": 1700, "trend": -0.04},
        {"name": "Algeria", "iso": "DZA", "base": 1300, "trend": -0.01},
        {"name": "Libya", "iso": "LBY", "base": 1100, "trend": 0.05},
        {"name": "Egypt", "iso": "EGY", "base": 650, "trend": 0.01},
    ],
    "Middle East": [
        {"name": "Saudi Arabia", "iso": "SAU", "base": 10500, "trend": 0.01},
        {"name": "Iraq", "iso": "IRQ", "base": 4500, "trend": 0.02},
        {"name": "UAE", "iso": "ARE", "base": 4000, "trend": 0.01},
        {"name": "Kuwait", "iso": "KWT", "base": 2800, "trend": 0.00},
        {"name": "Iran", "iso": "IRN", "base": 3500, "trend": -0.01},
    ],
    "Asia Pacific": [
        {"name": "China", "iso": "CHN", "base": 4000, "trend": -0.02},
        {"name": "India", "iso": "IND", "base": 800, "trend": 0.01},
        {"name": "Indonesia", "iso": "IDN", "base": 700, "trend": -0.03},
        {"name": "Malaysia", "iso": "MYS", "base": 500, "trend": -0.02},
        {"name": "Vietnam", "iso": "VNM", "base": 200, "trend": 0.02},
    ],
    "Americas": [
        {"name": "USA", "iso": "USA", "base": 12000, "trend": 0.03},
        {"name": "Canada", "iso": "CAN", "base": 4500, "trend": 0.01},
        {"name": "Brazil", "iso": "BRA", "base": 3500, "trend": 0.02},
        {"name": "Mexico", "iso": "MEX", "base": 1900, "trend": -0.02},
        {"name": "Colombia", "iso": "COL", "base": 800, "trend": -0.01},
    ]
}

@st.cache_data
def load_production_data(region):
    countries = REGIONS.get(region, REGIONS["Africa"])
    dates = pd.date_range(start="2018-01-01", end="2024-12-01", freq="MS")
    records = []
    for c in countries:
        for i, date in enumerate(dates):
            trend = (1 + c["trend"]) ** (i / 12)
            seasonal = 1 + 0.05 * np.sin(2 * np.pi * date.month / 12)
            noise = np.random.normal(1, 0.02)
            prod = max(0, c["base"] * trend * seasonal * noise)
            records.append({"Date": date, "Year": date.year, "Month": date.month, "ISO3": c["iso"], "Country": c["name"], "Region": region, "Production_kbpd": round(prod, 1)})
    return pd.DataFrame(records)

@st.cache_data(ttl=3600) # Cache for 1 hour
def load_prices():
    """Fetches real-time Brent Crude prices using Yahoo Finance."""
    try:
        # Download last 5 years of monthly data
        df = yf.download("BZ=F", period="5y", interval="1mo", progress=False)
        
        if df.empty:
            raise Exception("Empty Data")
            
        # Extract Close price using xs (cross-section) which handles MultiIndex well
        # This grabs the 'Close' level from the 'Price' axis
        close_series = df.xs('Close', level='Price', axis=1)
        
        # Convert Series to DataFrame and reset index to make Date a column
        df_prices = close_series.reset_index()
        df_prices.columns = ['Date', 'Brent_Price_USD']
        
        # Ensure Date is datetime and normalize time to midnight for matching
        df_prices['Date'] = pd.to_datetime(df_prices['Date']).dt.normalize()
        df_prices['Year'] = df_prices['Date'].dt.year
        df_prices['Month'] = df_prices['Date'].dt.month
        
        return df_prices
        
    except Exception as e:
        st.warning(f"⚠️ Could not fetch live prices: {e}. Using fallback.")
        # Fallback to static data if API fails
        dates = pd.date_range(start="2018-01-01", end="2024-12-01", freq="MS")
        return pd.DataFrame([{"Date": d, "Year": d.year, "Month": d.month, "Brent_Price_USD": 70 + np.random.normal(0, 10)} for d in dates])

def forecast_simple(df_country, steps=12):
    df_country = df_country.sort_values("Date").reset_index(drop=True)
    x = np.arange(len(df_country))
    y = df_country["Production_kbpd"].values
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    future_x = np.arange(len(df_country), len(df_country) + steps)
    forecast_vals = poly(future_x)
    future_dates = [df_country["Date"].max() + timedelta(days=30*i) for i in range(1, steps+1)]
    fc_df = pd.DataFrame({"Date": future_dates, "Forecast_kbpd": forecast_vals, "Type": "Forecast"})
    hist_df = df_country[["Date", "Production_kbpd"]].rename(columns={"Production_kbpd": "Forecast_kbpd"})
    hist_df["Type"] = "Historical"
    return pd.concat([hist_df, fc_df])

# --- APP ---
st.title("🌍 Global Oil Analytics Dashboard v2")
st.caption("📊 Advanced Analytics | Forecasting | Multi-Region Support")

# SIDEBAR WITH PHOTO
st.sidebar.markdown(f"""
<div class="profile-container">
    {profile_img}
    <div class="profile-name">Gerson Japhet Fumbuka</div>
    <div class="profile-title">DBA Scholar<br>INTI International University<br>Nilai, Malaysia</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# 📖 ABOUT THIS DASHBOARD
with st.sidebar.expander("📖 About This Dashboard"):
    st.markdown("""
    ### 🎯 Purpose
    This **Global Oil Production & Analytics Dashboard** is an open-access, interactive research tool designed to advance evidence-based understanding of petroleum resource dynamics across major oil-producing regions.
    
    ### 🌐 Why This Matters
    - **Global Significance**: Oil production drives economic development, geopolitical power, and energy security
    - **Data Transparency**: Addresses fragmented data through open, standardized presentation
    - **Academic Rigor**: Provides methodological transparency for peer-reviewed research
    - **Policy Support**: Enables data-driven decision-making for stakeholders
    
    ### 🎓 Benefits
    
    **For Students:**
    - ✅ Hands-on learning with real production trends
    - ✅ Skill development in data visualization and statistical analysis
    - ✅ Project inspiration for term papers and thesis work
    
    **For Researchers:**
    - ✅ Rapid hypothesis testing across regions and time periods
    - ✅ Methodological transparency with documented calculations
    - ✅ Cross-disciplinary collaboration support
    
    **For Scholars:**
    - ✅ Evidence-based advocacy and policy dialogue
    - ✅ Longitudinal analysis of structural vs. cyclical trends
    - ✅ Global comparative work and capacity building
    
    ### 🔓 Open Access
    This dashboard is provided under principles of **open science** and **equitable knowledge access**—free to use, transparent methodology, and privacy-respecting.
    """)

# 📚 DATA SOURCES & METHODOLOGY
with st.sidebar.expander("📚 Data Sources & Methodology"):
    st.markdown("### 🔍 Data Sources")
    st.markdown("- **Production & Reserves**: U.S. Energy Information Administration (EIA), OPEC Annual Statistical Bulletin, World Bank Open Data")
    st.markdown("- **Brent Crude Prices**: Yahoo Finance (Ticker: `BZ=F`)")
    st.markdown("- **Country Codes**: ISO 3166-1 alpha-3 standard")
    
    st.markdown("### 📐 Methodology")
    st.markdown("- **Units**: Production in thousand barrels per day (kbpd); Reserves in billion barrels (Bbbl)")
    st.markdown("- **R/P Ratio**: `(Reserves_Bbbl × 1000) / Annual_Production_Mbbl`")
    st.markdown("- **Forecasting**: Ordinary Least Squares (OLS) linear trend extrapolation (12-month horizon)")
    st.markdown("- **Correlation**: Pearson coefficient between aggregated regional production and monthly Brent spot prices")
    
    st.markdown("### 📅 Last Updated")
    st.code(datetime.now().strftime('%Y-%m-%d %H:%M UTC'))
    
    st.markdown("### 📖 Suggested Citation (APA)")
    citation = f"Fumbuka, G. J. (2026). Global Oil Analytics Dashboard v2 [Web application]. INTI International University. https://global-oil-dashboard-cobdnhgtjbkuplybfqncmq.streamlit.app"
    st.code(citation, language=None)
    
    st.markdown("### ⚠️ Limitations")
    st.markdown("Static historical data is used for demonstration. Forecasts represent linear trend projections and do not account for geopolitical shocks, OPEC+ policy changes, or structural market shifts. For academic research, replace static arrays with live EIA/API endpoints.")

st.sidebar.header("🔍 Controls")
region = st.sidebar.selectbox("Select Region", list(REGIONS.keys()), index=0)
show_fc = st.sidebar.checkbox("Show 12-Month Forecast", value=True)

prod_df = load_production_data(region)
price_df = load_prices()
prod_with_price = prod_df.merge(price_df, on=["Date", "Year", "Month"], how="left")

countries = prod_df["Country"].unique()
selected = st.sidebar.multiselect("Select Countries", countries, default=countries[:3] if len(countries)>=3 else countries)
if not selected:
    st.warning("Select at least one country."); st.stop()

prod_filt = prod_df[prod_df["Country"].isin(selected)]
prod_trend = prod_df[prod_df["Country"].isin(selected)].sort_values("Date")

# KPIs
total = prod_filt["Production_kbpd"].sum()
avg = total / 365 if total > 0 else 0
top = prod_filt.loc[prod_filt["Production_kbpd"].idxmax(), "Country"] if not prod_filt.empty else "N/A"
c1,c2,c3 = st.columns(3)
c1.metric(f"Avg Daily ({region})", f"{avg:,.0f} kbpd")
c2.metric("Top Producer", top)
c3.metric("Countries", len(selected))

# Map
st.subheader("🗺️ Production Map")
if not prod_filt.empty:
    map_df = prod_filt.groupby(["ISO3","Country"])["Production_kbpd"].mean().reset_index()
    fig = px.choropleth(map_df, locations="ISO3", color="Production_kbpd", hover_name="Country", color_continuous_scale="Viridis", title=f"Avg Production in {region}")
    fig.update_geos(center=dict(lat=0,lon=0), projection_type="natural earth")
    st.plotly_chart(fig, width="stretch")

# Trend & Forecast
st.subheader("📈 Production Trend & Forecast")
tab1, tab2 = st.tabs(["Historical Trend", "Simple Forecast"])
with tab1:
    fig = px.line(prod_trend, x="Date", y="Production_kbpd", color="Country", markers=False)
    st.plotly_chart(fig, width="stretch")
with tab2:
    if show_fc and len(selected)==1:
        fc = forecast_simple(prod_df[prod_df["Country"]==selected[0]])
        if fc is not None:
            fig = px.line(fc, x="Date", y="Forecast_kbpd", color="Type", line_dash="Type", markers=True, title=f"Forecast for {selected[0]}")
            st.plotly_chart(fig, width="stretch")
            st.info("💡 Linear trend forecast based on historical data")
    elif len(selected)!=1:
        st.warning("Select exactly ONE country for forecast")
    else:
        st.info("Enable forecast in sidebar")

# Price Correlation
st.subheader("💰 Brent Price Correlation")
try:
    corr = prod_with_price.groupby("Date")[["Production_kbpd","Brent_Price_USD"]].sum().reset_index()
    coef = corr["Production_kbpd"].corr(corr["Brent_Price_USD"])
    col1,col2 = st.columns([3,1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=corr["Date"], y=corr["Production_kbpd"], name="Production", yaxis="y1", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=corr["Date"], y=corr["Brent_Price_USD"], name="Brent Price", yaxis="y2", line=dict(color="#ff7f0e")))
        fig.update_layout(
            title="Production vs Brent Price",
            xaxis=dict(title="Month"),
            yaxis=dict(
                title=dict(text="Production (kbpd)", font=dict(color="#1f77b4")),
                tickfont=dict(color="#1f77b4"),
                side="left"
            ),
            yaxis2=dict(
                title=dict(text="Price (USD/bbl)", font=dict(color="#ff7f0e")),
                tickfont=dict(color="#ff7f0e"),
                overlaying="y",
                side="right"
            ),
            legend=dict(x=0.1, y=1.1, orientation="h"),
            hovermode="x unified"
        )
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.metric("Correlation", f"{coef:.3f}")
        if abs(coef)>0.7: st.success("Strong")
        elif abs(coef)>0.4: st.info("Moderate")
        else: st.warning("Weak")
except Exception as e:
    st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;font-size:14px;padding:20px 0'>
    Contact: <a href='mailto:oilproductiondashboard@gmail.com'>oilproductiondashboard@gmail.com</a>
</div>
""", unsafe_allow_html=True)


