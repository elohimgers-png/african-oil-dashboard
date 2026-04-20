

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import io
from fpdf import FPDF
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="🌍 Global Oil Analytics Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME CONFIGURATION ---
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- DATA GENERATION ENGINE ---

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
def load_global_production_data(selected_region):
    countries = REGIONS.get(selected_region, REGIONS["Africa"])
    
    dates = pd.date_range(start="2018-01-01", end="2024-12-01", freq="MS")
    records = []
    
    for c in countries:
        for i, date in enumerate(dates):
            trend_factor = (1 + c["trend"]) ** (i / 12)
            seasonal = 1 + 0.05 * np.sin(2 * np.pi * date.month / 12)
            noise = np.random.normal(1, 0.02)
            production = max(0, c["base"] * trend_factor * seasonal * noise)
            
            records.append({
                "Date": date, 
                "Year": date.year, 
                "Month": date.month,
                "ISO3": c["iso"], 
                "Country": c["name"],
                "Region": selected_region,
                "Production_kbpd": round(production, 1)
            })
    return pd.DataFrame(records)

@st.cache_data(ttl=86400)
def load_brent_prices():
    try:
        brent = yf.download("BZ=F", start="2018-01-01", end="2024-12-31", progress=False)
        if brent.empty:
            raise Exception("No data")
        brent_monthly = brent["Close"].resample("MS").mean().reset_index()
        brent_monthly.columns = ["Date", "Brent_Price_USD"]
        brent_monthly["Year"] = brent_monthly["Date"].dt.year
        brent_monthly["Month"] = brent_monthly["Date"].dt.month
        return brent_monthly[["Date", "Year", "Month", "Brent_Price_USD"]].dropna()
    except:
        dates = pd.date_range(start="2018-01-01", end="2024-12-01", freq="MS")
        prices = [{"Date": d, "Year": d.year, "Month": d.month, "Brent_Price_USD": 65 + np.random.normal(0, 5)} for d in dates]
        return pd.DataFrame(prices)

# --- FORECASTING FUNCTION (ARIMA) ---

def forecast_production_arima(df_country, steps=12):
    df_country = df_country.sort_values("Date")
    history = df_country["Production_kbpd"].values
    
    try:
        model = ARIMA(history, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        
        last_date = df_country["Date"].max()
        future_dates = [last_date + timedelta(days=30*i) for i in range(1, steps+1)]
        
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast_kbpd": forecast,
            "Type": "Forecast"
        })
        
        hist_df = df_country[["Date", "Production_kbpd"]].copy()
        hist_df.rename(columns={"Production_kbpd": "Forecast_kbpd"}, inplace=True)
        hist_df["Type"] = "Historical"
        
        return pd.concat([hist_df, forecast_df])
    except Exception as e:
        st.error(f"Forecasting error: {e}")
        return None

# --- MAIN APP ---

st.title("🌍 Global Oil Analytics Dashboard")
st.caption("📊 Advanced Analytics | Forecasting | Multi-Region Support")

# Sidebar Controls
st.sidebar.header("🔍 Global Controls")
selected_region = st.sidebar.selectbox("Select Region", list(REGIONS.keys()), index=0)
show_forecast = st.sidebar.checkbox("Show 12-Month Forecast (ARIMA)", value=True)

# Load Data
try:
    prod_df = load_global_production_data(selected_region)
    price_df = load_brent_prices()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Merge Price Data
prod_with_price = prod_df.merge(price_df, on=["Date", "Year", "Month"], how="left")

# Filter by Country
all_countries = prod_df["Country"].unique()
selected_countries = st.sidebar.multiselect("Select Countries", all_countries, default=all_countries[:3] if len(all_countries) >= 3 else all_countries)

if not selected_countries:
    st.warning("Please select at least one country.")
    st.stop()

# Filter Dataframes
prod_filt = prod_df[prod_df["Country"].isin(selected_countries)]
prod_trend = prod_df[prod_df["Country"].isin(selected_countries)].sort_values("Date")

# --- KPIs ---
total_prod = prod_filt["Production_kbpd"].sum()
avg_daily = total_prod / 365 if total_prod > 0 else 0
top_country = prod_filt.loc[prod_filt["Production_kbpd"].idxmax(), "Country"] if not prod_filt.empty else "N/A"

c1, c2, c3 = st.columns(3)
c1.metric(f"Avg Daily Production ({selected_region})", f"{avg_daily:,.0f} kbpd")
c2.metric("Top Producer", top_country)
c3.metric("Active Countries", len(selected_countries))

# --- VISUALIZATIONS ---

st.subheader("🗺️ Regional Production Map")
if not prod_filt.empty:
    map_data = prod_filt.groupby(["ISO3", "Country"])["Production_kbpd"].mean().reset_index()
    fig_map = px.choropleth(map_data, locations="ISO3", color="Production_kbpd",
                            hover_name="Country", color_continuous_scale="Viridis",
                            title=f"Avg Monthly Production in {selected_region}")
    fig_map.update_geos(center=dict(lat=0, lon=0), projection_type="natural earth")
    st.plotly_chart(fig_map, width="stretch")

st.subheader("📈 Production Trend & Forecast")
tab1, tab2 = st.tabs(["Historical Trend", "ARIMA Forecast"])

with tab1:
    fig_line = px.line(prod_trend, x="Date", y="Production_kbpd", color="Country", markers=False)
    fig_line.update_layout(xaxis_title="Month", yaxis_title="Production (kbpd)")
    st.plotly_chart(fig_line, width="stretch")

with tab2:
    if show_forecast and len(selected_countries) == 1:
        country_name = selected_countries[0]
        df_country = prod_df[prod_df["Country"] == country_name]
        forecast_result = forecast_production_arima(df_country)
        
        if forecast_result is not None:
            fig_forecast = px.line(forecast_result, x="Date", y="Forecast_kbpd", color="Type",
                                   line_dash="Type", markers=True,
                                   title=f"12-Month Production Forecast for {country_name}")
            fig_forecast.update_traces(line=dict(width=3))
            st.plotly_chart(fig_forecast, width="stretch")
            st.info("💡 *Forecast generated using ARIMA(1,1,1) model based on historical trends.*")
    elif len(selected_countries) != 1:
        st.warning("⚠️ Please select exactly **one** country to view the ARIMA forecast.")
    else:
        st.info("Enable 'Show 12-Month Forecast' in sidebar.")

# --- PRICE CORRELATION ---
st.subheader("💰 Brent Price Correlation")
try:
    corr_data = prod_with_price.groupby("Date")[["Production_kbpd", "Brent_Price_USD"]].sum().reset_index()
    corr_coef = corr_data["Production_kbpd"].corr(corr_data["Brent_Price_USD"])

    col_corr1, col_corr2 = st.columns([3, 1])
    with col_corr1:
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=corr_data["Date"], y=corr_data["Production_kbpd"],
                                       name="Total Production", yaxis="y1", line=dict(color="#1f77b4")))
        fig_dual.add_trace(go.Scatter(x=corr_data["Date"], y=corr_data["Brent_Price_USD"],
                                       name="Brent Price", yaxis="y2", line=dict(color="#ff7f0e")))
        fig_dual.update_layout(
            title="Production vs Brent Price (Dual Axis)",
            xaxis=dict(title="Month"),
            yaxis=dict(title=dict(text="Production (kbpd)", font=dict(color="#1f77b4")), tickfont=dict(color="#1f77b4"), side="left"),
            yaxis2=dict(title=dict(text="Price (USD/bbl)", font=dict(color="#ff7f0e")), tickfont=dict(color="#ff7f0e"), overlaying="y", side="right"),
            legend=dict(x=0.1, y=1.1, orientation="h"),
            hovermode="x unified"
        )
        st.plotly_chart(fig_dual, width="stretch")

    with col_corr2:
        st.metric("Correlation Coefficient", f"{corr_coef:.3f}")
        if abs(corr_coef) > 0.7: st.success("Strong correlation")
        elif abs(corr_coef) > 0.4: st.info("Moderate correlation")
        else: st.warning("Weak correlation")
except Exception as e:
    st.error(f"Error in correlation analysis: {e}")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px; padding: 20px 0;'>
    This web app was conceptualized and designed by <strong>Gerson Japhet Fumbuka</strong>, 
    a DBA scholar at INTI International University and Colleges, Nilai, Malaysia.<br>
    For any comments, please contact following email address: 
    <a href='mailto:oilproductiondashboard@gmail.com'>oilproductiondashboard@gmail.com</a>
</div>
""", unsafe_allow_html=True)

