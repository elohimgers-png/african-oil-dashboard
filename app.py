import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
import base64
import os
from prophet import Prophet
from prophet.plot import plot_components_plotly

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Global Oil Analytics Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mobile-optimized CSS
st.markdown("""
<style>
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1rem !important; }
        .stMetric { 
            font-size: 0.9rem !important; 
            padding: 8px !important;
        }
        .stButton>button {
            font-size: 0.9rem !important;
            padding: 0.5rem 1rem !important;
        }
        .stTextInput>div>div>input {
            font-size: 16px !important;
        }
        .stSelectbox>div>div>select {
            font-size: 16px !important;
        }
    }
    .stButton>button {
        min-height: 44px;
        min-width: 44px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .profile-container {
        text-align: center;
        padding: 20px 0;
    }
    .profile-name {
        font-size: 18px;
        font-weight: bold;
        color: #1a365d;
        margin-top: 15px;
    }
    .profile-title {
        font-size: 14px;
        color: #555;
        margin-top: 5px;
        line-height: 1.4;
    }
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1a365d;
        font-family: 'Helvetica Neue', sans-serif;
    }
    section[data-testid="stSidebar"] {
        width: 300px;
    }
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            width: 100%;
            max-width: 100vw;
        }
    }
    .stPlotlyChart {
        touch-action: manipulation;
    }
    .streamlit-expanderHeader {
        font-size: 1rem;
        padding: 10px;
    }
    body {
        overflow-x: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Profile section
def load_profile():
    profile_path = "profile.jpg"
    if os.path.exists(profile_path):
        with open(profile_path, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()
            return f'<img src="data:image/jpg;base64,{encoded}" style="width:120px;height:120px;border-radius:50%;object-fit:cover;border:3px solid #1a365d;">'
    return '<div style="width:120px;height:120px;border-radius:50%;background:#e2e8f0;display:flex;align-items:center;justify-content:center;color:#718096;font-size:14px;">👤</div>'

# Data loading functions
@st.cache_data(ttl=3600)
def load_production_data():
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/elohimgers-png/global-oil-dashboard/main/data/african_oil_production.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        dates = pd.date_range("2018-01-01", "2024-12-01", freq="MS")
        countries = ["Nigeria", "Angola", "Algeria", "Libya", "Egypt"]
        data = []
        for c in countries:
            base = np.random.uniform(800, 2000)
            for d in dates:
                data.append({
                    "Country": c,
                    "Date": d,
                    "Production_kbpd": max(0, base + np.random.normal(0, 50) + np.sin(d.month/12*2*np.pi)*30),
                    "Region": "Africa"
                })
        return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def load_prices():
    """Fetch Brent Crude prices with multiple fallbacks."""
    tickers = ["BZ=F", "BRENTOIL=X", "CL=F"]  # Try Brent Futures, Brent Spot, then WTI
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="5y", interval="1mo", progress=False)
            
            if not df.empty:
                # Handle MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.xs('Close', axis=1, level=0, drop_level=True)
                else:
                    if 'Close' in df.columns:
                        df = df[['Close']]
                    elif 'Adj Close' in df.columns:
                        df = df[['Adj Close']]
                    else:
                        continue  # Try next ticker
                
                df.columns = ['Brent_Price_USD']
                df = df.reset_index()
                df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
                df = df.dropna(subset=['Brent_Price_USD'])
                
                if not df.empty:
                    print(f"✅ Successfully fetched prices using ticker: {ticker}")
                    return df
                    
        except Exception as e:
            print(f"❌ Failed to fetch {ticker}: {e}")
            continue
    
    # If all tickers fail, use realistic fallback data
    st.warning("⚠️ Could not fetch live prices from any source. Using realistic fallback data.")
    dates = pd.date_range(start="2018-01-01", end="2024-12-01", freq="MS")
    
    # Generate realistic oil price pattern (mean-reverting around $75 with volatility)
    np.random.seed(42)  # For reproducibility
    base_price = 75.0
    volatility = 15.0
    prices = []
    
    for i, d in enumerate(dates):
        # Add some trend and seasonality
        trend = 0.002 * i  # Slight upward trend
        seasonal = 5.0 * np.sin(2 * np.pi * d.month / 12)  # Yearly seasonality
        noise = np.random.normal(0, volatility)
        price = max(20, min(150, base_price + trend + seasonal + noise))  # Clamp between $20-$150
        prices.append(price)
    
    fallback_df = pd.DataFrame({
        "Date": dates,
        "Brent_Price_USD": prices
    })
    
    return fallback_df


# Forecasting functions
def forecast_simple(df_country, steps=12):
    """Simple linear forecast for baseline."""
    x = np.arange(len(df_country))
    y = df_country["Production_kbpd"].values
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    future_x = np.arange(len(df_country), len(df_country) + steps)
    future_dates = [df_country["Date"].max() + timedelta(days=30*i) for i in range(1, steps+1)]
    fc_vals = poly(future_x)
    hist_df = df_country[["Date", "Production_kbpd"]].rename(columns={"Production_kbpd": "Forecast"})
    hist_df["Type"] = "Historical"
    fc_df = pd.DataFrame({"Date": future_dates, "Forecast": fc_vals, "Type": "Forecast"})
    return pd.concat([hist_df, fc_df])

def forecast_prophet(df_country, steps=12):
    """Advanced ML forecasting using Facebook Prophet."""
    try:
        from prophet import Prophet
        df = df_country[["Date", "Production_kbpd"]].copy()
        df.columns = ['ds', 'y']
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95)
        model.fit(df)
        future = model.make_future_dataframe(periods=steps, freq='MS')
        forecast = model.predict(future)
        viz_df = pd.DataFrame({
            'Date': forecast['ds'],
            'Forecast': forecast['yhat'],
            'Lower_Bound': forecast['yhat_lower'],
            'Upper_Bound': forecast['yhat_upper'],
            'Type': ['Historical' if d < df['ds'].max() else 'Forecast' for d in forecast['ds']]
        })
        return viz_df, model, forecast
    except ImportError:
        st.error("Prophet library not installed.")
        return None, None, None
    except Exception as e:
        st.error(f"Forecasting error: {e}")
        return None, None, None

def forecast_arima(df_country, steps=12):
    """ARIMA forecasting with proper error handling."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        df = df_country.set_index('Date')['Production_kbpd'].dropna()
        if len(df) < 10:
            st.error("Not enough data for ARIMA")
            return None, None
        try:
            model = ARIMA(df, order=(1, 1, 1))
            results = model.fit()
        except Exception as e:
            st.warning(f"ARIMA fitting issue: {e}. Using simpler model.")
            model = ARIMA(df, order=(1, 0, 1))
            results = model.fit()
        forecast_result = results.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=steps, freq='MS')
        hist_df = pd.DataFrame({
            'Date': df.index,
            'Forecast': df.values,
            'Lower_Bound': np.nan,
            'Upper_Bound': np.nan,
            'Type': 'Historical'
        })
        fc_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': forecast_mean.values,
            'Lower_Bound': conf_int.iloc[:, 0].values,
            'Upper_Bound': conf_int.iloc[:, 1].values,
            'Type': 'Forecast'
        })
        return pd.concat([hist_df, fc_df]), results
    except Exception as e:
        st.error(f"ARIMA Error: {str(e)}")
        return None, None

def calculate_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, and MAPE."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

# Main app
def main():
    # Sidebar
    # ================= SIDEBAR =================
    with st.sidebar:
        # Profile Section
        st.markdown('<div class="profile-container">', unsafe_allow_html=True)
        st.markdown(load_profile(), unsafe_allow_html=True)
        st.markdown('<div class="profile-name">Gerson Japhet Fumbuka</div>', unsafe_allow_html=True)
        st.markdown('<div class="profile-title">DBA Scholar<br>INTI International University<br>Nilai, Malaysia</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # About This Dashboard
        with st.expander("📖 About This Dashboard"):
            st.markdown("""
            **Global Oil Analytics Dashboard** provides:
            - Real-time oil production monitoring
            - ML-powered forecasting with Prophet
            - Price correlation analysis
            - Interactive visualizations
            - Mobile-optimized interface
            
            **Version:** 2.1  
            **Last Updated:** 2026
            """)
        
        # Data Sources & Methodology
        with st.expander("📚 Data Sources & Methodology"):
            st.markdown("""
            **Data Sources:**
            - African oil production data (EIA/OPEC)
            - Brent Crude prices (Yahoo Finance API)
            
            **Methodology:**
            - Prophet ML for time-series forecasting
            - ARIMA statistical modeling
            - Linear regression baseline
            - Pearson correlation for price analysis
            - ISO-3 country codes for mapping
            - 95% confidence intervals for uncertainty
            
            **Forecasting:**
            - 12-month ahead predictions
            - Yearly seasonality detection
            - RMSE-based model comparison
            """)
        
        st.divider()
        
        # Controls Section
        st.subheader("🎛️ Controls")
        
        # Region Selector
        st.markdown("**Select Region**")
        region = st.selectbox(
            "Region",
            ["Africa", "Middle East", "Asia", "Europe", "Americas"],
            index=0,
            help="Select geographic region"
        )
        
        st.divider()
        
        # Country Selection
        st.subheader("🌍 Select Countries")
        all_countries = ["Nigeria", "Angola", "Algeria", "Libya", "Egypt"]
        selected = st.multiselect(
            "Choose countries:",
            options=all_countries,
            default=["Nigeria"],
            help="Select one or more countries to analyze"
        )
        
        st.divider()
        
        # Forecast Options
        show_fc = st.checkbox("📈 Show 12-Month Forecast", value=True)
        
        st.divider()
        
        # User Guide & Instructions Manual
        with st.expander("❓ User Guide & Instructions Manual"):
            st.markdown("""
            ### 📖 Quick Start Guide
            
            **Step 1:** Select region from dropdown  
            **Step 2:** Choose countries to analyze  
            **Step 3:** Enable "Show 12-Month Forecast"  
            **Step 4:** Navigate tabs:
            - 🗺️ **Map**: Geographic production view
            - 📈 **Trend & Forecast**: ML predictions (select 1 country)
            - 💰 **Price Correlation**: Market relationship analysis
            - ⚠️ **Alerts**: >10% MoM drop warnings
            
            ### 💡 Pro Tips:
            
            - 📱 **Mobile:** Pinch to zoom charts
            - 🖱️ **Desktop:** Click legend to toggle series
            - 📊 **Analysis:** Single country required for forecast
            - 📥 **Data:** All metrics update in real-time
            
            ### 📚 Methodology Notes:
            
            **Forecasting Models:**
            - **Prophet:** Facebook ML model with seasonality
            - **ARIMA:** Classical statistical approach
            - **Linear:** Simple baseline regression
            
            **Performance Metrics:**
            - RMSE calculated on last 12 months
            - Lower RMSE = Better model
            - Best model highlighted in green
            
            **Correlation Analysis:**
            - Pearson coefficient (r)
            - |r| < 0.3: Weak
            - 0.3 ≤ |r| < 0.7: Moderate
            - |r| ≥ 0.7: Strong
            
            ---
            **Need Help?**  
            📧 oilproductiondashboard@gmail.com  
            DBA Scholar, INTI International University
            """)    
    # Main content
    st.title("🛢️ Global Oil Analytics Dashboard")
    st.caption("Real-time production monitoring • ML forecasting • Mobile-optimized")
    
    # Load data
    prod_df = load_production_data()
    price_df = load_prices()
    
    if selected:
        filtered = prod_df[prod_df["Country"].isin(selected)]
        
        # KPIs
        col1, col2, col3 = st.columns(3)
        total_prod = filtered["Production_kbpd"].sum()
        avg_prod = filtered["Production_kbpd"].mean()
        latest_price = price_df["Brent_Price_USD"].iloc[-1]
        
        col1.metric("Total Production", f"{total_prod:,.0f} kbpd")
        col2.metric("Avg per Country", f"{avg_prod:,.0f} kbpd")
        col3.metric("Brent Price", f"${latest_price:.2f}")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Production Map", "📈 Trend & Forecast", "💰 Price Correlation", "⚠️ Alerts"])
        
        # Tab 1: Map
        with tab1:
            fig_map = px.choropleth(
                filtered.groupby("Country")["Production_kbpd"].mean().reset_index(),
                locations="Country",
                locationmode="country names",
                color="Production_kbpd",
                color_continuous_scale="OrRd",
                title="Average Oil Production by Country"
            )
            st.plotly_chart(fig_map, width="stretch")
        
        # Tab 2: Forecast with Multi-Model Comparison
        with tab2:
            if show_fc and len(selected)==1:
                country_name = selected[0]
                country_df = prod_df[prod_df["Country"]==country_name].sort_values('Date')
                
                st.info("🤖 Multi-Model Forecasting Benchmark")
                
                model_choice = st.selectbox(
                    "Select Forecasting Model",
                    ["Prophet (ML)", "ARIMA (Statistical)", "Linear (Baseline)"],
                    index=0
                )
                
                st.write(f"**Running:** {model_choice}")
                
                fc_df = None
                
                if model_choice == "Prophet (ML)":
                    st.success("🧠 Using Prophet ML model with seasonality detection")
                    fc_df, model_obj, raw_fc = forecast_prophet(country_df)
                    
                elif model_choice == "ARIMA (Statistical)":
                    st.success("📈 Using ARIMA statistical model")
                    fc_df, model_obj = forecast_arima(country_df)
                    
                else:  # Linear
                    st.success("📉 Using Linear Regression baseline")
                    fc_df = forecast_simple(country_df)
                
                if fc_df is not None and not fc_df.empty:
                    hist_data = fc_df[fc_df['Type']=='Historical']
                    fc_data = fc_df[fc_df['Type']=='Forecast']
                    
                    fig = go.Figure()
                    
                    if not hist_data.empty:
                        fig.add_trace(go.Scatter(
                            x=hist_data['Date'],
                            y=hist_data['Forecast'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='#1f77b4', width=2)
                        ))
                    
                    if not fc_data.empty:
                        fig.add_trace(go.Scatter(
                            x=fc_data['Date'],
                            y=fc_data['Forecast'],
                            mode='lines',
                            name=f'{model_choice} Forecast',
                            line=dict(color='#ff7f0e', width=3, dash='dot')
                        ))
                        
                        if 'Lower_Bound' in fc_data.columns and not fc_data['Lower_Bound'].isna().all():
                            fig.add_trace(go.Scatter(
                                x=pd.concat([fc_data['Date'], fc_data['Date'][::-1]]),
                                y=pd.concat([fc_data['Upper_Bound'], fc_data['Lower_Bound'][::-1]]),
                                fill='toself',
                                fillcolor='rgba(255,127,14,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='95% CI'
                            ))
                    
                    fig.update_layout(
                        title=f"{model_choice} Forecast for {country_name}",
                        xaxis=dict(title="Month"),
                        yaxis=dict(title="Production (kbpd)"),
                        hovermode="x unified",
                        height=500
                    )
                    
                    st.plotly_chart(fig, width="stretch")
                    
                    # Model Performance Metrics
                    st.subheader("📊 Model Performance (Last 12 Months)")
                    actual_last_12 = country_df.tail(12)['Production_kbpd'].values
                    metrics_rows = []
                    
                    # Linear
                    x_train = np.arange(len(country_df)-12)
                    y_train = country_df.head(len(country_df)-12)['Production_kbpd'].values
                    coeffs_l = np.polyfit(x_train, y_train, 1)
                    pred_l = np.poly1d(coeffs_l)(np.arange(len(country_df)-12, len(country_df)))
                    rmse_l = np.sqrt(np.mean((actual_last_12 - pred_l)**2))
                    metrics_rows.append({"Model": "Linear", "RMSE": f"{rmse_l:.2f}"})
                    
                    # Prophet
                    try:
                        prophet_df = country_df[['Date', 'Production_kbpd']].rename(columns={'Date':'ds', 'Production_kbpd':'y'})
                        prop_mod = Prophet(yearly_seasonality=True, verbose=0)
                        prop_mod.fit(prophet_df)
                        prop_fc = prop_mod.predict(prophet_df.tail(12))
                        rmse_p = np.sqrt(np.mean((actual_last_12 - prop_fc['yhat'].values)**2))
                        metrics_rows.append({"Model": "Prophet", "RMSE": f"{rmse_p:.2f}"})
                    except:
                        metrics_rows.append({"Model": "Prophet", "RMSE": "N/A"})
                    
                    # ARIMA
                    try:
                        from statsmodels.tsa.arima.model import ARIMA
                        arima_mod = ARIMA(country_df['Production_kbpd'].values[:-12], order=(1,1,1))
                        arima_res = arima_mod.fit()
                        pred_a = arima_res.forecast(steps=12)
                        rmse_a = np.sqrt(np.mean((actual_last_12 - pred_a)**2))
                        metrics_rows.append({"Model": "ARIMA", "RMSE": f"{rmse_a:.2f}"})
                    except:
                        metrics_rows.append({"Model": "ARIMA", "RMSE": "N/A"})
                    
                    metrics_df = pd.DataFrame(metrics_rows)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    best = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model'] if 'N/A' not in metrics_df['RMSE'].values else "N/A"
                    if best != "N/A":
                        st.success(f"🏆 **Best Model:** {best}")
                        
                else:
                    st.warning("No forecast data generated")
                    
            elif len(selected)!=1:
                st.warning("⚠️ Select exactly ONE country for forecasting")
            else:
                st.info("Enable forecast in sidebar")
        
        # Tab 3: Correlation
        with tab3:
            st.subheader("💰 Brent Price Correlation")
            try:
                prod_with_price = filtered.merge(price_df[["Date", "Brent_Price_USD"]], on="Date", how="inner")
                if not prod_with_price.empty:
                    corr = prod_with_price.groupby("Date").agg({
                        "Production_kbpd": "sum",
                        "Brent_Price_USD": "mean"
                    }).reset_index()
                    coef = corr["Production_kbpd"].corr(corr["Brent_Price_USD"])
                    
                    fig_corr = go.Figure()
                    fig_corr.add_trace(go.Scatter(
                        x=corr["Date"],
                        y=corr["Production_kbpd"],
                        mode="lines",
                        name="Total Production",
                        line=dict(color="#1f77b4")
                    ))
                    fig_corr.add_trace(go.Scatter(
                        x=corr["Date"],
                        y=corr["Brent_Price_USD"],
                        mode="lines",
                        name="Brent Price",
                        line=dict(color="#ff7f0e"),
                        yaxis="y2"
                    ))
                    fig_corr.update_layout(
                        title="Production vs Brent Price",
                        yaxis=dict(title="Production (kbpd)"),
                        yaxis2=dict(title="Price (USD)", overlaying="y", side="right"),
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_corr, width="stretch")
                    st.metric("Correlation Coefficient", f"{coef:.3f}", 
                              delta="Weak" if abs(coef) < 0.3 else "Moderate" if abs(coef) < 0.7 else "Strong")
                else:
                    st.warning("No overlapping dates for correlation analysis")
            except Exception as e:
                st.error(f"Correlation error: {e}")
        
        # Tab 4: Alerts
        with tab4:
            st.subheader("⚠️ Production Drop Alerts (>10% MoM)")
            alerts = []
            for country in selected:
                cdf = prod_df[prod_df["Country"]==country].sort_values("Date")
                if len(cdf) >= 2:
                    latest = cdf.iloc[-1]["Production_kbpd"]
                    previous = cdf.iloc[-2]["Production_kbpd"]
                    change = ((latest - previous) / previous) * 100
                    if change < -10:
                        alerts.append(f"{country}: ▼ {abs(change):.1f}% drop")
            if alerts:
                for a in alerts:
                    st.error(f"🚨 {a}")
            else:
                st.success("✅ No significant drops detected")
    
    else:
        st.info("👈 Select at least one country in the sidebar to begin analysis")

if __name__ == "__main__":
    main()


