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
from fpdf import FPDF

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Global Oil Analytics Dashboard v2.2",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mobile-optimized CSS
st.markdown("""
<style>
    @media (max-width: 768px) {
        .main .block-container { padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1rem !important; }
        .stMetric { font-size: 0.9rem !important; padding: 8px !important; }
        .stButton>button { font-size: 0.9rem !important; padding: 0.5rem 1rem !important; }
        .stTextInput>div>div>input { font-size: 16px !important; }
        .stSelectbox>div>div>select { font-size: 16px !important; }
    }
    .stButton>button { min-height: 44px; min-width: 44px; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    .stMetric { background-color: #ffffff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .profile-container { text-align: center; padding: 20px 0; }
    .profile-name { font-size: 18px; font-weight: bold; color: #1a365d; margin-top: 15px; }
    .profile-title { font-size: 14px; color: #555; margin-top: 5px; line-height: 1.4; }
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1a365d; font-family: 'Helvetica Neue', sans-serif; }
    section[data-testid="stSidebar"] { width: 300px; }
    @media (max-width: 768px) { section[data-testid="stSidebar"] { width: 100%; max-width: 100vw; } }
    .stPlotlyChart { touch-action: manipulation; }
    .streamlit-expanderHeader { font-size: 1rem; padding: 10px; }
    body { overflow-x: hidden; }
</style>
""", unsafe_allow_html=True)

# Profile section
def load_profile():
    profile_path = "profile.jpg"
    if os.path.exists(profile_path):
        with open(profile_path, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()
            return f'<img src="data:image/jpeg;base64,{encoded}" style="width:120px;height:120px;border-radius:50%;object-fit:cover;border:3px solid #1a365d;">'
    return '<div style="width:120px;height:120px;border-radius:50%;background:#e2e8f0;display:flex;align-items:center;justify-content:center;color:#718096;font-size:14px;">👤</div>'

# PDF Report Generator
def generate_pdf_report(title, metrics_df):
    """Generate PDF report using FPDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, "Model Performance Metrics", ln=True)
    pdf.set_font('Helvetica', '', 10)
    col_width = 90
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(col_width, 8, 'Model', border=1, fill=True)
    pdf.cell(col_width, 8, 'RMSE', border=1, fill=True, ln=True)
    for _, row in metrics_df.iterrows():
        pdf.cell(col_width, 8, str(row['Model']), border=1)
        pdf.cell(col_width, 8, str(row['RMSE']), border=1, ln=True)
    pdf.ln(10)
    valid = metrics_df[metrics_df['RMSE'] != 'N/A']
    if not valid.empty:
        best_model = valid.loc[valid['RMSE'].astype(float).idxmin(), 'Model']
        best_rmse = valid.loc[valid['RMSE'].astype(float).idxmin(), 'RMSE']
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 10, f"Best Model: {best_model} (RMSE: {best_rmse})", ln=True)
    pdf.set_y(-20)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, "Global Oil Analytics Dashboard | Gerson Japhet Fumbuka | INTI International University", ln=True, align='C')
    return pdf.output(dest='S').encode('latin1')

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
                data.append({"Country": c, "Date": d, "Production_kbpd": max(0, base + np.random.normal(0, 50) + np.sin(d.month/12*2*np.pi)*30), "Region": "Africa"})
        return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def load_prices():
    tickers = ["BZ=F", "BRENTOIL=X", "CL=F"]
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="5y", interval="1mo", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.xs('Close', axis=1, level=0, drop_level=True)
                else:
                    if 'Close' in df.columns:
                        df = df[['Close']]
                    elif 'Adj Close' in df.columns:
                        df = df[['Adj Close']]
                    else:
                        continue
                df.columns = ['Brent_Price_USD']
                df = df.reset_index()
                df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
                df = df.dropna(subset=['Brent_Price_USD'])
                if not df.empty:
                    return df
        except:
            continue
    st.warning("⚠️ Could not fetch live prices. Using fallback data.")
    dates = pd.date_range(start="2018-01-01", end="2024-12-01", freq="MS")
    np.random.seed(42)
    base_price, volatility = 75.0, 15.0
    prices = []
    for i, d in enumerate(dates):
        trend = 0.002 * i
        seasonal = 5.0 * np.sin(2 * np.pi * d.month / 12)
        noise = np.random.normal(0, volatility)
        price = max(20, min(150, base_price + trend + seasonal + noise))
        prices.append(price)
    return pd.DataFrame({"Date": dates, "Brent_Price_USD": prices})

# Forecasting functions
def forecast_simple(df_country, steps=12):
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
    try:
        df = df_country[["Date", "Production_kbpd"]].copy()
        df.columns = ['ds', 'y']
        import logging
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
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
    except Exception as e:
        st.error(f"Prophet error: {e}")
        return None, None, None

def forecast_arima(df_country, steps=12):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        df = df_country.set_index('Date')['Production_kbpd'].dropna()
        if len(df) < 10:
            return None, None
        try:
            model = ARIMA(df, order=(1, 1, 1))
            results = model.fit()
        except:
            model = ARIMA(df, order=(1, 0, 1))
            results = model.fit()
        forecast_result = results.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=steps, freq='MS')
        hist_df = pd.DataFrame({'Date': df.index, 'Forecast': df.values, 'Lower_Bound': np.nan, 'Upper_Bound': np.nan, 'Type': 'Historical'})
        fc_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast_mean.values, 'Lower_Bound': conf_int.iloc[:, 0].values, 'Upper_Bound': conf_int.iloc[:, 1].values, 'Type': 'Forecast'})
        return pd.concat([hist_df, fc_df]), results
    except Exception as e:
        st.error(f"ARIMA Error: {e}")
        return None, None

# Main app
def main():
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
            **Global Oil Analytics Dashboard v2.2** provides:
            - Real-time oil production monitoring
            - ML-powered forecasting with Prophet
            - Price correlation analysis
            - Interactive visualizations
            - Mobile-optimized interface
            - CSV/PDF export functionality
            
            **Version:** 2.2  
            **Last Updated:** April 2026
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
            - ⚠️ **Alerts & Export**: Drop warnings & data downloads
            
            ### 💡 Pro Tips:
            
            - 📱 **Mobile:** Pinch to zoom charts
            - 🖱️ **Desktop:** Click legend to toggle series
            - 📊 **Analysis:** Single country required for forecast
            - 📥 **Data:** All metrics update in real-time
            - 📄 **Export:** Download CSV/PDF reports
            
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

    # ================= MAIN CONTENT =================
    st.title("🛢️ Global Oil Analytics Dashboard v2.2")
    st.caption("ML forecasting • Real-time monitoring • CSV/PDF Export")
    
    # Load data
    prod_df = load_production_data()
    price_df = load_prices()
    
    if not selected:
        st.info("👈 Select countries to begin")
        return
    
    filtered = prod_df[prod_df["Country"].isin(selected)]
    st.caption(f"📊 Showing: {', '.join(selected)}")
    
    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Production", f"{filtered['Production_kbpd'].sum():,.0f} kbpd")
    col2.metric("Avg per Country", f"{filtered['Production_kbpd'].mean():,.0f} kbpd")
    col3.metric("Brent Price", f"${price_df['Brent_Price_USD'].iloc[-1]:.2f}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map", "📈 Forecast", "💰 Correlation", "⚠️ Alerts & Export"])
    
    # Tab 1: Map
    with tab1:
        map_data = filtered.groupby("Country")["Production_kbpd"].mean().reset_index()
        fig_map = px.choropleth(
            map_data,
            locations="Country",
            locationmode="country names",
            color="Production_kbpd",
            color_continuous_scale="OrRd",
            title="Average Oil Production by Country",
            hover_name="Country"
        )
        st.plotly_chart(fig_map, width="stretch")
        csv_data = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Production CSV",
            data=csv_data,
            file_name=f"production_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Tab 2: Forecast
    with tab2:
        if show_fc and len(selected) == 1:
            country_name = selected[0]
            country_df = prod_df[prod_df["Country"] == country_name].sort_values('Date')
            st.info("🤖 Multi-Model Benchmark")
            
            model_choice = st.selectbox(
                "Select Forecasting Model",
                ["Prophet (ML)", "ARIMA (Statistical)", "Linear (Baseline)"],
                index=0
            )
            
            with st.spinner(f"⏳ Training {model_choice}..."):
                fc_df = None
                if model_choice == "Prophet (ML)":
                    fc_df, _, _ = forecast_prophet(country_df)
                elif model_choice == "ARIMA (Statistical)":
                    fc_df, _ = forecast_arima(country_df)
                else:
                    fc_df = forecast_simple(country_df)
            
            if fc_df is not None and not fc_df.empty:
                hist = fc_df[fc_df['Type'] == 'Historical']
                fc = fc_df[fc_df['Type'] == 'Forecast']
                
                fig = go.Figure()
                if not hist.empty:
                    fig.add_trace(go.Scatter(
                        x=hist['Date'], y=hist['Forecast'],
                        mode='lines', name='Historical',
                        line=dict(color='#1f77b4', width=2)
                    ))
                if not fc.empty:
                    fig.add_trace(go.Scatter(
                        x=fc['Date'], y=fc['Forecast'],
                        mode='lines', name=f'{model_choice} Forecast',
                        line=dict(color='#ff7f0e', width=3, dash='dot')
                    ))
                    if 'Lower_Bound' in fc.columns and not fc['Lower_Bound'].isna().all():
                        fig.add_trace(go.Scatter(
                            x=pd.concat([fc['Date'], fc['Date'][::-1]]),
                            y=pd.concat([fc['Upper_Bound'], fc['Lower_Bound'][::-1]]),
                            fill='toself',
                            fillcolor='rgba(255,127,14,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='95% CI'
                        ))
                
                fig.update_layout(
                    title=f"{model_choice} Forecast for {country_name}",
                    xaxis_title="Month",
                    yaxis_title="Production (kbpd)",
                    height=500,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, width="stretch")
                
                # Metrics & PDF Export
                st.subheader("📊 Model Performance (Last 12 Months)")
                try:
                    actual = country_df.tail(12)['Production_kbpd'].values
                    metrics = []
                    
                    # Linear RMSE
                    try:
                        x = np.arange(len(country_df) - 12)
                        y = country_df.head(len(country_df) - 12)['Production_kbpd'].values
                        pred = np.poly1d(np.polyfit(x, y, 1))(np.arange(len(country_df) - 12, len(country_df)))
                        rmse = np.sqrt(np.mean((actual - pred) ** 2))
                        metrics.append({"Model": "Linear", "RMSE": f"{rmse:.2f}"})
                    except:
                        metrics.append({"Model": "Linear", "RMSE": "N/A"})
                    
                    # Prophet RMSE
                    try:
                        import logging
                        logging.getLogger("prophet").setLevel(logging.ERROR)
                        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
                        df_p = country_df[['Date', 'Production_kbpd']].rename(columns={'Date': 'ds', 'Production_kbpd': 'y'})
                        m = Prophet(yearly_seasonality=True)
                        m.fit(df_p)
                        fc_p = m.predict(m.make_future_dataframe(periods=12, freq='MS'))
                        pred_p = fc_p.tail(12)['yhat'].values
                        if len(pred_p) == len(actual):
                            rmse = np.sqrt(np.mean((actual - pred_p) ** 2))
                            metrics.append({"Model": "Prophet", "RMSE": f"{rmse:.2f}"})
                        else:
                            metrics.append({"Model": "Prophet", "RMSE": "N/A"})
                    except:
                        metrics.append({"Model": "Prophet", "RMSE": "N/A"})
                    
                    # ARIMA RMSE
                    try:
                        from statsmodels.tsa.arima.model import ARIMA
                        train = country_df['Production_kbpd'].values[:-12]
                        model = ARIMA(train, order=(1, 1, 1)).fit()
                        pred_a = model.forecast(steps=12)
                        rmse = np.sqrt(np.mean((actual - pred_a) ** 2))
                        metrics.append({"Model": "ARIMA", "RMSE": f"{rmse:.2f}"})
                    except:
                        metrics.append({"Model": "ARIMA", "RMSE": "N/A"})
                    
                    metrics_df = pd.DataFrame(metrics)
                    st.dataframe(metrics_df, width="stretch")
                    
                    valid = metrics_df[metrics_df['RMSE'] != 'N/A']
                    if not valid.empty:
                        best = valid.loc[valid['RMSE'].astype(float).idxmin(), 'Model']
                        st.success(f"🏆 **Best Model:** {best}")
                        
                        # PDF Export Button
                        if st.button("📄 Generate PDF Report"):
                            pdf_data = generate_pdf_report(f"Forecast Report - {country_name}", metrics_df)
                            st.download_button(
                                label="📥 Download PDF",
                                data=pdf_data,
                                file_name=f"report_{country_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
                except Exception as e:
                    st.warning(f"⚠️ Metrics error: {e}")
            else:
                st.warning("No forecast data generated")
        elif len(selected) != 1:
            st.warning("⚠️ Select exactly ONE country for forecasting")
        else:
            st.info("Enable forecast in sidebar")
    
    # Tab 3: Correlation
    with tab3:
        st.subheader("💰 Brent Price Correlation")
        try:
            merged = filtered.merge(price_df[["Date", "Brent_Price_USD"]], on="Date", how="inner")
            if not merged.empty:
                corr = merged.groupby("Date").agg({
                    "Production_kbpd": "sum",
                    "Brent_Price_USD": "mean"
                }).reset_index()
                coef = corr["Production_kbpd"].corr(corr["Brent_Price_USD"])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=corr["Date"], y=corr["Production_kbpd"],
                    mode="lines", name="Total Production",
                    line=dict(color="#1f77b4", width=2), yaxis="y1"
                ))
                fig.add_trace(go.Scatter(
                    x=corr["Date"], y=corr["Brent_Price_USD"],
                    mode="lines", name="Brent Price",
                    line=dict(color="#ff7f0e", width=2), yaxis="y2"
                ))
                fig.update_layout(
                    title="Production vs Brent Price",
                    yaxis=dict(title="Production (kbpd)"),
                    yaxis2=dict(title="Price (USD)", overlaying="y", side="right"),
                    hovermode="x unified",
                    height=500
                )
                st.plotly_chart(fig, width="stretch")
                
                col1, col2 = st.columns(2)
                col1.metric("Correlation", f"{coef:.3f}")
                col2.metric("Strength", "Weak" if abs(coef) < 0.3 else "Moderate" if abs(coef) < 0.7 else "Strong")
                
                csv_corr = corr.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Correlation CSV",
                    data=csv_corr,
                    file_name=f"correlation_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ No overlapping dates for correlation analysis")
        except Exception as e:
            st.error(f"❌ Correlation error: {e}")
    
    # Tab 4: Alerts & Export
    with tab4:
        st.subheader("⚠️ Production Drop Alerts (>10% MoM)")
        alerts = []
        for country in selected:
            cdf = prod_df[prod_df["Country"] == country].sort_values("Date")
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
        
        st.divider()
        st.subheader("📦 Bulk Export")
        col1, col2 = st.columns(2)
        with col1:
            csv_all = prod_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 All Production Data",
                data=csv_all,
                file_name=f"all_production_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        with col2:
            csv_prices = price_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Price Data",
                data=csv_prices,
                file_name=f"prices_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()


