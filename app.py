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
import logging

# Suppress logs
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Global Oil Analytics Dashboard v3.0",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    @media (max-width: 768px) {
        .main .block-container { padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.2rem !important; }
        .stButton>button { min-height: 44px; min-width: 44px; }
    }
    .profile-container { text-align: center; padding: 20px 0; }
    .profile-name { font-size: 18px; font-weight: bold; color: #1a365d; margin-top: 15px; }
    .profile-title { font-size: 14px; color: #555; margin-top: 5px; line-height: 1.4; }
    .stMetric { background-color: #ffffff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# Profile
def load_profile():
    profile_path = "profile.jpg"
    if os.path.exists(profile_path):
        with open(profile_path, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()
            return f'<img src="image/jpg;base64,{encoded}" style="width:120px;height:120px;border-radius:50%;object-fit:cover;border:3px solid #1a365d;">'
    return '<div style="width:120px;height:120px;border-radius:50%;background:#e2e8f0;display:flex;align-items:center;justify-content:center;">👤</div>'

# Data
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
                data.append({"Country": c, "Date": d, "Production_kbpd": max(0, base + np.random.normal(0, 50)), "Region": "Africa"})
        return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def load_prices():
    try:
        df = yf.download("BZ=F", period="5y", interval="1mo", progress=False)
        if df.empty: raise Exception("Empty")
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs('Close', axis=1, level=0, drop_level=True)
            df.columns = ['Brent_Price_USD']
        else:
            df = df[['Close']]
            df.columns = ['Brent_Price_USD']
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
        return df.dropna(subset=['Brent_Price_USD'])
    except:
        dates = pd.date_range(start="2018-01-01", end="2024-12-01", freq="MS")
        return pd.DataFrame([{"Date": d, "Brent_Price_USD": 70 + np.random.normal(0, 10)} for d in dates])

# Forecast Functions
def forecast_linear(df_country, steps=12):
    try:
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
    except:
        return None

def forecast_prophet(df_country, steps=12):
    try:
        if df_country is None or df_country.empty or len(df_country) < 24:
            return None
        df = df_country[["Date", "Production_kbpd"]].copy()
        df.columns = ['ds', 'y']
        if df['ds'].isna().any() or df['y'].isna().any():
            return None
        model = Prophet(yearly_seasonality=True, interval_width=0.95, verbose=False)
        model.fit(df)
        future = model.make_future_dataframe(periods=steps, freq='MS')
        forecast = model.predict(future)
        if forecast is None or forecast.empty:
            return None
        return pd.DataFrame({
            'Date': forecast['ds'],
            'Forecast': forecast['yhat'],
            'Lower_Bound': forecast['yhat_lower'],
            'Upper_Bound': forecast['yhat_upper'],
            'Type': ['Historical' if d < df['ds'].max() else 'Forecast' for d in forecast['ds']]
        })
    except:
        return None

def forecast_arima(df_country, steps=12):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        df = df_country.set_index('Date')['Production_kbpd'].dropna()
        if len(df) < 10:
            return None
        model = ARIMA(df, order=(1, 1, 1))
        results = model.fit()
        forecast_result = results.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=steps, freq='MS')
        hist_df = pd.DataFrame({'Date': df.index, 'Forecast': df.values, 'Lower_Bound': np.nan, 'Upper_Bound': np.nan, 'Type': 'Historical'})
        fc_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast_mean.values, 'Lower_Bound': conf_int.iloc[:, 0].values, 'Upper_Bound': conf_int.iloc[:, 1].values, 'Type': 'Forecast'})
        return pd.concat([hist_df, fc_df])
    except:
        return None

def forecast_lstm(df_country, steps=12):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from sklearn.preprocessing import MinMaxScaler
        
        data = df_country["Production_kbpd"].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        look_back = 60
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        if len(X) == 0:
            return None
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, batch_size=1, epochs=1, verbose=0)
        
        last_sequence = scaled_data[-look_back:].reshape(1, look_back, 1)
        predictions = []
        for _ in range(steps):
            pred = model.predict(last_sequence, verbose=0)
            predictions.append(pred[0, 0])
            new_point = pred.reshape(1, 1, 1)
            last_sequence = np.append(last_sequence[:, 1:, :], new_point, axis=1)
        
        pred_scaled = np.array(predictions).reshape(-1, 1)
        pred_real = scaler.inverse_transform(pred_scaled)
        
        future_dates = [df_country["Date"].max() + timedelta(days=30*i) for i in range(1, steps+1)]
        fc_df = pd.DataFrame({"Date": future_dates, "Forecast": pred_real.flatten(), "Type": "Forecast"})
        hist_df = df_country[["Date", "Production_kbpd"]].rename(columns={"Production_kbpd": "Forecast"})
        hist_df["Type"] = "Historical"
        
        return pd.concat([hist_df, fc_df])
    except:
        return None

def evaluate_models(country_df):
    try:
        actual = country_df.tail(12)['Production_kbpd'].values
        train_df = country_df.head(len(country_df) - 12)
        if len(train_df) < 24:
            train_df = country_df
        metrics = {}
        
        # Linear
        try:
            lin_fc = forecast_linear(train_df, 12)
            if lin_fc is not None and len(lin_fc) >= 12:
                metrics['Linear'] = float(np.sqrt(np.mean((actual - lin_fc.tail(12)['Forecast'].values)**2)))
            else:
                metrics['Linear'] = 999.0
        except:
            metrics['Linear'] = 999.0

        # ARIMA
        try:
            arima_fc = forecast_arima(train_df, 12)
            if arima_fc is not None and len(arima_fc) >= 12:
                metrics['ARIMA'] = float(np.sqrt(np.mean((actual - arima_fc.tail(12)['Forecast'].values)**2)))
            else:
                metrics['ARIMA'] = 999.0
        except:
            metrics['ARIMA'] = 999.0

        # Prophet
        try:
            if len(train_df) >= 24:
                prop_fc = forecast_prophet(train_df, 12)
                if prop_fc is not None and len(prop_fc) >= 12 and 'Forecast' in prop_fc.columns:
                    pred = prop_fc.tail(12)['Forecast'].values
                    if len(pred) == len(actual) and not np.any(np.isnan(pred)):
                        metrics['Prophet'] = float(np.sqrt(np.mean((actual - pred)**2)))
                    else:
                        metrics['Prophet'] = 999.0
                else:
                    metrics['Prophet'] = 999.0
            else:
                metrics['Prophet'] = 999.0
        except:
            metrics['Prophet'] = 999.0

        # LSTM
        try:
            lstm_fc = forecast_lstm(train_df, 12)
            if lstm_fc is not None and len(lstm_fc) >= 12:
                metrics['LSTM'] = float(np.sqrt(np.mean((actual - lstm_fc.tail(12)['Forecast'].values)**2)))
            else:
                metrics['LSTM'] = 999.0
        except:
            metrics['LSTM'] = 999.0

        return metrics
    except:
        return {}

# Main
def main():
    with st.sidebar:
        st.markdown('<div class="profile-container">', unsafe_allow_html=True)
        st.markdown(load_profile(), unsafe_allow_html=True)
        st.markdown('<div class="profile-name">Gerson Japhet Fumbuka</div>', unsafe_allow_html=True)
        st.markdown('<div class="profile-title">DBA Scholar<br>INTI International University<br>Nilai, Malaysia</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        with st.expander("📖 About"):
            st.markdown("**v3.0**: Deep Learning + Multi-Model Benchmarking")
        
        with st.expander("📚 Methodology"):
            st.markdown("Prophet, LSTM, ARIMA, Linear | RMSE benchmarking")
        
        st.divider()
        st.subheader("🎛️ Controls")
        region = st.selectbox("Region", ["Africa", "Middle East", "Asia", "Europe", "Americas"], index=0)
        st.divider()
        st.subheader("🌍 Countries")
        all_countries = ["Nigeria", "Angola", "Algeria", "Libya", "Egypt"]
        selected = st.multiselect("Choose:", options=all_countries, default=["Nigeria"])
        st.divider()
        show_fc = st.checkbox("📈 Show Forecast", value=True)
        st.divider()
        with st.expander("❓ User Guide"):
            st.markdown("Select 1 country for forecast. LSTM may take 15-30s.")

    st.title("🛢️ Global Oil Analytics Dashboard v3.0")
    st.caption("Deep Learning • Multi-Model Benchmarking • Real-time")
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map", "📈 Forecast", "💰 Correlation", "⚠️ Alerts"])
    
    # Tab 1: Map
    with tab1:
        map_data = filtered.groupby("Country")["Production_kbpd"].mean().reset_index()
        country_iso = {"Nigeria": "NGA", "Angola": "AGO", "Algeria": "DZA", "Libya": "LBY", "Egypt": "EGY"}
        map_data["ISO"] = map_data["Country"].map(country_iso)
        fig_map = px.choropleth(map_data, locations="ISO", locationmode="ISO-3", color="Production_kbpd",
                                color_continuous_scale="OrRd", title="Oil Production", hover_name="Country")
        fig_map.update_geos(showland=True, landcolor="LightGray", showocean=True, oceancolor="LightBlue")
        st.plotly_chart(fig_map, width="stretch")
    
    # Tab 2: Forecast
    with tab2:
        if show_fc and len(selected)==1:
            country_name = selected[0]
            country_df = prod_df[prod_df["Country"]==country_name].sort_values('Date')
            st.info("🤖 Multi-Model Benchmark")
            
            model_choice = st.selectbox("Model", ["Prophet (ML)", "LSTM (Deep Learning)", "ARIMA (Statistical)", "Linear (Baseline)"], index=0)
            
            with st.spinner(f"⏳ Training {model_choice}..."):
                fc_df = None
                if model_choice == "Prophet (ML)":
                    fc_df = forecast_prophet(country_df)
                elif model_choice == "LSTM (Deep Learning)":
                    fc_df = forecast_lstm(country_df)
                elif model_choice == "ARIMA (Statistical)":
                    fc_df = forecast_arima(country_df)
                else:
                    fc_df = forecast_linear(country_df)
            
            if fc_df is not None:
                hist = fc_df[fc_df['Type']=='Historical']
                fc = fc_df[fc_df['Type']=='Forecast']
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Forecast'], mode='lines', name='Historical', line=dict(color='#1f77b4', width=2)))
                fig.add_trace(go.Scatter(x=fc['Date'], y=fc['Forecast'], mode='lines', name=f'{model_choice}', line=dict(color='#ff7f0e', width=3, dash='dot')))
                if 'Lower_Bound' in fc.columns and not fc['Lower_Bound'].isna().all():
                    fig.add_trace(go.Scatter(x=pd.concat([fc['Date'], fc['Date'][::-1]]), y=pd.concat([fc['Upper_Bound'], fc['Lower_Bound'][::-1]]), fill='toself', fillcolor='rgba(255,127,14,0.2)', line=dict(color='rgba(255,255,255,0)'), name='95% CI'))
                fig.update_layout(title=f"{model_choice} for {country_name}", xaxis_title="Month", yaxis_title="Production (kbpd)", height=500)
                st.plotly_chart(fig, width="stretch")
                
                st.subheader("📊 Model Comparison (RMSE)")
                with st.spinner("Calculating..."):
                    metrics = evaluate_models(country_df)
                if metrics:
                    df_m = pd.DataFrame(list(metrics.items()), columns=['Model', 'RMSE']).sort_values('RMSE')
                    st.dataframe(df_m, width="stretch")
                    st.success(f"🏆 Best: {df_m.iloc[0]['Model']}")
        elif len(selected)!=1:
            st.warning("⚠️ Select ONE country for forecast")
        else:
            st.info("Enable forecast in sidebar")
    
    # Tab 3: Correlation
    with tab3:
        st.subheader("💰 Price Correlation")
        try:
            merged = filtered.merge(price_df[["Date", "Brent_Price_USD"]], on="Date", how="inner")
            if not merged.empty:
                corr = merged.groupby("Date").agg({"Production_kbpd": "sum", "Brent_Price_USD": "mean"}).reset_index()
                coef = corr["Production_kbpd"].corr(corr["Brent_Price_USD"])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=corr["Date"], y=corr["Production_kbpd"], mode="lines", name="Production", line=dict(color="#1f77b4", width=2), yaxis="y1"))
                fig.add_trace(go.Scatter(x=corr["Date"], y=corr["Brent_Price_USD"], mode="lines", name="Brent Price", line=dict(color="#ff7f0e", width=2), yaxis="y2"))
                fig.update_layout(title="Production vs Price", yaxis=dict(title="Production (kbpd)"), yaxis2=dict(title="Price (USD)", overlaying="y", side="right"), hovermode="x unified", height=500)
                st.plotly_chart(fig, width="stretch")
                
                col1, col2 = st.columns(2)
                col1.metric("Correlation", f"{coef:.3f}")
                col2.metric("Strength", "Weak" if abs(coef)<0.3 else "Moderate" if abs(coef)<0.7 else "Strong")
            else:
                st.warning("⚠️ No overlapping dates")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    # Tab 4: Alerts
    with tab4:
        st.subheader("⚠️ Production Alerts (>10% MoM)")
        alerts = []
        for country in selected:
            cdf = prod_df[prod_df["Country"]==country].sort_values("Date")
            if len(cdf) >= 2:
                change = ((cdf.iloc[-1]["Production_kbpd"] - cdf.iloc[-2]["Production_kbpd"]) / cdf.iloc[-2]["Production_kbpd"]) * 100
                if change < -10:
                    alerts.append(f"{country}: ▼ {abs(change):.1f}%")
        if alerts:
            for a in alerts:
                st.error(f"🚨 {a}")
        else:
            st.success("✅ No significant drops")

if __name__ == "__main__":
    main()
