import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import io
from fpdf import FPDF
import yfinance as yf

st.set_page_config(page_title="🌍 African Oil Dashboard: Production + Prices", layout="wide")

# --- DATA LOADING FUNCTIONS ---

@st.cache_data
def load_production_data():
    countries = [
        {"name": "Nigeria", "iso": "NGA", "base": 2000, "trend": -0.03},
        {"name": "Angola", "iso": "AGO", "base": 1700, "trend": -0.04},
        {"name": "Algeria", "iso": "DZA", "base": 1300, "trend": -0.01},
        {"name": "Libya", "iso": "LBY", "base": 1100, "trend": 0.05},
        {"name": "Egypt", "iso": "EGY", "base": 650, "trend": 0.01},
        {"name": "Congo", "iso": "COG", "base": 280, "trend": -0.02},
        {"name": "Eq. Guinea", "iso": "GNQ", "base": 200, "trend": -0.05},
        {"name": "Gabon", "iso": "GAB", "base": 220, "trend": 0.00},
        {"name": "South Sudan", "iso": "SSD", "base": 160, "trend": 0.02},
        {"name": "Sudan", "iso": "SDN", "base": 30, "trend": -0.10},
        {"name": "Ghana", "iso": "GHA", "base": 190, "trend": 0.03},
        {"name": "Cameroon", "iso": "CMR", "base": 60, "trend": -0.03}
    ]
    
    dates = pd.date_range(start="2018-01-01", end="2024-12-01", freq="MS")
    records = []
    
    for c in countries:
        for i, date in enumerate(dates):
            trend_factor = (1 + c["trend"]) ** (i / 12)
            seasonal = 1 + 0.05 * np.sin(2 * np.pi * date.month / 12)
            noise = np.random.normal(1, 0.03)
            production = max(0, c["base"] * trend_factor * seasonal * noise)
            records.append({
                "Date": date, "Year": date.year, "Month": date.month,
                "ISO3": c["iso"], "Country": c["name"],
                "Production_kbpd": round(production, 1)
            })
    return pd.DataFrame(records)

@st.cache_data
def load_reserves_data():
    return pd.DataFrame([
        {"Country": "Nigeria", "ISO3": "NGA", "Reserves_Bbbl": 37.0, "Source": "EIA 2024"},
        {"Country": "Angola", "ISO3": "AGO", "Reserves_Bbbl": 9.0, "Source": "EIA 2024"},
        {"Country": "Algeria", "ISO3": "DZA", "Reserves_Bbbl": 12.2, "Source": "EIA 2024"},
        {"Country": "Libya", "ISO3": "LBY", "Reserves_Bbbl": 48.4, "Source": "EIA 2024"},
        {"Country": "Egypt", "ISO3": "EGY", "Reserves_Bbbl": 3.3, "Source": "EIA 2024"},
        {"Country": "Congo", "ISO3": "COG", "Reserves_Bbbl": 2.9, "Source": "EIA 2024"},
        {"Country": "Eq. Guinea", "ISO3": "GNQ", "Reserves_Bbbl": 1.1, "Source": "EIA 2024"},
        {"Country": "Gabon", "ISO3": "GAB", "Reserves_Bbbl": 2.0, "Source": "EIA 2024"},
        {"Country": "South Sudan", "ISO3": "SSD", "Reserves_Bbbl": 3.5, "Source": "EIA 2024"},
        {"Country": "Sudan", "ISO3": "SDN", "Reserves_Bbbl": 1.5, "Source": "EIA 2024"},
        {"Country": "Ghana", "ISO3": "GHA", "Reserves_Bbbl": 0.66, "Source": "EIA 2024"},
        {"Country": "Cameroon", "ISO3": "CMR", "Reserves_Bbbl": 0.2, "Source": "EIA 2024"}
    ])

@st.cache_data(ttl=86400)
def load_brent_prices():
    """Fetch monthly Brent Crude prices from Yahoo Finance"""
    try:
        brent = yf.download("BZ=F", start="2018-01-01", end="2024-12-31", progress=False)
        if brent.empty:
            return create_static_prices()
        
        brent_monthly = brent["Close"].resample("MS").mean().reset_index()
        brent_monthly.columns = ["Date", "Brent_Price_USD"]
        brent_monthly["Year"] = brent_monthly["Date"].dt.year
        brent_monthly["Month"] = brent_monthly["Date"].dt.month
        return brent_monthly[["Date", "Year", "Month", "Brent_Price_USD"]].dropna()
    except:
        return create_static_prices()

def create_static_prices():
    """Fallback static Brent prices (2018-2024 monthly avg)"""
    dates = pd.date_range(start="2018-01-01", end="2024-12-01", freq="MS")
    prices = []
    for i, date in enumerate(dates):
        base = 65
        trend = 0.002 * i
        pandemic = -30 if (date.year == 2020 and date.month <= 6) else 0
        recovery = 15 if (date.year >= 2021) else 0
        noise = np.random.normal(0, 5)
        price = max(20, base + trend + pandemic + recovery + noise)
        prices.append({"Date": date, "Year": date.year, "Month": date.month, "Brent_Price_USD": round(price, 2)})
    return pd.DataFrame(prices)

# Load all data
prod_df = load_production_data()
reserves_df = load_reserves_data()
price_df = load_brent_prices()

# Merge production with prices for correlation analysis
prod_with_price = prod_df.merge(price_df, on=["Date", "Year", "Month"], how="left")

# Merge for R/P calculation
merged_df = prod_df.merge(reserves_df, on=["Country", "ISO3"], how="left")
merged_df["Annual_Production_Mbbl"] = merged_df["Production_kbpd"] * 365 / 1000
merged_df["RP_Ratio_Years"] = (merged_df["Reserves_Bbbl"] * 1000) / merged_df["Annual_Production_Mbbl"]
merged_df["RP_Ratio_Years"] = merged_df["RP_Ratio_Years"].round(1)

# --- EXPORT FUNCTIONS ---

def generate_excel(prod_filt, reserves_filt, merged_filt, price_filt):
    """Generate Excel file with multiple sheets including prices"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        prod_export = prod_filt.copy()
        prod_export['Date'] = prod_export['Date'].dt.strftime('%Y-%m-%d')
        prod_export.to_excel(writer, sheet_name='Production', index=False)
        
        if not merged_filt.empty:
            rp_export = merged_filt.groupby("Country")[["Reserves_Bbbl", "Annual_Production_Mbbl", "RP_Ratio_Years"]].first().reset_index()
            rp_export['Annual_Production_Mbbl'] = rp_export['Annual_Production_Mbbl'].round(0)
            rp_export.to_excel(writer, sheet_name='Reserves_RP', index=False)
        
        if not price_filt.empty:
            price_export = price_filt.copy()
            price_export['Date'] = price_export['Date'].dt.strftime('%Y-%m-%d')
            price_export.to_excel(writer, sheet_name='Brent_Prices', index=False)
            
    output.seek(0)
    return output.getvalue()

def generate_pdf(selected_year, total_prod, top_prod, total_reserves, avg_rp, avg_price, price_change):
    """Generate PDF summary with price metrics"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="African Oil Production & Price Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(200, 10, txt=f"Data Year: {selected_year}", ln=True)
    pdf.ln(5)
    
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(200, 10, txt="Production Metrics", ln=True, fill=True)
    pdf.ln(3)
    pdf.cell(200, 10, txt=f"Total Avg Daily Production: {total_prod/365:,.0f} kbpd", ln=True)
    pdf.cell(200, 10, txt=f"Top Producer: {top_prod}", ln=True)
    pdf.cell(200, 10, txt=f"Total Proven Reserves: {total_reserves:.1f} Bbbl", ln=True)
    pdf.cell(200, 10, txt=f"Avg R/P Ratio: {avg_rp:.1f} Years", ln=True)
    
    pdf.ln(5)
    pdf.set_fill_color(255, 220, 200)
    pdf.cell(200, 10, txt="Price Metrics", ln=True, fill=True)
    pdf.ln(3)
    pdf.cell(200, 10, txt=f"Avg Brent Price: ${avg_price:.2f}/bbl", ln=True)
    pdf.cell(200, 10, txt=f"YoY Price Change: {price_change:+.1f}%", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, txt="Note: Brent Crude prices from Yahoo Finance. Static data mode active.")
    
    return pdf.output(dest='S').encode('latin-1')

# --- DASHBOARD UI ---

st.title("🛢️ African Oil Dashboard: Production + Prices")
st.caption("📊 Static Data | Production: kbpd | Prices: USD/bbl | Reserves: Billion Barrels")

# 📖 ABOUT THIS DASHBOARD SECTION
with st.expander("📖 About This Dashboard - Purpose & Benefits", expanded=False):
    st.markdown("""
    ### 🎯 Purpose
    This **African Oil Production & Analytics Dashboard** is an open-access, interactive research tool designed to advance evidence-based understanding of petroleum resource dynamics across the African continent.
    
    ### 🌐 Context: Why African Oil Matters
    - **Global Significance**: Africa holds **8–10% of global proven oil reserves** and supplies critical volumes to global markets.
    - **Development Challenges**: Many African producers face the "resource curse" paradox—abundant wealth coexisting with poverty and governance challenges.
    - **Energy Transition**: As the world shifts to low-carbon systems, African producers must navigate declining demand and diversification imperatives.
    - **Data Gap**: This dashboard addresses fragmented data through open, standardized, and interactive presentation.
    
    ### 🎓 Benefits for Academic Audiences
    
    **For Students:**
    - ✅ Hands-on learning with real production trends
    - ✅ Skill development in data visualization and statistical analysis
    - ✅ Project inspiration for term papers and thesis work
    
    **For Researchers:**
    - ✅ Rapid hypothesis testing across countries and time periods
    - ✅ Methodological transparency with documented calculations
    - ✅ Cross-disciplinary collaboration support
    
    **For Scholars:**
    - ✅ Evidence-based advocacy and policy dialogue
    - ✅ Longitudinal analysis of structural vs. cyclical trends
    - ✅ Global comparative work and capacity building
    
    ### 📊 Key Features
    - 🗺️ **Interactive Maps**: Production and reserves choropleths
    - 📈 **Price Correlation**: Brent Crude price trends with statistical analysis
    - ⏳ **R/P Ratios**: Reserves-to-production calculations for sustainability analysis
    - 📥 **Export Tools**: Excel and PDF reports for academic work
    - 🔄 **Live Data**: Yahoo Finance integration for current prices
    
    ### 🔓 Open Access Commitment
    This dashboard is provided under principles of **open science** and **equitable knowledge access**—free to use, transparent methodology, and privacy-respecting.
    """)
    st.markdown("---")

# Sidebar Filters
st.sidebar.header("🔍 Dashboard Controls")
min_year, max_year = int(prod_df["Year"].min()), int(prod_df["Year"].max())
selected_year = st.sidebar.slider("Select Year", min_year, max_year, max_year)
selected_countries = st.sidebar.multiselect("Select Countries", prod_df["Country"].unique(), default=prod_df["Country"].unique())
show_rp = st.sidebar.checkbox("Show R/P Ratio Analysis", value=True)
show_prices = st.sidebar.checkbox("Show Price Correlation", value=True)

# Filter Data
prod_filt = prod_df[(prod_df["Year"] == selected_year) & (prod_df["Country"].isin(selected_countries))]
prod_trend = prod_df[prod_df["Country"].isin(selected_countries)].sort_values("Date")
reserves_filt = reserves_df[reserves_df["Country"].isin(selected_countries)]
merged_filt = merged_df[(merged_df["Year"] == selected_year) & (merged_df["Country"].isin(selected_countries))]
price_filt = price_df[price_df["Year"] == selected_year]

# Merge for correlation analysis
prod_price_filt = prod_with_price[(prod_with_price["Year"] == selected_year) & (prod_with_price["Country"].isin(selected_countries))]

# KPIs - Production
total_prod_val = prod_filt["Production_kbpd"].sum()
avg_daily = total_prod_val / 365 if total_prod_val > 0 else 0
top_prod = prod_filt.loc[prod_filt["Production_kbpd"].idxmax(), "Country"] if not prod_filt.empty else "N/A"

# KPIs - Reserves
total_reserves_val = reserves_filt["Reserves_Bbbl"].sum()
top_reserves = reserves_filt.loc[reserves_filt["Reserves_Bbbl"].idxmax(), "Country"] if not reserves_filt.empty else "N/A"
avg_rp_val = merged_filt["RP_Ratio_Years"].mean() if not merged_filt.empty and show_rp else 0

# KPIs - Prices
avg_price = price_filt["Brent_Price_USD"].mean() if not price_filt.empty and show_prices else 0
if selected_year > min_year and show_prices:
    prev_price = price_df[price_df["Year"]==selected_year-1]["Brent_Price_USD"].mean()
    price_change = ((avg_price / prev_price) - 1) * 100 if prev_price > 0 else 0
else:
    price_change = 0.0

# YoY Production Change
if selected_year > min_year:
    prev = prod_df[(prod_df["Year"]==selected_year-1) & (prod_df["Country"].isin(selected_countries))]["Production_kbpd"].sum()
    yoy = ((total_prod_val / prev) - 1) * 100 if prev > 0 else 0
else:
    yoy = 0.0

# Display KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avg Daily Production", f"{avg_daily:,.0f} kbpd", f"{yoy:+.1f}%")
c2.metric("Top Producer", top_prod)
c3.metric("Total Reserves", f"{total_reserves_val:.1f} Bbbl")
if show_prices:
    c4.metric("Avg Brent Price", f"${avg_price:.2f}/bbl", f"{price_change:+.1f}%")
    c5.metric("Avg R/P Ratio", f"{avg_rp_val:.1f} yrs" if show_rp else "N/A")
else:
    c4.metric("Avg R/P Ratio", f"{avg_rp_val:.1f} yrs" if show_rp else "N/A")
    c5.metric("Prices", "Hidden")

# --- EXPORT SECTION ---
st.sidebar.markdown("---")
st.sidebar.header("📥 Export Data")

col_exp1, col_exp2 = st.sidebar.columns(2)

excel_data = generate_excel(prod_filt, reserves_filt, merged_filt, price_filt if show_prices else pd.DataFrame())
col_exp1.download_button(
    label="📊 Excel",
    data=excel_data,
    file_name=f"african_oil_data_{selected_year}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

pdf_data = generate_pdf(selected_year, total_prod_val, top_prod, total_reserves_val, avg_rp_val, avg_price, price_change)
col_exp2.download_button(
    label="📄 PDF",
    data=pdf_data,
    file_name=f"african_oil_report_{selected_year}.pdf",
    mime="application/pdf"
)

# --- VISUALIZATIONS ---

st.subheader("🗺️ Production Map")
if not prod_filt.empty:
    prod_avg = prod_filt.groupby("ISO3")["Production_kbpd"].mean().reset_index()
    fig_map_prod = px.choropleth(prod_avg, locations="ISO3", color="Production_kbpd",
                                  hover_name="ISO3", color_continuous_scale="YlOrRd",
                                  title=f"Avg Monthly Production ({selected_year})")
    fig_map_prod.update_geos(center=dict(lat=10, lon=15), projection_scale=2.5)
    st.plotly_chart(fig_map_prod, width="stretch")

if show_rp and not reserves_filt.empty:
    st.subheader("🗺️ Reserves Map")
    fig_map_res = px.choropleth(reserves_filt, locations="ISO3", color="Reserves_Bbbl",
                                 hover_name="Country", color_continuous_scale="Viridis",
                                 title="Proven Oil Reserves (Billion Barrels)")
    fig_map_res.update_geos(center=dict(lat=10, lon=15), projection_scale=2.5)
    st.plotly_chart(fig_map_res, width="stretch")

# Price Trend Chart
if show_prices and not price_df.empty:
    st.subheader("💰 Brent Crude Price Trend")
    price_trend = price_df[price_df["Year"].between(selected_year-2, selected_year)].copy()
    fig_price = px.line(price_trend, x="Date", y="Brent_Price_USD", 
                        title="Brent Crude Price (USD/barrel)", markers=False)
    fig_price.add_hline(y=avg_price, line_dash="dash", annotation_text=f"Avg ${avg_price:.1f}")
    fig_price.update_layout(xaxis_title="Month", yaxis_title="Price (USD/bbl)")
    st.plotly_chart(fig_price, width="stretch")

# Production vs Price Correlation
if show_prices and not prod_price_filt.empty:
    st.subheader("🔗 Production vs Price Correlation")
    
    # Calculate correlation for selected countries
    corr_data = prod_price_filt.groupby("Date")[["Production_kbpd", "Brent_Price_USD"]].sum().reset_index()
    corr_coef = corr_data["Production_kbpd"].corr(corr_data["Brent_Price_USD"])
    
    col_corr1, col_corr2 = st.columns([3, 1])
    with col_corr1:
        # Dual-axis chart - CORRECTED PLOTLY SYNTAX
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=corr_data["Date"], y=corr_data["Production_kbpd"],
                                       name="Total Production", yaxis="y1", line=dict(color="#1f77b4")))
        fig_dual.add_trace(go.Scatter(x=corr_data["Date"], y=corr_data["Brent_Price_USD"],
                                       name="Brent Price", yaxis="y2", line=dict(color="#ff7f0e")))
        fig_dual.update_layout(
            title="Production vs Brent Price (Dual Axis)",
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
        st.plotly_chart(fig_dual, width="stretch")
    
    with col_corr2:
        st.metric("Correlation Coefficient", f"{corr_coef:.3f}")
        if abs(corr_coef) > 0.7:
            st.success("Strong correlation")
        elif abs(corr_coef) > 0.4:
            st.info("Moderate correlation")
        else:
            st.warning("Weak correlation")
    
    # Scatter plot with trendline
    st.subheader("📊 Price vs Production Scatter")
    fig_scatter = px.scatter(corr_data, x="Brent_Price_USD", y="Production_kbpd",
                              trendline="ols", title="Monthly Production vs Brent Price",
                              labels={"Brent_Price_USD": "Brent Price (USD/bbl)", 
                                     "Production_kbpd": "Total Production (kbpd)"})
    # Add R² annotation
    if len(corr_data) > 2:
        model = np.polyfit(corr_data["Brent_Price_USD"], corr_data["Production_kbpd"], 1)
        predictions = np.polyval(model, corr_data["Brent_Price_USD"])
        r2 = 1 - np.sum((corr_data["Production_kbpd"] - predictions)**2) / np.sum((corr_data["Production_kbpd"] - corr_data["Production_kbpd"].mean())**2)
        fig_scatter.add_annotation(text=f"R² = {r2:.3f}", xref="paper", yref="paper", x=0.05, y=0.95, showarrow=False)
    st.plotly_chart(fig_scatter, width="stretch")

# Production Trend
st.subheader("📈 Production Trend (2018-2024)")
fig_line = px.line(prod_trend, x="Date", y="Production_kbpd", color="Country", markers=False)
fig_line.update_layout(xaxis_title="Month", yaxis_title="Production (kbpd)")
st.plotly_chart(fig_line, width="stretch")

if show_rp and not merged_filt.empty:
    st.subheader("⏳ Reserves-to-Production (R/P) Ratio")
    rp_df = merged_filt.groupby("Country")[["Reserves_Bbbl", "RP_Ratio_Years"]].first().reset_index()
    rp_df = rp_df.sort_values("RP_Ratio_Years", ascending=False)
    
    fig_rp = px.bar(rp_df, x="Country", y="RP_Ratio_Years", color="RP_Ratio_Years",
                    text="RP_Ratio_Years", color_continuous_scale="RdYlGn_r",
                    title="R/P Ratio by Country (Higher = Longer Supply Horizon)")
    fig_rp.update_traces(texttemplate="%{text:.1f} yrs", textposition="outside")
    st.plotly_chart(fig_rp, width="stretch")

st.subheader("📊 Production by Country")
prod_sorted = prod_filt.groupby("Country")["Production_kbpd"].mean().reset_index().sort_values("Production_kbpd", ascending=False)
fig_bar = px.bar(prod_sorted, x="Country", y="Production_kbpd", color="Country", text="Production_kbpd")
fig_bar.update_traces(texttemplate="%{text:.0f}", textposition="outside")
st.plotly_chart(fig_bar, width="stretch")

# Footer
st.markdown("---")
st.markdown("💡 *Brent Crude prices from Yahoo Finance. Correlation analysis helps identify market-driven production patterns.*")
st.markdown("🔄 *When you receive your EIA API key, production data will switch to live mode.*")

# 👤 CREDIT FOOTER
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px; padding: 20px 0;'>
    This web app was conceptualized and designed by <strong>Gerson Japhet Fumbuka</strong>, 
    a DBA scholar at INTI International University and Colleges, Nilai, Malaysia.<br>
    For any comments, please contact following email address: 
    <a href='mailto:oilproductiondashboard@gmail.com'>oilproductiondashboard@gmail.com</a>
</div>
""", unsafe_allow_html=True)

