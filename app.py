import streamlit as st
import pandas as pd
from pathlib import Path

# --- Configuration & CSS ---
st.set_page_config(page_title="WhaleWatch Equity", page_icon="🐋", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); padding: 2rem; border-radius: 12px; margin-bottom: 2rem; border: 1px solid rgba(0,212,170,0.2); }
.hero h1 { color: #00D4AA; font-weight: 800; margin:0; }
.hero p { color: #94a3b8; margin:0; }
div[data-testid="stMetricValue"] { color: #00D4AA; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "results" / "latest_signals.csv"
FII_FILE = BASE_DIR / "data" / "fii_dii.csv"
BULK_FILE = BASE_DIR / "data" / "bulk_deals.csv"

# --- Load Data ---
@st.cache_data(ttl=3600)
def load_data():
    signals = pd.read_csv(RESULTS_FILE) if RESULTS_FILE.exists() else pd.DataFrame()
    fii = pd.read_csv(FII_FILE) if FII_FILE.exists() else pd.DataFrame()
    bulk = pd.read_csv(BULK_FILE) if BULK_FILE.exists() else pd.DataFrame()
    return signals, fii, bulk

signals, fii, bulk = load_data()

# --- Header ---
st.markdown('<div class="hero"><h1>🐋 WhaleWatch Equity Engine</h1><p>Daily Institutional Conviction Tracker</p></div>', unsafe_allow_html=True)

if signals.empty:
    st.warning("Data is currently updating or missing. Please check back after market close.")
    st.stop()

# --- Top Metrics ---
bulls = len(signals[signals['Label'] == '🟢 Bullish'])
bears = len(signals[signals['Label'] == '🟠 Bearish'])
flagged = len(signals[signals['Label'] == '🔴 Flagged (Unsafe)'])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Scanned", len(signals))
c2.metric("🟢 Strong Bullish", bulls)
c3.metric("⚪ Neutral", len(signals) - bulls - bears - flagged)
c4.metric("🔴 Disqualified / Unsafe", flagged)

st.markdown("---")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🎯 Today's Signals", "🏦 FII/DII Flows", "📦 Bulk Deals"])

with tab1:
    st.subheader("High Conviction Stocks")
    
    # Filter UI
    label_filter = st.selectbox("Filter by Label:", signals['Label'].unique())
    filtered = signals[signals['Label'] == label_filter].copy()
    
    # Formatting
    filtered['close'] = filtered['close'].apply(lambda x: f"₹{x:,.2f}")
    filtered['turnover'] = (filtered['turnover'] / 10000000).apply(lambda x: f"₹{x:,.1f} Cr")
    
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "symbol": "Stock",
            "close": "Price",
            "delivery_pct": st.column_config.ProgressColumn("Delivery %", format="%.1f%%", min_value=0, max_value=100),
            "volume": st.column_config.NumberColumn("Volume"),
            "turnover": "Turnover",
            "Score": st.column_config.NumberColumn("Conviction Score (0-4)", help="Higher is better"),
            "Label": "Status"
        }
    )

with tab2:
    st.subheader("Market Regime (FII / DII)")
    if not fii.empty:
        st.bar_chart(fii.set_index('date')[['fii_net', 'dii_net']])
        st.dataframe(fii.sort_values('date', ascending=False), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Institutional Deals")
    if not bulk.empty:
        st.dataframe(bulk.sort_values('date', ascending=False), use_container_width=True, hide_index=True)
