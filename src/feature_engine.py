import streamlit as st
import pandas as pd
from pathlib import Path

# --- Configuration & CSS ---
st.set_page_config(page_title="WhaleWatch Equity", page_icon="🐋", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hero Section */
.hero-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem;
    border: 1px solid rgba(0, 212, 170, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.hero-header h1 {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, #00D4AA, #00B4D8, #00D4AA);
    background-size: 200% auto; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite; margin-bottom: 0.2rem;
}
@keyframes shimmer { 0%{background-position:0% center;} 100%{background-position:200% center;} }

/* Custom Button Styling to look like cards */
div.stButton > button {
    width: 100%; border-radius: 10px; height: 5rem;
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid rgba(255,255,255,0.1); color: white;
}
div.stButton > button:hover { border-color: #00D4AA; color: #00D4AA; }

/* FII/DII Cards */
.fii-box {
    padding: 1rem; border-radius: 8px; text-align: center; color: white;
    border: 1px solid rgba(255,255,255,0.1); margin-bottom: 0.5rem;
}
.bg-buy { background: rgba(34, 197, 94, 0.15); border-color: rgba(34, 197, 94, 0.3); }
.bg-sell { background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.3); }
.val-text { font-size: 1.4rem; font-weight: 700; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "results" / "latest_signals.csv"
FII_FILE = BASE_DIR / "data" / "fii_dii.csv"
BULK_FILE = BASE_DIR / "data" / "bulk_deals.csv"
HISTORY_FILE = BASE_DIR / "data" / "history.csv"

# --- Load Data ---
@st.cache_data(ttl=3600)
def load_data():
    signals = pd.read_csv(RESULTS_FILE) if RESULTS_FILE.exists() else pd.DataFrame()
    fii = pd.read_csv(FII_FILE) if FII_FILE.exists() else pd.DataFrame()
    bulk = pd.read_csv(BULK_FILE) if BULK_FILE.exists() else pd.DataFrame()
    history = pd.read_csv(HISTORY_FILE) if HISTORY_FILE.exists() else pd.DataFrame()
    return signals, fii, bulk, history

# Smart Investor Tagging for Bulk Deals
def tag_investor_type(client_name):
    name = str(client_name).upper()
    if any(x in name for x in ["MUTUAL FUND", "LIFE INSURANCE", "SBI", "HDFC", "ICICI", "AXIS", "KOTAK"]):
        return "DII / Domestic"
    elif any(x in name for x in ["MORGAN", "GOLDMAN", "CITI", "MERRILL", "SOCIETE", "VANGUARD", "BLACKROCK"]):
        return "FII / Foreign"
    return "Pro / Other"

signals, fii, bulk, history = load_data()

# --- Header ---
st.markdown(f"""
    <div class="hero-header">
        <h1>🐋 WhaleWatch Equity Engine</h1>
        <p>Institutional Tracking & Dark Pool Scanner</p>
    </div>
""", unsafe_allow_html=True)

if signals.empty:
    st.warning("Data is currently updating or missing. Please check back after market close.")
    st.stop()

# --- Session State for Filters ---
if "trend_filter" not in st.session_state: st.session_state.trend_filter = "All"
if "cat_filter" not in st.session_state: st.session_state.cat_filter = ["Large Cap", "Mid Cap", "Small Cap"]

# --- Sidebar Filters ---
with st.sidebar:
    st.header("⚙️ Scanner Settings")
    st.subheader("🔎 Filters")
    cats = st.multiselect(
        "Market Cap Class", 
        ["Large Cap", "Mid Cap", "Small Cap", "Penny"],
        default=st.session_state.cat_filter
    )
    st.session_state.cat_filter = cats

# --- Top Clickable Metrics ---
c1, c2, c3, c4 = st.columns(4)

total_count = len(signals)
bull_count = len(signals[signals["Label"] == "🟢 Bullish"])
bear_count = len(signals[signals["Label"] == "🟠 Bearish"])

c1.metric("Stocks Scanned", total_count)

with c2:
    if st.button(f"🟢 Bullish ({bull_count})", use_container_width=True):
        st.session_state.trend_filter = "🟢 Bullish"
with c3:
    if st.button(f"🟠 Bearish ({bear_count})", use_container_width=True):
        st.session_state.trend_filter = "🟠 Bearish"
with c4:
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.session_state.trend_filter = "All"

st.markdown("---")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🐋 Whale Radar", "📦 Deals & Flows", "📈 Stock Lookup"])

# --- TAB 1: WHALE RADAR ---
with tab1:
    # Apply Filters
    filtered = signals[signals["Category"].isin(st.session_state.cat_filter)].copy()
    if st.session_state.trend_filter != "All":
        filtered = filtered[filtered["Label"] == st.session_state.trend_filter]

    if filtered.empty:
        st.info("No stocks match the current filters.")
    else:
        # Display Formatting
        filtered['turnover_cr'] = (filtered['turnover'] / 10000000)
        
        st.dataframe(
            filtered[['symbol', 'Category', 'close', 'delivery_pct', 'vol_ratio', 'turnover_cr', 'Score', 'Label']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "symbol": "Stock",
                "Category": "Class",
                "close": st.column_config.NumberColumn("Price", format="₹%.2f"),
                "delivery_pct": st.column_config.ProgressColumn("Delivery %", format="%.1f%%", min_value=0, max_value=100),
                "vol_ratio": st.column_config.NumberColumn("Vol Spike", format="%.1fx"),
                "turnover_cr": st.column_config.NumberColumn("Turnover (Cr)", format="₹%.1f"),
                "Score": st.column_config.ProgressColumn("Conviction Score", min_value=-4, max_value=4),
                "Label": "Status"
            }
        )

# --- TAB 2: FLOWS & BULK DEALS ---
with tab2:
    st.markdown("### 🏦 FII / DII Net Flow (Last 30 Days)")
    if not fii.empty:
        latest_fii = fii.iloc[-1]
        fc1, fc2 = st.columns(2)
        with fc1:
            cls = "bg-buy" if latest_fii['fii_net'] > 0 else "bg-sell"
            st.markdown(f'<div class="fii-box {cls}"><div>FII NET FLOW</div><div class="val-text">₹ {latest_fii["fii_net"]:,.0f} Cr</div></div>', unsafe_allow_html=True)
        with fc2:
            cls = "bg-buy" if latest_fii['dii_net'] > 0 else "bg-sell"
            st.markdown(f'<div class="fii-box {cls}"><div>DII NET FLOW</div><div class="val-text">₹ {latest_fii["dii_net"]:,.0f} Cr</div></div>', unsafe_allow_html=True)

    st.markdown("### 📦 Smart Bulk Deals")
    if not bulk.empty:
        # Smart Processing
        bd = bulk.copy()
        bd.columns = [c.strip().upper() for c in bd.columns]
        client_col = next((c for c in bd.columns if "CLIENT" in c), None)
        
        if client_col:
            bd["Investor Type"] = bd[client_col].apply(tag_investor_type)
            itypes = st.multiselect("Filter Investor Type:", ["FII / Foreign", "DII / Domestic", "Pro / Other"], default=["FII / Foreign", "DII / Domestic"])
            st.dataframe(bd[bd["Investor Type"].isin(itypes)], use_container_width=True, hide_index=True)
        else:
            st.dataframe(bd, use_container_width=True, hide_index=True)

# --- TAB 3: STOCK LOOKUP ---
with tab3:
    if not history.empty:
        all_syms = sorted(history["symbol"].unique())
        search = st.selectbox("Search Stock:", [""] + all_syms)
        if search:
            stock_data = history[history["symbol"] == search].sort_values("date", ascending=False).head(20)
            if not stock_data.empty:
                latest = stock_data.iloc[0]
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Latest Price", f"₹{latest['close']}")
                sc2.metric("Latest Delivery", f"{latest.get('delivery_pct', 0):.1f}%")
                sc3.metric("Latest Volume", f"{latest['volume']:,}")
                
                st.line_chart(stock_data.set_index("date")["close"])
                st.dataframe(stock_data[['date', 'close', 'volume', 'delivery_pct']], use_container_width=True, hide_index=True)
