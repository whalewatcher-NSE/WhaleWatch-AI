"""
🐋 WhaleWatch AI — Institutional Activity Tracker for NSE 500
=============================================================
Tracks "Whale Moves" (high delivery %, abnormal volume, big price swings).
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os
import requests
import warnings
warnings.filterwarnings("ignore")

# =============================================
# CONFIGURATION
# =============================================
APP_TITLE = "WhaleWatch AI"
APP_ICON = "🐋"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
HISTORY_FILE = os.path.join(DATA_DIR, "whale_history.csv")
FII_DII_FILE = os.path.join(DATA_DIR, "fii_dii.csv")
BULK_DEALS_FILE = os.path.join(DATA_DIR, "bulk_deals.csv")

MIN_DELIVERY_PCT = 50.0
MIN_VOLUME_RATIO = 2.0
MIN_PRICE_CHANGE_PCT = 3.0
MA_WINDOW = 20

# Tiers
TIER_LARGE_PRICE = 500
TIER_LARGE_VOLUME = 500_000
TIER_MID_PRICE = 100
TIER_MID_VOLUME = 100_000
TIER_SMALL_PRICE = 20
TIER_SMALL_VOLUME = 25_000

# Headers for NSE
NSE_HEADERS = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'accept-language': 'en-US,en;q=0.9',
    'referer': 'https://www.nseindia.com/'
}

COLUMN_ALIASES = {
    "symbol": ["SYMBOL", "Symbol", "TckrSymb", "TCKR_SYMB"],
    "series": ["SERIES", "Series", "SrNm", "SERIES "],
    "open": ["OPEN_PRICE", "OPEN", "OpnPric", "OPEN_PRICE "],
    "high": ["HIGH_PRICE", "HIGH", "HghPric", "HIGH_PRICE "],
    "low": ["LOW_PRICE", "LOW", "LwPric", "LOW_PRICE "],
    "close": ["CLOSE_PRICE", "CLOSE", "ClsPric", "CLOSE_PRICE "],
    "prev_close": ["PREV_CLOSE", "PREVCLOSE", "PrvsClsgPric", "PREV_CLOSE ", "PREV_CLS_PRICE"],
    "volume": ["TTL_TRD_QNTY", "TOTTRDQTY", "TtlTradgVol", "TOTAL_TRADE_QUANTITY", "TTL_TRD_QNTY "],
    "delivery_qty": ["DELIV_QTY", "DlvryQty", "DELIVERY_QTY", "DELIV_QTY "],
    "delivery_pct": ["DELIV_PER", "DlvryPct", "DELIVERY_PCT", "DELIV_PER ", " DELIV_PER"],
    "date": ["DATE1", "DATE", "TIMESTAMP", "TradDt", "Date"],
}

# =============================================
# CUSTOM CSS
# =============================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.hero-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1rem;
    border: 1px solid rgba(0, 212, 170, 0.15); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.hero-header h1 {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(90deg, #00D4AA, #00B4D8, #00D4AA);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite; margin-bottom: 0.3rem;
}
@keyframes shimmer { 0%{background-position:0% center;} 100%{background-position:200% center;} }
.hero-header p { color: #94a3b8; font-size: 1rem; margin: 0; }
.freshness-banner {
    border-radius: 10px; padding: 0.7rem 1.2rem; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 10px; font-size: 0.88rem;
}
.freshness-live { background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.05)); border: 1px solid rgba(34,197,94,0.25); color: #22c55e; }
.freshness-stale { background: linear-gradient(135deg, rgba(234,179,8,0.1), rgba(234,179,8,0.05)); border: 1px solid rgba(234,179,8,0.25); color: #eab308; }
.freshness-empty { background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05)); border: 1px solid rgba(239,68,68,0.25); color: #ef4444; }
.fii-dii-card { border-radius: 12px; padding: 1rem 1.3rem; text-align: center; border: 1px solid rgba(0,212,170,0.1); }
.fii-buy { background: linear-gradient(145deg, #0a2e1a, #0e1a12); border-color: rgba(34,197,94,0.2); }
.fii-sell { background: linear-gradient(145deg, #2e0a0a, #1a0e0e); border-color: rgba(239,68,68,0.2); }
.fii-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.fii-value { font-size: 1.5rem; font-weight: 800; margin: 0.3rem 0; }
.fii-value.positive { color: #22c55e; }
.fii-value.negative { color: #ef4444; }
.fii-sub { font-size: 0.75rem; color: #475569; }
.metric-card {
    background: linear-gradient(145deg, #1a1f2e, #141820);
    border: 1px solid rgba(0, 212, 170, 0.1); border-radius: 12px;
    padding: 1.2rem 1.5rem; text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 800; color: #00D4AA; line-height: 1; }
.metric-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.4rem; }
.tier-badge { display: inline-block; font-weight: 600; font-size: 0.65rem; padding: 1px 8px; border-radius: 12px; margin-left: 6px; vertical-align: middle; }
.tier-large { background: #1a3a2a; color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.tier-mid { background: #1a2a3a; color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
.tier-small { background: #2a2a1a; color: #eab308; border: 1px solid rgba(234,179,8,0.3); }
.tier-penny { background: #2a1a1a; color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.section-divider { border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(0,212,170,0.3), transparent); margin: 1.5rem 0; }
.stock-card { background: linear-gradient(145deg, #1a1f2e, #141820); border: 1px solid rgba(0,212,170,0.12); border-radius: 14px; padding: 1.5rem; margin-bottom: 1rem; }
.live-dot { display: inline-block; width: 8px; height: 8px; background: #22c55e; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity:1; box-shadow:0 0 0 0 rgba(34,197,94,0.4); } 50% { opacity:0.7; box-shadow:0 0 0 6px rgba(34,197,94,0); } }
.app-footer { text-align: center; color: #475569; font-size: 0.75rem; padding: 2rem 0 1rem; border-top: 1px solid #1e2433; margin-top: 3rem; }
</style>
"""

# =============================================
# DATA LAYER
# =============================================
def get_nse_session():
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=5)
    except Exception:
        pass
    return s

def _resolve_column(df, canonical):
    aliases = COLUMN_ALIASES.get(canonical, [])
    for alias in aliases:
        for col in df.columns:
            if col.strip() == alias.strip():
                return col
    return None

def _normalize_dataframe(df):
    if df is None or df.empty: return None
    rename_map = {}
    for canonical in COLUMN_ALIASES:
        actual = _resolve_column(df, canonical)
        if actual and actual != canonical:
            rename_map[actual] = canonical
    df = df.rename(columns=rename_map)
    required = ["symbol", "close", "prev_close", "volume"]
    for col in required:
        if col not in df.columns: return None
    if "series" in df.columns:
        df = df[df["series"].str.strip() == "EQ"].copy()
    for col in ["open", "high", "low", "close", "prev_close", "volume", "delivery_qty", "delivery_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["symbol", "close", "volume"])
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_data(date_obj):
    date_str = date_obj.strftime("%d-%m-%Y")
    # Try library for Bhavcopy (assuming this part is not broken yet)
    try:
        from nselib import capital_market
        raw = capital_market.bhav_copy_with_delivery(date_str)
        df = _normalize_dataframe(raw)
        if df is not None and len(df) > 0:
            df["date"] = date_obj.strftime("%Y-%m-%d")
            return df
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fii_dii_data():
    """Fetch aggregate FII/DII trading activity via direct API."""
    try:
        session = get_nse_session()
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        r = session.get(url, timeout=5)
        data = r.json()
        result = {}
        for item in data:
            category = item.get("category", "").upper()
            buy = float(item.get("buyValue", 0))
            sell = float(item.get("sellValue", 0))
            net = buy - sell
            if "FII" in category or "FPI" in category:
                result["fii"] = {"buy": buy, "sell": sell, "net": net}
            elif "DII" in category:
                result["dii"] = {"buy": buy, "sell": sell, "net": net}
        if result:
            return result
    except Exception:
        pass
    
    # Fallback to file
    if os.path.exists(FII_DII_FILE):
        try:
            df = pd.read_csv(FII_DII_FILE)
            if not df.empty:
                latest = df.iloc[-1]
                return {
                    "fii": {"buy": float(latest.get("fii_buy",0)), "sell": float(latest.get("fii_sell",0)), "net": float(latest.get("fii_net",0))},
                    "dii": {"buy": float(latest.get("dii_buy",0)), "sell": float(latest.get("dii_sell",0)), "net": float(latest.get("dii_net",0))},
                }
        except Exception:
            pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bulk_deals():
    """Fetch today's bulk/block deals via direct API."""
    try:
        session = get_nse_session()
        url = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"
        r = session.get(url, timeout=5)
        data = r.json()
        frames = []
        if "BULK_DEALS_DATA" in data:
            b = pd.DataFrame(data["BULK_DEALS_DATA"])
            if not b.empty: frames.append(b)
        if "BLOCK_DEALS_DATA" in data:
            bl = pd.DataFrame(data["BLOCK_DEALS_DATA"])
            if not bl.empty: frames.append(bl)
        if frames:
            return pd.concat(frames, ignore_index=True)
    except Exception:
        pass

    # Fallback to file
    if os.path.exists(BULK_DEALS_FILE):
        try:
            df = pd.read_csv(BULK_DEALS_FILE)
            if not df.empty: return df
        except Exception:
            pass
    return None

def load_cached_history():
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            pass
    return None

def get_available_dates(history_df):
    if history_df is None or history_df.empty or "date" not in history_df.columns:
        return []
    return sorted(history_df["date"].dropna().unique().tolist(), reverse=True)

# =============================================
# LOGIC & TIERS
# =============================================
def classify_tier(close, vol):
    if close >= TIER_LARGE_PRICE and vol >= TIER_LARGE_VOLUME: return "🏛️ Large Cap"
    elif close >= TIER_MID_PRICE and vol >= TIER_MID_VOLUME: return "🏢 Mid Cap"
    elif close >= TIER_SMALL_PRICE and vol >= TIER_SMALL_VOLUME: return "🏠 Small Cap"
    return "⚠️ Penny"

def compute_volume_ma(history, window=MA_WINDOW):
    if history is None or history.empty or "volume" not in history.columns: return {}
    return history.groupby("symbol")["volume"].apply(lambda s: s.tail(window).mean()).to_dict()

def calculate_whale_score(dp, vr, pcp):
    d_norm = min(1.0, max(0.0, (dp - 50) / 40))
    v_norm = min(1.0, max(0.0, (vr - 2) / 6))
    p_norm = min(1.0, max(0.0, (abs(pcp) - 3) / 7))
    bonus = (d_norm * 0.40 + v_norm * 0.35 + p_norm * 0.25) * 70
    return round(30 + bonus, 1)

def detect_whale_moves(today_df, vol_ma):
    if today_df is None or today_df.empty: return pd.DataFrame()
    df = today_df.copy()
    df["price_change_pct"] = ((df["close"] - df["prev_close"]) / df["prev_close"] * 100).round(2)
    df["abs_price_change"] = df["price_change_pct"].abs()
    df["vol_20ma"] = df["symbol"].map(vol_ma)
    df["volume_ratio"] = np.where(df["vol_20ma"] > 0, (df["volume"] / df["vol_20ma"]).round(2), np.nan)
    if "delivery_pct" not in df.columns: df["delivery_pct"] = 0.0
    
    mask = (df["delivery_pct"] >= MIN_DELIVERY_PCT) & (df["volume_ratio"] >= MIN_VOLUME_RATIO) & (df["abs_price_change"] >= MIN_PRICE_CHANGE_PCT)
    whales = df[mask].copy()
    if whales.empty: return whales
    
    whales["whale_score"] = whales.apply(lambda r: calculate_whale_score(r["delivery_pct"], r["volume_ratio"], r["abs_price_change"]), axis=1)
    whales["avg_volume"] = whales["symbol"].map(vol_ma).fillna(whales["volume"])
    whales["tier"] = whales.apply(lambda r: classify_tier(r.get("close",0), r.get("avg_volume",0)), axis=1)
    return whales.sort_values("whale_score", ascending=False).reset_index(drop=True)

# =============================================
# UI
# =============================================
def render_header():
    st.markdown("""
        <div class="hero-header">
            <h1>🐋 WhaleWatch AI</h1>
            <p><span class="live-dot"></span>Tracking Institutional Activity in NSE 500</p>
        </div>
    """, unsafe_allow_html=True)

def render_fii_dii_cards(fii_dii):
    st.markdown("### 🏦 Institutional Flows (FII / DII)")
    if not fii_dii:
        st.caption("FII/DII data not available.")
        return
    fii, dii = fii_dii.get("fii", {}), fii_dii.get("dii", {})
    cols = st.columns(4)
    for i, (label, data) in enumerate([("FII / FPI", fii), ("DII / Domestic", dii)]):
        net = data.get("net", 0)
        cls = "fii-buy" if net >= 0 else "fii-sell"
        clr = "positive" if net >= 0 else "negative"
        txt = "Net Buy 📈" if net >= 0 else "Net Sell 📉"
        with cols[i]:
            st.markdown(f'<div class="fii-dii-card {cls}"><div class="fii-label">{label}</div><div class="fii-value {clr}">{"+" if net>=0 else ""}₹{abs(net):,.0f} Cr</div><div class="fii-sub">{txt}</div></div>', unsafe_allow_html=True)
    with cols[2]: st.metric("FII Gross Buy", f"₹{fii.get('buy',0):,.0f} Cr", f"Sell: ₹{fii.get('sell',0):,.0f} Cr")
    with cols[3]: st.metric("DII Gross Buy", f"₹{dii.get('buy',0):,.0f} Cr", f"Sell: ₹{dii.get('sell',0):,.0f} Cr")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_header()
    
    with st.sidebar:
        st.header("📅 Data Source")
        history = load_cached_history()
        dates = get_available_dates(history)
        sel_date = st.selectbox("Select Date", ["Live (Today)"] + dates)
        st.markdown("---")
        tier_filter = st.multiselect("Market Cap Tier", ["🏛️ Large Cap", "🏢 Mid Cap", "🏠 Small Cap", "⚠️ Penny"], default=["🏛️ Large Cap", "🏢 Mid Cap"])

    today = date.today()
    if sel_date == "Live (Today)":
        with st.spinner("📡 Scanning NSE live..."):
            raw_df = fetch_live_data(today)
            fii_dii = fetch_fii_dii_data()
            bulk = fetch_bulk_deals()
            disp_date = today.strftime("%Y-%m-%d")
    else:
        disp_date = sel_date
        raw_df = history[history["date"] == sel_date].copy() if history is not None else None
        fii_dii, bulk = None, None

    if raw_df is None or raw_df.empty:
        st.error("❌ No data available.")
        st.stop()
        
    st.caption(f"Showing data for: {disp_date}")
    
    # Process
    vol_ma = compute_volume_ma(history) if history is not None else {}
    if "whale_score" not in raw_df.columns:
        whales = detect_whale_moves(raw_df, vol_ma)
    else:
        whales = raw_df.copy() # History already has it
        if "tier" not in whales.columns:
             whales["tier"] = whales.apply(lambda r: classify_tier(r["close"], r.get("volume",0)), axis=1)

    filtered = whales[whales["tier"].isin(tier_filter)]
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Whales Detected", len(filtered))
    m2.metric("Bullish", len(filtered[filtered["price_change_pct"] > 0]))
    m3.metric("Bearish", len(filtered[filtered["price_change_pct"] < 0]))
    
    tab1, tab2, tab3 = st.tabs(["🐋 Whale Moves", "🏦 FII/DII", "📦 Bulk Deals"])
    
    with tab1:
        if filtered.empty:
            st.warning("No whales found.")
        else:
            st.dataframe(filtered[["symbol", "tier", "close", "price_change_pct", "volume_ratio", "delivery_pct", "whale_score"]].style.background_gradient(subset=["whale_score"], cmap="viridis"), use_container_width=True)

    with tab2: render_fii_dii_cards(fii_dii)
    with tab3: 
        if bulk is not None and not bulk.empty: st.dataframe(bulk, use_container_width=True)
        else: st.info("No bulk deals found.")

    st.markdown('<div class="app-footer">Built with ❤️ by WhaleWatch AI</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
