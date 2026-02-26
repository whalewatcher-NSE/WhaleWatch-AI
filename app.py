"""
🐋 WhaleWatch AI — Institutional Activity Tracker
=================================================
v2: Added F&O-filtered bulk deals, actionable alerts panel,
    improved investor classification, alert history.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import os
import json
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
FNO_LIST_FILE = os.path.join(DATA_DIR, "fno_stocks.json")
ALERTS_FILE = os.path.join(DATA_DIR, "alerts_today.json")

NSE_HEADERS = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'accept-language': 'en-US,en;q=0.9',
    'referer': 'https://www.nseindia.com/'
}

# =============================================
# CSS (PREMIUM DARK UI)
# =============================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

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

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px; padding: 1rem;
    text-align: center; transition: all 0.2s;
}
div[data-testid="stMetric"]:hover { border-color: rgba(0,212,170,0.3); transform: translateY(-2px); }

.fii-box {
    padding: 1rem; border-radius: 8px; text-align: center; color: white;
    border: 1px solid rgba(255,255,255,0.1); margin-bottom: 0.5rem;
}
.bg-buy { background: rgba(34, 197, 94, 0.15); border-color: rgba(34, 197, 94, 0.3); }
.bg-sell { background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.3); }
.val-text { font-size: 1.4rem; font-weight: 700; }

div.stButton > button {
    width: 100%; border-radius: 10px; height: 5rem;
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid rgba(255,255,255,0.1); color: white;
}
div.stButton > button:hover { border-color: #00D4AA; color: #00D4AA; }
div.stButton > button:focus { border-color: #00D4AA; box-shadow: 0 0 10px rgba(0,212,170,0.2); }

.badge { padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.b-large { background: rgba(34,197,94,0.2); color: #4ade80; border: 1px solid #22c55e; }
.b-mid { background: rgba(59,130,246,0.2); color: #60a5fa; border: 1px solid #3b82f6; }
.b-small { background: rgba(234,179,8,0.2); color: #facc15; border: 1px solid #eab308; }
.b-penny { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid #ef4444; }

.alert-card {
    background: linear-gradient(145deg, #1a2332, #0f172a);
    border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
    border-left: 4px solid;
}
.alert-buy { border-left-color: #22c55e; }
.alert-sell { border-left-color: #ef4444; }
.alert-symbol { font-size: 1.2rem; font-weight: 700; color: #00D4AA; }
.alert-detail { color: #94a3b8; font-size: 0.85rem; margin-top: 0.3rem; }
.fno-badge {
    background: rgba(168, 85, 247, 0.2); color: #c084fc;
    border: 1px solid #a855f7; padding: 2px 6px; border-radius: 4px;
    font-size: 0.7rem; font-weight: 600;
}
</style>
"""

# =============================================
# DATA ENGINE
# =============================================
def get_nse_session():
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    try: s.get("https://www.nseindia.com", timeout=2)
    except: pass
    return s


def load_fno_list():
    """Load cached F&O stock list."""
    if os.path.exists(FNO_LIST_FILE):
        try:
            with open(FNO_LIST_FILE) as f:
                data = json.load(f)
            return set(data.get("symbols", []))
        except:
            pass
    return set()


def load_alerts():
    """Load today's alerts."""
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE) as f:
                data = json.load(f)
            return data.get("alerts", [])
        except:
            pass
    return []


@st.cache_data(ttl=3600)
def fetch_fii_dii():
    try:
        s = get_nse_session()
        r = s.get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=5)
        data = r.json()
        out = {}
        for i in data:
            cat = i.get("category","").upper()
            val = float(i.get("buyValue",0)) - float(i.get("sellValue",0))
            if "FII" in cat or "FPI" in cat: out["fii"] = val
            elif "DII" in cat: out["dii"] = val
        return out
    except:
        if os.path.exists(FII_DII_FILE):
            try:
                df = pd.read_csv(FII_DII_FILE)
                if not df.empty:
                    last = df.iloc[-1]
                    return {"fii": float(last.get("fii_net",0)), "dii": float(last.get("dii_net",0))}
            except: pass
    return None


@st.cache_data(ttl=3600)
def fetch_bulk():
    # Try live first
    try:
        s = get_nse_session()
        r = s.get("https://www.nseindia.com/api/snapshot-capital-market-largedeal", timeout=5)
        d = r.json()
        frames = []
        if "BULK_DEALS_DATA" in d: frames.append(pd.DataFrame(d["BULK_DEALS_DATA"]))
        if "BLOCK_DEALS_DATA" in d: frames.append(pd.DataFrame(d["BLOCK_DEALS_DATA"]))
        if frames: return pd.concat(frames, ignore_index=True)
    except: pass

    # Fallback to saved CSV
    if os.path.exists(BULK_DEALS_FILE):
        try: return pd.read_csv(BULK_DEALS_FILE)
        except: pass
    return None


def load_data():
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except: pass
    if os.path.exists(DAILY_DIR):
        files = sorted([f for f in os.listdir(DAILY_DIR) if f.endswith(".csv")])
        if files:
            dfs = [pd.read_csv(os.path.join(DAILY_DIR, f)) for f in files[-10:]]
            return pd.concat(dfs, ignore_index=True) if dfs else None
    return None


# =============================================
# LOGIC & CLASSIFICATION
# =============================================
def classify_market_cap(row):
    turnover = row["close"] * row["volume"]
    price = row["close"]
    if turnover >= 500_000_000: return "Large Cap"
    elif turnover >= 100_000_000: return "Mid Cap"
    elif turnover >= 10_000_000: return "Small Cap"
    else:
        if price < 20 or turnover < 5_000_000: return "Penny"
        return "Small Cap"


# Investor classification (same lists as fetch_data_v2.py)
PROP_ALGO_KEYWORDS = [
    "NK SECURITIES RESEARCH", "JUNOMONETA FINSOL",
    "HRTI PRIVATE LIMITED", "QE SECURITIES",
    "IRAGE BROKING", "GRAVITON RESEARCH",
    "JUMP TRADING", "TOWER RESEARCH",
    "VIRTU FINANCIAL", "CITADEL SECURITIES",
    "ALPHAGREP", "QUADEYE", "OPTIVER",
    "PACE STOCK BROKING", "PACE COMMODITY",
    "CLT RESEARCH", "SILVERLEAF CAPITAL",
    "SHARE INDIA SECURITIES", "MICROCURVES TRADING",
    "MUSIGMA SECURITIES", "GRT STRATEGIC",
    "ELIXIR WEALTH", "QICAP MARKETS",
    "MATHISYS ADVISORS",
]

FII_KEYWORDS = [
    "MORGAN STANLEY", "GOLDMAN SACHS", "CITIGROUP", "MERRILL LYNCH",
    "SOCIETE GENERALE", "VANGUARD", "BLACKROCK", "BNP PARIBAS",
    "JPMORGAN", "JP MORGAN", "CREDIT SUISSE", "UBS ", "HSBC",
    "NOMURA", "BARCLAYS", "DEUTSCHE BANK", "CLSA", "MACQUARIE",
    "ABERDEEN", "TEMPLETON", "FIDELITY INTERNATIONAL",
    "INVESCO", "SCHRODERS", "CARLYLE", "KKR", "WARBURG",
    "GIC PRIVATE", "TEMASEK", "QATAR INVESTMENT",
    "ABU DHABI INVESTMENT", "ADIA", "NORGES BANK",
    "CANADA PENSION", "ONTARIO TEACHERS",
    "PLC", "LLC", "LP ", "L.P.", "GMBH", "B.V.", "PTE LTD",
    "FUND PCC", "VCC",
]

DII_KEYWORDS = [
    "MUTUAL FUND", "SBI LIFE", "SBI MUTUAL", "HDFC MUTUAL", "HDFC LIFE",
    "ICICI PRUDENTIAL", "ICICI LOMBARD", "AXIS MUTUAL", "KOTAK MUTUAL",
    "KOTAK MAHINDRA LIFE", "ADITYA BIRLA", "NIPPON INDIA",
    "UTI MUTUAL", "DSP MUTUAL", "TATA MUTUAL", "FRANKLIN TEMPLETON MUTUAL",
    "LIFE INSURANCE CORPORATION", "LIC OF INDIA", "LIC MF",
    "GENERAL INSURANCE", "NEW INDIA ASSURANCE",
    "NATIONAL INSURANCE", "UNITED INDIA INSURANCE",
    "EMPLOYEES PROVIDENT", "EPFO", "NPS TRUST",
    "IDFC MUTUAL", "MOTILAL OSWAL MUTUAL", "MIRAE ASSET",
    "EDELWEISS MUTUAL", "PGIM INDIA", "CANARA ROBECO",
    "SUNDARAM MUTUAL", "QUANTUM MUTUAL", "PPFAS MUTUAL",
    "BANDHAN MUTUAL", "BARODA BNP", "HSBC MUTUAL",
    "UNION MUTUAL", "MAHINDRA MANULIFE",
]


def tag_investor_type(client_name):
    """Improved investor classification."""
    name = str(client_name).upper().strip()

    for kw in PROP_ALGO_KEYWORDS:
        if kw in name:
            return "Prop/Algo"
    for kw in FII_KEYWORDS:
        if kw in name:
            return "FII"
    for kw in DII_KEYWORDS:
        if kw in name:
            return "DII"

    corp_suffixes = ["LIMITED", "LTD", "LLP", "PVT", "PRIVATE", "SECURITIES", "CAPITAL", "FUND"]
    if not any(s in name for s in corp_suffixes):
        return "Promoter/HNI"
    return "Other"


def process_whales(df, history):
    if df is None or df.empty: return pd.DataFrame()

    vol_ma = {}
    if history is not None and not history.empty:
        vol_ma = history.groupby("symbol")["volume"].apply(lambda x: x.tail(20).mean()).to_dict()

    df = df.copy()
    df["vol_avg"] = df["symbol"].map(vol_ma).fillna(df["volume"])
    df["vol_ratio"] = (df["volume"] / df["vol_avg"]).round(1)
    df["turnover_cr"] = ((df["close"] * df["volume"]) / 10000000).round(2)
    df["Category"] = df.apply(classify_market_cap, axis=1)

    mask = (df["delivery_pct"] >= 50) & (df["vol_ratio"] >= 2.0) & (df["price_change_pct"].abs() >= 3.0)
    whales = df[mask].copy()

    if whales.empty: return pd.DataFrame()

    whales["Score"] = (
        ((whales["delivery_pct"]-50)/40 * 0.4) +
        ((whales["vol_ratio"]-2)/6 * 0.35) +
        ((whales["price_change_pct"].abs()-3)/7 * 0.25)
    ) * 100
    whales["Score"] = whales["Score"].clip(40, 99).astype(int)
    whales["Signal"] = np.where(whales["price_change_pct"] > 0, "Bullish", "Bearish")

    # Add F&O flag
    fno_stocks = load_fno_list()
    if fno_stocks:
        whales["is_fno"] = whales["symbol"].isin(fno_stocks)

    return whales.sort_values("Score", ascending=False)


# =============================================
# MAIN INTERFACE
# =============================================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "trend_filter" not in st.session_state: st.session_state.trend_filter = "All"
    if "cat_filter" not in st.session_state: st.session_state.cat_filter = ["Large Cap", "Mid Cap", "Small Cap"]

    st.markdown(f"""
        <div class="hero-header">
            <h1>{APP_ICON} {APP_TITLE}</h1>
            <p>Institutional Tracking & Dark Pool Scanner</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Load Data ---
    history = load_data()
    fii_data = fetch_fii_dii()
    bulk_data = fetch_bulk()
    fno_stocks = load_fno_list()
    alerts = load_alerts()

    dates = sorted(history["date"].unique(), reverse=True) if history is not None else []

    with st.sidebar:
        st.header("⚙️ Scanner Settings")
        sel_date = st.selectbox("📅 Analysis Date", dates, index=0) if dates else str(date.today())

        st.markdown("---")
        st.subheader("🔎 Filters")
        cats = st.multiselect(
            "Market Cap Class",
            ["Large Cap", "Mid Cap", "Small Cap", "Penny"],
            default=st.session_state.cat_filter,
            key="cat_multiselect"
        )
        st.session_state.cat_filter = cats

        # F&O filter toggle
        fno_only = st.checkbox("F&O stocks only", value=False, help="Show only F&O eligible stocks")

        if fno_stocks:
            st.caption(f"📊 {len(fno_stocks)} F&O stocks loaded")

    # --- Main Analysis ---
    if history is not None and sel_date in dates:
        day_data = history[history["date"] == sel_date].copy()
        whales = process_whales(day_data, history)
    else:
        whales = pd.DataFrame()

    # --- Top Metrics ---
    c1, c2, c3, c4 = st.columns(4)
    total_count = len(whales) if not whales.empty else 0
    c1.metric("Whales Detected", total_count)

    bull_count = len(whales[whales["Signal"]=="Bullish"]) if not whales.empty else 0
    bear_count = len(whales[whales["Signal"]=="Bearish"]) if not whales.empty else 0

    with c2:
        if st.button(f"🟢 Bullish ({bull_count})", use_container_width=True):
            st.session_state.trend_filter = "Bullish"
    with c3:
        if st.button(f"🔴 Bearish ({bear_count})", use_container_width=True):
            st.session_state.trend_filter = "Bearish"
    with c4:
        if st.button("🔄 Reset Filters", use_container_width=True):
            st.session_state.trend_filter = "All"

    # --- FII / DII Banner ---
    st.markdown("### 🏦 Institutional Money Flow")
    if fii_data:
        fc1, fc2 = st.columns(2)
        f_val = fii_data.get("fii", 0)
        d_val = fii_data.get("dii", 0)

        with fc1:
            cls = "bg-buy" if f_val > 0 else "bg-sell"
            st.markdown(f'<div class="fii-box {cls}"><div>FII NET FLOW</div><div class="val-text">₹ {f_val:,.0f} Cr</div></div>', unsafe_allow_html=True)
        with fc2:
            cls = "bg-buy" if d_val > 0 else "bg-sell"
            st.markdown(f'<div class="fii-box {cls}"><div>DII NET FLOW</div><div class="val-text">₹ {d_val:,.0f} Cr</div></div>', unsafe_allow_html=True)

    # --- Alerts Panel (NEW) ---
    if alerts:
        st.markdown("### 🔔 Actionable Alerts — F&O Bulk/Block Deals")
        for alert in alerts:
            side_cls = "alert-buy" if alert.get("side", "").upper() == "BUY" else "alert-sell"
            side_emoji = "🟢 BOUGHT" if alert.get("side", "").upper() == "BUY" else "🔴 SOLD"
            inv_type = alert.get("investor_type", "Unknown")
            type_color = "#22c55e" if inv_type == "DII" else "#3b82f6" if inv_type == "FII" else "#94a3b8"

            st.markdown(f"""
                <div class="alert-card {side_cls}">
                    <span class="alert-symbol">{alert.get('symbol', '?')}</span>
                    <span class="fno-badge">F&O</span>
                    <span style="color: {type_color}; font-weight: 600; margin-left: 8px;">{inv_type}</span>
                    <div class="alert-detail">
                        {alert.get('investor', 'Unknown')} {side_emoji}
                        <strong>{alert.get('qty', 0):,}</strong> shares @ ₹{alert.get('price', 0):,.2f}
                        — Deal value: <strong>₹{alert.get('value_cr', 0):.1f} Cr</strong>
                        ({alert.get('deal_type', '?')})
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("")

    # --- Tabs ---
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "🐋 Whale Radar",
        "📦 F&O Bulk Deals",
        "📦 All Bulk Deals",
        "📈 Stock Lookup"
    ])

    # --- TAB 1: WHALE RADAR ---
    with tab1:
        if whales.empty:
            st.info("No whale activity detected for this date.")
        else:
            filtered_whales = whales[whales["Category"].isin(st.session_state.cat_filter)]
            if st.session_state.trend_filter != "All":
                filtered_whales = filtered_whales[filtered_whales["Signal"] == st.session_state.trend_filter]
            if fno_only and "is_fno" in filtered_whales.columns:
                filtered_whales = filtered_whales[filtered_whales["is_fno"] == True]

            if filtered_whales.empty:
                st.warning(f"No stocks found for current filters")
            else:
                display_cols = [
                    "symbol", "Category", "close", "price_change_pct",
                    "delivery_pct", "vol_ratio", "turnover_cr", "Score", "Signal"
                ]
                if "is_fno" in filtered_whales.columns:
                    display_cols.insert(1, "is_fno")

                st.dataframe(
                    filtered_whales[display_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "symbol": "Stock",
                        "is_fno": st.column_config.CheckboxColumn("F&O", default=False),
                        "Category": "Class",
                        "close": st.column_config.NumberColumn("Price", format="₹%.2f"),
                        "price_change_pct": st.column_config.NumberColumn("Change %", format="%.2f%%"),
                        "delivery_pct": st.column_config.ProgressColumn("Delivery %", min_value=0, max_value=100, format="%.1f%%"),
                        "vol_ratio": st.column_config.NumberColumn("Vol Spike", format="%.1fx"),
                        "turnover_cr": st.column_config.NumberColumn("Turnover (Cr)", format="₹%.1f"),
                        "Score": st.column_config.ProgressColumn("Whale Score", min_value=0, max_value=100),
                    }
                )

    # --- TAB 2: F&O BULK DEALS (NEW — the key addition) ---
    with tab2:
        if bulk_data is not None and not bulk_data.empty and fno_stocks:
            bd = bulk_data.copy()
            col_map = {c: c.strip() for c in bd.columns}
            bd.rename(columns=col_map, inplace=True)

            # Find symbol column
            sym_col = next((c for c in bd.columns if c.lower() == "symbol"), None)
            client_col = next((c for c in bd.columns if "client" in c.lower()), None)

            if sym_col:
                # Filter to F&O stocks only
                bd_fno = bd[bd[sym_col].isin(fno_stocks)].copy()

                if client_col:
                    bd_fno["Investor Type"] = bd_fno[client_col].apply(tag_investor_type)

                if bd_fno.empty:
                    st.info("No bulk/block deals in F&O stocks today. This is normal — most bulk deals are in small/micro caps.")
                else:
                    st.success(f"Found {len(bd_fno)} deals in {bd_fno[sym_col].nunique()} F&O stocks")

                    # Filter by investor type
                    if client_col:
                        itypes = st.multiselect(
                            "Filter Investor Type:",
                            ["FII", "DII", "Prop/Algo", "Promoter/HNI", "Other"],
                            default=["FII", "DII", "Promoter/HNI", "Other"],
                            key="fno_investor_filter"
                        )
                        bd_fno = bd_fno[bd_fno["Investor Type"].isin(itypes)]

                    st.dataframe(bd_fno, use_container_width=True, hide_index=True)
            else:
                st.warning("Could not identify symbol column in bulk deals data")
        elif not fno_stocks:
            st.warning("F&O stock list not available. Run `python fetch_data.py` to generate it.")
        else:
            st.info("No bulk deals data available today.")

    # --- TAB 3: ALL BULK DEALS ---
    with tab3:
        if bulk_data is not None and not bulk_data.empty:
            bd = bulk_data.copy()
            col_map = {c: c.strip() for c in bd.columns}
            bd.rename(columns=col_map, inplace=True)

            client_col = next((c for c in bd.columns if "client" in c.lower()), None)

            if client_col:
                bd["Investor Type"] = bd[client_col].apply(tag_investor_type)
                itypes = st.multiselect(
                    "Filter Investor Type:",
                    ["FII", "DII", "Prop/Algo", "Promoter/HNI", "Other"],
                    default=["FII", "DII", "Promoter/HNI"],
                    key="all_investor_filter"
                )
                filtered_bd = bd[bd["Investor Type"].isin(itypes)]
                st.dataframe(filtered_bd, use_container_width=True, hide_index=True)
            else:
                st.dataframe(bd, use_container_width=True)
        else:
            st.info("No bulk deals data available today.")

    # --- TAB 4: SEARCH ---
    with tab4:
        if history is not None:
            all_syms = sorted(history["symbol"].unique())
            search = st.selectbox("Search Stock:", [""] + all_syms)
            if search:
                stock_data = history[history["symbol"] == search].sort_values("date", ascending=False).head(10)
                if not stock_data.empty:
                    latest = stock_data.iloc[0]
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("Price", f"₹{latest['close']}")
                    sc2.metric("Delivery", f"{latest['delivery_pct']}%")
                    sc3.metric("Volume", f"{latest['volume']:,}")

                    # Show F&O badge
                    if fno_stocks and search in fno_stocks:
                        sc4.metric("F&O", "✅ Eligible")
                    else:
                        sc4.metric("F&O", "❌ Not in F&O")

                    st.line_chart(stock_data.set_index("date")["close"])
                    st.dataframe(stock_data, use_container_width=True, hide_index=True)

    st.caption("WhaleWatch AI v2.1 • F&O Filtered • Institutional Classification")


if __name__ == "__main__":
    main()
