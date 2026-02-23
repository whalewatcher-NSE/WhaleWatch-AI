"""
🐋 WhaleWatch AI — Institutional Activity Tracker for NSE 500
=============================================================
Tracks "Whale Moves" (high delivery %, abnormal volume, big price swings)
to spot potential institutional (FII/DII) activity in Indian stocks.
Enhanced with:
  • Data freshness indicator
  • Stock quality tiers (Large / Mid / Small Cap vs Penny)
  • FII/DII aggregate flows + Bulk/Block deal tracking
  • Historical date selector
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os
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
# Whale Score thresholds
MIN_DELIVERY_PCT = 50.0
MIN_VOLUME_RATIO = 2.0
MIN_PRICE_CHANGE_PCT = 3.0
MA_WINDOW = 20
# Stock quality tier thresholds
TIER_LARGE_PRICE = 500
TIER_LARGE_VOLUME = 500_000
TIER_MID_PRICE = 100
TIER_MID_VOLUME = 100_000
TIER_SMALL_PRICE = 20
TIER_SMALL_VOLUME = 25_000
# Column name mapping (handles nselib version differences)
COLUMN_ALIASES = {
    "symbol": ["SYMBOL", "Symbol", "TckrSymb", "TCKR_SYMB"],
    "series": ["SERIES", "Series", "SrNm", "SERIES "],
    "open": ["OPEN_PRICE", "OPEN", "OpnPric", "OPEN_PRICE "],
    "high": ["HIGH_PRICE", "HIGH", "HghPric", "HIGH_PRICE "],
    "low": ["LOW_PRICE", "LOW", "LwPric", "LOW_PRICE "],
    "close": ["CLOSE_PRICE", "CLOSE", "ClsPric", "CLOSE_PRICE "],
    "prev_close": [
        "PREV_CLOSE", "PREVCLOSE", "PrvsClsgPric",
        "PREV_CLOSE ", "PREV_CLS_PRICE",
    ],
    "volume": [
        "TTL_TRD_QNTY", "TOTTRDQTY", "TtlTradgVol",
        "TOTAL_TRADE_QUANTITY", "TTL_TRD_QNTY ",
    ],
    "delivery_qty": [
        "DELIV_QTY", "DlvryQty", "DELIVERY_QTY", "DELIV_QTY ",
    ],
    "delivery_pct": [
        "DELIV_PER", "DlvryPct", "DELIVERY_PCT",
        "DELIV_PER ", " DELIV_PER",
    ],
    "date": ["DATE1", "DATE", "TIMESTAMP", "TradDt", "Date"],
}
# =============================================
# CUSTOM CSS — Premium Dark Theme
# =============================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/* ---- Hero header ---- */
.hero-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(0, 212, 170, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
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
/* ---- Freshness banner ---- */
.freshness-banner {
    border-radius: 10px; padding: 0.7rem 1.2rem;
    margin-bottom: 1.2rem; display: flex;
    align-items: center; gap: 10px; font-size: 0.88rem;
}
.freshness-live {
    background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.05));
    border: 1px solid rgba(34,197,94,0.25); color: #22c55e;
}
.freshness-stale {
    background: linear-gradient(135deg, rgba(234,179,8,0.1), rgba(234,179,8,0.05));
    border: 1px solid rgba(234,179,8,0.25); color: #eab308;
}
.freshness-empty {
    background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.25); color: #ef4444;
}
/* ---- FII/DII cards ---- */
.fii-dii-card {
    border-radius: 12px; padding: 1rem 1.3rem; text-align: center;
    border: 1px solid rgba(0,212,170,0.1);
}
.fii-buy { background: linear-gradient(145deg, #0a2e1a, #0e1a12); border-color: rgba(34,197,94,0.2); }
.fii-sell { background: linear-gradient(145deg, #2e0a0a, #1a0e0e); border-color: rgba(239,68,68,0.2); }
.fii-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.fii-value { font-size: 1.5rem; font-weight: 800; margin: 0.3rem 0; }
.fii-value.positive { color: #22c55e; }
.fii-value.negative { color: #ef4444; }
.fii-sub { font-size: 0.75rem; color: #475569; }
/* ---- Metric cards ---- */
.metric-card {
    background: linear-gradient(145deg, #1a1f2e, #141820);
    border: 1px solid rgba(0, 212, 170, 0.1);
    border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,212,170,0.1); }
.metric-value { font-size: 2rem; font-weight: 800; color: #00D4AA; line-height: 1; }
.metric-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.4rem; }
/* ---- Tier badges ---- */
.tier-badge {
    display: inline-block; font-weight: 600; font-size: 0.65rem;
    padding: 1px 8px; border-radius: 12px; margin-left: 6px;
    vertical-align: middle;
}
.tier-large { background: #1a3a2a; color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.tier-mid { background: #1a2a3a; color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
.tier-small { background: #2a2a1a; color: #eab308; border: 1px solid rgba(234,179,8,0.3); }
.tier-penny { background: #2a1a1a; color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
/* ---- Confirmed badge ---- */
.confirmed-badge {
    display: inline-block; background: linear-gradient(135deg, #f59e0b, #d97706);
    color: #0E1117; font-weight: 700; font-size: 0.7rem;
    padding: 2px 10px; border-radius: 12px;
}
/* ---- Section divider ---- */
.section-divider {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,170,0.3), transparent);
    margin: 1.5rem 0;
}
/* ---- Stock card ---- */
.stock-card {
    background: linear-gradient(145deg, #1a1f2e, #141820);
    border: 1px solid rgba(0,212,170,0.12);
    border-radius: 14px; padding: 1.5rem; margin-bottom: 1rem;
}
/* ---- Live pulse ---- */
.live-dot {
    display: inline-block; width: 8px; height: 8px;
    background: #22c55e; border-radius: 50%; margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; box-shadow:0 0 0 0 rgba(34,197,94,0.4); }
    50% { opacity:0.7; box-shadow:0 0 0 6px rgba(34,197,94,0); }
}
/* ---- Footer ---- */
.app-footer {
    text-align: center; color: #475569; font-size: 0.75rem;
    padding: 2rem 0 1rem; border-top: 1px solid #1e2433; margin-top: 3rem;
}
/* ---- Mobile ---- */
@media (max-width: 768px) {
    .hero-header { padding: 1.2rem; }
    .hero-header h1 { font-size: 1.6rem; }
    .metric-value { font-size: 1.5rem; }
    .fii-value { font-size: 1.2rem; }
}
</style>
"""
# =============================================
# DATA LAYER
# =============================================
def _resolve_column(df: pd.DataFrame, canonical: str) -> str | None:
    aliases = COLUMN_ALIASES.get(canonical, [])
    for alias in aliases:
        for col in df.columns:
            if col.strip() == alias.strip():
                return col
    return None
def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    rename_map = {}
    for canonical in COLUMN_ALIASES:
        actual = _resolve_column(df, canonical)
        if actual and actual != canonical:
            rename_map[actual] = canonical
    df = df.rename(columns=rename_map)
    required = ["symbol", "close", "prev_close", "volume"]
    for col in required:
        if col not in df.columns:
            return None
    if "series" in df.columns:
        df = df[df["series"].str.strip() == "EQ"].copy()
    for col in ["open", "high", "low", "close", "prev_close",
                 "volume", "delivery_qty", "delivery_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["symbol", "close", "volume"])
    return df
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_data(date_obj: date) -> pd.DataFrame | None:
    date_str = date_obj.strftime("%d-%m-%Y")
    try:
        from nselib import capital_market
        raw = capital_market.bhav_copy_with_delivery(date_str)
        df = _normalize_dataframe(raw)
        if df is not None and len(df) > 0:
            df["date"] = date_obj.strftime("%Y-%m-%d")
            return df
    except Exception:
        pass
    try:
        from nselib import capital_market
        raw = capital_market.bhav_copy_equities(date_str)
        df = _normalize_dataframe(raw)
        if df is not None and len(df) > 0:
            df["date"] = date_obj.strftime("%Y-%m-%d")
            return df
    except Exception:
        pass
    return None
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fii_dii_data() -> dict | None:
    """Fetch aggregate FII/DII trading activity from NSE."""
    try:
        from nselib import capital_market
        raw = capital_market.fii_dii_trading_activity()
        if raw is not None and not raw.empty:
            result = {}
            for _, row in raw.iterrows():
                category = str(row.get("category", row.get("Category", ""))).strip()
                buy = pd.to_numeric(row.get("buyValue", row.get("BUY_VALUE", row.get("buy_value", 0))), errors="coerce") or 0
                sell = pd.to_numeric(row.get("sellValue", row.get("SELL_VALUE", row.get("sell_value", 0))), errors="coerce") or 0
                net = buy - sell
                if "FII" in category.upper() or "FPI" in category.upper():
                    result["fii"] = {"buy": buy, "sell": sell, "net": net}
                elif "DII" in category.upper():
                    result["dii"] = {"buy": buy, "sell": sell, "net": net}
            if result:
                return result
    except Exception:
        pass
    # Fallback: try loading from cached file
    if os.path.exists(FII_DII_FILE):
        try:
            df = pd.read_csv(FII_DII_FILE)
            if not df.empty:
                latest = df.iloc[-1]
                return {
                    "fii": {
                        "buy": float(latest.get("fii_buy", 0)),
                        "sell": float(latest.get("fii_sell", 0)),
                        "net": float(latest.get("fii_net", 0)),
                    },
                    "dii": {
                        "buy": float(latest.get("dii_buy", 0)),
                        "sell": float(latest.get("dii_sell", 0)),
                        "net": float(latest.get("dii_net", 0)),
                    },
                }
        except Exception:
            pass
    return None
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bulk_deals() -> pd.DataFrame | None:
    """Fetch today's bulk/block deals from NSE."""
    try:
        from nselib import capital_market
        raw = capital_market.bulk_deal_data()
        if raw is not None and not raw.empty:
            return raw
    except Exception:
        pass
    try:
        from nselib import capital_market
        raw = capital_market.block_deal_data()
        if raw is not None and not raw.empty:
            return raw
    except Exception:
        pass
    # Fallback: cached file
    if os.path.exists(BULK_DEALS_FILE):
        try:
            df = pd.read_csv(BULK_DEALS_FILE)
            if not df.empty:
                return df
        except Exception:
            pass
    return None
def load_cached_history() -> pd.DataFrame | None:
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            if not df.empty:
                return df
        except Exception:
            pass
    return None
def build_history_from_daily_csvs() -> pd.DataFrame | None:
    if not os.path.isdir(DAILY_DIR):
        return None
    all_files = sorted([f for f in os.listdir(DAILY_DIR) if f.endswith(".csv")])
    if not all_files:
        return None
    frames = []
    for f in all_files[-60:]:
        try:
            frames.append(pd.read_csv(os.path.join(DAILY_DIR, f)))
        except Exception:
            continue
    if frames:
        return pd.concat(frames, ignore_index=True)
    return None
def get_available_dates(history_df: pd.DataFrame | None) -> list[str]:
    """Get list of available dates from history, sorted newest first."""
    if history_df is None or history_df.empty or "date" not in history_df.columns:
        return []
    dates = sorted(history_df["date"].dropna().unique().tolist(), reverse=True)
    return dates
# =============================================
# STOCK QUALITY TIER ENGINE
# =============================================
def classify_tier(close_price: float, avg_volume: float) -> str:
    """Classify a stock into a quality tier based on price and liquidity."""
    if close_price >= TIER_LARGE_PRICE and avg_volume >= TIER_LARGE_VOLUME:
        return "🏛️ Large Cap"
    elif close_price >= TIER_MID_PRICE and avg_volume >= TIER_MID_VOLUME:
        return "🏢 Mid Cap"
    elif close_price >= TIER_SMALL_PRICE and avg_volume >= TIER_SMALL_VOLUME:
        return "🏠 Small Cap"
    else:
        return "⚠️ Penny"
def add_tier_to_dataframe(df: pd.DataFrame, vol_ma: dict) -> pd.DataFrame:
    """Add the 'tier' column to a DataFrame."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df["avg_volume"] = df["symbol"].map(vol_ma).fillna(df["volume"])
    df["tier"] = df.apply(
        lambda r: classify_tier(r.get("close", 0), r.get("avg_volume", 0)),
        axis=1,
    )
    return df
# =============================================
# WHALE SCORE ENGINE
# =============================================
def compute_volume_ma(history: pd.DataFrame, window: int = MA_WINDOW) -> dict:
    if history is None or history.empty or "volume" not in history.columns:
        return {}
    grouped = history.groupby("symbol")["volume"].apply(
        lambda s: s.tail(window).mean()
    )
    return grouped.to_dict()
def calculate_whale_score(delivery_pct: float, volume_ratio: float, price_change_pct: float) -> float:
    delivery_norm = min(1.0, max(0.0, (delivery_pct - 50) / 40))
    volume_norm = min(1.0, max(0.0, (volume_ratio - 2) / 6))
    price_norm = min(1.0, max(0.0, (abs(price_change_pct) - 3) / 7))
    bonus = (delivery_norm * 0.40 + volume_norm * 0.35 + price_norm * 0.25) * 70
    return round(30 + bonus, 1)
def detect_whale_moves(today_df: pd.DataFrame, vol_ma: dict) -> pd.DataFrame:
    if today_df is None or today_df.empty:
        return pd.DataFrame()
    df = today_df.copy()
    df["price_change_pct"] = (
        (df["close"] - df["prev_close"]) / df["prev_close"] * 100
    ).round(2)
    df["abs_price_change"] = df["price_change_pct"].abs()
    df["vol_20ma"] = df["symbol"].map(vol_ma)
    df["volume_ratio"] = np.where(
        df["vol_20ma"] > 0, (df["volume"] / df["vol_20ma"]).round(2), np.nan,
    )
    if "delivery_pct" not in df.columns:
        df["delivery_pct"] = 0.0
    whale_mask = (
        (df["delivery_pct"] >= MIN_DELIVERY_PCT)
        & (df["volume_ratio"] >= MIN_VOLUME_RATIO)
        & (df["abs_price_change"] >= MIN_PRICE_CHANGE_PCT)
    )
    whales = df[whale_mask].copy()
    if whales.empty:
        return whales
    whales["whale_score"] = whales.apply(
        lambda r: calculate_whale_score(
            r["delivery_pct"], r["volume_ratio"], r["abs_price_change"]
        ), axis=1,
    )
    whales["signal"] = np.where(
        whales["price_change_pct"] > 0, "🟢 Bullish", "🔴 Bearish"
    )
    whales = whales.sort_values("whale_score", ascending=False).reset_index(drop=True)
    return whales
# =============================================
# UI COMPONENTS
# =============================================
def render_header():
    st.markdown(
        """
        <div class="hero-header">
            <h1>🐋 WhaleWatch AI</h1>
            <p>
                <span class="live-dot"></span>
                Tracking Institutional Activity in NSE 500 &mdash;
                Powered by Delivery &amp; Volume Analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
def render_data_freshness(data_date_str: str):
    """Render a freshness banner showing how recent the data is."""
    today = date.today()
    now_ist = datetime.now()
    if data_date_str and data_date_str != "N/A":
        try:
            data_date = datetime.strptime(data_date_str, "%Y-%m-%d").date()
            days_old = (today - data_date).days
            if days_old == 0:
                css_class = "freshness-live"
                icon = "🟢"
                msg = f"Live data as of <strong>{data_date.strftime('%d %b %Y (%A)')}</strong>"
            elif days_old <= 3:
                css_class = "freshness-stale"
                icon = "🟡"
                msg = (
                    f"Last data: <strong>{data_date.strftime('%d %b %Y (%A)')}</strong>"
                    f" &mdash; {days_old} day{'s' if days_old > 1 else ''} ago"
                )
            else:
                css_class = "freshness-empty"
                icon = "🔴"
                msg = (
                    f"Stale data: <strong>{data_date.strftime('%d %b %Y')}</strong>"
                    f" &mdash; {days_old} days ago"
                )
            # Next update timing
            weekday = today.weekday()
            if weekday < 4:  # Mon-Thu → tonight or tomorrow
                next_update = "Tonight at 7:00 PM IST"
            elif weekday == 4:  # Friday
                if now_ist.hour < 19:
                    next_update = "Tonight at 7:00 PM IST"
                else:
                    next_update = "Monday at 7:00 PM IST"
            else:  # Weekend
                next_update = "Monday at 7:00 PM IST"
            st.markdown(
                f'<div class="freshness-banner {css_class}">'
                f'<span style="font-size:1.3rem;">{icon}</span>'
                f'<span>{msg} &nbsp;|&nbsp; Next update: {next_update}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            return
        except ValueError:
            pass
    # No data at all
    st.markdown(
        '<div class="freshness-banner freshness-empty">'
        '<span style="font-size:1.3rem;">🔴</span>'
        '<span><strong>No data available yet.</strong> '
        'Run the GitHub Action to fetch NSE data, or wait for the 7 PM IST auto-fetch.</span>'
        '</div>',
        unsafe_allow_html=True,
    )
def render_fii_dii_cards(fii_dii: dict | None):
    """Render FII/DII aggregate flow cards."""
    st.markdown("### 🏦 Institutional Flows (FII / DII)")
    if fii_dii is None:
        st.caption("FII/DII data not available — will appear after the next data fetch.")
        return
    fii = fii_dii.get("fii", {})
    dii = fii_dii.get("dii", {})
    cols = st.columns(4)
    # FII Net
    fii_net = fii.get("net", 0)
    fii_class = "fii-buy" if fii_net >= 0 else "fii-sell"
    fii_color = "positive" if fii_net >= 0 else "negative"
    fii_sign = "+" if fii_net >= 0 else ""
    fii_label_text = "Net Buy 📈" if fii_net >= 0 else "Net Sell 📉"
    with cols[0]:
        st.markdown(
            f'<div class="fii-dii-card {fii_class}">'
            f'<div class="fii-label">FII / FPI</div>'
            f'<div class="fii-value {fii_color}">{fii_sign}₹{abs(fii_net):,.0f} Cr</div>'
            f'<div class="fii-sub">{fii_label_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f'<div class="fii-dii-card" style="background:linear-gradient(145deg,#1a1f2e,#141820);">'
            f'<div class="fii-label">FII Buy</div>'
            f'<div class="fii-value positive">₹{fii.get("buy", 0):,.0f} Cr</div>'
            f'<div class="fii-sub">FII Sell: ₹{fii.get("sell", 0):,.0f} Cr</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    # DII Net
    dii_net = dii.get("net", 0)
    dii_class = "fii-buy" if dii_net >= 0 else "fii-sell"
    dii_color = "positive" if dii_net >= 0 else "negative"
    dii_sign = "+" if dii_net >= 0 else ""
    dii_label_text = "Net Buy 📈" if dii_net >= 0 else "Net Sell 📉"
    with cols[2]:
        st.markdown(
            f'<div class="fii-dii-card {dii_class}">'
            f'<div class="fii-label">DII</div>'
            f'<div class="fii-value {dii_color}">{dii_sign}₹{abs(dii_net):,.0f} Cr</div>'
            f'<div class="fii-sub">{dii_label_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            f'<div class="fii-dii-card" style="background:linear-gradient(145deg,#1a1f2e,#141820);">'
            f'<div class="fii-label">DII Buy</div>'
            f'<div class="fii-value positive">₹{dii.get("buy", 0):,.0f} Cr</div>'
            f'<div class="fii-sub">DII Sell: ₹{dii.get("sell", 0):,.0f} Cr</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
def render_metrics(whale_df: pd.DataFrame, total_stocks: int, data_date: str):
    n_whales = len(whale_df)
    n_bull = len(whale_df[whale_df["price_change_pct"] > 0]) if n_whales else 0
    n_bear = n_whales - n_bull
    avg_score = whale_df["whale_score"].mean() if n_whales else 0
    cols = st.columns(4)
    metrics = [
        ("🐋", str(n_whales), "Whale Moves"),
        ("📈", str(n_bull), "Bullish"),
        ("📉", str(n_bear), "Bearish"),
        ("🎯", f"{avg_score:.0f}", "Avg Score"),
    ]
    for col, (icon, val, label) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size:1.5rem;">{icon}</div>'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        f'<p style="text-align:center;color:#64748b;font-size:0.8rem;margin-top:0.6rem;">'
        f'Scanned <strong style="color:#94a3b8;">{total_stocks:,}</strong> stocks'
        f'</p>',
        unsafe_allow_html=True,
    )
def _tier_badge_html(tier: str) -> str:
    """Return HTML for a tier badge."""
    if "Large" in tier:
        return '<span class="tier-badge tier-large">LARGE CAP</span>'
    elif "Mid" in tier:
        return '<span class="tier-badge tier-mid">MID CAP</span>'
    elif "Small" in tier:
        return '<span class="tier-badge tier-small">SMALL CAP</span>'
    else:
        return '<span class="tier-badge tier-penny">PENNY</span>'
def render_whale_table(whale_df: pd.DataFrame, bulk_symbols: set):
    """Render the main whale moves table with tier filter."""
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 🏆 Top Whale Moves")
    if whale_df.empty:
        st.info(
            "🔍 No whale moves detected for this date. "
            "The market may be calm, it could be a holiday, "
            "or data hasn't been fetched yet."
        )
        return
    # Tier filter tabs
    tiers_present = whale_df["tier"].unique().tolist() if "tier" in whale_df.columns else []
    filter_options = ["All (excl. Penny)"] + [t for t in ["🏛️ Large Cap", "🏢 Mid Cap", "🏠 Small Cap"] if t in tiers_present]
    if "⚠️ Penny" in tiers_present:
        filter_options.append("⚠️ Penny (Risky)")
    selected_tier = st.radio(
        "Filter by stock quality:",
        filter_options,
        horizontal=True,
        key="tier_filter",
    )
    # Apply filter
    filtered = whale_df.copy()
    if selected_tier == "All (excl. Penny)":
        filtered = filtered[filtered["tier"] != "⚠️ Penny"]
    elif "Penny" in selected_tier:
        filtered = filtered[filtered["tier"] == "⚠️ Penny"]
    else:
        filtered = filtered[filtered["tier"] == selected_tier]
    if filtered.empty:
        st.info(f"No whale moves in the **{selected_tier}** category for this date.")
        return
    # Add confirmed institutional badge
    if bulk_symbols:
        filtered["institutional"] = filtered["symbol"].apply(
            lambda s: "🏦 Confirmed" if s in bulk_symbols else ""
        )
    # Build display columns
    display_cols = ["symbol", "tier", "close", "price_change_pct", "delivery_pct",
                    "volume_ratio", "whale_score", "signal"]
    if bulk_symbols:
        display_cols.append("institutional")
    available = [c for c in display_cols if c in filtered.columns]
    display_df = filtered[available].copy()
    rename = {
        "symbol": "Stock",
        "tier": "Tier",
        "close": "Close ₹",
        "price_change_pct": "Change %",
        "delivery_pct": "Delivery %",
        "volume_ratio": "Vol Ratio",
        "whale_score": "Score",
        "signal": "Signal",
        "institutional": "Inst. Deal",
    }
    display_df = display_df.rename(columns=rename)
    def style_row(row):
        styles = [""] * len(row)
        if "Signal" in row.index:
            if "Bullish" in str(row.get("Signal", "")):
                styles = ["color: #22c55e;"] * len(row)
            else:
                styles = ["color: #ef4444;"] * len(row)
        return styles
    format_dict = {
        "Close ₹": "₹{:,.2f}",
        "Change %": "{:+.2f}%",
        "Delivery %": "{:.1f}%",
        "Vol Ratio": "{:.1f}x",
        "Score": "{:.0f}",
    }
    styled = (
        display_df.style
        .apply(style_row, axis=1)
        .format(format_dict, na_rep="—")
        .set_properties(**{"text-align": "center", "font-size": "0.9rem"})
        .set_properties(subset=["Stock"], **{"text-align": "left", "font-weight": "600"})
    )
    st.dataframe(
        styled, use_container_width=True, hide_index=True,
        height=min(len(display_df) * 40 + 50, 600),
    )
    st.caption(
        f"Showing **{len(filtered)}** whale moves"
        + (f" · **{len([s for s in filtered['symbol'] if s in bulk_symbols])}** with confirmed bulk/block deals" if bulk_symbols else "")
    )
def render_bulk_deals(bulk_df: pd.DataFrame | None):
    """Render bulk/block deals section."""
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 📋 Bulk & Block Deals")
    if bulk_df is None or bulk_df.empty:
        st.caption("No bulk/block deal data available for today.")
        return
    with st.expander(f"View {len(bulk_df)} bulk/block deals", expanded=False):
        st.dataframe(bulk_df, use_container_width=True, hide_index=True, height=300)
def render_stock_search(history_df: pd.DataFrame, whale_df: pd.DataFrame):
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 🔎 Stock Lookup")
    if history_df is None or history_df.empty:
        st.warning(
            "📭 Historical data not yet available. "
            "Go to your GitHub repo → **Actions** tab → run the "
            "**🐋 Daily NSE Data Fetch** workflow with bootstrap enabled."
        )
        return
    all_symbols = sorted(history_df["symbol"].dropna().unique().tolist())
    search = st.text_input(
        "Search for a stock symbol",
        placeholder="e.g. RELIANCE, TCS, HDFCBANK ...",
        key="stock_search",
    )
    if not search:
        return
    query = search.strip().upper()
    matches = [s for s in all_symbols if query in s]
    if not matches:
        st.warning(f"No stocks found matching **{query}**")
        return
    selected = st.selectbox("Select stock:", matches, key="stock_select")
    if not selected:
        return
    stock_hist = history_df[history_df["symbol"] == selected].copy()
    if stock_hist.empty:
        st.info(f"No historical data for {selected}")
        return
    stock_hist = stock_hist.sort_values("date", ascending=False).head(30)
    # Tier badge
    tier = stock_hist.iloc[0].get("tier", "")
    tier_html = _tier_badge_html(tier) if tier else ""
    st.markdown(
        f'<div class="stock-card">'
        f'<h3 style="margin:0;color:#00D4AA;">{selected} {tier_html}</h3>'
        f'</div>',
        unsafe_allow_html=True,
    )
    latest = stock_hist.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Close", f"₹{latest.get('close', 0):,.2f}")
    with c2:
        st.metric("Delivery %", f"{latest.get('delivery_pct', 0):.1f}%")
    with c3:
        st.metric("Volume", f"{latest.get('volume', 0):,.0f}")
    with c4:
        is_whale = selected in whale_df["symbol"].values if not whale_df.empty else False
        st.metric("Whale Today?", "🐋 YES" if is_whale else "— No")
    with st.expander(f"📊 Last {len(stock_hist)} days — {selected}", expanded=True):
        hist_display = stock_hist[
            [c for c in ["date", "close", "volume", "delivery_pct"] if c in stock_hist.columns]
        ].copy()
        hist_display.columns = [c.replace("_", " ").title() for c in hist_display.columns]
        st.dataframe(hist_display, use_container_width=True, hide_index=True)
def render_sidebar(data_date: str, available_dates: list[str]):
    """Render sidebar with date selector, algorithm info, and about."""
    with st.sidebar:
        st.markdown("## 📅 Select Date")
        if available_dates:
            selected = st.selectbox(
                "View whale moves for:",
                available_dates,
                index=0,
                key="date_selector",
                format_func=lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%d %b %Y (%a)"),
            )
        else:
            selected = data_date
            st.caption("No historical dates available yet.")
        st.markdown("---")
        st.markdown("## ⚙️ Algorithm")
        st.markdown(
            f"""
            A stock is flagged as a **Whale Move** when
            **all three** conditions are met:
            | Rule | Threshold |
            |------|-----------|
            | Delivery % | ≥ **{MIN_DELIVERY_PCT}%** |
            | Volume vs 20d MA | ≥ **{MIN_VOLUME_RATIO}x** |
            | Price Change | ≥ **{MIN_PRICE_CHANGE_PCT}%** |
            """
        )
        st.markdown("---")
        st.markdown("## 📊 Stock Tiers")
        st.markdown(
            f"""
            | Tier | Price | Volume |
            |------|-------|--------|
            | 🏛️ Large | ≥ ₹{TIER_LARGE_PRICE} | ≥ {TIER_LARGE_VOLUME//1000}K |
            | 🏢 Mid | ≥ ₹{TIER_MID_PRICE} | ≥ {TIER_MID_VOLUME//1000}K |
            | 🏠 Small | ≥ ₹{TIER_SMALL_PRICE} | ≥ {TIER_SMALL_VOLUME//1000}K |
            | ⚠️ Penny | Below thresholds |  |
            """
        )
        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.markdown(
            """
            **WhaleWatch AI** analyses daily NSE delivery
            and volume data to surface stocks where
            institutional players may be active.
            *Not financial advice. For educational purposes only.*
            """
        )
        st.markdown(
            '<p style="color:#475569;font-size:0.7rem;text-align:center;">'
            "Built with ❤️ using Streamlit</p>",
            unsafe_allow_html=True,
        )
    return selected
def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            <strong>Disclaimer:</strong> WhaleWatch AI is an educational tool.
            It does not constitute financial advice. Always do your own research
            before making investment decisions.<br>
            Data sourced from NSE India &bull; Updated daily at 7 PM IST
        </div>
        """,
        unsafe_allow_html=True,
    )
# =============================================
# MAIN APP
# =============================================
def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_header()
    # -------------------------------------------
    # 1. Load historical data
    # -------------------------------------------
    history_df = load_cached_history()
    if history_df is None:
        history_df = build_history_from_daily_csvs()
    available_dates = get_available_dates(history_df)
    # -------------------------------------------
    # 2. Sidebar — date selector
    # -------------------------------------------
    today = date.today()
    default_date_str = today.strftime("%Y-%m-%d")
    selected_date_str = render_sidebar(default_date_str, available_dates)
    # -------------------------------------------
    # 3. Load data for selected date
    # -------------------------------------------
    today_df = None
    data_date_str = selected_date_str
    # Try loading from history for the selected date
    if history_df is not None and not history_df.empty and selected_date_str in history_df["date"].values:
        today_df = history_df[history_df["date"] == selected_date_str].copy()
    else:
        # Try live fetch
        with st.spinner("🔄 Fetching latest NSE data..."):
            for attempt_date in [today, today - timedelta(days=1), today - timedelta(days=2)]:
                if attempt_date.weekday() >= 5:
                    continue
                today_df = fetch_live_data(attempt_date)
                if today_df is not None and not today_df.empty:
                    data_date_str = attempt_date.strftime("%Y-%m-%d")
                    break
        # Last resort: use most recent from history
        if (today_df is None or today_df.empty) and available_dates:
            data_date_str = available_dates[0]
            today_df = history_df[history_df["date"] == data_date_str].copy()
    # -------------------------------------------
    # 4. Data freshness banner
    # -------------------------------------------
    render_data_freshness(data_date_str if today_df is not None and not today_df.empty else "N/A")
    # -------------------------------------------
    # 5. FII/DII flows
    # -------------------------------------------
    fii_dii = fetch_fii_dii_data()
    render_fii_dii_cards(fii_dii)
    # -------------------------------------------
    # 6. Compute whale scores + tiers
    # -------------------------------------------
    vol_ma = {}
    if history_df is not None and not history_df.empty:
        vol_ma = compute_volume_ma(history_df)
    elif today_df is not None and not today_df.empty:
        vol_ma = today_df.set_index("symbol")["volume"].to_dict()
    total_stocks = 0
    whale_df = pd.DataFrame()
    if today_df is not None and not today_df.empty:
        total_stocks = len(today_df)
        whale_df = detect_whale_moves(today_df, vol_ma)
        if not whale_df.empty:
            whale_df = add_tier_to_dataframe(whale_df, vol_ma)
    # -------------------------------------------
    # 7. Bulk/block deals
    # -------------------------------------------
    bulk_df = fetch_bulk_deals()
    bulk_symbols = set()
    if bulk_df is not None and not bulk_df.empty:
        # Try to extract symbols from bulk deal data
        for col in ["SYMBOL", "Symbol", "symbol", "SCRIP_NAME"]:
            if col in bulk_df.columns:
                bulk_symbols = set(bulk_df[col].dropna().str.strip().unique())
                break
    # -------------------------------------------
    # 8. Render dashboard
    # -------------------------------------------
    render_metrics(whale_df, total_stocks, data_date_str)
    render_whale_table(whale_df, bulk_symbols)
    render_bulk_deals(bulk_df)
    render_stock_search(history_df, whale_df)
    render_footer()
if __name__ == "__main__":
    main()
