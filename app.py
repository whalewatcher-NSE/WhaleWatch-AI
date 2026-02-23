"""
🐋 WhaleWatch AI — Institutional Activity Tracker for NSE 500
=============================================================
Tracks "Whale Moves" (high delivery %, abnormal volume, big price swings)
to spot potential institutional (FII/DII) activity in Indian stocks.
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
HISTORY_FILE = os.path.join(DATA_DIR, "whale_history.csv")
# Whale Score thresholds
MIN_DELIVERY_PCT = 50.0
MIN_VOLUME_RATIO = 2.0
MIN_PRICE_CHANGE_PCT = 3.0
MA_WINDOW = 20
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
/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
/* ---- Hide Streamlit branding ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/* ---- Hero header gradient ---- */
.hero-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(0, 212, 170, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.hero-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00D4AA, #00B4D8, #00D4AA);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    margin-bottom: 0.3rem;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.hero-header p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
}
/* ---- Metric cards ---- */
.metric-card {
    background: linear-gradient(145deg, #1a1f2e, #141820);
    border: 1px solid rgba(0, 212, 170, 0.1);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0, 212, 170, 0.1);
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #00D4AA;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.4rem;
}
/* ---- Whale badge ---- */
.whale-badge {
    display: inline-block;
    background: linear-gradient(135deg, #00D4AA, #00B4D8);
    color: #0E1117;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 2px 10px;
    border-radius: 20px;
}
/* ---- Score bar ---- */
.score-bar-bg {
    background: #1e2433;
    border-radius: 6px;
    overflow: hidden;
    height: 8px;
}
.score-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}
/* ---- Data table styling ---- */
.dataframe {
    border: none !important;
}
.dataframe th {
    background: #1a1f2e !important;
    color: #94a3b8 !important;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.5px;
}
.dataframe td {
    border-color: #1e2433 !important;
    font-size: 0.88rem;
}
/* ---- Status badges ---- */
.bullish { color: #22c55e; font-weight: 600; }
.bearish { color: #ef4444; font-weight: 600; }
/* ---- Section divider ---- */
.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,170,0.3), transparent);
    margin: 2rem 0;
}
/* ---- Search result card ---- */
.stock-card {
    background: linear-gradient(145deg, #1a1f2e, #141820);
    border: 1px solid rgba(0, 212, 170, 0.12);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
/* ---- Mobile responsive ---- */
@media (max-width: 768px) {
    .hero-header { padding: 1.2rem; }
    .hero-header h1 { font-size: 1.6rem; }
    .metric-value { font-size: 1.5rem; }
}
/* ---- Pulse animation for live dot ---- */
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(34,197,94,0); }
}
/* ---- Footer ---- */
.app-footer {
    text-align: center;
    color: #475569;
    font-size: 0.75rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid #1e2433;
    margin-top: 3rem;
}
</style>
"""
# =============================================
# DATA LAYER
# =============================================
def _resolve_column(df: pd.DataFrame, canonical: str) -> str | None:
    """Find the actual column name in a DataFrame given our alias list."""
    aliases = COLUMN_ALIASES.get(canonical, [])
    for alias in aliases:
        # Strip whitespace from DataFrame columns for matching
        for col in df.columns:
            if col.strip() == alias.strip():
                return col
    return None
def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    """Normalise column names from any nselib version to our standard schema."""
    if df is None or df.empty:
        return None
    rename_map = {}
    required = ["symbol", "close", "prev_close", "volume", "delivery_pct"]
    for canonical in COLUMN_ALIASES:
        actual = _resolve_column(df, canonical)
        if actual and actual != canonical:
            rename_map[actual] = canonical
    df = df.rename(columns=rename_map)
    # Check we have the critical columns
    for col in required:
        if col not in df.columns:
            return None
    # Filter equity series only
    if "series" in df.columns:
        df = df[df["series"].str.strip() == "EQ"].copy()
    # Coerce numerics
    for col in ["open", "high", "low", "close", "prev_close",
                 "volume", "delivery_qty", "delivery_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["symbol", "close", "volume"])
    return df
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_data(date_obj: date) -> pd.DataFrame | None:
    """Fetch bhavcopy + delivery data from NSE for a given date.
    Returns a normalised DataFrame or None on failure."""
    date_str = date_obj.strftime("%d-%m-%Y")
    # --- Attempt 1: nselib bhav_copy_with_delivery ---
    try:
        from nselib import capital_market
        raw = capital_market.bhav_copy_with_delivery(date_str)
        df = _normalize_dataframe(raw)
        if df is not None and len(df) > 0:
            df["date"] = date_obj.strftime("%Y-%m-%d")
            return df
    except Exception:
        pass
    # --- Attempt 2: nselib bhav_copy_equities ---
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
def get_last_n_trading_dates(n: int = 30) -> list[date]:
    """Return the last N weekday dates (approximate trading days)."""
    dates = []
    current = date.today()
    while len(dates) < n:
        current -= timedelta(days=1)
        if current.weekday() < 5:  # Mon–Fri
            dates.append(current)
    return dates
def load_cached_history() -> pd.DataFrame | None:
    """Load the master whale_history.csv from the data directory."""
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            if not df.empty:
                return df
        except Exception:
            pass
    return None
def build_history_from_daily_csvs() -> pd.DataFrame | None:
    """Fallback: reconstruct history from individual daily CSV files."""
    csv_dir = os.path.join(DATA_DIR, "daily")
    if not os.path.isdir(csv_dir):
        return None
    all_files = sorted(
        [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    )
    if not all_files:
        return None
    frames = []
    for f in all_files[-60:]:  # Last 60 files max
        try:
            frames.append(pd.read_csv(os.path.join(csv_dir, f)))
        except Exception:
            continue
    if frames:
        return pd.concat(frames, ignore_index=True)
    return None
# =============================================
# WHALE SCORE ENGINE
# =============================================
def compute_volume_ma(history: pd.DataFrame, window: int = MA_WINDOW) -> dict:
    """Compute the rolling volume MA per symbol from historical data.
    Returns {symbol: ma_volume}."""
    if history is None or history.empty or "volume" not in history.columns:
        return {}
    grouped = history.groupby("symbol")["volume"].apply(
        lambda s: s.tail(window).mean()
    )
    return grouped.to_dict()
def calculate_whale_score(
    delivery_pct: float,
    volume_ratio: float,
    price_change_pct: float,
) -> float:
    """Compute a weighted Whale Score (30–100).
    30 = barely meets thresholds, 100 = extreme conviction."""
    delivery_norm = min(1.0, max(0.0, (delivery_pct - 50) / 40))
    volume_norm = min(1.0, max(0.0, (volume_ratio - 2) / 6))
    price_norm = min(1.0, max(0.0, (abs(price_change_pct) - 3) / 7))
    bonus = (delivery_norm * 0.40 + volume_norm * 0.35 + price_norm * 0.25) * 70
    return round(30 + bonus, 1)
def detect_whale_moves(today_df: pd.DataFrame, vol_ma: dict) -> pd.DataFrame:
    """Flag whale moves in today's data and compute scores."""
    if today_df is None or today_df.empty:
        return pd.DataFrame()
    df = today_df.copy()
    # Price change %
    df["price_change_pct"] = (
        (df["close"] - df["prev_close"]) / df["prev_close"] * 100
    ).round(2)
    df["abs_price_change"] = df["price_change_pct"].abs()
    # Volume ratio vs 20-day MA
    df["vol_20ma"] = df["symbol"].map(vol_ma)
    df["volume_ratio"] = np.where(
        df["vol_20ma"] > 0,
        (df["volume"] / df["vol_20ma"]).round(2),
        np.nan,
    )
    # Ensure delivery_pct is present
    if "delivery_pct" not in df.columns:
        df["delivery_pct"] = 0.0
    # Apply Whale Move filter
    whale_mask = (
        (df["delivery_pct"] >= MIN_DELIVERY_PCT)
        & (df["volume_ratio"] >= MIN_VOLUME_RATIO)
        & (df["abs_price_change"] >= MIN_PRICE_CHANGE_PCT)
    )
    whales = df[whale_mask].copy()
    if whales.empty:
        return whales
    # Compute weighted Whale Score
    whales["whale_score"] = whales.apply(
        lambda r: calculate_whale_score(
            r["delivery_pct"], r["volume_ratio"], r["abs_price_change"]
        ),
        axis=1,
    )
    # Signal direction
    whales["signal"] = np.where(
        whales["price_change_pct"] > 0, "🟢 Bullish", "🔴 Bearish"
    )
    # Sort by score descending
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
def render_metrics(whale_df: pd.DataFrame, total_stocks: int, data_date: str):
    """Render the top KPI cards."""
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
                f"""
                <div class="metric-card">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown(
        f"""
        <p style="text-align:center; color:#64748b; font-size:0.8rem; margin-top:0.8rem;">
            Scanned <strong style="color:#94a3b8;">{total_stocks:,}</strong> stocks
            on <strong style="color:#94a3b8;">{data_date}</strong>
        </p>
        """,
        unsafe_allow_html=True,
    )
def _score_color(score: float) -> str:
    if score >= 75:
        return "#22c55e"
    elif score >= 50:
        return "#00D4AA"
    elif score >= 35:
        return "#eab308"
    else:
        return "#64748b"
def render_whale_table(whale_df: pd.DataFrame):
    """Render the main whale moves table."""
    if whale_df.empty:
        st.info("🔍 No whale moves detected for this date. The market may be calm or data hasn't been fetched yet.")
        return
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 🏆 Top Whale Moves")
    display_cols = [
        "symbol", "close", "price_change_pct", "delivery_pct",
        "volume_ratio", "whale_score", "signal",
    ]
    available = [c for c in display_cols if c in whale_df.columns]
    display_df = whale_df[available].copy()
    # Rename for display
    rename = {
        "symbol": "Stock",
        "close": "Close ₹",
        "price_change_pct": "Change %",
        "delivery_pct": "Delivery %",
        "volume_ratio": "Vol Ratio",
        "whale_score": "Score",
        "signal": "Signal",
    }
    display_df = display_df.rename(columns=rename)
    # Style the dataframe
    def style_row(row):
        styles = [""] * len(row)
        if "Signal" in row.index:
            if "Bullish" in str(row.get("Signal", "")):
                styles = ["color: #22c55e;"] * len(row)
            else:
                styles = ["color: #ef4444;"] * len(row)
        return styles
    styled = (
        display_df.style
        .apply(style_row, axis=1)
        .format({
            "Close ₹": "₹{:,.2f}",
            "Change %": "{:+.2f}%",
            "Delivery %": "{:.1f}%",
            "Vol Ratio": "{:.1f}x",
            "Score": "{:.0f}",
        }, na_rep="—")
        .set_properties(**{
            "text-align": "center",
            "font-size": "0.9rem",
        })
        .set_properties(subset=["Stock"], **{"text-align": "left", "font-weight": "600"})
    )
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(len(display_df) * 40 + 50, 600),
    )
def render_stock_search(history_df: pd.DataFrame, whale_df: pd.DataFrame):
    """Render the stock search section."""
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 🔎 Stock Lookup")
    if history_df is None or history_df.empty:
        st.warning("Historical data not yet available. Run the data pipeline first.")
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
    if selected:
        stock_hist = history_df[history_df["symbol"] == selected].copy()
        if stock_hist.empty:
            st.info(f"No historical data for {selected}")
            return
        stock_hist = stock_hist.sort_values("date", ascending=False).head(30)
        st.markdown(
            f"""
            <div class="stock-card">
                <h3 style="margin:0; color:#00D4AA;">{selected}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Key stats
        latest = stock_hist.iloc[0] if len(stock_hist) > 0 else None
        if latest is not None:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                close_val = latest.get("close", 0)
                st.metric("Close", f"₹{close_val:,.2f}")
            with c2:
                del_pct = latest.get("delivery_pct", 0)
                st.metric("Delivery %", f"{del_pct:.1f}%")
            with c3:
                vol = latest.get("volume", 0)
                st.metric("Volume", f"{vol:,.0f}")
            with c4:
                is_whale = selected in whale_df["symbol"].values if not whale_df.empty else False
                st.metric("Whale Today?", "🐋 YES" if is_whale else "—  No")
        # Historical table
        with st.expander(f"📊 Last {len(stock_hist)} days — {selected}", expanded=True):
            hist_display = stock_hist[
                [c for c in ["date", "close", "volume", "delivery_pct"] if c in stock_hist.columns]
            ].copy()
            hist_display.columns = [c.replace("_", " ").title() for c in hist_display.columns]
            st.dataframe(hist_display, use_container_width=True, hide_index=True)
def render_sidebar(data_date: str):
    """Render the sidebar with info and thresholds."""
    with st.sidebar:
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
        st.markdown("## 📅 Data Date")
        st.markdown(f"**{data_date}**")
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
        st.markdown("---")
        st.markdown(
            '<p style="color:#475569;font-size:0.7rem;text-align:center;">'
            "Built with ❤️ using Streamlit</p>",
            unsafe_allow_html=True,
        )
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
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    # Header
    render_header()
    # -------------------------------------------
    # 1. Load data
    # -------------------------------------------
    today = date.today()
    data_date_str = today.strftime("%Y-%m-%d")
    # Try cached history first (populated by GitHub Actions)
    history_df = load_cached_history()
    if history_df is None:
        history_df = build_history_from_daily_csvs()
    # Try fetching live data for today
    today_df = None
    with st.spinner("🔄 Fetching latest NSE data..."):
        # Try today first, then yesterday (in case market is closed)
        for attempt_date in [today, today - timedelta(days=1)]:
            if attempt_date.weekday() >= 5:
                continue
            today_df = fetch_live_data(attempt_date)
            if today_df is not None and not today_df.empty:
                data_date_str = attempt_date.strftime("%Y-%m-%d")
                break
    # If live fetch failed, try to extract latest day from history
    if (today_df is None or today_df.empty) and history_df is not None and not history_df.empty:
        latest_date = history_df["date"].max()
        today_df = history_df[history_df["date"] == latest_date].copy()
        data_date_str = str(latest_date)
    # -------------------------------------------
    # 2. Compute Whale Scores
    # -------------------------------------------
    vol_ma = {}
    if history_df is not None and not history_df.empty:
        vol_ma = compute_volume_ma(history_df)
    elif today_df is not None and not today_df.empty:
        # Without history, use today's volume as a rough baseline (not ideal)
        vol_ma = today_df.set_index("symbol")["volume"].to_dict()
    whale_df = pd.DataFrame()
    total_stocks = 0
    if today_df is not None and not today_df.empty:
        total_stocks = len(today_df)
        whale_df = detect_whale_moves(today_df, vol_ma)
    # -------------------------------------------
    # 3. Render UI
    # -------------------------------------------
    render_sidebar(data_date_str)
    render_metrics(whale_df, total_stocks, data_date_str)
    render_whale_table(whale_df)
    render_stock_search(history_df, whale_df)
    render_footer()
if __name__ == "__main__":
    main()
