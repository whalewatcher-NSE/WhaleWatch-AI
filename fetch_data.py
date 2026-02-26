"""
fetch_data.py — Standalone NSE Data Fetcher for GitHub Actions
==============================================================
v2: Added F&O filter for bulk deals, append-mode for bulk deals,
    improved investor classification, and actionable alert output.
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
import time
import argparse
import requests
import json

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
HISTORY_FILE = os.path.join(DATA_DIR, "whale_history.csv")
FII_DII_FILE = os.path.join(DATA_DIR, "fii_dii.csv")
BULK_DEALS_FILE = os.path.join(DATA_DIR, "bulk_deals.csv")
FNO_LIST_FILE = os.path.join(DATA_DIR, "fno_stocks.json")
ALERTS_FILE = os.path.join(DATA_DIR, "alerts_today.json")

# ---- Config ----
MAX_HISTORY_DAYS = 60

# Headers to mimic a browser (Crucial for NSE)
NSE_HEADERS = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'accept-language': 'en-US,en;q=0.9',
    'referer': 'https://www.nseindia.com/'
}

# Column aliases
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
}

# ========================================================
# Known investor classification (curated lists)
# ========================================================
# Major FII custodians and known foreign entities
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
    # Legal entity suffixes common in foreign entities
    "PLC", "LLC", "LP ", "L.P.", "GMBH", "B.V.", "PTE LTD",
    "FUND PCC", "VCC",
]

# Known DII names (mutual funds, insurance, domestic institutions)
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

# Known prop/algo trading desks (these dominate bulk deal data — not institutional signal)
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


def get_nse_session():
    """Create a session and initialize cookies from the homepage."""
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    return s


def resolve_column(df, canonical):
    aliases = COLUMN_ALIASES.get(canonical, [])
    for alias in aliases:
        for col in df.columns:
            if col.strip() == alias.strip():
                return col
    return None


def normalize_dataframe(df):
    if df is None or df.empty:
        return None
    rename_map = {}
    for canonical in COLUMN_ALIASES:
        actual = resolve_column(df, canonical)
        if actual and actual != canonical:
            rename_map[actual] = canonical
    df = df.rename(columns=rename_map)
    required = ["symbol", "close", "prev_close", "volume"]
    for col in required:
        if col not in df.columns:
            return None
    if "series" in df.columns:
        df = df[df["series"].str.strip() == "EQ"].copy()
    for col in ["open", "high", "low", "close", "prev_close", "volume", "delivery_qty", "delivery_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["symbol", "close", "volume"])
    return df


# ========================================================
# F&O Stock List
# ========================================================
def fetch_fno_list():
    """Fetch current F&O eligible stocks from NSE. Returns set of symbols."""
    print("  📡 Fetching F&O stock list...")

    # Try live API first
    try:
        session = get_nse_session()
        url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        r = session.get(url, timeout=10)
        data = r.json()
        symbols = [item["symbol"] for item in data.get("data", [])]
        if symbols:
            # Cache to file
            with open(FNO_LIST_FILE, "w") as f:
                json.dump({"updated": date.today().isoformat(), "symbols": symbols}, f)
            print(f"  ✅ Got {len(symbols)} F&O stocks (live)")
            return set(symbols)
    except Exception as e:
        print(f"  ⚠ Live F&O fetch failed: {e}")

    # Fallback to cached file
    if os.path.exists(FNO_LIST_FILE):
        try:
            with open(FNO_LIST_FILE) as f:
                cached = json.load(f)
            symbols = cached.get("symbols", [])
            print(f"  ℹ Using cached F&O list ({len(symbols)} stocks, from {cached.get('updated', '?')})")
            return set(symbols)
        except Exception:
            pass

    print("  ❌ No F&O list available")
    return set()


# ========================================================
# Investor Classification
# ========================================================
def classify_investor(client_name):
    """
    Classify a bulk deal client into: FII, DII, Prop/Algo, or Promoter/Other.
    Returns (type_label, confidence) tuple.
    """
    name = str(client_name).upper().strip()

    # Check prop/algo first (they dominate bulk deals and are NOT institutional signal)
    for kw in PROP_ALGO_KEYWORDS:
        if kw in name:
            return "Prop/Algo"

    # Check FII
    for kw in FII_KEYWORDS:
        if kw in name:
            return "FII"

    # Check DII
    for kw in DII_KEYWORDS:
        if kw in name:
            return "DII"

    # Heuristic: Individual names (no LTD/PVT/LLP) are likely promoter/HNI
    corp_suffixes = ["LIMITED", "LTD", "LLP", "PVT", "PRIVATE", "SECURITIES", "CAPITAL", "FUND"]
    is_corporate = any(s in name for s in corp_suffixes)

    if not is_corporate:
        return "Promoter/HNI"

    return "Other"


# ========================================================
# Data Fetch Functions
# ========================================================
def fetch_for_date(target_date):
    """Fetch NSE data for a specific date (Bhavcopy + Delivery)."""
    date_str = target_date.strftime("%d-%m-%Y")
    print(f"  📡 Fetching daily stock data for {date_str} ...")
    try:
        from nselib import capital_market
        raw = capital_market.bhav_copy_with_delivery(date_str)
        df = normalize_dataframe(raw)
        if df is not None and len(df) > 0:
            df["date"] = target_date.strftime("%Y-%m-%d")
            print(f"  ✅ Got {len(df)} stocks (bhav_copy_with_delivery)")
            return df
    except Exception:
        pass
    try:
        from nselib import capital_market
        raw = capital_market.bhav_copy_equities(date_str)
        df = normalize_dataframe(raw)
        if df is not None and len(df) > 0:
            df["date"] = target_date.strftime("%Y-%m-%d")
            print(f"  ✅ Got {len(df)} stocks (fallback: bhav_copy_equities)")
            return df
    except Exception:
        pass
    print(f"  ❌ No stock data available for {date_str}")
    return None


def fetch_fii_dii():
    """Fetch today's FII/DII flow data via direct API."""
    print("  📡 Fetching FII/DII activity...")
    today_str = date.today().strftime("%Y-%m-%d")
    result = {"date": today_str}

    try:
        session = get_nse_session()
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        r = session.get(url, timeout=10)
        data = r.json()

        for item in data:
            category = item.get("category", "").upper()
            buy = float(item.get("buyValue", 0))
            sell = float(item.get("sellValue", 0))
            net = buy - sell

            if "FII" in category or "FPI" in category:
                result.update({"fii_buy": buy, "fii_sell": sell, "fii_net": net})
            elif "DII" in category:
                result.update({"dii_buy": buy, "dii_sell": sell, "dii_net": net})

        if "fii_net" in result or "dii_net" in result:
            new_df = pd.DataFrame([result])
            if os.path.exists(FII_DII_FILE):
                try:
                    old_df = pd.read_csv(FII_DII_FILE)
                    old_df = old_df[old_df["date"] != today_str]
                    df = pd.concat([old_df, new_df], ignore_index=True)
                    df = df.tail(60)  # Keep 60 days instead of 30
                except Exception:
                    df = new_df
            else:
                df = new_df

            df.to_csv(FII_DII_FILE, index=False)
            print("  ✅ Saved FII/DII data")
            return True
    except Exception as e:
        print(f"  ⚠ Failed to fetch FII/DII: {e}")
    return False


def fetch_bulk_block_deals():
    """
    Fetch today's Bulk and Block deals via direct API.
    FIXED: Appends to existing CSV instead of overwriting.
    ADDED: F&O filtering and investor classification.
    """
    print("  📡 Fetching Bulk/Block deals...")
    frames = []

    try:
        session = get_nse_session()
        url = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"
        r = session.get(url, timeout=10)
        data = r.json()

        if "BULK_DEALS_DATA" in data:
            bulk = pd.DataFrame(data["BULK_DEALS_DATA"])
            if not bulk.empty:
                bulk["Deal_Type"] = "BULK"
                frames.append(bulk)
                print(f"  ✅ Got {len(bulk)} Bulk deals")

        if "BLOCK_DEALS_DATA" in data:
            block = pd.DataFrame(data["BLOCK_DEALS_DATA"])
            if not block.empty:
                block["Deal_Type"] = "BLOCK"
                frames.append(block)
                print(f"  ✅ Got {len(block)} Block deals")

    except Exception as e:
        print(f"  ⚠ Failed to fetch deals: {e}")

    if not frames:
        return False

    new_df = pd.concat(frames, ignore_index=True)

    # --- Enrich with investor classification ---
    client_col = next((c for c in new_df.columns if "client" in c.lower()), None)
    if client_col:
        new_df["investor_type"] = new_df[client_col].apply(classify_investor)

    # --- F&O flag ---
    fno_stocks = fetch_fno_list()
    symbol_col = next((c for c in new_df.columns if c.lower() == "symbol"), None)
    if symbol_col and fno_stocks:
        new_df["is_fno"] = new_df[symbol_col].isin(fno_stocks)
        fno_count = new_df["is_fno"].sum()
        print(f"  📊 F&O eligible deals: {fno_count}/{len(new_df)}")

    # --- Append to existing (dedup by date+symbol+client+buySell) ---
    if os.path.exists(BULK_DEALS_FILE):
        try:
            old_df = pd.read_csv(BULK_DEALS_FILE)
            # Identify the date column
            date_col = next((c for c in new_df.columns if c.lower() == "date"), None)
            if date_col and date_col in old_df.columns:
                # Get today's dates from new data to remove old entries for today
                new_dates = new_df[date_col].unique()
                old_df = old_df[~old_df[date_col].isin(new_dates)]
            combined = pd.concat([old_df, new_df], ignore_index=True)
            # Keep last 90 days of data
            if date_col and date_col in combined.columns:
                try:
                    combined["_parsed_date"] = pd.to_datetime(combined[date_col], format="mixed", dayfirst=True)
                    cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
                    combined = combined[combined["_parsed_date"] >= cutoff]
                    combined = combined.drop(columns=["_parsed_date"])
                except Exception:
                    pass
            combined.to_csv(BULK_DEALS_FILE, index=False)
            print(f"  💾 Appended bulk deals (total rows: {len(combined)})")
        except Exception as e:
            print(f"  ⚠ Append failed, overwriting: {e}")
            new_df.to_csv(BULK_DEALS_FILE, index=False)
    else:
        new_df.to_csv(BULK_DEALS_FILE, index=False)
        print(f"  💾 Saved {len(new_df)} bulk/block deals")

    # --- Generate alerts for F&O stocks with institutional activity ---
    generate_alerts(new_df, fno_stocks)

    return True


def generate_alerts(deals_df, fno_stocks):
    """
    Generate actionable alerts for put-selling decisions.
    Focus: F&O stocks where FII/DII (not prop desks) did bulk/block deals.
    """
    alerts = []
    today_str = date.today().isoformat()

    if deals_df is None or deals_df.empty or not fno_stocks:
        return

    symbol_col = next((c for c in deals_df.columns if c.lower() == "symbol"), None)
    client_col = next((c for c in deals_df.columns if "client" in c.lower()), None)
    qty_col = next((c for c in deals_df.columns if c.lower() == "qty"), None)
    price_col = next((c for c in deals_df.columns if c.lower() == "watp"), None)
    bs_col = next((c for c in deals_df.columns if c.lower() == "buysell" or c.lower() == "buysell"), None)

    if not all([symbol_col, client_col]):
        return

    # Filter: F&O stocks only, institutional investors only (FII/DII, not prop)
    fno_deals = deals_df[deals_df.get("is_fno", pd.Series(dtype=bool)) == True].copy()

    if fno_deals.empty:
        # Also check for large block deals in F&O stocks even from prop desks
        # Block deals > 5 lakh shares are noteworthy regardless of investor type
        print("  ℹ No FII/DII bulk deals in F&O stocks today")
    else:
        for _, row in fno_deals.iterrows():
            symbol = row.get(symbol_col, "?")
            client = row.get(client_col, "Unknown")
            inv_type = row.get("investor_type", "Unknown")
            qty = row.get(qty_col, 0) if qty_col else 0
            price = row.get(price_col, 0) if price_col else 0
            side = row.get(bs_col, "?") if bs_col else "?"
            deal_type = row.get("Deal_Type", "?")
            value_cr = (float(qty) * float(price)) / 1e7 if qty and price else 0

            alert = {
                "date": today_str,
                "symbol": symbol,
                "investor": client,
                "investor_type": inv_type,
                "side": side,
                "qty": int(qty) if qty else 0,
                "price": float(price) if price else 0,
                "value_cr": round(value_cr, 2),
                "deal_type": deal_type,
                "is_fno": True,
            }
            alerts.append(alert)

            # Print prominent alert
            action = "BOUGHT" if str(side).upper() == "BUY" else "SOLD"
            emoji = "🟢" if str(side).upper() == "BUY" else "🔴"
            print(f"\n  {emoji} ALERT: {inv_type} — {client}")
            print(f"     {action} {int(qty):,} shares of {symbol} @ ₹{float(price):,.2f}")
            print(f"     Deal value: ₹{value_cr:.1f} Cr ({deal_type})")

    # Save alerts
    if alerts:
        with open(ALERTS_FILE, "w") as f:
            json.dump({"date": today_str, "alerts": alerts}, f, indent=2)
        print(f"\n  🔔 Generated {len(alerts)} actionable alerts → {ALERTS_FILE}")
    else:
        # Save empty alerts file so dashboard knows we checked
        with open(ALERTS_FILE, "w") as f:
            json.dump({"date": today_str, "alerts": [], "note": "No F&O bulk/block deals today"}, f, indent=2)
        print("  ℹ No actionable F&O alerts today")


# ========================================================
# Daily CSV & History
# ========================================================
def save_daily_csv(df, target_date):
    os.makedirs(DAILY_DIR, exist_ok=True)
    filename = target_date.strftime("%Y-%m-%d") + ".csv"
    filepath = os.path.join(DAILY_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"  💾 Saved daily history: {filepath}")
    return filepath


def rebuild_history():
    if not os.path.isdir(DAILY_DIR):
        return
    csv_files = sorted([f for f in os.listdir(DAILY_DIR) if f.endswith(".csv")])
    if not csv_files:
        return
    files_to_keep = csv_files[-MAX_HISTORY_DAYS:]
    files_to_prune = csv_files[:-MAX_HISTORY_DAYS] if len(csv_files) > MAX_HISTORY_DAYS else []
    for old_file in files_to_prune:
        os.remove(os.path.join(DAILY_DIR, old_file))
    frames = []
    for f in files_to_keep:
        try:
            frames.append(pd.read_csv(os.path.join(DAILY_DIR, f)))
        except Exception:
            continue
    if frames:
        history = pd.concat(frames, ignore_index=True)
        if "prev_close" in history.columns and "close" in history.columns:
            history["price_change_pct"] = (
                (history["close"] - history["prev_close"]) / history["prev_close"] * 100
            ).round(2)
        history.to_csv(HISTORY_FILE, index=False)
        print(f"  📊 Rebuilt whale_history.csv — {len(history)} total rows")


def get_trading_dates(n=30):
    dates = []
    current = date.today()
    while len(dates) < n:
        current -= timedelta(days=1)
        if current.weekday() < 5:
            dates.append(current)
    return dates


# ========================================================
# Main Entry Points
# ========================================================
def run_today():
    today = date.today()
    if today.weekday() >= 5:
        print(f"📅 Today ({today}) is a weekend. Skipping.")
        return

    # 1. Fetch F&O list (cached, refreshed daily)
    fetch_fno_list()

    # 2. Fetch auxiliary data
    fetch_fii_dii()
    fetch_bulk_block_deals()

    # 3. Fetch main stock data
    daily_file = os.path.join(DAILY_DIR, today.strftime("%Y-%m-%d") + ".csv")
    if os.path.exists(daily_file):
        print(f"✅ Main stock data for {today} already exists. Skipping main fetch.")
        rebuild_history()
        return

    df = fetch_for_date(today)
    if df is not None and not df.empty:
        save_daily_csv(df, today)
        rebuild_history()
        print(f"\n🎉 Done! Fetched info for {len(df)} stocks on {today}")
    else:
        print(f"\n⚠ Could not fetch stock data for {today}.")


def run_bootstrap():
    print("🚀 Bootstrap mode: fetching last 30 trading days...\n")
    os.makedirs(DAILY_DIR, exist_ok=True)
    trading_dates = get_trading_dates(30)
    for i, td in enumerate(reversed(trading_dates)):
        daily_file = os.path.join(DAILY_DIR, td.strftime("%Y-%m-%d") + ".csv")
        if os.path.exists(daily_file):
            print(f"  ⏭  {td} already exists, skipping.")
            continue

        df = fetch_for_date(td)
        if df is not None and not df.empty:
            save_daily_csv(df, td)
        time.sleep(1.5)
    rebuild_history()
    fetch_fno_list()
    fetch_fii_dii()
    fetch_bulk_block_deals()
    print("\n🎉 Bootstrap complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    if args.bootstrap:
        run_bootstrap()
    else:
        run_today()


if __name__ == "__main__":
    main()
