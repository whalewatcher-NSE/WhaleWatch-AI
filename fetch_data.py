"""
fetch_data.py — Standalone NSE Data Fetcher for GitHub Actions
==============================================================
Enhanced version: Also fetches FII/DII aggregate flows and Bulk/Block deals.
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
import time
import argparse
# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
HISTORY_FILE = os.path.join(DATA_DIR, "whale_history.csv")
FII_DII_FILE = os.path.join(DATA_DIR, "fii_dii.csv")
BULK_DEALS_FILE = os.path.join(DATA_DIR, "bulk_deals.csv")
# ---- Config ----
MAX_HISTORY_DAYS = 60
# Column aliases (same as app.py)
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
    except Exception as e:
        pass
    try:
        from nselib import capital_market
        raw = capital_market.bhav_copy_equities(date_str)
        df = normalize_dataframe(raw)
        if df is not None and len(df) > 0:
            df["date"] = target_date.strftime("%Y-%m-%d")
            print(f"  ✅ Got {len(df)} stocks (fallback: bhav_copy_equities)")
            return df
    except Exception as e:
        pass
    print(f"  ❌ No stock data available for {date_str}")
    return None
def fetch_fii_dii():
    """Fetch today's FII/DII flow data and append to CSV."""
    print("  📡 Fetching FII/DII activity...")
    today_str = date.today().strftime("%Y-%m-%d")
    result = {"date": today_str}
    
    try:
        from nselib import capital_market
        raw = capital_market.fii_dii_trading_activity()
        if raw is not None and not raw.empty:
            for _, row in raw.iterrows():
                category = str(row.get("category", row.get("Category", ""))).strip()
                buy = pd.to_numeric(row.get("buyValue", row.get("BUY_VALUE", row.get("buy_value", 0))), errors="coerce") or 0
                sell = pd.to_numeric(row.get("sellValue", row.get("SELL_VALUE", row.get("sell_value", 0))), errors="coerce") or 0
                net = buy - sell
                
                if "FII" in category.upper() or "FPI" in category.upper():
                    result.update({"fii_buy": buy, "fii_sell": sell, "fii_net": net})
                elif "DII" in category.upper():
                    result.update({"dii_buy": buy, "dii_sell": sell, "dii_net": net})
            
            if "fii_net" in result or "dii_net" in result:
                new_df = pd.DataFrame([result])
                # Append to existing
                if os.path.exists(FII_DII_FILE):
                    try:
                        old_df = pd.read_csv(FII_DII_FILE)
                        # Remove today's date if already exists to avoid duplicates
                        old_df = old_df[old_df["date"] != today_str]
                        df = pd.concat([old_df, new_df], ignore_index=True)
                        # Keep last 30 days
                        df = df.tail(30)
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
    """Fetch today's Bulk and Block deals and save to a single CSV."""
    print("  📡 Fetching Bulk/Block deals...")
    frames = []
    
    try:
        from nselib import capital_market
        raw_bulk = capital_market.bulk_deal_data()
        if raw_bulk is not None and not raw_bulk.empty:
            raw_bulk["Deal_Type"] = "BULK"
            frames.append(raw_bulk)
            print(f"  ✅ Got {len(raw_bulk)} Bulk deals")
    except Exception as e:
        print(f"  ⚠ Failed to fetch bulk deals: {e}")
    try:
        from nselib import capital_market
        raw_block = capital_market.block_deal_data()
        if raw_block is not None and not raw_block.empty:
            raw_block["Deal_Type"] = "BLOCK"
            frames.append(raw_block)
            print(f"  ✅ Got {len(raw_block)} Block deals")
    except Exception as e:
        print(f"  ⚠ Failed to fetch block deals: {e}")
    if frames:
        df = pd.concat(frames, ignore_index=True)
        # We overwrite this file daily because we only care about "today's" deals
        df.to_csv(BULK_DEALS_FILE, index=False)
        print("  💾 Saved Bulk/Block deals to CSV")
        return True
    
    return False
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
def run_today():
    today = date.today()
    if today.weekday() >= 5:
        print(f"📅 Today ({today}) is a weekend. Skipping.")
        return
    # 1. Fetch auxiliary data (always run this to get latest deals/flows)
    fetch_fii_dii()
    fetch_bulk_block_deals()
    # 2. Fetch main stock data
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
        time.sleep(1.5)  # Rate limiting
    rebuild_history()
    # Also fetch today's auxiliary data during bootstrap
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
