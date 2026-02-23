"""
fetch_data.py — Standalone NSE Data Fetcher for GitHub Actions
==============================================================
Called daily by .github/workflows/scrape.yml at 7 PM IST.
Behaviour:
  1. Fetches today's bhavcopy + delivery data from NSE via nselib
  2. Saves a daily snapshot CSV to data/daily/YYYY-MM-DD.csv
  3. Recomputes the rolling whale_history.csv (last 60 trading days)
  4. Exits cleanly if today's data already exists (idempotent)
Usage:
  python fetch_data.py              # Fetch today's data
  python fetch_data.py --bootstrap  # Seed last 30 trading days
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
# ---- Thresholds (must match app.py) ----
MIN_DELIVERY_PCT = 50.0
MIN_VOLUME_RATIO = 2.0
MIN_PRICE_CHANGE_PCT = 3.0
MA_WINDOW = 20
MAX_HISTORY_DAYS = 60
# ---- Column aliases (same as app.py) ----
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
            print(f"  ⚠ Missing required column: {col}")
            return None
    if "series" in df.columns:
        df = df[df["series"].str.strip() == "EQ"].copy()
    for col in ["open", "high", "low", "close", "prev_close",
                 "volume", "delivery_qty", "delivery_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["symbol", "close", "volume"])
    return df
def fetch_for_date(target_date):
    """Fetch NSE data for a specific date. Returns normalised DataFrame or None."""
    date_str = target_date.strftime("%d-%m-%Y")
    print(f"  📡 Fetching data for {date_str} ...")
    try:
        from nselib import capital_market
        raw = capital_market.bhav_copy_with_delivery(date_str)
        df = normalize_dataframe(raw)
        if df is not None and len(df) > 0:
            df["date"] = target_date.strftime("%Y-%m-%d")
            print(f"  ✅ Got {len(df)} stocks")
            return df
    except Exception as e:
        print(f"  ⚠ bhav_copy_with_delivery failed: {e}")
    try:
        from nselib import capital_market
        raw = capital_market.bhav_copy_equities(date_str)
        df = normalize_dataframe(raw)
        if df is not None and len(df) > 0:
            df["date"] = target_date.strftime("%Y-%m-%d")
            print(f"  ✅ Got {len(df)} stocks (fallback)")
            return df
    except Exception as e:
        print(f"  ⚠ bhav_copy_equities also failed: {e}")
    print(f"  ❌ No data available for {date_str}")
    return None
def save_daily_csv(df, target_date):
    """Save a daily snapshot CSV."""
    os.makedirs(DAILY_DIR, exist_ok=True)
    filename = target_date.strftime("%Y-%m-%d") + ".csv"
    filepath = os.path.join(DAILY_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"  💾 Saved daily CSV: {filepath}")
    return filepath
def rebuild_history():
    """Rebuild whale_history.csv from daily CSVs (keep last MAX_HISTORY_DAYS)."""
    if not os.path.isdir(DAILY_DIR):
        print("  ⚠ No daily CSV directory found.")
        return
    csv_files = sorted([f for f in os.listdir(DAILY_DIR) if f.endswith(".csv")])
    if not csv_files:
        print("  ⚠ No daily CSVs found.")
        return
    # Keep only the last MAX_HISTORY_DAYS files
    files_to_keep = csv_files[-MAX_HISTORY_DAYS:]
    files_to_prune = csv_files[:-MAX_HISTORY_DAYS] if len(csv_files) > MAX_HISTORY_DAYS else []
    # Prune old files
    for old_file in files_to_prune:
        old_path = os.path.join(DAILY_DIR, old_file)
        os.remove(old_path)
        print(f"  🗑  Pruned old file: {old_file}")
    # Concatenate remaining files
    frames = []
    for f in files_to_keep:
        try:
            frames.append(pd.read_csv(os.path.join(DAILY_DIR, f)))
        except Exception as e:
            print(f"  ⚠ Error reading {f}: {e}")
    if frames:
        history = pd.concat(frames, ignore_index=True)
        # Add computed columns for the history file
        if "prev_close" in history.columns and "close" in history.columns:
            history["price_change_pct"] = (
                (history["close"] - history["prev_close"]) / history["prev_close"] * 100
            ).round(2)
        history.to_csv(HISTORY_FILE, index=False)
        print(f"  📊 Rebuilt whale_history.csv — {len(history)} rows from {len(files_to_keep)} days")
    else:
        print("  ⚠ No valid CSVs to combine.")
def get_trading_dates(n=30):
    """Return last N weekdays."""
    dates = []
    current = date.today()
    while len(dates) < n:
        current -= timedelta(days=1)
        if current.weekday() < 5:
            dates.append(current)
    return dates
def run_today():
    """Fetch today's data (idempotent — skips if already exists)."""
    today = date.today()
    # If today is weekend, skip
    if today.weekday() >= 5:
        print(f"📅 Today ({today}) is a weekend. Skipping.")
        return
    # Check if today's file already exists
    daily_file = os.path.join(DAILY_DIR, today.strftime("%Y-%m-%d") + ".csv")
    if os.path.exists(daily_file):
        print(f"✅ Data for {today} already exists. Skipping fetch.")
        rebuild_history()
        return
    # Fetch today's data
    df = fetch_for_date(today)
    if df is not None and not df.empty:
        save_daily_csv(df, today)
        rebuild_history()
        print(f"\n🎉 Done! Fetched {len(df)} stocks for {today}")
    else:
        print(f"\n⚠ Could not fetch data for {today}. NSE may not have published yet.")
def run_bootstrap():
    """Seed the last 30 trading days of data."""
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
        else:
            print(f"  ⏭  No data for {td} (may be a holiday)")
        # Rate-limit: be nice to NSE servers
        if i < len(trading_dates) - 1:
            time.sleep(2)
    rebuild_history()
    print("\n🎉 Bootstrap complete!")
def main():
    parser = argparse.ArgumentParser(description="WhaleWatch AI — NSE Data Fetcher")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Seed the last 30 trading days of data",
    )
    args = parser.parse_args()
    print("=" * 50)
    print("🐋 WhaleWatch AI — Data Fetcher")
    print("=" * 50)
    print()
    os.makedirs(DATA_DIR, exist_ok=True)
    if args.bootstrap:
        run_bootstrap()
    else:
        run_today()
if __name__ == "__main__":
    main()
