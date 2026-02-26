"""
fix_history.py — WhaleWatch Historical Data Fixer
==================================================
WHAT THIS FIXES:
  Your data/history/*.parquet files have delivery_qty and delivery_pct
  as all NaN. The analyze_strategy.py script correctly tries to
  recalculate delivery_pct = delivery_qty / volume, but if delivery_qty
  is NaN, the result is still NaN and the backtester finds 0 trades.

ROOT CAUSE:
  The original scrape used a basic bhavcopy function that does NOT
  include the MTO (delivery) file. This script uses
  nselib's bhav_copy_with_delivery() which downloads BOTH files
  and merges them together correctly.

HOW TO RUN:
  Via GitHub Actions → "🔧 Fix Historical Delivery Data" workflow
  OR locally: python fix_history.py --start_year 2021 --end_year 2025

OUTPUT:
  Replaces data/history/YYYY.parquet files with corrected versions.
  Backs up broken files as data/history/YYYY_broken_backup.parquet first.
  Prints a quality report at the end confirming delivery coverage %.

EXPECTED TIME:
  ~20 minutes per year on GitHub Actions.
  ~90 minutes total for 2021–2025.
"""

import os
import sys
import time
import shutil
import argparse
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths (flat repo structure — all files in root) ───────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
HISTORY_DIR = BASE_DIR / "data" / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Column aliases — every name NSE has used across different years and libraries
COLUMN_ALIASES = {
    "symbol":       ["SYMBOL", "Symbol", "TckrSymb", "TCKR_SYMB", "symbol"],
    "series":       ["SERIES", "Series", "SrNm", "series"],
    "open":         ["OPEN_PRICE", "OPEN", "OpnPric", "open"],
    "high":         ["HIGH_PRICE", "HIGH", "HghPric", "high"],
    "low":          ["LOW_PRICE",  "LOW",  "LwPric",  "low"],
    "close":        ["CLOSE_PRICE", "CLOSE", "ClsPric", "close"],
    "prev_close":   ["PREV_CLOSE", "PREVCLOSE", "PrvsClsgPric", "prev_close"],
    "volume":       ["TTL_TRD_QNTY", "TOTTRDQTY", "TtlTradgVol",
                     "TOTAL_TRADE_QUANTITY", "volume"],
    "delivery_qty": ["DELIV_QTY", "DlvryQty", "DELIVERY_QTY",
                     "Delivery Quantity", "delivery_qty"],
    "delivery_pct": ["DELIV_PER", "DlvryPct", "DELIVERY_PCT",
                     "Delivery %", "delivery_pct"],
    "turnover":     ["TOTTRDVAL", "TtlTrfVal", "TURNOVER", "turnover"],
    "trades":       ["TOTALTRADES", "TtlNbOfTxsExctd", "trades"],
    "isin":         ["ISIN", "isin"],
}


# ─────────────────────────────────────────────────────────────────────────────
#  COLUMN NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename every column to our standard lowercase names.
    Handles all the different column names NSE has used over the years.
    """
    if df is None or df.empty:
        return df

    # Build a reverse lookup: any alias → canonical name
    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            # Strip whitespace — NSE files often have trailing spaces
            rename_map[alias.strip()] = canonical

    # Apply: strip whitespace from actual column names first, then rename
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    After normalising column names:
    1. Filter to EQ series only
    2. Convert all numeric columns to numbers
    3. Remove stocks with no price or volume
    4. Calculate VWAP and delivery ratio if possible
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # EQ series only (removes derivatives, SME, etc.)
    if "series" in df.columns:
        df = df[df["series"].str.strip() == "EQ"].copy()

    if df.empty:
        return pd.DataFrame()

    # Force numeric types
    numeric_cols = ["open", "high", "low", "close", "prev_close",
                    "volume", "delivery_qty", "delivery_pct", "turnover", "trades"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Must have price and volume
    df = df.dropna(subset=["close", "volume"])
    df = df[df["close"] > 0]
    df = df[df["volume"] > 0]

    # Calculate delivery_ratio (this is what analyze_strategy.py needs)
    if "delivery_qty" in df.columns and "volume" in df.columns:
        df["delivery_ratio"] = df["delivery_qty"] / df["volume"]
    else:
        df["delivery_ratio"] = np.nan

    # Calculate VWAP if turnover available
    if "turnover" in df.columns:
        df["vwap"] = np.where(
            df["volume"] > 0,
            df["turnover"] / df["volume"],
            df["close"]
        )
        df["vwap_dev"] = np.where(
            df["vwap"] > 0,
            (df["close"] - df["vwap"]) / df["vwap"],
            np.nan
        )

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE DAY DOWNLOAD — tries nselib first, then direct NSE URLs
# ─────────────────────────────────────────────────────────────────────────────

def fetch_day_with_delivery(trade_date: date) -> pd.DataFrame | None:
    """
    Download Bhavcopy + Delivery data for one trading day.

    Method 1: nselib bhav_copy_with_delivery()
      This is the same library your fetch_data.py already uses.
      It downloads BOTH the bhavcopy and MTO delivery file together.

    Method 2: nselib bhav_copy_equities() + manual MTO download
      If method 1 fails, get the bhavcopy alone and try to get
      the MTO (delivery) file separately via direct HTTP.

    Method 3: Direct NSE archive URLs
      Last resort — downloads raw ZIP/CSV directly from NSE archives.
    """
    date_str = trade_date.strftime("%d-%m-%Y")   # Format nselib expects
    df = None

    # ── Method 1: nselib with delivery (best option) ──────────────────────────
    try:
        from nselib import capital_market
        raw = capital_market.bhav_copy_with_delivery(date_str)
        if raw is not None and not raw.empty and len(raw) > 10:
            df = normalize_columns(raw)
            df = clean_dataframe(df)
            if df is not None and not df.empty:
                # Verify delivery data actually came through
                if "delivery_qty" in df.columns and df["delivery_qty"].notna().sum() > 5:
                    df["date"] = pd.to_datetime(trade_date)
                    return df
    except Exception:
        pass

    # ── Method 2: nselib basic + manual MTO ───────────────────────────────────
    try:
        from nselib import capital_market
        raw = capital_market.bhav_copy_equities(date_str)
        if raw is not None and not raw.empty and len(raw) > 10:
            df = normalize_columns(raw)
            df = clean_dataframe(df)

            # Now try to get delivery data separately via direct download
            delivery_df = fetch_mto_direct(trade_date)
            if delivery_df is not None and not delivery_df.empty:
                df = df.merge(
                    delivery_df[["symbol", "delivery_qty", "delivery_pct"]],
                    on="symbol",
                    how="left",
                    suffixes=("", "_mto")
                )
                # Use MTO values if they exist and original is NaN
                if "delivery_qty_mto" in df.columns:
                    df["delivery_qty"] = df["delivery_qty"].fillna(df["delivery_qty_mto"])
                    df["delivery_pct"] = df["delivery_pct"].fillna(df["delivery_pct_mto"])
                    df = df.drop(columns=[c for c in df.columns if c.endswith("_mto")])
                    # Recalculate ratio with merged data
                    df["delivery_ratio"] = df["delivery_qty"] / df["volume"]

            if df is not None and not df.empty:
                df["date"] = pd.to_datetime(trade_date)
                return df
    except Exception:
        pass

    # ── Method 3: Direct NSE archive (last resort) ────────────────────────────
    try:
        df = fetch_bhavcopy_direct(trade_date)
        if df is not None and not df.empty:
            delivery_df = fetch_mto_direct(trade_date)
            if delivery_df is not None and not delivery_df.empty:
                df = df.merge(
                    delivery_df[["symbol", "delivery_qty", "delivery_pct"]],
                    on="symbol",
                    how="left"
                )
                df["delivery_ratio"] = df["delivery_qty"] / df["volume"]
            df["date"] = pd.to_datetime(trade_date)
            return df
    except Exception:
        pass

    return None


def fetch_mto_direct(trade_date: date) -> pd.DataFrame | None:
    """
    Download the MTO (delivery) file directly from NSE archives.
    This is the file that was MISSING from your original scrape.

    The MTO file format:
      Record type 90 = security-wise delivery data
      Fields: ..., symbol, series, ..., delivery_qty, delivery_pct, ...
    """
    import io
    import requests

    date_str_old = trade_date.strftime("%d%m%Y")   # 01012025
    date_str_new = trade_date.strftime("%Y%m%d")   # 20250101

    urls = [
        f"https://archives.nseindia.com/archives/equities/mto/MTO_{date_str_old}.DAT",
        f"https://nsearchives.nseindia.com/archives/equities/mto/MTO_{date_str_new}.DAT",
        f"https://nsearchives.nseindia.com/archives/equities/mto/MTO_{date_str_old}.DAT",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer":    "https://www.nseindia.com/",
    }

    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=8)
        time.sleep(0.5)
    except Exception:
        pass

    for url in urls:
        try:
            resp = session.get(url, headers=headers, timeout=12)
            if resp.status_code != 200 or len(resp.content) < 100:
                continue

            rows = []
            for line in resp.text.strip().split("\n"):
                parts = line.split(",")
                # Record type 90 = delivery position data
                if len(parts) >= 7 and parts[0].strip() == "90":
                    try:
                        rows.append({
                            "symbol":       parts[2].strip(),
                            "series":       parts[3].strip(),
                            "delivery_qty": float(parts[5].strip()),
                            "delivery_pct": float(parts[6].strip()),
                        })
                    except (ValueError, IndexError):
                        continue

            if rows:
                df = pd.DataFrame(rows)
                df = df[df["series"] == "EQ"][
                    ["symbol", "delivery_qty", "delivery_pct"]
                ].copy()
                if len(df) > 10:
                    return df

        except Exception:
            continue

    return None


def fetch_bhavcopy_direct(trade_date: date) -> pd.DataFrame | None:
    """
    Download Bhavcopy directly from NSE archives as a fallback.
    Handles both old ZIP format (pre-2025) and new CSV format (2025+).
    """
    import io
    import zipfile
    import requests

    date_str_long  = trade_date.strftime("%d%b%Y").upper()   # 01JAN2025
    date_str_short = trade_date.strftime("%Y%m%d")           # 20250101
    year = trade_date.year
    mon  = trade_date.strftime("%b").upper()

    urls = [
        f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date_str_short}_F_0000.csv.zip",
        f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date_str_short}_F_0000.csv",
        f"https://archives.nseindia.com/content/historical/EQUITIES/{year}/{mon}/cm{date_str_long}bhav.csv.zip",
        f"https://archives.nseindia.com/content/historical/EQUITIES/{year}/{mon}/cm{date_str_long}bhav.csv",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer":    "https://www.nseindia.com/",
    }

    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=8)
        time.sleep(0.5)
    except Exception:
        pass

    for url in urls:
        try:
            resp = session.get(url, headers=headers, timeout=15)
            if resp.status_code != 200 or len(resp.content) < 500:
                continue

            if url.endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                        with z.open(z.namelist()[0]) as f:
                            df = pd.read_csv(f)
                except zipfile.BadZipFile:
                    df = pd.read_csv(io.StringIO(resp.text))
            else:
                df = pd.read_csv(io.StringIO(resp.text))

            if df is not None and len(df) > 10:
                df = normalize_columns(df)
                df = clean_dataframe(df)
                if df is not None and not df.empty:
                    return df

        except Exception:
            continue

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  QUALITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_quality(filepath: Path) -> dict:
    """
    Read a parquet and check if delivery_qty is populated.
    Returns a dict describing the file quality.
    """
    if not filepath.exists():
        return {"exists": False, "delivery_coverage": 0.0, "rows": 0}

    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        return {"exists": True, "error": str(e), "delivery_coverage": 0.0, "rows": 0}

    rows    = len(df)
    days    = df["date"].nunique() if "date" in df.columns else 0
    symbols = df["symbol"].nunique() if "symbol" in df.columns else 0

    if "delivery_qty" not in df.columns:
        coverage = 0.0
    else:
        df["delivery_qty"] = pd.to_numeric(df["delivery_qty"], errors="coerce")
        coverage = df["delivery_qty"].notna().mean() * 100

    return {
        "exists":            True,
        "rows":              rows,
        "days":              days,
        "symbols":           symbols,
        "delivery_coverage": round(coverage, 1),
        # Broken = less than 5% of rows have delivery data
        "broken":            coverage < 5.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def get_weekdays(start: date, end: date) -> list:
    """Return all Monday–Friday dates between start and end inclusive."""
    dates, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            dates.append(cur)
        cur += timedelta(days=1)
    return dates


# ─────────────────────────────────────────────────────────────────────────────
#  FIX ONE YEAR
# ─────────────────────────────────────────────────────────────────────────────

def fix_year(year: int, force: bool = False) -> bool:
    """
    Re-download one full year with delivery data and save to parquet.

    Steps:
      1. Check if existing file already has good delivery data
      2. If not (or force=True), back up the broken file
      3. Download every weekday in the year, with retry logic
      4. Save the new fixed parquet
      5. Print coverage stats

    Returns True if successful.
    """
    out_path = HISTORY_DIR / f"{year}.parquet"

    # ── Skip if already good ──────────────────────────────────────────────────
    if not force:
        q = check_quality(out_path)
        if q.get("exists") and not q.get("broken"):
            print(f"  ✅ {year}: Already good "
                  f"(delivery={q['delivery_coverage']}%, rows={q['rows']:,}) — skipping")
            return True

    # ── Back up the broken file ───────────────────────────────────────────────
    if out_path.exists():
        backup = HISTORY_DIR / f"{year}_broken_backup.parquet"
        shutil.copy2(out_path, backup)
        print(f"  📦 Backed up broken file → {backup.name}")

    # ── Date range ────────────────────────────────────────────────────────────
    y_start = date(year, 1, 1)
    y_end   = min(date(year, 12, 31), date.today() - timedelta(days=1))

    if y_end < y_start:
        print(f"  ⏭  {year} hasn't started yet — skipping")
        return False

    weekdays = get_weekdays(y_start, y_end)
    print(f"\n{'='*55}")
    print(f"  Downloading {year} — up to {len(weekdays)} trading days")
    print(f"{'='*55}")

    # ── Download loop ─────────────────────────────────────────────────────────
    frames          = []
    days_success    = 0
    days_failed     = 0
    days_no_deliv   = 0

    for i, d in enumerate(weekdays):
        try:
            df_day = fetch_day_with_delivery(d)

            if df_day is None or df_day.empty:
                days_failed += 1
                # Print a dot every 5 failures so you know it's running
                if days_failed % 5 == 0:
                    print(f"    ... {days_failed} days with no data "
                          f"(likely holidays/weekends)")
                continue

            frames.append(df_day)
            days_success += 1

            # Track how many days have delivery data
            if "delivery_qty" in df_day.columns:
                has_deliv = df_day["delivery_qty"].notna().sum()
                if has_deliv < 5:
                    days_no_deliv += 1
            else:
                days_no_deliv += 1

            # Progress update every 50 successful days
            if days_success % 50 == 0:
                print(f"    ✔ {days_success} days downloaded so far...")

            # Polite delay — NSE rate limits aggressive scrapers
            # Slightly variable to avoid looking like a bot
            time.sleep(0.8 + (i % 4) * 0.15)

            # Refresh session every 80 days — NSE sessions expire
            if (i + 1) % 80 == 0:
                print(f"    🔄 Refreshing NSE session at day {i+1}...")
                time.sleep(3)

        except KeyboardInterrupt:
            print("\n  ⚠️  Interrupted by user")
            break
        except Exception as e:
            days_failed += 1
            continue

    # ── Save results ──────────────────────────────────────────────────────────
    if not frames:
        print(f"  ❌ {year}: No data collected at all. "
              f"Check your internet connection and nselib installation.")
        return False

    year_df = pd.concat(frames, ignore_index=True)
    year_df["date"] = pd.to_datetime(year_df["date"])
    year_df = year_df.sort_values(["date", "symbol"]).reset_index(drop=True)
    year_df = year_df.drop_duplicates(subset=["date", "symbol"])

    year_df.to_parquet(out_path, index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    size_mb = out_path.stat().st_size / 1024 / 1024

    if "delivery_qty" in year_df.columns:
        deliv_cov = year_df["delivery_qty"].notna().mean() * 100
    else:
        deliv_cov = 0.0

    print(f"\n  ✅ {year} SAVED")
    print(f"     Days with data:        {days_success}")
    print(f"     Days failed/holiday:   {days_failed}")
    print(f"     Days missing delivery: {days_no_deliv}")
    print(f"     Total rows:            {len(year_df):,}")
    print(f"     Unique stocks:         {year_df['symbol'].nunique()}")
    print(f"     Delivery data:         {deliv_cov:.1f}% of rows")
    print(f"     File size:             {size_mb:.1f} MB → {out_path.name}")

    if deliv_cov < 20:
        print(f"\n  ⚠️  Low delivery coverage ({deliv_cov:.1f}%).")
        print(f"     This can happen for older years (2021–2022) where NSE's")
        print(f"     MTO archive is incomplete. The bhavcopy data is still valid.")

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  QUALITY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_quality_report():
    """Print a summary table of all parquet files in data/history/."""
    files = sorted([f for f in HISTORY_DIR.glob("*.parquet")
                    if "backup" not in f.name])

    if not files:
        print("  No parquet files found in data/history/")
        return

    print("\n" + "="*68)
    print("  📊 WHALEWATCH DATA QUALITY REPORT")
    print("="*68)
    print(f"  {'FILE':<22} {'ROWS':>9} {'DAYS':>6} {'STOCKS':>8} "
          f"{'DELIVERY%':>11}  STATUS")
    print(f"  {'-'*22} {'-'*9} {'-'*6} {'-'*8} {'-'*11}  {'-'*8}")

    total_rows = 0
    for f in files:
        q = check_quality(f)
        if not q["exists"]:
            continue
        status = "❌ BROKEN" if q.get("broken") else "✅ GOOD"
        print(
            f"  {f.name:<22} "
            f"{q['rows']:>9,} "
            f"{q['days']:>6} "
            f"{q['symbols']:>8} "
            f"{q['delivery_coverage']:>10.1f}%  "
            f"{status}"
        )
        total_rows += q["rows"]

    print(f"  {'─'*68}")
    print(f"  {'TOTAL':<22} {total_rows:>9,}")
    print("="*68 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix historical parquet files — add missing delivery data"
    )
    parser.add_argument(
        "--start_year", type=int, default=2021,
        help="First year to fix (default: 2021)"
    )
    parser.add_argument(
        "--end_year", type=int, default=date.today().year,
        help="Last year to fix (default: current year)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if file already has good delivery data"
    )
    parser.add_argument(
        "--check_only", action="store_true",
        help="Only print quality report, do not download anything"
    )
    args = parser.parse_args()

    if args.check_only:
        print_quality_report()
        sys.exit(0)

    print("\n" + "="*55)
    print("  🐋 WHALEWATCH — HISTORICAL DATA FIX")
    print("="*55)
    print(f"  Years:       {args.start_year} → {args.end_year}")
    print(f"  Force:       {args.force}")
    print(f"  Output dir:  {HISTORY_DIR}")
    print("="*55)

    # Show what we currently have
    print("\n  Current state of data/history/:")
    for y in range(args.start_year, args.end_year + 1):
        q = check_quality(HISTORY_DIR / f"{y}.parquet")
        if q["exists"]:
            status = "BROKEN (will fix)" if q.get("broken") else "GOOD (will skip)"
            if args.force:
                status = "EXISTS (force re-download)"
            print(f"    {y}: {status} | delivery={q['delivery_coverage']}%")
        else:
            print(f"    {y}: NOT FOUND — will download")
    print()

    # Fix each year
    results = {}
    for year in range(args.start_year, args.end_year + 1):
        ok = fix_year(year, force=args.force)
        results[year] = ok

    # Final summary
    print_quality_report()

    good  = [y for y, ok in results.items() if ok]
    bad   = [y for y, ok in results.items() if not ok]

    print(f"  ✅ Fixed successfully: {good}")
    if bad:
        print(f"  ❌ Failed:             {bad}")
        print(f"     Failed years may have incomplete NSE archives.")
        print(f"     The backtester will still work for the successful years.")
    print(f"\n  🎯 Next step: Run analyze_strategy.py to verify the backtester\n")
