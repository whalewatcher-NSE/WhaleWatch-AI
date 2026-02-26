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
    if df is None or df.empty:
        return df

    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            rename_map[alias.strip()] = canonical

    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if "series" in df.columns:
        df = df[df["series"].str.strip() == "EQ"].copy()

    if df.empty:
        return pd.DataFrame()

    numeric_cols = ["open", "high", "low", "close", "prev_close",
                    "volume", "delivery_qty", "delivery_pct", "turnover", "trades"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close", "volume"])
    df = df[df["close"] > 0]
    df = df[df["volume"] > 0]

    if "delivery_qty" in df.columns and "volume" in df.columns:
        df["delivery_ratio"] = df["delivery_qty"] / df["volume"]
    else:
        df["delivery_ratio"] = np.nan

    if "turnover" in df.columns:
        df["vwap"] = np.where(df["volume"] > 0, df["turnover"] / df["volume"], df["close"])
        df["vwap_dev"] = np.where(df["vwap"] > 0, (df["close"] - df["vwap"]) / df["vwap"], np.nan)

    return df.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE DAY DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def fetch_day_with_delivery(trade_date: date) -> pd.DataFrame | None:
    date_str = trade_date.strftime("%d-%m-%Y")
    df = None

    try:
        from nselib import capital_market
        raw = capital_market.bhav_copy_with_delivery(date_str)
        if raw is not None and not raw.empty and len(raw) > 10:
            df = normalize_columns(raw)
            df = clean_dataframe(df)
            if df is not None and not df.empty:
                if "delivery_qty" in df.columns and df["delivery_qty"].notna().sum() > 5:
                    df["date"] = pd.to_datetime(trade_date)
                    return df
    except Exception:
        pass

    try:
        from nselib import capital_market
        raw = capital_market.bhav_copy_equities(date_str)
        if raw is not None and not raw.empty and len(raw) > 10:
            df = normalize_columns(raw)
            df = clean_dataframe(df)

            delivery_df = fetch_mto_direct(trade_date)
            if delivery_df is not None and not delivery_df.empty:
                df = df.merge(
                    delivery_df[["symbol", "delivery_qty", "delivery_pct"]],
                    on="symbol", how="left", suffixes=("", "_mto")
                )
                if "delivery_qty_mto" in df.columns:
                    df["delivery_qty"] = df["delivery_qty"].fillna(df["delivery_qty_mto"])
                    df["delivery_pct"] = df["delivery_pct"].fillna(df["delivery_pct_mto"])
                    df = df.drop(columns=[c for c in df.columns if c.endswith("_mto")])
                    df["delivery_ratio"] = df["delivery_qty"] / df["volume"]

            if df is not None and not df.empty:
                df["date"] = pd.to_datetime(trade_date)
                return df
    except Exception:
        pass

    try:
        df = fetch_bhavcopy_direct(trade_date)
        if df is not None and not df.empty:
            delivery_df = fetch_mto_direct(trade_date)
            if delivery_df is not None and not delivery_df.empty:
                df = df.merge(
                    delivery_df[["symbol", "delivery_qty", "delivery_pct"]],
                    on="symbol", how="left"
                )
                df["delivery_ratio"] = df["delivery_qty"] / df["volume"]
            df["date"] = pd.to_datetime(trade_date)
            return df
    except Exception:
        pass

    return None

def fetch_mto_direct(trade_date: date) -> pd.DataFrame | None:
    import requests
    date_str_old = trade_date.strftime("%d%m%Y")
    date_str_new = trade_date.strftime("%Y%m%d")

    urls = [
        f"https://archives.nseindia.com/archives/equities/mto/MTO_{date_str_old}.DAT",
        f"https://nsearchives.nseindia.com/archives/equities/mto/MTO_{date_str_new}.DAT",
    ]

    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"}
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=8)
        time.sleep(0.5)
    except Exception:
        pass

    for url in urls:
        try:
            resp = session.get(url, headers=headers, timeout=12)
            if resp.status_code != 200 or len(resp.content) < 100: continue

            rows = []
            for line in resp.text.strip().split("\n"):
                parts = line.split(",")
                if len(parts) >= 7 and parts[0].strip() == "90":
                    try:
                        rows.append({
                            "symbol":       parts[2].strip(),
                            "series":       parts[3].strip(),
                            "delivery_qty": float(parts[5].strip()),
                            "delivery_pct": float(parts[6].strip()),
                        })
                    except: continue

            if rows:
                df = pd.DataFrame(rows)
                df = df[df["series"] == "EQ"][["symbol", "delivery_qty", "delivery_pct"]].copy()
                if len(df) > 10: return df
        except Exception:
            continue

    return None

def fetch_bhavcopy_direct(trade_date: date) -> pd.DataFrame | None:
    import io, zipfile, requests
    date_str_long  = trade_date.strftime("%d%b%Y").upper()
    date_str_short = trade_date.strftime("%Y%m%d")
    year, mon  = trade_date.year, trade_date.strftime("%b").upper()

    urls = [
        f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date_str_short}_F_0000.csv.zip",
        f"https://archives.nseindia.com/content/historical/EQUITIES/{year}/{mon}/cm{date_str_long}bhav.csv.zip",
    ]

    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"}
    session = requests.Session()
    try: session.get("https://www.nseindia.com", headers=headers, timeout=8); time.sleep(0.5)
    except Exception: pass

    for url in urls:
        try:
            resp = session.get(url, headers=headers, timeout=15)
            if resp.status_code != 200 or len(resp.content) < 500: continue

            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f)

            if df is not None and len(df) > 10:
                df = normalize_columns(df)
                df = clean_dataframe(df)
                if df is not None and not df.empty: return df
        except Exception:
            continue

    return None

# ─────────────────────────────────────────────────────────────────────────────
#  QUALITY CHECK & UTILS
# ─────────────────────────────────────────────────────────────────────────────

def check_quality(filepath: Path) -> dict:
    if not filepath.exists():
        return {"exists": False, "delivery_coverage": 0.0, "rows": 0}

    try: df = pd.read_parquet(filepath)
    except Exception as e: return {"exists": True, "error": str(e), "delivery_coverage": 0.0, "rows": 0}

    rows = len(df)
    days = df["date"].nunique() if "date" in df.columns else 0
    symbols = df["symbol"].nunique() if "symbol" in df.columns else 0

    if "delivery_qty" not in df.columns:
        coverage = 0.0
    else:
        df["delivery_qty"] = pd.to_numeric(df["delivery_qty"], errors="coerce")
        coverage = df["delivery_qty"].notna().mean() * 100

    return {"exists": True, "rows": rows, "days": days, "symbols": symbols, 
            "delivery_coverage": round(coverage, 1), "broken": coverage < 5.0}

def get_weekdays(start: date, end: date) -> list:
    dates, cur = [], start
    while cur <= end:
        if cur.weekday() < 5: dates.append(cur)
        cur += timedelta(days=1)
    return dates

# ─────────────────────────────────────────────────────────────────────────────
#  FIX ONE YEAR
# ─────────────────────────────────────────────────────────────────────────────

def fix_year(year: int, force: bool = False) -> bool:
    out_path = HISTORY_DIR / f"{year}.parquet"

    if not force:
        q = check_quality(out_path)
        if q.get("exists") and not q.get("broken"):
            print(f"  ✅ {year}: Already good (delivery={q['delivery_coverage']}%) — skipping")
            return True

    if out_path.exists():
        backup = HISTORY_DIR / f"{year}_broken_backup.parquet"
        shutil.copy2(out_path, backup)

    y_start = date(year, 1, 1)
    y_end   = min(date(year, 12, 31), date.today() - timedelta(days=1))
    if y_end < y_start: return False

    weekdays = get_weekdays(y_start, y_end)
    print(f"\n{'='*55}\n  Downloading {year} — up to {len(weekdays)} trading days\n{'='*55}")

    frames, days_success, days_failed = [], 0, 0

    for i, d in enumerate(weekdays):
        try:
            df_day = fetch_day_with_delivery(d)
            if df_day is None or df_day.empty:
                days_failed += 1
                continue
            frames.append(df_day)
            days_success += 1

            if days_success % 50 == 0: print(f"    ✔ {days_success} days downloaded so far...")
            time.sleep(0.8 + (i % 4) * 0.15)
            if (i + 1) % 80 == 0: time.sleep(3)

        except KeyboardInterrupt: break
        except Exception:
            days_failed += 1
            continue

    if not frames: return False

    year_df = pd.concat(frames, ignore_index=True)
    year_df["date"] = pd.to_datetime(year_df["date"])
    year_df = year_df.sort_values(["date", "symbol"]).drop_duplicates(subset=["date", "symbol"]).reset_index(drop=True)
    year_df.to_parquet(out_path, index=False)

    deliv_cov = year_df["delivery_qty"].notna().mean() * 100 if "delivery_qty" in year_df.columns else 0.0
    print(f"\n  ✅ {year} SAVED. Days success: {days_success} | Delivery: {deliv_cov:.1f}%")
    return True

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2021)
    parser.add_argument("--end_year", type=int, default=date.today().year)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for year in range(args.start_year, args.end_year + 1):
        fix_year(year, force=args.force)
