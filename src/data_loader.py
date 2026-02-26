import os
import pandas as pd
from datetime import date
import requests
from pathlib import Path

# --- Setup Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DAILY_DIR = DATA_DIR / "daily"
HISTORY_FILE = DATA_DIR / "history.csv"
FII_FILE = DATA_DIR / "fii_dii.csv"
BULK_FILE = DATA_DIR / "bulk_deals.csv"

DATA_DIR.mkdir(exist_ok=True)
DAILY_DIR.mkdir(exist_ok=True)

HEADERS = {'user-agent': 'Mozilla/5.0', 'referer': 'https://www.nseindia.com/'}

# Robust Column Mapping
COLUMN_ALIASES = {
    "symbol":       ["SYMBOL", "Symbol", "symbol"],
    "series":       ["SERIES", "Series", "series"],
    "close":        ["CLOSE_PRICE", "CLOSE", "close", "last_price"],
    "volume":       ["TTL_TRD_QNTY", "TOTTRDQTY", "TOTAL_TRADE_QUANTITY", "volume"],
    "delivery_qty": ["DELIV_QTY", "DlvryQty", "DELIVERY_QTY", "delivery_qty"],
    "delivery_pct": ["DELIV_PER", "DlvryPct", "DELIVERY_PCT", "delivery_pct"],
}

def get_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    try: s.get("https://www.nseindia.com", timeout=5)
    except: pass
    return s

def fetch_daily_bhavcopy(today_str):
    try:
        from nselib import capital_market
        raw = capital_market.bhav_copy_with_delivery(today_str)
        if raw is not None and not raw.empty:
            # 1. Strip and lowercase all raw columns
            raw.columns = [str(c).strip().lower().replace(' ', '_') for c in raw.columns]
            
            # 2. Build mapping dictionary
            rename_map = {}
            for canonical, aliases in COLUMN_ALIASES.items():
                for alias in aliases:
                    rename_map[alias.lower().replace(' ', '_')] = canonical
                    
            # 3. Rename columns
            raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})
            
            # 4. Filter and select
            if 'series' in raw.columns:
                raw = raw[raw['series'].str.strip() == 'EQ']
                
            raw['date'] = pd.to_datetime(today_str, format="%d-%m-%Y")
            
            # Ensure columns exist before selecting
            required_cols = ['symbol', 'date', 'close', 'volume']
            missing = [c for c in required_cols if c not in raw.columns]
            if missing:
                print(f"Missing critical columns after mapping: {missing}")
                return None
                
            # Safely get delivery data if it exists
            final_cols = ['symbol', 'date', 'close', 'volume']
            if 'delivery_qty' in raw.columns: final_cols.append('delivery_qty')
            if 'delivery_pct' in raw.columns: final_cols.append('delivery_pct')
                
            return raw[final_cols]
    except Exception as e:
        print(f"Bhavcopy Error: {e}")
    return None

def fetch_institutional(session):
    # FII/DII
    try:
        r = session.get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=10)
        data = r.json()
        fii_net = sum([float(i['buyValue']) - float(i['sellValue']) for i in data if 'FII' in i.get('category','').upper()])
        dii_net = sum([float(i['buyValue']) - float(i['sellValue']) for i in data if 'DII' in i.get('category','').upper()])
        
        new_fii = pd.DataFrame([{'date': str(date.today()), 'fii_net': fii_net, 'dii_net': dii_net}])
        if FII_FILE.exists():
            old_fii = pd.read_csv(FII_FILE)
            old_fii = old_fii[old_fii['date'] != str(date.today())]
            pd.concat([old_fii, new_fii]).tail(30).to_csv(FII_FILE, index=False)
        else:
            new_fii.to_csv(FII_FILE, index=False)
    except Exception as e: print(f"FII Error: {e}")

    # Bulk/Block
    try:
        r = session.get("https://www.nseindia.com/api/snapshot-capital-market-largedeal", timeout=10)
        d = r.json()
        frames = []
        if "BULK_DEALS_DATA" in d: frames.append(pd.DataFrame(d["BULK_DEALS_DATA"]))
        if "BLOCK_DEALS_DATA" in d: frames.append(pd.DataFrame(d["BLOCK_DEALS_DATA"]))
        if frames:
            df = pd.concat(frames)
            df['date'] = str(date.today())
            if BULK_FILE.exists():
                old_bulk = pd.read_csv(BULK_FILE)
                old_bulk = old_bulk[old_bulk['date'] != str(date.today())]
                pd.concat([old_bulk, df]).tail(500).to_csv(BULK_FILE, index=False)
            else:
                df.to_csv(BULK_FILE, index=False)
    except Exception as e: print(f"Bulk Error: {e}")

def main():
    if date.today().weekday() >= 5:
        print("Weekend. Exiting.")
        return

    today_str = date.today().strftime("%d-%m-%Y")
    df = fetch_daily_bhavcopy(today_str)
    
    if df is not None and not df.empty:
        # Save daily
        df.to_csv(DAILY_DIR / f"{date.today()}.csv", index=False)
        
        # Update rolling history (last 30 days)
        files = sorted(list(DAILY_DIR.glob("*.csv")))[-30:]
        history_df = pd.concat([pd.read_csv(f) for f in files])
        history_df.to_csv(HISTORY_FILE, index=False)
        
        # Fetch FII/Bulk
        fetch_institutional(get_session())
        print(f"✅ Downloaded successfully for {today_str}")
        exit(0)
    else:
        print("❌ Data not published yet or mapping failed.")
        exit(1) # Triggers GitHub retry

if __name__ == "__main__":
    main()
