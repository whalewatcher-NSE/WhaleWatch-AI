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
            raw.columns = [c.strip().lower().replace(' ', '_') for c in raw.columns]
            
            # Map columns
            col_map = {'tottrdqty': 'volume', 'deliv_qty': 'delivery_qty', 'deliv_per': 'delivery_pct'}
            raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
            
            if 'series' in raw.columns:
                raw = raw[raw['series'] == 'EQ']
                
            raw['date'] = pd.to_datetime(today_str, format="%d-%m-%Y")
            return raw[['symbol', 'date', 'close', 'volume', 'delivery_qty', 'delivery_pct']]
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
        print("❌ Data not published yet.")
        exit(1) # Triggers GitHub retry

if __name__ == "__main__":
    main()
