import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

# --- Setup Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

HISTORY_FILE = DATA_DIR / "history.csv"
BULK_FILE = DATA_DIR / "bulk_deals.csv"

def calculate_signals():
    if not HISTORY_FILE.exists():
        print("No history file found.")
        return

    df = pd.read_csv(HISTORY_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])
    
    # Calculate Averages (using min_periods=1 so it works even on Day 1)
    df['vol_20d'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['close_prev'] = df.groupby('symbol')['close'].shift(1)
    df['turnover'] = df['close'] * df['volume']
    
    # Get latest day
    latest_date = df['date'].max()
    today_df = df[df['date'] == latest_date].copy()
    
    # --- HARD DISQUALIFIERS (The Red Flags) ---
    # 1. Illiquid (Turnover < 50 Lakhs)
    # 2. Pure Speculation (Delivery < 15%)
    # 3. Penny Stocks (Price < 20)
    today_df['Disqualified'] = np.where(
        (today_df['turnover'] < 5_000_000) | 
        (today_df['delivery_pct'] < 15) | 
        (today_df['close'] < 20), 
        True, False
    )

    # --- THE 7-SIGNAL CONVICTION SCORE ---
    today_df['Score'] = 0
    
    # Signal 1: Delivery
    today_df['Score'] += np.where(today_df['delivery_pct'] > 50, 1, 0)
    today_df['Score'] -= np.where(today_df['delivery_pct'] < 25, 1, 0)
    
    # Signal 2: Volume Surge
    today_df['Score'] += np.where(today_df['volume'] > (today_df['vol_20d'] * 2), 1, 0)
    
    # Signal 3: Price Momentum
    today_df['Score'] += np.where(today_df['close'] > today_df['close_prev'], 1, 0)
    today_df['Score'] -= np.where(today_df['close'] < (today_df['close_prev'] * 0.98), 1, 0)
    
    # Signal 4: Bulk Deals (Check if named institution bought today)
    bulk_buyers = []
    if BULK_FILE.exists():
        bulk = pd.read_csv(BULK_FILE)
        today_bulk = bulk[bulk['date'] == str(latest_date.date())]
        bulk_buyers = today_bulk['SYMBOL'].unique() if 'SYMBOL' in today_bulk.columns else []
    
    today_df['Score'] += np.where(today_df['symbol'].isin(bulk_buyers), 1, 0)
    
    # Labeling
    conditions = [
        today_df['Disqualified'] == True,
        today_df['Score'] >= 3,
        today_df['Score'] <= -1
    ]
    choices = ['🔴 Flagged (Unsafe)', '🟢 Bullish', '🟠 Bearish']
    today_df['Label'] = np.select(conditions, choices, default='⚪ Neutral')
    
    # Format and Save
    final_cols = ['symbol', 'close', 'delivery_pct', 'volume', 'turnover', 'Score', 'Label']
    final_df = today_df[final_cols].sort_values(['Score', 'turnover'], ascending=[False, False])
    
    final_df.to_csv(RESULTS_DIR / "latest_signals.csv", index=False)
    print(f"✅ Generated {len(final_df)} signals for {latest_date.date()}")

if __name__ == "__main__":
    calculate_signals()
