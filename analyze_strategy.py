"""
📊 WhaleWatch Master Analyst v2 (Bulletproof)
=============================================
1. Auto-detects column names (handles NSE raw formats).
2. Auto-scales Delivery % (0.5 -> 50.0).
3. DEBUG MODE: Prints sample data to the logs so you can see what's wrong.
"""
import pandas as pd
import numpy as np
import os
import itertools
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_FOLDER = "data/history" 

# --- STRATEGY GRID (54 Combinations) ---
GRID = {
    "entry_delivery": [30, 40, 50],
    "entry_vol_ratio": [1.5, 2.0, 3.0],
    "stop_loss": [3, 5],
    "take_profit": [10, 15, 20],
    "smart_exit": [True]
}

# --- SMART COLUMN MAPPING ---
def standardize_columns(df):
    """Renames varied column names to standard internal names."""
    # Map Common Variations -> Standard Name
    col_map = {
        # Date
        'date': 'date', 'timestamp': 'date', 'datetime': 'date', 'date1': 'date',
        # Symbol
        'symbol': 'symbol', 'ticker': 'symbol', 'instrument': 'symbol',
        # Prices
        'close': 'close', 'close_price': 'close', 'close price': 'close', 'clspric': 'close',
        'open': 'open', 'open_price': 'open',
        'high': 'high', 'high_price': 'high',
        'low': 'low', 'low_price': 'low',
        'prev_close': 'prev_close', 'prevclose': 'prev_close',
        # Volume
        'volume': 'volume', 'ttl_trd_qnty': 'volume', 'tottrdqty': 'volume', 'quantity': 'volume',
        # Delivery
        'delivery_pct': 'delivery_pct', 'deliv_per': 'delivery_pct', 'delivery_percentage': 'delivery_pct',
        'delivery_qty': 'delivery_qty', 'deliv_qty': 'delivery_qty', 'delivery quantity': 'delivery_qty'
    }
    
    # Normalize input cols to lowercase for matching
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Rename
    new_cols = {}
    for c in df.columns:
        if c in col_map:
            new_cols[c] = col_map[c]
    
    df = df.rename(columns=new_cols)
    return df

def load_data():
    """Robust Data Loader."""
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Error: Folder '{DATA_FOLDER}' not found.")
        return None
        
    all_files = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith(('.parquet', '.csv'))]
    print(f"📂 Found {len(all_files)} files. Loading...")
    
    dfs = []
    for f in tqdm(all_files):
        try:
            if f.endswith('.parquet'): df = pd.read_parquet(f)
            else: df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e: 
            print(f"⚠️ Error reading {f}: {e}")
            
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    
    # 1. Standardize Names
    df = standardize_columns(df)
    
    # 2. Validation Check (Debug)
    required = ['symbol', 'date', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ CRITICAL ERROR: Missing columns {missing}.")
        print(f"   Found columns: {list(df.columns)}")
        return None

    # 3. Ensure Types
    df['date'] = pd.to_datetime(df['date'])
    for col in ['close', 'volume', 'high', 'low']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # 4. Handle Delivery
    if 'delivery_pct' not in df.columns:
        if 'delivery_qty' in df.columns:
            df['delivery_pct'] = (df['delivery_qty'] / df['volume']) * 100
        else:
            print("⚠️ Warning: No Delivery Data found! Strategy will fail.")
            df['delivery_pct'] = 0.0
            
    # 5. Fix Scale (0.5 vs 50.0)
    # If max delivery is 1.0, it's decimal. Convert to percent.
    if df['delivery_pct'].max() <= 1.0:
        print("ℹ️ Detected decimal delivery (0.5), converting to percent (50.0)...")
        df['delivery_pct'] = df['delivery_pct'] * 100

    # 6. Calculate Indicators
    print("🔮 Calculating indicators...")
    df['turnover'] = df['close'] * df['volume']
    df['vol_ma'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    df['pct_change'] = df.groupby('symbol')['close'].pct_change() * 100
    
    # DEBUG PRINT
    print("\n🔎 DATA CHECK (First 3 Rows):")
    print(df[['date', 'symbol', 'close', 'delivery_pct', 'vol_ratio']].head(3).to_string())
    print("-" * 50)
         
    return df

def get_liquid_universe(df, year, top_n=100):
    subset = df[df['date'].dt.year == year]
    if subset.empty: return []
    avg_turnover = subset.groupby('symbol')['turnover'].mean()
    return avg_turnover.sort_values(ascending=False).head(top_n).index.tolist()

def simulate_trades(df, universe_list, params):
    # Filter for Liquid Stocks Only
    df_liquid = df[df['symbol'].isin(universe_list)].copy()
    if df_liquid.empty: return -99

    # Entry Signal
    mask = (
        (df_liquid['delivery_pct'] >= params['entry_delivery']) & 
        (df_liquid['vol_ratio'] >= params['entry_vol_ratio']) & 
        (df_liquid['pct_change'].abs() >= 3.0)
    )
    entries = df_liquid[mask]
    
    # DEBUG: Only print this once to see if signals are firing
    if len(entries) == 0 and params['entry_delivery'] == 30: 
        # Only warn on the loosest strategy
        return -99

    if entries.empty: return -99
    
    pnl_log = []
    
    for idx, row in entries.iterrows():
        symbol = row['symbol']
        buy_price = row['close']
        
        # Look forward 15 days
        future = df_liquid[(df_liquid['symbol'] == symbol) & (df_liquid['date'] > row['date'])].head(15)
        if future.empty: continue
        
        exit_price = future.iloc[-1]['close']
        
        for i, day in future.iterrows():
            if day['low'] <= buy_price * (1 - params['stop_loss']/100):
                exit_price = buy_price * (1 - params['stop_loss']/100); break
            if day['high'] >= buy_price * (1 + params['take_profit']/100):
                exit_price = buy_price * (1 + params['take_profit']/100); break
            if params['smart_exit'] and day['delivery_pct'] < 25:
                exit_price = day['close']; break
        
        pnl_log.append((exit_price - buy_price) / buy_price)
        
    return np.mean(pnl_log) if pnl_log else -99

def run_analysis():
    full_df = load_data()
    if full_df is None: return
    
    years = sorted(full_df['date'].dt.year.unique())
    print(f"📅 Data Years Found: {years}")
    
    keys, values = zip(*GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n⏩ STARTING ANALYSIS ({len(combinations)} Strategies per Year)")
    
    results_log = []
    cumulative_growth = 1.0
    
    for i in range(len(years) - 1):
        train_year = years[i]
        test_year = years[i+1]
        
        # Train
        train_universe = get_liquid_universe(full_df, train_year)
        train_data = full_df[full_df['date'].dt.year == train_year]
        
        best_score, best_config = -100, None
        
        for config in combinations:
            score = simulate_trades(train_data, train_universe, config)
            if score > best_score: best_score, best_config = score, config
            
        if not best_config or best_score == -99:
            print(f"Year {test_year} | ⚠️ NO TRADES FOUND during training.")
            continue

        # Test
        test_universe = get_liquid_universe(full_df, test_year)
        test_data = full_df[full_df['date'].dt.year == test_year]
        result = simulate_trades(test_data, test_universe, best_config)
        
        # Log
        res_str = f"{result*100:+.2f}%" if result != -99 else "No Trades"
        cfg_short = f"D>{best_config['entry_delivery']} V>{best_config['entry_vol_ratio']} TP>{best_config['take_profit']}"
        print(f"Year {test_year} | Best: {cfg_short} | Result: {res_str}")
        
        results_log.append({
            "Year": test_year,
            "Strategy": str(best_config),
            "Annual Return": result
        })
        
        if result != -99: cumulative_growth *= (1 + result)

    print(f"\n💰 CUMULATIVE RETURN: {(cumulative_growth - 1)*100:.2f}%")
    pd.DataFrame(results_log).to_csv("strategy_results.csv", index=False)

if __name__ == "__main__":
    run_analysis()
