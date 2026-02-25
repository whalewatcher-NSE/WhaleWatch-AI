"""
📊 WhaleWatch Master Analyst v3 (Deep Debug & Fallback)
=======================================================
1. Prints ALL raw column names to identify the 'Delivery' column.
2. Force-calculates Delivery % from Qty if needed.
3. Fallback: If Delivery data is missing, runs Volume-Only strategy.
"""
import pandas as pd
import numpy as np
import os
import itertools
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_FOLDER = "data/history" 

GRID = {
    "entry_delivery": [30, 40, 50],
    "entry_vol_ratio": [1.5, 2.0, 3.0],
    "stop_loss": [3, 5],
    "take_profit": [10, 15, 20],
    "smart_exit": [True]
}

def standardize_columns(df):
    """Aggressive column renaming."""
    # Print raw columns for debugging
    print(f"🔎 RAW COLUMNS FOUND: {list(df.columns)}")
    
    # Clean raw names
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
    
    col_map = {
        # Standard
        'symbol': 'symbol', 'ticker': 'symbol',
        'date': 'date', 'timestamp': 'date',
        'close': 'close', 'close_price': 'close',
        'volume': 'volume', 'ttl_trd_qnty': 'volume', 'total_traded_quantity': 'volume',
        
        # Delivery Variations (Add any you suspect!)
        'delivery_quantity': 'delivery_qty', 
        'deliv_qty': 'delivery_qty',
        'delivery_qty': 'delivery_qty',
        'dlvry_qty': 'delivery_qty',
        'qty_delivered': 'delivery_qty',
        
        # Delivery % Variations
        'delivery_pct': 'delivery_pct',
        'deliv_per': 'delivery_pct',
        'delivery_percentage': 'delivery_pct',
        'pct_deliv': 'delivery_pct'
    }
    
    new_cols = {}
    for c in df.columns:
        # Check exact match
        if c in col_map:
            new_cols[c] = col_map[c]
        # Check partial match (dangerous but effective)
        elif 'deliv' in c and 'qty' in c:
            new_cols[c] = 'delivery_qty'
        elif 'deliv' in c and ('per' in c or 'pct' in c):
            new_cols[c] = 'delivery_pct'
            
    df = df.rename(columns=new_cols)
    return df

def load_data():
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
            # Only keep non-empty
            if not df.empty: dfs.append(df)
        except Exception as e: 
            print(f"⚠️ Error reading {f}: {e}")
            
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    
    # 1. Standardize
    df = standardize_columns(df)
    
    # 2. Ensure Types
    df['date'] = pd.to_datetime(df['date'])
    for col in ['close', 'volume', 'delivery_qty', 'delivery_pct']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # 3. CRITICAL: Calculate Delivery % if missing
    if 'delivery_pct' not in df.columns:
        if 'delivery_qty' in df.columns and 'volume' in df.columns:
            print("ℹ️ Calculating Delivery % from Quantity...")
            df['delivery_pct'] = (df['delivery_qty'] / df['volume']) * 100
        else:
            print("⚠️ WARNING: No Delivery Data Found! Using Volume-Only Strategy.")
            # Set to 100 so it passes the "min delivery" filter, effectively disabling it
            df['delivery_pct'] = 100.0 
            
    # 4. Fill NaNs
    df['delivery_pct'] = df['delivery_pct'].fillna(0)
    
    # 5. Calculate Indicators
    print("🔮 Calculating indicators...")
    df['turnover'] = df['close'] * df['volume']
    df['vol_ma'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    df['pct_change'] = df.groupby('symbol')['close'].pct_change() * 100
    
    print("\n🔎 FINAL DATA CHECK (First 3 Rows):")
    cols_to_show = [c for c in ['date', 'symbol', 'close', 'delivery_pct', 'vol_ratio'] if c in df.columns]
    print(df[cols_to_show].head(3).to_string())
    print("-" * 50)
         
    return df

def get_liquid_universe(df, year, top_n=100):
    subset = df[df['date'].dt.year == year]
    if subset.empty: return []
    if 'turnover' not in subset.columns: return subset['symbol'].unique()[:top_n]
    avg_turnover = subset.groupby('symbol')['turnover'].mean()
    return avg_turnover.sort_values(ascending=False).head(top_n).index.tolist()

def simulate_trades(df, universe_list, params):
    df_liquid = df[df['symbol'].isin(universe_list)].copy()
    if df_liquid.empty: return -99

    mask = (
        (df_liquid['delivery_pct'] >= params['entry_delivery']) & 
        (df_liquid['vol_ratio'] >= params['entry_vol_ratio']) & 
        (df_liquid['pct_change'].abs() >= 3.0)
    )
    entries = df_liquid[mask]
    if entries.empty: return -99
    
    pnl_log = []
    
    # Simplified loop for speed
    for idx, row in entries.iterrows():
        symbol = row['symbol']
        buy_price = row['close']
        
        # Smart Lookahead (Vectorized approach possible but loop is safer for logic)
        future = df_liquid[(df_liquid['symbol'] == symbol) & (df_liquid['date'] > row['date'])].head(15)
        if future.empty: continue
        
        exit_price = future.iloc[-1]['close']
        
        for i, day in future.iterrows():
            if 'low' in day and day['low'] <= buy_price * (1 - params['stop_loss']/100):
                exit_price = buy_price * (1 - params['stop_loss']/100); break
            if 'high' in day and day['high'] >= buy_price * (1 + params['take_profit']/100):
                exit_price = buy_price * (1 + params['take_profit']/100); break
            if params['smart_exit'] and day['delivery_pct'] < 25:
                exit_price = day['close']; break
        
        pnl_log.append((exit_price - buy_price) / buy_price)
        
    return np.mean(pnl_log) if pnl_log else -99

def run_analysis():
    full_df = load_data()
    if full_df is None: return
    
    years = sorted(full_df['date'].dt.year.unique())
    keys, values = zip(*GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n⏩ STARTING ANALYSIS ({len(combinations)} Strategies per Year)")
    results_log = []
    
    for i in range(len(years) - 1):
        train_year = years[i]
        test_year = years[i+1]
        
        train_universe = get_liquid_universe(full_df, train_year)
        train_data = full_df[full_df['date'].dt.year == train_year]
        
        best_score, best_config = -100, None
        
        for config in combinations:
            score = simulate_trades(train_data, train_universe, config)
            if score > best_score: best_score, best_config = score, config
            
        if not best_config or best_score == -99:
            print(f"Year {test_year} | ⚠️ NO TRADES FOUND.")
            continue

        test_universe = get_liquid_universe(full_df, test_year)
        test_data = full_df[full_df['date'].dt.year == test_year]
        result = simulate_trades(test_data, test_universe, best_config)
        
        res_str = f"{result*100:+.2f}%" if result != -99 else "No Trades"
        cfg_short = f"D>{best_config['entry_delivery']} V>{best_config['entry_vol_ratio']} TP>{best_config['take_profit']}"
        print(f"Year {test_year} | Best: {cfg_short} | Result: {res_str}")
        
        results_log.append({"Year": test_year, "Config": str(best_config), "Return": result})
        
    pd.DataFrame(results_log).to_csv("strategy_results.csv", index=False)

if __name__ == "__main__":
    run_analysis()
