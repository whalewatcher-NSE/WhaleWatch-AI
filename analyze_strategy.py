"""
📊 WhaleWatch Master Analyst (Backward + Forward)
=================================================
1. Loads 2021-2026 Parquet Data from data/history.
2. Filters for 'NSE 100 Proxy' (Top 100 Liquid Stocks per year).
3. Performs Walk-Forward Analysis (The "Reality" Test).
"""
import pandas as pd
import numpy as np
import os
import itertools
from tqdm import tqdm

# --- CONFIGURATION ---
# We look for parquet files in this folder
DATA_FOLDER = "data/history" 

# --- STRATEGY GRID ---
# The script will test every combination here (3 x 3 x 2 x 3 = 54 combos)
GRID = {
    "entry_delivery": [30, 40, 50],       # Try buying at 30%, 40%, 50% delivery
    "entry_vol_ratio": [1.5, 2.0, 3.0],   # Try Volume 1.5x, 2x, 3x
    "stop_loss": [3, 5],                  # Tight vs Loose Stop Loss
    "take_profit": [10, 15, 20],          # Target profit
    "smart_exit": [True]                  # Always test "Whale Bail" exit
}

def load_data():
    """Loads Parquet files from data/history."""
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Error: Folder '{DATA_FOLDER}' not found.")
        return None
        
    all_files = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith('.parquet')]
    print(f"📂 Loading {len(all_files)} Parquet files from {DATA_FOLDER}...")
    
    dfs = []
    for f in tqdm(all_files):
        try:
            # Requires pyarrow installed
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e: 
            print(f"⚠️ Skipping {f}: {e}")
            
    if not dfs: 
        print("❌ No valid parquet files found.")
        return None
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Normalize Columns
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Ensure Date format
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # ----------------------------------------------------
    # CALCULATE INDICATORS ONCE (Vectorized = Fast)
    # ----------------------------------------------------
    print("🔮 Pre-calculating indicators...")
    
    # Turnover for NSE 100 Filter (Price * Volume)
    df['turnover'] = df['close'] * df['volume']
    
    # Strategy Indicators
    df['vol_ma'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    df['pct_change'] = df.groupby('symbol')['close'].pct_change() * 100
    
    # Handle Delivery
    if 'delivery_pct' not in df.columns and 'delivery_qty' in df.columns:
         df['delivery_pct'] = (df['delivery_qty'] / df['volume']) * 100
         
    return df

def get_liquid_universe(df, year, top_n=100):
    """
    Returns the list of Top 100 stocks by average turnover for a specific year.
    This effectively filters out penny stocks and illiquid names.
    """
    subset = df[df['date'].dt.year == year]
    if subset.empty: return []
    
    avg_turnover = subset.groupby('symbol')['turnover'].mean()
    return avg_turnover.sort_values(ascending=False).head(top_n).index.tolist()

def simulate_trades(df, universe_list, params):
    """
    Runs the strategy ONLY on the liquid universe for that period.
    """
    # 1. Filter: Only trade stocks in the "NSE 100 Proxy" list
    df_liquid = df[df['symbol'].isin(universe_list)].copy()
    
    if df_liquid.empty: return -99

    # 2. Entry Signal
    mask = (
        (df_liquid['delivery_pct'] >= params['entry_delivery']) & 
        (df_liquid['vol_ratio'] >= params['entry_vol_ratio']) & 
        (df_liquid['pct_change'].abs() >= 3.0)
    )
    entries = df_liquid[mask]
    
    if entries.empty: return -99
    
    pnl_log = []
    
    # 3. Simulate Trade Lifecycle
    # We iterate because "Path" matters (Stop Loss vs Take Profit)
    for idx, row in entries.iterrows():
        symbol = row['symbol']
        entry_date = row['date']
        buy_price = row['close']
        
        # Get next 15 days of data for this stock
        future = df_liquid[(df_liquid['symbol'] == symbol) & (df_liquid['date'] > entry_date)].head(15)
        
        if future.empty: continue
        
        exit_price = future.iloc[-1]['close'] # Default Exit (Time Limit)
        
        for i, day in future.iterrows():
            # Stop Loss (Did Low hit it?)
            if day['low'] <= buy_price * (1 - params['stop_loss']/100):
                exit_price = buy_price * (1 - params['stop_loss']/100)
                break
            # Take Profit (Did High hit it?)
            if day['high'] >= buy_price * (1 + params['take_profit']/100):
                exit_price = buy_price * (1 + params['take_profit']/100)
                break
            # Smart Exit (Whale Bail: Delivery drops < 25%)
            if params['smart_exit'] and day['delivery_pct'] < 25:
                exit_price = day['close']
                break
        
        pnl = (exit_price - buy_price) / buy_price
        pnl_log.append(pnl)
        
    return np.mean(pnl_log) if pnl_log else -99

def run_analysis():
    # 1. Load Data
    full_df = load_data()
    if full_df is None: return
    
    years = sorted(full_df['date'].dt.year.unique())
    print(f"📅 Data covers years: {years}")
    
    # Generate Combinations
    keys, values = zip(*GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"⚙️ Testing {len(combinations)} strategy variations per year...")

    print("\n⏩ STARTING WALK-FORWARD ANALYSIS")
    print("=" * 80)
    print(f"{'Year':<6} | {'Best Strategy (Learned from Prev Year)':<45} | {'Result':<10}")
    print("-" * 80)
    
    cumulative_growth = 1.0
    
    # Iterate through years (Train on Year X, Test on Year X+1)
    for i in range(len(years) - 1):
        train_year = years[i]
        test_year = years[i+1]
        
        # --- STEP 1: TRAINING (Optimize on Past) ---
        train_universe = get_liquid_universe(full_df, train_year, top_n=100)
        train_data = full_df[full_df['date'].dt.year == train_year]
        
        best_score = -100
        best_config = None
        
        # Quick Grid Search
        # (In a real scenario, we might use fewer combinations to save time)
        for config in combinations:
            score = simulate_trades(train_data, train_universe, config)
            if score > best_score:
                best_score = score
                best_config = config
        
        if best_config is None:
            print(f"{test_year:<6} | No valid trades found in training.")
            continue

        # --- STEP 2: TESTING (Trade on Future) ---
        test_universe = get_liquid_universe(full_df, test_year, top_n=100)
        test_data = full_df[full_df['date'].dt.year == test_year]
        
        # Run the "Best Config" on the "Test Year"
        result = simulate_trades(test_data, test_universe, best_config)
        
        # Format Output
        cfg_str = f"D>{best_config['entry_delivery']} V>{best_config['entry_vol_ratio']} TP>{best_config['take_profit']} SL>{best_config['stop_loss']}"
        res_str = f"{result*100:+.2f}%" if result != -99 else "No Trades"
        
        print(f"{test_year:<6} | {cfg_str:<45} | {res_str:<10}")
        
        if result != -99:
            cumulative_growth *= (1 + result)

    print("=" * 80)
    print(f"💰 CUMULATIVE RETURN (Compounded): {(cumulative_growth - 1)*100:.2f}%")
    print("   (This is the realistic result of adapting your strategy yearly)")

if __name__ == "__main__":
    run_analysis()
