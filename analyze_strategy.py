"""
📊 WhaleWatch Master Analyst (GitHub Actions Version)
=====================================================
1. Loads 2021-2026 Parquet Data from data/history.
2. Filters for 'NSE 100 Proxy' (Top 100 Liquid Stocks per year).
3. Performs Walk-Forward Analysis.
4. SAVES results to CSV for download.
"""
import pandas as pd
import numpy as np
import os
import itertools
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_FOLDER = "data/history" 

# --- STRATEGY GRID ---
GRID = {
    "entry_delivery": [30, 40, 50],
    "entry_vol_ratio": [1.5, 2.0, 3.0],
    "stop_loss": [3, 5],
    "take_profit": [10, 15, 20],
    "smart_exit": [True]
}

def load_data():
    """Loads Parquet files."""
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Error: Folder '{DATA_FOLDER}' not found.")
        return None
        
    all_files = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith('.parquet')]
    print(f"📂 Loading {len(all_files)} Parquet files...")
    
    dfs = []
    for f in tqdm(all_files):
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e: 
            print(f"⚠️ Skipping {f}: {e}")
            
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.lower().strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Pre-calc indicators
    print("🔮 Calculating indicators...")
    df['turnover'] = df['close'] * df['volume']
    df['vol_ma'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    df['pct_change'] = df.groupby('symbol')['close'].pct_change() * 100
    
    if 'delivery_pct' not in df.columns and 'delivery_qty' in df.columns:
         df['delivery_pct'] = (df['delivery_qty'] / df['volume']) * 100
         
    return df

def get_liquid_universe(df, year, top_n=100):
    subset = df[df['date'].dt.year == year]
    if subset.empty: return []
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
    for idx, row in entries.iterrows():
        symbol = row['symbol']
        buy_price = row['close']
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
    keys, values = zip(*GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print("\n⏩ STARTING WALK-FORWARD ANALYSIS")
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
            
        if not best_config: continue

        # Test
        test_universe = get_liquid_universe(full_df, test_year)
        test_data = full_df[full_df['date'].dt.year == test_year]
        result = simulate_trades(test_data, test_universe, best_config)
        
        # Log
        res_str = f"{result*100:+.2f}%" if result != -99 else "No Trades"
        print(f"Year {test_year} | Best: {best_config} | Result: {res_str}")
        
        results_log.append({
            "Year": test_year,
            "Strategy": str(best_config),
            "Annual Return": result
        })
        
        if result != -99: cumulative_growth *= (1 + result)

    print(f"\n💰 CUMULATIVE RETURN: {(cumulative_growth - 1)*100:.2f}%")
    
    # SAVE RESULTS
    pd.DataFrame(results_log).to_csv("strategy_results.csv", index=False)
    print("💾 Saved 'strategy_results.csv'")

if __name__ == "__main__":
    run_analysis()
