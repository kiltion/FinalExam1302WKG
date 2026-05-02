import os
import pandas as pd
import numpy as np

def build_features():
    # 1. Get the absolute path to the 'FINAL EXAM' root directory
    # This finds the directory where features.py lives and goes up one level
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config_path = os.path.join(base_dir, 'configs', 'dow30.txt')
    prices_dir = os.path.join(base_dir, 'data', 'raw', 'prices')
    fred_dir = os.path.join(base_dir, 'data', 'raw', 'fred')
    output_path = os.path.join(base_dir, 'data', 'processed', 'final_features.csv')

    print(f"Base Directory identified as: {base_dir}")

    if not os.path.exists(config_path):
        print(f"CRITICAL: Config not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        tickers = [t.strip().upper() for t in f.read().replace(',', '\n').splitlines() if t.strip()]

    all_stock_data = []

    for ticker in tickers:
        file_path = os.path.join(prices_dir, f"{ticker}.csv")
        
        if not os.path.exists(file_path):
            # Troubleshooting: Print exactly where it's looking
            print(f"SKIPPING: File not found for {ticker} at {file_path}")
            continue
            
        try:
            # Detect the header dynamically to handle yfinance multi-rows
            df_raw = pd.read_csv(file_path, header=None, nrows=10)
            header_row = None
            for i, row in df_raw.iterrows():
                if 'Close' in row.values:
                    header_row = i
                    break
            
            if header_row is None:
                continue

            df = pd.read_csv(file_path, header=header_row, index_col=0, parse_dates=True)
            
            # Clean MultiIndex and Tickers[cite: 1]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).strip() for c in df.columns]
            
            # Standardize columns and convert to numeric[cite: 1]
            # Use 'Adj Close' as 'Close' if it exists for better return accuracy
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop footer rows or empty rows[cite: 1]
            df = df.dropna(subset=['Close'])

            # --- Feature Engineering ---[cite: 1]
            df['ticker'] = ticker
            df['r_t'] = np.log(df['Close'] / df['Close'].shift(1))
            df['rolling_mean_5'] = df['r_t'].rolling(5).mean()
            df['rolling_mean_20'] = df['r_t'].rolling(20).mean()
            df['rolling_vol_10'] = df['r_t'].rolling(10).std()
            df['rolling_vol_20'] = df['r_t'].rolling(20).std()
            df['volume_change'] = np.log(df['Volume'] / df['Volume'].shift(1))
            df['target'] = (df['r_t'].shift(-1) > 0).astype(int)

            all_stock_data.append(df)
            print(f"SUCCESS: {ticker} loaded with {len(df)} rows.")

        except Exception as e:
            print(f"Error loading {ticker}: {e}")

    if not all_stock_data:
        print("Empty dataset. Check the paths printed above.")
        return

    full_df = pd.concat(all_stock_data)

# 4. Integrate Macro Data with robust joining
    macro_files = ["FEDFUNDS", "UNRATE", "CPIAUCSL", "VIXCLS"]
    for series in macro_files:
        macro_path = os.path.join(fred_dir, f"{series}.csv")
        if os.path.exists(macro_path):
            m_df = pd.read_csv(macro_path, index_col=0, parse_dates=True)
            
            # Convert 'value' to numeric (handles '.' or ' ' in FRED files)
            m_df['value'] = pd.to_numeric(m_df['value'], errors='coerce')
            
            # Create the lag to prevent data leakage
            macro_series = m_df['value'].rename(f"{series}_lag1").shift(1)
            
            # Join and immediately forward fill so monthly data hits every daily row
            full_df = full_df.join(macro_series, how='left')
            full_df[f"{series}_lag1"] = full_df[f"{series}_lag1"].ffill()

    # --- NEW CLEANUP LOGIC ---
    # 1. Drop columns that are ENTIRELY empty first (don't let one bad file kill the rows)
    all_null_cols = full_df.columns[full_df.isnull().all()]
    if len(all_null_cols) > 0:
        print(f"WARNING: These columns were 100% NaN and were removed: {list(all_null_cols)}")
        full_df.drop(columns=all_null_cols, inplace=True)

    # 2. Only now drop rows with remaining NaNs (usually just the first 20-30 rows)
    print(f"Shape before final drop: {full_df.shape}")
    full_df.dropna(inplace=True)
    print(f"Shape after final drop: {full_df.shape}")
    
    full_df.to_csv(output_path)

    # Final cleanup: drop NaNs from rolling/lagging[cite: 1]
# ... (After your macro join loop) ...

    # DEBUG: Check which columns are empty before dropping
    print("--- Data Integrity Check ---")
    for col in full_df.columns:
        null_count = full_df[col].isnull().sum()
        null_pct = (null_count / len(full_df)) * 100
        print(f"Column {col}: {null_count} NaNs ({null_pct:.1f}%)")

    # CRITICAL FIX: If a column is 100% NaN, drop the column, NOT the rows
    # This prevents one bad FRED series from deleting your whole project
    bad_cols = [col for col in full_df.columns if full_df[col].isnull().all()]
    if bad_cols:
        print(f"WARNING: Dropping empty columns: {bad_cols}")
        full_df.drop(columns=bad_cols, inplace=True)

    # Now drop only the rows with NaNs (like the first 20 days of history)
    print(f"Finalizing... Shape before row drop: {full_df.shape}")
    full_df.dropna(inplace=True) 
    print(f"Finalizing... Shape after row drop: {full_df.shape}")

    full_df.to_csv("data/processed/final_features.csv")