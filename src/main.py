import yfinance as yf
import os
import requests
import pandas as pd

def setup_directory_structure():
    # The list of directories required by the project specs
    directories = [
        'src', 
        'tests', 
        'configs', 
        'data/raw/prices', 
        'data/raw/fred', 
        'data/processed', 
        'reports'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

# 1. Load tickers from the specific config file path required[cite: 1]
def load_tickers(path='configs/dow30.txt'):
    if not os.path.exists(path):
        # Fallback/Error handling if you haven't made the file yet
        print(f"Error: {path} not found.")
        return []
    with open(path, 'r') as f:
        # Handles commas or newlines
        return [t.strip() for t in f.read().replace(',', '\n').splitlines() if t.strip()]

def download_stock_data():
    tickers = load_tickers()
    raw_path = "data/raw/prices/"
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        
    for ticker in tickers:
        file_path = f"{raw_path}{ticker}.csv"
        if os.path.exists(file_path):
            print(f"Skipping {ticker}, already cached.")
            continue
            
        print(f"Downloading {ticker}...")
        # Pulling data for the timeframe[cite: 1]
        data = yf.download(ticker, start="2015-01-01", end="2024-01-01")
        data.to_csv(file_path)

def download_fred_data():
    API_KEY = "253ed19af907aa7403ab34cd685cc880" 
    MACRO_SERIES = ["FEDFUNDS", "UNRATE", "CPIAUCSL", "VIXCLS"]
    raw_path = "data/raw/fred/"
    
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        
    for series_id in MACRO_SERIES:
        file_path = f"{raw_path}{series_id}.csv"
        if os.path.exists(file_path):
            print(f"Skipping {series_id}, already cached.")
            continue
            
        print(f"Downloading {series_id} from FRED...")
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={API_KEY}&file_type=json"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()['observations']
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df[['date', 'value']].set_index('date')
            df.to_csv(file_path)
        else:
            print(f"Failed to download {series_id}.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def run_experiment(df, features, model_type='rf'):
    tscv = TimeSeriesSplit(n_splits=3)
    results = {'auc': [], 'accuracy': [], 'precision': [], 'recall': []}
    
    # ... (your existing split and scaling logic)
    
    model.fit(X_train_scaled, y_train)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    preds = model.predict(X_test_scaled) # Need this for accuracy/precision
    
    results['auc'].append(roc_auc_score(y_test, probs))
    results['accuracy'].append(accuracy_score(y_test, preds))
    results['precision'].append(precision_score(y_test, preds))
    results['recall'].append(recall_score(y_test, preds))
    
    # Return means and the full list for your "metric-by-fold" table
    return results

from features import build_features
from models import perform_ablation_study, get_model_results_for_viz
from visualization import create_visualizations
from visualization import plot_prediction_overlay

if __name__ == "__main__":
    # 1. Build features
    build_features()
    
    # 2. Get results for the plots
    print("Training models for visualization...")
    viz_data = get_model_results_for_viz()
    
    # 3. Generate the ROC plots
    create_visualizations(viz_data)
    
    # 4. Run the ablation study
    impact = perform_ablation_study()
    print(f"Ablation Impact: {impact:.4f}")