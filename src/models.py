import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import os

def run_experiment(df, features, model_type='rf'):
    """Helper to run a single walk-forward experiment"""
    tscv = TimeSeriesSplit(n_splits=3) # Required for 'metric-by-fold'
    fold_aucs = []
    
    X = df[features]
    y = df['target']
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Preprocessing: Fit ONLY on training
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = LogisticRegression(max_iter=1000)
            
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        fold_aucs.append(roc_auc_score(y_test, probs))
        
    return np.mean(fold_aucs), fold_aucs

def get_model_results_for_viz():
    df = pd.read_csv("data/processed/final_features.csv", index_col=0, parse_dates=True).sort_index()
    df_numeric = df.select_dtypes(include=[np.number])
    features = [col for col in df_numeric.columns if col != 'target']
    
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])
    
    lr = LogisticRegression(max_iter=1000).fit(X_train, train['target'])
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, train['target'])
    
    # 2. Return the data structured EXACTLY for the visualization script
    return {
        'lr_probs': lr.predict_proba(X_test)[:, 1],
        'rf_probs': rf.predict_proba(X_test)[:, 1],
        'y_test': test['target'].values,
        'feature_names': features,
        'rf_metrics': {
            'importances': rf.feature_importances_
        }
    }

def perform_ablation_study():
    # Load and sort by date to respect time-series order
    df = pd.read_csv("data/processed/final_features.csv", index_col=0, parse_dates=True).sort_index()
    
    print(f"Dataset loaded. Shape: {df.shape}") # Debugging line
    if df.empty:
        raise ValueError("The feature file is empty. Check your join/dropna logic in features.py.")

    # Only use numeric columns for training
    # This automatically excludes 'ticker' and non-numeric junk
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Define feature groups
    all_features = [col for col in df_numeric.columns if col not in ['target']]
    macro_features = [col for col in all_features if "lag" in col] 
    non_macro_features = [col for col in all_features if col not in macro_features]
    
    # ... (inside perform_ablation_study) ...
    print(f"--- Starting Ablation Study with {len(all_features)} features ---")
    
    # 1. Full Model (All features)
    full_auc, full_folds = run_experiment(df, all_features, 'rf')
    print(f"Full Model Mean AUC: {full_auc:.4f}")
    
    # 2. Ablated Model (No Macro features)
    ablated_auc, ablated_folds = run_experiment(df, non_macro_features, 'rf')
    print(f"Ablated Model (No Macro) Mean AUC: {ablated_auc:.4f}")
    
    # Save results for the report
    results_df = pd.DataFrame({
        'Fold': [1, 2, 3],
        'Full_Model_AUC': full_folds,
        'No_Macro_AUC': ablated_folds
    })
    results_df.to_csv("reports/ablation_results.csv", index=False)
    
    # Return both the results for visualization and the impact score
    return full_auc - ablated_auc