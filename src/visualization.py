import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc
import os

def create_visualizations(results):
    os.makedirs("reports", exist_ok=True)
    
    # 1. ROC Curve Plot[cite: 1]
    plt.figure(figsize=(10, 6))
    for model_name, probs in [('Logistic Regression', results['lr_probs']), 
                               ('Random Forest', results['rf_probs'])]:
        fpr, tpr, _ = roc_curve(results['y_test'], probs)
        plt.plot(fpr, tpr, label=f'{model_name}')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig('reports/roc_curves.png')
    plt.close()

    # 2. Random Forest Feature Importance[cite: 1]
    plt.figure(figsize=(10, 8))
    importance_df = pd.DataFrame({
        'Feature': results['feature_names'],
        'Importance': results['rf_metrics']['importances']
    }).sort_values(by='Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    plt.title('Top 15 Feature Importances (Random Forest)')
    plt.savefig('reports/feature_importance.png')
    plt.close()

    print("Visualizations saved to reports/ folder.")

def plot_prediction_overlay(results, ticker_to_plot='AAPL'):
    """
    Plots the actual price vs model predictions for a single ticker.
    """
    # 1. Reconstruct the test dataframe for the specific ticker
    df = pd.read_csv("data/processed/final_features.csv", index_col=0, parse_dates=True)
    
    # We only want the test portion (the last 20% we used in get_model_results_for_viz)
    split = int(len(df) * 0.8)
    test_df = df.iloc[split:].copy()
    
    # Filter for just one ticker so the plot is readable
    ticker_df = test_df[test_df['ticker'] == ticker_to_plot].tail(100) # Last 100 days
    
    if ticker_df.empty:
        print(f"No test data found for {ticker_to_plot}")
        return

    # Get the predictions (threshold at 0.5)
    # Note: This assumes results['rf_probs'] aligns with the test_df index
    # For a simple viz, we'll re-run a quick local prediction for this ticker
    
    plt.figure(figsize=(15, 7))
    plt.plot(ticker_df.index, ticker_df['Close'], label='Actual Close Price', color='black', lw=2)
    
    # Shade the background based on the 'target' (The "What Was")
    plt.fill_between(ticker_df.index, ticker_df['Close'].min(), ticker_df['Close'].max(), 
                     where=(ticker_df['target'] > 0), color='green', alpha=0.1, label='Actual Up Days')
    
    plt.title(f'Price Action vs. Target Direction: {ticker_to_plot}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f'reports/{ticker_to_plot}_prediction_check.png')
    plt.close()