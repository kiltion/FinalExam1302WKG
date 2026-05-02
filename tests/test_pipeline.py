import os
import pandas as pd

def test_directory_structure():
    """Check if all required folders exist"""
    dirs = ['data/raw/prices', 'data/raw/fred', 'data/processed', 'reports', 'configs']
    for d in dirs:
        assert os.path.exists(d), f"Directory {d} is missing!"

def test_data_integrity():
    """Verify the feature file was created and has data"""
    feature_path = "data/processed/final_features.csv"
    assert os.path.exists(feature_path), "Final features file not found!"
    
    df = pd.read_csv(feature_path)
    assert not df.empty, "Feature file is empty!"
    assert 'target' in df.columns, "Target column missing from features!"
    # Ensure all 30 stocks are present (or at least most, depending on data quality)
    assert df['ticker'].nunique() > 25, "Too many tickers missing from final data!"

if __name__ == "__main__":
    print("Running basic pipeline tests...")
    try:
        test_directory_structure()
        test_data_integrity()
        print("✅ All tests passed!")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")