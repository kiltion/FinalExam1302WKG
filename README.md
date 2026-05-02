Project Overview
This project builds a complete data-science pipeline in Python to predict whether Dow 30 stocks will go "up" tomorrow based on historical price data and macroeconomic indicators. The pipeline handles everything from data ingestion and feature engineering to model training and walk-forward evaluation.
This is my file structure
.
├── configs/          # dow30.txt (ticker list)
├── data/
│   ├── raw/          # Cached prices and FRED downloads
│   └── processed/    # Final feature-engineered dataset
├── src/              # Python source code (features.py, model.py, etc.)
├── tests/            # Unit tests for clean/testable code
├── reports/          # Final PDF report, plots, and ablation results
├── main.py           # Single command to run the full pipeline
└── README.md         # Instructions
Hello! this works with the Dow industrial average, Y finance, FEDFUNDS, UNRATE, CPIAUCSL, and VIXCLS
this is designed with lagging and a walk forward evaluation in mind.
The only thing that has to run is just /src/main.py
