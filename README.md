# Stock Prediction — Multi-Ticker General Model

A Jupyter Notebook pipeline for collecting multi-ticker daily stock data, performing feature engineering (technical indicators), and preparing datasets for both regression (next-day price) and classification (price movement) tasks.

## Table of contents
- Project overview
- Notebook overview
- Key features
- Requirements
- Installation
- Usage
  - Open and run the notebook
  - Example: change tickers / period
- Inspect the prepared dataset
- Notes & next steps
- Contributing
- Contact

## Project overview
This repository contains a single notebook (Stock_prediction_project2.ipynb) that demonstrates:
- Downloading historical daily stock data for multiple tickers using yfinance.
- Combining per-ticker data into a single DataFrame with a `Ticker` column.
- Basic data checks and cleaning.
- Creating targets for:
  - `Target_Price`: next-day Close (regression).
  - `Price_Trend`: binary label when next-day percent change exceeds a threshold (classification).
- Computing common technical indicators used as features.

## Notebook (high level)
The notebook performs the following steps:
1. Data collection
   - Example tickers: ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL", "TSLA", ...]
   - Uses `yfinance.download` to fetch N years of daily data per ticker and concatenates rows into one DataFrame with a `Ticker` column.
2. Data cleaning
   - Checks for nulls and duplicates and ensures `Date` is a column.
3. Feature engineering
   - Creates `Target_Price = Close.shift(-1)`
   - Creates `Price_Trend` as a binary label based on next-day percent change (configurable threshold, e.g., > 2%).
   - Adds technical indicators via `add_technical_indicators(df)`, including:
     - Moving Averages: MA_10, MA_20, MA_50, MA_200
     - RSI_14
     - MACD and MACD_signal
     - ATR_14
     - Volatility_20 (rolling std)
   - Drops rows with NaNs produced by rolling calculations.
4. Data preview
   - Displays head() of the prepared DataFrame and final shape for quick inspection.

## Key features
- Aggregates multiple tickers into a single dataset while preserving the `Ticker` column.
- Produces both regression and classification targets for next-day prediction.
- Computes a standard set of technical indicators in a reusable function — suitable for prototyping and data-prep prior to modeling.

## Requirements
- Python 3.8+
- Primary packages (installable via pip):
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - yfinance
  - scikit-learn (if you plan to train models)
  - streamlit (optional; imported in notebook)
  - ta (optional; commented in the notebook for advanced indicators)
  - jupyterlab or notebook

## Installation
Install the packages with pip:
```bash
pip install numpy pandas matplotlib seaborn yfinance scikit-learn streamlit ta jupyterlab
```
Or create a virtual environment and install dependencies from a requirements file (if you add one):
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Open the notebook:
1. Launch Jupyter:
```bash
jupyter notebook
# or
jupyter lab
```
2. Open `Stock_prediction_project2.ipynb` and run the cells in order.

Example: change tickers / period
Inside the notebook, change the tickers list and period before running the fetch cell:
```python
tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL", "TSLA"]
period = "5y"  # or use start/end dates: start="2018-01-01", end="2023-01-01"
```

## Inspect the prepared dataset
The notebook shows a preview of the prepared DataFrame, including:
- Columns: Date, Ticker, Open, High, Low, Close, Adj Close, Volume, technical indicators..., Target_Price, Price_Trend
- Final shape and an example head() for quick verification.

## Notes & next steps
- Threshold for `Price_Trend` is configurable — tune it for different strategies or markets.
- The notebook provides a data-prep pipeline only. For modeling, add train/test splitting, feature scaling, cross-validation, and model training cells (e.g., using scikit-learn or other libraries).
- Consider adding:
  - A requirements.txt for reproducibility.
  - Unit tests for the indicator functions.
  - A Streamlit demo to visualize predictions and data interactively.
  - Support for intraday data if needed (yfinance has limitations).

## Contributing
Contributions are welcome. Suggested improvements:
- Add a requirements.txt and environment setup instructions.
- Add more technical indicators and examples of model training.
- Add a small sample dataset or caching to speed up repeated runs in CI.

## Contact
Created by codewithRahul01 — feel free to open an issue or pull request for suggestions, bug fixes, or feature requests.
