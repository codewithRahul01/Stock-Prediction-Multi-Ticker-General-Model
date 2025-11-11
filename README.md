# Stock-Prediction-Multi-Ticker-General-Model
A Jupyter notebook-based project that demonstrates multi-ticker stock data collection, feature engineering and preparation for both regression (next-day price) and classification (price movement) tasks. Data is collected from Yahoo Finance via yfinance and several technical indicators are computed to enrich the dataset.

Table of contents

Project overview
Notebook overview
Key features
Requirements
Installation
Usage
Open and run the notebook
Example: change tickers / period
Inspect prepared dataset
Notes & next steps
Contributing
License
Contact
Project overview This repository contains a notebook (Stock_prediction_project2.ipynb) that:

Downloads historical daily stock data for multiple tickers using yfinance.
Combines per-ticker data into a single dataframe.
Performs basic data checks and cleaning.
Creates targets for two tasks:
Target_Price: next-day close (regression)
Price_Trend: binary label if next-day change exceeds a threshold (classification)
Computes a set of technical indicators (moving averages, RSI, MACD, ATR, volatility) to use as features.
Notebook overview (high level)

Data collection
Tickers list (example): ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL", "TSLA", ...]
Uses yfinance.download to fetch 5 years of daily data per ticker and concatenates rows into one dataframe with a 'Ticker' column.
Data cleaning
Checks for nulls, duplicates and ensures Date is a column.
Feature engineering
Creates Target_Price as Close.shift(-1)
Creates Price_Trend as percent-change threshold (example > 2%)
Adds technical indicators via add_technical_indicators(df):
MA_10, MA_20, MA_50, MA_200
RSI_14
MACD and MACD_signal
ATR_14
Volatility_20 (rolling std)
Drops rows with NaNs created by rolling calculations.
Data preview
Shows head() of prepared dataframe and final shape.
Key features

Multi-ticker aggregation into a single dataset (Ticker column preserved)
Regression and classification target creation
Common technical indicators computed in a reusable function
Notebook is suitable as a data-prep and prototyping pipeline prior to model training or deployment
Requirements

Python 3.8+ recommended
Primary packages used (install via pip):
numpy
pandas
matplotlib
seaborn
yfinance
streamlit (imported in the notebook; optional)
scikit-learn (likely needed for modeling steps — install if training)
ta (optional; commented in notebook for advanced indicators)
jupyterlab or notebook
