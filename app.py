import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, accuracy_score
import matplotlib.pyplot as plt

# ==============================
# Paths for pretrained models
# ==============================
REGRESSOR_PATH = "stock_price_regressor.joblib"
CLASSIFIER_PATH = "stock_trend_classifier.joblib"

# ==============================
# Utility Functions
# ==============================
def load_model(path, model_type="regressor"):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Pretrained {model_type} model not found!")
        return None

def download_stock_data(tickers):
    all_data = []
    for ticker in tickers:
        st.write(f"Downloading data for {ticker} ...")
        df = yf.download(
            ticker,
            period="5y",
            interval="1d",
            multi_level_index=False
        )
        if df.empty:
            st.warning(f"No data found for {ticker}. Skipping.")
            continue
        df['Ticker'] = ticker
        df.reset_index(inplace=True)
        all_data.append(df)
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def add_technical_indicators(df):
    data = df.copy()
    for window in [10, 20, 50, 200]:
        data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    RS = roll_up / (roll_down + 1e-9)
    data['RSI_14'] = 100 - (100 / (1 + RS))
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR_14'] = tr.rolling(window=14).mean()
    data['Volatility_20'] = data['Close'].rolling(window=20).std()
    data = data.dropna().reset_index(drop=True)
    return data

def feature_engineering(df, tickers):
    df = add_technical_indicators(df)
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['RollingMean'] = df['Close'].rolling(window=3).mean()
    df['RollingStd'] = df['Close'].rolling(window=3).std()
    df['Target_Price'] = df['Close'].shift(-1)
    df['Price_Trend'] = ((df['Target_Price'] - df['Close']) / df['Close'] > 0.02).astype(int)
    df.dropna(inplace=True)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df[['Ticker']])
    ticker_encoded = encoder.transform(df[['Ticker']])
    ticker_df = pd.DataFrame(
        ticker_encoded,
        columns=encoder.get_feature_names_out(['Ticker']),
        index=df.index
    )
    df = pd.concat([df, ticker_df], axis=1)
    return df, encoder

def align_features(X, model):
    """Aligns X's columns to match the model's expected features."""
    if hasattr(model, "feature_names_in_"):
        expected = model.feature_names_in_
        # Add missing columns as zeros
        for col in expected:
            if col not in X.columns:
                X[col] = 0
        # Drop extra columns
        X = X[expected]
    return X

def plot_predictions(actual, predicted, title="Actual vs Predicted Prices"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(actual, label="Actual", color="blue")
    ax.plot(predicted, label="Predicted", color="orange")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

def plot_trend(df, pred_trend=None, window=50):
    # Show only the last `window` rows for clarity
    df_small = df.tail(window).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_small['Close'], label='Close Price', color='gray', linewidth=2)
    up = df_small[df_small['Price_Trend'] == 1]
    down = df_small[df_small['Price_Trend'] == 0]
    ax.scatter(up.index, up['Close'], color='limegreen', label='Actual Up', marker='^', s=80, edgecolor='black')
    ax.scatter(down.index, down['Close'], color='tomato', label='Actual Down', marker='v', s=80, edgecolor='black')
    if pred_trend is not None:
        pred_up = df_small[pred_trend[-window:] == 1]
        pred_down = df_small[pred_trend[-window:] == 0]
        ax.scatter(pred_up.index, pred_up['Close'], color='blue', label='Predicted Up', marker='^', s=40, alpha=0.6)
        ax.scatter(pred_down.index, pred_down['Close'], color='purple', label='Predicted Down', marker='v', s=40, alpha=0.6)
    ax.set_title('Actual vs Predicted Stock Trend (Last {} days)'.format(window))
    ax.set_xlabel('Index')
    ax.set_ylabel('Close Price')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    st.pyplot(fig)

def classification_report_to_df(y_true, y_pred):
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    # Only keep relevant rows (0, 1, accuracy, macro avg, weighted avg)
    keep_rows = [str(i) for i in sorted(set(y_true))] + ['accuracy', 'macro avg', 'weighted avg']
    df_report = df_report.loc[keep_rows]
    return df_report

# ==============================
# Main Streamlit App
# ==============================
st.set_page_config(page_title="Multi-Ticker Stock Predictor", page_icon="📈", layout="wide")
st.title("📈 Multi-Ticker Stock Prediction & Fine-Tuning")

st.markdown("""
**🧠 Concept:** Pretrained Multi-Ticker Model + Fine-Tuning  
Train on many tickers, then fine-tune for new ones — just like GPT/BERT for stocks!
""")

regressor = load_model(REGRESSOR_PATH, "regressor")
classifier = load_model(CLASSIFIER_PATH, "classifier")

tickers_input = st.text_input("Enter Ticker Symbols (comma-separated):", value="AAPL, TSLA, RELIANCE.NS")
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

features = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Lag1', 'Lag2', 'RollingMean', 'RollingStd',
    'MA_10', 'MA_20', 'MA_50', 'MA_200', 'RSI_14', 'MACD', 'MACD_signal', 'ATR_14', 'Volatility_20'
]

st.header("🔮 Predict Stock Prices & Trends")
if st.button("Predict"):
    if not regressor or not classifier:
        st.error("Model(s) not loaded. Please check your files.")
    else:
        df = download_stock_data(tickers)
        if df.empty:
            st.error("No valid data downloaded for the given tickers.")
        else:
            st.write("### Downloaded Data Preview")
            st.dataframe(df.head())
            df_prepared, encoder = feature_engineering(df, tickers)
            if df_prepared.empty:
                st.error("Not enough data after feature engineering.")
            else:
                features_full = features + list(encoder.get_feature_names_out(['Ticker']))
                X = df_prepared[features_full]
                X = align_features(X, regressor)
                y = df_prepared['Close']
                # Regression prediction
                preds = regressor.predict(X)
                mse = mean_squared_error(y, preds)
                mae = mean_absolute_error(y, preds)
                r2 = r2_score(y, preds)
                st.write("### Price Prediction Metrics")
                st.write(f"- MSE: {mse:.2f}")
                st.write(f"- MAE: {mae:.2f}")
                st.write(f"- R²: {r2:.2f}")
                plot_predictions(y.values, preds)
                df_prepared['Predicted_Price'] = preds
                # Classification prediction
                X_cls = align_features(X, classifier)
                trend_preds = classifier.predict(X_cls)
                acc = accuracy_score(df_prepared['Price_Trend'], trend_preds)
                st.write("### Trend Classification Metrics")
                st.write(f"- Accuracy: {acc:.2f}")

                # Show classification report as a table
                df_cls_report = classification_report_to_df(df_prepared['Price_Trend'], trend_preds)
                st.write("#### Classification Report")
                st.dataframe(df_cls_report.style.format("{:.2f}"))

                df_prepared['Predicted_Trend'] = trend_preds
                plot_trend(df_prepared, pred_trend=trend_preds)
                date_col = 'Date' if 'Date' in df_prepared.columns else df_prepared.index
                st.write("### Recent Predictions")
                st.dataframe(df_prepared[[date_col, 'Ticker', 'Close', 'Predicted_Price', 'Price_Trend', 'Predicted_Trend']].tail(10))

                # --- Tomorrow's Prediction Section ---
                st.subheader("📅 Tomorrow's Prediction & Suggestion")
                tomorrow_results = []
                for ticker in tickers:
                    df_ticker = df_prepared[df_prepared['Ticker'] == ticker]
                    if not df_ticker.empty:
                        last_row = df_ticker.iloc[-1]
                        tomorrow_price = last_row['Predicted_Price']
                        tomorrow_trend = last_row['Predicted_Trend']
                        action = "Hold" if tomorrow_trend == 1 else "Sell"
                        trend_text = "Up" if tomorrow_trend == 1 else "Down"
                        tomorrow_results.append({
                            "Ticker": ticker,
                            "Tomorrow Price": f"{tomorrow_price:.2f}",
                            "Trend": trend_text,
                            "Suggestion": action
                        })
                if tomorrow_results:
                    st.table(pd.DataFrame(tomorrow_results))
                else:
                    st.info("No tomorrow prediction available for the selected tickers.")

st.header("🔁 Fine-Tune Model on New Tickers")
if st.button("Fine-Tune on New Data"):
    if not regressor or not classifier:
        st.error("Model(s) not loaded. Please check your files.")
    else:
        df_new = download_stock_data(tickers)
        if df_new.empty:
            st.error("No valid data downloaded for the given tickers.")
        else:
            st.write("### New Data for Fine-Tuning")
            st.dataframe(df_new.head())
            df_new_prepared, encoder = feature_engineering(df_new, tickers)
            if df_new_prepared.empty:
                st.error("Not enough data after feature engineering.")
            else:
                features_full = features + list(encoder.get_feature_names_out(['Ticker']))
                X_new = df_new_prepared[features_full]
                y_new = df_new_prepared['Close']
                z_new = df_new_prepared['Price_Trend']
                fine_tuned = False
                errors = []
                # Fine-tune regressor
                try:
                    if hasattr(regressor, "partial_fit"):
                        X_new_reg = align_features(X_new, regressor)
                        regressor.partial_fit(X_new_reg, y_new)
                        fine_tuned = True
                    else:
                        errors.append("Regressor does not support partial_fit. Retrain externally if needed.")
                except Exception as e:
                    errors.append(f"Regressor error: {e}")
                # Fine-tune classifier
                try:
                    if hasattr(classifier, "partial_fit"):
                        X_new_cls = align_features(X_new, classifier)
                        classifier.partial_fit(X_new_cls, z_new)
                        fine_tuned = True
                    else:
                        errors.append("Classifier does not support partial_fit. Retrain externally if needed.")
                except Exception as e:
                    errors.append(f"Classifier error: {e}")
                if fine_tuned:
                    st.success("Model fine-tuned successfully on new data!")
                    joblib.dump(regressor, REGRESSOR_PATH)
                    joblib.dump(classifier, CLASSIFIER_PATH)
                else:
                    for err in errors:
                        st.warning(err)

st.info("💡 Tip: The pretrained model is like GPT — trained on many tickers, fine-tuned for specific stocks later.")