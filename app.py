import io
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import mplfinance as mpf

from stodir.forecast import fetch_data, add_features, predict_next_day

MODEL_PATH = "stodir_model.joblib"
#FIXME: This should come from config
HORIZONS = [2, 5, 60, 250, 1000]
PREDICTORS = [f"{h}_day" for h in HORIZONS]


@st.cache_resource
def load_model():
    """Loads the pre-trained model from disk, cached for performance."""
    if not os.path.exists(MODEL_PATH):
        return None
    model = joblib.load(MODEL_PATH)
    return model


def plot_candlestick(data: pd.DataFrame, ticker: str):
    """Displays a candlestick chart for the last 30 days."""
    buffer = io.BytesIO()
    mpf.plot(
        data.tail(30), type="candle", style="charles",
        title=f"{ticker} - Last 30 Days", ylabel="Price", savefig=buffer,
    )
    buffer.seek(0)
    st.image(buffer, caption=f"{ticker} Candlestick Chart", use_column_width=True)


def plot_closing_price(data: pd.DataFrame, ticker: str):
    """Displays a line chart of the closing price."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data["close"], label="Closing Price")
    ax.set_title(f"{ticker} Closing Price History")
    ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
    st.pyplot(fig)


def show_disclaimer():
    """Displays a disclaimer at the bottom of the app."""
    st.markdown("""
    <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; border: 1px solid #f5c6cb;">
        <strong>Disclaimer:</strong> This tool is for educational purposes only and is not financial advice. 
        Predictions are based on a generalized model and historical trends.
    </div>
    """, unsafe_allow_html=True)


def main():
    """Streamlit entry point for StoDir."""
    st.title("StoDir - Stock Direction Forecasting")

    model = load_model()

    if model is None:
        st.error(f"Model file not found at '{MODEL_PATH}'. Please run the training pipeline first using `python train.py`.")
        return

    ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()

    if st.button("Forecast"):
        if not ticker:
            st.warning("Please enter a stock ticker.")
            return

        with st.spinner(f"Fetching data and making prediction for {ticker}..."):
            try:
                raw_data = fetch_data(ticker)
                featured_data = add_features(raw_data.copy(), horizons=HORIZONS)

                # Perform prediction using the loaded model
                prediction = predict_next_day(model, featured_data, PREDICTORS)

                if prediction == "up":
                    st.success(f"Prediction for {ticker}: The stock price is likely to go UP tomorrow. 📈")
                else:
                    st.error(f"Prediction for {ticker}: The stock price is likely to go DOWN tomorrow. 📉")

                st.subheader("Data Visualization")
                plot_closing_price(raw_data, ticker)
                plot_candlestick(raw_data, ticker)

            except ValueError as e:
                st.error(f"Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

    show_disclaimer()


if __name__ == "__main__":
    main()
