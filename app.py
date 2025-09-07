import io

import yaml
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import mplfinance as mpf
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError, EntryNotFoundError
from urllib.error import HTTPError

from stodir.forecast import fetch_data, add_features, predict_next_day

REPO_ID = "AsifSayyed/stodir-forecast-model"
CONFIG_FILENAME = "config.yaml"
MODEL_FILENAME = "stodir_model.joblib"


@st.cache_resource
def load_config_and_model():
    """
    Loads the config and the pre-trained model from the Hugging Face Hub.
    The results are cached for performance.
    """
    try:
        # Download the config file
        config_path = hf_hub_download(repo_id=REPO_ID, filename=CONFIG_FILENAME)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Download the model file
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        model = joblib.load(model_path)
        return config, model

    except (RepositoryNotFoundError, EntryNotFoundError) as e:
        st.error(
            f"Error: Could not find the model or config file in the Hugging Face repository '{REPO_ID}'. "
            "Please ensure the repository and files exist and are public."
        )
        return None, None

    except HTTPError as e:
        st.error(
            "Error: A network issue occurred while trying to download the model. "
            "Please check your internet connection and try again."
        )
        return None, None

    except Exception as e:
        # A fallback for other unexpected errors (e.g., corrupt file, unpickling error)
        st.error(f"An unexpected error occurred while loading the model: {e}")
        return None, None


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

    config, model = load_config_and_model()
    # Using an assertion to ensure the model and config are loaded before proceeding.
    assert model is not None and config is not None, "Model or config failed to load. Cannot proceed."

    HORIZONS = config["features"]["horizons"]
    PREDICTORS = [f"{h}_day" for h in HORIZONS]

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
