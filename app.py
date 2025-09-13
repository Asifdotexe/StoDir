import io

import yaml
import joblib
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import mplfinance as mpf
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError, EntryNotFoundError, HfHubHTTPError, OfflineModeIsEnabled
from requests.exceptions import RequestException

from stodir.forecast import fetch_data, add_features, predict_next_day

REPO_ID = "AsifSayyed/stodir-forecast-model"
CONFIG_FILENAME = "config.yaml"
MODEL_FILENAME = "stodir_model.joblib"


st.set_page_config(
    page_title="StoDir Stock Forecast",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="📈"
)


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

    except (HfHubHTTPError, RequestException) as e:
        st.error(
            "Error: A network or Hub issue occurred while trying to download the model. "
            "Please check your internet connection and try again."
        )
        return None, None

    except OfflineModeIsEnabled:
        st.error("Error: Hugging Face Hub offline mode is enabled. Disable it or provide local artifacts.")
        return None, None


def plot_candlestick(data, ticker):
    """Displays a candlestick chart for the last 60 days."""
    fig, ax = mpf.plot(
        data.tail(60),
        type="candle",
        style="charles",
        title=f"{ticker} - Last 60 Days Candlestick",
        ylabel="Price (USD)",
        returnfig=True,
        figsize=(10, 4)
    )
    st.pyplot(fig)


def plot_closing_price(data, ticker):
    """Displays a line chart of the closing price."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data["close"], label="Closing Price")
    ax.set_title(f"{ticker} Closing Price History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)


def show_disclaimer():
    """Displays a non-distracting disclaimer at the bottom of the app."""
    st.divider()
    st.caption(
        "**Disclaimer:** This tool is for educational purposes only and is not financial advice. "
        "Market predictions are based solely on historical trends from a generalized model and do not account for all market factors. "
        "Use this information at your own risk."
    )


def main():
    """The main function that runs the Streamlit application."""
    st.title("StoDir - Stock Direction Forecaster")

    config, model = load_config_and_model()
    if model is None or config is None:
        st.error("Application cannot start because the model or configuration failed to load.")
        st.stop()

    HORIZONS = config["features"]["horizons"]
    PREDICTORS = [f"{h}_day" for h in HORIZONS]
    TRAINING_TICKERS = config["data"]["training_tickers"]

    ticker = st.selectbox("Select a stock for forecasting:", TRAINING_TICKERS)

    if st.button("Get Forecast", type="secondary", use_container_width=True):
        if not ticker:
            st.warning("Please enter a stock ticker.")
            return

        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # time to live set to 24 hours
                @st.cache_data(ttl=86400)
                def _get_company_name(ticker: str) -> str:
                    """Fetches the full company name for a given stock ticker.

                    :param ticker: The stock ticker symbol (e.g., "AAPL").
                    :returns: The full company name (e.g., "Apple Inc.") or the ticker as a fallback.
                    """
                    try:
                        return yf.Ticker(ticker).info.get("longName", ticker)
                    except Exception:
                        return ticker

                company_name = _get_company_name(ticker)

                raw_data = fetch_data(ticker)
                featured_data = add_features(raw_data.copy(), horizons=HORIZONS)

                # We need to update predict_next_day to return probability
                prediction, probability = predict_next_day(model, featured_data, PREDICTORS)
                latest_features = featured_data[PREDICTORS].iloc[-1]

                # --- Main Dashboard Layout ---
                st.header(f"Forecast for {company_name} ({ticker})")

                # Create two columns: one for summary, one for charts
                col1, col2 = st.columns((1, 2), gap="large")

                with col1:
                    # --- Forecast Summary Card ---
                    st.subheader("Prediction Summary")
                    with st.container(border=True):
                        if prediction == "up":
                            st.markdown("## 📈 **UP**")
                            confidence_text = f"The model predicts an upward price movement with **{probability:.2%}** confidence."
                        else:
                            st.markdown("## 📉 **DOWN**")
                            # Confidence in 'down' is 1 - P('up')
                            confidence_text = f"The model predicts a downward price movement with **{1-probability:.2%}** confidence." # type: ignore

                        st.write(confidence_text)
                        st.caption("Prediction for the next trading day.")

                    # --- Model Insights Card ---
                    st.subheader("Model Insights")
                    with st.container(border=True):
                        st.write("The prediction was based on these feature values:")
                        for horizon, value in zip(HORIZONS, latest_features):
                            st.metric(
                                label=f"Price vs. {horizon}-Day Avg.",
                                value=f"{value:.3f}"
                            )
                        st.caption("Ratios > 1 mean the current price is above the recent average.")

                with col2:
                    # --- Charting Area with Tabs ---
                    st.subheader("Historical Data")
                    tab1, tab2 = st.tabs(["Closing Price History", "Recent Candlestick"])
                    with tab1:
                        plot_closing_price(raw_data, ticker)
                    with tab2:
                        plot_candlestick(raw_data, ticker)

            except ValueError as e:
                st.error(f"Error analyzing {ticker}: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

    show_disclaimer()

if __name__ == "__main__":
    main()
