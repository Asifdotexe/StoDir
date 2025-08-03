# FILE: app.py

import io

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import mplfinance as mpf

from stodir.forecast import fetch_data, add_features, train_model, predict_next_day


def plot_candlestick(data: pd.DataFrame, ticker: str):
    """Displays a candlestick chart for the last 30 days.
    :param data: Historical stock data DataFrame.
    :param ticker: Stock ticker symbol.
    """
    buffer = io.BytesIO()
    mpf.plot(
        data.tail(30),
        type="candle",
        style="charles",
        title=f"{ticker} - Last 30 Days",
        ylabel="Price",
        savefig=buffer,
    )
    buffer.seek(0)
    st.image(buffer, caption=f"{ticker} Candlestick Chart", use_column_width=True)


def plot_closing_price(data: pd.DataFrame, ticker: str):
    """Displays a line chart of the closing price.
    :param data: Historical stock data DataFrame.
    :param ticker: Stock ticker symbol.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data["close"], label="Closing Price")
    ax.set_title(f"{ticker} Closing Price History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)


def show_disclaimer():
    """Displays a disclaimer at the bottom of the app."""
    disclaimer = """
    <div
    style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; border: 1px solid #f5c6cb;">
        <strong>Disclaimer:</strong> This tool is for educational purposes only and is not financial advice.
        Market predictions are based solely on historical trends and do not account for all market factors.
    </div>
    """
    st.markdown(disclaimer, unsafe_allow_html=True)


def main():
    """Streamlit entry point for StoDir."""
    st.title("StoDir - Stock Direction Forecasting")

    ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()

    if st.button("Forecast"):
        if not ticker:
            st.warning("Please enter a stock ticker.")
            return

        with st.spinner(f"Fetching and analyzing data for {ticker}..."):
            try:
                raw_data = fetch_data(ticker)
                featured_data = add_features(raw_data.copy())
                model, precision, predictors = train_model(featured_data)
                prediction = predict_next_day(model, featured_data, predictors)

                if prediction == "up":
                    st.success(f"Prediction for {ticker}: The stock price is likely to go UP tomorrow. 📈")
                else:
                    st.error(f"Prediction for {ticker}: The stock price is likely to go DOWN tomorrow. 📉")

                st.info(f"Model Precision: {precision:.2%}")

                st.subheader("Data Visualization")
                plot_closing_price(raw_data, ticker)
                plot_candlestick(raw_data, ticker)

            except ValueError as e:
                # This catches the error from fetch_data if the ticker is invalid
                st.error(f"Error: {e}")

    show_disclaimer()


if __name__ == "__main__":
    main()
