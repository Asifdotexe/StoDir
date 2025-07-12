import io

import mplfinance as mpf
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


def fetch_data(ticker: str, history_start: str = "1990-01-01") -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker using Yahoo Finance.

    :param ticker: Stock ticker symbol (e.g., 'AAPL').
    :param history_start: Date to start fetching historical data (YYYY-MM-DD).
    :return: DataFrame with historical stock data.
    """
    data = yf.Ticker(ticker).history(start=history_start)
    data.columns = data.columns.str.lower()
    return data


def add_features(data: pd.DataFrame,
                 horizons: list[int] = None) -> pd.DataFrame:
    """
    Add target variable and rolling average features to the data.

    :param data: Historical stock DataFrame.
    :param horizons: List of horizons to calculate rolling averages.
    :return: DataFrame with additional feature columns.
    """
    if horizons is None:
        horizons = [2, 5, 60, 250, 1000]

    data["tomorrow"] = data["close"].shift(-1)
    data["target"] = (data["tomorrow"] > data["close"]).astype(int)

    for horizon in horizons:
        rolling_average = data["close"].rolling(window=horizon).mean()
        data[f"{horizon}_day"] = rolling_average / data["close"]

    return data.dropna()


def train_model(data: pd.DataFrame,
                horizons: list[int] = None
                ) -> tuple[RandomForestClassifier, float]:
    """
    Train a RandomForestClassifier to predict stock price direction.

    :param data: Feature-engineered stock data.
    :param horizons: Horizons used for rolling average features.
    :return: Trained RandomForestClassifier and precision score.
    """
    if horizons is None:
        horizons = [2, 5, 60, 250, 1000]

    predictors = [f"{h}_day" for h in horizons]

    train_data = data.iloc[:-100]
    test_data = data.iloc[-100:]

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=50,
        random_state=42,
    )
    model.fit(train_data[predictors], train_data["target"])

    predictions = model.predict(test_data[predictors])
    precision = precision_score(test_data["target"], predictions)

    return model, precision


def predict_next_day(model: RandomForestClassifier,
                     data: pd.DataFrame, 
                     horizons: list[int] = None) -> str:
    """
    Predict the stock's direction for the next day.

    :param model: Trained RandomForestClassifier.
    :param data: Feature-engineered stock data.
    :param horizons: Horizons used for rolling average features.
    :return: 'up' if predicted to rise, 'down' otherwise.
    """
    if horizons is None:
        horizons = [2, 5, 60, 250, 1000]

    predictors = [f"{h}_day" for h in horizons]
    latest = data.iloc[[-1]]

    probability = model.predict_proba(latest[predictors])[0][1]
    return "up" if probability > 0.6 else "down"


def plot_candlestick(data: pd.DataFrame, ticker: str) -> None:
    """
    Display a candlestick chart for the last 30 days.

    :param data: Stock data.
    :param ticker: Stock ticker symbol.
    """
    buffer = io.BytesIO()

    mpf.plot(
        data[-30:],
        type="candle",
        style="charles",
        title=f"{ticker} - Last 30 Days Candlestick Chart",
        ylabel="Price",
        savefig=buffer,
    )

    buffer.seek(0)

    st.image(buffer, caption=f"{ticker} Candlestick Chart", 
             use_column_width=True)


def plot_closing_price(data: pd.DataFrame, ticker: str) -> None:
    """
    Display a line chart of the closing price.

    :param data: Stock data.
    :param ticker: Stock ticker symbol.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data["close"], label="Closing Price")
    ax.set_title(f"{ticker} Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)


def show_disclaimer() -> None:
    """
    Display a disclaimer at the bottom of the Streamlit app.
    """
    disclaimer = """
    <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; border: 1px solid #f5c6cb;">
        <strong>Disclaimer:</strong> This model predicts potential trends using historical data only.
        The stock market is influenced by countless external factors. This project is for educational
        purposes only and not for financial advice.
    </div>
    """
    st.markdown(disclaimer, unsafe_allow_html=True)


def main():
    """
    Streamlit entry point for StoDir: Stock Direction Forecasting Model.
    """
    st.title("StoDir - Stock Direction Forecasting Model")

    ticker = st.text_input("Enter Stock Ticker:", "AAPL")

    if ticker and st.button("Forecast"):
        data = fetch_data(ticker)
        st.write("Data shape before feature engineering:", data.shape)
        data = add_features(data)
        st.write("Data shape after feature engineering:", data.shape)
        model, precision = train_model(data)
        prediction = predict_next_day(model, data)

        if prediction == "up":
            st.success(f"Prediction for {ticker}: The stock price is predicted to go **UP** tomorrow. 📈")
        else:
            st.success(f"Prediction for {ticker}: The stock price is predicted to go **DOWN** tomorrow. 📉")

        st.write(f"Model Precision Score: **{precision:.2f}**")

        plot_closing_price(data, ticker)
        plot_candlestick(data, ticker)

    show_disclaimer()


if __name__ == "__main__":
    main()
