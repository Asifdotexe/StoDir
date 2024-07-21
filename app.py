import io
import yfinance as yf
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

def fetch_data(ticker: str, history_start: str = "1990-01-01") -> pd.DataFrame:
    """
    Fetches historical stock data for the given ticker from Yahoo Finance.
    """
    data = yf.Ticker(ticker).history(start=history_start)
    data.columns = data.columns.str.lower()
    return data

def add_features(data: pd.DataFrame, *horizons) -> pd.DataFrame:
    """
    Adds various features to the input DataFrame for stock price prediction.
    """
    data['tomorrow'] = data['close'].shift(-1)
    data['target'] = (data['tomorrow'] > data['close']).astype(int)
    horizons = [2, 5, 60, 250, 1000]
    for horizon in horizons:
        rolling_average = data['close'].rolling(horizon).mean()
        data[f'{horizon}_day'] = rolling_average / data['close']
    data = data.dropna(axis=0, how='any')
    return data

def train_model(data: pd.DataFrame, ticker: str) -> Tuple[RandomForestClassifier, float]:
    """
    Trains a RandomForestClassifier model on the provided historical stock data.
    """
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=42)
    train = data.iloc[:-100]
    test = data.iloc[-100:]
    predictors = [f'{h}_day' for h in [2, 5, 60, 250, 1000]]
    model.fit(train[predictors], train['target'])
    preds = model.predict(test[predictors])
    precision = precision_score(test['target'], preds)
    return model, precision

def predict_next_day(model, data: pd.DataFrame) -> str:
    """
    Predicts whether the stock price for the given ticker will go up or down tomorrow.
    """
    recent_data = data.iloc[-1:]
    features = [f'{h}_day' for h in [2, 5, 60, 250, 1000]]
    probabilities = model.predict_proba(recent_data[features])
    prediction = "up" if probabilities[0][1] > 0.6 else "down"
    return prediction

def plot_candlestick(data: pd.DataFrame, ticker: str) -> None:
    """
    Plots a candlestick chart of the last 30 days for the given stock ticker.
    """
    # Create an in-memory buffer to store the plot
    buffer = io.BytesIO()
    
    # Plot the candlestick chart
    mpf.plot(data[-30:], type='candle', style='charles', title=f"{ticker} Candlestick Chart of Last 30 days", ylabel='Price', savefig=buffer)
    
    # Set the buffer position to the start
    buffer.seek(0)
    
    # Display the plot using Streamlit
    st.image(buffer, caption=f"{ticker} Candlestick Chart of Last 30 days", use_column_width=True)

def plot_closing_price(data: pd.DataFrame, ticker: str) -> None:
    """
    Plots a line chart of the closing price for the given stock ticker.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(data['close'], label='Closing Price')
    plt.title(f'{ticker} Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    st.pyplot(fig)

def main():
    st.title('StoDir - Stock Direction Forecasting Model')
    
    ticker = st.text_input('Enter Stock Ticker:', 'AAPL')
    
    if ticker:
        if st.button('Forecast'):
            data = fetch_data(ticker)
            data = add_features(data)
            model, precision = train_model(data, ticker)
            prediction = predict_next_day(model, data)
            
            # Display prediction with success box and appropriate symbol
            if prediction == "up":
                st.success(f'Prediction for {ticker}: The stock price is predicted to go **up** tomorrow. 📈')
            else:
                st.success(f'Prediction for {ticker}: The stock price is predicted to go **down** tomorrow. 📉')

            st.write(f'Precision Score of the Model: {precision:.2f}')
            
            plot_closing_price(data, ticker)
            plot_candlestick(data, ticker)
            
    # Disclaimer at the bottom of the page
    disclaimer = """
    <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; border: 1px solid #f5c6cb;">
        <strong>Disclaimer:</strong> This model is designed to understand the patterns in temporal data and predict the likely trend. However, the stock market is influenced by numerous factors not considered in this model. Therefore, this prediction should not be treated as a source of truth. This project is intended for skill demonstration purposes only.
    </div>
    """
    st.markdown(disclaimer, unsafe_allow_html=True)


if __name__ == "__main__":
    main()