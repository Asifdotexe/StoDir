import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import argparse
import mplfinance as mpf

def fetch_data(ticker: str, 
               save_plot: bool=True,
               save_candlestick: bool=True,
               history_start: str = "1990-01-01"
) -> pd.DataFrame:
    """
    Fetches historical stock data for a given ticker symbol.

    Parameters:
    - ticker (str): The ticker symbol of the stock.
    - save_plot (bool, optional): If True, saves a plot of the closing prices to a file. Defaults to False.
    - save_candlestick (bool, optional): If True, saves a candlestick chart of the last 30 days to a file. Defaults to False.
    - history_start (str, optional): The start date for fetching the historical data. Defaults to "1990-01-01".

    Returns:
    - pd.DataFrame: A DataFrame containing the historical stock data.

    Note:
    - The function uses the yfinance library to fetch the data.
    - The function saves the plot in a directory named 'plots' with the filename as '{ticker}_closing_price.png' and '{ticker}_candlestick_chart.png'.
    - The function closes the plot after saving it.
    """
    data = yf.Ticker(ticker).history(start=history_start)
    data.columns = data.columns.str.lower()

    plot_dir = 'plots/'
    # Ensure the directory exists
    os.makedirs(plot_dir, exist_ok=True)

    if save_plot:
        # Save closing price plot
        plt.plot(data['close'])
        plt.title(f'Closing Price of {ticker} Over Time')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.savefig(f'{plot_dir}{ticker}_closing_price.png')
        plt.close()

    if save_candlestick:
        # Save candlestick chart for the last 30 days
        mpf.plot(data[-30:], type='candle', style='charles', 
                 title=f"{ticker} Candlestick Chart of Last 30 days", 
                 ylabel='Price', 
                 savefig=f'{plot_dir}{ticker}_candlestick_chart.png')

    return data
def add_features(data: pd.DataFrame, *horizons) -> pd.DataFrame:
    """
    Adds features to the input DataFrame for stock price prediction.
    
    Parameters:
    - data (pd.DataFrame): A DataFrame containing historical stock data.
    
    Returns:
    - pd.DataFrame: A DataFrame with added features for stock price prediction.
    
    The function adds the following features to the input DataFrame:
    1. 'tomorrow': The closing price of the next day.
    2. 'target': A binary indicator variable that takes the value 1 if the closing price of the next day is higher than the current closing price, and 0 otherwise.
    3. For each horizon in the list of horizons (2, 5, 60, 250, 1000), it calculates the rolling average of the closing price over that horizon and divides it by the current closing price to create a new feature with the name '{horizon}_day'.

    The function then drops any remaining columns from the DataFrame.
    """
    data['tomorrow'] = data['close'].shift(-1)
    data['target'] = (data['tomorrow'] > data['close']).astype(int)
    """ Horizon logic
     2 -> two days
     5 -> one week
     60 -> three months
     1000 -> 4 years
     """
    horizons = [2, 5, 60, 250, 1000]
    for horizon in horizons:
        rolling_average = data['close'].rolling(horizon).mean()
        data[f'{horizon}_day'] = rolling_average / data['close']
    data = data.dropna(axis=0, how='any')
    return data

def train_model(data: pd.DataFrame, ticker: str) -> RandomForestClassifier:
    
    plot_dir = 'plots/'
    # Ensure the directory exists
    os.makedirs(plot_dir, exist_ok=True)
    
    model = RandomForestClassifier(n_estimators=200,
                                   min_samples_split=50,
                                   random_state=42)
    train = data.iloc[:-100]
    test = data.iloc[-100:]
    
    predictors = [f'{h}_day' for h in [2, 5, 60, 250, 1000]]
    model.fit(train[predictors], train['target'])
    
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)
    precision = precision_score(test['target'], preds)
    print("Precision Score:", precision)
    
    combined = pd.concat([test['target'], preds], axis=1)
    combined.columns = ['Actual', 'Predicted']
    
    plt.figure(figsize=(14, 7))
    combined.plot()
    plt.title('Actual vs Predicted')
    plt.savefig(f'{plot_dir}{ticker}_actual_vs_predicted.png')
    plt.close()
    
    return model

def predict_next_day(model, data):
    """
    Predicts the direction of the next day's stock price based on the trained model and recent data.

    Parameters:
    - model (RandomForestClassifier): The trained Random Forest model for stock price prediction.
    - data (pd.DataFrame): A DataFrame containing recent historical stock data.

    Returns:
    - str: The predicted direction of the next day's stock price. Returns "up" if the probability of an upward movement is greater than 0.6, otherwise returns "down".

    The function selects the most recent data from the input DataFrame, extracts relevant features, and uses the trained model to predict the probabilities of an upward or downward movement.
    It then compares the probabilities to a threshold (0.6) and returns the corresponding prediction.
    """    
    recent_data = data.iloc[-1:]
    features = [f'{h}_day' for h in [2, 5, 60, 250, 1000]]
    probabilities = model.predict_proba(recent_data[features])
    prediction = "up" if probabilities[0][1] > 0.6 else "down"
    return prediction

def main():
    parser = argparse.ArgumentParser(description='Predict the next day stock price movement.')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('horizons', type=int, nargs='*', help='List of horizons to include as features (optional)')
    args = parser.parse_args()
    
    data = fetch_data(args.ticker)
    data = add_features(data)
    model = train_model(data, args.ticker)
    prediction = predict_next_day(model, data)
    
    print(f'The stock price for {args.ticker} is predicted to go {prediction} tomorrow.')
    print(f'Plots saved as {args.ticker}_closing_price.png and {args.ticker}_actual_vs_predicted.png')
    
if __name__ == "__main__":
    main()