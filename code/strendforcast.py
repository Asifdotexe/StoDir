import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import sys

def fetch_data(ticker: str, 
               save_plot: bool=False,
               history_start: str = "1990-01-01"
) -> pd.DataFrame:
    """
    Fetches historical stock data for a given ticker symbol.

    Parameters:
    - ticker (str): The ticker symbol of the stock.
    - save_plot (bool, optional): If True, saves a plot of the closing prices to a file. Defaults to False.
    - history_start (str, optional): The start date for fetching the historical data. Defaults to "1990-01-01".

    Returns:
    - pd.DataFrame: A DataFrame containing the historical stock data.

    Note:
    - The function uses the yfinance library to fetch the data.
    - The function saves the plot in a directory named 'plots' with the filename as '{ticker}_closing_price.png'.
    - The function closes the plot after saving it.
    """
    data = yf.Ticker(ticker).history(start=history_start)
    data.columns = data.columns.str.lower()
    if save_plot == True:
        plt.plot(data['Close'])
        plt.title(f'Closing Price of {ticker} Over Time')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.savefig(f'plots/{ticker}_closing_price.png')
        plt.close()
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
    data = data.drop()
    return data