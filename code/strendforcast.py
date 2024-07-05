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
    