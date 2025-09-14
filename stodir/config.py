PORTFOLIOS = [
    {
        "name": "sp500",
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "ticker_column_index": 0, # Ticker is in the first column
    },
    {
        "name": "nasdaq100",
        "url": "https://en.wikipedia.org/wiki/NASDAQ-100",
        "ticker_column_index": 0, # Ticker is in the second column
    },
    {
        "name": "nifty50",
        "url": "https://en.wikipedia.org/wiki/NIFTY_50",
        "ticker_column_index": 1,
        "ticker_suffix": ".NS", # yfinance requires a '.NS' suffix for Indian NSE stocks
    },
    {
        "name": "ftse100",
        "url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "ticker_column_index": 1,
        "ticker_suffix": ".L",  # yfinance requires a '.L' suffix for London Stock Exchange
    },
    {
        "name": "dax",
        "url": "https://en.wikipedia.org/wiki/DAX",
        "ticker_column_index": 3, # Ticker is in the fourth column ('Ticker')
        "ticker_suffix": ".DE", # yfinance requires a '.DE' suffix for Xetra (Germany)
    },
]
