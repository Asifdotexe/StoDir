# FILE: cli.py

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

from stodir.forecast import (add_features, fetch_data, predict_next_day,
                             train_model)


def save_plots(raw_data: pd.DataFrame, ticker):
    """Saves closing price and candlestick charts to the plots/ directory."""
    plot_dir = 'plots/'
    os.makedirs(plot_dir, exist_ok=True)  # Ensure the directory exists

    # Save closing price plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(raw_data.index, raw_data["close"], label="Closing Price")
    ax.set_title(f"{ticker} Closing Price History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    plt.savefig(f'{plot_dir}{ticker}_closing_price.png')
    plt.close(fig)
    print(f"Saved closing price chart to '{plot_dir}{ticker}_closing_price.png'")

    # Save candlestick chart
    mpf.plot(
        raw_data.tail(30),
        type='candle',
        style='charles',
        title=f"{ticker} - Last 30 Days",
        ylabel='Price',
        savefig=f'{plot_dir}{ticker}_candlestick_chart.png'
    )
    print(f"Saved candlestick chart to '{plot_dir}{ticker}_candlestick_chart.png'")


def main():
    """CLI entry point for StoDir."""
    parser = argparse.ArgumentParser(description='Forecast the next-day stock price direction.')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., GOOGL, MSFT).')
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"--- Running forecast for {ticker} ---")

    raw_data = fetch_data(ticker)
    print(f"Successfully fetched {len(raw_data)} data points.")
    featured_data = add_features(raw_data.copy())
    model, precision, predictors = train_model(featured_data)
    prediction = predict_next_day(model, featured_data, predictors)

    print("\n--- Forecast Results ---")
    print(f"Model Precision: {precision:.2%}")
    print(f"Prediction for next trading day: The stock is likely to go {prediction.upper()}.")

    print("\n--- Saving Plots ---")
    save_plots(raw_data, ticker)


if __name__ == "__main__":
    main()
