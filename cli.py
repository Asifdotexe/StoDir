import os
import argparse

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

from stodir.forecast import fetch_data, add_features, predict_next_day

MODEL_PATH = "stodir_model.joblib"
HORIZONS = [2, 5, 60, 250, 1000] # This should come from config
PREDICTORS = [f"{h}_day" for h in HORIZONS]

# (Keep your save_plots function as is)
def save_plots(raw_data: pd.DataFrame, ticker):
    """Saves closing price and candlestick charts to the plots/ directory."""
    plot_dir = 'plots/'
    os.makedirs(plot_dir, exist_ok=True)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(raw_data.index, raw_data["close"], label="Closing Price")
    ax.set_title(f"{ticker} Closing Price History")
    ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
    plt.savefig(f'{plot_dir}{ticker}_closing_price.png')
    plt.close(fig)
    print(f"Saved closing price chart to '{plot_dir}{ticker}_closing_price.png'")
    mpf.plot(raw_data.tail(30), type='candle', style='charles', title=f"{ticker} - Last 30 Days",
             ylabel='Price', savefig=f'{plot_dir}{ticker}_candlestick_chart.png')
    print(f"Saved candlestick chart to '{plot_dir}{ticker}_candlestick_chart.png'")


def main():
    """CLI entry point for StoDir."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        print("Please run the training pipeline first: `python train.py`")
        return
        
    model = joblib.load(MODEL_PATH)
    
    parser = argparse.ArgumentParser(description='Forecast the next-day stock price direction using a pre-trained model.')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., GOOGL, MSFT).')
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"--- Running forecast for {ticker} ---")

    try:
        raw_data = fetch_data(ticker)
        print(f"Successfully fetched {len(raw_data)} data points.")
        featured_data = add_features(raw_data.copy(), horizons=HORIZONS)
        
        prediction = predict_next_day(model, featured_data, PREDICTORS)

        print("\n--- Forecast Results ---")
        print(f"Prediction for next trading day: The stock is likely to go {prediction.upper()}.")

        print("\n--- Saving Plots ---")
        save_plots(raw_data, ticker)
        
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
