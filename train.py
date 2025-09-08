import yaml
import joblib
import pandas as pd
from sklearn.metrics import precision_score

from stodir.validation import backtest
from stodir.forecast import fetch_data, add_features

MODEL_SAVE_PATH = "artifacts/stodir_model.joblib"
CONFIG_PATH = "config.yaml"


def train_pipeline():
    """
    Full pipeline to train and save a generalized stock forecasting model.
    """
    print("--- Starting Model Training Pipeline ---")

        # Load configuration from YAML file
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Use values from config
    TRAINING_TICKERS = config["data"]["training_tickers"]
    HORIZONS = config["features"]["horizons"]
    PREDICTORS = [f"{h}_day" for h in HORIZONS]

    # Fetch and combine data for all training tickers
    all_data = []
    for ticker in TRAINING_TICKERS:
        print(f"Fetching data for {ticker}...")
        try:
            data = fetch_data(ticker)
            all_data.append(data)
        except ValueError as e:
            print(e)

    if not all_data:
        print("No data fetched. Aborting training.")
        return

    combined_data = pd.concat(all_data)

    # Add features to the combined dataset
    print("Engineering features...")
    featured_data = add_features(combined_data.copy(), horizons=HORIZONS)

    # Perform backtesting to validate the model
    print("Performing backtest for validation...")
    # Perform backtesting per ticker to validate the model
    print("Performing backtest for validation...")
    per_ticker_precisions = []
    for tkr, df in featured_data.groupby("ticker"):
        try:
            bt = backtest(df, PREDICTORS)
            prec = precision_score(bt["actual"], bt["predicted"])
            per_ticker_precisions.append(prec)
            print(f"{tkr}: {prec:.2%}")
        except Exception as e:
            print(f"Backtest skipped for {tkr}: {e}")
    if not per_ticker_precisions:
        print("Backtest failed for all tickers. Aborting training.")
        return
    precision = sum(per_ticker_precisions) / len(per_ticker_precisions)
    print("\n--- Backtest Validation Complete ---")
    print(f"Backtest Precision (avg across tickers): {precision:.2%}")

    # Train the final model on ALL available data
    print("\nTraining final model on all available data...")
    final_model, _, _ = train_model(featured_data, horizons=HORIZONS)

    # Serialize and save the final model
    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"Final model saved to '{MODEL_SAVE_PATH}'")

    print("\n--- Model Training Pipeline Complete ---")

if __name__ == "__main__":
    # Import train_model here to avoid circular dependency if it were in __init__
    from stodir.forecast import train_model
    train_pipeline()
