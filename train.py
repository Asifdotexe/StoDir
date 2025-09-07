import joblib
import pandas as pd
from sklearn.metrics import precision_score

from stodir.validation import backtest
from stodir.forecast import fetch_data, add_features

# Configuration: In a real project, this would come from a config file (see Issue #34)
TRAINING_TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA"]
HORIZONS = [2, 5, 60, 250, 1000]
PREDICTORS = [f"{h}_day" for h in HORIZONS]


def train_pipeline():
    """
    Full pipeline to train and save a generalized stock forecasting model.
    """
    print("--- Starting Model Training Pipeline ---")

    # 1. Fetch and combine data for all training tickers
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

    # 2. Add features to the combined dataset
    print("Engineering features...")
    featured_data = add_features(combined_data.copy(), horizons=HORIZONS)

    # 3. Perform backtesting to validate the model
    print("Performing backtest for validation...")
    backtest_results = backtest(featured_data, PREDICTORS)

    # Calculate and print overall precision from the backtest
    precision = precision_score(backtest_results["actual"], backtest_results["predicted"])
    print(f"\n--- Backtest Validation Complete ---")
    print(f"Backtest Precision Score: {precision:.2%}")
    print(f"This score reflects a more realistic measure of the model's historical performance.")

    # 4. Train the final model on ALL available data
    print("\nTraining final model on all available data...")
    final_model, _, _ = train_model(featured_data, horizons=HORIZONS)

    # 5. Serialize and save the final model
    model_filename = "stodir_model.joblib"
    joblib.dump(final_model, model_filename)
    print(f"Final model saved to '{model_filename}'")

    print("\n--- Model Training Pipeline Complete ---")

if __name__ == "__main__":
    # Import train_model here to avoid circular dependency if it were in __init__
    from stodir.forecast import train_model
    train_pipeline()
