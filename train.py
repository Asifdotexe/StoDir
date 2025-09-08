import os
import yaml
import joblib
import logging
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score

from stodir.validation import backtest
from stodir.forecast import fetch_data, add_features, train_model

MODEL_SAVE_PATH = "artifacts/stodir_model.joblib"
CONFIG_PATH = "config.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def train_pipeline():
    """
    Full pipeline to train and save a generalized stock forecasting model.
    """
    logger.info("--- Starting Model Training Pipeline ---")

    # Load configuration from YAML file
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Use values from config
    TRAINING_TICKERS = config["data"]["training_tickers"]
    HORIZONS = config["features"]["horizons"]
    PREDICTORS = [f"{h}_day" for h in HORIZONS]
    BACKTEST_START = config["backtesting"]["start"]
    BACKTEST_STEP = config["backtesting"]["step"]

    logger.info("Processing tickers individually to prevent data leakage...")
    all_featured_data = []
    for ticker in tqdm(TRAINING_TICKERS, desc="Fetching & Engineering Features"):
        try:
            # Fetch data for a single ticker
            data = fetch_data(ticker)

            # Engineer features on that single ticker's data
            # This ensures shift() operations do not cross ticker boundaries.
            featured = add_features(data.copy(), horizons=HORIZONS)
            featured['ticker'] = ticker
            all_featured_data.append(featured)

        except ValueError as e:
            logger.warning(f"Could not process data for {ticker}: {e}")

    if not all_featured_data:
        logger.error("No data could be processed. Aborting training.")
        return

    featured_data = pd.concat(all_featured_data).sort_index()
    logger.info(f"Successfully processed and combined data for {len(all_featured_data)} tickers.")

    logger.info("Performing backtest for validation...")
    per_ticker_precisions = []

    grouped_data = featured_data.groupby("ticker")
    for tkr, df in tqdm(grouped_data, desc="Backtesting tickers"):
        try:
            # Ensure there's enough data for this ticker to backtest
            if len(df) < (BACKTEST_START + BACKTEST_STEP):
                logger.warning(f"Skipping backtest for {tkr}: not enough historical data for a full run.")
                continue

            bt_results = backtest(df, PREDICTORS, start=BACKTEST_START, step=BACKTEST_STEP)

            # Check if the backtest produced any valid predictions.
            if bt_results.empty:
                logger.warning(f"Backtest for {tkr} produced no results, likely due to data gaps.")
                continue

            # Check if the model ever predicted the stock would go up.
            if bt_results["predicted"].sum() == 0:
                logger.warning(f"Model made no 'up' predictions for {tkr}. Precision is 0.")
                precision = 0.0
            else:
                precision = precision_score(bt_results["actual"], bt_results["predicted"])

            per_ticker_precisions.append(precision)
            logger.debug(f"Backtest precision for {tkr}: {precision:.2%}")

        except ValueError as e:
            logger.error(f"Backtest failed for {tkr} due to a data issue (likely a single class in a training split): {e}")

        except Exception as e:
            logger.error(f"An unexpected error occurred during backtest for {tkr}: {e}", exc_info=True)

    if not per_ticker_precisions:
        logger.error("Backtest failed for all tickers. Aborting training.")
        return

    avg_precision = sum(per_ticker_precisions) / len(per_ticker_precisions)
    print("\n--- Backtest Validation Complete ---")
    print(f"Backtest Precision (avg across tickers): {avg_precision:.2%}")

    # Train the final model on ALL available data
    logger.info("Training final model on all available data...")
    final_model, _, _ = train_model(featured_data.drop(columns=["ticker"]), horizons=HORIZONS)

    # Serialize and save the final model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH) or ".", exist_ok=True)
    joblib.dump({"model": final_model,
                 "horizons": HORIZONS,
                 "predictors": PREDICTORS,}, MODEL_SAVE_PATH)
    logger.info(f"Final model saved to '{MODEL_SAVE_PATH}'")

    logger.info("--- Model Training Pipeline Complete ---")

if __name__ == "__main__":
    train_pipeline()
