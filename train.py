import io
import os
import yaml
import joblib
import logging
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import precision_score
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from stodir.validation import backtest
from stodir.forecast import fetch_data, add_features, train_model

MODEL_SAVE_PATH = f"artifacts/stodir_model_{datetime.today().strftime('%Y%m%d')}.joblib"
CONFIG_PATH = "config.yaml"
DATA_DIR = "data"
CACHE_DIR = "data/cache"
RAW_DATA_CACHE = os.path.join(CACHE_DIR, "raw")
FEATURE_CACHE = os.path.join(CACHE_DIR, "features")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

log_capture = io.StringIO()
class WarningErrorHandler(logging.StreamHandler):
    def emit(self, record):
        if record.levelno >= logging.WARNING:  # WARNING=30, ERROR=40
            log_capture.write(self.format(record) + "\n")


we_handler = WarningErrorHandler()
we_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s"))
logger.addHandler(we_handler)


def load_tickers_from_config(config: dict) -> list[str]:
    """Loads all tickers from the portfolio files specified in the config.

    :param config: Dictionary containing the model configuration file
    """
    all_tickers = set()
    portfolio_files = config["data"]["portfolios"]

    for portfolio_name in portfolio_files:
        file_path = os.path.join(DATA_DIR, f"{portfolio_name}_tickers.txt")
        try:
            with open(file_path, "r") as f:
                tickers = [line.strip() for line in f if line.strip()]
                all_tickers.update(tickers)

            logger.info(f"Loaded {len(tickers)} tickers from {file_path}")

        except FileNotFoundError:
            logger.warning(f"Ticker file not found: {file_path}. Please run get_tickers.py.")
    return sorted(list(all_tickers))


def fetch_and_cache_ticker(ticker: str, start_date: str) -> pd.DataFrame | None:
    """
    Attempts to load the data for a single ticker using a local cache to avoid re-downloading

    :param ticker: String containing stock ticker e.g., AAPL for Apple
    :param start_date: String containing the date to fetch the data from.
    :return: Dataframe containing the data for the given ticker or None.
    """
    cache_path = os.path.join(RAW_DATA_CACHE, f"{ticker}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    try:
        data = fetch_data(ticker, history_start=start_date)
        data.to_parquet(cache_path)
        return data
    except Exception as e:
        logger.warning(f"Could not fetch data for {ticker}: {e}")
        return None


def run_backtest_for_ticker(args: tuple) -> tuple[str, float]:
    """Wrapper function to run backtest on a single ticker's data for multiprocessing."""
    ticker, featured_df, predictors, start, step = args
    try:
        if len(featured_df) < (start + step):
            raise ValueError("Not enough historical data for a full run.")

        bt_results = backtest(featured_df, predictors, start=start, step=step)

        if bt_results.empty or bt_results["predicted"].sum() == 0:
            return ticker, 0.0

        precision = precision_score(bt_results["actual"], bt_results["predicted"])
        return ticker, precision
    except Exception as e:
        logger.error(f"Backtest failed for {ticker}: {e}")
        return ticker, 0.0


def train_pipeline():
    """
    Full pipeline to train and save a generalized stock forecasting model.
    """
    logger.info("--- Starting Model Training Pipeline ---")

    os.makedirs(RAW_DATA_CACHE, exist_ok=True)
    os.makedirs(FEATURE_CACHE, exist_ok=True)

    # Load configuration from YAML file
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Use values from config
    TRAINING_TICKERS = load_tickers_from_config(config)
    if not TRAINING_TICKERS:
        logger.error("No tickers loaded. Aborting training.")
        return

    HORIZONS = config["features"]["horizons"]
    PREDICTORS = [f"{h}_day" for h in HORIZONS]
    BACKTEST_START = config["backtesting"]["start"]
    BACKTEST_STEP = config["backtesting"]["step"]
    START_DATE = config["features"]["start_date"]

    logger.info("Processing tickers individually to prevent data leakage...")

    logger.info(f"Fetching raw data for {len(TRAINING_TICKERS)} tickers (using cache)...")
    all_raw_data = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_and_cache_ticker, ticker, START_DATE): ticker for ticker in TRAINING_TICKERS}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching raw data"):
            ticker = futures[future]
            result = future.result()
            if result is not None and not result.empty:
                all_raw_data[ticker] = result

    logger.info("Engineering features...")
    all_featured_data = {}
    for ticker, raw_df in tqdm(all_raw_data.items(), desc="Engineering features"):
        cache_path = os.path.join(FEATURE_CACHE, f"{ticker}.parquet")
        if os.path.exists(cache_path):
            featured_df = pd.read_parquet(cache_path)
        else:
            featured_df = add_features(raw_df.copy(), horizons=HORIZONS)
            featured_df.to_parquet(cache_path)
        all_featured_data[ticker] = featured_df

    logger.info("Performing backtest for validation...")
    backtest_args = [
        (ticker, df, PREDICTORS, BACKTEST_START, BACKTEST_STEP)
        for ticker, df in all_featured_data.items()
    ]

    per_ticker_precisions = {}
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(run_backtest_for_ticker, backtest_args), total=len(backtest_args), desc="Backtesting tickers"))

    for ticker, precision in results:
        if precision > 0:
            per_ticker_precisions[ticker] = precision

    if not per_ticker_precisions:
        logger.error("Backtest failed for all tickers. Aborting training.")
        return

    avg_precision = sum(per_ticker_precisions.values()) / len(per_ticker_precisions)
    logger.info(f"Average Backtest Precision (across {len(per_ticker_precisions)} tickers): {avg_precision:.2%}")

    # --- Final Model Training ---
    logger.info("Training final model on all available data...")
    final_training_data = pd.concat(all_featured_data.values()).sort_index()
    final_model, _, _ = train_model(final_training_data, horizons=HORIZONS)

    joblib.dump(final_model, MODEL_SAVE_PATH)
    logger.info(f"Final model saved to '{MODEL_SAVE_PATH}'")
    logger.info("--- Model Training Pipeline Complete ---")

if __name__ == "__main__":
    train_pipeline()
