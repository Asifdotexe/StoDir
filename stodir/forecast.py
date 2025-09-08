import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


def fetch_data(ticker: str, history_start: str = "1990-01-01") -> pd.DataFrame:
    """
    Fetches historical stock data for a given ticker using Yahoo Finance.

    :param ticker: Stock ticker symbol (e.g., 'AAPL').
    :param history_start: Date to start fetching historical data (YYYY-MM-DD).
    :return: DataFrame with historical stock data.
    :raises ValueError: If the ticker is invalid or no data is found.
    """
    data = yf.Ticker(ticker).history(start=history_start)
    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Please check the symbol.")
    data.columns = data.columns.str.lower()
    return data


def add_features(data: pd.DataFrame, horizons: list[int] = None) -> pd.DataFrame:
    """
    Adds target variable and rolling average features to the data.

    :param data: Historical stock DataFrame.
    :param horizons: List of horizons to calculate rolling averages.
    :return: DataFrame with additional feature columns.
    """
    if horizons is None:
        horizons = [2, 5, 60, 250, 1000]

    data["tomorrow"] = data["close"].shift(-1)
    data["target"] = (data["tomorrow"] > data["close"]).astype(int)

    for horizon in horizons:
        rolling_average = data["close"].rolling(window=horizon).mean()
        data[f"{horizon}_day"] = rolling_average / data["close"]

    return data.dropna()


def train_model(data: pd.DataFrame, horizons: list[int] = None) -> tuple[RandomForestClassifier, float, list[str]]:
    """
    Trains a RandomForestClassifier and evaluates its precision.

    :param data: Feature-engineered stock data.
    :param horizons: Horizons used for rolling average features.
    :return: A tuple containing the trained model, its precision score, and the list of predictor columns.
    """
    if horizons is None:
        horizons = [2, 5, 60, 250, 1000]

    predictors = [f"{h}_day" for h in horizons]

    train_data = data.iloc[:-100]
    test_data = data.iloc[-100:]

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=50,
        random_state=42,
    )
    model.fit(train_data[predictors], train_data["target"])

    predictions = model.predict(test_data[predictors])
    precision = precision_score(test_data["target"], predictions)

    return model, precision, predictors


def predict_next_day(model: RandomForestClassifier, data: pd.DataFrame, predictors: list[str]) -> tuple[str, float]:
    """
    Predicts the stock's direction and the model's confidence.

    :param model: Trained RandomForestClassifier.
    :param data: The full feature-engineered stock data.
    :param predictors: List of feature names used for prediction.
    :return: A tuple containing the prediction ('up' or 'down') and the probability of the positive class ('up').
    """
    latest_data = data.iloc[[-1]]
    # Predict_proba returns probabilities for [class_0, class_1]
    # We want the probability for class_1 ('up')
    probability = model.predict_proba(latest_data[predictors])[0][1]

    prediction = "up" if probability > 0.6 else "down"
    return prediction, probability

