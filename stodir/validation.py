import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def backtest(data: pd.DataFrame, predictors: list[str], start: int = 250*4, step: int = 250) -> pd.DataFrame:
    """
    Performs a backtest on historical time-series data.

    This function simulates how a model would have performed historically by
    iterating through the data, training on a growing window of past data,
    and making predictions on the subsequent period.

    :param data: The full historical DataFrame with features and target.
    :param predictors: A list of the column names to be used as features.
    :param start: The minimum number of data points required for the initial training set. Defaults to ~4 years.
    :param step: The number of data points to predict forward in each iteration. Defaults to ~1 year.
    :return: A DataFrame containing the actual target values and the model's predictions, indexed by date.
    """
    all_predictions = []

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=50,
        random_state=42
    )

    for i in range(start, data.shape[0], step):
        # Train on all data up to the current point
        train_set = data.iloc[0:i]
        # Test on the next 'step' days
        test_set = data.iloc[i:(i + step)]

        model.fit(train_set[predictors], train_set["target"])

        preds = model.predict(test_set[predictors])
        preds_series = pd.Series(preds, index=test_set.index)

        # Combine actuals and predictions for this fold
        combined = pd.concat([test_set["target"], preds_series], axis=1)
        combined.columns = ["actual", "predicted"]

        all_predictions.append(combined)

    return pd.concat(all_predictions)

