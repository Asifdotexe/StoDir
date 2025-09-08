# Documentation: Backtesting Methodology

In financial forecasting, standard validation methods like a simple train-test split are unreliable. Financial data is a **time series**, where the order of events matters and underlying patterns (market regimes) constantly change. A model that worked well last year might fail completely during a market crash.

To get a realistic estimate of model performance, this project employs a robust time-series validation technique known as a **walk-forward backtest** (or expanding window validation).

## What's the flaw in a simple train-test split?
If we train a model on data from 1990-2023 and test it on 2024, our performance score only reflects a single market period. This gives us no confidence in how the model would have performed during major historical events like the 2008 financial crisis or the 2020 pandemic. The model could appear successful by chance alone.

## My Backtesting Approach
The `stodir.validation.backtest` function simulates how the model would have performed over many different periods in the past. It iteratively trains on a growing window of historical data and makes predictions on the period immediately following.

This process is visualized below:

```bash
Dataset: |=========================================|

Fold 1:
Train:   | T T T T |
Predict:           | P |

Fold 2:
Train:   | T T T T T |
Predict:             | P |

Fold 3:
Train:   | T T T T T T |
Predict:               | P |
...and so on.
```

The algorithm works as follows:

1. **Initialization:** The process begins after an initial "warm-up" period (e.g., the first 4 years of data).

2.  **Iteration:** The function loops through the rest of the dataset. In each loop:

    - The model is trained on all data observed up to that point.
    - The model then makes predictions for the next, unseen period.
    - These predictions are stored, and the window expands to include the data that was just predicted on.

3. **Aggregation:** This loop continues until all the data has been used for prediction. The final performance score is calculated on the complete set of out-of-sample predictions.

## Implementation Detaiks
The `stodir.validation.backtest` function's role is strictly to generate historical predictions, not to score them

- **Output:** The function returns a single pandas DataFrame containing two columns: `actual` (the true historical outcome) and `predicted` (the model's out-of-sample predictions).

- **Parameters:** The simulation can be configured with the `start` parameter (default `1000`, ≈4 years of initial training data) and the `step` parameter (default `250`, predicting ≈1 year forward per fold).

- **Usage:** To evaluate model performance, you should compute your desired metrics (e.g., precision, recall, ROC-AUC) using the DataFrame returned by this function. This design cleanly separates the simulation logic from the evaluation logic.

## Benefits of This Approach
- **Simulates Reality:** This process closely mimics how a model would actually be used: periodically retraining on new data to forecast the immediate future.

- **Prevents Lookahead Bias:** At no point in the simulation does the model have access to future information, making the evaluation fair and realistic.

- **Averages Over Time:** The final performance metric is calculated across many different years and market conditions, providing a much more stable and trustworthy estimate of the model's true predictive power.