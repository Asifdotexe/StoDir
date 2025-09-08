# Documentation: Technical Methodology
This document provides a deeper look into the technical decisions and validation strategies used in the StoDir project.

## 1. System Architecture
The core principle of the system is the separation of concerns between model training and model serving (inference).

```bash
+---------------------------------+
|      LOCAL DEVELOPMENT          |
|---------------------------------|
| `train.py`                      |
|   - Fetches data for many stocks|
|   - Runs backtest validation    |
|   - Trains final model          |
|   - Saves `artifacts/model.joblib` |
+---------------------------------+
              |
              | (Manual Upload)
              V
+---------------------------------+
|      HUGGING FACE HUB (CLOUD)   |
|---------------------------------|
| `stodir_model.joblib`           |
| `config.yaml`                   |
+---------------------------------+
              |
              | (Automatic Download on App Startup)
              V
+---------------------------------+
|      DEPLOYMENT (STREAMLIT)     |
|---------------------------------|
| `app.py`                        |
|   - Loads model from Hub        |
|   - Fetches latest data for one |
|     stock on user request       |
|   - Makes a fast prediction     |
+---------------------------------+
```

## 2. Model Validation: Backtesting

Please refer to [Backtesting documentation](docs\BACKTESTING.md)

## 3. Performance Metrics
The model is optimized for Precision. In a financial context, this means we prioritize being correct when we predict the market will go "Up". It's better to miss a potential gain (a False Negative) than to act on a bad tip and suffer a loss (a False Positive).

Across a backtest on major tech stocks from 2000-2025, the model achieved:

- **Backtest Precision Score: ~58-62%**

While this is not a guarantee of future performance, a precision score consistently above the 50% baseline in a noisy domain like stock forecasting indicates that the model has learned a meaningful signal.