# Documentation: System Architecture
This document provides a detailed look into the MLOps architecture of the StoDir project. The core principle is the separation of concerns between model training and model serving (inference).

## 1. The Training Pipeline `(train.py)`
This is an offline script responsible for creating the model artifact.

- **Data Aggregation:** It begins by fetching decades of historical data for a list of diverse, high-volume stocks defined in `config.yaml`. Training on a wide range of assets helps the model learn more generalizable market patterns rather than overfitting to the history of a single stock.

- **Feature Engineering:** It applies a series of transformations to the raw time-series data, primarily creating rolling average ratios as defined by the `horizons` in the config.

- **Model Validation (Backtesting):** Before training the final model, it runs a rigorous backtest using the `stodir.validation.backtest` function. This provides a reliable, historical performance metric (precision) for the chosen model and feature set. This step is critical for model selection and strategy evaluation.

- **Final Model Training:** After validation, a new `RandomForestClassifier` is trained on the entire combined dataset to make use of all available information.

- **Serialization:** The final, trained model object is serialized using `joblib` and saved to the local `artifacts/` directory. This file is the primary output of the entire pipeline.

## 2. The Artifact Store (Hugging Face Hub)
This serves as our model registry.

- Decoupling: By hosting the model on an external service, the application's source code (in Git) is decoupled from the model artifact. This is essential, as model files can be large and should not be stored in a Git repository.

- **Versioning:** The Hugging Face Hub provides a simple way to version models. If we retrain a better model, we can upload it as a new version without changing the application code.

- **Accessibility:** It provides a simple, secure, and reliable way for our deployed Streamlit application to download the model on startup, regardless of where the application is hosted.

## 3. The Inference Service (`app.py` & `cli.py`)
These are the lightweight, user-facing applications that consume the trained model.

- **Model Loading:** On startup, the application downloads the `stodir_model.joblib` from the Hugging Face Hub using the `huggingface-hub` library. This is cached in memory for efficiency.

- **Live Data Fetching:** When a user requests a forecast for a ticker (e.g., "TSLA"), the app performs a small, fast API call to `yfinance` to get only the most recent data needed for feature calculation.

- **Real-time Prediction:** It applies the same feature engineering steps as the training pipeline and feeds the resulting vector into the `model.predict_proba()` method.

- **Efficiency:** Because the computationally expensive training and backtesting are already done, the user-facing app is extremely fast and responsive. It only performs a quick data fetch and a single prediction.