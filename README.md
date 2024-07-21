# StoDir - Stock Direction Forecasting Model

This project implements a stock direction forecasting model that predicts whether the stock price will go up or down based on historical data. The model is built using Python and leverages various data processing, visualization, and machine learning techniques.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Analysis and Visualization](#data-analysis-and-visualization)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Backtesting](#backtesting)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

The goal of this project is to predict stock price movements for any given stock. The model forecasts whether the stock price will increase or decrease in the near future based on historical stock price data. This project utilizes Python libraries including Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn to perform data analysis, feature engineering, and model training.

## Dataset

The dataset used in this project consists of historical stock price data for a given stock symbol, fetched from Yahoo Finance using the `yfinance` library. The data includes the following columns:
- `open`: The opening price of the stock.
- `high`: The highest price of the stock during the trading day.
- `low`: The lowest price of the stock during the trading day.
- `close`: The closing price of the stock.
- `volume`: The total number of shares traded during the trading day.
- `dividends`: Payments made to shareholders.
- `stock splits`: Adjustments in the number of shares.

## Data Analysis and Visualization

Various visualizations are created to understand the data better:
- Line plots for `open`, `high`, `low`, `close`, and `volume` over time.
- Histograms to understand the distribution of each metric.
- Candlestick chart for the last 30 days of stock prices.
- Correlation heatmap to understand the relationships between different features.

## Feature Engineering

New features are created to improve the model's performance:
- `tomorrow`: The closing price of the next day.
- `target`: A binary indicator that takes the value 1 if the closing price of the next day is higher than the current closing price, and 0 otherwise.
- Rolling averages and trends for different horizons (2 days, 5 days, 60 days, 250 days, 1000 days).

## Model Training

A RandomForestClassifier from scikit-learn is used to train the model. The features used for training include rolling averages of closing prices over different horizons. The model is trained on a portion of the data and tested on the remaining data to evaluate its performance.

## Backtesting

Backtesting is performed to evaluate the model's performance over time. Predictions are made iteratively on different portions of the dataset to simulate a real-world trading scenario. The model’s precision score is calculated and visualized alongside actual vs. predicted results.

## Results

The results are presented through various visualizations:
- Line plots showing actual vs. predicted stock price movements.
- Precision score of the model, indicating the accuracy of predictions.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Asifdotexe/StoDir.git
    cd StoDir
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv stodir_env
    source stodir_env/bin/activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the project, follow these steps:

1. Run the script to fetch data, train the model, and make predictions:
    ```bash
    python StoDir.py <ticker> [horizons]
    ```
    - `<ticker>`: Stock ticker symbol (e.g., AAPL).
    - `[horizons]`: Optional list of horizons to include as features (e.g., 2 5 60 250 1000).

2. The script will save plots and predictions to files in the `plots` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
