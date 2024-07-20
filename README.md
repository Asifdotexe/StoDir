# StoDir - Stock Direction Forecasting Model

This project implements a stock direction forecasting model that forecasts whether the stock price will go up or down based on historical data. The model is built using Python and leverages various data processing, visualization, and machine learning techniques.

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

The goal of this project is to predict stock price movements for any particular stock. The model suggests whether the stock price will increase or decrease in the near future based on historical stock price data. This project uses various Python libraries including Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.

## Dataset

The dataset used in this project is historical stock price data for Apple Inc. from January 1, 2000, to June 28, 2024. The data is sourced from Yahoo Finance using the `yfinance` library.

The dataset includes the following columns:
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
- `target`: A flag indicating if the next day's closing price is higher than today's closing price.
- Rolling averages and trends for different horizons (2 days, 5 days, 60 days, 250 days, 1000 days).

## Model Training

A RandomForestClassifier from scikit-learn is used to train the model. The features used for training are:
- `open`
- `close`
- `high`
- `low`
- `volume`

The model is trained on a portion of the data and tested on the remaining data to evaluate its performance.

## Backtesting

Backtesting is performed to evaluate the model's performance over time. The predictions are made iteratively on different portions of the dataset to simulate a real-world trading scenario.

## Results

The model's precision score is evaluated, and the actual vs. predicted results are visualized using bar plots and line plots.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/stock-price-prediction.git
    cd stock-price-prediction
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the project, follow these steps:

1. Run the Jupyter Notebook containing the code:
    ```bash
    jupyter notebook
    ```

2. Open the notebook and execute the cells to perform data analysis, feature engineering, model training, and backtesting.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
