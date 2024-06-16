# Stock Market Prediction Model

## Objective

The objective of this project is to create a stock market prediction model that forecasts future trends of stock prices. This model aims to provide accurate predictions to assist in making informed trading decisions.

## Project Workflow

### 1. Data Collection

Data collection is a critical first step in the project. We gathered historical stock prices for the company Apple Inc. (AAPL) over the past several years. The data was sourced from Google Finance and includes the following features:
- Date
- Open
- High
- Low
- Close
- Volume
- Adjusted Close

[How to extract the data](https://support.google.com/docs/answer/3093281?hl=en-GB)

### 2. Feature Engineering

Feature engineering involves transforming raw data into meaningful features that improve the performance of the machine learning model. For this project, we performed the following steps:
- **Date Parsing**: Parsed the 'Date' column to a datetime format.
- **Moving Averages**: Calculated moving averages (e.g., 7-day, 30-day) to capture trends and smooth out short-term fluctuations.
- **Lag Features**: Created lag features to incorporate past stock prices into the model.

### 3. Model Selection

We employed a grid search with cross-validation to select the best regression model for the task. The models considered include:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor

### 4. Model Training

The model training process involves splitting the data into training and testing sets, training the model on the training set, and evaluating its performance on the testing set. We used Mean Squared Error (MSE) as the performance metric. The model with the lowest MSE was chosen as the best model.

### 5. Visualization

Finally, we visualized the model's predictions against the actual stock prices to assess its performance. We focused on the last 30 days of the dataset to understand the model's recent accuracy.

## Installation and Usage

To run this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/stock-market-prediction-model.git
   cd stock-market-prediction-model
   ```
   
2. **Install Dependencies**:
Make sure you have Python installed, then install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Run the Model**:
You can run the Jupyter notebooks provided in the repository to perform data preprocessing, model training, and visualization.

## Results

The best performing model was the RandomForestRegressor with the following parameters:
   ```
   n_estimators: 100
   max_depth: 20
   criterion: 'poisson'
   ```

The model achieved a Mean Squared Error (MSE) of **-13.826217** on the test set.
