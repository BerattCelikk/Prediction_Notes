---

📈 Bitcoin Price Prediction Using the ARIMA Model

This document comprehensively discusses the implementation of an ARIMA model to predict Bitcoin prices using Python. Below, each step is explained, including library imports, data downloading, stationarity checks, model training, and result visualization.

📚 Importing Libraries

import yfinance as yf  # Import yfinance library to fetch financial data
import pandas as pd  # Import pandas library for data manipulation
import numpy as np  # Import numpy library for numerical operations
from statsmodels.tsa.arima.model import ARIMA  # Import ARIMA model for time series analysis
from statsmodels.tsa.stattools import adfuller  # Import ADF test for stationarity check
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Import functions for model evaluation
import matplotlib.pyplot as plt  # Import matplotlib library for plotting graphs
from datetime import datetime  # Import datetime library for date manipulation

In this section, the libraries needed to retrieve and analyze Bitcoin price data are imported. Each library is briefly described below:

yfinance: For fetching financial data.

pandas: For data manipulation and analysis.

numpy: For numerical computations.

statsmodels: For statistical models and tests.

sklearn: For machine learning and model evaluation.

matplotlib: For data visualization.


📅 Getting the Current Date

end_date = datetime.now().strftime('%Y-%m-%d')  # Get the current date in YYYY-MM-DD format

This code retrieves the current date and assigns it to the variable end_date. This date will be used to fetch Bitcoin data.

📊 Downloading Bitcoin Data

btc_data = yf.download('BTC-USD', start='2023-01-01', end=end_date)  # Download Bitcoin data from Yahoo Finance

This section uses the Yahoo Finance API to download Bitcoin data for a specified date range, with the start date set to January 1, 2023.

📉 Using Closing Prices

btc_data = btc_data['Close'].asfreq('D')  # Filter to get closing prices and set to daily frequency

Here, we filter the data to use only the closing prices and set it to a daily frequency. This prepares the dataset to be used for model training.

📈 Stationarity Check

adf_result = adfuller(btc_data)  # Perform stationarity check using the ADF test
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')

if adf_result[1] > 0.05:
    print("The series is not stationary. Differencing is required.")
    btc_data = btc_data.diff().dropna()  # Apply differencing if the series is not stationary

In this part, the Augmented Dickey-Fuller (ADF) test is used to check the stationarity of the series. The ADF statistic and p-value are printed. If the p-value is greater than 0.05, it is concluded that the series is not stationary, and differencing is applied.

📊 Splitting the Data into Training and Testing Sets

train_size = int(len(btc_data) * 0.8)  # Calculate the size of the training set (80 percent)
train, test = btc_data[0:train_size], btc_data[train_size:]  # Split the data into training and testing sets

The data is split into training and testing sets at a ratio of 80% and 20%. The training set will be used to train the model, while the testing set will evaluate the accuracy of the model's predictions.

🔍 Finding the Best ARIMA Parameters

best_mse = float("inf")  # Initialize the best mean squared error as infinity
best_order = None  # Initialize the best ARIMA order as None

for p in range(3):  # Loop through possible values for p (0 to 2)
    for d in range(1):  # Use d=1 for differencing
        for q in range(3):  # Loop through possible values for q (0 to 2)
            try:
                model = ARIMA(train, order=(p, d, q), start_params=[0.1] * (p + q + 1))  # Create ARIMA model
                model_fit = model.fit(method='bfgs')  # Fit the model to the training data
                predictions = model_fit.forecast(steps=len(test))  # Make predictions for the test set
                
                mse = mean_squared_error(test, predictions)  # Calculate mean squared error
                if mse < best_mse:  # If the current MSE is less than the best MSE
                    best_mse = mse  # Update the best MSE
                    best_order = (p, d, q)  # Update the best parameters
            except Exception as e:
                print(f"Error for order (p={p}, d={d}, q={q}): {e}")  # Print error if occurs

This loop trains the model with different ARIMA parameters (p, d, q) and calculates the mean squared error (MSE) of the predictions on the test set. The parameters that yield the lowest MSE are recorded. The start_params argument is used to set the initial values for the model.

🏆 Printing the Best Parameters and Applying the Model

if best_order is not None:
    print(f'Best ARIMA parameters: p={best_order[0]}, d={best_order[1]}, q={best_order[2]} with MSE={best_mse:.2f}')  # Print the best parameters and MSE

    # Fit the best model and make predictions
    best_model = ARIMA(train, order=best_order)  # Create the best ARIMA model with the best parameters
    best_model_fit = best_model.fit()  # Fit the best model to the training data
    forecast = best_model_fit.forecast(steps=7)  # Make predictions for the next 7 days

The best ARIMA parameters and MSE value are printed. Then, the best model is created and fitted. This model makes predictions for the next 7 days.

💵 Storing Predictions

# Last closing price
last_close = btc_data.iloc[-1]  # Get the last closing price from the dataset

# Store the predicted prices in a DataFrame
predicted_prices = pd.DataFrame({
    'Date': pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=7),  # Generate dates for predictions
    'Predicted Price': forecast  # Add predicted prices
})

# Set the first prediction to the last closing price
predicted_prices.loc[0, 'Predicted Price'] = last_close  # Set the first prediction as the last closing price
predicted_prices['Predicted Price'] = predicted_prices['Predicted Price'].shift(-1)  # Shift predictions up by one

The last closing price is retrieved, and a DataFrame is created to hold the predicted prices. The first prediction is set as the last closing price, and the other predictions are shifted up by one position.

📉 Error Calculations

# Remove NaN values
predicted_prices = predicted_prices.dropna()  # Remove rows with NaN values

# Calculate MSE and MAE
mse = mean_squared_error(test[-len(predicted_prices):], predicted_prices['Predicted Price'])  # Calculate MSE for predictions
mae = mean_absolute_error(test[-len(predicted_prices):], predicted_prices['Predicted Price'])  # Calculate MAE for predictions

print(predicted_prices)

NaN values are removed, and both mean squared error (MSE) and mean absolute error (MAE) are calculated for the predictions. Finally, the predicted prices are printed.


---
