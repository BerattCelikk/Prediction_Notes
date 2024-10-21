---

📈 Bitcoin Price Prediction Using Prophet

🥇 Overview

This project implements a time series forecasting model to predict Bitcoin prices for the upcoming week using the Prophet library. By leveraging historical data sourced from Yahoo Finance, we can gain insights into future price trends and evaluate the model's performance through various metrics. The goal is to provide a robust forecasting tool that can assist in understanding market behavior.


---

🔍 Table of Contents

1. 🛠️ Prerequisites


2. 📦 Getting Started


3. 📊 Data Fetching


4. 🧹 Data Preparation


5. 📅 Splitting Data


6. 📈 Creating and Fitting the Model


7. 🧪 Evaluating Predictions


8. 📊 Visualizing the Results


9. 📝 Results Summary


10. 🔮 Future Work


11. ⚠️ Notes


12. 📜 References




---

🛠️ Prerequisites

To run this project, make sure you have the following Python packages installed:

yfinance: For fetching historical financial data.

pandas: For data manipulation and analysis.

prophet: For time series forecasting.

matplotlib: For data visualization.

scikit-learn: For model evaluation metrics.


You can install these packages using pip:

pip install yfinance pandas prophet matplotlib scikit-learn


---

📦 Getting Started

Follow these steps to clone the repository and set up the environment:

1. Clone the Repository:

git clone https://github.com/yourusername/bitcoin-price-prediction.git
cd bitcoin-price-prediction


2. Install Dependencies: Follow the prerequisites section to install the necessary packages.


3. Run the Prediction Script:

python predict_bitcoin.py




---

📊 Data Fetching

The data is fetched using the yfinance library, which retrieves historical Bitcoin price data from January 1, 2018, to the current date.

import yfinance as yf  # Importing the yfinance library to fetch financial data
from datetime import datetime  # Importing datetime to handle date objects

# Fetch Bitcoin data
today = datetime.now().strftime("%Y-%m-%d")  # Getting today's date in 'YYYY-MM-DD' format
btc_data = yf.download('BTC-USD', start='2018-01-01', end=today)  # Download Bitcoin price data from 2018 to today

📈 Data Structure

The downloaded data includes the following columns:

Open: The price at the start of the trading day.

High: The highest price during the trading day.

Low: The lowest price during the trading day.

Close: The price at the end of the trading day.

Volume: The total volume of trades during the day.

Adj Close: The adjusted closing price, which accounts for any corporate actions.


For our analysis, we will focus on the Close price.


---

🧹 Data Preparation

After fetching the data, we clean it by dropping any missing values and renaming the columns for compatibility with the Prophet model.

import pandas as pd  # Importing pandas for data manipulation

# Resetting index and keeping only the 'Close' price
btc_data = btc_data[['Close']].reset_index()  
btc_data.dropna(inplace=True)  # Dropping any rows with missing values

# Prepare data format for Prophet
btc_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)  # Renaming columns for Prophet

🧹 Data Cleaning Steps

1. Drop Missing Values: Ensure no rows contain NaN values.


2. Format Dates: Convert date columns to datetime format for accurate plotting and analysis.




---

📅 Splitting Data

The dataset is divided into training and testing sets, with the last 30 days used for testing the model.

# Split the data into the last 30 days for testing
train_data = btc_data[:-30]  # Training data (all data except the last 30 days)
test_data = btc_data[-30:]    # Test data (last 30 days)

📅 Why Split the Data?

Training Set: Used to train the model and capture patterns in the data.

Testing Set: Used to evaluate how well the model performs on unseen data.



---

📈 Creating and Fitting the Model

A Prophet model is created and fitted to the training data, with future predictions being made for the next 7 days.

from prophet import Prophet  # Importing the Prophet library for time series forecasting

# Create and fit the Prophet model
model = Prophet(yearly_seasonality=True, daily_seasonality=True)  # Create a Prophet model with yearly and daily seasonality
model.fit(train_data)  # Fit the model to the training data

# Create future dates for prediction
future = model.make_future_dataframe(periods=7)  # Creating a dataframe for the next 7 days for prediction
forecast = model.predict(future)  # Generate forecasts for future dates

📈 Model Parameters

Yearly Seasonality: Captures yearly trends in the data.

Daily Seasonality: Captures daily patterns, which may be relevant in cryptocurrency trading.



---

🧪 Evaluating Predictions

Predictions are compared to actual values from the test dataset, and error metrics are calculated to evaluate model performance.

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Importing metrics for evaluation

# Compare predicted values with actual test data
predicted_values = forecast['yhat'].iloc[-7:].values  # Extract the predicted values for the last 7 days
actual_values = test_data['y'].iloc[-7:].values  # Extract the actual values from the test data

# Calculate error metrics
mae = mean_absolute_error(actual_values, predicted_values)  # Calculate Mean Absolute Error
mse = mean_squared_error(actual_values, predicted_values)  # Calculate Mean Squared Error
r2 = r2_score(actual_values, predicted_values)  # Calculate R² Score

📊 Error Metrics Explained

Mean Absolute Error (MAE): Measures the average magnitude of the errors in a set of predictions, without considering their direction.

Mean Squared Error (MSE): Measures the average of the squares of the errors—i.e., the average squared difference between estimated values and actual values.

R² Score: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variable(s).



---

📊 Visualizing the Results

The results are visualized using Matplotlib, which plots actual and predicted values along with confidence intervals.

import matplotlib.pyplot as plt  # Importing matplotlib for data visualization

# Visualize the results
plt.figure(figsize=(14, 7))  # Create a new figure with a specified size

# Plot actual values
plt.plot(btc_data['ds'], btc_data['y'], label='Actual Values', color='blue', linewidth=2)  # Plotting actual closing prices

# Plot predicted values
plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Values', color='orange', linewidth=2)  # Plotting predicted prices

# Show confidence intervals
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.3, label='Prediction Interval')  # Filling the area between lower and upper bounds of predictions

# Indicate today's date on the plot
plt.axvline(x=pd.to_datetime(today), color='red', linestyle='--', label='Today\'s Date')  # Drawing a vertical line for today's date
plt.scatter(forecast['ds'].iloc[-7:], predicted_values, color='green', label='1 Week Predictions', s=100)  # Scatter plot for the predicted values

# Display error metrics on the plot
plt.text(forecast['ds'].iloc[-1], predicted_values[-1], f'MAE: {mae:.2f}', horizontalalignment='left', fontsize=10, color='black')  # Adding text for MAE on the plot

plt.title('Bitcoin Price Prediction for the Next Week')  # Title of the plot
plt.xlabel('Date')  # X-axis label
plt.ylabel('Price (USD)')  # Y-axis label
plt.legend()  # Displaying the legend
plt.grid()  # Adding a grid to the plot
plt.show()  # Show the plot

📊 Sample Visualization

The plot illustrates the actual closing prices (in blue), predicted values (in orange), and the prediction intervals (in gray). A vertical red line marks today's date for reference.


---

📝 Results Summary

At the end of the execution, the last seven days of predictions, error metrics, and the last closing price are printed.

# Print summary of results
print(f"Predicted Values for the Next Week: {predicted_values}")
print(f"Last Closing Price: {actual_values[-1]}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")


---

🔮 Future Work

Explore Additional Features: Investigate the impact of including additional features such as trading volume, social media sentiment analysis, and macroeconomic indicators to enhance prediction accuracy.

Model Comparison: Test and compare various forecasting models (e.g., ARIMA, LSTM) to identify which provides the best performance for Bitcoin price prediction.

Real-Time Predictions: Implement a real-time prediction system that updates forecasts based on the latest available data.

User Interface Development: Create a user-friendly application that allows users to visualize predictions and interact with the model.

Risk Analysis: Conduct a thorough risk analysis to better understand potential downsides and market volatility associated with Bitcoin investments.



---

⚠️ Notes

Disclaimer: This project is for educational purposes only and does not constitute financial advice. Users are encouraged to conduct their own research before making investment decisions.

Data Limitations: The accuracy of the predictions depends heavily on the quality and range of historical data. Sudden market changes and external factors may lead to unexpected results.


---






