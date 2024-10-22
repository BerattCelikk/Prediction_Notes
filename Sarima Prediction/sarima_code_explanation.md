---

🚀 BITCOIN PRICE PREDICTION USING THE SARIMA MODEL

This Python script employs the SARIMA (Seasonal Autoregressive Integrated Moving Average) model to predict Bitcoin prices for the upcoming week. It leverages historical price data sourced from Yahoo Finance, trains the SARIMA model on this data, and generates forecasts for the next seven days.


---

📥 1. DOWNLOADING BITCOIN DATA

We begin by importing necessary libraries and downloading historical Bitcoin price data. The time range is set to cover the last five years.

import yfinance as yf  # For downloading financial data from Yahoo Finance
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For plotting graphs
from statsmodels.tsa.statespace.sarimax import SARIMAX  # For the SARIMA model
from datetime import datetime, timedelta  # For handling date and time

# Set the current date as the end date
end_date = datetime.now()  # Current date

# Set the start date to 5 years ago from today
start_date = end_date - timedelta(days=365*5)  # 5 years ago from the current date

# Download the Bitcoin price data
btc_data = yf.download('BTC-USD', start=start_date, end=end_date)  # Download BTC price data

# Extract the 'Close' price
btc_close = btc_data['Close']  # Extract the 'Close' price data

Key Components:

Libraries Used:

yfinance: To fetch financial data.

pandas: For data manipulation and analysis.

matplotlib: For creating visualizations.

statsmodels: To implement the SARIMA model.

datetime: For handling date and time functions.


Data Range:

end_date: Set to the current date.

start_date: Calculated as five years prior to the current date.



The closing prices are extracted for model training.


---

📊 2. TRAINING THE SARIMA MODEL

Next, we initialize and fit the SARIMA model using the historical closing prices of Bitcoin. SARIMA incorporates seasonality into the traditional ARIMA model.

# Initialize the SARIMA model with parameters
sarima_model = SARIMAX(btc_close, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))  
# Fit the model
sarima_result = sarima_model.fit(disp=False)  # Fit the model, 'disp=False' suppresses output

SARIMA Parameters Explained:

ARIMA Part: order=(p, d, q)

p = 1: The number of lag observations included in the model (Auto-Regressive term).

d = 1: The number of times that the raw observations are differenced (Differencing).

q = 1: The size of the moving average window (Moving Average term).


Seasonal Part: seasonal_order=(P, D, Q, s)

P = 1: The number of seasonal lag observations included in the model.

D = 1: The number of seasonal differences.

Q = 1: The size of the seasonal moving average window.

s = 7: The length of the seasonal cycle (weekly seasonality).




---

📅 3. FORECASTING THE NEXT 7 DAYS

After training the model, we generate forecasts for the next seven days, starting from the day following the last available data point.

# Forecast the next 7 days
forecast = sarima_result.get_forecast(steps=7)  # Forecast for the next 7 days

# Create a date range for the forecast period (7 days after the last historical date)
forecast_index = pd.date_range(start=btc_close.index[-1] + timedelta(days=1), periods=7)  

# Get the predicted mean values for the forecasted days
forecast_values = forecast.predicted_mean  # Predicted mean values for the next 7 days

Forecasting Details:

get_forecast(steps=7): This function generates predictions for the next 7 days.

forecast_index: Creates a range of dates for the forecast period, starting the day after the last observed price.

forecast_values: Contains the predicted closing prices for the next week.



---

🗓️ 4. COMBINING HISTORICAL AND FORECASTED DATA

To visualize both historical and forecasted data, we combine them into a single dataset.

# Combine historical and forecast dates
combined_dates = btc_close.index.tolist() + forecast_index.tolist()  # Combine historical and forecast dates

# Combine historical and forecast values
combined_values = btc_close.tolist() + forecast_values.tolist()  # Combine historical and forecast values

Combined Data:

combined_dates: A list that includes both historical and forecast dates.

combined_values: A list that contains both historical and forecast price values.



---

🔮 5. PRINTING THE FORECASTED RESULTS

We then print the forecasted Bitcoin prices for the next seven days.

# Print the forecasted Bitcoin prices for the next 7 days
print("1 Week Bitcoin Forecast:")
for date, price in zip(forecast_index, forecast_values):
    print(f"Date: {date.date()}, Predicted Price: {price:.2f} USD")  # Print forecasted date and price

Example Output:

1 Week Bitcoin Forecast:
Date: 2024-10-22, Predicted Price: 67544.68 USD
Date: 2024-10-23, Predicted Price: 67669.15 USD


---

📉 6. PRINTING THE LAST WEEK'S ACTUAL CLOSING PRICES

To evaluate the model's accuracy, we also print the actual Bitcoin closing prices for the last week.

# Print the actual Bitcoin closing prices for the last 7 days
print("\nLast 1 Week Actual Bitcoin Closing Prices:")
last_week_actual = btc_close[-7:]  # Select the last 7 days of closing prices
for date, price in zip(last_week_actual.index, last_week_actual.values):
    print(f"Date: {date.date()}, Actual Closing Price: {price:.2f} USD")  # Print actual date and price

Example Output:

Last 1 Week Actual Bitcoin Closing Prices:
Date: 2024-10-15, Actual Closing Price: 67041.11 USD


---

📈 7. PLOTTING HISTORICAL AND FORECASTED DATA

Finally, we visualize the historical and forecasted Bitcoin prices using Matplotlib.

# Set the figure size for the plot
plt.figure(figsize=(10, 6))  

# Plot historical data
plt.plot(combined_dates, combined_values, label='Historical Bitcoin Prices')  

# Plot the 1-week forecast
plt.plot(forecast_index, forecast_values, label='1 Week Forecast', color='red')  

# Mark the last available historical data point
plt.axvline(x=btc_close.index[-1], color='gray', linestyle='--', label='Last Historical Date')  

# Display the legend
plt.legend()  

# Show the plot
plt.show()

Visualization Details:

Historical Prices: Plotted in the default color.

Forecast Prices: Shown in red to distinguish from historical data.

Vertical Line: Indicates the transition from historical data to forecast.



---

📝 SUMMARY

Historical Data: Downloaded using yfinance, encompassing the past five years of Bitcoin closing prices.

SARIMA Model: Trained using specific parameters that account for both trend and seasonality in the data.

Forecast: Generates predicted prices for the next week based on historical trends.

Visualization: Combines historical and forecasted data into a clear, informative plot for analysis.



---