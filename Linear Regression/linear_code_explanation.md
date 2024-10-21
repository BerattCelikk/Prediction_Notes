---

Bitcoin Price Prediction Using Linear Regression

This project uses historical Bitcoin data to predict future prices using Linear Regression. The model incorporates moving averages as additional features to enhance prediction accuracy. Below, you'll find the detailed steps involved in the code.

Libraries Used

import yfinance as yf  # Used to fetch financial data
import pandas as pd  # For data manipulation
import numpy as np  # For numerical calculations
from sklearn.linear_model import LinearRegression  # Linear regression model
import matplotlib.pyplot as plt  # For plotting
from datetime import datetime  # For working with dates

Explanation:

yfinance: Fetches financial data (here, Bitcoin prices).

pandas: For handling data structures, mainly DataFrame.

numpy: Provides support for numerical calculations.

LinearRegression: Model for linear regression from scikit-learn.

matplotlib: To visualize data using plots.

datetime: Deals with date and time-related functions.



---

1. Fetching Bitcoin Data

# Get today's date in 'YYYY-MM-DD' format
today = datetime.now().strftime('%Y-%m-%d')

# Download Bitcoin data from Yahoo Finance
btc_data = yf.download('BTC-USD', start='2022-01-01', end=today)
btc_data['Date'] = btc_data.index  # Add a 'Date' column based on the index

Explanation:

The code fetches Bitcoin data from Yahoo Finance starting from January 1, 2022, up to today's date.

today stores the current date in the format 'YYYY-MM-DD'.

btc_data contains the historical Bitcoin prices (Open, High, Low, Close, etc.) and the date is set as the index of the DataFrame.



---

2. Adding Features (Moving Averages)

# Add additional features: Moving Averages (MA)
btc_data['MA_7'] = btc_data['Close'].rolling(window=7).mean()  # 7-day moving average
btc_data['MA_30'] = btc_data['Close'].rolling(window=30).mean()  # 30-day moving average
btc_data.bfill(inplace=True)  # Backfill missing values

Explanation:

MA_7: 7-day moving average of Bitcoin’s closing price.

MA_30: 30-day moving average.

These moving averages are added as new columns in the DataFrame to be used as features in the model.

bfill ensures that any missing data is backfilled to avoid NaN values in the DataFrame.



---

3. Preparing Data for the Model

# Convert dates to numerical representation (days since the start)
btc_data['Days'] = (btc_data['Date'] - btc_data['Date'].min()).dt.days

# Define features (X) and target variable (y)
X = btc_data[['Days', 'MA_7', 'MA_30']]
y = btc_data['Close']

Explanation:

Days column is created to represent the number of days since the start of the dataset. This converts the date to a numerical value.

The features (X) include Days, MA_7, and MA_30.

The target variable (y) is the closing price of Bitcoin.



---

4. Training the Linear Regression Model

# Create the linear regression model
model = LinearRegression()
model.fit(X, y)  # Train the model

Explanation:

A LinearRegression model is initialized and then trained using the features (X) and target variable (y). This prepares the model to make predictions based on the historical data.



---

5. Predicting Future Prices

# Get the numerical representation of the last day
last_day = btc_data['Days'].max()

# Prepare features for the next 7 days
future_days = np.array(range(last_day + 1, last_day + 8)).reshape(-1, 1)
future_ma_7 = np.repeat(btc_data['MA_7'].iloc[-1], 7).reshape(-1, 1)
future_ma_30 = np.repeat(btc_data['MA_30'].iloc[-1], 7).reshape(-1, 1)

# Combine future features into a DataFrame
future_features = pd.DataFrame({
    'Days': future_days.flatten(),
    'MA_7': future_ma_7.flatten(),
    'MA_30': future_ma_30.flatten()
})

# Make predictions with the model
predictions = model.predict(future_features)

Explanation:

last_day: The last day in the dataset.

future_days: Prepares an array for the next 7 days.

future_ma_7 and future_ma_30: Repeats the last observed values of the moving averages.

Combines future days and moving averages into a DataFrame and makes predictions for the next 7 days.



---

6. Visualizing the Results

# Get the last actual closing price and date
last_close = btc_data['Close'].iloc[-1]
last_date = btc_data['Date'].iloc[-1]

# Create dates for the predictions
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)

# Add the last actual closing price to the predictions list
predictions_with_last_close = [last_close] + list(predictions)
future_dates_with_last_close = pd.date_range(start=last_date, periods=8)

# Visualize the results
plt.figure(figsize=(14, 7))
plt.plot(btc_data['Date'], y, label='Actual Price', color='blue')
plt.plot(future_dates_with_last_close, predictions_with_last_close, label='Predicted Price', color='orange')
plt.title('Bitcoin Price Prediction (Linear Regression)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

Explanation:

The code creates a visualization of the actual and predicted prices.

future_dates: The predicted dates for the next 7 days.

The last actual closing price is added to the list of predictions to show continuity.

The plot shows actual prices in blue and predicted prices in orange.



---

7. Displaying Predicted Prices

# Print the predicted prices as a DataFrame
predicted_prices = pd.DataFrame({'Date': future_dates_with_last_close, 'Predicted Price': predictions_with_last_close})
print(predicted_prices)

Explanation:

The predicted prices for the next 7 days are displayed in a DataFrame format.

This provides a clear tabular view of the future Bitcoin price predictions.



---

Final Output

At the end of the script, the model outputs the predicted prices for the next 7 days, starting from the last available closing price.


---

Important Note:

This project is for educational purposes only and should not be considered as financial advice.


---

