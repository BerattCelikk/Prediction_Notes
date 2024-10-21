---

🪙 Bitcoin Price Prediction Using Linear Regression

This project aims to predict future Bitcoin prices using historical data with a Linear Regression model. The model incorporates moving averages as additional features to enhance prediction accuracy. Below, you'll find detailed explanations and tips for each step of the code.

📚 Project Overview

Predicting prices of financial assets like Bitcoin provides significant advantages for decision-making. This project utilizes a straightforward yet effective approach using a Linear Regression model to forecast future Bitcoin prices. You will learn how to perform data processing, feature engineering, and prediction modeling.


---

📦 Libraries Used

import yfinance as yf  # Fetch financial data (Bitcoin prices)
import pandas as pd  # Data processing and manipulation
import numpy as np  # Numerical calculations
from sklearn.linear_model import LinearRegression  # Linear regression model
import matplotlib.pyplot as plt  # Visualization
from datetime import datetime  # Date and time operations

Explanation:

yfinance: Allows us to fetch historical data for financial assets like Bitcoin, which is crucial for financial modeling.

pandas: Used for organizing and manipulating data, especially with its powerful DataFrame structure.

numpy: A foundational library for numerical operations and calculations.

LinearRegression: This class from scikit-learn helps us learn trends in our dataset and make predictions.

matplotlib: Enables us to create visuals to make our data analysis and model results more comprehensible.

datetime: Used for managing date and time-related data, like the date we pull our dataset.



---

📥 1. Fetching Bitcoin Data

We begin by fetching data from Yahoo Finance. The dataset used in this project contains Bitcoin prices from 2022 to the present.

# Get today's date in 'YYYY-MM-DD' format
today = datetime.now().strftime('%Y-%m-%d')

# Download Bitcoin data from Yahoo Finance
btc_data = yf.download('BTC-USD', start='2022-01-01', end=today)
btc_data['Date'] = btc_data.index  # Add a 'Date' column based on the index

Tips:

The yf.download() method is used to pull historical price data for any financial asset via the Yahoo Finance API. This forms the backbone of our dataset.

Important: Properly defining the start and end dates during data retrieval is critical for the model to function effectively.



---

📈 2. Adding Features (Moving Averages)

Moving averages (MA) are popular financial indicators used in price forecasting. Here, we add both a 7-day and a 30-day moving average.

# Add additional features: Moving Averages (MA)
btc_data['MA_7'] = btc_data['Close'].rolling(window=7).mean()  # 7-day moving average
btc_data['MA_30'] = btc_data['Close'].rolling(window=30).mean()  # 30-day moving average
btc_data.bfill(inplace=True)  # Backfill missing values

Tips:

Moving Averages (MA): Short-term and long-term moving averages can indicate price trends and potential reversal points.

7-day MA: Monitors short-term price movements.

30-day MA: Indicates longer-term trends.


Why it Matters: Moving averages provide a foundation for price forecasting, helping the model become more accurate.

Handling Missing Data: The bfill() method backfills any missing data to prevent the model from failing due to NaN values.



---

🛠️ 3. Preparing Data for the Model

We need to prepare the data for our machine learning model. In this step, we convert date information into numerical values and separate independent and dependent variables.

# Convert dates to numerical representation (days since the start)
btc_data['Days'] = (btc_data['Date'] - btc_data['Date'].min()).dt.days

# Define features (X) and target variable (y)
X = btc_data[['Days', 'MA_7', 'MA_30']]
y = btc_data['Close']

Tips:

Numerical Representation of Dates: Converting date values to a numerical form (days elapsed) helps the model better understand time-related trends.

Independent and Dependent Variables: The target variable (y) is the Bitcoin closing price, while the independent variables (X) include the number of elapsed days, 7-day MA, and 30-day MA.



---

🧠 4. Training the Linear Regression Model

We train our linear regression model using the prepared data. This model learns the relationships in our data and is ready to predict future prices.

# Create the linear regression model
model = LinearRegression()
model.fit(X, y)  # Train the model

Tips:

Advantages of Linear Regression: A simple and fast model, linear regression performs well when there is a linear relationship present.

Model Training: The fit() function allows the model to learn patterns from the data. During training, the model understands the relationship between independent and dependent variables.



---

📅 5. Predicting Future Prices

After training, we predict Bitcoin prices for the next 7 days.

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

Tips:

Prediction Process: While forecasting future prices, maintaining the last observed values of moving averages ensures realistic predictions.

Data Preparation for Future: We create a new dataset containing future days and moving averages, which is used for the predictions.



---

📊 6. Visualizing the Results

We visualize the predicted Bitcoin prices alongside the actual prices.

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

Tips:

Creating Visuals: This code block creates a visualization of actual versus predicted prices.

Date Management: The future_dates variable generates dates for the upcoming 7 days.

Continuity: Including the last actual closing price ensures the prediction list appears continuous.



---

🖨️ 7. Displaying Predicted Prices

Finally, we display the predicted prices in a DataFrame format.

# Print the predicted prices as a DataFrame
predicted_prices = pd.DataFrame({'Date': future_dates_with_last_close, 'Predicted Price': predictions_with_last_close})
print(predicted_prices)

Tips:

Clear Display of Predictions: The predicted prices for the upcoming 7 days are presented in a DataFrame format, offering a clear view of future Bitcoin price predictions.



---

🚨 Important Note:

This project is for educational purposes only and should not be considered as financial advice.


---

