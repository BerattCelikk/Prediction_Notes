---

Bitcoin Price Prediction Using Linear Regression

This project aims to predict future Bitcoin prices using historical Bitcoin data with a Linear Regression model. The model enhances price prediction accuracy by incorporating moving averages as additional features. Below, you will find detailed explanations of each step of the code, along with key points to consider.

Project Overview

Predicting the prices of financial assets like Bitcoin provides significant advantages to investors in their decision-making processes. This project attempts to forecast future Bitcoin prices using a simple yet effective approach known as Linear Regression. You will learn how to perform data processing, feature engineering, and prediction modeling.


---

Libraries Used

The following libraries are used in this project:

import yfinance as yf  # Financial data retrieval (Bitcoin prices)
import pandas as pd  # Data manipulation and processing
import numpy as np  # Numerical calculations
from sklearn.linear_model import LinearRegression  # Linear regression model
import matplotlib.pyplot as plt  # Visualization
from datetime import datetime  # Date and time operations

Explanation:

yfinance: Allows us to retrieve historical data for financial assets like Bitcoin from Yahoo Finance. This is a critical step for financial modeling.

pandas: Used for organizing and manipulating data. It is particularly powerful for working with the DataFrame structure.

numpy: A fundamental library for numerical computations and operations, especially for array manipulation and vectorized calculations.

LinearRegression: A class from the scikit-learn library used for learning trends in our dataset and making predictions.

matplotlib: Helps create visual representations to make data analysis and model results more understandable.

datetime: Used for managing date and time data, particularly for operations related to data retrieval dates.



---

1. Fetching Bitcoin Data

We start by fetching the data from Yahoo Finance. The dataset used in this project includes Bitcoin prices from January 2022 to the present.

# Get today's date in 'YYYY-MM-DD' format
today = datetime.now().strftime('%Y-%m-%d')

# Download Bitcoin data from Yahoo Finance
btc_data = yf.download('BTC-USD', start='2022-01-01', end=today)
btc_data['Date'] = btc_data.index  # Add a 'Date' column based on the index

Key Point:

The yf.download() method is used to fetch historical price data for any financial asset via the Yahoo Finance API. This forms the foundation of our dataset.

Important Considerations: Correctly specifying the start and end dates when fetching data is critical for the model's proper functioning.



---

2. Adding Moving Averages

Moving averages (MA) are popular financial indicators used in predictions based on historical prices. Here, we add 7-day and 30-day moving averages.

# Add additional features: Moving Averages (MA)
btc_data['MA_7'] = btc_data['Close'].rolling(window=7).mean()  # 7-day moving average
btc_data['MA_30'] = btc_data['Close'].rolling(window=30).mean()  # 30-day moving average
btc_data.bfill(inplace=True)  # Backfill missing values

Key Point:

Moving Averages (MA): Short and long-term moving averages can indicate trends in prices and potential reversal points.

7-day MA: Monitors short-term price movements.

30-day MA: Reflects longer-term trends.


Why is this important?: Moving averages provide a basis for price predictions, helping the model to be more accurate.

Handling Missing Data: The bfill(inplace=True) command fills missing values afterward. This step prevents the model from being disrupted due to missing data.



---

3. Preparing Data for the Model

We need to prepare the data for our machine learning model. In this step, we convert date information into numerical values and separate independent variables from the dependent variable.

# Convert dates to numerical representation (days since the start)
btc_data['Days'] = (btc_data['Date'] - btc_data['Date'].min()).dt.days

# Define features (X) and target variable (y)
X = btc_data[['Days', 'MA_7', 'MA_30']]
y = btc_data['Close']

Key Point:

Numerical Representation of Dates: Converting date values to a numerical value (days since the start) helps the model better understand time-related trends.

Independent and Dependent Variables: The dependent variable (y) is the Bitcoin closing price that the model will predict. The independent variables (X) are the number of days, the 7-day MA, and the 30-day MA.



---

4. Training the Linear Regression Model

Now we train our linear regression model. This model learns the relationships in our data and prepares to predict future prices.

# Create the linear regression model
model = LinearRegression()
model.fit(X, y)  # Train the model

Key Point:

Advantages of Linear Regression: As a simple and fast model, linear regression performs well, especially when there is a linear relationship.

Model Training: The fit() function allows the model to learn patterns in the data. During training, the model learns the relationship between independent and dependent variables from historical data.



---

5. Predicting Future Prices

After training the model, we predict Bitcoin prices for the next 7 days.

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

Key Point:

Prediction Process: When predicting prices for future days, keeping the moving averages constant helps make the predictions more realistic.

Data Preparation for Future Predictions: A new dataset containing moving averages and day counts for the next 7 days is created, and the model uses this data to make predictions.



---

6. Visualizing Results

We visualize the predicted Bitcoin prices for better understanding.

# Get the last actual closing price and date
last_close = btc_data['Close'].iloc[-1]
last_date = btc_data['Date'].iloc[-1]

# Create dates for the predictions
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)

# Add the last actual closing price to the predictions list
predictions_with_last_close = [last_close] + list(predictions)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(btc_data['Date'], btc_data['Close'], label='Actual Prices', color='blue')
plt.plot(future_dates, predictions, label='Predicted Prices', color='orange', marker='o')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()

Key Point:

Visualization: Plotting both actual and predicted prices helps to assess the model's performance visually.

Understanding Trends: Visualizations provide insights into how well the model captures historical price movements and forecasts future prices.



---

Conclusion

This project demonstrates how to use historical Bitcoin price data to predict future prices using a linear regression model. The process includes data retrieval, feature engineering with moving averages, training a regression model, making predictions, and visualizing the results.

By following these steps, you can adapt this framework for other financial assets and enhance the model's accuracy with more sophisticated techniques or additional data features.


---

