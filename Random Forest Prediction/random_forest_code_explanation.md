---

🌟 Bitcoin Price Prediction Using Random Forest Model 🌟

<p align="center">
This project utilizes a **Random Forest Regressor** to predict Bitcoin's future prices using historical price data and key indicators like **returns**, **volatility**, and **moving averages**. The goal is to forecast Bitcoin prices for the next 7 days based on these features. Let's explore the code in detail! 🚀
</p>
---

📥 Step 1: Importing Required Libraries

<p align="center">
First, we need to import the required libraries for data manipulation, model building, and visualization.
</p>import yfinance as yf  # Fetch financial data from Yahoo Finance
import pandas as pd  # Data manipulation and analysis
from sklearn.model_selection import train_test_split, GridSearchCV  # Data splitting and hyperparameter tuning
from sklearn.ensemble import RandomForestRegressor  # Random Forest Regressor for predictions
from sklearn.metrics import mean_squared_error  # To evaluate model performance
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting graphs
from datetime import datetime, timedelta  # Handling date and time


---

📈 Step 2: Fetching Bitcoin Data

<p align="center">
We retrieve the Bitcoin price data from Yahoo Finance starting from January 1, 2020, until today's date.
</p>data = yf.download('BTC-USD', start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'))


---

🔍 Step 3: Feature Engineering

<p align="center">
In this step, we create several important features:
</p>data['Return'] = data['Close'].pct_change()  # Daily return
data['Volatility'] = data['Return'].rolling(window=7).std()  # 7-day rolling volatility
data['MA_5'] = data['Close'].rolling(window=5).mean()  # 5-day moving average
data['MA_10'] = data['Close'].rolling(window=10).mean()  # 10-day moving average
data['MA_30'] = data['Close'].rolling(window=30).mean()  # 30-day moving average
data.dropna(inplace=True)  # Remove NaN values

📝 Explanation:

Return helps capture the rate of change in the price.

Volatility gives a sense of price stability or instability over the last week.

Moving Averages smooth out short-term fluctuations to highlight longer-term trends.



---

✂️ Step 4: Data Preparation

<p align="center">
Next, we prepare the dataset by selecting the relevant features. The target variable is the next day's closing price, so we shift the closing price by one day.
</p>X = data[['Return', 'Volatility', 'MA_5', 'MA_10', 'MA_30']]  # Features
y = data['Close'].shift(-1)  # Target variable (shifted closing price)

X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)

📝 Explanation:

X: The independent variables or features.

y: The target or dependent variable (Bitcoin price).

train_test_split: Divides the data into training (80%) and testing (20%) sets.



---

🧠 Step 5: Model Training and Hyperparameter Tuning

<p align="center">
We tune the Random Forest model using **GridSearchCV**, testing different numbers of trees and depths for the trees.
</p>param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [None, 10, 20, 30],  # Max depth of the trees
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)  # Train the model

best_model = grid_search.best_estimator_  # Best model

📝 Explanation:

GridSearchCV: Used to automatically find the best combination of hyperparameters.

n_estimators: Controls the number of decision trees in the forest.

max_depth: Limits the depth of the trees to prevent overfitting.



---

🔮 Step 6: Making Predictions

<p align="center">
Now, we predict the Bitcoin prices for the testing set and evaluate the model's performance using **Mean Squared Error (MSE)**.
</p>predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Test MSE: {mse}')

📝 Explanation:

Mean Squared Error (MSE) is a commonly used loss function that tells us how well the model’s predictions match the actual values. The lower the MSE, the better the model.



---

📅 Step 7: Predicting Future Prices

<p align="center">
We now use the trained model to predict Bitcoin's price for the next 7 days, updating the input features iteratively.
</p>last_data = X.iloc[-1].values.reshape(1, -1)  # Last day's data
future_predictions = []  # Store future predictions

for day in range(7):  # Predict for 7 days
    next_prediction = best_model.predict(last_data)  # Predict next day's price
    future_predictions.append(next_prediction[0])  # Append prediction
    
    last_return = (next_prediction[0] - last_data[0][1]) / last_data[0][1]  # Calculate return
    last_volatility = np.std(np.append(data['Return'].values[-7:], last_return))  # Calculate volatility
    last_data = np.array([[last_return, last_volatility, last_data[0][2], last_data[0][3], last_data[0][4]]]).reshape(1, -1)  # Update features

📝 Explanation:

Iterative Predictions: We generate predictions one day at a time, using the model's previous prediction to update the feature set.



---

📊 Step 8: Visualizing Results

<p align="center">
Finally, we visualize the historical closing prices and predicted prices for the next week.
</p>plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Historical Closing Prices', color='blue')  # Historical prices

future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 8)]  # Future dates
plt.plot(future_dates, future_predictions, label='Predicted Prices', color='orange', marker='o')  # Future predictions

plt.title('Bitcoin Closing Prices and Daily Predictions')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

📝 Explanation:

Blue Line: Historical Bitcoin closing prices.

Orange Line: Predicted prices for the next 7 days.



---

🚀 Conclusion

<p align="center">
In this project, we successfully built a **Random Forest Regressor** model to predict Bitcoin's future prices. We also explored feature engineering and hyperparameter tuning to improve the model's accuracy. However, remember that this is not financial advice; cryptocurrency markets can be highly volatile and unpredictable. Stay safe! 📉📈
</p>
---

