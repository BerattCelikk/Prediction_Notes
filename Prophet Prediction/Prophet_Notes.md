# 📈 Time Series Forecasting with Prophet

Prophet is a time series forecasting tool developed by Facebook, widely used in business and finance. It allows users to analyze their data quickly and effectively. This document contains essential tips, details, and best practices to consider when using Prophet.

## 📊 What is Prophet?
- **Definition**: Prophet is an open-source library designed specifically for working with daily data. Its goal is to model seasonal trends and holiday effects in time series data and predict future values.

- **Use Cases**:
  - **Finance**: Forecasting financial data such as stock prices and exchange rates.
  - **Retail**: Sales forecasting and inventory management.
  - **Energy**: Consumption forecasting and demand management.
  - **Health**: Predicting patient numbers and disease spread.

- **Features**:
  - **Seasonality**: Automatically detects annual, weekly, or daily seasonalities.
  - **Holiday Effects**: Models the effects of special days.
  - **User-Friendly**: Provides a simple API for users to easily make predictions.

## 🚀 Getting Started
### 1. Installation
To install the Prophet library, run the following command in your terminal:
```bash
pip install prophet

2. Data Preparation

Organize your data into two main columns:

ds: Date (in datetime format)

y: Values to be predicted (numeric)


Example dataset creation:

import pandas as pd

# Example dataset
data = {
    'ds': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'y': [100, 150, 200, 250, 300]
}
df = pd.DataFrame(data)
df['ds'] = pd.to_datetime(df['ds'])  # Convert dates to datetime format

🔍 Model Creation

Steps to create a Prophet model and fit it to the data:


from prophet import Prophet

model = Prophet()
model.fit(df)  # Fit the model to the data

📅 Forecasting for the Future

Create future dates and make predictions:


future = model.make_future_dataframe(periods=365)  # Create dates for the next 365 days
forecast = model.predict(future)  # Make predictions

📈 Visualizing Forecast Results

To visualize the forecast results:


import matplotlib.pyplot as plt

fig = model.plot(forecast)
plt.title('Prophet Forecast Results')
plt.xlabel('Date')
plt.ylabel('Predicted Value')
plt.show()

🌟 Tips and Tricks

Adjust Seasonality

Use the add_seasonality function to examine seasonal effects. For example, you can add annual and weekly seasonalities:


model.add_seasonality(name='weekly', period=7, fourier_order=3)
model.add_seasonality(name='yearly', period=365.25, fourier_order=10)

Include Holidays

Use the add_country_holidays function to include special holidays in the model. For example:


from prophet import Prophet

holiday_df = pd.DataFrame({
    'holiday': 'my_holiday',
    'ds': pd.to_datetime(['2023-12-25', '2023-01-01']),
    'lower_window': 0,
    'upper_window': 1,
})
model = Prophet(holidays=holiday_df)

Incorporate Influencing Factors

Use the add_regressor function to include other variables in the model. Example:


df['extra_regressor'] = [value1, value2, value3]  # Extra variables
model.add_regressor('extra_regressor')

Evaluate Model Performance

Apply cross-validation to assess the model's performance, helping you test its generalizability:


from prophet.diagnostics import cross_validation

df_cv = cross_validation(model, initial='365 days', period='180 days', horizon='30 days')

Improve the Model

Tuning hyperparameters can enhance model performance. For instance, try adjusting changepoint_prior_scale and seasonality_prior_scale:


model = Prophet(changepoint_prior_scale=0.5, seasonality_prior_scale=10)

📊 Evaluating Results

Measuring Model Success

To evaluate prediction accuracy, you can use the following metrics:

MAE (Mean Absolute Error): The average absolute error of the predictions.

RMSE (Root Mean Squared Error): The square root of the mean of the squared errors.

MSE (Mean Squared Error): The average squared error.


Example calculation:

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

y_true = df['y']
y_pred = forecast['yhat'][:len(y_true)]  # Predictions for past values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"MAE: {mae}, RMSE: {rmse}")

🎓 Conclusion

Prophet is a powerful and user-friendly tool for time series forecasting. With proper data preparation and model tuning, you can make highly accurate predictions and optimize your business processes. For more information about Prophet, visit the official documentation.

🔗 Additional Resources

Facebook Prophet GitHub Page

Time Series Analysis and Forecasting

Machine Learning for Time Series Forecasting



---

> Note: This content should not be considered financial advice. Time series forecasting is a complex process and requires careful analysis.




