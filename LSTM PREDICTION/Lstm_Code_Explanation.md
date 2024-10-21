📊 Bitcoin Price Prediction using LSTM

import numpy as np  # 🔢 For numerical operations with arrays and matrices
import pandas as pd  # 📅 For data manipulation and analysis
import yfinance as yf  # 📈 To download financial data from Yahoo Finance
import matplotlib.pyplot as plt  # 📉 For data visualization
from sklearn.preprocessing import MinMaxScaler  # 📏 For scaling data
from tensorflow.keras.models import Sequential  # 🏗️ For building a linear stack of layers
from tensorflow.keras.layers import LSTM, Dense, Dropout  # 🧠 For LSTM layers and output
from tensorflow.keras.callbacks import EarlyStopping  # ⏱️ To stop training when performance stops improving
from sklearn.metrics import mean_squared_error, r2_score  # 📊 For evaluating model performance
from datetime import datetime  # 📅 For handling dates

# 📅 Getting Today's Date
def get_today_date():
    """Fetch today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")  # Fetches current date

# 📥 Downloading Bitcoin Data
def download_data(ticker, start_date, end_date):
    """Download historical data for the given ticker from Yahoo Finance."""
    return yf.download(ticker, start=start_date, end=end_date)  # Downloads historical Bitcoin prices

# 📊 Data Preprocessing
def preprocess_data(data):
    """Preprocess the data by scaling and creating training and testing sets."""
    data = data[['Close']]  # Extracts only the 'Close' prices for analysis
    scaler = MinMaxScaler(feature_range=(0, 1))  # Initializes the scaler
    scaled_data = scaler.fit_transform(data)  # Scales the data

    # 📊 Creating training and test data
    train_size = int(len(scaled_data) * 0.8)  # 80% of data for training
    train_data = scaled_data[:train_size]  # Training data subset
    x_train, y_train = [], []  # Lists for features and labels

    # 🔍 Creating input (x_train) and output (y_train) for the model
    for i in range(60, len(train_data)):  # Looping through training data
        x_train.append(train_data[i-60:i, 0])  # Last 60 days as input
        y_train.append(train_data[i, 0])  # Current price as output

    # 🛠️ Converting lists to NumPy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)  # Conversion for model input
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshaping for LSTM

    return x_train, y_train, scaled_data, train_size, data['Close'].values  # Returning necessary components

# 🏗️ Building the LSTM Model
def build_model(input_shape):
    """Create and compile the LSTM model."""
    model = Sequential()  # Initializes the Sequential model
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))  # First LSTM layer
    model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
    model.add(LSTM(units=100, return_sequences=True))  # Second LSTM layer
    model.add(Dropout(0.2))  # Another Dropout layer
    model.add(LSTM(units=100, return_sequences=False))  # Final LSTM layer
    model.add(Dropout(0.2))  # Final Dropout layer
    model.add(Dense(units=1))  # Output layer for price prediction

    model.compile(optimizer='adam', loss='mean_squared_error')  # Compiles model with Adam optimizer
    return model  # Returns the compiled model

# ⏱️ Early Stopping and Model Training
def train_model(model, x_train, y_train):
    """Train the LSTM model with early stopping."""
    early_stopping = EarlyStopping(monitor='loss', patience=5)  # Stops training if no improvement
    model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping])  # Trains the model

# 📈 Making Predictions
def make_predictions(model, scaled_data, train_size):
    """Generate predictions using the trained model."""
    test_data = scaled_data[train_size - 60:]  # Last 60 days for testing
    x_test, y_test = [], data['Close'][train_size:].values  # Input and actual output

    # 🔍 Creating input for the test set
    for i in range(60, len(test_data)):  # Looping through test data
        x_test.append(test_data[i-60:i, 0])  # Last 60 days as input

    # 🛠️ Converting test input to NumPy array
    x_test = np.array(x_test)  # Converts list to NumPy array
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshapes for LSTM

    # 📈 Making predictions with the model
    predictions = model.predict(x_test)  # Model predictions
    return predictions, y_test  # Returns predictions and actual prices

# 🎨 Visualizing Results
def visualize_results(data, y_test, predictions, train_size):
    """Visualize the actual and predicted prices."""
    plt.figure(figsize=(14, 5))  # Figure size for the plot
    plt.plot(data.index[train_size:], y_test, color='blue', label='Actual Prices')  # Actual prices in blue
    plt.plot(data.index[train_size:], predictions, color='red', label='Predicted Prices')  # Predicted prices in red
    plt.title('📈 Bitcoin Price Prediction')  # Title of the plot
    plt.xlabel('Date')  # X-axis label
    plt.ylabel('Price (USD)')  # Y-axis label
    plt.legend()  # Displays legend
    plt.show()  # Shows the plot

# 📏 Evaluating Model Performance
def evaluate_model(y_test, predictions):
    """Evaluate the model's performance using MSE and R² score."""
    mse = mean_squared_error(y_test, predictions)  # Calculate Mean Squared Error
    r2 = r2_score(y_test, predictions)  # Calculate R² score

    print(f'Mean Squared Error (MSE): {mse:.2f}')  # Display MSE
    print(f'R² Score: {r2:.2f}')  # Display R² score

# 🚀 Main Execution Flow
if __name__ == "__main__":
    today = get_today_date()  # Fetch today's date
    data = download_data("BTC-USD", "2020-01-01", today)  # Download Bitcoin data
    x_train, y_train, scaled_data, train_size, actual_prices = preprocess_data(data)  # Preprocess data

    model = build_model((x_train.shape[1], 1))  # Build model
    train_model(model, x_train, y_train)  # Train model

    predictions, y_test = make_predictions(model, scaled_data, train_size)  # Make predictions
    predictions = scaler.inverse_transform(predictions)  # Inverses scaling for actual prices

    visualize_results(data, y_test, predictions, train_size)  # Visualize results
    evaluate_model(y_test, predictions)  # Evaluate model performance

🚀 Enhancements and Features

1. Modular Functions: The code is now organized into functions, making it easier to read, test, and reuse components.


2. Hyperparameter Tuning: Consider incorporating a hyperparameter tuning strategy (like GridSearchCV) to find the best model parameters.


3. Model Evaluation: Added evaluation metrics to assess the model's performance, such as Mean Squared Error (MSE) and R² score.


4. Visualization: Enhanced the plotting function for better visual comparison between actual and predicted prices.


5. Scalability: The code can be easily expanded to include more features, such as additional indicators or a more complex model architecture (e.g., GRU or Bidirectional LSTM).


6. Error Handling: Consider implementing error handling to manage issues such as data download failures or model training issues.


7. Future Predictions: To extend this further, implement a forecasting capability that can predict future prices beyond the last date in your dataset.


8. Save and Load Model: Add functionality to save and load your trained model, so you can use it later without retraining.



📌 Notes:

Ensure you have all necessary libraries installed in your Python environment before running this code.

Adjust model parameters (like LSTM units, dropout rates, and training epochs) based on your specific data and use case for enhanced prediction accuracy.

Experiment with different architectures and tuning parameters for improved performance.


