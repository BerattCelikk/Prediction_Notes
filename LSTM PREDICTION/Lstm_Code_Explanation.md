# <p style="text-align:center; font-size: 32px;">📊 Bitcoin Price Prediction using LSTM</p>

```python
import numpy as np  # 🔢 For numerical operations with arrays and matrices
import pandas as pd  # 📅 For data manipulation and analysis
import yfinance as yf  # 📈 To download financial data from Yahoo Finance
import matplotlib.pyplot as plt  # 📉 For data visualization
from sklearn.preprocessing import MinMaxScaler  # 📏 For scaling data
from tensorflow.keras.models import Sequential  # 🏗️ For building a linear stack of layers
from tensorflow.keras.layers import LSTM, Dense, Dropout  # 🧠 For LSTM layers and output
from tensorflow.keras.callbacks import EarlyStopping  # ⏱️ To stop training when performance stops improving
from datetime import datetime, timedelta  # 📅 For handling dates
from sklearn.metrics import mean_squared_error, r2_score  # 📊 For evaluating model performance

<p style="text-align:center; font-size: 28px;">📅 Getting Today's Date</p>

# 🗓️ Getting today's date in YYYY-MM-DD format
today = datetime.now().strftime("%Y-%m-%d")  # Fetches current date

<p style="text-align:center; font-size: 28px;">📥 Downloading Bitcoin Data</p>

# ⬇️ Downloading Bitcoin data from Yahoo Finance from January 1, 2020, to today
data = yf.download("BTC-USD", start="2020-01-01", end=today)  # Downloads historical Bitcoin prices

<p style="text-align:center; font-size: 28px;">📊 Data Preprocessing</p>

# 📉 Keeping only the 'Close' prices from the data
data = data[['Close']]  # Extracts only the 'Close' prices for analysis

# 🔄 Scaling the data to a range of 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))  # Initializes the scaler
scaled_data = scaler.fit_transform(data)  # Scales the data

<p style="text-align:center; font-size: 28px;">🔍 Creating Training and Test Datasets</p>

# 📊 Creating training data
train_size = int(len(scaled_data) * 0.8)  # 80% of data for training
train_data = scaled_data[:train_size]  # Training data subset
x_train, y_train = [], []  # Lists for features and labels

# 🔍 Creating input (x_train) and output (y_train) for the model
for i in range(60, len(train_data)):  # Looping through training data
    x_train.append(train_data[i-60:i, 0])  # Last 60 days as input
    y_train.append(train_data[i, 0])  # Current price as output

# 🛠️ Converting lists to NumPy arrays
x_train, y_train = np.array(x_train), np.array(y_train)  # Conversion for model input

# 🔄 Reshaping input to 3D [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshaping for LSTM

<p style="text-align:center; font-size: 28px;">🏗️ Building the LSTM Model</p>

# 🏗️ Creating the LSTM model
model = Sequential()  # Initializes the Sequential model
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))  # First LSTM layer
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
model.add(LSTM(units=100, return_sequences=True))  # Second LSTM layer
model.add(Dropout(0.2))  # Another Dropout layer
model.add(LSTM(units=100, return_sequences=False))  # Final LSTM layer
model.add(Dropout(0.2))  # Final Dropout layer
model.add(Dense(units=1))  # Output layer for price prediction

# 🛠️ Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')  # Compiles model with Adam optimizer

<p style="text-align:center; font-size: 28px;">⏱️ Early Stopping and Model Training</p>

# ⏱️ EarlyStopping callback
early_stopping = EarlyStopping(monitor='loss', patience=5)  # Stops training if no improvement

# 🚀 Training the model
model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping])  # Trains the model

<p style="text-align:center; font-size: 28px;">📊 Preparing Test Data</p>

# 📊 Preparing test data
test_data = scaled_data[train_size - 60:]  # Last 60 days for testing
x_test, y_test = [], data['Close'][train_size:].values  # Input and actual output

# 🔍 Creating input for the test set
for i in range(60, len(test_data)):  # Looping through test data
    x_test.append(test_data[i-60:i, 0])  # Last 60 days as input

# 🛠️ Converting test input to NumPy array
x_test = np.array(x_test)  # Converts list to NumPy array
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshapes for LSTM

<p style="text-align:center; font-size: 28px;">📈 Making Predictions</p>

# 📈 Making predictions with the model
predictions = model.predict(x_test)  # Model predictions
predictions = scaler.inverse_transform(predictions)  # Inverses scaling for actual prices

<p style="text-align:center; font-size: 28px;">🎨 Visualizing Results</p>

# 🎨 Visualizing the results
plt.figure(figsize=(14, 5))  # Figure size for the plot
plt.plot(data.index[train_size:], y_test, color='blue', label='Actual Prices')  # Actual prices in blue
plt.plot(data.index[train_size:], predictions, color='red', label='Predicted Prices')  # Predicted prices in red
plt.title('📈 Bitcoin Price Prediction')  # Title of the plot
plt.xlabel('Date')  # X-axis label
plt.ylabel('Price (USD)')  # Y-axis label
plt.legend()  # Displays legend
plt.show()  # Shows the plot

<p style="text-align:center; font-size: 28px;">📏 Evaluating Model Performance</p>

# 📏 Evaluating Model Performance
mse = mean_squared_error(y_test, predictions)  # Calculate Mean Squared Error
r2 = r2_score(y_test, predictions)  # Calculate R² score

print(f'Mean Squared Error (MSE): {mse:.2f}')  # Display MSE
print(f'R² Score: {r2:.2f}')  # Display R² score

<p style="text-align:center; font-size: 28px;">📌 Notes</p>

Ensure you have all necessary libraries installed in your Python environment before running this code.

Adjust model parameters (like LSTM units, dropout rates, and training epochs) as needed to enhance prediction accuracy based on your specific data and use case.

Consider experimenting with different architectures, such as adding more layers or using bidirectional LSTM, to improve performance further.

You can also try different scaling methods, such as StandardScaler, or fine-tune the learning rate of the optimizer for better results.

### Additional Changes
- The section titles are now larger and centered for better visibility.
- Added headings for each code block for better organization.

