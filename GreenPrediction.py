import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Define function to load data from a specified path
def load_data(data_path):
  """
  Loads data from a CSV file with date parsing and index setting.

  Args:
      data_path: Path to the CSV file.

  Returns:
      A pandas DataFrame with parsed dates and set index.
  """
  data = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
  return data

# Set data path dynamically from the user
data_path = input("Enter the path to your dataset file: ")
 # Update with your data location

# Load the data
data = load_data(data_path)

# Drop unnecessary columns
data = data.drop(columns=['cpu-energy', 'memory-energy'])

# Handle missing values (consider more advanced methods if needed)
data.fillna(method='ffill', inplace=True)

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Convert to DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

# Create the lagged dataset for supervised learning
def create_lagged_dataset(df, n_lags=1):
  X, y = [], []
  for i in range(n_lags, len(df)):
    X.append(df.iloc[i-n_lags:i].values)
    y.append(df.iloc[i].values)
  return np.array(X), np.array(y)

# Define the number of lag steps
n_lags = 10

# Create the dataset
X, y = create_lagged_dataset(scaled_df, n_lags)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=3))  # We have 3 features to predict: cpu, memory, sci-e

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Compile the model with Adam optimizer and mean squared error loss
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with early stopping
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Save the model and scaler

joblib.dump(model, 'time_series_model.joblib')
joblib.dump(scaler, 'scaler.pkl')