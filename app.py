from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import argparse

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('time_series_model.joblib')
scaler = joblib.load('scaler.pkl')

# Function to predict future values
def predict_future(model, data, n_steps, n_lags):
    prediction_list = data[-n_lags:]
    for _ in range(n_steps):
        x = prediction_list[-n_lags:]
        x = x.reshape((1, n_lags, data.shape[1]))
        out = model.predict(x)[0]
        prediction_list = np.append(prediction_list, out.reshape(1, -1), axis=0)
    return prediction_list[n_lags:]

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    new_data = pd.DataFrame(content)
    new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
    new_data.set_index('timestamp', inplace=True)

    # Drop the unnecessary columns
    new_data = new_data.drop(columns=['cpu-energy', 'memory-energy'], errors='ignore')

    # Normalize the new data
    scaled_new_data = scaler.transform(new_data)

    # Access the last timestamp from the index
    last_timestamp = new_data.index[-1]

    # Define the number of lags and steps for future predictions
    n_lags = 5
    # Use the command-line arguments for n_steps and frequency
    n_steps = args.n_steps
    freq = args.freq

    # Predict the next n_steps data points
    future_predictions = predict_future(model, scaled_new_data, n_steps, n_lags)

    # Inverse transform the predictions to original scale
    future_predictions_inv = scaler.inverse_transform(future_predictions)

    # Convert predictions to DataFrame
    future_predictions_df = pd.DataFrame(future_predictions_inv, columns=['cpu', 'memory', 'sci-e'])

    # Generate timestamps for future predictions
    future_timestamps = pd.date_range(start=last_timestamp, periods=n_steps+1, freq=freq)[1:]
    future_predictions_df['timestamp'] = future_timestamps
    future_predictions_df.set_index('timestamp', inplace=True)

    # Convert the DataFrame to JSON
    result = future_predictions_df.reset_index().to_dict(orient='records')

    return jsonify(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series Prediction Server')
    parser.add_argument('--n_steps', type=int, default=300, help='Number of steps to predict')
    parser.add_argument('--freq', type=str, default='S', help='Frequency string for date_range')
    
    # Parse arguments
    args = parser.parse_args()

    app.run(debug=False, host="0.0.0.0", port=5001)
