# Energy Consumption Forecasting Model

A deep learning model using LSTM (Long Short-Term Memory) networks to forecast software energy consumption patterns based on CPU, Memory, and SciE metrics.

## Overview

The model analyzes historical data of software resource usage to predict future energy consumption patterns. It uses a dual-layer LSTM architecture followed by a dense layer for predictions.

## Model Architecture

```python
model = Sequential([
    LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    LSTM(50, return_sequences=False),
    Dense(3)
])
```

## Input Format

Send POST requests to the API endpoint with JSON payload containing:
```json
 {
   "timestamp": "02-08-2024 10:09",
   "cpu": 0.03556466,
   "memory": 1.604238,
   "cpu-energy": 0.00000151,
   "memory-energy": 0.000201236,
   "sci-e": 0.000202745
 },
```

## Output Format

The API returns predictions in JSON format:
```json
{
    "predictions": [
        {
            "cpu": 0.03081174,
            "memory": 1.604238,
            "sci-e": 0.000202744
        }
    ]
}
```

## Getting Started

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask application:
```bash
python app.py
```

3. Make predictions using Postman or any HTTP client by sending POST requests to the endpoint.

## Dependencies

- Flask
- pandas
- numpy
- scikit-learn
- tensorflow
- keras
- joblib

## Notes

- Ensure input data is preprocessed and normalized similarly to the training data
- Timestamp should be in the format "DD-MM-YYYY HH:MM:SS"
- Model has been trained on historical software energy consumption data