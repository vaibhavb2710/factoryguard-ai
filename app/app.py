from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import warnings

app = Flask(__name__)

# Load trained model with compatibility handling
MODEL_PATH = "../model/factoryguard_final_model.joblib"

def load_model():
    """Load model with scikit-learn version compatibility handling"""
    try:
        # Suppress DeprecationWarnings during loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

model = load_model()

THRESHOLD = 0.85   # change if needed

# Get all feature names that the model expects
def get_expected_features():
    """Get the feature names the model was trained on"""
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    return []

def compute_features(sensor_data):
    """
    Compute all required features from basic sensor data.
    sensor_data should contain: op_setting_1, op_setting_2, op_setting_3, sensor_1-21
    """
    # Create a DataFrame with the input
    df = pd.DataFrame([sensor_data])
    
    # Identify sensor columns
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
    
    # Compute rolling means for windows [1, 6, 12]
    for window in [1, 6, 12]:
        for sensor in sensor_cols:
            if sensor in df.columns:
                df[f"{sensor}_roll_mean_{window}"] = df[sensor]
    
    # Compute rolling std for windows [1, 6, 12]
    for window in [1, 6, 12]:
        for sensor in sensor_cols:
            if sensor in df.columns:
                df[f"{sensor}_roll_std_{window}"] = 0.0  # For single row, std is 0
    
    # Compute lags [1, 2]
    for lag in [1, 2]:
        for sensor in sensor_cols:
            if sensor in df.columns:
                df[f"{sensor}_lag_{lag}"] = np.nan  # For single row, lag is NaN
    
    return df

@app.route("/", methods=["GET"])
def home():
    return {"message": "FactoryGuard-AI Prediction API"}

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if data is None:
        return jsonify({"error": "No JSON received"}), 400

    try:
        # Compute all required features
        X = compute_features(data)
        
        # Reorder columns to match model's expected feature order
        expected_features = get_expected_features()
        if expected_features:
            # Fill missing columns with 0
            for col in expected_features:
                if col not in X.columns:
                    X[col] = 0.0
            # Reorder to match model's expectation
            X = X[expected_features]
        
        # Predict probability
        prob = model.predict_proba(X)[0][1]

        return jsonify({
            "failure_probability": float(prob),
            "will_fail_24h": bool(prob >= THRESHOLD)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
