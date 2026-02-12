from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model
MODEL_PATH = "../model/factoryguard_final_model.joblib"
model = joblib.load(MODEL_PATH)

THRESHOLD = 0.85   # change if needed

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if data is None:
        return jsonify({"error": "No JSON received"}), 400

    # Convert input to DataFrame
    X = pd.DataFrame([data])

    # Predict probability
    prob = model.predict_proba(X)[0][1]

    return jsonify({
        "failure_probability": float(prob),
        "will_fail_24h": bool(prob >= THRESHOLD)
    })


if __name__ == "__main__":
    app.run(debug=True)
