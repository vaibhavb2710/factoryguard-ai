"""
Retrain the model with current scikit-learn version to fix compatibility
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load training data
train_df = pd.read_csv("data/processed/train_features.csv")

# Prepare features and target
X = train_df.drop(['failure_24h', 'RUL', 'unit', 'time'], axis=1, errors='ignore')
y = train_df['failure_24h']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Drop columns with all NaN values
X = X.dropna(axis=1, how='all')
print(f"Features shape after dropping all-NaN columns: {X.shape}")

# Train model - using Logistic Regression for speed
print("\nTraining Logistic Regression model...")
model = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Fit on entire dataset
model.fit(X, y)

# Save model
output_path = "model/factoryguard_final_model.joblib"
joblib.dump(model, output_path)
print(f"\nModel saved to {output_path}")

# Test prediction
test_row = X.iloc[0:1]
pred_proba = model.predict_proba(test_row)[0][1]
print(f"Test prediction probability: {pred_proba:.4f}")
print("Model retrained successfully!")
