import pandas as pd
import numpy as np
#import jobio
import os
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("Loading data for CLV Regression model...")
df = pd.read_csv("data/processed_features.csv")

# Define target and features
target_col = 'actual_12m_value'
exclude_cols = ['customer_id', 'has_auto_ins', 'has_health_ins', 'has_life_ins', target_col]
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df[target_col]

# Split into train and evaluation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training LightGBM Regressor on {X_train.shape[0]} samples with {X_train.shape[1]} features...")

# Initialize and train LightGBM
clv_model = LGBMRegressor(
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1
)
clv_model.fit(X_train, y_train)

# Evaluate performance
y_pred = clv_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- CLV Model Evaluation Metrics ---")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"R² Variance Score: {r2:.4f} (An R² near 1.0 means high predictive accuracy)")

# Save artifacts for production deployment
os.makedirs("models", exist_ok=True)
import joblib
joblib.dump(clv_model, "models/clv_model.pkl")
joblib.dump(feature_cols, "models/feature_columns.pkl")
print("\nCLV model and feature blueprint saved to 'models/' directory.")