import pandas as pd
import numpy as np
import joblib
import os
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("Loading data for CLV Regression model...")
df = pd.read_csv("data/processed_features.csv")

# Define target
target_col = 'actual_12m_value'

# FIX: We allow the regressor to see the insurance ownership flags 
# because a bank knows its customers' existing policies when calculating value!
exclude_cols = ['customer_id', target_col]
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df[target_col]

# Split into train and evaluation sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Regressor on {X_train.shape[0]} samples with current policy features...")

# Initialize an optimized regressor
clv_model = LGBMRegressor(
    n_estimators=200,
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

print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"New R² Variance Score: {r2:.4f}")

# Save artifacts
os.makedirs("models", exist_ok=True)
joblib.dump(clv_model, "models/clv_model.pkl")
joblib.dump(feature_cols, "models/feature_columns_clv.pkl")
print("\nModel binaries updated in 'models/' directory.")