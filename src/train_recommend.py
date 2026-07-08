import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("Loading data for Insurance Propensity Classification...")
df = pd.read_csv("data/processed_features.csv")

# Define targets and features
target_cols = ['has_auto_ins', 'has_health_ins', 'has_life_ins']
exclude_cols = ['customer_id', 'actual_12m_value'] + target_cols
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df[target_cols]

# Split into train and evaluation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Multi-Output Random Forest Classifier...")

# Initialize multi-label framework wrapping a Random Forest
forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
classifier = MultiOutputClassifier(forest, n_jobs=-1)
classifier.fit(X_train, y_train)

# Predict and assess precision/recall matrix
y_pred = classifier.predict(X_test)

print("\n--- Insurance Propensity Classification Report ---")
for i, col in enumerate(target_cols):
    print(f"\nTarget Insurance Product: {col.upper()}")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i], target_names=['No Purchase', 'Likely Purchase']))

# Save model artifact
joblib.dump(classifier, "models/insurance_classifier.pkl")
print("Insurance cross-sell prediction engine saved to 'models/' directory.")