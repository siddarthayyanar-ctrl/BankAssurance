import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI application
app = FastAPI(
    title="Predictive Bancassurance Engine API",
    description="REST API serving Customer Lifetime Value and Insurance Cross-Sell predictions.",
    version="1.0"
)

# Global variables to hold our model artifacts
CLV_MODEL_PATH = "models/clv_model.pkl"
CLASSIFIER_PATH = "models/insurance_classifier.pkl"
FEATURES_CLV_PATH = "models/feature_columns_clv.pkl"
FEATURES_CLASS_PATH = "models/feature_columns.pkl"
DATA_PATH = "data/processed_features.csv"

# Load models and data at startup
if not (os.path.exists(CLV_MODEL_PATH) and os.path.exists(CLASSIFIER_PATH)):
    raise FileNotFoundError("Model binaries missing. Please run training scripts first.")

clv_model = joblib.load(CLV_MODEL_PATH)
insurance_classifier = joblib.load(CLASSIFIER_PATH)
clv_features_blueprint = joblib.load(FEATURES_CLV_PATH)
class_features_blueprint = joblib.load(FEATURES_CLASS_PATH)
features_df = pd.read_csv(DATA_PATH)

# Define request/response data structures using Pydantic
class PredictionRequest(BaseModel):
    customer_id: str

@app.get("/")
def read_root():
    return {"status": "online", "engine": "Bancassurance ML API v1.0"}

@app.post("/api/v1/predict")
def get_predictions(payload: PredictionRequest):
    cust_id = payload.customer_id
    
    # Extract the customer row from our preprocessed feature set
    customer_row = features_df[features_df['customer_id'] == cust_id]
    
    if customer_row.empty:
        raise HTTPException(status_code=404, detail=f"Customer ID {cust_id} not found in database.")
    
    # 1. Format inputs for the CLV Regressor
    X_clv = customer_row[clv_features_blueprint]
    predicted_clv = clv_model.predict(X_clv)[0]
    
    # 2. Format inputs for the Cross-Sell Classifier
    X_class = customer_row[class_features_blueprint]
    probabilities = insurance_classifier.predict_proba(X_class)
    
    # Extract the probability of "Class 1" (Likely Purchase) for each target product
    # probabilities is a list of arrays: [auto_probs, health_probs, life_probs]
    prob_auto = probabilities[0][0][1]
    prob_health = probabilities[1][0][1]
    prob_life = probabilities[2][0][1]
    
    # 3. Generate natural text "triggers" for the Relationship Manager
    recommendations = []
    if prob_auto > 0.4:
        recommendations.append({
            "product": "Automobile Insurance",
            "probability": round(float(prob_auto), 4),
            "trigger": "Elevated automotive spending or repair patterns identified over the last 90 days."
        })
    if prob_health > 0.4:
        recommendations.append({
            "product": "Health Insurance",
            "probability": round(float(prob_health), 4),
            "trigger": "Spike in pharmacy or medical merchant indicators, coupled with age demographic tier."
        })
    if prob_life > 0.4:
        recommendations.append({
            "product": "Comprehensive Life Policy",
            "probability": round(float(prob_life), 4),
            "trigger": "Substantial regular income tier with strong cash surplus / net balance retention."
        })
        
    # Sort recommendations by highest probability first
    recommendations = sorted(recommendations, key=lambda x: x['probability'], reverse=True)

    return {
        "customer_id": cust_id,
        "demographics": {
            "age": int(customer_row['age'].values[0]),
            "annual_income": float(customer_row['annual_income'].values[0]),
            "current_balance": float(customer_row['current_account_balance'].values[0])
        },
        "metrics": {
            "predicted_12m_clv": round(float(predicted_clv), 2)
        },
        "cross_sell_recommendations": recommendations
    }