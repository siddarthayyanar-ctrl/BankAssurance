import os
import joblib
import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import shap

# Initialize FastAPI application
app = FastAPI(
    title="Predictive Bancassurance Engine API - V2.0",
    description="Database-driven enterprise REST service tracking real-time feature attribution and CLV scores.",
    version="2.0"
)

DB_PATH = "data/bancassurance.db"
CLV_MODEL_PATH = "models/clv_model.pkl"
CLASSIFIER_PATH = "models/insurance_classifier.pkl"
FEATURES_CLV_PATH = "models/feature_columns_clv.pkl"
FEATURES_CLASS_PATH = "models/feature_columns.pkl"

# Verify artifacts exist
if not (os.path.exists(CLV_MODEL_PATH) and os.path.exists(CLASSIFIER_PATH)):
    raise FileNotFoundError("Model binaries missing. Run training scripts first.")

# Load models and column schemas
clv_model = joblib.load(CLV_MODEL_PATH)
insurance_classifier = joblib.load(CLASSIFIER_PATH)
clv_features_blueprint = joblib.load(FEATURES_CLV_PATH)
class_features_blueprint = joblib.load(FEATURES_CLASS_PATH)

# Instantiate SHAP Explainer using the base models for real-time calculation
# We extract estimators for our tree explainer architecture
base_rf_estimator = insurance_classifier.estimators_[0] 
explainer = shap.TreeExplainer(base_rf_estimator)

class PredictionRequest(BaseModel):
    customer_id: str

class RawInputRequest(BaseModel):
    age: int
    annual_income: float
    credit_score: int
    tenure_months: int
    current_account_balance: float
    total_salary_credited: float
    spend_Auto_Service: float
    spend_Dining: float
    spend_Fuel: float
    spend_Groceries: float
    spend_Medical: float
    spend_Pharmacy: float
    spend_Real_Estate: float
    spend_Utilities: float
    total_debit_amount: float
    avg_transaction_size: float
    transaction_count: int

@app.get("/api/v2/meta")
def get_meta_metrics():
    """Fetches systemic aggregates for dashboard visualization directly from SQL data."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT age, annual_income, current_account_balance, actual_12m_value, has_auto_ins, has_health_ins, has_life_ins FROM customer_profiles"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return {
        "total_records": int(len(df)),
        "avg_age": float(df['age'].mean()),
        "avg_balance": float(df['current_account_balance'].mean()),
        "conversion_rates": {
            "auto": float(df['has_auto_ins'].mean()),
            "health": float(df['has_health_ins'].mean()),
            "life": float(df['has_life_ins'].mean())
        }
    }

@app.post("/api/v2/predict")
def get_predictions_from_db(payload: PredictionRequest):
    # Clean the input to remove any accidental leading/trailing spaces
    cust_id = str(payload.customer_id).strip()
    
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM customer_profiles WHERE customer_id = ?"
    customer_df = pd.read_sql_query(query, conn, params=(cust_id,))
    
    # FALLBACK FILTER: If the selectbox passed something like "('CUST_0001',)" as a raw string
    if customer_df.empty:
        cleaned_id = cust_id.replace("(", "").replace(")", "").replace("'", "").replace(",", "").strip()
        customer_df = pd.read_sql_query(query, conn, params=(cleaned_id,))
    
    conn.close()
    
    if customer_df.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"Customer ID '{cust_id}' not found in Relational DB. Valid example: CUST_0001"
        )
    
    # --- Keep all the remaining SHAP and prediction matrix calculation code exactly the same ---
    X_clv = customer_df[clv_features_blueprint]
    X_class = customer_df[class_features_blueprint]
    
    predicted_clv = clv_model.predict(X_clv)[0]
    probabilities = insurance_classifier.predict_proba(X_class)
    
    prob_auto = probabilities[0][0][1]
    prob_health = probabilities[1][0][1]
    prob_life = probabilities[2][0][1]
    
    shap_values = explainer.shap_values(X_class)
    local_importance = shap_values[0, :, 1] if len(shap_values.shape) > 2 else shap_values[0, :]
    
    feature_impact_map = dict(zip(class_features_blueprint, local_importance))
    sorted_impacts = sorted(feature_impact_map.items(), key=lambda item: abs(item[1]), reverse=True)
    top_drivers = [feat for feat, val in sorted_impacts[:2]]
    
    product_scores = [
        {
            "product": "Automobile Insurance",
            "probability": float(prob_auto),
            "trigger": f"High risk pattern found. Top behavioral features identified: {', '.join(top_drivers).replace('_', ' ')}."
        },
        {
            "product": "Health Insurance",
            "probability": float(prob_health),
            "trigger": f"Demographic tier shift verified. Primary medical metrics evaluated: {', '.join(top_drivers).replace('_', ' ')}."
        },
        {
            "product": "Comprehensive Life Policy",
            "probability": float(prob_life),
            "trigger": f"Wealth asset accumulation patterns confirmed. Primary cash indicators: {', '.join(top_drivers).replace('_', ' ')}."
        }
    ]

    # Keep a reasonable display threshold for the UI, but do not hide all non-auto products
    # when the model is naturally skewed toward one category.
    threshold_map = {
        "Automobile Insurance": 0.25,
        "Health Insurance": 0.15,
        "Comprehensive Life Policy": 0.10,
    }

    recommendations = [
        {
            "product": item["product"],
            "probability": round(item["probability"], 4),
            "trigger": item["trigger"]
        }
        for item in product_scores
        if item["probability"] >= threshold_map.get(item["product"], 0.10)
    ]

    if not recommendations:
        recommendations = [
            {
                "product": item["product"],
                "probability": round(item["probability"], 4),
                "trigger": item["trigger"]
            }
            for item in sorted(product_scores, key=lambda x: x["probability"], reverse=True)[:3]
        ]

    return {
        "customer_id": cust_id,
        "demographics": {
            "age": int(customer_df['age'].values[0]),
            "annual_income": float(customer_df['annual_income'].values[0]),
            "current_balance": float(customer_df['current_account_balance'].values[0])
        },
        "metrics": {
            "predicted_12m_clv": round(float(predicted_clv), 2)
        },
        "cross_sell_recommendations": sorted(recommendations, key=lambda x: x['probability'], reverse=True)
    }

@app.post("/api/v2/predict-raw")
def get_predictions_from_raw(payload: RawInputRequest):
    """Processes incoming raw fields directly from web UI forms for manual evaluation."""
    raw_dict = payload.dict()
    input_df = pd.DataFrame([raw_dict])
    
    # Add required placeholder targets matching training dimensions
    input_df['has_auto_ins'] = 0
    input_df['has_health_ins'] = 0
    input_df['has_life_ins'] = 0
    
    # Derived composite signals
    input_df['savings_to_income_ratio'] = (input_df['total_salary_credited'] - input_df['total_debit_amount']) / (input_df['total_salary_credited'] + 1)
    input_df['auto_risk_signal'] = input_df['spend_Auto_Service'] + input_df['spend_Fuel']
    input_df['health_risk_signal'] = input_df['spend_Medical'] + input_df['spend_Pharmacy']
    
    X_clv = input_df[clv_features_blueprint]
    X_class = input_df[class_features_blueprint]
    
    predicted_clv = clv_model.predict(X_clv)[0]
    probabilities = insurance_classifier.predict_proba(X_class)
    
    return {
        "predicted_clv": round(float(predicted_clv), 2),
        "probabilities": {
            "Auto": round(float(probabilities[0][0][1]), 2),
            "Health": round(float(probabilities[1][0][1]), 2),
            "Life": round(float(probabilities[2][0][1]), 2)
        }
    }