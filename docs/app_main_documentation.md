# API Application Documentation - v2.0

## Overview

The file [app/main.py](../app/main.py) defines a FastAPI web service for serving Customer Lifetime Value (CLV) predictions and insurance cross-sell recommendations. It integrates with a SQLite relational database, loads pretrained model artifacts, and exposes REST endpoints for real-time customer scoring with SHAP-based feature attribution.

This production-grade API layer transforms customer features into actionable intelligence for relationship managers and downstream ML systems.

---

## Purpose

The application performs three main tasks:

1. **Model Inference**: Loads and executes pretrained CLV regression and insurance classification models.
2. **Database Integration**: Queries customer profiles from SQLite relational layer and performs real-time predictions.
3. **Explainability**: Computes SHAP values for local feature importance, generating interpretable recommendation triggers tied to specific behavioral drivers.

---

## Functional Workflow

### 1. Startup and Model Loading

When the service starts, it:

- validates that model artifact files exist in `models/` directory
- loads the CLV regression model from `models/clv_model.pkl`
- loads the insurance classifier from `models/insurance_classifier.pkl`
- loads feature blueprints from `models/feature_columns_clv.pkl` and `models/feature_columns.pkl`
- initializes a SHAP TreeExplainer using the base random forest estimator from the classifier

If the required model binaries are missing, the application raises a `FileNotFoundError` at startup.

### 2. Data Structures

The API defines two Pydantic request models:

- `PredictionRequest`: Single-field request for database lookup
  - `customer_id: str`

- `RawInputRequest`: 17-field request for manual feature entry
  - `age`, `annual_income`, `credit_score`, `tenure_months`
  - `current_account_balance`, `total_salary_credited`, `total_debit_amount`
  - Spending categories: `spend_Auto_Service`, `spend_Dining`, `spend_Fuel`, `spend_Groceries`, `spend_Medical`, `spend_Pharmacy`, `spend_Real_Estate`, `spend_Utilities`
  - `avg_transaction_size`, `transaction_count`

### 3. Endpoints

#### `GET /api/v2/meta`

Aggregates systemic metrics across the entire customer base for dashboard visualization and data drift monitoring.

**Response**:
- `total_records`: Number of customer profiles in database
- `avg_age`: Mean customer age
- `avg_balance`: Mean account balance
- `conversion_rates`: Dict with `auto`, `health`, `life` insurance penetration rates

#### `POST /api/v2/predict`

**Request**:
```json
{
  "customer_id": "CUST001"
}
```

**Response**:
- `customer_id`: Requested identifier
- `demographics`: Age, annual income, current account balance from database
- `metrics`: Predicted 12-month CLV score
- `cross_sell_recommendations`: Array of products with propensity scores and SHAP-driven behavioral triggers

Returns HTTP 404 if customer not found in SQLite database.

#### `POST /api/v2/predict-raw`

Accepts raw feature inputs for prospects or ad-hoc scoring scenarios outside the primary database.

**Request**: 17-field form containing customer behaviors (age, spending patterns, transaction metrics, etc.)

**Response**:
- `predicted_clv`: Continuous valuation estimate
- `probabilities`: Dict with Auto, Health, Life insurance propensity scores (0-1 range)

---

### 4. Prediction Logic

For database predictions (`/predict`):
1. Query customer profile from SQLite by `customer_id`
2. Extract CLV feature set, predict continuous value
3. Extract classifier feature set, compute insurance probabilities
4. Calculate SHAP values for local feature attribution
5. Filter recommendations where probability > 0.4
6. Map top-2 SHAP drivers to human-readable recommendation triggers
7. Return sorted by descending probability

For raw predictions (`/predict-raw`):
1. Compute derived signals (savings ratio, auto/health risk indicators)
2. Apply same CLV and classifier models
3. Return numeric predictions without SHAP explainability

---

## Dependencies

The service depends on:

**Artifacts**:
- `models/clv_model.pkl` — Regression model
- `models/insurance_classifier.pkl` — Multi-output classifier (3 binary classifiers)
- `models/feature_columns_clv.pkl` — Feature names for CLV
- `models/feature_columns.pkl` — Feature names for classifier

**Data**:
- `data/bancassurance.db` — SQLite database with `customer_profiles` table

**Python Libraries**:
- `fastapi` — Web framework
- `pydantic` — Request validation
- `joblib` — Model serialization
- `pandas` — Data manipulation
- `sqlite3` — Database queries
- `shap` — Feature attribution (TreeExplainer)
- `numpy` — Numeric operations

---

## How to Run

From the repository root:

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`.

---

## Key Design Decisions

- **SQLite Backend**: Relational storage avoids in-memory CSV bloat at startup.
- **SHAP Explainability**: TreeExplainer provides real-time feature attribution without LIME approximation overhead.
- **Dual Input Modes**: Database mode for known customers; raw mode for new prospects.
- **Probability Threshold**: Fixed at 0.4 for recommendation filtering (tunable if needed).
- **Multi-Output Classifier**: Three independent binary classifiers (Auto, Health, Life) allow product-specific signals.

---

## Troubleshooting

**404 Error**: Customer ID not found in database. Verify `customer_id` exists in `customer_profiles` table.

**Connection Error**: FastAPI backend not running or wrong port. Check `uvicorn` is active on port 8000.

**Missing Models**: FileNotFoundError at startup means model `.pkl` files missing. Run training scripts first.

---

## Summary

`app/main.py` provides a REST API that serves CLV predictions and insurance cross-sell guidance for individual customers. It loads pretrained models and preprocessed customer features, validates requests, and returns structured scoring output suitable for dashboarding, CRM integration, or relationship manager decision support.

---

## Streamlit UI (app/ui.py)

A lightweight Streamlit dashboard is provided in `app/ui.py` to visualize predictions and recommendations delivered by the FastAPI service. Key notes:

- **Purpose:** Query the `POST /api/v1/predict` endpoint and present CLV, demographics, and cross-sell recommendations in a relationship-manager-friendly layout.
- **Data Source:** The sidebar customer selector reads `data/processed_features.csv` to populate available `customer_id` values.
- **Run:** Start the FastAPI backend first, then from the repository root run:

```bash
uvicorn app.main:app --reload
streamlit run app/ui.py
```

- **API:** The UI expects the API at `http://127.0.0.1:8000/api/v1/predict`. Adjust `API_URL` in `app/ui.py` if your backend runs elsewhere.

- **Response Contract:** See the "Expected API response (JSON)" section at the top of this file — `app/ui.py` formats the dashboard around those keys.
