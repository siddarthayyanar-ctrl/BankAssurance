# API Application Documentation

## Overview

The file [app/main.py](../app/main.py) defines a FastAPI web service for serving CLV predictions and insurance cross-sell recommendations. It loads pretrained model artifacts and exposes a single prediction endpoint for customer scoring.

This file is intended to provide a lightweight API layer on top of the existing ML models, turning tabular customer features into actionable outputs for relationship managers or downstream systems.

---

## Purpose

The application performs three main tasks:

1. Loads saved model artifacts and preprocessed customer features.
2. Exposes HTTP endpoints for health checks and prediction requests.
3. Formats model output into a structured response including CLV, cross-sell probabilities, and recommendation triggers.

---

## Functional Workflow

### 1. Startup and Model Loading

When the service starts, it:

- validates that model artifact files exist
- loads the CLV regression model from `models/clv_model.pkl`
- loads the insurance classifier from `models/insurance_classifier.pkl`
- loads feature blueprints from `models/feature_columns_clv.pkl` and `models/feature_columns.pkl`
- reads the preprocessed feature dataset from `data/processed_features.csv`

If the required model binaries are missing, the application raises a `FileNotFoundError` at startup.

### 2. Data Structures

The API defines a Pydantic request model:

- `PredictionRequest` with a single field:
  - `customer_id: str`

This ensures that incoming requests are validated before prediction logic executes.

### 3. Endpoints

#### `GET /`

A health-check endpoint that returns a simple status message indicating the API is online.

#### `POST /api/v1/predict`

The prediction endpoint accepts a JSON body with `customer_id` and returns:

- the provided `customer_id`
- customer demographic values (`age`, `annual_income`, `current_balance`)
- the predicted 12-month CLV (`predicted_12m_clv`)
- insurance cross-sell recommendations with probability scores and human-readable triggers

If the requested `customer_id` is not found in the preprocessed dataset, the endpoint returns HTTP 404.

### 4. Prediction Logic

For a valid customer ID, the service:

- selects the customer row from `features_df`
- applies the CLV feature blueprint to prepare inputs for `clv_model`
- predicts the customer lifetime value
- applies the classifier feature blueprint to prepare inputs for `insurance_classifier`
- computes purchase probability estimates for auto, health, and life insurance
- creates recommendation triggers for any product with probability > 0.4
- sorts recommendations by descending probability

This makes the API suitable for scoring individual customers and returning both numeric and narrative outputs.

---

## Inputs Expected by the Service

The service depends on the following artifacts and data files:

- `models/clv_model.pkl`
- `models/insurance_classifier.pkl`
- `models/feature_columns_clv.pkl`
- `models/feature_columns.pkl`
- `data/processed_features.csv`

The feature dataset must contain a `customer_id` column as well as the numeric fields referenced in the response payload.

---

## Output Format

A successful `POST /api/v1/predict` response returns a JSON object containing:

- `customer_id`
- `demographics`
  - `age`
  - `annual_income`
  - `current_balance`
- `metrics`
  - `predicted_12m_clv`
- `cross_sell_recommendations`
  - list of recommendation objects with `product`, `probability`, and `trigger`

If no insurance recommendations pass the threshold, the `cross_sell_recommendations` list may be empty.

---

## Dependencies

This application uses the following Python libraries:

- `fastapi`
- `pydantic`
- `joblib`
- `pandas`

It also depends on saved LightGBM and scikit-learn artifacts created by the training pipeline.

---

## How to Run

Run the FastAPI app from the repository root using a command like:

```bash
uvicorn app.main:app --reload
```

Make sure the working directory is the repository root so the relative model and data paths resolve correctly.

---

## Notes and Considerations

- The app loads all model artifacts and data into memory at startup, so it is designed for small-to-medium-sized datasets and production-ready artifact sizes.
- The service does not currently support dynamic feature updates or model hot-reloading.
- The API assumes the processed data contains the demographic fields returned in the response.
- The threshold for creating cross-sell recommendations is currently fixed at `0.4`.

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
