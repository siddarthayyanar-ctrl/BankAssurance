Here is a complete, production-grade `README.md` for your project workspace. This document is written in clean, standard Markdown structure, making it ready to drop straight into your GitHub repository to showcase your engineering framework.

---

# Enterprise Predictive Bancassurance Engine (v2.0)

A high-performance, production-grade Machine Learning system built to optimize banking-insurance cross-sell propensities and forecast **12-Month Customer Lifetime Value (CLV)**. This platform replaces traditional flat-file dependencies with an optimized SQLite database abstraction layer, delivers explainable AI metrics via localized TreeSHAP calculus, and exposes real-time monitoring insights through an interactive multi-tab application interface.

---

## 🏗️ System Architecture

The ecosystem relies on an asynchronous decoupled layout separating data processing, database access, model inference, and presentation layers:

```text
    [ Raw Data Pipelines ]  ──►  [ SQL Relational Layer ] (SQLite Indexing)
                                            │
                                            ▼
   [ Streamlit Frontend ]   ◄──► [ FastAPI Service Node ] ◄──► [ Core ML Engine ]
   (User Interface Layer)       (REST Endpoints Pipeline)       (Scikit-Learn & SHAP)

```

1. **Database Relational Layer (`SQLite`)**: Houses the processed database rows. Optimized with unique customer indexes to assure immediate record lookups.
2. **REST API Service Node (`FastAPI`)**: Manages real-time model scoring pipelines, computes local SHAP feature attributions on the fly, and generates dynamic marketing triggers.
3. **Machine Learning Core (`Scikit-Learn` & `SHAP`)**: Employs parallel Random Forest architectures—a continuous target regressor for CLV and a multi-label classifier for product conversions.
4. **Interactive Dashboard (`Streamlit`)**: A multi-tab frontend serving Relationship Manager lookups, sandbox model simulations, and system telemetry metrics.

---

## 📊 Database Schema & Layout

Data properties are structured inside a single indexed table named `customer_profiles` inside `data/bancassurance.db`.

### Core Table Structure: `customer_profiles`

* **Primary Index Key**: `customer_id` (`TEXT`, Unique Indexed)
* **Demographics**: `age` (`INTEGER`), `annual_income` (`REAL`), `credit_score` (`INTEGER`), `tenure_months` (`INTEGER`)
* **Financial Positions**: `current_account_balance` (`REAL`), `total_salary_credited` (`REAL`), `total_debit_amount` (`REAL`)
* **Behavioral Spending Triggers**: `spend_Auto_Service` (`REAL`), `spend_Dining` (`REAL`), `spend_Fuel` (`REAL`), `spend_Groceries` (`REAL`), `spend_Medical` (`REAL`), `spend_Pharmacy` (`REAL`), `spend_Real_Estate` (`REAL`), `spend_Utilities` (`REAL`)
* **Operational Transaction Profiles**: `avg_transaction_size` (`REAL`), `transaction_count` (`INTEGER`)
* **Historical Ground-Truth Classes**: `has_auto_ins` (`INTEGER`), `has_health_ins` (`INTEGER`), `has_life_ins` (`INTEGER`), `actual_12m_value` (`REAL`)

---

## 🛠️ Workspace Directory Tree

Ensure your repository follows this module configuration structure:

```text
bancassurance-engine/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI Application Layer, SHAP Calculators & Database Queries
│   └── ui.py              # Streamlit Multi-Tab Presentation Interface (Portfolio/Sandbox/Monitor)
├── data/
│   ├── bancassurance.db   # Compiled Relational Database with Indexed Customer Profiles
│   └── processed_features.csv  # Feature Engineering Output (pre-migration)
├── models/
│   ├── clv_model.pkl           # LightGBM Regression binary for 12M CLV prediction
│   ├── insurance_classifier.pkl # RandomForest MultiOutputClassifier for 3 insurance products
│   ├── feature_columns_clv.pkl  # CLV feature names (saved by train_clv.py)
│   └── feature_columns.pkl      # Classifier feature names (manual generation from train_recommend.py)
├── src/
│   ├── data_gen.py
│   ├── db_migration.py    # SQLite Database Initialization & Population Tool
│   ├── demo.py
│   ├── features.py
│   ├── train_clv.py       # CLV Model Training Pipeline (LGBMRegressor)
│   └── train_recommend.py # Insurance Classifier Training (MultiOutputClassifier + RandomForest)
└── requirements.txt       # Python Package Dependencies

```

---

## 🚀 Setup & Execution Instructions

### 1. Environment Initialization

Clone the repository workspace and download the essential dependencies:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install system dependencies
pip install -r requirements.txt

```

> **Note on `requirements.txt**`: Ensure your file contains at least `fastapi`, `uvicorn`, `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `shap`, and `requests`.

### 2. Relational Database Migration

Inject your engineered flat files into the relational SQLite database layer:

```bash
python3 src/db_migration.py

```

*Expected Output:* `Migration Complete! Created database with [X] indexed relational customer entries.`

### 3. Launching the Back-End API Server

Spin up the FastAPI server utilizing the Uvicorn ASGI production engine:

```bash
uvicorn app.main:app --reload --port 8000

```

*Validation Check:* Open `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)` in your web browser to test out the interactive Swagger API documentation UI.

### 4. Running the Front-End Application UI

In a separate terminal window, launch the interactive user interface dashboard:

```bash
streamlit run app/ui.py

```

## 🤖 GenAI Context File (JSON)

- **File**: app/context/knowledge.json
- **Purpose**: Contains domain facts and short entries used by the GenAI assistant in the Streamlit Chat tab. The UI loads the top-level `context` string or joins items from the `entries` list.
- **Edit Instructions**: Open app/context/knowledge.json and update the `context` text or add/remove items in the `entries` array. Keep the JSON valid UTF-8.
- **Validation**: Quick check with Python:

```bash
python -c "import json; json.load(open('app/context/knowledge.json')); print('OK')"
```

- **Fallback**: If the file is missing or invalid, the app logs a warning and the GenAI assistant will still reply without the enriched context.

---

## ⚡ API Endpoint Specification Map

The FastAPI service node exposes three operational endpoint networks:

### 1. `POST /api/v2/predict`

* **Description**: Takes a database customer ID, queries records from SQLite, calculates ML cross-sell probabilities, and runs a local SHAP tree-attribution calculation to explain the top driving factors behind the prediction.
* **Payload Example**: `{"customer_id": "CUST_0001"}`

### 2. `POST /api/v2/predict-raw`

* **Description**: Processes manual raw data inputs directly from web forms for new prospects who don't have records in the database yet.
* **Payload Example**: Pass a full dictionary containing customer metrics (Age, Balances, Category Spending).

### 3. `GET /api/v2/meta`

* **Description**: Returns system-wide aggregates and baseline customer conversion metrics used to power the MLOps telemetry dashboard.

---

You can drop this directly into a `README.md` file in your root folder. It gives anyone looking at your project an immediate, crystal-clear understanding of the software engineering principles you put into building it!