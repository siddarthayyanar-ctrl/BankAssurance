# Recommendation / Insurance Propensity Training Documentation

## Overview

The file [src/train_recommend.py](../src/train_recommend.py) trains a multi-output classification model to predict which insurance products a customer is likely to purchase. It uses a Random Forest-based classifier wrapped in a multi-label framework so that it can predict three insurance outcomes simultaneously:

- whether the customer is likely to buy auto insurance
- whether the customer is likely to buy health insurance
- whether the customer is likely to buy life insurance

This script is the training step for the insurance cross-sell recommendation pipeline and is designed to turn customer behavior data into actionable propensity predictions.

---

## Purpose

The script performs four main tasks:

1. Loads preprocessed customer features from the data pipeline.
2. Defines the insurance purchase targets and input features.
3. Trains a multi-output classifier to estimate multiple purchase probabilities at once.
4. Evaluates performance and saves the trained model for future use.

---

## Functional Workflow

### 1. Data Loading

The script begins by reading the processed feature dataset:

- Input file: `data/processed_features.csv`
- Purpose: provides the feature set used for classification

This dataset is assumed to already contain engineered customer features and the target columns for insurance adoption.

### 2. Target and Feature Definition

The script identifies three target columns:

- `has_auto_ins`
- `has_health_ins`
- `has_life_ins`

These columns represent whether a customer has purchased each insurance product.

The script excludes the following from model features:

- `customer_id`
- `actual_12m_value`
- the three target columns themselves

All remaining columns are used as feature inputs.

The data is structured as:

- `X`: input features for the classifier
- `y`: multi-output target matrix for the three insurance products

### 3. Train/Test Split

The data is split into training and testing subsets using `train_test_split()`:

- test size: 20%
- random state: 42

This ensures the model is evaluated on data it did not see during training.

### 4. Model Training

The script uses a two-level modeling approach:

- A `RandomForestClassifier` serves as the base classifier.
- `MultiOutputClassifier` wraps it so the same model can learn all three targets at once.

The model configuration includes:

- `n_estimators=100`
- `max_depth=10`
- `random_state=42`
- `n_jobs=-1` for parallel training

This approach allows the model to learn a shared representation of customer behavior while predicting each insurance product independently.

### 5. Prediction and Evaluation

After training, the model predicts outcomes for the test set.

For each insurance target, the script prints a classification report with metrics such as:

- precision
- recall
- f1-score
- support

The report is generated for each of the three target labels separately, making it easier to evaluate performance by product type.

### 6. Model Artifact Saving

The trained classifier is exported as:

- `models/insurance_classifier.pkl`

This allows the model to be reused later for scoring new customers or integrating into a recommendation engine.

---

## Key Features

### Multi-Output Classification

One of the strongest features of this script is that it trains a single model capable of predicting multiple insurance products simultaneously. This is efficient and practical for cross-sell marketing scenarios.

### Random Forest-Based Learning

Random Forests are well-suited for tabular data and are robust to nonlinear relationships. They also provide good baseline predictive performance without requiring extensive feature engineering.

### Product-Specific Evaluation

The script does not provide only one global score. Instead, it produces a detailed report for each insurance segment, helping stakeholders understand which products the model predicts better.

### Reusable Model Export

The trained model is saved to disk so it can be deployed later in an application, dashboard, or batch scoring workflow.

### Simple End-to-End Training Pipeline

The script is compact and self-contained, making it easy to run and adapt for experimentation.

---

## Inputs Expected by the Script

The script expects the following:

- a processed CSV file at `data/processed_features.csv`
- columns representing customer-level features
- the three target columns:
  - `has_auto_ins`
  - `has_health_ins`
  - `has_life_ins`

If these columns are missing or renamed, the script will not work as intended.

---

## Outputs Produced by the Script

When the script finishes successfully, it prints:

- a training progress message
- a classification report for each target product
- a confirmation that the model artifact was saved

The saved artifact is:

- `models/insurance_classifier.pkl`

---

## Dependencies

This script relies on the following libraries:

- `pandas`
- `joblib`
- `scikit-learn`

---

## How to Run

Run the script from the repository root:

```bash
python3 src/train_recommend.py
```

Before running it, ensure that:

1. the processed data file exists at `data/processed_features.csv`
2. the repository root is the active working directory
3. the required Python dependencies have been installed

---

## Interpretation of Results

The classification report helps answer questions such as:

- Which insurance products are easier for the model to predict?
- Are some products more often missed by the model than others?
- Which products may require better features or a different model configuration?

Because the targets are binary, the model predicts whether a customer is likely or unlikely to purchase each product.

---

## Notes and Considerations

- The script uses relative paths, so it should be executed from the project root.
- The current implementation is a training script rather than a full deployment service.
- Performance depends heavily on the quality of the engineered features in `processed_features.csv`.
- The model is a baseline approach and may be improved with hyperparameter tuning or more advanced algorithms.

---

## Summary

This file is responsible for training a multi-output Random Forest classifier that predicts a customer’s likelihood of purchasing auto, health, and life insurance. It loads engineered features, trains the model, evaluates prediction quality for each insurance product, and saves the trained model for downstream recommendation or cross-sell use.
