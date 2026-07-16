# CLV Training Script Documentation

## Overview

The file [src/train_clv.py](../src/train_clv.py) trains a supervised regression model to predict customer lifetime value (CLV) from engineered banking and insurance features. It uses a LightGBM regressor to learn the relationship between historical customer behavior and the target column `actual_12m_value`.

This script is the final training stage in the CLV prediction pipeline. It assumes that feature engineering has already been completed and that the processed dataset exists at `data/processed_features.csv`.

---

## Purpose

The script performs three main tasks:

1. Loads a prepared dataset containing customer-level features.
2. Trains a machine learning regressor to predict future customer value.
3. Evaluates model quality and saves the trained model and feature list for reuse.

---

## Functional Workflow

### 1. Data Loading

The script begins by loading the processed feature file:

- Input file: `data/processed_features.csv`
- Purpose: provides the training data used for prediction

It reads the file using `pandas.read_csv()` and stores it in a DataFrame.

### 2. Target and Feature Definition

The script defines:

- `target_col = 'actual_12m_value'`
- `exclude_cols = ['customer_id', target_col]`

The target variable is the actual 12-month customer value. The script excludes the customer identifier and the target column from the feature matrix, while keeping the insurance ownership flags available as part of the input features.

The feature matrix is built as:

- `X = df[feature_cols]`
- `y = df[target_col]`

This separation ensures that the model learns from behavioral and financial attributes rather than identifiers or target leakage.

### 3. Train/Test Split

The data is split into training and evaluation subsets using `train_test_split()`:

- Test size: 20%
- Random state: 42

This gives the model a held-out dataset to measure generalization performance.

### 4. Model Training

A LightGBM regressor is initialized with the following settings:

- `n_estimators=200`: Number of boosting rounds
- `learning_rate=0.05`: Shrinkage (regularization) parameter
- `num_leaves=31`: Maximum leaves per tree
- `random_state=42`: Reproducibility seed

The model is trained on the full feature set, which includes existing insurance ownership flags. This is intentional: the bank knows its customers' current policies and can use them as signals for CLV prediction.

### 5. Model Evaluation

After training, the model is evaluated on the held-out test set:

- **RMSE** (Root Mean Squared Error): Measures average prediction error in dollars
- **R² Score**: Indicates percentage of variance explained by the model

Both metrics are printed to stdout for quick performance validation.

### 6. Model Serialization

Trained artifacts are saved to the `models/` directory:

- `models/clv_model.pkl` — The fitted LGBMRegressor
- `models/feature_columns_clv.pkl` — Feature column names for later alignment

The feature list is saved to ensure consistent feature ordering during inference.

---

## Output Files

After execution, the script produces:

- `models/clv_model.pkl`
- `models/feature_columns_clv.pkl`

Both files are required by [app/main.py](../app/main.py) at runtime to perform CLV predictions.

---

## Key Design Decisions

- **LightGBM over sklearn RandomForest**: Gradient boosting typically outperforms bagging for continuous targets with moderate datasets.
- **Insurance flags included**: Existing product ownership is retained in the feature set because it carries signal about customer value trajectory.
- **80/20 train-test split**: Sufficient for model evaluation without holding out too much training data.
- **Fixed random seed**: Ensures reproducibility across training runs.

---

## Dependencies

- `pandas` — Data loading and manipulation
- `numpy` — Numeric operations
- `joblib` — Model serialization
- `lightgbm` — LGBMRegressor implementation
- `scikit-learn` — train_test_split, metrics

- `n_estimators=200`
- `learning_rate=0.05`
- `num_leaves=31`
- `random_state=42`
- `verbose=-1`

The model is trained with the training subset using `fit(X_train, y_train)`.

### 5. Prediction and Evaluation

After training, the model predicts values for the test set:

- `y_pred = clv_model.predict(X_test)`

The script then computes:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE), calculated as the square root of the MSE
- R-squared ($R^2$)

These metrics indicate how well the model approximates actual customer values.

### 6. Model Artifact Saving

The script saves the trained model and the selected feature list to the `models/` directory:

- `models/clv_model.pkl`
- `models/feature_columns_clv.pkl`

This makes the model ready for later inference or deployment.

---

## Key Features

### Regression-Based CLV Prediction

The script is designed for numeric prediction rather than classification. It estimates a continuous value for customer lifetime value.

### Gradient Boosting with LightGBM

LightGBM is used because it is efficient, handles structured/tabular data well, and generally performs strongly on tabular regression tasks.

### Automated Evaluation Metrics

The script prints and calculates model performance metrics such as:

- RMSE for error magnitude
- R-squared for explanatory power

### Reusable Model Export

The model and feature definition are persisted so they can be loaded later in a production pipeline, API, or dashboard.

### Simple End-to-End Training Flow

The script requires minimal setup and performs the full training loop in one place.

---

## Inputs Expected by the Script

The script expects the following input structure:

- A CSV file named `data/processed_features.csv`
- A column named `actual_12m_value`
- A set of feature columns derived from engineering steps

The script will fail or behave unexpectedly if the required dataset or columns are missing.

---

## Output Artifacts

When successful, the script creates:

- `models/clv_model.pkl`: the trained LightGBM model
- `models/feature_columns_clv.pkl`: the ordered list of input features used during training

---

## Dependencies

The script depends on the following Python libraries:

- `pandas`
- `numpy`
- `lightgbm`
- `scikit-learn`
- `joblib`

---

## How to Run

Run this script from the repository root:

```bash
python src/train_clv.py
```

Before running it, ensure that:

1. The processed data file exists at `data/processed_features.csv`
2. The repository root is the current working directory
3. The required Python packages are installed

If the feature engineering step has not yet been run, it should be executed first using the feature generation script.

---

## Notes and Considerations

- The script uses relative file paths, so it should be executed from the project root.
- It does not currently support command-line arguments or configurable hyperparameters.
- It is a training script rather than a full production service; it focuses on model creation and evaluation.
- The saved model relies on the same feature columns used during training, so the feature order must remain consistent.

---

## Summary

This file is responsible for training and evaluating a LightGBM regression model that predicts customer lifetime value. It loads engineered customer features, trains the model, evaluates its accuracy, and saves the trained model and feature schema for downstream use.
