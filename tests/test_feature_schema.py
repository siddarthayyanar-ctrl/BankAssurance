import pandas as pd


def test_processed_features_do_not_include_string_model_features():
    df = pd.read_csv("data/processed_features.csv")

    feature_cols = [
        col
        for col in df.columns
        if col not in {"customer_id", "actual_12m_value", "has_auto_ins", "has_health_ins", "has_life_ins"}
    ]

    bad_columns = [col for col in feature_cols if df[col].dtype == object]

    assert not bad_columns, f"Non-numeric feature columns found: {bad_columns}"
