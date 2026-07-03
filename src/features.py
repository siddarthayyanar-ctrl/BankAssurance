import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def engineer_features():
    print("Starting feature engineering pipeline...")
    
    # 1. Load raw data paths
    customers_df = pd.read_csv("data/customers.csv")
    transactions_df = pd.read_csv("data/transactions.csv")
    #insurance_df = pd.read_csv("data/insurance_labels.csv")
    insurance_df = pd.read_csv("data/insurance.csv")
    
    # 2. Process Transactions Ledger
    print("Aggregating transactional behavioral data...")
    
    # Convert amounts to absolute values for expense calculation (excluding Salary)
    transactions_df['abs_amount'] = transactions_df['amount'].abs()
    
    # Separate debits (expenses) from credits (salary)
    debits = transactions_df[transactions_df['category'] != 'Salary']
    credits = transactions_df[transactions_df['category'] == 'Salary']
    
    # Calculate total and average monthly credit (Income validation)
    total_credits = credits.groupby('customer_id')['amount'].sum().reset_index(name='total_salary_credited')
    
    # Calculate spending patterns by category per customer
    category_spending = debits.pivot_table(
        index='customer_id', 
        columns='category', 
        values='abs_amount', 
        aggfunc='sum', 
        fill_value=0
    ).reset_index()
    
    # Rename columns to reflect spending metrics
    category_spending.columns = [
        f"spend_{col}" if col != 'customer_id' else col for col in category_spending.columns
    ]
    
    # Calculate baseline operational transaction metrics
    tx_metrics = debits.groupby('customer_id').agg(
        total_debit_amount=('abs_amount', 'sum'),
        avg_transaction_size=('abs_amount', 'mean'),
        transaction_count=('transaction_id', 'count')
    ).reset_index()
    
    # 3. Merge Features into a Unified Master Dataset
    print("Merging features with customer master profiles...")
    master_df = customers_df.merge(total_credits, on='customer_id', how='left')
    master_df = master_df.merge(category_spending, on='customer_id', how='left')
    master_df = master_df.merge(tx_metrics, on='customer_id', how='left')
    master_df = master_df.merge(insurance_df, on='customer_id', how='left')
    
    # Fill any NaNs resulting from customers who didn't spend in specific categories
    master_df.fillna(0, inplace=True)
    
    # 4. Create Ratios & Advanced Behavioral Indicators
    # Financial Stability Indicator: Savings Rate metric
    master_df['savings_to_income_ratio'] = (
        (master_df['total_salary_credited'] - master_df['total_debit_amount']) / 
        (master_df['total_salary_credited'] + 1) # Prevent division by zero
    )
    
    # Target-specific domain indicators
    master_df['auto_risk_signal'] = master_df['spend_Auto_Service'] + master_df['spend_Fuel']
    master_df['health_risk_signal'] = master_df['spend_Medical'] + master_df['spend_Pharmacy']
    
    # 5. Feature Scaling for ML Performance
    # We drop ID and raw target columns before scaling features
    exclude_cols = ['customer_id', 'has_auto_ins', 'has_health_ins', 'has_life_ins', 'actual_12m_value']
    feature_cols = [col for col in master_df.columns if col not in exclude_cols]
    
    print(f"Engineering completed for {len(feature_cols)} statistical features.")
    
    # Export the final processed dataset
    os.makedirs("data", exist_ok=True)
    master_df.to_csv("data/processed_features.csv", index=False)
    print("Master dataset saved successfully to 'data/processed_features.csv'!")

if __name__ == "__main__":
    engineer_features()