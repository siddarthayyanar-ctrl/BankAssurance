import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

#Initialize Faker and seed for reproducibility
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

NUM_CUSTOMERS = 2000
START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 12, 31)

print("Generating synthetic customer data...")

#--- 1. Generate Customer base profile ---
customers =[]
for i in range(NUM_CUSTOMERS):
    customer_id = f"CUST_{i+1:04d}"
    age = int(np.clip(np.random.normal(loc=35, scale=10), 18, 80))  # Age between 18 and 80

    #Income heavily corelated with age, but with some randomness
    base_income = np.random.lognormal(mean=10.5, sigma=0.4)  # Log-normal distribution for income
    age_modifier = (age / 35) if age < 35 else (60 / 35)
    annual_income = round(base_income * age_modifier, -2)

    credit_score = int(np.clip(np.random.normal(loc=700, scale=50), 300, 850))  # Credit score between 300 and 850
    tenure_months = random.randint(6, 120)  # Tenure between 6 month and 10 years
    account_balance = round(max(500, np.random.exponential(scale= annual_income / 12)), 2)
    
    customers.append({
        "customer_id": customer_id,
        "age": age,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "tenure_months": tenure_months,
        "current_account_balance": account_balance
    })

df_customers = pd.DataFrame(customers)
os.makedirs("data", exist_ok=True)
df_customers.to_csv("data/customers.csv", index=False)
print("Customer base profile generated and saved to 'data/customers.csv'.")
