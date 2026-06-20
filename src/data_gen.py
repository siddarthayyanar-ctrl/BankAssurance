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

#--- 2. Generate Transaction history ---
transactions = []
categories = ["Salary", "Groceries", "Fuel", "Auto_Service", "Medical", "Pharmacy", "Real_Estate", "Dining", "Utilities"]

for cust in customers:
    cust_id = cust["customer_id"]
    income = cust["annual_income"]
    age = cust["age"]

    # 6 months of monthly salary transactions
    monthly_salary = round(income / 12, 2)
    for month in range(6):
        salary_date = START_DATE + timedelta(days=month * 30 + random.randint(0, 3))
        transactions.append({
            "transaction_id": f"TX_{fake.uuid4()[:8]}",
            "customer_id": cust_id,
            "timestamp": salary_date.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": monthly_salary,
            "category": "Salary"
        })
        
        # Generate random day-to-day debit transactions
        num_tx = random.randint(40, 120)
        for _ in range(num_tx):
            tx_date = fake.date_time_between(start_date=START_DATE, end_date=END_DATE)
            category = random.choice(categories[1:]) # Exclude salary for debits
        
            # Inject deliberate consumer behavioral signals
            if category == "Auto_Service" and random.random() > 0.6:
                amount = round(random.uniform(200, 1500), 2)  # Car repair indicators
            elif category == "Real_Estate" and age > 30 and income > 80000 and random.random() > 0.95:
                amount = round(random.uniform(3000, 15000), 2) # Home buying indicators
            elif category == "Medical" and age > 50 and random.random() > 0.7:
                amount = round(random.uniform(500, 4000), 2)  # Health indicators
            else:
                amount = round(np.random.exponential(scale=50), 2) # Default minor spending

            transactions.append({
                "transaction_id": f"TX_{fake.uuid4()[:8]}",
                "customer_id": cust_id,
                "timestamp": tx_date.strftime("%Y-%m-%d %H:%M:%S"),
                "amount": -abs(amount), # Negative for debit
                "category": category
            })

df_transactions = pd.DataFrame(transactions)

# --- 3. GENERATE GROUND TRUTH INSURANCE TARGET LABELS ---
# Rules map transaction intents loosely to conversions, giving ML patterns to find
insurance = []
for cust in customers:
    cust_id = cust["customer_id"]
    cust_txs = df_transactions[df_transactions["customer_id"] == cust_id]
    
    # Base probabilities modified by behavioral telemetry
    has_car_expenses = "Auto_Service" in cust_txs["category"].values
    has_large_medical = cust_txs[cust_txs["category"] == "Medical"]["amount"].abs().max() > 1500
    has_property_payment = "Real_Estate" in cust_txs["category"].values
    
    has_auto_ins = 1 if (has_car_expenses and random.random() > 0.3) else (1 if random.random() > 0.85 else 0)
    has_health_ins = 1 if (has_large_medical or cust["age"] > 55) and random.random() > 0.4 else (1 if random.random() > 0.90 else 0)
    has_life_ins = 1 if (cust["annual_income"] > 100000 and cust["age"] > 35 and random.random() > 0.3) else 0
    
    # Calculate an evaluation matrix for the continuous 12-Month CLV metric
    # Customer Lifetime Value (CLV) is a metric that predicts the total net profit or revenue a business can expect to earn from a single customer throughout their entire relationship.
    # CLV = Average Purchase Value × Average Purchase Frequency × Customer Lifespan
    # Higher balances and policy counts mean premium value to a bancassurance entity
    num_policies = has_auto_ins + has_health_ins + has_life_ins
    actual_clv = round((cust["current_account_balance"] * 0.02) + (num_policies * random.randint(300, 800)), 2)

    insurance.append({
        "customer_id": cust_id,
        "has_auto_ins": has_auto_ins,
        "has_health_ins": has_health_ins,
        "has_life_ins": has_life_ins,
        "actual_12m_value": actual_clv
    })

df_insurance = pd.DataFrame(insurance)
os.makedirs("data", exist_ok=True)
df_customers.to_csv("data/customers.csv", index=False)
df_transactions.to_csv("data/transactions.csv", index=False)
df_insurance.to_csv("data/insurance.csv", index=False)
print("Data generated and saved to 'data/' directory.")
