import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

income = np.random.normal(65000, 20000, N).clip(15000, 200000)
balance = np.random.normal(3000, 2000, N).clip(0, 30000)
credit_limit = np.random.normal(8000, 4000, N).clip(1000, 50000)
utilization = (balance / credit_limit).clip(0, 1)
late_payments = np.random.poisson(0.8, N)
age = np.random.randint(21, 75, N)
tenure_years = np.random.gamma(3, 1.2, N).clip(0, 25)

segments = np.random.choice(["student","salaried","self-employed","retired"], size=N, p=[0.15,0.55,0.20,0.10])
regions = np.random.choice(["west","midwest","south","northeast"], size=N, p=[0.25,0.25,0.30,0.20])

risk_score = (
    0.6*utilization
    + 0.15*(late_payments > 1).astype(int)
    + 0.1*(income < 40000).astype(int)
    + 0.1*(tenure_years < 1).astype(int)
    + 0.05*(segments == "self-employed").astype(int)
)

threshold = np.quantile(risk_score, 0.7)  # top 30% labeled "bad"
risk_label = np.where(risk_score >= threshold, "bad", "good")

df = pd.DataFrame({
    "annual_income": income.round(0).astype(int),
    "current_balance": balance.round(2),
    "credit_limit": credit_limit.round(0).astype(int),
    "utilization_ratio": utilization.round(3),
    "late_payments_24m": late_payments,
    "age": age,
    "tenure_years": tenure_years.round(1),
    "customer_segment": segments,
    "region": regions,
    "risk_label": risk_label
})

df.to_csv("data/credit_risk_sample.csv", index=False)
print("Wrote data/credit_risk_sample.csv with shape:", df.shape)
