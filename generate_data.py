"""
SafeSphere AI - Synthetic Flood Dataset Generator
Generates a structured tabular dataset with realistic correlations between
environmental features and flood risk, so the model learns real signal
instead of memorizing noise.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1200

rainfall = np.random.gamma(shape=2.0, scale=40, size=N).clip(0, 300)          # mm
water_level = np.random.normal(2.5, 1.2, N).clip(0.2, 8.0)                    # m
flow_rate = np.random.normal(50, 25, N).clip(5, 200)                          # m3/s
temperature = np.random.normal(28, 4, N).clip(15, 42)                         # C
humidity = np.random.normal(70, 12, N).clip(30, 100)                          # %
elevation = np.random.normal(50, 30, N).clip(0, 200)                          # m
historical_flood_flag = np.random.binomial(1, 0.3, N)                         # 0/1

# Composite hazard score built from weighted, correlated factors + noise
raw_score = (
    0.35 * (rainfall / 300) +
    0.30 * (water_level / 8.0) +
    0.15 * (flow_rate / 200) +
    0.10 * (humidity / 100) +
    0.15 * historical_flood_flag -
    0.15 * (elevation / 200) +
    np.random.normal(0, 0.05, N)
)
raw_score = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())
risk_score = (raw_score * 100).round(1)

def bucket(s):
    if s < 40:
        return "LOW"
    elif s < 70:
        return "MODERATE"
    else:
        return "HIGH"

risk_level = [bucket(s) for s in risk_score]

df = pd.DataFrame({
    "rainfall": rainfall.round(1),
    "water_level": water_level.round(2),
    "flow_rate": flow_rate.round(1),
    "temperature": temperature.round(1),
    "humidity": humidity.round(1),
    "elevation": elevation.round(1),
    "historical_flood_flag": historical_flood_flag,
    "risk_score": risk_score,
    "risk_level": risk_level,
})

df.to_csv("flood_dataset.csv", index=False)
print(f"Generated {len(df)} rows -> flood_dataset.csv")
print(df["risk_level"].value_counts())
