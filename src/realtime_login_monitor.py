# src/realtime_login_monitor.py
import pandas as pd
import time
from sklearn.ensemble import IsolationForest

# Load processed features
df = pd.read_csv("data/processed/login_features.csv")

# Use only available columns for model
features = df[["login_hour", "is_after_hours"]]

# Train Isolation Forest on all logins
model = IsolationForest(
    n_estimators=100,       # you can reduce to 50 for faster demo
    contamination=0.05,     # expected anomaly ratio
    random_state=42
)
model.fit(features)

print("=== Real-time login monitoring started ===\n")

# Simulate login events one by one
for idx, row in df.iterrows():
    sample = row[["login_hour", "is_after_hours"]].values.reshape(1, -1)
    prediction = model.predict(sample)[0]

    if prediction == -1:
        print(f"[ALERT] Suspicious login detected for user {row['user']}")
    else:
        print(f"[OK] Normal login for user {row['user']}")

    # Simulate streaming delay (adjustable)
    time.sleep(0.3)  # 0.3 seconds per login
