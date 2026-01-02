import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("data/processed/login_features.csv")

X = df[['login_hour', 'is_after_hours']]

model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(X)
anomaly_rate = (df['anomaly'] == -1).mean()
print(f"Anomaly Rate: {anomaly_rate:.2%}")

df.to_csv("data/processed/anomaly_results.csv", index=False)
print("Anomaly detection completed.")


