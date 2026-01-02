import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/anomaly_results.csv")

plt.figure()
df['anomaly'].value_counts().plot(kind='bar')
plt.title("Login Anomaly Distribution")
plt.xlabel("Anomaly Label")
plt.ylabel("Count")
plt.show()
