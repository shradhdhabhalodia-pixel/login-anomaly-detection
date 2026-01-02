import pandas as pd

# Input and output files
INPUT_FILE = "data/processed/login_features.csv"
OUTPUT_FILE = "data/processed/login_time_anomalies.csv"

# Load processed login data
df = pd.read_csv(INPUT_FILE)

# STEP 1: Compute user-level login statistics
# - mean login hour
# - standard deviation of login hour
user_stats = df.groupby("user")["login_hour"].agg(["mean", "std"]).reset_index()
user_stats.rename(columns={"mean": "login_hour_mean", "std": "login_hour_std"}, inplace=True)

# Merge stats back to main dataframe
df = df.merge(user_stats, on="user", how="left")

# STEP 2: Flag time anomalies
# - If login hour is more than 2 standard deviations away from user's mean login hour
df["time_anomaly"] = ((df["login_hour"] - df["login_hour_mean"]).abs() > 2 * df["login_hour_std"]).astype(int)

# Optional: Combine with existing 'is_after_hours' for additional context
# 1 = suspicious (either after hours or abnormal time)
df["suspicious_login"] = ((df["is_after_hours"] == 1) | (df["time_anomaly"] == 1)).astype(int)

# STEP 3: Save results
df.to_csv(OUTPUT_FILE, index=False)

print(f"Time-based anomaly analysis completed. Results saved to {OUTPUT_FILE}")
