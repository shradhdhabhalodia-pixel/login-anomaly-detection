import pandas as pd
from sklearn.ensemble import IsolationForest

# -----------------------------
# STEP 1: Load preprocessed login features
# -----------------------------
df = pd.read_csv("data/processed/login_features.csv")

# Column names
USER_COL = "user"
FEATURE_COLS = ["login_hour", "is_after_hours"]  # Only available features

results = []

# -----------------------------
# STEP 2: Train per-user Isolation Forest
# -----------------------------
for user, user_df in df.groupby(USER_COL):
    if len(user_df) < 10:
        # Skip users with too few logins
        continue

    X = user_df[FEATURE_COLS]

    # Isolation Forest model
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # Assume 5% anomalies
        random_state=42
    )

    # Fit model and predict anomalies
    user_df = user_df.copy()
    user_df["anomaly"] = model.fit_predict(X)

    results.append(user_df)

# -----------------------------
# STEP 3: Combine all users
# -----------------------------
final_df = pd.concat(results, ignore_index=True)

# -----------------------------
# STEP 4: Save results
# -----------------------------
final_df.to_csv(
    "data/processed/login_anomaly_per_user.csv",
    index=False
)

print("Per-user anomaly detection completed successfully.")
