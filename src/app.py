import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Login Anomaly Detection", layout="wide")
st.title("Login Anomaly Detection Dashboard")

# -------------------------------
# Load data & train model ONCE
# -------------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("data/processed/login_features.csv")
    feature_cols = ["login_hour", "is_after_hours"]
    X = df[feature_cols]

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    model.fit(X)
    return model, feature_cols, df

model, feature_cols, df = load_model()

# -------------------------------
# Sidebar input (DYNAMIC INPUT)
# -------------------------------
st.sidebar.header("Enter Login Details")
user = st.sidebar.text_input("User ID", "MYB0686")
login_hour = st.sidebar.slider("Login Hour", 0, 23, 9)
is_after_hours = st.sidebar.selectbox("Is After Hours?", [0, 1])

# -------------------------------
# Prediction
# -------------------------------
input_df = pd.DataFrame(
    [[login_hour, is_after_hours]],
    columns=feature_cols
)

prediction = model.predict(input_df)[0]

if prediction == -1:
    st.error(f"⚠️ Suspicious login detected for user {user}")
else:
    st.success(f"✅ Normal login for user {user}")

# -------------------------------
# Show ALL logs with prediction
# -------------------------------
st.subheader("All Login Records with Anomaly Labels")

df_display = df.copy()
df_display["anomaly"] = model.predict(df[feature_cols])
df_display["status"] = df_display["anomaly"].map({1: "Normal", -1: "Suspicious"})

st.dataframe(df_display)

# -------------------------------
# Visualization
# -------------------------------
st.subheader("Anomaly Distribution")
st.bar_chart(df_display["status"].value_counts())
