import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="KNN Regressor", layout="wide")

MODEL_PATH = "models/knn_regressor.pkl"
SCALER_PATH = "models/scaler.pkl"

os.makedirs("models", exist_ok=True)

st.title("📈 K-Nearest Neighbors Regressor (KNN)")

# ---------------- DEFAULT DATASET ----------------
@st.cache_data
def load_default_data():
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

# ---------------- DATA SOURCE ----------------
choice = st.radio("📌 Choose Dataset", ["Default Dataset (California Housing)", "Upload CSV"])

df = None

if choice == "Upload CSV":
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
else:
    df = load_default_data()

# ---------------- MAIN ----------------
if df is not None:

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    target_col = st.selectbox("🎯 Select Target Column", df.columns, index=len(df.columns)-1)

    feature_cols = st.multiselect(
        "📌 Select Feature Columns",
        [c for c in df.columns if c != target_col],
        default=[c for c in df.columns if c != target_col]
    )

    if feature_cols and target_col:

        X = df[feature_cols]
        y = df[target_col]

        test_size = st.slider("🧪 Test Size", 0.1, 0.5, 0.2)
        k = st.slider("🔢 K Value", 1, 15, 5)

        # ---------------- TRAIN ----------------
        if st.button("🚀 Train Model"):

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Model
            model = KNeighborsRegressor(n_neighbors=k)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # Save
            joblib.dump(model, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)

            st.success("💾 Model saved successfully!")

            # ---------------- METRICS ----------------
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.success(f"📉 MSE: {mse:.4f}")
            st.success(f"📈 R² Score: {r2:.4f}")

            # ---------------- ACTUAL VS PREDICTED ----------------
            st.subheader("📊 Actual vs Predicted")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot(
                [y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                color="red"
            )
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

            # ---------------- ERROR DISTRIBUTION ----------------
            st.subheader("📉 Error Distribution")

            errors = y_test - y_pred

            fig2, ax2 = plt.subplots()
            ax2.hist(errors, bins=20)
            ax2.set_xlabel("Error")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)

            st.session_state["features"] = feature_cols

    # ---------------- PREDICTION ----------------
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):

        st.sidebar.header("🔮 Prediction Panel")

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        features = st.session_state.get("features", [])

        inputs = []

        for col in features:
            inputs.append(st.sidebar.number_input(col, value=0.0))

        if st.sidebar.button("Predict"):

            input_data = np.array(inputs).reshape(1, -1)
            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)

            st.sidebar.success(f"📈 Predicted Value: {prediction[0]:.2f}")

else:
    st.warning("📌 Please load dataset")