import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
st.set_page_config(page_title="KNN Classifier App", layout="wide")

MODEL_PATH = "models/knn_model.pkl"
SCALER_PATH = "models/scaler.pkl"

os.makedirs("models", exist_ok=True)

st.title("🧠 KNN Classifier - Breast Cancer Dataset")

# ---------------- DEFAULT DATASET (UPDATED) ----------------
@st.cache_data
def load_default_data():
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    return df

# ---------------- DATA SOURCE ----------------
choice = st.radio("📌 Choose Dataset", ["Breast Cancer (Default)", "Upload CSV"])

df = None

if choice == "Upload CSV":
    file = st.file_uploader("Upload your CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
else:
    df = load_default_data()

# ---------------- MAIN APP ----------------
if df is not None:

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.info("🎯 Target Mapping: 0 = Malignant | 1 = Benign")

    target_col = st.selectbox("🎯 Select Target Column", df.columns, index=len(df.columns)-1)

    feature_cols = st.multiselect(
        "📌 Select Feature Columns",
        [c for c in df.columns if c != target_col],
        default=[c for c in df.columns if c != target_col]
    )

    if feature_cols and target_col:

        X = df[feature_cols]
        y = df[target_col]

        # ---------------- SAFE CHECK ----------------
        if y.dtype != "object" and y.nunique() > 20:
            st.error("❌ Continuous target detected. This app supports ONLY classification datasets.")
            st.stop()

        y = y.astype(str)

        st.success("✅ Dataset is valid for KNN Classification")

        test_size = st.slider("🧪 Test Size", 0.1, 0.5, 0.2)
        k = st.slider("🔢 K Value", 1, 15, 5)

        # ---------------- TRAIN MODEL ----------------
        if st.button("🚀 Train Model"):

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Model
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # Save model
            joblib.dump(model, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)

            st.success("💾 Model saved successfully!")

            # ---------------- METRICS ----------------
            acc = accuracy_score(y_test, y_pred)
            st.success(f"🎯 Accuracy: {acc:.2f}")

            # ---------------- CONFUSION MATRIX ----------------
            st.subheader("📉 Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # ---------------- CLASS REPORT ----------------
            st.subheader("📄 Classification Report")
            st.text(classification_report(y_test, y_pred))

            # ---------------- VISUALIZATION ----------------
            st.subheader("📊 Actual vs Predicted")

            fig2, ax2 = plt.subplots()
            ax2.hist([y_test, y_pred], label=["Actual", "Predicted"])
            ax2.legend()
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

            result = "Benign (1)" if prediction[0] == "1" else "Malignant (0)"

            st.sidebar.success(f"🧠 Prediction: {result}")

else:
    st.warning("📌 Please load dataset")