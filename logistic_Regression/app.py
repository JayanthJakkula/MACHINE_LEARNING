import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# App Title
# -------------------------------
st.title("📊 Customer Churn Prediction App")
st.write("This app predicts whether a customer is likely to churn using Logistic Regression.")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("telco_dataset.csv")
df = load_data()

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Data Preprocessing
# -------------------------------
le = LabelEncoder()
df_encoded = df.copy()

for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# -------------------------------
# Features & Target
# -------------------------------
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Train Model
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------
# Metrics
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

# -------------------------------
# Display Metrics
# -------------------------------
st.subheader("📈 Model Performance")

st.write(f"**Accuracy:** {acc:.2f}")
st.write(f"**Precision:** {prec:.2f}")
st.write(f"**Recall:** {rec:.2f}")
st.write(f"**F1 Score:** {f1:.2f}")

st.subheader("📊 Confusion Matrix")
st.write(cm)

# -------------------------------
# Business Analysis
# -------------------------------
st.subheader("📌 Business Insights")

st.write(f"✅ **Churn customers correctly identified (TP):** {TP}")
st.write(f"❌ **Non-churn customers misclassified (FP):** {FP}")
st.write(f"⚠️ **Missed churn customers (FN):** {FN}")

st.info(
    "In churn prediction, missing a churn customer (False Negative) is more costly "
    "than wrongly flagging a loyal customer."
)

