import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


st.set_page_config(
    page_title="Customer Risk Prediction System (KNN)",
    layout="wide"
)


st.title("Customer Risk Prediction System (KNN)")
st.write("This system predicts customer risk by comparing them with similar customers.")


@st.cache_data
def load_data():
    df = pd.read_csv("credit_risk_dataset.csv")
    return df

df = load_data()

features = [
    "person_age",
    "person_income",
    "loan_amnt",
    "cb_person_cred_hist_length"
]

X = df[features]
y = df["loan_status"]

 
X = X.fillna(X.mean())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.sidebar.header("Enter Customer Details")

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.number_input("Annual Income", 10000, 200000, 50000)
loan_amount = st.sidebar.number_input("Loan Amount", 1000, 50000, 10000)

credit_history = st.sidebar.selectbox(
    "Credit History",
    ["Yes", "No"]
)

credit_value = 5 if credit_history == "Yes" else 0

k_value = st.sidebar.slider("K Value (Neighbors)", 1, 15, 5)


model = KNeighborsClassifier(n_neighbors=k_value)
model.fit(X_scaled, y)


if st.button("Predict Customer Risk"):

    new_customer = [[
        age,
        income,
        loan_amount,
        credit_value
    ]]

    new_scaled = scaler.transform(new_customer)
    prediction = model.predict(new_scaled)[0]

   
    st.subheader("Prediction Result")

    if prediction == 1:
        st.markdown(
            "<h2 style='color:red'>🔴 High Risk Customer</h2>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h2 style='color:green'>🟢 Low Risk Customer</h2>",
            unsafe_allow_html=True
        )



    st.subheader("Nearest Neighbors Explanation")

    nn = NearestNeighbors(n_neighbors=k_value)
    nn.fit(X_scaled)

    distances, indices = nn.kneighbors(new_scaled)

    neighbors_labels = y.iloc[indices[0]]

    high_risk = sum(neighbors_labels)
    low_risk = k_value - high_risk

    st.write(f"Number of neighbors considered: {k_value}")
    st.write(f"High Risk among neighbors: {high_risk}")
    st.write(f"Low Risk among neighbors: {low_risk}")

    majority = "High Risk" if high_risk > low_risk else "Low Risk"
    st.write(f"Majority Class: {majority}")

    st.write("Nearest Similar Customers:")

    neighbors_table = df.iloc[indices[0]][features + ["loan_status"]]
    st.dataframe(neighbors_table)

    st.subheader("Business Insight")

    st.info(
        "This decision is based on similarity with nearby customers in feature space. "
        "Customers who share similar age, income, loan amount, and credit history "
        "tend to show similar repayment behavior."
    )
