import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="Wholesale Customer Segmentation", layout="wide")

st.title("📦 Wholesale Customer Segmentation App")
st.write("Customers are grouped based on annual spending behavior using K-Means Clustering.")

# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Wholesale customers data.csv")

df = load_data()

# -----------------------------------
# TASK 1: DATA EXPLORATION
# -----------------------------------
st.subheader("🔹 Dataset Preview")
st.dataframe(df.head())
# -----------------------------------
# TASK 2: FEATURE SELECTION
# -----------------------------------
features = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]



X = df[features]

st.write("These features directly represent customer purchasing behavior.")

# -----------------------------------
# TASK 3: DATA PREPARATION
# -----------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.success("Data Standardized Successfully")

# -----------------------------------
# TASK 4 & 5: ELBOW METHOD
# -----------------------------------
st.subheader("🔹 Elbow Method (Optimal K)")

wcss = []
K_range = range(2,11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

fig1 = plt.figure()
plt.plot(K_range, wcss, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
st.pyplot(fig1)

k = st.slider("Select Number of Clusters (K)", 2, 6, 3)

st.write(f"Chosen K = {k}")

# -----------------------------------
# TASK 6: TRAIN MODEL & ASSIGN CLUSTERS
# -----------------------------------
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

st.subheader("🔹 Clustered Dataset")
st.dataframe(df.head())

# -----------------------------------
# TASK 7: VISUALIZATION
# -----------------------------------
st.subheader("🔹 Cluster Visualization (Milk vs Grocery)")

fig2 = plt.figure()
plt.scatter(df["Milk"], df["Grocery"], c=df["Cluster"])
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:,1], centers[:,2], marker="X")
plt.xlabel("Milk Spending")
plt.ylabel("Grocery Spending")
plt.title("Customer Segments")
st.pyplot(fig2)

# -----------------------------------
# TASK 8: CLUSTER PROFILING
# -----------------------------------
st.subheader("🔹 Cluster Profiling")

profile = df.groupby("Cluster")[features].mean()
st.dataframe(profile)

st.subheader("🧾 Cluster Summaries")

for i in profile.index:
    st.write(f"Cluster {i}:")
    st.write(profile.loc[i])
    st.write("----")

# -----------------------------------
# TASK 9: BUSINESS INSIGHTS
# -----------------------------------
st.subheader("🔹 Business Strategies")

st.write("""
Cluster 0 → Offer discounts & loyalty programs  
Cluster 1 → Bulk supply & inventory prioritization  
Cluster 2 → Promote fresh & frozen products  
""")

# -----------------------------------
# TASK 10: STABILITY CHECK
# -----------------------------------
st.subheader("🔹 Stability Test")

kmeans2 = KMeans(n_clusters=k, random_state=100)
clusters2 = kmeans2.fit_predict(X_scaled)

same = np.sum(clusters == clusters2)
st.write("Matching Assignments with new random state:", same, "out of", len(clusters))

st.write("Limitation: K-Means requires predefined K and is sensitive to outliers.")
