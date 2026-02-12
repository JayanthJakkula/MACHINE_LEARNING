import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage

# ------------------------------------------------------------
# 1️⃣ PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="News Topic Discovery Dashboard", layout="wide")

st.title("🟣 News Topic Discovery Dashboard")

st.markdown("""
This system uses *Hierarchical Clustering* to automatically group similar news articles based on textual similarity.
""")

st.info("👉 Discover hidden themes without defining categories upfront.")

# ------------------------------------------------------------
# 2️⃣ SIDEBAR CONTROLS
# ------------------------------------------------------------
st.sidebar.header("📂 Dataset Handling")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin1")
else:
    st.warning("Please upload a CSV file.")
    st.stop()


# Auto detect text column
text_columns = [col for col in df.columns if df[col].dtype == "object"]

if len(text_columns) == 0:
    st.error("No text column detected.")
    st.stop()

text_column = st.sidebar.selectbox("Select Text Column", text_columns)
documents = df[text_column].astype(str)

# ------------------------------------------------------------
# 📝 TEXT VECTORIZATION CONTROLS
# ------------------------------------------------------------
st.sidebar.header("📝 Text Vectorization")

max_features = st.sidebar.slider("Maximum TF-IDF Features", 100, 2000, 1000)

use_stopwords = st.sidebar.checkbox("Remove English Stopwords", value=True)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_option == "Unigrams":
    ngram_range = (1, 1)
elif ngram_option == "Bigrams":
    ngram_range = (2, 2)
else:
    ngram_range = (1, 2)

# ------------------------------------------------------------
# 🌳 HIERARCHICAL CONTROLS
# ------------------------------------------------------------
st.sidebar.header("🌳 Hierarchical Clustering")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

distance_metric = "euclidean"

subset_size = st.sidebar.slider(
    "Number of Articles for Dendrogram",
    20, 200, 100
)

# ------------------------------------------------------------
# TF-IDF TRANSFORMATION
# ------------------------------------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english" if use_stopwords else None,
    max_features=max_features,
    ngram_range=ngram_range
)

X = vectorizer.fit_transform(documents)

# ------------------------------------------------------------
# 3️⃣ GENERATE DENDROGRAM
# ------------------------------------------------------------
st.sidebar.markdown("---")
generate_button = st.sidebar.button("🟦 Generate Dendrogram")

if generate_button:

    X_subset = X[:subset_size].toarray()

    linkage_matrix = linkage(X_subset, method=linkage_method)

    st.subheader("🌳 Dendrogram")

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix)
    ax.set_xlabel("Article Index")
    ax.set_ylabel("Distance")
    st.pyplot(fig)

    st.info("""
    Look for large vertical gaps in the dendrogram.
    These indicate natural separation levels between topics.
    """)

# ------------------------------------------------------------
# 4️⃣ APPLY CLUSTERING
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🟩 Apply Clustering")

n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)

apply_button = st.sidebar.button("Apply Clustering")

if apply_button:

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
        metric=distance_metric
    )

    cluster_labels = model.fit_predict(X.toarray())

    df["Cluster"] = cluster_labels

    # --------------------------------------------------------
    # 5️⃣ PCA VISUALIZATION
    # --------------------------------------------------------
    st.subheader("📊 Cluster Visualization (PCA Projection)")

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X.toarray())

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    scatter = ax2.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=cluster_labels
    )

    ax2.set_xlabel("PCA Component 1")
    ax2.set_ylabel("PCA Component 2")
    ax2.set_title("2D Projection of News Clusters")
    st.pyplot(fig2)

    # --------------------------------------------------------
    # 6️⃣ CLUSTER SUMMARY
    # --------------------------------------------------------
    st.subheader("📋 Cluster Summary")

    feature_names = vectorizer.get_feature_names_out()

    summary_data = []

    for cluster_id in range(n_clusters):

        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_vectors = X[cluster_indices]

        mean_tfidf = np.mean(cluster_vectors.toarray(), axis=0)
        top_indices = mean_tfidf.argsort()[-10:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]

        sample_article = documents.iloc[cluster_indices[0]][:200]

        summary_data.append({
            "Cluster ID": cluster_id,
            "Number of Articles": len(cluster_indices),
            "Top Keywords": ", ".join(top_keywords),
            "Sample Snippet": sample_article
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df)

    # --------------------------------------------------------
    # 7️⃣ VALIDATION SECTION
    # --------------------------------------------------------
    st.subheader("📊 Clustering Quality")

    sil_score = silhouette_score(X.toarray(), cluster_labels)

    st.metric("Silhouette Score", round(sil_score, 3))

    st.markdown("""
    *Interpretation:*
    - Close to 1 → Well-separated clusters  
    - Close to 0 → Overlapping clusters  
    - Negative → Poor clustering
    """)

    # --------------------------------------------------------
    # 8️⃣ BUSINESS INTERPRETATION
    # --------------------------------------------------------
    st.subheader("💡 Business Interpretation")

    for row in summary_data:
        st.markdown(f"""
        🟣 *Cluster {row['Cluster ID']}*  
        Articles in this group primarily discuss:  
        {row['Top Keywords']}
        """)

    # --------------------------------------------------------
    # 9️⃣ USER GUIDANCE
    # --------------------------------------------------------
    st.info("""
    Articles grouped in the same cluster share similar vocabulary and themes.
    These clusters can be used for automatic tagging, recommendations, and content organization.
    """)