import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="PCA & SVD Virtual Lab", page_icon="🧪", layout="wide")

st.title("🧪 Virtual Lab: PCA & SVD")
st.caption("Interactive experiment for dimensionality reduction and matrix decomposition")

with st.sidebar:
    st.header("Lab Controls")
    experiment = st.selectbox("Choose experiment", ["PCA — Dimensionality Reduction", "SVD — Matrix Decomposition"])
    data_choice = st.selectbox("Dataset", ["Iris dataset", "Upload CSV"])
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"]) if data_choice == "Upload CSV" else None
    label_column = None
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        label_options = ["(No label)"] + list(raw.columns)
        label_column = st.selectbox("Label column (optional)", label_options)
        if label_column == "(No label)":
            label_column = None
        numeric_cols = raw.select_dtypes(include=np.number).columns.tolist()
        selected_cols = st.multiselect("Numeric feature columns", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
    else:
        iris = load_iris(as_frame=True)
        raw = iris.frame.copy()
        label_column = "target"
        selected_cols = list(iris.feature_names)

if data_choice == "Upload CSV" and uploaded is not None:
    if not selected_cols:
        st.warning("Select at least one numeric feature column.")
        st.stop()
    X_df = raw[selected_cols].dropna()
    labels = raw.loc[X_df.index, label_column].astype(str) if label_column else None
    label_title = label_column
else:
    iris = load_iris(as_frame=True)
    raw = iris.frame.copy()
    X_df = raw[iris.feature_names].copy()
    labels = raw["target"].map(dict(enumerate(iris.target_names))).astype(str)
    label_title = "Species"

if X_df.shape[1] < 2:
    st.warning("Choose at least 2 numeric features to run PCA/SVD.")
    st.stop()

X = X_df.to_numpy(dtype=float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n_features = X_scaled.shape[1]
max_components = min(X_scaled.shape[0], n_features)

st.markdown("### Experiment overview")
if experiment.startswith("PCA"):
    st.write("PCA transforms correlated features into principal components ordered by captured variance. Use the controls to reduce the data to 2D or 3D and compare the projection with the original features.")
else:
    st.write("SVD factorizes a matrix as **A = U Σ Vᵀ**. The singular values indicate how much structure each component contributes. Adjust the rank to inspect a low-rank reconstruction.")

tab1, tab2, tab3 = st.tabs(["Run Experiment", "Theory & Procedure", "Observation / Conclusion"])

with tab1:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Input data")
        st.write(f"**Rows:** {len(X_df)}")
        st.write(f"**Features:** {X_df.shape[1]}")
        st.dataframe(X_df.head(8), use_container_width=True)
        if labels is not None:
            st.write("**Class labels:**", labels.nunique())
        if experiment.startswith("PCA"):
            components = st.slider("Number of principal components", 2, min(3, max_components), 2)
            pca = PCA(n_components=components)
            projected = pca.fit_transform(X_scaled)
            variance = pca.explained_variance_ratio_
            st.metric("Variance retained", f"{variance.sum()*100:.2f}%")
            st.write("Explained variance by component")
            st.dataframe(pd.DataFrame({"Component": [f"PC{i+1}" for i in range(components)],
                                     "Variance (%)": variance*100}), hide_index=True)
            proj_df = pd.DataFrame(projected, columns=[f"PC{i+1}" for i in range(components)])
            if labels is not None:
                proj_df["Label"] = labels.to_numpy()
            st.download_button("Download PCA output CSV", proj_df.to_csv(index=False), "pca_projection.csv", "text/csv")
        else:
            rank = st.slider("Reconstruction rank", 1, max_components, min(2, max_components))
            U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
            reconstructed = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
            err = np.linalg.norm(X_scaled - reconstructed, ord="fro") / np.linalg.norm(X_scaled, ord="fro")
            st.metric("Relative reconstruction error", f"{err*100:.2f}%")
            st.write("Singular values")
            st.dataframe(pd.DataFrame({"Index": np.arange(1, len(s)+1), "Singular value": s}), hide_index=True)
            recon_df = pd.DataFrame(reconstructed, columns=X_df.columns)
            st.download_button("Download reconstructed CSV", recon_df.to_csv(index=False), "svd_reconstruction.csv", "text/csv")

    with right:
        st.subheader("Visualization")
        if experiment.startswith("PCA"):
            if components == 2:
                fig = px.scatter(proj_df, x="PC1", y="PC2", color="Label" if labels is not None else None,
                                title="PCA projection — 2D (PC1 vs PC2)")
            else:
                fig = px.scatter_3d(proj_df, x="PC1", y="PC2", z="PC3",
                                   color="Label" if labels is not None else None,
                                   title="PCA projection — 3D (PC1, PC2, PC3)")
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
            st.info("Each point is one sample. Nearby points have similar feature patterns after projection.")
        else:
            # Plot first two original feature columns, then compare original and rank-k reconstruction.
            plot_cols = list(X_df.columns[:2])
            plot_df = pd.DataFrame({"Feature 1": X_scaled[:, 0], "Feature 2": X_scaled[:, 1],
                                   "Type": "Original"})
            rec_df = pd.DataFrame({"Feature 1": reconstructed[:, 0], "Feature 2": reconstructed[:, 1],
                                   "Type": f"Rank-{rank} reconstruction"})
            compare = pd.concat([plot_df, rec_df], ignore_index=True)
            fig = px.scatter(compare, x="Feature 1", y="Feature 2", color="Type",
                             title="Original standardized data vs SVD reconstruction")
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("This 2D view uses the first two selected features. SVD reconstruction is computed using the chosen rank.")

with tab2:
    st.subheader("Theory")
    st.markdown("""
**PCA (Principal Component Analysis)**  
PCA reduces the number of features while preserving as much variance as possible. It first standardizes features, then finds orthogonal directions called principal components.

**SVD (Singular Value Decomposition)**  
For a matrix A, SVD produces three matrices:
- **U**: left singular vectors
- **Σ**: diagonal matrix of singular values
- **Vᵀ**: right singular vectors transposed

The rank-k approximation keeps the k largest singular values and reconstructs an approximate matrix.
""")
    st.subheader("Procedure")
    st.markdown("""
1. Load the Iris dataset or upload a CSV file.
2. Select numeric features and optionally a label column.
3. Run PCA or SVD from the sidebar.
4. Change the component count / reconstruction rank.
5. Observe the graph, variance retained, or reconstruction error.
6. Download the generated output CSV.
""")

with tab3:
    st.subheader("Observation")
    if experiment.startswith("PCA"):
        st.write("As the number of principal components increases, the percentage of explained variance generally increases. A 2D projection is easier to visualize, but it may lose some information compared with the original feature space.")
    else:
        st.write("As the reconstruction rank increases, the reconstructed matrix becomes closer to the original matrix and the relative reconstruction error generally decreases.")
    st.subheader("Conclusion")
    st.write("PCA is useful for visualizing and reducing high-dimensional data. SVD provides a matrix factorization and a practical way to create low-rank approximations. The best component count or rank depends on the dataset and the required balance between simplicity and information preservation.")
    st.caption("For a formal lab submission, record the dataset, selected features, component/rank setting, variance retained or reconstruction error, and screenshots of the resulting graph.")
