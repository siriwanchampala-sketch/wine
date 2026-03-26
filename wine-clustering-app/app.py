import streamlit as st
import numpy as np
from model import prepare_model
from image_model import extract_features
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image

st.title("🍷 Wine Clustering + Image Matching")

wine_names = {
    0: "🍷 Barolo",
    1: "🍷 Grignolino",
    2: "🍷 Barbera"
}

algo = st.selectbox(
    "เลือก Algorithm",
    ["K-Means", "DBSCAN", "Hierarchical", "GMM"]
)

# โหลดโมเดล
X_scaled, labels, scaler, centroids, cluster_to_wine = prepare_model(algo)

st.write("📊 จำนวน Cluster:", len(set(labels)))

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()
ax.scatter(X_pca[:,0], X_pca[:,1], c=labels)
ax.set_title(f"{algo} Clustering")

st.pyplot(fig)

# -------------------------
# Upload Image
# -------------------------
uploaded_file = st.file_uploader("อัปโหลดรูปไวน์", type=["jpg", "png"])

# 🔥 เตือน
if algo != "K-Means":
    st.warning("Image matching ใช้ได้ดีที่สุดกับ K-Means")

# -------------------------
# วิเคราะห์ภาพ
# -------------------------
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    st.write("🔍 กำลังวิเคราะห์...")

    # extract feature
    features = extract_features(uploaded_file)

    # ลด dimension
    features = features[:, :X_scaled.shape[1]]

    # ✅ ตรวจสอบก่อนใช้ centroid
    if centroids is not None:
        distances = euclidean_distances(features, centroids)
        cluster = np.argmin(distances)
        wine_label = cluster_to_wine.get(cluster, "Unknown")
        wine_name = wine_names.get(wine_label, "Unknown")
        st.success(f"""
🍷 ผลลัพธ์:
- Cluster: {cluster}
- สายพันธุ์: {wine_name}
""")
    else:
        st.error("Algorithm นี้ไม่รองรับ image matching (ใช้ได้กับ K-Means เท่านั้น)")