from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
from collections import Counter

def prepare_model(algo):
    data = load_wine()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    centroids = None

    if algo == "K-Means":
        model = KMeans(n_clusters=3, random_state=42)
        labels = model.fit_predict(X_scaled)
        centroids = model.cluster_centers_

    elif algo == "DBSCAN":
        model = DBSCAN(eps=1.5)
        labels = model.fit_predict(X_scaled)

    elif algo == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=3)
        labels = model.fit_predict(X_scaled)

    else:
        model = GaussianMixture(n_components=3)
        labels = model.fit_predict(X_scaled)

    # mapping
    cluster_to_wine = {}

    from collections import Counter
    import numpy as np

    for c in np.unique(labels):
        if c == -1:
            continue
        idx = np.where(labels == c)
        most_common = Counter(y[idx]).most_common(1)[0][0]
        cluster_to_wine[c] = most_common

    return X_scaled, labels, scaler, centroids, cluster_to_wine