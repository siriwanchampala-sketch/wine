from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

def load_data():
    data = load_wine()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, data.feature_names

def run_kmeans(X):
    model = KMeans(n_clusters=3, random_state=42)
    return model.fit_predict(X)

def run_dbscan(X):
    model = DBSCAN(eps=1.5, min_samples=5)
    return model.fit_predict(X)

def run_hierarchical(X):
    model = AgglomerativeClustering(n_clusters=3)
    return model.fit_predict(X)

def run_gmm(X):
    model = GaussianMixture(n_components=3)
    return model.fit_predict(X)