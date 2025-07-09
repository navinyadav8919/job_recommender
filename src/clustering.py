from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def apply_kmeans(X, n_clusters=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_scaled)
    return labels, model
