import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Dict, Any
from logger_config import get_logger
from .geo import coords_to_km

logger = get_logger()

def estimate_clusters(user_df: pd.DataFrame, config: Dict[str, Any], office_lat: float, office_lon: float) -> int:
    """Estimate optimal number of clusters using silhouette score with metric coordinates."""
    # Convert coordinates to km from office
    coords_km = []
    for _, user in user_df.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'], office_lat, office_lon)
        coords_km.append([lat_km, lon_km])

    coords_km = np.array(coords_km)

    max_clusters = min(10, len(user_df) // 2)
    if max_clusters < 2:
        return 1

    scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords_km)
        if len(set(cluster_labels)) > 1:
            score = silhouette_score(coords_km, cluster_labels)
            scores.append((n_clusters, score))

    if not scores:
        return 1

    best_n_clusters = max(scores, key=lambda item: item[1])[0]
    return best_n_clusters

def kmeans_clustering_metric(user_df: pd.DataFrame, n_clusters: int, office_lat: float, office_lon: float) -> List[int]:
    """Perform KMeans clustering using metric coordinates."""
    # Convert coordinates to km from office
    coords_km = []
    user_ids = []
    for _, user in user_df.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'], office_lat, office_lon)
        coords_km.append([lat_km, lon_km])
        user_ids.append(user['user_id'])

    # Sort by user_id for deterministic ordering
    sorted_data = sorted(zip(user_ids, coords_km), key=lambda x: x[0])
    coords_km = np.array([item[1] for item in sorted_data])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords_km)

    # Map labels back to original order
    label_map = {user_id: label for (user_id, _), label in zip(sorted_data, labels)}
    return [label_map[user_id] for user_id in user_df['user_id']]

def dbscan_clustering_metric(user_df: pd.DataFrame, eps_km: float, min_samples: int, office_lat: float, office_lon: float) -> List[int]:
    """Perform DBSCAN clustering using proper metric coordinates."""
    from scipy.spatial.distance import cdist

    # Convert coordinates to km from office
    coords_km = []
    for _, user in user_df.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'], office_lat, office_lon)
        coords_km.append([lat_km, lon_km])

    coords_km = np.array(coords_km)

    # Use DBSCAN with eps in km (no scaling needed now)
    dbscan = DBSCAN(eps=eps_km, min_samples=min_samples)
    labels = dbscan.fit_predict(coords_km)

    # Handle noise points: assign to nearest cluster if possible
    noise_mask = labels == -1
    if noise_mask.any():
        valid_labels = labels[~noise_mask]
        if len(valid_labels) > 0:
            # Find nearest cluster for each noise point
            for i in np.where(noise_mask)[0]:
                noise_point = coords_km[i]
                distances = cdist([noise_point], coords_km[~noise_mask])[0]
                nearest_cluster_idx = np.argmin(distances)
                labels[i] = valid_labels[nearest_cluster_idx]
        else:
            # If all points are noise, assign to a single cluster
            labels[:] = 0
    return labels.tolist()

def sweep_clustering(user_df: pd.DataFrame, config: Dict[str, Any]) -> List[int]:
    """Sweep algorithm: sort by bearing and group by capacity."""
    # Sort users by bearing from office
    sorted_df = user_df.sort_values('bearing_from_office')

    labels = []
    current_cluster = 0
    current_capacity = 0
    max_capacity = config.get('max_users_per_initial_cluster', 8)

    for idx, user in sorted_df.iterrows():
        if current_capacity >= max_capacity:
            current_cluster += 1
            current_capacity = 0

        labels.append(current_cluster)
        current_capacity += 1

    # Create label mapping back to original order
    result_labels = [-1] * len(user_df)
    for i, orig_idx in enumerate(sorted_df.index):
        result_labels[orig_idx] = labels[i]

    return result_labels