import os
import math
import requests
import numpy as np
import pandas as pd
import time
import json
from functools import lru_cache
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
from dotenv import load_dotenv
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from logger_config import get_logger
from progress_tracker import ProgressTracker

warnings.filterwarnings('ignore')

# Setup logging first
logger = get_logger()
progress = ProgressTracker()

# Import road_network module for route coherence scoring
try:
    import road_network as road_network_module
    # Create an instance of RoadNetwork class if it exists
    try:
        road_network = road_network_module.RoadNetwork('tricity_main_roads.graphml')
        logger.info("Successfully loaded RoadNetwork with GraphML data")
    except Exception as e:
        logger.warning(f"Could not create RoadNetwork instance: {e}. Using mock implementation.")

        class MockRoadNetwork:
            def get_route_coherence_score(self, driver_pos, user_positions, office_pos):
                if not user_positions:
                    return 1.0
                avg_dist_from_driver = sum(
                    haversine_distance(driver_pos[0], driver_pos[1], u[0], u[1])
                    for u in user_positions) / len(user_positions)
                avg_dist_from_office = sum(
                    haversine_distance(office_pos[0], office_pos[1], u[0], u[1])
                    for u in user_positions) / len(user_positions)

                score = max(0, 1.0 - (avg_dist_from_driver / 50.0) - (avg_dist_from_office / 100.0))
                return min(1.0, score)

            def is_user_on_route_path(self, driver_pos, current_user_positions, user_pos, office_pos,
                                    max_detour_ratio=1.3, route_type="balanced"):
                return True

            def get_road_distance(self, lat1, lon1, lat2, lon2):
                return haversine_distance(lat1, lon1, lat2, lon2)

            def find_nearest_road_node(self, lat, lon):
                return None, None

            def simplify_path_nodes(self, path, max_nodes=10):
                return path

        road_network = MockRoadNetwork()
except ImportError:
    logger.warning("road_network module not found. Road network features will be limited.")

    class MockRoadNetwork:
        def get_route_coherence_score(self, driver_pos, user_positions, office_pos):
            if not user_positions:
                return 1.0
            avg_dist_from_driver = sum(
                haversine_distance(driver_pos[0], driver_pos[1], u[0], u[1])
                for u in user_positions) / len(user_positions)
            avg_dist_from_office = sum(
                haversine_distance(office_pos[0], office_pos[1], u[0], u[1])
                for u in user_positions) / len(user_positions)

            score = max(0, 1.0 - (avg_dist_from_driver / 50.0) - (avg_dist_from_office / 100.0))
            return min(1.0, score)

        def is_user_on_route_path(self, driver_pos, current_user_positions, user_pos, office_pos,
                                max_detour_ratio=1.3, route_type="balanced"):
            return True

        def get_road_distance(self, lat1, lon1, lat2, lon2):
            return haversine_distance(lat1, lon1, lat2, lon2)

        def find_nearest_road_node(self, lat, lon):
            return None, None

        def simplify_path_nodes(self, path, max_nodes=10):
            return path

    road_network = MockRoadNetwork()

# Load and validate configuration
def load_and_validate_config():
    """Load configuration for locality-first balanced optimization"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    current_mode = "locality_first_balanced"
    mode_configs = cfg.get("mode_configs", {})
    mode_config = mode_configs.get("locality_first_balanced", {})

    logger.info(f"🏘️ Using optimization mode: LOCALITY-FIRST BALANCED")

    config = {}

    # Locality clustering parameters (Stage 1)
    config['LOCALITY_EPS_KM'] = max(0.5, float(mode_config.get("locality_eps_km", cfg.get("locality_eps_km", 2.0))))
    config['LOCALITY_MIN_SAMPLES'] = max(1, int(mode_config.get("locality_min_samples", cfg.get("locality_min_samples", 2))))

    # Directional clustering parameters (Stage 2)
    config['ANGULAR_SECTORS'] = max(6, int(mode_config.get("angular_sectors", cfg.get("angular_sectors", 8))))
    config['TURNING_THRESHOLD_DEGREES'] = max(15, float(mode_config.get("turning_threshold", cfg.get("turning_threshold", 45))))
    config['TORTUOSITY_THRESHOLD'] = max(1.2, float(mode_config.get("tortuosity_threshold", cfg.get("tortuosity_threshold", 1.6))))

    # Capacity-aware clustering parameters (Stage 3)
    config['MAX_USERS_PER_CLUSTER'] = max(4, int(mode_config.get("max_users_per_cluster", cfg.get("max_users_per_cluster", 8))))
    config['CAPACITY_SLACK_FACTOR'] = max(0.7, min(1.0, float(mode_config.get("capacity_slack", cfg.get("capacity_slack", 0.85)))))

    # Assignment cost weights (Stage 4)
    config['ALPHA_DISTANCE'] = max(0.1, float(mode_config.get("alpha_distance", cfg.get("alpha_distance", 1.0))))
    config['BETA_TURNING'] = max(0.005, float(mode_config.get("beta_turning", cfg.get("beta_turning", 0.02))))
    config['GAMMA_UTILIZATION'] = max(1.0, float(mode_config.get("gamma_utilization", cfg.get("gamma_utilization", 25.0))))
    config['DELTA_PRIORITY'] = max(0.0, float(mode_config.get("delta_priority", cfg.get("delta_priority", 0.1))))

    # Seat filling parameters (Stage 6)
    config['MAX_FILL_DISTANCE_KM'] = max(0.5, float(mode_config.get("max_fill_distance", cfg.get("max_fill_distance_km", 5.0))))
    config['MAX_DETOUR_RATIO'] = max(1.05, float(mode_config.get("max_detour_ratio", cfg.get("max_detour_ratio", 1.25))))
    config['COHERENCE_TOLERANCE'] = max(0.01, float(mode_config.get("coherence_tolerance", cfg.get("coherence_tolerance", 0.05))))

    # Global optimization parameters (Stage 7)
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(0.1, float(cfg.get("swap_improvement_threshold_km", 0.75)))
    config['MERGE_DISTANCE_KM'] = max(0.5, float(cfg.get("merge_distance_km", 3.5)))
    config['MERGE_BEARING_THRESHOLD'] = max(10, float(cfg.get("merge_bearing_threshold", 30)))

    # Office coordinates
    office_lat = float(os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489)))
    office_lon = float(os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))

    if not (-90 <= office_lat <= 90):
        logger.warning(f"Invalid office latitude {office_lat}, using default")
        office_lat = 30.6810489
    if not (-180 <= office_lon <= 180):
        logger.warning(f"Invalid office longitude {office_lon}, using default")
        office_lon = 76.7260711

    config['OFFICE_LAT'] = office_lat
    config['OFFICE_LON'] = office_lon

    # Latitude conversion factors
    config['LAT_TO_KM'] = 111.0
    config['LON_TO_KM'] = 111.0 * math.cos(math.radians(office_lat))

    logger.info(f"   🏘️ Locality EPS: {config['LOCALITY_EPS_KM']}km")
    logger.info(f"   📐 Angular sectors: {config['ANGULAR_SECTORS']}")
    logger.info(f"   🔄 Turning threshold: {config['TURNING_THRESHOLD_DEGREES']}°")
    logger.info(f"   👥 Max users per cluster: {config['MAX_USERS_PER_CLUSTER']}")

    return config

# Import core functions from assignment.py
from assignment import (
    validate_input_data, load_env_and_fetch_data, extract_office_coordinates,
    prepare_user_driver_dataframes, haversine_distance, calculate_bearing,
    bearing_difference, normalize_bearing_difference, coords_to_km,
    _get_all_drivers_as_unassigned, _convert_users_to_unassigned_format,
    get_progress_tracker
)

# Load configuration
_config = load_and_validate_config()

# Global constants from config
LOCALITY_EPS_KM = _config['LOCALITY_EPS_KM']
LOCALITY_MIN_SAMPLES = _config['LOCALITY_MIN_SAMPLES']
ANGULAR_SECTORS = _config['ANGULAR_SECTORS']
TURNING_THRESHOLD_DEGREES = _config['TURNING_THRESHOLD_DEGREES']
TORTUOSITY_THRESHOLD = _config['TORTUOSITY_THRESHOLD']
MAX_USERS_PER_CLUSTER = _config['MAX_USERS_PER_CLUSTER']
CAPACITY_SLACK_FACTOR = _config['CAPACITY_SLACK_FACTOR']
ALPHA_DISTANCE = _config['ALPHA_DISTANCE']
BETA_TURNING = _config['BETA_TURNING']
GAMMA_UTILIZATION = _config['GAMMA_UTILIZATION']
DELTA_PRIORITY = _config['DELTA_PRIORITY']
MAX_FILL_DISTANCE_KM = _config['MAX_FILL_DISTANCE_KM']
MAX_DETOUR_RATIO = _config['MAX_DETOUR_RATIO']
COHERENCE_TOLERANCE = _config['COHERENCE_TOLERANCE']
SWAP_IMPROVEMENT_THRESHOLD = _config['SWAP_IMPROVEMENT_THRESHOLD']
MERGE_DISTANCE_KM = _config['MERGE_DISTANCE_KM']
MERGE_BEARING_THRESHOLD = _config['MERGE_BEARING_THRESHOLD']
OFFICE_LAT = _config['OFFICE_LAT']
OFFICE_LON = _config['OFFICE_LON']

# ================== STAGE 1: LOCALITY CLUSTERING ==================

def derive_user_features(user_df, office_lat, office_lon):
    """Stage 1: Derive canonical user features from user data"""
    logger.info("🏗️ Stage 1: Deriving canonical user features...")

    user_df = user_df.copy()

    # Core distance and bearing features
    user_df['office_distance'] = user_df.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude'], office_lat, office_lon), 
        axis=1
    )

    user_df['bearing_to_office'] = user_df.apply(
        lambda row: calculate_bearing(row['latitude'], row['longitude'], office_lat, office_lon),
        axis=1
    )

    # Projection features for on-the-way ordering
    office_pos = np.array([office_lat, office_lon])
    user_positions = user_df[['latitude', 'longitude']].values

    # Calculate projection along main axis (office as reference)
    office_vectors = user_positions - office_pos
    office_distances = np.linalg.norm(office_vectors, axis=1)

    # Normalize vectors for projection calculation
    normalized_vectors = office_vectors / (office_distances.reshape(-1, 1) + 1e-10)

    # Use office direction as main axis
    main_axis = np.array([1.0, 0.0])  # East direction as default

    # Calculate projections
    projections = np.dot(normalized_vectors, main_axis) * office_distances
    perpendicular_distances = np.abs(np.cross(normalized_vectors, main_axis)) * office_distances

    user_df['projection_along_main_axis'] = projections
    user_df['perpendicular_distance'] = perpendicular_distances

    logger.info(f"✅ Derived features for {len(user_df)} users")
    return user_df

def create_locality_clusters(user_df):
    """Stage 1: Cluster users by geographic locality using DBSCAN"""
    logger.info("🏘️ Stage 1: Creating locality clusters...")

    if len(user_df) < 2:
        user_df['locality_cluster'] = 0
        return user_df

    # Use lat/lon coordinates for clustering
    coords = user_df[['latitude', 'longitude']].values

    # Convert to km for distance calculation
    coords_km = []
    for coord in coords:
        lat_km, lon_km = coords_to_km(coord[0], coord[1], OFFICE_LAT, OFFICE_LON)
        coords_km.append([lat_km, lon_km])
    coords_km = np.array(coords_km)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=LOCALITY_EPS_KM, min_samples=LOCALITY_MIN_SAMPLES)
    clusters = dbscan.fit_predict(coords_km)

    # Ensure close users are grouped together
    clusters = ensure_close_users_grouped(coords_km, clusters, max_distance_km=LOCALITY_EPS_KM / 2) # Use half of eps for closer grouping

    user_df['locality_cluster'] = clusters

    # Handle noise points (cluster = -1) by assigning them to nearest cluster or creating singleton clusters
    noise_mask = clusters == -1
    if noise_mask.any():
        noise_indices = np.where(noise_mask)[0]
        valid_clusters = clusters[~noise_mask]

        if len(valid_clusters) > 0:
            # Build KDTree for fast nearest neighbor search
            valid_coords = coords_km[~noise_mask]
            tree = KDTree(valid_coords)

            next_cluster_id = max(valid_clusters) + 1

            for noise_idx in noise_indices:
                noise_coord = coords_km[noise_idx].reshape(1, -1)
                distances, indices = tree.query(noise_coord, k=1)

                if distances[0][0] <= LOCALITY_EPS_KM * 1.5:  # Allow slightly larger distance for noise
                    # Assign to nearest cluster
                    nearest_cluster = valid_clusters[indices[0][0]]
                    user_df.iloc[noise_idx, user_df.columns.get_loc('locality_cluster')] = nearest_cluster
                else:
                    # Create new singleton cluster
                    user_df.iloc[noise_idx, user_df.columns.get_loc('locality_cluster')] = next_cluster_id
                    next_cluster_id += 1
        else:
            # All points are noise, assign sequential cluster IDs
            user_df.loc[noise_mask, 'locality_cluster'] = range(len(noise_indices))

    n_clusters = len(set(user_df['locality_cluster']))
    logger.info(f"🏘️ Created {n_clusters} locality clusters")

    # Log cluster sizes
    cluster_sizes = user_df['locality_cluster'].value_counts().sort_index()
    for cluster_id, size in cluster_sizes.items():
        logger.info(f"   Locality {cluster_id}: {size} users")

    return user_df

def ensure_close_users_grouped(coords_km, clusters, max_distance_km=1.0):
    """Ensure users within 1km of each other are in the same cluster"""
    logger.info(f"🔍 Ensuring users within {max_distance_km}km are grouped together...")

    # Build distance matrix for all points
    from scipy.spatial.distance import cdist
    distances = cdist(coords_km, coords_km, metric='euclidean')

    # Create groups of users that should be in same cluster
    close_groups = []
    processed = set()

    for i in range(len(coords_km)):
        if i in processed:
            continue

        # Find all users within max_distance_km of user i
        close_users = []
        for j in range(len(coords_km)):
            if distances[i][j] <= max_distance_km:
                close_users.append(j)
                processed.add(j)

        if len(close_users) > 1:
            close_groups.append(close_users)

    # Merge close users into same clusters
    modified_clusters = clusters.copy()
    next_cluster_id = max(clusters) + 1 if len(clusters) > 0 and max(clusters) >= 0 else 0

    for group in close_groups:
        # Assign all users in this group to the same cluster
        for user_idx in group:
            modified_clusters[user_idx] = next_cluster_id
        next_cluster_id += 1

        logger.info(f"   🤝 Grouped {len(group)} users within {max_distance_km}km into cluster {next_cluster_id-1}")

    return modified_clusters

# ================== STAGE 2: DIRECTIONAL SPLITTING ==================

def split_locality_by_direction(user_group, locality_id):
    """Stage 2: Split locality cluster by travel direction"""
    logger.info(f"📐 Stage 2: Directional splitting for locality {locality_id}")

    if len(user_group) <= 1:
        user_group['direction_cluster'] = 0
        return user_group

    # A. Angular sectoring approach
    direction_clusters = apply_angular_sectoring(user_group, locality_id)

    # B. Path projection + turning detection (for refinement)
    direction_clusters = refine_with_turning_detection(direction_clusters, locality_id)

    return direction_clusters

def apply_angular_sectoring(user_group, locality_id):
    """Angular sectoring approach for directional clustering"""
    user_group = user_group.copy()

    # Calculate cluster center bearing
    center_lat = user_group['latitude'].median()
    center_lon = user_group['longitude'].median()
    cluster_center_bearing = calculate_bearing(center_lat, center_lon, OFFICE_LAT, OFFICE_LON)

    # Calculate relative bearings
    relative_bearings = user_group['bearing_to_office'] - cluster_center_bearing
    relative_bearings = relative_bearings.apply(normalize_bearing_difference)

    # Create angular sectors
    sector_size = 360 / ANGULAR_SECTORS
    sector_ids = ((relative_bearings + 180) / sector_size).astype(int) % ANGULAR_SECTORS

    user_group['direction_cluster'] = sector_ids

    logger.info(f"   🎯 Locality {locality_id}: Split into {len(set(sector_ids))} angular sectors")
    return user_group

def refine_with_turning_detection(user_group, locality_id):
    """Refine directional clusters using turning detection"""
    refined_groups = []

    for direction_id in user_group['direction_cluster'].unique():
        direction_group = user_group[user_group['direction_cluster'] == direction_id].copy()

        if len(direction_group) <= 2:
            refined_groups.append(direction_group)
            continue

        # Sort by projection along main axis
        direction_group = direction_group.sort_values('projection_along_main_axis')

        # Detect turning points
        split_points = detect_turning_points(direction_group)

        # Split group at turning points
        if split_points:
            current_cluster_id = direction_id
            start_idx = 0

            for split_idx in split_points + [len(direction_group)]:
                subgroup = direction_group.iloc[start_idx:split_idx].copy()
                subgroup['direction_cluster'] = current_cluster_id
                refined_groups.append(subgroup)
                current_cluster_id += 100  # Offset to avoid conflicts
                start_idx = split_idx
        else:
            refined_groups.append(direction_group)

    result = pd.concat(refined_groups, ignore_index=True)

    n_direction_clusters = len(set(result['direction_cluster']))
    logger.info(f"   🔄 Locality {locality_id}: Refined to {n_direction_clusters} direction clusters")

    return result

def detect_turning_points(sorted_group):
    """Detect points where route would require significant turning"""
    if len(sorted_group) <= 2:
        return []

    turning_points = []
    positions = sorted_group[['latitude', 'longitude']].values

    for i in range(1, len(positions) - 1):
        # Calculate turning angle at point i
        prev_bearing = calculate_bearing(positions[i-1][0], positions[i-1][1], 
                                       positions[i][0], positions[i][1])
        next_bearing = calculate_bearing(positions[i][0], positions[i][1],
                                       positions[i+1][0], positions[i+1][1])

        turning_angle = abs(bearing_difference(prev_bearing, next_bearing))

        if turning_angle > TURNING_THRESHOLD_DEGREES:
            turning_points.append(i)

    # Check tortuosity for the whole sequence
    if len(positions) > 3:
        euclidean_distance = haversine_distance(positions[0][0], positions[0][1],
                                              positions[-1][0], positions[-1][1])

        total_route_distance = sum(
            haversine_distance(positions[i][0], positions[i][1], 
                             positions[i+1][0], positions[i+1][1])
            for i in range(len(positions) - 1)
        )

        tortuosity = total_route_distance / (euclidean_distance + 0.001)

        if tortuosity > TORTUOSITY_THRESHOLD:
            # Split at the middle point
            turning_points.append(len(positions) // 2)

    return sorted(set(turning_points))

# ================== STAGE 3: CAPACITY-AWARE SUBCLUSTERING ==================

def create_capacity_aware_subclusters(direction_group, available_capacities):
    """Stage 3: Create capacity-aware subclusters within direction groups"""
    if len(direction_group) <= 1:
        direction_group['capacity_cluster'] = 0
        return direction_group

    # Determine target cluster sizes based on available vehicle capacities
    max_capacity = max(available_capacities) if available_capacities else MAX_USERS_PER_CLUSTER
    target_size = min(int(max_capacity * CAPACITY_SLACK_FACTOR), MAX_USERS_PER_CLUSTER)

    # Sort by projection for contiguous grouping
    sorted_group = direction_group.sort_values('projection_along_main_axis').copy()

    # Use sweep algorithm to create contiguous capacity-aware clusters
    clusters = []
    current_cluster = []
    cluster_id = 0

    for idx, (_, user) in enumerate(sorted_group.iterrows()):
        current_cluster.append(user)

        # Check if we should close current cluster
        should_close = (
            len(current_cluster) >= target_size or  # Reached target size
            idx == len(sorted_group) - 1  # Last user
        )

        if should_close:
            cluster_df = pd.DataFrame(current_cluster)
            cluster_df['capacity_cluster'] = cluster_id
            clusters.append(cluster_df)

            current_cluster = []
            cluster_id += 1

    result = pd.concat(clusters, ignore_index=True) if clusters else direction_group.copy()

    if 'capacity_cluster' not in result.columns:
        result['capacity_cluster'] = 0

    return result

# ================== STAGE 4: DRIVER ASSIGNMENT ==================

def assign_best_driver_to_cluster(cluster_users, available_drivers, used_driver_ids, office_lat, office_lon):
    """Stage 4: Assign best driver to capacity cluster using balanced cost function"""
    cluster_size = len(cluster_users)

    if cluster_size == 0:
        return None

    best_driver = None
    best_cost = float('inf')
    best_sequence = None

    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids or driver['capacity'] < cluster_size:
            continue

        # Calculate balanced cost components
        route_distance, sequence, turning_penalty = calculate_route_metrics(
            driver, cluster_users, office_lat, office_lon)

        utilization = cluster_size / driver['capacity']
        utilization_penalty = (1 - utilization) * GAMMA_UTILIZATION

        priority_penalty = driver['priority'] * DELTA_PRIORITY

        # Balanced cost function
        total_cost = (ALPHA_DISTANCE * route_distance + 
                     BETA_TURNING * turning_penalty + 
                     utilization_penalty + 
                     priority_penalty)

        if total_cost < best_cost:
            best_cost = total_cost
            best_driver = driver
            best_sequence = sequence

    if best_driver is not None:
        used_driver_ids.add(best_driver['driver_id'])

        route = create_route_from_assignment(best_driver, best_sequence, office_lat, office_lon)

        utilization_pct = (cluster_size / best_driver['capacity']) * 100
        logger.info(f"   🚗 Assigned driver {best_driver['driver_id']}: "
                   f"{cluster_size}/{best_driver['capacity']} seats ({utilization_pct:.1f}%)")

        return route

    return None

def calculate_route_metrics(driver, cluster_users, office_lat, office_lon):
    """Calculate route distance, sequence, and turning penalty"""
    if len(cluster_users) == 0:
        return float('inf'), [], 0

    driver_pos = (driver['latitude'], driver['longitude'])
    office_pos = (office_lat, office_lon)

    # Get optimal sequence using projection-based ordering with 2-opt
    sequence = get_optimal_sequence(driver_pos, cluster_users, office_pos)

    # Calculate total route distance
    total_distance = 0
    turning_angles = []

    # Driver to first pickup
    if sequence:
        if hasattr(sequence[0], 'latitude'):
            first_user = sequence[0]
        else:
            first_user = sequence[0][1] if isinstance(sequence[0], tuple) else sequence[0]

        total_distance += haversine_distance(driver_pos[0], driver_pos[1],
                                           first_user['latitude'], first_user['longitude'])

    # Between pickups
    for i in range(len(sequence) - 1):
        current_user = sequence[i][1] if isinstance(sequence[i], tuple) else sequence[i]
        next_user = sequence[i + 1][1] if isinstance(sequence[i + 1], tuple) else sequence[i + 1]

        distance = haversine_distance(current_user['latitude'], current_user['longitude'],
                                    next_user['latitude'], next_user['longitude'])
        total_distance += distance

        # Calculate turning angle
        if i == 0:
            prev_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                           current_user['latitude'], current_user['longitude'])
        else:
            prev_pos_user = sequence[i - 1][1] if isinstance(sequence[i - 1], tuple) else sequence[i - 1]
            prev_bearing = calculate_bearing(prev_pos_user['latitude'], prev_pos_user['longitude'],
                                           current_user['latitude'], current_user['longitude'])

        next_bearing = calculate_bearing(current_user['latitude'], current_user['longitude'],
                                       next_user['latitude'], next_user['longitude'])

        turning_angle = abs(bearing_difference(prev_bearing, next_bearing))
        turning_angles.append(turning_angle)

    # Last pickup to office
    if sequence:
        last_user = sequence[-1][1] if isinstance(sequence[-1], tuple) else sequence[-1]
        total_distance += haversine_distance(last_user['latitude'], last_user['longitude'],
                                           office_lat, office_lon)

    # Calculate mean turning penalty
    turning_penalty = sum(turning_angles) / len(turning_angles) if turning_angles else 0

    return total_distance, sequence, turning_penalty

def get_optimal_sequence(driver_pos, cluster_users, office_pos):
    """Get optimal pickup sequence using projection-based ordering and 2-opt"""
    if len(cluster_users) <= 1:
        return cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)

    users_list = cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)

    # Initial ordering by projection along main axis
    main_axis_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    def projection_score(user):
        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1], user['latitude'], user['longitude'])
        bearing_alignment = math.cos(math.radians(abs(bearing_difference(user_bearing, main_axis_bearing))))
        distance = haversine_distance(driver_pos[0], driver_pos[1], user['latitude'], user['longitude'])
        return distance * bearing_alignment  # Closer points in the right direction get lower scores

    users_list.sort(key=projection_score)

    # Apply direction-aware 2-opt optimization
    return apply_direction_aware_2opt(users_list, driver_pos, office_pos)

def apply_direction_aware_2opt(sequence, driver_pos, office_pos):
    """Apply 2-opt with turning penalty"""
    if len(sequence) <= 2:
        return sequence

    improved = True
    max_iterations = 3
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        best_distance = calculate_total_distance(sequence, driver_pos, office_pos)
        best_turning = calculate_total_turning(sequence, driver_pos, office_pos)

        for i in range(len(sequence) - 1):
            for j in range(i + 2, len(sequence)):
                # Try 2-opt swap
                new_sequence = sequence[:i + 1] + sequence[i + 1:j + 1][::-1] + sequence[j + 1:]

                new_distance = calculate_total_distance(new_sequence, driver_pos, office_pos)
                new_turning = calculate_total_turning(new_sequence, driver_pos, office_pos)

                # Combined objective: distance + turning penalty
                old_objective = best_distance + BETA_TURNING * best_turning * 50  # Scale turning to distance units
                new_objective = new_distance + BETA_TURNING * new_turning * 50

                if new_objective < old_objective - 0.1:  # Improvement threshold
                    sequence = new_sequence
                    best_distance = new_distance
                    best_turning = new_turning
                    improved = True
                    break
            if improved:
                break

    return sequence

def calculate_total_distance(sequence, driver_pos, office_pos):
    """Calculate total route distance"""
    if not sequence:
        return 0

    total = haversine_distance(driver_pos[0], driver_pos[1], 
                              sequence[0]['latitude'], sequence[0]['longitude'])

    for i in range(len(sequence) - 1):
        total += haversine_distance(sequence[i]['latitude'], sequence[i]['longitude'],
                                   sequence[i + 1]['latitude'], sequence[i + 1]['longitude'])

    total += haversine_distance(sequence[-1]['latitude'], sequence[-1]['longitude'],
                               office_pos[0], office_pos[1])

    return total

def calculate_total_turning(sequence, driver_pos, office_pos):
    """Calculate total turning penalty"""
    if len(sequence) <= 1:
        return 0

    turning_angles = []

    for i in range(len(sequence) - 1):
        if i == 0:
            prev_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                           sequence[i]['latitude'], sequence[i]['longitude'])
        else:
            prev_bearing = calculate_bearing(sequence[i-1]['latitude'], sequence[i-1]['longitude'],
                                           sequence[i]['latitude'], sequence[i]['longitude'])

        next_bearing = calculate_bearing(sequence[i]['latitude'], sequence[i]['longitude'],
                                       sequence[i + 1]['latitude'], sequence[i + 1]['longitude'])

        turning_angle = abs(bearing_difference(prev_bearing, next_bearing))
        turning_angles.append(turning_angle)

    return sum(turning_angles) / len(turning_angles) if turning_angles else 0

def create_route_from_assignment(driver, sequence, office_lat, office_lon):
    """Create route object from driver and user sequence"""
    route = {
        'driver_id': str(driver['driver_id']),
        'vehicle_id': str(driver.get('vehicle_id', '')),
        'vehicle_type': int(driver['capacity']),
        'latitude': float(driver['latitude']),
        'longitude': float(driver['longitude']),
        'assigned_users': []
    }

    # Add users in sequence order
    for user in sequence:
        if isinstance(user, tuple):
            user = user[1]  # Extract user data if it's a tuple

        user_data = {
            'user_id': str(user['user_id']),
            'lat': float(user['latitude']),
            'lng': float(user['longitude']),
            'office_distance': float(user.get('office_distance', 0))
        }

        if pd.notna(user.get('first_name')):
            user_data['first_name'] = str(user['first_name'])
        if pd.notna(user.get('email')):
            user_data['email'] = str(user['email'])

        route['assigned_users'].append(user_data)

    return route

# ================== STAGE 5: SEQUENCE OPTIMIZATION ==================
# (Already handled in get_optimal_sequence with 2-opt)

# ================== STAGE 6: SEAT FILLING ==================

def path_aware_seat_filling(routes, unassigned_users_df, office_lat, office_lon):
    """Stage 6: Fill remaining seats with path-aware constraints"""
    logger.info("🪑 Stage 6: Path-aware seat filling...")

    if unassigned_users_df.empty:
        return routes, set()

    filled_user_ids = set()

    for route in routes:
        if len(route['assigned_users']) >= route['vehicle_type']:
            continue

        available_seats = route['vehicle_type'] - len(route['assigned_users'])
        route_center = calculate_route_center(route)

        # Get current route coherence
        current_coherence = calculate_route_coherence(route, office_lat, office_lon)

        # Find candidate users near the route
        candidate_users = []

        for _, user in unassigned_users_df.iterrows():
            if user['user_id'] in filled_user_ids:
                continue

            distance_to_route = haversine_distance(route_center[0], route_center[1],
                                                 user['latitude'], user['longitude'])

            if distance_to_route <= MAX_FILL_DISTANCE_KM:
                # Calculate insertion cost and coherence impact
                insertion_cost, new_coherence = calculate_insertion_impact(
                    route, user, office_lat, office_lon, current_coherence)

                # Check constraints
                coherence_ok = (new_coherence >= current_coherence - COHERENCE_TOLERANCE)
                detour_ok = (insertion_cost <= MAX_DETOUR_RATIO)

                if coherence_ok and detour_ok:
                    score = (0.6 * insertion_cost + 
                            0.3 * max(0, current_coherence - new_coherence) +
                            0.1 * distance_to_route)
                    candidate_users.append((score, user))

        # Fill seats with best candidates
        candidate_users.sort(key=lambda x: x[0])  # Lower score is better

        users_to_add = candidate_users[:available_seats]

        for score, user in users_to_add:
            user_data = {
                'user_id': str(user['user_id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': float(user.get('office_distance', 0))
            }

            if pd.notna(user.get('first_name')):
                user_data['first_name'] = str(user['first_name'])
            if pd.notna(user.get('email')):
                user_data['email'] = str(user['email'])

            route['assigned_users'].append(user_data)
            filled_user_ids.add(user['user_id'])

        if users_to_add:
            # Re-optimize sequence after adding users
            route = reoptimize_route_sequence(route, office_lat, office_lon)

            utilization = len(route['assigned_users']) / route['vehicle_type'] * 100
            logger.info(f"   🪑 Added {len(users_to_add)} users to route {route['driver_id']} "
                       f"({utilization:.1f}% utilization)")

    return routes, filled_user_ids

def calculate_route_center(route):
    """Calculate the center point of a route"""
    if not route['assigned_users']:
        return [route['latitude'], route['longitude']]

    lats = [user['lat'] for user in route['assigned_users']]
    lngs = [user['lng'] for user in route['assigned_users']]

    return [sum(lats) / len(lats), sum(lngs) / len(lngs)]

def calculate_route_coherence(route, office_lat, office_lon):
    """Calculate route coherence score"""
    if not route['assigned_users']:
        return 1.0

    driver_pos = (route['latitude'], route['longitude'])
    user_positions = [(user['lat'], user['lng']) for user in route['assigned_users']]
    office_pos = (office_lat, office_lon)

    return road_network.get_route_coherence_score(driver_pos, user_positions, office_pos)

def calculate_insertion_impact(route, new_user, office_lat, office_lon, current_coherence):
    """Calculate the impact of inserting a new user into the route"""
    # Create test route with new user
    test_route = route.copy()
    new_user_data = {
        'user_id': str(new_user['user_id']),
        'lat': float(new_user['latitude']),
        'lng': float(new_user['longitude']),
        'office_distance': float(new_user.get('office_distance', 0))
    }
    test_route['assigned_users'] = route['assigned_users'] + [new_user_data]

    # Calculate original and new distances
    original_distance = calculate_total_route_distance(route, office_lat, office_lon)
    new_distance = calculate_total_route_distance(test_route, office_lat, office_lon)

    insertion_cost = new_distance / (original_distance + 0.001)  # Detour ratio

    # Calculate new coherence
    new_coherence = calculate_route_coherence(test_route, office_lat, office_lon)

    return insertion_cost, new_coherence

def calculate_total_route_distance(route, office_lat, office_lon):
    """Calculate total distance for a route"""
    if not route['assigned_users']:
        return 0

    total = 0
    current_pos = (route['latitude'], route['longitude'])

    for user in route['assigned_users']:
        user_pos = (user['lat'], user['lng'])
        total += haversine_distance(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
        current_pos = user_pos

    # To office
    total += haversine_distance(current_pos[0], current_pos[1], office_lat, office_lon)

    return total

def reoptimize_route_sequence(route, office_lat, office_lon):
    """Re-optimize route sequence after adding users"""
    if len(route['assigned_users']) <= 1:
        return route

    # Convert to user list
    users = []
    for user_data in route['assigned_users']:
        user = {
            'user_id': user_data['user_id'],
            'latitude': user_data['lat'],
            'longitude': user_data['lng'],
            'office_distance': user_data.get('office_distance', 0)
        }
        if 'first_name' in user_data:
            user['first_name'] = user_data['first_name']
        if 'email' in user_data:
            user['email'] = user_data['email']
        users.append(user)

    # Get optimized sequence
    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (office_lat, office_lon)
    optimized_sequence = get_optimal_sequence(driver_pos, users, office_pos)

    # Update route with optimized sequence
    route['assigned_users'] = []
    for user in optimized_sequence:
        user_data = {
            'user_id': str(user['user_id']),
            'lat': float(user['latitude']),
            'lng': float(user['longitude']),
            'office_distance': float(user.get('office_distance', 0))
        }
        if 'first_name' in user:
            user_data['first_name'] = str(user['first_name'])
        if 'email' in user:
            user_data['email'] = str(user['email'])

        route['assigned_users'].append(user_data)

    return route

# ================== STAGE 7: GLOBAL OPTIMIZATION ==================

def local_swaps_and_merges(routes, office_lat, office_lon):
    """Stage 7: Perform local swaps and merges for global optimization"""
    logger.info("🔄 Stage 7: Local swaps and global optimization...")

    # Try local swaps first
    routes = perform_local_swaps(routes, office_lat, office_lon)

    # Try route merges
    routes = perform_route_merges(routes, office_lat, office_lon)

    # Split tortuous routes
    routes = split_tortuous_routes(routes, office_lat, office_lon)

    return routes

def perform_local_swaps(routes, office_lat, office_lon):
    """Perform beneficial user swaps between routes"""
    if len(routes) < 2:
        return routes

    swaps_made = 0

    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            route1 = routes[i]
            route2 = routes[j]

            if not route1['assigned_users'] or not route2['assigned_users']:
                continue

            # Try swapping users between routes
            best_swap = find_best_user_swap(route1, route2, office_lat, office_lon)

            if best_swap:
                # Perform the swap
                user1_idx, user2_idx, improvement = best_swap

                user1 = route1['assigned_users'].pop(user1_idx)
                user2 = route2['assigned_users'].pop(user2_idx)

                route1['assigned_users'].append(user2)
                route2['assigned_users'].append(user1)

                # Re-optimize sequences
                routes[i] = reoptimize_route_sequence(route1, office_lat, office_lon)
                routes[j] = reoptimize_route_sequence(route2, office_lat, office_lon)

                swaps_made += 1
                logger.info(f"   🔄 Swapped users between routes {route1['driver_id']} and {route2['driver_id']} "
                           f"(improvement: {improvement:.2f})")

    if swaps_made > 0:
        logger.info(f"   ✅ Completed {swaps_made} beneficial swaps")

    return routes

def find_best_user_swap(route1, route2, office_lat, office_lon):
    """Find the best user swap between two routes"""
    if not route1['assigned_users'] or not route2['assigned_users']:
        return None

    # Calculate original costs
    original_cost1 = calculate_total_route_distance(route1, office_lat, office_lon)
    original_cost2 = calculate_total_route_distance(route2, office_lat, office_lon)
    original_total = original_cost1 + original_cost2

    best_swap = None
    best_improvement = SWAP_IMPROVEMENT_THRESHOLD

    for i, user1 in enumerate(route1['assigned_users']):
        for j, user2 in enumerate(route2['assigned_users']):
            # Create test routes with swapped users
            test_route1 = route1.copy()
            test_route2 = route2.copy()

            test_route1['assigned_users'] = route1['assigned_users'].copy()
            test_route2['assigned_users'] = route2['assigned_users'].copy()

            # Perform swap
            test_route1['assigned_users'][i] = user2
            test_route2['assigned_users'][j] = user1

            # Calculate new costs
            new_cost1 = calculate_total_route_distance(test_route1, office_lat, office_lon)
            new_cost2 = calculate_total_route_distance(test_route2, office_lat, office_lon)
            new_total = new_cost1 + new_cost2

            improvement = original_total - new_total

            if improvement > best_improvement:
                best_improvement = improvement
                best_swap = (i, j, improvement)

    return best_swap

def perform_route_merges(routes, office_lat, office_lon):
    """Merge compatible routes to improve utilization"""
    merged_routes = []
    used_indices = set()

    for i, route1 in enumerate(routes):
        if i in used_indices:
            continue

        best_merge = None
        best_score = float('inf')

        for j, route2 in enumerate(routes):
            if j <= i or j in used_indices:
                continue

            # Check merge compatibility
            if can_merge_routes(route1, route2, office_lat, office_lon):
                merge_score = calculate_merge_score(route1, route2, office_lat, office_lon)

                if merge_score < best_score:
                    best_score = merge_score
                    best_merge = j

        if best_merge is not None:
            # Perform merge
            merged_route = merge_routes(route1, routes[best_merge], office_lat, office_lon)
            merged_routes.append(merged_route)
            used_indices.add(i)
            used_indices.add(best_merge)

            total_users = len(merged_route['assigned_users'])
            utilization = (total_users / merged_route['vehicle_type']) * 100
            logger.info(f"   🔗 Merged routes {route1['driver_id']} + {routes[best_merge]['driver_id']} "
                       f"= {total_users}/{merged_route['vehicle_type']} seats ({utilization:.1f}%)")
        else:
            merged_routes.append(route1)
            used_indices.add(i)

    return merged_routes

def can_merge_routes(route1, route2, office_lat, office_lon):
    """Check if two routes can be merged"""
    total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
    max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])

    if total_users > max_capacity:
        return False

    # Check distance between route centers
    center1 = calculate_route_center(route1)
    center2 = calculate_route_center(route2)
    center_distance = haversine_distance(center1[0], center1[1], center2[0], center2[1])

    if center_distance > MERGE_DISTANCE_KM:
        return False

    # Check bearing compatibility
    bearing1 = calculate_route_bearing(route1, office_lat, office_lon)
    bearing2 = calculate_route_bearing(route2, office_lat, office_lon)
    bearing_diff = abs(bearing_difference(bearing1, bearing2))

    if bearing_diff > MERGE_BEARING_THRESHOLD:
        return False

    return True

def calculate_route_bearing(route, office_lat, office_lon):
    """Calculate average bearing for a route"""
    if not route['assigned_users']:
        return calculate_bearing(route['latitude'], route['longitude'], office_lat, office_lon)

    center = calculate_route_center(route)
    return calculate_bearing(center[0], center[1], office_lat, office_lon)

def calculate_merge_score(route1, route2, office_lat, office_lon):
    """Calculate score for merging two routes (lower is better)"""
    # Distance component
    center1 = calculate_route_center(route1)
    center2 = calculate_route_center(route2)
    distance_score = haversine_distance(center1[0], center1[1], center2[0], center2[1])

    # Utilization component
    total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
    max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])
    utilization = total_users / max_capacity
    utilization_score = (1 - utilization) * 10  # Penalty for low utilization

    return distance_score + utilization_score

def merge_routes(route1, route2, office_lat, office_lon):
    """Merge two routes into one"""
    # Use the larger capacity vehicle
    if route1['vehicle_type'] >= route2['vehicle_type']:
        merged_route = route1.copy()
    else:
        merged_route = route2.copy()

    # Combine user lists
    all_users = route1['assigned_users'] + route2['assigned_users']
    merged_route['assigned_users'] = all_users

    # Re-optimize sequence
    merged_route = reoptimize_route_sequence(merged_route, office_lat, office_lon)

    return merged_route

def split_tortuous_routes(routes, office_lat, office_lon):
    """Split routes that are too tortuous"""
    final_routes = []

    for route in routes:
        if len(route['assigned_users']) <= 2:
            final_routes.append(route)
            continue

        # Check if route needs splitting
        tortuosity = calculate_route_tortuosity(route, office_lat, office_lon)
        turning_score = calculate_route_turning_score(route, office_lat, office_lon)

        if (tortuosity > TORTUOSITY_THRESHOLD or 
            turning_score > TURNING_THRESHOLD_DEGREES * 1.5):

            # Split the route
            split_routes = split_route_by_direction(route, office_lat, office_lon)
            final_routes.extend(split_routes)

            logger.info(f"   ✂️ Split tortuous route {route['driver_id']} into {len(split_routes)} routes")
        else:
            final_routes.append(route)

    return final_routes

def calculate_route_tortuosity(route, office_lat, office_lon):
    """Calculate tortuosity ratio for a route"""
    if len(route['assigned_users']) <= 1:
        return 1.0

    # Total route distance
    total_distance = calculate_total_route_distance(route, office_lat, office_lon)

    # Direct distance from start to end
    start_pos = (route['latitude'], route['longitude'])
    end_pos = (office_lat, office_lon)
    direct_distance = haversine_distance(start_pos[0], start_pos[1], end_pos[0], end_pos[1])

    return total_distance / (direct_distance + 0.001)

def calculate_route_turning_score(route, office_lat, office_lon):
    """Calculate average turning angle for a route"""
    if len(route['assigned_users']) <= 1:
        return 0

    users = [{'latitude': route['latitude'], 'longitude': route['longitude']}]
    users.extend([{'latitude': u['lat'], 'longitude': u['lng']} for u in route['assigned_users']])
    users.append({'latitude': office_lat, 'longitude': office_lon})

    turning_angles = []

    for i in range(1, len(users) - 1):
        prev_bearing = calculate_bearing(users[i-1]['latitude'], users[i-1]['longitude'],
                                       users[i]['latitude'], users[i]['longitude'])
        next_bearing = calculate_bearing(users[i]['latitude'], users[i]['longitude'],
                                       users[i+1]['latitude'], users[i+1]['longitude'])

        turning_angle = abs(bearing_difference(prev_bearing, next_bearing))
        turning_angles.append(turning_angle)

    return sum(turning_angles) / len(turning_angles) if turning_angles else 0

def split_route_by_direction(route, office_lat, office_lon):
    """Split a route into multiple routes by direction"""
    if len(route['assigned_users']) <= 2:
        return [route]

    # Group users by similar direction
    users_with_bearings = []
    for user in route['assigned_users']:
        bearing = calculate_bearing(user['lat'], user['lng'], office_lat, office_lon)
        users_with_bearings.append((bearing, user))

    # Sort by bearing
    users_with_bearings.sort(key=lambda x: x[0])

    # Find split points where bearing changes significantly
    split_points = []
    for i in range(len(users_with_bearings) - 1):
        bearing_diff = abs(bearing_difference(users_with_bearings[i][0], 
                                            users_with_bearings[i+1][0]))
        if bearing_diff > TURNING_THRESHOLD_DEGREES:
            split_points.append(i + 1)

    # Create split routes
    if not split_points:
        return [route]

    split_routes = []
    start_idx = 0

    for split_idx in split_points + [len(users_with_bearings)]:
        if split_idx > start_idx:
            split_users = [user for _, user in users_with_bearings[start_idx:split_idx]]

            split_route = route.copy()
            split_route['assigned_users'] = split_users
            split_route = reoptimize_route_sequence(split_route, office_lat, office_lon)
            split_routes.append(split_route)

            start_idx = split_idx

    return split_routes if split_routes else [route]

# ================== STAGE 8: FALLBACK AND DRIVER INJECTION ==================

def handle_unassigned_users(unassigned_users_df, available_drivers_df, routes, office_lat, office_lon):
    """Handle remaining unassigned users with fallback strategies"""
    logger.info("🆘 Stage 8: Fallback and driver injection...")

    if unassigned_users_df.empty:
        return routes, []

    # First, try aggressive filling of existing routes
    routes, filled_ids = aggressive_route_filling(routes, unassigned_users_df, office_lat, office_lon)

    remaining_users = unassigned_users_df[~unassigned_users_df['user_id'].isin(filled_ids)]

    if remaining_users.empty:
        return routes, []

    # Then, inject spare drivers for remaining users
    new_routes = inject_spare_drivers(remaining_users, available_drivers_df, office_lat, office_lon)
    routes.extend(new_routes)

    # Update remaining users
    assigned_in_new_routes = set()
    for route in new_routes:
        for user in route['assigned_users']:
            assigned_in_new_routes.add(user['user_id'])

    final_unassigned = remaining_users[~remaining_users['user_id'].isin(assigned_in_new_routes)]

    unassigned_list = []
    for _, user in final_unassigned.iterrows():
        user_data = {
            'user_id': str(user['user_id']),
            'lat': float(user['latitude']),
            'lng': float(user['longitude']),
            'first_name': str(user.get('first_name', '')),
            'email': str(user.get('email', ''))
        }
        unassigned_list.append(user_data)

    logger.info(f"🆘 Fallback complete: {len(unassigned_list)} users remain unassigned")

    return routes, unassigned_list

def aggressive_route_filling(routes, unassigned_users_df, office_lat, office_lon):
    """Aggressively fill existing routes with relaxed constraints"""
    filled_ids = set()

    for route in routes:
        if len(route['assigned_users']) >= route['vehicle_type']:
            continue

        available_seats = route['vehicle_type'] - len(route['assigned_users'])
        route_center = calculate_route_center(route)

        # Find users within expanded distance
        candidates = []
        for _, user in unassigned_users_df.iterrows():
            if user['user_id'] in filled_ids:
                continue

            distance = haversine_distance(route_center[0], route_center[1],
                                        user['latitude'], user['longitude'])

            if distance <= MAX_FILL_DISTANCE_KM * 1.5:  # Relaxed distance
                candidates.append((distance, user))

        # Fill with closest users
        candidates.sort(key=lambda x: x[0])
        users_to_add = candidates[:available_seats]

        for distance, user in users_to_add:
            user_data = {
                'user_id': str(user['user_id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': float(user.get('office_distance', 0))
            }

            if pd.notna(user.get('first_name')):
                user_data['first_name'] = str(user['first_name'])
            if pd.notna(user.get('email')):
                user_data['email'] = str(user['email'])

            route['assigned_users'].append(user_data)
            filled_ids.add(user['user_id'])

        if users_to_add:
            route = reoptimize_route_sequence(route, office_lat, office_lon)

    return routes, filled_ids

def inject_spare_drivers(unassigned_users_df, available_drivers_df, office_lat, office_lon):
    """Inject spare drivers to create routes for remaining users"""
    if available_drivers_df.empty or unassigned_users_df.empty:
        return []

    new_routes = []
    used_driver_ids = set()
    assigned_user_ids = set()

    # Sort drivers by priority (lowest first for spare drivers)
    sorted_drivers = available_drivers_df.sort_values('priority')

    for _, driver in sorted_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids or assigned_user_ids.issuperset(set(unassigned_users_df['user_id'])):
            continue

        remaining_users = unassigned_users_df[~unassigned_users_df['user_id'].isin(assigned_user_ids)]

        if remaining_users.empty:
            break

        # Find users near this driver
        driver_pos = (driver['latitude'], driver['longitude'])
        nearby_users = []

        for _, user in remaining_users.iterrows():
            distance = haversine_distance(driver_pos[0], driver_pos[1],
                                        user['latitude'], user['longitude'])

            if distance <= MAX_FILL_DISTANCE_KM * 2:  # Very relaxed for injection
                nearby_users.append((distance, user))

        if nearby_users:
            # Take up to capacity users, sorted by distance
            nearby_users.sort(key=lambda x: x[0])
            users_for_route = [user for _, user in nearby_users[:driver['capacity']]]

            # Create new route
            route = {
                'driver_id': str(driver['driver_id']),
                'vehicle_id': str(driver.get('vehicle_id', '')),
                'vehicle_type': int(driver['capacity']),
                'latitude': float(driver['latitude']),
                'longitude': float(driver['longitude']),
                'assigned_users': []
            }

            for user in users_for_route:
                user_data = {
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0))
                }

                if pd.notna(user.get('first_name')):
                    user_data['first_name'] = str(user['first_name'])
                if pd.notna(user.get('email')):
                    user_data['email'] = str(user['email'])

                route['assigned_users'].append(user_data)
                assigned_user_ids.add(user['user_id'])

            route = reoptimize_route_sequence(route, office_lat, office_lon)
            new_routes.append(route)
            used_driver_ids.add(driver['driver_id'])

            utilization = len(route['assigned_users']) / driver['capacity'] * 100
            logger.info(f"   🚗 Injected driver {driver['driver_id']}: "
                       f"{len(route['assigned_users'])}/{driver['capacity']} seats ({utilization:.1f}%)")

    return new_routes

# ================== MAIN ASSIGNMENT FUNCTION ==================

def run_assignment_balance(source_id: str, parameter: int = 1, string_param: str = "", choice: str = ""):
    """Main entry point for locality-first balanced optimization assignment"""
    return run_assignment_balance_internal(source_id, parameter, string_param, choice)

def run_assignment_balance_internal(source_id: str, parameter: int = 1, string_param: str = "", choice: str = ""):
    """
    Main assignment function implementing the locality-first blueprint:
    1. Cluster by locality first (geographic proximity)
    2. Split by main travel direction within each locality
    3. Form capacity-aware clusters within each direction group
    4. Assign best driver per cluster using balanced cost function
    5. Optimize sequence within cluster using direction-aware 2-opt
    6. Fill remaining seats with path-aware constraints
    7. Global optimization with swaps and merges
    8. Fallback and driver injection for remaining users
    """
    start_time = time.time()

    # Reload configuration
    global _config
    _config = load_and_validate_config()

    logger.info(f"🚀 Starting LOCALITY-FIRST BALANCED assignment for source_id: {source_id}")
    logger.info(f"📋 Parameters: {parameter}, String: {string_param}, Choice: {choice}")

    try:
        # Load and validate data
        data = load_env_and_fetch_data(source_id, parameter, string_param)

        # Edge case handling
        users = data.get('users', [])
        if not users:
            logger.warning("⚠️ No users found - returning empty assignment")
            return {
                "status": "true",
                "execution_time": time.time() - start_time,
                "data": [],
                "unassignedUsers": [],
                "unassignedDrivers": _get_all_drivers_as_unassigned(data),
                "clustering_analysis": {"method": "No Users", "clusters": 0},
                "optimization_mode": "locality_first_balanced",
                "parameter": parameter,
            }

        # Get all drivers
        all_drivers = []
        if "drivers" in data:
            drivers_data = data["drivers"]
            all_drivers.extend(drivers_data.get("driversUnassigned", []))
            all_drivers.extend(drivers_data.get("driversAssigned", []))
        else:
            all_drivers.extend(data.get("driversUnassigned", []))
            all_drivers.extend(data.get("driversAssigned", []))

        if not all_drivers:
            logger.warning("⚠️ No drivers available - all users unassigned")
            unassigned_users = _convert_users_to_unassigned_format(users)
            return {
                "status": "true",
                "execution_time": time.time() - start_time,
                "data": [],
                "unassignedUsers": unassigned_users,
                "unassignedDrivers": [],
                "clustering_analysis": {"method": "No Drivers", "clusters": 0},
                "optimization_mode": "locality_first_balanced",
                "parameter": parameter,
            }

        logger.info(f"📥 Data loaded - Users: {len(users)}, Drivers: {len(all_drivers)}")

        # Extract office coordinates and validate data
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        logger.info("✅ Data validation passed")

        # Prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)
        logger.info(f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")

        # STAGE 1: Derive features and create locality clusters
        user_df = derive_user_features(user_df, office_lat, office_lon)
        user_df = create_locality_clusters(user_df)

        # STAGE 2 & 3: Process each locality cluster
        routes = []
        assigned_user_ids = set()
        used_driver_ids = set()

        # Get available vehicle capacities for capacity-aware clustering
        available_capacities = driver_df['capacity'].tolist()

        for locality_id in user_df['locality_cluster'].unique():
            locality_users = user_df[user_df['locality_cluster'] == locality_id].copy()
            logger.info(f"🏘️ Processing locality {locality_id} with {len(locality_users)} users")

            # STAGE 2: Split by direction within locality
            locality_users = split_locality_by_direction(locality_users, locality_id)

            # STAGE 3 & 4: Create capacity clusters and assign drivers
            for direction_id in locality_users['direction_cluster'].unique():
                direction_group = locality_users[locality_users['direction_cluster'] == direction_id].copy()

                if len(direction_group) == 0:
                    continue

                # Create capacity-aware subclusters
                direction_group = create_capacity_aware_subclusters(direction_group, available_capacities)

                # Assign drivers to each capacity cluster
                for capacity_id in direction_group['capacity_cluster'].unique():
                    cluster_users = direction_group[direction_group['capacity_cluster'] == capacity_id]

                    if len(cluster_users) == 0:
                        continue

                    # Get available drivers
                    available_drivers = driver_df[~driver_df['driver_id'].isin(used_driver_ids)]

                    if available_drivers.empty:
                        logger.warning(f"No more drivers available for cluster in locality {locality_id}")
                        break

                    # STAGE 4: Assign best driver
                    route = assign_best_driver_to_cluster(
                        cluster_users, available_drivers, used_driver_ids, office_lat, office_lon)

                    if route:
                        routes.append(route)
                        for user in route['assigned_users']:
                            assigned_user_ids.add(user['user_id'])

        logger.info(f"✅ Initial assignment: {len(routes)} routes, {len(assigned_user_ids)} users assigned")

        # STAGE 6: Path-aware seat filling
        unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()
        routes, filled_user_ids = path_aware_seat_filling(routes, unassigned_users_df, office_lat, office_lon)
        assigned_user_ids.update(filled_user_ids)

        # STAGE 7: Global optimization
        routes = local_swaps_and_merges(routes, office_lat, office_lon)

        # STAGE 8: Handle remaining unassigned users
        remaining_unassigned = user_df[~user_df['user_id'].isin(assigned_user_ids)]
        available_drivers_df = driver_df[~driver_df['driver_id'].isin(used_driver_ids)]

        routes, unassigned_users = handle_unassigned_users(
            remaining_unassigned, available_drivers_df, routes, office_lat, office_lon)

        # Filter out empty routes and build unassigned drivers list
        filtered_routes = [r for r in routes if r['assigned_users']]
        assigned_driver_ids = {route['driver_id'] for route in filtered_routes}
        unassigned_drivers_df = driver_df[~driver_df['driver_id'].isin(assigned_driver_ids)]

        unassigned_drivers = []
        for _, driver in unassigned_drivers_df.iterrows():
            driver_data = {
                'driver_id': str(driver.get('driver_id', '')),
                'capacity': int(driver.get('capacity', 0)),
                'vehicle_id': str(driver.get('vehicle_id', '')),
                'latitude': float(driver.get('latitude', 0.0)),
                'longitude': float(driver.get('longitude', 0.0))
            }
            unassigned_drivers.append(driver_data)

        execution_time = time.time() - start_time

        # Final metrics
        total_users_in_api = len(users)
        users_assigned = sum(len(r['assigned_users']) for r in filtered_routes)
        users_unassigned = len(unassigned_users)

        logger.info(f"✅ Locality-first balanced assignment complete in {execution_time:.2f}s")
        logger.info(f"📊 Final routes: {len(filtered_routes)}")
        logger.info(f"🎯 Users assigned: {users_assigned}")
        logger.info(f"👥 Users unassigned: {users_unassigned}")

        # Build enhanced response with rich data
        company_info = data.get("company", {})
        shift_info = data.get("shift", {})

        # Enhance routes with driver and user details
        enhanced_routes = []
        for route in filtered_routes:
            enhanced_route = route.copy()

            # Add driver details
            driver_id = route['driver_id']
            if "drivers" in data:
                all_drivers_data = data["drivers"].get("driversUnassigned", []) + data["drivers"].get("driversAssigned", [])
            else:
                all_drivers_data = data.get("driversUnassigned", []) + data.get("driversAssigned", [])

            for driver in all_drivers_data:
                if str(driver.get('id', driver.get('sub_user_id', ''))) == driver_id:
                    enhanced_route.update({
                        'first_name': driver.get('first_name', ''),
                        'last_name': driver.get('last_name', ''),
                        'email': driver.get('email', ''),
                        'vehicle_name': driver.get('vehicle_name', ''),
                        'vehicle_no': driver.get('vehicle_no', ''),
                        'capacity': driver.get('capacity', ''),
                        'chasis_no': driver.get('chasis_no', ''),
                        'color': driver.get('color', ''),
                        'registration_no': driver.get('registration_no', ''),
                        'shift_type_id': driver.get('shift_type_id', '')
                    })
                    break

            # Enhance user information
            enhanced_users = []
            original_users = data.get('users', [])

            for user in route['assigned_users']:
                enhanced_user = user.copy()
                for orig_user in original_users:
                    if str(orig_user.get('id', orig_user.get('sub_user_id', ''))) == user['user_id']:
                        enhanced_user.update({
                            'address': orig_user.get('address', ''),
                            'employee_shift': orig_user.get('employee_shift', ''),
                            'shift_type': orig_user.get('shift_type', ''),
                            'last_name': orig_user.get('last_name', ''),
                            'phone': orig_user.get('phone', '')
                        })
                        break
                enhanced_users.append(enhanced_user)

            enhanced_route['assigned_users'] = enhanced_users
            enhanced_routes.append(enhanced_route)

        # Enhance unassigned data
        enhanced_unassigned_users = []
        original_users = data.get('users', [])

        for user in unassigned_users:
            enhanced_user = user.copy()
            for orig_user in original_users:
                if str(orig_user.get('id', orig_user.get('sub_user_id', ''))) == user['user_id']:
                    enhanced_user.update({
                        'address': orig_user.get('address', ''),
                        'employee_shift': orig_user.get('employee_shift', ''),
                        'shift_type': orig_user.get('shift_type', ''),
                        'last_name': orig_user.get('last_name', ''),
                        'phone': orig_user.get('phone', '')
                    })
                    break
            enhanced_unassigned_users.append(enhanced_user)

        enhanced_unassigned_drivers = []
        if "drivers" in data:
            all_drivers_data = data["drivers"].get("driversUnassigned", []) + data["drivers"].get("driversAssigned", [])
        else:
            all_drivers_data = data.get("driversUnassigned", []) + data.get("driversAssigned", [])

        for driver in unassigned_drivers:
            enhanced_driver = driver.copy()
            for orig_driver in all_drivers_data:
                if str(orig_driver.get('id', orig_driver.get('sub_user_id', ''))) == driver['driver_id']:
                    enhanced_driver.update({
                        'first_name': orig_driver.get('first_name', ''),
                        'last_name': orig_driver.get('last_name', ''),
                        'email': orig_driver.get('email', ''),
                        'vehicle_name': orig_driver.get('vehicle_name', ''),
                        'vehicle_no': orig_driver.get('vehicle_no', ''),
                        'chasis_no': orig_driver.get('chasis_no', ''),
                        'color': orig_driver.get('color', ''),
                        'registration_no': orig_driver.get('registration_no', ''),
                        'shift_type_id': orig_driver.get('shift_type_id', '')
                    })
                    break
            enhanced_unassigned_drivers.append(enhanced_driver)

        clustering_results = {
            "method": "locality_first_balanced", 
            "clusters": len(user_df['locality_cluster'].unique()) if not user_df.empty else 0
        }

        return {
            "status": "true",
            "execution_time": execution_time,
            "company": company_info,
            "shift": shift_info,
            "data": enhanced_routes,
            "unassignedUsers": enhanced_unassigned_users,
            "unassignedDrivers": enhanced_unassigned_drivers,
            "clustering_analysis": clustering_results,
            "optimization_mode": "locality_first_balanced",
            "parameter": parameter,
        }

    except requests.exceptions.RequestException as req_err:
        logger.error(f"API request failed: {req_err}")
        return {"status": "false", "details": str(req_err), "data": []}
    except ValueError as val_err:
        logger.error(f"Data validation error: {val_err}")
        return {"status": "false", "details": str(val_err), "data": []}
    except Exception as e:
        logger.error(f"Assignment failed: {e}", exc_info=True)
        return {"status": "false", "details": str(e), "data": []}