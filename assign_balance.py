
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
from scipy.spatial.distance import cdist
from dotenv import load_dotenv
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from logger_config import get_logger

warnings.filterwarnings('ignore')

# Setup logging
logger = get_logger()


# Load and validate configuration with truly balanced optimization settings
def load_and_validate_config():
    """Load configuration with truly balanced optimization settings"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    # Use balanced mode configuration
    current_mode = "balanced_optimization"
    mode_configs = cfg.get("mode_configs", {})
    mode_config = mode_configs.get("balanced_optimization", {})

    logger.info(f"🎯 Using optimization mode: TRUE BALANCED OPTIMIZATION")

    config = {}

    # Truly balanced distance configurations (midpoint between route and capacity)
    config['MAX_FILL_DISTANCE_KM'] = max(0.1, float(mode_config.get("max_fill_distance_km", cfg.get("max_fill_distance_km", 6.5))))  # Route: 5.0, Capacity: 8.0
    config['MERGE_DISTANCE_KM'] = max(0.1, float(mode_config.get("merge_distance_km", cfg.get("merge_distance_km", 4.0))))  # Route: 3.0, Capacity: 5.0
    config['DBSCAN_EPS_KM'] = max(0.1, float(cfg.get("dbscan_eps_km", 2.0)))  # Route: 1.5, Capacity: 2.5
    config['OVERFLOW_PENALTY_KM'] = max(0.0, float(cfg.get("overflow_penalty_km", 7.5)))  # Route: 10.0, Capacity: 5.0
    config['DISTANCE_ISSUE_THRESHOLD'] = max(0.1, float(cfg.get("distance_issue_threshold_km", 10.0)))  # Route: 8.0, Capacity: 12.0
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(0.0, float(cfg.get("swap_improvement_threshold_km", 0.75)))  # Route: 0.5, Capacity: 1.0

    # Balanced utilization thresholds (midpoint between route and capacity)
    config['MIN_UTIL_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("min_util_threshold", 0.65))))  # Route: 0.5, Capacity: 0.8
    config['LOW_UTILIZATION_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("low_utilization_threshold", 0.6))))  # Route: 0.5, Capacity: 0.7

    # Balanced integer configurations
    config['MIN_SAMPLES_DBSCAN'] = max(1, int(cfg.get("min_samples_dbscan", 2)))
    config['MAX_SWAP_ITERATIONS'] = max(1, int(cfg.get("max_swap_iterations", 4)))  # Route: 3, Capacity: 5
    config['MAX_USERS_FOR_FALLBACK'] = max(1, int(cfg.get("max_users_for_fallback", 4)))  # Route: 3, Capacity: 5
    config['FALLBACK_MIN_USERS'] = max(1, int(cfg.get("fallback_min_users", 2)))
    config['FALLBACK_MAX_USERS'] = max(1, int(cfg.get("fallback_max_users", 7)))  # Route: 7, Capacity: 10

    # Balanced angle configurations (midpoint between route and capacity)
    config['MAX_BEARING_DIFFERENCE'] = max(0, min(180, float(mode_config.get("max_bearing_difference", cfg.get("max_bearing_difference", 32.5)))))  # Route: 20, Capacity: 45
    config['MAX_TURNING_ANGLE'] = max(0, min(180, float(mode_config.get("max_allowed_turning_score", cfg.get("max_allowed_turning_score", 47.5)))))  # Route: 35, Capacity: 60

    # Balanced cost penalties (midpoint between route and capacity)
    config['UTILIZATION_PENALTY_PER_SEAT'] = max(0.0, float(mode_config.get("utilization_penalty_per_seat", cfg.get("utilization_penalty_per_seat", 3.5))))  # Route: 2.0, Capacity: 5.0

    # Office coordinates with environment variable fallbacks
    office_lat = float(os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489)))
    office_lon = float(os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))

    # Validate coordinate bounds
    if not (-90 <= office_lat <= 90):
        logger.warning(f"Invalid office latitude {office_lat}, using default")
        office_lat = 30.6810489
    if not (-180 <= office_lon <= 180):
        logger.warning(f"Invalid office longitude {office_lon}, using default")
        office_lon = 76.7260711

    config['OFFICE_LAT'] = office_lat
    config['OFFICE_LON'] = office_lon

    # Truly balanced optimization parameters (50/50 split)
    config['optimization_mode'] = "balanced_optimization"
    config['aggressive_merging'] = mode_config.get('aggressive_merging', None)  # Moderate approach
    config['capacity_weight'] = mode_config.get('capacity_weight', 3.0)  # Route: 1.0, Capacity: 5.0
    config['direction_weight'] = mode_config.get('direction_weight', 2.0)  # Route: 3.0, Capacity: 1.0

    # Balanced clustering parameters (midpoint between route and capacity)
    config['clustering_method'] = cfg.get('clustering_method', 'adaptive')
    config['min_cluster_size'] = max(2, cfg.get('min_cluster_size', 2))
    config['use_sweep_algorithm'] = cfg.get('use_sweep_algorithm', True)  # Route: True, Capacity: False
    config['angular_sectors'] = cfg.get('angular_sectors', 8)  # Route: 8, Capacity: 6
    config['max_users_per_initial_cluster'] = cfg.get('max_users_per_initial_cluster', 10)  # Route: 8, Capacity: 12
    config['max_users_per_cluster'] = cfg.get('max_users_per_cluster', 7)  # Route: 7, Capacity: 10

    # Balanced optimization parameters (midpoint values)
    config['zigzag_penalty_weight'] = mode_config.get('zigzag_penalty_weight', cfg.get('zigzag_penalty_weight', 1.75))  # Route: 3.0, Capacity: 0.5
    config['route_split_turning_threshold'] = cfg.get('route_split_turning_threshold', 57.5)  # Route: 35, Capacity: 80
    config['max_tortuosity_ratio'] = cfg.get('max_tortuosity_ratio', 1.7)  # Route: 1.4, Capacity: 2.0
    config['route_split_consistency_threshold'] = cfg.get('route_split_consistency_threshold', 0.5)  # Route: 0.7, Capacity: 0.3
    config['merge_tortuosity_improvement_required'] = cfg.get('merge_tortuosity_improvement_required', None)  # Balanced

    # Latitude conversion factor for distance normalization
    config['LAT_TO_KM'] = 111.0
    config['LON_TO_KM'] = 111.0 * math.cos(math.radians(office_lat))

    logger.info(f"   📊 Max bearing difference: {config['MAX_BEARING_DIFFERENCE']}°")
    logger.info(f"   📊 Max turning score: {config['MAX_TURNING_ANGLE']}°")
    logger.info(f"   📊 Max fill distance: {config['MAX_FILL_DISTANCE_KM']}km")
    logger.info(f"   📊 Capacity weight: {config['capacity_weight']}")
    logger.info(f"   📊 Direction weight: {config['direction_weight']}")

    return config


# Import only essential functions from assignment.py
from assignment import (
    validate_input_data, load_env_and_fetch_data, extract_office_coordinates,
    prepare_user_driver_dataframes, haversine_distance, bearing_difference,
    calculate_bearing_vectorized, calculate_bearing, calculate_bearings_and_features,
    coords_to_km, dbscan_clustering_metric, kmeans_clustering_metric, estimate_clusters,
    _get_all_drivers_as_unassigned, _convert_users_to_unassigned_format, analyze_assignment_quality,
    get_progress_tracker, normalize_bearing_difference
)

# Load validated configuration - always balanced optimization
_config = load_and_validate_config()
MAX_FILL_DISTANCE_KM = _config['MAX_FILL_DISTANCE_KM']
MERGE_DISTANCE_KM = _config['MERGE_DISTANCE_KM']
MIN_UTIL_THRESHOLD = _config['MIN_UTIL_THRESHOLD']
DBSCAN_EPS_KM = _config['DBSCAN_EPS_KM']
MIN_SAMPLES_DBSCAN = _config['MIN_SAMPLES_DBSCAN']
MAX_BEARING_DIFFERENCE = _config['MAX_BEARING_DIFFERENCE']
SWAP_IMPROVEMENT_THRESHOLD = _config['SWAP_IMPROVEMENT_THRESHOLD']
MAX_SWAP_ITERATIONS = _config['MAX_SWAP_ITERATIONS']
UTILIZATION_PENALTY_PER_SEAT = _config['UTILIZATION_PENALTY_PER_SEAT']
OVERFLOW_PENALTY_KM = _config['OVERFLOW_PENALTY_KM']
DISTANCE_ISSUE_THRESHOLD = _config['DISTANCE_ISSUE_THRESHOLD']
LOW_UTILIZATION_THRESHOLD = _config['LOW_UTILIZATION_THRESHOLD']
MAX_USERS_FOR_FALLBACK = _config['MAX_USERS_FOR_FALLBACK']
FALLBACK_MIN_USERS = _config['FALLBACK_MIN_USERS']
FALLBACK_MAX_USERS = _config['FALLBACK_MAX_USERS']
OFFICE_LAT = _config['OFFICE_LAT']
OFFICE_LON = _config['OFFICE_LON']


# INDEPENDENT BALANCED CLUSTERING FUNCTIONS
def create_balanced_geographic_clusters(user_df, office_lat, office_lon, config):
    """Create truly balanced geographic clusters - independent implementation"""
    if len(user_df) == 0:
        return user_df

    logger.info("  🎯 Creating balanced geographic clusters...")
    
    # Calculate features including bearings
    user_df = calculate_bearings_and_features(user_df, office_lat, office_lon)
    
    # Balanced clustering approach - combine both sector and distance clustering
    if len(user_df) > 4:
        labels = balanced_hybrid_clustering(user_df, office_lat, office_lon, config)
    else:
        labels = simple_balanced_clustering(user_df, config)
    
    user_df['geo_cluster'] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    logger.info(f"  ✅ Created {n_clusters} balanced geographic clusters")
    return user_df


def balanced_hybrid_clustering(user_df, office_lat, office_lon, config):
    """Balanced hybrid clustering - combines directional and spatial approaches"""
    n_sectors = config.get('angular_sectors', 8)
    sector_angle = 360.0 / n_sectors
    
    # Convert to metric coordinates
    coords_km = []
    bearings = []
    for _, user in user_df.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'], office_lat, office_lon)
        coords_km.append([lat_km, lon_km])
        bearings.append(user['bearing_from_office'])
    
    coords_km = np.array(coords_km)
    bearings = np.array(bearings)
    
    # Create balanced feature matrix - 50% spatial, 50% directional
    spatial_weight = 1.0
    directional_weight = 1.0
    
    # Directional features
    bearing_sin = np.sin(np.radians(bearings)) * directional_weight
    bearing_cos = np.cos(np.radians(bearings)) * directional_weight
    
    # Combined features for balanced clustering
    features = np.column_stack([
        coords_km * spatial_weight,  # Spatial component
        bearing_sin.reshape(-1, 1),  # Directional component
        bearing_cos.reshape(-1, 1)   # Directional component
    ])
    
    # Use DBSCAN with balanced parameters
    eps_balanced = config.get('DBSCAN_EPS_KM', 2.0)
    dbscan = DBSCAN(eps=eps_balanced, min_samples=2)
    labels = dbscan.fit_predict(features)
    
    # Handle noise points by assigning to nearest cluster
    noise_mask = labels == -1
    if noise_mask.any() and len(set(labels)) > 1:
        valid_labels = labels[~noise_mask]
        if len(valid_labels) > 0:
            for i in np.where(noise_mask)[0]:
                noise_point = features[i]
                distances = cdist([noise_point], features[~noise_mask])[0]
                nearest_idx = np.argmin(distances)
                labels[i] = valid_labels[nearest_idx]
    
    return labels


def simple_balanced_clustering(user_df, config):
    """Simple balanced clustering for small datasets"""
    max_cluster_size = config.get('max_users_per_cluster', 7)
    
    # Sort by bearing for directional consistency
    sorted_users = user_df.sort_values('bearing_from_office')
    
    labels = []
    current_cluster = 0
    current_size = 0
    
    for _ in sorted_users.iterrows():
        if current_size >= max_cluster_size:
            current_cluster += 1
            current_size = 0
        
        labels.append(current_cluster)
        current_size += 1
    
    # Map back to original order
    result_labels = [-1] * len(user_df)
    for i, orig_idx in enumerate(sorted_users.index):
        result_labels[orig_idx] = labels[i]
    
    return result_labels


def create_balanced_capacity_subclusters(user_df, office_lat, office_lon, config):
    """Create balanced capacity subclusters - independent implementation"""
    if len(user_df) == 0:
        return user_df

    logger.info("  🚗 Creating balanced capacity-based sub-clusters...")

    user_df['sub_cluster'] = -1
    sub_cluster_counter = 0
    max_bearing_diff = config.get('MAX_BEARING_DIFFERENCE', 32.5)

    for geo_cluster in user_df['geo_cluster'].unique():
        if geo_cluster == -1:
            continue

        geo_cluster_users = user_df[user_df['geo_cluster'] == geo_cluster]
        max_users_per_cluster = config.get('max_users_per_cluster', 7)

        if len(geo_cluster_users) <= max_users_per_cluster:
            user_df.loc[geo_cluster_users.index, 'sub_cluster'] = sub_cluster_counter
            sub_cluster_counter += 1
        else:
            # Balanced sub-clustering approach
            sub_cluster_counter = balanced_capacity_splitting(
                geo_cluster_users, user_df, sub_cluster_counter, config, max_bearing_diff
            )

    logger.info(f"  ✅ Created {user_df['sub_cluster'].nunique()} balanced capacity-based sub-clusters")
    return user_df


def balanced_capacity_splitting(geo_cluster_users, user_df, sub_cluster_counter, config, max_bearing_diff):
    """Split clusters using balanced approach - considers both capacity and direction"""
    sorted_users = geo_cluster_users.sort_values('bearing_from_office')
    max_users_per_cluster = config.get('max_users_per_cluster', 7)
    
    current_cluster_users = []
    
    for idx, (user_idx, user) in enumerate(sorted_users.iterrows()):
        # Check both capacity and bearing constraints
        capacity_exceeded = len(current_cluster_users) >= max_users_per_cluster
        bearing_exceeded = False
        
        if current_cluster_users and len(current_cluster_users) >= 2:
            # Check bearing spread
            all_bearings = [u[1]['bearing_from_office'] for u in current_cluster_users] + [user['bearing_from_office']]
            bearing_spread = calculate_balanced_bearing_spread(all_bearings)
            bearing_exceeded = bearing_spread > max_bearing_diff
        
        # Split if either constraint is exceeded
        if capacity_exceeded or bearing_exceeded:
            # Assign current cluster
            for cluster_user_idx, _ in current_cluster_users:
                user_df.loc[cluster_user_idx, 'sub_cluster'] = sub_cluster_counter
            sub_cluster_counter += 1
            current_cluster_users = []
        
        current_cluster_users.append((user_idx, user))
    
    # Assign remaining users
    if current_cluster_users:
        for cluster_user_idx, _ in current_cluster_users:
            user_df.loc[cluster_user_idx, 'sub_cluster'] = sub_cluster_counter
        sub_cluster_counter += 1
    
    return sub_cluster_counter


def calculate_balanced_bearing_spread(bearings):
    """Calculate bearing spread with balanced constraints"""
    if len(bearings) <= 1:
        return 0
    
    bearings = sorted(bearings)
    max_gap = 0
    
    for i in range(len(bearings)):
        gap = bearings[(i + 1) % len(bearings)] - bearings[i]
        if gap < 0:
            gap += 360
        max_gap = max(max_gap, gap)
    
    return 360 - max_gap if max_gap > 180 else max_gap


# INDEPENDENT BALANCED DRIVER ASSIGNMENT
def assign_drivers_balanced_approach(user_df, driver_df, office_lat, office_lon):
    """Truly balanced driver assignment - independent implementation"""
    logger.info("🚗 Step 3: Balanced driver assignment (50% route efficiency + 50% capacity)...")

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Balanced sorting: equal priority to capacity and availability
    available_drivers = driver_df.sort_values(['capacity', 'priority'], ascending=[False, True])
    all_unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()

    # Process each sub-cluster with balanced optimization
    for sub_cluster_id in sorted(user_df['sub_cluster'].unique()):
        cluster_users = user_df[user_df['sub_cluster'] == sub_cluster_id]
        unassigned_in_cluster = cluster_users[~cluster_users['user_id'].isin(assigned_user_ids)]

        if unassigned_in_cluster.empty:
            continue

        # Balanced approach to large clusters
        max_driver_capacity = int(available_drivers['capacity'].max()) if not available_drivers.empty else 0
        
        if max_driver_capacity > 0 and len(unassigned_in_cluster) > max_driver_capacity:
            # Split large clusters with balanced criteria
            parts = math.ceil(len(unassigned_in_cluster) / max_driver_capacity)
            split_clusters = balanced_cluster_splitting(unassigned_in_cluster, parts, office_lat, office_lon)
            
            for split_cluster in split_clusters:
                route = assign_best_driver_balanced(split_cluster, available_drivers, used_driver_ids, office_lat, office_lon)
                if route:
                    routes.append(route)
                    assigned_user_ids.update(u['user_id'] for u in route['assigned_users'])
            continue

        # Check for balanced quality and split if needed
        if needs_balanced_splitting(unassigned_in_cluster, office_lat, office_lon):
            split_routes = balanced_bearing_split(unassigned_in_cluster, available_drivers, used_driver_ids, office_lat, office_lon)
            for route in split_routes:
                if route:
                    routes.append(route)
                    assigned_user_ids.update(u['user_id'] for u in route['assigned_users'])
            continue

        # Standard balanced assignment
        route = assign_best_driver_balanced(unassigned_in_cluster, available_drivers, used_driver_ids, office_lat, office_lon)
        if route:
            routes.append(route)
            assigned_user_ids.update(u['user_id'] for u in route['assigned_users'])

    logger.info(f"  ✅ Created {len(routes)} routes with balanced optimization")
    return routes, assigned_user_ids


def balanced_cluster_splitting(cluster_users, parts, office_lat, office_lon):
    """Split clusters using balanced criteria (spatial + directional)"""
    coords_km = []
    bearings = []
    rows = []
    
    for _, user in cluster_users.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'], office_lat, office_lon)
        coords_km.append([lat_km, lon_km])
        bearings.append(calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude']))
        rows.append(user)
    
    # Balanced feature matrix
    features = np.column_stack([
        coords_km,  # 50% spatial
        np.sin(np.radians(bearings)),  # 25% directional
        np.cos(np.radians(bearings))   # 25% directional
    ])
    
    n_clusters = min(parts, len(features))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Group users by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(rows[i])
    
    return [pd.DataFrame(users) for users in clusters.values() if users]


def needs_balanced_splitting(cluster_users, office_lat, office_lon):
    """Check if cluster needs splitting based on balanced criteria"""
    if len(cluster_users) <= 2:
        return False
    
    bearings = []
    for _, user in cluster_users.iterrows():
        bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
        bearings.append(bearing)
    
    bearing_spread = calculate_balanced_bearing_spread(bearings)
    return bearing_spread > MAX_BEARING_DIFFERENCE


def balanced_bearing_split(cluster_users, available_drivers, used_driver_ids, office_lat, office_lon):
    """Split cluster by bearing using balanced approach"""
    coords_km = []
    bearings = []
    
    for _, user in cluster_users.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'], office_lat, office_lon)
        coords_km.append([lat_km, lon_km])
        bearings.append(calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude']))
    
    # Balanced clustering with equal weight to spatial and directional
    features = np.column_stack([
        coords_km,
        np.sin(np.radians(bearings)),
        np.cos(np.radians(bearings))
    ])
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    split_routes = []
    for split_id in range(2):
        split_users = cluster_users.iloc[labels == split_id]
        if len(split_users) > 0:
            route = assign_best_driver_balanced(split_users, available_drivers, used_driver_ids, office_lat, office_lon)
            if route:
                split_routes.append(route)
    
    return split_routes


def assign_best_driver_balanced(cluster_users, available_drivers, used_driver_ids, office_lat, office_lon):
    """Assign best driver using truly balanced scoring (50% capacity + 50% route)"""
    cluster_size = len(cluster_users)
    
    best_driver = None
    best_score = float('inf')
    best_sequence = None
    
    # Balanced weights - exactly 50/50 split
    capacity_weight = _config.get('capacity_weight', 3.0)
    direction_weight = _config.get('direction_weight', 2.0)
    
    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids:
            continue
        
        if driver['capacity'] < cluster_size:
            continue
        
        # Calculate balanced route metrics
        route_cost, sequence, turning_score = calculate_balanced_route_cost(driver, cluster_users, office_lat, office_lon)
        
        # Balanced scoring components
        utilization = cluster_size / driver['capacity']
        
        # 50% Route efficiency component
        route_efficiency_score = route_cost * 0.5 + turning_score * direction_weight * 0.02
        
        # 50% Capacity component  
        capacity_score = (1.0 - utilization) * capacity_weight * 2.0
        
        # Balanced bonuses
        utilization_bonus = 0
        if utilization >= 0.85:
            utilization_bonus = -1.5
        elif utilization >= 0.7:
            utilization_bonus = -0.75
        
        route_bonus = 0
        if turning_score <= 35:
            route_bonus = -1.0
        elif turning_score <= 45:
            route_bonus = -0.5
        
        # Priority component (small tiebreaker)
        priority_score = driver['priority'] * 0.1
        
        # Truly balanced total score: equal weight to both objectives
        total_score = (route_efficiency_score * 0.5 + capacity_score * 0.5 + 
                      priority_score + utilization_bonus + route_bonus)
        
        if total_score < best_score:
            best_score = total_score
            best_driver = driver
            best_sequence = sequence
    
    if best_driver is not None:
        used_driver_ids.add(best_driver['driver_id'])
        
        route = {
            'driver_id': str(best_driver['driver_id']),
            'vehicle_id': str(best_driver.get('vehicle_id', '')),
            'vehicle_type': int(best_driver['capacity']),
            'latitude': float(best_driver['latitude']),
            'longitude': float(best_driver['longitude']),
            'assigned_users': []
        }
        
        # Add users in optimal sequence
        for seq_user in best_sequence:
            user_data = {
                'user_id': str(seq_user['user_id']),
                'lat': float(seq_user['latitude']),
                'lng': float(seq_user['longitude']),
                'office_distance': float(seq_user.get('office_distance', 0))
            }
            
            # Add optional fields
            if 'first_name' in seq_user and pd.notna(seq_user['first_name']):
                user_data['first_name'] = str(seq_user['first_name'])
            if 'email' in seq_user and pd.notna(seq_user['email']):
                user_data['email'] = str(seq_user['email'])
            
            route['assigned_users'].append(user_data)
        
        # Optimize and update metrics
        route = optimize_balanced_route_sequence(route, office_lat, office_lon)
        update_balanced_route_metrics(route, office_lat, office_lon)
        
        utilization = len(route['assigned_users']) / route['vehicle_type']
        turning_score = route.get('turning_score', 0)
        logger.info(f"    ⚖️ Balanced assignment - Driver {best_driver['driver_id']}: {len(route['assigned_users'])}/{route['vehicle_type']} seats ({utilization*100:.1f}%, {turning_score:.1f}° turn)")
        
        return route
    
    return None


def calculate_balanced_route_cost(driver, cluster_users, office_lat, office_lon):
    """Calculate route cost with balanced optimization (equal weight to distance and direction)"""
    if len(cluster_users) == 0:
        return float('inf'), [], 0
    
    driver_pos = (driver['latitude'], driver['longitude'])
    office_pos = (office_lat, office_lon)
    
    # Get balanced pickup sequence
    sequence = calculate_balanced_optimal_sequence(driver_pos, cluster_users, office_pos)
    
    # Calculate total route distance
    total_distance = 0
    bearing_differences = []
    
    # Driver to first pickup
    if sequence:
        first_user = sequence[0]
        total_distance += haversine_distance(driver_pos[0], driver_pos[1], 
                                           first_user['latitude'], first_user['longitude'])
    
    # Between pickups
    for i in range(len(sequence) - 1):
        current_user = sequence[i]
        next_user = sequence[i + 1]
        
        distance = haversine_distance(current_user['latitude'], current_user['longitude'],
                                    next_user['latitude'], next_user['longitude'])
        total_distance += distance
        
        # Calculate bearing differences for turning score
        if i == 0:
            prev_bearing = calculate_bearing(driver_pos[0], driver_pos[1], 
                                           current_user['latitude'], current_user['longitude'])
        else:
            prev_pos = (sequence[i-1]['latitude'], sequence[i-1]['longitude'])
            prev_bearing = calculate_bearing(prev_pos[0], prev_pos[1],
                                           current_user['latitude'], current_user['longitude'])
        
        next_bearing = calculate_bearing(current_user['latitude'], current_user['longitude'],
                                       next_user['latitude'], next_user['longitude'])
        
        bearing_diff = bearing_difference(prev_bearing, next_bearing)
        bearing_differences.append(bearing_diff)
    
    # Last pickup to office
    if sequence:
        last_user = sequence[-1]
        total_distance += haversine_distance(last_user['latitude'], last_user['longitude'],
                                           office_lat, office_lon)
    
    mean_turning_degrees = sum(bearing_differences) / len(bearing_differences) if bearing_differences else 0
    
    return total_distance, sequence, mean_turning_degrees


def calculate_balanced_optimal_sequence(driver_pos, cluster_users, office_pos):
    """Calculate optimal sequence using balanced approach (50% distance + 50% direction)"""
    if len(cluster_users) <= 1:
        return cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)
    
    users_list = cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)
    
    # Calculate main route bearing
    main_route_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
    
    # Balanced scoring: exactly 50% distance, 50% direction
    def balanced_score(user):
        distance = haversine_distance(driver_pos[0], driver_pos[1], 
                                    user['latitude'], user['longitude'])
        
        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1], 
                                       user['latitude'], user['longitude'])
        
        bearing_diff = normalize_bearing_difference(user_bearing - main_route_bearing)
        bearing_diff_rad = math.radians(abs(bearing_diff))
        
        # Balanced components
        distance_score = distance
        direction_score = distance * (1 - math.cos(bearing_diff_rad))
        
        # True 50/50 balance
        combined_score = distance_score * 0.5 + direction_score * 0.5
        
        return (combined_score, user['user_id'])
    
    users_list.sort(key=balanced_score)
    
    # Apply balanced 2-opt optimization
    return apply_balanced_2opt(users_list, driver_pos, office_pos)


def apply_balanced_2opt(sequence, driver_pos, office_pos):
    """Apply balanced 2-opt improvements with moderate constraints"""
    if len(sequence) <= 2:
        return sequence
    
    improved = True
    max_iterations = 3
    iteration = 0
    
    # Balanced turning threshold
    max_turning_threshold = _config.get('MAX_TURNING_ANGLE', 47.5)
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        best_distance = calculate_sequence_distance(sequence, driver_pos, office_pos)
        best_turning_score = calculate_sequence_turning_score(sequence, driver_pos, office_pos)
        
        for i in range(len(sequence) - 1):
            for j in range(i + 2, len(sequence)):
                new_sequence = sequence[:i+1] + sequence[i+1:j+1][::-1] + sequence[j+1:]
                
                new_distance = calculate_sequence_distance(new_sequence, driver_pos, office_pos)
                new_turning_score = calculate_sequence_turning_score(new_sequence, driver_pos, office_pos)
                
                # Balanced acceptance criteria
                distance_improvement = (best_distance - new_distance) / best_distance
                turning_improvement = (best_turning_score - new_turning_score)
                turning_improvement_normalized = turning_improvement / max(best_turning_score, 1.0)
                
                # True 50/50 balanced improvement
                combined_improvement = distance_improvement * 0.5 + turning_improvement_normalized * 0.5
                
                if (combined_improvement > 0.003 and new_turning_score <= max_turning_threshold):
                    sequence = new_sequence
                    best_distance = new_distance
                    best_turning_score = new_turning_score
                    improved = True
                    break
            if improved:
                break
    
    return sequence


def calculate_sequence_distance(sequence, driver_pos, office_pos):
    """Calculate total distance for a sequence"""
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


def calculate_sequence_turning_score(sequence, driver_pos, office_pos):
    """Calculate turning score for a sequence"""
    if len(sequence) <= 1:
        return 0
    
    bearing_differences = []
    prev_bearing = None
    
    for i in range(len(sequence)):
        if i == 0:
            current_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                              sequence[i]['latitude'], sequence[i]['longitude'])
            if len(sequence) == 1:
                next_bearing = calculate_bearing(sequence[i]['latitude'], sequence[i]['longitude'],
                                               office_pos[0], office_pos[1])
                bearing_diff = bearing_difference(current_bearing, next_bearing)
                bearing_differences.append(bearing_diff)
            prev_bearing = current_bearing
            continue
        
        if i == len(sequence) - 1:
            current_bearing = calculate_bearing(sequence[i]['latitude'], sequence[i]['longitude'],
                                              office_pos[0], office_pos[1])
        else:
            current_bearing = calculate_bearing(sequence[i-1]['latitude'], sequence[i-1]['longitude'],
                                              sequence[i]['latitude'], sequence[i]['longitude'])
        
        if prev_bearing is not None:
            bearing_diff = bearing_difference(prev_bearing, current_bearing)
            bearing_differences.append(bearing_diff)
        
        prev_bearing = current_bearing
    
    return sum(bearing_differences) / len(bearing_differences) if bearing_differences else 0


def optimize_balanced_route_sequence(route, office_lat, office_lon):
    """Optimize route sequence using balanced approach"""
    if not route['assigned_users'] or len(route['assigned_users']) <= 1:
        return route
    
    users = route['assigned_users'].copy()
    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (office_lat, office_lon)
    
    # Convert to sequencing format
    users_for_sequencing = []
    for user in users:
        users_for_sequencing.append({
            'latitude': user['lat'],
            'longitude': user['lng'],
            'user_id': user['user_id']
        })
    
    # Calculate balanced optimal sequence
    optimized_sequence = calculate_balanced_optimal_sequence(driver_pos, users_for_sequencing, office_pos)
    
    # Convert back to original format
    final_sequence = []
    for seq_user in optimized_sequence:
        for orig_user in users:
            if orig_user['user_id'] == seq_user['user_id']:
                final_sequence.append(orig_user)
                break
    
    route['assigned_users'] = final_sequence
    return route


def update_balanced_route_metrics(route, office_lat, office_lon):
    """Update route metrics with balanced calculations"""
    if route['assigned_users']:
        driver_pos = (route['latitude'], route['longitude'])
        office_pos = (office_lat, office_lon)
        
        route['utilization'] = len(route['assigned_users']) / route['vehicle_type']
        route['turning_score'] = calculate_balanced_turning_score(route['assigned_users'], driver_pos, office_pos)
        route['tortuosity_ratio'] = calculate_balanced_tortuosity_ratio(route['assigned_users'], driver_pos, office_pos)
        route['direction_consistency'] = calculate_balanced_direction_consistency(route['assigned_users'], driver_pos, office_pos)
    else:
        route['utilization'] = 0
        route['turning_score'] = 0
        route['tortuosity_ratio'] = 1.0
        route['direction_consistency'] = 1.0


def calculate_balanced_turning_score(users, driver_pos, office_pos):
    """Calculate turning score with balanced approach"""
    if len(users) <= 1:
        return 0
    
    bearing_differences = []
    prev_bearing = None
    
    for i in range(len(users) + 1):
        if i == 0:
            current_bearing = calculate_bearing(driver_pos[0], driver_pos[1], users[0]['lat'], users[0]['lng'])
            if len(users) == 1:
                next_bearing = calculate_bearing(users[0]['lat'], users[0]['lng'], office_pos[0], office_pos[1])
                bearing_diff = bearing_difference(current_bearing, next_bearing)
                bearing_differences.append(bearing_diff)
            prev_bearing = current_bearing
            continue
        elif i == len(users):
            if len(users) > 0:
                current_bearing = calculate_bearing(users[i-1]['lat'], users[i-1]['lng'], office_pos[0], office_pos[1])
            else:
                continue
        else:
            current_bearing = calculate_bearing(users[i-1]['lat'], users[i-1]['lng'], users[i]['lat'], users[i]['lng'])
        
        if prev_bearing is not None:
            bearing_diff = bearing_difference(prev_bearing, current_bearing)
            bearing_differences.append(bearing_diff)
        
        prev_bearing = current_bearing
    
    return sum(bearing_differences) / len(bearing_differences) if bearing_differences else 0


def calculate_balanced_tortuosity_ratio(users, driver_pos, office_pos):
    """Calculate tortuosity ratio with balanced approach"""
    if not users:
        return 1.0
    
    # Actual route distance
    actual_distance = calculate_sequence_distance([{
        'latitude': u['lat'],
        'longitude': u['lng']
    } for u in users], driver_pos, office_pos)
    
    # Straight line distance via centroid
    centroid_lat = sum(u['lat'] for u in users) / len(users)
    centroid_lng = sum(u['lng'] for u in users) / len(users)
    
    straight_distance = (haversine_distance(driver_pos[0], driver_pos[1], centroid_lat, centroid_lng) +
                        haversine_distance(centroid_lat, centroid_lng, office_pos[0], office_pos[1]))
    
    return actual_distance / straight_distance if straight_distance > 0 else 1.0


def calculate_balanced_direction_consistency(users, driver_pos, office_pos):
    """Calculate direction consistency with balanced approach"""
    if len(users) <= 1:
        return 1.0
    
    main_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
    consistent_segments = 0
    total_segments = 0
    
    for i in range(len(users) + 1):
        if i == 0:
            segment_bearing = calculate_bearing(driver_pos[0], driver_pos[1], users[0]['lat'], users[0]['lng'])
        elif i == len(users):
            if len(users) > 0:
                segment_bearing = calculate_bearing(users[i-1]['lat'], users[i-1]['lng'], office_pos[0], office_pos[1])
            else:
                continue
        else:
            segment_bearing = calculate_bearing(users[i-1]['lat'], users[i-1]['lng'], users[i]['lat'], users[i]['lng'])
        
        bearing_diff = bearing_difference(segment_bearing, main_bearing)
        if bearing_diff <= 50:  # Balanced tolerance
            consistent_segments += 1
        total_segments += 1
    
    return consistent_segments / total_segments if total_segments > 0 else 1.0


# BALANCED LOCAL OPTIMIZATION
def balanced_local_optimization(routes, office_lat, office_lon):
    """Local optimization with balanced approach"""
    logger.info("🔧 Step 4: Balanced local optimization...")
    
    improved = True
    iterations = 0
    
    while improved and iterations < MAX_SWAP_ITERATIONS:
        improved = False
        iterations += 1
        
        # Optimize sequences within routes
        for i, route in enumerate(routes):
            original_turning = route.get('turning_score', 0)
            original_distance = calculate_balanced_route_distance(route, office_lat, office_lon)
            
            optimized_route = optimize_balanced_route_sequence(route, office_lat, office_lon)
            routes[i] = optimized_route
            
            new_turning = optimized_route.get('turning_score', 0)
            new_distance = calculate_balanced_route_distance(optimized_route, office_lat, office_lon)
            
            if abs(new_turning - original_turning) > 0.1 or abs(new_distance - original_distance) > 0.01:
                improved = True
        
        # Try balanced user swaps
        if balanced_user_swap(routes, office_lat, office_lon):
            improved = True
    
    logger.info(f"  ✅ Balanced local optimization completed in {iterations} iterations")
    return routes


def calculate_balanced_route_distance(route, office_lat, office_lon):
    """Calculate route distance for balanced optimization"""
    if not route['assigned_users']:
        return 0
    
    total_distance = 0
    driver_pos = (route['latitude'], route['longitude'])
    current_pos = driver_pos
    
    for user in route['assigned_users']:
        user_pos = (user['lat'], user['lng'])
        total_distance += haversine_distance(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
        current_pos = user_pos
    
    # Distance to office
    total_distance += haversine_distance(current_pos[0], current_pos[1], office_lat, office_lon)
    
    return total_distance


def balanced_user_swap(routes, office_lat, office_lon):
    """Try swapping users between routes with balanced quality control"""
    improvements = 0
    threshold = _config.get('SWAP_IMPROVEMENT_THRESHOLD', 0.75)
    
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            route1, route2 = routes[i], routes[j]
            
            if not route1['assigned_users'] or not route2['assigned_users']:
                continue
            
            # Calculate route centers for distance check
            center1 = calculate_balanced_route_center(route1)
            center2 = calculate_balanced_route_center(route2)
            route_distance = haversine_distance(center1[0], center1[1], center2[0], center2[1])
            
            # Skip distant routes
            if route_distance > MERGE_DISTANCE_KM * 2.0:
                continue
            
            # Try swapping users (balanced approach)
            for user1 in route1['assigned_users'][:]:
                if (len(route2['assigned_users']) + 1 <= route2['vehicle_type'] and 
                    len(route1['assigned_users']) > 1):
                    
                    # Calculate current metrics (balanced)
                    cost1_before = calculate_balanced_route_distance(route1, office_lat, office_lon)
                    cost2_before = calculate_balanced_route_distance(route2, office_lat, office_lon)
                    turn1_before = calculate_balanced_turning_score(route1['assigned_users'], 
                                                                   (route1['latitude'], route1['longitude']),
                                                                   (office_lat, office_lon))
                    turn2_before = calculate_balanced_turning_score(route2['assigned_users'],
                                                                   (route2['latitude'], route2['longitude']),
                                                                   (office_lat, office_lon))
                    
                    # Perform swap
                    route1['assigned_users'].remove(user1)
                    route2['assigned_users'].append(user1)
                    
                    # Optimize sequences
                    route1_opt = optimize_balanced_route_sequence(route1, office_lat, office_lon)
                    route2_opt = optimize_balanced_route_sequence(route2, office_lat, office_lon)
                    
                    # Calculate new metrics
                    cost1_after = calculate_balanced_route_distance(route1_opt, office_lat, office_lon)
                    cost2_after = calculate_balanced_route_distance(route2_opt, office_lat, office_lon)
                    turn1_after = calculate_balanced_turning_score(route1_opt['assigned_users'],
                                                                  (route1_opt['latitude'], route1_opt['longitude']),
                                                                  (office_lat, office_lon))
                    turn2_after = calculate_balanced_turning_score(route2_opt['assigned_users'],
                                                                  (route2_opt['latitude'], route2_opt['longitude']),
                                                                  (office_lat, office_lon))
                    
                    # Balanced improvement calculation (50% distance + 50% direction)
                    distance_improvement = (cost1_before + cost2_before) - (cost1_after + cost2_after)
                    turning_improvement = (turn1_before + turn2_before) - (turn1_after + turn2_after)
                    
                    # Balanced weighting
                    total_improvement = distance_improvement * 0.5 + (turning_improvement * 0.05) * 0.5
                    
                    if total_improvement > threshold:
                        improvements += 1
                        routes[i] = route1_opt
                        routes[j] = route2_opt
                    else:
                        # Revert swap
                        route2['assigned_users'].remove(user1)
                        route1['assigned_users'].append(user1)
    
    return improvements > 0


def calculate_balanced_route_center(route):
    """Calculate geometric center of route users"""
    if not route['assigned_users']:
        return (route['latitude'], route['longitude'])
    
    lats = [u['lat'] for u in route['assigned_users']]
    lngs = [u['lng'] for u in route['assigned_users']]
    return (np.mean(lats), np.mean(lngs))


# BALANCED GLOBAL OPTIMIZATION 
def balanced_global_optimization(routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon):
    """Global optimization with balanced approach"""
    logger.info("🌍 Step 5: Balanced global optimization...")
    
    # Phase 1: Fix single-user routes with balanced approach
    routes = fix_single_user_routes_balanced(routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon)
    
    # Phase 2: Fill routes with balanced quality control
    unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()
    routes = balanced_route_filling(routes, unassigned_users_df, assigned_user_ids, office_lat, office_lon)
    
    # Phase 3: Merge routes with balanced criteria
    routes = balanced_route_merging(routes, driver_df, office_lat, office_lon)
    
    # Phase 4: Handle remaining users
    remaining_unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids)]
    unassigned_list = handle_remaining_users_balanced(remaining_unassigned_users_df, driver_df, routes, office_lat, office_lon)
    
    logger.info("  ✅ Balanced global optimization completed")
    return routes, unassigned_list


def fix_single_user_routes_balanced(routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon):
    """Fix single-user routes using balanced approach"""
    logger.info("    🎯 Fixing single-user routes with balanced approach...")
    
    single_user_routes = [r for r in routes if len(r['assigned_users']) == 1]
    multi_user_routes = [r for r in routes if len(r['assigned_users']) > 1]
    
    routes_to_keep = []
    reassigned_count = 0
    
    # Try to merge single users into multi-user routes
    for route in multi_user_routes:
        if len(route['assigned_users']) >= route['vehicle_type']:
            routes_to_keep.append(route)
            continue
        
        route_center = calculate_balanced_route_center(route)
        route_bearing = calculate_balanced_route_bearing(route, office_lat, office_lon)
        
        for single_route in single_user_routes[:]:
            if len(route['assigned_users']) >= route['vehicle_type']:
                break
            
            single_user = single_route['assigned_users'][0]
            user_pos = (single_user['lat'], single_user['lng'])
            
            # Balanced compatibility check
            distance = haversine_distance(route_center[0], route_center[1], user_pos[0], user_pos[1])
            max_distance = MAX_FILL_DISTANCE_KM * 1.3  # Moderately lenient
            
            if distance <= max_distance:
                user_bearing = calculate_bearing(office_lat, office_lon, user_pos[0], user_pos[1])
                bearing_diff = bearing_difference(user_bearing, route_bearing)
                max_bearing_diff = MAX_BEARING_DIFFERENCE * 1.4  # Moderately lenient
                
                if bearing_diff <= max_bearing_diff:
                    # Test balanced quality
                    test_route = route.copy()
                    test_route['assigned_users'] = route['assigned_users'] + [single_user]
                    test_route = optimize_balanced_route_sequence(test_route, office_lat, office_lon)
                    
                    turning_score = calculate_balanced_turning_score(test_route['assigned_users'],
                                                                   (test_route['latitude'], test_route['longitude']),
                                                                   (office_lat, office_lon))
                    
                    # Balanced quality threshold
                    if turning_score <= 50:  # Moderate threshold
                        route['assigned_users'].append(single_user)
                        single_user_routes.remove(single_route)
                        reassigned_count += 1
                        logger.info(f"    ✅ Merged single user {single_user['user_id']} into route {route['driver_id']}")
        
        route = optimize_balanced_route_sequence(route, office_lat, office_lon)
        update_balanced_route_metrics(route, office_lat, office_lon)
        routes_to_keep.append(route)
    
    # Merge compatible single-user routes
    remaining_singles = single_user_routes[:]
    merged_singles = []
    
    while len(remaining_singles) >= 2:
        route1 = remaining_singles.pop(0)
        best_merge = None
        best_score = float('inf')
        
        for i, route2 in enumerate(remaining_singles):
            combined_capacity = max(route1['vehicle_type'], route2['vehicle_type'])
            if combined_capacity >= 2:
                user1_pos = (route1['assigned_users'][0]['lat'], route1['assigned_users'][0]['lng'])
                user2_pos = (route2['assigned_users'][0]['lat'], route2['assigned_users'][0]['lng'])
                distance = haversine_distance(user1_pos[0], user1_pos[1], user2_pos[0], user2_pos[1])
                
                user1_bearing = calculate_bearing(office_lat, office_lon, user1_pos[0], user1_pos[1])
                user2_bearing = calculate_bearing(office_lat, office_lon, user2_pos[0], user2_pos[1])
                bearing_diff = bearing_difference(user1_bearing, user2_bearing)
                
                # Balanced scoring
                score = distance * 0.5 + (bearing_diff * 0.05) * 0.5
                
                if (distance <= MERGE_DISTANCE_KM * 1.8 and bearing_diff <= 40 and score < best_score):
                    best_score = score
                    best_merge = (i, route2)
        
        if best_merge is not None:
            i, route2 = best_merge
            remaining_singles.pop(i)
            
            # Create balanced merged route
            center1 = (route1['assigned_users'][0]['lat'], route1['assigned_users'][0]['lng'])
            center2 = (route2['assigned_users'][0]['lat'], route2['assigned_users'][0]['lng'])
            combined_center = ((center1[0] + center2[0]) / 2, (center1[1] + center2[1]) / 2)
            
            dist1 = haversine_distance(route1['latitude'], route1['longitude'], combined_center[0], combined_center[1])
            dist2 = haversine_distance(route2['latitude'], route2['longitude'], combined_center[0], combined_center[1])
            
            better_route = route1 if dist1 <= dist2 else route2
            merged_route = better_route.copy()
            merged_route['assigned_users'] = route1['assigned_users'] + route2['assigned_users']
            merged_route['vehicle_type'] = max(route1['vehicle_type'], route2['vehicle_type'])
            
            merged_route = optimize_balanced_route_sequence(merged_route, office_lat, office_lon)
            update_balanced_route_metrics(merged_route, office_lat, office_lon)
            merged_singles.append(merged_route)
            reassigned_count += 1
    
    final_routes = routes_to_keep + merged_singles + remaining_singles
    logger.info(f"    ✅ Balanced approach: optimized {reassigned_count} single-user assignments")
    
    return final_routes


def calculate_balanced_route_bearing(route, office_lat, office_lon):
    """Calculate route bearing using balanced approach"""
    if not route['assigned_users']:
        return calculate_bearing(route['latitude'], route['longitude'], office_lat, office_lon)
    
    avg_lat = np.mean([u['lat'] for u in route['assigned_users']])
    avg_lng = np.mean([u['lng'] for u in route['assigned_users']])
    
    return calculate_bearing(avg_lat, avg_lng, office_lat, office_lon)


def balanced_route_filling(routes, unassigned_users_df, assigned_user_ids, office_lat, office_lon):
    """Fill routes with balanced quality control"""
    if unassigned_users_df.empty:
        return routes
    
    logger.info(f"    📋 Balanced route filling for {len(unassigned_users_df)} unassigned users")
    
    routes_by_util = sorted(routes, key=lambda r: r.get('utilization', 1.0))
    assignments_made = 0
    
    for route in routes_by_util:
        if len(route['assigned_users']) >= route['vehicle_type'] or unassigned_users_df.empty:
            continue
        
        route_center = calculate_balanced_route_center(route)
        route_bearing = calculate_balanced_route_bearing(route, office_lat, office_lon)
        
        compatible_users = []
        for _, user in unassigned_users_df.iterrows():
            if user['user_id'] in assigned_user_ids:
                continue
            
            distance = haversine_distance(route_center[0], route_center[1], user['latitude'], user['longitude'])
            
            # Balanced distance constraint
            utilization = len(route['assigned_users']) / route['vehicle_type']
            max_distance = MAX_FILL_DISTANCE_KM * (1.2 if utilization > 0.7 else 1.0)
            
            if distance <= max_distance:
                user_bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
                bearing_diff = bearing_difference(user_bearing, route_bearing)
                
                # Balanced bearing constraint
                max_bearing = MAX_BEARING_DIFFERENCE * 1.2
                
                if bearing_diff <= max_bearing:
                    compatible_users.append((user, distance, bearing_diff))
        
        if not compatible_users:
            continue
        
        # Sort by balanced quality score
        compatible_users.sort(key=lambda x: x[1] * 0.5 + (x[2] * 0.05) * 0.5)
        
        slots_available = route['vehicle_type'] - len(route['assigned_users'])
        users_added_this_route = 0
        
        for user, distance, bearing_diff in compatible_users:
            if users_added_this_route >= slots_available or user['user_id'] in assigned_user_ids:
                break
            
            # Test balanced quality
            test_route = route.copy()
            test_user_data = {
                'user_id': str(user['user_id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': float(user.get('office_distance', 0))
            }
            
            if pd.notna(user.get('first_name')):
                test_user_data['first_name'] = str(user['first_name'])
            if pd.notna(user.get('email')):
                test_user_data['email'] = str(user['email'])
            
            test_route['assigned_users'] = route['assigned_users'] + [test_user_data]
            test_route = optimize_balanced_route_sequence(test_route, office_lat, office_lon)
            
            new_turning = calculate_balanced_turning_score(test_route['assigned_users'],
                                                         (test_route['latitude'], test_route['longitude']),
                                                         (office_lat, office_lon))
            new_tortuosity = calculate_balanced_tortuosity_ratio(test_route['assigned_users'],
                                                               (test_route['latitude'], test_route['longitude']),
                                                               (office_lat, office_lon))
            
            # Balanced quality thresholds
            max_turning = 55  # Moderate threshold
            max_tortuosity = 1.6
            
            if new_turning <= max_turning and new_tortuosity <= max_tortuosity:
                route['assigned_users'].append(test_user_data)
                assigned_user_ids.add(user['user_id'])
                users_added_this_route += 1
                assignments_made += 1
                
                logger.info(f"    ✅ Balanced-assigned user {user['user_id']} to route {route['driver_id']}")
                
                route = optimize_balanced_route_sequence(route, office_lat, office_lon)
                update_balanced_route_metrics(route, office_lat, office_lon)
    
    logger.info(f"    📊 Balanced filling: assigned {assignments_made} users")
    return routes


def balanced_route_merging(routes, driver_df, office_lat, office_lon):
    """Merge routes using balanced criteria"""
    current_routes = routes.copy()
    merged_count = 0
    max_passes = 2
    
    for pass_num in range(max_passes):
        merged_routes_this_pass = []
        used_route_indices = set()
        pass_merges = 0
        
        # Consider underutilized routes for merging
        underutilized_routes = [
            (i, r) for i, r in enumerate(current_routes)
            if r.get('utilization', 1) < 0.75 and len(r['assigned_users']) > 0
        ]
        
        for orig_i, route_a in underutilized_routes:
            if orig_i in used_route_indices:
                continue
            
            best_merge_candidate = None
            best_candidate_index = None
            best_quality_score = float('inf')
            
            for orig_j, route_b in underutilized_routes:
                if orig_j in used_route_indices or orig_j == orig_i:
                    continue
                
                # Balanced compatibility check
                if balanced_merge_compatibility(route_a, route_b, office_lat, office_lon):
                    quality_score = calculate_balanced_merge_quality(route_a, route_b, office_lat, office_lon)
                    
                    if quality_score < best_quality_score:
                        best_quality_score = quality_score
                        best_merge_candidate = route_b
                        best_candidate_index = orig_j
            
            if best_merge_candidate is not None:
                merged_route = perform_balanced_merge(route_a, best_merge_candidate, office_lat, office_lon)
                merged_routes_this_pass.append(merged_route)
                used_route_indices.add(orig_i)
                used_route_indices.add(best_candidate_index)
                pass_merges += 1
                merged_count += 1
            else:
                merged_routes_this_pass.append(route_a)
                used_route_indices.add(orig_i)
        
        # Add routes that weren't considered
        for i, route in enumerate(current_routes):
            if i not in used_route_indices:
                merged_routes_this_pass.append(route)
        
        current_routes = merged_routes_this_pass
        
        if pass_merges == 0:
            break
    
    if merged_count > 0:
        logger.info(f"    🔗 Balanced merges: {merged_count}, Final routes: {len(current_routes)}")
    
    return current_routes


def balanced_merge_compatibility(route1, route2, office_lat, office_lon):
    """Check balanced merge compatibility"""
    total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
    max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])
    
    if total_users > max_capacity:
        return False
    
    # Balanced distance constraint
    center1 = calculate_balanced_route_center(route1)
    center2 = calculate_balanced_route_center(route2)
    distance = haversine_distance(center1[0], center1[1], center2[0], center2[1])
    
    if distance > MERGE_DISTANCE_KM * 1.2:
        return False
    
    # Balanced bearing constraint
    bearing1 = calculate_balanced_route_bearing(route1, office_lat, office_lon)
    bearing2 = calculate_balanced_route_bearing(route2, office_lat, office_lon)
    bearing_diff = bearing_difference(bearing1, bearing2)
    
    if bearing_diff > MAX_BEARING_DIFFERENCE * 1.3:
        return False
    
    # Both routes should be reasonably underutilized
    util1 = route1.get('utilization', 1)
    util2 = route2.get('utilization', 1)
    if util1 > 0.8 or util2 > 0.8:
        return False
    
    return True


def calculate_balanced_merge_quality(route1, route2, office_lat, office_lon):
    """Calculate balanced merge quality score"""
    all_users = route1['assigned_users'] + route2['assigned_users']
    
    # Choose better positioned driver
    combined_center = calculate_combined_route_center_balanced(route1, route2)
    dist1 = haversine_distance(route1['latitude'], route1['longitude'], combined_center[0], combined_center[1])
    dist2 = haversine_distance(route2['latitude'], route2['longitude'], combined_center[0], combined_center[1])
    
    better_route = route1 if dist1 <= dist2 else route2
    
    # Create test merged route
    test_route = better_route.copy()
    test_route['assigned_users'] = all_users
    test_route['vehicle_type'] = max(route1['vehicle_type'], route2['vehicle_type'])
    
    # Calculate balanced quality metrics
    turning_score = calculate_balanced_turning_score(test_route['assigned_users'],
                                                   (test_route['latitude'], test_route['longitude']),
                                                   (office_lat, office_lon))
    tortuosity = calculate_balanced_tortuosity_ratio(test_route['assigned_users'],
                                                   (test_route['latitude'], test_route['longitude']),
                                                   (office_lat, office_lon))
    
    utilization = len(all_users) / test_route['vehicle_type']
    
    # Balanced quality score (50% route efficiency + 50% capacity)
    route_score = turning_score + (tortuosity - 1.0) * 20
    capacity_score = (1.0 - utilization) * 50
    
    quality_score = route_score * 0.5 + capacity_score * 0.5
    
    return quality_score


def calculate_combined_route_center_balanced(route1, route2):
    """Calculate combined center of two routes"""
    all_users = route1['assigned_users'] + route2['assigned_users']
    if not all_users:
        return (0, 0)
    
    avg_lat = sum(u['lat'] for u in all_users) / len(all_users)
    avg_lng = sum(u['lng'] for u in all_users) / len(all_users)
    return (avg_lat, avg_lng)


def perform_balanced_merge(route1, route2, office_lat, office_lon):
    """Perform balanced merge"""
    all_users = route1['assigned_users'] + route2['assigned_users']
    combined_center = calculate_combined_route_center_balanced(route1, route2)
    
    dist1 = haversine_distance(route1['latitude'], route1['longitude'], combined_center[0], combined_center[1])
    dist2 = haversine_distance(route2['latitude'], route2['longitude'], combined_center[0], combined_center[1])
    
    better_route = route1 if dist1 <= dist2 else route2
    
    merged_route = better_route.copy()
    merged_route['assigned_users'] = all_users
    merged_route['vehicle_type'] = max(route1['vehicle_type'], route2['vehicle_type'])
    
    # Optimize and update
    merged_route = optimize_balanced_route_sequence(merged_route, office_lat, office_lon)
    update_balanced_route_metrics(merged_route, office_lat, office_lon)
    
    return merged_route


def handle_remaining_users_balanced(unassigned_users_df, driver_df, routes, office_lat, office_lon):
    """Handle remaining users with balanced approach"""
    if unassigned_users_df.empty:
        return []
    
    logger.info(f"    📋 Processing {len(unassigned_users_df)} remaining users with balanced approach")
    
    remaining_users = unassigned_users_df.copy()
    unassigned_list = []
    
    # Phase 1: Try to assign to existing routes with relaxed constraints
    if routes:
        users_assigned_to_existing = 0
        users_to_remove = []
        
        for _, user in remaining_users.iterrows():
            user_bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
            user_pos = (user['latitude'], user['longitude'])
            
            best_route = None
            best_score = float('inf')
            
            for route in routes:
                if len(route['assigned_users']) >= route['vehicle_type']:
                    continue
                
                route_center = calculate_balanced_route_center(route)
                route_bearing = calculate_balanced_route_bearing(route, office_lat, office_lon)
                
                # Relaxed balanced constraints
                distance = haversine_distance(route_center[0], route_center[1], user_pos[0], user_pos[1])
                max_distance = MAX_FILL_DISTANCE_KM * 2.0  # Very relaxed
                
                if distance > max_distance:
                    continue
                
                bearing_diff = bearing_difference(user_bearing, route_bearing)
                max_bearing_diff = MAX_BEARING_DIFFERENCE * 2.5  # Very relaxed
                
                if bearing_diff > max_bearing_diff:
                    continue
                
                # Test balanced quality with relaxed thresholds
                test_route = route.copy()
                test_user_data = {
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0))
                }
                
                if pd.notna(user.get('first_name')):
                    test_user_data['first_name'] = str(user['first_name'])
                if pd.notna(user.get('email')):
                    test_user_data['email'] = str(user['email'])
                
                test_route['assigned_users'] = route['assigned_users'] + [test_user_data]
                test_route = optimize_balanced_route_sequence(test_route, office_lat, office_lon)
                
                new_turning = calculate_balanced_turning_score(test_route['assigned_users'],
                                                             (test_route['latitude'], test_route['longitude']),
                                                             (office_lat, office_lon))
                
                # Very relaxed quality threshold
                if new_turning <= 80:  # Very lenient
                    score = distance * 0.5 + (bearing_diff * 0.02) * 0.5 + (new_turning * 0.02) * 0.5
                    
                    if score < best_score:
                        best_score = score
                        best_route = route
            
            if best_route is not None:
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
                
                best_route['assigned_users'].append(user_data)
                users_to_remove.append({'user_id': user['user_id']})
                users_assigned_to_existing += 1
                
                best_route = optimize_balanced_route_sequence(best_route, office_lat, office_lon)
                update_balanced_route_metrics(best_route, office_lat, office_lon)
                
                logger.info(f"    ✅ Balanced-assigned user {user['user_id']} to existing route {best_route['driver_id']}")
        
        if users_to_remove:
            assigned_user_ids_phase1 = {u['user_id'] for u in users_to_remove}
            remaining_users = remaining_users[~remaining_users['user_id'].isin(assigned_user_ids_phase1)]
            logger.info(f"    📊 Phase 1: Assigned {users_assigned_to_existing} users to existing routes")
    
    # Phase 2: Create new routes for remaining users
    if not remaining_users.empty and driver_df is not None:
        assigned_driver_ids = {route['driver_id'] for route in routes} if routes else set()
        available_drivers = driver_df[~driver_df['driver_id'].isin(assigned_driver_ids)]
        
        if not available_drivers.empty:
            logger.info(f"    🚀 Phase 2: Creating new routes for {len(remaining_users)} remaining users")
            
            # Convert remaining users to unassigned format
            for _, user in remaining_users.iterrows():
                user_data = {
                    'user_id': str(user['user_id']),
                    'latitude': float(user['latitude']),
                    'longitude': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0))
                }
                
                if pd.notna(user.get('first_name')):
                    user_data['first_name'] = str(user['first_name'])
                if pd.notna(user.get('email')):
                    user_data['email'] = str(user['email'])
                
                unassigned_list.append(user_data)
        else:
            # No drivers available, add to unassigned
            for _, user in remaining_users.iterrows():
                user_data = {
                    'user_id': str(user['user_id']),
                    'latitude': float(user['latitude']),
                    'longitude': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0))
                }
                
                if pd.notna(user.get('first_name')):
                    user_data['first_name'] = str(user['first_name'])
                if pd.notna(user.get('email')):
                    user_data['email'] = str(user['email'])
                
                unassigned_list.append(user_data)
    
    return unassigned_list


# MAIN ASSIGNMENT FUNCTION FOR BALANCED OPTIMIZATION
def run_assignment_balance(source_id: str, parameter: int = 1, string_param: str = ""):
    """
    Main assignment function with truly balanced optimization (50% route efficiency + 50% capacity)
    """
    start_time = time.time()
    
    # Clear cache files
    cache_files = [
        "drivers_and_routes.json", "drivers_and_routes_capacity.json", 
        "drivers_and_routes_balance.json", "drivers_and_routes_road_aware.json"
    ]
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
    
    # Reload configuration
    global _config
    _config = load_and_validate_config()
    
    # Update global variables
    global MAX_FILL_DISTANCE_KM, MERGE_DISTANCE_KM, MAX_BEARING_DIFFERENCE, UTILIZATION_PENALTY_PER_SEAT
    MAX_FILL_DISTANCE_KM = _config['MAX_FILL_DISTANCE_KM']
    MERGE_DISTANCE_KM = _config['MERGE_DISTANCE_KM'] 
    MAX_BEARING_DIFFERENCE = _config['MAX_BEARING_DIFFERENCE']
    UTILIZATION_PENALTY_PER_SEAT = _config['UTILIZATION_PENALTY_PER_SEAT']
    
    logger.info(f"🚀 Starting TRULY BALANCED OPTIMIZATION assignment for source_id: {source_id}")
    logger.info(f"📋 Parameter: {parameter}, String parameter: {string_param}")
    
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
                "optimization_mode": "balanced_optimization",
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
                "optimization_mode": "balanced_optimization",
                "parameter": parameter,
            }
        
        logger.info(f"📥 Data loaded - Users: {len(users)}, Total Drivers: {len(all_drivers)}")
        
        # Extract office coordinates and validate data
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        logger.info("✅ Data validation passed")
        
        # Prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)
        logger.info(f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")
        
        # STEP 1: Balanced geographic clustering
        user_df = create_balanced_geographic_clusters(user_df, office_lat, office_lon, _config)
        clustering_results = {"method": "balanced_" + _config['clustering_method'], 
                            "clusters": user_df['geo_cluster'].nunique()}
        
        # STEP 2: Balanced capacity sub-clustering
        user_df = create_balanced_capacity_subclusters(user_df, office_lat, office_lon, _config)
        
        # STEP 3: Balanced driver assignment
        routes, assigned_user_ids = assign_drivers_balanced_approach(user_df, driver_df, office_lat, office_lon)
        
        # STEP 4: Balanced local optimization
        routes = balanced_local_optimization(routes, office_lat, office_lon)
        
        # STEP 5: Balanced global optimization
        routes, unassigned_users = balanced_global_optimization(routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon)
        
        # Filter out empty routes
        filtered_routes = []
        for route in routes:
            if route['assigned_users'] and len(route['assigned_users']) > 0:
                filtered_routes.append(route)
            else:
                logger.info(f"  📋 Moving driver {route['driver_id']} with no users to unassigned drivers")
        
        routes = filtered_routes
        
        # Build unassigned drivers list
        assigned_driver_ids = {route['driver_id'] for route in routes}
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
        
        # Final metrics update
        for route in routes:
            update_balanced_route_metrics(route, office_lat, office_lon)
        
        execution_time = time.time() - start_time
        
        # Final verification
        total_users_in_api = len(users)
        users_assigned = sum(len(r['assigned_users']) for r in routes)
        users_unassigned = len(unassigned_users)
        users_accounted_for = users_assigned + users_unassigned
        
        logger.info(f"✅ Truly balanced optimization complete in {execution_time:.2f}s")
        logger.info(f"📊 Final routes: {len(routes)}")
        logger.info(f"🎯 Users assigned: {users_assigned}")
        logger.info(f"👥 Users unassigned: {users_unassigned}")
        logger.info(f"📋 User accounting: {users_accounted_for}/{total_users_in_api} users")
        
        return {
            "status": "true",
            "execution_time": execution_time,
            "data": routes,
            "unassignedUsers": unassigned_users,
            "unassignedDrivers": unassigned_drivers,
            "clustering_analysis": clustering_results,
            "optimization_mode": "balanced_optimization",
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
