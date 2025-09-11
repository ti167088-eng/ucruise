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
from logger_config import get_logger, start_session

# Start new session with cleared logs
logger = start_session()

warnings.filterwarnings('ignore')

# Import custom logging and progress tracking
from logger_config import get_logger
from progress_tracker import get_progress_tracker

# File context for logging
FILE_CONTEXT = "ASSIGN_CAPACITY.PY (CAPACITY OPTIMIZATION)"


# Load and validate configuration with capacity optimization settings
def load_and_validate_config():
    """Load configuration with capacity optimization settings"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger = get_logger()
        logger.warning(
            f"Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    # Always use capacity mode
    current_mode = "capacity_optimization"

    # Get capacity optimization configuration
    mode_configs = cfg.get("mode_configs", {})
    mode_config = mode_configs.get("capacity_optimization", {})

    logger = get_logger()
    logger.info(f"🎯 Using optimization mode: CAPACITY OPTIMIZATION")

    # Validate and set configuration with mode-specific overrides
    config = {}

    # Distance configurations with mode overrides (more lenient for capacity filling)
    config['MAX_FILL_DISTANCE_KM'] = max(
        0.1,
        float(
            mode_config.get("max_fill_distance_km",
                            cfg.get("max_fill_distance_km", 8.0))))
    config['MERGE_DISTANCE_KM'] = max(
        0.1,
        float(
            mode_config.get("merge_distance_km",
                            cfg.get("merge_distance_km", 5.0))))
    config['DBSCAN_EPS_KM'] = max(0.1, float(cfg.get("dbscan_eps_km", 2.5)))
    config['OVERFLOW_PENALTY_KM'] = max(
        0.0, float(cfg.get("overflow_penalty_km", 5.0)))
    config['DISTANCE_ISSUE_THRESHOLD'] = max(
        0.1, float(cfg.get("distance_issue_threshold_km", 12.0)))
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(
        0.0, float(cfg.get("swap_improvement_threshold_km", 1.0)))

    # Utilization thresholds (more aggressive for capacity)
    config['MIN_UTIL_THRESHOLD'] = max(
        0.0, min(1.0, float(cfg.get("min_util_threshold", 0.8))))
    config['LOW_UTILIZATION_THRESHOLD'] = max(
        0.0, min(1.0, float(cfg.get("low_utilization_threshold", 0.7))))

    # Integer configurations (must be positive)
    config['MIN_SAMPLES_DBSCAN'] = max(1, int(cfg.get("min_samples_dbscan",
                                                      2)))
    config['MAX_SWAP_ITERATIONS'] = max(1,
                                        int(cfg.get("max_swap_iterations", 5)))
    config['MAX_USERS_FOR_FALLBACK'] = max(
        1, int(cfg.get("max_users_for_fallback", 5)))
    config['FALLBACK_MIN_USERS'] = max(1, int(cfg.get("fallback_min_users",
                                                      3)))
    config['FALLBACK_MAX_USERS'] = max(1, int(cfg.get("fallback_max_users",
                                                      10)))

    # Angle configurations with mode overrides (more lenient for capacity)
    config['MAX_BEARING_DIFFERENCE'] = max(
        0,
        min(
            180,
            float(
                mode_config.get("max_bearing_difference",
                                cfg.get("max_bearing_difference", 45)))))
    config['MAX_TURNING_ANGLE'] = max(
        0,
        min(
            180,
            float(
                mode_config.get("max_allowed_turning_score",
                                cfg.get("max_allowed_turning_score", 60)))))

    # Cost penalties with mode overrides (prioritize capacity over route quality)
    config['UTILIZATION_PENALTY_PER_SEAT'] = max(
        0.0,
        float(
            mode_config.get("utilization_penalty_per_seat",
                            cfg.get("utilization_penalty_per_seat", 5.0))))

    # Office coordinates with environment variable fallbacks
    office_lat = float(
        os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489)))
    office_lon = float(
        os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))

    # Validate coordinate bounds
    if not (-90 <= office_lat <= 90):
        logger.warning(f"Invalid office latitude {office_lat}, using default")
        office_lat = 30.6810489
    if not (-180 <= office_lon <= 180):
        logger.warning(f"Invalid office longitude {office_lon}, using default")
        office_lon = 76.7260711

    config['OFFICE_LAT'] = office_lat
    config['OFFICE_LON'] = office_lon

    # Capacity optimization parameters
    config['optimization_mode'] = "capacity_optimization"
    config['aggressive_merging'] = mode_config.get('aggressive_merging', True)
    config['capacity_weight'] = mode_config.get('capacity_weight', 5.0)
    config['direction_weight'] = mode_config.get('direction_weight', 1.0)

    # Clustering and optimization parameters with mode overrides
    config['clustering_method'] = cfg.get('clustering_method', 'adaptive')
    config['min_cluster_size'] = max(2, cfg.get('min_cluster_size', 2))
    config['use_sweep_algorithm'] = cfg.get(
        'use_sweep_algorithm', False)  # Less directional for capacity
    config['angular_sectors'] = cfg.get('angular_sectors',
                                        6)  # Fewer sectors for larger groups
    config['max_users_per_initial_cluster'] = cfg.get(
        'max_users_per_initial_cluster', 12)
    config['max_users_per_cluster'] = cfg.get('max_users_per_cluster', 10)

    # Capacity optimization parameters
    config['zigzag_penalty_weight'] = mode_config.get(
        'zigzag_penalty_weight', cfg.get('zigzag_penalty_weight',
                                         0.5))  # Very low
    config['route_split_turning_threshold'] = cfg.get(
        'route_split_turning_threshold', 80)  # Very high
    config['max_tortuosity_ratio'] = cfg.get('max_tortuosity_ratio',
                                             2.0)  # Very lenient
    config['route_split_consistency_threshold'] = cfg.get(
        'route_split_consistency_threshold', 0.3)  # Very low
    config['merge_tortuosity_improvement_required'] = cfg.get(
        'merge_tortuosity_improvement_required', False)

    # Latitude conversion factor for distance normalization
    config['LAT_TO_KM'] = 111.0
    config['LON_TO_KM'] = 111.0 * math.cos(math.radians(office_lat))

    logger.info(
        f"   📊 Max bearing difference: {config['MAX_BEARING_DIFFERENCE']}°")
    logger.info(f"   📊 Max turning score: {config['MAX_TURNING_ANGLE']}°")
    logger.info(f"   📊 Max fill distance: {config['MAX_FILL_DISTANCE_KM']}km")
    logger.info(f"   📊 Capacity weight: {config['capacity_weight']}")
    logger.info(f"   📊 Direction weight: {config['direction_weight']}")

    return config


# Import all other functions from assignment.py (keeping the same structure)
# NOTE: These imports are placeholders and assume the existence of a file named `assignment.py`
# with the specified functions. In a real scenario, these functions would need to be
# either in the same file or properly imported from their respective modules.
try:
    from assignment import (
        validate_input_data, load_env_and_fetch_data, extract_office_coordinates,
        prepare_user_driver_dataframes, haversine_distance, bearing_difference,
        calculate_bearing_vectorized, calculate_bearing,
        calculate_bearings_and_features, coords_to_km, dbscan_clustering_metric,
        kmeans_clustering_metric, estimate_clusters, create_geographic_clusters,
        sweep_clustering, polar_sector_clustering, create_capacity_subclusters,
        create_bearing_aware_subclusters, calculate_bearing_spread,
        normalize_bearing_difference, calculate_sequence_distance,
        calculate_sequence_turning_score_improved,
        apply_strict_direction_aware_2opt, calculate_optimal_sequence_improved,
        split_cluster_by_bearing_metric, apply_route_splitting,
        split_route_by_bearing_improved, create_sub_route_improved,
        calculate_users_center_improved, local_optimization,
        optimize_route_sequence_improved, calculate_route_cost_improved,
        calculate_route_turning_score_improved,
        calculate_direction_consistency_improved, try_user_swap_improved,
        calculate_route_center_improved, update_route_metrics_improved,
        calculate_tortuosity_ratio_improved, global_optimization,
        fix_single_user_routes_improved, calculate_average_bearing_improved,
        quality_controlled_route_filling, quality_preserving_route_merging,
        strict_merge_compatibility_improved, calculate_merge_quality_score,
        perform_quality_merge_improved, enhanced_route_splitting,
        intelligent_route_splitting_improved, split_by_bearing_clusters_improved,
        split_by_distance_clusters_improved, create_split_routes_improved,
        find_best_driver_for_group, outlier_detection_and_reassignment,
        try_reassign_outlier, handle_remaining_users_improved,
        find_best_driver_for_cluster_improved, final_pass_merge,
        calculate_combined_route_center, _get_all_drivers_as_unassigned,
        _convert_users_to_unassigned_format)
except ImportError:
    logger.warning("Could not import all functions from 'assignment.py'. "
                   "Ensure it exists and contains the necessary functions.")
    # Define dummy functions to allow the script to run without crashing
    # In a real application, you would handle this import error more robustly
    def dummy_func(*args, **kwargs):
        pass
    def dummy_func_with_return(*args, **kwargs):
        return []
    def dummy_func_with_return_none(*args, **kwargs):
        return None
    def dummy_func_with_return_dict(*args, **kwargs):
        return {}
    def dummy_func_with_return_metrics(*args, **kwargs):
        return {"error": "Assignment failed"}

    globals().update({
        "validate_input_data": dummy_func,
        "load_env_and_fetch_data": lambda *a, **kw: {"users": [], "drivers": {"driversUnassigned": [], "driversAssigned": []}},
        "extract_office_coordinates": lambda *a, **kw: (30.6810489, 76.7260711),
        "prepare_user_driver_dataframes": lambda *a, **kw: (pd.DataFrame(), pd.DataFrame()),
        "haversine_distance": lambda *a, **kw: 0.0,
        "bearing_difference": lambda *a, **kw: 0.0,
        "calculate_bearing_vectorized": dummy_func,
        "calculate_bearing": lambda *a, **kw: 0.0,
        "calculate_bearings_and_features": dummy_func,
        "coords_to_km": lambda *a, **kw: (0.0, 0.0),
        "dbscan_clustering_metric": dummy_func,
        "kmeans_clustering_metric": dummy_func,
        "estimate_clusters": dummy_func,
        "create_geographic_clusters": lambda *a, **kw: pd.DataFrame(),
        "sweep_clustering": dummy_func,
        "polar_sector_clustering": dummy_func,
        "create_capacity_subclusters": lambda *a, **kw: pd.DataFrame(),
        "create_bearing_aware_subclusters": dummy_func,
        "calculate_bearing_spread": lambda *a, **kw: 0.0,
        "normalize_bearing_difference": lambda *a, **kw: 0.0,
        "calculate_sequence_distance": lambda *a, **kw: 0.0,
        "calculate_sequence_turning_score_improved": lambda *a, **kw: 0.0,
        "apply_strict_direction_aware_2opt": dummy_func,
        "calculate_optimal_sequence_improved": lambda *a, **kw: [],
        "split_cluster_by_bearing_metric": dummy_func,
        "apply_route_splitting": dummy_func,
        "split_route_by_bearing_improved": dummy_func,
        "create_sub_route_improved": dummy_func,
        "calculate_users_center_improved": lambda *a, **kw: (0.0, 0.0),
        "local_optimization": lambda *a, **kw: [],
        "optimize_route_sequence_improved": lambda *a, **kw: {},
        "calculate_route_cost_improved": lambda *a, **kw: 0.0,
        "calculate_route_turning_score_improved": lambda *a, **kw: 0.0,
        "calculate_direction_consistency_improved": lambda *a, **kw: 0.0,
        "try_user_swap_improved": lambda *a, **kw: False,
        "calculate_route_center_improved": lambda *a, **kw: (0.0, 0.0),
        "update_route_metrics_improved": dummy_func,
        "calculate_tortuosity_ratio_improved": lambda *a, **kw: 0.0,
        "global_optimization": lambda *a, **kw: ([], []),
        "fix_single_user_routes_improved": dummy_func,
        "calculate_average_bearing_improved": lambda *a, **kw: 0.0,
        "quality_controlled_route_filling": dummy_func,
        "quality_preserving_route_merging": dummy_func,
        "strict_merge_compatibility_improved": lambda *a, **kw: False,
        "calculate_merge_quality_score": lambda *a, **kw: 0.0,
        "perform_quality_merge_improved": dummy_func,
        "enhanced_route_splitting": dummy_func,
        "intelligent_route_splitting_improved": dummy_func,
        "split_by_bearing_clusters_improved": dummy_func,
        "split_by_distance_clusters_improved": dummy_func,
        "create_split_routes_improved": dummy_func,
        "find_best_driver_for_group": lambda *a, **kw: None,
        "outlier_detection_and_reassignment": dummy_func,
        "try_reassign_outlier": lambda *a, **kw: False,
        "handle_remaining_users_improved": lambda *a, **kw: [],
        "find_best_driver_for_cluster_improved": lambda *a, **kw: None,
        "final_pass_merge": dummy_func,
        "calculate_combined_route_center": lambda *a, **kw: (0.0, 0.0),
        "_get_all_drivers_as_unassigned": lambda *a, **kw: [],
        "_convert_users_to_unassigned_format": lambda *a, **kw: [],
        "analyze_assignment_quality": dummy_func_with_return_metrics,
    })


# Load validated configuration - always capacity optimization
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


def assign_drivers_by_priority_capacity_focused(user_df, driver_df, office_lat,
                                                office_lon):
    """
    ROUTE-FIRST APPROACH: Create optimal user groups first, then assign best drivers to each route
    """
    logger = get_logger()
    logger.info("🚗 Step 3: ROUTE-FIRST capacity-focused assignment...")

    # PHASE 1: Create optimal user routes based on capacity needs and geography
    logger.info("  🎯 PHASE 1: Creating optimal user routes without drivers")
    
    optimal_user_routes = create_capacity_optimized_user_routes(user_df, driver_df, office_lat, office_lon)
    
    # PHASE 2: Assign best available drivers to each route
    logger.info("  🎯 PHASE 2: Assigning best drivers to created routes")
    
    final_routes = assign_drivers_to_user_routes(optimal_user_routes, driver_df, office_lat, office_lon)

    logger.info(f"  ✅ ROUTE-FIRST assignment complete: {len(final_routes)} routes created")

    # Calculate final stats
    total_seats = sum(r['vehicle_type'] for r in final_routes)
    total_users = sum(len(r['assigned_users']) for r in final_routes)
    overall_utilization = (total_users / total_seats * 100) if total_seats > 0 else 0

    logger.info(f"  📊 Overall seat utilization: {total_users}/{total_seats} ({overall_utilization:.1f}%)")

    assigned_user_ids = set()
    for route in final_routes:
        assigned_user_ids.update(u['user_id'] for u in route['assigned_users'])

    return final_routes, assigned_user_ids


def create_capacity_optimized_user_routes(user_df, driver_df, office_lat, office_lon):
    """
    Create optimal user routes based on capacity needs and geography, without assigning drivers yet
    """
    logger = get_logger()
    
    # Get available vehicle capacities to guide route creation
    available_capacities = sorted(driver_df['capacity'].tolist(), reverse=True)
    capacity_counts = driver_df['capacity'].value_counts().to_dict()
    
    logger.info(f"  📊 Available capacities: {capacity_counts}")
    
    optimal_routes = []
    remaining_users = user_df.copy()
    
    # Strategy: Create routes that maximize capacity utilization
    for target_capacity in available_capacities:
        if remaining_users.empty or capacity_counts.get(target_capacity, 0) <= 0:
            continue
            
        # Create routes for this capacity size
        routes_for_capacity = create_routes_for_target_capacity(
            remaining_users, target_capacity, office_lat, office_lon
        )
        
        for route_users in routes_for_capacity:
            if len(route_users) >= max(2, target_capacity * 0.6):  # At least 60% utilization
                optimal_routes.append({
                    'target_capacity': target_capacity,
                    'users': route_users,
                    'utilization': len(route_users) / target_capacity
                })
                
                # Remove assigned users
                assigned_ids = {u['user_id'] for u in route_users}
                remaining_users = remaining_users[~remaining_users['user_id'].isin(assigned_ids)]
                
                # Decrease available count for this capacity
                capacity_counts[target_capacity] -= 1
                
                logger.info(f"    ✅ Created route with {len(route_users)} users for capacity {target_capacity} ({len(route_users)/target_capacity*100:.1f}%)")
                
                if capacity_counts[target_capacity] <= 0:
                    break
    
    # Handle remaining users by creating additional routes or merging into existing ones
    if not remaining_users.empty:
        logger.info(f"  🔄 Handling {len(remaining_users)} remaining users")
        
        # Try to add remaining users to existing routes with available capacity
        for _, user in remaining_users.iterrows():
            best_route = None
            best_score = float('inf')
            
            for route in optimal_routes:
                if len(route['users']) < route['target_capacity']:
                    # Calculate compatibility score
                    route_center = calculate_users_center_from_list(route['users'])
                    distance = haversine_distance(route_center[0], route_center[1],
                                                user['latitude'], user['longitude'])
                    
                    if distance < best_score and distance <= MAX_FILL_DISTANCE_KM * 2:
                        best_score = distance
                        best_route = route
            
            if best_route:
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
                    
                best_route['users'].append(user_data)
                best_route['utilization'] = len(best_route['users']) / best_route['target_capacity']
                logger.info(f"    ➕ Added user {user['user_id']} to existing route")
    
    logger.info(f"  📋 Created {len(optimal_routes)} optimal user routes")
    return optimal_routes


def create_routes_for_target_capacity(users_df, target_capacity, office_lat, office_lon):
    """
    Create routes for a specific target capacity using geographic clustering
    """
    if users_df.empty:
        return []
    
    # Convert to list format for easier handling
    users_list = []
    for _, user in users_df.iterrows():
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
        users_list.append(user_data)
    
    # Use DBSCAN clustering to group nearby users
    coords_km = []
    for user in users_list:
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'], office_lat, office_lon)
        coords_km.append([lat_km, lon_km])
    
    if len(coords_km) < 2:
        return [users_list] if users_list else []
    
    coords_km = np.array(coords_km)
    eps_km = _config.get('DBSCAN_EPS_KM', 2.5)
    
    # Adjust clustering to create groups close to target capacity
    dbscan = DBSCAN(eps=eps_km, min_samples=max(2, target_capacity // 3))
    labels = dbscan.fit_predict(coords_km)
    
    # Group users by cluster
    clustered_routes = {}
    for i, label in enumerate(labels):
        if label not in clustered_routes:
            clustered_routes[label] = []
        clustered_routes[label].append(users_list[i])
    
    # Split large clusters and merge small ones
    final_routes = []
    
    for cluster_users in clustered_routes.values():
        if len(cluster_users) > target_capacity:
            # Split large clusters
            num_splits = math.ceil(len(cluster_users) / target_capacity)
            chunk_size = len(cluster_users) // num_splits
            
            for i in range(0, len(cluster_users), chunk_size):
                chunk = cluster_users[i:i + chunk_size]
                if len(chunk) >= 2:  # Minimum viable route
                    final_routes.append(chunk)
        elif len(cluster_users) >= 2:
            final_routes.append(cluster_users)
    
    return final_routes


def assign_drivers_to_user_routes(user_routes, driver_df, office_lat, office_lon):
    """
    Assign the best available drivers to pre-created user routes
    """
    logger = get_logger()
    
    final_routes = []
    used_driver_ids = set()
    available_drivers = driver_df.copy()
    
    # Sort user routes by utilization (highest first) to prioritize well-utilized routes
    user_routes.sort(key=lambda r: r['utilization'], reverse=True)
    
    for user_route in user_routes:
        target_capacity = user_route['target_capacity']
        route_users = user_route['users']
        
        if not route_users:
            continue
            
        # Calculate route center for driver selection
        route_center = calculate_users_center_from_list(route_users)
        
        # Find best available driver for this route
        best_driver = find_best_driver_for_route(
            route_users, route_center, target_capacity, available_drivers, 
            used_driver_ids, office_lat, office_lon
        )
        
        if best_driver is not None:
            # Create final route with assigned driver
            route = {
                'driver_id': str(best_driver['driver_id']),
                'vehicle_id': str(best_driver.get('vehicle_id', '')),
                'vehicle_type': int(best_driver['capacity']),
                'latitude': float(best_driver['latitude']),
                'longitude': float(best_driver['longitude']),
                'assigned_users': []
            }
            
            # Add users to route in optimized sequence
            for user in route_users:
                user_data = {
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0))
                }
                if user.get('first_name'):
                    user_data['first_name'] = str(user['first_name'])
                if user.get('email'):
                    user_data['email'] = str(user['email'])
                route['assigned_users'].append(user_data)
            
            # Optimize sequence and update metrics
            route = optimize_route_sequence_improved(route, office_lat, office_lon)
            update_route_metrics_improved(route, office_lat, office_lon)
            
            final_routes.append(route)
            used_driver_ids.add(best_driver['driver_id'])
            
            utilization = len(route['assigned_users']) / route['vehicle_type'] * 100
            logger.info(f"  ✅ Assigned driver {best_driver['driver_id']} to route with {len(route['assigned_users'])} users ({utilization:.1f}%)")
        else:
            logger.warning(f"  ⚠️ No suitable driver found for route with {len(route_users)} users")
    
    return final_routes


def find_best_driver_for_route(route_users, route_center, target_capacity, 
                              available_drivers, used_driver_ids, office_lat, office_lon):
    """
    Find the best available driver for a pre-created route
    """
    best_driver = None
    best_score = float('inf')
    
    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids:
            continue
            
        # Must have sufficient capacity
        if driver['capacity'] < len(route_users):
            continue
        
        # Calculate suitability score
        distance_to_route = haversine_distance(
            driver['latitude'], driver['longitude'],
            route_center[0], route_center[1]
        )
        
        # Distance from driver to office
        distance_to_office = haversine_distance(
            driver['latitude'], driver['longitude'], office_lat, office_lon
        )
        
        # Calculate route bearing alignment
        driver_bearing = calculate_bearing(office_lat, office_lon, 
                                         driver['latitude'], driver['longitude'])
        route_bearing = calculate_bearing(office_lat, office_lon,
                                        route_center[0], route_center[1])
        bearing_diff = bearing_difference(driver_bearing, route_bearing)
        
        # Capacity utilization factor
        utilization = len(route_users) / driver['capacity']
        
        # Combined score (lower is better)
        distance_score = distance_to_route * 0.6  # Primary factor
        bearing_score = bearing_diff * 0.02       # Secondary factor
        utilization_score = (1.0 - utilization) * 3.0  # Penalty for underutilization
        priority_score = driver.get('priority', 1) * 0.1
        
        total_score = distance_score + bearing_score + utilization_score + priority_score
        
        if total_score < best_score:
            best_score = total_score
            best_driver = driver
    
    return best_driver


def calculate_users_center_from_list(users_list):
    """Calculate center point from a list of users"""
    if not users_list:
        return (0, 0)
    
    avg_lat = sum(u['latitude'] for u in users_list) / len(users_list)
    avg_lng = sum(u['longitude'] for u in users_list) / len(users_list)
    return (avg_lat, avg_lng)


def assign_best_driver_to_cluster_capacity_focused(cluster_users,
                                                   available_drivers,
                                                   used_driver_ids,
                                                   assigned_user_ids,
                                                   office_lat, office_lon):
    """Find and assign the best available driver with MAXIMUM capacity utilization focus"""
    cluster_size = len(cluster_users)

    best_driver = None
    best_score = -float(
        'inf')  # Changed to maximize score instead of minimize cost
    best_sequence = None

    # Ultra-aggressive capacity optimization weights
    capacity_weight = _config.get('capacity_weight',
                                  5.0) * 2  # Double the capacity weight
    direction_weight = _config.get('direction_weight',
                                   1.0) * 0.5  # Halve direction importance

    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids:
            continue

        # Accept any driver that can fit the users (no over-capacity rejection)
        if driver['capacity'] < cluster_size:
            continue

        # Calculate utilization score (the main factor)
        utilization = cluster_size / driver['capacity']
        utilization_score = utilization * capacity_weight * 10  # Massive utilization bonus

        # Distance penalty (but much smaller)
        route_cost, sequence, mean_turning_degrees = calculate_route_cost_capacity_focused(
            driver, cluster_users, office_lat, office_lon)
        distance_penalty = route_cost * 0.1  # Very small impact

        # Priority bonus (prefer higher priority drivers but with small impact)
        priority_bonus = (
            5 - driver['priority']) * 0.1  # Small priority preference

        # Minimal zigzag penalty
        zigzag_penalty = mean_turning_degrees * 0.01  # Almost no zigzag penalty

        # SEAT FILLING BONUS: Extra bonus for near-perfect utilization
        if utilization >= 0.9:  # 90%+ utilization
            seat_filling_bonus = 20
        elif utilization >= 0.8:  # 80%+ utilization
            seat_filling_bonus = 10
        elif utilization >= 0.7:  # 70%+ utilization
            seat_filling_bonus = 5
        else:
            seat_filling_bonus = 0

        # Calculate total score (higher is better for seat filling)
        total_score = utilization_score + seat_filling_bonus + priority_bonus - distance_penalty - zigzag_penalty

        if total_score > best_score:
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

        # Add ALL users from cluster (prioritize seat filling over sequence optimization)
        if hasattr(cluster_users, 'iterrows'):
            users_to_add = list(cluster_users.iterrows())
        else:
            users_to_add = [(i, user) for i, user in enumerate(cluster_users)]

        for _, user in users_to_add:
            # Add users to route with duplicate check
            if user['user_id'] in assigned_user_ids:
                logger = get_logger()
                logger.warning(
                    f"  ⚠️ User {user['user_id']} already assigned, skipping")
                continue

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

        # Quick sequence optimization (but don't remove users for it)
        route = optimize_route_sequence_improved(route, office_lat, office_lon)
        update_route_metrics_improved(route, office_lat, office_lon)

        utilization = len(route['assigned_users']) / route['vehicle_type']
        logger = get_logger()
        logger.info(
            f"    🚛 Assigned driver {best_driver['driver_id']}: {len(route['assigned_users'])}/{route['vehicle_type']} seats ({utilization*100:.1f}% utilization)"
        )

        return route

    return None


def calculate_route_cost_capacity_focused(driver, cluster_users, office_lat,
                                          office_lon):
    """Calculate route cost with capacity optimization focus (allows more zigzag)"""
    if len(cluster_users) == 0:
        return float('inf'), [], 0

    driver_pos = (driver['latitude'], driver['longitude'])
    office_pos = (office_lat, office_lon)

    # Get optimal pickup sequence with capacity focus (less strict on turning)
    sequence = calculate_optimal_sequence_capacity_focused(
        driver_pos, cluster_users, office_pos)

    # Calculate total route distance
    total_distance = 0
    bearing_differences = []

    # Driver to first pickup
    if sequence:
        first_user = sequence[0]
        total_distance += haversine_distance(driver_pos[0], driver_pos[1],
                                             first_user['latitude'],
                                             first_user['longitude'])

    # Between pickups - calculate bearing differences
    for i in range(len(sequence) - 1):
        current_user = sequence[i]
        next_user = sequence[i + 1]

        distance = haversine_distance(current_user['latitude'],
                                      current_user['longitude'],
                                      next_user['latitude'],
                                      next_user['longitude'])
        total_distance += distance

        # Calculate bearing difference between segments (less penalty)
        if i == 0:
            prev_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                             current_user['latitude'],
                                             current_user['longitude'])
        else:
            prev_pos = (sequence[i - 1]['latitude'],
                        sequence[i - 1]['longitude'])
            prev_bearing = calculate_bearing(prev_pos[0], prev_pos[1],
                                             current_user['latitude'],
                                             current_user['longitude'])

        next_bearing = calculate_bearing(current_user['latitude'],
                                         current_user['longitude'],
                                         next_user['latitude'],
                                         next_user['longitude'])

        bearing_diff = bearing_difference(prev_bearing, next_bearing)
        bearing_differences.append(bearing_diff)

    # Last pickup to office
    if sequence:
        last_user = sequence[-1]
        total_distance += haversine_distance(last_user['latitude'],
                                             last_user['longitude'],
                                             office_lat, office_lon)

    # Calculate mean turning angle (but give it less weight)
    mean_turning_degrees = sum(bearing_differences) / len(
        bearing_differences) if bearing_differences else 0

    return total_distance, sequence, mean_turning_degrees


def calculate_optimal_sequence_capacity_focused(driver_pos, cluster_users,
                                                office_pos):
    """Calculate sequence prioritizing capacity utilization over route efficiency"""
    if len(cluster_users) <= 1:
        return cluster_users.to_dict('records') if hasattr(
            cluster_users, 'to_dict') else list(cluster_users)

    users_list = cluster_users.to_dict('records') if hasattr(
        cluster_users, 'to_dict') else list(cluster_users)

    # For capacity mode, use simpler distance-based sorting instead of bearing projection
    def distance_from_driver_score(user):
        distance = haversine_distance(driver_pos[0], driver_pos[1],
                                      user['latitude'], user['longitude'])
        return (distance, user['user_id'])  # Sort by distance, then by user_id

    users_list.sort(key=distance_from_driver_score)

    # Apply much more lenient 2-opt with focus on distance rather than turning
    return apply_lenient_2opt_capacity_focused(users_list, driver_pos,
                                               office_pos)


def is_directionally_consistent(users, driver_pos, office_pos, lenient=False):
    """Check if a group of users maintains directional consistency - very lenient for capacity mode"""
    if len(users) <= 2:  # Always allow 1-2 users
        return True

    office_lat, office_lon = office_pos

    # Calculate bearings from office to each user
    bearings = []
    for user in users:
        bearing = calculate_bearing(office_lat, office_lon, user['latitude'],
                                    user['longitude'])
        bearings.append(bearing)

    # Very lenient bearing spread for capacity optimization
    max_bearing_spread = 120 if lenient else 90  # Much more lenient

    if len(bearings) >= 2:
        bearing_diffs = []
        for i in range(len(bearings)):
            for j in range(i + 1, len(bearings)):
                diff = bearing_difference(bearings[i], bearings[j])
                bearing_diffs.append(diff)

        max_diff = max(bearing_diffs) if bearing_diffs else 0
        if max_diff > max_bearing_spread:
            return False

    return True


def is_directionally_consistent_from_dicts(user_dicts,
                                           driver_pos,
                                           office_pos,
                                           lenient=False):
    """Check directional consistency for user dictionaries - very lenient for capacity mode"""
    if len(user_dicts) <= 2:  # Always allow 1-2 users
        return True

    office_lat, office_lon = office_pos

    # Calculate bearings from office to each user
    bearings = []
    for user in user_dicts:
        bearing = calculate_bearing(office_lat, office_lon, user['lat'],
                                    user['lng'])
        bearings.append(bearing)

    # Very lenient bearing spread for capacity optimization
    max_bearing_spread = 120 if lenient else 90  # Much more lenient

    if len(bearings) >= 2:
        bearing_diffs = []
        for i in range(len(bearings)):
            for j in range(i + 1, len(bearings)):
                diff = bearing_difference(bearings[i], bearings[j])
                bearing_diffs.append(diff)

        max_diff = max(bearing_diffs) if bearing_diffs else 0
        if max_diff > max_bearing_spread:
            return False

    return True


def apply_lenient_2opt_capacity_focused(sequence, driver_pos, office_pos):
    """Apply very lenient 2-opt improvements focused on distance rather than turning"""
    if len(sequence) <= 2:
        return sequence

    improved = True
    max_iterations = 2  # Fewer iterations for capacity mode
    iteration = 0

    # Very lenient turning angle threshold
    max_turning_threshold = _config.get('MAX_TURNING_ANGLE',
                                        60) * 2  # Much more lenient

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        best_distance = calculate_sequence_distance(sequence, driver_pos,
                                                    office_pos)

        for i in range(len(sequence) - 1):
            for j in range(i + 2, len(sequence)):
                # Try 2-opt swap
                new_sequence = sequence[:i +
                                        1] + sequence[i + 1:j +
                                                      1][::-1] + sequence[j +
                                                                          1:]

                # Check if new sequence maintains directional consistency
                if not is_directionally_consistent(
                        new_sequence, driver_pos, office_pos, lenient=True):
                    continue

                # Calculate new metrics
                new_distance = calculate_sequence_distance(
                    new_sequence, driver_pos, office_pos)

                # For capacity mode, only care about distance improvement
                if new_distance < best_distance * 0.99:  # Even more lenient distance improvement
                    sequence = new_sequence
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break

    return sequence


def final_pass_merge_capacity_focused(routes, config, office_lat, office_lon):
    """
    DIRECTIONAL capacity-focused final-pass merge: Fill seats while maintaining directional consistency
    """
    logger = get_logger()
    logger.info("🔄 Step 6: DIRECTIONAL capacity-focused final-pass merge...")

    merged_routes = []
    used = set()

    # Balanced thresholds for capacity + direction
    MERGE_BEARING_THRESHOLD = 60  # More lenient directional requirement for capacity
    MERGE_DISTANCE_KM = config.get("MERGE_DISTANCE_KM",
                                   5.0) * 2  # More distance tolerance for capacity

    for i, r1 in enumerate(routes):
        if i in used:
            continue

        best_merge = None
        best_total_utilization = len(r1['assigned_users']) / r1[
            'vehicle_type']  # Start with current utilization

        for j, r2 in enumerate(routes):
            if j <= i or j in used:
                continue

            # 1. Check direction similarity (more lenient for capacity)
            b1 = calculate_average_bearing_improved(r1, office_lat, office_lon)
            b2 = calculate_average_bearing_improved(r2, office_lat, office_lon)
            bearing_diff = bearing_difference(b1, b2)

            # More lenient for capacity optimization
            if bearing_diff > MERGE_BEARING_THRESHOLD:
                continue

            # 2. Check centroid distance (more lenient)
            c1 = calculate_route_center_improved(r1)
            c2 = calculate_route_center_improved(r2)
            centroid_distance = haversine_distance(c1[0], c1[1], c2[0], c2[1])

            if centroid_distance > MERGE_DISTANCE_KM:
                continue

            # 3. Check if we can fit all users in the larger vehicle
            total_users = len(r1['assigned_users']) + len(r2['assigned_users'])
            max_capacity = max(r1['vehicle_type'], r2['vehicle_type'])

            if total_users > max_capacity:
                continue

            # 4. Check directional consistency of merged route (more lenient)
            all_users = r1['assigned_users'] + r2['assigned_users']
            driver_pos = (r1['latitude'], r1['longitude']
                          ) if r1['vehicle_type'] >= r2['vehicle_type'] else (
                              r2['latitude'], r2['longitude'])

            if not is_directionally_consistent_from_dicts(
                    all_users, driver_pos,
                (office_lat, office_lon), lenient=True):
                continue

            # 5. Calculate total utilization after merge
            merged_utilization = total_users / max_capacity

            # Prioritize higher utilization while maintaining direction
            if merged_utilization > best_total_utilization:
                # Choose the driver with larger capacity or better position
                if r1['vehicle_type'] >= r2['vehicle_type']:
                    better_route = r1
                else:
                    better_route = r2

                # Create test merged route
                test_route = better_route.copy()
                test_route['assigned_users'] = all_users
                test_route['vehicle_type'] = max_capacity

                # Quick sequence optimization with directional check
                test_route = optimize_route_sequence_improved(
                    test_route, office_lat, office_lon)

                # Accept merge if directional and has good utilization
                best_total_utilization = merged_utilization
                best_merge = (j, test_route)

        if best_merge is not None:
            j, merged_route = best_merge
            merged_routes.append(merged_route)
            used.add(i)
            used.add(j)

            utilization_pct = len(merged_route['assigned_users']
                                  ) / merged_route['vehicle_type'] * 100
            logger.info(
                f"  🧭 DIRECTIONAL merge: routes {r1['driver_id']} + {routes[j]['driver_id']} = {len(merged_route['assigned_users'])}/{merged_route['vehicle_type']} seats ({utilization_pct:.1f}%)"
            )
        else:
            merged_routes.append(r1)
            used.add(i)

    # Final statistics
    total_seats = sum(r['vehicle_type'] for r in merged_routes)
    total_users = sum(len(r['assigned_users']) for r in merged_routes)
    overall_utilization = (total_users / total_seats *
                           100) if total_seats > 0 else 0

    logger.info(
        f"  🎯 DIRECTIONAL capacity merge complete: {len(routes)} → {len(merged_routes)} routes"
    )
    logger.info(
        f"  🧭 Final directional seat utilization: {total_users}/{total_seats} ({overall_utilization:.1f}%)"
    )

    return merged_routes


def global_directional_optimization(routes, office_lat, office_lon):
    """
    Global optimization that swaps users between routes to improve directional consistency
    while maintaining capacity constraints
    """
    logger = get_logger()
    logger.info(f"  🔄 Starting global directional optimization on {len(routes)} routes")
    
    improvements_made = 0
    max_iterations = 3
    
    for iteration in range(max_iterations):
        iteration_improvements = 0
        logger.info(f"    🔄 Iteration {iteration + 1}/{max_iterations}")
        
        # Calculate initial quality metrics for all routes
        route_qualities = []
        for i, route in enumerate(routes):
            if len(route['assigned_users']) == 0:
                continue
                
            # Calculate directional consistency
            consistency = calculate_direction_consistency_improved(
                route['assigned_users'],
                (route['latitude'], route['longitude']),
                (office_lat, office_lon)
            )
            
            # Calculate turning score
            turning = calculate_route_turning_score_improved(
                route['assigned_users'],
                (route['latitude'], route['longitude']),
                (office_lat, office_lon)
            )
            
            route_qualities.append((i, consistency, turning, len(route['assigned_users'])))
        
        # Sort by quality - worst routes first
        route_qualities.sort(key=lambda x: (x[1], -x[2]))  # Low consistency, high turning = bad
        
        # Try to improve the worst routes by swapping users
        for route_idx, consistency, turning, user_count in route_qualities[:len(routes)//2]:
            if consistency > 0.7 and turning < 45:  # Already good enough
                continue
                
            route = routes[route_idx]
            route_bearing = calculate_average_bearing_improved(route, office_lat, office_lon)
            
            # Find users in this route that don't fit the direction
            outlier_users = []
            good_users = []
            
            for user in route['assigned_users']:
                user_bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])
                bearing_diff = bearing_difference(route_bearing, user_bearing)
                
                if bearing_diff > 35:  # User doesn't fit route direction
                    outlier_users.append(user)
                else:
                    good_users.append(user)
            
            if not outlier_users:
                continue
                
            # Try to swap outlier users with better-fitting users from other routes
            for outlier_user in outlier_users:
                best_swap = None
                best_improvement = 0
                outlier_bearing = calculate_bearing(office_lat, office_lon, outlier_user['lat'], outlier_user['lng'])
                
                # Look for a better route for this outlier user
                for other_idx, other_route in enumerate(routes):
                    if other_idx == route_idx or len(other_route['assigned_users']) == 0:
                        continue
                        
                    other_bearing = calculate_average_bearing_improved(other_route, office_lat, office_lon)
                    outlier_fit = bearing_difference(outlier_bearing, other_bearing)
                    
                    if outlier_fit > 35:  # Still doesn't fit
                        continue
                    
                    # Check if other route has capacity or if we can swap
                    if len(other_route['assigned_users']) < other_route['vehicle_type']:
                        # Direct move possible
                        improvement = 40 - outlier_fit  # How much better it fits
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_swap = (other_idx, None, improvement)
                    else:
                        # Look for a user to swap back
                        for other_user in other_route['assigned_users']:
                            other_user_bearing = calculate_bearing(office_lat, office_lon, other_user['lat'], other_user['lng'])
                            
                            # How well would other_user fit in original route?
                            other_user_fit = bearing_difference(other_user_bearing, route_bearing)
                            
                            # How well would outlier fit in other route?
                            outlier_fit_new = bearing_difference(outlier_bearing, other_bearing)
                            
                            # Calculate total improvement
                            current_badness = bearing_difference(outlier_bearing, route_bearing) + bearing_difference(other_user_bearing, other_bearing)
                            new_badness = outlier_fit_new + other_user_fit
                            improvement = current_badness - new_badness
                            
                            if improvement > best_improvement and improvement > 10:  # Significant improvement
                                best_improvement = improvement
                                best_swap = (other_idx, other_user, improvement)
                
                # Execute the best swap found
                if best_swap:
                    other_route_idx, swap_user, improvement = best_swap
                    other_route = routes[other_route_idx]
                    
                    # Remove outlier from original route
                    route['assigned_users'].remove(outlier_user)
                    
                    if swap_user:
                        # Swap users
                        other_route['assigned_users'].remove(swap_user)
                        route['assigned_users'].append(swap_user)
                        logger.info(f"    🔄 Swapped user {outlier_user['user_id']} (route {route['driver_id']}) with user {swap_user['user_id']} (route {other_route['driver_id']})")
                    else:
                        logger.info(f"    ➡️ Moved user {outlier_user['user_id']} from route {route['driver_id']} to route {other_route['driver_id']}")
                    
                    # Add outlier to new route
                    other_route['assigned_users'].append(outlier_user)
                    
                    # Re-optimize sequences for both routes
                    route = optimize_route_sequence_improved(route, office_lat, office_lon)
                    other_route = optimize_route_sequence_improved(other_route, office_lat, office_lon)
                    update_route_metrics_improved(route, office_lat, office_lon)
                    update_route_metrics_improved(other_route, office_lat, office_lon)
                    
                    routes[route_idx] = route
                    routes[other_route_idx] = other_route
                    
                    iteration_improvements += 1
                    improvements_made += 1
                    break  # Try next outlier user
        
        logger.info(f"    ✅ Iteration {iteration + 1}: {iteration_improvements} improvements")
        
        if iteration_improvements == 0:
            break  # No more improvements possible
    
    # Clean up empty routes
    non_empty_routes = [route for route in routes if len(route['assigned_users']) > 0]
    removed_routes = len(routes) - len(non_empty_routes)
    
    if removed_routes > 0:
        logger.info(f"  🧹 Removed {removed_routes} empty routes after optimization")
    
    logger.info(f"  ✅ Global directional optimization complete: {improvements_made} total improvements")
    
    # Calculate final quality metrics
    good_routes = 0
    total_routes = len(non_empty_routes)
    
    for route in non_empty_routes:
        if len(route['assigned_users']) == 0:
            continue
            
        consistency = calculate_direction_consistency_improved(
            route['assigned_users'],
            (route['latitude'], route['longitude']),
            (office_lat, office_lon)
        )
        
        turning = calculate_route_turning_score_improved(
            route['assigned_users'],
            (route['latitude'], route['longitude']),
            (office_lat, office_lon)
        )
        
        if consistency > 0.6 and turning < 50:  # Acceptable quality
            good_routes += 1
    
    quality_percentage = (good_routes / total_routes * 100) if total_routes > 0 else 0
    logger.info(f"  📊 Final route quality: {good_routes}/{total_routes} routes ({quality_percentage:.1f}%) meet quality standards")
    
    return non_empty_routes


def analyze_assignment_quality(result):
    """Analyze the quality of the assignment with capacity focus"""
    logger = get_logger()

    if result['status'] != 'true':
        return {"error": "Assignment failed"}

    routes = result['data']
    unassigned_users = result.get('unassignedUsers', [])

    # Capacity analysis
    total_capacity = sum(r['vehicle_type'] for r in routes)
    total_assigned = sum(len(r['assigned_users']) for r in routes)
    overall_utilization = (total_assigned / total_capacity * 100) if total_capacity > 0 else 0

    # Route utilization distribution
    utilizations = []
    for route in routes:
        util = len(route['assigned_users']) / route['vehicle_type']
        utilizations.append(util)

    avg_utilization = np.mean(utilizations) if utilizations else 0

    quality_metrics = {
        "capacity_metrics": {
            "total_capacity": total_capacity,
            "total_assigned": total_assigned,
            "overall_utilization": overall_utilization,
            "average_route_utilization": avg_utilization * 100,
            "unassigned_count": len(unassigned_users),
            "assignment_rate": (total_assigned / (total_assigned + len(unassigned_users)) * 100) if (total_assigned + len(unassigned_users)) > 0 else 0
        },
        "route_quality": {
            "total_routes": len(routes),
            "avg_users_per_route": total_assigned / len(routes) if routes else 0,
            "single_user_routes": len([r for r in routes if len(r['assigned_users']) == 1]),
            "full_routes": len([r for r in routes if len(r['assigned_users']) == r['vehicle_type']])
        }
    }

    logger.info(f"📊 Assignment Quality Analysis:")
    logger.info(f"   🎯 Overall utilization: {overall_utilization:.1f}%")
    logger.info(f"   📈 Assignment rate: {quality_metrics['capacity_metrics']['assignment_rate']:.1f}%")
    logger.info(f"   🚗 Total routes: {len(routes)}")
    logger.info(f"   ❌ Unassigned users: {len(unassigned_users)}")

    return quality_metrics


# MAIN ASSIGNMENT FUNCTION FOR CAPACITY OPTIMIZATION
def run_assignment_capacity(source_id: str,
                            parameter: int = 1,
                            string_param: str = ""):
    """
    Main assignment function optimized for capacity utilization:
    - Prioritizes filling vehicle seats over route efficiency
    - Allows more flexible routes and higher turning angles
    - Maximizes utilization across all vehicles
    - Reduces unassigned users through aggressive seat filling
    """
    start_time = time.time()

    # Clear any cached data files to ensure fresh assignment
    cache_files = [
        "drivers_and_routes.json", "drivers_and_routes_capacity.json",
        "drivers_and_routes_balance.json", "drivers_and_routes_road_aware.json"
    ]

    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)

    # Reload configuration for capacity optimization
    global _config
    _config = load_and_validate_config()

    # Update global variables from new config
    global MAX_FILL_DISTANCE_KM, MERGE_DISTANCE_KM, MAX_BEARING_DIFFERENCE, UTILIZATION_PENALTY_PER_SEAT
    MAX_FILL_DISTANCE_KM = _config['MAX_FILL_DISTANCE_KM']
    MERGE_DISTANCE_KM = _config['MERGE_DISTANCE_KM']
    MAX_BEARING_DIFFERENCE = _config['MAX_BEARING_DIFFERENCE']
    UTILIZATION_PENALTY_PER_SEAT = _config['UTILIZATION_PENALTY_PER_SEAT']

    logger = get_logger()
    logger.info(
        f"🚀 Starting CAPACITY OPTIMIZATION assignment for source_id: {source_id}"
    )
    logger.info(f"📋 Parameter: {parameter}, String parameter: {string_param}")

    try:
        # Load and validate data
        data = load_env_and_fetch_data(source_id, parameter, string_param)

        # Edge case handling
        users = data.get('users', [])
        if not users:
            logger.warning("No users found - returning empty assignment")
            return {
                "status": "true",
                "execution_time": time.time() - start_time,
                "data": [],
                "unassignedUsers": [],
                "unassignedDrivers": _get_all_drivers_as_unassigned(data),
                "clustering_analysis": {
                    "method": "No Users",
                    "clusters": 0
                },
                "optimization_mode": "capacity_optimization",
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
            logger.warning("No drivers available - all users unassigned")
            unassigned_users = _convert_users_to_unassigned_format(users)
            return {
                "status": "true",
                "execution_time": time.time() - start_time,
                "data": [],
                "unassignedUsers": unassigned_users,
                "unassignedDrivers": [],
                "clustering_analysis": {
                    "method": "No Drivers",
                    "clusters": 0
                },
                "optimization_mode": "capacity_optimization",
                "parameter": parameter,
            }

        logger.info(
            f"📥 Data loaded - Users: {len(users)}, Total Drivers: {len(all_drivers)}"
        )

        # Extract office coordinates and validate data
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        logger.info("✅ Data validation passed")

        # Prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)

        logger.info(
            f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}"
        )

        # STEP 1: Geographic clustering (less strict for capacity mode)
        user_df = create_geographic_clusters(user_df, office_lat, office_lon,
                                             _config)
        clustering_results = {
            "method": "capacity_focused_" + _config['clustering_method'],
            "clusters": user_df['geo_cluster'].nunique()
        }

        # STEP 2: Capacity-based sub-clustering (more lenient for capacity filling)
        user_df = create_capacity_subclusters(user_df, office_lat, office_lon,
                                              _config)

        # STEP 3: Capacity-focused driver assignment
        routes, assigned_user_ids = assign_drivers_by_priority_capacity_focused(
            user_df, driver_df, office_lat, office_lon)

        # STEP 4: Local optimization (less strict on turning)
        routes = local_optimization(routes, office_lat, office_lon)

        # STEP 5: Aggressive global optimization for capacity filling
        routes, unassigned_users = global_optimization(routes, user_df,
                                                       assigned_user_ids,
                                                       driver_df, office_lat,
                                                       office_lon)

        # STEP 6: Aggressive final-pass merge for maximum capacity utilization
        routes = final_pass_merge_capacity_focused(routes, _config, office_lat,
                                                   office_lon)

        # STEP 7: EMERGENCY ASSIGNMENT PASS - Try to assign remaining unassigned users more aggressively
        if unassigned_users:
            logger.info(f"🚨 EMERGENCY PASS: Attempting to assign {len(unassigned_users)} remaining users")

            emergency_assignments = 0
            remaining_unassigned = []

            for unassigned_user in unassigned_users:
                user_assigned = False

                # Try to fit into ANY route with available capacity, regardless of direction/distance
                for route in routes:
                    if len(route['assigned_users']) < route['vehicle_type']:
                        # Add user to this route
                        user_data = {
                            'user_id': str(unassigned_user['user_id']),
                            'lat': float(unassigned_user['lat']),
                            'lng': float(unassigned_user['lng']),
                            'office_distance': float(unassigned_user.get('office_distance', 0))
                        }

                        if unassigned_user.get('first_name'):
                            user_data['first_name'] = str(unassigned_user['first_name'])
                        if unassigned_user.get('email'):
                            user_data['email'] = str(unassigned_user['email'])

                        route['assigned_users'].append(user_data)
                        emergency_assignments += 1
                        user_assigned = True

                        logger.info(f"🚨 EMERGENCY: Assigned user {unassigned_user['user_id']} to route {route['driver_id']}")
                        break

                if not user_assigned:
                    # Try to create new route with available driver
                    assigned_driver_ids = {route['driver_id'] for route in routes}
                    available_emergency_drivers = driver_df[~driver_df['driver_id'].isin(assigned_driver_ids)]

                    if not available_emergency_drivers.empty:
                        emergency_driver = available_emergency_drivers.iloc[0]

                        user_data = {
                            'user_id': str(unassigned_user['user_id']),
                            'lat': float(unassigned_user['lat']),
                            'lng': float(unassigned_user['lng']),
                            'office_distance': float(unassigned_user.get('office_distance', 0))
                        }

                        if unassigned_user.get('first_name'):
                            user_data['first_name'] = str(unassigned_user['first_name'])
                        if unassigned_user.get('email'):
                            user_data['email'] = str(unassigned_user['email'])

                        emergency_route = {
                            'driver_id': str(emergency_driver['driver_id']),
                            'vehicle_id': str(emergency_driver.get('vehicle_id', '')),
                            'vehicle_type': int(emergency_driver['capacity']),
                            'latitude': float(emergency_driver['latitude']),
                            'longitude': float(emergency_driver['longitude']),
                            'assigned_users': [user_data]
                        }

                        update_route_metrics_improved(emergency_route, office_lat, office_lon)
                        routes.append(emergency_route)
                        emergency_assignments += 1
                        user_assigned = True

                        logger.info(f"🚨 EMERGENCY: Created new route for user {unassigned_user['user_id']} with driver {emergency_driver['driver_id']}")

                if not user_assigned:
                    remaining_unassigned.append(unassigned_user)

            unassigned_users = remaining_unassigned
            logger.info(f"🚨 EMERGENCY PASS COMPLETE: {emergency_assignments} users assigned, {len(unassigned_users)} still unassigned")

        # STEP 8: GLOBAL DIRECTIONAL OPTIMIZATION - Fix poor routes created by emergency assignment
        logger.info("🎯 GLOBAL DIRECTIONAL OPTIMIZATION: Improving route quality through intelligent swapping")
        routes = global_directional_optimization(routes, office_lat, office_lon)

        # Filter out routes with no assigned users and move those drivers to unassigned
        filtered_routes = []
        empty_route_driver_ids = set()

        for route in routes:
            if route['assigned_users'] and len(route['assigned_users']) > 0:
                filtered_routes.append(route)
            else:
                empty_route_driver_ids.add(route['driver_id'])
                logger.info(
                    f"  📋 Moving driver {route['driver_id']} with no users to unassigned drivers"
                )

        routes = filtered_routes

        # Build unassigned drivers list (including drivers from empty routes)
        assigned_driver_ids = {route['driver_id'] for route in routes}
        unassigned_drivers_df = driver_df[~driver_df['driver_id'].
                                          isin(assigned_driver_ids)]
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

        # Final metrics update for all routes
        for route in routes:
            update_route_metrics_improved(route, office_lat, office_lon)

        execution_time = time.time() - start_time

        # Final user count verification with duplicate detection
        total_users_in_api = len(users)

        # Check for duplicate assignments
        all_assigned_user_ids = []
        for route in routes:
            for user in route['assigned_users']:
                all_assigned_user_ids.append(user['user_id'])

        # Find duplicates
        seen = set()
        duplicates = set()
        for user_id in all_assigned_user_ids:
            if user_id in seen:
                duplicates.add(user_id)
            seen.add(user_id)

        if duplicates:
            logger.error(f"🚨 DUPLICATE ASSIGNMENTS DETECTED: {duplicates}")
            # Remove duplicates - keep only first occurrence
            for route in routes:
                unique_users = []
                seen_in_route = set()
                for user in route['assigned_users']:
                    if user['user_id'] not in seen_in_route:
                        unique_users.append(user)
                        seen_in_route.add(user['user_id'])
                route['assigned_users'] = unique_users

        users_assigned = sum(len(r['assigned_users']) for r in routes)
        users_unassigned = len(unassigned_users)
        users_accounted_for = users_assigned + users_unassigned

        # Validate total doesn't exceed input
        if users_assigned > total_users_in_api:
            logger.error(
                f"🚨 ASSIGNMENT ERROR: {users_assigned} users assigned but only {total_users_in_api} users provided!"
            )

        result = {
            "status": "true",
            "execution_time": execution_time,
            "data": routes,
            "unassignedUsers": unassigned_users,
            "unassignedDrivers": unassigned_drivers,
            "clustering_analysis": clustering_results,
            "optimization_mode": "capacity_optimization",
            "parameter": parameter,
        }

        quality_analysis = analyze_assignment_quality(result)
        logger.info(f"✅ CAPACITY OPTIMIZATION assignment completed:")
        logger.info(f"   🚗 {len(routes)} routes created")
        logger.info(f"   👥 {users_assigned} users assigned")
        logger.info(f"   ❌ {users_unassigned} users unassigned")
        logger.info(f"   📊 Overall utilization: {quality_analysis['capacity_metrics']['overall_utilization']:.1f}%")
        logger.info(f"   📈 Assignment rate: {quality_analysis['capacity_metrics']['assignment_rate']:.1f}%")

        return result

    except requests.exceptions.RequestException as req_err:
        logger.error(f"API request failed: {req_err}")
        return {"status": "false", "details": str(req_err), "data": []}
    except ValueError as val_err:
        logger.error(f"Data validation error: {val_err}")
        return {"status": "false", "details": str(val_err), "data": []}
    except Exception as e:
        logger.error(f"Assignment failed: {e}", exc_info=True)
        return {"status": "false", "details": str(e), "data": []}