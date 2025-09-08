
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


# Load and validate configuration with balanced optimization settings
def load_and_validate_config():
    """Load configuration with balanced optimization settings"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    # Always use balanced mode
    current_mode = "balanced_optimization"

    # Get balanced optimization configuration
    mode_configs = cfg.get("mode_configs", {})
    mode_config = mode_configs.get("balanced_optimization", {})

    logger.info(f"🎯 Using optimization mode: BALANCED OPTIMIZATION")

    # Validate and set configuration with mode-specific overrides (true balance between capacity and route)
    config = {}

    # Distance configurations - balanced between capacity (8km) and route (6km) = 7km
    config['MAX_FILL_DISTANCE_KM'] = max(0.1, float(mode_config.get("max_fill_distance_km", cfg.get("max_fill_distance_km", 7.0))))
    config['MERGE_DISTANCE_KM'] = max(0.1, float(mode_config.get("merge_distance_km", cfg.get("merge_distance_km", 4.25))))  # Between 3.5 and 5.0
    config['DBSCAN_EPS_KM'] = max(0.1, float(cfg.get("dbscan_eps_km", 2.25)))  # Between 2.0 and 2.5
    config['OVERFLOW_PENALTY_KM'] = max(0.0, float(cfg.get("overflow_penalty_km", 6.0)))  # Between 5.0 and 7.0
    config['DISTANCE_ISSUE_THRESHOLD'] = max(0.1, float(cfg.get("distance_issue_threshold_km", 11.0)))  # Between 10.0 and 12.0
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(0.0, float(cfg.get("swap_improvement_threshold_km", 0.85)))  # Between 0.7 and 1.0

    # Utilization thresholds - balanced between capacity (0.8) and route (0.65) = 0.725
    config['MIN_UTIL_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("min_util_threshold", 0.725))))
    config['LOW_UTILIZATION_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("low_utilization_threshold", 0.65))))

    # Integer configurations (must be positive)
    config['MIN_SAMPLES_DBSCAN'] = max(1, int(cfg.get("min_samples_dbscan", 2)))
    config['MAX_SWAP_ITERATIONS'] = max(1, int(cfg.get("max_swap_iterations", 4)))  # Between 4 and 5
    config['MAX_USERS_FOR_FALLBACK'] = max(1, int(cfg.get("max_users_for_fallback", 4)))
    config['FALLBACK_MIN_USERS'] = max(1, int(cfg.get("fallback_min_users", 2)))
    config['FALLBACK_MAX_USERS'] = max(1, int(cfg.get("fallback_max_users", 8)))

    # Angle configurations - balanced between capacity (45°) and route (30°) = 37.5°
    config['MAX_BEARING_DIFFERENCE'] = max(0, min(180, float(mode_config.get("max_bearing_difference", cfg.get("max_bearing_difference", 37.5)))))
    config['MAX_TURNING_ANGLE'] = max(0, min(180, float(mode_config.get("max_allowed_turning_score", cfg.get("max_allowed_turning_score", 50)))))  # Between 40° and 60°

    # Cost penalties - balanced between capacity (5.0) and route (3.0) = 4.0
    config['UTILIZATION_PENALTY_PER_SEAT'] = max(0.0, float(mode_config.get("utilization_penalty_per_seat", cfg.get("utilization_penalty_per_seat", 4.0))))

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

    # Balanced optimization parameters - true 50/50 balance
    config['optimization_mode'] = "balanced_optimization"
    config['aggressive_merging'] = mode_config.get('aggressive_merging', None)  # Moderate approach
    config['capacity_weight'] = mode_config.get('capacity_weight', 3.0)  # Between 2.5 and 5.0
    config['direction_weight'] = mode_config.get('direction_weight', 2.0)  # Between 1.0 and 2.5

    # Clustering and optimization parameters with balanced overrides
    config['clustering_method'] = cfg.get('clustering_method', 'adaptive')
    config['min_cluster_size'] = max(2, cfg.get('min_cluster_size', 2))
    config['use_sweep_algorithm'] = cfg.get('use_sweep_algorithm', True)  # Between False and True
    config['angular_sectors'] = cfg.get('angular_sectors', 8)  # Between 6 and 10
    config['max_users_per_initial_cluster'] = cfg.get('max_users_per_initial_cluster', 10)  # Between 8 and 12
    config['max_users_per_cluster'] = cfg.get('max_users_per_cluster', 8)  # Between 7 and 10

    # Balanced optimization parameters
    config['zigzag_penalty_weight'] = mode_config.get('zigzag_penalty_weight', cfg.get('zigzag_penalty_weight', 1.25))  # Between 0.5 and 2.0
    config['route_split_turning_threshold'] = cfg.get('route_split_turning_threshold', 62.5)  # Between 45 and 80
    config['max_tortuosity_ratio'] = cfg.get('max_tortuosity_ratio', 1.75)  # Between 1.5 and 2.0
    config['route_split_consistency_threshold'] = cfg.get('route_split_consistency_threshold', 0.45)  # Between 0.3 and 0.6
    config['merge_tortuosity_improvement_required'] = cfg.get('merge_tortuosity_improvement_required', None)

    # Latitude conversion factor for distance normalization
    config['LAT_TO_KM'] = 111.0
    config['LON_TO_KM'] = 111.0 * math.cos(math.radians(office_lat))

    logger.info(f"   📊 Max bearing difference: {config['MAX_BEARING_DIFFERENCE']}°")
    logger.info(f"   📊 Max turning score: {config['MAX_TURNING_ANGLE']}°")
    logger.info(f"   📊 Max fill distance: {config['MAX_FILL_DISTANCE_KM']}km")
    logger.info(f"   📊 Capacity weight: {config['capacity_weight']}")
    logger.info(f"   📊 Direction weight: {config['direction_weight']}")

    return config


# Import all other functions from assignment.py (keeping the same structure)
from assignment import (
    validate_input_data, load_env_and_fetch_data, extract_office_coordinates,
    prepare_user_driver_dataframes, haversine_distance, bearing_difference,
    calculate_bearing_vectorized, calculate_bearing, calculate_bearings_and_features,
    coords_to_km, dbscan_clustering_metric, kmeans_clustering_metric, estimate_clusters,
    create_geographic_clusters, sweep_clustering, polar_sector_clustering,
    create_capacity_subclusters, create_bearing_aware_subclusters, calculate_bearing_spread,
    normalize_bearing_difference, calculate_sequence_distance, calculate_sequence_turning_score_improved,
    apply_strict_direction_aware_2opt, split_cluster_by_bearing_metric, apply_route_splitting, 
    split_route_by_bearing_improved, create_sub_route_improved, calculate_users_center_improved, 
    local_optimization, optimize_route_sequence_improved, calculate_route_cost_improved, 
    calculate_route_turning_score_improved, calculate_direction_consistency_improved, 
    try_user_swap_improved, calculate_route_center_improved, update_route_metrics_improved, 
    calculate_tortuosity_ratio_improved, global_optimization, fix_single_user_routes_improved, 
    calculate_average_bearing_improved, quality_controlled_route_filling, quality_preserving_route_merging, 
    strict_merge_compatibility_improved, calculate_merge_quality_score, perform_quality_merge_improved, 
    enhanced_route_splitting, intelligent_route_splitting_improved, split_by_bearing_clusters_improved, 
    split_by_distance_clusters_improved, create_split_routes_improved, find_best_driver_for_group, 
    outlier_detection_and_reassignment, try_reassign_outlier, handle_remaining_users_improved, 
    find_best_driver_for_cluster_improved, calculate_combined_route_center, _get_all_drivers_as_unassigned, 
    _convert_users_to_unassigned_format, analyze_assignment_quality, get_progress_tracker
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


def assign_drivers_by_priority_balanced(user_df, driver_df, office_lat, office_lon):
    """
    Capacity-focused balanced driver assignment: Prioritize filling seats while maintaining reasonable routes
    """
    logger.info("🚗 Step 3: Capacity-focused balanced assignment (80% capacity, 20% route)...")

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Sort by capacity DESC first to prioritize larger vehicles for better utilization
    available_drivers = driver_df.sort_values(['capacity', 'priority'], 
                                              ascending=[False, True])

    # Collect ALL unassigned users into one pool for maximum capacity filling
    all_unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()

    # More aggressive capacity-focused approach
    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids or all_unassigned_users.empty:
            continue

        vehicle_capacity = int(driver['capacity'])
        driver_pos = (driver['latitude'], driver['longitude'])

        # Calculate main route bearing
        main_route_bearing = calculate_bearing(office_lat, office_lon, driver['latitude'], driver['longitude'])

        # More lenient constraints to maximize capacity filling
        users_for_vehicle = []
        max_distance_limit = MAX_FILL_DISTANCE_KM * 2.2  # Much more lenient for capacity
        max_bearing_deviation = 50  # More lenient bearing for capacity

        # Collect candidate users with capacity-focused scoring
        candidate_users = []
        for _, user in all_unassigned_users.iterrows():
            distance = haversine_distance(driver['latitude'], driver['longitude'], 
                                        user['latitude'], user['longitude'])

            office_to_user_bearing = calculate_bearing(office_lat, office_lon, 
                                                     user['latitude'], user['longitude'])

            bearing_diff_from_main = bearing_difference(main_route_bearing, office_to_user_bearing)

            # Capacity-focused acceptance criteria (much more lenient)
            if (distance <= max_distance_limit and 
                bearing_diff_from_main <= max_bearing_deviation):
                # Capacity-focused score: heavily weight distance, less on direction
                score = distance * 0.8 + bearing_diff_from_main * 0.05  # Heavy preference for distance
                candidate_users.append((score, user))

        # Sort by capacity-focused score and try to fill to maximum capacity
        candidate_users.sort(key=lambda x: x[0])
        # Capacity-focused target: aim for 95% or full capacity
        target_capacity = min(vehicle_capacity, max(2, int(vehicle_capacity * 0.98)))

        for score, user in candidate_users:
            if len(users_for_vehicle) >= target_capacity:
                break
            users_for_vehicle.append(user)

        # More lenient directional consistency check
        if len(users_for_vehicle) >= 2:
            if not is_directionally_consistent_balanced(users_for_vehicle, driver_pos, (office_lat, office_lon)):
                # Apply more lenient filtering that keeps more users
                users_for_vehicle = filter_for_directional_consistency_balanced_lenient(users_for_vehicle, driver_pos, (office_lat, office_lon))

        # Lower minimum utilization threshold to assign more routes
        min_users_required = max(2, int(vehicle_capacity * 0.4))  # Only 40% minimum
        if len(users_for_vehicle) >= min_users_required:
            cluster_df = pd.DataFrame(users_for_vehicle)
            route = assign_best_driver_to_cluster_balanced(
                cluster_df, pd.DataFrame([driver]), used_driver_ids, office_lat, office_lon)

            if route:
                routes.append(route)
                assigned_user_ids.update(u['user_id'] for u in route['assigned_users'])

                # Remove assigned users from pool
                assigned_ids_set = {u['user_id'] for u in route['assigned_users']}
                all_unassigned_users = all_unassigned_users[~all_unassigned_users['user_id'].isin(assigned_ids_set)]

                utilization = len(route['assigned_users']) / vehicle_capacity * 100
                logger.info(f"  ⚖️ Driver {driver['driver_id']}: {len(route['assigned_users'])}/{vehicle_capacity} seats ({utilization:.1f}%) - Capacity-Focused")

    # Second pass: Fill remaining seats with balanced constraints (from both approaches)
    remaining_users = all_unassigned_users[~all_unassigned_users['user_id'].isin(assigned_user_ids)]

    for route in routes:
        if remaining_users.empty:
            break

        available_seats = route['vehicle_type'] - len(route['assigned_users'])
        if available_seats <= 0:
            continue

        # Balanced seat filling
        route_bearing = calculate_average_bearing_improved(route, office_lat, office_lon)
        route_center = calculate_route_center_improved(route)

        compatible_users = []
        for _, user in remaining_users.iterrows():
            distance = haversine_distance(route_center[0], route_center[1],
                                        user['latitude'], user['longitude'])

            user_bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
            bearing_diff = bearing_difference(route_bearing, user_bearing)

            # Balanced criteria for seat filling: between capacity (lenient) and route (strict)
            max_fill_distance = MAX_FILL_DISTANCE_KM * 1.9  # Between 1.8 and 2.0
            max_fill_bearing = 42  # Between 40 and 45

            if distance <= max_fill_distance and bearing_diff <= max_fill_bearing:
                # Test directional consistency
                test_users = route['assigned_users'] + [
                    {
                        'user_id': str(user['user_id']),
                        'lat': float(user['latitude']),
                        'lng': float(user['longitude'])
                    }
                ]

                driver_pos = (route['latitude'], route['longitude'])
                if is_directionally_consistent_from_dicts_balanced(
                        test_users, driver_pos, (office_lat, office_lon)):
                    score = distance * 0.7 + bearing_diff * 0.1  # Balanced scoring
                    compatible_users.append((score, user))

        # Fill available seats
        compatible_users.sort(key=lambda x: x[0])
        users_to_add = []

        for score, user in compatible_users[:available_seats]:
            users_to_add.append(user)

        # Add users to route
        for user in users_to_add:
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

        if users_to_add:
            assigned_ids = {u['user_id'] for u in users_to_add}
            remaining_users = remaining_users[~remaining_users['user_id'].isin(assigned_ids)]

            route = optimize_route_sequence_improved(route, office_lat, office_lon)
            utilization = len(route['assigned_users']) / route['vehicle_type'] * 100
            logger.info(f"  ⚖️ Route {route['driver_id']}: Added {len(users_to_add)} users, now {len(route['assigned_users'])}/{route['vehicle_type']} ({utilization:.1f}%)")

    logger.info(f"  ✅ Balanced assignment: {len(routes)} routes with true capacity + route balance")

    # Calculate balanced metrics
    total_seats = sum(r['vehicle_type'] for r in routes)
    total_users = sum(len(r['assigned_users']) for r in routes)
    overall_utilization = (total_users / total_seats * 100) if total_seats > 0 else 0

    logger.info(f"  ⚖️ Balanced utilization: {total_users}/{total_seats} ({overall_utilization:.1f}%)")

    return routes, assigned_user_ids


def assign_best_driver_to_cluster_balanced(cluster_users, available_drivers, used_driver_ids, office_lat, office_lon):
    """Find and assign the best available driver with CAPACITY-FOCUSED optimization"""
    cluster_size = len(cluster_users)

    best_driver = None
    best_score = float('inf')
    best_sequence = None

    # Capacity-focused weights - heavily prioritize utilization
    capacity_weight = _config.get('capacity_weight', 3.0) * 2.0  # Double capacity importance
    direction_weight = _config.get('direction_weight', 2.0) * 0.3  # Reduce route importance

    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids:
            continue

        # More lenient capacity check - allow slight over-assignment if beneficial
        if driver['capacity'] < cluster_size:
            continue

        # Calculate route metrics with capacity focus
        route_cost, sequence, mean_turning_degrees = calculate_route_cost_balanced(
            driver, cluster_users, office_lat, office_lon
        )

        # Capacity-focused scoring approach
        utilization = cluster_size / driver['capacity']

        # Heavily weight capacity utilization
        distance_score = route_cost * 0.2  # Reduce distance importance to 20%
        direction_score = mean_turning_degrees * direction_weight * 0.008  # Reduce direction importance
        capacity_score = (1.0 - utilization) * capacity_weight * 8.0  # MASSIVE penalty for underutilization

        # Utilization bonus for high capacity usage
        if utilization >= 0.9:
            utilization_bonus = -5.0  # Big bonus for 90%+ utilization
        elif utilization >= 0.8:
            utilization_bonus = -3.0  # Medium bonus for 80%+ utilization
        elif utilization >= 0.7:
            utilization_bonus = -1.0  # Small bonus for 70%+ utilization
        else:
            utilization_bonus = 0.0

        # Priority component (very small)
        priority_score = driver['priority'] * 0.05

        # Capacity-focused total score (heavily penalize underutilization)
        total_score = distance_score + direction_score + capacity_score + priority_score + utilization_bonus

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

        # Add all users from cluster with balanced approach
        if hasattr(cluster_users, 'iterrows'):
            users_to_add = list(cluster_users.iterrows())
        else:
            users_to_add = [(i, user) for i, user in enumerate(cluster_users)]

        for _, user in users_to_add:
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

        # Balanced optimization of sequence
        route = optimize_route_sequence_improved(route, office_lat, office_lon)
        update_route_metrics_improved(route, office_lat, office_lon)

        utilization = len(route['assigned_users']) / route['vehicle_type']
        logger.info(f"    ⚖️ Balanced assignment - Driver {best_driver['driver_id']}: {len(route['assigned_users'])}/{route['vehicle_type']} seats ({utilization*100:.1f}%)")

        return route

    return None


def calculate_route_cost_balanced(driver, cluster_users, office_lat, office_lon):
    """Calculate route cost with balanced optimization - weight both distance and direction"""
    if len(cluster_users) == 0:
        return float('inf'), [], 0

    driver_pos = (driver['latitude'], driver['longitude'])
    office_pos = (office_lat, office_lon)

    # Get optimal pickup sequence with balanced focus
    sequence = calculate_optimal_sequence_balanced(driver_pos, cluster_users, office_pos)

    # Calculate total route distance
    total_distance = 0
    bearing_differences = []

    # Driver to first pickup
    if sequence:
        first_user = sequence[0]
        total_distance += haversine_distance(
            driver_pos[0], driver_pos[1], 
            first_user['latitude'], first_user['longitude']
        )

    # Between pickups - calculate bearing differences
    for i in range(len(sequence) - 1):
        current_user = sequence[i]
        next_user = sequence[i + 1]

        distance = haversine_distance(
            current_user['latitude'], current_user['longitude'],
            next_user['latitude'], next_user['longitude']
        )
        total_distance += distance

        # Calculate bearing difference between segments
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
        total_distance += haversine_distance(
            last_user['latitude'], last_user['longitude'],
            office_lat, office_lon
        )

    # Calculate mean turning angle with balanced weight
    mean_turning_degrees = sum(bearing_differences) / len(bearing_differences) if bearing_differences else 0

    return total_distance, sequence, mean_turning_degrees


def calculate_optimal_sequence_balanced(driver_pos, cluster_users, office_pos):
    """Calculate sequence with balanced optimization - combine distance and direction preferences"""
    if len(cluster_users) <= 1:
        return cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)

    users_list = cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)

    # Calculate main route bearing (driver to office)
    main_route_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    # Balanced scoring: weight both distance and direction
    def balanced_score(user):
        # Distance component
        distance = haversine_distance(driver_pos[0], driver_pos[1], 
                                    user['latitude'], user['longitude'])

        # Direction component
        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1], 
                                       user['latitude'], user['longitude'])

        bearing_diff = normalize_bearing_difference(user_bearing - main_route_bearing)
        bearing_diff_rad = math.radians(abs(bearing_diff))

        # Balanced weight: 60% distance, 40% direction
        distance_score = distance
        direction_score = distance * (1 - math.cos(bearing_diff_rad))

        combined_score = distance_score * 0.6 + direction_score * 0.4

        return (combined_score, user['user_id'])

    users_list.sort(key=balanced_score)

    # Apply balanced 2-opt optimization
    return apply_balanced_2opt(users_list, driver_pos, office_pos)


def apply_balanced_2opt(sequence, driver_pos, office_pos):
    """Apply balanced 2-opt improvements - weight both distance and direction"""
    if len(sequence) <= 2:
        return sequence

    improved = True
    max_iterations = 3
    iteration = 0

    # Calculate main bearing from driver to office
    main_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    # Balanced turning angle threshold
    max_turning_threshold = 52  # Between route (47°) and capacity (60°)

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        best_distance = calculate_sequence_distance(sequence, driver_pos, office_pos)
        best_turning_score = calculate_sequence_turning_score_improved(sequence, driver_pos, office_pos)

        for i in range(len(sequence) - 1):
            for j in range(i + 2, len(sequence)):
                # Try 2-opt swap
                new_sequence = sequence[:i+1] + sequence[i+1:j+1][::-1] + sequence[j+1:]

                # Calculate new metrics
                new_distance = calculate_sequence_distance(new_sequence, driver_pos, office_pos)
                new_turning_score = calculate_sequence_turning_score_improved(new_sequence, driver_pos, office_pos)

                # Balanced acceptance criteria
                distance_improvement = (best_distance - new_distance) / best_distance
                turning_improvement = (best_turning_score - new_turning_score)

                # Convert turning improvement to same scale as distance
                turning_improvement_normalized = turning_improvement / max(best_turning_score, 1.0)

                # Balanced weight: 60% distance, 40% direction
                combined_improvement = distance_improvement * 0.6 + turning_improvement_normalized * 0.4

                # Accept if combined improvement is positive and turning stays reasonable
                if (combined_improvement > 0.003 and  # Small positive improvement
                    new_turning_score <= max_turning_threshold):

                    sequence = new_sequence
                    best_distance = new_distance
                    best_turning_score = new_turning_score
                    improved = True
                    break
            if improved:
                break

    return sequence


def is_directionally_consistent_balanced(users, driver_pos, office_pos):
    """Check if a group of users maintains directional consistency with balanced constraints"""
    if len(users) <= 1:
        return True

    office_lat, office_lon = office_pos

    # Calculate bearings from office to each user
    bearings = []
    for user in users:
        bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
        bearings.append(bearing)

    # Balanced bearing spread: between capacity (60°) and route (45°) = 52.5°
    max_bearing_spread = 52.5

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


def filter_for_directional_consistency_balanced(users, driver_pos, office_pos):
    """Filter users to maintain directional consistency with balanced approach"""
    if len(users) <= 2:
        return users

    office_lat, office_lon = office_pos

    # Calculate bearings and find the most consistent group
    user_bearings = []
    for user in users:
        bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
        user_bearings.append((user, bearing))

    # Sort by bearing
    user_bearings.sort(key=lambda x: x[1])

    # Find largest group within bearing spread constraint
    max_bearing_spread = 52.5  # Balanced constraint
    best_group = []
    
    for start_idx in range(len(user_bearings)):
        current_group = [user_bearings[start_idx][0]]
        start_bearing = user_bearings[start_idx][1]
        
        for end_idx in range(start_idx + 1, len(user_bearings)):
            end_bearing = user_bearings[end_idx][1]
            if bearing_difference(start_bearing, end_bearing) <= max_bearing_spread:
                current_group.append(user_bearings[end_idx][0])
            else:
                break
        
        if len(current_group) > len(best_group):
            best_group = current_group

    return best_group


def filter_for_directional_consistency_balanced_lenient(users, driver_pos, office_pos):
    """Filter users with very lenient directional consistency for capacity focus"""
    if len(users) <= 3:
        return users

    office_lat, office_lon = office_pos

    # Calculate bearings and find the most consistent group
    user_bearings = []
    for user in users:
        bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
        user_bearings.append((user, bearing))

    # Sort by bearing
    user_bearings.sort(key=lambda x: x[1])

    # Find largest group within very lenient bearing spread constraint
    max_bearing_spread = 70  # Very lenient for capacity focus
    best_group = []
    
    for start_idx in range(len(user_bearings)):
        current_group = [user_bearings[start_idx][0]]
        start_bearing = user_bearings[start_idx][1]
        
        for end_idx in range(start_idx + 1, len(user_bearings)):
            end_bearing = user_bearings[end_idx][1]
            if bearing_difference(start_bearing, end_bearing) <= max_bearing_spread:
                current_group.append(user_bearings[end_idx][0])
            else:
                break
        
        if len(current_group) > len(best_group):
            best_group = current_group

    # If no good group found, just take the largest consecutive group by distance
    if len(best_group) < len(users) * 0.7:  # If we lose more than 30% of users
        # Sort by distance from driver instead
        user_distances = []
        driver_lat, driver_lon = driver_pos
        for user in users:
            distance = haversine_distance(driver_lat, driver_lon, user['latitude'], user['longitude'])
            user_distances.append((user, distance))
        
        user_distances.sort(key=lambda x: x[1])
        # Take the closest 80% of users
        take_count = max(2, int(len(users) * 0.8))
        best_group = [user for user, _ in user_distances[:take_count]]

    return best_group


def is_directionally_consistent_from_dicts_balanced(user_dicts, driver_pos, office_pos):
    """Check directional consistency for user dictionaries with balanced constraints"""
    if len(user_dicts) <= 1:
        return True

    office_lat, office_lon = office_pos

    # Calculate bearings from office to each user
    bearings = []
    for user in user_dicts:
        bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])
        bearings.append(bearing)

    # Balanced bearing spread
    max_bearing_spread = 52.5

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


def final_pass_merge_balanced(routes, config, office_lat, office_lon):
    """
    Capacity-focused balanced merge: Prioritize seat utilization over route perfection
    """
    logger.info("🔄 Step 6: Capacity-focused balanced merge (prioritize seat filling)...")

    merged_routes = []
    used = set()

    # More lenient thresholds to allow more merging for capacity
    MERGE_BEARING_THRESHOLD = 45  # More lenient for capacity
    MERGE_DISTANCE_KM = config.get("MERGE_DISTANCE_KM", 4.25) * 1.8  # Much more lenient
    MERGE_TURNING_THRESHOLD = 65  # More lenient for capacity
    MERGE_TORTUOSITY_THRESHOLD = 2.1  # Much more lenient for capacity

    for i, r1 in enumerate(routes):
        if i in used:
            continue

        best_merge = None
        best_balanced_score = float('inf')

        for j, r2 in enumerate(routes):
            if j <= i or j in used:
                continue

            # 1. Direction compatibility check (balanced)
            b1 = calculate_average_bearing_improved(r1, office_lat, office_lon)
            b2 = calculate_average_bearing_improved(r2, office_lat, office_lon)
            bearing_diff = bearing_difference(b1, b2)

            if bearing_diff > MERGE_BEARING_THRESHOLD:
                continue

            # 2. Distance compatibility check (balanced)
            c1 = calculate_route_center_improved(r1)
            c2 = calculate_route_center_improved(r2)
            centroid_distance = haversine_distance(c1[0], c1[1], c2[0], c2[1])

            if centroid_distance > MERGE_DISTANCE_KM:
                continue

            # 3. Capacity check
            total_users = len(r1['assigned_users']) + len(r2['assigned_users'])
            max_capacity = max(r1['vehicle_type'], r2['vehicle_type'])

            if total_users > max_capacity:
                continue

            # 4. Directional consistency check (from capacity approach)
            all_users = r1['assigned_users'] + r2['assigned_users']
            driver_pos = (r1['latitude'], r1['longitude']) if r1['vehicle_type'] >= r2['vehicle_type'] else (r2['latitude'], r2['longitude'])

            if not is_directionally_consistent_from_dicts_balanced(all_users, driver_pos, (office_lat, office_lon)):
                continue

            # 5. Quality assessment with balanced criteria
            combined_center = calculate_combined_route_center(r1, r2)
            dist1 = haversine_distance(r1['latitude'], r1['longitude'], combined_center[0], combined_center[1])
            dist2 = haversine_distance(r2['latitude'], r2['longitude'], combined_center[0], combined_center[1])

            better_route = r1 if dist1 <= dist2 else r2

            # Create test merged route
            test_route = better_route.copy()
            test_route['assigned_users'] = r1['assigned_users'] + r2['assigned_users']
            test_route['vehicle_type'] = max_capacity

            # Optimize sequence for merged route
            test_route = optimize_route_sequence_improved(test_route, office_lat, office_lon)

            # Calculate balanced quality metrics
            turning_score = calculate_route_turning_score_improved(
                test_route['assigned_users'],
                (test_route['latitude'], test_route['longitude']),
                (office_lat, office_lon)
            )

            tortuosity = calculate_tortuosity_ratio_improved(
                test_route['assigned_users'],
                (test_route['latitude'], test_route['longitude']),
                (office_lat, office_lon)
            )

            utilization = total_users / max_capacity

            # Balanced acceptance criteria
            efficiency_acceptable = (turning_score <= MERGE_TURNING_THRESHOLD and 
                                   tortuosity <= MERGE_TORTUOSITY_THRESHOLD)
            capacity_acceptable = utilization >= 0.65  # Balanced utilization requirement

            # Accept if meets minimum criteria, prioritizing capacity heavily
            efficiency_acceptable_min = (turning_score <= MERGE_TURNING_THRESHOLD * 1.2 and 
                                       tortuosity <= MERGE_TORTUOSITY_THRESHOLD * 1.1)  # More lenient
            capacity_acceptable_min = utilization >= 0.5  # Lower threshold
            
            if efficiency_acceptable_min and capacity_acceptable_min:
                # Capacity-focused scoring: 20% efficiency, 80% capacity
                efficiency_score = turning_score * 0.3 + (tortuosity - 1.0) * 15
                capacity_score = (1.0 - utilization) * 200  # Much higher penalty for underutilization

                # Massive capacity bonus for high utilization
                if utilization >= 0.95:
                    capacity_bonus = -50  # Huge bonus
                elif utilization >= 0.9:
                    capacity_bonus = -30  # Big bonus
                elif utilization >= 0.8:
                    capacity_bonus = -15  # Medium bonus
                else:
                    capacity_bonus = 0

                # Heavily weight capacity
                balanced_score = efficiency_score * 0.2 + capacity_score * 0.8 + capacity_bonus

                if balanced_score < best_balanced_score:
                    best_balanced_score = balanced_score
                    best_merge = (j, test_route)

        if best_merge is not None:
            j, merged_route = best_merge
            merged_routes.append(merged_route)
            used.add(i)
            used.add(j)

            utilization_pct = len(merged_route['assigned_users']) / merged_route['vehicle_type'] * 100
            turning = merged_route.get('turning_score', 0)
            logger.info(f"  ⚖️ Balanced merge: {r1['driver_id']} + {routes[j]['driver_id']} = {len(merged_route['assigned_users'])}/{merged_route['vehicle_type']} seats ({utilization_pct:.1f}%, {turning:.1f}° turn)")
        else:
            merged_routes.append(r1)
            used.add(i)

    # Final statistics
    total_seats = sum(r['vehicle_type'] for r in merged_routes)
    total_users = sum(len(r['assigned_users']) for r in merged_routes)
    avg_utilization = (total_users / total_seats * 100) if total_seats > 0 else 0
    avg_turning = np.mean([r.get('turning_score', 0) for r in merged_routes])

    logger.info(f"  ⚖️ Balanced merge: {len(routes)} → {len(merged_routes)} routes")
    logger.info(f"  ⚖️ Final balance: {avg_utilization:.1f}% utilization, {avg_turning:.1f}° avg turning")

    return merged_routes


# MAIN ASSIGNMENT FUNCTION FOR BALANCED OPTIMIZATION
def run_assignment_balance(source_id: str, parameter: int = 1, string_param: str = ""):
    """
    Main assignment function optimized for balanced approach:
    - True balance between capacity utilization and route efficiency  
    - Combines the best of assign_capacity.py and assign_route.py
    - Moderate constraints balancing both objectives
    """
    start_time = time.time()

    # Clear any cached data files to ensure fresh assignment
    cache_files = [
        "drivers_and_routes.json",
        "drivers_and_routes_capacity.json", 
        "drivers_and_routes_balance.json",
        "drivers_and_routes_road_aware.json"
    ]

    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)

    # Reload configuration for balanced optimization
    global _config
    _config = load_and_validate_config()

    # Update global variables from new config
    global MAX_FILL_DISTANCE_KM, MERGE_DISTANCE_KM, MAX_BEARING_DIFFERENCE, UTILIZATION_PENALTY_PER_SEAT
    MAX_FILL_DISTANCE_KM = _config['MAX_FILL_DISTANCE_KM']
    MERGE_DISTANCE_KM = _config['MERGE_DISTANCE_KM'] 
    MAX_BEARING_DIFFERENCE = _config['MAX_BEARING_DIFFERENCE']
    UTILIZATION_PENALTY_PER_SEAT = _config['UTILIZATION_PENALTY_PER_SEAT']

    logger.info(f"🚀 Starting BALANCED OPTIMIZATION assignment for source_id: {source_id}")
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
                "clustering_analysis": {
                    "method": "No Users",
                    "clusters": 0
                },
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
                "clustering_analysis": {
                    "method": "No Drivers",
                    "clusters": 0
                },
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

        # STEP 1: Geographic clustering with balanced approach
        user_df = create_geographic_clusters(user_df, office_lat, office_lon, _config)
        clustering_results = {"method": "balanced_" + _config['clustering_method'], 
                            "clusters": user_df['geo_cluster'].nunique()}

        # STEP 2: Capacity-based sub-clustering with balanced constraints
        user_df = create_capacity_subclusters(user_df, office_lat, office_lon, _config)

        # STEP 3: Balanced driver assignment (combines capacity + route approaches)
        routes, assigned_user_ids = assign_drivers_by_priority_balanced(
            user_df, driver_df, office_lat, office_lon)

        # STEP 4: Local optimization with balanced approach
        routes = local_optimization(routes, office_lat, office_lon)

        # STEP 5: Balanced global optimization
        routes, unassigned_users = global_optimization(routes, user_df,
                                                       assigned_user_ids, driver_df, office_lat, office_lon)

        # STEP 6: Balanced final-pass merge
        routes = final_pass_merge_balanced(routes, _config, office_lat, office_lon)

        # Filter out routes with no assigned users and move those drivers to unassigned
        filtered_routes = []
        empty_route_driver_ids = set()

        for route in routes:
            if route['assigned_users'] and len(route['assigned_users']) > 0:
                filtered_routes.append(route)
            else:
                empty_route_driver_ids.add(route['driver_id'])
                logger.info(f"  📋 Moving driver {route['driver_id']} with no users to unassigned drivers")

        routes = filtered_routes

        # Build unassigned drivers list (including drivers from empty routes)
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

        # Final metrics update for all routes
        for route in routes:
            update_route_metrics_improved(route, office_lat, office_lon)

        execution_time = time.time() - start_time

        # Final user count verification
        total_users_in_api = len(users)
        users_assigned = sum(len(r['assigned_users']) for r in routes)
        users_unassigned = len(unassigned_users)
        users_accounted_for = users_assigned + users_unassigned

        logger.info(f"✅ Balanced optimization complete in {execution_time:.2f}s")
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
