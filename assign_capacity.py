
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
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load and validate configuration with capacity optimization settings
def load_and_validate_config():
    """Load configuration with capacity optimization settings"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    # Always use capacity mode
    current_mode = "capacity_optimization"

    # Get capacity optimization configuration
    mode_configs = cfg.get("mode_configs", {})
    mode_config = mode_configs.get("capacity_optimization", {})

    print(f"🎯 Using optimization mode: CAPACITY OPTIMIZATION")
    
    # Validate and set configuration with mode-specific overrides
    config = {}

    # Distance configurations with mode overrides (more lenient for capacity filling)
    config['MAX_FILL_DISTANCE_KM'] = max(0.1, float(mode_config.get("max_fill_distance_km", cfg.get("max_fill_distance_km", 8.0))))
    config['MERGE_DISTANCE_KM'] = max(0.1, float(mode_config.get("merge_distance_km", cfg.get("merge_distance_km", 5.0))))
    config['DBSCAN_EPS_KM'] = max(0.1, float(cfg.get("dbscan_eps_km", 2.5)))
    config['OVERFLOW_PENALTY_KM'] = max(0.0, float(cfg.get("overflow_penalty_km", 5.0)))
    config['DISTANCE_ISSUE_THRESHOLD'] = max(0.1, float(cfg.get("distance_issue_threshold_km", 12.0)))
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(0.0, float(cfg.get("swap_improvement_threshold_km", 1.0)))

    # Utilization thresholds (more aggressive for capacity)
    config['MIN_UTIL_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("min_util_threshold", 0.8))))
    config['LOW_UTILIZATION_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("low_utilization_threshold", 0.7))))

    # Integer configurations (must be positive)
    config['MIN_SAMPLES_DBSCAN'] = max(1, int(cfg.get("min_samples_dbscan", 2)))
    config['MAX_SWAP_ITERATIONS'] = max(1, int(cfg.get("max_swap_iterations", 5)))
    config['MAX_USERS_FOR_FALLBACK'] = max(1, int(cfg.get("max_users_for_fallback", 5)))
    config['FALLBACK_MIN_USERS'] = max(1, int(cfg.get("fallback_min_users", 3)))
    config['FALLBACK_MAX_USERS'] = max(1, int(cfg.get("fallback_max_users", 10)))

    # Angle configurations with mode overrides (more lenient for capacity)
    config['MAX_BEARING_DIFFERENCE'] = max(0, min(180, float(mode_config.get("max_bearing_difference", cfg.get("max_bearing_difference", 45)))))
    config['MAX_TURNING_ANGLE'] = max(0, min(180, float(mode_config.get("max_allowed_turning_score", cfg.get("max_allowed_turning_score", 60)))))

    # Cost penalties with mode overrides (prioritize capacity over route quality)
    config['UTILIZATION_PENALTY_PER_SEAT'] = max(0.0, float(mode_config.get("utilization_penalty_per_seat", cfg.get("utilization_penalty_per_seat", 5.0))))

    # Office coordinates with environment variable fallbacks
    office_lat = float(os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489)))
    office_lon = float(os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))

    # Validate coordinate bounds
    if not (-90 <= office_lat <= 90):
        print(f"Warning: Invalid office latitude {office_lat}, using default")
        office_lat = 30.6810489
    if not (-180 <= office_lon <= 180):
        print(f"Warning: Invalid office longitude {office_lon}, using default")
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
    config['use_sweep_algorithm'] = cfg.get('use_sweep_algorithm', False)  # Less directional for capacity
    config['angular_sectors'] = cfg.get('angular_sectors', 6)  # Fewer sectors for larger groups
    config['max_users_per_initial_cluster'] = cfg.get('max_users_per_initial_cluster', 12)
    config['max_users_per_cluster'] = cfg.get('max_users_per_cluster', 10)

    # Capacity optimization parameters
    config['zigzag_penalty_weight'] = mode_config.get('zigzag_penalty_weight', cfg.get('zigzag_penalty_weight', 0.5))  # Very low
    config['route_split_turning_threshold'] = cfg.get('route_split_turning_threshold', 80)  # Very high
    config['max_tortuosity_ratio'] = cfg.get('max_tortuosity_ratio', 2.0)  # Very lenient
    config['route_split_consistency_threshold'] = cfg.get('route_split_consistency_threshold', 0.3)  # Very low
    config['merge_tortuosity_improvement_required'] = cfg.get('merge_tortuosity_improvement_required', False)
    
    # Latitude conversion factor for distance normalization
    config['LAT_TO_KM'] = 111.0
    config['LON_TO_KM'] = 111.0 * math.cos(math.radians(office_lat))

    print(f"   📊 Max bearing difference: {config['MAX_BEARING_DIFFERENCE']}°")
    print(f"   📊 Max turning score: {config['MAX_TURNING_ANGLE']}°")
    print(f"   📊 Max fill distance: {config['MAX_FILL_DISTANCE_KM']}km")
    print(f"   📊 Capacity weight: {config['capacity_weight']}")
    print(f"   📊 Direction weight: {config['direction_weight']}")

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
    apply_strict_direction_aware_2opt, calculate_optimal_sequence_improved,
    split_cluster_by_bearing_metric, apply_route_splitting, split_route_by_bearing_improved,
    create_sub_route_improved, calculate_users_center_improved, local_optimization,
    optimize_route_sequence_improved, calculate_route_cost_improved, calculate_route_turning_score_improved,
    calculate_direction_consistency_improved, try_user_swap_improved, calculate_route_center_improved,
    update_route_metrics_improved, calculate_tortuosity_ratio_improved, global_optimization,
    fix_single_user_routes_improved, calculate_average_bearing_improved, quality_controlled_route_filling,
    quality_preserving_route_merging, strict_merge_compatibility_improved, calculate_merge_quality_score,
    perform_quality_merge_improved, enhanced_route_splitting, intelligent_route_splitting_improved,
    split_by_bearing_clusters_improved, split_by_distance_clusters_improved, create_split_routes_improved,
    find_best_driver_for_group, outlier_detection_and_reassignment, try_reassign_outlier,
    handle_remaining_users_improved, find_best_driver_for_cluster_improved, final_pass_merge,
    calculate_combined_route_center, _get_all_drivers_as_unassigned, _convert_users_to_unassigned_format,
    analyze_assignment_quality
)

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


def assign_drivers_by_priority_capacity_focused(user_df, driver_df, office_lat, office_lon):
    """
    Directional capacity-focused assignment: Fill every seat possible while maintaining directional consistency
    """
    print("🚗 Step 3: DIRECTIONAL capacity-focused driver assignment...")

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Sort drivers by capacity (descending), then by priority - prioritize largest vehicles
    available_drivers = driver_df.sort_values(['capacity', 'priority'], 
                                              ascending=[False, True])

    # Collect ALL unassigned users into large pools for maximum seat filling
    all_unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()

    # Directional clustering - pack users tightly but enforce direction consistency
    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids or all_unassigned_users.empty:
            continue

        vehicle_capacity = int(driver['capacity'])
        driver_pos = (driver['latitude'], driver['longitude'])
        
        # Calculate bearing from office to driver (main direction for this route)
        main_route_bearing = calculate_bearing(office_lat, office_lon, driver['latitude'], driver['longitude'])
        
        # Find users in the same general direction with directional constraints
        users_for_vehicle = []
        max_distance_limit = MAX_FILL_DISTANCE_KM * 2  # Allow more distance for capacity
        max_bearing_deviation = 45  # Allow 45 degrees deviation for capacity mode
        
        # Calculate distances and bearings from driver to all unassigned users
        candidate_users = []
        for _, user in all_unassigned_users.iterrows():
            distance = haversine_distance(driver['latitude'], driver['longitude'], 
                                        user['latitude'], user['longitude'])
            
            # Calculate bearing from driver to user
            user_bearing = calculate_bearing(driver['latitude'], driver['longitude'], 
                                           user['latitude'], user['longitude'])
            
            # Calculate bearing from office to user (to ensure general same direction)
            office_to_user_bearing = calculate_bearing(office_lat, office_lon, 
                                                     user['latitude'], user['longitude'])
            
            # Check directional consistency
            bearing_diff_from_main = bearing_difference(main_route_bearing, office_to_user_bearing)
            
            # Accept users that are in the same general direction
            if (distance <= max_distance_limit and 
                bearing_diff_from_main <= max_bearing_deviation):
                candidate_users.append((distance, bearing_diff_from_main, user))
        
        # Sort by distance first, then by bearing alignment
        candidate_users.sort(key=lambda x: (x[0], x[1]))
        
        # Take users to fill the vehicle while maintaining direction
        for distance, bearing_diff, user in candidate_users:
            if len(users_for_vehicle) >= vehicle_capacity:
                break
            
            # Additional directional check: ensure new user doesn't create zigzag
            if len(users_for_vehicle) > 0:
                # Check if adding this user maintains route directional consistency
                test_users = users_for_vehicle + [user]
                if not is_directionally_consistent(test_users, driver_pos, (office_lat, office_lon)):
                    continue
            
            users_for_vehicle.append(user)

        # Try to fill remaining seats with nearby users even if slightly off-direction
        if len(users_for_vehicle) < vehicle_capacity and len(candidate_users) > len(users_for_vehicle):
            remaining_candidates = [user for _, _, user in candidate_users[len(users_for_vehicle):]]
            
            for user in remaining_candidates[:vehicle_capacity - len(users_for_vehicle)]:
                # More lenient check for remaining seats
                test_users = users_for_vehicle + [user]
                if is_directionally_consistent(test_users, driver_pos, (office_lat, office_lon), lenient=True):
                    users_for_vehicle.append(user)

        if len(users_for_vehicle) >= 2:  # Minimum viable route
            # Create route with directionally consistent users
            cluster_df = pd.DataFrame(users_for_vehicle)
            route = assign_best_driver_to_cluster_capacity_focused(
                cluster_df, pd.DataFrame([driver]), used_driver_ids, office_lat, office_lon)
            
            if route:
                routes.append(route)
                assigned_user_ids.update(u['user_id'] for u in route['assigned_users'])
                
                # Remove assigned users from the pool
                assigned_ids_set = {u['user_id'] for u in route['assigned_users']}
                all_unassigned_users = all_unassigned_users[~all_unassigned_users['user_id'].isin(assigned_ids_set)]
                
                print(f"  🧭 Driver {driver['driver_id']}: {len(route['assigned_users'])}/{vehicle_capacity} seats filled ({len(route['assigned_users'])/vehicle_capacity*100:.1f}%) - Directional")

    # Second pass: Try to fill remaining seats with directional constraints
    remaining_users = all_unassigned_users[~all_unassigned_users['user_id'].isin(assigned_user_ids)]
    
    for route in routes:
        if remaining_users.empty:
            break
            
        available_seats = route['vehicle_type'] - len(route['assigned_users'])
        if available_seats <= 0:
            continue
        
        # Get current route direction
        route_bearing = calculate_average_bearing_improved(route, office_lat, office_lon)
        route_center = calculate_route_center_improved(route)
        
        # Find remaining users that fit the route direction
        compatible_users = []
        for _, user in remaining_users.iterrows():
            distance = haversine_distance(route_center[0], route_center[1],
                                        user['latitude'], user['longitude'])
            
            # Check if user is in same direction as route
            user_bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
            bearing_diff = bearing_difference(route_bearing, user_bearing)
            
            if distance <= MAX_FILL_DISTANCE_KM * 1.5 and bearing_diff <= 30:  # Directional constraint
                # Test if adding this user maintains directional consistency
                test_users = route['assigned_users'] + [{
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude'])
                }]
                
                driver_pos = (route['latitude'], route['longitude'])
                if is_directionally_consistent_from_dicts(test_users, driver_pos, (office_lat, office_lon)):
                    compatible_users.append((distance, user))
        
        # Sort by distance and add users
        compatible_users.sort(key=lambda x: x[0])
        users_to_add = []
        
        for distance, user in compatible_users[:available_seats]:
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
            # Remove assigned users from remaining pool
            assigned_ids = {u['user_id'] for u in users_to_add}
            remaining_users = remaining_users[~remaining_users['user_id'].isin(assigned_ids)]
            
            # Re-optimize route sequence
            route = optimize_route_sequence_improved(route, office_lat, office_lon)
            
            print(f"  🎯 Route {route['driver_id']}: Added {len(users_to_add)} directional users, now {len(route['assigned_users'])}/{route['vehicle_type']} seats")

    print(f"  ✅ DIRECTIONAL capacity assignment: {len(routes)} routes with directional seat filling")
    
    # Calculate and display utilization stats
    total_seats = sum(r['vehicle_type'] for r in routes)
    total_users = sum(len(r['assigned_users']) for r in routes)
    overall_utilization = (total_users / total_seats * 100) if total_seats > 0 else 0
    
    print(f"  📊 Overall seat utilization: {total_users}/{total_seats} ({overall_utilization:.1f}%)")
    
    return routes, assigned_user_ids


def assign_best_driver_to_cluster_capacity_focused(cluster_users, available_drivers, used_driver_ids, office_lat, office_lon):
    """Find and assign the best available driver with MAXIMUM capacity utilization focus"""
    cluster_size = len(cluster_users)

    best_driver = None
    best_score = -float('inf')  # Changed to maximize score instead of minimize cost
    best_sequence = None

    # Ultra-aggressive capacity optimization weights
    capacity_weight = _config.get('capacity_weight', 5.0) * 2  # Double the capacity weight
    direction_weight = _config.get('direction_weight', 1.0) * 0.5  # Halve direction importance

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
            driver, cluster_users, office_lat, office_lon
        )
        distance_penalty = route_cost * 0.1  # Very small distance impact

        # Priority bonus (prefer higher priority drivers but with small impact)
        priority_bonus = (5 - driver['priority']) * 0.1  # Small priority preference

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

        # Quick sequence optimization (but don't remove users for it)
        route = optimize_route_sequence_improved(route, office_lat, office_lon)
        update_route_metrics_improved(route, office_lat, office_lon)
        
        utilization = len(route['assigned_users']) / route['vehicle_type']
        print(f"    🚛 Assigned driver {best_driver['driver_id']}: {len(route['assigned_users'])}/{route['vehicle_type']} seats ({utilization*100:.1f}% utilization)")
        
        return route

    return None


def calculate_route_cost_capacity_focused(driver, cluster_users, office_lat, office_lon):
    """Calculate route cost with capacity optimization focus (allows more zigzag)"""
    if len(cluster_users) == 0:
        return float('inf'), [], 0

    driver_pos = (driver['latitude'], driver['longitude'])
    office_pos = (office_lat, office_lon)

    # Get optimal pickup sequence with capacity focus (less strict on turning)
    sequence = calculate_optimal_sequence_capacity_focused(driver_pos, cluster_users, office_pos)

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

        # Calculate bearing difference between segments (less penalty)
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

    # Calculate mean turning angle (but give it less weight)
    mean_turning_degrees = sum(bearing_differences) / len(bearing_differences) if bearing_differences else 0

    return total_distance, sequence, mean_turning_degrees


def calculate_optimal_sequence_capacity_focused(driver_pos, cluster_users, office_pos):
    """Calculate sequence prioritizing capacity utilization over route efficiency"""
    if len(cluster_users) <= 1:
        return cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)

    users_list = cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)

    # For capacity mode, use simpler distance-based sorting instead of bearing projection
    def distance_from_driver_score(user):
        distance = haversine_distance(driver_pos[0], driver_pos[1], 
                                    user['latitude'], user['longitude'])
        return (distance, user['user_id'])  # Sort by distance, then by user_id

    users_list.sort(key=distance_from_driver_score)

    # Apply much more lenient 2-opt with focus on distance rather than turning
    return apply_lenient_2opt_capacity_focused(users_list, driver_pos, office_pos)


def is_directionally_consistent(users, driver_pos, office_pos, lenient=False):
    """Check if a group of users maintains directional consistency"""
    if len(users) <= 1:
        return True
    
    office_lat, office_lon = office_pos
    
    # Calculate bearings from office to each user
    bearings = []
    for user in users:
        bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
        bearings.append(bearing)
    
    # Check bearing spread
    max_bearing_spread = 60 if lenient else 45  # Allow more spread in lenient mode
    
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


def is_directionally_consistent_from_dicts(user_dicts, driver_pos, office_pos, lenient=False):
    """Check directional consistency for user dictionaries (with lat/lng keys)"""
    if len(user_dicts) <= 1:
        return True
    
    office_lat, office_lon = office_pos
    
    # Calculate bearings from office to each user
    bearings = []
    for user in user_dicts:
        bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])
        bearings.append(bearing)
    
    # Check bearing spread
    max_bearing_spread = 60 if lenient else 45  # Allow more spread in lenient mode
    
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
    max_turning_threshold = _config.get('MAX_TURNING_ANGLE', 60) * 2  # Much more lenient

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        best_distance = calculate_sequence_distance(sequence, driver_pos, office_pos)
        
        for i in range(len(sequence) - 1):
            for j in range(i + 2, len(sequence)):
                # Try 2-opt swap
                new_sequence = sequence[:i+1] + sequence[i+1:j+1][::-1] + sequence[j+1:]

                # Check if new sequence maintains directional consistency
                if not is_directionally_consistent(new_sequence, driver_pos, office_pos, lenient=True):
                    continue

                # Calculate new metrics
                new_distance = calculate_sequence_distance(new_sequence, driver_pos, office_pos)
                
                # For capacity mode, only care about distance improvement
                if new_distance < best_distance * 0.99:  # Even more lenient distance improvement
                    sequence = new_sequence
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break

    return sequence


# MAIN ASSIGNMENT FUNCTION FOR CAPACITY OPTIMIZATION
def run_assignment_capacity(source_id: str, parameter: int = 1, string_param: str = ""):
    """
    Main assignment function optimized for capacity utilization:
    - Prioritizes filling vehicle seats over route efficiency
    - Allows zigzag routes and higher turning angles
    - Maximizes utilization across all vehicles
    """
    start_time = time.time()

    # Reload configuration for capacity optimization
    global _config
    _config = load_and_validate_config()
    
    # Update global variables from new config
    global MAX_FILL_DISTANCE_KM, MERGE_DISTANCE_KM, MAX_BEARING_DIFFERENCE, UTILIZATION_PENALTY_PER_SEAT
    MAX_FILL_DISTANCE_KM = _config['MAX_FILL_DISTANCE_KM']
    MERGE_DISTANCE_KM = _config['MERGE_DISTANCE_KM'] 
    MAX_BEARING_DIFFERENCE = _config['MAX_BEARING_DIFFERENCE']
    UTILIZATION_PENALTY_PER_SEAT = _config['UTILIZATION_PENALTY_PER_SEAT']

    print(f"🚀 Starting CAPACITY OPTIMIZATION assignment for source_id: {source_id}")
    print(f"📋 Parameter: {parameter}, String parameter: {string_param}")

    try:
        # Load and validate data
        data = load_env_and_fetch_data(source_id, parameter, string_param)

        # Edge case handling
        users = data.get('users', [])
        if not users:
            print("⚠️ No users found - returning empty assignment")
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
            print("⚠️ No drivers available - all users unassigned")
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

        print(f"📥 Data loaded - Users: {len(users)}, Total Drivers: {len(all_drivers)}")

        # Extract office coordinates and validate data
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        print("✅ Data validation passed")

        # Prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)
        
        print(f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")

        # STEP 1: Geographic clustering (less strict for capacity mode)
        user_df = create_geographic_clusters(user_df, office_lat, office_lon, _config)
        clustering_results = {"method": "capacity_focused_" + _config['clustering_method'], 
                            "clusters": user_df['geo_cluster'].nunique()}

        # STEP 2: Capacity-based sub-clustering (more lenient for capacity filling)
        user_df = create_capacity_subclusters(user_df, office_lat, office_lon, _config)

        # STEP 3: Capacity-focused driver assignment
        routes, assigned_user_ids = assign_drivers_by_priority_capacity_focused(
            user_df, driver_df, office_lat, office_lon)

        # STEP 4: Local optimization (less strict on turning)
        routes = local_optimization(routes, office_lat, office_lon)

        # STEP 5: Aggressive global optimization for capacity filling
        routes, unassigned_users = global_optimization(routes, user_df,
                                                       assigned_user_ids, driver_df, office_lat, office_lon)

        # STEP 6: Aggressive final-pass merge for maximum capacity utilization
        routes = final_pass_merge_capacity_focused(routes, _config, office_lat, office_lon)

        # Filter out routes with no assigned users and move those drivers to unassigned
        filtered_routes = []
        empty_route_driver_ids = set()
        
        for route in routes:
            if route['assigned_users'] and len(route['assigned_users']) > 0:
                filtered_routes.append(route)
            else:
                empty_route_driver_ids.add(route['driver_id'])
                print(f"  📋 Moving driver {route['driver_id']} with no users to unassigned drivers")
        
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
        
        print(f"✅ Capacity optimization complete in {execution_time:.2f}s")
        print(f"📊 Final routes: {len(routes)}")
        print(f"🎯 Users assigned: {users_assigned}")
        print(f"👥 Users unassigned: {users_unassigned}")
        print(f"📋 User accounting: {users_accounted_for}/{total_users_in_api} users")

        return {
            "status": "true",
            "execution_time": execution_time,
            "data": routes,
            "unassignedUsers": unassigned_users,
            "unassignedDrivers": unassigned_drivers,
            "clustering_analysis": clustering_results,
            "optimization_mode": "capacity_optimization",
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


def final_pass_merge_capacity_focused(routes, config, office_lat, office_lon):
    """
    DIRECTIONAL capacity-focused final-pass merge: Fill seats while maintaining directional consistency
    """
    print("🔄 Step 6: DIRECTIONAL capacity-focused final-pass merge...")

    merged_routes = []
    used = set()
    
    # Balanced thresholds for capacity + direction
    MERGE_BEARING_THRESHOLD = 35  # Stricter directional requirement
    MERGE_DISTANCE_KM = config.get("MERGE_DISTANCE_KM", 5.0) * 1.5  # Moderate distance tolerance

    for i, r1 in enumerate(routes):
        if i in used:
            continue

        best_merge = None
        best_total_utilization = len(r1['assigned_users']) / r1['vehicle_type']  # Start with current utilization

        for j, r2 in enumerate(routes):
            if j <= i or j in used:
                continue

            # 1. Check direction similarity (strict for directional consistency)
            b1 = calculate_average_bearing_improved(r1, office_lat, office_lon)
            b2 = calculate_average_bearing_improved(r2, office_lat, office_lon)
            bearing_diff = bearing_difference(b1, b2)

            # Reject if not in same general direction
            if bearing_diff > MERGE_BEARING_THRESHOLD:
                continue

            # 2. Check centroid distance
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

            # 4. Check directional consistency of merged route
            all_users = r1['assigned_users'] + r2['assigned_users']
            driver_pos = (r1['latitude'], r1['longitude']) if r1['vehicle_type'] >= r2['vehicle_type'] else (r2['latitude'], r2['longitude'])
            
            if not is_directionally_consistent_from_dicts(all_users, driver_pos, (office_lat, office_lon), lenient=True):
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
                test_route = optimize_route_sequence_improved(test_route, office_lat, office_lon)

                # Accept merge if directional and has good utilization
                best_total_utilization = merged_utilization
                best_merge = (j, test_route)

        if best_merge is not None:
            j, merged_route = best_merge
            merged_routes.append(merged_route)
            used.add(i)
            used.add(j)
            
            utilization_pct = len(merged_route['assigned_users']) / merged_route['vehicle_type'] * 100
            print(f"  🧭 DIRECTIONAL merge: routes {r1['driver_id']} + {routes[j]['driver_id']} = {len(merged_route['assigned_users'])}/{merged_route['vehicle_type']} seats ({utilization_pct:.1f}%)")
        else:
            merged_routes.append(r1)
            used.add(i)

    # ADDITIONAL AGGRESSIVE PASS: Try to redistribute users to maximize overall utilization
    print("  🎯 Additional seat-filling pass...")
    
    # Sort routes by utilization (ascending) to fill up underutilized routes first
    merged_routes_with_util = []
    for route in merged_routes:
        utilization = len(route['assigned_users']) / route['vehicle_type']
        merged_routes_with_util.append((utilization, route))
    
    merged_routes_with_util.sort(key=lambda x: x[0])  # Sort by utilization ascending
    
    # Try to move users from high-utilization routes to low-utilization routes for better overall filling
    for i, (util_low, route_low) in enumerate(merged_routes_with_util):
        if util_low >= 0.9:  # Already well utilized
            continue
            
        available_seats = route_low['vehicle_type'] - len(route_low['assigned_users'])
        if available_seats <= 0:
            continue
        
        # Look for users in higher-utilization routes that could be moved
        for j, (util_high, route_high) in enumerate(merged_routes_with_util[i+1:], start=i+1):
            if len(route_high['assigned_users']) <= 1:  # Don't empty routes completely
                continue
                
            # Find users in the high-util route that are closer to the low-util route
            users_to_consider = []
            route_low_center = calculate_route_center_improved(route_low)
            
            for user in route_high['assigned_users']:
                distance_to_low = haversine_distance(route_low_center[0], route_low_center[1],
                                                   user['lat'], user['lng'])
                if distance_to_low <= MERGE_DISTANCE_KM:  # Close enough to move
                    users_to_consider.append((distance_to_low, user))
            
            # Move closest users to fill the low-utilization route
            users_to_consider.sort(key=lambda x: x[0])
            users_moved = 0
            
            for distance, user in users_to_consider[:available_seats]:
                # Move the user
                route_high['assigned_users'].remove(user)
                route_low['assigned_users'].append(user)
                users_moved += 1
                available_seats -= 1
                
                if available_seats <= 0:
                    break
            
            if users_moved > 0:
                # Re-optimize both routes
                route_low = optimize_route_sequence_improved(route_low, office_lat, office_lon)
                route_high = optimize_route_sequence_improved(route_high, office_lat, office_lon)
                
                # Update utilizations
                new_util_low = len(route_low['assigned_users']) / route_low['vehicle_type']
                new_util_high = len(route_high['assigned_users']) / route_high['vehicle_type']
                
                merged_routes_with_util[i] = (new_util_low, route_low)
                merged_routes_with_util[j] = (new_util_high, route_high)
                
                print(f"    🔄 Redistributed {users_moved} users: Route {route_high['driver_id']} → Route {route_low['driver_id']}")
                print(f"       New utilizations: {new_util_low*100:.1f}%, {new_util_high*100:.1f}%")
                
                if available_seats <= 0:
                    break

    # Extract final routes
    final_routes = [route for _, route in merged_routes_with_util]
    
    # Final statistics
    total_seats = sum(r['vehicle_type'] for r in final_routes)
    total_users = sum(len(r['assigned_users']) for r in final_routes)
    overall_utilization = (total_users / total_seats * 100) if total_seats > 0 else 0
    
    print(f"  🎯 DIRECTIONAL capacity merge complete: {len(routes)} → {len(final_routes)} routes")
    print(f"  🧭 Final directional seat utilization: {total_users}/{total_seats} ({overall_utilization:.1f}%)")
    
    return final_routes
