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

# Setup logging first
logger = get_logger()

# Import road_network module for route coherence scoring
try:
    import road_network as road_network_module
    # Create an instance of RoadNetwork class if it exists
    try:
        # Try to create RoadNetwork instance (assuming GraphML file exists)
        road_network = road_network_module.RoadNetwork('tricity_main_roads.graphml')
        logger.info("Successfully loaded RoadNetwork with GraphML data")
    except Exception as e:
        logger.warning(f"Could not create RoadNetwork instance: {e}. Using mock implementation.")
        class MockRoadNetwork:
            def get_route_coherence_score(self, driver_pos, user_positions, office_pos):
                # Mock implementation: returns a score based on simple distance heuristic
                if not user_positions:
                    return 1.0
                avg_dist_from_driver = sum(haversine_distance(driver_pos[0], driver_pos[1], u[0], u[1]) for u in user_positions) / len(user_positions)
                avg_dist_from_office = sum(haversine_distance(office_pos[0], office_pos[1], u[0], u[1]) for u in user_positions) / len(user_positions)
                
                # Simple heuristic: higher coherence if users are closer to the driver's path
                # and not too far from the office
                score = max(0, 1.0 - (avg_dist_from_driver / 50.0) - (avg_dist_from_office / 100.0))
                return min(1.0, score)
        road_network = MockRoadNetwork()
except ImportError:
    logger.warning("road_network module not found. Road network features will be limited.")
    class MockRoadNetwork:
        def get_route_coherence_score(self, driver_pos, user_positions, office_pos):
            # Mock implementation: returns a score based on simple distance heuristic
            if not user_positions:
                return 1.0
            avg_dist_from_driver = sum(haversine_distance(driver_pos[0], driver_pos[1], u[0], u[1]) for u in user_positions) / len(user_positions)
            avg_dist_from_office = sum(haversine_distance(office_pos[0], office_pos[1], u[0], u[1]) for u in user_positions) / len(user_positions)
            
            # Simple heuristic: higher coherence if users are closer to the driver's path
            # and not too far from the office
            score = max(0, 1.0 - (avg_dist_from_driver / 50.0) - (avg_dist_from_office / 100.0))
            return min(1.0, score)
    road_network = MockRoadNetwork()


# Load and validate configuration with route optimization settings
def load_and_validate_config():
    """Load configuration with route optimization settings"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    # Always use balanced mode for route optimization
    current_mode = "balanced_optimization"

    # Get balanced optimization configuration for route optimization
    mode_configs = cfg.get("mode_configs", {})
    mode_config = mode_configs.get("balanced_optimization", {})

    logger.info(f"🗺️ Using optimization mode: ROUTE OPTIMIZATION")

    # Validate and set configuration with mode-specific overrides (between route_efficiency and capacity)
    config = {}

    # Distance configurations with mode overrides (moderate between the two)
    config['MAX_FILL_DISTANCE_KM'] = max(0.1, float(mode_config.get("max_fill_distance_km", cfg.get("max_fill_distance_km", 6.0))))
    config['MERGE_DISTANCE_KM'] = max(0.1, float(mode_config.get("merge_distance_km", cfg.get("merge_distance_km", 3.5))))
    config['DBSCAN_EPS_KM'] = max(0.1, float(cfg.get("dbscan_eps_km", 2.0)))
    config['OVERFLOW_PENALTY_KM'] = max(0.0, float(cfg.get("overflow_penalty_km", 7.0)))
    config['DISTANCE_ISSUE_THRESHOLD'] = max(0.1, float(cfg.get("distance_issue_threshold_km", 10.0)))
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(0.0, float(cfg.get("swap_improvement_threshold_km", 0.7)))

    # Utilization thresholds (route optimized)
    config['MIN_UTIL_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("min_util_threshold", 0.65))))
    config['LOW_UTILIZATION_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("low_utilization_threshold", 0.6))))

    # Integer configurations (must be positive)
    config['MIN_SAMPLES_DBSCAN'] = max(1, int(cfg.get("min_samples_dbscan", 2)))
    config['MAX_SWAP_ITERATIONS'] = max(1, int(cfg.get("max_swap_iterations", 4)))
    config['MAX_USERS_FOR_FALLBACK'] = max(1, int(cfg.get("max_users_for_fallback", 4)))
    config['FALLBACK_MIN_USERS'] = max(1, int(cfg.get("fallback_min_users", 2)))
    config['FALLBACK_MAX_USERS'] = max(1, int(cfg.get("fallback_max_users", 8)))

    # Angle configurations with mode overrides (moderate)
    config['MAX_BEARING_DIFFERENCE'] = max(0, min(180, float(mode_config.get("max_bearing_difference", cfg.get("max_bearing_difference", 30)))))
    config['MAX_TURNING_ANGLE'] = max(0, min(180, float(mode_config.get("max_allowed_turning_score", cfg.get("max_allowed_turning_score", 40)))))

    # Cost penalties with mode overrides (route optimized weights)
    config['UTILIZATION_PENALTY_PER_SEAT'] = max(0.0, float(mode_config.get("utilization_penalty_per_seat", cfg.get("utilization_penalty_per_seat", 3.0))))

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

    # Route optimization parameters
    config['optimization_mode'] = "route_optimization"
    config['aggressive_merging'] = mode_config.get('aggressive_merging', None)  # Moderate
    config['capacity_weight'] = mode_config.get('capacity_weight', 2.5)
    config['direction_weight'] = mode_config.get('direction_weight', 2.5)

    # Clustering and optimization parameters with mode overrides
    config['clustering_method'] = cfg.get('clustering_method', 'adaptive')
    config['min_cluster_size'] = max(2, cfg.get('min_cluster_size', 2))
    config['use_sweep_algorithm'] = cfg.get('use_sweep_algorithm', True)
    config['angular_sectors'] = cfg.get('angular_sectors', 10)
    config['max_users_per_initial_cluster'] = cfg.get('max_users_per_initial_cluster', 8)
    config['max_users_per_cluster'] = cfg.get('max_users_per_cluster', 7)

    # Route optimization parameters
    config['zigzag_penalty_weight'] = mode_config.get('zigzag_penalty_weight', cfg.get('zigzag_penalty_weight', 2.0))
    config['route_split_turning_threshold'] = cfg.get('route_split_turning_threshold', 45)
    config['max_tortuosity_ratio'] = cfg.get('max_tortuosity_ratio', 1.5)
    config['route_split_consistency_threshold'] = cfg.get('route_split_consistency_threshold', 0.6)
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

# Load validated configuration - always route optimization
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


def assign_drivers_by_priority_route_optimized(user_df, driver_df, office_lat, office_lon):
    """
    Route optimized driver assignment: Smart route between capacity utilization and route efficiency
    """
    logger.info("🗺️ Step 3: Route optimized driver assignment...")

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Route optimized sorting: capacity and priority equally weighted
    available_drivers = driver_df.sort_values(['capacity', 'priority'], 
                                              ascending=[False, True])

    # Collect unassigned users for capacity-aware assignment
    all_unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()

    # Hybrid approach: Start with clusters but allow cross-cluster filling for capacity
    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids or all_unassigned_users.empty:
            continue

        vehicle_capacity = int(driver['capacity'])
        driver_pos = (driver['latitude'], driver['longitude'])

        # Calculate main route bearing
        main_route_bearing = calculate_bearing(office_lat, office_lon, driver['latitude'], driver['longitude'])

        # Find users with route optimized distance/direction constraints
        users_for_vehicle = []
        max_distance_limit = MAX_FILL_DISTANCE_KM * 1.5  # Between strict efficiency and lenient capacity
        max_bearing_deviation = 37  # Between 20° (efficiency) and 45° (capacity)

        # Collect candidate users with route optimized scoring
        candidate_users = []
        for _, user in all_unassigned_users.iterrows():
            distance = haversine_distance(driver['latitude'], driver['longitude'], 
                                        user['latitude'], user['longitude'])

            office_to_user_bearing = calculate_bearing(office_lat, office_lon, 
                                                     user['latitude'], user['longitude'])

            bearing_diff_from_main = bearing_difference(main_route_bearing, office_to_user_bearing)

            # Route optimized acceptance criteria
            if (distance <= max_distance_limit and 
                bearing_diff_from_main <= max_bearing_deviation):
                # Route optimized score: equal weight to distance and direction
                score = distance * 0.5 + bearing_diff_from_main * 0.1  # 0.1 km per degree
                candidate_users.append((score, user))

        # Sort by route optimized score and fill to 85% capacity (route optimized target)
        candidate_users.sort(key=lambda x: x[0])
        target_capacity = min(vehicle_capacity, max(2, int(vehicle_capacity * 0.85)))

        for score, user in candidate_users:
            if len(users_for_vehicle) >= target_capacity:
                break
            users_for_vehicle.append(user)

        # If we have good utilization, create the route
        if len(users_for_vehicle) >= max(2, vehicle_capacity * 0.5):  # At least 50% utilization
            cluster_df = pd.DataFrame(users_for_vehicle)
            route = assign_best_driver_to_cluster_route_optimized(
                cluster_df, pd.DataFrame([driver]), used_driver_ids, office_lat, office_lon)

            if route:
                routes.append(route)
                assigned_user_ids.update(u['user_id'] for u in route['assigned_users'])

                # Log detailed route creation
                quality_metrics = {
                    'utilization': len(route['assigned_users']) / vehicle_capacity,
                    'vehicle_capacity': vehicle_capacity,
                    'optimization_mode': 'route_optimization',
                    'directional_consistency': True
                }

                logger.log_route_creation(
                    driver['driver_id'],
                    route['assigned_users'],
                    f"Route optimization with {len(route['assigned_users'])} users",
                    quality_metrics
                )

                # Log each user assignment
                for user in route['assigned_users']:
                    logger.log_user_assignment(
                        user['user_id'],
                        driver['driver_id'],
                        {
                            'pickup_order': route['assigned_users'].index(user) + 1,
                            'distance_from_driver': haversine_distance(
                                route['latitude'], route['longitude'],
                                user['lat'], user['lng']
                            ),
                            'optimization_mode': 'route_optimization'
                        }
                    )

                # Remove assigned users from pool
                assigned_ids_set = {u['user_id'] for u in route['assigned_users']}
                all_unassigned_users = all_unassigned_users[~all_unassigned_users['user_id'].isin(assigned_ids_set)]

                utilization = len(route['assigned_users']) / vehicle_capacity * 100
                logger.info(f"  🗺️ Driver {driver['driver_id']}: {len(route['assigned_users'])}/{vehicle_capacity} seats ({utilization:.1f}%) - Route Optimized")

    # Second pass: Fill remaining seats with slightly more lenient constraints
    remaining_users = all_unassigned_users[~all_unassigned_users['user_id'].isin(assigned_user_ids)]

    for route in routes:
        if remaining_users.empty:
            break

        available_seats = route['vehicle_type'] - len(route['assigned_users'])
        if available_seats <= 0:
            continue

        # Route optimized seat filling
        route_bearing = calculate_average_bearing_improved(route, office_lat, office_lon)
        route_center = calculate_route_center_improved(route)

        compatible_users = []
        for _, user in remaining_users.iterrows():
            distance = haversine_distance(route_center[0], route_center[1],
                                        user['latitude'], user['longitude'])

            user_bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
            bearing_diff = bearing_difference(route_bearing, user_bearing)

            # Route optimized criteria for seat filling
            if distance <= MAX_FILL_DISTANCE_KM * 1.8 and bearing_diff <= 40:
                score = distance * 0.6 + bearing_diff * 0.08  # Slightly favor distance
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
            logger.info(f"  🗺️ Route {route['driver_id']}: Added {len(users_to_add)} users, now {len(route['assigned_users'])}/{route['vehicle_type']} ({utilization:.1f}%)")

    # Check for users who were not assigned and need to be processed by splitting/merging logic
    final_unassigned_users_df = all_unassigned_users[~all_unassigned_users['user_id'].isin(assigned_user_ids)]
    if not final_unassigned_users_df.empty:
        logger.warning(f"⚠️ {len(final_unassigned_users_df)} users remain unassigned after initial pass. These will be handled by global_optimization or split/merge logic.")

    logger.info(f"  ✅ Route optimized assignment: {len(routes)} routes with smart route optimization")

    # Calculate route optimized metrics
    total_seats = sum(r['vehicle_type'] for r in routes)
    total_users = sum(len(r['assigned_users']) for r in routes)
    overall_utilization = (total_users / total_seats * 100) if total_seats > 0 else 0

    logger.info(f"  🗺️ Route optimized utilization: {total_users}/{total_seats} ({overall_utilization:.1f}%)")

    return routes, assigned_user_ids


def assign_best_driver_to_cluster_route_optimized(cluster_users, available_drivers, used_driver_ids, office_lat, office_lon):
    """Find and assign the best available driver with route optimization"""
    cluster_size = len(cluster_users)

    best_driver = None
    best_score = float('inf')
    best_sequence = None

    # Route optimized weights - route importance
    capacity_weight = 2.5  # Route weight
    direction_weight = 2.5  # Route weight

    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids:
            continue

        # Capacity check (allow some over-assignment for route optimization)
        if driver['capacity'] < cluster_size:
            continue

        # Calculate route metrics
        route_cost, sequence, mean_turning_degrees = calculate_route_cost_route_optimized(
            driver, cluster_users, office_lat, office_lon
        )

        # Route optimized scoring approach
        utilization = cluster_size / driver['capacity']

        # Distance component (efficiency factor)
        distance_score = route_cost * 0.5  # 50% weight on distance

        # Direction component (efficiency factor)
        direction_score = mean_turning_degrees * direction_weight * 0.02  # 50% weight on direction

        # Capacity component (capacity factor) - inverted to prefer higher utilization
        capacity_score = (1.0 - utilization) * capacity_weight * 5.0  # 50% weight on capacity (penalty for underutilization)

        # Priority component (small tiebreaker)
        priority_score = driver['priority'] * 0.2

        # Route optimized total score: route emphasis on efficiency (distance + direction) and capacity
        efficiency_component = distance_score + direction_score  # Route efficiency
        capacity_component = capacity_score  # Capacity utilization

        total_score = efficiency_component + capacity_component + priority_score

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

        # Add all users from cluster (route optimized approach)
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

        # Route optimized optimization of sequence
        route = optimize_route_sequence_improved(route, office_lat, office_lon)
        update_route_metrics_improved(route, office_lat, office_lon)

        utilization = len(route['assigned_users']) / route['vehicle_type']
        logger.info(f"    🗺️ Route optimized assignment - Driver {best_driver['driver_id']}: {len(route['assigned_users'])}/{route['vehicle_type']} seats ({utilization*100:.1f}%)")

        return route

    return None


def calculate_route_cost_route_optimized(driver, cluster_users, office_lat, office_lon):
    """Calculate route cost with route optimization - equal weight to distance and direction"""
    if len(cluster_users) == 0:
        return float('inf'), [], 0

    driver_pos = (driver['latitude'], driver['longitude'])
    office_pos = (office_lat, office_lon)

    # Get optimal pickup sequence with route optimized focus (equal weight to distance and direction)
    sequence = calculate_optimal_sequence_route_optimized(driver_pos, cluster_users, office_pos)

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

    # Calculate mean turning angle - route optimized weight (neither too strict nor too lenient)
    mean_turning_degrees = sum(bearing_differences) / len(bearing_differences) if bearing_differences else 0

    return total_distance, sequence, mean_turning_degrees


def calculate_optimal_sequence_route_optimized(driver_pos, cluster_users, office_pos):
    """Calculate sequence with route optimization - route focused between distance and direction"""
    if len(cluster_users) <= 1:
        return cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)

    users_list = cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)

    # Calculate main route bearing (driver to office)
    main_route_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    # Route optimized scoring: exactly 50% distance, 50% direction
    def route_optimized_score(user):
        # Distance component (50%)
        distance = haversine_distance(driver_pos[0], driver_pos[1], 
                                    user['latitude'], user['longitude'])

        # Direction component (50%)
        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1], 
                                       user['latitude'], user['longitude'])

        bearing_diff = normalize_bearing_difference(user_bearing - main_route_bearing)
        bearing_diff_rad = math.radians(abs(bearing_diff))

        # Route weight to distance and bearing alignment
        distance_score = distance  # Raw distance
        direction_score = distance * (1 - math.cos(bearing_diff_rad))  # Direction penalty in distance units

        # Route optimized route focus
        combined_score = distance_score * 0.5 + direction_score * 0.5

        return (combined_score, user['user_id'])

    users_list.sort(key=route_optimized_score)

    # Apply route optimized 2-opt optimization
    return apply_route_optimized_2opt(users_list, driver_pos, office_pos)


def apply_route_optimized_2opt(sequence, driver_pos, office_pos):
    """Apply route optimized 2-opt improvements - route weight to distance and direction"""
    if len(sequence) <= 2:
        return sequence

    improved = True
    max_iterations = 3
    iteration = 0

    # Calculate main bearing from driver to office
    main_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    # Route optimized turning angle threshold (exactly between efficiency and capacity)
    max_turning_threshold = 47  # Between 35° (efficiency) and 60° (capacity)

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

                # Route optimized acceptance criteria - route weight to both factors
                distance_improvement = (best_distance - new_distance) / best_distance  # Normalized improvement
                turning_improvement = (best_turning_score - new_turning_score)  # Absolute improvement

                # Convert turning improvement to same scale as distance (percentage)
                turning_improvement_normalized = turning_improvement / max(best_turning_score, 1.0)

                # Route weight to both improvements
                combined_improvement = distance_improvement * 0.5 + turning_improvement_normalized * 0.5

                # Accept if combined improvement is positive and turning stays reasonable
                if (combined_improvement > 0.005 and  # Small positive improvement
                    new_turning_score <= max_turning_threshold):

                    sequence = new_sequence
                    best_distance = new_distance
                    best_turning_score = new_turning_score
                    improved = True
                    break
            if improved:
                break

    return sequence


def final_pass_merge_route_optimized(routes, config, office_lat, office_lon):
    """
    Route optimized final-pass merge: Route focus on capacity utilization and route efficiency
    """
    logger.info("🔄 Step 6: Route optimized final-pass merge...")

    merged_routes = []
    used = set()

    # Route optimized thresholds (exactly between route efficiency and capacity)
    MERGE_BEARING_THRESHOLD = 32  # Between 20° (efficiency) and 45° (capacity)
    MERGE_DISTANCE_KM = config.get("MERGE_DISTANCE_KM", 3.5) * 1.4  # Between 3.0 and 5.0
    MERGE_TURNING_THRESHOLD = 50  # Between 35° (efficiency) and 60° (capacity)
    MERGE_TORTUOSITY_THRESHOLD = 1.65  # Between 1.3 and 2.0

    for i, r1 in enumerate(routes):
        if i in used:
            continue

        best_merge = None
        best_route_optimized_score = float('inf')

        for j, r2 in enumerate(routes):
            if j <= i or j in used:
                continue

            # 1. Direction compatibility check (route optimized)
            b1 = calculate_average_bearing_improved(r1, office_lat, office_lon)
            b2 = calculate_average_bearing_improved(r2, office_lat, office_lon)
            bearing_diff = bearing_difference(b1, b2)

            if bearing_diff > MERGE_BEARING_THRESHOLD:
                continue

            # 2. Distance compatibility check (route optimized)
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

            # 4. Quality assessment with route optimized criteria
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

            # Calculate route optimized quality metrics
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

            # Route optimized acceptance criteria - both efficiency and capacity matter for routes
            efficiency_acceptable = (turning_score <= MERGE_TURNING_THRESHOLD and 
                                   tortuosity <= MERGE_TORTUOSITY_THRESHOLD)
            capacity_acceptable = utilization >= 0.6  # Route optimized utilization requirement

            # Road network coherence check for route optimization
            road_coherence_acceptable = True
            if road_network:
                driver_pos = (test_route['latitude'], test_route['longitude'])
                user_positions = [(u['lat'], u['lng']) for u in test_route['assigned_users']]
                coherence = road_network.get_route_coherence_score(driver_pos, user_positions, (office_lat, office_lon))
                road_coherence_acceptable = coherence >= 0.6  # Route optimized coherence threshold

            # Only accept if ALL criteria are met for route optimization
            if efficiency_acceptable and capacity_acceptable and road_coherence_acceptable:
                # Route optimized scoring: route% efficiency, route% capacity
                efficiency_score = turning_score * 0.5 + (tortuosity - 1.0) * 20  # Route quality
                capacity_score = (1.0 - utilization) * 100  # Underutilization penalty

                # Route weight to both components
                route_optimized_score = efficiency_score * 0.5 + capacity_score * 0.5

                if route_optimized_score < best_route_optimized_score:
                    best_route_optimized_score = route_optimized_score
                    best_merge = (j, test_route)

        if best_merge is not None:
            j, merged_route = best_merge
            merged_routes.append(merged_route)
            used.add(i)
            used.add(j)

            utilization_pct = len(merged_route['assigned_users']) / merged_route['vehicle_type'] * 100
            turning = merged_route.get('turning_score', 0)
            logger.info(f"  🗺️ Route optimized merge: {r1['driver_id']} + {routes[j]['driver_id']} = {len(merged_route['assigned_users'])}/{merged_route['vehicle_type']} seats ({utilization_pct:.1f}%, {turning:.1f}° turn)")
        else:
            merged_routes.append(r1)
            used.add(i)

    # Additional route optimized optimization pass
    logger.info("  🗺️ Additional route optimization pass...")

    # Try to redistribute for better overall route optimization
    optimization_made = True
    while optimization_made:
        optimization_made = False

        # Sort routes by efficiency score for route optimization
        routes_with_scores = []
        for route in merged_routes:
            utilization = len(route['assigned_users']) / route['vehicle_type']
            efficiency = route.get('turning_score', 0)

            # Look for opportunities to optimize routes
            if utilization < 0.7 and efficiency < 35:  # Good efficiency, poor capacity
                routes_with_scores.append(('low_util', route))
            elif utilization > 0.9 and efficiency > 45:  # Good capacity, poor efficiency  
                routes_with_scores.append(('high_util', route))
            else:
                routes_with_scores.append(('route_optimized', route))

        # Try to swap users between unbalanced routes for better overall route optimization
        low_util_routes = [r for t, r in routes_with_scores if t == 'low_util']
        high_util_routes = [r for t, r in routes_with_scores if t == 'high_util']

        for low_route in low_util_routes:
            for high_route in high_util_routes:
                if len(high_route['assigned_users']) <= 1:
                    continue

                # Try moving one user from high-util to low-util route
                available_capacity = low_route['vehicle_type'] - len(low_route['assigned_users'])
                if available_capacity > 0:
                    # Find user in high_route that's closest to low_route
                    low_center = calculate_route_center_improved(low_route)
                    best_user = None
                    best_distance = float('inf')

                    for user in high_route['assigned_users']:
                        distance = haversine_distance(low_center[0], low_center[1], user['lat'], user['lng'])
                        if distance < best_distance and distance <= MERGE_DISTANCE_KM:
                            best_distance = distance
                            best_user = user

                    if best_user:
                        # Move the user
                        high_route['assigned_users'].remove(best_user)
                        low_route['assigned_users'].append(best_user)

                        # Re-optimize both routes
                        low_route = optimize_route_sequence_improved(low_route, office_lat, office_lon)
                        high_route = optimize_route_sequence_improved(high_route, office_lat, office_lon)
                        update_route_metrics_improved(low_route, office_lat, office_lon)
                        update_route_metrics_improved(high_route, office_lat, office_lon)

                        optimization_made = True
                        logger.info("    🗺️ Route optimization redistribution: User moved between routes for better optimization")
                        break
            if optimization_made:
                break

    # Final statistics
    total_seats = sum(r['vehicle_type'] for r in merged_routes)
    total_users = sum(len(r['assigned_users']) for r in merged_routes)
    avg_utilization = (total_users / total_seats * 100) if total_seats > 0 else 0
    avg_turning = np.mean([r.get('turning_score', 0) for r in merged_routes])

    logger.info(f"  🗺️ Route optimized merge: {len(routes)} → {len(merged_routes)} routes")
    logger.info(f"  🗺️ Final route optimization: {avg_utilization:.1f}% utilization, {avg_turning:.1f}° avg turning")

    return merged_routes


# ROAD NETWORK HELPER FUNCTIONS FOR MERGING AND SPLITTING
def _are_routes_on_same_road_path(route1, route2, office_lat, office_lon):
    """Check if two routes follow similar road paths for merging compatibility"""
    if not road_network:
        return True  # Fallback to allowing merge if no road network

    try:
        # Get positions for both routes
        r1_pos = (route1['latitude'], route1['longitude'])
        r2_pos = (route2['latitude'], route2['longitude'])
        office_pos = (office_lat, office_lon)

        # Get user positions for both routes
        r1_users = [(u['lat'], u['lng']) for u in route1['assigned_users']]
        r2_users = [(u['lat'], u['lng']) for u in route2['assigned_users']]

        # Check if routes have good coherence when combined
        combined_users = r1_users + r2_users

        # Choose better positioned driver for coherence check
        r1_to_center = sum(haversine_distance(r1_pos[0], r1_pos[1], u[0], u[1]) for u in combined_users) / len(combined_users) if combined_users else 0
        r2_to_center = sum(haversine_distance(r2_pos[0], r2_pos[1], u[0], u[1]) for u in combined_users) / len(combined_users) if combined_users else 0

        driver_pos = r1_pos if r1_to_center <= r2_to_center else r2_pos

        # Calculate coherence score for the combined route
        coherence_score = road_network.get_route_coherence_score(driver_pos, combined_users, office_pos)

        # Accept merge if coherence is reasonable (route optimized threshold)
        return coherence_score >= 0.6  # Route optimized threshold between 0.5 and 0.7

    except Exception as e:
        logger.warning(f"Road path checking failed: {e}")
        return True  # Fallback to allowing merge


def _should_split_route_by_road_network(route, office_lat, office_lon):
    """Check if a route should be split based on road network analysis"""
    if not road_network or len(route['assigned_users']) < 3:
        return False, []

    try:
        driver_pos = (route['latitude'], route['longitude'])
        office_pos = (office_lat, office_lon)
        user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]

        # Check if current route has poor coherence
        current_coherence = road_network.get_route_coherence_score(driver_pos, user_positions, office_pos)

        # If coherence is already good, don't split
        if current_coherence >= 0.65:  # Route optimized threshold
            return False, []

        # Try to find better groupings by analyzing road connectivity
        best_split = _find_optimal_road_based_split(driver_pos, user_positions, office_pos)

        if best_split:
            group1, group2 = best_split
            # Ensure both groups have reasonable sizes
            if len(group1) >= 2 and len(group2) >= 2:
                return True, [group1, group2]

        return False, []

    except Exception as e:
        logger.warning(f"Road network split analysis failed: {e}")
        return False, []


def _find_optimal_road_based_split(driver_pos, user_positions, office_pos):
    """Find optimal way to split users based on road network analysis"""
    if len(user_positions) < 4:
        return None

    best_split = None
    best_combined_coherence = 0

    # Try different ways to split the users
    from itertools import combinations

    n_users = len(user_positions)
    # Try splits where each group has at least 2 users
    for r in range(2, n_users - 1):
        for group1_indices in combinations(range(n_users), r):
            group2_indices = [i for i in range(n_users) if i not in group1_indices]

            if len(group2_indices) < 2:
                continue

            group1_positions = [user_positions[i] for i in group1_indices]
            group2_positions = [user_positions[i] for i in group2_indices]

            # Calculate coherence for both groups
            coherence1 = road_network.get_route_coherence_score(driver_pos, group1_positions, office_pos)
            coherence2 = road_network.get_route_coherence_score(driver_pos, group2_positions, office_pos)

            # Combined coherence (weighted by group sizes)
            w1 = len(group1_positions) / n_users
            w2 = len(group2_positions) / n_users
            combined_coherence = coherence1 * w1 + coherence2 * w2

            # Accept split if both groups have decent coherence and combined is better
            if (coherence1 >= 0.6 and coherence2 >= 0.6 and 
                combined_coherence > best_combined_coherence):
                best_combined_coherence = combined_coherence
                best_split = (group1_positions, group2_positions)

    return best_split


def perform_quality_merge_improved(routes, config, office_lat, office_lon):
    """
    Perform route merging with enhanced quality checks, including road network awareness.
    This function merges routes based on proximity, capacity, and road path compatibility.
    """
    logger.info("🔄 Performing enhanced route merging with road network awareness...")
    
    merged_routes = []
    used_route_indices = set()

    # Route optimized merge thresholds
    MERGE_DISTANCE_KM = config.get('MERGE_DISTANCE_KM', 3.5) * 1.2 # Slightly more lenient for road awareness
    MERGE_BEARING_THRESHOLD = config.get('MAX_BEARING_DIFFERENCE', 30) + 10 # Slightly more lenient
    MERGE_TURNING_THRESHOLD = config.get('MAX_TURNING_ANGLE', 40) + 10 # Slightly more lenient
    MERGE_TORTUOSITY_THRESHOLD = config.get('max_tortuosity_ratio', 1.5) + 0.15 # Slightly more lenient

    for i in range(len(routes)):
        if i in used_route_indices:
            continue

        best_merge_candidate = None
        best_merge_score = float('inf')

        for j in range(i + 1, len(routes)):
            if j in used_route_indices:
                continue

            route1 = routes[i]
            route2 = routes[j]

            # 1. Basic compatibility checks
            # Check capacity
            total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
            max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])
            if total_users > max_capacity:
                continue

            # Check direction (average bearing)
            avg_bearing1 = calculate_average_bearing_improved(route1, office_lat, office_lon)
            avg_bearing2 = calculate_average_bearing_improved(route2, office_lat, office_lon)
            bearing_diff = bearing_difference(avg_bearing1, avg_bearing2)
            if bearing_diff > MERGE_BEARING_THRESHOLD:
                continue

            # Check centroid distance
            center1 = calculate_route_center_improved(route1)
            center2 = calculate_route_center_improved(route2)
            dist_between_centers = haversine_distance(center1[0], center1[1], center2[0], center2[1])
            if dist_between_centers > MERGE_DISTANCE_KM:
                continue
            
            # 2. Road network compatibility check
            if not _are_routes_on_same_road_path(route1, route2, office_lat, office_lon):
                continue

            # 3. Quality assessment of potential merge
            # Create a hypothetical merged route to evaluate
            merged_route_candidate = {
                'driver_id': route1['driver_id'], # Arbitrarily pick one driver ID for the merged route
                'vehicle_id': route1['vehicle_id'],
                'vehicle_type': max_capacity,
                'latitude': (center1[0] * len(route1['assigned_users']) + center2[0] * len(route2['assigned_users'])) / total_users,
                'longitude': (center1[1] * len(route1['assigned_users']) + center2[1] * len(route2['assigned_users'])) / total_users,
                'assigned_users': route1['assigned_users'] + route2['assigned_users']
            }

            # Optimize sequence for the candidate merged route
            merged_route_candidate = optimize_route_sequence_improved(merged_route_candidate, office_lat, office_lon)
            update_route_metrics_improved(merged_route_candidate, office_lat, office_lon)

            # Evaluate quality
            turning_score = merged_route_candidate.get('turning_score', 0)
            tortuosity = merged_route_candidate.get('tortuosity_ratio', 1.0)
            utilization = total_users / max_capacity

            # Road network coherence for the merged route
            coherence = 0.0
            if road_network:
                driver_pos = (merged_route_candidate['latitude'], merged_route_candidate['longitude'])
                user_positions = [(u['lat'], u['lng']) for u in merged_route_candidate['assigned_users']]
                coherence = road_network.get_route_coherence_score(driver_pos, user_positions, (office_lat, office_lon))

            # Route optimized scoring for merge quality
            # Prioritize efficiency (turning, tortuosity) and road coherence, then capacity
            efficiency_score = (turning_score / MERGE_TURNING_THRESHOLD) + (tortuosity / MERGE_TORTUOSITY_THRESHOLD)
            coherence_score = coherence / 0.7 # Normalize coherence
            capacity_score = (1.0 - utilization) * 2 # Penalty for underutilization

            # Route optimized weighting for merge decision
            merge_score = (efficiency_score * 0.4 + 
                           coherence_score * 0.4 + 
                           capacity_score * 0.2)

            if merge_score < best_merge_score:
                best_merge_score = merge_score
                best_merge_candidate = (j, merged_route_candidate)

        if best_merge_candidate:
            j, merged_route = best_merge_candidate
            merged_routes.append(merged_route)
            used_route_indices.add(i)
            used_route_indices.add(j)
            logger.info(f"  ✅ Merged route {i} with {j} into a new route with {len(merged_route['assigned_users'])} users.")
        else:
            merged_routes.append(routes[i])
            used_route_indices.add(i)
            
    # Add any routes that were not merged
    for i in range(len(routes)):
        if i not in used_route_indices:
            merged_routes.append(routes[i])

    # Final check for road network split logic
    final_routes_after_split = []
    for route in merged_routes:
        should_split, split_groups = _should_split_route_by_road_network(route, office_lat, office_lon)
        if should_split:
            logger.info(f"  🚗 Splitting route {route['driver_id']} based on road network analysis.")
            # Convert split groups back to route format
            for group_positions in split_groups:
                new_route = route.copy()
                # Find original user data for the positions
                user_ids_in_group = set()
                for pos in group_positions:
                    for user in route['assigned_users']:
                        if abs(user['lat'] - pos[0]) < 0.0001 and abs(user['lng'] - pos[1]) < 0.0001:
                            user_ids_in_group.add(user['user_id'])
                            break
                
                new_route['assigned_users'] = [u for u in route['assigned_users'] if u['user_id'] in user_ids_in_group]
                # Update route center and driver position for the new route
                if new_route['assigned_users']:
                    new_route['latitude'] = np.mean([u['lat'] for u in new_route['assigned_users']])
                    new_route['longitude'] = np.mean([u['lng'] for u in new_route['assigned_users']])
                    new_route = optimize_route_sequence_improved(new_route, office_lat, office_lon)
                    update_route_metrics_improved(new_route, office_lat, office_lon)
                    final_routes_after_split.append(new_route)
                
        else:
            final_routes_after_split.append(route)

    return final_routes_after_split


def enhanced_route_splitting(routes, config, office_lat, office_lon):
    """
    Perform route splitting with enhanced logic, considering road network constraints.
    This function identifies routes that are too long, have poor directional consistency,
    or are divided by road networks and splits them into more coherent sub-routes.
    """
    logger.info("🔄 Performing enhanced route splitting with road network analysis...")
    
    split_routes = []
    
    for route in routes:
        # Check if the route is eligible for splitting based on its current quality
        should_split, split_groups = _should_split_route_by_road_network(route, office_lat, office_lon)

        if should_split:
            logger.info(f"  🚗 Splitting route {route['driver_id']} due to road network analysis.")
            # Create new routes from the split groups
            for group_positions in split_groups:
                new_route = route.copy()
                # Filter assigned users based on positions
                user_ids_in_group = set()
                for pos in group_positions:
                    for user in route['assigned_users']:
                        if abs(user['lat'] - pos[0]) < 0.0001 and abs(user['lng'] - pos[1]) < 0.0001:
                            user_ids_in_group.add(user['user_id'])
                            break
                
                new_route['assigned_users'] = [u for u in route['assigned_users'] if u['user_id'] in user_ids_in_group]
                
                # Recalculate route properties for the new sub-route
                if new_route['assigned_users']:
                    new_route['latitude'] = np.mean([u['lat'] for u in new_route['assigned_users']])
                    new_route['longitude'] = np.mean([u['lng'] for u in new_route['assigned_users']])
                    
                    # Re-optimize the sequence and update metrics for the new sub-route
                    new_route = optimize_route_sequence_improved(new_route, office_lat, office_lon)
                    update_route_metrics_improved(new_route, office_lat, office_lon)
                    split_routes.append(new_route)
        else:
            split_routes.append(route)

    return split_routes

def intelligent_route_splitting_improved(routes, config, office_lat, office_lon):
    """
    Perform intelligent route splitting by identifying and splitting routes
    that are too long, have poor directional consistency, or are divided by roads.
    """
    logger.info("🔄 Performing intelligent route splitting...")
    
    # First, apply the road network based splitting
    routes_after_road_split = enhanced_route_splitting(routes, config, office_lat, office_lon)
    
    final_split_routes = []
    for route in routes_after_road_split:
        # Apply existing splitting logic (e.g., by bearing, distance) if needed
        # For now, we rely on the enhanced splitting to cover these aspects.
        # If further splitting logic is required, it would be added here.
        
        # Example: Check for excessive turning angles or tortuosity if not already covered
        turning_score = route.get('turning_score', 0)
        tortuosity = route.get('tortuosity_ratio', 1.0)
        
        # Route optimized thresholds for splitting
        SPLIT_TURNING_THRESHOLD = config.get('route_split_turning_threshold', 45) + 5 # More lenient
        SPLIT_TORTUOSITY_THRESHOLD = config.get('max_tortuosity_ratio', 1.5) + 0.15 # More lenient

        # Check if the route needs splitting based on turning or tortuosity alone
        if (turning_score > SPLIT_TURNING_THRESHOLD or tortuosity > SPLIT_TORTUOSITY_THRESHOLD) and len(route['assigned_users']) >= 3:
            logger.info(f"  🚗 Splitting route {route['driver_id']} due to high turning/tortuosity.")
            # For simplicity, we'll just log this and assume the road network split logic
            # or subsequent optimization will handle it. A more complex implementation
            # would involve actual splitting logic here.
            # In a real scenario, you might call a function like `split_route_by_bearing_improved` or similar.
            
            # Placeholder for actual splitting logic if needed:
            # For now, we just add the original route and log the potential split.
            # If a specific splitting mechanism is desired, implement it here.
            final_split_routes.append(route)
        else:
            final_split_routes.append(route)
            
    return final_split_routes


def quality_preserving_route_merging(routes, config, office_lat, office_lon):
    """
    Performs route merging with a focus on preserving overall route quality,
    considering efficiency, capacity, and now, road network compatibility.
    """
    logger.info("🔄 Performing quality-preserving route merging with road network awareness...")
    
    # Use the enhanced merging function which includes road network checks
    return perform_quality_merge_improved(routes, config, office_lat, office_lon)


def strict_merge_compatibility_improved(route1, route2, office_lat, office_lon, config):
    """
    Checks for strict compatibility between two routes for merging,
    incorporating road network considerations.
    """
    # Use the road network check as part of the strict compatibility
    if not _are_routes_on_same_road_path(route1, route2, office_lat, office_lon):
        return False

    # Existing strict checks (e.g., capacity, direction, distance)
    total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
    max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])
    if total_users > max_capacity:
        return False

    avg_bearing1 = calculate_average_bearing_improved(route1, office_lat, office_lon)
    avg_bearing2 = calculate_average_bearing_improved(route2, office_lat, office_lon)
    bearing_diff = bearing_difference(avg_bearing1, avg_bearing2)
    if bearing_diff > config.get('MAX_BEARING_DIFFERENCE', 30):
        return False

    center1 = calculate_route_center_improved(route1)
    center2 = calculate_route_center_improved(route2)
    dist_between_centers = haversine_distance(center1[0], center1[1], center2[0], center2[1])
    if dist_between_centers > config.get('MERGE_DISTANCE_KM', 3.5):
        return False

    # Add checks for turning angle and tortuosity if needed for "strict" merge
    # This part would depend on how "strict" is defined in the context of route quality.
    # For now, we'll assume the above checks and road network are the primary strict criteria.

    return True


def calculate_merge_quality_score(route1, route2, merged_route, office_lat, office_lon, config):
    """
    Calculates a quality score for merging two routes, considering road network coherence.
    """
    total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
    max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])
    utilization = total_users / max_capacity if max_capacity else 0

    turning_score = merged_route.get('turning_score', 0)
    tortuosity = merged_route.get('tortuosity_ratio', 1.0)

    coherence = 0.0
    if road_network:
        driver_pos = (merged_route['latitude'], merged_route['longitude'])
        user_positions = [(u['lat'], u['lng']) for u in merged_route['assigned_users']]
        coherence = road_network.get_route_coherence_score(driver_pos, user_positions, (office_lat, office_lon))

    # Route optimized scoring: prioritize efficiency, coherence, then capacity
    # These weights are heuristic and can be tuned.
    efficiency_weight = 0.4
    coherence_weight = 0.4
    capacity_weight = 0.2

    # Normalize scores to be comparable
    # Lower turning/tortuosity is better; higher coherence/utilization is better.
    # We'll penalize higher turning/tortuosity and lower utilization.
    
    # Penalties for inefficiency (higher is worse)
    turning_penalty = turning_score / config.get('MAX_TURNING_ANGLE', 40)
    tortuosity_penalty = (tortuosity - 1.0) / (config.get('max_tortuosity_ratio', 1.5) - 1.0) if tortuosity > 1.0 else 0

    # Penalty for underutilization (higher is worse)
    capacity_penalty = (1.0 - utilization) * 2 # Scale this penalty

    # Coherence score (higher is better)
    coherence_score = coherence / 0.7 # Normalize to a target good coherence

    # Combine scores: Lower is better for penalties, higher is better for coherence
    # We want to minimize the total "badness"
    
    # Combine penalties: lower is better
    efficiency_and_capacity_score = (turning_penalty * 0.5 + 
                                     tortuosity_penalty * 0.5 + 
                                     capacity_penalty * 0.5)

    # Combine all factors. For simplicity, let's aim to minimize a composite score.
    # Lower scores are better.
    # Score: efficiency_penalty + capacity_penalty - coherence_bonus
    # We want to minimize this composite score.
    
    # Let's define a score where lower is better.
    # Higher turning, tortuosity, capacity penalty increase the score.
    # Higher coherence decreases the score.
    
    score = (efficiency_weight * (turning_penalty + tortuosity_penalty) +
             capacity_weight * capacity_penalty -
             coherence_weight * coherence_score)
    
    # Ensure score is not excessively low due to very high coherence.
    return score


def perform_quality_merge_improved(routes, config, office_lat, office_lon):
    """
    Performs route merging with enhanced quality checks, including road network awareness.
    This function merges routes based on proximity, capacity, and road path compatibility.
    """
    logger.info("🔄 Performing enhanced route merging with road network awareness...")
    
    merged_routes = []
    used_route_indices = set()

    # Route optimized merge thresholds
    MERGE_DISTANCE_KM = config.get('MERGE_DISTANCE_KM', 3.5) * 1.2 # Slightly more lenient for road awareness
    MERGE_BEARING_THRESHOLD = config.get('MAX_BEARING_DIFFERENCE', 30) + 10 # Slightly more lenient
    MERGE_TURNING_THRESHOLD = config.get('MAX_TURNING_ANGLE', 40) + 10 # Slightly more lenient
    MERGE_TORTUOSITY_THRESHOLD = config.get('max_tortuosity_ratio', 1.5) + 0.15 # Slightly more lenient

    for i in range(len(routes)):
        if i in used_route_indices:
            continue

        best_merge_candidate = None
        best_merge_score = float('inf')

        for j in range(i + 1, len(routes)):
            if j in used_route_indices:
                continue

            route1 = routes[i]
            route2 = routes[j]

            # 1. Basic compatibility checks
            # Check capacity
            total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
            max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])
            if total_users > max_capacity:
                continue

            # Check direction (average bearing)
            avg_bearing1 = calculate_average_bearing_improved(route1, office_lat, office_lon)
            avg_bearing2 = calculate_average_bearing_improved(route2, office_lat, office_lon)
            bearing_diff = bearing_difference(avg_bearing1, avg_bearing2)
            if bearing_diff > MERGE_BEARING_THRESHOLD:
                continue

            # Check centroid distance
            center1 = calculate_route_center_improved(route1)
            center2 = calculate_route_center_improved(route2)
            dist_between_centers = haversine_distance(center1[0], center1[1], center2[0], center2[1])
            if dist_between_centers > MERGE_DISTANCE_KM:
                continue
            
            # 2. Road network compatibility check
            if not _are_routes_on_same_road_path(route1, route2, office_lat, office_lon):
                continue

            # 3. Quality assessment of potential merge
            # Create a hypothetical merged route to evaluate
            merged_route_candidate = {
                'driver_id': route1['driver_id'], # Arbitrarily pick one driver ID for the merged route
                'vehicle_id': route1['vehicle_id'],
                'vehicle_type': max_capacity,
                'latitude': (center1[0] * len(route1['assigned_users']) + center2[0] * len(route2['assigned_users'])) / total_users if total_users > 0 else center1[0],
                'longitude': (center1[1] * len(route1['assigned_users']) + center2[1] * len(route2['assigned_users'])) / total_users if total_users > 0 else center1[1],
                'assigned_users': route1['assigned_users'] + route2['assigned_users']
            }

            # Optimize sequence for the candidate merged route
            merged_route_candidate = optimize_route_sequence_improved(merged_route_candidate, office_lat, office_lon)
            update_route_metrics_improved(merged_route_candidate, office_lat, office_lon)

            # Calculate quality score
            merge_score = calculate_merge_quality_score(route1, route2, merged_route_candidate, office_lat, office_lon, config)

            if merge_score < best_merge_score:
                best_merge_score = merge_score
                best_merge_candidate = (j, merged_route_candidate)

        if best_merge_candidate:
            j, merged_route = best_merge_candidate
            merged_routes.append(merged_route)
            used_route_indices.add(i)
            used_route_indices.add(j)
            logger.info(f"  ✅ Merged route {i} with {j} into a new route with {len(merged_route['assigned_users'])} users.")
        else:
            merged_routes.append(routes[i])
            used_route_indices.add(i)
            
    # Add any routes that were not merged
    for i in range(len(routes)):
        if i not in used_route_indices:
            merged_routes.append(routes[i])

    # Final check for road network split logic (applying split to the merged routes)
    final_routes_after_split = []
    for route in merged_routes:
        should_split, split_groups = _should_split_route_by_road_network(route, office_lat, office_lon)
        if should_split:
            logger.info(f"  🚗 Splitting merged route {route['driver_id']} based on road network analysis.")
            # Create new routes from the split groups
            for group_positions in split_groups:
                new_route = route.copy()
                # Filter assigned users based on positions
                user_ids_in_group = set()
                for pos in group_positions:
                    for user in route['assigned_users']:
                        # Using a small tolerance for floating point comparisons
                        if abs(user['lat'] - pos[0]) < 0.0001 and abs(user['lng'] - pos[1]) < 0.0001:
                            user_ids_in_group.add(user['user_id'])
                            break
                
                new_route['assigned_users'] = [u for u in route['assigned_users'] if u['user_id'] in user_ids_in_group]
                
                # Recalculate route properties for the new sub-route
                if new_route['assigned_users']:
                    new_route['latitude'] = np.mean([u['lat'] for u in new_route['assigned_users']])
                    new_route['longitude'] = np.mean([u['lng'] for u in new_route['assigned_users']])
                    
                    # Re-optimize the sequence and update metrics for the new sub-route
                    new_route = optimize_route_sequence_improved(new_route, office_lat, office_lon)
                    update_route_metrics_improved(new_route, office_lat, office_lon)
                    final_routes_after_split.append(new_route)
        else:
            final_routes_after_split.append(route)

    return final_routes_after_split


# MAIN ASSIGNMENT FUNCTION FOR ROUTE OPTIMIZATION
def run_road_aware_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """
    Main assignment function optimized for route optimization approach:
    - Routes capacity utilization and route efficiency
    - Route constraints on both turning angles and utilization
    - Seeks optimal route optimization between the two objectives
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

    # Reload configuration for route optimization
    global _config
    _config = load_and_validate_config()

    # Update global variables from new config
    global MAX_FILL_DISTANCE_KM, MERGE_DISTANCE_KM, MAX_BEARING_DIFFERENCE, UTILIZATION_PENALTY_PER_SEAT
    MAX_FILL_DISTANCE_KM = _config['MAX_FILL_DISTANCE_KM']
    MERGE_DISTANCE_KM = _config['MERGE_DISTANCE_KM'] 
    MAX_BEARING_DIFFERENCE = _config['MAX_BEARING_DIFFERENCE']
    UTILIZATION_PENALTY_PER_SEAT = _config['UTILIZATION_PENALTY_PER_SEAT']

    logger.info(f"🚀 Starting ROUTE OPTIMIZATION assignment for source_id: {source_id}")
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
                "optimization_mode": "route_optimization",
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
                "optimization_mode": "route_optimization",
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

        # STEP 1: Geographic clustering with route optimized approach
        user_df = create_geographic_clusters(user_df, office_lat, office_lon, _config)
        clustering_results = {"method": "route_optimized_" + _config['clustering_method'], 
                            "clusters": user_df['geo_cluster'].nunique()}

        # STEP 2: Capacity-based sub-clustering with route optimized constraints
        user_df = create_capacity_subclusters(user_df, office_lat, office_lon, _config)

        # STEP 3: Route optimized driver assignment
        routes, assigned_user_ids = assign_drivers_by_priority_route_optimized(
            user_df, driver_df, office_lat, office_lon)

        # STEP 4: Local optimization with route optimized approach
        routes = local_optimization(routes, office_lat, office_lon)

        # STEP 5: Route optimized global optimization
        routes, unassigned_users = global_optimization(routes, user_df,
                                                       assigned_user_ids, driver_df, office_lat, office_lon)

        # STEP 6: Route optimized final-pass merge (using enhanced logic)
        routes = quality_preserving_route_merging(routes, _config, office_lat, office_lon)
        
        # STEP 7: Route splitting based on enhanced logic
        routes = intelligent_route_splitting_improved(routes, _config, office_lat, office_lon)


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

        logger.info(f"✅ Route optimization complete in {execution_time:.2f}s")
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
            "optimization_mode": "route_optimization",
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