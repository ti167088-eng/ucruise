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

# File context for logging
FILE_CONTEXT = "ASSIGN_BALANCE.PY (BALANCED OPTIMIZATION)"

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

            def is_user_on_route_path(self, driver_pos, current_user_positions, user_pos, office_pos, max_detour_ratio=1.5, route_type="optimization"):
                # Mock implementation: always returns True for simplicity in mock
                return True

            def get_road_distance(self, lat1, lon1, lat2, lon2):
                # Mock implementation: returns haversine distance
                return haversine_distance(lat1, lon1, lat2, lon2)

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

        def is_user_on_route_path(self, driver_pos, current_user_positions, user_pos, office_pos, max_detour_ratio=1.5, route_type="optimization"):
            # Mock implementation: always returns True for simplicity in mock
            return True

        def get_road_distance(self, lat1, lon1, lat2, lon2):
            # Mock implementation: returns haversine distance
            return haversine_distance(lat1, lon1, lat2, lon2)

    road_network = MockRoadNetwork()


# Load and validate configuration with balanced optimization settings
def load_and_validate_config():
    """Load configuration with balanced optimization settings for assign_balance"""
    logger.step_start("STEP 1: LOAD AND VALIDATE CONFIGURATION", FILE_CONTEXT)
    try:
        with open('config.json') as f:
            cfg = json.load(f)
            logger.info("Configuration loaded from config.json")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    # Use balanced optimization mode for assign_balance
    current_mode = "balanced_optimization"

    # Get balanced optimization configuration
    mode_configs = cfg.get("mode_configs", {})
    mode_config = mode_configs.get("balanced_optimization", {})

    logger.info(f"🎯 Using optimization mode: BALANCED ROUTE OPTIMIZATION")

    # Validate and set configuration with balanced route optimization defaults
    config = {}

    # Distance configurations with balanced route optimization defaults
    config['MAX_FILL_DISTANCE_KM'] = max(0.1, float(mode_config.get("max_fill_distance_km", cfg.get("max_fill_distance_km", 6.0))))
    config['MERGE_DISTANCE_KM'] = max(0.1, float(mode_config.get("merge_distance_km", cfg.get("merge_distance_km", 3.5))))
    config['DBSCAN_EPS_KM'] = max(0.1, float(cfg.get("dbscan_eps_km", 2.0)))
    config['OVERFLOW_PENALTY_KM'] = max(0.0, float(cfg.get("overflow_penalty_km", 7.0)))
    config['DISTANCE_ISSUE_THRESHOLD'] = max(0.1, float(cfg.get("distance_issue_threshold_km", 10.0)))
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(0.0, float(cfg.get("swap_improvement_threshold_km", 0.7)))

    # Utilization thresholds (balanced for route optimization)
    config['MIN_UTIL_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("min_util_threshold", 0.65))))
    config['LOW_UTILIZATION_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("low_utilization_threshold", 0.6))))

    # Integer configurations (must be positive)
    config['MIN_SAMPLES_DBSCAN'] = max(1, int(cfg.get("min_samples_dbscan", 2)))
    config['MAX_SWAP_ITERATIONS'] = max(1, int(cfg.get("max_swap_iterations", 4)))
    config['MAX_USERS_FOR_FALLBACK'] = max(1, int(cfg.get("max_users_for_fallback", 4)))
    config['FALLBACK_MIN_USERS'] = max(1, int(cfg.get("fallback_min_users", 2)))
    config['FALLBACK_MAX_USERS'] = max(1, int(cfg.get("fallback_max_users", 8)))

    # Angle configurations with balanced route optimization defaults
    config['MAX_BEARING_DIFFERENCE'] = max(0, min(180, float(mode_config.get("max_bearing_difference", cfg.get("max_bearing_difference", 30)))))
    config['MAX_TURNING_ANGLE'] = max(0, min(180, float(mode_config.get("max_allowed_turning_score", cfg.get("max_allowed_turning_score", 40)))))

    # Cost penalties with balanced route optimization weights
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

    # Balanced route optimization parameters
    config['optimization_mode'] = "balanced_route_optimization"
    config['aggressive_merging'] = mode_config.get('aggressive_merging', None)  # Moderate
    config['capacity_weight'] = mode_config.get('capacity_weight', 2.5)
    config['direction_weight'] = mode_config.get('direction_weight', 2.5)

    # Clustering and optimization parameters with balanced route optimization defaults
    config['clustering_method'] = cfg.get('clustering_method', 'adaptive')
    config['min_cluster_size'] = max(2, cfg.get('min_cluster_size', 2))
    config['use_sweep_algorithm'] = cfg.get('use_sweep_algorithm', True)
    config['angular_sectors'] = cfg.get('angular_sectors', 10)
    config['max_users_per_initial_cluster'] = cfg.get('max_users_per_initial_cluster', 8)
    config['max_users_per_cluster'] = cfg.get('max_users_per_cluster', 7)

    # Balanced route optimization parameters
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

# Load validated configuration - always balanced route optimization
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


def run_balanced_route_assignment(source_id, parameter=1, string_param=""):
    """Main function for balanced route optimization assignment"""
    logger.step_start("STEP 1: DATA LOADING AND PREPARATION", FILE_CONTEXT)
    try:
        # Load and validate data
        data = load_env_and_fetch_data(source_id, parameter, string_param)
        validate_input_data(data)

        # Extract coordinates and prepare dataframes
        office_lat, office_lon = extract_office_coordinates(data)
        user_df, driver_df = prepare_user_driver_dataframes(data)

        logger.info(f"📊 Assignment setup: {len(user_df)} users, {len(driver_df)} drivers")

        if user_df.empty:
            logger.warning("No users to assign")
            return {
                "status": "true",
                "data": [],
                "unassignedUsers": [],
                "clustering_analysis": {"method": "none", "clusters": 0}
            }

        if driver_df.empty:
            logger.warning("No drivers available")
            return {
                "status": "true", 
                "data": [],
                "unassignedUsers": _convert_users_to_unassigned_format(user_df),
                "clustering_analysis": {"method": "none", "clusters": 0}
            }

        logger.step_start("STEP 2: GEOGRAPHIC CLUSTERING", FILE_CONTEXT)
        # Step 2: Create geographic clusters
        user_df = create_geographic_clusters(user_df, office_lat, office_lon, _config)

        logger.step_start("STEP 2A: CAPACITY SUBCLUSTERING", FILE_CONTEXT)
        # Step 3: Create capacity subclusters
        user_df = create_capacity_subclusters(user_df, office_lat, office_lon, _config)

        logger.step_start("STEP 3: BALANCED ROUTE-OPTIMIZED DRIVER ASSIGNMENT", FILE_CONTEXT)
        # Step 4: Balanced route optimized driver assignment
        routes, assigned_user_ids = assign_drivers_balanced_route_optimized(user_df, driver_df, office_lat, office_lon)

        logger.step_start("STEP 4: LOCAL OPTIMIZATION", FILE_CONTEXT)
        # Step 5: Local optimization
        routes = local_optimization(routes, office_lat, office_lon)

        logger.step_start("STEP 5: GLOBAL OPTIMIZATION", FILE_CONTEXT)
        # Step 6: Global optimization
        unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids)]
        routes, unassigned_list = global_optimization(routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon)

        logger.step_start("STEP 6: ENHANCED ROUTE SPLITTING AND MERGING", FILE_CONTEXT)
        # Step 7: Enhanced route splitting and merging
        routes = enhanced_route_splitting(routes, driver_df, office_lat, office_lon)
        routes = quality_preserving_route_merging(routes, _config, office_lat, office_lon)

        # Final quality analysis
        result = {
            "status": "true",
            "data": routes,
            "unassignedUsers": unassigned_list,
            "clustering_analysis": {
                "method": "balanced_route_optimization",
                "clusters": user_df['sub_cluster'].nunique() if 'sub_cluster' in user_df.columns else 0
            }
        }

        quality_analysis = analyze_assignment_quality(result)
        logger.info(f"✅ BALANCED ROUTE OPTIMIZATION assignment completed:")
        logger.info(f"   🚗 {len(routes)} routes created")
        logger.info(f"   👥 {sum(len(r['assigned_users']) for r in routes)} users assigned")
        logger.info(f"   ❌ {len(unassigned_list)} users unassigned")

        return result

    except Exception as e:
        logger.error(f"❌ Error running assignment: {e}")
        raise


def assign_drivers_balanced_route_optimized(user_df, driver_df, office_lat, office_lon):
    """
    Balanced route optimized driver assignment: Optimal balance between capacity and route efficiency
    """
    logger = get_logger()
    logger.step_start("STEP 3: BALANCED ROUTE-OPTIMIZED DRIVER ASSIGNMENT", FILE_CONTEXT)

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Balanced route optimization sorting: capacity and priority equally weighted
    available_drivers = driver_df.sort_values(['capacity', 'priority'], 
                                              ascending=[False, True])

    # Collect unassigned users for capacity-aware assignment
    all_unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()

    # Balanced approach: Optimize both route efficiency and capacity utilization
    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids or all_unassigned_users.empty:
            continue

        vehicle_capacity = int(driver['capacity'])
        driver_pos = (driver['latitude'], driver['longitude'])

        # Calculate main route bearing
        main_route_bearing = calculate_bearing(office_lat, office_lon, driver['latitude'], driver['longitude'])

        # Find users with balanced route optimization constraints
        users_for_vehicle = []
        max_distance_limit = MAX_FILL_DISTANCE_KM * 1.4  # Balanced between strict and lenient
        max_bearing_deviation = 35  # Balanced between route efficiency (20°) and capacity (45°)

        # Collect candidate users with enhanced road-aware scoring
        candidate_users = []
        for _, user in all_unassigned_users.iterrows():
            distance = haversine_distance(driver['latitude'], driver['longitude'], 
                                        user['latitude'], user['longitude'])

            office_to_user_bearing = calculate_bearing(office_lat, office_lon, 
                                                     user['latitude'], user['longitude'])

            bearing_diff = bearing_difference(main_route_bearing, office_to_user_bearing)

            # Primary check: distance and bearing
            if distance > max_distance_limit or bearing_diff > max_bearing_deviation:
                continue

            # ENHANCED road path validation with balanced route optimization
            is_on_route_path = True
            route_efficiency_score = 0.0

            if road_network:
                try:
                    # Check if user is actually on the driver's route to office
                    driver_pos = (driver['latitude'], driver['longitude'])
                    user_pos = (user['latitude'], user['longitude'])
                    office_pos = (office_lat, office_lon)

                    # Balanced road network validation - not too strict, not too lenient
                    is_on_route_path = road_network.is_user_on_route_path(
                        driver_pos, [], user_pos, office_pos, 
                        max_detour_ratio=1.2,  # Balanced detour ratio
                        route_type="balanced_route_optimization"
                    )

                    # Balanced backtracking prevention
                    if is_on_route_path:
                        driver_to_office_dist = road_network.get_road_distance(
                            driver['latitude'], driver['longitude'], office_lat, office_lon)
                        driver_to_user_dist = road_network.get_road_distance(
                            driver['latitude'], driver['longitude'], user['latitude'], user['longitude'])
                        user_to_office_dist = road_network.get_road_distance(
                            user['latitude'], user['longitude'], office_lat, office_lon)

                        # Balanced backtracking check
                        detour_penalty = (driver_to_user_dist + user_to_office_dist) / driver_to_office_dist
                        if detour_penalty > 1.2:  # Balanced detour tolerance
                            is_on_route_path = False

                        # Calculate route efficiency score for ranking
                        route_efficiency_score = 1.0 / max(1.0, detour_penalty)

                except Exception as e:
                    logger.warning(f"Road path validation failed for user {user['user_id']}: {e}")
                    is_on_route_path = False

            if is_on_route_path:
                # Balanced scoring: optimize both route efficiency and capacity
                distance_score = distance * 0.4  # Balanced weight
                bearing_score = bearing_diff * 0.06  # Balanced penalty per degree  
                efficiency_score = (1.0 - route_efficiency_score) * 3.0  # Balanced penalty for inefficient routes

                total_score = distance_score + bearing_score + efficiency_score
                candidate_users.append((total_score, user))

        # Sort by balanced route optimization score and fill to 80% capacity (balanced target)
        candidate_users.sort(key=lambda x: x[0])
        target_capacity = min(vehicle_capacity, max(2, int(vehicle_capacity * 0.8)))

        for score, user in candidate_users:
            if len(users_for_vehicle) >= target_capacity:
                break
            users_for_vehicle.append(user)

        # If we have good utilization, create the route
        if len(users_for_vehicle) >= max(2, vehicle_capacity * 0.6):  # At least 60% utilization
            cluster_df = pd.DataFrame(users_for_vehicle)
            route = assign_best_driver_to_cluster_balanced_route_optimized(
                cluster_df, pd.DataFrame([driver]), used_driver_ids, office_lat, office_lon)

            if route:
                routes.append(route)
                assigned_user_ids.update(u['user_id'] for u in route['assigned_users'])

                # Remove assigned users from pool
                assigned_ids_set = {u['user_id'] for u in route['assigned_users']}
                all_unassigned_users = all_unassigned_users[~all_unassigned_users['user_id'].isin(assigned_ids_set)]

                utilization = len(route['assigned_users']) / vehicle_capacity * 100
                logger.info(f"  🎯 Driver {driver['driver_id']}: {len(route['assigned_users'])}/{vehicle_capacity} seats ({utilization:.1f}%) - Balanced Route Optimized")

    # Second pass: Fill remaining seats with balanced constraints
    remaining_users = all_unassigned_users[~all_unassigned_users['user_id'].isin(assigned_user_ids)]

    for route in routes:
        if remaining_users.empty:
            break

        available_seats = route['vehicle_type'] - len(route['assigned_users'])
        if available_seats <= 0:
            continue

        # Balanced route optimization seat filling
        route_bearing = calculate_average_bearing_improved(route, office_lat, office_lon)
        route_center = calculate_route_center_improved(route)

        compatible_users = []
        for _, user in remaining_users.iterrows():
            distance = haversine_distance(route_center[0], route_center[1],
                                        user['latitude'], user['longitude'])

            user_bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
            bearing_diff = bearing_difference(route_bearing, user_bearing)

            # Balanced route optimization criteria for seat filling with road validation
            if distance <= MAX_FILL_DISTANCE_KM * 1.6 and bearing_diff <= 38:
                # Additional road network validation for seat filling
                is_compatible = True
                if road_network:
                    try:
                        # Check if adding this user maintains route coherence
                        current_user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]
                        test_user_positions = current_user_positions + [(user['latitude'], user['longitude'])]

                        route_driver_pos = (route['latitude'], route['longitude'])
                        office_pos = (office_lat, office_lon)

                        # Calculate coherence with and without the new user
                        current_coherence = road_network.get_route_coherence_score(
                            route_driver_pos, current_user_positions, office_pos)
                        new_coherence = road_network.get_route_coherence_score(
                            route_driver_pos, test_user_positions, office_pos)

                        # Only accept if coherence doesn't decrease significantly
                        if new_coherence < current_coherence - 0.12:  # Balanced coherence requirement
                            is_compatible = False

                        # Also check direct path compatibility
                        if is_compatible:
                            is_compatible = road_network.is_user_on_route_path(
                                route_driver_pos, current_user_positions, 
                                (user['latitude'], user['longitude']), office_pos,
                                max_detour_ratio=1.3
                            )

                    except Exception as e:
                        logger.warning(f"Road coherence check failed: {e}")
                        is_compatible = False

                if is_compatible:
                    score = distance * 0.5 + bearing_diff * 0.07  # Balanced scoring
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
            update_route_metrics_improved(route, office_lat, office_lon)

            utilization = len(route['assigned_users']) / route['vehicle_type'] * 100
            logger.info(f"  🎯 Route {route['driver_id']}: Added {len(users_to_add)} users, now {len(route['assigned_users'])}/{route['vehicle_type']} ({utilization:.1f}%)")

    # Check for users who were not assigned and need to be processed by splitting/merging logic
    final_unassigned_users_df = all_unassigned_users[~all_unassigned_users['user_id'].isin(assigned_user_ids)]
    if not final_unassigned_users_df.empty:
        logger.warning(f"⚠️ {len(final_unassigned_users_df)} users remain unassigned after initial pass. These will be handled by global_optimization or split/merge logic.")

    logger.info(f"  ✅ Balanced route optimized assignment: {len(routes)} routes with optimal balance")

    # Calculate balanced route optimization metrics
    total_seats = sum(r['vehicle_type'] for r in routes)
    total_users = sum(len(r['assigned_users']) for r in routes)
    overall_utilization = (total_users / total_seats * 100) if total_seats > 0 else 0

    logger.info(f"  🎯 Balanced route optimization utilization: {total_users}/{total_seats} ({overall_utilization:.1f}%)")

    return routes, assigned_user_ids


def assign_best_driver_to_cluster_balanced_route_optimized(cluster_users, available_drivers, used_driver_ids, office_lat, office_lon):
    """Find and assign the best available driver with balanced route optimization"""
    cluster_size = len(cluster_users)

    best_driver = None
    best_score = float('inf')
    best_sequence = None

    # Balanced route optimization weights
    capacity_weight = _config.get('capacity_weight', 2.5)
    direction_weight = _config.get('direction_weight', 2.5)

    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids:
            continue

        # Capacity check (allow some over-assignment for balanced route optimization)
        if driver['capacity'] < cluster_size:
            continue

        # Calculate route metrics
        route_cost, sequence, mean_turning_degrees = calculate_route_cost_balanced_route_optimized(
            driver, cluster_users, office_lat, office_lon
        )

        # Balanced route optimization scoring approach
        utilization = cluster_size / driver['capacity']

        # Distance component (balanced efficiency factor)
        distance_score = route_cost * 0.5  # 50% weight on distance

        # Direction component (balanced efficiency factor)
        direction_score = mean_turning_degrees * direction_weight * 0.02  # Balanced weight on direction

        # Capacity component (balanced capacity factor) - inverted to prefer higher utilization
        capacity_score = (1.0 - utilization) * capacity_weight * 4.0  # Balanced weight on capacity

        # Priority component (small tiebreaker)
        priority_score = driver['priority'] * 0.15

        # Balanced total score: equal emphasis on efficiency and capacity
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

        # Add all users from cluster (balanced route optimization approach)
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

        # Balanced route optimization of sequence
        route = optimize_route_sequence_improved(route, office_lat, office_lon)
        update_route_metrics_improved(route, office_lat, office_lon)

        utilization = len(route['assigned_users']) / route['vehicle_type']
        logger.info(f"    🎯 Balanced route optimized assignment - Driver {best_driver['driver_id']}: {len(route['assigned_users'])}/{route['vehicle_type']} seats ({utilization:.1f}%)")

        return route

    return None


def calculate_route_cost_balanced_route_optimized(driver, cluster_users, office_lat, office_lon):
    """Calculate route cost with balanced route optimization - equal weight to distance and direction"""
    if len(cluster_users) == 0:
        return float('inf'), [], 0

    driver_pos = (driver['latitude'], driver['longitude'])
    office_pos = (office_lat, office_lon)

    # Get optimal pickup sequence with balanced route optimization focus (equal weight to distance and direction)
    sequence = calculate_optimal_sequence_balanced_route_optimized(driver_pos, cluster_users, office_pos)

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

    # Calculate mean turning angle - balanced route optimization weight
    mean_turning_degrees = sum(bearing_differences) / len(bearing_differences) if bearing_differences else 0

    return total_distance, sequence, mean_turning_degrees


def calculate_optimal_sequence_balanced_route_optimized(driver_pos, cluster_users, office_pos):
    """Calculate sequence with balanced route optimization - balanced focus between distance and direction"""
    if len(cluster_users) <= 1:
        return cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)

    users_list = cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)

    # Calculate main route bearing (driver to office)
    main_route_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    # Balanced route optimization scoring: exactly 50% distance, 50% direction
    def balanced_route_optimized_score(user):
        # Distance component (50%)
        distance = haversine_distance(driver_pos[0], driver_pos[1], 
                                    user['latitude'], user['longitude'])

        # Direction component (50%)
        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1], 
                                       user['latitude'], user['longitude'])

        bearing_diff = normalize_bearing_difference(user_bearing - main_route_bearing)
        bearing_diff_rad = math.radians(abs(bearing_diff))

        # Balanced weight to distance and bearing alignment
        distance_score = distance  # Raw distance
        direction_score = distance * (1 - math.cos(bearing_diff_rad))  # Direction penalty in distance units

        # Balanced route optimization focus
        combined_score = distance_score * 0.5 + direction_score * 0.5

        return (combined_score, user['user_id'])

    users_list.sort(key=balanced_route_optimized_score)

    # Apply balanced route optimization 2-opt optimization
    return apply_balanced_route_optimized_2opt(users_list, driver_pos, office_pos)


def apply_balanced_route_optimized_2opt(sequence, driver_pos, office_pos):
    """Apply balanced route optimization 2-opt improvements - balanced weight to distance and direction"""
    if len(sequence) <= 2:
        return sequence

    improved = True
    max_iterations = 3
    iteration = 0

    # Calculate main bearing from driver to office
    main_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    # Balanced route optimization turning angle threshold
    max_turning_threshold = 45  # Balanced between efficiency (35°) and capacity (60°)

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

                # Balanced route optimization acceptance criteria - balanced weight to both factors
                distance_improvement = (best_distance - new_distance) / best_distance  # Normalized improvement
                turning_improvement = (best_turning_score - new_turning_score)  # Absolute improvement

                # Convert turning improvement to same scale as distance (percentage)
                turning_improvement_normalized = turning_improvement / max(best_turning_score, 1.0)

                # Balanced weight to both improvements
                combined_improvement = distance_improvement * 0.5 + turning_improvement_normalized * 0.5

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


def run_balanced_assignment(source_id, parameter=1, string_param=""):
    """
    Main entry point for balanced route optimization assignment
    """
    import traceback
    from datetime import datetime
    
    logger = get_logger()
    logger.info("=" * 80, FILE_CONTEXT)
    logger.info(f"🚀 STARTING BALANCED ROUTE OPTIMIZATION ASSIGNMENT", FILE_CONTEXT)
    logger.info(f"📋 Source ID: {source_id} | Started at: {datetime.now().strftime('%H:%M:%S')}", FILE_CONTEXT)
    logger.info("=" * 80, FILE_CONTEXT)

    try:
        # Use the existing function that's already defined
        return run_balanced_route_assignment(source_id, parameter, string_param)

    except Exception as e:
        error_msg = f"❌ Error in balanced route optimization assignment: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {
            "status": "false",
            "error": error_msg,
            "data": [],
            "unassignedUsers": []
        }


# Expose the main function for use by other modules
def run_assignment_balance(source_id, parameter=1, string_param=""):
    """Main entry point for balanced route optimization assignment"""
    return run_balanced_assignment(source_id, parameter, string_param)