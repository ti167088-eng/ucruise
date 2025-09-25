
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
        # Try to create RoadNetwork instance (assuming GraphML file exists)
        road_network = road_network_module.RoadNetwork(
            'tricity_main_roads.graphml')
        logger.info("Successfully loaded RoadNetwork with GraphML data")
    except Exception as e:
        logger.warning(
            f"Could not create RoadNetwork instance: {e}. Using mock implementation."
        )

        class MockRoadNetwork:

            def get_route_coherence_score(self, driver_pos, user_positions,
                                          office_pos):
                # Mock implementation: returns a score based on simple distance heuristic
                if not user_positions:
                    return 1.0
                avg_dist_from_driver = sum(
                    haversine_distance(driver_pos[0], driver_pos[1], u[0],
                                       u[1])
                    for u in user_positions) / len(user_positions)
                avg_dist_from_office = sum(
                    haversine_distance(office_pos[0], office_pos[1], u[0],
                                       u[1])
                    for u in user_positions) / len(user_positions)

                # Simple heuristic: higher coherence if users are closer to the driver's path
                # and not too far from the office
                score = max(
                    0, 1.0 - (avg_dist_from_driver / 50.0) -
                    (avg_dist_from_office / 100.0))
                return min(1.0, score)

            def is_user_on_route_path(self,
                                      driver_pos,
                                      current_user_positions,
                                      user_pos,
                                      office_pos,
                                      max_detour_ratio=1.3,
                                      route_type="balanced"):
                # Mock implementation: always returns True for simplicity in mock
                return True

            def get_road_distance(self, lat1, lon1, lat2, lon2):
                # Mock implementation: returns haversine distance
                return haversine_distance(lat1, lon1, lat2, lon2)

            def find_nearest_road_node(self, lat, lon):
                # Mock implementation
                return None, None

            def simplify_path_nodes(self, path, max_nodes=10):
                # Mock implementation
                return path

        road_network = MockRoadNetwork()
except ImportError:
    logger.warning(
        "road_network module not found. Road network features will be limited."
    )

    class MockRoadNetwork:

        def get_route_coherence_score(self, driver_pos, user_positions,
                                      office_pos):
            # Mock implementation: returns a score based on simple distance heuristic
            if not user_positions:
                return 1.0
            avg_dist_from_driver = sum(
                haversine_distance(driver_pos[0], driver_pos[1], u[0],
                                   u[1])
                for u in user_positions) / len(user_positions)
            avg_dist_from_office = sum(
                haversine_distance(office_pos[0], office_pos[1], u[0],
                                   u[1])
                for u in user_positions) / len(user_positions)

            # Simple heuristic: higher coherence if users are closer to the driver's path
            # and not too far from the office
            score = max(
                0, 1.0 - (avg_dist_from_driver / 50.0) -
                (avg_dist_from_office / 100.0))
            return min(1.0, score)

        def is_user_on_route_path(self,
                                  driver_pos,
                                  current_user_positions,
                                  user_pos,
                                  office_pos,
                                  max_detour_ratio=1.3,
                                  route_type="balanced"):
            # Mock implementation: always returns True for simplicity in mock
            return True

        def get_road_distance(self, lat1, lon1, lat2, lon2):
            # Mock implementation: returns haversine distance
            return haversine_distance(lat1, lon1, lat2, lon2)

        def find_nearest_road_node(self, lat, lon):
            # Mock implementation
            return None, None

        def simplify_path_nodes(self, path, max_nodes=10):
            # Mock implementation
            return path

    road_network = MockRoadNetwork()


# Load and validate configuration with balanced optimization settings
def load_and_validate_config():
    """Load configuration with balanced optimization settings"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(
            f"Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    # Always use balanced mode for this assignment
    current_mode = "balanced_optimization"

    # Get balanced optimization configuration
    mode_configs = cfg.get("mode_configs", {})
    mode_config = mode_configs.get("balanced_optimization", {})

    logger.info(f"⚖️ Using optimization mode: BALANCED OPTIMIZATION")

    # Validate and set configuration with balanced overrides
    config = {}

    # Distance configurations with balanced overrides (between route_efficiency and capacity)
    config['MAX_FILL_DISTANCE_KM'] = max(
        0.1,
        float(
            mode_config.get("max_fill_distance_km",
                            cfg.get("max_fill_distance_km", 6.5))))  # Between 5.0 and 8.0
    config['MERGE_DISTANCE_KM'] = max(
        0.1,
        float(
            mode_config.get("merge_distance_km",
                            cfg.get("merge_distance_km", 4.0))))  # Between 3.0 and 5.0
    config['DBSCAN_EPS_KM'] = max(0.1, float(cfg.get("dbscan_eps_km", 2.0)))
    config['OVERFLOW_PENALTY_KM'] = max(
        0.0, float(cfg.get("overflow_penalty_km", 7.5)))
    config['DISTANCE_ISSUE_THRESHOLD'] = max(
        0.1, float(cfg.get("distance_issue_threshold_km", 10.0)))
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(
        0.0, float(cfg.get("swap_improvement_threshold_km", 0.75)))

    # Utilization thresholds (balanced)
    config['MIN_UTIL_THRESHOLD'] = max(
        0.0, min(1.0, float(cfg.get("min_util_threshold", 0.65))))  # Between 0.5 and 0.8
    config['LOW_UTILIZATION_THRESHOLD'] = max(
        0.0, min(1.0, float(cfg.get("low_utilization_threshold", 0.65))))

    # Integer configurations (must be positive)
    config['MIN_SAMPLES_DBSCAN'] = max(1, int(cfg.get("min_samples_dbscan",
                                                      2)))
    config['MAX_SWAP_ITERATIONS'] = max(1,
                                        int(cfg.get("max_swap_iterations", 4)))
    config['MAX_USERS_FOR_FALLBACK'] = max(
        1, int(cfg.get("max_users_for_fallback", 4)))
    config['FALLBACK_MIN_USERS'] = max(1, int(cfg.get("fallback_min_users",
                                                      2)))
    config['FALLBACK_MAX_USERS'] = max(1, int(cfg.get("fallback_max_users",
                                                      8)))

    # Angle configurations with balanced overrides
    config['MAX_BEARING_DIFFERENCE'] = max(
        0,
        min(
            180,
            float(
                mode_config.get("max_bearing_difference",
                                cfg.get("max_bearing_difference", 30)))))  # Between 20 and 45
    config['MAX_TURNING_ANGLE'] = max(
        0,
        min(
            180,
            float(
                mode_config.get("max_allowed_turning_score",
                                cfg.get("max_allowed_turning_score", 45)))))  # Between 35 and 60

    # Cost penalties with balanced overrides
    config['UTILIZATION_PENALTY_PER_SEAT'] = max(
        0.0,
        float(
            mode_config.get("utilization_penalty_per_seat",
                            cfg.get("utilization_penalty_per_seat", 3.5))))  # Between 2.0 and 5.0

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

    # Balanced optimization parameters
    config['optimization_mode'] = "balanced_optimization"
    config['aggressive_merging'] = mode_config.get('aggressive_merging', None)  # Moderate
    config['capacity_weight'] = mode_config.get('capacity_weight', 2.5)  # Equal weight
    config['direction_weight'] = mode_config.get('direction_weight', 2.5)  # Equal weight

    # Clustering and optimization parameters with balanced overrides
    config['clustering_method'] = cfg.get('clustering_method', 'adaptive')
    config['min_cluster_size'] = max(2, cfg.get('min_cluster_size', 2))
    config['use_sweep_algorithm'] = cfg.get('use_sweep_algorithm', True)
    config['angular_sectors'] = cfg.get('angular_sectors', 8)  # Between 6 and 10
    config['max_users_per_initial_cluster'] = cfg.get(
        'max_users_per_initial_cluster', 9)  # Between 8 and 12
    config['max_users_per_cluster'] = cfg.get('max_users_per_cluster', 8)  # Between 7 and 10

    # Balanced optimization parameters
    config['zigzag_penalty_weight'] = mode_config.get(
        'zigzag_penalty_weight', cfg.get('zigzag_penalty_weight', 1.75))  # Between 0.5 and 3.0
    config['route_split_turning_threshold'] = cfg.get(
        'route_split_turning_threshold', 55)  # Between 35 and 80
    config['max_tortuosity_ratio'] = cfg.get('max_tortuosity_ratio', 1.65)  # Between 1.4 and 2.0
    config['route_split_consistency_threshold'] = cfg.get(
        'route_split_consistency_threshold', 0.5)  # Between 0.3 and 0.7
    config['merge_tortuosity_improvement_required'] = cfg.get(
        'merge_tortuosity_improvement_required', None)

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
from assignment import (
    validate_input_data, load_env_and_fetch_data, extract_office_coordinates,
    prepare_user_driver_dataframes, haversine_distance, bearing_difference,
    calculate_bearing,
    calculate_bearings_and_features, coords_to_km, dbscan_clustering_metric,
    kmeans_clustering_metric, estimate_clusters, create_geographic_clusters,
    sweep_clustering, polar_sector_clustering, create_capacity_subclusters,
    create_bearing_aware_subclusters, calculate_bearing_spread,
    normalize_bearing_difference, calculate_sequence_distance,
    calculate_sequence_turning_score_improved,
    apply_strict_direction_aware_2opt, split_cluster_by_bearing_metric,
    apply_route_splitting, split_route_by_bearing_improved,
    create_sub_route_improved, calculate_users_center_improved,
    local_optimization, optimize_route_sequence_improved,
    calculate_route_cost_improved, calculate_route_turning_score_improved,
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
    find_best_driver_for_cluster_improved, calculate_combined_route_center,
    _get_all_drivers_as_unassigned, _convert_users_to_unassigned_format,
    analyze_assignment_quality, get_progress_tracker,
    validate_route_path_coherence, reoptimize_route_with_road_awareness)

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
    Balanced driver assignment: Perfect balance between route efficiency and capacity utilization
    Maintains strict directional consistency while maximizing seat filling
    """
    logger.info("⚖️ Step 3: Balanced driver assignment...")

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Balanced sorting: capacity and priority with equal importance
    available_drivers = driver_df.sort_values(['capacity', 'priority'],
                                              ascending=[False, True])

    # Collect unassigned users for balanced assignment
    all_unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()

    # Balanced approach: Combine efficient clustering with capacity awareness
    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids or all_unassigned_users.empty:
            continue

        vehicle_capacity = int(driver['capacity'])
        driver_pos = (driver['latitude'], driver['longitude'])

        # Calculate main route bearing for directional consistency
        main_route_bearing = calculate_bearing(office_lat, office_lon,
                                               driver['latitude'],
                                               driver['longitude'])

        # Find users with balanced distance/direction constraints
        users_for_vehicle = []
        max_distance_limit = MAX_FILL_DISTANCE_KM * 1.3  # Balanced between strict and lenient
        max_bearing_deviation = 32  # Balanced between 20° (efficiency) and 45° (capacity)

        # Collect candidate users with balanced scoring
        candidate_users = []
        for _, user in all_unassigned_users.iterrows():
            distance = haversine_distance(driver['latitude'],
                                          driver['longitude'],
                                          user['latitude'], user['longitude'])

            office_to_user_bearing = calculate_bearing(office_lat, office_lon,
                                                       user['latitude'], user['longitude'])

            bearing_diff = bearing_difference(main_route_bearing,
                                              office_to_user_bearing)

            # Primary check: balanced distance and bearing requirements
            if distance > max_distance_limit or bearing_diff > max_bearing_deviation:
                continue

            # Enhanced road path validation with balanced strictness
            is_on_route_path = True
            route_efficiency_score = 0.0

            if road_network:
                try:
                    # Check if user is on the driver's route to office
                    driver_pos = (driver['latitude'], driver['longitude'])
                    user_pos = (user['latitude'], user['longitude'])
                    office_pos = (office_lat, office_lon)

                    # Balanced road network validation
                    is_on_route_path = road_network.is_user_on_route_path(
                        driver_pos,
                        [],
                        user_pos,
                        office_pos,
                        max_detour_ratio=1.25,  # Balanced strictness
                        route_type="balanced")

                    # Balanced backtracking prevention
                    if is_on_route_path:
                        driver_to_office_dist = road_network.get_road_distance(
                            driver['latitude'], driver['longitude'],
                            office_lat, office_lon)
                        driver_to_user_dist = road_network.get_road_distance(
                            driver['latitude'], driver['longitude'],
                            user['latitude'], user['longitude'])
                        user_to_office_dist = road_network.get_road_distance(
                            user['latitude'], user['longitude'], office_lat,
                            office_lon)

                        # Balanced backtracking check
                        detour_penalty = (
                            driver_to_user_dist +
                            user_to_office_dist) / driver_to_office_dist
                        if detour_penalty > 1.25:  # Balanced tolerance (between 1.15 and 1.3)
                            is_on_route_path = False

                        # Calculate route efficiency score for ranking
                        route_efficiency_score = 1.0 / max(
                            1.0, detour_penalty)

                except Exception as e:
                    logger.warning(
                        f"Road path validation failed for user {user['user_id']}: {e}"
                    )
                    is_on_route_path = False

            if is_on_route_path:
                # Balanced scoring: equal weight to distance, bearing, and efficiency
                distance_score = distance * 0.4  # Balanced weight
                bearing_score = bearing_diff * 0.07  # Balanced penalty per degree
                efficiency_score = (
                    1.0 - route_efficiency_score
                ) * 3.0  # Balanced penalty for inefficient routes

                total_score = distance_score + bearing_score + efficiency_score
                candidate_users.append((total_score, user))

        # Sort by balanced score and fill to optimal capacity (75-85% target)
        candidate_users.sort(key=lambda x: x[0])
        target_capacity = min(vehicle_capacity,
                              max(2, int(vehicle_capacity * 0.8)))  # Balanced target

        for score, user in candidate_users:
            if len(users_for_vehicle) >= target_capacity:
                break
            users_for_vehicle.append(user)

        # If we have good balanced utilization, create the route
        if len(users_for_vehicle) >= max(2, vehicle_capacity * 0.6):  # Balanced minimum utilization
            # Ensure we don't exceed capacity
            users_for_vehicle = users_for_vehicle[:vehicle_capacity]
            cluster_df = pd.DataFrame(users_for_vehicle)
            route = assign_best_driver_to_cluster_balanced(
                cluster_df, pd.DataFrame([driver]), used_driver_ids,
                office_lat, office_lon)

            if route:
                routes.append(route)
                assigned_user_ids.update(u['user_id']
                                         for u in route['assigned_users'])

                # Log detailed route creation
                quality_metrics = {
                    'utilization':
                    len(route['assigned_users']) / vehicle_capacity,
                    'vehicle_capacity': vehicle_capacity,
                    'optimization_mode': 'balanced_optimization',
                    'directional_consistency': True
                }

                logger.log_route_creation(
                    driver['driver_id'], route['assigned_users'],
                    f"Balanced optimization with {len(route['assigned_users'])} users",
                    quality_metrics, "assign_balance.py")

                # Log each user assignment
                for user in route['assigned_users']:
                    logger.log_user_assignment(
                        user['user_id'], driver['driver_id'], {
                            'pickup_order':
                            route['assigned_users'].index(user) + 1,
                            'distance_from_driver':
                            haversine_distance(route['latitude'],
                                               route['longitude'], user['lat'],
                                               user['lng']),
                            'optimization_mode':
                            'balanced_optimization'
                        }, "assign_balance.py")

                # Remove assigned users from pool
                assigned_ids_set = {
                    u['user_id']
                    for u in route['assigned_users']
                }
                all_unassigned_users = all_unassigned_users[
                    ~all_unassigned_users['user_id'].isin(assigned_ids_set)]

                utilization = len(
                    route['assigned_users']) / vehicle_capacity * 100
                logger.info(
                    f"  ⚖️ Driver {driver['driver_id']}: {len(route['assigned_users'])}/{vehicle_capacity} seats ({utilization:.1f}%) - Balanced"
                )

    # Second pass: Fill remaining seats with balanced constraints
    remaining_users = all_unassigned_users[~all_unassigned_users['user_id'].
                                           isin(assigned_user_ids)]

    for route in routes:
        if remaining_users.empty:
            break

        available_seats = route['vehicle_type'] - len(route['assigned_users'])
        if available_seats <= 0:
            continue

        # Balanced seat filling
        route_bearing = calculate_average_bearing_improved(
            route, office_lat, office_lon)
        route_center = calculate_route_center_improved(route)

        compatible_users = []
        for _, user in remaining_users.iterrows():
            distance = haversine_distance(route_center[0], route_center[1],
                                          user['latitude'], user['longitude'])

            user_bearing = calculate_bearing(office_lat, office_lon,
                                             user['latitude'],
                                             user['longitude'])
            bearing_diff = bearing_difference(route_bearing, user_bearing)

            # Balanced criteria for seat filling with road validation
            if distance <= MAX_FILL_DISTANCE_KM * 1.6 and bearing_diff <= 35:
                # Balanced road network validation for seat filling
                is_compatible = True
                if road_network:
                    try:
                        # Check if adding this user maintains route coherence
                        current_user_positions = [
                            (u['lat'], u['lng'])
                            for u in route['assigned_users']
                        ]
                        test_user_positions = current_user_positions + [
                            (user['latitude'], user['longitude'])
                        ]

                        route_driver_pos = (route['latitude'], route['longitude'])
                        office_pos = (office_lat, office_lon)

                        # Calculate coherence with and without the new user
                        current_coherence = road_network.get_route_coherence_score(
                            route_driver_pos, current_user_positions,
                            office_pos)
                        new_coherence = road_network.get_route_coherence_score(
                            route_driver_pos, test_user_positions, office_pos)

                        # Balanced: Only accept if coherence doesn't decrease significantly
                        if new_coherence < current_coherence - 0.08:  # Balanced coherence requirement
                            is_compatible = False
                            logger.debug(
                                f"User {user['user_id']} rejected: coherence drop ({current_coherence:.2f} -> {new_coherence:.2f})"
                            )

                        # Balanced: Check direct path compatibility
                        if is_compatible:
                            is_compatible = road_network.is_user_on_route_path(
                                route_driver_pos,
                                current_user_positions,
                                (user['latitude'], user['longitude']),
                                office_pos,
                                max_detour_ratio=1.25,  # Balanced detour ratio
                                route_type="balanced")
                            if not is_compatible:
                                logger.debug(
                                    f"User {user['user_id']} rejected: not on route path"
                                )

                        # Balanced: Check if user creates excessive backtracking
                        if is_compatible and current_user_positions:
                            # Get distance from route center to office
                            route_center = calculate_route_center_improved(
                                route)
                            center_to_office = haversine_distance(
                                route_center[0], route_center[1], office_lat,
                                office_lon)
                            user_to_office = haversine_distance(
                                user['latitude'], user['longitude'],
                                office_lat, office_lon)

                            # Reject if user is significantly further from office than route center
                            if user_to_office > center_to_office + 3.0:  # Balanced tolerance
                                is_compatible = False
                                logger.debug(
                                    f"User {user['user_id']} rejected: creates backtracking ({user_to_office:.2f}km vs {center_to_office:.2f}km)"
                                )

                    except Exception as e:
                        logger.warning(f"Road coherence check failed: {e}")
                        is_compatible = False

                if is_compatible:
                    score = distance * 0.7 + bearing_diff * 0.06  # Balanced scoring
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
            remaining_users = remaining_users[~remaining_users['user_id'].
                                              isin(assigned_ids)]

            route = optimize_route_sequence_improved(route, office_lat, office_lon)
            utilization = len(
                route['assigned_users']) / route['vehicle_type'] * 100
            logger.info(
                f"  ⚖️ Route {route['driver_id']}: Added {len(users_to_add)} users, now {len(route['assigned_users'])}/{route['vehicle_type']} ({utilization:.1f}%)"
            )

    # Check for users who were not assigned and need to be processed by global optimization
    final_unassigned_users_df = all_unassigned_users[
        ~all_unassigned_users['user_id'].isin(assigned_user_ids)]
    if not final_unassigned_users_df.empty:
        logger.warning(
            f"⚠️ {len(final_unassigned_users_df)} users remain unassigned after initial pass. These will be handled by global_optimization with driver injection."
        )

    logger.info(
        f"  ✅ Balanced assignment: {len(routes)} routes with perfect balance"
    )

    # Calculate balanced metrics
    total_seats = sum(r['vehicle_type'] for r in routes)
    total_users = sum(len(r['assigned_users']) for r in routes)
    overall_utilization = (total_users / total_seats *
                           100) if total_seats > 0 else 0

    logger.info(
        f"  ⚖️ Balanced utilization: {total_users}/{total_seats} ({overall_utilization:.1f}%)"
    )

    return routes, assigned_user_ids


def assign_best_driver_to_cluster_balanced(cluster_users,
                                           available_drivers,
                                           used_driver_ids, office_lat,
                                           office_lon):
    """Find and assign the best available driver with balanced optimization"""
    cluster_size = len(cluster_users)

    best_driver = None
    best_score = float('inf')
    best_sequence = None

    # Balanced weights - equal importance to capacity and direction
    capacity_weight = 2.5  # Balanced weight
    direction_weight = 2.5  # Balanced weight

    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids:
            continue

        # Strict capacity check - never exceed capacity
        if driver['capacity'] < cluster_size:
            continue

        # Calculate route metrics
        route_cost, sequence, mean_turning_degrees = calculate_route_cost_balanced(
            driver, cluster_users, office_lat, office_lon)

        # Balanced scoring approach
        utilization = cluster_size / driver['capacity']

        # Distance component (efficiency factor)
        distance_score = route_cost * 0.6  # Balanced weight on distance

        # Direction component (efficiency factor)
        direction_score = mean_turning_degrees * direction_weight * 0.015  # Balanced weight

        # Capacity component (capacity factor) - penalty for underutilization
        capacity_score = (
            1.0 - utilization
        ) * capacity_weight * 3.5  # Balanced weight on capacity

        # Priority component (small tiebreaker)
        priority_score = driver['priority'] * 0.2

        # Balanced total score: equal emphasis on efficiency and capacity
        efficiency_component = distance_score + direction_score
        capacity_component = capacity_score

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

        # Add all users from cluster (balanced approach)
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
        logger.info(
            f"    ⚖️ Balanced assignment - Driver {best_driver['driver_id']}: {len(route['assigned_users'])}/{route['vehicle_type']} seats ({utilization*100:.1f}%)"
        )

        return route

    return None


def calculate_route_cost_balanced(driver, cluster_users, office_lat, office_lon):
    """Calculate route cost with balanced optimization - equal weight to distance and direction"""
    if len(cluster_users) == 0:
        return float('inf'), [], 0

    driver_pos = (driver['latitude'], driver['longitude'])
    office_pos = (office_lat, office_lon)

    # Get optimal pickup sequence with balanced focus (equal weight to distance and direction)
    sequence = calculate_optimal_sequence_balanced(
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

        # Calculate bearing difference between segments
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

    # Calculate mean turning angle - balanced weight (neither too strict nor too lenient)
    mean_turning_degrees = sum(bearing_differences) / len(
        bearing_differences) if bearing_differences else 0

    return total_distance, sequence, mean_turning_degrees


def calculate_optimal_sequence_balanced(driver_pos, cluster_users, office_pos):
    """Calculate sequence with balanced optimization - equal focus on distance and direction"""
    if len(cluster_users) <= 1:
        return cluster_users.to_dict('records') if hasattr(
            cluster_users, 'to_dict') else list(cluster_users)

    users_list = cluster_users.to_dict('records') if hasattr(
        cluster_users, 'to_dict') else list(cluster_users)

    # Calculate main route bearing (driver to office)
    main_route_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                           office_pos[0], office_pos[1])

    # Balanced scoring: exactly 50% distance, 50% direction
    def balanced_score(user):
        # Distance component (50%)
        distance = haversine_distance(driver_pos[0], driver_pos[1],
                                      user['latitude'], user['longitude'])

        # Direction component (50%)
        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                         user['latitude'], user['longitude'])

        bearing_diff = normalize_bearing_difference(user_bearing -
                                                    main_route_bearing)
        bearing_diff_rad = math.radians(abs(bearing_diff))

        # Equal weight to distance and bearing alignment
        distance_score = distance  # Raw distance
        direction_score = distance * (1 - math.cos(bearing_diff_rad)
                                      )  # Direction penalty in distance units

        # Balanced optimization: exactly 50/50
        combined_score = distance_score * 0.5 + direction_score * 0.5

        return (combined_score, user['user_id'])

    users_list.sort(key=balanced_score)

    # Apply balanced 2-opt optimization
    return apply_balanced_2opt(users_list, driver_pos, office_pos)


def apply_balanced_2opt(sequence, driver_pos, office_pos):
    """Apply balanced 2-opt improvements - equal weight to distance and direction"""
    if len(sequence) <= 2:
        return sequence

    improved = True
    max_iterations = 3
    iteration = 0

    # Calculate main bearing from driver to office
    main_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                     office_pos[0], office_pos[1])

    # Balanced turning angle threshold (exactly between efficiency and capacity)
    max_turning_threshold = 45  # Between 35° (efficiency) and 60° (capacity)

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        best_distance = calculate_sequence_distance(sequence, driver_pos,
                                                    office_pos)
        best_turning_score = calculate_sequence_turning_score_improved(
            sequence, driver_pos, office_pos)

        for i in range(len(sequence) - 1):
            for j in range(i + 2, len(sequence)):
                # Try 2-opt swap
                new_sequence = sequence[:i +
                                        1] + sequence[i + 1:j +
                                                      1][::-1] + sequence[j +
                                                                          1:]

                # Calculate new metrics
                new_distance = calculate_sequence_distance(
                    new_sequence, driver_pos, office_pos)
                new_turning_score = calculate_sequence_turning_score_improved(
                    new_sequence, driver_pos, office_pos)

                # Balanced acceptance criteria - equal weight to both factors
                distance_improvement = (
                    best_distance -
                    new_distance) / best_distance  # Normalized improvement
                turning_improvement = (best_turning_score - new_turning_score
                                       )  # Absolute improvement

                # Convert turning improvement to same scale as distance (percentage)
                turning_improvement_normalized = turning_improvement / max(
                    best_turning_score, 1.0)

                # Equal weight to both improvements
                combined_improvement = distance_improvement * 0.5 + turning_improvement_normalized * 0.5

                # Accept if combined improvement is positive and turning stays reasonable
                if (combined_improvement > 0.003
                        and  # Small positive improvement
                        new_turning_score <= max_turning_threshold):

                    sequence = new_sequence
                    best_distance = new_distance
                    best_turning_score = new_turning_score
                    improved = True
                    break
            if improved:
                break

    return sequence


def final_pass_merge_balanced(routes, config, office_lat, office_lon):
    """
    Balanced final-pass merge: Equal focus on capacity utilization and route efficiency
    """
    logger.info("🔄 Step 6: Balanced final-pass merge...")

    merged_routes = []
    used = set()

    # Balanced thresholds (exactly between route efficiency and capacity)
    MERGE_BEARING_THRESHOLD = 32  # Between 20° (efficiency) and 45° (capacity)
    MERGE_DISTANCE_KM = config.get("MERGE_DISTANCE_KM",
                                   4.0) * 1.25  # Between 3.0 and 5.0
    MERGE_TURNING_THRESHOLD = 47  # Between 35° (efficiency) and 60° (capacity)
    MERGE_TORTUOSITY_THRESHOLD = 1.55  # Between 1.3 and 2.0

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

            # 3. Capacity check - use the larger vehicle's capacity as the merged capacity
            total_users = len(r1['assigned_users']) + len(r2['assigned_users'])
            # Choose the driver with larger capacity for the merged route
            if r1['vehicle_type'] >= r2['vehicle_type']:
                merged_capacity = r1['vehicle_type']
                better_driver_route = r1
            else:
                merged_capacity = r2['vehicle_type']
                better_driver_route = r2

            if total_users > merged_capacity:
                continue

            # 4. Quality assessment with balanced criteria
            combined_center = calculate_combined_route_center(r1, r2)
            dist1 = haversine_distance(r1['latitude'], r1['longitude'],
                                       combined_center[0], combined_center[1])
            dist2 = haversine_distance(r2['latitude'], r2['longitude'],
                                       combined_center[0], combined_center[1])

            better_route = r1 if dist1 <= dist2 else r2

            # Create test merged route
            test_route = better_route.copy()
            test_route[
                'assigned_users'] = r1['assigned_users'] + r2['assigned_users']
            test_route['vehicle_type'] = max(r1['vehicle_type'], r2['vehicle_type'])

            # Optimize sequence for merged route
            test_route = optimize_route_sequence_improved(
                test_route, office_lat, office_lon)

            # Calculate balanced quality metrics
            turning_score = calculate_route_turning_score_improved(
                test_route['assigned_users'],
                (test_route['latitude'], test_route['longitude']),
                (office_lat, office_lon))

            tortuosity = calculate_tortuosity_ratio_improved(
                test_route['assigned_users'],
                (test_route['latitude'], test_route['longitude']),
                (office_lat, office_lon))

            utilization = total_users / merged_capacity

            # Road network coherence check for balanced optimization
            road_coherence_acceptable = True
            path_validation_acceptable = True

            if road_network:
                driver1_pos = (r1['latitude'], r1['longitude'])
                user1_positions = [(u['lat'], u['lng'])
                                   for u in r1['assigned_users']]
                driver2_pos = (r2['latitude'], r2['longitude'])
                user2_positions = [(u['lat'], u['lng'])
                                   for u in r2['assigned_users']]
                office_pos = (office_lat, office_lon)

                # Check overall route coherence
                coherence1 = road_network.get_route_coherence_score(
                    driver1_pos, user1_positions, office_pos)
                coherence2 = road_network.get_route_coherence_score(
                    driver2_pos, user2_positions, office_pos)

                # Balanced: Both routes must have reasonable coherence
                if coherence1 < 0.35 or coherence2 < 0.35:
                    logger.info(
                        f"Routes rejected: coherence too low ({coherence1:.2f}, {coherence2:.2f})"
                    )
                    road_coherence_acceptable = False

                # Balanced: Check if combined route would have acceptable coherence
                combined_user_positions = user1_positions + user2_positions
                combined_center = calculate_combined_route_center(r1, r2)

                # Use the better positioned driver
                dist1_to_combined = haversine_distance(driver1_pos[0], driver1_pos[1],
                                                       combined_center[0],
                                                       combined_center[1])
                dist2_to_combined = haversine_distance(driver2_pos[0], driver2_pos[1],
                                                       combined_center[0],
                                                       combined_center[1])

                better_driver_pos = driver1_pos if dist1_to_combined <= dist2_to_combined else driver2_pos

                combined_coherence = road_network.get_route_coherence_score(
                    better_driver_pos, combined_user_positions,
                    office_pos)

                # Balanced: Combined route must maintain reasonable coherence
                if combined_coherence < 0.3:
                    logger.info(
                        f"Routes rejected: combined coherence too low ({combined_coherence:.2f})"
                    )
                    road_coherence_acceptable = False

                # Balanced: Bearing check
                bearing1 = calculate_average_bearing_improved(r1, office_lat,
                                                              office_lon)
                bearing2 = calculate_average_bearing_improved(r2, office_lat,
                                                              office_lon)
                bearing_diff = bearing_difference(bearing1, bearing2)

                if bearing_diff > 35:  # Balanced bearing requirement
                    logger.info(
                        f"Routes rejected: bearing difference too high ({bearing_diff:.1f}°)"
                    )
                    road_coherence_acceptable = False

                # Balanced: Distance check
                road_distance = road_network.get_road_distance(c1[0], c1[1],
                                                               c2[0], c2[1])
                if road_distance > 4.5:  # Balanced distance requirement
                    logger.info(
                        f"Routes rejected: centers too far apart ({road_distance:.2f}km)"
                    )
                    road_coherence_acceptable = False

                # Validate path sequence for the combined route
                if road_coherence_acceptable and len(combined_user_positions) > 1:
                    try:
                        # Check if the route sequence makes sense from road network perspective
                        sequence_valid = True
                        for idx, user_pos in enumerate(combined_user_positions):
                            # Check if each user is on the path from driver to office
                            is_on_path = road_network.is_user_on_route_path(
                                better_driver_pos,
                                combined_user_positions[:idx],
                                user_pos,
                                office_pos,
                                max_detour_ratio=1.25)
                            if not is_on_path:
                                sequence_valid = False
                                break
                        path_validation_acceptable = sequence_valid
                    except Exception as e:
                        logger.warning(f"Path sequence validation failed: {e}")
                        path_validation_acceptable = False
                else:
                    path_validation_acceptable = False # If coherence failed, path check is also invalid

            # Balanced acceptance criteria - equal weight to both factors
            efficiency_acceptable = (turning_score <= MERGE_TURNING_THRESHOLD
                                     and tortuosity
                                     <= MERGE_TORTUOSITY_THRESHOLD)
            capacity_acceptable = utilization >= 0.65  # Balanced utilization requirement

            # Only accept if ALL criteria are met for balanced optimization
            if (efficiency_acceptable and capacity_acceptable and
                    road_coherence_acceptable and path_validation_acceptable):
                # Balanced scoring: 50% efficiency, 50% capacity
                efficiency_score = turning_score * 0.5 + (
                    tortuosity - 1.0) * 20  # Route quality
                capacity_score = (
                    1.0 - utilization
                ) * 100  # Underutilization penalty

                # Equal weight to both components
                balanced_score = efficiency_score * 0.5 + capacity_score * 0.5

                if balanced_score < best_balanced_score:
                    best_balanced_score = balanced_score
                    best_merge = (j, test_route)

        if best_merge is not None:
            j, merged_route = best_merge
            merged_routes.append(merged_route)
            used.add(i)
            used.add(j)
            utilization_pct = len(merged_route['assigned_users']
                                  ) / merged_route['vehicle_type'] * 100
            turning = merged_route.get('turning_score', 0)
            logger.info(
                f"  ⚖️ Balanced merge: {routes[i]['driver_id']} + {routes[j]['driver_id']} = {len(merged_route['assigned_users'])}/{merged_route['vehicle_type']} seats ({utilization_pct:.1f}%, {turning:.1f}° turn)"
            )
        else:
            merged_routes.append(r1)
            used.add(i)

    # Add any routes that were not merged
    for i in range(len(routes)):
        if i not in used:
            merged_routes.append(routes[i])

    return merged_routes


def global_optimization_with_driver_injection(routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon):
    """
    Enhanced global optimization with driver injection for unassigned users
    Focus on getting all users assigned while maintaining route quality
    """
    logger.info("🌍 Step 5: Enhanced Global optimization with driver injection...")

    # Get unassigned users
    unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()
    
    if unassigned_users_df.empty:
        logger.info("✅ All users already assigned, proceeding with standard optimization")
        return global_optimization(routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon)

    logger.info(f"🎯 Found {len(unassigned_users_df)} unassigned users to handle")

    # Phase 1: Try to fill existing routes first
    logger.info("  📈 Phase 1: Filling existing routes...")
    routes = quality_controlled_route_filling(routes, unassigned_users_df,
                                              assigned_user_ids, office_lat,
                                              office_lon)
    
    # Update assigned users after filling
    for route in routes:
        for user in route['assigned_users']:
            assigned_user_ids.add(user['user_id'])
    
    # Check remaining unassigned users
    remaining_unassigned = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()
    
    if remaining_unassigned.empty:
        logger.info("✅ All users assigned after route filling!")
        return routes, []

    logger.info(f"🚗 Phase 2: Injecting drivers for {len(remaining_unassigned)} remaining users")
    
    # Phase 2: Inject additional drivers for remaining users
    used_driver_ids = {route['driver_id'] for route in routes}
    available_drivers = driver_df[~driver_df['driver_id'].isin(used_driver_ids)].copy()
    
    if available_drivers.empty:
        logger.warning("❌ No more drivers available for injection")
        unassigned_list = handle_remaining_users_improved(
            remaining_unassigned, driver_df, routes, office_lat, office_lon)
        return routes, unassigned_list
    
    logger.info(f"📋 Available drivers for injection: {len(available_drivers)}")
    
    # Create geographic clusters for remaining users
    remaining_unassigned = calculate_bearings_and_features(remaining_unassigned, office_lat, office_lon)
    
    # Sort available drivers by capacity and priority
    available_drivers = available_drivers.sort_values(['capacity', 'priority'], ascending=[False, True])
    
    additional_routes = []
    newly_assigned_ids = set()
    
    # Try to assign drivers to unassigned users
    for _, driver in available_drivers.iterrows():
        if newly_assigned_ids.issuperset(set(remaining_unassigned['user_id'])):
            break  # All users assigned
            
        remaining_users = remaining_unassigned[~remaining_unassigned['user_id'].isin(newly_assigned_ids)]
        if remaining_users.empty:
            break
            
        vehicle_capacity = int(driver['capacity'])
        driver_pos = (driver['latitude'], driver['longitude'])
        
        # Calculate main route bearing for directional consistency
        main_route_bearing = calculate_bearing(office_lat, office_lon,
                                               driver['latitude'],
                                               driver['longitude'])
        
        # Find compatible users for this driver
        compatible_users = []
        max_distance_limit = MAX_FILL_DISTANCE_KM * 1.4  # Slightly more lenient for driver injection
        max_bearing_deviation = 35  # Balanced bearing constraint
        
        for _, user in remaining_users.iterrows():
            distance = haversine_distance(driver['latitude'], driver['longitude'],
                                          user['latitude'], user['longitude'])
            
            office_to_user_bearing = calculate_bearing(office_lat, office_lon,
                                                       user['latitude'], user['longitude'])
            
            bearing_diff = bearing_difference(main_route_bearing, office_to_user_bearing)
            
            # Check basic compatibility
            if distance <= max_distance_limit and bearing_diff <= max_bearing_deviation:
                # Road network validation for driver injection
                is_compatible = True
                if road_network:
                    try:
                        driver_pos_tuple = (driver['latitude'], driver['longitude'])
                        user_pos_tuple = (user['latitude'], user['longitude'])
                        office_pos_tuple = (office_lat, office_lon)
                        
                        is_compatible = road_network.is_user_on_route_path(
                            driver_pos_tuple,
                            [],
                            user_pos_tuple,
                            office_pos_tuple,
                            max_detour_ratio=1.3,  # Slightly more lenient for injection
                            route_type="balanced")
                    except Exception as e:
                        logger.warning(f"Road validation failed for driver injection: {e}")
                        is_compatible = True  # Be lenient on validation failures
                
                if is_compatible:
                    score = distance * 0.6 + bearing_diff * 0.08
                    compatible_users.append((score, user))
        
        if compatible_users:
            # Sort by score and take up to capacity
            compatible_users.sort(key=lambda x: x[0])
            users_to_assign = []
            
            for score, user in compatible_users[:vehicle_capacity]:
                users_to_assign.append(user)
            
            if len(users_to_assign) >= 1:  # Accept even single-user routes for complete coverage
                # Create new route
                cluster_df = pd.DataFrame(users_to_assign)
                new_route = assign_best_driver_to_cluster_balanced(
                    cluster_df, pd.DataFrame([driver]), used_driver_ids,
                    office_lat, office_lon)
                
                if new_route:
                    additional_routes.append(new_route)
                    for user in new_route['assigned_users']:
                        newly_assigned_ids.add(user['user_id'])
                        assigned_user_ids.add(user['user_id'])
                    
                    utilization = len(new_route['assigned_users']) / vehicle_capacity * 100
                    logger.info(
                        f"  🚗 Injected driver {driver['driver_id']}: {len(new_route['assigned_users'])}/{vehicle_capacity} seats ({utilization:.1f}%)"
                    )
    
    # Add newly created routes to the main routes list
    routes.extend(additional_routes)
    
    # Phase 3: Standard global optimization for route quality
    logger.info("  🔧 Phase 3: Standard route optimization...")
    routes = fix_single_user_routes_improved(routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon)
    routes = quality_preserving_route_merging(routes, driver_df, office_lat, office_lon)
    routes = enhanced_route_splitting(routes, driver_df, office_lat, office_lon)
    
    # Handle any remaining unassigned users
    final_unassigned = user_df[~user_df['user_id'].isin(assigned_user_ids)]
    unassigned_list = handle_remaining_users_improved(
        final_unassigned, driver_df, routes, office_lat, office_lon)
    
    logger.info(f"  ✅ Global optimization with injection complete: {len(unassigned_list)} users remain unassigned")
    
    return routes, unassigned_list


# MAIN ASSIGNMENT FUNCTION FOR BALANCED OPTIMIZATION
def run_assignment_balance(source_id: str, parameter: int = 1, string_param: str = ""):
    """Main entry point for balanced optimization assignment"""
    return run_assignment_balance_internal(source_id, parameter, string_param)

def run_assignment_balance_internal(source_id: str, parameter: int = 1, string_param: str = ""):
    """
    Main assignment function optimized for balanced route efficiency and capacity utilization:
    - Equal weight to route efficiency and capacity utilization
    - Maintains strict directional consistency (no opposite directions)
    - Focuses on assigning all users through driver injection and global optimization
    - Perfect balance between efficient routes and seat filling
    """
    start_time = time.time()

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

        # STEP 1: Geographic clustering (balanced approach)
        user_df = create_geographic_clusters(user_df, office_lat, office_lon, _config)
        clustering_results = {"method": "balanced_" + _config['clustering_method'], 
                            "clusters": user_df['geo_cluster'].nunique()}

        # STEP 2: Balanced sub-clustering
        user_df = create_capacity_subclusters(user_df, office_lat, office_lon, _config)

        # STEP 3: Balanced driver assignment
        routes, assigned_user_ids = assign_drivers_by_priority_balanced(
            user_df, driver_df, office_lat, office_lon)

        # STEP 4: Local optimization (balanced approach)
        routes = local_optimization(routes, office_lat, office_lon)

        # STEP 5: Enhanced global optimization with driver injection
        routes, unassigned_users = global_optimization_with_driver_injection(
            routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon)

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

        # Extract additional data for rich response
        company_info = data.get("company", {})
        shift_info = data.get("shift", {})

        # Enhance route data with rich information
        enhanced_routes = []
        for route in routes:
            enhanced_route = route.copy()

            # Add enhanced driver information
            driver_id = route['driver_id']
            driver_info = None

            # Find driver in original data
            if "drivers" in data:
                all_drivers_data = data["drivers"].get("driversUnassigned", []) + data["drivers"].get("driversAssigned", [])
            else:
                all_drivers_data = data.get("driversUnassigned", []) + data.get("driversAssigned", [])

            for driver in all_drivers_data:
                if str(driver.get('id', driver.get('sub_user_id', ''))) == driver_id:
                    driver_info = driver
                    break

            if driver_info:
                # Add driver details directly to route level instead of nested driver_info
                enhanced_route.update({
                    'first_name': driver_info.get('first_name', ''),
                    'last_name': driver_info.get('last_name', ''),
                    'email': driver_info.get('email', ''),
                    'vehicle_name': driver_info.get('vehicle_name', ''),
                    'vehicle_no': driver_info.get('vehicle_no', ''),
                    'capacity': driver_info.get('capacity', ''),
                    'chasis_no': driver_info.get('chasis_no', ''),
                    'color': driver_info.get('color', ''),
                    'registration_no': driver_info.get('registration_no', ''),
                    'shift_type_id': driver_info.get('shift_type_id', '')
                })

            # Enhance user information
            enhanced_users = []
            original_users = data.get('users', [])

            for user in route['assigned_users']:
                enhanced_user = user.copy()

                # Find user in original data
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

        # Enhance unassigned users
        enhanced_unassigned_users = []
        original_users = data.get('users', [])

        for user in unassigned_users:
            enhanced_user = user.copy()

            # Find user in original data
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

        # Enhance unassigned drivers
        enhanced_unassigned_drivers = []
        if "drivers" in data:
            all_drivers_data = data["drivers"].get("driversUnassigned", []) + data["drivers"].get("driversAssigned", [])
        else:
            all_drivers_data = data.get("driversUnassigned", []) + data.get("driversAssigned", [])

        for driver in unassigned_drivers:
            enhanced_driver = driver.copy()

            # Find driver in original data
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

        return {
            "status": "true",
            "execution_time": execution_time,
            "company": company_info,
            "shift": shift_info,
            "data": enhanced_routes,
            "unassignedUsers": enhanced_unassigned_users,
            "unassignedDrivers": enhanced_unassigned_drivers,
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
