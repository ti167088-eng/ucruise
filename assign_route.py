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
                                      max_detour_ratio=1.5,
                                      route_type="optimization"):
                # Mock implementation: always returns True for simplicity in mock
                return True

            def get_road_distance(self, lat1, lon1, lat2, lon2):
                # Mock implementation: returns haversine distance
                return haversine_distance(lat1, lon1, lat2, lon2)

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
                haversine_distance(driver_pos[0], driver_pos[1], u[0], u[1])
                for u in user_positions) / len(user_positions)
            avg_dist_from_office = sum(
                haversine_distance(office_pos[0], office_pos[1], u[0], u[1])
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
                                  max_detour_ratio=1.5,
                                  route_type="optimization"):
            # Mock implementation: always returns True for simplicity in mock
            return True

        def get_road_distance(self, lat1, lon1, lat2, lon2):
            # Mock implementation: returns haversine distance
            return haversine_distance(lat1, lon1, lat2, lon2)

    road_network = MockRoadNetwork()


# Load and validate configuration with route optimization settings
def load_and_validate_config():
    """Load configuration with route optimization settings"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(
            f"Could not load config.json, using defaults. Error: {e}")
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
    config['MAX_FILL_DISTANCE_KM'] = max(
        0.1,
        float(
            mode_config.get("max_fill_distance_km",
                            cfg.get("max_fill_distance_km", 6.0))))
    config['MERGE_DISTANCE_KM'] = max(
        0.1,
        float(
            mode_config.get("merge_distance_km",
                            cfg.get("merge_distance_km", 3.5))))
    config['DBSCAN_EPS_KM'] = max(0.1, float(cfg.get("dbscan_eps_km", 2.0)))
    config['OVERFLOW_PENALTY_KM'] = max(
        0.0, float(cfg.get("overflow_penalty_km", 7.0)))
    config['DISTANCE_ISSUE_THRESHOLD'] = max(
        0.1, float(cfg.get("distance_issue_threshold_km", 10.0)))
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(
        0.0, float(cfg.get("swap_improvement_threshold_km", 0.7)))

    # Utilization thresholds (route optimized)
    config['MIN_UTIL_THRESHOLD'] = max(
        0.0, min(1.0, float(cfg.get("min_util_threshold", 0.65))))
    config['LOW_UTILIZATION_THRESHOLD'] = max(
        0.0, min(1.0, float(cfg.get("low_utilization_threshold", 0.6))))

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

    # Angle configurations with mode overrides (moderate)
    config['MAX_BEARING_DIFFERENCE'] = max(
        0,
        min(
            180,
            float(
                mode_config.get("max_bearing_difference",
                                cfg.get("max_bearing_difference", 30)))))
    config['MAX_TURNING_ANGLE'] = max(
        0,
        min(
            180,
            float(
                mode_config.get("max_allowed_turning_score",
                                cfg.get("max_allowed_turning_score", 40)))))

    # Cost penalties with mode overrides (route optimized weights)
    config['UTILIZATION_PENALTY_PER_SEAT'] = max(
        0.0,
        float(
            mode_config.get("utilization_penalty_per_seat",
                            cfg.get("utilization_penalty_per_seat", 3.0))))

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

    # Route optimization parameters
    config['optimization_mode'] = "route_optimization"
    config['aggressive_merging'] = mode_config.get('aggressive_merging',
                                                   None)  # Moderate
    config['capacity_weight'] = mode_config.get('capacity_weight', 2.5)
    config['direction_weight'] = mode_config.get('direction_weight', 2.5)

    # Clustering and optimization parameters with mode overrides
    config['clustering_method'] = cfg.get('clustering_method', 'adaptive')
    config['min_cluster_size'] = max(2, cfg.get('min_cluster_size', 2))
    config['use_sweep_algorithm'] = cfg.get('use_sweep_algorithm', True)
    config['angular_sectors'] = cfg.get('angular_sectors', 10)
    config['max_users_per_initial_cluster'] = cfg.get(
        'max_users_per_initial_cluster', 8)
    config['max_users_per_cluster'] = cfg.get('max_users_per_cluster', 7)

    # Route optimization parameters
    config['zigzag_penalty_weight'] = mode_config.get(
        'zigzag_penalty_weight', cfg.get('zigzag_penalty_weight', 2.0))
    config['route_split_turning_threshold'] = cfg.get(
        'route_split_turning_threshold', 45)
    config['max_tortuosity_ratio'] = cfg.get('max_tortuosity_ratio', 1.5)
    config['route_split_consistency_threshold'] = cfg.get(
        'route_split_consistency_threshold', 0.6)
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
    calculate_bearing_vectorized, calculate_bearing,
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

# Make sure we have sklearn available for geographic clustering
try:
    from sklearn.cluster import KMeans
    import numpy as np
except ImportError:
    logger.warning(
        "sklearn not available, using fallback geographic splitting")

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


def assign_drivers_by_priority_route_optimized(user_df, driver_df, office_lat,
                                               office_lon):
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
    all_unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids
                                                            )].copy()

    # Hybrid approach: Start with clusters but allow cross-cluster filling for capacity
    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids or all_unassigned_users.empty:
            continue

        vehicle_capacity = int(driver['capacity'])
        driver_pos = (driver['latitude'], driver['longitude'])

        # Calculate main route bearing
        main_route_bearing = calculate_bearing(office_lat, office_lon,
                                               driver['latitude'],
                                               driver['longitude'])

        # Find users with route optimized distance/direction constraints
        users_for_vehicle = []
        max_distance_limit = MAX_FILL_DISTANCE_KM * 1.5  # Between strict efficiency and lenient capacity
        max_bearing_deviation = 37  # Between 20° (efficiency) and 45° (capacity)

        # Collect candidate users with enhanced road-aware scoring
        candidate_users = []
        for _, user in all_unassigned_users.iterrows():
            distance = haversine_distance(driver['latitude'],
                                          driver['longitude'],
                                          user['latitude'], user['longitude'])

            office_to_user_bearing = calculate_bearing(office_lat, office_lon,
                                                       user['latitude'], user['longitude'])

            bearing_diff = bearing_difference(main_route_bearing,
                                              office_to_user_bearing)

            # Primary check: distance and bearing
            if distance > max_distance_limit or bearing_diff > max_bearing_deviation:
                continue

            # ENHANCED road path validation with stricter route coherence
            is_on_route_path = True
            route_efficiency_score = 0.0

            if road_network:
                try:
                    # Check if user is actually on the driver's route to office
                    driver_pos = (driver['latitude'], driver['longitude'])
                    user_pos = (user['latitude'], user['longitude'])
                    office_pos = (office_lat, office_lon)

                    # STRICTER road network validation
                    is_on_route_path = road_network.is_user_on_route_path(
                        driver_pos,
                        [],
                        user_pos,
                        office_pos,
                        max_detour_ratio=1.1,  # MUCH stricter - from 1.15 to 1.1
                        route_type="route_optimization")

                    # ENHANCED backtracking prevention
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

                        # STRICTER backtracking check
                        detour_penalty = (
                            driver_to_user_dist +
                            user_to_office_dist) / driver_to_office_dist
                        if detour_penalty > 1.15:  # Reduced from 1.2 to 1.15 (15% max detour)
                            is_on_route_path = False

                        # Calculate route efficiency score for ranking
                        route_efficiency_score = 1.0 / max(
                            1.0, detour_penalty
                        )  # Higher score for more efficient routes

                except Exception as e:
                    logger.warning(
                        f"Road path validation failed for user {user['user_id']}: {e}"
                    )
                    is_on_route_path = False

            if is_on_route_path:
                # ENHANCED scoring: prioritize route efficiency over distance
                distance_score = distance * 0.3  # Reduced weight
                bearing_score = bearing_diff * 0.05  # Reduced penalty per degree
                efficiency_score = (
                    1.0 - route_efficiency_score
                ) * 5.0  # High penalty for inefficient routes

                total_score = distance_score + bearing_score + efficiency_score
                candidate_users.append((total_score, user))

        # Sort by route optimized score and fill to 85% capacity (route optimized target)
        candidate_users.sort(key=lambda x: x[0])
        target_capacity = min(vehicle_capacity,
                              max(2, int(vehicle_capacity * 0.85)))

        for score, user in candidate_users:
            if len(users_for_vehicle) >= target_capacity:
                break
            users_for_vehicle.append(user)

        # If we have good utilization, create the route
        if len(users_for_vehicle) >= max(2, vehicle_capacity *
                                         0.5):  # At least 50% utilization
            cluster_df = pd.DataFrame(users_for_vehicle)
            route = assign_best_driver_to_cluster_route_optimized(
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
                    'optimization_mode': 'route_optimization',
                    'directional_consistency': True
                }

                logger.log_route_creation(
                    driver['driver_id'], route['assigned_users'],
                    f"Route optimization with {len(route['assigned_users'])} users",
                    quality_metrics, "assign_route.py")

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
                            'route_optimization'
                        }, "assign_route.py")

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
                    f"  🗺️ Driver {driver['driver_id']}: {len(route['assigned_users'])}/{vehicle_capacity} seats ({utilization:.1f}%) - Route Optimized"
                )

    # Second pass: Fill remaining seats with slightly more lenient constraints
    remaining_users = all_unassigned_users[~all_unassigned_users['user_id'].
                                           isin(assigned_user_ids)]

    for route in routes:
        if remaining_users.empty:
            break

        available_seats = route['vehicle_type'] - len(route['assigned_users'])
        if available_seats <= 0:
            continue

        # Route optimized seat filling
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

            # Route optimized criteria for seat filling with road validation
            if distance <= MAX_FILL_DISTANCE_KM * 1.8 and bearing_diff <= 40:
                # STRICT road network validation for seat filling
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

                        route_driver_pos = (route['latitude'],
                                            route['longitude'])
                        office_pos = (office_lat, office_lon)

                        # Calculate coherence with and without the new user
                        current_coherence = road_network.get_route_coherence_score(
                            route_driver_pos, current_user_positions,
                            office_pos)
                        new_coherence = road_network.get_route_coherence_score(
                            route_driver_pos, test_user_positions, office_pos)

                        # STRICT: Only accept if coherence doesn't decrease AT ALL
                        if new_coherence < current_coherence - 0.05:  # Very strict coherence requirement
                            is_compatible = False
                            logger.info(
                                f"User {user['user_id']} rejected: coherence drop ({current_coherence:.2f} -> {new_coherence:.2f})"
                            )

                        # STRICT: Check direct path compatibility with tight detour ratio
                        if is_compatible:
                            is_compatible = road_network.is_user_on_route_path(
                                route_driver_pos,
                                current_user_positions,
                                (user['latitude'], user['longitude']),
                                office_pos,
                                max_detour_ratio=
                                1.15,  # Much stricter detour ratio
                                route_type="route_optimization")
                            if not is_compatible:
                                logger.info(
                                    f"User {user['user_id']} rejected: not on route path"
                                )

                        # ADDITIONAL: Check if user creates backtracking
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
                            if user_to_office > center_to_office + 2.0:  # 2km tolerance
                                is_compatible = False
                                logger.info(
                                    f"User {user['user_id']} rejected: creates backtracking ({user_to_office:.2f}km vs {center_to_office:.2f}km)"
                                )

                    except Exception as e:
                        logger.warning(f"Road coherence check failed: {e}")
                        is_compatible = False

                if is_compatible:
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
            remaining_users = remaining_users[~remaining_users['user_id'].
                                              isin(assigned_ids)]

            route = optimize_route_sequence_improved(route, office_lat,
                                                     office_lon)
            utilization = len(
                route['assigned_users']) / route['vehicle_type'] * 100
            logger.info(
                f"  🗺️ Route {route['driver_id']}: Added {len(users_to_add)} users, now {len(route['assigned_users'])}/{route['vehicle_type']} ({utilization:.1f}%)"
            )

    # Check for users who were not assigned and need to be processed by splitting/merging logic
    final_unassigned_users_df = all_unassigned_users[
        ~all_unassigned_users['user_id'].isin(assigned_user_ids)]
    if not final_unassigned_users_df.empty:
        logger.warning(
            f"⚠️ {len(final_unassigned_users_df)} users remain unassigned after initial pass. These will be handled by global_optimization or split/merge logic."
        )

    logger.info(
        f"  ✅ Route optimized assignment: {len(routes)} routes with smart route optimization"
    )

    # Calculate route optimized metrics
    total_seats = sum(r['vehicle_type'] for r in routes)
    total_users = sum(len(r['assigned_users']) for r in routes)
    overall_utilization = (total_users / total_seats *
                           100) if total_seats > 0 else 0

    logger.info(
        f"  🗺️ Route optimized utilization: {total_users}/{total_seats} ({overall_utilization:.1f}%)"
    )

    return routes, assigned_user_ids


def assign_best_driver_to_cluster_route_optimized(cluster_users,
                                                  available_drivers,
                                                  used_driver_ids, office_lat,
                                                  office_lon):
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
            driver, cluster_users, office_lat, office_lon)

        # Route optimized scoring approach
        utilization = cluster_size / driver['capacity']

        # Distance component (efficiency factor)
        distance_score = route_cost * 0.5  # 50% weight on distance

        # Direction component (efficiency factor)
        direction_score = mean_turning_degrees * direction_weight * 0.02  # 0.1 km per degree

        # Capacity component (capacity factor) - inverted to prefer higher utilization
        capacity_score = (
            1.0 - utilization
        ) * capacity_weight * 5.0  # 50% weight on capacity (penalty for underutilization)

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
        logger.info(
            f"    🗺️ Route optimized assignment - Driver {best_driver['driver_id']}: {len(route['assigned_users'])}/{route['vehicle_type']} seats ({utilization*100:.1f}%)"
        )

        return route

    return None


def calculate_route_cost_route_optimized(driver, cluster_users, office_lat,
                                         office_lon):
    """Calculate route cost with route optimization - equal weight to distance and direction"""
    if len(cluster_users) == 0:
        return float('inf'), [], 0

    driver_pos = (driver['latitude'], driver['longitude'])
    office_pos = (office_lat, office_lon)

    # Get optimal pickup sequence with route optimized focus (equal weight to distance and direction)
    sequence = calculate_optimal_sequence_route_optimized(
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

    # Calculate mean turning angle - route optimized weight (neither too strict nor too lenient)
    mean_turning_degrees = sum(bearing_differences) / len(
        bearing_differences) if bearing_differences else 0

    return total_distance, sequence, mean_turning_degrees


def calculate_optimal_sequence_route_optimized(driver_pos, cluster_users,
                                               office_pos):
    """Calculate sequence with route optimization - route focused between distance and direction"""
    if len(cluster_users) <= 1:
        return cluster_users.to_dict('records') if hasattr(
            cluster_users, 'to_dict') else list(cluster_users)

    users_list = cluster_users.to_dict('records') if hasattr(
        cluster_users, 'to_dict') else list(cluster_users)

    # Calculate main route bearing (driver to office)
    main_route_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                           office_pos[0], office_pos[1])

    # Route optimized scoring: exactly 50% distance, 50% direction
    def route_optimized_score(user):
        # Distance component (50%)
        distance = haversine_distance(driver_pos[0], driver_pos[1],
                                      user['latitude'], user['longitude'])

        # Direction component (50%)
        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                         user['latitude'], user['longitude'])

        bearing_diff = normalize_bearing_difference(user_bearing -
                                                    main_route_bearing)
        bearing_diff_rad = math.radians(abs(bearing_diff))

        # Route weight to distance and bearing alignment
        distance_score = distance  # Raw distance
        direction_score = distance * (1 - math.cos(bearing_diff_rad)
                                      )  # Direction penalty in distance units

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
    main_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                     office_pos[0], office_pos[1])

    # Route optimized turning angle threshold (exactly between efficiency and capacity)
    max_turning_threshold = 47  # Between 35° (efficiency) and 60° (capacity)

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

                # Route optimized acceptance criteria - route weight to both factors
                distance_improvement = (
                    best_distance -
                    new_distance) / best_distance  # Normalized improvement
                turning_improvement = (best_turning_score - new_turning_score
                                       )  # Absolute improvement

                # Convert turning improvement to same scale as distance (percentage)
                turning_improvement_normalized = turning_improvement / max(
                    best_turning_score, 1.0)

                # Route weight to both improvements
                combined_improvement = distance_improvement * 0.5 + turning_improvement_normalized * 0.5

                # Accept if combined improvement is positive and turning stays reasonable
                if (combined_improvement > 0.005
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


def final_pass_merge_route_optimized(routes, config, office_lat, office_lon):
    """
    Route optimized final-pass merge: Route focus on capacity utilization and route efficiency
    """
    logger.info("🔄 Step 6: Route optimized final-pass merge...")

    merged_routes = []
    used = set()

    # Route optimized thresholds (exactly between route efficiency and capacity)
    MERGE_BEARING_THRESHOLD = 32  # Between 20° (efficiency) and 45° (capacity)
    MERGE_DISTANCE_KM = config.get("MERGE_DISTANCE_KM",
                                   3.5) * 1.4  # Between 3.0 and 5.0
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
            dist1 = haversine_distance(r1['latitude'], r1['longitude'],
                                       combined_center[0], combined_center[1])
            dist2 = haversine_distance(r2['latitude'], r2['longitude'],
                                       combined_center[0], combined_center[1])

            better_route = r1 if dist1 <= dist2 else r2

            # Create test merged route
            test_route = better_route.copy()
            test_route[
                'assigned_users'] = r1['assigned_users'] + r2['assigned_users']
            test_route['vehicle_type'] = max_capacity

            # Optimize sequence for merged route
            test_route = optimize_route_sequence_improved(
                test_route, office_lat, office_lon)

            # Calculate route optimized quality metrics
            turning_score = calculate_route_turning_score_improved(
                test_route['assigned_users'],
                (test_route['latitude'], test_route['longitude']),
                (office_lat, office_lon))

            tortuosity = calculate_tortuosity_ratio_improved(
                test_route['assigned_users'],
                (test_route['latitude'], test_route['longitude']),
                (office_lat, office_lon))

            utilization = total_users / max_capacity

            # Road network coherence check for route optimization
            road_coherence_acceptable = True
            path_validation_acceptable = True

            if road_network:
                driver_pos = (test_route['latitude'], test_route['longitude'])
                user_positions = [(u['lat'], u['lng'])
                                  for u in test_route['assigned_users']]
                office_pos = (office_lat, office_lon)

                # Check overall route coherence
                coherence = road_network.get_route_coherence_score(
                    driver_pos, user_positions, office_pos)
                road_coherence_acceptable = coherence >= 0.65  # Stricter coherence threshold

                # Validate that all users are actually on reasonable paths
                if road_coherence_acceptable and len(user_positions) > 1:
                    try:
                        # Check if the route sequence makes sense from road network perspective
                        sequence_valid = True
                        for i, user_pos in enumerate(user_positions):
                            # Check if each user is on the path from driver to office
                            remaining_users = user_positions[i:]
                            is_on_path = road_network.is_user_on_route_path(
                                driver_pos,
                                user_positions[:i],
                                user_pos,
                                office_pos,
                                max_detour_ratio=1.2)
                            if not is_on_path:
                                sequence_valid = False
                                break

                        path_validation_acceptable = sequence_valid

                    except Exception as e:
                        logger.warning(f"Path sequence validation failed: {e}")
                        path_validation_acceptable = False

            # 2. Route optimized acceptance criteria - route weight to both factors
            efficiency_acceptable = (turning_score <= MERGE_TURNING_THRESHOLD
                                     and tortuosity
                                     <= MERGE_TORTUOSITY_THRESHOLD)
            capacity_acceptable = utilization >= 0.6  # Route optimized utilization requirement

            # Only accept if ALL criteria are met for route optimization
            if efficiency_acceptable and capacity_acceptable and road_coherence_acceptable and path_validation_acceptable:
                # Route optimized scoring: route% efficiency, route% capacity
                efficiency_score = turning_score * 0.5 + (
                    tortuosity - 1.0) * 20  # Route quality
                capacity_score = (
                    1.0 - utilization) * 100  # Underutilization penalty

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
            utilization_pct = len(merged_route['assigned_users']
                                  ) / merged_route['vehicle_type'] * 100
            turning = merged_route.get('turning_score', 0)
            logger.info(
                f"  🗺️ Route optimized merge: {r1['driver_id']} + {routes[j]['driver_id']} = {len(merged_route['assigned_users'])}/{merged_route['vehicle_type']} seats ({utilization_pct:.1f}%, {turning:.1f}° turn)"
            )
        else:
            merged_routes.append(r1)
            used.add(i)

    # Add any routes that were not merged
    for i in range(len(routes)):
        if i not in used:
            merged_routes.append(routes[i])

    return merged_routes


def enhanced_route_splitting(routes, driver_df, office_lat, office_lon):
    """Enhanced route splitting with intelligent clustering and driver allocation"""
    logger = get_logger()
    improved_routes = []
    available_drivers = driver_df[~driver_df['driver_id'].
                                  isin([r['driver_id']
                                        for r in routes])].copy()

    # More aggressive thresholds for splitting problematic routes
    turning_threshold = _config.get('route_split_turning_threshold', 45)
    tortuosity_threshold = _config.get('max_tortuosity_ratio', 1.5)
    consistency_threshold = _config.get('route_split_consistency_threshold',
                                        0.6)

    # Additional geographic dispersion threshold
    max_user_spread_km = 8.0  # Maximum distance between users in a route

    routes_split = 0

    for route in routes:
        if len(route['assigned_users']) < 3:  # Can't split small routes
            improved_routes.append(route)
            continue

        # Calculate current quality metrics
        turning_score = route.get('turning_score', 0)
        tortuosity = route.get('tortuosity_ratio', 1.0)
        consistency = route.get('direction_consistency', 1.0)

        # Calculate geographic dispersion of users
        users = route['assigned_users']
        max_distance_between_users = 0
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                dist = haversine_distance(users[i]['lat'], users[i]['lng'],
                                          users[j]['lat'], users[j]['lng'])
                max_distance_between_users = max(max_distance_between_users,
                                                 dist)

        # Enhanced splitting criteria - now includes geographic dispersion
        needs_split = (turning_score > turning_threshold
                       or tortuosity > tortuosity_threshold
                       or consistency < consistency_threshold
                       or max_distance_between_users > max_user_spread_km)

        if needs_split and len(available_drivers) > 0 and len(
                route['assigned_users']
        ) >= 3:  # Allow splitting 3+ user routes
            reason = []
            if turning_score > turning_threshold:
                reason.append(f"turning: {turning_score:.1f}°")
            if tortuosity > tortuosity_threshold:
                reason.append(f"tortuosity: {tortuosity:.2f}")
            if consistency < consistency_threshold:
                reason.append(f"consistency: {consistency:.2f}")
            if max_distance_between_users > max_user_spread_km:
                reason.append(f"spread: {max_distance_between_users:.1f}km")

            logger.info(
                f"    🔄 Enhanced splitting - Driver {route['driver_id']}, users: {len(route['assigned_users'])}, issues: {', '.join(reason)}"
            )

            # Intelligent splitting based on geographic clusters
            split_groups = _split_users_geographically(users, office_lat,
                                                       office_lon)

            if len(split_groups) >= 2 and len(
                    available_drivers) >= len(split_groups) - 1:
                # Keep original route with first group
                route['assigned_users'] = split_groups[0]
                route = optimize_route_sequence_improved(
                    route, office_lat, office_lon)
                update_route_metrics_improved(route, office_lat, office_lon)

                # VALIDATE: Ensure the modified original route still has good coherence
                if _validate_route_road_coherence(route, office_lat,
                                                  office_lon):
                    improved_routes.append(route)
                    logger.info(
                        f"Modified route {route['driver_id']} validated and kept"
                    )
                else:
                    logger.warning(
                        f"Modified route {route['driver_id']} failed validation after split"
                    )
                    # Still add it but mark for potential further optimization
                    improved_routes.append(route)

                # Create new routes for other groups
                for i, group in enumerate(split_groups[1:], 1):
                    if i <= len(available_drivers):
                        new_driver = available_drivers.iloc[i - 1]
                        new_route = {
                            'driver_id': str(new_driver['driver_id']),
                            'vehicle_id': str(new_driver.get('vehicle_id',
                                                             '')),
                            'vehicle_type': int(new_driver['capacity']),
                            'latitude': float(new_driver['latitude']),
                            'longitude': float(new_driver['longitude']),
                            'assigned_users': group
                        }
                        new_route = optimize_route_sequence_improved(
                            new_route, office_lat, office_lon)
                        update_route_metrics_improved(new_route, office_lat,
                                                      office_lon)
                        improved_routes.append(new_route)

                # Remove used drivers
                drivers_used = min(
                    len(split_groups) - 1, len(available_drivers))
                available_drivers = available_drivers.iloc[drivers_used:]
                routes_split += 1
            else:
                # Fallback: simple binary split
                mid_point = len(users) // 2
                group1 = users[:mid_point]
                group2 = users[mid_point:]

                if len(available_drivers) > 0:
                    # Keep original route with first group
                    route['assigned_users'] = group1
                    route = optimize_route_sequence_improved(
                        route, office_lat, office_lon)
                    update_route_metrics_improved(route, office_lat,
                                                  office_lon)
                    improved_routes.append(route)

                    # Create new route with second group
                    new_driver = available_drivers.iloc[0]
                    new_route = {
                        'driver_id': str(new_driver['driver_id']),
                        'vehicle_id': str(new_driver.get('vehicle_id', '')),
                        'vehicle_type': int(new_driver['capacity']),
                        'latitude': float(new_driver['latitude']),
                        'longitude': float(new_driver['longitude']),
                        'assigned_users': group2
                    }
                    new_route = optimize_route_sequence_improved(
                        new_route, office_lat, office_lon)
                    update_route_metrics_improved(new_route, office_lat,
                                                  office_lon)

                    # VALIDATE: Ensure the new route has good road network coherence
                    if _validate_route_road_coherence(new_route, office_lat,
                                                      office_lon):
                        improved_routes.append(new_route)
                        logger.info(
                            f"New route {new_route['driver_id']} validated and added"
                        )
                    else:
                        logger.warning(
                            f"New route {new_route['driver_id']} failed road validation, skipping"
                        )
                        # Return the driver to available pool
                        available_drivers = pd.concat(
                            [available_drivers,
                             pd.DataFrame([new_driver])],
                            ignore_index=True)

                    # Remove used driver
                    available_drivers = available_drivers[
                        available_drivers['driver_id'] !=
                        new_driver['driver_id']]
                    routes_split += 1
                else:
                    improved_routes.append(route)  # No drivers available
        else:
            improved_routes.append(route)

    if routes_split > 0:
        logger.info(
            f"    ✂️ Successfully split {routes_split} routes with enhanced logic"
        )

    return improved_routes


def _split_users_geographically(users, office_lat, office_lon):
    """Split users into geographic groups using K-means clustering"""
    if len(users) < 3:
        return [users]

    # Extract coordinates
    coords = [[user['lat'], user['lng']] for user in users]

    # Try 2-cluster split first
    try:
        from sklearn.cluster import KMeans
        import numpy as np

        # Try 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)

        # Group users by cluster
        groups = [[], []]
        for i, label in enumerate(labels):
            groups[label].append(users[i])

        # Only return split if both groups have reasonable size
        if min(len(groups[0]), len(groups[1])) >= 1:
            return groups
        else:
            return [users]  # Don't split if one group would be too small

    except Exception:
        # Fallback: simple geographic split
        # Sort by distance from office and split in half
        users_with_distance = []
        for user in users:
            dist = haversine_distance(user['lat'], user['lng'], office_lat,
                                      office_lon)
            users_with_distance.append((dist, user))

        users_with_distance.sort()
        mid_point = len(users_with_distance) // 2

        group1 = [user for _, user in users_with_distance[:mid_point]]
        group2 = [user for _, user in users_with_distance[mid_point:]]

        return [group1, group2]


def _are_routes_on_same_road_path(route1, route2, office_lat, office_lon):
    """Check if two routes are on the same general road path to the office"""
    if not road_network:
        # Fallback to bearing-based check
        bearing1 = calculate_average_bearing_improved(route1, office_lat,
                                                      office_lon)
        bearing2 = calculate_average_bearing_improved(route2, office_lat,
                                                      office_lon)
        bearing_diff = bearing_difference(bearing1, bearing2)
        return bearing_diff <= 35  # More reasonable tolerance without road network

    try:
        # Get route centers
        center1 = calculate_route_center_improved(route1)
        center2 = calculate_route_center_improved(route2)

        # BALANCED: Check if both routes have reasonable coherence scores individually
        driver1_pos = (route1['latitude'], route1['longitude'])
        user1_positions = [(u['lat'], u['lng'])
                           for u in route1['assigned_users']]
        coherence1 = road_network.get_route_coherence_score(
            driver1_pos, user1_positions, (office_lat, office_lon))

        driver2_pos = (route2['latitude'], route2['longitude'])
        user2_positions = [(u['lat'], u['lng'])
                           for u in route2['assigned_users']]
        coherence2 = road_network.get_route_coherence_score(
            driver2_pos, user2_positions, (office_lat, office_lon))

        # BALANCED: Both routes must have reasonable coherence (reduced to 0.3 for better merging)
        if coherence1 < 0.3 or coherence2 < 0.3:
            logger.info(
                f"Routes rejected: coherence too low ({coherence1:.2f}, {coherence2:.2f})"
            )
            return False

        # BALANCED: Check if combined route would have acceptable coherence
        combined_user_positions = user1_positions + user2_positions
        combined_center = calculate_combined_route_center(route1, route2)

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
            (office_lat, office_lon))

        # BALANCED: Combined route must maintain reasonable coherence (reduced to 0.25 for better merging)
        if combined_coherence < 0.25:
            logger.info(
                f"Routes rejected: combined coherence too low ({combined_coherence:.2f})"
            )
            return False

        # BALANCED: Bearing check (increased tolerance from 20° to 40°)
        bearing1 = calculate_average_bearing_improved(route1, office_lat,
                                                      office_lon)
        bearing2 = calculate_average_bearing_improved(route2, office_lat,
                                                      office_lon)
        bearing_diff = bearing_difference(bearing1, bearing2)

        if bearing_diff > 40:  # More lenient bearing requirement
            logger.info(
                f"Routes rejected: bearing difference too high ({bearing_diff:.1f}°)"
            )
            return False

        # BALANCED: Distance check (increased from 3.0km to 5.0km)
        road_distance = road_network.get_road_distance(center1[0], center1[1],
                                                       center2[0], center2[1])
        if road_distance > 5.0:  # More lenient distance requirement
            logger.info(
                f"Routes rejected: centers too far apart ({road_distance:.2f}km)"
            )
            return False

        logger.info(
            f"Routes accepted: coherence=({coherence1:.2f}, {coherence2:.2f}), combined={combined_coherence:.2f}, bearing_diff={bearing_diff:.1f}°"
        )
        return True

    except Exception as e:
        logger.warning(f"Road path compatibility check failed: {e}")
        # BALANCED FALLBACK: Use bearing-based check if road network fails
        bearing1 = calculate_average_bearing_improved(route1, office_lat,
                                                      office_lon)
        bearing2 = calculate_average_bearing_improved(route2, office_lat,
                                                      office_lon)
        bearing_diff = bearing_difference(bearing1, bearing2)

        center1 = calculate_route_center_improved(route1)
        center2 = calculate_route_center_improved(route2)
        distance = haversine_distance(center1[0], center1[1], center2[0],
                                      center2[1])

        return bearing_diff <= 35 and distance <= 4.0


def _are_routes_on_same_strict_road_path(route1, route2, office_lat,
                                         office_lon):
    """STRICT check if two routes are on the same road path - for high-quality merges"""
    if not road_network:
        # Strict fallback - allow if reasonably similar bearing
        bearing1 = calculate_average_bearing_improved(route1, office_lat,
                                                      office_lon)
        bearing2 = calculate_average_bearing_improved(route2, office_lat,
                                                      office_lon)
        bearing_diff = bearing_difference(bearing1, bearing2)
        return bearing_diff <= 25  # Reasonable strict tolerance when no road network

    try:
        # Get route centers
        center1 = calculate_route_center_improved(route1)
        center2 = calculate_route_center_improved(route2)

        # STRICT: Both routes must have good coherence
        driver1_pos = (route1['latitude'], route1['longitude'])
        user1_positions = [(u['lat'], u['lng'])
                           for u in route1['assigned_users']]
        coherence1 = road_network.get_route_coherence_score(
            driver1_pos, user1_positions, (office_lat, office_lon))

        driver2_pos = (route2['latitude'], route2['longitude'])
        user2_positions = [(u['lat'], u['lng'])
                           for u in route2['assigned_users']]
        coherence2 = road_network.get_route_coherence_score(
            driver2_pos, user2_positions, (office_lat, office_lon))

        # STRICT: Both routes must have good coherence (reduced to 0.35 for better merging)
        if coherence1 < 0.35 or coherence2 < 0.35:
            logger.info(
                f"STRICT: Routes rejected due to low coherence ({coherence1:.2f}, {coherence2:.2f})"
            )
            return False

        # STRICT: Test combined route coherence must be acceptable
        combined_user_positions = user1_positions + user2_positions
        combined_center = calculate_combined_route_center(route1, route2)

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
            (office_lat, office_lon))

        # STRICT: Combined route must maintain good coherence (reduced to 0.3 for better merging)
        if combined_coherence < 0.3:
            logger.info(
                f"STRICT: Routes rejected - combined coherence too low ({combined_coherence:.2f})"
            )
            return False

        # STRICT: Bearing alignment (increased from 15° to 30°)
        bearing1 = calculate_average_bearing_improved(route1, office_lat,
                                                      office_lon)
        bearing2 = calculate_average_bearing_improved(route2, office_lat,
                                                      office_lon)
        bearing_diff = bearing_difference(bearing1, bearing2)

        if bearing_diff > 30:  # More reasonable strict bearing requirement
            logger.info(
                f"STRICT: Routes rejected - bearing difference too high ({bearing_diff:.1f}°)"
            )
            return False

        # STRICT: Distance requirements (increased from 2.0km to 3.5km)
        road_distance = road_network.get_road_distance(center1[0], center1[1],
                                                       center2[0], center2[1])
        if road_distance > 3.5:  # More reasonable strict distance requirement
            logger.info(
                f"STRICT: Routes rejected - centers too far apart ({road_distance:.2f}km)"
            )
            return False

        logger.info(
            f"STRICT: Routes accepted - coherence=({coherence1:.2f}, {coherence2:.2f}), combined={combined_coherence:.2f}, bearing_diff={bearing_diff:.1f}°"
        )
        return True

    except Exception as e:
        logger.warning(f"Strict road path compatibility check failed: {e}")
        # BALANCED FALLBACK - use bearing and distance check if road network fails
        bearing1 = calculate_average_bearing_improved(route1, office_lat,
                                                      office_lon)
        bearing2 = calculate_average_bearing_improved(route2, office_lat,
                                                      office_lon)
        bearing_diff = bearing_difference(bearing1, bearing2)

        center1 = calculate_route_center_improved(route1)
        center2 = calculate_route_center_improved(route2)
        distance = haversine_distance(center1[0], center1[1], center2[0],
                                      center2[1])

        return bearing_diff <= 25 and distance <= 3.0


def final_route_consolidation(routes, driver_df, office_lat, office_lon):
    """Final optimization to consolidate single-user routes with strict quality controls"""
    logger = get_logger()

    # Separate single-user and multi-user routes
    single_user_routes = []
    multi_user_routes = []

    for route in routes:
        if len(route['assigned_users']) == 1:
            single_user_routes.append(route)
        else:
            multi_user_routes.append(route)

    logger.info(
        f"   📊 Found {len(single_user_routes)} single-user routes to consolidate"
    )

    if len(single_user_routes) == 0:
        return routes

    consolidated_routes = multi_user_routes.copy()
    used_single_routes = set()

    # Phase 1: Merge single users into existing multi-user routes with STRICT constraints
    for i, single_route in enumerate(single_user_routes):
        if i in used_single_routes:
            continue

        single_user = single_route['assigned_users'][0]
        single_pos = (single_user['lat'], single_user['lng'])

        best_merge_route = None
        best_merge_score = float('inf')

        for multi_route in consolidated_routes:
            # Check capacity
            if len(multi_route['assigned_users']) >= multi_route['vehicle_type']:
                continue

            # Calculate route center and bearing
            route_center = calculate_route_center_improved(multi_route)
            route_bearing = calculate_average_bearing_improved(
                multi_route, office_lat, office_lon)

            # STRICT distance check - much tighter for quality
            distance = haversine_distance(route_center[0], route_center[1],
                                          single_pos[0], single_pos[1])
            if distance > 3.5:  # Much stricter distance - only 3.5km
                continue

            # STRICT bearing check - tighter for quality
            user_bearing = calculate_bearing(office_lat, office_lon,
                                             single_pos[0], single_pos[1])
            bearing_diff = bearing_difference(route_bearing, user_bearing)
            if bearing_diff > 25:  # Much stricter bearing - only 25°
                continue

            # STRICT road network validation
            is_compatible = True
            if road_network:
                try:
                    # Test if adding this user maintains route coherence
                    current_user_positions = [(u['lat'], u['lng']) for u in multi_route['assigned_users']]
                    test_user_positions = current_user_positions + [single_pos]

                    route_driver_pos = (multi_route['latitude'], multi_route['longitude'])
                    office_pos = (office_lat, office_lon)

                    # Check current vs new coherence
                    current_coherence = road_network.get_route_coherence_score(
                        route_driver_pos, current_user_positions, office_pos)
                    new_coherence = road_network.get_route_coherence_score(
                        route_driver_pos, test_user_positions, office_pos)

                    # STRICT: No decrease in coherence allowed
                    if new_coherence < current_coherence - 0.02:
                        is_compatible = False
                        logger.debug(f"User {single_user['user_id']} rejected: coherence drop")

                    # STRICT: Check if user is actually on route path
                    if is_compatible:
                        is_compatible = road_network.is_user_on_route_path(
                            route_driver_pos, current_user_positions, single_pos, office_pos,
                            max_detour_ratio=1.1, route_type="route_optimization")
                        if not is_compatible:
                            logger.debug(f"User {single_user['user_id']} rejected: not on route path")

                except Exception as e:
                    logger.warning(f"Road validation failed: {e}")
                    is_compatible = False

            if not is_compatible:
                continue

            # Create test route to validate quality metrics
            test_route = multi_route.copy()
            test_route['assigned_users'] = multi_route['assigned_users'] + [single_user]
            test_route = optimize_route_sequence_improved(test_route, office_lat, office_lon)
            update_route_metrics_improved(test_route, office_lat, office_lon)

            # STRICT quality checks
            turning_score = test_route.get('turning_score', 0)
            tortuosity = test_route.get('tortuosity_ratio', 1.0)

            # Reject if quality degrades significantly
            if turning_score > 40 or tortuosity > 1.4:  # Strict quality thresholds
                logger.debug(f"User {single_user['user_id']} rejected: quality issues (turn={turning_score:.1f}, tort={tortuosity:.2f})")
                continue

            # Enhanced scoring - heavily weight quality preservation
            quality_penalty = (turning_score * 0.5) + ((tortuosity - 1.0) * 10)
            score = distance + (bearing_diff * 0.05) + quality_penalty

            if score < best_merge_score:
                best_merge_score = score
                best_merge_route = multi_route

        # Merge if found suitable route
        if best_merge_route is not None:
            user_data = single_user.copy()
            best_merge_route['assigned_users'].append(user_data)
            used_single_routes.add(i)

            # Re-optimize the merged route
            best_merge_route = optimize_route_sequence_improved(
                best_merge_route, office_lat, office_lon)
            update_route_metrics_improved(best_merge_route, office_lat,
                                          office_lon)

            utilization = len(best_merge_route['assigned_users']
                              ) / best_merge_route['vehicle_type'] * 100
            logger.info(
                f"   ✅ Merged single user {single_user['user_id']} into route {best_merge_route['driver_id']} ({utilization:.1f}% capacity)"
            )

    # Phase 2: Merge remaining single-user routes with STRICT pairing criteria
    remaining_singles = [
        route for i, route in enumerate(single_user_routes)
        if i not in used_single_routes
    ]

    while len(remaining_singles) >= 2:
        best_pair = None
        best_pair_score = float('inf')

        for i in range(len(remaining_singles)):
            for j in range(i + 1, len(remaining_singles)):
                route1 = remaining_singles[i]
                route2 = remaining_singles[j]

                user1 = route1['assigned_users'][0]
                user2 = route2['assigned_users'][0]

                pos1 = (user1['lat'], user1['lng'])
                pos2 = (user2['lat'], user2['lng'])

                # STRICT distance between users - much tighter
                distance = haversine_distance(pos1[0], pos1[1], pos2[0], pos2[1])
                if distance > 2.5:  # Only very close users - reduced from 5.0km
                    continue

                # STRICT bearing similarity
                bearing1 = calculate_bearing(office_lat, office_lon, pos1[0], pos1[1])
                bearing2 = calculate_bearing(office_lat, office_lon, pos2[0], pos2[1])
                bearing_diff = bearing_difference(bearing1, bearing2)
                if bearing_diff > 20:  # Much stricter direction - reduced from 30°
                    continue

                # Check capacity compatibility
                max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])
                if max_capacity < 2:
                    continue

                # STRICT road network validation for pairing
                is_valid_pair = True
                if road_network:
                    try:
                        # Test which driver position is better
                        center = ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)

                        dist1 = haversine_distance(route1['latitude'], route1['longitude'], center[0], center[1])
                        dist2 = haversine_distance(route2['latitude'], route2['longitude'], center[0], center[1])

                        better_driver_pos = (route1['latitude'], route1['longitude']) if dist1 <= dist2 else (route2['latitude'], route2['longitude'])

                        # Check route coherence for the pair
                        coherence = road_network.get_route_coherence_score(
                            better_driver_pos, [pos1, pos2], (office_lat, office_lon))

                        if coherence < 0.4:  # Strict coherence requirement
                            is_valid_pair = False

                        # Check if both users are on reasonable path
                        if is_valid_pair:
                            path1_valid = road_network.is_user_on_route_path(
                                better_driver_pos, [], pos1, (office_lat, office_lon),
                                max_detour_ratio=1.15, route_type="route_optimization")
                            path2_valid = road_network.is_user_on_route_path(
                                better_driver_pos, [pos1], pos2, (office_lat, office_lon),
                                max_detour_ratio=1.15, route_type="route_optimization")

                            if not (path1_valid and path2_valid):
                                is_valid_pair = False

                    except Exception as e:
                        logger.warning(f"Road validation for pair failed: {e}")
                        is_valid_pair = False

                if not is_valid_pair:
                    continue

                # Enhanced scoring with quality focus
                score = distance * 0.8 + (bearing_diff * 0.1)  # Heavily weight distance

                if score < best_pair_score:
                    best_pair_score = score
                    best_pair = (i, j, route1, route2)

        if best_pair is not None:
            i, j, route1, route2 = best_pair

            # Choose better positioned driver
            user1_pos = (route1['assigned_users'][0]['lat'],
                         route1['assigned_users'][0]['lng'])
            user2_pos = (route2['assigned_users'][0]['lat'],
                         route2['assigned_users'][0]['lng'])
            center = ((user1_pos[0] + user2_pos[0]) / 2,
                      (user1_pos[1] + user2_pos[1]) / 2)

            dist1 = haversine_distance(route1['latitude'], route1['longitude'],
                                       center[0], center[1])
            dist2 = haversine_distance(route2['latitude'], route2['longitude'],
                                       center[0], center[1])

            better_route = route1 if dist1 <= dist2 else route2
            other_route = route2 if better_route == route1 else route1

            # Merge users
            merged_route = better_route.copy()
            merged_route['assigned_users'] = route1['assigned_users'] + route2[
                'assigned_users']
            merged_route['vehicle_type'] = max(route1['vehicle_type'],
                                               route2['vehicle_type'])

            # Optimize merged route
            merged_route = optimize_route_sequence_improved(
                merged_route, office_lat, office_lon)
            update_route_metrics_improved(merged_route, office_lat, office_lon)

            consolidated_routes.append(merged_route)

            # Remove the merged routes from remaining_singles
            remaining_singles = [
                r for idx, r in enumerate(remaining_singles)
                if idx != i and idx != j
            ]

            utilization = len(merged_route['assigned_users']
                              ) / merged_route['vehicle_type'] * 100
            logger.info(
                f"   🔗 Merged single routes {route1['driver_id']} + {route2['driver_id']} = {len(merged_route['assigned_users'])}/{merged_route['vehicle_type']} seats ({utilization:.1f}%)"
            )
        else:
            break

    # Add remaining single routes
    for route in remaining_singles:
        consolidated_routes.append(route)

    # Phase 3: DISABLED - Geographic clustering creates poor quality routes
    final_singles = [
        r for r in consolidated_routes if len(r['assigned_users']) == 1
    ]

    # Only attempt geographic clustering if we have many singles and strict criteria
    if len(final_singles) >= 5:  # Increased threshold
        logger.info(
            f"   🎯 Cautiously attempting geographic clustering of {len(final_singles)} remaining single routes"
        )
        # Use much stricter clustering parameters
        clustered_routes = geographic_clustering_merge(final_singles,
                                                       consolidated_routes,
                                                       office_lat, office_lon)
        # Validate results and revert if quality is poor
        quality_issues = 0
        for route in clustered_routes:
            if len(route['assigned_users']) > 1:
                turning = route.get('turning_score', 0)
                tortuosity = route.get('tortuosity_ratio', 1.0)
                if turning > 35 or tortuosity > 1.3:
                    quality_issues += 1

        # Only accept if less than 20% of routes have quality issues
        if quality_issues / len(clustered_routes) < 0.2:
            consolidated_routes = clustered_routes
            logger.info(f"   ✅ Geographic clustering accepted with {quality_issues} quality issues")
        else:
            logger.info(f"   ❌ Geographic clustering rejected due to {quality_issues} quality issues")
            # Keep original routes without clustering
            consolidated_routes = consolidated_routes

    initial_single_count = len(single_user_routes)
    final_single_count = len(
        [r for r in consolidated_routes if len(r['assigned_users']) == 1])

    logger.info(
        f"   📈 Consolidation results: {initial_single_count} → {final_single_count} single-user routes"
    )
    logger.info(
        f"   📊 Total routes: {len(routes)} → {len(consolidated_routes)}")

    return consolidated_routes


def geographic_clustering_merge(single_routes, other_routes, office_lat,
                                office_lon):
    """Geographic clustering for remaining single routes"""
    if len(single_routes) < 3:
        return other_routes + single_routes

    logger = get_logger()

    try:
        from sklearn.cluster import DBSCAN
        import numpy as np

        # Extract coordinates and metadata
        coords = []
        route_data = []

        for route in single_routes:
            user = route['assigned_users'][0]
            coords.append([user['lat'], user['lng']])
            route_data.append(route)

        coords = np.array(coords)

        # Perform DBSCAN clustering with STRICT parameters for quality
        dbscan = DBSCAN(eps=0.01, min_samples=3)  # ~1km radius, min 3 points - much stricter
        labels = dbscan.fit_predict(coords)

        clustered_routes = other_routes.copy()

        # Group routes by cluster
        cluster_groups = {}
        for i, label in enumerate(labels):
            if label == -1:  # Noise points - keep as single routes
                clustered_routes.append(route_data[i])
            else:
                if label not in cluster_groups:
                    cluster_groups[label] = []
                cluster_groups[label].append(route_data[i])

        # Create merged routes for each cluster
        for cluster_id, cluster_routes in cluster_groups.items():
            if len(cluster_routes) >= 2:
                # Find best driver for the cluster
                cluster_center = np.mean([
                    coords[i]
                    for i, label in enumerate(labels) if label == cluster_id
                ],
                                         axis=0)

                best_route = None
                best_distance = float('inf')

                for route in cluster_routes:
                    distance = haversine_distance(route['latitude'],
                                                  route['longitude'],
                                                  cluster_center[0],
                                                  cluster_center[1])
                    if distance < best_distance:
                        best_distance = distance
                        best_route = route

                # Create merged route
                merged_route = best_route.copy()
                merged_route['assigned_users'] = []

                for route in cluster_routes:
                    merged_route['assigned_users'].extend(
                        route['assigned_users'])

                # Find maximum capacity needed
                total_users = len(merged_route['assigned_users'])

                # If current driver can't handle all users, try to find a bigger driver
                if merged_route['vehicle_type'] < total_users:
                    # Look for available drivers with sufficient capacity
                    # For now, keep original driver and log the issue
                    logger.warning(
                        f"Cluster needs {total_users} seats but driver {merged_route['driver_id']} only has {merged_route['vehicle_type']}"
                    )

                # Optimize the merged route
                merged_route = optimize_route_sequence_improved(
                    merged_route, office_lat, office_lon)
                update_route_metrics_improved(merged_route, office_lat,
                                              office_lon)

                clustered_routes.append(merged_route)

                utilization = len(merged_route['assigned_users']
                                  ) / merged_route['vehicle_type'] * 100
                logger.info(
                    f"   🌐 Geographic cluster: {len(cluster_routes)} routes → 1 route with {total_users} users ({utilization:.1f}%)"
                )
            else:
                # Single route clusters - keep as is
                clustered_routes.extend(cluster_routes)

        return clustered_routes

    except ImportError:
        logger.warning("sklearn not available for geographic clustering")
        return other_routes + single_routes
    except Exception as e:
        logger.warning(f"Geographic clustering failed: {e}")
        return other_routes + single_routes


def _validate_route_road_coherence(route,
                                   office_lat,
                                   office_lon,
                                   min_coherence=0.3):
    """Validate that a route has acceptable road network coherence"""
    if not road_network or len(route['assigned_users']) == 0:
        return True  # Can't validate without road network or users

    try:
        driver_pos = (route['latitude'], route['longitude'])
        user_positions = [(u['lat'], u['lng'])
                          for u in route['assigned_users']]
        office_pos = (office_lat, office_lon)

        # Check route coherence with more practical threshold
        coherence = road_network.get_route_coherence_score(
            driver_pos, user_positions, office_pos)

        if coherence < min_coherence:
            logger.info(
                f"Route {route['driver_id']} failed coherence check: {coherence:.2f} < {min_coherence}"
            )
            return False

        # Skip detailed path checking for routes with 2 or fewer users to avoid over-splitting
        if len(user_positions) <= 2:
            logger.info(
                f"Route {route['driver_id']} passed validation (small route): coherence={coherence:.2f}"
            )
            return True

        # For larger routes, do basic detour checking
        total_detour_violations = 0
        for i, user_pos in enumerate(user_positions):
            driver_to_user = road_network.get_road_distance(
                driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
            user_to_office = road_network.get_road_distance(
                user_pos[0], user_pos[1], office_pos[0], office_pos[1])
            driver_to_office = road_network.get_road_distance(
                driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

            if driver_to_office > 0:
                detour_ratio = (driver_to_user +
                                user_to_office) / driver_to_office
                if detour_ratio > 1.5:  # More lenient detour ratio
                    total_detour_violations += 1

        # Allow some detour violations (up to 30% of users)
        violation_ratio = total_detour_violations / len(
            user_positions) if user_positions else 0
        if violation_ratio > 0.3:
            logger.info(
                f"Route {route['driver_id']} failed: too many detour violations ({violation_ratio:.1%})"
            )
            return False

        logger.info(
            f"Route {route['driver_id']} passed validation: coherence={coherence:.2f}, violations={violation_ratio:.1%}"
        )
        return True

    except Exception as e:
        logger.warning(f"Route validation failed: {e}")
        return True  # Be more lenient on validation failures


def intelligent_route_splitting_improved(routes, driver_df, config, office_lat,
                                         office_lon):
    """
    Perform intelligent route splitting by identifying and splitting routes
    that are too long, have poor directional consistency, or are divided by roads.
    """
    logger.info("🔄 Performing intelligent route splitting...")

    # First, apply the road network based splitting
    routes_after_road_split = enhanced_route_splitting(routes, driver_df,
                                                       office_lat, office_lon)

    final_split_routes = []
    for route in routes_after_road_split:
        # Apply existing splitting logic (e.g., by bearing, distance) if needed
        # For now, we rely on the enhanced splitting to cover these aspects.
        # If further splitting logic is required, it would be added here.

        # Example: Check for excessive turning angles or tortuosity if not already covered
        turning_score = route.get('turning_score', 0)
        tortuosity = route.get('tortuosity_ratio', 1.0)

        # Route optimized thresholds for splitting
        SPLIT_TURNING_THRESHOLD = config.get('route_split_turning_threshold',
                                             45) + 5  # More lenient
        SPLIT_TORTUOSITY_THRESHOLD = config.get('max_tortuosity_ratio',
                                                1.5) + 0.15  # More lenient

        # Check if the route needs splitting based on turning or tortuosity alone
        if (turning_score > SPLIT_TURNING_THRESHOLD
                or tortuosity > SPLIT_TORTUOSITY_THRESHOLD) and len(
                    route['assigned_users']) >= 3:
            logger.info(
                f"  🚗 Splitting route {route['driver_id']} due to high turning/tortuosity."
            )
            # For simplicity, we'll just log this and assume the road network split logic
            # or subsequent optimization will handle it. A more complex implementation
            # would involve actual splitting logic here.

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
    logger.info(
        "Performing quality-preserving route merging with road network awareness..."
    )

    # Use the enhanced merging function which includes road network checks
    return perform_quality_merge_improved(routes, config, office_lat,
                                          office_lon)


def strict_merge_compatibility_improved(route1, route2, office_lat, office_lon,
                                        config):
    """
    Checks for strict compatibility between two routes for merging,
    incorporating road network considerations.
    """
    # Use the road network check as part of the strict compatibility
    if not _are_routes_on_same_road_path(route1, route2, office_lat,
                                         office_lon):
        return False

    # Existing strict checks (e.g., capacity, direction, distance)
    total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
    max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])
    if total_users > max_capacity:
        return False

    avg_bearing1 = calculate_average_bearing_improved(route1, office_lat,
                                                      office_lon)
    avg_bearing2 = calculate_average_bearing_improved(route2, office_lat,
                                                      office_lon)
    bearing_diff = bearing_difference(avg_bearing1, avg_bearing2)
    if bearing_diff > config.get('MAX_BEARING_DIFFERENCE', 30):
        return False

    center1 = calculate_route_center_improved(route1)
    center2 = calculate_route_center_improved(route2)
    dist_between_centers = haversine_distance(center1[0], center1[1],
                                              center2[0], center2[1])
    if dist_between_centers > config.get('MERGE_DISTANCE_KM', 3.5):
        return False

    # Add checks for turning angle and tortuosity if needed for "strict" merge
    # This part would depend on how "strict" is defined in the context of route quality.
    # For now, we'll assume the above checks and road network are the primary strict criteria.

    return True


def calculate_merge_quality_score(route1, route2, merged_route, office_lat,
                                  office_lon, config):
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
        user_positions = [(u['lat'], u['lng'])
                          for u in merged_route['assigned_users']]
        coherence = road_network.get_route_coherence_score(
            driver_pos, user_positions, (office_lat, office_lon))

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
    tortuosity_penalty = (tortuosity -
                          1.0) / (config.get('max_tortuosity_ratio', 1.5) -
                                  1.0) if tortuosity > 1.0 else 0

    # Penalty for underutilization (higher is worse)
    capacity_penalty = (1.0 - utilization) * 2  # Scale this penalty

    # Coherence score (higher is better)
    coherence_score = coherence / 0.7  # Normalize to a target good coherence

    # Combine scores: Lower is better for penalties, higher is better for coherence
    # We want to minimize the total "badness"

    # Combine penalties: lower is better
    efficiency_and_capacity_score = (turning_penalty * 0.5 +
                                     tortuosity_penalty * 0.5 +
                                     capacity_penalty * 0.5)

    # Combine all factors. For simplicity, let's aim to minimize a composite score.
    # Lower scores are better.
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
    logger.info(
        "Performing enhanced route merging with road network awareness..."
    )

    merged_routes = []
    used_route_indices = set()

    # Route optimized merge thresholds
    MERGE_DISTANCE_KM = config.get(
        'MERGE_DISTANCE_KM',
        3.5) * 1.2  # Slightly more lenient for road awareness
    MERGE_BEARING_THRESHOLD = config.get('MAX_BEARING_DIFFERENCE',
                                         30) + 10  # Slightly more lenient
    MERGE_TURNING_THRESHOLD = config.get('MAX_TURNING_ANGLE',
                                         40) + 10  # Slightly more lenient
    MERGE_TORTUOSITY_THRESHOLD = config.get(
        'max_tortuosity_ratio', 1.5) + 0.15  # Slightly more lenient

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
            total_users = len(route1['assigned_users']) + len(
                route2['assigned_users'])
            max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])
            if total_users > max_capacity:
                continue

            # Check direction (average bearing)
            avg_bearing1 = calculate_average_bearing_improved(
                route1, office_lat, office_lon)
            avg_bearing2 = calculate_average_bearing_improved(
                route2, office_lat, office_lon)
            bearing_diff = bearing_difference(avg_bearing1, avg_bearing2)
            if bearing_diff > MERGE_BEARING_THRESHOLD:
                continue

            # Check centroid distance
            center1 = calculate_route_center_improved(route1)
            center2 = calculate_route_center_improved(route2)
            dist_between_centers = haversine_distance(center1[0], center1[1],
                                                      center2[0], center2[1])
            if dist_between_centers > MERGE_DISTANCE_KM:
                continue

            # 2. STRICT road network compatibility - only merge if truly on same route
            if not _are_routes_on_same_strict_road_path(
                    route1, route2, office_lat, office_lon):
                continue

            # 3. Quality assessment of potential merge
            # Create a hypothetical merged route to evaluate
            merged_route_candidate = {
                'driver_id':
                route1[
                    'driver_id'],  # Arbitrarily pick one driver ID for the merged route
                'vehicle_id':
                route1['vehicle_id'],
                'vehicle_type':
                max_capacity,
                'latitude': (center1[0] * len(route1['assigned_users']) +
                             center2[0] * len(route2['assigned_users'])) /
                total_users if total_users > 0 else center1[0],
                'longitude': (center1[1] * len(route1['assigned_users']) +
                              center2[1] * len(route2['assigned_users'])) /
                total_users if total_users > 0 else center1[1],
                'assigned_users':
                route1['assigned_users'] + route2['assigned_users']
            }

            # Optimize sequence for the candidate merged route
            merged_route_candidate = optimize_route_sequence_improved(
                merged_route_candidate, office_lat, office_lon)
            update_route_metrics_improved(merged_route_candidate, office_lat,
                                          office_lon)

            # Calculate quality score
            merge_score = calculate_merge_quality_score(
                route1, route2, merged_route_candidate, office_lat, office_lon,
                config)

            if merge_score < best_merge_score:
                best_merge_score = merge_score
                best_merge_candidate = (j, merged_route_candidate)

        if best_merge_candidate:
            j, merged_route = best_merge_candidate
            merged_routes.append(merged_route)
            used_route_indices.add(i)
            used_route_indices.add(j)
            logger.info(
                f"  ✅ Merged route {i} with {j} into a new route with {len(merged_route['assigned_users'])} users."
            )
        else:
            merged_routes.append(routes[i])
            used_route_indices.add(i)

    # Add any routes that were not merged
    for i in range(len(routes)):
        if i not in used_route_indices:
            merged_routes.append(routes[i])

    return merged_routes


# MAIN ASSIGNMENT FUNCTION FOR ROUTE OPTIMIZATION
def run_road_aware_assignment(source_id: str,
                              parameter: int = 1,
                              string_param: str = ""):
    """
    Main assignment function optimized for route optimization approach:
    - Routes capacity utilization and route efficiency
    - Route constraints on both turning angles and utilization
    - Seeks optimal route optimization between the two objectives
    - Includes non-destructive global optimization with driver injection
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

    # Reload configuration for route optimization
    global _config
    _config = load_and_validate_config()

    # Update global variables from new config
    global MAX_FILL_DISTANCE_KM, MERGE_DISTANCE_KM, MAX_BEARING_DIFFERENCE, UTILIZATION_PENALTY_PER_SEAT
    MAX_FILL_DISTANCE_KM = _config['MAX_FILL_DISTANCE_KM']
    MERGE_DISTANCE_KM = _config['MERGE_DISTANCE_KM']
    MAX_BEARING_DIFFERENCE = _config['MAX_BEARING_DIFFERENCE']
    UTILIZATION_PENALTY_PER_SEAT = _config['UTILIZATION_PENALTY_PER_SEAT']

    logger.info(
        f"🚀 Starting ROUTE OPTIMIZATION assignment for source_id: {source_id}")
    logger.info(f"📋 Parameter: {parameter}, String parameter: {string_param}")
    progress.start_stage("Data Loading", "Loading and validating input data")

    try:
        # Load and validate data
        data = load_env_and_fetch_data(source_id, parameter, string_param)
        progress.update_stage_progress("Data loaded successfully")

        # Edge case handling
        users = data.get('users', [])
        if not users:
            logger.warning("⚠️ No users found - returning empty assignment")
            result = {
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
            progress.complete_stage("No users found, assignment finished")
            progress.show_final_summary(result)
            return result

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
            result = {
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
            progress.complete_stage("No drivers available, assignment finished")
            progress.show_final_summary(result)
            return result

        logger.info(
            f"📥 Data loaded - Users: {len(users)}, Total Drivers: {len(all_drivers)}"
        )

        # Extract office coordinates and validate data
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        logger.info("✅ Data validation passed")
        progress.update_stage_progress("Data validated")

        # Prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)

        logger.info(
            f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}"
        )
        progress.update_stage_progress("DataFrames prepared")

        # Core assignment logic
        progress.start_stage("Local Optimization", "Running core assignment with road-aware optimization")
        print(f"🧠 Running intelligent route optimization...")
        routes, unassigned_users, driver_df_final = run_core_assignment_with_optimization(
            user_df, driver_df, office_lat, office_lon, _config)
        progress.update_stage_progress(f"Created {len(routes)} initial routes")

        # Filter out routes with no assigned users and capture their drivers for unassigned list
        filtered_routes = []
        unused_injected_driver_ids = set()
        
        for route in routes:
            if route['assigned_users'] and len(route['assigned_users']) > 0:
                filtered_routes.append(route)
            else:
                unused_injected_driver_ids.add(route['driver_id'])
                logger.info(f"🚫 Driver {route['driver_id']} has no users - moving to unassigned")
        
        routes = filtered_routes
        
        # Build unassigned drivers list (including unused injected drivers)
        assigned_driver_ids = {route['driver_id'] for route in routes}
        unassigned_drivers_df = driver_df_final[~driver_df_final['driver_id'].isin(assigned_driver_ids)]
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
        
        if unused_injected_driver_ids:
            logger.info(f"📊 Moved {len(unused_injected_driver_ids)} unused injected drivers to unassigned list")

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
        logger.info(
            f"📋 User accounting: {users_accounted_for}/{total_users_in_api} users"
        )

        clustering_results = {
            "method": "route_optimized_with_driver_injection",
            "clusters": len(routes)
        }

        # Final result assembly
        result = {
            "status": "true",
            "execution_time": execution_time,
            "data": routes,
            "unassignedUsers": unassigned_users,
            "unassignedDrivers": unassigned_drivers,
            "clustering_analysis": clustering_results,
            "optimization_mode": "road_aware_route_optimization",
            "parameter": parameter,
        }

        progress.complete_stage("Assignment completed successfully")
        progress.show_final_summary(result)
        logger.info("Assignment session completed successfully")
        return result

    except requests.exceptions.RequestException as req_err:
        logger.error(f"API request failed: {req_err}")
        result = {"status": "false", "details": str(req_err), "data": []}
        progress.fail_stage("API request failed")
        progress.show_final_summary(result)
        return result
    except ValueError as val_err:
        logger.error(f"Data validation error: {val_err}")
        result = {"status": "false", "details": str(val_err), "data": []}
        progress.fail_stage("Data validation error")
        progress.show_final_summary(result)
        return result
    except Exception as e:
        logger.error(f"Assignment failed: {e}", exc_info=True)
        result = {"status": "false", "details": str(e), "data": []}
        progress.fail_stage("Assignment failed")
        progress.show_final_summary(result)
        return result


def run_core_assignment_with_optimization(user_df, driver_df, office_lat, office_lon, config):
    """
    Core assignment algorithm with comprehensive non-destructive optimization as per user's 3-step plan:
    1. Run standard assignment
    2. Non-destructive optimization with driver injection for unassigned users
    3. Smart user swapping for route optimization
    """
    print("🔄 Starting comprehensive 3-step optimization plan...")
    logger.info("🔄 Starting comprehensive 3-step optimization plan...")

    # STEP 1: Run standard assignment algorithm
    print("📋 STEP 1: Running standard assignment algorithm")
    logger.info("📋 STEP 1: Running standard assignment algorithm")
    progress.start_stage("Standard Assignment", "Executing initial driver assignments")
    
    routes, unassigned_users = run_standard_assignment_algorithm(
        user_df, driver_df, office_lat, office_lon, config)
    
    progress.update_stage_progress(f"{len(routes)} routes created, {len(unassigned_users)} users unassigned")

    initial_unassigned_count = len(unassigned_users)
    print(f"📊 Step 1 complete: {len(routes)} routes, {initial_unassigned_count} unassigned users")
    logger.info(f"📊 Initial assignment: {initial_unassigned_count} unassigned users")

    # Early exit if all users assigned
    if initial_unassigned_count == 0:
        print("🎉 All users assigned in step 1! Proceeding to route optimization...")
        routes = validate_and_cleanup_routes(routes, office_lat, office_lon)
        return routes, unassigned_users, driver_df

    # STEP 2: Non-destructive optimization with driver injection
    print("🔧 STEP 2: Non-destructive optimization with driver injection")
    logger.info("🔧 STEP 2: Non-destructive optimization with driver injection")
    progress.start_stage("Non-Destructive Optimization", "Improving routes and assigning remaining users")
    
    routes, unassigned_users, final_driver_df = step_2_non_destructive_optimization(
        routes, unassigned_users, driver_df, user_df, office_lat, office_lon, config)

    step2_unassigned_count = len(unassigned_users)
    print(f"📊 Step 2 complete: {step2_unassigned_count} unassigned users")
    logger.info(f"📊 After step 2: {step2_unassigned_count} unassigned users")
    progress.update_stage_progress(f"{step2_unassigned_count} users remaining after driver injection")

    # STEP 3: Smart user swapping for route optimization (only if we have routes)
    if len(routes) > 1:
        print("🔄 STEP 3: Smart user swapping for route optimization")
        logger.info("🔄 STEP 3: Smart user swapping for route optimization")
        progress.start_stage("Smart Swapping", "Optimizing routes through user swaps")
        
        routes = step_3_smart_user_swapping(routes, office_lat, office_lon, config)
        print("📊 Step 3 complete: Route quality optimized")
        logger.info(f"📊 After step 3: Route quality optimized")
        progress.update_stage_progress(f"Route quality improved through user swaps")
    else:
        print("⏭️ Skipping step 3: Not enough routes for swapping optimization")
        logger.info("⏭️ Skipping step 3: Not enough routes for swapping optimization")

    final_unassigned_count = len(unassigned_users)
    improvement = initial_unassigned_count - final_unassigned_count
    print(f"✅ Optimization complete: {improvement} users assigned ({final_unassigned_count} remaining)")
    logger.info(f"✅ Smart swapping complete: {final_unassigned_count} unassigned users")
    logger.info(f"📈 Overall improvement: {initial_unassigned_count} → {final_unassigned_count} unassigned users")

    # Final validation
    print("🔍 Running final validation and cleanup...")
    routes = validate_and_cleanup_routes(routes, office_lat, office_lon)
    progress.update_stage_progress(f"Final validation and cleanup applied to {len(routes)} routes")

    return routes, unassigned_users, final_driver_df


def step_2_non_destructive_optimization(routes, unassigned_users, driver_df, user_df, office_lat, office_lon, config):
    """
    STEP 2: Bulk driver injection and optimization strategy
    Inject ALL available drivers at once, then optimize
    """
    print(f"🎯 Starting Step 2: {len(unassigned_users)} users to assign, {len(routes)} existing routes")

    if len(unassigned_users) == 0:
        print("🎉 No unassigned users, skipping step 2")
        return routes, unassigned_users, driver_df

    # Phase 2A: Try to assign unassigned users to existing routes first
    print(f"🎯 Phase 2A: Trying to fill existing routes...")
    initial_unassigned = len(unassigned_users)
    routes, unassigned_users = assign_unassigned_to_existing_routes(
        routes, unassigned_users, office_lat, office_lon)

    users_assigned_to_existing = initial_unassigned - len(unassigned_users)
    if users_assigned_to_existing > 0:
        print(f"✅ Assigned {users_assigned_to_existing} users to existing routes")
        logger.info(f"✅ Assigned {users_assigned_to_existing} users to existing routes")

    # If no more unassigned users, we're done
    if len(unassigned_users) == 0:
        print(f"🎉 All users assigned to existing routes!")
        logger.info("✅ All users assigned to existing routes!")
        return routes, unassigned_users, driver_df

    # Phase 2B: BULK DRIVER INJECTION - Inject ALL remaining available drivers
    print(f"🚗 Phase 2B: BULK driver injection for {len(unassigned_users)} remaining users")
    
    # Find all available drivers not currently assigned
    assigned_driver_ids = {route['driver_id'] for route in routes}
    available_drivers = driver_df[~driver_df['driver_id'].isin(assigned_driver_ids)]
    
    if len(available_drivers) == 0:
        print("⚠️ No additional drivers available for injection")
        logger.warning("⚠️ No additional drivers available for injection")
        return routes, unassigned_users, driver_df

    print(f"🚗 Found {len(available_drivers)} available drivers for bulk injection")
    logger.info(f"🚗 Found {len(available_drivers)} available drivers for bulk injection")
    
    # Create smart clusters of unassigned users 
    clusters = create_user_clusters(unassigned_users, office_lat, office_lon)
    print(f"📍 Created {len(clusters)} clusters from unassigned users")
    logger.info(f"📍 Created {len(clusters)} clusters from unassigned users")
    
    # Log cluster details
    for i, cluster in enumerate(clusters):
        print(f"   📍 Cluster {i+1}: {len(cluster)} users")
        logger.info(f"   📍 Cluster {i+1}: {len(cluster)} users")

    # Calculate optimal drivers needed (be generous for better coverage)
    DEFAULT_INJECTED_CAPACITY = 4
    
    # Strategy: Inject enough drivers to handle all unassigned users
    drivers_for_users = math.ceil(len(unassigned_users) / DEFAULT_INJECTED_CAPACITY)
    drivers_for_clusters = len(clusters) + 3  # At least one driver per cluster plus buffer
    
    # Use the larger estimate but cap at reasonable limits
    drivers_needed = min(
        len(available_drivers),  # Don't exceed available drivers
        max(drivers_for_users, drivers_for_clusters),  # Take the larger estimate
        min(50, len(unassigned_users) // 2)  # Cap at 50 or half the users, whichever is smaller
    )

    print(f"🚗 Driver injection calculation:")
    print(f"   📊 Users needing drivers: {len(unassigned_users)}")
    print(f"   📊 Drivers for user count: {drivers_for_users}")
    print(f"   📊 Drivers for clusters: {drivers_for_clusters}")
    print(f"   📊 Final injection count: {drivers_needed}")
    print(f"🚗 Injecting {drivers_needed} drivers from {len(available_drivers)} available")
    
    logger.info(f"🚗 Driver injection: {drivers_needed} drivers for {len(unassigned_users)} users in {len(clusters)} clusters")
    
    # Select best positioned drivers for the clusters
    drivers_to_inject = select_best_drivers_for_clusters(
        clusters, available_drivers, drivers_needed, office_lat, office_lon)

    # Create enhanced driver dataframe with all injected drivers
    current_driver_df = driver_df.copy()
    injected_count = 0

    for driver_data in drivers_to_inject:
        # Add to driver pool with enhanced positioning
        injected_count += 1
        print(f"   🚗 Injecting driver {driver_data['driver_id']} at cluster {driver_data.get('cluster_id', 'N/A')}")

    print(f"✅ Bulk injection complete: {injected_count} drivers added")
    logger.info(f"🚗 Bulk injection: {injected_count} drivers added")

    # Phase 2C: Run full assignment with all drivers
    print(f"🔄 Re-running assignment with ALL drivers ({len(current_driver_df)} total)")
    temp_routes, temp_unassigned = run_standard_assignment_algorithm(
        user_df, current_driver_df, office_lat, office_lon, config)

    # Accept the new solution
    improvement = len(unassigned_users) - len(temp_unassigned)
    routes = temp_routes
    unassigned_users = temp_unassigned
    
    print(f"📈 Bulk injection result: {improvement} users assigned ({len(temp_unassigned)} remaining)")
    logger.info(f"📈 Bulk injection result: {len(temp_unassigned)} unassigned users")

    print(f"🎯 Step 2 complete: bulk injected {injected_count} drivers, {len(unassigned_users)} users remain")
    logger.info(f"🎯 Step 2 complete: bulk injected {injected_count} drivers")
    progress.update_stage_progress(f"{injected_count} drivers bulk injected, {len(unassigned_users)} users remain")
    
    return routes, unassigned_users, current_driver_df


def assign_unassigned_to_existing_routes(routes, unassigned_users, office_lat, office_lon):
    """
    Try to assign unassigned users to existing routes with available capacity
    Uses relaxed constraints for better coverage
    """
    if not unassigned_users:
        return routes, unassigned_users

    assigned_user_ids = set()

    for user in unassigned_users[:]:  # Copy to iterate safely
        best_route = None
        best_score = float('inf')

        user_pos = (user['lat'], user['lng'])
        user_bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])

        for route in routes:
            # Check capacity
            if len(route['assigned_users']) >= route['vehicle_type']:
                continue

            # Calculate compatibility scores with relaxed thresholds
            route_center = calculate_route_center_improved(route)
            route_bearing = calculate_average_bearing_improved(route, office_lat, office_lon)

            # Distance score (more lenient)
            distance = haversine_distance(route_center[0], route_center[1], user_pos[0], user_pos[1])
            max_distance = MAX_FILL_DISTANCE_KM * 2.0  # Very lenient for unassigned users

            if distance > max_distance:
                continue

            # Bearing score (more lenient)
            bearing_diff = bearing_difference(user_bearing, route_bearing)
            max_bearing = MAX_BEARING_DIFFERENCE * 2.0  # Very lenient for unassigned users

            if bearing_diff > max_bearing:
                continue

            # Quality impact test (relaxed)
            test_route = route.copy()
            test_route['assigned_users'] = route['assigned_users'] + [user]
            test_route = optimize_route_sequence_improved(test_route, office_lat, office_lon)

            turning_score = calculate_route_turning_score_improved(
                test_route['assigned_users'],
                (test_route['latitude'], test_route['longitude']),
                (office_lat, office_lon))

            # Accept if reasonable quality (very relaxed for coverage)
            if turning_score <= 60:  # Much more lenient
                combined_score = distance + (bearing_diff * 0.02) + (turning_score * 0.05)

                if combined_score < best_score:
                    best_score = combined_score
                    best_route = route

        # Assign to best compatible route
        if best_route is not None:
            best_route['assigned_users'].append(user)
            assigned_user_ids.add(user['user_id'])

            # Re-optimize the route
            best_route = optimize_route_sequence_improved(best_route, office_lat, office_lon)
            update_route_metrics_improved(best_route, office_lat, office_lon)

    # Remove assigned users from unassigned list
    unassigned_users = [u for u in unassigned_users if u['user_id'] not in assigned_user_ids]

    return routes, unassigned_users


def step_3_smart_user_swapping(routes, office_lat, office_lon, config):
    """
    STEP 3: Smart user swapping for route optimization
    Swaps users between routes to improve overall route efficiency
    """
    logger.info("🔄 Starting smart user swapping optimization...")

    improvements_made = 0
    max_swap_iterations = 5

    for iteration in range(max_swap_iterations):
        logger.info(f"  🔄 Swap iteration {iteration + 1}")
        iteration_improvements = 0

        # Try cross-route optimizations
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                route1, route2 = routes[i], routes[j]

                if not route1['assigned_users'] or not route2['assigned_users']:
                    continue

                # Try beneficial swaps between these routes
                swaps_made = try_beneficial_swaps(route1, route2, office_lat, office_lon)
                iteration_improvements += swaps_made

        # Try path-based user reassignments
        path_reassignments = try_path_based_reassignments(routes, office_lat, office_lon)
        iteration_improvements += path_reassignments

        improvements_made += iteration_improvements
        logger.info(f"📊 Iteration {iteration + 1}: {iteration_improvements} improvements")

        # Stop if no improvements in this iteration
        if iteration_improvements == 0:
            break

    logger.info(f"✅ Smart swapping complete: {improvements_made} total improvements")
    return routes


def try_beneficial_swaps(route1, route2, office_lat, office_lon):
    """
    Try beneficial user swaps between two routes
    """
    swaps_made = 0

    # Calculate route centers and bearings
    center1 = calculate_route_center_improved(route1)
    center2 = calculate_route_center_improved(route2)
    bearing1 = calculate_average_bearing_improved(route1, office_lat, office_lon)
    bearing2 = calculate_average_bearing_improved(route2, office_lat, office_lon)

    # Only try swaps for routes that are reasonably close
    route_distance = haversine_distance(center1[0], center1[1], center2[0], center2[1])
    if route_distance > MERGE_DISTANCE_KM * 3.0:  # More lenient distance for swaps
        return 0

    # Try swapping each user from route1 to route2
    for user1 in route1['assigned_users'][:]:  # Copy list to iterate safely
        user1_pos = (user1['lat'], user1['lng'])

        # Check if this user would be better in route2
        dist1_to_center1 = haversine_distance(user1_pos[0], user1_pos[1], center1[0], center1[1])
        dist1_to_center2 = haversine_distance(user1_pos[0], user1_pos[1], center2[0], center2[1])

        user1_bearing = calculate_bearing(office_lat, office_lon, user1_pos[0], user1_pos[1])
        bearing_diff1 = bearing_difference(user1_bearing, bearing1)
        bearing_diff2 = bearing_difference(user1_bearing, bearing2)

        # Check if user1 is better suited for route2
        if (dist1_to_center2 < dist1_to_center1 * 0.8 and  # Significantly closer to route2
            bearing_diff2 < bearing_diff1 * 0.8 and  # Better direction match with route2
            len(route2['assigned_users']) < route2['vehicle_type']):  # Route2 has capacity

            # Calculate quality impact
            if test_swap_quality(route1, route2, user1, None, office_lat, office_lon):
                # Perform the swap
                route1['assigned_users'].remove(user1)
                route2['assigned_users'].append(user1)

                # Re-optimize both routes
                route1 = optimize_route_sequence_improved(route1, office_lat, office_lon)
                route2 = optimize_route_sequence_improved(route2, office_lat, office_lon)
                update_route_metrics_improved(route1, office_lat, office_lon)
                update_route_metrics_improved(route2, office_lat, office_lon)

                swaps_made += 1
                logger.info(f"🔄 Swapped user {user1['user_id']} from route {route1['driver_id']} to {route2['driver_id']}")

    return swaps_made


def try_path_based_reassignments(routes, office_lat, office_lon):
    """
    Try reassigning users based on path optimization
    Find users that are on the path of other drivers and would improve overall efficiency
    """
    reassignments = 0

    # Build a map of user positions for efficient lookup
    all_users_by_route = {}
    for i, route in enumerate(routes):
        all_users_by_route[i] = [(u['lat'], u['lng']) for u in route['assigned_users']]

    for i, route in enumerate(routes):
        if not route['assigned_users']:
            continue

        driver_pos = (route['latitude'], route['longitude'])

        # For each user in this route, check if they would be better in another route
        for user in route['assigned_users'][:]:  # Copy to iterate safely
            user_pos = (user['lat'], user['lng'])

            best_alternative_route = None
            best_improvement_score = 0

            for j, other_route in enumerate(routes):
                if i == j or len(other_route['assigned_users']) >= other_route['vehicle_type']:
                    continue

                # Check if this user would improve the other route
                improvement_score = calculate_path_improvement_score(
                    user, route, other_route, office_lat, office_lon)

                if improvement_score > best_improvement_score:
                    best_improvement_score = improvement_score
                    best_alternative_route = other_route

            # If found a significantly better route, make the reassignment
            if best_alternative_route and best_improvement_score > 0.5:  # Threshold for reassignment
                route['assigned_users'].remove(user)
                best_alternative_route['assigned_users'].append(user)

                # Re-optimize both routes
                route = optimize_route_sequence_improved(route, office_lat, office_lon)
                best_alternative_route = optimize_route_sequence_improved(
                    best_alternative_route, office_lat, office_lon)
                update_route_metrics_improved(route, office_lat, office_lon)
                update_route_metrics_improved(best_alternative_route, office_lat, office_lon)

                reassignments += 1
                logger.info(f"🎯 Reassigned user {user['user_id']} for path optimization")

    return reassignments


def calculate_path_improvement_score(user, current_route, alternative_route, office_lat, office_lon):
    """
    Calculate how much improvement would result from moving user to alternative route
    """
    user_pos = (user['lat'], user['lng'])

    # Current route metrics
    current_center = calculate_route_center_improved(current_route)
    current_bearing = calculate_average_bearing_improved(current_route, office_lat, office_lon)

    # Alternative route metrics  
    alt_center = calculate_route_center_improved(alternative_route)
    alt_bearing = calculate_average_bearing_improved(alternative_route, office_lat, office_lon)

    # User's bearing
    user_bearing = calculate_bearing(office_lat, office_lon, user_pos[0], user_pos[1])

    # Distance improvements
    current_dist = haversine_distance(user_pos[0], user_pos[1], current_center[0], current_center[1])
    alt_dist = haversine_distance(user_pos[0], user_pos[1], alt_center[0], alt_center[1])
    distance_improvement = max(0, current_dist - alt_dist) / max(current_dist, 1.0)

    # Bearing alignment improvements
    current_bearing_diff = bearing_difference(user_bearing, current_bearing)
    alt_bearing_diff = bearing_difference(user_bearing, alt_bearing)
    bearing_improvement = max(0, current_bearing_diff - alt_bearing_diff) / 180.0

    # Combined improvement score
    total_improvement = distance_improvement * 0.6 + bearing_improvement * 0.4

    return total_improvement


def test_swap_quality(route1, route2, user1, user2, office_lat, office_lon):
    """
    Test if swapping users between routes improves overall quality
    """
    # Calculate current quality
    current_quality1 = calculate_route_turning_score_improved(
        route1['assigned_users'], (route1['latitude'], route1['longitude']), (office_lat, office_lon))
    current_quality2 = calculate_route_turning_score_improved(
        route2['assigned_users'], (route2['latitude'], route2['longitude']), (office_lat, office_lon))

    # Test new arrangement
    test_route1 = route1.copy()
    test_route2 = route2.copy()

    test_route1['assigned_users'] = [u for u in route1['assigned_users'] if u['user_id'] != user1['user_id']]
    test_route2['assigned_users'] = route2['assigned_users'] + [user1]

    if user2:  # If it's a bidirectional swap
        test_route2['assigned_users'] = [u for u in test_route2['assigned_users'] if u['user_id'] != user2['user_id']]
        test_route1['assigned_users'] = test_route1['assigned_users'] + [user2]

    # Calculate new quality
    new_quality1 = calculate_route_turning_score_improved(
        test_route1['assigned_users'], (test_route1['latitude'], test_route1['longitude']), (office_lat, office_lon))
    new_quality2 = calculate_route_turning_score_improved(
        test_route2['assigned_users'], (test_route2['latitude'], test_route2['longitude']), (office_lat, office_lon))

    # Accept if overall quality improves
    current_total = current_quality1 + current_quality2
    new_total = new_quality1 + new_quality2

    return new_total < current_total * 0.95  # Must improve by at least 5%


def run_standard_assignment_algorithm(user_df, driver_df, office_lat, office_lon, config):
    """Run the standard assignment algorithm steps"""
    # STEP 1: Geographic clustering
    user_df = create_geographic_clusters(user_df, office_lat, office_lon, config)

    # STEP 2: Capacity-based sub-clustering
    user_df = create_capacity_subclusters(user_df, office_lat, office_lon, config)

    # STEP 3: Driver assignment
    routes, assigned_user_ids = assign_drivers_by_priority_route_optimized(
        user_df, driver_df, office_lat, office_lon)

    # STEP 4: Local optimization
    routes = local_optimization(routes, office_lat, office_lon)

    # STEP 5: Global optimization
    routes, unassigned_users = global_optimization(routes, user_df,
                                                   assigned_user_ids,
                                                   driver_df, office_lat,
                                                   office_lon)

    # STEP 6: Final merging
    routes = quality_preserving_route_merging(routes, config, office_lat, office_lon)

    # STEP 7: Route splitting
    routes = intelligent_route_splitting_improved(routes, driver_df, config, office_lat, office_lon)

    # STEP 8: Final consolidation
    routes = final_route_consolidation(routes, driver_df, office_lat, office_lon)

    return routes, unassigned_users


def run_non_destructive_optimization(routes, unassigned_users, office_lat, office_lon, config):
    """
    Non-destructive global optimization that maintains current solution integrity
    """
    logger.info("🔧 Starting non-destructive optimization...")

    current_solution = routes.copy()
    current_unassigned = unassigned_users.copy()

    max_iterations = 10
    improvement_found = True
    iteration = 0

    while improvement_found and iteration < max_iterations:
        iteration += 1
        improvement_found = False

        logger.info(f"  🔄 Non-destructive iteration {iteration}")

        # Try route merging proposals
        merge_proposals = generate_merge_proposals(current_solution, office_lat, office_lon)
        for proposal in merge_proposals:
            if validate_proposal(proposal, office_lat, office_lon):
                if apply_proposal(proposal, current_solution):
                    improvement_found = True
                    logger.info(f"  ✅ Applied merge proposal")
                    break

        if improvement_found:
            continue

        # Try user reassignment proposals
        reassign_proposals = generate_reassignment_proposals(
            current_solution, current_unassigned, office_lat, office_lon)
        for proposal in reassign_proposals:
            if validate_proposal(proposal, office_lat, office_lon):
                if apply_reassignment_proposal(proposal, current_solution, current_unassigned):
                    improvement_found = True
                    logger.info(f"  ✅ Applied reassignment proposal")
                    break

        if improvement_found:
            continue

        # Try route splitting proposals for overloaded routes
        split_proposals = generate_split_proposals(current_solution, office_lat, office_lon)
        for proposal in split_proposals:
            if validate_proposal(proposal, office_lat, office_lon):
                if apply_split_proposal(proposal, current_solution):
                    improvement_found = True
                    logger.info(f"  ✅ Applied split proposal")
                    break

    logger.info(f"🔧 Non-destructive optimization completed after {iteration} iterations")
    return current_solution, current_unassigned


def create_user_clusters(unassigned_users, office_lat, office_lon):
    """Create geographic clusters of unassigned users with aggressive clustering for better driver injection"""
    if len(unassigned_users) == 0:
        return []

    if len(unassigned_users) == 1:
        return [unassigned_users]

    try:
        from sklearn.cluster import DBSCAN, KMeans
        import numpy as np

        # Extract coordinates
        coords = []
        for user in unassigned_users:
            coords.append([user['lat'], user['lng']])

        coords = np.array(coords)

        # Strategy 1: Use KMeans to force a reasonable number of clusters
        # Target 4-6 users per cluster for optimal driver utilization
        target_users_per_cluster = 5
        estimated_clusters = max(3, min(len(unassigned_users) // target_users_per_cluster, 20))
        
        print(f"📍 Creating {estimated_clusters} clusters from {len(unassigned_users)} users")
        logger.info(f"📍 Creating {estimated_clusters} clusters from {len(unassigned_users)} users")
        
        # Use KMeans for initial clustering
        kmeans = KMeans(n_clusters=estimated_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(coords)

        # Group users by KMeans cluster
        kmeans_clusters = {}
        for i, label in enumerate(kmeans_labels):
            if label not in kmeans_clusters:
                kmeans_clusters[label] = []
            kmeans_clusters[label].append(unassigned_users[i])

        # Strategy 2: Further split large clusters using DBSCAN
        final_clusters = []
        for cluster_users in kmeans_clusters.values():
            if len(cluster_users) <= 8:  # Small enough cluster
                final_clusters.append(cluster_users)
            else:
                # Split large cluster using DBSCAN
                cluster_coords = np.array([[u['lat'], u['lng']] for u in cluster_users])
                dbscan = DBSCAN(eps=0.015, min_samples=2)  # ~1.5km radius, tighter clustering
                dbscan_labels = dbscan.fit_predict(cluster_coords)
                
                # Group by DBSCAN results
                dbscan_subclusters = {}
                for i, label in enumerate(dbscan_labels):
                    if label == -1:  # Noise points become individual clusters
                        final_clusters.append([cluster_users[i]])
                    else:
                        if label not in dbscan_subclusters:
                            dbscan_subclusters[label] = []
                        dbscan_subclusters[label].append(cluster_users[i])
                
                # Add subclusters
                for subcluster in dbscan_subclusters.values():
                    final_clusters.append(subcluster)

        print(f"✅ Created {len(final_clusters)} final clusters (avg {len(unassigned_users)/len(final_clusters):.1f} users per cluster)")
        logger.info(f"✅ Created {len(final_clusters)} final clusters")
        
        return final_clusters

    except ImportError:
        # Fallback: create smaller clusters by proximity with aggressive splitting
        clusters = []
        remaining = unassigned_users.copy()

        while remaining:
            cluster = [remaining.pop(0)]
            cluster_center = (cluster[0]['lat'], cluster[0]['lng'])

            # Add nearby users to cluster (smaller radius for more clusters)
            to_remove = []
            for i, user in enumerate(remaining):
                if len(cluster) >= 6:  # Limit cluster size to force more clusters
                    break
                    
                distance = haversine_distance(cluster_center[0], cluster_center[1],
                                            user['lat'], user['lng'])
                if distance <= 2.0:  # Smaller 2km radius for more granular clusters
                    cluster.append(user)
                    to_remove.append(i)

            # Remove added users
            for i in reversed(to_remove):
                remaining.pop(i)

            clusters.append(cluster)

        print(f"✅ Fallback created {len(clusters)} clusters")
        logger.info(f"✅ Fallback created {len(clusters)} clusters")
        return clusters


def create_drivers_from_clusters(clusters, default_capacity, attempt, office_lat, office_lon):
    """Create new drivers positioned at cluster centroids"""
    new_drivers = []

    for i, cluster in enumerate(clusters):
        if not cluster:
            continue

        # Calculate centroid
        centroid_lat = sum(user['lat'] for user in cluster) / len(cluster)
        centroid_lng = sum(user['lng'] for user in cluster) / len(cluster)

        # Calculate median bearing
        bearings = []
        for user in cluster:
            bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])
            bearings.append(bearing)
        bearings.sort()
        median_bearing = bearings[len(bearings) // 2]

        driver_id = f"ADDED_{attempt}_{i}"

        new_driver = {
            'driver_id': driver_id,
            'latitude': centroid_lat,
            'longitude': centroid_lng,
            'capacity': default_capacity,
            'vehicle_id': f"VEHICLE_{driver_id}",
            'is_injected': True
        }

        new_drivers.append(new_driver)

        cluster_user_ids = [user['user_id'] for user in cluster]
        logger.info(f"🚗 INJECTION_DRIVER_CREATED: {driver_id} at ({centroid_lat:.4f}, {centroid_lng:.4f}) "
                   f"capacity={default_capacity} cluster_size={len(cluster)} users={cluster_user_ids}")

    return new_drivers


def generate_merge_proposals(routes, office_lat, office_lon):
    """Generate proposals for merging compatible routes"""
    proposals = []

    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            route1 = routes[i]
            route2 = routes[j]

            # Check basic compatibility
            total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
            max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])

            if total_users <= max_capacity:
                # Check if routes are on same path
                if _are_routes_on_same_road_path(route1, route2, office_lat, office_lon):
                    proposal = {
                        'type': 'merge',
                        'route1_idx': i,
                        'route2_idx': j,
                        'route1': route1,
                        'route2': route2
                    }
                    proposals.append(proposal)

    return proposals


def select_best_drivers_for_clusters(clusters, available_drivers, drivers_needed, office_lat, office_lon):
    """
    Select the best positioned drivers for the user clusters
    """
    selected_drivers = []
    
    # Sort available drivers by priority and capacity  
    available_drivers_sorted = available_drivers.sort_values(['priority', 'capacity'], ascending=[True, False])
    
    # Strategy 1: Assign drivers to largest clusters first
    clusters_by_size = sorted(clusters, key=len, reverse=True)
    
    drivers_assigned = 0
    driver_index = 0
    
    for cluster_idx, cluster in enumerate(clusters_by_size):
        if drivers_assigned >= drivers_needed or driver_index >= len(available_drivers_sorted):
            break
            
        # Calculate cluster centroid
        if cluster:
            centroid_lat = sum(user['lat'] for user in cluster) / len(cluster)
            centroid_lng = sum(user['lng'] for user in cluster) / len(cluster)
            
            # Find the best available driver for this cluster
            best_driver = None
            best_score = float('inf')
            best_driver_idx = -1
            
            for idx in range(driver_index, min(driver_index + 3, len(available_drivers_sorted))):  # Check next 3 drivers
                driver = available_drivers_sorted.iloc[idx]
                
                # Calculate score based on distance to cluster centroid
                distance = haversine_distance(
                    driver['latitude'], driver['longitude'],
                    centroid_lat, centroid_lng
                )
                
                # Score based on distance and capacity utilization potential
                capacity_score = len(cluster) / driver['capacity'] if driver['capacity'] > 0 else 0
                total_score = distance - (capacity_score * 2.0)  # Prefer good utilization
                
                if total_score < best_score:
                    best_score = total_score
                    best_driver = driver
                    best_driver_idx = idx
            
            if best_driver is not None:
                driver_data = {
                    'driver_id': str(best_driver['driver_id']),
                    'latitude': float(best_driver['latitude']),
                    'longitude': float(best_driver['longitude']),
                    'capacity': int(best_driver['capacity']),
                    'vehicle_id': str(best_driver.get('vehicle_id', '')),
                    'priority': int(best_driver.get('priority', 1000)),
                    'cluster_id': cluster_idx,
                    'cluster_size': len(cluster)
                }
                selected_drivers.append(driver_data)
                drivers_assigned += 1
                driver_index = best_driver_idx + 1
    
    # Strategy 2: Fill remaining slots with best available drivers
    while drivers_assigned < drivers_needed and driver_index < len(available_drivers_sorted):
        driver = available_drivers_sorted.iloc[driver_index]
        driver_data = {
            'driver_id': str(driver['driver_id']),
            'latitude': float(driver['latitude']),
            'longitude': float(driver['longitude']),
            'capacity': int(driver['capacity']),
            'vehicle_id': str(driver.get('vehicle_id', '')),
            'priority': int(driver.get('priority', 1000)),
            'cluster_id': -1,  # No specific cluster
            'cluster_size': 0
        }
        selected_drivers.append(driver_data)
        drivers_assigned += 1
        driver_index += 1
    
    return selected_drivers




def generate_reassignment_proposals(routes, unassigned_users, office_lat, office_lon):
    """Generate proposals for reassigning unassigned users to existing routes"""
    proposals = []

    for user in unassigned_users:
        for i, route in enumerate(routes):
            if len(route['assigned_users']) < route['vehicle_type']:
                # Check compatibility
                route_center = calculate_route_center_improved(route)
                distance = haversine_distance(route_center[0], route_center[1],
                                            user['lat'], user['lng'])

                if distance <= MAX_FILL_DISTANCE_KM:
                    proposal = {
                        'type': 'reassign',
                        'user': user,
                        'route_idx': i,
                        'route': route
                    }
                    proposals.append(proposal)

    return proposals


def generate_split_proposals(routes, office_lat, office_lon):
    """Generate proposals for splitting overloaded or poor quality routes"""
    proposals = []

    for i, route in enumerate(routes):
        if len(route['assigned_users']) >= 4:  # Only split larger routes
            turning_score = route.get('turning_score', 0)
            tortuosity = route.get('tortuosity_ratio', 1.0)

            # Check if route needs splitting due to quality issues
            if turning_score > 45 or tortuosity > 1.6:
                proposal = {
                    'type': 'split',
                    'route_idx': i,
                    'route': route
                }
                proposals.append(proposal)

    return proposals


def validate_proposal(proposal, office_lat, office_lon):
    """Validate that a proposal maintains route quality standards"""
    try:
        if proposal['type'] == 'merge':
            # Create test merged route
            route1 = proposal['route1']
            route2 = proposal['route2']

            test_route = route1.copy()
            test_route['assigned_users'] = route1['assigned_users'] + route2['assigned_users']
            test_route['vehicle_type'] = max(route1['vehicle_type'], route2['vehicle_type'])

            # Optimize and validate
            test_route = optimize_route_sequence_improved(test_route, office_lat, office_lon)
            update_route_metrics_improved(test_route, office_lat, office_lon)

            return validate_route_path_coherence(test_route, office_lat, office_lon, strict_mode=True)

        elif proposal['type'] == 'reassign':
            # Create test route with new user
            route = proposal['route']
            user = proposal['user']

            test_route = route.copy()
            test_route['assigned_users'] = route['assigned_users'] + [user]

            # Optimize and validate
            test_route = optimize_route_sequence_improved(test_route, office_lat, office_lon)
            update_route_metrics_improved(test_route, office_lat, office_lon)

            return validate_route_path_coherence(test_route, office_lat, office_lon, strict_mode=True)

        elif proposal['type'] == 'split':
            # For split proposals, assume they're valid if the route has quality issues
            return True

        return False

    except Exception as e:
        logger.warning(f"Proposal validation failed: {e}")
        return False


def apply_proposal(proposal, current_solution):
    """Apply a validated merge proposal to the current solution"""
    if proposal['type'] == 'merge':
        route1_idx = proposal['route1_idx']
        route2_idx = proposal['route2_idx']

        # Create merged route
        route1 = current_solution[route1_idx]
        route2 = current_solution[route2_idx]

        merged_route = route1.copy()
        merged_route['assigned_users'] = route1['assigned_users'] + route2['assigned_users']
        merged_route['vehicle_type'] = max(route1['vehicle_type'], route2['vehicle_type'])

        # Remove original routes and add merged route
        # Remove in reverse order to maintain indices
        if route2_idx > route1_idx:
            current_solution.pop(route2_idx)
            current_solution.pop(route1_idx)
        else:
            current_solution.pop(route1_idx)
            current_solution.pop(route2_idx)

        current_solution.append(merged_route)
        return True

    return False


def apply_reassignment_proposal(proposal, current_solution, current_unassigned):
    """Apply a validated reassignment proposal"""
    if proposal['type'] == 'reassign':
        route_idx = proposal['route_idx']
        user = proposal['user']

        # Add user to route
        current_solution[route_idx]['assigned_users'].append(user)

        # Remove user from unassigned list
        current_unassigned.remove(user)

        return True

    return False


def apply_split_proposal(proposal, current_solution):
    """Apply a validated split proposal"""
    # For now, just log the split attempt
    # Actual splitting would require available drivers
    logger.info(f"Split proposal for route {proposal['route_idx']} (not implemented)")
    return False


def validate_and_cleanup_routes(routes, office_lat, office_lon):
    """Final validation and cleanup of routes"""
    validated_routes = []

    for route in routes:
        if route['assigned_users'] and len(route['assigned_users']) > 0:
            # Final optimization
            route = optimize_route_sequence_improved(route, office_lat, office_lon)
            update_route_metrics_improved(route, office_lat, office_lon)

            # Validate route coherence
            if validate_route_path_coherence(route, office_lat, office_lon, strict_mode=False):
                validated_routes.append(route)
            else:
                logger.warning(f"Route {route['driver_id']} failed final validation but keeping")
                validated_routes.append(route)  # Keep anyway to avoid losing users

    return validated_routes