import os
import math
import requests
import numpy as np
import pandas as pd
import time
import json
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import warnings
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ordering system for optimal pickup sequences
try:
    from ordering import apply_route_ordering
    ORDERING_AVAILABLE = True
except ImportError:
    ORDERING_AVAILABLE = False
    logger.warning("Ordering system not available - routes will not have optimal pickup sequences")

# Import algorithm-level caching system
try:
    from algorithm.algorithm_cache import get_algorithm_cache
    ALGORITHM_CACHE_AVAILABLE = True
except ImportError:
    ALGORITHM_CACHE_AVAILABLE = False
    logger.warning("Algorithm cache system not available - will run without caching")

# Import required base functions from assignment.py
from algorithm.base.base import (
    validate_input_data, load_env_and_fetch_data, extract_office_coordinates,
    prepare_user_driver_dataframes, haversine_distance, bearing_difference,
    calculate_bearing_vectorized, coords_to_km, optimize_route_sequence_improved,
    update_route_metrics_improved, calculate_average_bearing_improved,
    _get_all_drivers_as_unassigned, _convert_users_to_unassigned_format
)


# =============================================================================
# USER DATA PRESERVATION WRAPPERS (Prevent Duplicate Creation)
# =============================================================================

def optimize_route_sequence_safe(route, office_lat, office_lon):
    """
    SAFE wrapper for route optimization that preserves ALL user data
    Prevents creation of incomplete user objects during optimization
    """
    if not route.get('assigned_users'):
        return route

    # Store original complete user objects
    original_users = []
    user_id_mapping = {}  # Maps user_id -> original complete user object

    for user in route['assigned_users']:
        user_id = user.get('user_id', user.get('id', ''))
        if user_id:
            # Create a deep copy to preserve all original data
            import copy
            original_user = copy.deepcopy(user)
            original_users.append(original_user)
            user_id_mapping[user_id] = original_user

    try:
        # Call the original optimization function
        optimized_route = optimize_route_sequence_improved(route, office_lat, office_lon)

        # Restore complete user data to the optimized route
        restored_users = []
        for optimized_user in optimized_route.get('assigned_users', []):
            user_id = optimized_user.get('user_id', optimized_user.get('id', ''))

            if user_id in user_id_mapping:
                # Use the complete original user object
                complete_user = copy.deepcopy(user_id_mapping[user_id])
                # Preserve optimization-specific fields
                complete_user['pickup_order'] = optimized_user.get('pickup_order', len(restored_users) + 1)
                restored_users.append(complete_user)
            else:
                # Fallback: use optimized user if original not found
                restored_users.append(optimized_user)

        optimized_route['assigned_users'] = restored_users
        return optimized_route

    except Exception as e:
        print(f"    ⚠️ Route optimization failed, returning original route: {e}")

        # Even when optimization fails, ensure coordinate format consistency
        for user in route.get('assigned_users', []):
            if 'latitude' in user or 'longitude' in user:
                user['lat'] = float(user.get('latitude', user.get('lat', 0)))
                user['lng'] = float(user.get('longitude', user.get('lng', 0)))
                user.pop('latitude', None)
                user.pop('longitude', None)

            # Ensure pickup_order is assigned based on position if missing
            if 'pickup_order' not in user:
                for idx, route_user in enumerate(route.get('assigned_users', [])):
                    if route_user.get('user_id') == user.get('user_id'):
                        user['pickup_order'] = idx + 1
                        break

        return route


def update_route_metrics_safe(route, office_lat, office_lon):
    """
    SAFE wrapper for route metrics that preserves user data
    """
    try:
        return update_route_metrics_improved(route, office_lat, office_lon)
    except Exception as e:
        print(f"    ⚠️ Route metrics update failed: {e}")
        return route


# =============================================================================
# CAPACITY-OPTIMIZED CONFIGURATION (Balance.py Style with Capacity Branding)
# =============================================================================

def load_capacity_config():
    """
    Load capacity-optimized configuration with balance.py simplicity
    Maintains capacity branding while using proven balance.py approach
    """
    # Find the config file relative to this script's location
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(script_dir, 'config.json')

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load config.json from {config_path}, using defaults. Error: {e}")
        cfg = {}

    # Always use capacity mode
    current_mode = "capacity_optimization"

    # Get capacity optimization configuration
    mode_configs = cfg.get("mode_configs", {})
    mode_config = mode_configs.get("capacity_optimization", {})

    print(f"-> Using CAPACITY-OPTIMIZED algorithm (Balance.py Architecture)")

    # Main configuration with capacity branding
    config = {}

    # Geographic clustering parameters (Balance.py style)
    config['GEOGRAPHIC_CLUSTER_RADIUS_KM'] = float(mode_config.get("geographic_cluster_radius_km", cfg.get("geographic_cluster_radius_km", 1.2)))
    config['MAX_CLUSTER_SIZE'] = int(mode_config.get("max_cluster_size", cfg.get("max_cluster_size", 6)))
    config['MIN_SAMPLES_DBSCAN'] = 1  # Individual users allowed as clusters

    # Capacity matching parameters (Capacity-branded)
    config['DISTANCE_WEIGHT'] = float(mode_config.get("distance_weight", cfg.get("distance_weight", 0.7)))
    config['CAPACITY_WASTE_WEIGHT'] = float(mode_config.get("capacity_waste_weight", cfg.get("capacity_waste_weight", 0.3)))
    config['MIN_CAPACITY_UTILIZATION'] = float(mode_config.get("min_capacity_utilization", cfg.get("min_capacity_utilization", 0.4)))

    # Route optimization parameters
    config['MAX_ON_ROUTE_DETOUR_KM'] = float(mode_config.get("max_on_route_detour_km", cfg.get("max_on_route_detour_km", 2.0)))
    config['BEARING_TOLERANCE_DEGREES'] = float(mode_config.get("bearing_tolerance_degrees", cfg.get("bearing_tolerance_degrees", 45)))
    config['DIRECTION_TOLERANCE_DEGREES'] = float(mode_config.get("direction_tolerance_degrees", cfg.get("direction_tolerance_degrees", 60)))

    # Route merging parameters
    config['MAX_MERGE_DISTANCE_KM'] = float(mode_config.get("max_merge_distance_km", cfg.get("max_merge_distance_km", 4.0)))
    config['MERGE_SCORE_THRESHOLD'] = float(mode_config.get("merge_score_threshold", cfg.get("merge_score_threshold", 1.5)))
    config['SMALL_ROUTE_THRESHOLD'] = int(mode_config.get("small_route_threshold", cfg.get("small_route_threshold", 2)))

    # Capacity-specific parameters
    config['CAPACITY_PRIORITY_WEIGHT'] = float(mode_config.get("capacity_priority_weight", cfg.get("capacity_priority_weight", 2.0)))
    config['OVERFLOW_PENALTY_PER_SEAT'] = float(mode_config.get("overflow_penalty_per_seat", cfg.get("overflow_penalty_per_seat", 3.0)))
    config['UTILIZATION_BONUS_THRESHOLD'] = float(mode_config.get("utilization_bonus_threshold", cfg.get("utilization_bonus_threshold", 0.8)))

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

    print(f"   Geographic cluster radius: {config['GEOGRAPHIC_CLUSTER_RADIUS_KM']}km")
    print(f"   Max cluster size: {config['MAX_CLUSTER_SIZE']} users")
    print(f"   Distance weight: {config['DISTANCE_WEIGHT']}, Capacity waste weight: {config['CAPACITY_WASTE_WEIGHT']}")
    print(f"   Max on-route detour: {config['MAX_ON_ROUTE_DETOUR_KM']}km")
    print(f"   CAPACITY-OPTIMIZED: Geographic simplicity with capacity intelligence")

    return config

# Load capacity configuration
CONFIG = load_capacity_config()


# =============================================================================
# CORE CAPACITY FUNCTIONS (Balance.py Architecture with Capacity Branding)
# =============================================================================

def cluster_users_by_proximity(user_df, office_lat, office_lon):
    """
    CAPACITY-OPTIMIZED: Geographic clustering prioritizing closest neighbors
    Uses hierarchical clustering to group nearby users, always choosing closest pairs first
    """
    print(f"  🗺️  CAPACITY clustering: Grouping nearby users (radius: {CONFIG['GEOGRAPHIC_CLUSTER_RADIUS_KM']}km)...")

    try:
        if len(user_df) < 2:
            user_df['geo_cluster'] = 0
            print(f"    📍 Created 1 geographic cluster for {len(user_df)} users")
            return user_df

        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist, squareform

        # Convert coordinates to km for accurate distance calculation
        coords = user_df[['latitude', 'longitude']].values
        coords_km = np.array([coords_to_km(coord[0], coord[1], office_lat, office_lon) for coord in coords])

        # Calculate pairwise distances
        distances = pdist(coords_km, metric='euclidean')
        
        # Use hierarchical clustering with single linkage (closest neighbors first)
        linkage_matrix = linkage(distances, method='single')
        
        # Form clusters based on distance threshold
        clusters = fcluster(linkage_matrix, CONFIG['GEOGRAPHIC_CLUSTER_RADIUS_KM'], criterion='distance')
        
        user_df['geo_cluster'] = clusters

        # Split oversized clusters to prevent large geographic spreads (Capacity enhancement)
        user_df = split_oversized_clusters_capacity(user_df)

        print(f"    📍 Created {len(user_df['geo_cluster'].unique())} geographic clusters (closest-neighbor priority)")
        for cluster_id in user_df['geo_cluster'].unique():
            cluster_size = len(user_df[user_df['geo_cluster'] == cluster_id])
            print(f"      Cluster {cluster_id}: {cluster_size} users")

        return user_df

    except Exception as e:
        logger.error(f"Error in capacity geographic clustering: {e}")
        # Fallback: assign each user to individual cluster
        user_df['geo_cluster'] = range(len(user_df))
        return user_df


def split_oversized_clusters_capacity(user_df):
    """
    CAPACITY-OPTIMIZED: Split oversized clusters with capacity awareness
    Enhanced balance.py approach with capacity consideration
    """
    cluster_sizes = user_df['geo_cluster'].value_counts()
    oversized_clusters = cluster_sizes[cluster_sizes > CONFIG['MAX_CLUSTER_SIZE']].index

    if len(oversized_clusters) == 0:
        return user_df

    print(f"    ✂️  Splitting {len(oversized_clusters)} oversized clusters (max: {CONFIG['MAX_CLUSTER_SIZE']} users)")

    next_cluster_id = user_df['geo_cluster'].max() + 1

    for cluster_id in oversized_clusters:
        cluster_users = user_df[user_df['geo_cluster'] == cluster_id].copy()

        # Use capacity-aware splitting strategy
        if len(cluster_users) <= CONFIG['MAX_CLUSTER_SIZE'] * 2:
            # Simple split into two groups
            mid_point = len(cluster_users) // 2
            cluster_users.loc[cluster_users.index[mid_point:], 'geo_cluster'] = next_cluster_id
            next_cluster_id += 1
        else:
            # Use geographic splitting for very large clusters
            center_lat = cluster_users['latitude'].mean()
            center_lon = cluster_users['longitude'].mean()

            distances = []
            for _, user in cluster_users.iterrows():
                dist = haversine_distance(user['latitude'], user['longitude'], center_lat, center_lon)
                distances.append(dist)

            cluster_users = cluster_users.copy()
            cluster_users['dist_to_center'] = distances

            # Sort by distance and split
            cluster_users = cluster_users.sort_values('dist_to_center')
            chunk_size = CONFIG['MAX_CLUSTER_SIZE']

            for i in range(0, len(cluster_users), chunk_size):
                if i == 0:
                    # Keep first chunk in original cluster
                    continue
                else:
                    user_df.loc[cluster_users.index[i:i+chunk_size], 'geo_cluster'] = next_cluster_id
                    next_cluster_id += 1

    return user_df


def assign_individual_users_fallback(cluster_users, available_drivers, office_lat, office_lon):
    """
    CAPACITY ENHANCEMENT: Assign individual users from rejected clusters to available drivers
    Key balance.py-style fallback mechanism to maximize user assignment
    """
    assignments = []

    if cluster_users.empty or available_drivers.empty:
        return assignments

    print(f"    🔧 Individual fallback: trying to assign {len(cluster_users)} users individually...")

    # Sort users by distance from office for priority (closest first)
    office_distance_cache = {}
    def get_office_distance(lat, lon):
        cache_key = (lat, lon)
        if cache_key not in office_distance_cache:
            office_distance_cache[cache_key] = haversine_distance(lat, lon, office_lat, office_lon)
        return office_distance_cache[cache_key]

    cluster_users = cluster_users.copy()
    cluster_users['office_distance'] = cluster_users.apply(
        lambda row: get_office_distance(row['latitude'], row['longitude']), axis=1
    )
    cluster_users = cluster_users.sort_values('office_distance')

    # For each user, try to find the best available driver
    for _, user in cluster_users.iterrows():
        user_id = user['user_id']
        user_lat = user['latitude']
        user_lon = user['longitude']

        # Find drivers with at least 1 seat available
        suitable_drivers = available_drivers[available_drivers['capacity'] >= 1].copy()

        if suitable_drivers.empty:
            print(f"      ❌ User {user_id}: No drivers with any capacity available")
            assignments.append({'user_id': user_id, 'route': None, 'driver_id': None})
            continue

        # Find the best driver (closest with minimum waste)
        best_driver = None
        best_score = float('inf')

        for _, driver in suitable_drivers.iterrows():
            # Distance to user
            driver_to_user = haversine_distance(
                driver['latitude'], driver['longitude'],
                user_lat, user_lon
            )

            # Capacity waste (prefer smaller vehicles for single users)
            waste = driver['capacity'] - 1

            # Composite score (distance + waste penalty)
            score = driver_to_user + (waste * 0.5)

            if score < best_score:
                best_score = score
                best_driver = driver

        if best_driver is not None:
            # Create a route for this individual user
            route = create_route_from_cluster_capacity(
                cluster_users[cluster_users['user_id'] == user_id],
                best_driver,
                office_lat,
                office_lon
            )

            if route:
                assignments.append({
                    'user_id': user_id,
                    'route': route,
                    'driver_id': best_driver['driver_id']
                })

                # Remove this driver from available drivers
                available_drivers = available_drivers[available_drivers['driver_id'] != best_driver['driver_id']]
                print(f"      ✅ User {user_id}: Assigned to driver {best_driver['driver_id']} (score: {best_score:.2f})")
            else:
                assignments.append({'user_id': user_id, 'route': None, 'driver_id': None})
                print(f"      ❌ User {user_id}: Failed to create route")
        else:
            assignments.append({'user_id': user_id, 'route': None, 'driver_id': None})
            print(f"      ❌ User {user_id}: No suitable driver found")

    successful_assignments = sum(1 for a in assignments if a['route'] is not None)
    print(f"    📊 Individual fallback: {successful_assignments}/{len(assignments)} users assigned successfully")

    return assignments


def split_cluster_for_capacity_match(cluster_users, available_drivers, office_lat, office_lon):
    """
    CAPACITY ENHANCEMENT: Split cluster to match available driver capacities
    Key balance.py-style adaptive clustering
    """
    if available_drivers.empty:
        return None

    cluster_size = len(cluster_users)
    max_available_capacity = available_drivers['capacity'].max()

    print(f"      🔧 Cluster size: {cluster_size}, Max available capacity: {max_available_capacity}")

    if max_available_capacity == 0:
        return None

    # Create sub-clusters that fit available driver capacities
    sub_clusters = []
    remaining_users = cluster_users.copy()

    # Sort users by proximity to create geographically coherent sub-clusters
    if len(remaining_users) > 1:
        center_lat = remaining_users['latitude'].mean()
        center_lon = remaining_users['longitude'].mean()

        remaining_users['distance_from_center'] = remaining_users.apply(
            lambda row: haversine_distance(row['latitude'], row['longitude'], center_lat, center_lon),
            axis=1
        )
        remaining_users = remaining_users.sort_values('distance_from_center')

    # Create sub-clusters based on available driver capacities
    driver_capacity_counts = available_drivers['capacity'].value_counts().sort_index(ascending=False)

    for capacity, driver_count in driver_capacity_counts.items():
        while len(remaining_users) >= capacity and driver_count > 0:
            # Take the closest users for the next sub-cluster
            sub_cluster = remaining_users.head(capacity).copy()
            sub_clusters.append(sub_cluster)
            remaining_users = remaining_users.iloc[capacity:]
            driver_count -= 1

            print(f"      📍 Created sub-cluster of {capacity} users")

            if len(remaining_users) == 0:
                break

    # Handle remaining users with smaller capacity drivers
    if len(remaining_users) > 0:
        min_capacity = min(available_drivers['capacity'])
        if len(remaining_users) <= min_capacity:
            sub_clusters.append(remaining_users)
            print(f"      📍 Created final sub-cluster of {len(remaining_users)} users")
        else:
            # Split remaining users into smallest possible groups
            for i in range(0, len(remaining_users), min_capacity):
                end_idx = min(i + min_capacity, len(remaining_users))
                sub_clusters.append(remaining_users.iloc[i:end_idx])
                print(f"      📍 Created small sub-cluster of {end_idx - i} users")

    if not sub_clusters:
        return None

    print(f"      ✂️  Split cluster into {len(sub_clusters)} sub-clusters")

    # Return the first sub-cluster as representative (main cluster assignment will handle others)
    # The other sub-clusters will be processed as individual clusters in the main loop
    return sub_clusters[0]


def assign_cab_to_cluster_capacity(cluster_users, available_drivers, office_lat, office_lon):
    """
    CAPACITY-OPTIMIZED: Smart capacity matching with distance-sensitive scoring
    Enhanced balance.py approach with capacity prioritization
    """
    cluster_size = len(cluster_users)

    if cluster_size == 0:
        return None

    print(f"    🚗 Finding optimal cab for cluster of {cluster_size} users...")

    # Filter suitable drivers (capacity constraint)
    suitable_drivers = available_drivers[available_drivers['capacity'] >= cluster_size].copy()

    if suitable_drivers.empty:
        print(f"    ❌ No cab available for cluster of {cluster_size} users")
        # CAPACITY ENHANCEMENT: Try to split cluster to match available driver capacities
        print(f"    🔄 Attempting cluster splitting for capacity match...")
        return split_cluster_for_capacity_match(cluster_users, available_drivers, office_lat, office_lon)

    # Calculate cluster center
    cluster_center = (cluster_users['latitude'].mean(), cluster_users['longitude'].mean())

    # Calculate capacity-optimized composite score for each driver
    best_driver = None
    best_score = float('inf')

    for _, driver in suitable_drivers.iterrows():
        # Calculate distance component
        driver_to_cluster = haversine_distance(
            driver['latitude'], driver['longitude'],
            cluster_center[0], cluster_center[1]
        )

        # Calculate capacity efficiency (Capacity enhancement)
        capacity_waste = driver['capacity'] - cluster_size
        utilization = cluster_size / driver['capacity']

        # Capacity bonus for good utilization
        capacity_bonus = 0
        if utilization >= CONFIG['UTILIZATION_BONUS_THRESHOLD']:
            capacity_bonus = -CONFIG['CAPACITY_PRIORITY_WEIGHT']  # Negative bonus = better score

        # CAPACITY-OPTIMIZED composite score:
        # Balance.py base + Capacity enhancements
        score = (
            driver_to_cluster * CONFIG['DISTANCE_WEIGHT'] +                    # Distance component
            capacity_waste * CONFIG['CAPACITY_WASTE_WEIGHT'] +                 # Waste penalty
            capacity_bonus                                                   # Capacity efficiency bonus
        )

        print(f"      Driver {driver['driver_id']}: dist={driver_to_cluster:.1f}km, waste={capacity_waste}, util={utilization:.2f}, score={score:.2f}")

        if score < best_score:
            best_score = score
            best_driver = driver

    if best_driver is not None:
        utilization = cluster_size / best_driver['capacity']
        print(f"    ✅ Selected Driver {best_driver['driver_id']} (capacity {best_driver['capacity']}, utilization {utilization:.1%})")

    return best_driver


def optimize_geographic_distribution_capacity(routes, office_lat, office_lon, driver_df):
    """
    CAPACITY-OPTIMIZED: Reallocate users between nearby routes for better geographic distribution
    Balance.py approach with capacity constraints and efficiency focus
    """
    print(f"  🗺️  CAPACITY optimization: Improving geographic distribution...")

    if len(routes) < 2:
        return routes

    optimization_iterations = 0
    max_iterations = 3

    while optimization_iterations < max_iterations:
        improvements_made = 0

        for i, route1 in enumerate(routes):
            for j, route2 in enumerate(routes):
                if i >= j:
                    continue

                # Try moving users from route1 to route2 (Capacity-aware)
                improvement = try_user_reallocation_capacity(route1, route2, office_lat, office_lon)
                if improvement > 0:
                    improvements_made += 1

        if improvements_made == 0:
            break

        optimization_iterations += 1
        print(f"    🔄 Optimization iteration {optimization_iterations}: {improvements_made} improvements")

    print(f"  ✅ Geographic optimization complete: {optimization_iterations} iterations")
    return routes


def try_user_reallocation_capacity(route1, route2, office_lat, office_lon):
    """
    CAPACITY-OPTIMIZED: Try moving users between routes for better geographic distribution
    Balance.py logic with capacity constraints
    """
    max_improvement = 0
    best_user_to_move = None

    # Check capacity constraints
    if len(route2['assigned_users']) >= route2['vehicle_type']:
        return 0

    route1_center = calculate_route_center(route1['assigned_users'])
    route2_center = calculate_route_center(route2['assigned_users'])

    for user in route1['assigned_users'][:]:  # Copy to avoid modification during iteration
        user_pos = (user['lat'], user['lng'])

        # Check if user is closer to route2 center
        dist_to_route1 = haversine_distance(user_pos[0], user_pos[1], route1_center[0], route1_center[1])
        dist_to_route2 = haversine_distance(user_pos[0], user_pos[1], route2_center[0], route2_center[1])

        if dist_to_route2 < dist_to_route1 - 0.5:  # 500m threshold
            geographic_improvement = dist_to_route1 - dist_to_route2

            # Capacity efficiency bonus
            route1_util_after = (len(route1['assigned_users']) - 1) / route1['vehicle_type']
            route2_util_after = (len(route2['assigned_users']) + 1) / route2['vehicle_type']

            # Prefer moves that improve both routes' utilization
            capacity_improvement = 0
            if route1_util_after >= route2_util_after:
                capacity_improvement = 0.5

            total_improvement = geographic_improvement + capacity_improvement

            if total_improvement > max_improvement:
                max_improvement = total_improvement
                best_user_to_move = user

    # Perform the move if beneficial
    if best_user_to_move and max_improvement > 0.3:  # Minimum threshold
        route1['assigned_users'].remove(best_user_to_move)
        route2['assigned_users'].append(best_user_to_move)

        # Optimize both routes
        optimize_route_sequence_safe(route1, office_lat, office_lon)
        optimize_route_sequence_safe(route2, office_lat, office_lon)

        return max_improvement

    return 0


def swap_users_for_geographic_separation_capacity(routes):
    """
    CAPACITY-OPTIMIZED: Multi-pass user swapping for better geographic separation
    Balance.py algorithm with capacity constraints
    """
    print(f"  🔄 CAPACITY basic swapping: Improving user geographic distribution...")

    max_swaps = 5
    improvements_per_pass = []

    for swap_pass in range(max_swaps):
        improvements_made = 0

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                route1, route2 = routes[i], routes[j]

                # Try asymmetric swaps (Capacity enhancement)
                swap_improvement = try_asymmetric_swap_capacity(route1, route2)
                if swap_improvement > 0:
                    improvements_made += swap_improvement

        improvements_per_pass.append(improvements_made)

        if improvements_made < 2:  # Stop if minimal improvements
            break

    total_improvements = sum(improvements_per_pass)
    print(f"  ✅ Basic user swapping complete: {total_improvements} total improvements over {len(improvements_per_pass)} passes")

    return routes


def advanced_user_swapping_capacity(routes, office_lat, office_lon):
    """
    CAPACITY-OPTIMIZED: Advanced multi-pass user swapping with swap candidates
    Enhanced balance.py approach with comprehensive optimization
    """
    print(f"  🔄 CAPACITY advanced swapping: Comprehensive user optimization...")

    total_improvements = 0
    max_iterations = 3

    for iteration in range(max_iterations):
        iteration_improvements = 0
        swap_candidates_found = 0

        # Find all potential swap candidates between route pairs
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                route1, route2 = routes[i], routes[j]

                # Skip if both routes are at capacity
                if (len(route1['assigned_users']) >= route1['vehicle_type'] and
                    len(route2['assigned_users']) >= route2['vehicle_type']):
                    continue

                # Find swap candidates
                candidates = find_user_swap_candidates_capacity(route1, route2)
                swap_candidates_found += len(candidates)

                # Execute best swaps
                for candidate in candidates[:2]:  # Limit to top 2 swaps per pair
                    if perform_user_swap_capacity(route1, route2, candidate):
                        iteration_improvements += candidate['improvement']

        # Re-optimize all routes after swaps
        for route in routes:
            if route['assigned_users']:
                optimize_route_sequence_improved(route, office_lat, office_lon)

        total_improvements += iteration_improvements

        print(f"    Advanced swapping iteration {iteration + 1}: {iteration_improvements:.2f} improvements, {swap_candidates_found} candidates")

        if iteration_improvements < 1.0:  # Stop if minimal improvements
            break

    print(f"  ✅ Advanced user swapping complete: {total_improvements:.2f} total improvements")
    return routes


def try_asymmetric_swap_capacity(route1, route2):
    """
    CAPACITY-OPTIMIZED: Try asymmetric user swaps between routes
    Balance.py logic with capacity validation
    """
    best_improvement = 0
    best_swap = None

    # Calculate current route centers
    route1_center = calculate_route_center(route1['assigned_users'])
    route2_center = calculate_route_center(route2['assigned_users'])

    # Try moving users from route1 to route2
    for user1 in route1['assigned_users'][:]:
        if len(route2['assigned_users']) >= route2['vehicle_type']:
            break  # No capacity in route2

        user1_pos = (user1['lat'], user1['lng'])

        # Check if user1 is closer to route2
        dist1_to_r1 = haversine_distance(user1_pos[0], user1_pos[1], route1_center[0], route1_center[1])
        dist1_to_r2 = haversine_distance(user1_pos[0], user1_pos[1], route2_center[0], route2_center[1])

        if dist1_to_r2 < dist1_to_r1 - 0.3:  # 300m improvement threshold
            improvement = dist1_to_r1 - dist1_to_r2

            # Capacity utilization bonus
            r1_util_after = (len(route1['assigned_users']) - 1) / route1['vehicle_type']
            r2_util_after = (len(route2['assigned_users']) + 1) / route2['vehicle_type']

            if r1_util_after >= 0.6 and r2_util_after >= 0.6:
                improvement += 0.5  # Bonus for maintaining good utilization

            if improvement > best_improvement:
                best_improvement = improvement
                best_swap = ('move', user1, None, route1, route2)

    # Perform the best swap if found (More permissive threshold)
    if best_swap and best_improvement > 0.2:
        swap_type, user1, user2, from_route, to_route = best_swap

        if swap_type == 'move':
            from_route['assigned_users'].remove(user1)
            to_route['assigned_users'].append(user1)

            # Optimize both routes
            optimize_route_sequence_safe(from_route, CONFIG['OFFICE_LAT'], CONFIG['OFFICE_LON'])
            optimize_route_sequence_safe(to_route, CONFIG['OFFICE_LAT'], CONFIG['OFFICE_LON'])

            return best_improvement

    return 0


def merge_small_routes_with_nearby_capacity(routes, office_lat, office_lon):
    """
    CAPACITY-OPTIMIZED: Direction-aware merging of small routes with compatible routes
    Enhanced balance.py approach with capacity efficiency focus
    """
    print(f"  🔗 CAPACITY merging: Combining small routes with nearby routes...")

    # Sort routes to prioritize merging small routes first
    route_data = []
    for i, route in enumerate(routes):
        route_data.append({
            'index': i,
            'route': route,
            'user_count': len(route['assigned_users']),
            'capacity': route['vehicle_type'],
            'available_capacity': route['vehicle_type'] - len(route['assigned_users']),
            'utilization': len(route['assigned_users']) / route['vehicle_type']
        })

    route_data.sort(key=lambda x: x['user_count'])

    merged_routes = []
    used_route_indices = set()
    merges_performed = 0

    for i, route1_data in enumerate(route_data):
        if route1_data['index'] in used_route_indices:
            continue

        # Focus on small routes or underutilized routes (Capacity enhancement)
        if (route1_data['user_count'] > CONFIG['SMALL_ROUTE_THRESHOLD'] and
            route1_data['utilization'] >= CONFIG['MIN_CAPACITY_UTILIZATION']):
            merged_routes.append(route1_data['route'])
            used_route_indices.add(route1_data['index'])
            continue

        best_merge_candidate = None
        best_merge_score = float('inf')

        # Look for merge candidates
        for j, route2_data in enumerate(route_data):
            if i == j or route2_data['index'] in used_route_indices:
                continue

            # Check capacity constraint
            combined_users = route1_data['user_count'] + route2_data['user_count']
            max_capacity = max(route1_data['capacity'], route2_data['capacity'])

            if combined_users > max_capacity:
                continue

            # Calculate CAPACITY-OPTIMIZED merge score
            route1_center = calculate_route_center(route1_data['route']['assigned_users'])
            route2_center = calculate_route_center(route2_data['route']['assigned_users'])

            geographic_distance = haversine_distance(
                route1_center[0], route1_center[1],
                route2_center[0], route2_center[1]
            )

            if geographic_distance > CONFIG['MAX_MERGE_DISTANCE_KM']:
                continue

            # Direction compatibility scoring
            compatibility_score = calculate_direction_compatibility_capacity(
                route1_data['route'], route2_data['route'], office_lat, office_lon
            )

            # Capacity efficiency bonus (Capacity enhancement)
            combined_utilization = combined_users / max_capacity
            capacity_bonus = 0
            if combined_utilization >= CONFIG['UTILIZATION_BONUS_THRESHOLD']:
                capacity_bonus = -1.0  # Negative bonus = better score

            # Calculate merge score (lower is better)
            merge_score = (
                geographic_distance * 0.3 +           # Geographic proximity
                (1 - compatibility_score) * 2.0 +     # Direction alignment
                capacity_bonus                         # Capacity efficiency
            )

            # Bonuses for small routes and good utilization
            if route1_data['user_count'] <= 1 or route2_data['user_count'] <= 1:
                merge_score -= 1.0

            if merge_score < best_merge_score and merge_score < CONFIG['MERGE_SCORE_THRESHOLD']:
                best_merge_candidate = route2_data
                best_merge_score = merge_score

        # Perform merge if score is good
        if best_merge_candidate:
            merged_route = perform_route_merge_capacity(
                route1_data['route'], best_merge_candidate['route'], office_lat, office_lon
            )
            if merged_route:
                merged_routes.append(merged_route)
                used_route_indices.add(route1_data['index'])
                used_route_indices.add(best_merge_candidate['index'])
                merges_performed += 1
                print(f"    🔗 Merged routes: {route1_data['user_count']} + {best_merge_candidate['user_count']} = {len(merged_route['assigned_users'])} users")
                continue

        # No merge found, keep original
        merged_routes.append(route1_data['route'])
        used_route_indices.add(route1_data['index'])

    print(f"  ✅ Route merging complete: {merges_performed} merges performed")
    return merged_routes


def calculate_direction_compatibility_capacity(route1, route2, office_lat, office_lon):
    """
    CAPACITY-OPTIMIZED: Calculate direction compatibility between routes
    Enhanced balance.py approach with capacity consideration
    """
    try:
        # Calculate average bearings for both routes
        bearing1 = calculate_average_bearing_improved(route1, office_lat, office_lon)
        bearing2 = calculate_average_bearing_improved(route2, office_lat, office_lon)

        # Calculate bearing difference
        bearing_diff = bearing_difference(bearing1, bearing2)

        # Normalize to 0-1 scale (higher = more compatible)
        compatibility = 1.0 - (bearing_diff / CONFIG['DIRECTION_TOLERANCE_DEGREES'])
        compatibility = max(0.0, min(1.0, compatibility))

        return compatibility

    except Exception as e:
        logger.error(f"Error calculating direction compatibility: {e}")
        return 0.5  # Default to neutral


def perform_route_merge_capacity(route1, route2, office_lat, office_lon):
    """
    CAPACITY-OPTIMIZED: Perform safe merge of two routes
    Enhanced balance.py approach with capacity validation
    """
    try:
        # Choose the route with larger capacity as the base
        base_route = route1 if route1['vehicle_type'] >= route2['vehicle_type'] else route2
        other_route = route2 if route1['vehicle_type'] >= route2['vehicle_type'] else route1

        # Create merged route
        merged_route = base_route.copy()
        merged_route['assigned_users'] = route1['assigned_users'] + route2['assigned_users']

        # CRITICAL: Validate capacity constraint
        if len(merged_route['assigned_users']) > merged_route['vehicle_type']:
            logger.error(f"MERGE ERROR: Combined {len(merged_route['assigned_users'])} users exceeds capacity {merged_route['vehicle_type']}")
            return None

        # Optimize merged route sequence
        merged_route = optimize_route_sequence_safe(merged_route, office_lat, office_lon)

        # Update route metrics
        update_route_metrics_improved(merged_route, office_lat, office_lon)

        return merged_route

    except Exception as e:
        logger.error(f"Error performing route merge: {e}")
        return None


def fill_remaining_seats_with_cluster_check_capacity(routes, unassigned_users_df, office_lat, office_lon):
    """
    CAPACITY-OPTIMIZED: Smart seat filling choosing CLOSEST and BEST-ALIGNED users
    Prioritizes users with minimal detour and best geographic/bearing alignment
    """
    print(f"  🪑 CAPACITY seat filling: Filling remaining seats with closest user priority...")

    filled_user_ids = set()
    seats_filled = 0

    # Calculate route centers and bearings for intelligent matching
    route_centers = []
    route_bearings = []
    for route in routes:
        if route['assigned_users']:
            center = calculate_route_center(route['assigned_users'])
            route_centers.append(center)
            # Calculate route bearing
            bearing = calculate_average_bearing_improved(route, office_lat, office_lon)
            route_bearings.append(bearing)
        else:
            route_centers.append((route['latitude'], route['longitude']))
            route_bearings.append(0)

    # Sort unassigned users: single users from sparse areas first (Balance.py priority)
    unassigned_users_df = unassigned_users_df.copy()
    if 'geo_cluster' in unassigned_users_df.columns:
        unassigned_users_df['cluster_size'] = unassigned_users_df.groupby('geo_cluster')['user_id'].transform('count')
        unassigned_users_df = unassigned_users_df.sort_values('cluster_size', ascending=True)

    for route_idx, route in enumerate(routes):
        available_seats = route['vehicle_type'] - len(route['assigned_users'])
        if available_seats <= 0:
            continue

        driver_pos = (route['latitude'], route['longitude'])
        office_pos = (office_lat, office_lon)
        route_center = route_centers[route_idx]
        route_bearing = route_bearings[route_idx]

        print(f"    🚗 Route {route['driver_id']}: {available_seats} seats available")

        # Find ALL candidates and score them comprehensively
        candidates = []
        for _, user in unassigned_users_df.iterrows():
            if user['user_id'] in filled_user_ids:
                continue

            user_pos = (user['latitude'], user['longitude'])
            
            # Calculate distance to route center
            dist_to_route = haversine_distance(user_pos[0], user_pos[1], route_center[0], route_center[1])

            # Check cluster ownership (but be permissive)
            belongs_to_other_cluster = False
            min_dist_to_other = float('inf')
            for other_idx, other_center in enumerate(route_centers):
                if other_idx == route_idx:
                    continue
                dist_to_other = haversine_distance(user_pos[0], user_pos[1], other_center[0], other_center[1])
                min_dist_to_other = min(min_dist_to_other, dist_to_other)
                # Only prevent if user is significantly closer to another route
                if dist_to_other < dist_to_route - 0.3:  # 300m threshold
                    belongs_to_other_cluster = True
                    break

            # Override cluster ownership if route has capacity and user is reasonably close
            if belongs_to_other_cluster and (available_seats > 2 or dist_to_route < 1.5):
                belongs_to_other_cluster = False

            if belongs_to_other_cluster:
                continue

            # Check if user is on the way to office
            on_way, detour_distance = is_user_on_way_to_office_capacity(
                user_pos, driver_pos, office_pos, route['assigned_users']
            )

            if not on_way or detour_distance > CONFIG['MAX_ON_ROUTE_DETOUR_KM']:
                continue

            # Calculate bearing alignment score
            user_bearing = calculate_bearing(route_center[0], route_center[1], user_pos[0], user_pos[1])
            bearing_diff = abs(bearing_difference(user_bearing, route_bearing))
            bearing_alignment_score = 1.0 - (bearing_diff / 180.0)  # 0-1 scale

            # Calculate capacity efficiency
            new_utilization = (len(route['assigned_users']) + 1) / route['vehicle_type']
            
            # COMPREHENSIVE SCORING: Prioritize closest + best-aligned users
            # Lower score = better candidate
            composite_score = (
                dist_to_route * 0.4 +           # Geographic proximity (40%)
                detour_distance * 0.3 +         # Route efficiency (30%)
                (1 - bearing_alignment_score) * 0.2 +  # Bearing alignment (20%)
                (1 - new_utilization) * 0.1     # Capacity utilization (10%)
            )

            candidates.append({
                'user': user,
                'composite_score': composite_score,
                'dist_to_route': dist_to_route,
                'detour_distance': detour_distance,
                'bearing_alignment': bearing_alignment_score,
                'utilization': new_utilization
            })

        # Sort by composite score - BEST candidates first
        candidates.sort(key=lambda x: x['composite_score'])

        # Fill seats with BEST candidates
        route_seats_filled = 0
        for candidate in candidates:
            if route_seats_filled >= available_seats:
                break

            user = candidate['user']
            
            # Add user to route
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
            route_seats_filled += 1
            seats_filled += 1

            print(f"      ✅ Added user {user['user_id']} (score: {candidate['composite_score']:.2f}, "
                  f"dist: {candidate['dist_to_route']:.1f}km, bearing: {candidate['bearing_alignment']:.2f})")

        # Re-optimize route if seats were filled
        if route_seats_filled > 0:
            route = optimize_route_sequence_safe(route, office_lat, office_lon)
            update_route_metrics_safe(route, office_lat, office_lon)
            print(f"    🪑 Filled {route_seats_filled} seats in route {route['driver_id']} with BEST-FIT users")

    print(f"  ✅ Seat filling complete: {seats_filled} total seats filled (closest + best-aligned priority)")
    return routes, filled_user_ids


def is_user_on_way_to_office_capacity(user_pos, driver_pos, office_pos, current_route_users):
    """
    CAPACITY-OPTIMIZED: Check if user is on the way to office
    Enhanced balance.py approach with capacity route consideration
    """
    try:
        # Calculate detour if this user is added
        route_with_user = current_route_users + [{'lat': user_pos[0], 'lng': user_pos[1]}]

        # Calculate route distance with user
        route_distance_with_user = calculate_total_route_distance_capacity(route_with_user, driver_pos, office_pos)

        # Calculate original route distance
        original_route_distance = calculate_total_route_distance_capacity(current_route_users, driver_pos, office_pos)

        # Calculate detour
        detour_distance = route_distance_with_user - original_route_distance

        # Check if detour is acceptable
        on_way = detour_distance <= CONFIG['MAX_ON_ROUTE_DETOUR_KM']

        return on_way, detour_distance

    except Exception as e:
        logger.error(f"Error checking if user is on way: {e}")
        return False, float('inf')


def calculate_total_route_distance_capacity(route_users, driver_pos, office_pos):
    """
    CAPACITY-OPTIMIZED: Calculate total route distance
    Simplified balance.py approach for detour calculations
    """
    if not route_users:
        return haversine_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    total_distance = 0
    current_pos = driver_pos

    # Driver to first pickup
    first_user = route_users[0]
    user_pos = (first_user['lat'], first_user['lng'])
    total_distance += haversine_distance(current_pos[0], current_pos[1], user_pos[0], user_pos[1])

    # Between pickups
    for i in range(len(route_users) - 1):
        current_user = route_users[i]
        next_user = route_users[i + 1]
        current_pos = (current_user['lat'], current_user['lng'])
        next_pos = (next_user['lat'], next_user['lng'])
        total_distance += haversine_distance(current_pos[0], current_pos[1], next_pos[0], next_pos[1])

    # Last pickup to office
    last_user = route_users[-1]
    last_pos = (last_user['lat'], last_user['lng'])
    total_distance += haversine_distance(last_pos[0], last_pos[1], office_pos[0], office_pos[1])

    return total_distance


def calculate_route_center(users):
    """
    Calculate the geographic center of a route's users
    Balance.py utility function
    """
    if not users:
        return None

    # Support both lat/lng and latitude/longitude formats
    center_lat = 0
    center_lon = 0
    valid_users = 0

    for user in users:
        try:
            lat = float(user.get('lat', user.get('latitude', 0)))
            lon = float(user.get('lng', user.get('longitude', 0)))

            if lat != 0 or lon != 0:  # Skip invalid coordinates
                center_lat += lat
                center_lon += lon
                valid_users += 1
        except (ValueError, TypeError):
            continue

    if valid_users == 0:
        return None

    center_lat /= valid_users
    center_lon /= valid_users

    return (center_lat, center_lon)


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing from point 1 to point 2 in degrees
    Returns bearing in range [0, 360) where 0 = North, 90 = East, etc.
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Calculate bearing
    delta_lon = lon2_rad - lon1_rad
    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))

    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)

    # Convert to 0-360 range
    return (bearing_deg + 360) % 360


def create_route_from_cluster_capacity(cluster_users, driver, office_lat, office_lon):
    """
    CAPACITY-OPTIMIZED: Create route from cluster with capacity validation
    Balance.py structure with capacity safety checks
    """
    try:
        # CRITICAL: Validate capacity constraint
        cluster_size = len(cluster_users)
        # Support both capacity and vehicle_capacity field names
        driver_capacity = int(driver.get('capacity', driver.get('vehicle_capacity', 0)))

        if cluster_size > driver_capacity:
            logger.error(f"ROUTE ERROR: Cluster {cluster_size} users exceeds driver capacity {driver_capacity}")
            return None

        route = {
            'driver_id': str(driver['driver_id']),
            'vehicle_id': str(driver.get('vehicle_id', '')),
            'vehicle_type': driver_capacity,  # Use validated capacity
            'latitude': float(driver['latitude']),
            'longitude': float(driver['longitude']),
            'assigned_users': []
        }

        # Add all users from the cluster
        for _, user in cluster_users.iterrows():
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

        # Optimize sequence
        route = optimize_route_sequence_safe(route, office_lat, office_lon)
        update_route_metrics_safe(route, office_lat, office_lon)

        # FINAL SAFETY CHECK
        if len(route['assigned_users']) > route['vehicle_type']:
            logger.error(f"CRITICAL ERROR: Route created with {len(route['assigned_users'])} users but capacity is {route['vehicle_type']}")
            return None

        return route

    except Exception as e:
        logger.error(f"Error creating route from cluster: {e}")
        return None


# =============================================================================
# ADVANCED CAPACITY OPTIMIZATION FUNCTIONS (Balance.py Enhanced)
# =============================================================================

def identify_nearby_route_groups_capacity(routes):
    """
    CAPACITY-OPTIMIZED: Group nearby routes for batch optimization
    Advanced balance.py approach with capacity-aware grouping
    """
    if len(routes) < 2:
        return [routes]

    route_centers = []
    for route in routes:
        center = calculate_route_center(route['assigned_users'])
        if center is None:
            center = (route['latitude'], route['longitude'])
        route_centers.append(center)

    # Build adjacency matrix based on proximity
    route_groups = []
    used_indices = set()

    for i, route in enumerate(routes):
        if i in used_indices:
            continue

        current_group = [route]
        used_indices.add(i)

        # Find all nearby routes
        for j in range(i + 1, len(routes)):
            if j in used_indices:
                continue

            # Check if routes are close enough
            distance = haversine_distance(
                route_centers[i][0], route_centers[i][1],
                route_centers[j][0], route_centers[j][1]
            )

            if distance <= CONFIG['MAX_MERGE_DISTANCE_KM'] * 1.5:  # Slightly larger radius for grouping
                # Check capacity compatibility
                combined_users = len(route['assigned_users']) + len(routes[j]['assigned_users'])
                max_capacity = max(route['vehicle_type'], routes[j]['vehicle_type'])

                if combined_users <= max_capacity:
                    current_group.append(routes[j])
                    used_indices.add(j)

        route_groups.append(current_group)

    return route_groups


def reorganize_users_geographically_capacity(route_groups, office_lat, office_lon):
    """
    CAPACITY-OPTIMIZED: Advanced user reorganization between nearby routes
    Balance.py approach with capacity efficiency focus
    """
    improvements_made = 0

    for group in route_groups:
        if len(group) < 2:
            continue

        # Calculate all user centers and route centers
        all_users = []
        route_user_indices = []

        for route_idx, route in enumerate(group):
            for user_idx, user in enumerate(route['assigned_users']):
                all_users.append(user)
                route_user_indices.append((route_idx, user_idx))

        # Try to improve geographic assignment
        group_improvements = 0
        max_iterations = min(3, len(group))

        for _ in range(max_iterations):
            iteration_improvements = 0

            for user_idx in range(len(all_users)):
                user = all_users[user_idx]
                current_route_idx, current_user_idx = route_user_indices[user_idx]
                current_route = group[current_route_idx]

                user_pos = (user['lat'], user['lng'])

                # Find best route for this user
                best_route_idx = current_route_idx
                best_improvement = 0

                for test_route_idx, test_route in enumerate(group):
                    if test_route_idx == current_route_idx:
                        continue

                    # Check capacity constraint
                    if len(test_route['assigned_users']) >= test_route['vehicle_type']:
                        continue

                    test_center = calculate_route_center(test_route['assigned_users'])
                    current_center = calculate_route_center(current_route['assigned_users'])

                    dist_to_current = haversine_distance(user_pos[0], user_pos[1], current_center[0], current_center[1])
                    dist_to_test = haversine_distance(user_pos[0], user_pos[1], test_center[0], test_center[1])

                    if dist_to_test < dist_to_current - 0.3:  # 300m improvement threshold
                        improvement = dist_to_current - dist_to_test

                        # Capacity efficiency bonus
                        current_util_after = (len(current_route['assigned_users']) - 1) / current_route['vehicle_type']
                        test_util_after = (len(test_route['assigned_users']) + 1) / test_route['vehicle_type']

                        capacity_bonus = 0
                        if current_util_after >= CONFIG['MIN_CAPACITY_UTILIZATION'] and test_util_after >= CONFIG['MIN_CAPACITY_UTILIZATION']:
                            capacity_bonus = 0.5

                        total_improvement = improvement + capacity_bonus

                        if total_improvement > best_improvement:
                            best_improvement = total_improvement
                            best_route_idx = test_route_idx

                # Perform the move if beneficial (More permissive threshold)
                if best_improvement > 0.3 and best_route_idx != current_route_idx:
                    # Move user
                    group[current_route_idx]['assigned_users'].pop(current_user_idx)
                    group[best_route_idx]['assigned_users'].append(user)
                    route_user_indices[user_idx] = (best_route_idx, len(group[best_route_idx]['assigned_users']) - 1)
                    iteration_improvements += 1

            group_improvements += iteration_improvements

            if iteration_improvements == 0:
                break

        improvements_made += group_improvements

        # Re-optimize all routes in the group
        for route in group:
            if route['assigned_users']:
                optimize_route_sequence_improved(route, office_lat, office_lon)

    return improvements_made


def find_user_swap_candidates_capacity(route1, route2):
    """
    CAPACITY-OPTIMIZED: Find optimal swap candidates between routes
    Advanced balance.py approach with capacity constraints
    """
    candidates = []

    # Check capacity constraints for swapping
    if len(route1['assigned_users']) >= route1['vehicle_type'] and len(route2['assigned_users']) >= route2['vehicle_type']:
        return candidates

    route1_center = calculate_route_center(route1['assigned_users'])
    route2_center = calculate_route_center(route2['assigned_users'])

    # Try swapping users from route1 to route2
    for i, user1 in enumerate(route1['assigned_users']):
        if len(route2['assigned_users']) >= route2['vehicle_type']:
            break

        user1_pos = (user1['lat'], user1['lng'])
        dist1_to_r1 = haversine_distance(user1_pos[0], user1_pos[1], route1_center[0], route1_center[1])
        dist1_to_r2 = haversine_distance(user1_pos[0], user1_pos[1], route2_center[0], route2_center[1])

        if dist1_to_r2 < dist1_to_r1 - 0.3:  # Improvement threshold
            # Try finding user2 to swap back
            for j, user2 in enumerate(route2['assigned_users']):
                if len(route1['assigned_users']) >= route1['vehicle_type']:
                    break

                user2_pos = (user2['lat'], user2['lng'])
                dist2_to_r2 = haversine_distance(user2_pos[0], user2_pos[1], route2_center[0], route2_center[1])
                dist2_to_r1 = haversine_distance(user2_pos[0], user2_pos[1], route1_center[0], route1_center[1])

                if dist2_to_r1 < dist2_to_r2 - 0.3:  # Mutual improvement
                    # Calculate total improvement
                    total_improvement = (dist1_to_r1 - dist1_to_r2) + (dist2_to_r2 - dist2_to_r1)

                    # Capacity efficiency bonus
                    r1_util_after = len(route1['assigned_users']) / route1['vehicle_type']  # Users stay same in swap
                    r2_util_after = len(route2['assigned_users']) / route2['vehicle_type']

                    capacity_bonus = 0
                    if r1_util_after >= CONFIG['MIN_CAPACITY_UTILIZATION'] and r2_util_after >= CONFIG['MIN_CAPACITY_UTILIZATION']:
                        capacity_bonus = 0.3

                    total_improvement += capacity_bonus

                    if total_improvement > 0.3:  # More permissive
                        candidates.append({
                            'type': 'swap',
                            'user1': (i, user1),
                            'user2': (j, user2),
                            'improvement': total_improvement
                        })

    # Also consider asymmetric moves
    for i, user1 in enumerate(route1['assigned_users']):
        if len(route2['assigned_users']) >= route2['vehicle_type']:
            break

        user1_pos = (user1['lat'], user1['lng'])
        dist1_to_r1 = haversine_distance(user1_pos[0], user1_pos[1], route1_center[0], route1_center[1])
        dist1_to_r2 = haversine_distance(user1_pos[0], user1_pos[1], route2_center[0], route2_center[1])

        if dist1_to_r2 < dist1_to_r1 - 0.5:  # Higher threshold for asymmetric move
            improvement = dist1_to_r1 - dist1_to_r2

            # Capacity utilization bonus for move
            r1_util_after = (len(route1['assigned_users']) - 1) / route1['vehicle_type']
            r2_util_after = (len(route2['assigned_users']) + 1) / route2['vehicle_type']

            capacity_bonus = 0
            if r1_util_after >= CONFIG['MIN_CAPACITY_UTILIZATION'] and r2_util_after >= CONFIG['MIN_CAPACITY_UTILIZATION']:
                capacity_bonus = 0.4

            total_improvement = improvement + capacity_bonus

            if total_improvement > 0.4:  # More permissive
                candidates.append({
                    'type': 'asymmetric_move',
                    'from_route': 1,
                    'to_route': 2,
                    'user': (i, user1),
                    'improvement': total_improvement
                })

    # Sort candidates by improvement
    candidates.sort(key=lambda x: x['improvement'], reverse=True)
    return candidates


def perform_user_swap_capacity(route1, route2, swap_candidate):
    """
    CAPACITY-OPTIMIZED: Execute user swap with capacity validation
    Balance.py approach with safety checks
    """
    try:
        if swap_candidate['type'] == 'swap':
            i, user1 = swap_candidate['user1']
            j, user2 = swap_candidate['user2']

            # Validate indices
            if i >= len(route1['assigned_users']) or j >= len(route2['assigned_users']):
                return False

            # Perform swap
            route1['assigned_users'][i] = user2
            route2['assigned_users'][j] = user1

            # Re-optimize both routes
            optimize_route_sequence_safe(route1, CONFIG['OFFICE_LAT'], CONFIG['OFFICE_LON'])
            optimize_route_sequence_safe(route2, CONFIG['OFFICE_LAT'], CONFIG['OFFICE_LON'])

            return True

        elif swap_candidate['type'] == 'asymmetric_move':
            i, moved_user = swap_candidate['user']
            from_route = route1 if swap_candidate['from_route'] == 1 else route2
            to_route = route2 if swap_candidate['from_route'] == 1 else route1

            # Validate capacity
            if len(to_route['assigned_users']) >= to_route['vehicle_type']:
                return False

            # Perform move
            if i < len(from_route['assigned_users']):
                moved_user = from_route['assigned_users'].pop(i)
                to_route['assigned_users'].append(moved_user)

                # Re-optimize both routes
                optimize_route_sequence_safe(from_route, CONFIG['OFFICE_LAT'], CONFIG['OFFICE_LON'])
                optimize_route_sequence_safe(to_route, CONFIG['OFFICE_LAT'], CONFIG['OFFICE_LON'])

                return True

        return False

    except Exception as e:
        logger.error(f"Error performing user swap: {e}")
        return False


def consolidate_very_close_routes_capacity(routes, office_lat, office_lon):
    """
    CAPACITY-OPTIMIZED: Merge very close routes for capacity efficiency
    Balance.py approach with aggressive consolidation
    """
    if len(routes) < 2:
        return routes

    route_centers = []
    for route in routes:
        center = calculate_route_center(route['assigned_users'])
        if center is None:
            center = (route['latitude'], route['longitude'])
        route_centers.append(center)

    merged_routes = []
    used_indices = set()
    consolidations = 0

    for i, route1 in enumerate(routes):
        if i in used_indices:
            continue

        best_merge = None
        best_distance = float('inf')

        # Find closest route that can be merged
        for j, route2 in enumerate(routes):
            if i >= j or j in used_indices:
                continue

            # Check capacity constraint
            combined_users = len(route1['assigned_users']) + len(route2['assigned_users'])
            max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])

            if combined_users > max_capacity:
                continue

            # Calculate distance
            distance = haversine_distance(
                route_centers[i][0], route_centers[i][1],
                route_centers[j][0], route_centers[j][1]
            )

            # Very close routes threshold (more aggressive)
            if distance < CONFIG['MAX_MERGE_DISTANCE_KM'] * 0.5:  # Half the normal threshold
                if distance < best_distance:
                    best_distance = distance
                    best_merge = j

        # Perform merge if very close
        if best_merge is not None:
            route2 = routes[best_merge]

            # Choose route with larger capacity
            base_route = route1 if route1['vehicle_type'] >= route2['vehicle_type'] else route2

            merged_route = base_route.copy()
            merged_route['assigned_users'] = route1['assigned_users'] + route2['assigned_users']

            # Validate capacity
            if len(merged_route['assigned_users']) <= merged_route['vehicle_type']:
                merged_route = optimize_route_sequence_safe(merged_route, office_lat, office_lon)
                merged_routes.append(merged_route)
                used_indices.add(i)
                used_indices.add(best_merge)
                consolidations += 1
                print(f"    🔗 Consolidated very close routes: {len(route1['assigned_users'])} + {len(route2['assigned_users'])} = {len(merged_route['assigned_users'])} users")
            else:
                # Fallback: keep original routes
                merged_routes.append(route1)
                used_indices.add(i)
        else:
            merged_routes.append(route1)
            used_indices.add(i)

    print(f"  ✅ Very close route consolidation complete: {consolidations} consolidations")
    return merged_routes


def apply_balance_style_hard_consolidation(routes, office_lat, office_lon):
    """
    BALANCE.PY STYLE: Hard consolidation for users within 0.5km MUST be in same route
    This is the key balance.py mechanism for maximizing assignment
    """
    print(f"    🎯 Balance.py-style hard consolidation: Users within 0.5km must be together...")

    if len(routes) < 2:
        return routes

    consolidation_threshold = 0.5  # km - Balance.py threshold
    consolidations = 0
    routes_changed = True

    # Multiple passes to catch all consolidations
    while routes_changed:
        routes_changed = False

        # Find users across different routes that are very close to each other
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                route1 = routes[i]
                route2 = routes[j]

                # Skip if either route is now invalid
                if not route1['assigned_users'] or not route2['assigned_users']:
                    continue

                combined_users = len(route1['assigned_users']) + len(route2['assigned_users'])
                max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])

                # Only consider if combined users fit in one vehicle
                if combined_users > max_capacity:
                    continue

                # Check if any users between routes are within consolidation threshold
                should_consolidate = False
                for user1 in route1['assigned_users']:
                    for user2 in route2['assigned_users']:
                        # Support both lat/lng and latitude/longitude formats
                        lat1 = user1.get('latitude', user1.get('lat', 0))
                        lon1 = user1.get('longitude', user1.get('lng', 0))
                        lat2 = user2.get('latitude', user2.get('lat', 0))
                        lon2 = user2.get('longitude', user2.get('lng', 0))

                        distance = haversine_distance(lat1, lon1, lat2, lon2)

                        if distance <= consolidation_threshold:
                            should_consolidate = True
                            print(f"      🔗 Found users {distance:.2f}km apart - consolidating routes")
                            break

                    if should_consolidate:
                        break

                if should_consolidate:
                    # Perform consolidation
                    base_route = route1 if route1['vehicle_type'] >= route2['vehicle_type'] else route2
                    other_route = route2 if base_route == route1 else route1

                    # Merge users
                    merged_route = base_route.copy()
                    merged_route['assigned_users'] = route1['assigned_users'] + route2['assigned_users']

                    # Re-optimize the merged route
                    if merged_route['assigned_users']:
                        optimize_route_sequence_improved(merged_route, office_lat, office_lon)
                        update_route_metrics_improved(merged_route, office_lat, office_lon)

                    # Replace routes
                    routes = [r for k, r in enumerate(routes) if k != i and k != j]
                    routes.append(merged_route)

                    consolidations += 1
                    routes_changed = True
                    break

            if routes_changed:
                break

    print(f"    ✅ Balance.py-style consolidation: {consolidations} consolidations completed")
    return routes


def apply_final_user_fallback_capacity(routes, unassigned_users_df, available_drivers, office_lat, office_lon):
    """
    BALANCE.PY STYLE: Ultimate fallback to maximize user assignment
    This is the final safety net - assign users even if it means breaking some optimization rules
    """
    print(f"    🚨 Ultimate fallback: {len(unassigned_users_df)} users need assignment")

    if unassigned_users_df.empty:
        return routes, unassigned_users_df

    if available_drivers is None or available_drivers.empty:
        print(f"    ❌ No available drivers for fallback assignment")
        return routes, unassigned_users_df

    # Get currently used driver IDs to exclude them
    used_driver_ids = {route['driver_id'] for route in routes}
    fallback_drivers = available_drivers[~available_drivers['driver_id'].isin(used_driver_ids)]

    if fallback_drivers.empty:
        print(f"    ❌ No remaining drivers for fallback assignment")
        return routes, unassigned_users_df

    fallback_assignments = 0
    remaining_users = unassigned_users_df.copy()

    # BALANCE.PY STYLE: Ultra-permissive assignment - prioritize coverage over optimization
    for _, user in remaining_users.iterrows():
        if fallback_drivers.empty:
            break

        user_id = user['user_id']
        user_lat = user['latitude']
        user_lon = user['longitude']

        # Balance.py approach: Find FIRST available driver with ANY capacity
        # No complex scoring, no distance optimization - just assign!
        best_driver = None

        for _, driver in fallback_drivers.iterrows():
            if driver['capacity'] > 0:  # Just need ANY capacity
                best_driver = driver
                break  # Take the first available driver

        if best_driver is not None:
            # Create individual route for this user
            single_user_df = pd.DataFrame([user])
            route = create_route_from_cluster_capacity(
                single_user_df, best_driver, office_lat, office_lon
            )

            if route:
                routes.append(route)
                fallback_assignments += 1

                # Remove this driver from available pool
                fallback_drivers = fallback_drivers[fallback_drivers['driver_id'] != best_driver['driver_id']]

                print(f"      ✅ BALANCE.PY style: Assigned user {user_id} to driver {best_driver['driver_id']}")
            else:
                print(f"      ❌ Failed to create route for user {user_id}")
        else:
            print(f"      ❌ No driver available for user {user_id}")

    # Update remaining unassigned users
    assigned_user_ids = {route['driver_id'] for route in routes}
    # Actually we need to track user IDs, not driver IDs
    all_assigned_user_ids = set()
    for route in routes:
        for user in route['assigned_users']:
            user_id = user.get('user_id', user.get('id', ''))
            if user_id:
                all_assigned_user_ids.add(str(user_id))

    still_unassigned = unassigned_users_df[~unassigned_users_df['user_id'].isin(all_assigned_user_ids)]

    print(f"    📊 Ultimate fallback results: {fallback_assignments} additional users assigned, {len(still_unassigned)} still unassigned")

    return routes, still_unassigned


def apply_aggressive_capacity_merging_capacity(routes, unassigned_users_df, office_lat, office_lon):
    """
    BALANCE.PY STYLE: Aggressive capacity-based merging for 100% assignment
    - Merge directionally similar routes regardless of current user count
    - Fill unassigned users into any route going in their direction with capacity
    - Prioritize user assignment over route separation
    """
    print(f"    🔥 Starting aggressive capacity merging: {len(routes)} routes, {len(unassigned_users_df)} unassigned users")

    if len(routes) < 2 and unassigned_users_df.empty:
        return routes, unassigned_users_df

    # STEP 1: Aggressive route merging by direction and capacity
    print(f"      🔗 Phase 1: Aggressive route merging by direction...")
    routes = merge_routes_by_direction_and_capacity(routes, office_lat, office_lon)

    # STEP 2: Aggressive unassigned user filling by direction
    if not unassigned_users_df.empty:
        print(f"      🧩 Phase 2: Direction-based user filling...")
        routes, remaining_unassigned = fill_users_by_direction_capacity(
            routes, unassigned_users_df, office_lat, office_lon
        )
    else:
        remaining_unassigned = unassigned_users_df

    # STEP 3: Final capacity optimization - combine any routes that can fit
    print(f"      ⚡ Phase 3: Final capacity optimization...")
    routes = apply_final_capacity_optimization(routes, office_lat, office_lon)

    print(f"    ✅ Aggressive merging complete: {len(routes)} routes, {len(remaining_unassigned)} unassigned users remaining")
    return routes, remaining_unassigned


def merge_routes_by_direction_and_capacity(routes, office_lat, office_lon):
    """
    AGGRESSIVE MERGE: Combine routes going in same direction if capacity allows
    Much more permissive than regular merging
    """
    if len(routes) < 2:
        return routes

    merged_routes = []
    used_indices = set()
    office_pos = (office_lat, office_lon)

    # Calculate route directions and centers
    route_data = []
    for i, route in enumerate(routes):
        if not route['assigned_users']:
            continue

        center = calculate_route_center(route['assigned_users'])
        if center:
            # Calculate bearing from route center to office
            bearing = calculate_bearing(center[0], center[1], office_lat, office_lon)
            route_data.append({
                'index': i,
                'route': route,
                'center': center,
                'bearing': bearing,
                'user_count': len(route['assigned_users']),
                'capacity': route['vehicle_type']
            })

    # Try to merge routes aggressively
    for i, route1_info in enumerate(route_data):
        if route1_info['index'] in used_indices:
            continue

        # Look for merge candidates
        merge_candidates = []
        for j, route2_info in enumerate(route_data):
            if i >= j or route2_info['index'] in used_indices:
                continue

            # Check if routes have compatible directions (very permissive - 60 degree tolerance)
            bearing_diff = abs(route1_info['bearing'] - route2_info['bearing'])
            bearing_diff = min(bearing_diff, 360 - bearing_diff)

            # Check distance between route centers (very permissive - 5km)
            distance = haversine_distance(
                route1_info['center'][0], route1_info['center'][1],
                route2_info['center'][0], route2_info['center'][1]
            )

            # Check if combined users fit in larger vehicle
            total_users = route1_info['user_count'] + route2_info['user_count']
            max_capacity = max(route1_info['capacity'], route2_info['capacity'])

            # CRITICAL: Office-relative quadrant validation to prevent extreme geographic crossing
            # Get office-relative positions for both routes
            route1_relative_x = route1_info['center'][1] - office_lon
            route2_relative_x = route2_info['center'][1] - office_lon

            # Determine if routes are on same side of office (left/right)
            route1_on_left = route1_relative_x < 0
            route2_on_left = route2_relative_x < 0

            # STRICT MERGING CRITERIA: Prevent extreme left-right crossing
            # 1. Routes must be on same side of office OR both very close to office centerline
            same_side = route1_on_left == route2_on_left
            close_to_centerline = abs(route1_relative_x) < 0.02 and abs(route2_relative_x) < 0.02  # Within ~2km

            # 2. Stricter bearing requirement (35° instead of 60°)
            compatible_direction = bearing_diff < 35

            # 3. Stricter distance requirement (3km instead of 5km)
            reasonable_distance = distance < 3.0

            # 4. Final validation: All conditions must be met to prevent extreme crossing
            geographic_coherence = same_side or close_to_centerline

            if (total_users <= max_capacity and  # Fits in larger vehicle
                geographic_coherence and         # Same side of office or near centerline
                compatible_direction and         # Stricter direction compatibility
                reasonable_distance):           # Closer distance requirement

                merge_candidates.append({
                    'route2_info': route2_info,
                    'total_users': total_users,
                    'bearing_diff': bearing_diff,
                    'distance': distance,
                    'efficiency': total_users / max_capacity  # Higher is better
                })

        # Sort by efficiency and merge the best candidate
        if merge_candidates:
            merge_candidates.sort(key=lambda x: x['efficiency'], reverse=True)
            best_merge = merge_candidates[0]

            # Perform the merge
            route1_info['route']['assigned_users'].extend(
                best_merge['route2_info']['route']['assigned_users']
            )

            # Use the larger vehicle
            larger_route = route1_info if route1_info['capacity'] >= best_merge['route2_info']['capacity'] else best_merge['route2_info']
            route1_info['route']['vehicle_type'] = larger_route['capacity']

            # Re-optimize the merged route
            if route1_info['route']['assigned_users']:
                optimize_route_sequence_improved(route1_info['route'], office_lat, office_lon)
                update_route_metrics_improved(route1_info['route'], office_lat, office_lon)

            used_indices.add(route1_info['index'])
            used_indices.add(best_merge['route2_info']['index'])
            merged_routes.append(route1_info['route'])

            print(f"      🔗 Aggressively merged routes {route1_info['index']}+{best_merge['route2_info']['index']}: "
                  f"{best_merge['total_users']} users, bearing diff: {best_merge['bearing_diff']:.1f}°")

    # Add unmerged routes
    for route_info in route_data:
        if route_info['index'] not in used_indices:
            merged_routes.append(route_info['route'])

    return merged_routes


def fill_users_by_direction_capacity(routes, unassigned_users_df, office_lat, office_lon):
    """
    AGGRESSIVE FILLING: Put unassigned users into any route going in their direction with capacity
    """
    if unassigned_users_df.empty:
        return routes, unassigned_users_df

    office_pos = (office_lat, office_lon)
    filled_user_ids = set()

    # Calculate route directions and available capacity
    route_data = []
    for route in routes:
        if not route['assigned_users']:
            continue

        center = calculate_route_center(route['assigned_users'])
        if center:
            bearing = calculate_bearing(center[0], center[1], office_lat, office_lon)
            route_data.append({
                'route': route,
                'center': center,
                'bearing': bearing,
                'available_capacity': route['vehicle_type'] - len(route['assigned_users'])
            })

    # For each unassigned user, try to find a route going in their direction
    filled_routes = []
    for _, route_info in enumerate(route_data):
        if route_info['available_capacity'] <= 0:
            filled_routes.append(route_info['route'])
            continue

        users_to_add = []
        for _, user in unassigned_users_df.iterrows():
            if str(user['user_id']) in filled_user_ids:
                continue

            user_pos = (user['latitude'], user['longitude'])

            # Calculate user bearing to office
            user_bearing = calculate_bearing(user_pos[0], user_pos[1], office_lat, office_lon)

            # Check if user is going in same direction as route (much stricter - 25 degree tolerance)
            bearing_diff = abs(route_info['bearing'] - user_bearing)
            bearing_diff = min(bearing_diff, 360 - bearing_diff)

            # Check distance from route center (much stricter - 3km instead of 8km)
            distance = haversine_distance(
                route_info['center'][0], route_info['center'][1],
                user_pos[0], user_pos[1]
            )

            # CRITICAL: Office-relative quadrant validation to prevent extreme left-right crossing
            # Get office-relative positions for route and user
            route_center = route_info['center']
            route_relative_x = route_center[1] - office_lon  # Longitude difference from office
            user_relative_x = user_pos[1] - office_lon       # Longitude difference from office

            # Determine if route and user are on same side of office (left/right)
            route_on_left = route_relative_x < 0
            user_on_left = user_relative_x < 0

            # STRICT CRITERIA: Prevent extreme geographic crossing
            # 1. Same side of office (left/right) OR very close to office centerline
            same_side = route_on_left == user_on_left
            close_to_centerline = abs(route_relative_x) < 0.02 and abs(user_relative_x) < 0.02  # Within ~2km of office longitude

            # 2. Much stricter bearing requirement (25° instead of 45°)
            good_directional_alignment = bearing_diff < 25

            # 3. Much stricter distance requirement (3km instead of 8km)
            reasonable_distance = distance < 3.0

            # 4. Final validation: All conditions must be met to prevent extreme crossing
            geographic_coherence = same_side or close_to_centerline

            if geographic_coherence and good_directional_alignment and reasonable_distance:
                users_to_add.append(user)
                filled_user_ids.add(str(user['user_id']))

                if len(users_to_add) >= route_info['available_capacity']:
                    break

        # Add users to route - preserve complete user data
        if users_to_add:
            for user in users_to_add:
                # Preserve the complete user object from the dataframe
                user_dict = user.to_dict() if hasattr(user, 'to_dict') else dict(user)

                # Ensure essential fields are present
                complete_user = {
                    'user_id': str(user_dict.get('user_id', '')),
                    'lat': float(user_dict.get('latitude', user_dict.get('lat', 0))),
                    'lng': float(user_dict.get('longitude', user_dict.get('lng', 0))),
                    # Preserve all original user data
                    'first_name': user_dict.get('first_name', ''),
                    'last_name': user_dict.get('last_name', ''),
                    'email': user_dict.get('email', ''),
                    'phone': user_dict.get('phone', ''),
                    'address': user_dict.get('address', ''),
                    'employee_shift': user_dict.get('employee_shift', ''),
                    'shift_type': user_dict.get('shift_type', ''),
                    'office_distance': user_dict.get('office_distance', 0)
                }

                route_info['route']['assigned_users'].append(complete_user)

            # Re-optimize route with new users
            optimize_route_sequence_safe(route_info['route'], office_lat, office_lon)
            update_route_metrics_safe(route_info['route'], office_lat, office_lon)

            print(f"      🧩 Directionally filled {len(users_to_add)} users into route {route_info['route']['driver_id']} "
                  f"(now {len(route_info['route']['assigned_users'])}/{route_info['route']['vehicle_type']})")

        filled_routes.append(route_info['route'])

    # Calculate remaining unassigned users
    remaining_unassigned = unassigned_users_df[~unassigned_users_df['user_id'].isin(filled_user_ids)]

    return filled_routes, remaining_unassigned


def apply_final_capacity_optimization(routes, office_lat, office_lon):
    """
    FINAL OPTIMIZATION: One last pass to combine any routes that can fit together
    """
    if len(routes) < 2:
        return routes

    optimized_routes = routes.copy()
    improved = True
    iteration = 0

    while improved and iteration < 3:  # Max 3 iterations
        improved = False
        iteration += 1

        for i in range(len(optimized_routes)):
            for j in range(i + 1, len(optimized_routes)):
                if i >= len(optimized_routes) or j >= len(optimized_routes):
                    continue

                route1 = optimized_routes[i]
                route2 = optimized_routes[j]

                combined_users = len(route1['assigned_users']) + len(route2['assigned_users'])
                max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])

                # CRITICAL: Add strict geographic validation before merging
                # This prevents extreme left-right crossing in final optimization
                if combined_users <= max_capacity:
                    # Calculate route centers for geographic validation
                    route1_center = calculate_route_center(route1['assigned_users'])
                    route2_center = calculate_route_center(route2['assigned_users'])

                    if route1_center and route2_center:
                        # Get office-relative positions for both routes
                        route1_relative_x = route1_center[1] - office_lon
                        route2_relative_x = route2_center[1] - office_lon

                        # Determine if routes are on same side of office (left/right)
                        route1_on_left = route1_relative_x < 0
                        route2_on_left = route2_relative_x < 0

                        # STRICT FINAL MERGE CRITERIA: Prevent extreme geographic crossing
                        # 1. Routes must be on same side of office OR both very close to office centerline
                        same_side = route1_on_left == route2_on_left
                        close_to_centerline = abs(route1_relative_x) < 0.015 and abs(route2_relative_x) < 0.015  # Within ~1.5km

                        # 2. Routes must be reasonably close to each other (2km max for final merge)
                        distance_between_routes = haversine_distance(
                            route1_center[0], route1_center[1],
                            route2_center[0], route2_center[1]
                        )
                        reasonable_distance = distance_between_routes < 2.0

                        # 3. Final geographic validation
                        geographic_coherence = same_side or close_to_centerline

                        # Only merge if all geographic criteria are met
                        if not (geographic_coherence and reasonable_distance):
                            print(f"      🚫 BLOCKED extreme geographic merge: routes on opposite sides ({distance_between_routes:.1f}km apart)")
                            continue
                    # Use the route with larger capacity as base
                    base_route = route1 if route1['vehicle_type'] >= route2['vehicle_type'] else route2

                    merged_route = base_route.copy()
                    merged_route['assigned_users'] = route1['assigned_users'] + route2['assigned_users']

                    # Re-optimize
                    if merged_route['assigned_users']:
                        optimize_route_sequence_improved(merged_route, office_lat, office_lon)
                        update_route_metrics_improved(merged_route, office_lat, office_lon)

                    # Replace routes
                    optimized_routes = [r for k, r in enumerate(optimized_routes) if k != i and k != j]
                    optimized_routes.append(merged_route)

                    improved = True
                    print(f"      ⚡ Final optimization merged routes: {combined_users} users into capacity {max_capacity}")
                    break

            if improved:
                break

    return optimized_routes


def apply_hard_consolidation_capacity(routes, office_lat, office_lon):
    """
    CAPACITY-OPTIMIZED: Final hard consolidation pass for maximum capacity efficiency
    Balance.py approach with aggressive optimization
    """
    print(f"  ⚡ CAPACITY hard consolidation: Final optimization pass...")

    if len(routes) < 2:
        return routes

    # First, apply balance.py-style 0.5km hard consolidation
    routes = apply_balance_style_hard_consolidation(routes, office_lat, office_lon)

    # Then, consolidate very close routes
    routes = consolidate_very_close_routes_capacity(routes, office_lat, office_lon)

    # Then try aggressive merging of small/underutilized routes
    max_iterations = 2
    improvements_performed = 0

    for iteration in range(max_iterations):
        iteration_improvements = 0

        # Identify route groups for batch optimization
        route_groups = identify_nearby_route_groups_capacity(routes)

        # Apply advanced reorganization
        reorg_improvements = reorganize_users_geographically_capacity(route_groups, office_lat, office_lon)
        iteration_improvements += reorg_improvements

        # Try final merging pass
        merged_count = 0
        route_pairs = []

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                combined_users = len(routes[i]['assigned_users']) + len(routes[j]['assigned_users'])
                max_capacity = max(routes[i]['vehicle_type'], routes[j]['vehicle_type'])

                if combined_users <= max_capacity:
                    route_pairs.append((i, j, combined_users, max_capacity))

        # Sort by combined utilization (prefer higher utilization)
        route_pairs.sort(key=lambda x: (x[2] / x[3]), reverse=True)

        for i, j, combined_users, max_capacity in route_pairs:
            if i >= len(routes) or j >= len(routes):
                continue

            # Calculate merge score
            center1 = calculate_route_center(routes[i]['assigned_users'])
            center2 = calculate_route_center(routes[j]['assigned_users'])

            if center1 and center2:
                distance = haversine_distance(center1[0], center1[1], center2[0], center2[1])
                utilization = combined_users / max_capacity

                # Aggressive merging criteria
                if (distance < CONFIG['MAX_MERGE_DISTANCE_KM'] * 1.2 and  # More lenient distance
                    utilization >= CONFIG['MIN_CAPACITY_UTILIZATION']):  # Good utilization

                    # Perform merge
                    base_route = routes[i] if routes[i]['vehicle_type'] >= routes[j]['vehicle_type'] else routes[j]

                    merged_route = base_route.copy()
                    merged_route['assigned_users'] = routes[i]['assigned_users'] + routes[j]['assigned_users']

                    if len(merged_route['assigned_users']) <= merged_route['vehicle_type']:
                        merged_route = optimize_route_sequence_safe(merged_route, office_lat, office_lon)

                        # Remove original routes and add merged route
                        routes = [r for idx, r in enumerate(routes) if idx not in [i, j]]
                        routes.append(merged_route)

                        merged_count += 1
                        iteration_improvements += 1
                        break  # Break to avoid index issues

        improvements_performed += iteration_improvements

        if iteration_improvements < 2:  # Stop if minimal improvements
            break

        print(f"    Hard consolidation iteration {iteration + 1}: {iteration_improvements} improvements")

    total_improvements = improvements_performed + reorg_improvements
    print(f"  ✅ Hard consolidation complete: {total_improvements} total improvements, {len(routes)} final routes")

    return routes


def enhance_natural_user_clustering_capacity(routes, office_lat, office_lon):
    """
    CAPACITY-OPTIMIZED: GLOBAL Geographic User Reorganization
    True global swapping that consolidates entire geographic areas

    Instead of moving individual users, this identifies geographic clusters
    and reassigns them to the most appropriate routes for maximum cohesion
    """
    print(f"  🌍 GLOBAL geographic reorganization for {len(routes)} routes...")

    if len(routes) < 2:
        return routes

    reorganization_count = 0

    # Step 1: Identify geographic clusters across ALL routes using DBSCAN
    all_users = []
    for route_idx, route in enumerate(routes):
        for user_idx, user in enumerate(route['assigned_users']):
            user_id = user.get('user_id', user.get('id', ''))
            user_lat = float(user.get('latitude', user.get('lat', 0)))
            user_lon = float(user.get('longitude', user.get('lng', 0)))

            all_users.append({
                'user': user,
                'user_id': user_id,
                'route_idx': route_idx,
                'user_idx': user_idx,
                'lat': user_lat,
                'lon': user_lon,
                'route': route
            })

    if len(all_users) == 0:
        return routes

    # Apply DBSCAN to find natural geographic clusters (1.5km radius)
    user_coordinates = [(u['lat'], u['lon']) for u in all_users]
    clustering = DBSCAN(eps=0.015, min_samples=2).fit(user_coordinates)  # ~1.5km

    # Assign cluster labels to users
    for i, user in enumerate(all_users):
        user['geo_cluster'] = clustering.labels_[i]

    # Group users by geographic cluster
    geo_clusters = {}
    for user in all_users:
        cluster_id = user['geo_cluster']
        if cluster_id == -1:  # Noise points - skip
            continue
        if cluster_id not in geo_clusters:
            geo_clusters[cluster_id] = []
        geo_clusters[cluster_id].append(user)

    print(f"    🗺️ Found {len(geo_clusters)} geographic clusters across routes")

    # Step 2: For each geographic cluster, find the BEST route assignment
    for cluster_id, cluster_users in geo_clusters.items():
        if len(cluster_users) < 2:  # Skip single-user clusters
            continue

        # Calculate cluster center
        center_lat = sum(u['lat'] for u in cluster_users) / len(cluster_users)
        center_lon = sum(u['lon'] for u in cluster_users) / len(cluster_users)

        # Find which routes these users currently belong to
        current_routes = list(set(u['route_idx'] for u in cluster_users))

        # If all users already in same route, skip
        if len(current_routes) == 1:
            continue

        print(f"      🎯 Cluster {cluster_id}: {len(cluster_users)} users across {len(current_routes)} routes")

        # Find the BEST route for this cluster based on multiple factors
        best_route = None
        best_score = -float('inf')

        for route_idx, route in enumerate(routes):
            if not route['assigned_users']:
                continue

            # Calculate route center
            route_center = calculate_route_center(route['assigned_users'])
            if not route_center:
                continue

            # Factor 1: Distance between cluster center and route center
            distance_score = max(0, (5 - haversine_distance(center_lat, center_lon, route_center[0], route_center[1])) / 5)

            # Factor 2: Capacity availability
            available_capacity = route['vehicle_type'] - len(route['assigned_users'])
            capacity_score = max(0, available_capacity / route['vehicle_type'])

            # Factor 3: Geographic alignment (bearing)
            route_bearing = calculate_route_bearing(route)
            cluster_bearing = calculate_cluster_bearing(center_lat, center_lon, cluster_users)
            bearing_diff = abs(route_bearing - cluster_bearing)
            if bearing_diff > 180:
                bearing_diff = 360 - bearing_diff
            alignment_score = max(0, (90 - bearing_diff) / 90)

            # Combined score (prioritize distance and capacity)
            total_score = distance_score * 0.5 + capacity_score * 0.4 + alignment_score * 0.1

            if total_score > best_score and available_capacity >= len(cluster_users):
                best_score = total_score
                best_route = route_idx

        # Step 3: If we found a better route, move ALL cluster users there
        if best_route is not None and best_route not in current_routes:
            # CRITICAL FIX: Check for duplicate assignments before moving
            users_to_move = []

            for user in cluster_users:
                if user['route_idx'] == best_route:
                    continue  # Skip users already in the best route

                # CRITICAL: Verify user hasn't been moved already by another cluster
                current_user_route = None
                for route_idx, route in enumerate(routes):
                    if route['assigned_users']:
                        for assigned_user in route['assigned_users']:
                            if assigned_user.get('user_id', assigned_user.get('id', '')) == user['user_id']:
                                current_user_route = route_idx
                                break
                    if current_user_route is not None:
                        break

                # Only move if user is still in original route and not duplicated
                if current_user_route == user['route_idx']:
                    users_to_move.append(user)
                else:
                    print(f"        ⚠️ User {user['user_id']} already moved or duplicated - skipping")

            # Move verified users
            for user in users_to_move:
                # Remove from current route
                old_route = routes[user['route_idx']]
                old_route['assigned_users'] = [
                    u for u in old_route['assigned_users']
                    if u.get('user_id', u.get('id', '')) != user['user_id']
                ]

                # CRITICAL: Double-check user not already in target route
                user_already_in_target = False
                for assigned_user in routes[best_route]['assigned_users']:
                    if assigned_user.get('user_id', assigned_user.get('id', '')) == user['user_id']:
                        user_already_in_target = True
                        break

                if not user_already_in_target:
                    # Add to best route
                    routes[best_route]['assigned_users'].append(user['user'])
                    print(f"        🔄 Moving user {user['user_id']} from route {old_route['driver_id']} to route {routes[best_route]['driver_id']}")
                    reorganization_count += 1
                else:
                    print(f"        ⚠️ User {user['user_id']} already exists in target route - skipping")

            print(f"      ✅ Cluster {cluster_id}: Moved {len(users_to_move)} users to route {routes[best_route]['driver_id']} (score: {best_score:.2f})")

    # Step 4: Remove empty routes and re-optimize all routes
    optimized_routes = []
    for route in routes:
        if route['assigned_users']:
            # Re-optimize the route sequence
            if len(route['assigned_users']) > 1:
                route = optimize_route_sequence_safe(route, office_lat, office_lon)
                update_route_metrics_safe(route, office_lat, office_lon)
            optimized_routes.append(route)
        else:
            print(f"      🗑️ Removed empty route {route['driver_id']}")

    print(f"  ✅ Global reorganization complete: {reorganization_count} user movements, {len(optimized_routes)} final routes")

    return optimized_routes


def calculate_route_bearing(route):
    """Calculate the dominant bearing of a route based on user sequence"""
    users = route['assigned_users']
    if len(users) < 2:
        return 0

    bearings = []
    for i in range(len(users) - 1):
        user1_lat = float(users[i].get('latitude', users[i].get('lat', 0)))
        user1_lon = float(users[i].get('longitude', users[i].get('lng', 0)))
        user2_lat = float(users[i + 1].get('latitude', users[i + 1].get('lat', 0)))
        user2_lon = float(users[i + 1].get('longitude', users[i + 1].get('lng', 0)))

        bearing = calculate_bearing(user1_lat, user1_lon, user2_lat, user2_lon)
        bearings.append(bearing)

    return sum(bearings) / len(bearings) if bearings else 0


def calculate_cluster_bearing(center_lat, center_lon, cluster_users):
    """Calculate the dominant bearing of users in a cluster from center"""
    bearings = []
    for user in cluster_users:
        bearing = calculate_bearing(center_lat, center_lon, user['lat'], user['lon'])
        bearings.append(bearing)

    return sum(bearings) / len(bearings) if bearings else 0


def remove_duplicate_users_from_routes(routes):
    """
    CRITICAL FIX: Remove duplicate users from routes and ensure each user appears exactly once
    This function enforces data integrity across all routes and PRESERVES ORIGINAL USER DATA
    """
    print(f"  🔧 CRITICAL: Removing duplicate users from {len(routes)} routes...")

    # Track all assigned users globally with completeness scoring
    global_assigned_users = {}
    duplicates_found = 0
    routes_cleaned = 0

    # First pass: score each user by data completeness and keep the most complete version
    for route_idx, route in enumerate(routes):
        if not route.get('assigned_users'):
            continue

        clean_users = []
        for user in route['assigned_users']:
            user_id = user.get('user_id', user.get('id', ''))

            if not user_id:
                continue  # Skip users without valid IDs

            # Calculate user data completeness score and show details for debugging
            completeness_score = 0
            details = []
            if user.get('first_name'):
                completeness_score += 1
                details.append('first_name')
            if user.get('last_name'):
                completeness_score += 1
                details.append('last_name')
            if user.get('email'):
                completeness_score += 1
                details.append('email')
            if user.get('phone'):
                completeness_score += 1
                details.append('phone')
            if user.get('address'):
                completeness_score += 1
                details.append('address')
            if user.get('employee_shift'):
                completeness_score += 1
                details.append('employee_shift')

            # Debug: Show user completeness on first few duplicates
            if user_id in global_assigned_users and completeness_score != 6:
                print(f"        📊 User {user_id} details: {details} (score: {completeness_score}/6)")

            user_info = {
                'user': user,
                'route_idx': route_idx,
                'driver_id': route.get('driver_id', 'unknown'),
                'completeness': completeness_score
            }

            if user_id not in global_assigned_users:
                # First time seeing this user - keep them
                global_assigned_users[user_id] = user_info
                clean_users.append(user)
            else:
                # Duplicate found! Compare completeness scores
                existing_user = global_assigned_users[user_id]
                if completeness_score > existing_user['completeness']:
                    # This user has more complete data - replace the existing one
                    print(f"      🔄 UPGRADING: User {user_id} has better data (score: {completeness_score} > {existing_user['completeness']})")
                    print(f"         Keeping route {route.get('driver_id', 'unknown')}, removing from route {existing_user['driver_id']}")

                    global_assigned_users[user_id] = user_info
                    clean_users.append(user)

                    # Mark existing route for cleaning
                    existing_route_idx = existing_user['route_idx']
                    if existing_route_idx < len(routes):
                        existing_route = routes[existing_route_idx]
                        if existing_route.get('assigned_users'):
                            existing_route['assigned_users'] = [
                                u for u in existing_route['assigned_users']
                                if u.get('user_id', u.get('id', '')) != user_id
                            ]
                else:
                    # Existing user has better or equal data - remove this duplicate
                    duplicates_found += 1
                    original_route = existing_user['driver_id']
                    print(f"      🚨 DUPLICATE: User {user_id} found in route {route.get('driver_id', 'unknown')}, keeping more complete version in route {original_route}")

        # Update route with deduplicated users
        if len(clean_users) != len(route['assigned_users']):
            route['assigned_users'] = clean_users
            routes_cleaned += 1

    print(f"  ✅ Smart deduplication complete: {duplicates_found} duplicates removed, {routes_cleaned} routes cleaned, {len(global_assigned_users)} unique users preserved")

    return routes


def apply_final_geographic_user_optimization_capacity(routes, unassigned_users_df, office_lat, office_lon):
    """
    CAPACITY-OPTIMIZED: Final Geographic User Optimization with Bearing Analysis
    Ultra-advanced balance.py-style optimization for maximum geographic efficiency

    1. Analyze user-route distance mismatches
    2. Calculate bearing alignments between users and route directions
    3. Perform intelligent swaps to optimize geographic coverage
    4. Maintain 100% assignment while maximizing efficiency

    Returns: (optimized_routes, optimization_count)
    """
    print(f"  🎯 Final geographic optimization: {len(routes)} routes, {len(unassigned_users_df)} unassigned")

    if len(routes) < 2:
        return routes, 0

    optimization_count = 0

    # Create route direction analysis for bearing calculations
    route_analysis = []
    for route in routes:
        if len(route['assigned_users']) >= 2:
            # Calculate route bearing based on user sequence
            bearings = []
            users = route['assigned_users']

            # Calculate bearings between consecutive users
            for i in range(len(users) - 1):
                user1_lat = float(users[i].get('latitude', 0))
                user1_lon = float(users[i].get('longitude', 0))
                user2_lat = float(users[i + 1].get('latitude', 0))
                user2_lon = float(users[i + 1].get('longitude', 0))

                bearing = calculate_bearing(user1_lat, user1_lon, user2_lat, user2_lon)
                bearings.append(bearing)

            # Calculate dominant bearing direction
            avg_bearing = sum(bearings) / len(bearings) if bearings else 0

            route_analysis.append({
                'route': route,
                'avg_bearing': avg_bearing,
                'bearings': bearings,
                'center': calculate_route_center(users)
            })
        else:
            # Single user routes - use direction from office to user
            if route['assigned_users']:
                user = route['assigned_users'][0]
                user_lat = float(user.get('latitude', 0))
                user_lon = float(user.get('longitude', 0))
                bearing = calculate_bearing(office_lat, office_lon, user_lat, user_lon)

                route_analysis.append({
                    'route': route,
                    'avg_bearing': bearing,
                    'bearings': [bearing],
                    'center': calculate_route_center([user])
                })
            else:
                route_analysis.append({
                    'route': route,
                    'avg_bearing': 0,
                    'bearings': [],
                    'center': None
                })

    # Analyze user-route mismatches and bearing alignments
    users_to_swap = []

    for route_info in route_analysis:
        route = route_info['route']
        route_bearing = route_info['avg_bearing']
        route_center = route_info['center']

        for user in route['assigned_users']:
            user_lat = float(user.get('latitude', 0))
            user_lon = float(user.get('longitude', 0))
            user_id = user.get('user_id', user.get('id', ''))

            # Calculate distance from user to route center
            if route_center:
                distance_to_center = haversine_distance(user_lat, user_lon, route_center[0], route_center[1])
            else:
                distance_to_center = 0

            # Calculate bearing from route center to user
            if route_center:
                bearing_to_user = calculate_bearing(route_center[0], route_center[1], user_lat, user_lon)
                bearing_diff = abs(bearing_to_user - route_bearing)
                if bearing_diff > 180:
                    bearing_diff = 360 - bearing_diff
            else:
                bearing_diff = 0

            # Find better routes for this user
            better_routes = []
            for other_route_info in route_analysis:
                if other_route_info['route']['driver_id'] == route['driver_id']:
                    continue

                other_route = other_route_info['route']
                other_center = other_route_info['center']
                other_bearing = other_route_info['avg_bearing']

                if not other_center:
                    continue

                # Check capacity
                if len(other_route['assigned_users']) >= other_route['vehicle_type']:
                    continue

                # Calculate distance to other route center
                distance_to_other = haversine_distance(user_lat, user_lon, other_center[0], other_center[1])

                # Calculate bearing alignment
                bearing_to_other_user = calculate_bearing(other_center[0], other_center[1], user_lat, user_lon)
                other_bearing_diff = abs(bearing_to_other_user - other_bearing)
                if other_bearing_diff > 180:
                    other_bearing_diff = 360 - other_bearing_diff

                # Optimization score: prioritize distance reduction + bearing alignment
                distance_improvement = distance_to_center - distance_to_other
                bearing_improvement = bearing_diff - other_bearing_diff

                # Strict geographic cohesion check before any swap
                geographic_misalignment = bearing_diff > 40 or distance_to_other > 4.5

                if geographic_misalignment:
                    continue  # Skip this route - too geographically misaligned

                # Combined score with stricter requirements
                combined_score = distance_improvement * 0.8 + bearing_improvement * 0.2

                if combined_score > 0.8:  # Higher threshold for swaps
                    better_routes.append({
                        'route': other_route,
                        'distance_improvement': distance_improvement,
                        'bearing_improvement': bearing_improvement,
                        'combined_score': combined_score,
                        'distance_to_other': distance_to_other
                    })

            # Sort by combined improvement score
            better_routes.sort(key=lambda x: x['combined_score'], reverse=True)

            if better_routes:
                users_to_swap.append({
                    'user': user,
                    'user_id': user_id,
                    'current_route': route,
                    'distance_to_current': distance_to_center,
                    'bearing_diff': bearing_diff,
                    'best_alternative': better_routes[0]
                })

    # Sort by highest improvement score
    users_to_swap.sort(key=lambda x: x['best_alternative']['combined_score'], reverse=True)

    # Perform optimized swaps
    print(f"    🔄 Found {len(users_to_swap)} users for potential bearing optimization")

    for swap_info in users_to_swap:
        user = swap_info['user']
        user_id = swap_info['user_id']
        current_route = swap_info['current_route']
        target_route = swap_info['best_alternative']['route']

        # Final capacity check
        if len(target_route['assigned_users']) >= target_route['vehicle_type']:
            continue

        # Ensure user hasn't been moved already
        user_in_current = False
        for u in current_route['assigned_users']:
            if u.get('user_id', u.get('id', '')) == user_id:
                user_in_current = True
                break

        if not user_in_current:
            continue

        # Perform the swap
        print(f"      🎯 Moving user {user_id} to route {target_route['driver_id']}")
        print(f"         Distance improvement: {swap_info['best_alternative']['distance_improvement']:.2f}km")
        print(f"         Bearing improvement: {swap_info['best_alternative']['bearing_improvement']:.1f}°")

        # Remove user from current route
        current_route['assigned_users'] = [
            u for u in current_route['assigned_users']
            if u.get('user_id', u.get('id', '')) != user_id
        ]

        # Add user to target route
        target_route['assigned_users'].append(user)

        optimization_count += 1

        # Re-optimize affected routes
        if current_route['assigned_users']:
            current_route = optimize_route_sequence_improved(current_route, office_lat, office_lon)
            update_route_metrics_improved(current_route, office_lat, office_lon)

        if target_route['assigned_users']:
            target_route = optimize_route_sequence_improved(target_route, office_lat, office_lon)
            update_route_metrics_improved(target_route, office_lat, office_lon)

    # Try to assign unassigned users to routes with bearing alignment
    if not unassigned_users_df.empty:
        print(f"    🎯 Trying to assign {len(unassigned_users_df)} unassigned users with bearing optimization...")

        for _, user in unassigned_users_df.iterrows():
            user_lat = float(user.get('latitude', 0))
            user_lon = float(user.get('longitude', 0))
            user_id = user.get('user_id', '')

            best_route = None
            best_score = -float('inf')

            for route_info in route_analysis:
                route = route_info['route']
                route_center = route_info['center']
                route_bearing = route_info['avg_bearing']

                # Check capacity
                if len(route['assigned_users']) >= route['vehicle_type']:
                    continue

                if not route_center:
                    continue

                # Calculate distance to route
                distance = haversine_distance(user_lat, user_lon, route_center[0], route_center[1])

                # Calculate bearing alignment
                bearing_to_user = calculate_bearing(route_center[0], route_center[1], user_lat, user_lon)
                bearing_diff = abs(bearing_to_user - route_bearing)
                if bearing_diff > 180:
                    bearing_diff = 360 - bearing_diff

                # Combined score with STRICT bearing constraints (geographic cohesion priority)
                bearing_score = max(0, (45 - bearing_diff) / 45)  # Much stricter - only 45° tolerance
                distance_score = max(0, (8 - distance) / 8)  # Shorter distance range for better cohesion

                # Geographic cohesion requirements
                geographic_cohesion = bearing_diff <= 35 and distance <= 4.0  # Strict constraints

                # Only consider if geographic cohesion is maintained
                if geographic_cohesion:
                    combined_score = distance_score * 0.7 + bearing_score * 0.3  # Prioritize distance more

                    if combined_score > best_score:
                        best_score = combined_score
                        best_route = route

            if best_route:
                # Assign user to best route
                user_dict = user.to_dict() if hasattr(user, 'to_dict') else dict(user)
                best_route['assigned_users'].append(user_dict)

                print(f"      ✅ Assigned unassigned user {user_id} to route {best_route['driver_id']}")
                if route_center:
                    print(f"         Distance: {haversine_distance(user_lat, user_lon, route_center[0], route_center[1]):.2f}km, Score: {best_score:.2f}")
                else:
                    print(f"         Distance: N/A, Score: {best_score:.2f}")

                optimization_count += 1

    # Final route optimization and cleanup
    optimized_routes = []
    for route in routes:
        if route['assigned_users']:
            # Final re-optimization
            route = optimize_route_sequence_safe(route, office_lat, office_lon)
            update_route_metrics_safe(route, office_lat, office_lon)
            optimized_routes.append(route)
        else:
            # Remove empty routes
            print(f"      🗑️ Removing empty route {route['driver_id']}")

    print(f"  ✅ Final bearing optimization complete: {optimization_count} improvements, {len(optimized_routes)} final routes")

    return optimized_routes, optimization_count


# =============================================================================
# MAIN CAPACITY ASSIGNMENT FUNCTION (Balance.py Architecture)
# =============================================================================

def run_assignment_capacity(source_id: str, parameter: int = 1, string_param: str = "", choice: str = ""):
    """
    Main entry point for CAPACITY-OPTIMIZED assignment
    Balance.py 5-step architecture with capacity branding and intelligence
    """
    return run_capacity_assignment_simplified(source_id, parameter, string_param, choice)


def run_capacity_assignment_simplified(source_id: str, parameter: int = 1, string_param: str = "", choice: str = ""):
    """
    CAPACITY-OPTIMIZED: Simplified capacity assignment with Balance.py 5-step algorithm
    1. Geographic clustering (group nearby users)
    2. Smart capacity matching (send right-sized cabs)
    3. Advanced geographic optimization
    4. Direction-aware route merging
    5. Priority-based seat filling
    """
    start_time = time.time()

    print(f"🚀 Starting CAPACITY-OPTIMIZED assignment (Balance.py Architecture)")
    print(f"📋 Source: {source_id}, Parameter: {parameter}, String: {string_param}")

    try:
        # Load and validate data
        data = load_env_and_fetch_data(source_id, parameter, string_param)

        # Algorithm-level caching check
        db_name = source_id if source_id and source_id != "1" else data.get("db", "default")
        cached_result = None

        if ALGORITHM_CACHE_AVAILABLE:
            try:
                # Initialize cache for this algorithm
                cache = get_algorithm_cache(db_name, "capacity")

                # Generate current data signature
                current_signature = cache.generate_data_signature(data, {
                    'parameter': parameter,
                    'string_param': string_param,
                    'choice': choice,
                    'algorithm': 'capacity'
                })

                # Check for cached result
                cached_result = cache.get_cached_result(current_signature)

                if cached_result is not None:
                    print("⚡ FAST RESPONSE: Using cached algorithm result")
                    cached_result['_execution_time'] = 0.001
                    cached_result['_cache_hit'] = True
                    return cached_result

            except Exception as e:
                print(f"Cache system error: {e} - proceeding with algorithm execution")

        # Edge case handling
        users = data.get('users', [])
        if not users:
            print("⚠️ No users found - returning empty assignment")
            return create_empty_response(data, start_time)

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
            return create_no_drivers_response(data, start_time)

        print(f"📥 Data loaded - Users: {len(users)}, Drivers: {len(all_drivers)}")

        # Extract office coordinates and validate data
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        print("✅ Data validation passed")

        # Prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)

        print(f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")

        # ========================================
        # CAPACITY-OPTIMIZED 5-STEP ALGORITHM
        # ========================================

        # STEP 1: Geographic clustering (Balance.py Step 1)
        print("\n📍 STEP 1: CAPACITY geographic clustering...")
        user_df = cluster_users_by_proximity(user_df, office_lat, office_lon)

        # STEP 2: Smart capacity matching and initial assignment (Balance.py Step 2)
        print("\n🚗 STEP 2: CAPACITY smart matching...")
        routes = []
        assigned_user_ids = set()
        used_driver_ids = set()

        available_drivers = driver_df.copy()

        for cluster_id in user_df['geo_cluster'].unique():
            if cluster_id == -1:  # Skip noise points
                continue

            cluster_users = user_df[user_df['geo_cluster'] == cluster_id]
            assigned_driver = assign_cab_to_cluster_capacity(cluster_users, available_drivers, office_lat, office_lon)

            if assigned_driver is not None:
                route = create_route_from_cluster_capacity(cluster_users, assigned_driver, office_lat, office_lon)

                if route:
                    routes.append(route)
                    used_driver_ids.add(assigned_driver['driver_id'])

                    # Track assigned users
                    for _, user in cluster_users.iterrows():
                        assigned_user_ids.add(user['user_id'])

                    utilization = len(route['assigned_users']) / route['vehicle_type'] * 100
                    print(f"  ✅ Route {route['driver_id']}: {len(route['assigned_users'])}/{route['vehicle_type']} ({utilization:.1f}% util)")
            else:
                # CAPACITY ENHANCEMENT: Fallback to individual user assignment
                print(f"    🔄 Cluster rejected - trying individual user assignment for {len(cluster_users)} users...")
                individual_assignments = assign_individual_users_fallback(cluster_users, available_drivers, office_lat, office_lon)

                for assignment in individual_assignments:
                    if assignment['route']:
                        routes.append(assignment['route'])
                        used_driver_ids.add(assignment['driver_id'])
                        assigned_user_ids.add(assignment['user_id'])

                        utilization = len(assignment['route']['assigned_users']) / assignment['route']['vehicle_type'] * 100
                        print(f"  ✅ Individual assignment {assignment['route']['driver_id']}: {len(assignment['route']['assigned_users'])}/{assignment['route']['vehicle_type']} ({utilization:.1f}% util)")
                    else:
                        print(f"  ❌ Could not assign user {assignment['user_id']} individually")

            # Update available drivers
            available_drivers = driver_df[~driver_df['driver_id'].isin(used_driver_ids)]

        # STEP 3: Advanced geographic optimization (Balance.py Step 3)
        print("\n🗺️  STEP 3: CAPACITY geographic optimization...")
        routes = optimize_geographic_distribution_capacity(routes, office_lat, office_lon, driver_df)

        # STEP 3.1: Advanced user swapping with capacity constraints (Balance.py Enhanced)
        print("\n🔄 STEP 3.1: CAPACITY advanced user swapping...")
        routes = advanced_user_swapping_capacity(routes, office_lat, office_lon)

        # STEP 3.2: Geographic reorganization of route groups (Balance.py Enhanced)
        print("\n🎯 STEP 3.2: CAPACITY geographic reorganization...")
        route_groups = identify_nearby_route_groups_capacity(routes)
        reorg_improvements = reorganize_users_geographically_capacity(route_groups, office_lat, office_lon)
        print(f"    📍 Geographic reorganization: {reorg_improvements} improvements")

        # STEP 3.5: Direction-aware route merging (Balance.py Step 3.5)
        print("\n🔗 STEP 3.5: CAPACITY route merging...")
        routes = merge_small_routes_with_nearby_capacity(routes, office_lat, office_lon)

        # STEP 3.8: Hard consolidation for maximum capacity efficiency (Balance.py Enhanced)
        print("\n⚡ STEP 3.8: CAPACITY hard consolidation...")
        routes = apply_hard_consolidation_capacity(routes, office_lat, office_lon)

        # STEP 4: Priority-based seat filling (Balance.py Step 4)
        print("\n🪑 STEP 4: CAPACITY seat filling...")
        unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids)]

        if not unassigned_users_df.empty:
            routes, filled_ids = fill_remaining_seats_with_cluster_check_capacity(routes, unassigned_users_df, office_lat, office_lon)
            assigned_user_ids.update(filled_ids)

        # STEP 4.5: FINAL USER FALLBACK (Balance.py style - prioritize coverage)
        print("\n🚨 STEP 4.5: FINAL user fallback for maximum coverage...")

        # Recalculate currently available drivers for fallback
        used_driver_ids = {route['driver_id'] for route in routes}
        currently_available_drivers = driver_df[~driver_df['driver_id'].isin(used_driver_ids)]

        routes, unassigned_users_remaining = apply_final_user_fallback_capacity(
            routes, unassigned_users_df, currently_available_drivers, office_lat, office_lon
        )
        print(f"  ✅ Final fallback assigned additional users: {len(unassigned_users_df) - len(unassigned_users_remaining)}")

        # CRITICAL FIX: Recalculate assigned_user_ids based on actual routes to fix tracking
        assigned_user_ids = set()
        for route in routes:
            for user in route['assigned_users']:
                user_id = user.get('user_id', user.get('id', ''))
                if user_id:
                    assigned_user_ids.add(str(user_id))

        # Recalculate final unassigned users based on actual assignments
        final_unassigned_df = user_df[~user_df['user_id'].isin(assigned_user_ids)]
        print(f"  📊 Final assignment check: {len(assigned_user_ids)} users assigned, {len(final_unassigned_df)} truly unassigned")

        # STEP 4.8: AGGRESSIVE CAPACITY-BASED ROUTE MERGING & DIRECTIONAL FILLING
        print("\n🔥 STEP 4.8: AGGRESSIVE capacity-based merging for 100% assignment...")
        routes, final_unassigned_df = apply_aggressive_capacity_merging_capacity(
            routes, final_unassigned_df, office_lat, office_lon
        )
        print(f"  ✅ Aggressive merging: {len(routes)} final routes, {len(final_unassigned_df)} remaining unassigned")

        # CRITICAL: Update assigned_user_ids with final merged routes for accurate reporting
        assigned_user_ids = set()
        for route in routes:
            for user in route['assigned_users']:
                user_id = user.get('user_id', user.get('id', ''))
                if user_id:
                    assigned_user_ids.add(str(user_id))

        print(f"  🎯 FINAL ACCURATE COUNT: {len(assigned_user_ids)} users actually assigned, {len(final_unassigned_df)} truly unassigned")

        # STEP 4.85: GLOBAL GEOGRAPHIC USER REORGANIZATION
        print("\n🌍 STEP 4.85: GLOBAL geographic user reorganization...")
        routes = enhance_natural_user_clustering_capacity(routes, office_lat, office_lon)

        # Update assigned_user_ids after natural clustering
        assigned_user_ids = set()
        for route in routes:
            for user in route['assigned_users']:
                user_id = user.get('user_id', user.get('id', ''))
                if user_id:
                    assigned_user_ids.add(str(user_id))

        # Recalculate unassigned users after clustering
        final_unassigned_df = user_df[~user_df['user_id'].isin(assigned_user_ids)]
        print(f"    📊 After clustering: {len(assigned_user_ids)} assigned, {len(final_unassigned_df)} unassigned")

        # STEP 4.9: FINAL GEOGRAPHIC USER OPTIMIZATION WITH BEARING ANALYSIS
        print("\n🎯 STEP 4.9: FINAL geographic user optimization with bearing analysis...")

        if len(routes) > 1:
            routes, bearing_optimization_count = apply_final_geographic_user_optimization_capacity(
                routes, final_unassigned_df, office_lat, office_lon
            )
            print(f"  ✅ Final bearing optimization: {bearing_optimization_count} improvements")

            # Update assigned_user_ids again for final accurate reporting
            assigned_user_ids = set()
            for route in routes:
                for user in route['assigned_users']:
                    user_id = user.get('user_id', user.get('id', ''))
                    if user_id:
                        assigned_user_ids.add(str(user_id))

            # Recalculate final unassigned users after bearing optimization
            final_unassigned_df = user_df[~user_df['user_id'].isin(assigned_user_ids)]
        else:
            print(f"    ℹ️ Single route - skipping bearing optimization")
            bearing_optimization_count = 0

        # STEP 4.95: CRITICAL DUPLICATE USER REMOVAL
        print("\n🔧 STEP 4.95: CRITICAL duplicate user removal...")
        routes = remove_duplicate_users_from_routes(routes)

        # STEP 5: Apply optimal pickup ordering (Balance.py Step 5)
        print("\n🎯 STEP 5: CAPACITY optimal ordering...")
        if ORDERING_AVAILABLE and routes:
            try:
                logger.info(f"Applying optimal pickup ordering to {len(routes)} routes")
                routes = apply_route_ordering(routes, office_lat, office_lon, db_name=db_name, algorithm_name="capacity_optimized")
                logger.info("Optimal pickup ordering applied successfully")
            except Exception as e:
                logger.error(f"Failed to apply optimal ordering: {e}")

        # Build final response with capacity branding
        execution_time = time.time() - start_time

        print(f"\n✅ CAPACITY-OPTIMIZED assignment complete in {execution_time:.2f}s")
        print(f"📊 Final routes: {len(routes)}")
        print(f"👥 Users assigned: {len(assigned_user_ids)}")
        print(f"🎯 Using Balance.py architecture with capacity intelligence")

        # Save result to algorithm cache if available
        result = create_enhanced_response_capacity(routes, data, execution_time, assigned_user_ids, parameter, string_param, choice)

        if ALGORITHM_CACHE_AVAILABLE and cached_result is None:
            try:
                cache = get_algorithm_cache(db_name, "capacity")

                # Regenerate signature for cache storage
                current_signature = cache.generate_data_signature(data, {
                    'parameter': parameter,
                    'string_param': string_param,
                    'choice': choice,
                    'algorithm': 'capacity'
                })

                # Save the complete result to cache
                cache_result = result.copy()
                cache_result['_cache_metadata'] = {
                    'cached': True,
                    'cache_timestamp': time.time(),
                    'data_signature': current_signature
                }

                cache.save_result_to_cache(cache_result, current_signature)
                print("💾 Algorithm result saved to cache for future use")

            except Exception as e:
                print(f"Failed to save result to cache: {e}")

        return result

    except requests.exceptions.RequestException as req_err:
        logger.error(f"API request failed: {req_err}")
        from algorithm.response.response_builder import create_error_response
        return create_error_response(
            error_message=f"API request failed: {req_err}",
            execution_time=time.time() - start_time,
            optimization_mode="capacity",
            parameter=parameter,
            string_param=string_param,
            choice=choice
        )
    except ValueError as val_err:
        logger.error(f"Data validation error: {val_err}")
        from algorithm.response.response_builder import create_error_response
        return create_error_response(
            error_message=f"Data validation error: {val_err}",
            execution_time=time.time() - start_time,
            optimization_mode="capacity",
            parameter=parameter,
            string_param=string_param,
            choice=choice
        )
    except Exception as e:
        logger.error(f"Assignment failed: {e}", exc_info=True)
        from algorithm.response.response_builder import create_error_response
        return create_error_response(
            error_message=f"Assignment failed: {e}",
            execution_time=time.time() - start_time,
            optimization_mode="capacity",
            parameter=parameter,
            string_param=string_param,
            choice=choice
        )


def create_enhanced_response_capacity(routes, data, execution_time, assigned_user_ids, parameter=1, string_param="", choice=""):
    """
    Create enhanced response with capacity branding
    Balance.py response format with capacity intelligence
    """
    # Filter out routes with no assigned users
    filtered_routes = []
    empty_route_driver_ids = set()

    for route in routes:
        if route['assigned_users'] and len(route['assigned_users']) > 0:
            filtered_routes.append(route)
        else:
            empty_route_driver_ids.add(route['driver_id'])
            print(f"  📋 Moving driver {route['driver_id']} with no users to unassigned drivers")

    routes = filtered_routes

    # Build unassigned users list (using corrected assigned_user_ids from actual routes)
    users = data.get('users', [])
    unassigned_users = []
    for user in users:
        if str(user.get('id', user.get('sub_user_id', ''))) not in assigned_user_ids:
            unassigned_user = {
                'user_id': str(user.get('id', user.get('sub_user_id', ''))),
                'lat': float(user.get('latitude', 0)),
                'lng': float(user.get('longitude', 0)),
                'office_distance': float(user.get('office_distance', 0))
            }
            if user.get('first_name'):
                unassigned_user['first_name'] = str(user['first_name'])
            if user.get('email'):
                unassigned_user['email'] = str(user['email'])
            unassigned_users.append(unassigned_user)

    # Build unassigned drivers list
    assigned_driver_ids = {route['driver_id'] for route in routes}
    all_drivers = []
    if "drivers" in data:
        drivers_data = data["drivers"]
        all_drivers.extend(drivers_data.get("driversUnassigned", []))
        all_drivers.extend(drivers_data.get("driversAssigned", []))
    else:
        all_drivers.extend(data.get("driversUnassigned", []))
        all_drivers.extend(data.get("driversAssigned", []))

    unassigned_drivers = []
    for driver in all_drivers:
        driver_id = str(driver.get('id', driver.get('sub_user_id', '')))
        if driver_id not in assigned_driver_ids:
            driver_data = {
                'driver_id': driver_id,
                'capacity': int(driver.get('capacity', 0)),
                'vehicle_id': str(driver.get('vehicle_id', '')),
                'latitude': float(driver.get('latitude', 0.0)),
                'longitude': float(driver.get('longitude', 0.0))
            }
            unassigned_drivers.append(driver_data)

    # Final metrics update for all routes
    office_lat, office_lon = extract_office_coordinates(data)
    for route in routes:
        update_route_metrics_safe(route, office_lat, office_lon)

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

        for driver in all_drivers:
            if str(driver.get('id', driver.get('sub_user_id', ''))) == driver_id:
                driver_info = driver
                break

        if driver_info:
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

            for orig_user in original_users:
                if str(orig_user.get('id', orig_user.get('sub_user_id', ''))) == user['user_id']:
                    # Get original pickup_order if it exists and user doesn't have it
                    if 'pickup_order' not in enhanced_user and 'pickup_order' in orig_user:
                        enhanced_user['pickup_order'] = orig_user['pickup_order']

                    # Update fields but preserve coordinate format consistency
                    enhanced_user.update({
                        'address': orig_user.get('address', ''),
                        'employee_shift': orig_user.get('employee_shift', ''),
                        'shift_type': orig_user.get('shift_type', ''),
                        'last_name': orig_user.get('last_name', ''),
                        'phone': orig_user.get('phone', '')
                    })
                    break

            # Ensure coordinates are always in lat/lng format
            if 'latitude' in enhanced_user or 'longitude' in enhanced_user:
                enhanced_user['lat'] = float(enhanced_user.get('latitude', enhanced_user.get('lat', 0)))
                enhanced_user['lng'] = float(enhanced_user.get('longitude', enhanced_user.get('lng', 0)))
                # Remove old coordinate format to maintain consistency
                enhanced_user.pop('latitude', None)
                enhanced_user.pop('longitude', None)

            enhanced_users.append(enhanced_user)

        enhanced_route['assigned_users'] = enhanced_users
        enhanced_routes.append(enhanced_route)

    # Enhance unassigned users
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

    # Enhance unassigned drivers
    enhanced_unassigned_drivers = []
    for driver in unassigned_drivers:
        enhanced_driver = driver.copy()

        for orig_driver in all_drivers:
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

    # Clustering analysis with capacity branding
    clustering_results = {
        "method": "capacity_optimized_geographic_clustering",
        "clusters": len(routes),
        "algorithm": "Balance.py Architecture with Capacity Intelligence",
        "optimization_focus": "Geographic proximity with capacity efficiency"
    }

    # Import the standardized response builder
    from algorithm.response.response_builder import (
        build_standard_response,
        save_standardized_response,
        log_response_metrics
    )

    # Build standardized response (clustering_analysis is removed as per new standards)
    result = build_standard_response(
        status="true",
        execution_time=execution_time,
        routes=enhanced_routes,
        unassigned_users=enhanced_unassigned_users,
        unassigned_drivers=enhanced_unassigned_drivers,
        optimization_mode="capacity",
        parameter=parameter,
        company=company_info,
        shift=shift_info,
        string_param=string_param,
        choice=choice
    )

    # Save standardized response
    save_standardized_response(result, "drivers_and_routes.json")

    # Log metrics for monitoring
    log_response_metrics(result, "capacity_optimized")

    return result


def create_empty_response(data, start_time):
    """Create response for no users scenario"""
    from algorithm.response.response_builder import build_standard_response
    return build_standard_response(
        status="true",
        execution_time=time.time() - start_time,
        routes=[],
        unassigned_users=[],
        unassigned_drivers=_get_all_drivers_as_unassigned(data),
        optimization_mode="capacity",
        parameter=1
    )


def create_no_drivers_response(data, start_time):
    """Create response for no drivers scenario"""
    from algorithm.response.response_builder import build_standard_response
    unassigned_users = _convert_users_to_unassigned_format(data.get('users', []))

    return build_standard_response(
        status="true",
        execution_time=time.time() - start_time,
        routes=[],
        unassigned_users=unassigned_users,
        unassigned_drivers=[],
        optimization_mode="capacity",
        parameter=1
    )