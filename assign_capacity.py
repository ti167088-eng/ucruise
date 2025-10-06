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
from scipy.spatial import KDTree
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

    print(f"🎯 Using optimization mode: CAPACITY OPTIMIZATION WITH BLUEPRINT")

    # Validate and set configuration with mode-specific overrides
    config = {}

    # Distance configurations with mode overrides (more lenient for capacity filling)
    config['MAX_FILL_DISTANCE_KM'] = max(0.1, float(mode_config.get("max_fill_distance_km", cfg.get("max_fill_distance_km", 8.0))))
    config['MERGE_DISTANCE_KM'] = max(0.1, float(mode_config.get("merge_distance_km", cfg.get("merge_distance_km", 5.0))))
    config['DBSCAN_EPS_KM'] = max(0.1, float(cfg.get("dbscan_eps_km", 2.5)))
    config['OVERFLOW_PENALTY_KM'] = max(0.0, float(cfg.get("overflow_penalty_km", 5.0)))
    config['DISTANCE_ISSUE_THRESHOLD'] = max(0.1, float(cfg.get("distance_issue_threshold_km", 12.0)))
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(0.0, float(cfg.get("swap_improvement_threshold_km", 1.0)))

    # Blueprint-specific parameters - PROXIMITY-FIRST approach
    config['MICRO_CLUSTER_RADIUS_KM'] = 2.0  # Increased radius to keep nearby users together
    config['DIRECTIONAL_COHERENCE_THRESHOLD'] = 75  # More relaxed bearing spread for proximity priority
    config['PATH_INSERT_MAX_DETOUR_RATIO'] = 0.15  # 15% max detour for insertions (more lenient)
    config['PATH_INSERT_MAX_BEARING_DIFF'] = 45  # More relaxed bearing difference for insertions
    config['FINAL_MERGE_BEARING_THRESHOLD'] = 60  # More relaxed bearing threshold for final merges
    config['FINAL_MERGE_DISTANCE_THRESHOLD'] = 3.0  # Increased distance threshold for final merges
    config['MIN_UTILIZATION_IMPROVEMENT'] = 0.02  # 2% min improvement for merges (more aggressive)

    # Scoring weights for driver selection
    config['ALPHA_UTILIZATION'] = 1.0
    config['BETA_WASTE_PENALTY'] = 1.0
    config['GAMMA_DISTANCE_PENALTY'] = 0.5
    config['DELTA_BEARING_PENALTY'] = 0.3

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

    print(f"   📊 Proximity-first micro-cluster radius: {config['MICRO_CLUSTER_RADIUS_KM']}km")
    print(f"   📊 Relaxed directional coherence: {config['DIRECTIONAL_COHERENCE_THRESHOLD']}°")
    print(f"   📊 Lenient max detour ratio: {config['PATH_INSERT_MAX_DETOUR_RATIO']*100}%")
    print(f"   📊 Capacity weight: {config['capacity_weight']}")
    print(f"   🎯 PRIORITY: Nearby users first, direction secondary")

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


# =============================================================================
# BLUEPRINT IMPLEMENTATION - NEW FUNCTIONS
# =============================================================================

def calculate_local_density(coords_km, point_idx, radius_km=3.0):
    """Calculate local density around a point (users per km²)"""
    point = coords_km[point_idx]
    distances = np.sqrt(np.sum((coords_km - point)**2, axis=1))
    neighbors = np.sum(distances <= radius_km)
    area = np.pi * radius_km**2
    return neighbors / area if area > 0 else 0


def adaptive_micro_cluster_radius(user_df, office_lat, office_lon, base_radius=1.2):
    """
    Adaptive micro-cluster radius based on local density
    Dense zones: smaller radius, Sparse zones: larger radius
    """
    if user_df.empty:
        return base_radius
    
    # Convert to km
    coords = user_df[['latitude', 'longitude']].values
    lat_to_km = 111.0
    lon_to_km = 111.0 * math.cos(math.radians(office_lat))
    
    coords_km = coords.copy()
    coords_km[:, 0] = coords[:, 0] * lat_to_km
    coords_km[:, 1] = coords[:, 1] * lon_to_km
    
    # Calculate average density
    densities = []
    sample_size = min(50, len(coords_km))  # Sample for efficiency
    sample_indices = np.random.choice(len(coords_km), sample_size, replace=False)
    
    for idx in sample_indices:
        density = calculate_local_density(coords_km, idx, radius_km=3.0)
        densities.append(density)
    
    avg_density = np.mean(densities) if densities else 1.0
    
    # Adapt radius: high density → smaller radius, low density → larger radius
    if avg_density > 5.0:  # Dense zone
        adaptive_radius = base_radius * 0.7
    elif avg_density > 2.0:  # Medium density
        adaptive_radius = base_radius
    else:  # Sparse zone
        adaptive_radius = base_radius * 1.5
    
    print(f"  📊 Adaptive radius: {adaptive_radius:.2f}km (density: {avg_density:.2f} users/km²)")
    return adaptive_radius


def micro_cluster_users(user_df, office_lat, office_lon, r_micro_km=1.2):
    """
    A. Micro-clustering: Create tight atomic clusters that should never be split
    PRIORITY: Nearby users first, regardless of direction
    Uses adaptive radius based on local density
    """
    if user_df.empty:
        return []
    
    # Calculate adaptive radius based on local density
    adaptive_radius = adaptive_micro_cluster_radius(user_df, office_lat, office_lon, r_micro_km)
    
    print(f"  🔬 Creating proximity-first micro-clusters with adaptive radius {adaptive_radius:.2f}km...")

    # Prepare coordinates for DBSCAN
    coords = user_df[['latitude', 'longitude']].values

    # Convert to km using lat/lon scaling
    lat_to_km = 111.0
    lon_to_km = 111.0 * math.cos(math.radians(office_lat))

    coords_km = coords.copy()
    coords_km[:, 0] = coords[:, 0] * lat_to_km  # lat to km
    coords_km[:, 1] = coords[:, 1] * lon_to_km  # lon to km

    # Apply DBSCAN clustering with adaptive radius
    dbscan = DBSCAN(eps=adaptive_radius, min_samples=1)  # min_samples=1 to avoid noise points
    cluster_labels = dbscan.fit_predict(coords_km)

    # Group users by cluster
    micro_clusters = []
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:  # Noise points get individual clusters
            noise_indices = np.where(cluster_labels == cluster_id)[0]
            for idx in noise_indices:
                user_data = user_df.iloc[idx]
                micro_clusters.append([{
                    'user_id': user_data['user_id'],
                    'latitude': user_data['latitude'],
                    'longitude': user_data['longitude'],
                    'office_distance': user_data.get('office_distance', 0),
                    'first_name': user_data.get('first_name', ''),
                    'email': user_data.get('email', '')
                }])
        else:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_users = []
            for idx in cluster_indices:
                user_data = user_df.iloc[idx]
                cluster_users.append({
                    'user_id': user_data['user_id'],
                    'latitude': user_data['latitude'],
                    'longitude': user_data['longitude'],
                    'office_distance': user_data.get('office_distance', 0),
                    'first_name': user_data.get('first_name', ''),
                    'email': user_data.get('email', '')
                })
            micro_clusters.append(cluster_users)

    print(f"    📊 Created {len(micro_clusters)} proximity-first micro-clusters")
    for i, mc in enumerate(micro_clusters):
        print(f"      Micro-cluster {i}: {len(mc)} users")

    return micro_clusters


def aggregate_micro_clusters_into_groups(micro_clusters, drivers, office_lat, office_lon, config):
    """
    B. Cluster aggregation → capacity groups using PROXIMITY-FIRST Best-Fit Decreasing
    PRIORITY: Fill nearby clusters first, direction is secondary
    """
    print(f"  📦 Aggregating micro-clusters with PROXIMITY-FIRST approach...")

    if not micro_clusters or drivers.empty:
        return []

    # Sort micro-clusters by size (largest first) for efficient packing
    micro_sizes = sorted(micro_clusters, key=len, reverse=True)

    # Sort vehicles by capacity (largest first)
    vehicles = drivers.sort_values('capacity', ascending=False)

    groups = []  # Each group will be assigned to a vehicle

    for mc in micro_sizes:
        mc_size = len(mc)
        mc_center = calculate_micro_cluster_center(mc)
        mc_bearing = calculate_micro_cluster_bearing(mc, office_lat, office_lon)

        placed = False
        best_group = None
        best_proximity_score = float('inf')

        # PROXIMITY-FIRST: Try to place in existing groups based on proximity
        for group in groups:
            if mc_size <= group['remaining_capacity']:
                # Calculate proximity score (distance between cluster centers)
                group_center = calculate_group_center(group)
                proximity_distance = haversine_distance(
                    mc_center[0], mc_center[1], 
                    group_center[0], group_center[1]
                )

                # Relaxed directional check (only reject if extremely different)
                direction_ok = direction_compatible_relaxed(mc, group, 60, office_lat, office_lon)  # 60° threshold instead of 30°

                if direction_ok and proximity_distance < best_proximity_score:
                    best_proximity_score = proximity_distance
                    best_group = group

        if best_group is not None:
            # Add to the closest compatible group
            best_group['micro_clusters'].append(mc)
            best_group['total_users'].extend(mc)
            best_group['remaining_capacity'] -= mc_size
            best_group['bearings'].append(mc_bearing)
            placed = True
            print(f"    🎯 Added {mc_size} users to existing group (proximity: {best_proximity_score:.1f}km)")

        if not placed:
            # Create new group - find smallest vehicle that can fit this micro-cluster
            suitable_vehicle = None
            for _, vehicle in vehicles.iterrows():
                if vehicle['capacity'] >= mc_size:
                    suitable_vehicle = vehicle
                    break

            if suitable_vehicle is not None:
                new_group = {
                    'target_vehicle': suitable_vehicle,
                    'target_capacity': int(suitable_vehicle['capacity']),
                    'micro_clusters': [mc],
                    'total_users': list(mc),
                    'remaining_capacity': int(suitable_vehicle['capacity']) - mc_size,
                    'bearings': [mc_bearing]
                }
                groups.append(new_group)
                print(f"    🆕 Created new group for {mc_size} users")
            else:
                # No suitable vehicle - this shouldn't happen with proper data
                print(f"    ⚠️ Warning: No vehicle can fit micro-cluster of size {mc_size}")

    print(f"    📊 Created {len(groups)} proximity-optimized capacity groups")
    for i, group in enumerate(groups):
        utilization = len(group['total_users']) / group['target_capacity'] * 100
        print(f"      Group {i}: {len(group['total_users'])}/{group['target_capacity']} users ({utilization:.1f}%)")

    return groups


def calculate_micro_cluster_bearing(micro_cluster, office_lat, office_lon):
    """Calculate average bearing for a micro-cluster"""
    if not micro_cluster:
        return 0.0

    bearings = []
    for user in micro_cluster:
        bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
        bearings.append(bearing)

    # Calculate average bearing (handling circular nature)
    x_sum = sum(math.cos(math.radians(b)) for b in bearings)
    y_sum = sum(math.sin(math.radians(b)) for b in bearings)

    avg_bearing = math.degrees(math.atan2(y_sum, x_sum))
    if avg_bearing < 0:
        avg_bearing += 360

    return avg_bearing


def calculate_micro_cluster_center(micro_cluster):
    """Calculate geographic center of a micro-cluster"""
    if not micro_cluster:
        return (0, 0)

    avg_lat = sum(user['latitude'] for user in micro_cluster) / len(micro_cluster)
    avg_lon = sum(user['longitude'] for user in micro_cluster) / len(micro_cluster)
    return (avg_lat, avg_lon)


def calculate_group_center(group):
    """Calculate geographic center of a group"""
    if not group['total_users']:
        return (0, 0)

    avg_lat = sum(user['latitude'] for user in group['total_users']) / len(group['total_users'])
    avg_lon = sum(user['longitude'] for user in group['total_users']) / len(group['total_users'])
    return (avg_lat, avg_lon)


def direction_compatible_relaxed(micro_cluster, group, threshold_degrees, office_lat, office_lon):
    """
    C. Relaxed directional compatibility check - prioritizes proximity over strict direction
    """
    if not group['bearings']:
        return True  # Empty group is always compatible

    mc_bearing = calculate_micro_cluster_bearing(micro_cluster, office_lat, office_lon)

    # Check bearing spread if we add this micro-cluster
    all_bearings = group['bearings'] + [mc_bearing]

    # Calculate bearing spread
    max_diff = 0
    for i in range(len(all_bearings)):
        for j in range(i + 1, len(all_bearings)):
            diff = bearing_difference(all_bearings[i], all_bearings[j])
            max_diff = max(max_diff, diff)

    return max_diff <= threshold_degrees


def direction_compatible(micro_cluster, group, threshold_degrees, office_lat, office_lon):
    """
    C. Directional compatibility check (strict version for fallback)
    """
    return direction_compatible_relaxed(micro_cluster, group, threshold_degrees, office_lat, office_lon)


def score_driver_for_group(group, driver, office_lat, office_lon, config):
    """
    D. Driver selection scoring function
    """
    group_size = len(group['total_users'])
    vehicle_capacity = int(driver['capacity'])

    if group_size > vehicle_capacity:
        return -float('inf')  # Cannot fit

    # Scoring components
    utilization = group_size / vehicle_capacity
    utilization_score = 1000 * utilization

    waste_penalty = 200 * (vehicle_capacity - group_size)

    # Calculate route distance
    route_distance = calculate_group_route_distance(group['total_users'], driver, office_lat, office_lon)
    distance_penalty = route_distance * 1.0

    # Calculate bearing penalty
    group_bearing_spread = calculate_group_bearing_spread(group, office_lat, office_lon)
    bearing_penalty = max(0, group_bearing_spread - 20) * 5

    # Apply weights
    alpha = config['ALPHA_UTILIZATION']
    beta = config['BETA_WASTE_PENALTY']
    gamma = config['GAMMA_DISTANCE_PENALTY']
    delta = config['DELTA_BEARING_PENALTY']

    score = (alpha * utilization_score - 
             beta * waste_penalty - 
             gamma * distance_penalty - 
             delta * bearing_penalty)

    return score


def calculate_group_route_distance(users, driver, office_lat, office_lon):
    """Calculate total route distance for a group"""
    if not users:
        return 0

    # Simple approximation: driver to center of users + users to office
    center_lat = sum(u['latitude'] for u in users) / len(users)
    center_lon = sum(u['longitude'] for u in users) / len(users)

    driver_to_center = haversine_distance(driver['latitude'], driver['longitude'], center_lat, center_lon)
    center_to_office = haversine_distance(center_lat, center_lon, office_lat, office_lon)

    return driver_to_center + center_to_office


def calculate_group_bearing_spread(group, office_lat, office_lon):
    """Calculate bearing spread for a group"""
    if not group['bearings']:
        return 0

    max_diff = 0
    for i in range(len(group['bearings'])):
        for j in range(i + 1, len(group['bearings'])):
            diff = bearing_difference(group['bearings'][i], group['bearings'][j])
            max_diff = max(max_diff, diff)

    return max_diff


def calculate_centroid(users):
    """Calculate geographic centroid of a list of users"""
    if not users:
        return (0, 0)
    
    avg_lat = sum(u.get('lat', u.get('latitude', 0)) for u in users) / len(users)
    avg_lon = sum(u.get('lng', u.get('longitude', 0)) for u in users) / len(users)
    return (avg_lat, avg_lon)


def is_closer_to_recipient(user, donor_centroid, recipient_centroid, slack=0.15):
    """
    Proximity rule: Only transfer if user is closer to recipient centroid
    Allows small slack for roughly equal distances
    """
    user_lat = user.get('lat', user.get('latitude', 0))
    user_lon = user.get('lng', user.get('longitude', 0))
    
    dist_to_recipient = haversine_distance(user_lat, user_lon, recipient_centroid[0], recipient_centroid[1])
    dist_to_donor = haversine_distance(user_lat, user_lon, donor_centroid[0], donor_centroid[1])
    
    # Allow transfer if closer to recipient or within slack tolerance
    return dist_to_recipient < dist_to_donor * (1 + slack)


def check_donor_minimum(donor_route, min_remaining_factor=2):
    """
    Donor constraint: Never drop donor below minimum remaining threshold
    min_remaining = max(1, capacity - min_remaining_factor)
    """
    current_users = len(donor_route['assigned_users'])
    capacity = donor_route['vehicle_type']
    min_remaining = max(1, capacity - min_remaining_factor)
    
    return current_users > min_remaining


def path_insert_user_into_route(user, route, office_lat, office_lon, max_detour_ratio=0.07, donor_route=None):
    """
    F. Path-aware insertion with strict detour limits and proximity checks
    """
    if route['vehicle_type'] <= len(route['assigned_users']):
        return False, route  # No capacity

    # Check proximity rule if donor route provided
    if donor_route is not None:
        donor_centroid = calculate_centroid(donor_route['assigned_users'])
        recipient_centroid = calculate_centroid(route['assigned_users'])
        
        if not is_closer_to_recipient(user, donor_centroid, recipient_centroid, slack=0.15):
            return False, route  # User not closer to recipient

        # Check donor minimum remaining constraint
        if not check_donor_minimum(donor_route, min_remaining_factor=2):
            return False, route  # Would violate donor minimum

    # Calculate current route distance
    current_distance = calculate_route_total_distance(route, office_lat, office_lon)

    # Calculate user bearing
    user_bearing = calculate_bearing(office_lat, office_lon, user.get('latitude', user.get('lat', 0)), 
                                     user.get('longitude', user.get('lng', 0)))
    route_avg_bearing = calculate_average_bearing_improved(route, office_lat, office_lon)

    bearing_diff = bearing_difference(user_bearing, route_avg_bearing)

    # Soft penalty instead of hard reject - use scoring
    bearing_penalty = bearing_diff * 0.5  # Penalty per degree

    # Try inserting user at different positions
    best_position = None
    best_distance = float('inf')

    current_users = route['assigned_users'].copy()

    for i in range(len(current_users) + 1):
        # Create test route with user inserted at position i
        test_users = current_users[:i] + [user] + current_users[i:]
        test_route = route.copy()
        test_route['assigned_users'] = test_users

        # Calculate new distance
        new_distance = calculate_route_total_distance(test_route, office_lat, office_lon)

        if new_distance < best_distance:
            best_distance = new_distance
            best_position = i

    # Calculate score-based acceptance with soft penalties
    if current_distance > 0:
        added_distance = best_distance - current_distance
        detour_ratio = added_distance / current_distance
        
        # Capacity gain (positive)
        capacity_gain = 1.0
        
        # Combined score with configurable weights
        Wc = 10.0  # Capacity weight
        Wd = 1.0   # Distance weight
        Wb = 0.3   # Bearing weight
        
        score = capacity_gain * Wc - added_distance * Wd - bearing_penalty * Wb
        
        # Accept if score is positive and detour is within extended tolerance
        if score > 0 and detour_ratio <= max_detour_ratio * 1.5:  # Extended tolerance with scoring
            # Accept insertion
            new_route = route.copy()
            new_users = current_users[:best_position] + [user] + current_users[best_position:]
            new_route['assigned_users'] = new_users
            new_route = optimize_route_sequence_improved(new_route, office_lat, office_lon)
            return True, new_route

    return False, route


def calculate_route_total_distance(route, office_lat, office_lon):
    """Calculate total distance for a route"""
    if not route['assigned_users']:
        return 0

    total_distance = 0

    # Driver to first pickup
    first_user = route['assigned_users'][0]
    total_distance += haversine_distance(
        route['latitude'], route['longitude'],
        first_user['lat'], first_user['lng']
    )

    # Between pickups
    for i in range(len(route['assigned_users']) - 1):
        current_user = route['assigned_users'][i]
        next_user = route['assigned_users'][i + 1]
        total_distance += haversine_distance(
            current_user['lat'], current_user['lng'],
            next_user['lat'], next_user['lng']
        )

    # Last pickup to office
    last_user = route['assigned_users'][-1]
    total_distance += haversine_distance(
        last_user['lat'], last_user['lng'],
        office_lat, office_lon
    )

    return total_distance


def create_route_from_group(group, driver, office_lat, office_lon):
    """Create a route structure from a capacity group and driver"""
    route = {
        'driver_id': str(driver['driver_id']),
        'vehicle_type': int(driver['capacity']),
        'latitude': float(driver['latitude']),
        'longitude': float(driver['longitude']),
        'assigned_users': [],
        'total_distance': 0,
        'avg_bearing': 0,
        'bearing_spread': 0,
        'direction_consistency': 0,
        'turning_score': 0
    }

    # Add users from group
    for user in group['total_users']:
        user_dict = {
            'user_id': str(user['user_id']),
            'lat': float(user['latitude']),
            'lng': float(user['longitude']),
            'latitude': float(user['latitude']),
            'longitude': float(user['longitude']),
            'office_distance': float(user.get('office_distance', 0)),
            'first_name': str(user.get('first_name', '')),
            'email': str(user.get('email', ''))
        }
        route['assigned_users'].append(user_dict)

    # Optimize route sequence
    route = optimize_route_sequence_improved(route, office_lat, office_lon)
    update_route_metrics_improved(route, office_lat, office_lon)

    return route


def split_micro_cluster_by_bearing(micro_cluster, max_capacity):
    """
    E. Micro-cluster splitting by bearing (rare operation)
    """
    if len(micro_cluster) <= max_capacity:
        return [micro_cluster]

    print(f"    ⚠️ Splitting micro-cluster of size {len(micro_cluster)} (max capacity: {max_capacity})")

    # Calculate bearings for all users in micro-cluster
    user_bearings = []
    for user in micro_cluster:
        bearing = calculate_bearing(OFFICE_LAT, OFFICE_LON, user['latitude'], user['longitude'])
        user_bearings.append((bearing, user))

    # Sort by bearing
    user_bearings.sort(key=lambda x: x[0])

    # Split into chunks of max_capacity
    split_clusters = []
    for i in range(0, len(user_bearings), max_capacity):
        chunk = [user for _, user in user_bearings[i:i + max_capacity]]
        split_clusters.append(chunk)

    return split_clusters


# =============================================================================
# MAIN BLUEPRINT ASSIGNMENT FUNCTION
# =============================================================================

def assign_drivers_blueprint_approach(user_df, driver_df, office_lat, office_lon):
    """
    Main assignment function following the exact blueprint ordering
    """
    print("🚗 Step 3: BLUEPRINT capacity assignment with atomic micro-clusters...")
    
    start_time = time.time()
    routes = []
    used_driver_ids = set()

    # STEP 1: Micro-cluster users into tight atomic groups
    micro_clusters = micro_cluster_users(user_df, office_lat, office_lon, 
                                       _config['MICRO_CLUSTER_RADIUS_KM'])

    if not micro_clusters:
        print("  ⚠️ No micro-clusters created")
        return routes, set()

    # STEP 2: Split oversized micro-clusters if needed
    max_vehicle_capacity = driver_df['capacity'].max() if not driver_df.empty else 10
    final_micro_clusters = []

    for mc in micro_clusters:
        if len(mc) > max_vehicle_capacity:
            split_clusters = split_micro_cluster_by_bearing(mc, max_vehicle_capacity)
            final_micro_clusters.extend(split_clusters)
        else:
            final_micro_clusters.append(mc)

    print(f"  📊 After splitting: {len(final_micro_clusters)} micro-clusters")

    # STEP 3: Aggregate micro-clusters into capacity groups using bin-packing
    available_drivers = driver_df[~driver_df['driver_id'].isin(used_driver_ids)]
    capacity_groups = aggregate_micro_clusters_into_groups(
        final_micro_clusters, available_drivers, office_lat, office_lon, _config)

    # STEP 4: Match groups to drivers with capacity-aware scoring
    assigned_user_ids = set()

    for group in capacity_groups:
        if available_drivers.empty:
            break

        best_driver = None
        best_score = -float('inf')

        for _, driver in available_drivers.iterrows():
            if driver['driver_id'] in used_driver_ids:
                continue

            score = score_driver_for_group(group, driver, office_lat, office_lon, _config)

            if score > best_score:
                best_score = score
                best_driver = driver

        if best_driver is not None:
            # Create route
            route = create_route_from_group(group, best_driver, office_lat, office_lon)

            if route:
                routes.append(route)
                used_driver_ids.add(best_driver['driver_id'])

                for user in route['assigned_users']:
                    assigned_user_ids.add(user['user_id'])

                utilization = len(route['assigned_users']) / route['vehicle_type'] * 100
                print(f"  ✅ Driver {best_driver['driver_id']}: {len(route['assigned_users'])}/{route['vehicle_type']} seats ({utilization:.1f}%) - Atomic grouping")

        # Update available drivers
        available_drivers = driver_df[~driver_df['driver_id'].isin(used_driver_ids)]

    # STEP 5: Two-pass assignment strategy
    print("  🛣️ Pass A: Capacity-first strict assignment (proximity + low bearing penalty)...")
    
    remaining_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]

    # Pass A: Strict - only users closer to recipient with low bearing penalty
    for _, user in remaining_users.iterrows():
        user_dict = {
            'user_id': str(user['user_id']),
            'lat': float(user['latitude']),
            'lng': float(user['longitude']),
            'latitude': float(user['latitude']),
            'longitude': float(user['longitude']),
            'office_distance': float(user.get('office_distance', 0)),
            'first_name': str(user.get('first_name', '')),
            'email': str(user.get('email', ''))
        }

        # Find best route with strict criteria
        best_route = None
        best_score = -float('inf')
        
        for route in routes:
            if len(route['assigned_users']) >= route['vehicle_type']:
                continue
            
            # Calculate proximity to route centroid
            route_centroid = calculate_centroid(route['assigned_users'])
            dist_to_route = haversine_distance(
                user['latitude'], user['longitude'],
                route_centroid[0], route_centroid[1]
            )
            
            # Calculate bearing difference
            user_bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
            route_bearing = calculate_average_bearing_improved(route, office_lat, office_lon)
            bearing_diff = bearing_difference(user_bearing, route_bearing)
            
            # Strict criteria: low bearing penalty
            if bearing_diff <= 30:  # Strict bearing threshold for Pass A
                capacity_score = (route['vehicle_type'] - len(route['assigned_users'])) * 10
                proximity_score = 1.0 / (1.0 + dist_to_route)
                bearing_score = 1.0 - (bearing_diff / 30)
                
                score = capacity_score + proximity_score * 5 + bearing_score * 2
                
                if score > best_score:
                    best_score = score
                    best_route = route

        if best_route:
            success, new_route = path_insert_user_into_route(
                user_dict, best_route, office_lat, office_lon, 
                _config['PATH_INSERT_MAX_DETOUR_RATIO'])

            if success:
                best_route.clear()
                best_route.update(new_route)
                assigned_user_ids.add(user['user_id'])
                print(f"    ✅ Pass A: Inserted user {user['user_id']} (strict criteria)")

    # Pass B: Bearing-fallback with lenient criteria
    print("  🛣️ Pass B: Bearing-fallback lenient assignment (same direction ±30°)...")
    
    remaining_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]
    
    for _, user in remaining_users.iterrows():
        user_dict = {
            'user_id': str(user['user_id']),
            'lat': float(user['latitude']),
            'lng': float(user['longitude']),
            'latitude': float(user['latitude']),
            'longitude': float(user['longitude']),
            'office_distance': float(user.get('office_distance', 0)),
            'first_name': str(user.get('first_name', '')),
            'email': str(user.get('email', ''))
        }

        # Find routes in same bearing sector with available capacity
        user_bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
        
        best_route = None
        best_score = -float('inf')
        
        for route in routes:
            if len(route['assigned_users']) >= route['vehicle_type']:
                continue
            
            route_bearing = calculate_average_bearing_improved(route, office_lat, office_lon)
            bearing_diff = bearing_difference(user_bearing, route_bearing)
            
            # Lenient bearing threshold for Pass B (±30°)
            if bearing_diff <= 45:
                # Score with soft penalties
                route_centroid = calculate_centroid(route['assigned_users'])
                dist_to_route = haversine_distance(
                    user['latitude'], user['longitude'],
                    route_centroid[0], route_centroid[1]
                )
                
                capacity_gain = 1.0
                distance_penalty = dist_to_route * 0.5
                bearing_penalty = bearing_diff * 0.2
                
                score = capacity_gain * 10 - distance_penalty - bearing_penalty
                
                if score > best_score:
                    best_score = score
                    best_route = route

        if best_route and best_score > 0:
            success, new_route = path_insert_user_into_route(
                user_dict, best_route, office_lat, office_lon, 
                _config['PATH_INSERT_MAX_DETOUR_RATIO'] * 1.3)  # Slightly more lenient

            if success:
                best_route.clear()
                best_route.update(new_route)
                assigned_user_ids.add(user['user_id'])
                print(f"    ✅ Pass B: Inserted user {user['user_id']} (lenient bearing)")
            else:
                print(f"    ⏭️ Could not insert user {user['user_id']} - detour constraints")
    
    # STEP 3: Allow controlled overflow for large micro-clusters
    print("  ⚡ Controlled overflow pass for remaining large groups...")

    remaining_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]

    if not remaining_users.empty:
        # Group remaining users into micro-clusters
        overflow_clusters = micro_cluster_users(remaining_users, office_lat, office_lon, 
                                               _config['MICRO_CLUSTER_RADIUS_KM'])

        for cluster in overflow_clusters:
            if len(cluster) >= 3:  # Only for groups of 3+
                # Find nearest route
                cluster_center = calculate_micro_cluster_center(cluster)
                best_route = None
                min_distance = float('inf')

                for route in routes:
                    route_center = calculate_route_center_improved(route)
                    distance = haversine_distance(
                        cluster_center[0], cluster_center[1],
                        route_center[0], route_center[1]
                    )

                    if distance < min_distance:
                        min_distance = distance
                        best_route = route

                # Allow temporary overflow
                if best_route and min_distance <= 4.0:  # Within 4km
                    overflow_count = len(cluster)
                    for user in cluster:
                        user_dict = {
                            'user_id': str(user['user_id']),
                            'lat': float(user['latitude']),
                            'lng': float(user['longitude']),
                            'latitude': float(user['latitude']),
                            'longitude': float(user['longitude']),
                            'office_distance': float(user.get('office_distance', 0)),
                            'first_name': str(user.get('first_name', '')),
                            'email': str(user.get('email', ''))
                        }
                        best_route['assigned_users'].append(user_dict)
                        assigned_user_ids.add(user['user_id'])
                        best_route['overflow'] = best_route.get('overflow', 0) + 1

                    print(f"    ⚠️ Overflow: Added {overflow_count} users to route {best_route['driver_id']} (temp overflow)")

    # STEP 4: Strong local swap optimization
    print("  🔄 Local swap optimization pass...")

    max_swap_iterations = _config.get('MAX_SWAP_ITERATIONS', 5)
    for iteration in range(max_swap_iterations):
        improvements_made = 0

        for i, r1 in enumerate(routes):
            for j, r2 in enumerate(routes):
                if j <= i:
                    continue

                # Try swapping users between routes to improve utilization
                for u1 in r1['assigned_users'][:]:
                    for u2 in r2['assigned_users'][:]:
                        # Calculate swap benefit
                        old_util = (len(r1['assigned_users']) / r1['vehicle_type'] + 
                                   len(r2['assigned_users']) / r2['vehicle_type']) / 2

                        # Simulate swap
                        new_r1_size = len(r1['assigned_users'])
                        new_r2_size = len(r2['assigned_users'])

                        if new_r1_size <= r1['vehicle_type'] and new_r2_size <= r2['vehicle_type']:
                            new_util = (new_r1_size / r1['vehicle_type'] + 
                                       new_r2_size / r2['vehicle_type']) / 2

                            if new_util > old_util + 0.05:  # 5% improvement threshold
                                # Perform swap
                                r1['assigned_users'].remove(u1)
                                r2['assigned_users'].remove(u2)
                                r1['assigned_users'].append(u2)
                                r2['assigned_users'].append(u1)
                                improvements_made += 1
                                break

        if improvements_made == 0:
            break
        print(f"    🔄 Iteration {iteration + 1}: {improvements_made} swaps made")

    # STEP 5: Global optimization - fix overflow and rebalance
    print("  🌐 Global optimization - rebalancing overflow...")

    overflow_routes = [r for r in routes if r.get('overflow', 0) > 0]
    for route in overflow_routes:
        overflow_count = route['overflow']
        # Try to move overflow users to other routes
        for _ in range(overflow_count):
            if len(route['assigned_users']) <= route['vehicle_type']:
                break

            # Find user farthest from route center
            route_center = calculate_route_center_improved(route)
            farthest_user = None
            max_dist = 0

            for user in route['assigned_users']:
                dist = haversine_distance(
                    user['lat'], user['lng'],
                    route_center[0], route_center[1]
                )
                if dist > max_dist:
                    max_dist = dist
                    farthest_user = user

            if farthest_user:
                # Try to insert into another route
                inserted = False
                for other_route in routes:
                    if other_route['driver_id'] == route['driver_id']:
                        continue

                    success, new_route = path_insert_user_into_route(
                        farthest_user, other_route, office_lat, office_lon, 0.3)

                    if success:
                        # Remove user by creating new list without the user
                        route['assigned_users'] = [u for u in route['assigned_users'] if u['user_id'] != farthest_user['user_id']]
                        other_route.clear()
                        other_route.update(new_route)
                        inserted = True
                        print(f"    ✅ Moved overflow user from {route['driver_id']} to {other_route['driver_id']}")
                        break

                if not inserted:
                    break


    # STEP 6: Final local optimization pass
    print("  🔧 Final local optimization: pairwise swaps and limited LNS...")
    
    # Recompute all centroids before optimization
    for route in routes:
        route['centroid'] = calculate_centroid(route['assigned_users'])
    
    # Pairwise swap optimization
    improvements = 0
    for iteration in range(3):  # Limited iterations
        swap_made = False
        
        for i, route1 in enumerate(routes):
            for j, route2 in enumerate(routes):
                if j <= i:
                    continue
                
                # Create copies of user lists to avoid modification during iteration
                users1 = route1['assigned_users'][:]
                users2 = route2['assigned_users'][:]
                
                # Try swapping users between routes
                for user1 in users1:
                    for user2 in users2:
                        # Verify users are still in their respective routes
                        if user1 not in route1['assigned_users'] or user2 not in route2['assigned_users']:
                            continue
                        
                        # Calculate current objective
                        old_util = (len(route1['assigned_users']) / route1['vehicle_type'] + 
                                   len(route2['assigned_users']) / route2['vehicle_type']) / 2
                        
                        old_dist1 = calculate_route_total_distance(route1, office_lat, office_lon)
                        old_dist2 = calculate_route_total_distance(route2, office_lat, office_lon)
                        old_total_dist = old_dist1 + old_dist2
                        
                        # Check proximity constraints before swap
                        if not is_closer_to_recipient(user1, route1['centroid'], route2['centroid'], slack=0.2):
                            continue
                        if not is_closer_to_recipient(user2, route2['centroid'], route1['centroid'], slack=0.2):
                            continue
                        
                        # Check donor minimums
                        if len(route1['assigned_users']) <= max(1, route1['vehicle_type'] - 2):
                            continue
                        if len(route2['assigned_users']) <= max(1, route2['vehicle_type'] - 2):
                            continue
                        
                        # Simulate swap - create new lists
                        new_users1 = [u for u in route1['assigned_users'] if u['user_id'] != user1['user_id']]
                        new_users1.append(user2)
                        
                        new_users2 = [u for u in route2['assigned_users'] if u['user_id'] != user2['user_id']]
                        new_users2.append(user1)
                        
                        # Temporarily update routes
                        route1['assigned_users'] = new_users1
                        route2['assigned_users'] = new_users2
                        
                        # Recalculate centroids
                        route1['centroid'] = calculate_centroid(route1['assigned_users'])
                        route2['centroid'] = calculate_centroid(route2['assigned_users'])
                        
                        # Calculate new objective
                        new_util = (len(route1['assigned_users']) / route1['vehicle_type'] + 
                                   len(route2['assigned_users']) / route2['vehicle_type']) / 2
                        
                        new_dist1 = calculate_route_total_distance(route1, office_lat, office_lon)
                        new_dist2 = calculate_route_total_distance(route2, office_lat, office_lon)
                        new_total_dist = new_dist1 + new_dist2
                        
                        # Check if swap improves global objective
                        util_improvement = new_util - old_util
                        dist_improvement = old_total_dist - new_total_dist
                        
                        # Combined improvement (prioritize utilization)
                        improvement = util_improvement * 10 + dist_improvement * 0.5
                        
                        if improvement > 0.05:  # Threshold for accepting swap
                            swap_made = True
                            improvements += 1
                            print(f"    🔄 Swap improved: util +{util_improvement:.3f}, dist -{dist_improvement:.2f}km")
                        else:
                            # Revert swap - restore original lists
                            route1['assigned_users'] = users1[:]
                            route2['assigned_users'] = users2[:]
                            route1['centroid'] = calculate_centroid(route1['assigned_users'])
                            route2['centroid'] = calculate_centroid(route2['assigned_users'])
        
        if not swap_made:
            break
    
    print(f"    ✅ Local optimization: {improvements} beneficial swaps made")
    
    # Limited LNS-style remove & reinsert (3-6 micro-clusters)
    print("  🔧 LNS-style remove & reinsert for local minima escape...")
    
    # Identify underutilized routes for reinsert candidates
    underutil_routes = [r for r in routes if len(r['assigned_users']) / r['vehicle_type'] < 0.7]
    
    if len(underutil_routes) >= 2:
        # Remove small clusters from underutilized routes
        removed_clusters = []
        
        for route in underutil_routes[:3]:  # Limit to 3 routes
            if len(route['assigned_users']) > 2:
                # Remove smallest cluster (last 2-3 users) - create new list
                cluster_size = min(3, len(route['assigned_users']) - 1)
                removed = route['assigned_users'][-cluster_size:]
                route['assigned_users'] = route['assigned_users'][:-cluster_size]  # Slice creates new list
                removed_clusters.append(removed)
                route['centroid'] = calculate_centroid(route['assigned_users'])
        
        # Reinsert removed clusters optimally
        for cluster in removed_clusters:
            best_route = None
            best_score = -float('inf')
            
            for route in routes:
                if len(route['assigned_users']) + len(cluster) > route['vehicle_type']:
                    continue
                
                # Calculate insertion score
                route_centroid = calculate_centroid(route['assigned_users'])
                cluster_centroid = calculate_centroid(cluster)
                
                dist = haversine_distance(cluster_centroid[0], cluster_centroid[1],
                                         route_centroid[0], route_centroid[1])
                
                capacity_score = (route['vehicle_type'] - len(route['assigned_users'])) / route['vehicle_type']
                proximity_score = 1.0 / (1.0 + dist)
                
                score = capacity_score * 10 + proximity_score * 5
                
                if score > best_score:
                    best_score = score
                    best_route = route
            
            if best_route:
                best_route['assigned_users'].extend(cluster)
                best_route['centroid'] = calculate_centroid(best_route['assigned_users'])
                print(f"    🔄 Reinserted cluster of {len(cluster)} users")
    
    # STEP 7: Blueprint final merge with adaptive thresholds
    routes = final_merge_blueprint(routes, _config, office_lat, office_lon)

    # STEP 8: Metric checks & safety validation
    print("  📊 Computing final metrics and safety checks...")
    
    total_capacity = sum(r['vehicle_type'] for r in routes)
    total_assigned = sum(len(r['assigned_users']) for r in routes)
    overall_utilization = (total_assigned / total_capacity * 100) if total_capacity > 0 else 0
    
    # Calculate average detour per route
    total_detour = 0
    route_count = 0
    
    for route in routes:
        if not route['assigned_users']:
            continue
        
        # Simple detour: route distance vs direct distance
        route_dist = calculate_route_total_distance(route, office_lat, office_lon)
        
        # Direct distance from driver to office via centroid
        centroid = calculate_centroid(route['assigned_users'])
        direct_dist = (haversine_distance(route['latitude'], route['longitude'], 
                                         centroid[0], centroid[1]) +
                      haversine_distance(centroid[0], centroid[1], office_lat, office_lon))
        
        detour = route_dist - direct_dist if direct_dist > 0 else 0
        total_detour += detour
        route_count += 1
    
    avg_detour = total_detour / route_count if route_count > 0 else 0
    
    # Calculate average bearing spread
    total_bearing_spread = 0
    for route in routes:
        if len(route['assigned_users']) > 1:
            bearings = []
            for user in route['assigned_users']:
                bearing = calculate_bearing(office_lat, office_lon, 
                                          user.get('lat', user.get('latitude', 0)),
                                          user.get('lng', user.get('longitude', 0)))
                bearings.append(bearing)
            
            max_spread = 0
            for i in range(len(bearings)):
                for j in range(i + 1, len(bearings)):
                    spread = bearing_difference(bearings[i], bearings[j])
                    max_spread = max(max_spread, spread)
            
            total_bearing_spread += max_spread
    
    avg_bearing_spread = total_bearing_spread / len(routes) if routes else 0
    
    # Safety checks
    print(f"  📊 Final Metrics:")
    print(f"    • Overall utilization: {overall_utilization:.1f}%")
    print(f"    • Average detour: {avg_detour:.2f}km")
    print(f"    • Average bearing spread: {avg_bearing_spread:.1f}°")
    
    # Business limits validation
    DETOUR_LIMIT = 8.0  # km
    BEARING_SPREAD_LIMIT = 90  # degrees
    MIN_UTILIZATION = 65  # percent
    
    warnings = []
    if avg_detour > DETOUR_LIMIT:
        warnings.append(f"⚠️ Average detour {avg_detour:.2f}km exceeds limit {DETOUR_LIMIT}km")
    if avg_bearing_spread > BEARING_SPREAD_LIMIT:
        warnings.append(f"⚠️ Bearing spread {avg_bearing_spread:.1f}° exceeds limit {BEARING_SPREAD_LIMIT}°")
    if overall_utilization < MIN_UTILIZATION:
        warnings.append(f"⚠️ Utilization {overall_utilization:.1f}% below minimum {MIN_UTILIZATION}%")
    
    if warnings:
        print("  ⚠️ Safety warnings:")
        for warning in warnings:
            print(f"    {warning}")
    else:
        print("  ✅ All safety checks passed")

    # Final user count verification
    total_users_in_api = len(user_df)
    users_assigned = len(assigned_user_ids)

    print(f"✅ Blueprint capacity optimization complete in {time.time() - start_time:.2f}s")
    print(f"📊 Final routes: {len(routes)}")
    print(f"🎯 Users assigned: {users_assigned}")

    return routes, assigned_user_ids

    


def final_merge_blueprint(routes, config, office_lat, office_lon):
    """
    G. Final merge with geographic proximity priority
    """
    print("🔄 Step 6: Blueprint final merge with geographic priority...")

    if len(routes) < 2:
        return routes

    merged_routes = []
    used = set()

    for i, r1 in enumerate(routes):
        if i in used:
            continue

        best_merge = None
        best_score = float('-inf')

        for j, r2 in enumerate(routes):
            if j <= i or j in used:
                continue

            # Check merge conditions
            total_users = len(r1['assigned_users']) + len(r2['assigned_users'])
            max_capacity = max(r1['vehicle_type'], r2['vehicle_type'])

            if total_users > max_capacity:
                continue

            # Calculate centroid distance (PRIORITY)
            c1 = calculate_route_center_improved(r1)
            c2 = calculate_route_center_improved(r2)
            centroid_distance = haversine_distance(c1[0], c1[1], c2[0], c2[1])

            if centroid_distance > config['FINAL_MERGE_DISTANCE_THRESHOLD']:
                continue

            # Check bearing difference (more lenient)
            b1 = calculate_average_bearing_improved(r1, office_lat, office_lon)
            b2 = calculate_average_bearing_improved(r2, office_lat, office_lon)
            bearing_diff = bearing_difference(b1, b2)

            if bearing_diff > config['FINAL_MERGE_BEARING_THRESHOLD']:
                continue

            # Calculate merge score prioritizing proximity and utilization
            merged_utilization = total_users / max_capacity
            distance_score = 1.0 / (1.0 + centroid_distance)  # Higher score for closer routes
            utilization_score = merged_utilization
            bearing_score = 1.0 - (bearing_diff / config['FINAL_MERGE_BEARING_THRESHOLD'])

            # Weighted score: 50% proximity, 40% utilization, 10% direction
            merge_score = (distance_score * 0.5 + utilization_score * 0.4 + bearing_score * 0.1)

            if merge_score > best_score:
                best_merge = j
                best_score = merge_score

        if best_merge is not None:
            # Perform merge
            r2 = routes[best_merge]
            better_route = r1 if r1['vehicle_type'] >= r2['vehicle_type'] else r2

            merged_route = better_route.copy()
            merged_route['assigned_users'] = r1['assigned_users'] + r2['assigned_users']
            merged_route = optimize_route_sequence_improved(merged_route, office_lat, office_lon)

            merged_routes.append(merged_route)
            used.add(i)
            used.add(best_merge)

            utilization_pct = len(merged_route['assigned_users']) / merged_route['vehicle_type'] * 100
            print(f"  🔗 Blueprint merge: routes {r1['driver_id']} + {r2['driver_id']} = {len(merged_route['assigned_users'])}/{merged_route['vehicle_type']} seats ({utilization_pct:.1f}%)")
        else:
            merged_routes.append(r1)
            used.add(i)

    total_seats = sum(r['vehicle_type'] for r in merged_routes)
    total_users = sum(len(r['assigned_users']) for r in merged_routes)
    overall_utilization = (total_users / total_seats * 100) if total_seats > 0 else 0

    print(f"  🎯 Blueprint merge complete: {len(routes)} → {len(merged_routes)} routes")
    print(f"  📊 Final utilization: {total_users}/{total_seats} ({overall_utilization:.1f}%)")

    return merged_routes


# =============================================================================
# MAIN ASSIGNMENT FUNCTION FOR BLUEPRINT CAPACITY OPTIMIZATION
# =============================================================================

def run_assignment_capacity(source_id: str, parameter: int = 1, string_param: str = "", choice: str = ""):
    """Main entry point for blueprint capacity optimization assignment"""
    return run_assignment_capacity_internal(source_id, parameter, string_param, choice)

def run_assignment_capacity_internal(source_id: str, parameter: int = 1, string_param: str = "", choice: str = ""):
    """
    Main assignment function implementing the blueprint approach:
    - Micro-clustering for atomic groups
    - Capacity-aware group aggregation
    - Directional validation throughout
    - Path-aware insertion with strict limits
    """
    start_time = time.time()

    # Reload configuration for capacity optimization
    global _config
    _config = load_and_validate_config()

    print(f"🚀 Starting BLUEPRINT CAPACITY OPTIMIZATION for source_id: {source_id}")
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
                    "method": "Blueprint - No Users",
                    "clusters": 0
                },
                "optimization_mode": "blueprint_capacity_optimization",
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
                    "method": "Blueprint - No Drivers",
                    "clusters": 0
                },
                "optimization_mode": "blueprint_capacity_optimization",
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

        # BLUEPRINT ASSIGNMENT EXECUTION
        
        clustering_results = {
            "method": "blueprint_micro_clustering",
            "clusters": 0
        }

        # STEP 1: Blueprint assignment with micro-clustering
        routes, assigned_user_ids = assign_drivers_blueprint_approach(
            user_df, driver_df, office_lat, office_lon)

        clustering_results["clusters"] = len(routes)

        # STEP 2: Handle remaining unassigned users with geographic-aware residual grouping
        unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids)]
        used_driver_ids = {route['driver_id'] for route in routes}
        available_drivers_df = driver_df[~driver_df['driver_id'].isin(used_driver_ids)]

        if not unassigned_users_df.empty and not available_drivers_df.empty:
            print("  🔄 Geographic-aware residual grouping for remaining users...")

            # Try to insert remaining users into existing routes first (geographic priority)
            for _, user in unassigned_users_df.iterrows():
                user_dict = {
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'latitude': float(user['latitude']),
                    'longitude': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0)),
                    'first_name': str(user.get('first_name', '')),
                    'email': str(user.get('email', ''))
                }

                # Find closest route by geographic distance
                best_route = None
                min_distance = float('inf')

                for route in routes:
                    if len(route['assigned_users']) >= route['vehicle_type']:
                        continue

                    # Calculate distance to route center
                    route_center = calculate_group_center({'total_users': route['assigned_users']})
                    distance = haversine_distance(
                        user['latitude'], user['longitude'],
                        route_center[0], route_center[1]
                    )

                    if distance < min_distance and distance <= 3.0:  # Within 3km
                        min_distance = distance
                        best_route = route

                if best_route:
                    # Try to insert into closest route
                    success, new_route = path_insert_user_into_route(
                        user_dict, best_route, office_lat, office_lon, 
                        _config['PATH_INSERT_MAX_DETOUR_RATIO'])

                    if success:
                        best_route.clear()
                        best_route.update(new_route)
                        assigned_user_ids.add(user['user_id'])
                        print(f"    📍 Inserted user {user['user_id']} into geographic-nearest route")

            # Update unassigned users after geographic insertion
            unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids)]

            # Create new routes for truly remaining users
            if not unassigned_users_df.empty:
                remaining_routes, additional_assigned = handle_remaining_users_improved(
                    unassigned_users_df, available_drivers_df, routes, office_lat, office_lon)

                routes.extend(remaining_routes)
                assigned_user_ids.update(additional_assigned)

        # STEP 3: Allow controlled overflow for large micro-clusters
        print("  ⚡ Controlled overflow pass for remaining large groups...")

        remaining_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]

        if not remaining_users.empty:
            # Group remaining users into micro-clusters
            overflow_clusters = micro_cluster_users(remaining_users, office_lat, office_lon, 
                                                   _config['MICRO_CLUSTER_RADIUS_KM'])

            for cluster in overflow_clusters:
                if len(cluster) >= 3:  # Only for groups of 3+
                    # Find nearest route
                    cluster_center = calculate_micro_cluster_center(cluster)
                    best_route = None
                    min_distance = float('inf')

                    for route in routes:
                        route_center = calculate_route_center_improved(route)
                        distance = haversine_distance(
                            cluster_center[0], cluster_center[1],
                            route_center[0], route_center[1]
                        )

                        if distance < min_distance:
                            min_distance = distance
                            best_route = route

                    # Allow temporary overflow
                    if best_route and min_distance <= 4.0:  # Within 4km
                        overflow_count = len(cluster)
                        for user in cluster:
                            user_dict = {
                                'user_id': str(user['user_id']),
                                'lat': float(user['latitude']),
                                'lng': float(user['longitude']),
                                'latitude': float(user['latitude']),
                                'longitude': float(user['longitude']),
                                'office_distance': float(user.get('office_distance', 0)),
                                'first_name': str(user.get('first_name', '')),
                                'email': str(user.get('email', ''))
                            }
                            best_route['assigned_users'].append(user_dict)
                            assigned_user_ids.add(user['user_id'])
                            best_route['overflow'] = best_route.get('overflow', 0) + 1

                        print(f"    ⚠️ Overflow: Added {overflow_count} users to route {best_route['driver_id']} (temp overflow)")

        # STEP 4: Strong local swap optimization
        print("  🔄 Local swap optimization pass...")

        max_swap_iterations = _config.get('MAX_SWAP_ITERATIONS', 5)
        for iteration in range(max_swap_iterations):
            improvements_made = 0

            for i, r1 in enumerate(routes):
                for j, r2 in enumerate(routes):
                    if j <= i:
                        continue

                    # Try swapping users between routes to improve utilization
                    for u1 in r1['assigned_users'][:]:
                        for u2 in r2['assigned_users'][:]:
                            # Calculate swap benefit
                            old_util = (len(r1['assigned_users']) / r1['vehicle_type'] + 
                                       len(r2['assigned_users']) / r2['vehicle_type']) / 2

                            # Simulate swap
                            new_r1_size = len(r1['assigned_users'])
                            new_r2_size = len(r2['assigned_users'])

                            if new_r1_size <= r1['vehicle_type'] and new_r2_size <= r2['vehicle_type']:
                                new_util = (new_r1_size / r1['vehicle_type'] + 
                                           new_r2_size / r2['vehicle_type']) / 2

                                if new_util > old_util + 0.05:  # 5% improvement threshold
                                    # Perform swap - create new lists to avoid remove errors
                                    r1['assigned_users'] = [u for u in r1['assigned_users'] if u['user_id'] != u1['user_id']]
                                    r1['assigned_users'].append(u2)
                                    r2['assigned_users'] = [u for u in r2['assigned_users'] if u['user_id'] != u2['user_id']]
                                    r2['assigned_users'].append(u1)
                                    improvements_made += 1
                                    break

            if improvements_made == 0:
                break
            print(f"    🔄 Iteration {iteration + 1}: {improvements_made} swaps made")

        # STEP 5: Global optimization - fix overflow and rebalance
        print("  🌐 Global optimization - rebalancing overflow...")

        overflow_routes = [r for r in routes if r.get('overflow', 0) > 0]
        for route in overflow_routes:
            overflow_count = route['overflow']
            # Try to move overflow users to other routes
            for _ in range(overflow_count):
                if len(route['assigned_users']) <= route['vehicle_type']:
                    break

                # Find user farthest from route center
                route_center = calculate_route_center_improved(route)
                farthest_user = None
                max_dist = 0

                for user in route['assigned_users']:
                    dist = haversine_distance(
                        user['lat'], user['lng'],
                        route_center[0], route_center[1]
                    )
                    if dist > max_dist:
                        max_dist = dist
                        farthest_user = user

                if farthest_user:
                    # Try to insert into another route
                    inserted = False
                    for other_route in routes:
                        if other_route['driver_id'] == route['driver_id']:
                            continue

                        success, new_route = path_insert_user_into_route(
                            farthest_user, other_route, office_lat, office_lon, 0.3)

                        if success:
                            route['assigned_users'].remove(farthest_user)
                            other_route.clear()
                            other_route.update(new_route)
                            inserted = True
                            print(f"    ✅ Moved overflow user from {route['driver_id']} to {other_route['driver_id']}")
                            break

                    if not inserted:
                        break

        # STEP 6: Blueprint final merge with adaptive thresholds
        routes = final_merge_blueprint(routes, _config, office_lat, office_lon)

        # Build unassigned users list
        unassigned_users = []
        for _, user in user_df.iterrows():
            if user['user_id'] not in assigned_user_ids:
                unassigned_user = {
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0))
                }
                if pd.notna(user.get('first_name')):
                    unassigned_user['first_name'] = str(user['first_name'])
                if pd.notna(user.get('email')):
                    unassigned_user['email'] = str(user['email'])
                unassigned_users.append(unassigned_user)

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

        # Final metrics update for all routes
        for route in routes:
            update_route_metrics_improved(route, office_lat, office_lon)

        execution_time = time.time() - start_time

        # Final user count verification
        total_users_in_api = len(user_df) # Use user_df length for accurate count of processed users
        users_assigned = sum(len(r['assigned_users']) for r in routes)
        users_unassigned = len(unassigned_users)
        users_accounted_for = users_assigned + users_unassigned

        print(f"✅ Blueprint capacity optimization complete in {execution_time:.2f}s")
        print(f"📊 Final routes: {len(routes)}")
        print(f"🎯 Users assigned: {users_assigned}")
        print(f"👥 Users unassigned: {users_unassigned}")
        print(f"📋 User accounting: {users_accounted_for}/{total_users_in_api} users")

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
            "optimization_mode": "blueprint_capacity_optimization",
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