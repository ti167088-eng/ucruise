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

# Import road_network module for minimal usage in global optimization only
try:
    import road_network as road_network_module
    try:
        road_network = road_network_module.RoadNetwork('tricity_main_roads.graphml')
        logger.info("Successfully loaded RoadNetwork for global optimization")
    except Exception as e:
        logger.warning(f"Could not create RoadNetwork instance: {e}. Using mock implementation.")

        class MockRoadNetwork:
            def get_route_coherence_score(self, driver_pos, user_positions, office_pos):
                return 0.8  # Neutral score

            def is_user_on_route_path(self, driver_pos, current_user_positions, user_pos, office_pos, max_detour_ratio=1.5, route_type="optimization"):
                return True  # Always allow for simplicity

        road_network = MockRoadNetwork()
except ImportError:
    logger.warning("road_network module not found. Using mock implementation.")

    class MockRoadNetwork:
        def get_route_coherence_score(self, driver_pos, user_positions, office_pos):
            return 0.8

        def is_user_on_route_path(self, driver_pos, current_user_positions, user_pos, office_pos, max_detour_ratio=1.5, route_type="optimization"):
            return True

    road_network = MockRoadNetwork()


# Load and validate configuration for balanced optimization
def load_and_validate_config():
    """Load configuration with balanced optimization settings"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    config = {}

    # Balanced parameters - middle ground between capacity and efficiency
    config['MAX_FILL_DISTANCE_KM'] = 5.0
    config['MERGE_DISTANCE_KM'] = 4.0
    config['DBSCAN_EPS_KM'] = 2.0
    config['MAX_BEARING_DIFFERENCE'] = 30
    config['MAX_TURNING_ANGLE'] = 40
    config['MIN_UTIL_THRESHOLD'] = 0.6
    config['LOW_UTILIZATION_THRESHOLD'] = 0.5

    # Office coordinates
    office_lat = float(os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489)))
    office_lon = float(os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))

    config['OFFICE_LAT'] = office_lat
    config['OFFICE_LON'] = office_lon

    # Clustering parameters
    config['clustering_method'] = 'adaptive'
    config['max_users_per_cluster'] = 6
    config['angular_sectors'] = 8

    # Weights for balanced optimization
    config['capacity_weight'] = 1.0
    config['direction_weight'] = 1.0

    # Conversion factors
    config['LAT_TO_KM'] = 111.0
    config['LON_TO_KM'] = 111.0 * math.cos(math.radians(office_lat))

    logger.info(f"🎯 Using optimization mode: BALANCED OPTIMIZATION")
    logger.info(f"   📊 Max bearing difference: {config['MAX_BEARING_DIFFERENCE']}°")
    logger.info(f"   📊 Max turning score: {config['MAX_TURNING_ANGLE']}°")
    logger.info(f"   📊 Max fill distance: {config['MAX_FILL_DISTANCE_KM']}km")

    return config


# Import basic functions from assignment.py
from assignment import (
    validate_input_data, load_env_and_fetch_data, extract_office_coordinates,
    prepare_user_driver_dataframes, haversine_distance, bearing_difference,
    calculate_bearing, _get_all_drivers_as_unassigned, _convert_users_to_unassigned_format
)

# Load validated configuration
_config = load_and_validate_config()


def run_assignment_balance(source_id: str, parameter: int = 1, string_param: str = ""):
    """Main entry point for balanced optimization assignment"""
    start_time = time.time()

    logger.info(f"🚀 Starting BALANCED OPTIMIZATION assignment for source_id: {source_id}")
    logger.info(f"📋 Parameter: {parameter}, String parameter: {string_param}")

    try:
        # Load and validate data
        data = load_env_and_fetch_data(source_id, parameter, string_param)

        users = data.get('users', [])
        if not users:
            logger.info("⚠️ No users found - returning empty assignment")
            return {
                "status": "true",
                "execution_time": time.time() - start_time,
                "data": [],
                "unassignedUsers": [],
                "unassignedDrivers": _get_all_drivers_as_unassigned(data),
                "clustering_analysis": {"method": "No Users", "clusters": 0},
                "optimization_mode": "balanced_optimization",
                "parameter": parameter,
                "string_param": string_param
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
            logger.info("⚠️ No drivers available - all users unassigned")
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
                "string_param": string_param
            }

        logger.info(f"📥 Data loaded - Users: {len(users)}, Total Drivers: {len(all_drivers)}")

        # Extract office coordinates and validate data
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        logger.info("✅ Data validation passed")

        # Prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)

        logger.info(f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")

        # STEP 1: Geographic clustering
        user_df = create_geographic_clusters(user_df, office_lat, office_lon)
        clustering_results = {"method": "balanced_geographic", "clusters": user_df['geo_cluster'].nunique()}

        # STEP 2: Capacity-based sub-clustering
        user_df = create_capacity_subclusters(user_df)

        # STEP 3: Balanced driver assignment
        routes, assigned_user_ids = assign_drivers_balanced(user_df, driver_df, office_lat, office_lon)

        # STEP 4: Local optimization
        routes = local_optimization(routes, office_lat, office_lon)

        # STEP 5: Global optimization (with road network for swapping only)
        routes, unassigned_users = global_optimization(routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon)

        # STEP 6: Route merging (BEFORE splitting - important!)
        routes = merge_compatible_routes(routes, office_lat, office_lon)

        # STEP 7: Route splitting (AFTER merging - only for severe cases)
        routes = split_poor_routes(routes, driver_df, office_lat, office_lon)

        # STEP 8: Handle remaining unassigned users
        unassigned_users = handle_remaining_unassigned(unassigned_users, user_df, assigned_user_ids)

        # Filter out routes with no users - these drivers should be unassigned
        valid_routes = []
        empty_route_drivers = []
        
        for route in routes:
            if route['assigned_users'] and len(route['assigned_users']) > 0:
                valid_routes.append(route)
            else:
                # Driver has no users, should be unassigned
                empty_route_drivers.append(route['driver_id'])
                logger.info(f"    📋 Moving driver {route['driver_id']} to unassigned (no users)")
        
        routes = valid_routes

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

        # Update final metrics
        for route in routes:
            update_route_metrics(route, office_lat, office_lon)

        execution_time = time.time() - start_time

        users_assigned = sum(len(r['assigned_users']) for r in routes)
        users_unassigned = len(unassigned_users)

        logger.info(f"✅ Balanced optimization complete in {execution_time:.2f}s")
        logger.info(f"📊 Final routes: {len(routes)}")
        logger.info(f"🎯 Users assigned: {users_assigned}")
        logger.info(f"👥 Users unassigned: {users_unassigned}")

        return {
            "status": "true",
            "execution_time": execution_time,
            "data": routes,
            "unassignedUsers": unassigned_users,
            "unassignedDrivers": unassigned_drivers,
            "clustering_analysis": clustering_results,
            "optimization_mode": "balanced_optimization",
            "parameter": parameter,
            "string_param": string_param
        }

    except Exception as e:
        logger.error(f"Assignment failed: {e}", exc_info=True)
        return {"status": "false", "details": str(e), "data": [], "parameter": parameter, "string_param": string_param}


def create_geographic_clusters(user_df, office_lat, office_lon):
    """Enhanced geographic clustering that prioritizes proximity and road connectivity"""
    if len(user_df) == 0:
        return user_df

    logger.info("  🗺️ Creating enhanced geographic clusters with proximity and road awareness...")

    # Calculate bearings and distances from office to users
    user_df = user_df.copy()
    user_df['bearing_from_office'] = user_df.apply(
        lambda row: calculate_bearing(office_lat, office_lon, row['latitude'], row['longitude']), axis=1
    )
    user_df['office_distance'] = user_df.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude'], office_lat, office_lon), axis=1
    )

    # Phase 1: Find tight proximity clusters (users very close to each other)
    coords = [[row['latitude'], row['longitude']] for _, row in user_df.iterrows()]
    
    # DBSCAN with 1km radius for very tight clusters
    from sklearn.cluster import DBSCAN
    tight_clustering = DBSCAN(eps=0.01, min_samples=2)  # ~1km radius for tight groups
    tight_labels = tight_clustering.fit_predict(coords)
    
    user_df['tight_cluster'] = tight_labels
    
    # Phase 2: Expand clusters to include nearby users on similar routes
    expanded_clustering = DBSCAN(eps=0.025, min_samples=2)  # ~2.5km radius for route expansion
    expanded_labels = expanded_clustering.fit_predict(coords)
    
    user_df['expanded_cluster'] = expanded_labels
    
    cluster_counter = 0
    
    # Process tight clusters first (highest priority)
    processed_users = set()
    
    for tight_cluster_id in sorted(user_df['tight_cluster'].unique()):
        if tight_cluster_id == -1:
            continue
            
        tight_users = user_df[user_df['tight_cluster'] == tight_cluster_id]
        
        # Check bearing spread within tight cluster
        bearings = tight_users['bearing_from_office'].values
        bearing_spread = max(bearings) - min(bearings)
        if bearing_spread > 180:
            bearing_spread = 360 - bearing_spread
            
        if bearing_spread <= 50:  # More lenient for tight clusters
            user_df.loc[tight_users.index, 'geo_cluster'] = cluster_counter
            processed_users.update(tight_users.index)
            cluster_counter += 1
            logger.info(f"    ✅ Created tight cluster {cluster_counter-1} with {len(tight_users)} users (spread: {bearing_spread:.1f}°)")
    
    # Process expanded clusters for remaining users
    for expanded_cluster_id in sorted(user_df['expanded_cluster'].unique()):
        if expanded_cluster_id == -1:
            continue
            
        expanded_users = user_df[user_df['expanded_cluster'] == expanded_cluster_id]
        unprocessed_expanded = expanded_users[~expanded_users.index.isin(processed_users)]
        
        if len(unprocessed_expanded) >= 2:
            bearings = unprocessed_expanded['bearing_from_office'].values
            bearing_spread = max(bearings) - min(bearings)
            if bearing_spread > 180:
                bearing_spread = 360 - bearing_spread
                
            if bearing_spread <= 35:  # Directional consistency for expanded clusters
                user_df.loc[unprocessed_expanded.index, 'geo_cluster'] = cluster_counter
                processed_users.update(unprocessed_expanded.index)
                cluster_counter += 1
                logger.info(f"    ✅ Created expanded cluster {cluster_counter-1} with {len(unprocessed_expanded)} users (spread: {bearing_spread:.1f}°)")
            else:
                # Split by bearing for large spread expanded clusters
                sorted_users = unprocessed_expanded.sort_values('bearing_from_office')
                mid_point = len(sorted_users) // 2
                
                group1 = sorted_users.iloc[:mid_point]
                group2 = sorted_users.iloc[mid_point:]
                
                if len(group1) >= 2:
                    user_df.loc[group1.index, 'geo_cluster'] = cluster_counter
                    processed_users.update(group1.index)
                    cluster_counter += 1
                    
                if len(group2) >= 2:
                    user_df.loc[group2.index, 'geo_cluster'] = cluster_counter
                    processed_users.update(group2.index)
                    cluster_counter += 1
    
    # Handle remaining isolated users by directional sectors
    unprocessed_users = user_df[~user_df.index.isin(processed_users)]
    if len(unprocessed_users) > 0:
        logger.info(f"    🔄 Processing {len(unprocessed_users)} isolated users by direction...")
        
        # Group isolated users by bearing sectors
        n_sectors = 16  # Fine-grained sectors for isolated users
        sector_angle = 360.0 / n_sectors
        
        sector_groups = {}
        for _, user in unprocessed_users.iterrows():
            sector = int(user['bearing_from_office'] // sector_angle)
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(user.name)
        
        # Assign clusters to sector groups
        for sector, user_indices in sector_groups.items():
            if len(user_indices) >= 2:
                user_df.loc[user_indices, 'geo_cluster'] = cluster_counter
                cluster_counter += 1
            else:
                # Single users get individual clusters
                for idx in user_indices:
                    user_df.loc[idx, 'geo_cluster'] = cluster_counter
                    cluster_counter += 1

    logger.info(f"  ✅ Created {user_df['geo_cluster'].nunique()} enhanced geographic clusters with proximity priority")
    return user_df


def create_capacity_subclusters(user_df):
    """Split large clusters into capacity-appropriate subclusters"""
    if len(user_df) == 0:
        return user_df

    logger.info("  🚗 Creating capacity-based sub-clusters...")

    user_df['sub_cluster'] = -1
    sub_cluster_counter = 0
    max_users_per_cluster = _config['max_users_per_cluster']

    for geo_cluster in user_df['geo_cluster'].unique():
        cluster_users = user_df[user_df['geo_cluster'] == geo_cluster]

        if len(cluster_users) <= max_users_per_cluster:
            user_df.loc[cluster_users.index, 'sub_cluster'] = sub_cluster_counter
            sub_cluster_counter += 1
        else:
            # Split large clusters by distance from office
            cluster_users = cluster_users.copy()
            cluster_users['office_distance'] = cluster_users.apply(
                lambda row: haversine_distance(row['latitude'], row['longitude'],
                                             _config['OFFICE_LAT'], _config['OFFICE_LON']), axis=1
            )
            cluster_users = cluster_users.sort_values('office_distance')

            n_subclusters = math.ceil(len(cluster_users) / max_users_per_cluster)
            users_per_subcluster = len(cluster_users) // n_subclusters

            for i in range(n_subclusters):
                start_idx = i * users_per_subcluster
                end_idx = start_idx + users_per_subcluster if i < n_subclusters - 1 else len(cluster_users)
                subcluster_users = cluster_users.iloc[start_idx:end_idx]
                user_df.loc[subcluster_users.index, 'sub_cluster'] = sub_cluster_counter
                sub_cluster_counter += 1

    logger.info(f"  ✅ Created {user_df['sub_cluster'].nunique()} capacity-based sub-clusters")
    return user_df


def assign_drivers_balanced(user_df, driver_df, office_lat, office_lon):
    """Balanced driver assignment focusing on equilibrium between capacity and efficiency"""
    logger.info("⚖️ Step 3: Balanced driver assignment...")

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Sort drivers by capacity (descending) then priority (ascending)
    available_drivers = driver_df.sort_values(['capacity', 'priority'], ascending=[False, True])

    # Process each sub-cluster
    for sub_cluster_id in sorted(user_df['sub_cluster'].unique()):
        cluster_users = user_df[user_df['sub_cluster'] == sub_cluster_id]
        unassigned_in_cluster = cluster_users[~cluster_users['user_id'].isin(assigned_user_ids)]

        if unassigned_in_cluster.empty:
            continue

        route = assign_best_driver_to_cluster(unassigned_in_cluster, available_drivers,
                                            used_driver_ids, office_lat, office_lon)
        if route:
            routes.append(route)
            assigned_user_ids.update(u['user_id'] for u in route['assigned_users'])

    logger.info(f"  ✅ Created {len(routes)} initial routes with balanced assignment")
    return routes, assigned_user_ids


def assign_best_driver_to_cluster(cluster_users, available_drivers, used_driver_ids, office_lat, office_lon):
    """Find best driver using balanced scoring with more inclusive assignment"""
    cluster_size = len(cluster_users)
    
    # Don't create routes for empty clusters
    if cluster_size == 0:
        return None
    
    # More lenient bearing validation - allow some spread for nearby users
    bearings = cluster_users['bearing_from_office'].values
    bearing_spread = max(bearings) - min(bearings)
    if bearing_spread > 180:  # Handle circular nature
        bearing_spread = 360 - bearing_spread
    
    # Only reject if bearing spread is extremely large (users are in opposite directions)
    if bearing_spread > 50 and cluster_size > 4:  # More lenient threshold
        logger.warning(f"    ⚠️ Large cluster with high bearing spread ({bearing_spread:.1f}°) - attempting assignment anyway")
        
    best_driver = None
    best_score = float('inf')

    # Calculate cluster center and main bearing
    cluster_center_lat = cluster_users['latitude'].mean()
    cluster_center_lon = cluster_users['longitude'].mean()
    cluster_main_bearing = cluster_users['bearing_from_office'].mean()

    # Try multiple passes with increasingly lenient criteria
    for pass_num in range(3):
        for _, driver in available_drivers.iterrows():
            if driver['driver_id'] in used_driver_ids or driver['capacity'] < cluster_size:
                continue

            # Distance score
            distance_to_cluster = haversine_distance(driver['latitude'], driver['longitude'],
                                                   cluster_center_lat, cluster_center_lon)

            # More flexible utilization scoring
            utilization = cluster_size / driver['capacity']
            if utilization < 0.4:  # More lenient minimum
                utilization_penalty = (0.4 - utilization) * 10
            elif utilization > 0.95:
                utilization_penalty = (utilization - 0.95) * 20
            else:
                utilization_penalty = 0

            # Progressive bearing tolerance
            driver_to_office_bearing = calculate_bearing(driver['latitude'], driver['longitude'],
                                                       office_lat, office_lon)
            bearing_diff = bearing_difference(driver_to_office_bearing, cluster_main_bearing)
            
            # Bearing thresholds get more lenient with each pass
            bearing_thresholds = [30, 45, 60]  # More lenient than before
            if bearing_diff > bearing_thresholds[pass_num]:
                continue
                
            direction_penalty = bearing_diff * 0.15  # Reduced penalty

            # Driver position relative to cluster-office line
            cluster_to_office_bearing = calculate_bearing(cluster_center_lat, cluster_center_lon, office_lat, office_lon)
            driver_to_cluster_bearing = calculate_bearing(driver['latitude'], driver['longitude'], 
                                                        cluster_center_lat, cluster_center_lon)
            route_alignment_diff = bearing_difference(driver_to_cluster_bearing, cluster_to_office_bearing)
            route_alignment_penalty = abs(route_alignment_diff) * 0.05  # Reduced penalty

            # Total score with more balanced penalties
            total_score = (distance_to_cluster * 0.6 + utilization_penalty + direction_penalty + 
                          route_alignment_penalty + driver['priority'] * 0.3)

            if total_score < best_score:
                best_score = total_score
                best_driver = driver

        # If we found a driver, break out of passes
        if best_driver is not None:
            break

    if best_driver is not None and cluster_size > 0:
        used_driver_ids.add(best_driver['driver_id'])

        route = {
            'driver_id': str(best_driver['driver_id']),
            'vehicle_id': str(best_driver.get('vehicle_id', '')),
            'vehicle_type': int(best_driver['capacity']),
            'latitude': float(best_driver['latitude']),
            'longitude': float(best_driver['longitude']),
            'assigned_users': []
        }

        # Add users to route
        for _, user in cluster_users.iterrows():
            user_data = {
                'user_id': str(user['user_id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': float(user.get('office_distance', 0)),
                'bearing_from_office': float(user['bearing_from_office'])
            }
            if pd.notna(user.get('first_name')):
                user_data['first_name'] = str(user['first_name'])
            if pd.notna(user.get('email')):
                user_data['email'] = str(user['email'])

            route['assigned_users'].append(user_data)

        # Optimize pickup sequence with balanced approach
        route = optimize_route_sequence(route, office_lat, office_lon)

        utilization = len(route['assigned_users']) / route['vehicle_type']
        logger.info(f"    ⚖️ Driver {best_driver['driver_id']}: {len(route['assigned_users'])}/{route['vehicle_type']} seats ({utilization*100:.1f}%)")

        return route

    # If no driver found, log the issue
    logger.warning(f"    ⚠️ No suitable driver found for cluster of {cluster_size} users (bearing spread: {bearing_spread:.1f}°)")
    return None


def optimize_route_sequence(route, office_lat, office_lon):
    """Optimize pickup sequence using simple nearest neighbor with bearing consideration"""
    if not route['assigned_users'] or len(route['assigned_users']) <= 1:
        return route

    users = route['assigned_users']
    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (office_lat, office_lon)

    # Calculate main route bearing
    main_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    # Sort users by projection along main route direction
    def route_score(user):
        user_pos = (user['lat'], user['lng'])
        distance = haversine_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])

        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
        bearing_diff = bearing_difference(user_bearing, main_bearing)

        # Balanced scoring - consider both distance and direction
        return distance * 0.7 + bearing_diff * 0.02

    users.sort(key=route_score)
    route['assigned_users'] = users

    return route


def optimize_route_sequence_ultra_strict(route, office_lat, office_lon):
    """Ultra-strict route sequence optimization - maintains perfect bearing consistency"""
    if not route['assigned_users'] or len(route['assigned_users']) <= 1:
        return route

    users = route['assigned_users']
    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (office_lat, office_lon)

    # Calculate main route bearing (driver to office)
    main_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
    
    # Step 1: Filter users that are too far off the main bearing
    valid_users = []
    for user in users:
        user_bearing = user.get('bearing_from_office', 
                               calculate_bearing(office_pos[0], office_pos[1], user['lat'], user['lng']))
        bearing_deviation = bearing_difference(user_bearing, main_bearing)
        
        if abs(bearing_deviation) <= 20:  # Only users within 20° of main bearing
            user['bearing_deviation'] = bearing_deviation
            user['office_distance'] = haversine_distance(office_pos[0], office_pos[1], user['lat'], user['lng'])
            valid_users.append(user)
    
    if not valid_users:
        logger.warning(f"    ⚠️ No users within bearing tolerance for driver {route['driver_id']}")
        route['assigned_users'] = users  # Keep original if no valid users found
        return route
    
    # Step 2: Sort by distance from office (farthest first - natural pickup order)
    valid_users.sort(key=lambda x: x['office_distance'], reverse=True)
    
    # Step 3: Apply bearing-consistent ordering
    final_sequence = []
    remaining_users = valid_users.copy()
    
    # Start from the user farthest from office
    if remaining_users:
        current_user = remaining_users.pop(0)
        final_sequence.append(current_user)
        current_pos = (current_user['lat'], current_user['lng'])
        
        # Add remaining users in bearing-consistent order
        while remaining_users:
            best_user = None
            best_score = float('inf')
            
            for user in remaining_users:
                user_pos = (user['lat'], user['lng'])
                
                # Distance from current position
                distance = haversine_distance(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
                
                # Bearing from current position to user
                segment_bearing = calculate_bearing(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
                
                # How well does this segment align with main route?
                bearing_consistency = abs(bearing_difference(segment_bearing, main_bearing))
                
                # Prefer users that maintain bearing consistency and are closer
                score = distance * 0.7 + bearing_consistency * 0.3
                
                if score < best_score:
                    best_score = score
                    best_user = user
            
            if best_user:
                final_sequence.append(best_user)
                remaining_users.remove(best_user)
                current_pos = (best_user['lat'], best_user['lng'])
            else:
                # Add remaining users in distance order if no good bearing match
                final_sequence.extend(remaining_users)
                break
    
    # Clean up temporary fields
    for user in final_sequence:
        if 'bearing_deviation' in user:
            del user['bearing_deviation']
        if 'office_distance' in user:
            del user['office_distance']
    
    route['assigned_users'] = final_sequence
    
    # Final validation - check for zigzag pattern
    if len(final_sequence) > 2:
        zigzag_score = calculate_zigzag_score(final_sequence, driver_pos, office_pos)
        if zigzag_score > 30:  # High zigzag detected
            logger.warning(f"    ⚠️ High zigzag score ({zigzag_score:.1f}°) for route {route['driver_id']}")
            # Try distance-only sorting as fallback
            route['assigned_users'] = sorted(users, key=lambda x: 
                haversine_distance(office_pos[0], office_pos[1], x['lat'], x['lng']), reverse=True)
    
    return route


def calculate_zigzag_score(users, driver_pos, office_pos):
    """Calculate zigzag score - average turning angle in the route"""
    if len(users) <= 1:
        return 0
    
    route_points = [driver_pos] + [(u['lat'], u['lng']) for u in users] + [office_pos]
    turning_angles = []
    
    for i in range(1, len(route_points) - 1):
        p1, p2, p3 = route_points[i-1], route_points[i], route_points[i+1]
        
        bearing1 = calculate_bearing(p1[0], p1[1], p2[0], p2[1])
        bearing2 = calculate_bearing(p2[0], p2[1], p3[0], p3[1])
        
        turning_angle = abs(bearing_difference(bearing1, bearing2))
        turning_angles.append(turning_angle)
    
    return np.mean(turning_angles) if turning_angles else 0


def validate_no_zigzag(users, driver_pos, office_pos, main_bearing):
    """Ensure route doesn't zigzag - maintain consistent direction"""
    if len(users) <= 1:
        return users
    
    validated_sequence = []
    current_pos = driver_pos
    remaining_users = users.copy()
    
    while remaining_users:
        best_user = None
        best_score = float('inf')
        
        for user in remaining_users:
            user_pos = (user['lat'], user['lng'])
            
            # Calculate bearing from current position to user
            current_to_user_bearing = calculate_bearing(current_pos[0], current_pos[1], 
                                                       user_pos[0], user_pos[1])
            
            # Check bearing consistency with main route
            bearing_deviation = abs(bearing_difference(current_to_user_bearing, main_bearing))
            
            # Distance factor
            distance = haversine_distance(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
            
            # Prefer users that maintain bearing consistency
            score = distance + bearing_deviation * 0.5  # Weight bearing consistency
            
            if score < best_score:
                best_score = score
                best_user = user
        
        if best_user:
            validated_sequence.append(best_user)
            remaining_users.remove(best_user)
            current_pos = (best_user['lat'], best_user['lng'])
        else:
            # Fallback - add remaining users in order
            validated_sequence.extend(remaining_users)
            break
    
    return validated_sequence


def local_optimization(routes, office_lat, office_lon):
    """Local optimization - simple user swapping between nearby routes"""
    logger.info("🔧 Step 4: Local optimization...")

    improved = True
    iterations = 0
    max_iterations = 3

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        # Optimize sequence within each route
        for route in routes:
            route = optimize_route_sequence(route, office_lat, office_lon)

        # Try simple user swaps between nearby routes
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if try_user_swap(routes[i], routes[j], office_lat, office_lon):
                    improved = True

    logger.info(f"  ✅ Local optimization completed in {iterations} iterations")
    return routes


def try_user_swap(route1, route2, office_lat, office_lon):
    """Try swapping users between two routes if it improves overall quality"""
    if not route1['assigned_users'] or not route2['assigned_users']:
        return False

    # Check if routes are nearby
    center1 = calculate_route_center(route1)
    center2 = calculate_route_center(route2)
    distance = haversine_distance(center1[0], center1[1], center2[0], center2[1])

    if distance > _config['MERGE_DISTANCE_KM'] * 1.5:
        return False

    # Try swapping one user from route1 to route2
    for user1 in route1['assigned_users'][:]:
        if len(route2['assigned_users']) >= route2['vehicle_type']:
            continue

        # Calculate current costs
        cost1_before = calculate_route_cost(route1, office_lat, office_lon)
        cost2_before = calculate_route_cost(route2, office_lat, office_lon)

        # Perform swap
        route1['assigned_users'].remove(user1)
        route2['assigned_users'].append(user1)

        # Optimize sequences
        route1 = optimize_route_sequence(route1, office_lat, office_lon)
        route2 = optimize_route_sequence(route2, office_lat, office_lon)

        # Calculate new costs
        cost1_after = calculate_route_cost(route1, office_lat, office_lon)
        cost2_after = calculate_route_cost(route2, office_lat, office_lon)

        total_improvement = (cost1_before + cost2_before) - (cost1_after + cost2_after)

        if total_improvement > 0.5:  # Keep if improvement > 0.5km
            return True
        else:
            # Revert swap
            route2['assigned_users'].remove(user1)
            route1['assigned_users'].append(user1)

    return False


def global_optimization(routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon):
    """Global optimization with road network for intelligent user swapping"""
    logger.info("🌍 Step 5: Global optimization...")

    # Handle single-user routes first
    routes = fix_single_user_routes(routes, office_lat, office_lon)

    # Fill underutilized routes with road network assistance
    unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()
    routes = fill_underutilized_routes_with_road_network(routes, unassigned_users_df,
                                                        assigned_user_ids, office_lat, office_lon)

    # Build final unassigned list
    final_unassigned = user_df[~user_df['user_id'].isin(assigned_user_ids)]
    unassigned_list = []
    for _, user in final_unassigned.iterrows():
        user_data = {
            'user_id': str(user['user_id']),
            'lat': float(user['latitude']),
            'lng': float(user['longitude'])
        }
        if pd.notna(user.get('first_name')):
            user_data['first_name'] = str(user['first_name'])
        if pd.notna(user.get('email')):
            user_data['email'] = str(user['email'])
        unassigned_list.append(user_data)

    logger.info("  ✅ Global optimization completed")
    return routes, unassigned_list


def fill_underutilized_routes_with_road_network(routes, unassigned_users_df, assigned_user_ids, office_lat, office_lon):
    """Enhanced user assignment with proximity and path prioritization"""
    users_added = 0
    
    # PRIORITY PHASE: Find users very close to existing route users
    logger.info(f"    🎯 Priority phase: Finding users close to existing route users...")
    for route in routes:
        if len(route['assigned_users']) >= route['vehicle_type']:
            continue
        
        available_seats = route['vehicle_type'] - len(route['assigned_users'])
        close_users = []
        
        # Find users very close to any user in this route
        for _, unassigned_user in unassigned_users_df.iterrows():
            if unassigned_user['user_id'] in assigned_user_ids:
                continue
                
            min_distance_to_route_user = float('inf')
            for route_user in route['assigned_users']:
                distance = haversine_distance(
                    unassigned_user['latitude'], unassigned_user['longitude'],
                    route_user['lat'], route_user['lng']
                )
                min_distance_to_route_user = min(min_distance_to_route_user, distance)
            
            # If very close (within 1.5km), prioritize
            if min_distance_to_route_user <= 1.5:
                user_bearing = calculate_bearing(office_lat, office_lon, 
                                               unassigned_user['latitude'], unassigned_user['longitude'])
                route_bearing = calculate_average_bearing(route, office_lat, office_lon)
                bearing_diff = bearing_difference(user_bearing, route_bearing)
                
                if bearing_diff <= 40:  # Reasonable bearing compatibility
                    close_users.append((min_distance_to_route_user, unassigned_user))
        
        # Add closest users first
        close_users.sort(key=lambda x: x[0])
        added_to_route = 0
        for distance, user in close_users[:available_seats]:
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
            users_added += 1
            added_to_route += 1

        if added_to_route > 0:
            route = optimize_route_sequence(route, office_lat, office_lon)
            logger.info(f"      🎯 Added {added_to_route} close users to route {route['driver_id']}")
    
    # PATH-BASED PHASE: Find users on route paths using road network
    logger.info(f"    🛣️ Path-based phase: Finding users on route paths...")
    for route in routes:
        if len(route['assigned_users']) >= route['vehicle_type']:
            continue
        
        available_seats = route['vehicle_type'] - len(route['assigned_users'])
        path_users = []
        
        driver_pos = (route['latitude'], route['longitude'])
        current_users = [(u['lat'], u['lng']) for u in route['assigned_users']]
        office_pos = (office_lat, office_lon)
        
        for _, unassigned_user in unassigned_users_df.iterrows():
            if unassigned_user['user_id'] in assigned_user_ids:
                continue
                
            candidate_pos = (unassigned_user['latitude'], unassigned_user['longitude'])
            
            # Check if user is on the route path
            try:
                if road_network.is_user_on_route_path(driver_pos, current_users,
                                                    candidate_pos, office_pos,
                                                    max_detour_ratio=1.6,
                                                    route_type="balanced"):
                    distance_to_route = haversine_distance(
                        calculate_route_center(route)[0], calculate_route_center(route)[1],
                        candidate_pos[0], candidate_pos[1]
                    )
                    
                    if distance_to_route <= 6.0:  # Within reasonable distance
                        user_bearing = calculate_bearing(office_lat, office_lon, 
                                                       candidate_pos[0], candidate_pos[1])
                        route_bearing = calculate_average_bearing(route, office_lat, office_lon)
                        bearing_diff = bearing_difference(user_bearing, route_bearing)
                        
                        if bearing_diff <= 35:
                            path_score = distance_to_route + bearing_diff * 0.1
                            path_users.append((path_score, unassigned_user))
            except Exception as e:
                logger.warning(f"Path check failed: {e}")
                continue
        
        # Add best path users
        path_users.sort(key=lambda x: x[0])
        added_to_route = 0
        for score, user in path_users[:available_seats]:
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
            users_added += 1
            added_to_route += 1

        if added_to_route > 0:
            route = optimize_route_sequence(route, office_lat, office_lon)
            logger.info(f"      🛣️ Added {added_to_route} path users to route {route['driver_id']}")
    
    # STANDARD PHASE: Progressive relaxation for remaining assignments
    for pass_num in range(3):
        logger.info(f"    🔄 Pass {pass_num + 1}: Standard filling with progressive relaxation...")
        
        distance_thresholds = [5.0, 7.0, 10.0]
        bearing_thresholds = [30, 40, 50]
        max_distance = distance_thresholds[pass_num]
        max_bearing_diff = bearing_thresholds[pass_num]
        
        for route in routes:
            if len(route['assigned_users']) >= route['vehicle_type']:
                continue

            available_seats = route['vehicle_type'] - len(route['assigned_users'])
            route_center = calculate_route_center(route)
            route_bearing = calculate_average_bearing(route, office_lat, office_lon)

            # Find compatible unassigned users
            candidates = []
            for _, user in unassigned_users_df.iterrows():
                if user['user_id'] in assigned_user_ids:
                    continue

                distance = haversine_distance(route_center[0], route_center[1],
                                            user['latitude'], user['longitude'])

                if distance <= max_distance:
                    user_bearing = calculate_bearing(office_lat, office_lon, 
                                                   user['latitude'], user['longitude'])
                    bearing_diff = bearing_difference(route_bearing, user_bearing)
                    
                    if bearing_diff <= max_bearing_diff:
                        score = distance + bearing_diff * 0.1
                        candidates.append((score, distance, user))

            # Add best candidates
            candidates.sort(key=lambda x: x[0])
            added_to_route = 0
            for score, distance, user in candidates[:available_seats]:
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
                users_added += 1
                added_to_route += 1

            if added_to_route > 0:
                route = optimize_route_sequence(route, office_lat, office_lon)
                logger.info(f"      ➕ Added {added_to_route} users to route {route['driver_id']}")

    # EMERGENCY ROUTES: Create new routes for remaining clustered users
    remaining_unassigned = unassigned_users_df[~unassigned_users_df['user_id'].isin(assigned_user_ids)]
    if len(remaining_unassigned) >= 2:
        logger.info(f"    🚗 Attempting to create new routes for {len(remaining_unassigned)} remaining users...")
        new_routes = create_emergency_routes_for_unassigned(remaining_unassigned, routes, office_lat, office_lon)
        if new_routes:
            routes.extend(new_routes)
            for new_route in new_routes:
                for user in new_route['assigned_users']:
                    assigned_user_ids.add(user['user_id'])
                    users_added += 1

    if users_added > 0:
        logger.info(f"    ✅ Successfully assigned {users_added} additional users with enhanced proximity and path detection")

    return routes


def create_emergency_routes_for_unassigned(unassigned_users_df, existing_routes, office_lat, office_lon):
    """Create new routes for clusters of unassigned users using available drivers"""
    if len(unassigned_users_df) < 2:
        return []
    
    # Get assigned driver IDs
    assigned_driver_ids = {route['driver_id'] for route in existing_routes}
    
    # For now, we'll try to find the best clusters among unassigned users
    # In a real implementation, you'd need access to available drivers here
    # This is a placeholder that identifies potential clusters
    
    coords = [[row['latitude'], row['longitude']] for _, row in unassigned_users_df.iterrows()]
    
    # Use DBSCAN to find clusters of unassigned users
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=0.02, min_samples=2)  # ~2km radius, min 2 users
    cluster_labels = clustering.fit_predict(coords)
    
    potential_routes = []
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:  # Skip noise points
            continue
            
        cluster_users = unassigned_users_df[cluster_labels == cluster_id]
        if len(cluster_users) >= 2:
            logger.info(f"      🎯 Found cluster of {len(cluster_users)} unassigned users for potential new route")
            # In a real implementation, you'd assign the best available driver here
            # For now, we'll just log the opportunity
    
    return potential_routes  # Return empty for now, but structure is ready


def fix_single_user_routes(routes, office_lat, office_lon):
    """Merge single-user routes with compatible multi-user routes"""
    single_routes = [r for r in routes if len(r['assigned_users']) == 1]
    multi_routes = [r for r in routes if len(r['assigned_users']) > 1]

    for single_route in single_routes[:]:
        single_user = single_route['assigned_users'][0]
        best_merge_route = None
        best_distance = float('inf')

        for multi_route in multi_routes:
            if len(multi_route['assigned_users']) >= multi_route['vehicle_type']:
                continue

            route_center = calculate_route_center(multi_route)
            distance = haversine_distance(single_user['lat'], single_user['lng'],
                                        route_center[0], route_center[1])

            if distance < best_distance and distance <= _config['MAX_FILL_DISTANCE_KM']:
                # Check direction compatibility
                route_bearing = calculate_average_bearing(multi_route, office_lat, office_lon)
                user_bearing = calculate_bearing(office_lat, office_lon,
                                               single_user['lat'], single_user['lng'])
                bearing_diff = bearing_difference(route_bearing, user_bearing)

                if bearing_diff <= _config['MAX_BEARING_DIFFERENCE'] * 1.5:
                    best_distance = distance
                    best_merge_route = multi_route

        if best_merge_route is not None:
            best_merge_route['assigned_users'].append(single_user)
            best_merge_route = optimize_route_sequence(best_merge_route, office_lat, office_lon)
            routes.remove(single_route)

    return routes


def merge_compatible_routes(routes, office_lat, office_lon):
    """Enhanced route merging with road network awareness and proximity priority"""
    logger.info("🔗 Step 6: Enhanced road-aware route merging...")

    # Phase 1: Priority merging for nearby users
    logger.info("    🎯 Phase 1: Priority merging for geographically close routes...")
    current_routes = merge_nearby_routes(routes, office_lat, office_lon)
    
    # Phase 2: Road-path based merging
    logger.info("    🛣️ Phase 2: Road-path based merging...")
    current_routes = merge_routes_on_same_path(current_routes, office_lat, office_lon)
    
    # Phase 3: Traditional compatibility merging
    logger.info("    ⚖️ Phase 3: Traditional compatibility merging...")
    current_routes = perform_traditional_merging(current_routes, office_lat, office_lon)
    
    logger.info(f"  🔗 Enhanced merging: {len(routes)} → {len(current_routes)} routes")
    return current_routes


def merge_nearby_routes(routes, office_lat, office_lon):
    """Merge routes with users who are very close to each other"""
    merged_routes = []
    used = set()
    merges_count = 0
    
    for i, route1 in enumerate(routes):
        if i in used:
            continue
            
        best_merge_route = None
        best_merge_index = None
        min_user_distance = float('inf')
        
        # Look for routes with users very close to this route's users
        for j, route2 in enumerate(routes):
            if j <= i or j in used:
                continue
                
            # Check capacity
            total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
            max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])
            if total_users > max_capacity:
                continue
            
            # Find minimum distance between any two users from different routes
            min_distance = float('inf')
            for user1 in route1['assigned_users']:
                for user2 in route2['assigned_users']:
                    distance = haversine_distance(user1['lat'], user1['lng'], user2['lat'], user2['lng'])
                    min_distance = min(min_distance, distance)
            
            # If users are very close (within 2km), consider merging
            if min_distance <= 2.0 and min_distance < min_user_distance:
                # Check if merged route would be acceptable
                test_route = route1.copy()
                test_route['assigned_users'] = route1['assigned_users'] + route2['assigned_users']
                test_route['vehicle_type'] = max_capacity
                test_route = optimize_route_sequence(test_route, office_lat, office_lon)
                
                turning_score = calculate_turning_score(test_route, office_lat, office_lon)
                if turning_score <= 60:  # More lenient for nearby users
                    min_user_distance = min_distance
                    best_merge_route = route2
                    best_merge_index = j
        
        if best_merge_route is not None:
            # Perform merge
            better_route = route1 if route1['vehicle_type'] >= best_merge_route['vehicle_type'] else best_merge_route
            merged_route = better_route.copy()
            merged_route['assigned_users'] = route1['assigned_users'] + best_merge_route['assigned_users']
            merged_route['vehicle_type'] = max(route1['vehicle_type'], best_merge_route['vehicle_type'])
            merged_route = optimize_route_sequence(merged_route, office_lat, office_lon)
            
            merged_routes.append(merged_route)
            used.add(i)
            used.add(best_merge_index)
            merges_count += 1
            logger.info(f"      ✅ Merged nearby routes (min distance: {min_user_distance:.1f}km)")
        else:
            merged_routes.append(route1)
            used.add(i)
    
    logger.info(f"    🎯 Phase 1: {merges_count} nearby route merges")
    return merged_routes


def merge_routes_on_same_path(routes, office_lat, office_lon):
    """Merge routes where users are on the same road path"""
    merged_routes = []
    used = set()
    merges_count = 0
    
    for i, route1 in enumerate(routes):
        if i in used:
            continue
        
        best_merge_route = None
        best_merge_index = None
        best_path_score = float('inf')
        
        route1_center = calculate_route_center(route1)
        route1_bearing = calculate_average_bearing(route1, office_lat, office_lon)
        
        for j, route2 in enumerate(routes):
            if j <= i or j in used:
                continue
                
            # Check capacity
            total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
            max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])
            if total_users > max_capacity:
                continue
            
            route2_center = calculate_route_center(route2)
            route2_bearing = calculate_average_bearing(route2, office_lat, office_lon)
            
            # Check if routes are on similar paths
            center_distance = haversine_distance(route1_center[0], route1_center[1], route2_center[0], route2_center[1])
            bearing_diff = bearing_difference(route1_bearing, route2_bearing)
            
            if center_distance <= 4.0 and bearing_diff <= 25:
                # Use road network to check if they're on the same path
                try:
                    if road_network and hasattr(road_network, 'is_user_on_route_path'):
                        # Check if route2 users are on route1's path
                        on_path_count = 0
                        for user2 in route2['assigned_users']:
                            existing_positions = [(u['lat'], u['lng']) for u in route1['assigned_users']]
                            if road_network.is_user_on_route_path(
                                (route1['latitude'], route1['longitude']),
                                existing_positions,
                                (user2['lat'], user2['lng']),
                                (office_lat, office_lon),
                                max_detour_ratio=1.8,
                                route_type="balanced"
                            ):
                                on_path_count += 1
                        
                        path_compatibility = on_path_count / len(route2['assigned_users'])
                        if path_compatibility >= 0.6:  # 60% of users on same path
                            path_score = center_distance + bearing_diff * 0.1 - (path_compatibility * 10)
                            if path_score < best_path_score:
                                best_path_score = path_score
                                best_merge_route = route2
                                best_merge_index = j
                except Exception as e:
                    logger.warning(f"Road network check failed: {e}")
                    continue
        
        if best_merge_route is not None:
            # Perform merge
            better_route = route1 if route1['vehicle_type'] >= best_merge_route['vehicle_type'] else best_merge_route
            merged_route = better_route.copy()
            merged_route['assigned_users'] = route1['assigned_users'] + best_merge_route['assigned_users']
            merged_route['vehicle_type'] = max(route1['vehicle_type'], best_merge_route['vehicle_type'])
            merged_route = optimize_route_sequence(merged_route, office_lat, office_lon)
            
            merged_routes.append(merged_route)
            used.add(i)
            used.add(best_merge_index)
            merges_count += 1
            logger.info(f"      ✅ Merged routes on same path (score: {best_path_score:.1f})")
        else:
            merged_routes.append(route1)
            used.add(i)
    
    logger.info(f"    🛣️ Phase 2: {merges_count} same-path route merges")
    return merged_routes


def perform_traditional_merging(routes, office_lat, office_lon):
    """Traditional compatibility-based merging"""
    current_routes = routes.copy()
    total_merges = 0
    
    for pass_num in range(2):  # Reduced passes since we already did specific merging
        merged_routes = []
        used = set()
        merges_in_pass = 0

        # More conservative thresholds since we already handled priority cases
        distance_thresholds = [3.0, 5.0]
        bearing_thresholds = [20, 30]
        
        for i, route1 in enumerate(current_routes):
            if i in used:
                continue

            best_merge = None
            best_score = float('inf')

            for j, route2 in enumerate(current_routes):
                if j <= i or j in used:
                    continue

                # Basic capacity check
                total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
                max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])
                if total_users > max_capacity:
                    continue

                # Distance compatibility check
                center1 = calculate_route_center(route1)
                center2 = calculate_route_center(route2)
                distance = haversine_distance(center1[0], center1[1], center2[0], center2[1])
                if distance > distance_thresholds[pass_num]:
                    continue

                # Bearing compatibility check
                bearing1 = calculate_average_bearing(route1, office_lat, office_lon)
                bearing2 = calculate_average_bearing(route2, office_lat, office_lon)
                bearing_diff = bearing_difference(bearing1, bearing2)
                if bearing_diff > bearing_thresholds[pass_num]:
                    continue

                # Test merge quality
                test_route = route1.copy()
                test_route['assigned_users'] = route1['assigned_users'] + route2['assigned_users']
                test_route['vehicle_type'] = max_capacity
                test_route = optimize_route_sequence(test_route, office_lat, office_lon)
                
                merged_turning = calculate_turning_score(test_route, office_lat, office_lon)
                if merged_turning > 50:  # Conservative threshold
                    continue

                # Calculate merge score
                merge_score = distance * 0.4 + bearing_diff * 0.2 + merged_turning * 0.4
                
                if merge_score < best_score:
                    best_score = merge_score
                    best_merge = j

            if best_merge is not None:
                # Perform merge
                better_route = route1 if route1['vehicle_type'] >= current_routes[best_merge]['vehicle_type'] else current_routes[best_merge]
                merged_route = better_route.copy()
                merged_route['assigned_users'] = route1['assigned_users'] + current_routes[best_merge]['assigned_users']
                merged_route['vehicle_type'] = max(route1['vehicle_type'], current_routes[best_merge]['vehicle_type'])
                merged_route = optimize_route_sequence(merged_route, office_lat, office_lon)
                merged_routes.append(merged_route)
                used.add(i)
                used.add(best_merge)
                merges_in_pass += 1
                total_merges += 1
            else:
                merged_routes.append(route1)
                used.add(i)

        current_routes = merged_routes
        
        if merges_in_pass == 0:
            break

    logger.info(f"    ⚖️ Phase 3: {total_merges} traditional merges")
    return current_routes


def split_poor_routes(routes, driver_df, office_lat, office_lon):
    """Enhanced route splitting with road connectivity analysis and stricter criteria"""
    logger.info("✂️ Step 7: Road-aware route splitting...")

    improved_routes = []
    available_drivers = driver_df[~driver_df['driver_id'].isin([r['driver_id'] for r in routes])].copy()
    routes_split = 0

    for route in routes:
        if len(route['assigned_users']) < 4:  # Only split larger routes
            improved_routes.append(route)
            continue

        # Calculate comprehensive quality metrics
        turning_score = calculate_turning_score(route, office_lat, office_lon)
        zigzag_score = calculate_zigzag_score(route['assigned_users'], 
                                            (route['latitude'], route['longitude']), 
                                            (office_lat, office_lon))
        bearing_spread = calculate_route_bearing_spread(route, office_lat, office_lon)
        
        # NEW: Check road connectivity issues
        road_connectivity_problems = analyze_route_connectivity(route, office_lat, office_lon)
        
        # Determine if route needs splitting with stricter criteria
        needs_split = False
        split_reason = []
        
        # Much stricter thresholds
        if turning_score > 80:  # Very high turning threshold
            needs_split = True
            split_reason.append(f"excessive turning ({turning_score:.1f}°)")
            
        if zigzag_score > 80:  # Very high zigzag threshold
            needs_split = True
            split_reason.append(f"excessive zigzag ({zigzag_score:.1f}°)")
            
        if bearing_spread > 60:  # Very high bearing spread
            needs_split = True
            split_reason.append(f"excessive bearing spread ({bearing_spread:.1f}°)")
        
        # NEW: Road connectivity criterion - split if users require completely different routes
        if road_connectivity_problems['severity'] > 0.7:  # High connectivity problems
            needs_split = True
            split_reason.append(f"poor road connectivity ({road_connectivity_problems['severity']:.2f})")

        # Only split if we have available drivers and severe problems
        if needs_split and len(route['assigned_users']) >= 4 and len(available_drivers) > 0:
            logger.info(f"  ✂️ Splitting route {route['driver_id']}: {', '.join(split_reason)}")
            
            # Intelligent splitting based on the main problem
            if road_connectivity_problems['severity'] > 0.7:
                # Split based on road connectivity clusters
                split_routes = split_route_by_road_connectivity(route, available_drivers, office_lat, office_lon, road_connectivity_problems)
            elif bearing_spread > 60:
                # Split by bearing clusters
                split_routes = split_route_by_bearing(route, available_drivers, office_lat, office_lon)
            else:
                # Split by geographic distance
                split_routes = split_route_by_distance(route, available_drivers, office_lat, office_lon)
            
            if len(split_routes) > 1:
                improved_routes.extend(split_routes)
                routes_split += 1
                # Update available drivers
                for split_route in split_routes:
                    available_drivers = available_drivers[available_drivers['driver_id'] != split_route['driver_id']]
            else:
                improved_routes.append(route)
        else:
            improved_routes.append(route)

    if routes_split > 0:
        logger.info(f"  ✂️ Successfully split {routes_split} routes with road connectivity analysis")
    else:
        logger.info(f"  ✂️ No routes required splitting under strict criteria")
    
    return improved_routes


def analyze_route_connectivity(route, office_lat, office_lon):
    """Analyze road connectivity issues in a route"""
    users = route['assigned_users']
    if len(users) < 2:
        return {'severity': 0.0, 'problem_areas': []}
    
    connectivity_problems = {'severity': 0.0, 'problem_areas': []}
    
    try:
        driver_pos = (route['latitude'], route['longitude'])
        office_pos = (office_lat, office_lon)
        
        # Check road distances vs straight line distances for route segments
        total_detour_ratio = 0.0
        segment_count = 0
        problem_segments = []
        
        # Driver to first user
        if users:
            first_user = users[0]
            road_distance = road_network.get_road_distance(driver_pos[0], driver_pos[1], first_user['lat'], first_user['lng'])
            straight_distance = haversine_distance(driver_pos[0], driver_pos[1], first_user['lat'], first_user['lng'])
            
            if straight_distance > 0:
                detour_ratio = road_distance / straight_distance
                total_detour_ratio += detour_ratio
                segment_count += 1
                
                if detour_ratio > 2.5:  # Major detour
                    problem_segments.append({
                        'type': 'driver_to_first',
                        'detour_ratio': detour_ratio,
                        'positions': [driver_pos, (first_user['lat'], first_user['lng'])]
                    })
        
        # Between users
        for i in range(len(users) - 1):
            user1 = users[i]
            user2 = users[i + 1]
            
            road_distance = road_network.get_road_distance(user1['lat'], user1['lng'], user2['lat'], user2['lng'])
            straight_distance = haversine_distance(user1['lat'], user1['lng'], user2['lat'], user2['lng'])
            
            if straight_distance > 0:
                detour_ratio = road_distance / straight_distance
                total_detour_ratio += detour_ratio
                segment_count += 1
                
                if detour_ratio > 2.5:  # Major detour between users
                    problem_segments.append({
                        'type': 'user_to_user',
                        'detour_ratio': detour_ratio,
                        'positions': [(user1['lat'], user1['lng']), (user2['lat'], user2['lng'])]
                    })
        
        # Last user to office
        if users:
            last_user = users[-1]
            road_distance = road_network.get_road_distance(last_user['lat'], last_user['lng'], office_pos[0], office_pos[1])
            straight_distance = haversine_distance(last_user['lat'], last_user['lng'], office_pos[0], office_pos[1])
            
            if straight_distance > 0:
                detour_ratio = road_distance / straight_distance
                total_detour_ratio += detour_ratio
                segment_count += 1
                
                if detour_ratio > 2.5:  # Major detour to office
                    problem_segments.append({
                        'type': 'last_to_office',
                        'detour_ratio': detour_ratio,
                        'positions': [(last_user['lat'], last_user['lng']), office_pos]
                    })
        
        # Calculate overall connectivity severity
        if segment_count > 0:
            avg_detour_ratio = total_detour_ratio / segment_count
            problem_ratio = len(problem_segments) / segment_count
            
            # Severity based on average detour and proportion of problem segments
            connectivity_problems['severity'] = min(1.0, (avg_detour_ratio - 1.0) * 0.5 + problem_ratio * 0.5)
            connectivity_problems['problem_areas'] = problem_segments
        
    except Exception as e:
        logger.warning(f"Road connectivity analysis failed: {e}")
        connectivity_problems['severity'] = 0.0
    
    return connectivity_problems


def split_route_by_road_connectivity(route, available_drivers, office_lat, office_lon, connectivity_problems):
    """Split route based on road connectivity issues"""
    users = route['assigned_users']
    if len(users) < 4 or not connectivity_problems['problem_areas']:
        return [route]
    
    # Identify users that are causing connectivity problems
    problem_users = set()
    for problem in connectivity_problems['problem_areas']:
        if problem['type'] == 'user_to_user' and problem['detour_ratio'] > 3.0:
            # Mark users involved in high-detour segments
            for pos in problem['positions']:
                for user in users:
                    if abs(user['lat'] - pos[0]) < 0.001 and abs(user['lng'] - pos[1]) < 0.001:
                        problem_users.add(user['user_id'])
    
    if not problem_users:
        return [route]
    
    # Split users into two groups: connected and problematic
    connected_users = [u for u in users if u['user_id'] not in problem_users]
    problematic_users = [u for u in users if u['user_id'] in problem_users]
    
    # Ensure minimum group sizes
    if len(connected_users) < 2 or len(problematic_users) < 2:
        return [route]
    
    split_routes = []
    
    # Keep original route with connected users
    route_copy = route.copy()
    route_copy['assigned_users'] = connected_users
    route_copy = optimize_route_sequence(route_copy, office_lat, office_lon)
    split_routes.append(route_copy)
    
    # Create new route for problematic users if driver available
    if len(available_drivers) > 0 and len(problematic_users) >= 2:
        best_driver = available_drivers.iloc[0]
        
        new_route = {
            'driver_id': str(best_driver['driver_id']),
            'vehicle_id': str(best_driver.get('vehicle_id', '')),
            'vehicle_type': int(best_driver['capacity']),
            'latitude': float(best_driver['latitude']),
            'longitude': float(best_driver['longitude']),
            'assigned_users': problematic_users
        }
        
        new_route = optimize_route_sequence(new_route, office_lat, office_lon)
        split_routes.append(new_route)
        
        logger.info(f"    ✂️ Split route based on connectivity: {len(connected_users)} + {len(problematic_users)} users")
    
    return split_routes if len(split_routes) > 1 else [route]


def calculate_route_bearing_spread(route, office_lat, office_lon):
    """Calculate the bearing spread of users in a route"""
    if len(route['assigned_users']) <= 1:
        return 0
    
    bearings = []
    for user in route['assigned_users']:
        bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])
        bearings.append(bearing)
    
    bearing_spread = max(bearings) - min(bearings)
    if bearing_spread > 180:  # Handle circular nature
        bearing_spread = 360 - bearing_spread
    
    return bearing_spread


def split_route_by_bearing(route, available_drivers, office_lat, office_lon):
    """Split route by bearing clusters"""
    users = route['assigned_users']
    if len(users) < 3 or len(available_drivers) == 0:
        return [route]
    
    # Calculate bearings for all users
    user_bearings = []
    for user in users:
        bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])
        user_bearings.append((user, bearing))
    
    # Sort by bearing
    user_bearings.sort(key=lambda x: x[1])
    
    # Find split point (largest bearing gap)
    best_split = len(users) // 2  # Default to middle
    largest_gap = 0
    
    for i in range(1, len(user_bearings)):
        bearing_gap = user_bearings[i][1] - user_bearings[i-1][1]
        if bearing_gap > 180:  # Handle circular nature
            bearing_gap = 360 - bearing_gap
        if bearing_gap > largest_gap and i >= len(users) // 3:  # Don't create tiny groups
            largest_gap = bearing_gap
            best_split = i
    
    # Split users
    group1 = [item[0] for item in user_bearings[:best_split]]
    group2 = [item[0] for item in user_bearings[best_split:]]
    
    # Create routes
    routes = []
    
    # Keep original route with first group
    route['assigned_users'] = group1
    route = optimize_route_sequence_ultra_strict(route, office_lat, office_lon)
    routes.append(route)
    
    # Create new route with second group
    if group2 and len(available_drivers) > 0:
        new_driver = available_drivers.iloc[0]
        new_route = {
            'driver_id': str(new_driver['driver_id']),
            'vehicle_id': str(new_driver.get('vehicle_id', '')),
            'vehicle_type': int(new_driver['capacity']),
            'latitude': float(new_driver['latitude']),
            'longitude': float(new_driver['longitude']),
            'assigned_users': group2
        }
        new_route = optimize_route_sequence_ultra_strict(new_route, office_lat, office_lon)
        routes.append(new_route)
    
    return routes


def split_route_by_distance(route, available_drivers, office_lat, office_lon):
    """Split route by distance from office"""
    users = route['assigned_users']
    if len(users) < 3 or len(available_drivers) == 0:
        return [route]
    
    # Calculate distances and sort
    users_with_distance = []
    for user in users:
        distance = haversine_distance(office_lat, office_lon, user['lat'], user['lng'])
        users_with_distance.append((user, distance))
    
    users_with_distance.sort(key=lambda x: x[1])
    
    # Split at median distance
    mid_point = len(users) // 2
    group1 = [item[0] for item in users_with_distance[:mid_point]]
    group2 = [item[0] for item in users_with_distance[mid_point:]]
    
    routes = []
    
    # Keep original route with first group
    route['assigned_users'] = group1
    route = optimize_route_sequence_ultra_strict(route, office_lat, office_lon)
    routes.append(route)
    
    # Create new route with second group
    if group2 and len(available_drivers) > 0:
        new_driver = available_drivers.iloc[0]
        new_route = {
            'driver_id': str(new_driver['driver_id']),
            'vehicle_id': str(new_driver.get('vehicle_id', '')),
            'vehicle_type': int(new_driver['capacity']),
            'latitude': float(new_driver['latitude']),
            'longitude': float(new_driver['longitude']),
            'assigned_users': group2
        }
        new_route = optimize_route_sequence_ultra_strict(new_route, office_lat, office_lon)
        routes.append(new_route)
    
    return routes


def handle_remaining_unassigned(unassigned_users, user_df, assigned_user_ids):
    """Handle remaining unassigned users"""
    final_unassigned = user_df[~user_df['user_id'].isin(assigned_user_ids)]
    unassigned_list = []

    for _, user in final_unassigned.iterrows():
        user_data = {
            'user_id': str(user['user_id']),
            'lat': float(user['latitude']),
            'lng': float(user['longitude'])
        }
        if pd.notna(user.get('first_name')):
            user_data['first_name'] = str(user['first_name'])
        if pd.notna(user.get('email')):
            user_data['email'] = str(user['email'])
        unassigned_list.append(user_data)

    return unassigned_list


# Helper functions
def calculate_route_center(route):
    """Calculate the center of a route"""
    if not route['assigned_users']:
        return (route['latitude'], route['longitude'])

    lats = [u['lat'] for u in route['assigned_users']]
    lngs = [u['lng'] for u in route['assigned_users']]
    return (np.mean(lats), np.mean(lngs))


def calculate_average_bearing(route, office_lat, office_lon):
    """Calculate average bearing of route users from office"""
    if not route['assigned_users']:
        return calculate_bearing(office_lat, office_lon, route['latitude'], route['longitude'])

    bearings = []
    for user in route['assigned_users']:
        bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])
        bearings.append(bearing)

    return np.mean(bearings)


def calculate_route_cost(route, office_lat, office_lon):
    """Calculate total route distance"""
    if not route['assigned_users']:
        return 0

    total_cost = 0
    current_pos = (route['latitude'], route['longitude'])

    for user in route['assigned_users']:
        user_pos = (user['lat'], user['lng'])
        total_cost += haversine_distance(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
        current_pos = user_pos

    # Add distance to office
    total_cost += haversine_distance(current_pos[0], current_pos[1], office_lat, office_lon)

    return total_cost


def calculate_turning_score(route, office_lat, office_lon):
    """Calculate average turning angle for a route"""
    if len(route['assigned_users']) <= 1:
        return 0

    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (office_lat, office_lon)
    route_points = [driver_pos] + [(u['lat'], u['lng']) for u in route['assigned_users']] + [office_pos]

    turning_angles = []
    for i in range(1, len(route_points) - 1):
        p1, p2, p3 = route_points[i-1], route_points[i], route_points[i+1]

        bearing1 = calculate_bearing(p1[0], p1[1], p2[0], p2[1])
        bearing2 = calculate_bearing(p2[0], p2[1], p3[0], p3[1])

        turning_angle = bearing_difference(bearing1, bearing2)
        turning_angles.append(turning_angle)

    return np.mean(turning_angles) if turning_angles else 0


def update_route_metrics(route, office_lat, office_lon):
    """Update route metrics"""
    if route['assigned_users']:
        route['utilization'] = len(route['assigned_users']) / route['vehicle_type']
        route['turning_score'] = calculate_turning_score(route, office_lat, office_lon)
        route['total_distance'] = calculate_route_cost(route, office_lat, office_lon)
    else:
        route['utilization'] = 0
        route['turning_score'] = 0
        route['total_distance'] = 0