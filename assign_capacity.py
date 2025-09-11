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

# Disable road network for better bearing-based routing
road_network = None
logger = get_logger()
logger.info("🎯 Using bearing-focused routing without road network dependency")


# Bearing-focused routing class
class BearingFocusedRouting:

    def get_road_distance(self, lat1, lon1, lat2, lon2):
        return haversine_distance(lat1, lon1, lat2, lon2) * 1.2  # Slightly conservative estimate

    def is_user_on_route_path(self,
                              driver_pos,
                              existing_users,
                              candidate_pos,
                              office_pos,
                              max_detour_ratio=1.10,
                              route_type="capacity"):
        """Ultra-strict bearing-based validation for directional consistency"""

        # Calculate main route bearing (office to driver direction, reversed)
        main_bearing = calculate_bearing(office_pos[0], office_pos[1], driver_pos[0], driver_pos[1])

        # Calculate candidate bearing from office
        candidate_bearing = calculate_bearing(office_pos[0], office_pos[1], candidate_pos[0], candidate_pos[1])

        # ULTRA STRICT: Maximum 12 degrees bearing difference
        bearing_diff = bearing_difference(main_bearing, candidate_bearing)
        if bearing_diff > 12:
            return False

        # Distance progression check - candidate should be closer to office than driver
        driver_to_office = haversine_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
        candidate_to_office = haversine_distance(candidate_pos[0], candidate_pos[1], office_pos[0], office_pos[1])

        # Must be making progress toward office
        if candidate_to_office > driver_to_office + 0.5:  # 500m tolerance
            return False

        # ULTRA STRICT detour check
        driver_to_candidate = haversine_distance(driver_pos[0], driver_pos[1], candidate_pos[0], candidate_pos[1])
        total_distance = driver_to_candidate + candidate_to_office

        if driver_to_office > 0:
            detour_ratio = total_distance / driver_to_office
            if detour_ratio > 1.08:  # Maximum 8% detour
                return False

        # If existing users, check bearing consistency within the route
        if existing_users:
            for user_pos in existing_users:
                user_bearing = calculate_bearing(office_pos[0], office_pos[1], user_pos[0], user_pos[1])
                user_bearing_diff = bearing_difference(candidate_bearing, user_bearing)
                if user_bearing_diff > 8:  # Maximum 8 degrees between users in same route
                    return False

        return True

    def get_route_coherence_score(self, driver_pos, user_positions, office_pos):
        """Calculate coherence based purely on bearing consistency"""
        if not user_positions:
            return 1.0

        # Calculate bearing from office to driver
        main_bearing = calculate_bearing(office_pos[0], office_pos[1], driver_pos[0], driver_pos[1])

        # Check bearing consistency for all users
        total_deviation = 0.0
        for user_pos in user_positions:
            user_bearing = calculate_bearing(office_pos[0], office_pos[1], user_pos[0], user_pos[1])
            deviation = bearing_difference(main_bearing, user_bearing)
            total_deviation += deviation

        if len(user_positions) == 0:
            return 1.0

        avg_deviation = total_deviation / len(user_positions)

        # Score based on average deviation (lower is better)
        if avg_deviation <= 10:
            return 1.0
        elif avg_deviation <= 20:
            return 0.8
        elif avg_deviation <= 30:
            return 0.6
        else:
            return 0.3


# Always use bearing-focused routing for better directional consistency
road_network = BearingFocusedRouting()


# Load and validate configuration
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

    # Always use capacity optimization mode
    current_mode = "capacity_optimization"

    # Get capacity optimization configuration
    mode_configs = cfg.get("mode_configs", {})
    mode_config = mode_configs.get("capacity_optimization", {})

    logger = get_logger()
    logger.info("🎯 Using optimization mode: CAPACITY OPTIMIZATION")

    # Validate and set configuration with mode-specific overrides
    config = {}

    # Distance configurations - stricter for bearing-focused routing
    config['MAX_FILL_DISTANCE_KM'] = max(
        0.1,
        float(
            mode_config.get("max_fill_distance_km",
                            cfg.get("max_fill_distance_km", 8.0))))  # Reduced from 15.0
    config['MERGE_DISTANCE_KM'] = max(
        0.1,
        float(
            mode_config.get("merge_distance_km",
                            cfg.get("merge_distance_km", 6.0))))  # Reduced from 10.0
    config['DBSCAN_EPS_KM'] = max(0.1, float(cfg.get("dbscan_eps_km", 2.0)))  # Reduced from 3.0
    config['MIN_SAMPLES_DBSCAN'] = max(1, int(cfg.get("min_samples_dbscan",
                                                      2)))

    # Angle configurations - much stricter for bearing-focused routing
    config['MAX_BEARING_DIFFERENCE'] = max(
        0,
        min(
            180,
            float(
                mode_config.get("max_bearing_difference",
                                cfg.get("max_bearing_difference", 15)))))  # Reduced from 60
    config['MAX_TURNING_ANGLE'] = max(
        0,
        min(
            180,
            float(
                mode_config.get("max_allowed_turning_score",
                                cfg.get("max_allowed_turning_score", 30)))))  # Reduced from 90

    # Capacity optimization specific settings
    config['capacity_weight'] = mode_config.get('capacity_weight', 10.0)
    config['direction_weight'] = mode_config.get('direction_weight', 1.0)
    config['aggressive_merging'] = mode_config.get('aggressive_merging', True)

    # Clustering parameters
    config['max_users_per_cluster'] = cfg.get('max_users_per_cluster', 8)
    config['min_cluster_size'] = cfg.get('min_cluster_size', 2)

    # Office coordinates
    office_lat = float(
        os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489)))
    office_lon = float(
        os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))
    config['OFFICE_LAT'] = office_lat
    config['OFFICE_LON'] = office_lon

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


# Load validated configuration
_config = load_and_validate_config()


def validate_input_data(data):
    """Comprehensive data validation"""
    logger = get_logger()
    if not isinstance(data, dict):
        raise ValueError("API response must be a dictionary")

    users = data.get("users", [])
    if not users:
        raise ValueError("No users found in API response")

    # Validate each user
    for i, user in enumerate(users):
        if not isinstance(user, dict):
            raise ValueError(f"User {i} must be a dictionary")

        required_fields = ["id", "latitude", "longitude"]
        for field in required_fields:
            if field not in user:
                raise ValueError(f"User {i} missing required field: {field}")

        try:
            lat = float(user["latitude"])
            lon = float(user["longitude"])
            if not (-90 <= lat <= 90):
                raise ValueError(f"User {i} invalid latitude: {lat}")
            if not (-180 <= lon <= 180):
                raise ValueError(f"User {i} invalid longitude: {lon}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"User {i} invalid coordinates: {e}")

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
        raise ValueError("No drivers found in API response")

    # Validate drivers
    for i, driver in enumerate(all_drivers):
        if not isinstance(driver, dict):
            raise ValueError(f"Driver {i} must be a dictionary")

        required_fields = ["id", "capacity", "latitude", "longitude"]
        for field in required_fields:
            if field not in driver:
                raise ValueError(f"Driver {i} missing required field: {field}")

        try:
            lat = float(driver["latitude"])
            lon = float(driver["longitude"])
            capacity = int(driver["capacity"])
            if not (-90 <= lat <= 90):
                raise ValueError(f"Driver {i} invalid latitude: {lat}")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Driver {i} invalid longitude: {lon}")
            if capacity <= 0:
                raise ValueError(f"Driver {i} invalid capacity: {capacity}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Driver {i} invalid data: {e}")

    logger.info(
        f"✅ Input data validation passed - {len(users)} users, {len(all_drivers)} drivers"
    )


def load_env_and_fetch_data(source_id: str,
                            parameter: int = 1,
                            string_param: str = ""):
    """Load environment variables and fetch data from API"""
    logger = get_logger()
    env_path = ".env"
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        raise FileNotFoundError(f".env file not found at {env_path}")

    BASE_API_URL = os.getenv("API_URL")
    API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
    if not BASE_API_URL or not API_AUTH_TOKEN:
        raise ValueError("Both API_URL and API_AUTH_TOKEN must be set in .env")

    API_URL = f"{BASE_API_URL.rstrip('/')}/{source_id}/{parameter}/{string_param}"
    headers = {
        "Authorization": f"Bearer {API_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    logger.info(f"📡 Making API call to: {API_URL}")
    resp = requests.get(API_URL, headers=headers)
    resp.raise_for_status()

    try:
        payload = resp.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"API returned invalid JSON: {str(e)}")

    if not payload.get("status") or "data" not in payload:
        raise ValueError(
            "Unexpected response format: 'status' or 'data' missing")

    data = payload["data"]
    data["_parameter"] = parameter
    data["_string_param"] = string_param

    # Handle nested drivers structure
    if "drivers" in data:
        drivers_data = data["drivers"]
        data["driversUnassigned"] = drivers_data.get("driversUnassigned", [])
        data["driversAssigned"] = drivers_data.get("driversAssigned", [])
    else:
        data["driversUnassigned"] = data.get("driversUnassigned", [])
        data["driversAssigned"] = data.get("driversAssigned", [])

    return data


def extract_office_coordinates(data):
    """Extract dynamic office coordinates from API data"""
    company_data = data.get("company", {})
    office_lat = float(company_data.get("latitude", _config['OFFICE_LAT']))
    office_lon = float(company_data.get("longitude", _config['OFFICE_LON']))
    return office_lat, office_lon


def prepare_user_driver_dataframes(data):
    """Prepare user and driver dataframes from API data"""
    # Prepare user DataFrame
    users = data.get("users", [])
    user_data = []
    for user in users:
        user_data.append({
            'user_id':
            str(user.get('id', '')),
            'latitude':
            float(user.get('latitude', 0.0)),
            'longitude':
            float(user.get('longitude', 0.0)),
            'first_name':
            str(user.get('first_name', '')),
            'email':
            str(user.get('email', '')),
            'office_distance':
            float(user.get('office_distance', 0.0))
        })

    user_df = pd.DataFrame(user_data)

    # Prepare driver DataFrame
    all_drivers = []
    if "drivers" in data:
        drivers_data = data["drivers"]
        all_drivers.extend(drivers_data.get("driversUnassigned", []))
        all_drivers.extend(drivers_data.get("driversAssigned", []))
    else:
        all_drivers.extend(data.get("driversUnassigned", []))
        all_drivers.extend(data.get("driversAssigned", []))

    driver_data = []
    for i, driver in enumerate(all_drivers):
        driver_data.append({
            'driver_id': str(driver.get('id', '')),
            'latitude': float(driver.get('latitude', 0.0)),
            'longitude': float(driver.get('longitude', 0.0)),
            'capacity': int(driver.get('capacity', 1)),
            'vehicle_id': str(driver.get('vehicle_id', '')),
            'priority': i + 1
        })

    driver_df = pd.DataFrame(driver_data)

    return user_df, driver_df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(
        dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371
    return c * r


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point A to B in degrees"""
    lat1, lat2 = map(math.radians, [lat1, lat2])
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
        lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def bearing_difference(b1, b2):
    """Compute minimum difference between two bearings"""
    diff = abs(b1 - b2) % 360
    return min(diff, 360 - diff)


# STEP 1: CLUSTERING
def step1_clustering(user_df, office_lat, office_lon, config):
    """Step 1: Create geographic clusters with capacity-friendly approach"""
    logger = get_logger()
    logger.info(
        "================================================================================"
    )
    logger.info("🚀 STARTING STEP 1: GEOGRAPHIC CLUSTERING")
    logger.info(
        "================================================================================"
    )

    if len(user_df) == 0:
        return user_df

    # Add bearing information
    user_df['bearing_from_office'] = user_df.apply(
        lambda row: calculate_bearing(office_lat, office_lon, row['latitude'],
                                      row['longitude']),
        axis=1)

    # Use DBSCAN for initial clustering with lenient parameters
    eps_km = config['DBSCAN_EPS_KM']
    min_samples = config['MIN_SAMPLES_DBSCAN']

    # Convert to metric coordinates for clustering
    coords_km = []
    for _, user in user_df.iterrows():
        lat_km = (user['latitude'] - office_lat) * config['LAT_TO_KM']
        lon_km = (user['longitude'] - office_lon) * config['LON_TO_KM']
        coords_km.append([lat_km, lon_km])

    coords_km = np.array(coords_km)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps_km, min_samples=min_samples)
    labels = dbscan.fit_predict(coords_km)

    # Handle noise points by assigning them to nearest cluster
    noise_mask = labels == -1
    if noise_mask.any():
        valid_labels = labels[~noise_mask]
        if len(valid_labels) > 0:
            for i in np.where(noise_mask)[0]:
                noise_point = coords_km[i]
                distances = cdist([noise_point], coords_km[~noise_mask])[0]
                nearest_cluster_idx = np.argmin(distances)
                labels[i] = valid_labels[nearest_cluster_idx]
        else:
            labels[:] = 0

    user_df['geo_cluster'] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    logger.info(
        f"✅ COMPLETED STEP 1: GEOGRAPHIC CLUSTERING - {n_clusters} clusters created"
    )
    return user_df


# STEP 2: SUBCLUSTERING
def step2_subclustering(user_df, office_lat, office_lon, config):
    """Step 2: Create capacity-aware subclusters"""
    logger = get_logger()
    logger.info(
        "================================================================================"
    )
    logger.info("🚀 STARTING STEP 2: CAPACITY-AWARE SUBCLUSTERING")
    logger.info(
        "================================================================================"
    )

    if len(user_df) == 0:
        return user_df

    user_df['sub_cluster'] = -1
    sub_cluster_counter = 0
    max_users_per_cluster = config.get('max_users_per_cluster', 8)

    for geo_cluster in user_df['geo_cluster'].unique():
        if geo_cluster == -1:
            continue

        geo_cluster_users = user_df[user_df['geo_cluster'] == geo_cluster]

        if len(geo_cluster_users) <= max_users_per_cluster:
            user_df.loc[geo_cluster_users.index,
                        'sub_cluster'] = sub_cluster_counter
            sub_cluster_counter += 1
        else:
            # Split large clusters into capacity-sized chunks
            n_subclusters = math.ceil(
                len(geo_cluster_users) / max_users_per_cluster)

            # Use K-means to split into subclusters
            coords = []
            for _, user in geo_cluster_users.iterrows():
                lat_km = (user['latitude'] - office_lat) * config['LAT_TO_KM']
                lon_km = (user['longitude'] - office_lon) * config['LON_TO_KM']
                coords.append([lat_km, lon_km])

            coords = np.array(coords)

            if n_subclusters > 1:
                kmeans = KMeans(n_clusters=n_subclusters,
                                random_state=42,
                                n_init=10)
                subcluster_labels = kmeans.fit_predict(coords)

                for i, (idx, _) in enumerate(geo_cluster_users.iterrows()):
                    user_df.loc[
                        idx,
                        'sub_cluster'] = sub_cluster_counter + subcluster_labels[
                            i]

                sub_cluster_counter += n_subclusters
            else:
                user_df.loc[geo_cluster_users.index,
                            'sub_cluster'] = sub_cluster_counter
                sub_cluster_counter += 1

    logger.info(
        f"✅ COMPLETED STEP 2: CAPACITY-AWARE SUBCLUSTERING - {user_df['sub_cluster'].nunique()} subclusters created"
    )
    return user_df


# STEP 3: ASSIGNING ROUTES
def step3_assign_routes(user_df, driver_df, office_lat, office_lon):
    """Step 3: Assign routes with capacity optimization focus and bearing coherence"""
    logger = get_logger()
    logger.info(
        "================================================================================"
    )
    logger.info(
        "🚀 STARTING STEP 3: INITIAL ROUTE ASSIGNMENT WITH BEARING COHERENCE")
    logger.info(
        "================================================================================"
    )

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Sort drivers by capacity (largest first) for capacity optimization
    available_drivers = driver_df.sort_values(['capacity', 'priority'],
                                              ascending=[False, True])

    logger.info(f"🔍 DEBUG: Processing {user_df['sub_cluster'].nunique()} sub-clusters")
    logger.info(f"🔍 DEBUG: Available drivers: {len(available_drivers)}")
    
    # Process each sub-cluster with bearing coherence check
    for sub_cluster_id in sorted(user_df['sub_cluster'].unique()):
        if sub_cluster_id == -1:
            continue

        cluster_users = user_df[user_df['sub_cluster'] == sub_cluster_id]
        unassigned_in_cluster = cluster_users[~cluster_users['user_id'].
                                              isin(assigned_user_ids)]

        if unassigned_in_cluster.empty:
            continue

        logger.info(f"🔍 DEBUG: Processing sub-cluster {sub_cluster_id} with {len(unassigned_in_cluster)} users")
        
        # Log user details in this cluster
        for _, user in unassigned_in_cluster.iterrows():
            user_bearing = user['bearing_from_office']
            logger.info(f"  🧭 User {user['user_id']}: bearing={user_bearing:.1f}°, lat={user['latitude']:.6f}, lng={user['longitude']:.6f}")

        # Check bearing coherence within cluster
        if len(unassigned_in_cluster) > 1:
            bearings = unassigned_in_cluster['bearing_from_office'].values
            bearing_diffs = []
            for i in range(len(bearings)):
                for j in range(i + 1, len(bearings)):
                    diff = bearing_difference(bearings[i], bearings[j])
                    bearing_diffs.append(diff)
                    logger.info(f"    🔄 Bearing diff between users {unassigned_in_cluster.iloc[i]['user_id']} and {unassigned_in_cluster.iloc[j]['user_id']}: {diff:.1f}°")

            max_bearing_diff = max(bearing_diffs) if bearing_diffs else 0
            logger.info(f"  📏 Max bearing difference in cluster {sub_cluster_id}: {max_bearing_diff:.1f}° (threshold: {_config['MAX_BEARING_DIFFERENCE']}°)")

            # If bearing spread is too large, split the cluster
            if max_bearing_diff > _config['MAX_BEARING_DIFFERENCE']:
                logger.info(
                    f"  📐 SPLITTING cluster {sub_cluster_id} due to bearing spread ({max_bearing_diff:.1f}°)"
                )
                split_clusters = split_cluster_by_bearing(
                    unassigned_in_cluster, office_lat, office_lon)

                logger.info(f"  🔗 Split into {len(split_clusters)} sub-clusters")
                for i, split_cluster in enumerate(split_clusters):
                    logger.info(f"    📦 Split cluster {i+1}: {len(split_cluster)} users")
                    for _, user in split_cluster.iterrows():
                        logger.info(f"      👤 User {user['user_id']}: bearing={user['bearing_from_office']:.1f}°")
                    
                    route = assign_driver_to_cluster(split_cluster,
                                                     available_drivers,
                                                     used_driver_ids,
                                                     office_lat, office_lon)
                    if route:
                        routes.append(route)
                        assigned_user_ids.update(
                            u['user_id'] for u in route['assigned_users'])
                        used_driver_ids.add(route['driver_id'])
                        logger.info(f"    ✅ Route created for split cluster {i+1} with driver {route['driver_id']}")
                    else:
                        logger.warning(f"    ❌ Failed to create route for split cluster {i+1}")
                continue

        # Normal assignment for coherent clusters
        logger.info(f"  ✅ Cluster {sub_cluster_id} is directionally coherent, assigning to single driver")
        route = assign_driver_to_cluster(unassigned_in_cluster,
                                         available_drivers, used_driver_ids,
                                         office_lat, office_lon)
        if route:
            routes.append(route)
            assigned_user_ids.update(u['user_id']
                                     for u in route['assigned_users'])
            used_driver_ids.add(route['driver_id'])
            logger.info(f"  ✅ Route created for cluster {sub_cluster_id} with driver {route['driver_id']}")
        else:
            logger.warning(f"  ❌ Failed to create route for cluster {sub_cluster_id}")

    logger.info(
        f"✅ COMPLETED STEP 3: INITIAL ROUTE ASSIGNMENT - {len(routes)} routes created"
    )
    
    # Log final route details for validation
    logger.info("🔍 DEBUG: Final routes from Step 3:")
    for i, route in enumerate(routes, 1):
        logger.info(f"  Route {i} - Driver {route['driver_id']}:")
        driver_bearing = calculate_bearing(office_lat, office_lon, route['latitude'], route['longitude'])
        logger.info(f"    🧭 Driver bearing: {driver_bearing:.1f}°")
        logger.info(f"    👥 Users ({len(route['assigned_users'])}):")
        for user in route['assigned_users']:
            user_bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])
            logger.info(f"      - User {user['user_id']}: bearing={user_bearing:.1f}°")
    
    return routes, assigned_user_ids, used_driver_ids


def validate_cluster_directional_consistency(cluster_users, office_lat, office_lon):
    """Validate that cluster has strict directional consistency"""
    if len(cluster_users) <= 1:
        return True

    # Calculate all bearings from office to users
    bearings = []
    for _, user in cluster_users.iterrows():
        bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
        bearings.append(bearing)

    # Check bearing spread
    min_bearing = min(bearings)
    max_bearing = max(bearings)

    # Handle wraparound case
    bearing_spread = max_bearing - min_bearing
    if bearing_spread > 180:
        bearing_spread = 360 - bearing_spread

    # STRICT: Maximum 20 degree spread within cluster
    if bearing_spread > 20:
        return False

    # Check distance consistency (users should be in similar distance range)
    distances = [haversine_distance(user['latitude'], user['longitude'], office_lat, office_lon)
                for _, user in cluster_users.iterrows()]

    min_distance = min(distances)
    max_distance = max(distances)

    # STRICT: Distance range should not be too large
    if max_distance - min_distance > 8.0:  # 8km max range
        return False

    return True


def split_cluster_by_bearing(cluster_users, office_lat, office_lon):
    """Split cluster by bearing to ensure STRICT directional coherence"""
    if len(cluster_users) <= 1:
        return [cluster_users]

    # Calculate bearings for all users
    bearings = []
    for _, user in cluster_users.iterrows():
        bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
        bearings.append(bearing)

    # Use K-means on bearing values (converted to 2D coordinates)
    coords_with_bearing = []
    for _, user in cluster_users.iterrows():
        bearing_rad = math.radians(user['bearing_from_office'])
        # Weight bearing very heavily for strict directional clustering
        x = math.cos(bearing_rad) * 5.0  # Increased weight
        y = math.sin(bearing_rad) * 5.0

        # Reduced geographic weight to prioritize direction
        lat_norm = (user['latitude'] - office_lat) * _config['LAT_TO_KM'] * 0.05
        lon_norm = (user['longitude'] - office_lon) * _config['LON_TO_KM'] * 0.05

        coords_with_bearing.append([x, y, lat_norm, lon_norm])

    coords_with_bearing = np.array(coords_with_bearing)

    # Split into 2 groups
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords_with_bearing)

    group1 = cluster_users[labels == 0]
    group2 = cluster_users[labels == 1]

    # Validate that both groups maintain directional consistency
    valid_groups = []
    for group in [group1, group2]:
        if validate_cluster_directional_consistency(group, office_lat, office_lon):
            valid_groups.append(group)
        else:
            # Further split if still not consistent
            if len(group) > 2:
                # Split into individual routes if necessary
                for _, user in group.iterrows():
                    single_user_df = group[group['user_id'] == user['user_id']]
                    valid_groups.append(single_user_df)
            else:
                valid_groups.append(group)

    return valid_groups if valid_groups else [group1, group2]


def assign_driver_to_cluster(cluster_users, available_drivers, used_driver_ids,
                             office_lat, office_lon):
    """Assign best driver to a cluster with STRICT directional validation"""
    if cluster_users.empty:
        logger.warning("🔍 assign_driver_to_cluster: Empty cluster_users")
        return None

    logger.info(f"🔍 assign_driver_to_cluster: Processing cluster with {len(cluster_users)} users")
    
    best_driver = None
    best_score = float('inf')
    candidates_checked = 0
    candidates_rejected = {}

    # Calculate cluster properties with strict directional validation
    cluster_center = (cluster_users['latitude'].mean(),
                      cluster_users['longitude'].mean())
    cluster_bearing = cluster_users['bearing_from_office'].mean()

    logger.info(f"  🧭 Cluster center: ({cluster_center[0]:.6f}, {cluster_center[1]:.6f})")
    logger.info(f"  🧭 Cluster bearing: {cluster_bearing:.1f}°")

    # STRICT: Validate cluster directional consistency before assignment
    is_consistent = validate_cluster_directional_consistency(cluster_users, office_lat, office_lon)
    logger.info(f"  🔍 Cluster directional consistency: {'✅ PASS' if is_consistent else '❌ FAIL'}")
    
    if not is_consistent:
        logger.warning(f"❌ Cluster failed strict directional validation, skipping assignment")
        return None

    for _, driver in available_drivers.iterrows():
        candidates_checked += 1
        driver_id = driver['driver_id']
        
        logger.info(f"  🚗 Checking driver {driver_id} (candidate {candidates_checked})")
        
        if driver['driver_id'] in used_driver_ids:
            candidates_rejected[driver_id] = "Already used"
            logger.info(f"    ❌ Driver {driver_id}: Already used")
            continue

        # Check if driver can accommodate all users
        if driver['capacity'] < len(cluster_users):
            candidates_rejected[driver_id] = f"Insufficient capacity ({driver['capacity']} < {len(cluster_users)})"
            logger.info(f"    ❌ Driver {driver_id}: Insufficient capacity ({driver['capacity']} < {len(cluster_users)})")
            continue

        # Check bearing compatibility - ULTRA STRICT directional validation
        driver_bearing = calculate_bearing(office_lat, office_lon,
                                           driver['latitude'],
                                           driver['longitude'])
        bearing_diff = bearing_difference(driver_bearing, cluster_bearing)
        
        logger.info(f"    🧭 Driver {driver_id}: bearing={driver_bearing:.1f}°, diff from cluster={bearing_diff:.1f}°")

        # ULTRA strict bearing requirement for initial assignment
        if bearing_diff > 5:  # Maximum 5 degrees deviation for drivers
            candidates_rejected[driver_id] = f"Bearing diff too large ({bearing_diff:.1f}° > 5°)"
            logger.info(f"    ❌ Driver {driver_id}: Bearing diff too large ({bearing_diff:.1f}° > 5°)")
            continue

        # Additional check: driver should be positioned reasonably for the direction
        # Driver should be in roughly the same direction as the cluster from office
        driver_to_office_bearing = calculate_bearing(driver['latitude'], driver['longitude'], office_lat, office_lon)
        office_to_cluster_bearing = calculate_bearing(office_lat, office_lon, cluster_center[0], cluster_center[1])
        driver_position_compatibility = bearing_difference(driver_to_office_bearing, office_to_cluster_bearing)

        logger.info(f"    🔄 Driver {driver_id}: position compatibility={driver_position_compatibility:.1f}°")

        if driver_position_compatibility > 15:  # Driver must be positioned sensibly
            candidates_rejected[driver_id] = f"Position incompatible ({driver_position_compatibility:.1f}° > 15°)"
            logger.info(f"    ❌ Driver {driver_id}: Position incompatible ({driver_position_compatibility:.1f}° > 15°)")
            continue

        # Calculate driver to cluster center distance
        distance = haversine_distance(driver['latitude'], driver['longitude'],
                                      cluster_center[0], cluster_center[1])

        logger.info(f"    📏 Driver {driver_id}: distance to cluster={distance:.2f}km")

        # STRICT: Driver should be in reasonable proximity to cluster
        max_distance = _config['MAX_FILL_DISTANCE_KM'] * 0.5
        if distance > max_distance:  # Reduced tolerance
            candidates_rejected[driver_id] = f"Too far from cluster ({distance:.2f}km > {max_distance:.2f}km)"
            logger.info(f"    ❌ Driver {driver_id}: Too far from cluster ({distance:.2f}km > {max_distance:.2f}km)")
            continue

        # Stronger capacity utilization bonus
        utilization = len(cluster_users) / driver['capacity']
        utilization_bonus = utilization * _config['capacity_weight'] * 1.5

        # INCREASED direction penalty for strict directional consistency
        direction_penalty = bearing_diff * (_config['direction_weight'] * 2.0) / 50.0

        # Combined score (lower is better)
        score = distance + direction_penalty - utilization_bonus

        logger.info(f"    📊 Driver {driver_id}: score={score:.3f} (distance={distance:.2f}, direction_penalty={direction_penalty:.3f}, utilization_bonus={utilization_bonus:.3f})")

        if score < best_score:
            best_score = score
            best_driver = driver
            logger.info(f"    ⭐ Driver {driver_id}: NEW BEST CANDIDATE with score {score:.3f}")

    logger.info(f"  📈 Summary: Checked {candidates_checked} drivers")
    logger.info(f"  📈 Rejected drivers: {len(candidates_rejected)}")
    for driver_id, reason in candidates_rejected.items():
        logger.info(f"    ❌ {driver_id}: {reason}")

    if best_driver is None:
        logger.warning(f"  ❌ No suitable driver found for cluster")
        return None

    logger.info(f"  🎯 SELECTED: Driver {best_driver['driver_id']} with score {best_score:.3f}")

    # Create route
    route = {
        'driver_id': str(best_driver['driver_id']),
        'vehicle_id': str(best_driver.get('vehicle_id', '')),
        'vehicle_type': int(best_driver['capacity']),
        'latitude': float(best_driver['latitude']),
        'longitude': float(best_driver['longitude']),
        'assigned_users': []
    }

    # Add all users from this cluster, sorted by office distance
    cluster_users_sorted = cluster_users.sort_values('office_distance',
                                                     ascending=False)

    logger.info(f"  👥 Adding {len(cluster_users_sorted)} users to route:")
    for _, user in cluster_users_sorted.iterrows():
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
        
        user_bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
        logger.info(f"    👤 User {user['user_id']}: bearing={user_bearing:.1f}°, office_distance={user.get('office_distance', 0):.2f}km")

    utilization = len(route['assigned_users']) / route['vehicle_type'] * 100
    avg_bearing = cluster_users['bearing_from_office'].mean()
    logger.info(
        f"  ✅ ROUTE CREATED: Driver {route['driver_id']}: {len(route['assigned_users'])}/{route['vehicle_type']} users ({utilization:.1f}%) - Avg Bearing: {avg_bearing:.1f}°"
    )

    return route


# STEP 4: ROAD NETWORK ON-THE-WAY PICKUP
def step4_road_network_pickup(routes, user_df, assigned_user_ids, office_lat,
                              office_lon):
    """Step 4: Use road network to pick up users on the way with STRICT directional validation"""
    logger = get_logger()
    logger.info(
        "================================================================================"
    )
    logger.info("🚀 STARTING STEP 4: STRICT DIRECTIONAL ON-THE-WAY PICKUP")
    logger.info(
        "================================================================================"
    )

    total_pickups = 0
    unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]
    
    logger.info(f"🔍 Step 4 DEBUG: {len(unassigned_users)} unassigned users available for pickup")
    logger.info(f"🔍 Step 4 DEBUG: Processing {len(routes)} existing routes")

    for route_idx, route in enumerate(routes):
        logger.info(f"  🚗 Processing Route {route_idx + 1} - Driver {route['driver_id']}")
        logger.info(f"    📊 Current capacity: {len(route['assigned_users'])}/{route['vehicle_type']}")
        
        if len(route['assigned_users']) >= route['vehicle_type']:
            logger.info(f"    ❌ Route is full, skipping")
            continue  # Route is full

        driver_pos = (route['latitude'], route['longitude'])
        office_pos = (office_lat, office_lon)
        existing_users = [(u['lat'], u['lng'])
                          for u in route['assigned_users']]

        # Calculate STRICT route bearing - use existing users' average bearing
        route_corridor_bearing = calculate_strict_route_bearing(route, office_lat, office_lon)
        
        logger.info(f"    🧭 Route corridor bearing: {route_corridor_bearing:.1f}°")

        # Get current route corridor boundaries
        existing_users_bearings = []
        for existing_user in route['assigned_users']:
            user_bearing = calculate_bearing(office_lat, office_lon, existing_user['lat'], existing_user['lng'])
            existing_users_bearings.append(user_bearing)
            logger.info(f"      👤 Existing user {existing_user['user_id']}: bearing={user_bearing:.1f}°")

        # Calculate corridor boundaries (if we have existing users)
        if existing_users_bearings:
            corridor_min = min(existing_users_bearings)
            corridor_max = max(existing_users_bearings)
            corridor_spread = corridor_max - corridor_min

            # Handle wraparound
            if corridor_spread > 180:
                corridor_spread = 360 - corridor_spread
                # Adjust corridor boundaries for wraparound case
                if corridor_spread < 10:  # Only if tight corridor
                    corridor_min, corridor_max = corridor_max, corridor_min + 360
                    
            logger.info(f"    📏 Corridor boundaries: [{corridor_min:.1f}°, {corridor_max:.1f}°] (spread: {corridor_spread:.1f}°)")
        else:
            corridor_min = route_corridor_bearing - 5
            corridor_max = route_corridor_bearing + 5
            corridor_spread = 10
            logger.info(f"    📏 New corridor boundaries: [{corridor_min:.1f}°, {corridor_max:.1f}°] (default)")

        users_to_add = []
        users_evaluated = 0
        users_rejected = {}

        for _, user in unassigned_users.iterrows():
            users_evaluated += 1
            user_id = user['user_id']
            
            if user['user_id'] in assigned_user_ids:
                users_rejected[user_id] = "Already assigned"
                continue

            if len(route['assigned_users']) + len(users_to_add) >= route['vehicle_type']:
                users_rejected[user_id] = "Route would be full"
                break

            candidate_pos = (user['latitude'], user['longitude'])
            logger.info(f"      🔍 Evaluating User {user_id} for pickup")

            # STEP 1: ULTRA STRICT corridor validation
            candidate_bearing = calculate_bearing(office_lat, office_lon, candidate_pos[0], candidate_pos[1])
            logger.info(f"        🧭 User {user_id}: bearing={candidate_bearing:.1f}°")

            # Check if candidate is within the established corridor
            bearing_diff_from_corridor = bearing_difference(route_corridor_bearing, candidate_bearing)
            logger.info(f"        📏 User {user_id}: corridor deviation={bearing_diff_from_corridor:.1f}° (max: 5°)")
            
            if bearing_diff_from_corridor > 5:  # ULTRA strict - maximum 5 degrees from corridor center
                users_rejected[user_id] = f"Corridor deviation {bearing_diff_from_corridor:.1f}° > 5°"
                logger.info(f"        ❌ User {user_id}: REJECTED - corridor deviation {bearing_diff_from_corridor:.1f}° > 5°")
                continue

            # If we have existing users, candidate must fit within their corridor boundaries
            if existing_users_bearings:
                # Check if candidate bearing fits within existing corridor (with small expansion)
                corridor_tolerance = 3  # Allow 3 degrees expansion
                adjusted_min = corridor_min - corridor_tolerance
                adjusted_max = corridor_max + corridor_tolerance

                logger.info(f"        📏 User {user_id}: checking corridor boundaries [{adjusted_min:.1f}°, {adjusted_max:.1f}°]")

                # Handle wraparound case
                if adjusted_max - adjusted_min > 360:
                    # Wide corridor due to wraparound - check differently
                    if not (candidate_bearing >= adjusted_min or candidate_bearing <= adjusted_max - 360):
                        users_rejected[user_id] = f"Outside corridor boundaries (wraparound) {candidate_bearing:.1f}°"
                        logger.info(f"        ❌ User {user_id}: REJECTED - outside corridor boundaries (wraparound)")
                        continue
                else:
                    # Normal case
                    if not (adjusted_min <= candidate_bearing <= adjusted_max):
                        users_rejected[user_id] = f"Outside corridor boundaries {candidate_bearing:.1f}° not in [{adjusted_min:.1f}°, {adjusted_max:.1f}°]"
                        logger.info(f"        ❌ User {user_id}: REJECTED - outside corridor boundaries {candidate_bearing:.1f}° not in [{adjusted_min:.1f}°, {adjusted_max:.1f}°]")
                        continue

            # STEP 2: Progressive distance validation - MUST be making progress toward office
            driver_to_office_distance = haversine_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
            candidate_to_office_distance = haversine_distance(candidate_pos[0], candidate_pos[1], office_pos[0], office_pos[1])

            logger.info(f"        📏 User {user_id}: distance validation - candidate_to_office={candidate_to_office_distance:.2f}km, driver_to_office={driver_to_office_distance:.2f}km")

            # Candidate must be closer to office than driver (strict progression)
            if candidate_to_office_distance >= driver_to_office_distance + 0.2:  # Reduced tolerance to 200m
                users_rejected[user_id] = f"Not progressing toward office {candidate_to_office_distance:.2f}km vs {driver_to_office_distance:.2f}km"
                logger.info(f"        ❌ User {user_id}: REJECTED - not progressing toward office {candidate_to_office_distance:.2f}km vs {driver_to_office_distance:.2f}km")
                continue

            # STEP 3: ULTRA STRICT detour validation
            driver_to_candidate = haversine_distance(driver_pos[0], driver_pos[1], candidate_pos[0], candidate_pos[1])
            candidate_to_office = haversine_distance(candidate_pos[0], candidate_pos[1], office_pos[0], office_pos[1])

            if driver_to_office_distance > 0:
                detour_ratio = (driver_to_candidate + candidate_to_office) / driver_to_office_distance
                logger.info(f"        📏 User {user_id}: detour ratio={detour_ratio:.3f} (max: 1.08)")
                
                if detour_ratio > 1.08:  # ULTRA strict 8% detour limit (reduced from 15%)
                    users_rejected[user_id] = f"Detour ratio {detour_ratio:.3f} > 1.08"
                    logger.info(f"        ❌ User {user_id}: REJECTED - detour ratio {detour_ratio:.3f} > 1.08")
                    continue

            # STEP 4: Proximity to existing route validation
            if existing_users:
                min_distance_to_route = min(haversine_distance(candidate_pos[0], candidate_pos[1], u[0], u[1]) for u in existing_users)
                max_distance_allowed = _config['MAX_FILL_DISTANCE_KM'] * 0.3
                logger.info(f"        📏 User {user_id}: min distance to route={min_distance_to_route:.2f}km (max: {max_distance_allowed:.2f}km)")
                
                if min_distance_to_route > max_distance_allowed:  # Reduced from 0.4
                    users_rejected[user_id] = f"Too far from route {min_distance_to_route:.2f}km > {max_distance_allowed:.2f}km"
                    logger.info(f"        ❌ User {user_id}: REJECTED - too far from route {min_distance_to_route:.2f}km")
                    continue

            # STEP 5: ULTRA STRICT road network validation
            road_network_valid = road_network.is_user_on_route_path(driver_pos, existing_users, candidate_pos, office_pos, max_detour_ratio=1.06, route_type="capacity")
            logger.info(f"        🛣️ User {user_id}: road network validation={'✅ PASS' if road_network_valid else '❌ FAIL'}")
            
            if not road_network_valid:
                users_rejected[user_id] = "Failed road network validation"
                logger.info(f"        ❌ User {user_id}: REJECTED - failed road network validation")
                continue

            # STEP 6: Calculate quality score for ranking
            direction_score = bearing_diff_from_corridor / 5.0  # Normalize to 0-1
            efficiency_score = (detour_ratio - 1.0) * 20  # Detour penalty
            proximity_score = min_distance_to_route / _config['MAX_FILL_DISTANCE_KM'] if existing_users else 0

            combined_score = direction_score * 0.6 + efficiency_score * 0.3 + proximity_score * 0.1
            users_to_add.append((user, bearing_diff_from_corridor, driver_to_candidate, combined_score))

            logger.info(f"        ✅ User {user_id}: ACCEPTED for route {route['driver_id']} - corridor_dev={bearing_diff_from_corridor:.1f}°, detour={detour_ratio:.3f}, score={combined_score:.3f}")

        logger.info(f"    📊 Route {route['driver_id']} evaluation summary:")
        logger.info(f"      👥 Users evaluated: {users_evaluated}")
        logger.info(f"      ✅ Users accepted: {len(users_to_add)}")
        logger.info(f"      ❌ Users rejected: {len(users_rejected)}")
        
        for user_id, reason in users_rejected.items():
            logger.info(f"        ❌ {user_id}: {reason}")

        # Sort by combined quality score (lower is better) for proper tie-breaking
        users_to_add.sort(key=lambda x: x[3])  # Sort by combined_score

        # Add the picked up users
        for user_tuple in users_to_add:
            if len(user_tuple) == 4:
                user, bearing_diff, projection, combined_score = user_tuple
            else:
                # Handle case where tuple doesn't have expected length
                user = user_tuple[0]
                bearing_diff = user_tuple[1] if len(user_tuple) > 1 else 0
                projection = user_tuple[2] if len(user_tuple) > 2 else 0
                combined_score = 0
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
            total_pickups += 1

        if users_to_add:
            bearing_diffs = []
            projections = []
            for user_tuple in users_to_add:
                if len(user_tuple) >= 4:
                    bearing_diffs.append(user_tuple[1])
                    projections.append(user_tuple[2])
                else:
                    bearing_diffs.append(0)
                    projections.append(0)

            avg_bearing_diff = np.mean(bearing_diffs) if bearing_diffs else 0
            avg_projection = np.mean(projections) if projections else 0
            logger.info(
                f"  -> Route {route['driver_id']} (bearing: {route_corridor_bearing:.1f}°): Picked up {len(users_to_add)} users (avg bearing diff: {avg_bearing_diff:.1f}°, avg projection: {avg_projection:.2f}km)"
            )

    logger.info(
        f"✅ COMPLETED STEP 4: BEARING-AWARE ON-THE-WAY PICKUP - {total_pickups} users picked up"
    )
    return routes, assigned_user_ids


# STEP 5: CAPACITY-BASED MERGING
def step5_capacity_merging(routes, office_lat, office_lon):
    """Step 5: Merge routes with STRICT directional consistency"""
    logger = get_logger()
    logger.info(
        "================================================================================"
    )
    logger.info("🚀 STARTING STEP 5: STRICT DIRECTIONAL CAPACITY MERGING")
    logger.info(
        "================================================================================"
    )

    merges_completed = 0
    max_distance = _config['MERGE_DISTANCE_KM'] * 0.8  # More restrictive distance
    max_bearing_diff = 6  # ULTRA strict bearing difference for merging - maximum 6 degrees

    improved = True
    while improved:
        improved = False
        new_routes = []
        used_indices = set()

        for i, route1 in enumerate(routes):
            if i in used_indices:
                continue

            best_merge = None
            best_score = float('inf')

            for j, route2 in enumerate(routes):
                if j <= i or j in used_indices:
                    continue

                # Check if merge is possible
                total_users = len(route1['assigned_users']) + len(
                    route2['assigned_users'])
                max_capacity = max(route1['vehicle_type'],
                                   route2['vehicle_type'])

                if total_users > max_capacity:
                    continue

                # STRICT: Check bearing compatibility first (most important)
                bearing1 = calculate_strict_route_bearing(route1, office_lat, office_lon)
                bearing2 = calculate_strict_route_bearing(route2, office_lat, office_lon)
                bearing_diff = bearing_difference(bearing1, bearing2)

                if bearing_diff > max_bearing_diff:
                    continue

                # Check distance between route centers
                center1 = calculate_route_center(route1)
                center2 = calculate_route_center(route2)
                distance = haversine_distance(center1[0], center1[1],
                                              center2[0], center2[1])

                if distance > max_distance:
                    continue

                # STRICT: Validate that merged route would maintain directional integrity
                test_merged_users = route1['assigned_users'] + route2['assigned_users']
                if not validate_merged_route_direction(test_merged_users, office_lat, office_lon):
                    continue

                # ENHANCED: Comprehensive merge coherence check
                all_users_positions = [(u['lat'], u['lng']) for u in route1['assigned_users']] + [(u['lat'], u['lng']) for u in route2['assigned_users']]

                # Check route coherence score
                coherence = road_network.get_route_coherence_score(
                    (route1['latitude'], route1['longitude']),
                    all_users_positions,
                    (office_lat, office_lon)
                )

                # Calculate combined detour ratio
                combined_detour_ratio = _calculate_combined_detour_ratio(
                    (route1['latitude'], route1['longitude']), all_users_positions, (office_lat, office_lon)
                )

                # Calculate turning score for combined route
                combined_turning_score = _calculate_combined_turning_score(
                    (route1['latitude'], route1['longitude']), all_users_positions, (office_lat, office_lon)
                )

                # STRICT merge criteria - all must pass
                if coherence < 0.65:  # Increased from 0.55
                    continue
                if combined_detour_ratio > 1.3:  # Max 30% detour for merged route
                    continue
                if combined_turning_score > 45:  # Max 45 degrees average turning
                    continue

                # Calculate merge score (lower is better)
                utilization_improvement = (total_users / max_capacity) - max(
                    len(route1['assigned_users']) / route1['vehicle_type'],
                    len(route2['assigned_users']) / route2['vehicle_type'])

                # Bonus for high utilization, penalty for bearing difference
                score = distance + (bearing_diff *
                                    0.1) - (utilization_improvement * 50)

                if score < best_score and utilization_improvement > 0.1:  # Require meaningful improvement
                    best_score = score
                    best_merge = (j, route2, total_users / max_capacity)

            if best_merge is not None:
                j, route2, new_utilization = best_merge

                # Create merged route with better positioned driver
                center1 = calculate_route_center(route1)
                center2 = calculate_route_center(route2)
                combined_center = ((center1[0] + center2[0]) / 2,
                                   (center1[1] + center2[1]) / 2)

                dist1 = haversine_distance(route1['latitude'],
                                           route1['longitude'],
                                           combined_center[0],
                                           combined_center[1])
                dist2 = haversine_distance(route2['latitude'],
                                           route2['longitude'],
                                           combined_center[0],
                                           combined_center[1])

                better_route = route1 if dist1 <= dist2 else route2
                merged_route = better_route.copy()

                # Combine users and sort by office distance
                all_users = route1['assigned_users'] + route2['assigned_users']
                all_users.sort(key=lambda u: u.get('office_distance', 0),
                               reverse=True)

                merged_route['assigned_users'] = all_users
                merged_route['vehicle_type'] = max(route1['vehicle_type'],
                                                   route2['vehicle_type'])

                new_routes.append(merged_route)
                used_indices.add(i)
                used_indices.add(j)
                merges_completed += 1
                improved = True

                bearing1 = calculate_route_bearing(route1, office_lat,
                                                   office_lon)
                bearing2 = calculate_route_bearing(route2, office_lat,
                                                   office_lon)
                logger.info(
                    f"  -> Merged routes {route1['driver_id']} (bearing: {bearing1:.1f}°) + {route2['driver_id']} (bearing: {bearing2:.1f}°) = {new_utilization*100:.1f}% utilization"
                )
            else:
                new_routes.append(route1)
                used_indices.add(i)

        routes = new_routes

    logger.info(
        f"✅ COMPLETED STEP 5: BEARING-AWARE CAPACITY MERGING - {merges_completed} merges completed"
    )
    return routes


def calculate_route_bearing(route, office_lat, office_lon):
    """Calculate average bearing of a route"""
    if not route['assigned_users']:
        return calculate_bearing(office_lat, office_lon, route['latitude'],
                                 route['longitude'])

    bearings = []
    for user in route['assigned_users']:
        bearing = calculate_bearing(office_lat, office_lon, user['lat'],
                                    user['lng'])
        bearings.append(bearing)

    return np.mean(bearings)


def calculate_strict_route_bearing(route, office_lat, office_lon):
    """Calculate strict route bearing with validation"""
    if not route['assigned_users']:
        return calculate_bearing(office_lat, office_lon, route['latitude'], route['longitude'])

    # Use the bearing from office to the centroid of existing users
    if len(route['assigned_users']) == 1:
        user = route['assigned_users'][0]
        return calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])

    # Calculate centroid of existing users
    total_lat = sum(u['lat'] for u in route['assigned_users'])
    total_lng = sum(u['lng'] for u in route['assigned_users'])
    centroid_lat = total_lat / len(route['assigned_users'])
    centroid_lng = total_lng / len(route['assigned_users'])

    return calculate_bearing(office_lat, office_lon, centroid_lat, centroid_lng)


def validate_strict_directional_consistency(driver_pos, existing_users, candidate_pos, office_pos, route_bearing):
    """Validate that adding a candidate maintains strict directional consistency"""

    # Calculate candidate bearing from office
    candidate_bearing = calculate_bearing(office_pos[0], office_pos[1], candidate_pos[0], candidate_pos[1])

    # STRICT: Candidate must be in same general direction (max 15 degrees deviation)
    bearing_diff = bearing_difference(route_bearing, candidate_bearing)
    if bearing_diff > 15:
        return False

    # STRICT: Check that candidate doesn't create backtracking
    if existing_users:
        # Calculate distances to office for all existing users
        existing_distances = [haversine_distance(u[0], u[1], office_pos[0], office_pos[1]) for u in existing_users]
        candidate_distance = haversine_distance(candidate_pos[0], candidate_pos[1], office_pos[0], office_pos[1])

        min_existing_distance = min(existing_distances)
        max_existing_distance = max(existing_distances)

        # Candidate must fit within the distance range (no outliers)
        if candidate_distance < min_existing_distance - 1.0 or candidate_distance > max_existing_distance + 1.0:
            return False

    # STRICT: Progressive distance check - must be moving toward office
    driver_to_office = haversine_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
    candidate_to_office = haversine_distance(candidate_pos[0], candidate_pos[1], office_pos[0], office_pos[1])

    # Candidate should be closer to office than driver (allowing small tolerance)
    if candidate_to_office > driver_to_office + 0.3:  # 300m tolerance
        return False

    return True


def validate_route_directional_integrity(route, office_lat, office_lon):
    """Validate that entire route maintains directional integrity"""
    if len(route['assigned_users']) <= 1:
        return True

    # Calculate route bearing
    route_bearing = calculate_strict_route_bearing(route, office_lat, office_lon)

    # Check each user against route bearing
    max_allowed_deviation = 12  # Very strict - 12 degrees max

    for user in route['assigned_users']:
        user_bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])
        bearing_diff = bearing_difference(route_bearing, user_bearing)

        if bearing_diff > max_allowed_deviation:
            return False

    # Check distance progression (users should be roughly in order of distance from office)
    distances = [haversine_distance(u['lat'], u['lng'], office_lat, office_lon) for u in route['assigned_users']]

    # Allow some flexibility but no major reversals
    for i in range(len(distances) - 1):
        if distances[i] < distances[i + 1] - 2.0:  # 2km tolerance for distance progression
            return False

    return True


def calculate_route_center(route):
    """Calculate the center point of users in a route"""
    if not route['assigned_users']:
        return (route['latitude'], route['longitude'])

    lats = [u['lat'] for u in route['assigned_users']]
    lngs = [u['lng'] for u in route['assigned_users']]
    return (np.mean(lats), np.mean(lngs))


def validate_merged_route_direction(merged_users, office_lat, office_lon):
    """Validate that merged route maintains strict directional consistency"""
    if len(merged_users) <= 1:
        return True

    # Calculate bearings for all users
    bearings = []
    for user in merged_users:
        bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])
        bearings.append(bearing)

    # Check bearing spread - should be very tight
    min_bearing = min(bearings)
    max_bearing = max(bearings)

    # Handle wraparound case (crossing 0/360 degrees)
    bearing_spread = max_bearing - min_bearing
    if bearing_spread > 180:
        bearing_spread = 360 - bearing_spread

    # STRICT: Maximum 20 degree spread for merged route
    if bearing_spread > 20:
        return False

    # Check distance progression consistency
    distances = [haversine_distance(u['lat'], u['lng'], office_lat, office_lon) for u in merged_users]
    distances.sort()

    # Check for reasonable distance distribution (no huge gaps)
    for i in range(len(distances) - 1):
        gap = distances[i + 1] - distances[i]
        if gap > 5.0:  # 5km max gap between consecutive users
            return False

    return True


def _calculate_combined_detour_ratio(driver_pos, user_positions, office_pos):
    """Calculate detour ratio for a combined route"""
    if not user_positions:
        return 1.0

    # Calculate total route distance
    total_distance = 0.0
    current_pos = driver_pos

    for user_pos in user_positions:
        total_distance += haversine_distance(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
        current_pos = user_pos

    total_distance += haversine_distance(current_pos[0], current_pos[1], office_pos[0], office_pos[1])

    # Calculate direct distance
    direct_distance = haversine_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    return total_distance / direct_distance if direct_distance > 0 else 1.0


def _calculate_combined_turning_score(driver_pos, user_positions, office_pos):
    """Calculate average turning score for a combined route"""
    if len(user_positions) < 2:
        return 0.0

    route_points = [driver_pos] + user_positions + [office_pos]
    total_turning = 0.0
    turning_segments = 0

    for i in range(len(route_points) - 2):
        # Calculate bearings for consecutive segments
        bearing1 = calculate_bearing(route_points[i][0], route_points[i][1],
                                   route_points[i+1][0], route_points[i+1][1])
        bearing2 = calculate_bearing(route_points[i+1][0], route_points[i+1][1],
                                   route_points[i+2][0], route_points[i+2][1])

        # Calculate turning angle
        turning_angle = bearing_difference(bearing1, bearing2)
        total_turning += turning_angle
        turning_segments += 1

    return total_turning / turning_segments if turning_segments > 0 else 0.0


# STEP 6: LOCAL OPTIMIZATION
def step6_local_optimization(routes, office_lat, office_lon):
    """Step 6: Local optimization of routes"""
    logger = get_logger()
    logger.info(
        "================================================================================"
    )
    logger.info("🚀 STARTING STEP 6: LOCAL OPTIMIZATION")
    logger.info(
        "================================================================================"
    )

    improved_routes = 0

    for route in routes:
        # Optimize user sequence within route
        if len(route['assigned_users']) > 1:
            # Simple optimization: sort by distance from office
            route['assigned_users'].sort(
                key=lambda u: haversine_distance(u['lat'], u['lng'],
                                                 office_lat, office_lon),
                reverse=True  # Farthest first for pickup
            )
            improved_routes += 1

    logger.info(
        f"✅ COMPLETED STEP 6: LOCAL OPTIMIZATION - {improved_routes} routes improved"
    )
    return routes


# STEP 7: GLOBAL OPTIMIZATION
def step7_global_optimization(routes, user_df, assigned_user_ids, driver_df,
                              office_lat, office_lon):
    """Step 7: Global optimization with swapping and merging"""
    logger = get_logger()
    logger.info(
        "================================================================================"
    )
    logger.info("🚀 STARTING STEP 7: GLOBAL OPTIMIZATION")
    logger.info(
        "================================================================================"
    )

    # Phase 1: User swapping for better capacity utilization
    swaps_made = 0
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            route1, route2 = routes[i], routes[j]

            # Check if swapping users could improve utilization
            if try_capacity_swap(route1, route2):
                swaps_made += 1

    if swaps_made > 0:
        logger.info(
            f"  -> Completed capacity-based swapping: {swaps_made} swaps made")

    # Phase 2: Final merging attempt
    routes = step5_capacity_merging(routes, office_lat, office_lon)

    # Phase 3: Handle remaining unassigned users
    unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]
    unassigned_list = []

    if not unassigned_users.empty:
        unassigned_list = handle_remaining_unassigned_users(
            unassigned_users, driver_df, routes, office_lat, office_lon)

    # Phase 4: POST-ASSIGNMENT LOCAL OPTIMIZATION
    logger.info(f"  -> Starting post-assignment local optimization")
    routes = post_assignment_local_optimization(routes, office_lat, office_lon)

    logger.info(
        f"✅ COMPLETED STEP 7: GLOBAL OPTIMIZATION - Capacity optimization completed"
    )
    return routes, unassigned_list


def post_assignment_local_optimization(routes, office_lat, office_lon):
    """Post-assignment local optimization to fix greedy assignment issues"""
    logger = get_logger()
    logger.info("🔄 Starting post-assignment local optimization")

    improvements_made = 0
    max_iterations = 3

    for iteration in range(max_iterations):
        iteration_improvements = 0

        # Phase 1: Single user moves between nearby routes
        for i in range(len(routes)):
            for j in range(len(routes)):
                if i == j or len(routes[i]['assigned_users']) <= 1:
                    continue

                # Check if routes are nearby
                center_i = calculate_route_center(routes[i])
                center_j = calculate_route_center(routes[j])
                distance = haversine_distance(center_i[0], center_i[1], center_j[0], center_j[1])

                if distance > _config['MERGE_DISTANCE_KM']:
                    continue

                # Try moving users from route i to route j
                if try_single_user_move(routes[i], routes[j], office_lat, office_lon):
                    iteration_improvements += 1

        # Phase 2: User swaps between routes
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if not routes[i]['assigned_users'] or not routes[j]['assigned_users']:
                    continue

                # Check if routes are nearby
                center_i = calculate_route_center(routes[i])
                center_j = calculate_route_center(routes[j])
                distance = haversine_distance(center_i[0], center_i[1], center_j[0], center_j[1])

                if distance > _config['MERGE_DISTANCE_KM']:
                    continue

                # Try swapping users between routes
                if try_user_swap_between_routes(routes[i], routes[j], office_lat, office_lon):
                    iteration_improvements += 1

        improvements_made += iteration_improvements
        logger.info(f"  -> Iteration {iteration + 1}: {iteration_improvements} improvements")

        if iteration_improvements == 0:
            break

    logger.info(f"  -> Total improvements made: {improvements_made}")
    return routes


def try_single_user_move(source_route, target_route, office_lat, office_lon):
    """Try moving a single user from source to target route if it improves combined cost"""
    if len(target_route['assigned_users']) >= target_route['vehicle_type']:
        return False

    best_improvement = 0
    best_user_idx = None

    # Calculate current costs
    current_cost_source = calculate_route_total_cost(source_route, office_lat, office_lon)
    current_cost_target = calculate_route_total_cost(target_route, office_lat, office_lon)
    current_total_cost = current_cost_source + current_cost_target

    # Try moving each user
    for user_idx, user in enumerate(source_route['assigned_users']):
        # Check if user satisfies direction constraints for target route
        if not check_user_direction_compatibility(user, target_route, office_lat, office_lon):
            continue

        # Create temporary routes
        temp_source = source_route.copy()
        temp_target = target_route.copy()
        temp_source['assigned_users'] = [u for i, u in enumerate(source_route['assigned_users']) if i != user_idx]
        temp_target['assigned_users'] = target_route['assigned_users'] + [user]

        # Calculate new costs
        new_cost_source = calculate_route_total_cost(temp_source, office_lat, office_lon)
        new_cost_target = calculate_route_total_cost(temp_target, office_lat, office_lon)
        new_total_cost = new_cost_source + new_cost_target

        improvement = current_total_cost - new_total_cost
        if improvement > best_improvement and improvement > 0.5:  # Minimum improvement threshold
            best_improvement = improvement
            best_user_idx = user_idx

    # Apply best move if found
    if best_user_idx is not None:
        user_to_move = source_route['assigned_users'].pop(best_user_idx)
        target_route['assigned_users'].append(user_to_move)
        logger.info(f"    -> Moved user {user_to_move['user_id']} from route {source_route['driver_id']} to {target_route['driver_id']} (improvement: {best_improvement:.2f}km)")
        return True

    return False


def try_user_swap_between_routes(route1, route2, office_lat, office_lon):
    """Try swapping users between two routes if it improves combined cost"""
    if not route1['assigned_users'] or not route2['assigned_users']:
        return False

    best_improvement = 0
    best_swap = None

    # Calculate current costs
    current_cost1 = calculate_route_total_cost(route1, office_lat, office_lon)
    current_cost2 = calculate_route_total_cost(route2, office_lat, office_lon)
    current_total_cost = current_cost1 + current_cost2

    # Try all possible single-user swaps
    for i, user1 in enumerate(route1['assigned_users']):
        for j, user2 in enumerate(route2['assigned_users']):
            # Check direction compatibility
            if not (check_user_direction_compatibility(user1, route2, office_lat, office_lon) and
                    check_user_direction_compatibility(user2, route1, office_lat, office_lon)):
                continue

            # Create temporary routes with swapped users
            temp_route1 = route1.copy()
            temp_route2 = route2.copy()
            temp_route1['assigned_users'] = [user2 if idx == i else u for idx, u in enumerate(route1['assigned_users'])]
            temp_route2['assigned_users'] = [user1 if idx == j else u for idx, u in enumerate(route2['assigned_users'])]

            # Calculate new costs
            new_cost1 = calculate_route_total_cost(temp_route1, office_lat, office_lon)
            new_cost2 = calculate_route_total_cost(temp_route2, office_lat, office_lon)
            new_total_cost = new_cost1 + new_cost2

            improvement = current_total_cost - new_total_cost
            if improvement > best_improvement and improvement > 0.3:  # Minimum improvement threshold
                best_improvement = improvement
                best_swap = (i, j, user1, user2)

    # Apply best swap if found
    if best_swap is not None:
        i, j, user1, user2 = best_swap
        route1['assigned_users'][i] = user2
        route2['assigned_users'][j] = user1
        logger.info(f"    -> Swapped users {user1['user_id']} ↔ {user2['user_id']} between routes {route1['driver_id']} and {route2['driver_id']} (improvement: {best_improvement:.2f}km)")
        return True

    return False


def check_user_direction_compatibility(user, route, office_lat, office_lon):
    """Check if user is directionally compatible with route"""
    if not route['assigned_users']:
        return True

    # Calculate route bearing
    route_bearing = calculate_route_bearing(route, office_lat, office_lon)

    # Calculate user bearing from office
    user_bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])

    # Check bearing compatibility
    bearing_diff = bearing_difference(route_bearing, user_bearing)
    return bearing_diff <= _config['MAX_BEARING_DIFFERENCE']


def calculate_route_total_cost(route, office_lat, office_lon):
    """Calculate total cost (distance + turning penalty) for a route"""
    if not route['assigned_users']:
        return 0.0

    # Calculate total distance
    total_distance = 0.0
    current_pos = (route['latitude'], route['longitude'])

    for user in route['assigned_users']:
        user_pos = (user['lat'], user['lng'])
        total_distance += haversine_distance(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
        current_pos = user_pos

    total_distance += haversine_distance(current_pos[0], current_pos[1], office_lat, office_lon)

    # Calculate turning penalty
    turning_score = _calculate_combined_turning_score((route['latitude'], route['longitude']),
                                                     [(u['lat'], u['lng']) for u in route['assigned_users']],
                                                     (office_lat, office_lon))

    # Combine distance and turning penalty (1 degree ≈ 0.1km penalty)
    return total_distance + (turning_score * 0.1)


def try_capacity_swap(route1, route2):
    """Try swapping users between routes to improve capacity utilization"""
    if not route1['assigned_users'] or not route2['assigned_users']:
        return False

    util1 = len(route1['assigned_users']) / route1['vehicle_type']
    util2 = len(route2['assigned_users']) / route2['vehicle_type']

    # Only swap if one route is underutilized and the other has capacity
    if util1 >= 0.8 and util2 >= 0.8:
        return False

    # Try moving one user from the fuller route to the emptier route
    if util1 > util2 and len(
            route2['assigned_users']) < route2['vehicle_type']:
        # Move user from route1 to route2
        user_to_move = route1['assigned_users'].pop()
        route2['assigned_users'].append(user_to_move)
        return True
    elif util2 > util1 and len(
            route1['assigned_users']) < route1['vehicle_type']:
        # Move user from route2 to route1
        user_to_move = route2['assigned_users'].pop()
        route1['assigned_users'].append(user_to_move)
        return True

    return False


def handle_remaining_unassigned_users(unassigned_users, driver_df, routes,
                                      office_lat, office_lon):
    """Handle remaining unassigned users by creating new routes or filling existing ones"""
    logger = get_logger()
    unassigned_list = []
    used_driver_ids = {route['driver_id'] for route in routes}
    available_drivers = driver_df[~driver_df['driver_id'].isin(used_driver_ids
                                                               )]

    logger.info(f"    Processing {len(unassigned_users)} unassigned users")

    # Phase 1: Try to fill existing routes with available capacity
    for route in routes:
        if len(route['assigned_users']) >= route['vehicle_type']:
            continue

        route_center = calculate_route_center(route)
        available_seats = route['vehicle_type'] - len(route['assigned_users'])

        users_to_add = []
        for _, user in unassigned_users.iterrows():
            if len(users_to_add) >= available_seats:
                break

            distance = haversine_distance(route_center[0], route_center[1],
                                          user['latitude'], user['longitude'])

            # More lenient distance for capacity optimization
            if distance <= _config['MAX_FILL_DISTANCE_KM']:
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

        # Remove assigned users from unassigned list
        if users_to_add:
            assigned_ids = [u['user_id'] for u in users_to_add]
            unassigned_users = unassigned_users[~unassigned_users['user_id'].
                                                isin(assigned_ids)]

    # Phase 2: Create new routes for remaining users
    remaining_unassigned = unassigned_users.copy()

    for _, driver in available_drivers.iterrows():
        if remaining_unassigned.empty:
            break

        # Take up to driver capacity users
        users_for_route = remaining_unassigned.head(driver['capacity'])

        if len(users_for_route) > 0:
            # Create new route
            new_route = {
                'driver_id': str(driver['driver_id']),
                'vehicle_id': str(driver.get('vehicle_id', '')),
                'vehicle_type': int(driver['capacity']),
                'latitude': float(driver['latitude']),
                'longitude': float(driver['longitude']),
                'assigned_users': []
            }

            for _, user in users_for_route.iterrows():
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

                new_route['assigned_users'].append(user_data)

            routes.append(new_route)
            assigned_ids = [row[1]['user_id'] for row in users_for_route.iterrows()]
            remaining_unassigned = remaining_unassigned[
                ~remaining_unassigned['user_id'].isin(assigned_ids)]

            logger.info(
                f"    Created new route for driver {driver['driver_id']} with {len(users_for_route)} users"
            )

    # Add remaining users to unassigned list
    for _, user in remaining_unassigned.iterrows():
        unassigned_user = {
            'user_id': str(user['user_id']),
            'latitude': float(user['latitude']),
            'longitude': float(user['longitude']),
            'first_name': str(user.get('first_name', '')),
            'email': str(user.get('email', '')),
            'office_distance': float(user.get('office_distance', 0))
        }
        unassigned_list.append(unassigned_user)

    return unassigned_list


def update_route_metrics(route, office_lat, office_lon):
    """Update route metrics"""
    if route['assigned_users']:
        route['centroid'] = calculate_route_center(route)
        route['utilization'] = len(
            route['assigned_users']) / route['vehicle_type']
        route['turning_score'] = 0  # Simplified for capacity optimization
        route['tortuosity_ratio'] = 1.0
        route['direction_consistency'] = 1.0
    else:
        route['centroid'] = [route['latitude'], route['longitude']]
        route['utilization'] = 0
        route['turning_score'] = 0
        route['tortuosity_ratio'] = 1.0
        route['direction_consistency'] = 1.0


def _get_all_drivers_as_unassigned(data):
    """Convert all drivers to unassigned format"""
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
        driver_data = {
            'driver_id': str(driver.get('id', '')),
            'capacity': int(driver.get('capacity', 0)),
            'vehicle_id': str(driver.get('vehicle_id', '')),
            'latitude': float(driver.get('latitude', 0.0)),
            'longitude': float(driver.get('longitude', 0.0))
        }
        unassigned_drivers.append(driver_data)

    return unassigned_drivers


def _convert_users_to_unassigned_format(users):
    """Convert users to unassigned format"""
    unassigned_users = []
    for user in users:
        user_data = {
            'user_id': str(user.get('id', '')),
            'latitude': float(user.get('latitude', 0.0)),
            'longitude': float(user.get('longitude', 0.0)),
            'first_name': str(user.get('first_name', '')),
            'email': str(user.get('email', '')),
            'office_distance': float(user.get('office_distance', 0.0))
        }
        unassigned_users.append(user_data)

    return unassigned_users


# MAIN ASSIGNMENT FUNCTION
def run_assignment_capacity(source_id: str,
                            parameter: int = 1,
                            string_param: str = ""):
    """
    Main assignment function with 7-step capacity optimization approach
    """
    start_time = time.time()
    logger = get_logger()
    logger.info("=" * 80)
    logger.info(f"🚀 STARTING CAPACITY OPTIMIZATION")
    logger.info(
        f"📋 Source ID: {source_id} | Parameter: {parameter} | String: {string_param}"
    )
    logger.info("=" * 80)

    try:
        # Load and validate data
        data = load_env_and_fetch_data(source_id, parameter, string_param)

        # Handle edge cases
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

        # Prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)

        # Execute 7-step process
        # STEP 1: Clustering
        user_df = step1_clustering(user_df, office_lat, office_lon, _config)

        # STEP 2: Subclustering
        user_df = step2_subclustering(user_df, office_lat, office_lon, _config)

        # STEP 3: Assigning routes
        routes, assigned_user_ids, used_driver_ids = step3_assign_routes(
            user_df, driver_df, office_lat, office_lon)

        # STEP 4: Road network pickup
        routes, assigned_user_ids = step4_road_network_pickup(
            routes, user_df, assigned_user_ids, office_lat, office_lon)

        # STEP 5: Capacity-based merging
        routes = step5_capacity_merging(routes, office_lat, office_lon)

        # STEP 6: Local optimization
        routes = step6_local_optimization(routes, office_lat, office_lon)

        # STEP 7: Global optimization
        routes, unassigned_users = step7_global_optimization(
            routes, user_df, assigned_user_ids, driver_df, office_lat,
            office_lon)

        # Filter out empty routes
        filtered_routes = [
            route for route in routes if route['assigned_users']
        ]

        # Build unassigned drivers list
        assigned_driver_ids = {route['driver_id'] for route in filtered_routes}
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

        # Final metrics update
        for route in filtered_routes:
            update_route_metrics(route, office_lat, office_lon)

        execution_time = time.time() - start_time

        # Final statistics
        total_users_assigned = sum(
            len(r['assigned_users']) for r in filtered_routes)
        total_capacity = sum(r['vehicle_type'] for r in filtered_routes)
        overall_utilization = (total_users_assigned / total_capacity *
                               100) if total_capacity > 0 else 0

        logger.info(
            f"✅ Capacity optimization completed in {execution_time:.2f}s")
        logger.info(f"📊 Final routes: {len(filtered_routes)}")
        logger.info(f"🎯 Users assigned: {total_users_assigned}")
        logger.info(f"👥 Users unassigned: {len(unassigned_users)}")
        logger.info(
            f"📈 Overall capacity utilization: {overall_utilization:.1f}%")

        clustering_results = {
            "method": "capacity_optimization",
            "clusters": user_df['geo_cluster'].nunique()
        }

        return {
            "status": "true",
            "execution_time": execution_time,
            "data": filtered_routes,
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