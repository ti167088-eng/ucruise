import math
import numpy as np
from typing import Tuple, List
from logger_config import get_logger

logger = get_logger()

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth in kilometers."""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371
    return c * r

def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing from point A to B in degrees."""
    lat1, lat2 = map(math.radians, [lat1, lat2])
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def calculate_bearing_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized calculation of bearing from point A to B in degrees."""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def bearing_difference(b1: float, b2: float) -> float:
    """Compute minimum difference between two bearings."""
    diff = abs(b1 - b2) % 360
    return min(diff, 360 - diff)

def coords_to_km(lat: float, lon: float, office_lat: float, office_lon: float) -> Tuple[float, float]:
    """Convert lat/lon coordinates to km from office using local approximation."""
    LAT_TO_KM = 111.0
    LON_TO_KM = 111.0 * math.cos(math.radians(office_lat))

    lat_km = (lat - office_lat) * LAT_TO_KM
    lon_km = (lon - office_lon) * LON_TO_KM
    return lat_km, lon_km

def normalize_bearing_difference(diff: float) -> float:
    """Normalize bearing difference to [-180, 180] range."""
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff

def calculate_centroid(coordinates: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculate the centroid of a list of coordinates."""
    if not coordinates:
        return (0, 0)

    avg_lat = sum(coord[0] for coord in coordinates) / len(coordinates)
    avg_lon = sum(coord[1] for coord in coordinates) / len(coordinates)
    return (avg_lat, avg_lon)