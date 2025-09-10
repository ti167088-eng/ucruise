import networkx as nx
import numpy as np
from functools import lru_cache
from typing import List, Tuple, Optional, Dict, Any
import math
import logging
import json
from collections import defaultdict, OrderedDict
import weakref
import threading
from datetime import datetime, timedelta
from logger_config import get_logger

logger = get_logger()

class RoadNetworkConfig:
    """Centralized configuration management for road network parameters"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file with defaults"""
        defaults = {
            "road_network": {
                "max_search_radius_km": 5.0,
                "fallback_search_radius_km": 10.0,
                "max_detour_ratio": 3.0,
                "cache_size_limit": 50000,
                "cache_cleanup_threshold": 40000,
                "coherence_weights": {
                    "directional_penalty": 0.3,
                    "backtrack_penalty": 0.4,
                    "corridor_bonus": 0.2,
                    "sequence_penalty": 0.1
                },
                "distance_validation": {
                    "max_reasonable_ratio": 5.0,
                    "min_road_efficiency": 0.1
                },
                "projection": {
                    "tile_size_degrees": 1.0,
                    "use_adaptive_projection": True
                }
            }
        }
        
        try:
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
                # Merge with defaults
                config = defaults.copy()
                if "road_network" in file_config:
                    config["road_network"].update(file_config["road_network"])
                return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load config file {self.config_file}: {e}. Using defaults.")
            return defaults
    
    def _validate_config(self):
        """Validate configuration values"""
        rn_config = self.config["road_network"]
        
        # Validate numeric ranges
        if rn_config["max_search_radius_km"] <= 0:
            rn_config["max_search_radius_km"] = 5.0
        if rn_config["max_detour_ratio"] < 1.0:
            rn_config["max_detour_ratio"] = 3.0
        if rn_config["cache_size_limit"] < 1000:
            rn_config["cache_size_limit"] = 50000
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


class AdaptiveProjection:
    """Adaptive coordinate projection for different geographic regions"""
    
    def __init__(self, center_lat: float, center_lon: float):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.lat_to_m = 111320.0  # meters per degree latitude
        self.lon_to_m = 111320.0 * math.cos(math.radians(center_lat))
    
    def to_meters(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert lat/lon to local metric coordinates"""
        x = (lon - self.center_lon) * self.lon_to_m
        y = (lat - self.center_lat) * self.lat_to_m
        return (x, y)
    
    def from_meters(self, x: float, y: float) -> Tuple[float, float]:
        """Convert local metric coordinates back to lat/lon"""
        lon = x / self.lon_to_m + self.center_lon
        lat = y / self.lat_to_m + self.center_lat
        return (lat, lon)


class EnhancedCache:
    """Enhanced caching system with size limits, directional awareness, and invalidation"""
    
    def __init__(self, max_size: int = 50000, cleanup_threshold: int = 40000):
        self.max_size = max_size
        self.cleanup_threshold = cleanup_threshold
        self._cache = OrderedDict()
        self._access_times = {}
        self._lock = threading.RLock()
        self._creation_time = datetime.now()
    
    def get(self, key: Tuple, default=None):
        """Get value with LRU access tracking"""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                self._access_times[key] = datetime.now()
                return value
            return default
    
    def set(self, key: Tuple, value: Any):
        """Set value with automatic cleanup"""
        with self._lock:
            self._cache[key] = value
            self._access_times[key] = datetime.now()
            
            if len(self._cache) > self.max_size:
                self._cleanup()
    
    def _cleanup(self):
        """Intelligent cache cleanup based on access patterns"""
        target_size = self.cleanup_threshold
        current_time = datetime.now()
        
        # Calculate scores for each entry (lower score = more likely to be removed)
        scores = []
        for key in self._cache:
            access_time = self._access_times.get(key, self._creation_time)
            age_hours = (current_time - access_time).total_seconds() / 3600
            # Score based on recency (recent items have higher scores)
            score = 1.0 / (1.0 + age_hours)
            scores.append((score, key))
        
        # Sort by score and remove lowest scoring items
        scores.sort()
        items_to_remove = len(scores) - target_size
        
        for i in range(items_to_remove):
            key = scores[i][1]
            del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
    
    def invalidate_pattern(self, pattern_func):
        """Invalidate cache entries matching a pattern"""
        with self._lock:
            keys_to_remove = []
            for key in self._cache:
                if pattern_func(key):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()


class RoadNetwork:
    """Enhanced road network with improved graph processing, caching, and error handling"""

    def __init__(self, graphml_path: str, config_file: str = 'config.json'):
        """Initialize road network with comprehensive error handling and configuration"""
        self.config = RoadNetworkConfig(config_file)
        self.graphml_path = graphml_path
        self.graph = None
        self.node_positions = {}
        self.projection = None
        self._kdtree = None
        self._node_list = []
        self._node_coords_m = []
        
        # Enhanced caching system
        cache_size = self.config.get('road_network.cache_size_limit', 50000)
        cleanup_threshold = self.config.get('road_network.cache_cleanup_threshold', 40000)
        self._path_cache = EnhancedCache(cache_size, cleanup_threshold)
        self._distance_cache = EnhancedCache(cache_size, cleanup_threshold)
        self._coherence_cache = EnhancedCache(cache_size // 4, cleanup_threshold // 4)
        
        # Road type weights for realistic routing
        self.road_type_weights = {
            'motorway': 0.8, 'trunk': 0.9, 'primary': 1.0, 'secondary': 1.1,
            'tertiary': 1.2, 'residential': 1.3, 'service': 1.5, 'track': 2.0
        }
        
        # Graph quality metrics
        self.graph_stats = {}
        
        try:
            self._load_and_process_graph()
        except Exception as e:
            logger.error(f"Failed to initialize road network: {e}")
            raise

    def _load_and_process_graph(self):
        """Load and process graph with comprehensive error handling"""
        try:
            self.graph = nx.read_graphml(self.graphml_path)
            logger.info(f"Loaded graph with {len(self.graph.nodes)} nodes, {self.graph.number_of_edges()} edges")
        except Exception as e:
            raise Exception(f"Failed to load GraphML file {self.graphml_path}: {e}")
        
        # Convert to undirected if needed
        if self.graph.is_directed():
            self.graph = self.graph.to_undirected()
            logger.info("Converted directed graph to undirected")
        
        # Handle MultiGraph conversion more intelligently
        if self.graph.is_multigraph():
            self.graph = self._convert_multigraph_intelligently()
        
        # Extract and validate node positions
        self._extract_node_positions()
        
        # Set up adaptive projection
        if self.node_positions:
            center_lat = np.mean([pos[0] for pos in self.node_positions.values()])
            center_lon = np.mean([pos[1] for pos in self.node_positions.values()])
            self.projection = AdaptiveProjection(center_lat, center_lon)
        
        # Process edge weights with road type awareness
        self._process_edge_weights()
        
        # Build spatial index
        self._build_spatial_index()
        
        # Calculate graph statistics
        self._calculate_graph_stats()

    def _convert_multigraph_intelligently(self):
        """Convert MultiGraph to simple Graph while preserving important connections"""
        logger.info("Converting MultiGraph to simple Graph with intelligent edge selection...")
        
        simple_graph = nx.Graph()
        simple_graph.add_nodes_from(self.graph.nodes(data=True))
        
        edges_processed = 0
        edges_merged = 0
        
        # Group edges by node pairs
        edge_groups = defaultdict(list)
        for u, v, key, data in self.graph.edges(data=True, keys=True):
            edge_groups[(u, v)].append(data)
        
        for (u, v), edge_list in edge_groups.items():
            if len(edge_list) == 1:
                simple_graph.add_edge(u, v, **edge_list[0])
            else:
                # Merge multiple edges intelligently
                merged_data = self._merge_edge_data(edge_list)
                simple_graph.add_edge(u, v, **merged_data)
                edges_merged += len(edge_list) - 1
            
            edges_processed += len(edge_list)
        
        logger.info(f"Processed {edges_processed} edges, merged {edges_merged} duplicate connections")
        return simple_graph

    def _merge_edge_data(self, edge_list: List[Dict]) -> Dict:
        """Intelligently merge data from multiple edges between same nodes"""
        merged = edge_list[0].copy()
        
        # For length/weight, use minimum (shortest path)
        if len(edge_list) > 1:
            lengths = [e.get('length', float('inf')) for e in edge_list]
            weights = [e.get('weight', float('inf')) for e in edge_list]
            
            if any(l != float('inf') for l in lengths):
                merged['length'] = min(l for l in lengths if l != float('inf'))
            if any(w != float('inf') for w in weights):
                merged['weight'] = min(w for w in weights if w != float('inf'))
            
            # For road type, prefer higher importance roads
            road_types = [e.get('highway', 'unknown') for e in edge_list]
            merged['highway'] = self._select_best_road_type(road_types)
        
        return merged

    def _select_best_road_type(self, road_types: List[str]) -> str:
        """Select the most important road type from a list"""
        importance_order = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service', 'track']
        
        for road_type in importance_order:
            if road_type in road_types:
                return road_type
        
        return road_types[0] if road_types else 'unknown'

    def _extract_node_positions(self):
        """Extract node positions with validation"""
        self.node_positions = {}
        invalid_nodes = 0
        
        for node, data in self.graph.nodes(data=True):
            try:
                # Try different coordinate attribute names
                lat = data.get('lat') or data.get('y') or data.get('d4')
                lon = data.get('lon') or data.get('x') or data.get('d5')

                if lat is not None and lon is not None:
                    lat, lon = float(lat), float(lon)
                    
                    # Validate coordinate bounds
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        self.node_positions[node] = (lat, lon)
                    else:
                        invalid_nodes += 1
                else:
                    invalid_nodes += 1
            except (ValueError, TypeError):
                invalid_nodes += 1
        
        if invalid_nodes > 0:
            logger.warning(f"Found {invalid_nodes} nodes with invalid coordinates")
        
        logger.info(f"Extracted positions for {len(self.node_positions)} nodes")

    def _process_edge_weights(self):
        """Process edge weights with road type and traffic condition awareness"""
        edges_updated = 0
        
        for u, v, data in self.graph.edges(data=True):
            if u in self.node_positions and v in self.node_positions:
                pos1 = self.node_positions[u]
                pos2 = self.node_positions[v]
                
                # Calculate base distance
                base_distance = self._calculate_distance(pos1[0], pos1[1], pos2[0], pos2[1])
                
                # Apply road type factor
                road_type = data.get('highway', 'residential')
                type_factor = self.road_type_weights.get(road_type, 1.0)
                
                # Apply speed limit factor if available
                speed_factor = self._calculate_speed_factor(data)
                
                # Calculate final weight
                final_weight = base_distance * type_factor * speed_factor
                data['weight'] = final_weight
                data['base_distance'] = base_distance
                
                edges_updated += 1
        
        logger.info(f"Updated weights for {edges_updated} edges with road type awareness")

    def _calculate_speed_factor(self, edge_data: Dict) -> float:
        """Calculate speed factor based on road attributes"""
        # Default speed factor
        speed_factor = 1.0
        
        # Consider max speed if available
        max_speed = edge_data.get('maxspeed')
        if max_speed:
            try:
                speed_val = float(max_speed.replace('mph', '').replace('kmh', '').strip())
                # Normalize to a factor (higher speed = less travel time factor)
                if 'mph' in str(max_speed):
                    speed_val *= 1.609  # Convert to km/h
                
                # Speed factor: faster roads have lower factors (less time cost)
                if speed_val > 0:
                    speed_factor = max(0.5, min(2.0, 50.0 / speed_val))
            except (ValueError, AttributeError):
                pass
        
        return speed_factor

    def _build_spatial_index(self):
        """Build enhanced spatial index with adaptive projection"""
        if not self.node_positions or not self.projection:
            logger.warning("Cannot build spatial index: missing node positions or projection")
            return
        
        self._node_list = []
        self._node_coords_m = []
        
        for node_id, (lat, lon) in self.node_positions.items():
            self._node_list.append(node_id)
            x, y = self.projection.to_meters(lat, lon)
            self._node_coords_m.append((x, y))
        
        try:
            from scipy.spatial import cKDTree
            self._kdtree = cKDTree(self._node_coords_m)
            logger.info(f"Built spatial index with {len(self._node_coords_m)} nodes")
        except ImportError:
            logger.warning("scipy not available, spatial queries will be slower")
            self._kdtree = None

    def _calculate_graph_stats(self):
        """Calculate comprehensive graph statistics"""
        self.graph_stats = {
            'num_nodes': len(self.graph.nodes),
            'num_edges': self.graph.number_of_edges(),
            'num_positioned_nodes': len(self.node_positions),
            'connectivity': nx.is_connected(self.graph),
            'num_components': nx.number_connected_components(self.graph)
        }
        
        if self.node_positions:
            lats = [pos[0] for pos in self.node_positions.values()]
            lons = [pos[1] for pos in self.node_positions.values()]
            self.graph_stats.update({
                'lat_bounds': (min(lats), max(lats)),
                'lon_bounds': (min(lons), max(lons)),
                'coverage_area_km2': self._estimate_coverage_area(lats, lons)
            })

    def _estimate_coverage_area(self, lats: List[float], lons: List[float]) -> float:
        """Estimate coverage area in square kilometers"""
        if len(lats) < 3:
            return 0.0
        
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        
        # Rough approximation
        center_lat = (max(lats) + min(lats)) / 2
        lat_km = lat_range * 111.0
        lon_km = lon_range * 111.0 * math.cos(math.radians(center_lat))
        
        return lat_km * lon_km

    def find_nearest_road_node(self, lat: float, lon: float, max_search_km: Optional[float] = None) -> Tuple[Optional[str], float]:
        """Enhanced nearest node search with fallback strategies"""
        if max_search_km is None:
            max_search_km = self.config.get('road_network.max_search_radius_km', 5.0)
        
        # Primary search using spatial index
        if self._kdtree and self.projection:
            node_id, distance_km = self._spatial_search(lat, lon, max_search_km)
            if node_id is not None:
                return node_id, distance_km
        
        # Fallback search with expanded radius
        fallback_radius = self.config.get('road_network.fallback_search_radius_km', 10.0)
        if max_search_km < fallback_radius:
            logger.info(f"Primary search failed, trying fallback search with {fallback_radius}km radius")
            return self._brute_force_search(lat, lon, fallback_radius)
        
        return None, float('inf')

    def _spatial_search(self, lat: float, lon: float, max_search_km: float) -> Tuple[Optional[str], float]:
        """Spatial index-based search"""
        try:
            x, y = self.projection.to_meters(lat, lon)
            max_distance_m = max_search_km * 1000
            
            distance_m, idx = self._kdtree.query((x, y))
            
            if distance_m <= max_distance_m:
                node_id = self._node_list[idx]
                distance_km = distance_m / 1000.0
                return node_id, distance_km
        except Exception as e:
            logger.warning(f"Spatial search failed: {e}")
        
        return None, float('inf')

    def _brute_force_search(self, lat: float, lon: float, max_search_km: float) -> Tuple[Optional[str], float]:
        """Brute force search as fallback"""
        best_node = None
        best_distance = float('inf')
        
        for node_id, (node_lat, node_lon) in self.node_positions.items():
            distance = self._calculate_distance(lat, lon, node_lat, node_lon)
            if distance < best_distance and distance <= max_search_km:
                best_distance = distance
                best_node = node_id
        
        return best_node, best_distance

    def get_road_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Enhanced road distance calculation with validation and caching"""
        # Create cache key
        cache_key = (round(lat1, 6), round(lon1, 6), round(lat2, 6), round(lon2, 6))
        
        # Check cache first
        cached_result = self._distance_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Calculate distance
        result = self._calculate_road_distance_internal(lat1, lon1, lat2, lon2)
        
        # Validate result
        validated_result = self._validate_distance_result(lat1, lon1, lat2, lon2, result)
        
        # Cache result
        self._distance_cache.set(cache_key, validated_result)
        
        return validated_result

    def _calculate_road_distance_internal(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Internal road distance calculation"""
        try:
            # Find nearest nodes with enhanced search
            node1, dist1 = self.find_nearest_road_node(lat1, lon1)
            node2, dist2 = self.find_nearest_road_node(lat2, lon2)
            
            # Check if nodes are valid and reasonably close
            max_node_distance = self.config.get('road_network.max_search_radius_km', 5.0)
            
            if not node1 or not node2:
                return self._calculate_distance(lat1, lon1, lat2, lon2)
            
            if dist1 > max_node_distance or dist2 > max_node_distance:
                # If nodes are very far, use hybrid approach
                return self._hybrid_distance_calculation(lat1, lon1, lat2, lon2, node1, node2, dist1, dist2)
            
            if node1 == node2:
                return self._calculate_distance(lat1, lon1, lat2, lat2)
            
            # Calculate path distance
            path_distance = self._get_shortest_path_distance(node1, node2)
            if path_distance == float('inf'):
                return self._calculate_distance(lat1, lon1, lat2, lon2)
            
            # Add access distances
            access_distance1 = self._calculate_distance(lat1, lon1, *self.node_positions[node1])
            access_distance2 = self._calculate_distance(lat2, lon2, *self.node_positions[node2])
            
            return path_distance + access_distance1 + access_distance2
            
        except Exception as e:
            logger.warning(f"Road distance calculation failed: {e}")
            return self._calculate_distance(lat1, lon1, lat2, lon2)

    def _hybrid_distance_calculation(self, lat1: float, lon1: float, lat2: float, lon2: float, 
                                   node1: str, node2: str, dist1: float, dist2: float) -> float:
        """Hybrid calculation when nodes are far from query points"""
        straight_distance = self._calculate_distance(lat1, lon1, lat2, lon2)
        
        try:
            path_distance = self._get_shortest_path_distance(node1, node2)
            if path_distance != float('inf'):
                # Weight the road distance by how close the nodes are to the query points
                node_weight = 1.0 / (1.0 + dist1 + dist2)
                return straight_distance * (1 - node_weight) + (path_distance + dist1 + dist2) * node_weight
        except Exception:
            pass
        
        return straight_distance

    def _get_shortest_path_distance(self, node1: str, node2: str) -> float:
        """Get shortest path distance with caching"""
        # Create directional cache key
        cache_key = (min(node1, node2), max(node1, node2))
        
        cached_result = self._path_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            distance = nx.shortest_path_length(self.graph, node1, node2, weight='weight')
            self._path_cache.set(cache_key, distance)
            return distance
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self._path_cache.set(cache_key, float('inf'))
            return float('inf')

    def _validate_distance_result(self, lat1: float, lon1: float, lat2: float, lon2: float, road_distance: float) -> float:
        """Validate distance calculation results"""
        straight_distance = self._calculate_distance(lat1, lon1, lat2, lon2)
        
        if straight_distance == 0:
            return 0
        
        ratio = road_distance / straight_distance
        max_ratio = self.config.get('road_network.distance_validation.max_reasonable_ratio', 3.0)  # Reduced from 5.0
        min_efficiency = self.config.get('road_network.distance_validation.min_road_efficiency', 0.2)  # Increased from 0.1

        # Check if ratio is reasonable
        if ratio > max_ratio:
            logger.warning(f"Unreasonable road distance ratio: {ratio:.2f}, using conservative estimate")
            # More conservative fallback - use 1.5x straight distance instead of uncapped ratio
            return straight_distance * 1.5

        # Check minimum efficiency
        if ratio < min_efficiency:
            logger.warning(f"Road distance too efficient: {ratio:.2f}, using straight distance")
            return straight_distance

        return road_distance

    def get_route_coherence_score(self, driver_pos: Tuple[float, float], 
                                user_positions: List[Tuple[float, float]], 
                                office_pos: Tuple[float, float]) -> float:
        """Enhanced coherence scoring with proper factor weighting"""
        if not user_positions:
            return 1.0
        
        # Create cache key
        positions_key = tuple([driver_pos] + user_positions + [office_pos])
        cache_key = hash(positions_key)
        
        cached_result = self._coherence_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = self._calculate_coherence_internal(driver_pos, user_positions, office_pos)
        self._coherence_cache.set(cache_key, result)
        
        return result

    def _calculate_coherence_internal(self, driver_pos: Tuple[float, float], 
                                    user_positions: List[Tuple[float, float]], 
                                    office_pos: Tuple[float, float]) -> float:
        """Internal coherence calculation with balanced scoring"""
        route_points = [driver_pos] + user_positions + [office_pos]
        
        # Calculate base efficiency
        total_road_distance = 0.0
        total_straight_distance = 0.0
        
        for i in range(len(route_points) - 1):
            lat1, lon1 = route_points[i]
            lat2, lon2 = route_points[i + 1]
            
            road_dist = self.get_road_distance(lat1, lon1, lat2, lon2)
            straight_dist = self._calculate_distance(lat1, lon1, lat2, lon2)
            
            total_road_distance += road_dist
            total_straight_distance += straight_dist
        
        if total_straight_distance == 0:
            return 1.0
        
        base_efficiency = total_straight_distance / total_road_distance
        
        # Get weights from configuration
        weights = self.config.get('road_network.coherence_weights', {})
        directional_weight = weights.get('directional_penalty', 0.3)
        backtrack_weight = weights.get('backtrack_penalty', 0.4)
        corridor_weight = weights.get('corridor_bonus', 0.2)
        sequence_weight = weights.get('sequence_penalty', 0.1)
        
        # Calculate component scores
        directional_penalty = self._calculate_directional_penalty(route_points) * directional_weight
        backtrack_penalty = self._calculate_backtrack_penalty(route_points, office_pos) * backtrack_weight
        corridor_bonus = self._calculate_corridor_bonus(route_points, office_pos) * corridor_weight
        sequence_penalty = self._calculate_sequence_penalty(user_positions, office_pos) * sequence_weight
        
        # Combine scores
        final_score = base_efficiency - directional_penalty - backtrack_penalty + corridor_bonus - sequence_penalty
        
        return max(0.0, min(1.0, final_score))

    def _calculate_corridor_bonus(self, route_points: List[Tuple[float, float]], office_pos: Tuple[float, float]) -> float:
        """Calculate bonus for corridor-like routes"""
        if len(route_points) < 3:
            return 0.0
        
        # Check if route generally heads toward office
        driver_pos = route_points[0]
        main_bearing = self._calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
        
        consistent_segments = 0
        total_segments = 0
        
        for i in range(len(route_points) - 1):
            segment_bearing = self._calculate_bearing(route_points[i][0], route_points[i][1], 
                                                    route_points[i+1][0], route_points[i+1][1])
            bearing_diff = abs(self._normalize_bearing_difference(segment_bearing - main_bearing))
            
            if bearing_diff <= 45:  # Within 45 degrees of main direction
                consistent_segments += 1
            total_segments += 1
        
        if total_segments == 0:
            return 0.0
        
        consistency_ratio = consistent_segments / total_segments
        return consistency_ratio * 0.3  # Maximum 0.3 bonus

    def _calculate_sequence_penalty(self, user_positions: List[Tuple[float, float]], office_pos: Tuple[float, float]) -> float:
        """Calculate penalty for non-sequential routing"""
        if len(user_positions) <= 1:
            return 0.0
        
        office_distances = [self._calculate_distance(pos[0], pos[1], office_pos[0], office_pos[1]) 
                           for pos in user_positions]
        
        violations = 0
        for i in range(len(office_distances) - 1):
            if office_distances[i] < office_distances[i + 1] - 1.0:  # 1km tolerance
                violations += 1
        
        if len(office_distances) <= 1:
            return 0.0
        
        violation_ratio = violations / (len(office_distances) - 1)
        return violation_ratio * 0.2  # Maximum 0.2 penalty

    def is_user_on_route_path(self, driver_pos: Tuple[float, float], 
                            existing_users: List[Tuple[float, float]], 
                            candidate_pos: Tuple[float, float], 
                            office_pos: Tuple[float, float], 
                            max_detour_ratio: float = 1.25, 
                            route_type: str = "balanced") -> bool:
        """Enhanced path checking with multiple validation strategies"""
        try:
            # Primary check: detour ratio
            if self._check_detour_ratio(driver_pos, candidate_pos, office_pos, max_detour_ratio):
                return True
            
            # Secondary check: proximity to optimal path
            if self._check_path_proximity(driver_pos, candidate_pos, office_pos):
                return True
            
            # Tertiary check: route coherence impact
            if existing_users and self._check_coherence_impact(driver_pos, existing_users, candidate_pos, office_pos):
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Path checking failed: {e}")
            # Strict fallback - if road checking fails, reject the assignment
            return False

    def _check_detour_ratio(self, driver_pos: Tuple[float, float], candidate_pos: Tuple[float, float], 
                           office_pos: Tuple[float, float], max_detour_ratio: float) -> bool:
        """Check if detour ratio is acceptable"""
        driver_to_candidate = self.get_road_distance(driver_pos[0], driver_pos[1], candidate_pos[0], candidate_pos[1])
        candidate_to_office = self.get_road_distance(candidate_pos[0], candidate_pos[1], office_pos[0], office_pos[1])
        driver_to_office = self.get_road_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
        
        if driver_to_office <= 0:
            return False
        
        total_distance = driver_to_candidate + candidate_to_office
        detour_ratio = total_distance / driver_to_office
        
        return detour_ratio <= max_detour_ratio

    def _check_path_proximity(self, driver_pos: Tuple[float, float], candidate_pos: Tuple[float, float], 
                            office_pos: Tuple[float, float]) -> bool:
        """Check if candidate is close to optimal path"""
        try:
            # Find path between driver and office
            driver_node, _ = self.find_nearest_road_node(driver_pos[0], driver_pos[1])
            office_node, _ = self.find_nearest_road_node(office_pos[0], office_pos[1])
            candidate_node, candidate_dist = self.find_nearest_road_node(candidate_pos[0], candidate_pos[1])
            
            if not all([driver_node, office_node, candidate_node]) or candidate_dist > 1.0:  # Reduced from 2.0km
                return False

            try:
                path = nx.shortest_path(self.graph, driver_node, office_node, weight='weight')
                return candidate_node in path or self._is_near_path(candidate_node, path)
            except nx.NetworkXNoPath:
                return False
                
        except Exception:
            return False

    def _is_near_path(self, candidate_node: str, path: List[str]) -> bool:
        """Check if candidate node is near the path"""
        if candidate_node not in self.node_positions:
            return False
        
        candidate_pos = self.node_positions[candidate_node]
        
        for path_node in path:
            if path_node in self.node_positions:
                path_pos = self.node_positions[path_node]
                distance = self._calculate_distance(candidate_pos[0], candidate_pos[1], path_pos[0], path_pos[1])
                if distance <= 0.5:  # Reduced from 1km to 500m
                    return True
        
        return False

    def _check_coherence_impact(self, driver_pos: Tuple[float, float], existing_users: List[Tuple[float, float]], 
                              candidate_pos: Tuple[float, float], office_pos: Tuple[float, float]) -> bool:
        """Check if adding candidate improves or maintains route coherence"""
        current_coherence = self.get_route_coherence_score(driver_pos, existing_users, office_pos)
        new_coherence = self.get_route_coherence_score(driver_pos, existing_users + [candidate_pos], office_pos)
        
        # Allow if coherence doesn't decrease significantly
        return new_coherence >= current_coherence - 0.1

    def get_optimal_pickup_sequence(self, driver_pos: Tuple[float, float], 
                                  user_positions: List[Tuple[float, float]], 
                                  office_pos: Tuple[float, float]) -> List[int]:
        """Enhanced pickup sequence optimization"""
        if len(user_positions) <= 1:
            return list(range(len(user_positions)))
        
        # Use road-aware nearest neighbor with 2-opt improvement
        sequence = self._nearest_neighbor_road_aware(driver_pos, user_positions, office_pos)
        sequence = self._apply_2opt_improvement(sequence, user_positions, driver_pos, office_pos)
        
        return sequence

    def _nearest_neighbor_road_aware(self, driver_pos: Tuple[float, float], 
                                   user_positions: List[Tuple[float, float]], 
                                   office_pos: Tuple[float, float]) -> List[int]:
        """Road-aware nearest neighbor construction"""
        remaining_users = list(range(len(user_positions)))
        sequence = []
        current_pos = driver_pos
        
        while remaining_users:
            best_user_idx = None
            best_score = float('inf')
            
            for user_idx in remaining_users:
                user_pos = user_positions[user_idx]
                
                # Road distance to user
                distance_score = self.get_road_distance(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
                
                # Progress toward office
                current_to_office = self.get_road_distance(current_pos[0], current_pos[1], office_pos[0], office_pos[1])
                user_to_office = self.get_road_distance(user_pos[0], user_pos[1], office_pos[0], office_pos[1])
                progress_score = max(0, current_to_office - user_to_office) * -2.0
                
                # Directional consistency
                main_bearing = self._calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
                user_bearing = self._calculate_bearing(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
                bearing_diff = abs(self._normalize_bearing_difference(user_bearing - main_bearing))
                direction_penalty = (bearing_diff / 180.0) * 1.5
                
                total_score = distance_score + progress_score + direction_penalty
                
                if total_score < best_score:
                    best_score = total_score
                    best_user_idx = user_idx
            
            if best_user_idx is not None:
                sequence.append(best_user_idx)
                remaining_users.remove(best_user_idx)
                current_pos = user_positions[best_user_idx]
            else:
                break
        
        return sequence

    def _apply_2opt_improvement(self, sequence: List[int], user_positions: List[Tuple[float, float]], 
                              driver_pos: Tuple[float, float], office_pos: Tuple[float, float], 
                              max_iterations: int = 10) -> List[int]:
        """Apply 2-opt improvement with road distances"""
        if len(sequence) <= 2:
            return sequence
        
        current_sequence = sequence.copy()
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            current_distance = self._calculate_sequence_total_distance(current_sequence, user_positions, driver_pos, office_pos)
            
            for i in range(len(current_sequence) - 1):
                for j in range(i + 1, len(current_sequence)):
                    # Create new sequence by reversing segment
                    new_sequence = current_sequence[:i+1] + current_sequence[i+1:j+1][::-1] + current_sequence[j+1:]
                    new_distance = self._calculate_sequence_total_distance(new_sequence, user_positions, driver_pos, office_pos)
                    
                    if new_distance < current_distance:
                        current_sequence = new_sequence
                        current_distance = new_distance
                        improved = True
                        break
                
                if improved:
                    break
        
        return current_sequence

    def _calculate_sequence_total_distance(self, sequence: List[int], user_positions: List[Tuple[float, float]], 
                                         driver_pos: Tuple[float, float], office_pos: Tuple[float, float]) -> float:
        """Calculate total distance for a pickup sequence"""
        total_distance = 0.0
        current_pos = driver_pos
        
        for user_idx in sequence:
            user_pos = user_positions[user_idx]
            total_distance += self.get_road_distance(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
            current_pos = user_pos
        
        total_distance += self.get_road_distance(current_pos[0], current_pos[1], office_pos[0], office_pos[1])
        return total_distance

    @staticmethod
    def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in km"""
        R = 6371.0  # Earth radius in kilometers
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two points in degrees"""
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2_rad - lon1_rad
        x = math.sin(dlon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        initial_bearing = math.atan2(x, y)
        return (math.degrees(initial_bearing) + 360) % 360

    def _normalize_bearing_difference(self, diff: float) -> float:
        """Normalize bearing difference to [-180, 180] range"""
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    def _calculate_directional_penalty(self, route_points: List[Tuple[float, float]]) -> float:
        """Calculate penalty for excessive directional changes"""
        if len(route_points) < 3:
            return 0.0
        
        bearings = []
        for i in range(len(route_points) - 1):
            bearing = self._calculate_bearing(route_points[i][0], route_points[i][1], 
                                            route_points[i+1][0], route_points[i+1][1])
            bearings.append(bearing)
        
        if len(bearings) < 2:
            return 0.0
        
        total_change = 0.0
        for i in range(len(bearings) - 1):
            change = abs(self._normalize_bearing_difference(bearings[i+1] - bearings[i]))
            if change > 45:  # Significant turn
                total_change += (change - 45) / 135.0  # Normalize
        
        return min(1.0, total_change / max(1, len(bearings) - 1))

    def _calculate_backtrack_penalty(self, route_points: List[Tuple[float, float]], office_pos: Tuple[float, float]) -> float:
        """Calculate penalty for backtracking away from office"""
        if len(route_points) < 2:
            return 0.0
        
        penalty = 0.0
        prev_distance = self._calculate_distance(route_points[0][0], route_points[0][1], office_pos[0], office_pos[1])
        
        for i in range(1, len(route_points) - 1):  # Exclude final office position
            current_distance = self._calculate_distance(route_points[i][0], route_points[i][1], office_pos[0], office_pos[1])
            
            if current_distance > prev_distance + 0.5:  # 500m tolerance
                backtrack_amount = current_distance - prev_distance
                penalty += min(0.3, backtrack_amount / 10.0)  # Cap penalty
            
            prev_distance = current_distance
        
        return min(1.0, penalty)

    def invalidate_cache(self, pattern: Optional[str] = None):
        """Invalidate cache entries"""
        if pattern is None:
            self._path_cache.clear()
            self._distance_cache.clear()
            self._coherence_cache.clear()
        else:
            # Implement pattern-based invalidation if needed
            pass

    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        stats = self.graph_stats.copy()
        stats.update({
            'cache_sizes': {
                'path_cache': len(self._path_cache._cache),
                'distance_cache': len(self._distance_cache._cache),
                'coherence_cache': len(self._coherence_cache._cache)
            },
            'has_spatial_index': self._kdtree is not None,
            'projection_center': (self.projection.center_lat, self.projection.center_lon) if self.projection else None
        })
        return stats