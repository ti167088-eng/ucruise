
"""
Road network factory to centralize RoadNetwork instantiation.
"""
import os
from typing import Optional
from logger_config import get_logger

logger = get_logger()

import os
from typing import List, Tuple, Optional
from logger_config import get_logger
from .geo import haversine_distance

logger = get_logger()

class MockRoadNetwork:
    """Mock implementation of RoadNetwork for fallback when GraphML data is unavailable."""
    
    def get_route_coherence_score(self, driver_pos: Tuple[float, float], 
                                user_positions: List[Tuple[float, float]], 
                                office_pos: Tuple[float, float]) -> float:
        """Mock implementation: returns a score based on simple distance heuristic."""
        if not user_positions:
            return 1.0
            
        avg_dist_from_driver = sum(
            haversine_distance(driver_pos[0], driver_pos[1], u[0], u[1]) 
            for u in user_positions
        ) / len(user_positions)
        
        avg_dist_from_office = sum(
            haversine_distance(office_pos[0], office_pos[1], u[0], u[1]) 
            for u in user_positions
        ) / len(user_positions)

        # Simple heuristic: higher coherence if users are closer to the driver's path
        # and not too far from the office
        score = max(0, 1.0 - (avg_dist_from_driver / 50.0) - (avg_dist_from_office / 100.0))
        return min(1.0, score)

    def is_user_on_route_path(self, driver_pos: Tuple[float, float], 
                            current_user_positions: List[Tuple[float, float]], 
                            user_pos: Tuple[float, float], 
                            office_pos: Tuple[float, float], 
                            max_detour_ratio: float = 1.5, 
                            route_type: str = "optimization") -> bool:
        """Mock implementation: simple distance-based heuristic."""
        driver_to_office = haversine_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
        driver_to_user = haversine_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
        user_to_office = haversine_distance(user_pos[0], user_pos[1], office_pos[0], office_pos[1])
        
        # Check if detour is reasonable
        detour_ratio = (driver_to_user + user_to_office) / driver_to_office if driver_to_office > 0 else 1.0
        return detour_ratio <= max_detour_ratio

    def get_road_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Mock implementation: returns haversine distance."""
        return haversine_distance(lat1, lon1, lat2, lon2)

class RoadNetworkFactory:
    """Factory for creating and managing RoadNetwork instances."""
    
    _instance = None
    _mock_instance = None
    
    @classmethod
    def get_road_network(cls, graphml_file: str = 'tricity_main_roads.graphml', 
                        use_mock: bool = False) -> 'RoadNetwork':
        """
        Get RoadNetwork instance (singleton pattern).
        
        Args:
            graphml_file: Path to GraphML file
            use_mock: If True, return mock implementation
            
        Returns:
            RoadNetwork instance (real or mock)
        """
        if use_mock:
            return cls._get_mock_network()
        
        if cls._instance is None:
            cls._instance = cls._create_network(graphml_file)
        
        return cls._instance
    
    @classmethod
    def _create_network(cls, graphml_file: str) -> 'RoadNetwork':
        """Create RoadNetwork instance with proper error handling."""
        try:
            # Import here to avoid circular imports
            import road_network as road_network_module
            
            if not os.path.exists(graphml_file):
                logger.warning(f"GraphML file {graphml_file} not found, using mock implementation")
                return cls._get_mock_network()
            
            network = road_network_module.RoadNetwork(graphml_file)
            logger.info(f"Successfully loaded RoadNetwork from {graphml_file}")
            return network
            
        except Exception as e:
            logger.warning(f"Could not create RoadNetwork instance: {e}. Using mock implementation.")
            return cls._get_mock_network()
    
    @classmethod
    def _get_mock_network(cls):
        """Get mock RoadNetwork implementation."""
        if cls._mock_instance is None:
            cls._mock_instance = MockRoadNetwork()
            logger.info("Created mock RoadNetwork instance")
        
        return cls._mock_instance
    
    @classmethod
    def clear_cache(cls):
        """Clear cached instances (useful for testing)."""
        cls._instance = None
        cls._mock_instance = None

class MockRoadNetwork:
    """Mock implementation of RoadNetwork for fallback use."""
    
    def get_route_coherence_score(self, driver_pos, user_positions, office_pos):
        """Mock implementation: returns a score based on simple distance heuristic."""
        if not user_positions:
            return 1.0
        
        # Simple heuristic: lower score for more spread out users
        from .geo import haversine_distance
        
        total_distance = 0
        for i, pos1 in enumerate(user_positions):
            for pos2 in user_positions[i+1:]:
                total_distance += haversine_distance(pos1[0], pos1[1], pos2[0], pos2[1])
        
        # Normalize score (arbitrary heuristic)
        if len(user_positions) <= 1:
            return 1.0
        
        avg_distance = total_distance / (len(user_positions) * (len(user_positions) - 1) / 2)
        coherence = max(0.1, 1.0 - (avg_distance / 10.0))  # Arbitrary normalization
        
        return min(1.0, coherence)
    
    def get_route_score(self, route_points):
        """Mock route scoring."""
        if len(route_points) <= 1:
            return 1.0
        
        # Simple score based on total distance
        from .geo import haversine_distance
        
        total_distance = 0
        for i in range(len(route_points) - 1):
            total_distance += haversine_distance(
                route_points[i][0], route_points[i][1],
                route_points[i+1][0], route_points[i+1][1]
            )
        
        # Return inverse distance score (shorter routes = higher score)
        return max(0.1, 1.0 / (1.0 + total_distance / 10.0))
