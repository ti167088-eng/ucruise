
import threading
import time
from logger_config import get_logger

logger = get_logger()

class RoadNetworkManager:
    """Manages road network loading in background"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.road_network = None
            self._loading = False
            self._loaded = False
            self._load_error = None
            self._mock_network = None
            self._initialized = True
    
    def start_loading(self, graphml_path='tricity_main_roads.graphml'):
        """Start loading road network in background"""
        if self._loading or self._loaded:
            return
        
        self._loading = True
        
        def load_worker():
            try:
                logger.info("🔄 Starting background road network loading...")
                
                # Try to import and load road network
                try:
                    import road_network as road_network_module
                    self.road_network = road_network_module.RoadNetwork(graphml_path)
                    self._loaded = True
                    logger.info("✅ Road network loaded successfully in background")
                except Exception as e:
                    logger.warning(f"Could not load road network: {e}. Using mock implementation.")
                    self._create_mock_network()
                    self._loaded = True
                    
            except Exception as e:
                logger.error(f"❌ Background road network loading failed: {e}")
                self._load_error = e
                self._create_mock_network()
                self._loaded = True
            finally:
                self._loading = False
        
        thread = threading.Thread(target=load_worker, daemon=True)
        thread.start()
        logger.info("🚀 Road network loading started in background")
    
    def _create_mock_network(self):
        """Create mock road network for fallback"""
        from assignment import haversine_distance
        
        class MockRoadNetwork:
            def get_route_coherence_score(self, driver_pos, user_positions, office_pos):
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
                
                score = max(0, 1.0 - (avg_dist_from_driver / 50.0) - (avg_dist_from_office / 100.0))
                return min(1.0, score)
            
            def is_user_on_route_path(self, driver_pos, current_user_positions, user_pos, office_pos, max_detour_ratio=1.5, route_type="optimization"):
                return True
            
            def get_road_distance(self, lat1, lon1, lat2, lon2):
                return haversine_distance(lat1, lon1, lat2, lon2)
            
            def find_nearest_road_node(self, lat, lon):
                return None, None
            
            def simplify_path_nodes(self, path, max_nodes=10):
                return path
        
        self.road_network = MockRoadNetwork()
    
    def get_road_network(self, wait_timeout=5):
        """Get road network, optionally waiting for it to load"""
        if self._loaded:
            return self.road_network
        
        if self._loading and wait_timeout > 0:
            logger.info(f"⏳ Waiting up to {wait_timeout}s for road network to load...")
            start_time = time.time()
            
            while self._loading and (time.time() - start_time) < wait_timeout:
                time.sleep(0.1)
        
        # If still not loaded, create mock network
        if not self._loaded:
            logger.warning("Road network not ready, using mock implementation")
            self._create_mock_network()
            self._loaded = True
        
        return self.road_network
    
    def is_loaded(self):
        """Check if road network is loaded"""
        return self._loaded
    
    def is_loading(self):
        """Check if road network is currently loading"""
        return self._loading

# Global instance
road_network_manager = RoadNetworkManager()
