
import json
import os
import math
from typing import Dict, Any, Optional


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with defaults"""
        try:
            with open(self.config_file) as f:
                cfg = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            cfg = {}
        
        return cfg
    
    def _validate_config(self):
        """Validate configuration values"""
        # Add validation logic as needed
        pass
    
    def get_mode_config(self, mode: str) -> Dict[str, Any]:
        """Get configuration for a specific optimization mode"""
        base_config = self._get_base_config()
        mode_configs = self._config.get("mode_configs", {})
        mode_specific = mode_configs.get(mode, {})
        
        # Merge base config with mode-specific overrides
        config = base_config.copy()
        config.update(mode_specific)
        
        return config
    
    def _get_base_config(self) -> Dict[str, Any]:
        """Get base configuration common to all modes"""
        office_lat = float(os.getenv("OFFICE_LAT", self._config.get("office_latitude", 30.6810489)))
        office_lon = float(os.getenv("OFFICE_LON", self._config.get("office_longitude", 76.7260711)))
        
        # Validate coordinate bounds
        if not (-90 <= office_lat <= 90):
            office_lat = 30.6810489
        if not (-180 <= office_lon <= 180):
            office_lon = 76.7260711
        
        return {
            'OFFICE_LAT': office_lat,
            'OFFICE_LON': office_lon,
            'LAT_TO_KM': 111.0,
            'LON_TO_KM': 111.0 * math.cos(math.radians(office_lat)),
            
            # Distance configurations
            'MAX_FILL_DISTANCE_KM': max(0.1, float(self._config.get("max_fill_distance_km", 5.0))),
            'MERGE_DISTANCE_KM': max(0.1, float(self._config.get("merge_distance_km", 3.0))),
            'DBSCAN_EPS_KM': max(0.1, float(self._config.get("dbscan_eps_km", 1.5))),
            'OVERFLOW_PENALTY_KM': max(0.0, float(self._config.get("overflow_penalty_km", 10.0))),
            'DISTANCE_ISSUE_THRESHOLD': max(0.1, float(self._config.get("distance_issue_threshold_km", 8.0))),
            'SWAP_IMPROVEMENT_THRESHOLD': max(0.0, float(self._config.get("swap_improvement_threshold_km", 0.5))),
            
            # Utilization thresholds
            'MIN_UTIL_THRESHOLD': max(0.0, min(1.0, float(self._config.get("min_util_threshold", 0.5)))),
            'LOW_UTILIZATION_THRESHOLD': max(0.0, min(1.0, float(self._config.get("low_utilization_threshold", 0.5)))),
            
            # Integer configurations
            'MIN_SAMPLES_DBSCAN': max(1, int(self._config.get("min_samples_dbscan", 2))),
            'MAX_SWAP_ITERATIONS': max(1, int(self._config.get("max_swap_iterations", 3))),
            'MAX_USERS_FOR_FALLBACK': max(1, int(self._config.get("max_users_for_fallback", 3))),
            'FALLBACK_MIN_USERS': max(1, int(self._config.get("fallback_min_users", 2))),
            'FALLBACK_MAX_USERS': max(1, int(self._config.get("fallback_max_users", 7))),
            
            # Angle configurations
            'MAX_BEARING_DIFFERENCE': max(0, min(180, float(self._config.get("max_bearing_difference", 20)))),
            'MAX_TURNING_ANGLE': max(0, min(180, float(self._config.get("max_allowed_turning_score", 35)))),
            
            # Cost penalties
            'UTILIZATION_PENALTY_PER_SEAT': max(0.0, float(self._config.get("utilization_penalty_per_seat", 2.0))),
            
            # Clustering parameters
            'clustering_method': self._config.get('clustering_method', 'adaptive'),
            'min_cluster_size': max(2, self._config.get('min_cluster_size', 3)),
            'use_sweep_algorithm': self._config.get('use_sweep_algorithm', True),
            'angular_sectors': self._config.get('angular_sectors', 8),
            'max_users_per_initial_cluster': self._config.get('max_users_per_initial_cluster', 8),
            'max_users_per_cluster': self._config.get('max_users_per_cluster', 7),
            
            # Route optimization parameters
            'zigzag_penalty_weight': self._config.get('zigzag_penalty_weight', 3.0),
            'route_split_turning_threshold': self._config.get('route_split_turning_threshold', 35),
            'max_tortuosity_ratio': self._config.get('max_tortuosity_ratio', 1.4),
            'route_split_consistency_threshold': self._config.get('route_split_consistency_threshold', 0.7),
            'merge_tortuosity_improvement_required': self._config.get('merge_tortuosity_improvement_required', True)
        }


# Global config manager instance
config_manager = ConfigManager()


def get_config(mode: str) -> Dict[str, Any]:
    """Get configuration for a specific mode"""
    return config_manager.get_mode_config(mode)
