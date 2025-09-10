
"""
Centralized configuration management to eliminate duplication across modules.
"""
import json
import os
from typing import Dict, Any, Optional
from utils.logger_config import get_logger

import json
import os
from typing import Dict, Any

logger = get_logger()

class ConfigManager:
    """Centralized configuration manager with mode-specific configs and key normalization."""
    
    _config_cache = None
    _config_file = "config.json"
    
    @classmethod
    def load_config(cls, config_file: str = None) -> Dict[str, Any]:
        """Load and cache configuration from JSON file."""
        if config_file:
            cls._config_file = config_file
            
        if cls._config_cache is None:
            try:
                with open(cls._config_file, 'r') as f:
                    cls._config_cache = json.load(f)
                logger.info(f"Configuration loaded from {cls._config_file}")
            except FileNotFoundError:
                logger.warning(f"Configuration file {cls._config_file} not found, using defaults")
                cls._config_cache = cls._get_default_config()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {cls._config_file}: {e}")
                raise
                
        return cls._config_cache
    
    @classmethod
    def _get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration when config file is not available."""
        return {
            "max_fill_distance_km": 5.0,
            "merge_distance_km": 3.0,
            "max_bearing_difference": 20,
            "max_turning_score": 35,
            "office_latitude": 30.6810489,
            "office_longitude": 76.7260711,
            "clustering_method": "adaptive",
            "use_sweep_algorithm": True,
            "mode_configs": {
                "route_efficiency": {
                    "max_fill_distance_km": 5.0,
                    "max_bearing_difference": 20,
                    "max_allowed_turning_score": 35
                },
                "capacity_optimization": {
                    "max_fill_distance_km": 8.0,
                    "max_bearing_difference": 45,
                    "max_allowed_turning_score": 60
                },
                "balanced_optimization": {
                    "max_fill_distance_km": 6.0,
                    "max_bearing_difference": 30,
                    "max_allowed_turning_score": 40
                }
            }
        }
    
    @classmethod
    def get_config(cls, mode: str = None) -> Dict[str, Any]:
        """Get configuration for specific mode with key normalization."""
        config = cls.load_config()
        
        # Start with base config
        result = config.copy()
        
        # Apply mode-specific overrides if mode is specified
        if mode and 'mode_configs' in config and mode in config['mode_configs']:
            mode_config = config['mode_configs'][mode]
            result.update(mode_config)
            logger.info(f"Applied mode-specific config for: {mode}")
        
        # Normalize key names (map legacy keys to new ones)
        result = cls._normalize_keys(result)
        
        return result
    
    @classmethod
    def _normalize_keys(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize configuration keys to handle legacy naming."""
        key_mappings = {
            # Map legacy keys to normalized keys
            'max_users_for_initial_cluster': 'max_users_per_initial_cluster',
            'max_bearing_diff': 'max_bearing_difference',
            'max_turn_score': 'max_turning_score',
            'max_fill_dist': 'max_fill_distance',
            'max_allowed_turning_score': 'max_turning_score',
            # Add more mappings as needed
        }
        
        normalized = config.copy()
        
        for legacy_key, new_key in key_mappings.items():
            if legacy_key in normalized:
                normalized[new_key] = normalized[legacy_key]
                logger.debug(f"Normalized key: {legacy_key} -> {new_key}")
        
        return normalized
    
    @classmethod
    def get(cls, key: str, default: Any = None, mode: str = None) -> Any:
        """Get specific config value with optional default."""
        config = cls.get_config(mode)
        return config.get(key, default)
    
    @classmethod
    def clear_cache(cls):
        """Clear cached configuration (useful for testing)."""
        cls._config_cache = None
