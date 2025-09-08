
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import pandas as pd
from data.config import get_config


class BaseOptimizer(ABC):
    """Abstract base class for all optimization algorithms"""
    
    def __init__(self, mode: str):
        self.mode = mode
        self.config = get_config(mode)
        self.logger = None
    
    @abstractmethod
    def optimize(self, user_df: pd.DataFrame, driver_df: pd.DataFrame, 
                office_lat: float, office_lon: float) -> Tuple[List[Dict[str, Any]], set]:
        """
        Main optimization method that each algorithm must implement
        
        Args:
            user_df: DataFrame of users to assign
            driver_df: DataFrame of available drivers
            office_lat: Office latitude
            office_lon: Office longitude
            
        Returns:
            Tuple of (routes, assigned_user_ids)
        """
        pass
    
    def set_logger(self, logger):
        """Set the logger instance"""
        self.logger = logger
    
    def get_mode_config(self) -> Dict[str, Any]:
        """Get configuration for this optimization mode"""
        return self.config
