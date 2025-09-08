
from typing import Dict, Any, List, Tuple
import pandas as pd
from .base_optimizer import BaseOptimizer


class RouteEfficiencyOptimizer(BaseOptimizer):
    """Route efficiency optimization strategy - wraps existing assignment.py logic"""
    
    def __init__(self):
        super().__init__("route_efficiency")
    
    def optimize(self, user_df: pd.DataFrame, driver_df: pd.DataFrame, 
                office_lat: float, office_lon: float) -> Tuple[List[Dict[str, Any]], set]:
        """Run route efficiency optimization using existing assignment.py logic"""
        
        # Import the existing functions to maintain exact same logic
        from assignment import (
            create_geographic_clusters, create_capacity_subclusters,
            assign_drivers_by_priority, local_optimization, global_optimization
        )
        
        if self.logger:
            self.logger.info("🚗 Running Route Efficiency Optimization")
        
        # Step 1: Geographic clustering
        user_df = create_geographic_clusters(user_df, office_lat, office_lon, self.config)
        
        # Step 2: Capacity sub-clustering  
        user_df = create_capacity_subclusters(user_df, office_lat, office_lon, self.config)
        
        # Step 3: Driver assignment
        routes, assigned_user_ids = assign_drivers_by_priority(user_df, driver_df, office_lat, office_lon)
        
        # Step 4: Local optimization
        routes = local_optimization(routes, office_lat, office_lon)
        
        # Step 5: Global optimization
        routes, unassigned_users = global_optimization(routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon)
        
        return routes, assigned_user_ids
