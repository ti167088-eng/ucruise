
from typing import Dict, Any
from .base_optimizer import BaseOptimizer
from .route_efficiency_optimizer import RouteEfficiencyOptimizer


class OptimizerFactory:
    """Factory class for creating optimization algorithms"""
    
    _optimizers = {
        "route_efficiency": RouteEfficiencyOptimizer,
        "capacity_optimization": None,  # Will wrap existing assign_capacity.py
        "balanced_optimization": None,  # Will wrap existing assign_balance.py  
        "route_optimization": None,     # Will wrap existing assign_route.py
    }
    
    @classmethod
    def create_optimizer(cls, mode: str) -> BaseOptimizer:
        """Create an optimizer instance for the given mode"""
        optimizer_class = cls._optimizers.get(mode)
        
        if optimizer_class is None:
            # Fallback to route efficiency for now
            # In a full implementation, you would create wrappers for other modes too
            return RouteEfficiencyOptimizer()
        
        return optimizer_class()
    
    @classmethod
    def get_available_modes(cls) -> list:
        """Get list of available optimization modes"""
        return list(cls._optimizers.keys())
