
import time
from typing import Dict, Any, Tuple, List
from data.config import get_config
from core.validation import validate_input_data
from data.models import AssignmentResult
from logger_config import get_logger


class AssignmentService:
    """Main assignment service that orchestrates different optimization strategies"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def run_assignment(self, source_id: str, parameter: int = 1, string_param: str = "") -> Dict[str, Any]:
        """Main assignment entry point that routes to appropriate algorithm"""
        start_time = time.time()
        
        try:
            # Load and validate data
            from assignment import load_env_and_fetch_data, extract_office_coordinates, prepare_user_driver_dataframes
            
            data = load_env_and_fetch_data(source_id, parameter, string_param)
            validate_input_data(data)
            
            # Extract office coordinates
            office_lat, office_lon = extract_office_coordinates(data)
            
            # Prepare dataframes
            user_df, driver_df = prepare_user_driver_dataframes(data)
            
            # Determine algorithm based on API data
            algorithm_priority = data.get("_algorithm_priority")
            optimization_mode = self._determine_optimization_mode(algorithm_priority)
            
            self.logger.info(f"🎯 Detected optimization mode: {optimization_mode}")
            
            # Route to appropriate algorithm
            if optimization_mode == "capacity_optimization":
                from assign_capacity import run_assignment_capacity
                result = run_assignment_capacity(source_id, parameter, string_param)
            elif optimization_mode == "balanced_optimization":
                from assign_balance import run_assignment_balance
                result = run_assignment_balance(source_id, parameter, string_param)
            elif optimization_mode == "route_optimization":
                from assign_route import run_assignment_route
                result = run_assignment_route(source_id, parameter, string_param)
            else:
                # Use modular route efficiency optimizer
                from algorithms.optimizer_factory import OptimizerFactory
                optimizer = OptimizerFactory.create_optimizer(optimization_mode)
                optimizer.set_logger(self.logger)
                
                # Run optimization
                routes, assigned_user_ids = optimizer.optimize(user_df, driver_df, office_lat, office_lon)
                
                # Handle unassigned users and build response
                result = self._build_response(routes, assigned_user_ids, user_df, driver_df, data, 
                                            optimization_mode, parameter, string_param, start_time)
            
            # Ensure result has execution time
            if "execution_time" not in result:
                result["execution_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Assignment failed: {e}", exc_info=True)
            return {
                "status": "false",
                "details": str(e),
                "data": [],
                "parameter": parameter,
                "string_param": string_param,
                "execution_time": execution_time
            }
    
    def _build_response(self, routes, assigned_user_ids, user_df, driver_df, data, 
                       optimization_mode, parameter, string_param, start_time):
        """Build assignment response"""
        from assignment import _get_all_drivers_as_unassigned, _convert_users_to_unassigned_format
        
        execution_time = time.time() - start_time
        
        # Build unassigned users list
        unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids)]
        unassigned_users = _convert_users_to_unassigned_format(unassigned_users_df.to_dict('records'))
        
        # Build unassigned drivers list
        assigned_driver_ids = {route['driver_id'] for route in routes}
        unassigned_drivers_df = driver_df[~driver_df['driver_id'].isin(assigned_driver_ids)]
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
        
        # Calculate clustering analysis
        clustering_results = {
            "method": f"{optimization_mode}_clustering",
            "clusters": user_df.get('geo_cluster', pd.Series()).nunique() if 'geo_cluster' in user_df else 0
        }
        
        return {
            "status": "true",
            "execution_time": execution_time,
            "data": routes,
            "unassignedUsers": unassigned_users,
            "unassignedDrivers": unassigned_drivers,
            "clustering_analysis": clustering_results,
            "optimization_mode": optimization_mode,
            "parameter": parameter,
            "string_param": string_param
        }
    
    def _determine_optimization_mode(self, algorithm_priority: Any) -> str:
        """Determine optimization mode based on API priority value"""
        if algorithm_priority is None:
            return "route_efficiency"  # Default
        
        try:
            priority = int(algorithm_priority)
            if priority == 1:
                return "capacity_optimization"
            elif priority == 2:
                return "balanced_optimization"
            elif priority == 3:
                return "route_optimization"
            else:
                return "route_efficiency"
        except (ValueError, TypeError):
            return "route_efficiency"


# Global service instance
assignment_service = AssignmentService()


def run_assignment(source_id: str, parameter: int = 1, string_param: str = "") -> Dict[str, Any]:
    """Main entry point for assignment"""
    return assignment_service.run_assignment(source_id, parameter, string_param)
