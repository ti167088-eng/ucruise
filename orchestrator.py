
#!/usr/bin/env python3
"""
Route Assignment Orchestrator - Main coordination system
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logger_config import get_logger
from algorithms.assign_route import run_route_assignment
from algorithms.assign_balance import run_balanced_assignment  
from algorithms.assign_capacity import run_capacity_assignment

FILE_CONTEXT = "ORCHESTRATOR.PY (MAIN COORDINATION)"

def log_session_start():
    """Log the start of a new assignment session"""
    import datetime
    import random
    
    logger = get_logger()
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + str(random.randint(100, 999))
    
    logger.info("=" * 80, FILE_CONTEXT)
    logger.info("ROUTE ASSIGNMENT SESSION STARTED", FILE_CONTEXT)
    logger.info(f"Session ID: {session_id}", FILE_CONTEXT)
    logger.info("=" * 80, FILE_CONTEXT)
    
    return session_id

def assignment_mode(source_id, mode, string_param=""):
    """
    Route assignment mode dispatcher
    
    Args:
        source_id: Source identifier
        mode: Assignment mode (1=route, 2=balance, 3=capacity, 4=legacy)
        string_param: Additional parameters
    """
    logger = get_logger()
    
    # Mode mapping for clear identification
    mode_map = {
        1: ("ROUTE EFFICIENCY", "ASSIGNMENT.PY (ROUTE EFFICIENCY)"),
        2: ("BALANCED ROUTE OPTIMIZATION", "ASSIGN_BALANCE.PY (BALANCED)"),
        3: ("CAPACITY OPTIMIZATION", "ASSIGN_CAPACITY.PY (CAPACITY)"),
        4: ("LEGACY ASSIGNMENT", "ASSIGNMENT.PY (LEGACY)")
    }
    
    mode_name, file_context = mode_map.get(mode, ("UNKNOWN", "UNKNOWN"))
    
    logger.info("=" * 60, FILE_CONTEXT)
    logger.info(f"🎯 ASSIGNMENT MODE: {mode_name} | FILE: {file_context}", FILE_CONTEXT)
    logger.info("=" * 60, FILE_CONTEXT)
    
    try:
        if mode == 1:
            # Route efficiency mode - using assign_route.py
            return run_route_assignment(source_id, 1, string_param)
        elif mode == 2:
            # Balanced route optimization mode - using assign_balance.py  
            return run_balanced_assignment(source_id, 1, string_param)
        elif mode == 3:
            # Capacity optimization mode - using assign_capacity.py
            return run_capacity_assignment(source_id, 1, string_param)
        elif mode == 4:
            # Legacy assignment mode - using assignment.py directly
            import assignment
            return assignment.run_route_efficiency_assignment(source_id, 1, string_param)
        else:
            error_msg = f"❌ Invalid assignment mode: {mode}. Valid modes: 1-4"
            logger.error(error_msg, FILE_CONTEXT)
            return {
                "status": "false",
                "error": error_msg,
                "data": []
            }
            
    except Exception as e:
        error_msg = f"❌ Error in assignment mode {mode}: {str(e)}"
        logger.error(error_msg, FILE_CONTEXT)
        import traceback
        logger.error(traceback.format_exc(), FILE_CONTEXT)
        return {
            "status": "false", 
            "error": error_msg,
            "data": []
        }

def orchestrator_routing_assignment(source_id, mode=1, string_param="Evening shift"):
    """
    Main orchestrator function for routing assignments
    
    Args:
        source_id: Source identifier for the assignment
        mode: Assignment mode (default: 1 for route efficiency)
        string_param: Additional parameters (default: "Evening shift")
    """
    logger = get_logger()
    
    # Log session start
    session_id = log_session_start()
    
    # Log the orchestrator call
    logger.info(f"🎯 Orchestrator routing assignment: source_id={source_id}, mode={mode}", FILE_CONTEXT)
    
    try:
        # Call the assignment mode dispatcher
        result = assignment_mode(source_id, mode, string_param)
        
        # Log completion
        if result.get("status") == "true":
            routes_count = len(result.get("data", []))
            unassigned_count = len(result.get("unassignedUsers", []))
            logger.info(f"✅ Assignment completed: {routes_count} routes, {unassigned_count} unassigned users", FILE_CONTEXT)
        else:
            logger.error(f"❌ Assignment failed: {result.get('error', 'Unknown error')}", FILE_CONTEXT)
        
        return result
        
    except Exception as e:
        error_msg = f"❌ Orchestrator error: {str(e)}"
        logger.error(error_msg, FILE_CONTEXT)
        import traceback
        logger.error(traceback.format_exc(), FILE_CONTEXT)
        return {
            "status": "false",
            "error": error_msg,
            "data": [],
            "session_id": session_id
        }

def run_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """
    Main assignment dispatcher function
    """
    return orchestrator_routing_assignment(source_id, parameter, string_param)

def main():
    """Main entry point"""
    logger = get_logger()
    logger.info("🚀 Route Assignment System Starting...", FILE_CONTEXT)
    
    # Example usage - replace with your actual parameters
    result = orchestrator_routing_assignment(
        source_id="UC_frontdev",
        mode=1,  # Route efficiency mode
        string_param="Evening shift"
    )
    
    logger.info(f"📋 Final result status: {result.get('status')}", FILE_CONTEXT)
    return result

if __name__ == "__main__":
    main()
