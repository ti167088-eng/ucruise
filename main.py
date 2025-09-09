# Updated the assign_drivers function to support multiple optimization modes
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from assignment import run_assignment
from assign_route import run_road_aware_assignment
from assign_capacity import run_assignment_capacity
from assign_balance import run_assignment_balance
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AssignmentRequest(BaseModel):
    source_id: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/assign-drivers/{source_id}/{parameter}/{string_param}")
def assign_drivers(source_id: str, parameter: int, string_param: str):
    try:
        from logger_config import get_logger
        logger = get_logger()

        # Use automatic API-based routing like run_and_view.py
        logger.info(f"🤖 Using AUTOMATIC algorithm detection from API response")
        logger.info(f"📋 Parameters: {parameter}, String: {string_param}")
        
        # The run_assignment function will automatically route to the correct algorithm
        # based on _algorithm_priority from the API response:
        # Priority 1 → assign_capacity.py (Capacity Optimization)
        # Priority 2 → assign_balance.py (Balanced Optimization) 
        # Priority 3 → assign_route.py (Road-Aware Routing)
        # Default → assignment.py (Route Efficiency)
        
        result = run_assignment(source_id, parameter, string_param)
        optimization_mode = result.get("optimization_mode", "auto_detected")

        # Ensure the result includes the optimization mode used
        if isinstance(result, dict):
            result["optimization_mode_used"] = optimization_mode
            result["parameter"] = parameter
            result["string_param"] = string_param

        # Save results to appropriate file based on optimization mode
        if result["status"] == "true":
            filename_map = {
                "route_efficiency": "drivers_and_routes.json",
                "capacity_optimization": "drivers_and_routes_capacity.json", 
                "balanced_optimization": "drivers_and_routes_balance.json",
                "road_aware_route_optimization": "drivers_and_routes_road_aware.json",
                "route_efficiency_default": "drivers_and_routes.json"
            }

            filename = filename_map.get(optimization_mode, "drivers_and_routes.json")

            with open(filename, "w") as f:
                import json
                json.dump(result["data"], f, indent=2)

            logger.info(f"✅ Results saved to {filename}")

        return result

    except Exception as e:
        from logger_config import get_logger
        logger = get_logger()
        logger.critical(f"Server error in assign_drivers: {e}")
        return {
            "status": "false", 
            "details": f"Server error: {str(e)}", 
            "data": [], 
            "parameter": parameter, 
            "string_param": string_param,
            "optimization_mode_used": "error"
        }

@app.get("/routes")
def get_routes():
    if os.path.exists("drivers_and_routes.json"):
        return FileResponse("drivers_and_routes.json", media_type="application/json")
    else:
        return {"status": "false", "message": "No routes data available. Run assignment first.", "data": []}

@app.get("/visualize", response_class=HTMLResponse)
def get_visualization():
    return FileResponse("visualize.html")

@app.get("/")
def root():
    return {
        "message": "Driver Assignment API with Multiple Optimization Modes",
        "endpoints": [
            "/assign-drivers/{source_id}/{parameter}/{string_param}",
            "/routes", 
            "/visualize", 
            "/health"
        ],
        "optimization_modes": {
            "automatic_detection": "System automatically selects algorithm based on API ride_settings priority",
            "priority_1": "Capacity Optimization (assign_capacity.py) - Maximizes seat utilization",
            "priority_2": "Balanced Optimization (assign_balance.py) - 50/50 route efficiency + capacity",
            "priority_3": "Road-Aware Routing (assign_route.py) - Uses road network data",
            "default": "Route Efficiency (assignment.py) - Prioritizes straight routes"
        },
        "usage": "Algorithm is automatically selected from API response _algorithm_priority value"
    }