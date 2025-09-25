from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import json

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

@app.post("/assign-drivers/{source_id}/{parameter}/{string_param}/{ridesetting}")
def assign_drivers(source_id: str, parameter: int, string_param: str, ridesetting: str):
    try:
        print(f"🚗 Starting assignment for source_id: {source_id}, parameter: {parameter}, string_param: {string_param}, ridesetting: {ridesetting}")
        
        # Import here to avoid circular imports
        from assignment import load_env_and_fetch_data
        
        # First, get the data to determine which algorithm to use
        try:
            data = load_env_and_fetch_data(source_id, parameter, string_param, ridesetting)
            
            # Ensure data is a dictionary
            if not isinstance(data, dict):
                raise ValueError(f"Expected dictionary from API, got {type(data)}")
            
            # Get algorithm priority from ride_settings
            ride_settings = data.get("ride_settings", {})
            if isinstance(ride_settings, dict):
                pic_priority = ride_settings.get("pic_priority")
                drop_priority = ride_settings.get("drop_priority")
                algorithm_priority = pic_priority if pic_priority is not None else drop_priority
            else:
                algorithm_priority = None
            
            print(f"🤖 Detected algorithm priority: {algorithm_priority}")
            
            # Route to appropriate algorithm based on priority
            if algorithm_priority == 1:
                print("🎪 Using CAPACITY OPTIMIZATION (Priority 1)")
                from assign_capacity import run_assignment_capacity
                result = run_assignment_capacity(source_id, parameter, string_param, ridesetting)
                result["optimization_mode"] = "capacity_optimization"
                
            elif algorithm_priority == 2:
                print("⚖️ Using BALANCED OPTIMIZATION (Priority 2)")
                from assign_balance import run_assignment_balance
                result = run_assignment_balance(source_id, parameter, string_param, ridesetting)
                result["optimization_mode"] = "balanced_optimization"
                
            elif algorithm_priority == 3:
                print("🗺️ Using ROAD-AWARE ROUTING (Priority 3)")
                from assign_route import run_road_aware_assignment
                result = run_road_aware_assignment(source_id, parameter, string_param, ridesetting)
                result["optimization_mode"] = "road_aware_route_optimization"
                
            else:
                print("🎯 Using ROUTE EFFICIENCY (Default)")
                from assignment import run_assignment
                result = run_assignment(source_id, parameter, string_param, ridesetting)
                result["optimization_mode"] = "route_efficiency_default"
                
        except Exception as api_error:
            print(f"⚠️ API error, using default algorithm: {api_error}")
            from assignment import run_assignment
            result = run_assignment(source_id, parameter, string_param, ridesetting)
            result["optimization_mode"] = "route_efficiency_default"

        # Ensure result has proper structure
        if not isinstance(result, dict):
            result = {"status": "false", "details": "Invalid result format", "data": []}

        # Add parameters to result
        result["parameter"] = parameter
        result["string_param"] = string_param
        result["ridesetting"] = ridesetting

        # Ensure data is always an array
        if "data" not in result:
            result["data"] = []
        elif result["data"] is None:
            result["data"] = []
        elif not isinstance(result["data"], list):
            result["data"] = []

        if result["status"] == "true":
            print(f"✅ Assignment successful. Routes: {len(result['data'])}")
            
            # Save to the main file for /routes endpoint
            try:
                with open("drivers_and_routes.json", "w") as f:
                    json.dump(result["data"], f, indent=2)
                print("💾 Results saved to drivers_and_routes.json")
            except Exception as save_error:
                print(f"⚠️ Failed to save results: {save_error}")
        else:
            print(f"❌ Assignment failed: {result.get('details', 'Unknown error')}")

        return result

    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "false", 
            "details": f"Server error: {str(e)}", 
            "data": [], 
            "parameter": parameter, 
            "string_param": string_param,
            "ridesetting": ridesetting,
            "optimization_mode": "error"
        }

@app.get("/routes")
def get_routes():
    # Check for the main route file first
    if os.path.exists("drivers_and_routes.json"):
        try:
            with open("drivers_and_routes.json", 'r') as f:
                data = json.load(f)
                # Ensure data is valid
                if isinstance(data, list):
                    return FileResponse("drivers_and_routes.json", media_type="application/json")
                else:
                    return {"status": "true", "data": data if data else [], "message": "Data loaded from drivers_and_routes.json"}
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error reading drivers_and_routes.json: {e}")
    
    # Check other route files as backup
    route_files = [
        "drivers_and_routes_capacity.json",
        "drivers_and_routes_balance.json",
        "drivers_and_routes_road_aware.json"
    ]

    for filename in route_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return FileResponse(filename, media_type="application/json")
                    else:
                        return {"status": "true", "data": data if data else [], "message": f"Data loaded from {filename}"}
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error reading {filename}: {e}")
                continue

    # Return empty but valid response
    return {"status": "false", "message": "No valid routes data available. Run assignment first.", "data": []}

@app.get("/visualize", response_class=HTMLResponse)
def get_visualization():
    return FileResponse("visualize.html")

@app.get("/")
def root():
    return {
        "message": "Driver Assignment API with Multiple Optimization Modes",
        "endpoints": [
            "/assign-drivers/{source_id}/{parameter}/{string_param}/{ridesetting}",
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