from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import json
import threading

# Start loading road network in background immediately on startup
from road_network_manager import road_network_manager
print("🗺️ Road network loading started in background...")

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


def run_all_assignments(source_id: str, parameter: int, string_param: str,
                        choice: str):
    """Runs all assignment algorithms in parallel and saves their results."""
    results = {}
    
    # Clear existing route files
    route_files_to_clear = [
        "drivers_and_routes.json", "drivers_and_routes_capacity.json",
        "drivers_and_routes_balance.json", "drivers_and_routes_road_aware.json",
        "drivers_and_routes_all_modes.json"
    ]
    for filename in route_files_to_clear:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"🗑️ Cleared cached file: {filename}")

    def run_and_save(algorithm_name, func, filename):
        try:
            print(f"🚗 Starting {algorithm_name} assignment...")
            # Pass parameters and choice to each assignment function
            result = func(source_id, parameter, string_param, choice)
            
            # Ensure result has proper structure
            if not isinstance(result, dict):
                result = {
                    "status": "false",
                    "details": "Invalid result format",
                    "data": []
                }

            # Add parameters to result
            result["parameter"] = parameter
            result["string_param"] = string_param
            result["choice"] = choice
            result["optimization_mode"] = algorithm_name # Set the specific mode

            # Ensure data is always an array
            if "data" not in result:
                result["data"] = []
            elif result["data"] is None:
                result["data"] = []
            elif not isinstance(result["data"], list):
                result["data"] = []

            results[algorithm_name] = result

            # Save individual mode results
            with open(filename, "w") as f:
                json.dump(result, f, indent=2)
            print(f"💾 Results for {algorithm_name} saved to {filename}")

        except Exception as e:
            print(f"❌ Error running {algorithm_name}: {e}")
            import traceback
            traceback.print_exc()
            results[algorithm_name] = {
                "status": "false",
                "details": f"Error: {str(e)}",
                "data": [],
                "parameter": parameter,
                "string_param": string_param,
                "choice": choice,
                "optimization_mode": algorithm_name
            }

    # Import here to avoid circular imports and ensure they are available
    from assignment import run_assignment
    from assign_capacity import run_assignment_capacity
    from assign_balance import run_assignment_balance
    from assign_route import run_road_aware_assignment

    threads = []
    threads.append(
        threading.Thread(target=run_and_save,
                         args=("route_efficiency_default", run_assignment,
                               "drivers_and_routes.json")))
    threads.append(
        threading.Thread(target=run_and_save,
                         args=("capacity_optimization", run_assignment_capacity,
                               "drivers_and_routes_capacity.json")))
    threads.append(
        threading.Thread(target=run_and_save,
                         args=("balanced_optimization", run_assignment_balance,
                               "drivers_and_routes_balance.json")))
    threads.append(
        threading.Thread(target=run_and_save,
                         args=("road_aware_route_optimization",
                               run_road_aware_assignment,
                               "drivers_and_routes_road_aware.json")))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # Combine all results into a single JSON file
    all_modes_result = {
        "all_modes_results": results,
        "parameter": parameter,
        "string_param": string_param,
        "choice": choice
    }
    try:
        with open("drivers_and_routes_all_modes.json", "w") as f:
            json.dump(all_modes_result, f, indent=2)
        print("💾 Combined results for all modes saved to drivers_and_routes_all_modes.json")
    except Exception as e:
        print(f"⚠️ Failed to save combined results: {e}")

    return all_modes_result


@app.post("/assign-drivers/{source_id}/{parameter}/{string_param}/{choice}")
def assign_drivers(source_id: str, parameter: int, string_param: str,
                   choice: str):
    try:
        print(
            f"🚗 Starting assignment for source_id: {source_id}, parameter: {parameter}, string_param: {string_param}, choice: {choice}"
        )

        # Handle empty choice parameter
        if not choice or choice.strip() == "":
            choice = " "  # Use space instead of empty string

        # Check if ALL_MODES is enabled. We'll assume this is determined by a specific choice value for now, e.g., "ALL_MODES"
        if choice == "ALL_MODES":
            print("🚀 Running ALL_MODES...")
            return run_all_assignments(source_id, parameter, string_param, choice)
        else:
            # Existing logic for single assignment
            try:
                # First, get the data to determine which algorithm to use
                from assignment import load_env_and_fetch_data
                data = load_env_and_fetch_data(source_id, parameter, string_param,
                                               choice)

                # Get algorithm priority from ride_settings
                ride_settings = data.get("ride_settings", {})
                pic_priority = ride_settings.get("pic_priority")
                drop_priority = ride_settings.get("drop_priority")

                # Use pic_priority first, then drop_priority, then default to None
                algorithm_priority = pic_priority
                if algorithm_priority is None:
                    algorithm_priority = drop_priority

                print(f"🤖 Detected algorithm priority: {algorithm_priority}")
                print(f"🔍 Raw ride_settings: {ride_settings}")

                result = {}
                # Route to appropriate algorithm based on priority
                if algorithm_priority == 1:
                    print("🎪 Using CAPACITY OPTIMIZATION (Priority 1)")
                    from assign_capacity import run_assignment_capacity
                    result = run_assignment_capacity(source_id, parameter,
                                                     string_param, choice)
                    result["optimization_mode"] = "capacity_optimization"

                elif algorithm_priority == 2:
                    print("⚖️ Using BALANCED OPTIMIZATION (Priority 2)")
                    from assign_balance import run_assignment_balance
                    result = run_assignment_balance(source_id, parameter,
                                                    string_param, choice)
                    result["optimization_mode"] = "balanced_optimization"

                elif algorithm_priority == 3:
                    print("🗺️ Using ROAD-AWARE ROUTING (Priority 3)")
                    from assign_route import run_road_aware_assignment
                    result = run_road_aware_assignment(source_id, parameter,
                                                       string_param, choice)
                    result["optimization_mode"] = "road_aware_route_optimization"

                else:
                    print(
                        f"🎯 Using ROUTE EFFICIENCY (Default) - priority was: {algorithm_priority}"
                    )
                    from assignment import run_assignment
                    result = run_assignment(source_id, parameter, string_param,
                                            choice)
                    result["optimization_mode"] = "route_efficiency_default"

            except Exception as api_error:
                print(f"⚠️ API error, using default algorithm: {api_error}")
                from assignment import run_assignment
                result = run_assignment(source_id, parameter, string_param, choice)
                result["optimization_mode"] = "route_efficiency_default"

            # Ensure result has proper structure
            if not isinstance(result, dict):
                result = {
                    "status": "false",
                    "details": "Invalid result format",
                    "data": []
                }

            # Add parameters to result
            result["parameter"] = parameter
            result["string_param"] = string_param
            result["choice"] = choice

            # Ensure data is always an array
            if "data" not in result:
                result["data"] = []
            elif result["data"] is None:
                result["data"] = []
            elif not isinstance(result["data"], list):
                result["data"] = []

            if result["status"] == "true":
                print(f"✅ Assignment successful. Routes: {len(result['data'])}")

                # Save to the main file for /routes endpoint if not ALL_MODES
                try:
                    with open("drivers_and_routes.json", "w") as f:
                        json.dump(result["data"], f, indent=2)
                    print("💾 Results saved to drivers_and_routes.json")
                except Exception as save_error:
                    print(f"⚠️ Failed to save results: {save_error}")
            else:
                print(
                    f"❌ Assignment failed: {result.get('details', 'Unknown error')}"
                )

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
            "choice": choice,
            "optimization_mode": "error"
        }


@app.get("/routes")
def get_routes():
    # Check for all modes file first
    if os.path.exists("drivers_and_routes_all_modes.json"):
        try:
            with open("drivers_and_routes_all_modes.json", 'r') as f:
                all_modes_data = json.load(f)
                
                # Transform to expected format for frontend
                if "all_modes_results" in all_modes_data:
                    modes_dict = {}
                    for mode_name, mode_result in all_modes_data["all_modes_results"].items():
                        # Extract just the data array from each mode
                        if isinstance(mode_result, dict) and "data" in mode_result:
                            modes_dict[mode_name] = {
                                "status": mode_result.get("status", "true"),
                                "data": mode_result["data"] if isinstance(mode_result["data"], list) else [],
                                "optimization_mode": mode_result.get("optimization_mode", mode_name)
                            }
                    
                    return {
                        "all_modes": True,
                        "modes": modes_dict
                    }
                    
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error reading drivers_and_routes_all_modes.json: {e}")

    # Check for the main route file
    if os.path.exists("drivers_and_routes.json"):
        try:
            with open("drivers_and_routes.json", 'r') as f:
                data = json.load(f)
                # Ensure data is an array
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "data" in data:
                    return data["data"] if isinstance(data["data"], list) else []
                else:
                    return []
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
                        return data
                    elif isinstance(data, dict) and "data" in data:
                        return data["data"] if isinstance(data["data"], list) else []
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error reading {filename}: {e}")
                continue

    # Return empty array (valid format for frontend)
    return []


@app.get("/visualize", response_class=HTMLResponse)
def get_visualization():
    return FileResponse("visualize.html")


@app.get("/")
def root():
    return {
        "message":
        "Driver Assignment API with Multiple Optimization Modes",
        "endpoints": [
            "/assign-drivers/{source_id}/{parameter}/{string_param}/{choice}",
            "/routes", "/visualize", "/health"
        ],
        "optimization_modes": {
            "all_modes":
            "Runs all assignment algorithms (Capacity, Balanced, Road-Aware, Default) in parallel and serves combined results.",
            "automatic_detection":
            "System automatically selects algorithm based on API ride_settings priority.",
            "priority_1":
            "Capacity Optimization (assign_capacity.py) - Maximizes seat utilization.",
            "priority_2":
            "Balanced Optimization (assign_balance.py) - 50/50 route efficiency + capacity.",
            "priority_3":
            "Road-Aware Routing (assign_route.py) - Uses road network data.",
            "default":
            "Route Efficiency (assignment.py) - Prioritizes straight routes."
        },
        "usage":
        "To run all modes, set choice to 'ALL_MODES' in the /assign-drivers endpoint. The /routes endpoint will then serve combined data. The visualization can then switch between modes."
    }