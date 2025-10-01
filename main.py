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


@app.post("/assign-drivers/{source_id}/{parameter}/{string_param}/{choice}")
def assign_drivers(source_id: str, parameter: int, string_param: str,
                   choice: str):
    try:
        print(
            f"🚗 Starting assignment for source_id: {source_id}, parameter: {parameter}, string_param: {string_param}, choice: {choice}"
        )

        # Clear any existing route files to ensure fresh assignment
        route_files = [
            "drivers_and_routes.json", "drivers_and_routes_capacity.json",
            "drivers_and_routes_balance.json",
            "drivers_and_routes_road_aware.json"
        ]
        for filename in route_files:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"🗑️ Cleared cached file: {filename}")

        # Import here to avoid circular imports
        from assignment import load_env_and_fetch_data

        # Handle empty choice parameter
        if not choice or choice.strip() == "":
            choice = " "  # Use space instead of empty string
            
        # First, get the data to determine which algorithm to use
        try:
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

            # Save to the main file for /routes endpoint
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
    # Check for the main route file first
    if os.path.exists("drivers_and_routes.json"):
        try:
            with open("drivers_and_routes.json", 'r') as f:
                data = json.load(f)
                # Ensure data is valid
                if isinstance(data, list):
                    return FileResponse("drivers_and_routes.json",
                                        media_type="application/json")
                else:
                    return {
                        "status": "true",
                        "data": data if data else [],
                        "message": "Data loaded from drivers_and_routes.json"
                    }
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error reading drivers_and_routes.json: {e}")

    # Check other route files as backup
    route_files = [
        "drivers_and_routes_capacity.json", "drivers_and_routes_balance.json",
        "drivers_and_routes_road_aware.json"
    ]

    for filename in route_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return FileResponse(filename,
                                            media_type="application/json")
                    else:
                        return {
                            "status": "true",
                            "data": data if data else [],
                            "message": f"Data loaded from {filename}"
                        }
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error reading {filename}: {e}")
                continue

    # Return empty but valid response
    return {
        "status": "false",
        "message": "No valid routes data available. Run assignment first.",
        "data": []
    }


@app.get("/visualize", response_class=HTMLResponse)
def get_visualization():
    return FileResponse("visualize.html")


@app.post("/compare-all-modes/{source_id}/{parameter}/{string_param}/{choice}")
def compare_all_optimization_modes(source_id: str, parameter: int, string_param: str, choice: str):
    """
    Run all three optimization modes in parallel and return results for comparison
    """
    try:
        print(f"🔄 Starting parallel optimization comparison for source_id: {source_id}")

        # Import optimization modules
        from assignment import run_assignment
        from assign_capacity import run_assignment_capacity
        from assign_balance import run_assignment_balance
        from assign_route import run_road_aware_assignment
        import concurrent.futures
        import time

        start_time = time.time()

        # Define optimization functions
        optimization_functions = {
            "route_efficiency": lambda: run_assignment(source_id, parameter, string_param, choice),
            "capacity_optimization": lambda: run_assignment_capacity(source_id, parameter, string_param, choice),
            "balanced_optimization": lambda: run_assignment_balance(source_id, parameter, string_param, choice),
            "road_aware_routing": lambda: run_road_aware_assignment(source_id, parameter, string_param, choice)
        }

        # Run all optimizations in parallel
        results = {}
        errors = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all optimization tasks
            future_to_mode = {
                executor.submit(func): mode 
                for mode, func in optimization_functions.items()
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_mode):
                mode = future_to_mode[future]
                try:
                    result = future.result(timeout=300)  # 5-minute timeout per mode
                    results[mode] = result
                    print(f"✅ {mode} completed successfully")
                except Exception as e:
                    error_msg = str(e)
                    errors[mode] = error_msg
                    print(f"❌ {mode} failed: {error_msg}")
                    # Still add a placeholder result
                    results[mode] = {
                        "status": "false",
                        "error": error_msg,
                        "data": [],
                        "optimization_mode": mode
                    }

        total_time = time.time() - start_time

        # Create comparison summary
        comparison_summary = {
            "execution_summary": {
                "total_execution_time": round(total_time, 2),
                "successful_modes": len([r for r in results.values() if r.get("status") == "true"]),
                "failed_modes": len(errors),
                "errors": errors
            },
            "mode_comparison": {}
        }

        # Generate comparison metrics for successful results
        for mode, result in results.items():
            if result.get("status") == "true" and result.get("data"):
                routes = result["data"]
                total_routes = len(routes)
                total_users_assigned = sum(len(route.get("assigned_users", [])) for route in routes)
                total_capacity = sum(route.get("vehicle_type", 0) for route in routes)
                overall_utilization = (total_users_assigned / total_capacity * 100) if total_capacity > 0 else 0

                unassigned_users = len(result.get("unassignedUsers", []))
                unassigned_drivers = len(result.get("unassignedDrivers", []))

                comparison_summary["mode_comparison"][mode] = {
                    "total_routes": total_routes,
                    "users_assigned": total_users_assigned,
                    "users_unassigned": unassigned_users,
                    "drivers_unassigned": unassigned_drivers,
                    "overall_utilization_percent": round(overall_utilization, 1),
                    "execution_time": result.get("execution_time", 0),
                    "clustering_method": result.get("clustering_analysis", {}).get("method", "unknown")
                }
            else:
                comparison_summary["mode_comparison"][mode] = {
                    "status": "failed",
                    "error": result.get("error", "Unknown error")
                }

        # Find best performing mode
        successful_modes = {
            mode: metrics for mode, metrics in comparison_summary["mode_comparison"].items()
            if "total_routes" in metrics
        }

        if successful_modes:
            # Best by utilization
            best_utilization_mode = max(successful_modes.items(), 
                                      key=lambda x: x[1]["overall_utilization_percent"])

            # Best by users assigned
            best_assignment_mode = max(successful_modes.items(),
                                     key=lambda x: x[1]["users_assigned"])

            comparison_summary["recommendations"] = {
                "best_utilization": {
                    "mode": best_utilization_mode[0],
                    "utilization": best_utilization_mode[1]["overall_utilization_percent"]
                },
                "best_user_assignment": {
                    "mode": best_assignment_mode[0],
                    "users_assigned": best_assignment_mode[1]["users_assigned"]
                }
            }

        print(f"🎉 Parallel optimization comparison completed in {total_time:.2f}s")

        return {
            "status": "true",
            "message": "All optimization modes completed",
            "comparison_summary": comparison_summary,
            "detailed_results": {
                "route_efficiency": results.get("route_efficiency"),
                "capacity_optimization": results.get("capacity_optimization"), 
                "balanced_optimization": results.get("balanced_optimization"),
                "road_aware_routing": results.get("road_aware_routing")
            },
            "parameters": {
                "source_id": source_id,
                "parameter": parameter,
                "string_param": string_param,
                "choice": choice
            }
        }

    except Exception as e:
        print(f"❌ Parallel optimization comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "false",
            "error": f"Parallel comparison failed: {str(e)}",
            "parameters": {
                "source_id": source_id,
                "parameter": parameter,
                "string_param": string_param,
                "choice": choice
            }
        }


@app.get("/")
def root():
    return {
        "message":
        "Driver Assignment API with Multiple Optimization Modes",
        "endpoints": [
            "/assign-drivers/{source_id}/{parameter}/{string_param}/{choice}",
            "/compare-all-modes/{source_id}/{parameter}/{string_param}/{choice}",
            "/routes", "/visualize", "/health"
        ],
        "optimization_modes": {
            "automatic_detection":
            "System automatically selects algorithm based on API ride_settings priority",
            "priority_1":
            "Capacity Optimization (assign_capacity.py) - Maximizes seat utilization",
            "priority_2":
            "Balanced Optimization (assign_balance.py) - 50/50 route efficiency + capacity",
            "priority_3":
            "Road-Aware Routing (assign_route.py) - Uses road network data",
            "default":
            "Route Efficiency (assignment.py) - Prioritizes straight routes"
        },
        "usage":
        "Use /compare-all-modes/ to run all optimization modes in parallel for comparison"
    }