import subprocess
import threading
import time
import webbrowser
import sys
import argparse
import os
import json
import requests
import urllib.parse
from logger_config import clear_logs

SOURCE_ID = "UC_logisticllp"  # <-- Replace with your real source_id
PARAMETER = 1  # Example numerical parameter
STRING_PARAM = "Evening shift" # Example string parameter (no URL encoding)
CHOICE = " " # Example choice parameter (use "1" to match main.py behavior)
ALL_MODES = True  # Set to True to run all algorithms in parallel

def start_fastapi():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI on 0.0.0.0:5000")
    subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--reload", "--log-level", "info"])

def launch_browser():
    """Launch browser after server starts"""
    time.sleep(5)  # Wait for server to start
    try:
        webbrowser.open("http://localhost:5000/visualize")
        print("🌐 Browser opened at: http://localhost:5000/visualize")
    except Exception as e:
        print(f"⚠️  Could not auto-open browser: {e}")
        print("   Please manually visit: http://localhost:5000/visualize")

def run_single_mode(mode_name, source_id, parameter, string_param, choice):
    """Run a single assignment mode with its own config"""
    import shutil

    # Create mode-specific config backup
    config_backup = f"config_{mode_name}.json"

    try:
        # Load and modify config for this mode
        with open('config.json', 'r') as f:
            config = json.load(f)

        # Set mode-specific optimization
        if mode_name == "route_efficiency":
            config['optimization_mode'] = 'route_efficiency'
        elif mode_name == "capacity":
            config['optimization_mode'] = 'capacity_optimization'
        elif mode_name == "balanced":
            config['optimization_mode'] = 'balanced_optimization'

        # Save mode-specific config
        with open(config_backup, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"🔄 Running {mode_name.upper()} mode...")

        # Run the appropriate assignment module directly
        if mode_name == "route_efficiency":
            from assignment import run_assignment
            result = run_assignment(source_id, parameter, string_param, choice)
        elif mode_name == "capacity":
            from assign_capacity import run_assignment_capacity
            result = run_assignment_capacity(source_id, parameter, string_param, choice)
        elif mode_name == "balanced":
            from assign_balance import run_assignment_balance
            result = run_assignment_balance(source_id, parameter, string_param, choice)

        # Save mode-specific results
        output_file = f"drivers_and_routes_{mode_name}.json"
        if result and result.get("status") == "true":
            with open(output_file, 'w') as f:
                json.dump(result["data"], f, indent=2)
            print(f"✅ {mode_name.upper()} results saved to {output_file}")

        return result

    except Exception as e:
        print(f"❌ Error running {mode_name} mode: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_all_modes_parallel(source_id, parameter, string_param, choice):
    """Run all three modes in parallel"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import copy

    print("\n🚀 Running ALL MODES in parallel...")
    print("="*60)

    # Clear any existing cached routes
    clear_cached_routes()

    modes = ["route_efficiency", "capacity", "balanced"]
    results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_mode = {
            executor.submit(run_single_mode, mode, source_id, parameter, string_param, choice): mode 
            for mode in modes
        }

        # Collect results as they complete
        for future in as_completed(future_to_mode):
            mode = future_to_mode[future]
            try:
                result = future.result()
                results[mode] = result
                if result and result.get("status") == "true":
                    routes_count = len(result.get("data", []))
                    print(f"✅ {mode.upper()}: {routes_count} routes created")
            except Exception as e:
                print(f"❌ {mode.upper()} failed: {e}")
                results[mode] = None

    # Create combined results file
    combined_results = {
        "all_modes": True,
        "modes": {}
    }

    for mode, result in results.items():
        if result and result.get("status") == "true":
            combined_results["modes"][mode] = {
                "status": result["status"],
                "data": result["data"],
                "execution_time": result.get("execution_time", 0),
                "optimization_mode": result.get("optimization_mode", mode),
                "unassignedUsers": result.get("unassignedUsers", []),
                "unassignedDrivers": result.get("unassignedDrivers", [])
            }

    # Save combined results
    with open("drivers_and_routes_all_modes.json", 'w') as f:
        json.dump(combined_results, f, indent=2)

    print("\n✅ All modes completed!")
    print(f"📊 Combined results saved to drivers_and_routes_all_modes.json")

    return combined_results

def call_assignment_api(source_id, parameter, string_param, choice):
    """Call the FastAPI assignment endpoint"""
    # URL encode the string parameter to handle spaces
    import urllib.parse
    encoded_string_param = urllib.parse.quote(string_param)

    url = f"http://localhost:5000/assign-drivers/{source_id}/{parameter}/{encoded_string_param}/{choice}"

    try:
        print(f"📡 Calling assignment API: {url}")
        print("⏳ Processing assignment (this may take 30-60 seconds)...")
        print("🔄 Server is running assignment algorithm...")

        # Clear any existing cached routes before making the call
        clear_cached_routes()

        response = requests.post(url, timeout=1800)  # Increased timeout
        response.raise_for_status()

        result = response.json()
        print("✅ API response received successfully!")
        return result

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to FastAPI server. Make sure it's running on port 5000.")
        return None
    except requests.exceptions.Timeout:
        print("❌ API request timed out. The assignment is taking longer than expected.")
        print("💡 Try running 'python test_analysis.py' for direct assignment testing")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON response: {e}")
        return None

def clear_cached_routes():
    """Clear any cached route files to ensure fresh data"""
    route_files = [
        "drivers_and_routes.json",
        "drivers_and_routes_capacity.json",
        "drivers_and_routes_balance.json",
        "drivers_and_routes_road_aware.json"
    ]

    for filename in route_files:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"🗑️ Cleared cached file: {filename}")
            except Exception as e:
                print(f"⚠️ Could not clear {filename}: {e}")

def display_detailed_analytics(result, algorithm_name):
    """Display comprehensive analytics in terminal with enhanced formatting"""
    print("\n" + "🎯" + "="*78 + "🎯")
    print(f"📊 ROUTEFLOW - INTELLIGENT ASSIGNMENT DASHBOARD ({algorithm_name})")
    print("🎯" + "="*78 + "🎯")

    if result["status"] != "true":
        print("❌ Assignment failed - no analytics available")
        return

    # Basic metrics
    routes = result["data"]
    unassigned_users = result.get("unassignedUsers", [])
    unassigned_drivers = result.get("unassignedDrivers", [])

    total_assigned = sum(len(route["assigned_users"]) for route in routes)
    total_users = total_assigned + len(unassigned_users)

    # Handle no users case
    if total_users == 0:
        print("\n📈 SYSTEM OVERVIEW")
        print("─" * 50)
        print(f"   🚗 Active Routes Created: {len(routes)}")
        print(f"   👥 Users Successfully Assigned: 0")
        print(f"   ⚠️  Users Unassigned: 0")
        print(f"   🚙 Drivers Deployed: 0")
        print(f"   💤 Drivers Available: {len(unassigned_drivers)}")
        print(f"   🏆 Total Fleet Capacity: 0 passengers")
        print(f"\nℹ️  No users found for assignment")
        print(f"   📊 System ready for user assignment when users are available")
        return

    total_capacity = sum(route["vehicle_type"] for route in routes)

    # Enhanced Overview Section
    print(f"\n📈 SYSTEM OVERVIEW")
    print("─" * 50)
    print(f"   🤖 Algorithm Used: {result.get('optimization_mode', 'Unknown')}")
    print(f"   🚗 Active Routes Created: {len(routes)}")
    print(f"   👥 Users Successfully Assigned: {total_assigned}")
    print(f"   ⚠️  Users Unassigned: {len(unassigned_users)}")
    print(f"   🚙 Drivers Deployed: {len(routes)}")
    print(f"   💤 Drivers Available: {len(unassigned_drivers)}")
    print(f"   🏆 Total Fleet Capacity: {total_capacity} passengers")
    if total_capacity > 0:
        print(f"   📊 Overall Capacity Utilization: {(total_assigned/total_capacity*100):.1f}%")
    else:
        print(f"   📊 Overall Capacity Utilization: N/A (no capacity available)")

    # Route Performance Analysis
    print(f"\n🚗 DETAILED ROUTE PERFORMANCE ANALYSIS")
    print("─" * 50)

    for i, route in enumerate(routes):
        assigned = len(route["assigned_users"])
        capacity = route["vehicle_type"]
        utilization = assigned / capacity if capacity > 0 else 0

        # Efficiency categorization
        if utilization >= 0.8:
            efficiency_icon = "🟢"
            efficiency_label = "EXCELLENT"
        elif utilization >= 0.6:
            efficiency_icon = "🟡"
            efficiency_label = "GOOD"
        elif utilization >= 0.4:
            efficiency_icon = "🟠"
            efficiency_label = "FAIR"
        else:
            efficiency_icon = "🔴"
            efficiency_label = "NEEDS OPTIMIZATION"

        print(f"   Route {i+1:2d}: {efficiency_icon} {efficiency_label:15} | "
              f"{assigned}/{capacity} users ({utilization*100:5.1f}%) | "
              f"Driver: {route['driver_id']} | Vehicle: {route.get('vehicle_id', 'N/A')}")

    if unassigned_users:
        print(f"\n⚠️  UNASSIGNED USERS REQUIRING ATTENTION")
        print("─" * 50)
        for i, user in enumerate(unassigned_users[:5]):  # Show first 5
            print(f"   {i+1}. User {user['user_id']}: Location ({user.get('lat', 'N/A')}, {user.get('lng', 'N/A')})")
        if len(unassigned_users) > 5:
            print(f"   ... and {len(unassigned_users) - 5} more users need manual assignment")

    print("\n" + "🎯" + "="*78 + "🎯")
    print("🌐 ACCESS FULL INTERACTIVE DASHBOARD: http://localhost:5000/visualize")
    print("📊 Real-time analytics, route optimization, and performance monitoring available")
    print("🎯" + "="*78 + "🎯\n")

def wait_for_server():
    """Wait for FastAPI server to be ready"""
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:5000/health", timeout=2)
            if response.status_code == 200:
                print("✅ FastAPI server is ready!")
                print("🌐 Server URL: http://localhost:5000")
                print("📊 Health check: PASSED")
                return True
        except requests.exceptions.RequestException:
            pass

        if attempt == 0:
            print("⏳ Waiting for FastAPI server to start...")
        if attempt % 5 == 0 and attempt > 0:
            print(f"⏳ Still waiting... ({attempt + 1}/{max_attempts}) - Server starting...")
        time.sleep(2)

    print("❌ Server failed to start within timeout period")
    print("💡 Try manually running: uvicorn main:app --host 0.0.0.0 --port 5000")
    return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Driver Assignment Dashboard')
    parser.add_argument('--source-id', type=str, default=SOURCE_ID,
                       help=f'Source ID for API (default: {SOURCE_ID})')
    parser.add_argument('--parameter', type=int, default=PARAMETER,
                       help=f'Parameter value (default: {PARAMETER})')
    parser.add_argument('--string-param', type=str, default=STRING_PARAM,
                       help=f'String parameter (default: {STRING_PARAM})')
    parser.add_argument('--choice', type=str, default=CHOICE,
                       help=f'Choice parameter (default: {CHOICE})')
    parser.add_argument('--all-modes', action='store_true', default=ALL_MODES,
                       help='Run all assignment modes in parallel (default: True if ALL_MODES is True)')
    args = parser.parse_args()

    ALL_MODES = args.all_modes # Update ALL_MODES based on command line argument

    # Clear logs at the start
    clear_logs()

    print("🚀 Starting Driver Assignment Dashboard...")
    print(f"📍 Source ID: {args.source_id}")
    print(f"📊 Parameter: {args.parameter}")
    print(f"📝 String Parameter: {args.string_param}")
    print(f"🎯 Choice: {args.choice}")

    try:
        print("\n🔧 Starting FastAPI Server...")

        # Start server in background
        server_thread = threading.Thread(target=start_fastapi, daemon=True)
        server_thread.start()

        # Wait for server to be ready
        if not wait_for_server():
            print("❌ Failed to start server")
            exit(1)

        print("\n🤖 Running Assignment...")
        print("=" * 60)
        print(f"🔄 Using parameters: source_id={args.source_id}, parameter={args.parameter}, string_param='{args.string_param}', choice={args.choice}")
        print(f"🎯 ALL_MODES: {ALL_MODES}")

        if ALL_MODES:
            # Run all modes in parallel
            result = run_all_modes_parallel(args.source_id, args.parameter, args.string_param, args.choice)

            if not result or not result.get("modes"):
                print("❌ All modes assignment failed")
                exit(1)

            print("\n📊 ALL MODES SUMMARY:")
            print("="*60)
            for mode, mode_result in result["modes"].items():
                routes_count = len(mode_result.get("data", []))
                exec_time = mode_result.get("execution_time", 0)
                print(f"   {mode.upper():20} | {routes_count:3} routes | {exec_time:.1f}s")
            print("="*60)

        else:
            # Call the assignment API (single mode)
            result = call_assignment_api(args.source_id, args.parameter, args.string_param, args.choice)

            if not result:
                print("❌ Assignment API call failed")
                print("💡 The server may still be processing. Check http://localhost:5000/routes for results")
                print("💡 Or try running 'python test_analysis.py' for direct testing")
                exit(1)

            print(f"📋 Result status: {result.get('status', 'unknown')}")
            print(f"🔍 Result keys: {list(result.keys())}")

            if result["status"] == "true":
                # Get the algorithm name from the result
                algorithm_name = result.get("optimization_mode", "AUTO-DETECTED ALGORITHM")
                algorithm_name = algorithm_name.replace("_", " ").upper()

                print(f"✅ {algorithm_name} assignment completed successfully!")
                print(f"📊 Routes Created: {len(result.get('data', []))}")

                # Display detailed analytics
                display_detailed_analytics(result, algorithm_name)

            else:
                print("❌ Assignment failed:")
                print(f"   Error: {result.get('details', 'Unknown error')}")
                print(f"   Please check your configuration and API credentials")
                exit(1)

        print("\n🌐 Launching Dashboard...")

        # Launch browser
        browser_thread = threading.Thread(target=launch_browser, daemon=True)
        browser_thread.start()

        print("\n✅ Dashboard is running!")
        print("📱 Dashboard URL: http://localhost:5000/visualize")
        print("📊 API Endpoint: http://localhost:5000/routes")
        print("🔍 Health Check: http://localhost:5000/health")
        print("\n⌨️  Press Ctrl+C to stop the server")

        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down dashboard...")
            print("👋 Goodbye!")

    except KeyboardInterrupt:
        print("\n🛑 Dashboard startup interrupted by user")
        exit(0)
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print(f"📋 Error type: {type(e).__name__}")
        exit(1)