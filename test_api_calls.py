
import requests
import json
import time
import urllib.parse
from datetime import datetime

# Configuration - Update these to match your API setup
import os
import platform

# Auto-detect if running locally or in Replit
if platform.system() == "Windows" or os.path.exists("C:\\"):
    # Running on local Windows machine - use your Replit URL
    BASE_URL = "https://your-repl-name.your-username.replit.dev"  # Replace with your actual Replit URL
    print("🖥️ Detected Windows environment - using Replit URL")
else:
    # Running in Replit environment
    BASE_URL = "http://0.0.0.0:5000"
    print("☁️ Detected Replit environment - using local server")

SOURCE_ID = "UC_logisticllp"
PARAMETER = 1
STRING_PARAM = "Evening shift"
RIDESETTING = " "  # Space character as used in your main.py

def test_health_endpoint():
    """Test if the server is responding"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False

def test_debug_endpoint():
    """Test the debug endpoint to check API configuration"""
    print("\n🔍 Testing debug endpoint...")
    # URL encode parameters properly
    encoded_string_param = urllib.parse.quote(STRING_PARAM, safe='')
    encoded_ridesetting = urllib.parse.quote(RIDESETTING, safe='')
    
    url = f"{BASE_URL}/debug-api/{SOURCE_ID}/{PARAMETER}/{encoded_string_param}/{encoded_ridesetting}"
    
    try:
        response = requests.get(url, timeout=30)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            debug_data = response.json()
            print(f"   Base URL: {debug_data.get('base_url')}")
            print(f"   Final URL: {debug_data.get('final_url')}")
            print(f"   Has Token: {debug_data.get('has_token')}")
            print(f"   Token Length: {debug_data.get('token_length')}")
            print(f"   Original String Param: {debug_data.get('encoded_params', {}).get('original_string_param')}")
            print(f"   Encoded String Param: {debug_data.get('encoded_params', {}).get('encoded_string_param')}")
            return True
        else:
            print(f"   ❌ Debug endpoint failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Debug endpoint error: {e}")
        return False

def test_assignment_endpoint():
    """Test the main assignment endpoint"""
    print("\n🚗 Testing assignment endpoint...")
    
    # URL encode parameters properly
    encoded_string_param = urllib.parse.quote(STRING_PARAM, safe='')
    encoded_ridesetting = urllib.parse.quote(RIDESETTING, safe='')
    
    url = f"{BASE_URL}/assign-drivers/{SOURCE_ID}/{PARAMETER}/{encoded_string_param}/{encoded_ridesetting}"
    
    print(f"   Making POST request to: {url}")
    print(f"   Timeout: 120 seconds")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, timeout=120)
        elapsed_time = time.time() - start_time
        
        print(f"   Status: {response.status_code}")
        print(f"   Response time: {elapsed_time:.2f} seconds")
        print(f"   Content-Type: {response.headers.get('content-type', 'unknown')}")
        print(f"   Content-Length: {len(response.text)} characters")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"   ✅ Assignment successful!")
                print(f"   Status: {result.get('status')}")
                print(f"   Execution time: {result.get('execution_time', 0):.2f}s")
                print(f"   Routes created: {len(result.get('data', []))}")
                print(f"   Unassigned users: {len(result.get('unassignedUsers', []))}")
                print(f"   Unassigned drivers: {len(result.get('unassignedDrivers', []))}")
                print(f"   Optimization mode: {result.get('optimization_mode', 'unknown')}")
                return result
            except json.JSONDecodeError as e:
                print(f"   ❌ Invalid JSON response: {e}")
                print(f"   Raw response: {response.text[:500]}...")
                return None
        else:
            print(f"   ❌ Assignment failed with status {response.status_code}")
            print(f"   Error response: {response.text[:500]}...")
            return None
            
    except requests.exceptions.Timeout:
        elapsed_time = time.time() - start_time
        print(f"   ❌ Request timed out after {elapsed_time:.2f} seconds")
        print(f"   This indicates the API is taking too long to respond")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ Connection error: {e}")
        print(f"   Make sure the FastAPI server is running on port 5000")
        return None
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"   ❌ Unexpected error after {elapsed_time:.2f} seconds: {e}")
        return None

def test_routes_endpoint():
    """Test the routes endpoint"""
    print("\n📊 Testing routes endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/routes", timeout=30)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                routes_data = response.json()
                if isinstance(routes_data, list):
                    print(f"   ✅ Routes available: {len(routes_data)} routes")
                else:
                    print(f"   ✅ Routes response: {routes_data.get('message', 'Success')}")
                return True
            except json.JSONDecodeError:
                print(f"   ✅ Routes file download successful")
                return True
        else:
            print(f"   ❌ Routes endpoint failed: {response.text[:200]}...")
            return False
    except Exception as e:
        print(f"   ❌ Routes endpoint error: {e}")
        return False

def run_comprehensive_test():
    """Run all tests in sequence"""
    print("🚀 API ENDPOINT TESTING - COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    print(f"Testing server: {BASE_URL}")
    print(f"Source ID: {SOURCE_ID}")
    print(f"Parameter: {PARAMETER}")
    print(f"String Param: '{STRING_PARAM}'")
    print(f"Ride Setting: '{RIDESETTING}'")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test 1: Health check
    health_ok = test_health_endpoint()
    if not health_ok:
        print("\n❌ Server is not responding. Please start the FastAPI server first:")
        print("   uvicorn main:app --host 0.0.0.0 --port 5000")
        return
    
    # Test 2: Debug endpoint (if available)
    debug_ok = test_debug_endpoint()
    
    # Test 3: Main assignment endpoint
    assignment_result = test_assignment_endpoint()
    
    # Test 4: Routes endpoint
    routes_ok = test_routes_endpoint()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Health Check: {'PASS' if health_ok else 'FAIL'}")
    print(f"{'✅' if debug_ok else '❌'} Debug Endpoint: {'PASS' if debug_ok else 'FAIL'}")
    print(f"{'✅' if assignment_result else '❌'} Assignment Endpoint: {'PASS' if assignment_result else 'FAIL'}")
    print(f"{'✅' if routes_ok else '❌'} Routes Endpoint: {'PASS' if routes_ok else 'FAIL'}")
    
    if assignment_result:
        print("\n🎉 ALL TESTS PASSED - API is working correctly!")
    else:
        print("\n⚠️ ISSUES DETECTED:")
        if not health_ok:
            print("   - Server is not running or not accessible")
        if not assignment_result:
            print("   - Assignment endpoint is failing (check .env file and API credentials)")
            print("   - This is likely where your cURL timeout issue is occurring")
        if not routes_ok:
            print("   - Routes endpoint has issues")
            
        print("\n💡 DEBUGGING SUGGESTIONS:")
        print("   1. Check your .env file has correct API_URL and API_AUTH_TOKEN")
        print("   2. Verify the external API is accessible from your server")
        print("   3. Check server logs for detailed error messages")
        print("   4. Test the external API directly with curl:")
        print(f"      curl -H 'Authorization: Bearer YOUR_TOKEN' 'YOUR_API_URL/{SOURCE_ID}/{PARAMETER}/{urllib.parse.quote(STRING_PARAM)}/{urllib.parse.quote(RIDESETTING)}'")

def test_different_parameters():
    """Test with different parameter combinations"""
    print("\n🔄 Testing different parameter combinations...")
    
    test_cases = [
        ("UC_logisticllp", 1, "Evening shift", " "),
        ("UC_logisticllp", 1, "Morning shift", " "),
        ("UC_logisticllp", 2, "Evening shift", " "),
        ("UC_unify_dev", 1, "Evening shift", " "),
    ]
    
    for i, (source_id, param, string_param, ridesetting) in enumerate(test_cases, 1):
        print(f"\n   Test Case {i}: {source_id}, {param}, '{string_param}', '{ridesetting}'")
        
        encoded_string_param = urllib.parse.quote(string_param, safe='')
        encoded_ridesetting = urllib.parse.quote(ridesetting, safe='')
        
        url = f"{BASE_URL}/assign-drivers/{source_id}/{param}/{encoded_string_param}/{encoded_ridesetting}"
        
        try:
            start_time = time.time()
            response = requests.post(url, timeout=60)
            elapsed_time = time.time() - start_time
            
            print(f"      Status: {response.status_code}, Time: {elapsed_time:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                routes_count = len(result.get('data', []))
                print(f"      ✅ Success: {routes_count} routes created")
            else:
                print(f"      ❌ Failed: {response.text[:100]}...")
                
        except requests.exceptions.Timeout:
            print(f"      ❌ Timeout after 60 seconds")
        except Exception as e:
            print(f"      ❌ Error: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test API endpoints')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive test suite')
    parser.add_argument('--different-params', action='store_true', help='Test different parameter combinations')
    parser.add_argument('--source-id', default=SOURCE_ID, help='Source ID to test')
    parser.add_argument('--parameter', type=int, default=PARAMETER, help='Parameter to test')
    parser.add_argument('--string-param', default=STRING_PARAM, help='String parameter to test')
    parser.add_argument('--ridesetting', default=RIDESETTING, help='Ridesetting parameter to test')
    
    args = parser.parse_args()
    
    # Update global variables if provided
    SOURCE_ID = args.source_id
    PARAMETER = args.parameter
    STRING_PARAM = args.string_param
    RIDESETTING = args.ridesetting
    
    if args.different_params:
        test_different_parameters()
    elif args.comprehensive:
        run_comprehensive_test()
    else:
        # Default: run comprehensive test
        run_comprehensive_test()
