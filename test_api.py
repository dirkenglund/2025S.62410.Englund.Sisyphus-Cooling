#!/usr/bin/env python3
"""
Comprehensive test script for the Sisyphus Cooling Simulation API
"""
import json
import sys
import requests

# Base URL for the API
BASE_URL = "http://localhost:4000"

def call_endpoint(endpoint, params=None, expected_status=200, test_name=""):
    """Test an API endpoint and validate its response"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if params:
            response = requests.get(url, params=params)
        else:
            response = requests.get(url)
            
        # Validate status code
        if response.status_code != expected_status:
            print(f"❌ {test_name} failed: Expected status {expected_status}, got {response.status_code}")
            return False
            
        # For successful responses, validate JSON format
        if expected_status == 200:
            try:
                data = response.json()
                print(f"✅ {test_name} passed: {endpoint} returned valid JSON")
                return data
            except json.JSONDecodeError:
                print(f"❌ {test_name} failed: {endpoint} did not return valid JSON")
                return False
        else:
            print(f"✅ {test_name} passed with expected status {expected_status}")
            return True
    except requests.RequestException as e:
        print(f"❌ {test_name} failed: {str(e)}")
        return False

def test_main_page():
    """Test the main HTML page"""
    url = BASE_URL
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("✅ Main page test passed: HTML returned successfully")
            return True
        else:
            print(f"❌ Main page test failed: Status code {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Main page test failed: {str(e)}")
        return False

def test_constants_api():
    """Test the constants API endpoint"""
    data = call_endpoint("/api/constants", test_name="Constants API")
    if data:
        # Validate expected constants are present
        expected_keys = [
            "RB87_MASS", "WAVELENGTH_D2", "WAVENUMBER_D2", "GAMMA_D2", 
            "RECOIL_ENERGY", "RECOIL_TEMPERATURE", "DOPPLER_TEMPERATURE"
        ]
        missing_keys = [key for key in expected_keys if key not in data]
        if missing_keys:
            print(f"❌ Constants API validation failed: Missing keys {missing_keys}")
            return False
        else:
            print("✅ Constants API validation passed: All expected keys present")
            return True
    return False

def test_world_model_api():
    """Test the world model API endpoint"""
    data = call_endpoint("/api/world-model", test_name="World Model API")
    if data:
        # Validate expected structure
        if "nodes" not in data or "links" not in data:
            print("❌ World Model API validation failed: Missing 'nodes' or 'links' keys")
            return False
        else:
            print(f"✅ World Model API validation passed: Found {len(data['nodes'])} nodes and {len(data['links'])} links")
            return True
    return False

def test_entity_info_api():
    """Test the entity info API endpoint"""
    data = call_endpoint("/api/entity-info/RECOIL_TEMPERATURE", test_name="Entity Info API")
    if data:
        # Validate expected structure
        expected_keys = ["name", "description", "value", "related"]
        missing_keys = [key for key in expected_keys if key not in data]
        if missing_keys:
            print(f"❌ Entity Info API validation failed: Missing keys {missing_keys}")
            return False
        else:
            print("✅ Entity Info API validation passed")
            return True
    return False

def test_simulation_api():
    """Test the simulation API endpoint"""
    params = {
        "initialTemp": 100,
        "detuning": -3,
        "rabiFreq": 1
    }
    data = call_endpoint("/api/simulate-cooling", params=params, test_name="Simulation API")
    if data:
        # Validate expected structure and data validity
        expected_keys = ["times", "sisyphus_temps", "doppler_temps"]
        missing_keys = [key for key in expected_keys if key not in data]
        
        if missing_keys:
            print(f"❌ Simulation API validation failed: Missing keys {missing_keys}")
            return False
        
        # Check that temperatures are decreasing (cooling)
        sisyphus_temps = data["sisyphus_temps"]
        doppler_temps = data["doppler_temps"]
        
        if len(sisyphus_temps) < 2 or len(doppler_temps) < 2:
            print("❌ Simulation API validation failed: Not enough temperature data points")
            return False
        
        sisyphus_decreasing = all(t1 >= t2 for t1, t2 in zip(sisyphus_temps[:-1], sisyphus_temps[1:]))
        doppler_decreasing = all(t1 >= t2 for t1, t2 in zip(doppler_temps[:-1], doppler_temps[1:]))
        
        if not sisyphus_decreasing:
            print("❌ Simulation API validation failed: Sisyphus temperatures are not decreasing")
            return False
        
        if not doppler_decreasing:
            print("❌ Simulation API validation failed: Doppler temperatures are not decreasing")
            return False
        
        # Verify that Sisyphus cooling reaches lower temperatures than Doppler cooling
        if sisyphus_temps[-1] >= doppler_temps[-1]:
            print("❌ Simulation API validation failed: Sisyphus cooling should reach lower final temperature than Doppler cooling")
            return False
        
        print(f"✅ Simulation API validation passed: Initial temp: {sisyphus_temps[0]:.2f}μK, Final temp: {sisyphus_temps[-1]:.2f}μK")
        return True
    
    return False

def run_all_tests():
    """Run all tests and report results"""
    print("🔍 Running comprehensive tests for Sisyphus Cooling Simulation API\n")
    
    tests = [
        ("Main Page", test_main_page),
        ("Constants API", test_constants_api),
        ("World Model API", test_world_model_api),
        ("Entity Info API", test_entity_info_api),
        ("Simulation API", test_simulation_api)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 Testing {name}...")
        result = test_func()
        results.append((name, result))
    
    # Print summary
    print("\n📊 Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
    
    print(f"\n{passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
