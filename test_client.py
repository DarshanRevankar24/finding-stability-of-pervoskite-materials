"""
Test client for Perovskite Stability Predictor API
Run this after starting the FastAPI server with: uvicorn main:app --reload
"""

import requests
import json
from typing import Dict, List

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint."""
    print("\n" + "="*60)
    print("TESTING HEALTH CHECK")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_single_prediction(formula: str, include_details: bool = False):
    """Test single prediction."""
    print("\n" + "="*60)
    print(f"TESTING PREDICTION: {formula}")
    print("="*60)
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json={
            "formula": formula,
            "include_details": include_details
        }
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    
    return result

def test_batch_prediction(formulas: List[str]):
    """Test batch prediction."""
    print("\n" + "="*60)
    print("TESTING BATCH PREDICTION")
    print("="*60)
    
    response = requests.post(
        f"{BASE_URL}/batch_predict",
        params={"include_details": True},
        json=formulas
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Total: {result['total']}, Successful: {result['successful']}")
    
    for pred in result['predictions']:
        if 'error' not in pred:
            print(f"\n{pred['formula']}: {pred['energy_above_hull']} meV/atom - {pred['stability_class']}")
        else:
            print(f"\n{pred['formula']}: ERROR - {pred['error']}")

def test_database_info():
    """Test database info endpoints."""
    print("\n" + "="*60)
    print("TESTING DATABASE INFO")
    print("="*60)
    
    # Get elements
    response = requests.get(f"{BASE_URL}/database/elements")
    elements = response.json()
    print(f"A-site elements: {len(elements['a_site'])}")
    print(f"B-site elements: {len(elements['b_site'])}")
    print(f"Sample A-site: {elements['a_site'][:10]}")
    
    # Get features
    response = requests.get(f"{BASE_URL}/database/features")
    features = response.json()
    print(f"\nTotal features: {features['count']}")
    print(f"Sample features: {list(features['sample_means'].keys())[:5]}")

def run_comprehensive_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# PEROVSKITE STABILITY PREDICTOR - COMPREHENSIVE TESTS")
    print("#"*60)
    
    # Test 1: Health check
    test_health()
    
    # Test 2: Database info
    test_database_info()
    
    # Test 3: Classic perovskites
    print("\n" + "#"*60)
    print("# CLASSIC PEROVSKITES")
    print("#"*60)
    
    test_single_prediction("BaTiO3", include_details=True)
    test_single_prediction("SrTiO3", include_details=True)
    test_single_prediction("CaTiO3", include_details=True)
    
    # Test 4: Halide perovskites
    print("\n" + "#"*60)
    print("# HALIDE PEROVSKITES")
    print("#"*60)
    
    test_single_prediction("CsPbI3", include_details=True)
    
    # Test 5: Mixed compositions
    print("\n" + "#"*60)
    print("# MIXED COMPOSITIONS")
    print("#"*60)
    
    test_single_prediction("Ba0.5Sr0.5TiO3", include_details=True)
    
    # Test 6: Batch prediction
    print("\n" + "#"*60)
    print("# BATCH PREDICTION")
    print("#"*60)
    
    batch_formulas = [
        "BaTiO3",
        "SrTiO3",
        "CaTiO3",
        "PbTiO3",
        "LaAlO3",
        "NdGaO3"
    ]
    test_batch_prediction(batch_formulas)
    
    # Test 7: Error handling
    print("\n" + "#"*60)
    print("# ERROR HANDLING")
    print("#"*60)
    
    print("\nTesting invalid formula...")
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"formula": "123Invalid"}
    )
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    print("\nTesting empty formula...")
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"formula": ""}
    )
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    try:
        run_comprehensive_tests()
        print("\n" + "#"*60)
        print("# ALL TESTS COMPLETED")
        print("#"*60)
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server.")
        print("Make sure the server is running with: uvicorn main:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")