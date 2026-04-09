import sys
import os
import dill
import numpy as np
import traceback
import odl

# Ensure the target module is in the path
sys.path.append(os.path.dirname(__file__))

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def test_load_and_preprocess_data():
    """
    Test script for load_and_preprocess_data.
    Handles stochastic output (noisy data) by relaxing checks on the first return value,
    while strictly checking deterministic outputs (ground truth and geometry).
    """
    
    # 1. Define Data Paths
    data_paths = ['/data/yjh/spectral_ct_examples-master_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    outer_path = None
    inner_path = None

    # 2. Identify Files
    for p in data_paths:
        if 'parent_function' in p:
            inner_path = p
        else:
            outer_path = p

    if not outer_path:
        print("Error: No outer data file found (standard_data_load_and_preprocess_data.pkl).")
        sys.exit(1)

    try:
        # 3. Load Outer Data
        print(f"Loading Outer Data: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # 4. Execute Function
        print("Executing load_and_preprocess_data with outer arguments...")
        actual_result = load_and_preprocess_data(*outer_args, **outer_kwargs)

        # 5. Handle Scenarios
        if inner_path:
            # Scenario B: Factory Pattern (Not applicable for this specific function signature, but kept for robustness)
            print(f"Loading Inner Data: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_result = inner_data['output']
            
            print("Executing Inner Function (Operator)...")
            actual_result = actual_result(*inner_args, **inner_kwargs)
        else:
            # Scenario A: Standard Function
            print("No inner data found. Treating as Standard Function (Scenario A).")
            expected_result = outer_data['output']

        # 6. Verification
        print("Verifying results...")

        # The function returns tuple: (data_noisy, gt_images, geometry)
        # data_noisy [0] is STOCHASTIC (contains random noise). We cannot expect exact match.
        # gt_images [1] and geometry [2] should be DETERMINISTIC.

        if isinstance(expected_result, tuple) and len(expected_result) == 3:
            
            # Check 1: Noisy Data (Relaxed Check)
            # We check shape and type, but ignore values because of random noise
            actual_noisy = actual_result[0]
            expected_noisy = expected_result[0]
            
            if actual_noisy.shape != expected_noisy.shape:
                print(f"TEST FAILED: Shape mismatch for data_noisy. Expected {expected_noisy.shape}, got {actual_noisy.shape}")
                sys.exit(1)
            print("Check [0] (Noisy Data): Shape match passed. Values skipped due to stochastic noise.")

            # Check 2: Ground Truth Images (Strict Check)
            passed_gt, msg_gt = recursive_check(expected_result[1], actual_result[1])
            if not passed_gt:
                print(f"TEST FAILED: Deterministic mismatch at output[1] (GT Images): {msg_gt}")
                sys.exit(1)
            print("Check [1] (GT Images): Passed.")

            # Check 3: Geometry (Strict Check)
            # ODL geometries might need custom comparison or recursive_check handles them if they are standard objects
            # If recursive_check fails on ODL objects due to strict equality, we might need to compare their string repr or attributes
            # recursive_check usually handles deep comparison well.
            passed_geo, msg_geo = recursive_check(expected_result[2], actual_result[2])
            if not passed_geo:
                 # Fallback for ODL objects if strict equality fails but they represent the same space
                str_exp = str(expected_result[2])
                str_act = str(actual_result[2])
                if str_exp == str_act:
                    print("Check [2] (Geometry): Passed via string representation.")
                else:
                    print(f"TEST FAILED: Deterministic mismatch at output[2] (Geometry): {msg_geo}")
                    sys.exit(1)
            else:
                print("Check [2] (Geometry): Passed.")

        else:
            # Fallback for unexpected return structure
            passed, msg = recursive_check(expected_result, actual_result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    except Exception as e:
        print(f"Execution Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_load_and_preprocess_data()