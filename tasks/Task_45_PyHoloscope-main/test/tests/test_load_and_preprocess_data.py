import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_load_and_preprocess_data import load_and_preprocess_data
    from verification_utils import recursive_check
except ImportError as e:
    print(f"[FATAL] Import error: {e}")
    sys.exit(1)

def main():
    data_paths = ['/data/yjh/PyHoloscope-main_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # 1. Identify Data Files
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if 'standard_data_load_and_preprocess_data.pkl' in p:
            outer_path = p
        elif 'parent_function_load_and_preprocess_data' in p:
            inner_paths.append(p)

    if not outer_path:
        print("[SKIP] No standard_data_load_and_preprocess_data.pkl found. Skipping test.")
        sys.exit(0)

    print(f"[INFO] Loading data from {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"[FATAL] Failed to load pickle file: {e}")
        sys.exit(1)

    # 2. Extract Inputs
    args = outer_data.get('args', [])
    kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # 3. Execute Target Function
    print("[INFO] Executing load_and_preprocess_data...")
    try:
        actual_result = load_and_preprocess_data(*args, **kwargs)
    except Exception as e:
        print("[FATAL] Execution failed with error:")
        traceback.print_exc()
        sys.exit(1)

    # 4. Scenario Analysis
    # The function load_and_preprocess_data returns (processed_img, raw_holo, raw_back) directly.
    # It is NOT a factory pattern returning a closure, based on the provided reference code.
    # Therefore, we treat this as Scenario A: Direct comparison.

    # 5. Verification
    print("[INFO] Verifying results...")
    
    # Check if result is a tuple as expected
    if not isinstance(actual_result, tuple):
        print(f"[ERROR] Expected output to be a tuple, got {type(actual_result)}")
        sys.exit(1)

    passed, msg = recursive_check(expected_output, actual_result)
    
    if passed:
        print("[SUCCESS] Test Passed: Output matches expected data.")
        sys.exit(0)
    else:
        print(f"[FAILURE] Test Failed: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()