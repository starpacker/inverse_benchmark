import sys
import os
import dill
import traceback
import numpy as np

# Add path for imports
sys.path.insert(0, '/data/yjh/pymiescat_sandbox_sandbox/run_code')

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/pymiescat_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    print(f"[INFO] Outer data path: {outer_path}")
    print(f"[INFO] Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and execute function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data successfully")
        print(f"[INFO] Outer data keys: {outer_data.keys()}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"[INFO] Number of outer args: {len(outer_args)}")
    print(f"[INFO] Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        result = evaluate_results(*outer_args, **outer_kwargs)
        print(f"[INFO] Function executed successfully")
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if result is callable (factory pattern - Scenario B)
    if callable(result) and len(inner_paths) > 0:
        print("[INFO] Detected factory/closure pattern (Scenario B)")
        
        # Load inner data and execute the returned callable
        inner_path = inner_paths[0]
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"[INFO] Loaded inner data successfully")
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected_output = inner_data.get('output')
        
        try:
            result = result(*inner_args, **inner_kwargs)
            print(f"[INFO] Inner function executed successfully")
        except Exception as e:
            print(f"ERROR: Failed to execute inner function: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("[INFO] Detected simple function pattern (Scenario A)")
    
    # Phase 2: Verification
    try:
        passed, msg = recursive_check(expected_output, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        print(f"[DEBUG] Expected type: {type(expected_output)}")
        print(f"[DEBUG] Result type: {type(result)}")
        if isinstance(expected_output, dict) and isinstance(result, dict):
            print(f"[DEBUG] Expected keys: {expected_output.keys()}")
            print(f"[DEBUG] Result keys: {result.keys()}")
            for key in expected_output:
                if key in result:
                    exp_val = expected_output[key]
                    res_val = result[key]
                    if exp_val != res_val:
                        print(f"[DEBUG] Mismatch at key '{key}': expected={exp_val}, got={res_val}")
        sys.exit(1)

if __name__ == "__main__":
    main()