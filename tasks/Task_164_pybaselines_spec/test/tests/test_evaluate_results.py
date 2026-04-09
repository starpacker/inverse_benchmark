import sys
import os
import dill
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """
    Test script for evaluate_results function.
    
    This function is a simple function (Scenario A) - it takes inputs
    and returns metrics. No factory/closure pattern detected.
    """
    
    data_paths = ['/data/yjh/pybaselines_spec_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Filter paths to identify outer (main function) and inner (closure) data
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"[ERROR] Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if this is an inner/closure data file (contains 'parent_function' or 'parent_')
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("[ERROR] Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    print(f"[INFO] Outer data path: {outer_path}")
    print(f"[INFO] Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and execute function
    try:
        print("[INFO] Loading outer data...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"[INFO] Outer function name: {outer_data.get('func_name', 'unknown')}")
        print(f"[INFO] Outer args count: {len(outer_args)}")
        print(f"[INFO] Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("[INFO] Executing evaluate_results function...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print(f"[INFO] Function executed successfully")
        print(f"[INFO] Result type: {type(result)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory pattern (result is callable and we have inner paths)
    if len(inner_paths) > 0 and callable(result):
        # Scenario B: Factory/Closure Pattern
        print("[INFO] Detected factory/closure pattern - result is callable")
        
        for inner_path in inner_paths:
            try:
                print(f"[INFO] Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_output = inner_data.get('output', None)
                
                print(f"[INFO] Inner function name: {inner_data.get('func_name', 'unknown')}")
                print(f"[INFO] Inner args count: {len(inner_args)}")
                print(f"[INFO] Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the closure/operator
                print("[INFO] Executing closure/operator...")
                actual_result = result(*inner_args, **inner_kwargs)
                
                # Compare with expected output
                print("[INFO] Comparing results...")
                passed, msg = recursive_check(inner_output, actual_result)
                
                if not passed:
                    print(f"[FAIL] Verification failed: {msg}")
                    sys.exit(1)
                else:
                    print(f"[PASS] Inner execution verified successfully")
                    
            except Exception as e:
                print(f"[ERROR] Failed during inner execution: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - compare direct output
        print("[INFO] Simple function pattern - comparing direct output")
        
        try:
            print("[INFO] Comparing results...")
            passed, msg = recursive_check(outer_output, result)
            
            if not passed:
                print(f"[FAIL] Verification failed: {msg}")
                sys.exit(1)
            else:
                print(f"[PASS] Output verified successfully")
                
        except Exception as e:
            print(f"[ERROR] Failed during verification: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()