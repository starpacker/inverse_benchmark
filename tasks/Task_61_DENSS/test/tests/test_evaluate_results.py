import sys
import os
import dill
import numpy as np
import traceback

# Add the path to find the module
sys.path.insert(0, '/data/yjh/DENSS_sandbox_sandbox/run_code')

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    """Main test function for evaluate_results."""
    
    data_paths = ['/data/yjh/DENSS_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer (main function) and inner (closure/operator) data files
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
    
    # Phase 1: Load outer data and run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("[INFO] Successfully loaded outer data file")
    except Exception as e:
        print(f"ERROR: Failed to load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"[INFO] Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"[INFO] Number of args: {len(outer_args)}")
    print(f"[INFO] Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        print("[INFO] Executing evaluate_results...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print("[INFO] Function executed successfully")
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory/closure pattern (Scenario B) or simple function (Scenario A)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("[INFO] Detected factory/closure pattern (Scenario B)")
        
        # Verify result is callable
        if not callable(result):
            print(f"ERROR: Expected callable operator, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner data file
        for inner_path in inner_paths:
            print(f"[INFO] Processing inner data: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print("[INFO] Successfully loaded inner data file")
            except Exception as e:
                print(f"ERROR: Failed to load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output')
            
            try:
                print("[INFO] Executing operator with inner args...")
                inner_result = agent_operator(*inner_args, **inner_kwargs)
                print("[INFO] Operator executed successfully")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify inner result
            try:
                passed, msg = recursive_check(inner_expected, inner_result)
                if not passed:
                    print(f"VERIFICATION FAILED for inner execution: {msg}")
                    sys.exit(1)
                print(f"[INFO] Inner execution verification passed")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function
        print("[INFO] Detected simple function pattern (Scenario A)")
        
        # Verify result against expected output
        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"VERIFICATION FAILED: {msg}")
                sys.exit(1)
            print("TEST PASSED")
            sys.exit(0)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()