import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/learned_pd_ct_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Identify outer and inner paths
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
    
    # Phase 1: Load outer data and execute the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"[INFO] Outer args count: {len(outer_args)}")
    print(f"[INFO] Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        result = evaluate_results(*outer_args, **outer_kwargs)
        print(f"[INFO] Function executed successfully")
    except Exception as e:
        print(f"ERROR: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner paths (factory/closure pattern)
    if inner_paths:
        # Scenario B: Factory pattern
        print(f"[INFO] Detected factory pattern with {len(inner_paths)} inner path(s)")
        
        # Check if result is callable
        if not callable(result):
            print(f"ERROR: Expected callable operator but got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output')
            
            try:
                inner_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"[INFO] Inner operator executed successfully")
            except Exception as e:
                print(f"ERROR: Inner operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare inner result
            passed, msg = recursive_check(inner_expected, inner_result)
            if not passed:
                print(f"VERIFICATION FAILED (inner): {msg}")
                sys.exit(1)
            print(f"[INFO] Inner verification passed")
    else:
        # Scenario A: Simple function - compare result directly
        print(f"[INFO] Simple function pattern detected")
        
        passed, msg = recursive_check(expected_output, result)
        if not passed:
            print(f"VERIFICATION FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()