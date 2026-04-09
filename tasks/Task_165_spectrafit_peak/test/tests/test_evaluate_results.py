import sys
import os
import dill
import traceback

# Add the path to allow imports
sys.path.insert(0, '/data/yjh/spectrafit_peak_sandbox_sandbox/run_code')

import numpy as np

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    """Main test function for evaluate_results."""
    
    data_paths = ['/data/yjh/spectrafit_peak_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file 'standard_data_evaluate_results.pkl'")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute the function
    print(f"[TEST] Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"[TEST] Outer data loaded successfully")
    print(f"[TEST] Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"[TEST] Number of args: {len(outer_args)}")
    print(f"[TEST] Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Check if this is a factory/closure pattern (Scenario B) or simple function (Scenario A)
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"[TEST] Detected factory/closure pattern with {len(inner_paths)} inner data file(s)")
        
        # Execute outer function to get operator/closure
        print("[TEST] Phase 1: Executing evaluate_results to get operator...")
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute evaluate_results (outer): {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)
        
        print(f"[TEST] Got callable operator: {type(agent_operator)}")
        
        # Phase 2: Load inner data and execute operator
        for inner_path in inner_paths:
            print(f"[TEST] Loading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output')
            
            print(f"[TEST] Inner data loaded successfully")
            print(f"[TEST] Executing operator with inner args/kwargs...")
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            print("[TEST] Comparing results with expected output...")
            try:
                passed, msg = recursive_check(inner_expected, result)
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            
            print(f"[TEST] Inner test passed for: {os.path.basename(inner_path)}")
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function
        print("[TEST] Detected simple function pattern (no inner data)")
        
        print("[TEST] Executing evaluate_results...")
        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        print(f"[TEST] Function executed successfully")
        print(f"[TEST] Result type: {type(result)}")
        
        # Compare results
        print("[TEST] Comparing results with expected output...")
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()