import sys
import os
import dill
import traceback

# Add path for imports
sys.path.insert(0, '/data/yjh/pysyd_astero_sandbox_sandbox/run_code')

import numpy as np

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    """Main test function for evaluate_results."""
    
    data_paths = ['/data/yjh/pysyd_astero_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
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
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    print(f"Loading outer data from: {outer_path}")
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
    
    print(f"Function name from data: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(outer_args)}")
    print(f"Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    print("Executing evaluate_results...")
    try:
        result = evaluate_results(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if result is callable (factory pattern)
    if callable(result) and not isinstance(result, type) and inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("Detected factory/closure pattern - result is callable")
        agent_operator = result
        
        # Load inner data
        inner_path = inner_paths[0]  # Use first inner path
        print(f"Loading inner data from: {inner_path}")
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected_output = inner_data.get('output')
        
        print(f"Inner function name: {inner_data.get('func_name', 'unknown')}")
        print(f"Number of inner args: {len(inner_args)}")
        print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
        
        # Execute the operator
        print("Executing the returned operator...")
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function - result is already the output
        print("Detected simple function pattern")
    
    # Phase 2: Verification
    print("Verifying results...")
    print(f"Result type: {type(result)}")
    print(f"Expected type: {type(expected_output)}")
    
    if isinstance(result, dict):
        print(f"Result keys: {list(result.keys())}")
    if isinstance(expected_output, dict):
        print(f"Expected keys: {list(expected_output.keys())}")
    
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
        
        # Additional debug info
        if isinstance(result, dict) and isinstance(expected_output, dict):
            print("\nDetailed comparison:")
            for key in set(list(result.keys()) + list(expected_output.keys())):
                res_val = result.get(key, '<MISSING>')
                exp_val = expected_output.get(key, '<MISSING>')
                match = "✓" if res_val == exp_val else "✗"
                print(f"  {match} {key}: result={res_val}, expected={exp_val}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()