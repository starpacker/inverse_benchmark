import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/saxnerf_ct_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner paths
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
    
    # Phase 1: Load outer data and execute function
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
    outer_output = outer_data.get('output', None)
    
    print(f"Outer data loaded successfully.")
    print(f"  Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"  Args count: {len(outer_args)}")
    print(f"  Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    print("\nExecuting evaluate_results with outer data...")
    try:
        result = evaluate_results(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Function executed successfully.")
    print(f"Result type: {type(result)}")
    
    # Check if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Verify result is callable
        if not callable(result):
            print(f"ERROR: Expected callable result for factory pattern, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner path
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner data loaded successfully.")
            print(f"  Function name: {inner_data.get('func_name', 'unknown')}")
            print(f"  Args count: {len(inner_args)}")
            print(f"  Kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner data
            print("\nExecuting agent_operator with inner data...")
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            print(f"Operator executed successfully.")
            print(f"Actual result type: {type(actual_result)}")
            print(f"Expected result type: {type(expected)}")
            
            # Compare results
            print("\nComparing results...")
            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"ERROR: Failed during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed: {msg}")
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function execution")
        
        expected = outer_output
        
        print(f"Result type: {type(result)}")
        print(f"Expected type: {type(expected)}")
        
        # Compare results
        print("\nComparing results...")
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: Failed during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print(f"TEST PASSED: {msg}")
            sys.exit(0)


if __name__ == "__main__":
    main()