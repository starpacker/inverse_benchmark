import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_sirt_reconstruct import sirt_reconstruct
from verification_utils import recursive_check


def main():
    """
    Test script for sirt_reconstruct function.
    """
    # Data paths provided
    data_paths = ['/data/yjh/TIGRE_sandbox_sandbox/run_code/std_data/standard_data_sirt_reconstruct.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_sirt_reconstruct.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file 'standard_data_sirt_reconstruct.pkl'")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute sirt_reconstruct
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
    
    print(f"Outer function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of outer args: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute sirt_reconstruct with outer data
    print("\nExecuting sirt_reconstruct with outer data...")
    try:
        result = sirt_reconstruct(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute sirt_reconstruct: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is a factory pattern or simple function
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"\nDetected factory pattern with {len(inner_paths)} inner data file(s)")
        
        # Check if result is callable (operator)
        if not callable(result):
            print("WARNING: Result is not callable, treating as simple function output")
            # Fall back to Scenario A
            expected = outer_output
        else:
            print("Result is callable, executing with inner data...")
            agent_operator = result
            
            # Load and execute with inner data
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
                
                print(f"Inner function name: {inner_data.get('func_name', 'unknown')}")
                print(f"Number of inner args: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner data
                print("\nExecuting operator with inner data...")
                try:
                    result = agent_operator(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"ERROR: Failed to execute operator: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                # Compare results
                print("\nComparing results...")
                try:
                    passed, msg = recursive_check(expected, result)
                except Exception as e:
                    print(f"ERROR: Verification failed with exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed: {msg}")
            
            print("\nTEST PASSED")
            sys.exit(0)
    
    # Scenario A: Simple function (no inner paths or non-callable result)
    print("\nScenario A: Simple function output comparison")
    expected = outer_output
    
    # Compare results
    print("\nComparing results...")
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
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