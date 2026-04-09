import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_build_meas_pattern_std import build_meas_pattern_std

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for build_meas_pattern_std"""
    
    # Data paths provided
    data_paths = ['/home/yjh/pyeit_sandbox/examples/run_code/std_data/standard_data_build_meas_pattern_std.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_build_meas_pattern_std.pkl':
            outer_path = path
    
    # Validate that we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_build_meas_pattern_std.pkl in data_paths")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute the function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    try:
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        print(f"Outer args: {len(outer_args)} positional arguments")
        print(f"Outer kwargs: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"ERROR: Failed to extract args/kwargs from outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the target function
    try:
        print("Executing build_meas_pattern_std with outer args/kwargs...")
        result = build_meas_pattern_std(*outer_args, **outer_kwargs)
        print("Function executed successfully")
    except Exception as e:
        print(f"ERROR: Failed to execute build_meas_pattern_std: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Check if result is callable (operator/closure)
        if not callable(result):
            print("WARNING: Result is not callable, but inner paths exist. This may indicate a test configuration issue.")
            # Fall back to Scenario A behavior
            expected = outer_data.get('output')
        else:
            # Load inner data and execute the operator
            for inner_path in inner_paths:
                try:
                    print(f"Loading inner data from: {inner_path}")
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                except Exception as e:
                    print(f"ERROR: Failed to load inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                try:
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    print(f"Inner args: {len(inner_args)} positional arguments")
                    print(f"Inner kwargs: {list(inner_kwargs.keys())}")
                except Exception as e:
                    print(f"ERROR: Failed to extract args/kwargs from inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                try:
                    print("Executing operator with inner args/kwargs...")
                    result = result(*inner_args, **inner_kwargs)
                    print("Operator executed successfully")
                except Exception as e:
                    print(f"ERROR: Failed to execute operator: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                expected = inner_data.get('output')
                
                # Verify this inner execution
                try:
                    print("Verifying results...")
                    passed, msg = recursive_check(expected, result)
                    if not passed:
                        print(f"TEST FAILED: {msg}")
                        sys.exit(1)
                    print(f"Inner test passed for: {inner_path}")
                except Exception as e:
                    print(f"ERROR: Verification failed with exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
            
            print("TEST PASSED")
            sys.exit(0)
    else:
        # Scenario A: Simple Function
        print("Scenario A detected: Simple function test")
        expected = outer_data.get('output')
    
    # Final verification for Scenario A (or fallback)
    try:
        print("Verifying results...")
        passed, msg = recursive_check(expected, result)
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()