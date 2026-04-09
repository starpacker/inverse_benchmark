import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for forward_operator."""
    
    # Data paths provided
    data_paths = ['/home/yjh/lenstronomy_host_decomp_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    # Scenario A: Simple function - only outer path exists
    if outer_path is None:
        print("ERROR: No outer data file found (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute forward_operator
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the forward_operator function
    try:
        print("Executing forward_operator with outer args/kwargs...")
        result = forward_operator(*outer_args, **outer_kwargs)
        print(f"forward_operator returned type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine test scenario based on inner paths
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # The result from forward_operator should be callable
        if not callable(result):
            print(f"ERROR: Expected forward_operator to return a callable, got {type(result)}")
            # In some cases, forward_operator might directly return the result
            # Let's check if we should compare directly
            print("Attempting direct comparison instead...")
            expected = outer_output
            passed, msg = recursive_check(expected, result)
            if passed:
                print("TEST PASSED (direct comparison)")
                sys.exit(0)
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        
        agent_operator = result
        
        # Process each inner data file
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_output = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                print("Executing agent_operator with inner args/kwargs...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                print("Comparing results...")
                passed, msg = recursive_check(inner_output, actual_result)
                
                if not passed:
                    print(f"TEST FAILED for {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
                else:
                    print(f"TEST PASSED for {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR processing inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nALL TESTS PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple function - compare result directly with outer output
        print("\nScenario A detected: Simple function comparison")
        
        expected = outer_output
        
        print("Comparing results...")
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)


if __name__ == '__main__':
    main()