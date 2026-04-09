import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_build_exc_pattern_std import build_exc_pattern_std

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for build_exc_pattern_std"""
    
    # Data paths provided
    data_paths = ['/home/yjh/pyeit_sandbox/examples/run_code/std_data/standard_data_build_exc_pattern_std.pkl']
    
    # Separate outer and inner paths based on naming convention
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_build_exc_pattern_std.pkl':
            outer_path = path
    
    # Validate that we have the outer path
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_build_exc_pattern_std.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    try:
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
    except Exception as e:
        print(f"ERROR: Failed to extract args/kwargs from outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the target function
    try:
        print("Executing build_exc_pattern_std with outer args/kwargs...")
        agent_operator = build_exc_pattern_std(*outer_args, **outer_kwargs)
        print(f"Function returned type: {type(agent_operator)}")
    except Exception as e:
        print(f"ERROR: Failed to execute build_exc_pattern_std: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator but got {type(agent_operator)}")
            sys.exit(1)
        
        # Process each inner data file
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner args: {inner_args}")
                print(f"Inner kwargs: {inner_kwargs}")
            except Exception as e:
                print(f"ERROR: Failed to extract args/kwargs from inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                print("Executing agent_operator with inner args/kwargs...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
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
    else:
        # Scenario A: Simple Function
        print("Scenario A detected: Simple function test")
        
        result = agent_operator
        expected = expected_output
        
        # Verify results
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
    
    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()