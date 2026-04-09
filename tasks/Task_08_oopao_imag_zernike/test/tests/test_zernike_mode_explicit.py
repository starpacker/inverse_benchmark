import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_zernike_mode_explicit import zernike_mode_explicit
from verification_utils import recursive_check


def main():
    """
    Test script for zernike_mode_explicit function.
    
    Analyzes data paths to determine if this is:
    - Scenario A: Simple function (only standard_data_zernike_mode_explicit.pkl)
    - Scenario B: Factory/Closure pattern (outer + inner data files)
    """
    
    # Data paths provided
    data_paths = ['/home/yjh/oopao_zernike_sandbox/run_code/std_data/standard_data_zernike_mode_explicit.pkl']
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if this is an inner data file (contains 'parent_function')
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        # Check if this is the outer data file (exact match pattern)
        elif basename == 'standard_data_zernike_mode_explicit.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_zernike_mode_explicit.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("Successfully loaded outer data file.")
    except Exception as e:
        print(f"ERROR: Failed to load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    try:
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"ERROR: Failed to extract args/kwargs from outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the target function with outer args/kwargs
    try:
        print("Executing zernike_mode_explicit with outer args/kwargs...")
        agent_result = zernike_mode_explicit(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(agent_result)}")
    except Exception as e:
        print(f"ERROR: Failed to execute zernike_mode_explicit: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify results
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\n--- Scenario B: Factory/Closure Pattern Detected ---")
        
        # Verify that agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from zernike_mode_explicit, got {type(agent_result)}")
            sys.exit(1)
        
        agent_operator = agent_result
        
        # Process each inner data file
        for inner_path in inner_paths:
            print(f"\nProcessing inner data: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print("Successfully loaded inner data file.")
            except Exception as e:
                print(f"ERROR: Failed to load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output', None)
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            except Exception as e:
                print(f"ERROR: Failed to extract args/kwargs from inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Execute the operator with inner args/kwargs
            try:
                print("Executing agent_operator with inner args/kwargs...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Operator executed successfully. Result type: {type(actual_result)}")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                print("Comparing actual result with expected output...")
                passed, msg = recursive_check(expected_output, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Verification passed for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"ERROR: Failed during result comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function
        print("\n--- Scenario A: Simple Function Pattern Detected ---")
        
        # The result from Phase 1 IS the result to compare
        actual_result = agent_result
        expected_output = outer_output
        
        # Compare results
        try:
            print("Comparing actual result with expected output...")
            passed, msg = recursive_check(expected_output, actual_result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("Verification passed.")
                print("\nTEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR: Failed during result comparison: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()