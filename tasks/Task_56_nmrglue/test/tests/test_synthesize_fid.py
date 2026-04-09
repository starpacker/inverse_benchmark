import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_synthesize_fid import synthesize_fid
from verification_utils import recursive_check

def main():
    """
    Unit test for synthesize_fid function.
    
    This test analyzes the data files to determine if we're dealing with:
    - Scenario A: Simple function (only standard_data_synthesize_fid.pkl exists)
    - Scenario B: Factory/Closure pattern (both outer and inner data files exist)
    """
    
    # Data paths provided
    data_paths = ['/data/yjh/nmrglue_sandbox_sandbox/run_code/std_data/standard_data_synthesize_fid.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"WARNING: Data file not found: {path}")
            continue
            
        basename = os.path.basename(path)
        
        # Check if this is an inner data file (contains 'parent_function')
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        # Check if this is the outer data file (exact match pattern)
        elif basename == 'standard_data_synthesize_fid.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_synthesize_fid.pkl)")
        sys.exit(1)
    
    print(f"Outer data file: {outer_path}")
    print(f"Inner data files: {inner_paths}")
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data for function: {outer_data.get('func_name', 'unknown')}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function with outer data
    try:
        agent_result = synthesize_fid(*outer_args, **outer_kwargs)
        print("Successfully executed synthesize_fid with outer data")
    except Exception as e:
        print(f"ERROR: Failed to execute synthesize_fid: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("\n=== Scenario B: Factory/Closure Pattern ===")
        
        # Verify the result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable operator, got {type(agent_result)}")
            sys.exit(1)
        
        print("Agent result is callable, proceeding with inner data execution")
        
        # Process each inner data file
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output', None)
            
            # Execute the operator with inner data
            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
                print("Successfully executed agent operator with inner data")
            except Exception as e:
                print(f"ERROR: Failed to execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = recursive_check(expected_output, actual_result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner data verification passed: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\n=== Scenario A: Simple Function ===")
        
        expected_output = outer_output
        actual_result = agent_result
        
        # Verify results
        try:
            passed, msg = recursive_check(expected_output, actual_result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            print("Outer data verification passed")
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()