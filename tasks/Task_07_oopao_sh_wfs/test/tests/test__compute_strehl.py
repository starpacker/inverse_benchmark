import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent__compute_strehl import _compute_strehl

# Import verification utility
from verification_utils import recursive_check


def main():
    """
    Main test function for _compute_strehl.
    """
    # Define data paths
    data_paths = ['/home/yjh/oopao_sh_sandbox/run_code/std_data/standard_data__compute_strehl.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        # Check if this is an inner data file (contains 'parent_function' or 'parent_')
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data__compute_strehl.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data__compute_strehl.pkl)")
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
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer data function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    print("Executing _compute_strehl with outer data...")
    try:
        result = _compute_strehl(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute _compute_strehl: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario based on inner paths
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"Detected Scenario B: Factory/Closure pattern with {len(inner_paths)} inner data file(s)")
        
        # Verify the result is callable (it's an operator/closure)
        if not callable(result):
            print(f"ERROR: Expected callable operator from _compute_strehl, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner data file
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
            
            print(f"Inner data function name: {inner_data.get('func_name', 'unknown')}")
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner args
            print("Executing agent_operator with inner data...")
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            print("Verifying results...")
            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed for: {inner_path}")
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple function")
        
        expected = outer_output
        actual_result = result
        
        # Verify results
        print("Verifying results...")
        try:
            passed, msg = recursive_check(expected, actual_result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == "__main__":
    main()