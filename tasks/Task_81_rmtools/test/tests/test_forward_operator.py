import sys
import os
import dill
import traceback
import numpy as np

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check


def main():
    """Main test function for forward_operator."""
    
    # Data paths provided
    data_paths = ['/data/yjh/rmtools_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"Loaded outer data for function: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(outer_args)}")
    print(f"Number of kwargs: {len(outer_kwargs)}")
    
    # Execute the forward_operator function
    try:
        result = forward_operator(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory/closure pattern (Scenario B)
    if inner_paths:
        # Scenario B: Factory pattern - result should be callable
        if not callable(result):
            print("ERROR: Expected callable result for factory pattern, but got non-callable")
            sys.exit(1)
        
        agent_operator = result
        print("Factory pattern detected - agent_operator is callable")
        
        # Process each inner path
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            print(f"Loaded inner data for function: {inner_data.get('func_name', 'unknown')}")
            
            # Execute the agent operator with inner args
            try:
                inner_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner data")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify inner result
            try:
                passed, msg = recursive_check(inner_expected, inner_result)
            except Exception as e:
                print(f"ERROR: Verification failed with exception")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner execution")
                print(f"Failure message: {msg}")
                sys.exit(1)
            
            print(f"Inner execution verification passed")
    else:
        # Scenario A: Simple function - compare result directly
        print("Simple function pattern detected")
        
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print("TEST FAILED")
            print(f"Failure message: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()