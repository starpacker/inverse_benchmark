import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check


def main():
    """Main test function for forward_operator."""
    
    # Data paths provided
    data_paths = ['/data/yjh/bisip_sip_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
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
    
    # Phase 1: Load outer data and execute forward_operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args: {len(outer_args)} positional arguments")
    print(f"Outer kwargs: {list(outer_kwargs.keys())}")
    
    # Execute the forward_operator function
    try:
        result = forward_operator(*outer_args, **outer_kwargs)
        print("Successfully executed forward_operator")
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is Scenario B (factory pattern) or Scenario A (simple function)
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # The result should be callable (an operator)
        if not callable(result):
            print(f"ERROR: Expected callable operator from forward_operator, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner path
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner args: {len(inner_args)} positional arguments")
            print(f"Inner kwargs: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner data
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent_operator with inner data")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed for: {inner_path}")
            except Exception as e:
                print(f"ERROR: Comparison failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function call")
        
        expected = outer_output
        actual_result = result
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, actual_result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Comparison failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()