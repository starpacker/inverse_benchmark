import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/DENSS_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']

def main():
    """Main test function for forward_operator."""
    
    # Step 1: Analyze data paths to determine test strategy
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_path = path
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data path: {inner_path}")
    
    # Step 2: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("Successfully loaded outer data")
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
    
    # Step 3: Execute forward_operator with outer data
    try:
        result = forward_operator(*outer_args, **outer_kwargs)
        print("Successfully executed forward_operator")
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Determine which scenario we're in
    if inner_path is not None and os.path.exists(inner_path):
        # Scenario B: Factory/Closure pattern
        print("Scenario B: Factory/Closure pattern detected")
        
        # Check if result is callable (operator)
        if not callable(result):
            print("ERROR: Expected callable operator from forward_operator, but got non-callable")
            sys.exit(1)
        
        agent_operator = result
        
        # Load inner data
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print("Successfully loaded inner data")
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output', None)
        
        # Execute the operator with inner data
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Successfully executed agent_operator with inner data")
        except Exception as e:
            print(f"ERROR: Failed to execute agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Scenario A: Simple function pattern detected")
        expected = outer_output
    
    # Step 5: Compare results
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Comparison failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == '__main__':
    main()