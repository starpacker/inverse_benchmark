import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/mudic_dic_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']

def main():
    """Main test function for forward_operator"""
    
    # Step 1: Categorize data files
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
    
    # Step 2: Load outer data
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
    outer_output = outer_data.get('output')
    
    print(f"Outer data loaded successfully.")
    print(f"  - Function name: {outer_data.get('func_name')}")
    print(f"  - Number of args: {len(outer_args)}")
    print(f"  - Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Step 3: Execute the forward_operator function
    print("Executing forward_operator with outer data...")
    try:
        result = forward_operator(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("forward_operator executed successfully.")
    
    # Step 4: Determine if this is a factory pattern or simple function
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Detected factory pattern with {len(inner_paths)} inner data file(s)")
        
        # Check if result is callable (an operator)
        if not callable(result):
            print("ERROR: Expected forward_operator to return a callable, but got non-callable result")
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
            expected = inner_data.get('output')
            
            print(f"Inner data loaded successfully.")
            print(f"  - Function name: {inner_data.get('func_name')}")
            print(f"  - Number of args: {len(inner_args)}")
            print(f"  - Kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner data
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
            
            print(f"Inner test passed: {msg}")
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function - result is the output to compare
        print("Detected simple function pattern (no inner data files)")
        
        expected = outer_output
        
        # Verify results
        print("Verifying results...")
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        
        print(f"TEST PASSED: {msg}")
        sys.exit(0)

if __name__ == "__main__":
    main()