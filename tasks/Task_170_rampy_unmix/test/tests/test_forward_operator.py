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
    data_paths = ['/data/yjh/rampy_unmix_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Identify outer path (main function data) and inner path (closure/operator data)
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if os.path.exists(path):
            basename = os.path.basename(path)
            if 'parent_function' in basename or 'parent_' in basename:
                inner_paths.append(path)
            elif basename == 'standard_data_forward_operator.pkl':
                outer_path = path
    
    # Check if we have the required data file
    if outer_path is None:
        print("ERROR: Could not find standard_data_forward_operator.pkl")
        sys.exit(1)
    
    print(f"Found outer path: {outer_path}")
    print(f"Found inner paths: {inner_paths}")
    
    try:
        # Phase 1: Load outer data and execute function
        print("\n=== Phase 1: Loading outer data ===")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Function name: {outer_data.get('func_name')}")
        print(f"Number of args: {len(outer_args)}")
        print(f"Kwargs keys: {list(outer_kwargs.keys())}")
        
        # Execute the function
        print("\n=== Executing forward_operator ===")
        result = forward_operator(*outer_args, **outer_kwargs)
        
        # Determine expected output and actual result based on scenario
        if inner_paths:
            # Scenario B: Factory/Closure pattern
            print("\n=== Scenario B: Factory/Closure Pattern ===")
            
            # Check if result is callable (an operator)
            if callable(result):
                print("Result is callable (operator created)")
                agent_operator = result
                
                # Load inner data and execute operator
                inner_path = inner_paths[0]
                print(f"Loading inner data from: {inner_path}")
                
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner function name: {inner_data.get('func_name')}")
                print("Executing operator with inner data...")
                
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            else:
                # Result is not callable, use outer output for comparison
                print("Result is not callable, using outer data for comparison")
                actual_result = result
                expected = outer_output
        else:
            # Scenario A: Simple function
            print("\n=== Scenario A: Simple Function ===")
            actual_result = result
            expected = outer_output
        
        # Phase 2: Verification
        print("\n=== Phase 2: Verification ===")
        
        if actual_result is not None:
            if hasattr(actual_result, 'shape'):
                print(f"Actual result shape: {actual_result.shape}")
            if hasattr(actual_result, 'dtype'):
                print(f"Actual result dtype: {actual_result.dtype}")
        
        if expected is not None:
            if hasattr(expected, 'shape'):
                print(f"Expected result shape: {expected.shape}")
            if hasattr(expected, 'dtype'):
                print(f"Expected result dtype: {expected.dtype}")
        
        # Use recursive_check for comparison
        passed, msg = recursive_check(expected, actual_result)
        
        if passed:
            print("\n" + "="*50)
            print("TEST PASSED")
            print("="*50)
            sys.exit(0)
        else:
            print("\n" + "="*50)
            print("TEST FAILED")
            print(f"Verification message: {msg}")
            print("="*50)
            sys.exit(1)
            
    except Exception as e:
        print("\n" + "="*50)
        print("TEST FAILED WITH EXCEPTION")
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("="*50)
        sys.exit(1)

if __name__ == "__main__":
    main()