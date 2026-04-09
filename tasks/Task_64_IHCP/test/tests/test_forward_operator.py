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
    data_paths = ['/data/yjh/IHCP_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    # Verify outer path exists
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data file: {outer_path}")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Loaded outer data from: {outer_path}")
    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
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
    
    print("Successfully executed forward_operator")
    
    # Check if this is Scenario B (factory/closure pattern)
    # Filter inner_paths to only existing files
    existing_inner_paths = [p for p in inner_paths if os.path.exists(p)]
    
    if existing_inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Detected Scenario B: Found {len(existing_inner_paths)} inner data file(s)")
        
        # Check if the result is callable (operator/closure)
        if callable(result):
            agent_operator = result
            
            # Process each inner data file
            for inner_path in existing_inner_paths:
                print(f"\nProcessing inner data: {inner_path}")
                
                try:
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                except Exception as e:
                    print(f"ERROR: Failed to load inner data file: {inner_path}")
                    print(f"Exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                # Execute the operator with inner data
                try:
                    actual_result = agent_operator(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"ERROR: Failed to execute agent_operator with inner data")
                    print(f"Exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                # Compare results
                try:
                    passed, msg = recursive_check(expected, actual_result)
                except Exception as e:
                    print(f"ERROR: Failed during recursive_check")
                    print(f"Exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                if not passed:
                    print(f"TEST FAILED for inner data: {inner_path}")
                    print(f"Failure message: {msg}")
                    sys.exit(1)
                
                print(f"Verification passed for: {inner_path}")
            
            print("\nTEST PASSED")
            sys.exit(0)
        else:
            # Result is not callable, treat as Scenario A
            print("Result is not callable, falling back to Scenario A")
    
    # Scenario A: Simple Function (or fallback)
    print("Using Scenario A: Direct comparison with outer data output")
    
    expected = outer_output
    
    # Compare results
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Failed during recursive_check")
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