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
    
    # Define data paths
    data_paths = ['/data/yjh/nmrglue_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Analyze data paths to determine test scenario
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
        print(f"Successfully loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer data function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(outer_args)}")
    print(f"Number of kwargs: {len(outer_kwargs)}")
    
    # Execute forward_operator with outer data
    try:
        result = forward_operator(*outer_args, **outer_kwargs)
        print("Successfully executed forward_operator")
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine test scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("\n=== Scenario B: Factory/Closure Pattern Detected ===")
        
        # Check if result is callable (operator)
        if not callable(result):
            print("WARNING: Result is not callable, but inner paths exist.")
            print("Falling back to Scenario A behavior.")
            # Fall back to Scenario A
            expected = outer_output
        else:
            agent_operator = result
            print(f"Agent operator type: {type(agent_operator)}")
            
            # Process each inner path
            all_passed = True
            for inner_path in inner_paths:
                print(f"\nProcessing inner data: {inner_path}")
                
                try:
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    print(f"Successfully loaded inner data")
                except Exception as e:
                    print(f"ERROR: Failed to load inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_output = inner_data.get('output', None)
                
                print(f"Inner data function name: {inner_data.get('func_name', 'unknown')}")
                
                try:
                    inner_result = agent_operator(*inner_args, **inner_kwargs)
                    print("Successfully executed agent_operator with inner data")
                except Exception as e:
                    print(f"ERROR: Failed to execute agent_operator: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                # Verify inner result
                try:
                    passed, msg = recursive_check(inner_output, inner_result)
                    if not passed:
                        print(f"VERIFICATION FAILED for {inner_path}:")
                        print(msg)
                        all_passed = False
                    else:
                        print(f"Verification passed for {inner_path}")
                except Exception as e:
                    print(f"ERROR: Verification failed with exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
            
            if all_passed:
                print("\n=== TEST PASSED ===")
                sys.exit(0)
            else:
                print("\n=== TEST FAILED ===")
                sys.exit(1)
    
    # Scenario A: Simple Function (no inner paths, or fallback)
    print("\n=== Scenario A: Simple Function ===")
    expected = outer_output
    
    # Verify result
    try:
        passed, msg = recursive_check(expected, result)
        if not passed:
            print("VERIFICATION FAILED:")
            print(msg)
            sys.exit(1)
        else:
            print("\n=== TEST PASSED ===")
            sys.exit(0)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()