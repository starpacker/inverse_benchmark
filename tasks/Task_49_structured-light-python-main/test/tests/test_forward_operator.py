import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_forward_operator import forward_operator
    from verification_utils import recursive_check
except ImportError:
    print("Error: Could not import 'forward_operator' from 'agent_forward_operator' or 'recursive_check' from 'verification_utils'.")
    print("Make sure these files are in the same directory.")
    sys.exit(1)

def main():
    data_paths = ['/data/yjh/structured-light-python-main_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Identify Data Files
    outer_path = None
    inner_path = None

    for path in data_paths:
        if 'standard_data_forward_operator.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_forward_operator' in path:
            inner_path = path

    if not outer_path:
        print("Error: standard_data_forward_operator.pkl not found in provided paths.")
        sys.exit(1)

    # --- Phase 1: Load Outer Data and Execute Function ---
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_result = outer_data.get('output')

    print("Executing forward_operator with outer arguments...")
    try:
        # Run the target function
        actual_result = forward_operator(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Error executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Phase 2: Check for Factory Pattern (Inner Data) ---
    # In this specific case, forward_operator returns a dictionary of data directly (Scenario A),
    # NOT a callable/closure. However, we keep logic robust for both.
    
    if inner_path:
        # Scenario B: The function returned a callable (Closure/Factory)
        print(f"Inner data path found: {inner_path}. Treating result as a callable operator.")
        
        if not callable(actual_result):
            print("Error: Inner data file exists, implying a factory pattern, but forward_operator did not return a callable.")
            sys.exit(1)

        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"Error loading inner pickle file: {e}")
            traceback.print_exc()
            sys.exit(1)

        inner_args = inner_data.get('args', [])
        inner_kwargs = inner_data.get('kwargs', {})
        expected_final_result = inner_data.get('output')

        print("Executing the returned operator with inner arguments...")
        try:
            final_actual_result = actual_result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"Error executing the returned operator: {e}")
            traceback.print_exc()
            sys.exit(1)
            
        print("Verifying final result against inner data output...")
        is_correct, fail_msg = recursive_check(expected_final_result, final_actual_result)

    else:
        # Scenario A: The function returned the data directly
        print("No inner data path found. Treating result as final output.")
        
        # In this specific problem context, forward_operator is a simulation function returning data.
        print("Verifying result against outer data output...")
        is_correct, fail_msg = recursive_check(expected_outer_result, actual_result)

    # --- Final Verification ---
    if is_correct:
        print("TEST PASSED: Output matches expected data.")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {fail_msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()