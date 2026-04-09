import sys
import os
import dill
import numpy as np
import traceback
import scipy.fft
import scipy.ndimage

# Add current directory to path so we can import the agent code
sys.path.append(os.path.dirname(__file__))

# Import the target function
try:
    from agent_radon_transform_logic import radon_transform_logic
except ImportError:
    print("Error: Could not import 'radon_transform_logic' from 'agent_radon_transform_logic.py'.")
    sys.exit(1)

# Import verification utils
try:
    from verification_utils import recursive_check
except ImportError:
    print("Error: Could not import 'recursive_check' from 'verification_utils.py'.")
    sys.exit(1)

def main():
    # Defined data paths from instructions
    data_paths = ['/data/yjh/tomopy-master_sandbox/run_code/std_data/standard_data_radon_transform_logic.pkl']
    
    # Analyze paths to distinguish between simple function execution and factory pattern
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if 'standard_data_radon_transform_logic.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_radon_transform_logic' in path:
            inner_paths.append(path)

    if not outer_path:
        print("Error: standard_data_radon_transform_logic.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading data from {outer_path}...")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

    # Execution Phase
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)

        print("Executing radon_transform_logic with loaded arguments...")
        
        # Scenario A: Simple function execution (since the provided code doesn't return a function/closure)
        # Based on the provided code, radon_transform_logic returns 'sinogram' (numpy array), not a callable.
        # So we treat this as a direct execution scenario.
        
        actual_result = radon_transform_logic(*outer_args, **outer_kwargs)
        
        # Scenario B check: If the result is callable and we have inner paths, we would execute the result.
        # However, the provided reference code shows it returns a numpy array.
        if callable(actual_result) and inner_paths:
            print("Detected Factory Pattern. Executing returned operator...")
            operator = actual_result
            
            # For simplicity in this template, we test the first inner capture if multiple exist
            inner_path = inner_paths[0]
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
                
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            actual_result = operator(*inner_args, **inner_kwargs)
            
    except Exception:
        traceback.print_exc()
        print("Error during execution of radon_transform_logic.")
        sys.exit(1)

    # Verification Phase
    print("Verifying results...")
    try:
        is_correct, fail_msg = recursive_check(expected_output, actual_result)
    except Exception:
        traceback.print_exc()
        print("Error during verification logic.")
        sys.exit(1)

    if is_correct:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {fail_msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()