import sys
import os
import dill
import numpy as np
import traceback

# Ensure the current directory is in the path to import local modules
sys.path.append(os.getcwd())

try:
    from agent_phasor_filter_median_impl import phasor_filter_median_impl
except ImportError:
    print("Error: Could not import 'phasor_filter_median_impl' from 'agent_phasor_filter_median_impl.py'")
    sys.exit(1)

try:
    from verification_utils import recursive_check
except ImportError:
    print("Error: Could not import 'recursive_check' from 'verification_utils.py'")
    sys.exit(1)

def main():
    # 1. Data File Configuration
    # Based on the prompt's provided data paths
    target_data_path = '/data/yjh/phasorpy-main_sandbox/run_code/std_data/standard_data_phasor_filter_median_impl.pkl'
    
    # 2. Load Data
    if not os.path.exists(target_data_path):
        print(f"Error: Data file not found at {target_data_path}")
        sys.exit(1)

    try:
        with open(target_data_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Extract Inputs and Expected Outputs
    # Structure from gen_data_code: {'func_name': ..., 'args': ..., 'kwargs': ..., 'output': ...}
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output')
    func_name = data.get('func_name', 'phasor_filter_median_impl')

    print(f"Testing function: {func_name}")
    print(f"Data source: {target_data_path}")

    # 4. Execute Function
    try:
        # Scenario A: Simple Function Execution
        # The function returns (mean, real, imag) directly.
        actual_result = phasor_filter_median_impl(*args, **kwargs)
    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verification
    try:
        is_correct, msg = recursive_check(expected_output, actual_result)
        
        if is_correct:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            # Optional: Print shape/type info for debugging
            if isinstance(actual_result, tuple) and isinstance(expected_output, tuple):
                print(f"Expected tuple len: {len(expected_output)}, Actual: {len(actual_result)}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Verification process failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()