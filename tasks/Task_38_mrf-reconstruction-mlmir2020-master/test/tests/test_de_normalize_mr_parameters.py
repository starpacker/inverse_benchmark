import sys
import os
import dill
import numpy as np
import traceback

# Add project root to sys.path to ensure imports work
# We assume the script is running in an environment where the project root is accessible
# or relevant paths are set.
# Based on the file path structure provided, we try to add the directory containing the agent.
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    sys.path.append('/data/yjh/mrf-reconstruction-mlmir2020-master_sandbox/run_code')
except Exception:
    pass

try:
    from agent_de_normalize_mr_parameters import de_normalize_mr_parameters
    from verification_utils import recursive_check
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def run_test():
    # Define data paths
    data_paths = ['/data/yjh/mrf-reconstruction-mlmir2020-master_sandbox/run_code/std_data/standard_data_de_normalize_mr_parameters.pkl']
    
    # Identify the main outer data file
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if 'standard_data_de_normalize_mr_parameters.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_de_normalize_mr_parameters' in path:
            inner_paths.append(path)

    if not outer_path:
        print("Error: Main data file 'standard_data_de_normalize_mr_parameters.pkl' not found.")
        sys.exit(1)

    print(f"Loading data from {outer_path}...")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract args and kwargs
    args = outer_data.get('args', [])
    kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')

    print("Executing de_normalize_mr_parameters...")
    try:
        # Scenario A: The function is a standard function that returns data directly
        # based on the provided signature in the prompt.
        actual_result = de_normalize_mr_parameters(*args, **kwargs)
        
        # Scenario B check: If the function returned a callable (closure pattern),
        # we would need to execute it with inner data. However, looking at the 
        # provided target code, it returns a numpy array directly.
        # We proceed with direct verification.
        
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_output, actual_result)
    except Exception as e:
        print(f"Verification process failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()