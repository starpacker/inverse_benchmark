import sys
import os
import dill
import numpy as np
import traceback
import h5py
from scipy import signal

# Add the directory containing the agent to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
# Import verification utility
from verification_utils import recursive_check

def test_load_and_preprocess_data():
    """
    Test script for load_and_preprocess_data function using captured pickle data.
    """
    data_paths = ['/data/yjh/us-beamform-linarray-master_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # 1. Identify Data Files
    # Based on the provided path, we seem to have Scenario A (Simple Function execution)
    # However, we will implement logic to handle both standard inputs and potential closure patterns if they arise.
    
    outer_path = None
    inner_paths = []

    for path in data_paths:
        if 'parent_function' in path:
            inner_paths.append(path)
        else:
            outer_path = path

    if not outer_path:
        print("Error: No standard data file found (outer function inputs).")
        sys.exit(1)

    print(f"Loading data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

    # 2. Extract Args and Kwargs for the outer function
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    # 3. Handle File Path Dependencies
    # The first argument is likely a file path to an h5 file. 
    # The pickle captured the path used during recording. We must ensure this file exists
    # or mock the behavior if the file is missing but we want to test logic.
    # However, in this specific provided context, we assume the environment mirrors the recording environment
    # or the file path is accessible. 
    
    # Check if the first arg is a path and if it exists
    if outer_args and isinstance(outer_args[0], str):
        data_file_path = outer_args[0]
        if not os.path.exists(data_file_path):
            print(f"Warning: Data file path in args not found: {data_file_path}")
            # In a real CI/CD, we might need to download this file or skip. 
            # For this strict test generation, we proceed, noting that FileNotFoundError might occur.

    print("Running load_and_preprocess_data with loaded arguments...")
    try:
        # 4. Execute the function
        actual_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
    except Exception as e:
        print("Execution failed with error:")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verification Logic
    # Scenario A: The function returns data directly (tuples of numpy arrays).
    # Scenario B: The function returns a closure (callable).
    
    if callable(actual_result) and not isinstance(actual_result, (np.ndarray, tuple, list)):
        print("Detected Factory/Closure pattern.")
        if not inner_paths:
            print("Error: Function returned a callable, but no inner data files (parent_function) were provided to test it.")
            sys.exit(1)
            
        # If we had inner paths (Scenario B), we would iterate them here.
        # Given the provided data list only has the main pkl, this block is defensive.
        for inner_path in inner_paths:
             # Logic for inner execution would go here (load args, call actual_result(*args), check output)
             pass
    
    else:
        # Scenario A: Direct comparison
        print("Detected direct return value.")
        
        # We need to compare actual_result with expected_outer_output
        passed, msg = recursive_check(expected_outer_output, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

if __name__ == "__main__":
    test_load_and_preprocess_data()