import sys
import os
import dill
import numpy as np
import traceback

# Add the directory containing the agent code to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
# Import verification utility
from verification_utils import recursive_check

def test_load_and_preprocess_data():
    """
    Unit test for load_and_preprocess_data using captured execution data.
    """
    data_paths = ['/data/yjh/MPIRF-master_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # 1. Identify Data Files
    outer_data_path = None
    inner_data_path = None

    for path in data_paths:
        if 'parent_function' in path:
            inner_data_path = path
        elif 'standard_data_load_and_preprocess_data.pkl' in path:
            outer_data_path = path

    if not outer_data_path:
        print("Error: Standard outer data file not found in provided paths.")
        sys.exit(1)

    print(f"Loading outer data from: {outer_data_path}")
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

    # 2. Reconstruct Operator / Execute Function
    print("Executing load_and_preprocess_data with captured arguments...")
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Execute the function
        actual_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        
    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Handle Scenario A vs B
    expected_result = None

    if inner_data_path:
        # Scenario B: Factory Pattern (Not expected based on provided paths, but implemented for robustness)
        print(f"Detected inner data file: {inner_data_path}")
        if not callable(actual_result):
            print("Error: Expected a callable (factory/closure) from outer execution, but got a concrete result.")
            sys.exit(1)
            
        try:
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
                
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            
            print("Executing inner operator...")
            actual_final_result = actual_result(*inner_args, **inner_kwargs)
            expected_result = inner_data['output']
            actual_result = actual_final_result # Update for comparison
            
        except Exception as e:
            print(f"Inner execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple Function (Target scenario)
        print("No inner data file found. Treating as direct function execution.")
        expected_result = outer_data['output']

    # 4. Verification
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_result, actual_result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"Verification process failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_load_and_preprocess_data()