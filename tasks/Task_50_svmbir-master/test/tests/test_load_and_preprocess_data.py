import sys
import os
import dill
import numpy as np
import traceback

# Add the directory containing the agent code to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data

# Import verification utility
from verification_utils import recursive_check

def test_load_and_preprocess_data():
    """
    Unit test for load_and_preprocess_data based on captured data.
    """
    data_paths = ['/data/yjh/svmbir-master_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']

    # 1. Identify Data Files
    outer_path = None
    inner_paths = []

    for path in data_paths:
        if 'parent_function' in path:
            inner_paths.append(path)
        else:
            outer_path = path

    if outer_path is None:
        print("Error: No outer data file found (standard_data_load_and_preprocess_data.pkl).")
        sys.exit(1)

    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Execute Outer Function
    # This recreates the agent/result using the same inputs as recorded.
    print("Executing load_and_preprocess_data with loaded arguments...")
    try:
        # Note: The provided reference code for load_and_preprocess_data uses np.random.normal.
        # Since the random seed was fixed in the generation code (via _fix_seeds_), 
        # we expect deterministic output if we run in the same environment.
        # However, due to floating point or environment differences, exact equality might fail 
        # if the check is too strict, but recursive_check handles tolerances.
        
        # We need to ensure seeds are reset if the function relies on randomness internally
        # akin to how the generation code did.
        np.random.seed(42) 
        
        actual_result = load_and_preprocess_data(*outer_data['args'], **outer_data['kwargs'])
    except Exception as e:
        print(f"Error executing function: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Determine Execution Scenario
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        # The result of the outer function is an operator (callable) that needs to be tested
        # against the inner data inputs.
        print(f"detected factory pattern with {len(inner_paths)} inner calls.")
        
        if not callable(actual_result):
            print(f"Error: Expected outer function to return a callable (operator), but got {type(actual_result)}")
            sys.exit(1)
            
        operator = actual_result
        
        for inner_path in inner_paths:
            print(f"Testing inner execution: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"Error loading inner data: {e}")
                sys.exit(1)
                
            try:
                # Execute the operator
                inner_result = operator(*inner_data['args'], **inner_data['kwargs'])
                
                # Compare
                passed, msg = recursive_check(inner_data['output'], inner_result)
                if not passed:
                    print(f"Inner Check Failed: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"Error executing inner operator: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple Function
        # The outer function returns the final result (data tuples).
        print("Scenario A: Direct result comparison.")
        
        expected_result = outer_data['output']
        
        # Verify
        passed, msg = recursive_check(expected_result, actual_result)
        if not passed:
            print(f"Check Failed: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    test_load_and_preprocess_data()