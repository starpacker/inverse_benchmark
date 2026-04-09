import sys
import os
import dill
import torch
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Add the directory containing the agent code to the path if necessary
# Assuming the agent code is in the current directory or accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the target function
from agent_evaluate_results import evaluate_results

# Import verification utility
from verification_utils import recursive_check

# -------------------------------------------------------------------------
# HELPER INJECTIONS FOR DILL
# -------------------------------------------------------------------------
# We need to ensure that any custom classes or functions used in the pickled 
# data (especially from the generator environment) are available here so 
# dill.load doesn't crash. Based on the provided code, we need to mock or 
# define the classes if they were pickled by value, but typically standard 
# pickling by reference just needs imports.
#
# However, `dill` often pickles the entire dependency graph. If the original 
# environment had specific helper functions in the global scope that are 
# referenced inside the pickled closure/object, we might need to be careful.
#
# Given the provided context, `evaluate_results` is a straightforward function
# that runs metrics and plotting. It doesn't appear to return a closure (Factory pattern),
# nor does the data list suggest inner files. 
# 
# Let's handle the specific definitions that might be missing if dill serialized 
# objects that rely on globals.
# -------------------------------------------------------------------------

def load_data(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def main():
    # 1. Configuration
    data_paths = ['/data/yjh/caustics-main_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # 2. Strategy Determination
    # We look for the "outer" data file which corresponds to the main function call.
    outer_data_path = None
    inner_data_paths = []

    for path in data_paths:
        if "parent_function" in path:
            inner_data_paths.append(path)
        elif "evaluate_results.pkl" in path:
            outer_data_path = path

    if not outer_data_path:
        print("Error: standard_data_evaluate_results.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading test data from: {outer_data_path}")
    try:
        outer_data = load_data(outer_data_path)
    except Exception as e:
        print(f"Failed to load pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Execution
    try:
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)

        print("Executing evaluate_results...")
        
        # Scenario A: Simple Function (No inner files found)
        # Based on the provided data paths, we only have the main file.
        # However, evaluate_results prints to stdout/saves a file and returns None usually.
        # We will capture the return value and compare it.
        
        # If the function creates a closure (Scenario B), the result would be a callable,
        # and we would then use inner_data_paths to test that callable.
        # But evaluate_results docstring suggests it calculates metrics and plots.
        
        actual_result = evaluate_results(*args, **kwargs)
        
        # Scenario Check: Did it return a callable?
        if callable(actual_result) and inner_data_paths:
            print("Detected Factory Pattern. Testing inner operator...")
            # If we had inner paths, we would iterate them. 
            # Since the prompt only provided the outer path, we proceed with Scenario A logic.
            # But just in case logic is needed for robustness:
            for inner_path in inner_data_paths:
                print(f"  Running inner test from {inner_path}")
                inner_data = load_data(inner_path)
                i_args = inner_data.get('args', [])
                i_kwargs = inner_data.get('kwargs', {})
                i_expected = inner_data.get('output')
                
                i_actual = actual_result(*i_args, **i_kwargs)
                
                passed, msg = recursive_check(i_expected, i_actual)
                if not passed:
                    print(f"Inner Comparison Failed: {msg}")
                    sys.exit(1)
            
            # If we processed inner files, we assume the outer result itself (the operator)
            # doesn't need direct value comparison against the pickle, or the pickle 
            # contains the function object which is hard to compare.
            # However, for a factory, often the outer output IS the operator.
            print("TEST PASSED (Factory Mode)")
            sys.exit(0)

        # Scenario A: Direct Result Comparison
        # For evaluate_results, the output is likely None (void function) or metrics.
        print("Comparing results...")
        passed, msg = recursive_check(expected_output, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            # If the expected output is None and we got None, recursive_check handles it.
            # If there's a mismatch, we fail.
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()