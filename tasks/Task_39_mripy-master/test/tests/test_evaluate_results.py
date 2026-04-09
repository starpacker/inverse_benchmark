import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the directory containing the agent code to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

# Define data paths
data_paths = ['/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test():
    # 1. Identify File Types
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        if 'standard_data_evaluate_results.pkl' in p:
            outer_path = p
        elif 'parent_function' in p:
            inner_paths.append(p)

    if not outer_path:
        print("Error: standard_data_evaluate_results.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading data from {outer_path}...")
    try:
        outer_data = load_pkl(outer_path)
    except Exception as e:
        print(f"Failed to load pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')

    # 2. Execute Function
    print("Executing evaluate_results with loaded arguments...")
    try:
        # Based on the provided code and data analysis, this is Scenario A: Simple Function.
        # However, we structure it to handle the possibility of it being a factory if inner paths existed.
        
        # Execute Outer
        result = evaluate_results(*outer_args, **outer_kwargs)

        # 3. Check for Scenario B (Factory Pattern)
        # If the result is callable and we have inner paths, it's a factory. 
        # But given only one path, we assume the immediate result is the final output.
        if callable(result) and inner_paths:
            # This block handles the factory case if inner files were present
            print("Detected Factory pattern. Executing inner functions...")
            agent_operator = result
            
            for inner_p in inner_paths:
                inner_data = load_pkl(inner_p)
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output')
                
                inner_result = agent_operator(*inner_args, **inner_kwargs)
                
                passed, msg = recursive_check(inner_expected, inner_result)
                if not passed:
                    print(f"Inner Verification FAILED for {inner_p}: {msg}")
                    sys.exit(1)
            print("All inner verifications passed.")
            
        else:
            # Scenario A: Immediate Result
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"Verification FAILED: {msg}")
                sys.exit(1)
            
        print("TEST PASSED")
        sys.exit(0)

    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()