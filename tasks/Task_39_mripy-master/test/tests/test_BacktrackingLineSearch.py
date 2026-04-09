import sys
import os
import dill
import numpy as np
import torch
import traceback
from agent_BacktrackingLineSearch import BacktrackingLineSearch
from verification_utils import recursive_check

# --- Helper Injection Section ---
# The deserialized functions (f, df) depend on these helpers. 
# We must define them globally so dill can find them during execution.

def dim_match(s1, s2):
    """
    Helper function potentially required by the deserialized operators.
    Matches dimensions between sensitivity maps and images.
    """
    if len(s1) == len(s2):
        return (s1, s2)
    elif len(s1) > len(s2):
        s2_new = list(s2) + [1] * (len(s1) - len(s2))
        return (s1, tuple(s2_new))
    else:
        s1_new = list(s1) + [1] * (len(s2) - len(s1))
        return (tuple(s1_new), s2)

# Ensure 'dim_match' is available in the global namespace for dill-loaded functions
globals()['dim_match'] = dim_match

# --------------------------------

def load_data(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test():
    # 1. Paths
    # Note: The prompt provided this specific path.
    outer_data_path = '/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_BacktrackingLineSearch.pkl'
    
    if not os.path.exists(outer_data_path):
        print(f"Error: Data file not found at {outer_data_path}")
        sys.exit(1)

    try:
        # 2. Load Data
        print(f"Loading data from: {outer_data_path}")
        outer_data = load_data(outer_data_path)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
        
        # 3. Execution
        # BacktrackingLineSearch is a standalone optimization step, not a factory.
        # It takes functions f, df and current state x, p, returning (alpha, iterations).
        print("Executing BacktrackingLineSearch...")
        
        # We need to handle potential GPU tensors in args if the environment is CPU-only,
        # though the context says GPU is available.
        # The recursive_check handles tensor comparisons.
        
        actual_result = BacktrackingLineSearch(*outer_args, **outer_kwargs)

        # 4. Verification
        print("Verifying results...")
        passed, msg = recursive_check(expected_output, actual_result)

        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"Execution Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()