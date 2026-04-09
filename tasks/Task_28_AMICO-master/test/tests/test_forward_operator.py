import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the module is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the target function
from agent_forward_operator import forward_operator, Scheme

# Import verification utility
from verification_utils import recursive_check

def test_forward_operator():
    """
    Test script for forward_operator.
    Strategy: Scenario A (Direct Function Execution)
    """
    
    # 1. Define Data Paths
    outer_path = '/data/yjh/AMICO-master_sandbox/run_code/std_data/standard_data_forward_operator.pkl'
    
    if not os.path.exists(outer_path):
        print(f"Skipping test: Data file not found at {outer_path}")
        sys.exit(0)

    try:
        # 2. Load Data
        print(f"Loading data from {outer_path}...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)

        # 3. Execution
        # Note: The data args might contain a Scheme object serialized by dill. 
        # If the class definition in the data file differs slightly from the one in agent_forward_operator,
        # there might be issues. However, since the provided code includes the Scheme class, 
        # and dill handles class serialization often by reference or value, we proceed.
        
        # Checking if args[0] is a Scheme object or path. If it's a path that doesn't exist, we might need to mock or handle it.
        # But usually in these pickles, the object itself is serialized.
        
        print("Executing forward_operator...")
        actual_output = forward_operator(*args, **kwargs)

        # 4. Verification
        print("Verifying results...")
        passed, msg = recursive_check(expected_output, actual_output)

        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_forward_operator()