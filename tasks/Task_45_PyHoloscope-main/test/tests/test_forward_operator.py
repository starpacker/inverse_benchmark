import sys
import os
import dill
import numpy as np
import scipy.signal
import math
import traceback

# Import the target function
from agent_forward_operator import forward_operator
# Import verification utility
from verification_utils import recursive_check

# -------------------------------------------------------------------------
# HELPER INJECTION
# -------------------------------------------------------------------------
# The data generation code utilizes specific decorators and potential globals.
# To ensure 'dill.load' works correctly without MissingDefinition errors,
# we may need to define empty placeholders or mocks if the pickle file 
# references them. However, for standard numpy data, this is often not needed
# unless the pickle contains function closures wrapping these decorators.

# -------------------------------------------------------------------------
# TEST LOGIC
# -------------------------------------------------------------------------
def run_test():
    # 1. Define Data Paths
    # Based on the user instruction, we have one primary data file.
    # However, we must check if this is a factory pattern or a direct function call.
    outer_path = '/data/yjh/PyHoloscope-main_sandbox/run_code/std_data/standard_data_forward_operator.pkl'
    
    # We search for potential inner data files (for closure/factory pattern)
    base_dir = os.path.dirname(outer_path)
    inner_files = [
        os.path.join(base_dir, f) for f in os.listdir(base_dir) 
        if 'standard_data_parent_function_forward_operator' in f and f.endswith('.pkl')
    ]
    
    # If inner files exist, we likely have a factory pattern (Scenario B).
    # If not, it's a simple function call (Scenario A).
    # Looking at the source code of `forward_operator`, it returns `prop_field` (np.ndarray),
    # not a function. It calculates FFTs and returns the result immediately.
    # Therefore, this is almost certainly Scenario A (Simple Function).
    # We will write robust logic to handle both, just in case the provided pickle
    # implies a different usage in the data generation context.
    
    is_factory = len(inner_files) > 0

    print(f"Test Configuration: Factory Mode = {is_factory}")
    if is_factory:
        print(f"Found inner data files: {inner_files}")
    
    # 2. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("Outer data loaded successfully.")
    except Exception as e:
        print(f"FAILED to load outer data: {e}")
        sys.exit(1)

    # 3. Execution Phase 1: Call the primary function
    print("\n--- Executing Phase 1: forward_operator ---")
    try:
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        
        # Execute
        phase1_result = forward_operator(*args, **kwargs)
        print("Phase 1 execution successful.")
    except Exception as e:
        print(f"Phase 1 execution FAILED with error:")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification Logic
    actual_result = None
    expected_result = None

    if is_factory:
        # SCENARIO B: Factory Pattern
        # The result of Phase 1 is a callable (the operator)
        agent_operator = phase1_result
        if not callable(agent_operator):
            print(f"FAILURE: Expected a callable operator in Factory Mode, but got {type(agent_operator)}")
            sys.exit(1)
            
        # Load Inner Data (Taking the first one found, assuming consistent behavior)
        inner_path = inner_files[0]
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"FAILED to load inner data from {inner_path}: {e}")
            sys.exit(1)
            
        print("\n--- Executing Phase 2: Inner Operator ---")
        try:
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            
            actual_result = agent_operator(*inner_args, **inner_kwargs)
            expected_result = inner_data['output']
            print("Phase 2 execution successful.")
        except Exception as e:
            print(f"Phase 2 execution FAILED with error:")
            traceback.print_exc()
            sys.exit(1)
            
    else:
        # SCENARIO A: Simple Function
        # The result of Phase 1 is the final data
        actual_result = phase1_result
        expected_result = outer_data['output']

    # 5. Comparison
    print("\n--- Verifying Results ---")
    try:
        passed, msg = recursive_check(expected_result, actual_result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"Verification process FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()