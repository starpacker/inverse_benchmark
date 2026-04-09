import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the directory containing the agent code to sys.path
sys.path.append(os.path.abspath('.'))

# Import the target function
from agent_find_max_discrepancy import find_max_discrepancy
from verification_utils import recursive_check

# Mock or Import necessary dependencies for dill loading
# Since dill might try to load classes from modules it serialized
try:
    import brighteyes_ism.simulation.PSF_sim as psf_sim
except ImportError:
    # If the original environment module structure isn't exactly preserved, 
    # we might need to mock this class if it's just a data holder.
    # However, usually the environment is set up. 
    # Let's attempt to define a dummy structure if import fails, 
    # but based on instructions, environment is assumed ready.
    pass

def test_find_max_discrepancy():
    data_paths = ['/data/yjh/s2ISM-main_sandbox/run_code/std_data/standard_data_find_max_discrepancy.pkl']
    
    # 1. Identify File Types
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        if 'parent_function' in p:
            inner_paths.append(p)
        else:
            outer_path = p

    if not outer_path:
        print("Error: No outer data file found (standard_data_find_max_discrepancy.pkl).")
        sys.exit(1)

    print(f"Loading data from {outer_path}...")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Reconstruct / Execute
    # Scenario A: Simple Function (The output is the final result)
    # Scenario B: Factory (The output is a callable)
    
    print("Executing find_max_discrepancy with loaded arguments...")
    try:
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        # Run the function
        actual_result = find_max_discrepancy(*args, **kwargs)
        
        # Check if the result is a callable (Scenario B) or a value (Scenario A)
        if callable(actual_result) and inner_paths:
            # Scenario B: It's a factory/closure pattern
            print("Detected closure pattern. Executing inner function(s)...")
            
            for inner_p in inner_paths:
                print(f"  Loading inner data from {inner_p}...")
                with open(inner_p, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output')
                
                # Execute the closure
                closure_result = actual_result(*inner_args, **inner_kwargs)
                
                # Verify
                passed, msg = recursive_check(inner_expected, closure_result)
                if not passed:
                    print(f"FAILED (Inner execution): {inner_p}")
                    print(msg)
                    sys.exit(1)
                else:
                    print(f"  PASSED: {inner_p}")
            
            print("All inner tests passed.")
            
        else:
            # Scenario A: It's a simple function return
            print("Comparing direct results...")
            passed, msg = recursive_check(expected_output, actual_result)
            if not passed:
                print("FAILED: Output mismatch.")
                print(msg)
                sys.exit(1)
            else:
                print("TEST PASSED")

    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_find_max_discrepancy()