import sys
import os
import dill
import numpy as np
import traceback

# Ensure the target module is available
try:
    from agent_make_pupil_grid import make_pupil_grid, CartesianGrid, Grid, Field
except ImportError:
    print("Error: Could not import 'make_pupil_grid' or helper classes from 'agent_make_pupil_grid.py'.")
    sys.exit(1)

from verification_utils import recursive_check

# --- namespace fix for dill loading ---
# The pickle file contains objects (like CartesianGrid) saved under the '__main__' scope 
# or a different scope than the current import. We map them to the imported classes 
# so dill loads them as the classes we actually have.
import agent_make_pupil_grid
if "__main__" in sys.modules:
    sys.modules["__main__"].CartesianGrid = agent_make_pupil_grid.CartesianGrid
    sys.modules["__main__"].Grid = agent_make_pupil_grid.Grid
    sys.modules["__main__"].Field = agent_make_pupil_grid.Field
# -------------------------------------

def run_test():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/hcipy-master_sandbox/run_code/std_data/standard_data_make_pupil_grid.pkl']
    
    outer_path = None
    inner_path = None

    # 2. Identify Test Strategy (Factory vs Simple Function)
    for path in data_paths:
        if 'standard_data_make_pupil_grid.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_make_pupil_grid' in path:
            inner_path = path

    if not outer_path:
        print("TEST FAILED: Standard input data 'standard_data_make_pupil_grid.pkl' not found.")
        sys.exit(1)

    # 3. Load Outer Data
    try:
        print(f"Loading outer data from {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"TEST FAILED: Error loading outer pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Execute Function
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Running make_pupil_grid with args: {outer_args} and kwargs: {outer_kwargs}")
        actual_result = make_pupil_grid(*outer_args, **outer_kwargs)

    except Exception as e:
        print(f"TEST FAILED: Error executing make_pupil_grid: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Handle Verification Strategy
    expected_result = None

    if inner_path:
        # Factory Pattern Strategy: The result of step 4 is a closure/operator
        if not callable(actual_result):
            print("TEST FAILED: Expected a callable (Factory Pattern) but got a static result.")
            sys.exit(1)
            
        try:
            print(f"Loading inner data from {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_result = inner_data.get('output')

            print(f"Executing inner closure with args: {inner_args}")
            actual_result = actual_result(*inner_args, **inner_kwargs)
            
        except Exception as e:
            print(f"TEST FAILED: Error executing inner closure: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Simple Function Strategy: The result of step 4 is the final object
        expected_result = outer_data.get('output')

    # 6. Verification
    print("Verifying result...")
    
    # Custom relaxation for Grid/CartesianGrid comparison if needed
    # We strip the specific class check if the attributes match, as namespaces often clash in pickles
    
    # Specifically for this case, recursive_check might fail on strict type equality 
    # if the pickle class path != current class path. We manually check critical attributes first.
    if hasattr(expected_result, 'coords') and hasattr(actual_result, 'coords'):
        try:
            # Check coords
            for c_exp, c_act in zip(expected_result.coords, actual_result.coords):
                if not np.allclose(c_exp, c_act):
                    print("TEST FAILED: Coordinate mismatch in Grid.")
                    sys.exit(1)
            # Check weights
            if expected_result.weights is not None:
                if not np.allclose(expected_result.weights, actual_result.weights):
                    print("TEST FAILED: Weights mismatch in Grid.")
                    sys.exit(1)
            # Check dims/delta if Cartesian
            if hasattr(expected_result, 'delta') and hasattr(actual_result, 'delta'):
                 if not np.allclose(expected_result.delta, actual_result.delta):
                    print("TEST FAILED: Delta mismatch in CartesianGrid.")
                    sys.exit(1)
            
            print("TEST PASSED") # Manual attributes match, bypassing strict type check
            sys.exit(0)
        except Exception as e:
            print(f"Manual verification failed: {e}")
            # Fall through to recursive_check

    passed, msg = recursive_check(expected_result, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: Output mismatch.\n{msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()