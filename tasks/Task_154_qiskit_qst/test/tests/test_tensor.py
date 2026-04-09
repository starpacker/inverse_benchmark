import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_tensor import tensor

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for tensor."""
    
    # Data paths provided
    data_paths = ['/data/yjh/qiskit_qst_sandbox_sandbox/run_code/std_data/standard_data_tensor.pkl']
    
    # Determine outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_tensor.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_tensor.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    # Execute tensor function
    try:
        agent_result = tensor(*outer_args, **outer_kwargs)
        print("Successfully executed tensor function")
    except Exception as e:
        print(f"ERROR: Failed to execute tensor function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario based on inner paths
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        # The agent_result should be callable
        if not callable(agent_result):
            print("ERROR: Expected callable result for factory pattern, but got non-callable")
            sys.exit(1)
        
        # Load inner data and execute
        inner_path = inner_paths[0]  # Use first inner path
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"Loaded inner data from: {inner_path}")
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output')
        
        try:
            result = agent_result(*inner_args, **inner_kwargs)
            print("Successfully executed inner function call")
        except Exception as e:
            print(f"ERROR: Failed to execute inner function: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple Function
        result = agent_result
        expected = outer_output
    
    # Phase 2: Verification
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()