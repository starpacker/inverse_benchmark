import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

# Global settings for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

def test_forward_operator():
    """
    Unit test for forward_operator using captured standard data.
    """
    # 1. Define Data Paths
    data_paths = ['/data/yjh/mrf-reconstruction-mlmir2020-master_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # 2. Identify the Data File
    # Based on the provided list, we only have one file.
    # Scenario A: The function returns a value directly, not a closure.
    # Looking at the function signature: `forward_operator(model, x, ndim_y, device)` returns `y_pred`.
    # This confirms Scenario A.
    
    target_path = None
    for path in data_paths:
        if path.endswith('standard_data_forward_operator.pkl'):
            target_path = path
            break
            
    if not target_path:
        print("Error: standard_data_forward_operator.pkl not found in provided paths.")
        sys.exit(1)

    # 3. Load Data
    try:
        with open(target_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading data file {target_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Extract Inputs and Expected Output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output')

    # 5. Execution
    try:
        # The stored args/kwargs might contain tensors on specific devices or detached.
        # We pass them directly to the function under test.
        actual_output = forward_operator(*args, **kwargs)
    except Exception as e:
        print(f"Error executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 6. Verification
    try:
        passed, msg = recursive_check(expected_output, actual_output)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_forward_operator()