import sys
import os
import dill
import numpy as np
import torch
import traceback
from verification_utils import recursive_check
from agent_gaussian_kernel import gaussian_kernel

# Set fixed seeds for reproducibility, matching generation environment
def fix_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

fix_seeds()

def test_gaussian_kernel():
    """
    Test script for gaussian_kernel.
    Strategy: Scenario A (Simple Function)
    The function returns a numpy array directly, not a closure.
    We execute the function with stored args and compare the result.
    """
    
    data_paths = ['/data/yjh/semiblindpsfdeconv-master_sandbox/run_code/std_data/standard_data_gaussian_kernel.pkl']
    
    # Filter for the main data file
    outer_path = None
    for p in data_paths:
        if 'standard_data_gaussian_kernel.pkl' in p:
            outer_path = p
            break
            
    if not outer_path or not os.path.exists(outer_path):
        print(f"Skipping test: Data file not found at {outer_path}")
        sys.exit(0)

    try:
        # Load the data
        with open(outer_path, 'rb') as f:
            data = dill.load(f)
        
        func_name = data.get('func_name', '')
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        expected_output = data.get('output', None)
        
        print(f"Loaded data for function: {func_name}")
        print(f"Args: {args}")
        print(f"Kwargs: {kwargs.keys()}")

        # Execute the function
        actual_result = gaussian_kernel(*args, **kwargs)

        # Verification
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
    test_gaussian_kernel()