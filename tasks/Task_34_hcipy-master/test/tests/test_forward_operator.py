import sys
import os
import dill
import numpy as np
import torch
import traceback

# Add current directory to path to ensure local imports work
sys.path.append(os.getcwd())

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

# --- MOCK / HELPER INJECTION SECTION ---
# The error "NameError: name 'Field' is not defined" suggests that the deserialized 
# objects (dm/atmosphere) contain code that references 'Field'. 
# We need to define or import Field so the unpickled code can find it.

class Field(np.ndarray):
    """
    Mock or Minimal implementation of hcipy.Field to satisfy the unpickled objects.
    In a real scenario, this would likely be 'from hcipy import Field'.
    Since we don't have the library installed in this isolated check, we mimic the structure
    expected by the traceback: Field(data, grid).
    """
    def __new__(cls, arr, grid):
        obj = np.asarray(arr).view(cls)
        obj.grid = grid
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.grid = getattr(obj, 'grid', None)

# Inject into global namespace so dill-loaded functions can find it
globals()['Field'] = Field

# ---------------------------------------

def test_forward_operator():
    """
    Test script for forward_operator.
    """
    
    # 1. Define Data Paths
    data_paths = ['/data/yjh/hcipy-master_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # 2. Strategy: Single Execution (Scenario A)
    # The provided data path list only contains the main function data, 
    # and the function signature implies it returns a result (PSF), not a closure.
    
    path = data_paths[0]
    print(f"Loading data from {path}...")
    
    if not os.path.exists(path):
        print(f"Error: Data file not found at {path}")
        sys.exit(1)
        
    try:
        with open(path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    # Extract inputs and expected output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_result = data.get('output', None)
    
    print("Executing forward_operator...")
    try:
        # 3. Execution
        actual_result = forward_operator(*args, **kwargs)
        
        # 4. Verification
        print("Verifying results...")
        passed, msg = recursive_check(expected_result, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_forward_operator()