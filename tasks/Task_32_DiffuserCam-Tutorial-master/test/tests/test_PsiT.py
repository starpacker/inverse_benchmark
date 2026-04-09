import sys
import os
import dill
import numpy as np
import traceback

# Handle optional torch dependency to prevent ModuleNotFoundError
try:
    import torch
except ImportError:
    torch = None

from agent_PsiT import PsiT
from verification_utils import recursive_check

def test_PsiT():
    """
    Test script for PsiT function.
    Scenario: Simple Function (Direct execution) based on code analysis.
    """
    
    # 1. Define Data Paths
    data_paths = ['/data/yjh/DiffuserCam-Tutorial-master_sandbox/run_code/std_data/standard_data_PsiT.pkl']
    
    target_path = None
    for path in data_paths:
        if 'standard_data_PsiT.pkl' in path:
            target_path = path
            break
            
    if not target_path or not os.path.exists(target_path):
        print(f"Skipping test: Data file not found at {target_path}")
        sys.exit(0)

    print(f"Loading data from {target_path}...")

    # 2. Load Data
    try:
        with open(target_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Extract Inputs and Expected Outputs
    try:
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        expected_output = data.get('output')
        
        # Determine if this is a closure pattern or direct execution
        # Based on the provided code, PsiT returns a numpy array, so it is likely direct execution.
        # We execute the function with loaded args.
        
        print(f"Executing PsiT with {len(args)} args and {len(kwargs)} kwargs...")
        
        # 4. Execute Function
        actual_output = PsiT(*args, **kwargs)
        
        # 5. Verify Results
        passed, msg = recursive_check(expected_output, actual_output)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_PsiT()