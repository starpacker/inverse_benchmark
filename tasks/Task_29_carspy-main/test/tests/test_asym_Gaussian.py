import sys
import os
import dill
import numpy as np
import torch
import traceback

# Add current directory to path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the target function
try:
    from agent_asym_Gaussian import asym_Gaussian
except ImportError:
    print("Error: Could not import 'asym_Gaussian' from 'agent_asym_Gaussian.py'")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # If verification_utils is missing, we can define a fallback or fail. 
    # Provided instructions imply it exists.
    print("Error: Could not import 'recursive_check' from 'verification_utils.py'")
    sys.exit(1)

def run_test():
    # Provided data paths
    data_paths = ['/data/yjh/carspy-main_sandbox/run_code/std_data/standard_data_asym_Gaussian.pkl']
    
    # Strategy:
    # The function asym_Gaussian returns a numpy array, not a callable.
    # Therefore, we are in "Scenario A": Simple Function Execution.
    # We look for the main data file.
    
    target_path = None
    for p in data_paths:
        if p.endswith('standard_data_asym_Gaussian.pkl'):
            target_path = p
            break
            
    if not target_path:
        print(f"Error: Could not find 'standard_data_asym_Gaussian.pkl' in provided paths: {data_paths}")
        sys.exit(1)
        
    if not os.path.exists(target_path):
        print(f"Error: File does not exist at path: {target_path}")
        sys.exit(1)
        
    print(f"Loading data from {target_path}...")
    try:
        with open(target_path, 'rb') as f:
            data_payload = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    # Extract inputs
    args = data_payload.get('args', [])
    kwargs = data_payload.get('kwargs', {})
    expected_result = data_payload.get('output')
    
    print("Executing asym_Gaussian with loaded arguments...")
    try:
        # Execute the function
        actual_result = asym_Gaussian(*args, **kwargs)
    except Exception as e:
        print(f"Error during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    # Verification
    print("Verifying results...")
    try:
        is_correct, msg = recursive_check(expected_result, actual_result)
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if is_correct:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()