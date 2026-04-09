import sys
import os
import dill
import numpy as np
import traceback

# Handle optional torch import to prevent ModuleNotFoundError
try:
    import torch
except ImportError:
    torch = None

# Add current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

try:
    from agent_unitsphere2cart_1d import unitsphere2cart_1d
except ImportError:
    print("CRITICAL: Could not import 'unitsphere2cart_1d' from 'agent_unitsphere2cart_1d'.")
    sys.exit(1)

try:
    from verification_utils import recursive_check
except ImportError:
    print("CRITICAL: Could not import 'recursive_check' from 'verification_utils'.")
    sys.exit(1)

def main():
    # Define the data path based on the prompt's instructions
    data_paths = ['/data/yjh/dmipy-master_sandbox/run_code/std_data/standard_data_unitsphere2cart_1d.pkl']
    
    # We look for the main data file.
    # Since unitsphere2cart_1d returns an array (not a callable), we expect Scenario A (Simple Function).
    target_path = None
    for p in data_paths:
        if 'standard_data_unitsphere2cart_1d.pkl' in p:
            target_path = p
            break
            
    if not target_path or not os.path.exists(target_path):
        print(f"Test Skipped: Data file not found at {target_path}")
        sys.exit(0)

    print(f"Loading data from {target_path}...")
    try:
        with open(target_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"CRITICAL: Failed to load data with dill: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract inputs and expected output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_result = data.get('output')

    print(f"Executing unitsphere2cart_1d with {len(args)} args and {len(kwargs)} kwargs...")
    
    try:
        # Execute the function
        actual_result = unitsphere2cart_1d(*args, **kwargs)
    except Exception as e:
        print(f"CRITICAL: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Verify the result
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_result, actual_result)
    except Exception as e:
        print(f"CRITICAL: Verification logic failed: {e}")
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