import sys
import os
import dill
import numpy as np
import traceback
import torch

# Ensure the current directory is in sys.path to import the agent and verification utils
sys.path.append(os.getcwd())

try:
    from agent_envel_detect import envel_detect
    from verification_utils import recursive_check
except ImportError as e:
    print(f"[ERROR] Could not import modules: {e}")
    sys.exit(1)

def run_test():
    # 1. Define Data Paths
    # Based on the prompt, we are looking for the standard data file.
    data_path = '/data/yjh/us-beamform-linarray-master_sandbox/run_code/std_data/standard_data_envel_detect.pkl'

    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found at: {data_path}")
        print("Cannot run verification without input data.")
        sys.exit(1)

    # 2. Load Data
    try:
        with open(data_path, 'rb') as f:
            data = dill.load(f)
        
        # Extract inputs and expected output
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        expected_output = data.get('output', None)
        
        print(f"[INFO] Loaded data from {os.path.basename(data_path)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load data via dill: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Execution Strategy
    # The target function 'envel_detect' is a simple processing function (Scenario A),
    # not a factory/closure. We simply run it with the loaded arguments.
    
    try:
        print("[INFO] Executing envel_detect...")
        actual_output = envel_detect(*args, **kwargs)
        
    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    try:
        print("[INFO] Verifying results...")
        passed, msg = recursive_check(expected_output, actual_output)
        
        if passed:
            print("[SUCCESS] TEST PASSED")
            sys.exit(0)
        else:
            print(f"[FAILURE] TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Verification logic failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()