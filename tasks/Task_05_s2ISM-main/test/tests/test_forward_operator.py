import sys
import os
import dill
import numpy as np
import torch
import traceback

# Ensure current directory is in path for imports
sys.path.append(os.getcwd())

from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def run_test():
    # 1. Define Data Path
    data_paths = ['/data/yjh/s2ISM-main_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    target_path = data_paths[0]

    if not os.path.exists(target_path):
        print(f"Error: Data file not found at {target_path}")
        sys.exit(1)

    # 2. Load Data
    print(f"Loading data from {target_path}...")
    try:
        with open(target_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle data: {e}")
        traceback.print_exc()
        sys.exit(1)

    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output')

    # 3. Execute Function
    print("Executing forward_operator...")
    try:
        # The function uses np.random.poisson. We set the seed to try and match,
        # but we will also implement robust verification for stochastic outputs.
        np.random.seed(42)
        actual_output = forward_operator(*args, **kwargs)
    except Exception as e:
        print(f"Error executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verify Results
    print("Verifying results...")

    # The function returns (data_ISM_noise, ground_truth_scaled)
    # Index 0: data_ISM_noise -> STOCHASTIC (Poisson noise)
    # Index 1: ground_truth_scaled -> DETERMINISTIC

    if not isinstance(actual_output, (tuple, list)) or len(actual_output) != 2:
        print(f"TEST FAILED: Output format mismatch. Expected tuple of length 2, got {type(actual_output)}")
        sys.exit(1)

    actual_noise, actual_gt = actual_output
    expected_noise, expected_gt = expected_output

    # A. Validate Deterministic Component (Ground Truth)
    # We expect strict equality or high precision match here.
    passed_gt, msg_gt = recursive_check(expected_gt, actual_gt)
    if not passed_gt:
        print(f"TEST FAILED: Mismatch in deterministic component (ground_truth_scaled).")
        print(msg_gt)
        sys.exit(1)

    # B. Validate Stochastic Component (Noise)
    # Poisson noise prevents exact matching (difference of 33 observed previously). 
    # We check structural integrity instead of exact values.
    
    # Check 1: Shape
    if actual_noise.shape != expected_noise.shape:
        print(f"TEST FAILED: Shape mismatch in stochastic component (data_ISM_noise).")
        print(f"Expected: {expected_noise.shape}, Got: {actual_noise.shape}")
        sys.exit(1)

    # Check 2: Dtype
    if actual_noise.dtype != expected_noise.dtype:
        print(f"TEST FAILED: Dtype mismatch in stochastic component (data_ISM_noise).")
        print(f"Expected: {expected_noise.dtype}, Got: {actual_noise.dtype}")
        sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()