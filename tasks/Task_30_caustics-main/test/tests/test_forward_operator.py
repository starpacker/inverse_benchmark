import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_forward_operator import forward_operator
from verification_utils import recursive_check

# List of data paths provided
data_paths = ['/data/yjh/caustics-main_sandbox/run_code/std_data/standard_data_forward_operator.pkl']

def run_test():
    print("Starting test_forward_operator.py")
    
    # 1. Identify the Data File
    # Since forward_operator returns a Tensor (Scenario A), we expect one main data file.
    outer_path = None
    for p in data_paths:
        if p.endswith('standard_data_forward_operator.pkl'):
            outer_path = p
            break
    
    if outer_path is None:
        print("Test Skipped: standard_data_forward_operator.pkl not found in provided paths.")
        sys.exit(0)

    # 2. Load Data
    print(f"Loading test data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"FATAL: Failed to load pickle file. Error: {e}")
        sys.exit(1)

    # 3. Extract Inputs and Expected Output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output', None)

    # 4. Device & Environment Handling
    # The 'device' arg (args[1]) might be a specific GPU not available here.
    # We must ensure compatibility.
    current_args = list(args)
    if len(current_args) >= 2:
        # Check the requested device
        req_device = current_args[1]
        
        # If the input was serialized as a string or device object
        if isinstance(req_device, str):
            req_device = torch.device(req_device)
        
        # Fallback to CPU if CUDA is requested but unavailable
        if isinstance(req_device, torch.device):
            if req_device.type == 'cuda' and not torch.cuda.is_available():
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                current_args[1] = torch.device('cpu')
                # Also ensure the input tensor x (args[0]) is on CPU
                if isinstance(current_args[0], torch.Tensor):
                    current_args[0] = current_args[0].cpu()
            elif req_device.type == 'cuda' and torch.cuda.is_available():
                # Ensure we use a valid device index if the original one is out of bounds
                # or just map to current device. Simpler to trust torch unless it errors.
                pass
    
    # Re-tuple args
    args = tuple(current_args)

    # 5. Execute Target Function
    print("Executing forward_operator...")
    try:
        actual_output = forward_operator(*args, **kwargs)
    except Exception as e:
        traceback.print_exc()
        print(f"FATAL: Execution of forward_operator failed. Error: {e}")
        sys.exit(1)

    # 6. Verification
    print("Verifying output...")
    try:
        # We might need to move tensors to CPU for comparison if they aren't already
        if isinstance(actual_output, torch.Tensor):
            actual_output = actual_output.cpu()
        if isinstance(expected_output, torch.Tensor):
            expected_output = expected_output.cpu()

        passed, msg = recursive_check(expected_output, actual_output)
    except Exception as e:
        traceback.print_exc()
        print(f"FATAL: Verification logic failed. Error: {e}")
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()