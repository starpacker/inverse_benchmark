import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.getcwd())

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def run_test():
    # Data paths provided in the prompt
    data_paths = ['/data/yjh/caustics-main_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']

    # Identify files
    outer_data_path = None
    inner_data_path = None

    for path in data_paths:
        if "parent_function" in path:
            inner_data_path = path
        elif "standard_data_load_and_preprocess_data.pkl" in path:
            outer_data_path = path

    if not outer_data_path:
        print("Error: Standard outer data file not found in paths.")
        sys.exit(1)

    # Load outer data (Args to create the operator/result)
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data file: {e}")
        sys.exit(1)

    print(f"Loaded outer data from {outer_data_path}")
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    # Phase 1: Execute target function
    try:
        print("Executing load_and_preprocess_data with outer args...")
        # We need to ensure args are on the correct device if they are tensors, 
        # but here the arg is likely just 'device'.
        # The function signature is `load_and_preprocess_data(device: torch.device)`
        
        # Check if we need to map device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Since we are running in a possibly different environment, we might want to override the device arg
        # or just pass the args as is if they are compatible.
        # Looking at signature: def load_and_preprocess_data(device: torch.device)
        # We will override the first arg to be the current available device to ensure robustness
        
        if len(outer_args) > 0 and isinstance(outer_args[0], torch.device):
             outer_args = (device,) + outer_args[1:]
        elif 'device' in outer_kwargs:
             outer_kwargs['device'] = device
        
        # NOTE: If outer_args is empty, we might need to manually pass device if it was passed as positional
        # However, usually the dill capture captures the exact args used. 
        # If the recorded run used 'cuda:0' and this machine has it, fine. If not, might fail.
        # Let's trust the input or force it if it fails.
        
        # Let's try to inspect arguments.
        # If the first argument is a device, we replace it with the current valid device.
        new_outer_args = []
        for arg in outer_args:
            if isinstance(arg, torch.device):
                new_outer_args.append(device)
            else:
                new_outer_args.append(arg)
        outer_args = tuple(new_outer_args)
        
        if 'device' in outer_kwargs:
             outer_kwargs['device'] = device

        actual_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Verification Strategy
    if inner_data_path:
        # Scenario B: The function returned a callable (Closure/Factory pattern)
        print("Scenario B detected: Testing inner operator execution.")
        
        try:
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"Error loading inner data: {e}")
            sys.exit(1)
            
        inner_args = inner_data.get('args', [])
        inner_kwargs = inner_data.get('kwargs', {})
        expected_inner_output = inner_data.get('output', None)

        if not callable(actual_result):
            print(f"Error: Expected a callable operator, but got {type(actual_result)}")
            sys.exit(1)

        try:
            print("Executing inner operator...")
            final_result = actual_result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"Inner operator execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
            
        # Compare against inner output
        passed, msg = recursive_check(expected_inner_output, final_result)
        
    else:
        # Scenario A: Simple execution (The function returns data directly)
        print("Scenario A detected: Verifying direct output.")
        # Compare against outer output
        # Note: The output contains Tensors. recursive_check handles tolerances.
        
        # Ensure expected output is moved to the same device for comparison if they are tensors
        def move_to_device(obj, dev):
            if isinstance(obj, torch.Tensor):
                return obj.to(dev)
            if isinstance(obj, (list, tuple)):
                return type(obj)(move_to_device(x, dev) for x in obj)
            return obj
            
        expected_outer_output = move_to_device(expected_outer_output, device)
        final_result = actual_result
        
        passed, msg = recursive_check(expected_outer_output, final_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()