import sys
import os
import dill
import torch
import numpy as np
import traceback
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

# --- Dependency Injection for Dill Deserialization ---
# We must redefine classes here exactly as they appear in the source 
# so dill can reconstruct objects if they are instances of these classes in the pickle.

KEY_FINGERPRINTS = 'fingerprints'
KEY_MR_PARAMS = 'mr_params'
ID_MAP_FF = 'FFmap'
ID_MAP_T1H2O = 'T1H2Omap'
ID_MAP_T1FAT = 'T1FATmap'
ID_MAP_B0 = 'B0map'
ID_MAP_B1 = 'B1map'
MR_PARAMS = (ID_MAP_FF, ID_MAP_T1H2O, ID_MAP_T1FAT, ID_MAP_B0, ID_MAP_B1)
FILE_NAME_FINGERPRINTS = 'fingerprints.npy'
FILE_NAME_PARAMETERS = 'parameters.npy'
FILE_NAME_PARAMETERS_MIN = 'parameters_mins.pkl'
FILE_NAME_PARAMETERS_MAX = 'parameters_maxs.pkl'

class NumpyMRFDataset(data.Dataset):
    def __init__(self, dataset_dir: str, index_selection: list=None, transform=None) -> None:
        super().__init__()
        # Implementation details omitted for brevity as they are just needed for type resolution
        pass

class InvertibleModule(nn.Module):
    def forward(self, x, rev=False): pass

class SequenceINN(InvertibleModule):
    def __init__(self, *modules): super().__init__()
    def forward(self, x, rev=False): pass

class F_fully_connected_small(nn.Module):
    def __init__(self, size_in, size, internal_size=None, dropout=0.0): super().__init__()
    def forward(self, x): pass

class RNVPCouplingBlock(InvertibleModule):
    def __init__(self, dims_in, subnet_constructor): super().__init__()
    def forward(self, x, rev=False): pass

class PermuteRandom(InvertibleModule):
    def __init__(self, dims_in, seed=None): super().__init__()
    def forward(self, x, rev=False): pass

# Inject these into global scope for dill
globals()['NumpyMRFDataset'] = NumpyMRFDataset
globals()['SequenceINN'] = SequenceINN
globals()['InvertibleModule'] = InvertibleModule
globals()['F_fully_connected_small'] = F_fully_connected_small
globals()['RNVPCouplingBlock'] = RNVPCouplingBlock
globals()['PermuteRandom'] = PermuteRandom

def run_test():
    # 1. Setup Data Paths
    data_paths = ['/data/yjh/mrf-reconstruction-mlmir2020-master_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    outer_path = None
    inner_path = None

    for path in data_paths:
        if 'parent_function' in path:
            inner_path = path
        elif 'standard_data_load_and_preprocess_data.pkl' in path:
            outer_path = path

    if not outer_path:
        print("Test Skipped: No standard input data found (standard_data_load_and_preprocess_data.pkl).")
        sys.exit(0)

    # 2. Load Outer Data (Arguments for the function itself)
    print(f"Loading data from {outer_path}...")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"Test Failed: Corrupt pickle file. {e}")
        print("The recording process likely failed to write the full file.")
        sys.exit(1)
    except Exception as e:
        print(f"Test Failed: Error loading pickle file. {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Execution Strategy
    # This function returns multiple objects (dataloader, model, optimizer, etc.)
    # It is a direct execution scenario (Scenario A), not a factory pattern returning a closure.
    
    print("Executing load_and_preprocess_data...")
    try:
        # Extract arguments
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        
        # Run the function
        actual_result = load_and_preprocess_data(*args, **kwargs)
        
        expected_result = outer_data.get('output')
    except Exception as e:
        print(f"Test Failed: Execution error in load_and_preprocess_data. {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    print("Verifying results...")
    
    # Special Handling for Complex Return Types (Tuple of DL, Model, Optim, Dict, Dataset, Device)
    # We cannot strictly compare memory addresses or deep object states for things like Dataloaders/Optimizers
    # easily via equality. We will check structure and key properties.
    
    try:
        passed = True
        msg = ""

        # Check tuple length
        if len(actual_result) != len(expected_result):
            passed = False
            msg = f"Length mismatch: Expected {len(expected_result)}, got {len(actual_result)}"
        else:
            # Unpack for clarity: (dataloader, model, optimizer, dims, dataset, device)
            act_dl, act_model, act_optim, act_dims, act_ds, act_dev = actual_result
            exp_dl, exp_model, exp_optim, exp_dims, exp_ds, exp_dev = expected_result

            # 1. Check Dimensions Dictionary (Critical Logic)
            res_dims, msg_dims = recursive_check(exp_dims, act_dims)
            if not res_dims:
                passed = False
                msg += f"\nDimensions mismatch: {msg_dims}"

            # 2. Check Device
            # Note: Device might differ if test env has no GPU but recorded env did.
            # We usually relax this check or just ensure it's a torch.device
            if str(act_dev) != str(exp_dev):
                print(f"Warning: Device mismatch (Expected {exp_dev}, Got {act_dev}). Ignoring if hardware differs.")

            # 3. Check Model Architecture (Basic check)
            if str(act_model) != str(exp_model):
                # String representation of model architecture should match
                passed = False
                msg += f"\nModel architecture mismatch."

            # 4. Check Dataset Length
            if len(act_ds) != len(exp_ds):
                passed = False
                msg += f"\nDataset length mismatch: Expected {len(exp_ds)}, Got {len(act_ds)}"

        if not passed:
            print(f"Test Failed: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"Test Failed: Verification logic error. {e}")
        traceback.print_exc()
        sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()