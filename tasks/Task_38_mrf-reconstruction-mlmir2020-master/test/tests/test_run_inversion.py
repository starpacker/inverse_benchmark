import sys
import os
import dill
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import traceback

# -------------------------------------------------------------------------
# 1. INJECT TARGET FUNCTION
# -------------------------------------------------------------------------
from agent_run_inversion import run_inversion

# -------------------------------------------------------------------------
# 2. INJECT REFEREE (EVALUATION LOGIC)
# -------------------------------------------------------------------------
KEY_FINGERPRINTS = 'fingerprints'
KEY_MR_PARAMS = 'mr_params'

ID_MAP_FF = 'FFmap'
ID_MAP_T1H2O = 'T1H2Omap'
ID_MAP_T1FAT = 'T1FATmap'
ID_MAP_B0 = 'B0map'
ID_MAP_B1 = 'B1map'

MR_PARAMS = (ID_MAP_FF, ID_MAP_T1H2O, ID_MAP_T1FAT, ID_MAP_B0, ID_MAP_B1)

def de_normalize(data: np.ndarray, minmax_tuple: tuple):
    return data * (minmax_tuple[1] - minmax_tuple[0]) + minmax_tuple[0]

def de_normalize_mr_parameters(data: np.ndarray, mr_param_ranges, mr_params=MR_PARAMS):
    data_de_normalized = data.copy()
    for idx, mr_param in enumerate(mr_params):
        if mr_param in mr_param_ranges:
             data_de_normalized[:, idx] = de_normalize(data[:, idx], mr_param_ranges[mr_param])
    return data_de_normalized

def forward_operator(model, x, ndim_y, device):
    """
    Simulates the forward Bloch process (learning-based).
    Maps parameters x -> fingerprint y.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()
    
    x = x.to(device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
        
    current_bs = x.size(0)
    ndim_x = x.size(1)
    
    pad_len = ndim_y - ndim_x
    if pad_len < 0:
        raise ValueError("Parameter dimension cannot be larger than fingerprint dimension for this INN architecture.")
        
    pad_x = torch.zeros(current_bs, pad_len, device=device)
    x_padded = torch.cat((x, pad_x), dim=1)
    
    # Forward pass through INN (x -> y)
    y_pred = model(x_padded, rev=False)
    
    return y_pred

# Note: The evaluation function in the prompt reference calls 'run_inversion' internally.
# However, for this test script, we are testing 'run_inversion' itself.
# We will adapt 'evaluate_results' to serve as a checker for the OUTPUT of run_inversion,
# or we will use it to validate the ecosystem.
# 
# Looking at the provided `evaluate_results` signature:
# evaluate_results(model, dataloader, dataset, dims, device, epochs)
# It trains a model and then runs inversion.
#
# Our `run_inversion` function is just the inference step.
# The data inputs for `run_inversion` captured by the decorator are likely:
# (model, y, ndim_x, device).
#
# The output of `run_inversion` is the reconstructed parameters `x_rec`.
# The standard output in the .pkl file is the ground truth `x_rec` produced by the original run.
#
# Evaluation Strategy:
# Since `run_inversion` is deterministic (inference), we should compare the Agent's result
# with the Standard result directly using MSE/L1, rather than retraining a whole model.
# Retraining (as seen in `evaluate_results`) is too heavy for a unit test of an inference function.
# However, we will use the logic from `evaluate_results` regarding denormalization and error printing.

def simple_evaluate_inversion(x_rec, x_gt_reference, dataset_info=None):
    """
    Compares the Agent's reconstruction against the Standard implementation's reconstruction.
    If dataset info is available (ranges), it tries to print physical values.
    """
    x_rec_np = x_rec.cpu().numpy() if isinstance(x_rec, torch.Tensor) else x_rec
    x_std_np = x_gt_reference.cpu().numpy() if isinstance(x_gt_reference, torch.Tensor) else x_gt_reference
    
    # Calculate MSE between Agent and Standard execution
    mse = np.mean((x_rec_np - x_std_np)**2)
    print(f"MSE between Agent and Standard Output: {mse:.8f}")
    
    # If this were a full pipeline test, we'd compare against Ground Truth labels (y -> x_GT).
    # But here we are validating code integrity: Agent Code vs Standard Code on same inputs.
    return mse

# -------------------------------------------------------------------------
# 3. TEST EXECUTION
# -------------------------------------------------------------------------
def main():
    data_paths = ['/data/yjh/mrf-reconstruction-mlmir2020-master_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # 1. Parse Data Paths
    outer_path = None
    inner_path = None
    
    for p in data_paths:
        if "parent_function" in p:
            inner_path = p
        else:
            outer_path = p

    if not outer_path:
        print("Error: Primary data file (standard_data_run_inversion.pkl) not found.")
        sys.exit(1)

    print(f"Loading primary data: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    # 2. Reconstruct Call
    # Pattern 1: Direct Execution
    if not inner_path:
        print("Execution Pattern: Direct Call")
        args = outer_data['args']
        kwargs = outer_data['kwargs']
        std_output = outer_data['output']
        
        try:
            # Run Agent
            agent_output = run_inversion(*args, **kwargs)
        except Exception as e:
            print("Agent execution failed.")
            traceback.print_exc()
            sys.exit(1)
            
        # 3. Evaluation
        # Since run_inversion is a deterministic inference function, 
        # we check if the agent output matches the recorded standard output.
        # We allow a tiny floating point tolerance.
        
        score_diff = simple_evaluate_inversion(agent_output, std_output)
        
        # Threshold: effectively 0 for deterministic code, but allowing for small float diffs
        threshold = 1e-5
        
        if score_diff > threshold:
            print(f"FAILURE: Agent output differs significantly from standard. MSE: {score_diff}")
            sys.exit(1)
        else:
            print("SUCCESS: Agent output matches standard output integrity.")
            sys.exit(0)

    else:
        # Pattern 2: Chained Execution (Factory/Closure)
        print("Execution Pattern: Chained (Factory/Closure)")
        print(f"Loading inner data: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
            
        outer_args = outer_data['args']
        outer_kwargs = outer_data['kwargs']
        
        inner_args = inner_data['args']
        inner_kwargs = inner_data['kwargs']
        std_output = inner_data['output']
        
        try:
            # 1. Run Outer to get Operator
            operator = run_inversion(*outer_args, **outer_kwargs)
            
            # 2. Run Operator with Inner args
            agent_output = operator(*inner_args, **inner_kwargs)
            
        except Exception as e:
            print("Agent execution failed in chained mode.")
            traceback.print_exc()
            sys.exit(1)

        # 3. Evaluation
        score_diff = simple_evaluate_inversion(agent_output, std_output)
        threshold = 1e-5
        
        if score_diff > threshold:
            print(f"FAILURE: Agent output differs significantly from standard. MSE: {score_diff}")
            sys.exit(1)
        else:
            print("SUCCESS: Agent output matches standard output integrity.")
            sys.exit(0)

if __name__ == "__main__":
    main()