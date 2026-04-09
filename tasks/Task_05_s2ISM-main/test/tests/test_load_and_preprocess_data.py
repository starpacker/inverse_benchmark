import sys
import os
import dill
import numpy as np
import traceback
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

# --- Helpers to ensure dill loads data correctly ---
import torch
import functools
import inspect
import json
import random
from scipy.special import jv, kl_div
from scipy.stats import pearsonr
from scipy.signal import argrelmin, argrelmax
import brighteyes_ism.simulation.PSF_sim as psf_sim
import brighteyes_ism.simulation.Tubulin_sim as st
import brighteyes_ism.analysis.Tools_lib as tools

# Mock or ensure helpers used during data generation are present
# This block is critical if the dill file contains references to these functions
def _fix_seeds_(seed=42):
    if np:
        np.random.seed(seed)
    random.seed(seed)
    if torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

# Ensure seeds are fixed exactly as they were likely fixed during generation
# However, the previous error showed a large mismatch in the phantom generation (tubulin).
# Tubulin simulation often uses random processes. If the random state isn't identical,
# the phantom will differ significantly.
_fix_seeds_(42)

def main():
    data_paths = ['/data/yjh/s2ISM-main_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # 1. Identify File Strategy
    outer_path = None
    inner_path = None
    
    for p in data_paths:
        if 'parent_function' in p:
            inner_path = p
        elif 'load_and_preprocess_data.pkl' in p:
            outer_path = p

    if not outer_path:
        print("TEST FAILED: Standard data file not found.")
        sys.exit(1)

    print(f"Outer Data: {outer_path}")
    print(f"Inner Data: {inner_path}")

    try:
        # 2. Load Outer Data
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)

        # 3. Execution Strategy
        if inner_path:
            # Scenario B: Factory Pattern
            print("Scenario B: Factory pattern detected.")
            # Step 1: Create the operator
            print("Creating agent operator...")
            _fix_seeds_(42) # Reset seed before first call
            agent_operator = load_and_preprocess_data(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"TEST FAILED: Expected a callable (factory), got {type(agent_operator)}")
                sys.exit(1)
                
            # Step 2: Load inner args
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output', None)
            
            # Step 3: Execute inner
            print("Executing inner function...")
            _fix_seeds_(42) # Reset seed might be needed if inner uses randomness, though less likely for inner closures usually
            actual_output = agent_operator(*inner_args, **inner_kwargs)
            
        else:
            # Scenario A: Direct Execution
            print("Scenario A: Direct function execution verification.")
            _fix_seeds_(42) # Reset seed immediately before execution
            actual_output = load_and_preprocess_data(*outer_args, **outer_kwargs)

        # 4. Verification
        # NOTE: phTub (output[0]) is generated stochastically. 
        # While we fix seeds, different environments or library versions might yield different random streams.
        # Ideally, we check equality. If strict equality fails for the phantom, we might need a looser check 
        # or acknowledge that stochastic tests are fragile without exact environment replication.
        # For now, we rely on recursive_check with standard tolerance.
        
        print("Verifying results...")
        passed, msg = recursive_check(expected_output, actual_output)

        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            # If it's the phantom (index 0) failing significantly, it confirms randomness issues.
            # However, the requirement is to fail if it doesn't match the recorded data.
            sys.exit(1)

    except Exception as e:
        print(f"TEST FAILED: Execution error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()