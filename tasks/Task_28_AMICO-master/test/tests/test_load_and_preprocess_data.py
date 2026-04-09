import sys
import os
import dill
import numpy as np
import traceback
import nibabel as nib
import warnings
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

# --- CRITICAL: Define Helper Classes for Dill Deserialization ---
# The pickle file contains instances of 'Scheme'. We must define it here exactly
# as it was defined during data generation so dill can reconstruct it.
class Scheme:
    def __init__(self, filename, b0_thr=0):
        self.version = None
        if isinstance(filename, str):
            if not os.path.isfile(filename):
                raise FileNotFoundError(f'Scheme file "{filename}" not found')
            try:
                self.raw = np.loadtxt(filename)
            except Exception as e:
                raise RuntimeError(f'Could not load scheme file: {e}')
            with open(filename, 'r') as f:
                first_line = f.readline()
                if 'VERSION: STEJSKALTANNER' in first_line:
                    self.version = 1
                else:
                    self.version = 0
        else:
            self.raw = filename
            self.version = 0 if self.raw.shape[1] <= 4 else 1
        self.nS = self.raw.shape[0]
        if self.version == 0:
            self.b = self.raw[:, 3]
            self.g = np.ones(self.nS)
        else:
            self.g = self.raw[:, 3]
            self.Delta = self.raw[:, 4]
            self.delta = self.raw[:, 5]
            self.TE = self.raw[:, 6]
            gamma = 267598700.0
            self.b = (gamma * self.delta * self.g) ** 2 * (self.Delta - self.delta / 3.0) * 1e-06
        self.raw[:, :3] /= np.linalg.norm(self.raw[:, :3], axis=1)[:, None] + 1e-16
        self.b0_idx = np.where(self.b <= b0_thr)[0]
        self.dwi_idx = np.where(self.b > b0_thr)[0]
        self.b0_count = len(self.b0_idx)
        self.dwi_count = len(self.dwi_idx)
        b_rounded = np.round(self.b[self.dwi_idx], -2)
        unique_b = np.unique(b_rounded)
        self.shells = []
        for ub in unique_b:
            idx = self.dwi_idx[b_rounded == ub]
            shell = {'b': np.mean(self.b[idx]), 'idx': idx}
            if self.version == 1:
                shell['G'] = np.mean(self.g[idx])
                shell['Delta'] = np.mean(self.Delta[idx])
                shell['delta'] = np.mean(self.delta[idx])
                shell['TE'] = np.mean(self.TE[idx])
            shell['grad'] = self.raw[idx, :3]
            self.shells.append(shell)

# ----------------------------------------------------------------

def run_test():
    print("Starting test_load_and_preprocess_data.py...")
    
    data_paths = ['/data/yjh/AMICO-master_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # 1. Strategy: Identify file types
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if 'standard_data_load_and_preprocess_data.pkl' in p:
            outer_path = p
        elif 'parent_function_load_and_preprocess_data' in p:
            inner_paths.append(p)
    
    if not outer_path:
        print("CRITICAL ERROR: No standard_data_load_and_preprocess_data.pkl found.")
        sys.exit(1)

    print("Test Configuration:")
    print(f"  Outer Data: {outer_path}")
    print(f"  Inner Data: {inner_paths}")

    # 2. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except EOFError:
        print("CRITICAL ERROR: Outer data file is corrupted or incomplete (EOFError).")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load outer pickle: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # 3. Execute Target Function
    try:
        print("Executing load_and_preprocess_data with loaded args...")
        actual_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Scenario Determination
    # Since load_and_preprocess_data typically returns data tuples (not a function),
    # this is likely Scenario A (Simple Function).
    # However, if it returns a callable, we handle Scenario B.
    
    final_result = actual_result
    
    if callable(actual_result) and inner_paths:
        print("Detected Factory Pattern (Scenario B).")
        # Just take the first inner path for this test
        inner_path = inner_paths[0]
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output', None)
            
            print(f"Executing inner function from {inner_path}...")
            final_result = actual_result(*inner_args, **inner_kwargs)
            
        except Exception as e:
            print(f"Inner execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    elif callable(actual_result) and not inner_paths:
        print("Warning: Function returned a callable but no inner data found. Comparing callables directly (may fail).")
    else:
        print("Detected Simple Execution (Scenario A).")

    # 5. Verification
    print("Verifying results...")
    try:
        is_correct, msg = recursive_check(expected_output, final_result)
        if is_correct:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"Verification process error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()