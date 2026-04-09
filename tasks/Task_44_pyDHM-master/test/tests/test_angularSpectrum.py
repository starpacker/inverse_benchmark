import sys
import os
import dill
import numpy as np
import traceback
import torch

# Import the target function
from agent_angularSpectrum import angularSpectrum
from verification_utils import recursive_check

def load_data(file_path):
    """Robust data loading helper."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"Data file is empty (0 bytes): {file_path}")
    
    with open(file_path, "rb") as f:
        try:
            return dill.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to unpickle data from {file_path}. Error: {e}")

def run_test():
    # 1. DATA FILE ANALYSIS
    # Only standard_data_angularSpectrum.pkl is expected based on the function signature (Scenario A).
    # However, we implement logic to support Scenario B if the data dictates it.
    
    data_paths = ['/data/yjh/pyDHM-master_sandbox/run_code/std_data/standard_data_angularSpectrum.pkl']
    
    outer_path = None
    inner_path = None

    for path in data_paths:
        if "standard_data_angularSpectrum.pkl" in path:
            outer_path = path
        elif "parent_function" in path and "angularSpectrum" in path:
            inner_path = path

    if not outer_path:
        print("CRITICAL: Primary data file (standard_data_angularSpectrum.pkl) not found in paths.")
        sys.exit(1)

    print(f"Primary data file: {outer_path}")
    print(f"Secondary data file (if factory pattern): {inner_path}")

    # 2. PHASE 1: RECONSTRUCT OPERATOR / EXECUTE FUNCTION
    try:
        outer_data = load_data(outer_path)
    except Exception as e:
        print(f"Error: {e}")
        print("CRITICAL: Failed to load primary data. Cannot proceed.")
        sys.exit(1)

    outer_args = outer_data.get("args", [])
    outer_kwargs = outer_data.get("kwargs", {})
    
    # We need to determine if this is Scenario A (Result is output) or B (Result is operator)
    # Based on the provided code, angularSpectrum returns a numpy array (Scenario A).
    
    print("Executing angularSpectrum with primary arguments...")
    try:
        # If the function expects numpy arrays but got lists (common in JSON/pickle), convert them if necessary.
        # The function internal logic does `field = np.array(field)`, so it handles lists/arrays gracefully.
        phase1_result = angularSpectrum(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution failed during Phase 1: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. PHASE 2: VERIFICATION STRATEGY
    final_result = None
    expected_output = None

    # Check if the result implies a Factory Pattern (Scenario B)
    if callable(phase1_result) and not isinstance(phase1_result, (np.ndarray, torch.Tensor)):
        print("Detected Factory Pattern: Phase 1 returned a callable.")
        if not inner_path:
            print("CRITICAL: Factory pattern detected but no inner data file (parent_function) provided.")
            sys.exit(1)
            
        # Load Inner Data
        try:
            inner_data = load_data(inner_path)
        except Exception as e:
            print(f"Error loading inner data: {e}")
            sys.exit(1)
            
        inner_args = inner_data.get("args", [])
        inner_kwargs = inner_data.get("kwargs", {})
        expected_output = inner_data.get("output")
        
        print("Executing Inner Operator (Phase 2)...")
        try:
            final_result = phase1_result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"Execution failed during Phase 2 (Inner Operator): {e}")
            traceback.print_exc()
            sys.exit(1)
            
    else:
        # Scenario A: The function returned the final result directly
        print("Detected Direct Execution: Phase 1 returned data (not a callable).")
        final_result = phase1_result
        expected_output = outer_data.get("output")
        
        # If inner_path exists but we aren't using it, warn
        if inner_path:
            print("Warning: Inner data file found but not used (Function did not return a callable).")

    # 4. COMPARISON
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_output, final_result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"Verification process crashed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()