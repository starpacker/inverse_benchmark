import sys
import os
import dill
import numpy as np
import traceback
import xml.etree.ElementTree as ET

# Add current directory to path so we can import the agent
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_load_and_preprocess_data import load_and_preprocess_data
except ImportError:
    print("Error: Could not import 'load_and_preprocess_data' from 'agent_load_and_preprocess_data.py'")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    print("Error: Could not import 'recursive_check' from 'verification_utils'")
    sys.exit(1)

def run_test():
    # 1. Configuration
    data_paths = ['/data/yjh/oct-cbort-main_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    outer_path = None
    inner_paths = []

    # 2. Sort paths into Outer (Factory creation) and Inner (Closure execution)
    for p in data_paths:
        if "parent_function" in p:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("Error: No outer data file found (standard_data_load_and_preprocess_data.pkl).")
        sys.exit(1)

    print(f"Loading outer data from: {outer_path}")

    # 3. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file {outer_path}: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    # 4. Execute Target Function (Scenario A or B Phase 1)
    print("Executing load_and_preprocess_data with outer arguments...")
    try:
        actual_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
    except Exception as e:
        print("Error during execution of load_and_preprocess_data:")
        traceback.print_exc()
        sys.exit(1)

    # 5. Determine Verification Strategy
    # The provided data path list only contains the outer file.
    # We check if the result is callable (Scenario B - Factory) or data (Scenario A - Direct).
    # Based on the provided code, load_and_preprocess_data returns (raw_frame_ch1, raw_frame_ch2, settings), 
    # which is NOT callable. This is Scenario A (Simple Function).

    # Scenario B check (Closure) - Just in case logic changes in future or implicit requirements
    if inner_paths:
        print(f"Found inner data paths, implying Factory pattern. {len(inner_paths)} files found.")
        if not callable(actual_result):
            print("Error: Inner data paths exist, expecting a callable (operator) from outer function, but got data.")
            sys.exit(1)
            
        # If we had inner paths, we would loop through them here.
        # Since the provided list only has the outer path, this block is skipped in this specific run.
        # But for robustness:
        for inner_p in inner_paths:
            print(f"Testing inner data: {inner_p}")
            with open(inner_p, 'rb') as f:
                inner_data = dill.load(f)
            
            i_args = inner_data.get('args', [])
            i_kwargs = inner_data.get('kwargs', {})
            i_expected = inner_data.get('output')
            
            try:
                i_actual = actual_result(*i_args, **i_kwargs)
            except Exception as e:
                print(f"Error executing inner closure for {inner_p}:")
                traceback.print_exc()
                sys.exit(1)
                
            passed, msg = recursive_check(i_expected, i_actual)
            if not passed:
                print(f"Verification FAILED for inner file {inner_p}: {msg}")
                sys.exit(1)
        
        print("TEST PASSED (Factory Pattern)")
        sys.exit(0)

    # Scenario A: Direct Data Verification
    print("Verifying direct output...")
    passed, msg = recursive_check(expected_outer_output, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()