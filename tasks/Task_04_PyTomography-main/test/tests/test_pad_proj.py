import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the module can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_pad_proj import pad_proj
except ImportError:
    print("Error: Could not import 'pad_proj' from 'agent_pad_proj.py'")
    sys.exit(1)

from verification_utils import recursive_check

def test_pad_proj():
    # Defined data paths
    data_paths = ['/data/yjh/PyTomography-main_sandbox/run_code/std_data/standard_data_pad_proj.pkl']

    # 1. File Logic Setup
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if 'parent_function' in p:
            inner_paths.append(p)
        else:
            # We assume the file ending in _pad_proj.pkl without parent_function is the outer/main call
            if p.endswith('standard_data_pad_proj.pkl'):
                outer_path = p
    
    # Fallback if specific filtering logic misses the single file provided
    if outer_path is None and len(data_paths) == 1 and not inner_paths:
        outer_path = data_paths[0]

    if not outer_path:
        print("Error: Could not identify the main data file (standard_data_pad_proj.pkl).")
        sys.exit(1)

    # 2. Phase 1: Reconstruct Operator / Execute Main Function
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
            
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        print("Executing pad_proj with loaded arguments...")
        # Execute the primary function
        result_phase_1 = pad_proj(*outer_args, **outer_kwargs)
        
    except Exception as e:
        traceback.print_exc()
        print(f"CRITICAL ERROR during Phase 1 execution: {e}")
        sys.exit(1)

    # 3. Phase 2: Execution & Verification
    # Scenario B (Factory Pattern) - If inner files exist, Phase 1 result is an operator
    if inner_paths:
        print(f"Scenario B detected: {len(inner_paths)} inner files found. Treating Phase 1 result as an operator.")
        
        if not callable(result_phase_1):
            print(f"Error: Inner files exist, but pad_proj returned {type(result_phase_1)} instead of a callable.")
            sys.exit(1)

        for inner_path in inner_paths:
            print(f"  Verifying against inner file: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                # Execute the operator returned in Phase 1
                actual_result = result_phase_1(*inner_args, **inner_kwargs)
                
                # Compare
                passed, msg = recursive_check(expected_output, actual_result)
                if not passed:
                    print(f"  FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"  PASSED: Inner verification successful.")
                    
            except Exception as e:
                traceback.print_exc()
                print(f"Error during inner execution for {inner_path}: {e}")
                sys.exit(1)

    # Scenario A (Simple Function) - No inner files, Phase 1 result is the final output
    else:
        print("Scenario A detected: No inner files. Comparing Phase 1 result directly.")
        expected_output = outer_data.get('output')
        
        passed, msg = recursive_check(expected_output, result_phase_1)
        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    test_pad_proj()