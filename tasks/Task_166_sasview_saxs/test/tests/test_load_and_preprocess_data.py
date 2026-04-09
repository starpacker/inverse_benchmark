import sys
import os
import dill
import traceback

# Add the necessary path for imports
sys.path.insert(0, '/data/yjh/sasview_saxs_sandbox_sandbox/run_code')

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def main():
    """Main test function for load_and_preprocess_data."""
    
    data_paths = ['/data/yjh/sasview_saxs_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"[ERROR] Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if this is an inner path (contains parent_function pattern)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("[ERROR] Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    print(f"[INFO] Outer path: {outer_path}")
    print(f"[INFO] Inner paths: {inner_paths}")
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        print("[INFO] Loading outer data...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"[INFO] Outer args: {outer_args}")
        print(f"[INFO] Outer kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("[INFO] Executing load_and_preprocess_data...")
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print(f"[INFO] Function executed successfully, result type: {type(agent_result)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Detected Scenario B: Factory/Closure pattern")
        
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"[ERROR] Expected callable from outer function, got {type(agent_result)}")
            sys.exit(1)
        
        agent_operator = agent_result
        
        for inner_path in inner_paths:
            try:
                print(f"[INFO] Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                print(f"[INFO] Inner args: {inner_args}")
                print(f"[INFO] Inner kwargs: {inner_kwargs}")
                
                # Execute the operator with inner args
                print("[INFO] Executing operator with inner args...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                print("[INFO] Comparing results...")
                passed, msg = recursive_check(expected_output, actual_result)
                
                if not passed:
                    print(f"[FAILED] Verification failed for {inner_path}")
                    print(f"[FAILED] Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"[PASSED] Verification passed for {inner_path}")
                    
            except Exception as e:
                print(f"[ERROR] Failed processing inner path {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("[INFO] Detected Scenario A: Simple function")
        
        expected_output = outer_output
        actual_result = agent_result
        
        # Compare results
        print("[INFO] Comparing results...")
        try:
            passed, msg = recursive_check(expected_output, actual_result)
            
            if not passed:
                print(f"[FAILED] Verification failed")
                print(f"[FAILED] Message: {msg}")
                sys.exit(1)
            else:
                print("[PASSED] Verification passed")
                
        except Exception as e:
            print(f"[ERROR] Failed during verification: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()