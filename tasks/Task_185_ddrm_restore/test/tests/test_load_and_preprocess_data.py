import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for load_and_preprocess_data."""
    
    # Data paths provided
    data_paths = ['/data/yjh/ddrm_restore_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    # Verify outer path exists
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    print(f"[TEST] Loading outer data from: {outer_path}")
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"[TEST] Outer data loaded successfully")
    print(f"[TEST] Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"[TEST] Args: {outer_args}")
    print(f"[TEST] Kwargs: {outer_kwargs}")
    
    # Execute the function to get the result/operator
    try:
        print("[TEST] Executing load_and_preprocess_data...")
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("[TEST] Function executed successfully")
    except Exception as e:
        print(f"ERROR: Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if we're in Scenario A or B
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("[TEST] Detected Factory/Closure pattern (Scenario B)")
        
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from load_and_preprocess_data, got {type(agent_result)}")
            sys.exit(1)
        
        # Process each inner path
        for inner_path in inner_paths:
            if not os.path.exists(inner_path):
                print(f"WARNING: Inner data file does not exist: {inner_path}")
                continue
            
            print(f"[TEST] Loading inner data from: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"[TEST] Inner function name: {inner_data.get('func_name', 'unknown')}")
            
            # Execute the operator with inner args
            try:
                print("[TEST] Executing operator with inner args...")
                result = agent_result(*inner_args, **inner_kwargs)
                print("[TEST] Operator executed successfully")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify result
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"VERIFICATION FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"[TEST] Inner verification passed: {msg}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("[TEST] Detected Simple Function pattern (Scenario A)")
        
        result = agent_result
        expected = outer_output
        
        # Verify result
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"VERIFICATION FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"[TEST] Verification passed: {msg}")
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()