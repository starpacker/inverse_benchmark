import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/ddrm_restore_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    print(f"[INFO] Outer data path: {outer_path}")
    print(f"[INFO] Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and execute function
    try:
        print("[PHASE 1] Loading outer data...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"[INFO] Outer args count: {len(outer_args)}")
        print(f"[INFO] Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("[PHASE 1] Executing evaluate_results...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print(f"[INFO] Function returned type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is a factory pattern (Scenario B) or simple function (Scenario A)
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("[INFO] Detected factory/closure pattern (Scenario B)")
        
        # The result should be callable
        if not callable(result):
            print(f"ERROR: Expected callable result for factory pattern, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Phase 2: Load inner data and execute the operator
        for inner_path in inner_paths:
            try:
                print(f"[PHASE 2] Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"[INFO] Inner args count: {len(inner_args)}")
                print(f"[INFO] Inner kwargs keys: {list(inner_kwargs.keys())}")
                
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                print("[PHASE 2] Executing agent operator...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"[INFO] Operator returned type: {type(actual_result)}")
                
            except Exception as e:
                print(f"ERROR: Failed to execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verification
            try:
                print("[VERIFICATION] Comparing results...")
                passed, msg = recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"[INFO] Inner test passed: {msg}")
                    
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple Function
        print("[INFO] Detected simple function pattern (Scenario A)")
        
        expected = outer_output
        
        # Verification
        try:
            print("[VERIFICATION] Comparing results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"TEST PASSED: {msg}")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()