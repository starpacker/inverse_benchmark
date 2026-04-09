import sys
import os
import dill
import traceback
import numpy as np

# Handle optional torch dependency to prevent ModuleNotFoundError
try:
    import torch
except ImportError:
    torch = None

# Ensure the current directory is in sys.path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_forward_operator import forward_operator
    from verification_utils import recursive_check
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_forward_operator():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/storm-analysis-master_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # 2. Logic to Separate Outer (Factory/Function) and Inner (Closure execution) data
    outer_path = None
    inner_paths = []
    
    target_func_name = "forward_operator"
    
    for p in data_paths:
        if f"standard_data_{target_func_name}.pkl" in p:
            outer_path = p
        elif f"standard_data_parent_function_{target_func_name}_" in p:
            inner_paths.append(p)
            
    if not outer_path:
        print(f"Skipping test: No outer data file found for {target_func_name}")
        # If no data is present, we often treat this as a pass or skip in CI, 
        # but here we'll exit cleanly.
        sys.exit(0)

    print(f"Loading Outer Data from: {outer_path}")
    try:
        with open(outer_path, "rb") as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Phase 1: Run the Outer Function (The Agent)
    print("Executing Phase 1: Reconstructing/Running Operator...")
    try:
        outer_args = outer_data.get("args", [])
        outer_kwargs = outer_data.get("kwargs", {})
        
        # Execute the function
        # Note: If forward_operator is a factory, this returns the closure.
        # If it's a standard function, this returns the final result.
        result_phase_1 = forward_operator(*outer_args, **outer_kwargs)
        
    except Exception as e:
        print(f"Phase 1 Execution Failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Phase 2: Execution & Verification
    if inner_paths:
        # Scenario B: Factory Pattern
        # result_phase_1 is expected to be a callable (the closure)
        if not callable(result_phase_1):
            print(f"Error: Expected a callable from Phase 1 due to presence of inner data, but got {type(result_phase_1)}")
            sys.exit(1)
            
        print(f"Phase 1 successful. Found {len(inner_paths)} inner data files. Testing closures...")
        
        for i_path in inner_paths:
            print(f"  Testing inner file: {os.path.basename(i_path)}")
            try:
                with open(i_path, "rb") as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get("args", [])
                inner_kwargs = inner_data.get("kwargs", {})
                expected_inner = inner_data.get("output")
                
                # Execute closure
                actual_inner = result_phase_1(*inner_args, **inner_kwargs)
                
                # Verify
                passed, msg = recursive_check(expected_inner, actual_inner)
                if not passed:
                    print(f"  FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("  Passed.")
                    
            except Exception as e:
                print(f"  Inner Execution Failed for {i_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
                
    else:
        # Scenario A: Simple Function
        # result_phase_1 is the actual result
        print("Phase 1 successful. No inner data found (Standard Function mode). Verifying result...")
        expected_outer = outer_data.get("output")
        
        passed, msg = recursive_check(expected_outer, result_phase_1)
        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")

if __name__ == "__main__":
    test_forward_operator()