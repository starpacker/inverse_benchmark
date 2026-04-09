import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add current directory to path to ensure local imports work
sys.path.append(os.getcwd())

try:
    from agent_interp2d import interp2d
    from verification_utils import recursive_check
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_interp2d():
    # 1. DATA FILE ANALYSIS
    data_paths = ['/data/yjh/caustics-main_sandbox/run_code/std_data/standard_data_interp2d.pkl']
    
    outer_path = None
    inner_path = None
    
    for p in data_paths:
        if "standard_data_interp2d.pkl" in p:
            outer_path = p
        elif "parent_function" in p and "interp2d" in p:
            inner_path = p

    if not outer_path:
        print("CRITICAL: Outer data file 'standard_data_interp2d.pkl' not found in paths.")
        sys.exit(1)

    # 2. LOAD OUTER DATA
    print(f"Loading outer data from {outer_path}...")
    try:
        with open(outer_path, "rb") as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load outer pickle: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    outer_expected = outer_data.get('output', None)

    # 3. PHASE 1: RECONSTRUCT / EXECUTE OPERATOR
    print("Executing interp2d with outer arguments...")
    try:
        # Check if GPU is available and move args if necessary (optional robustness)
        # Assuming args match the environment or torch handles it.
        phase1_result = interp2d(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution of interp2d failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. PHASE 2: EXECUTION & VERIFICATION
    final_result = None
    expected_result = None

    if inner_path:
        # Scenario B: Factory/Closure Pattern
        print(f"Inner path detected: {inner_path}. Treating Phase 1 result as callable operator.")
        
        if not callable(phase1_result):
            print(f"Error: Expected callable from outer execution, got {type(phase1_result)}")
            sys.exit(1)
            
        print(f"Loading inner data from {inner_path}...")
        try:
            with open(inner_path, "rb") as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"Failed to load inner pickle: {e}")
            sys.exit(1)
            
        inner_args = inner_data.get('args', [])
        inner_kwargs = inner_data.get('kwargs', {})
        expected_result = inner_data.get('output', None)
        
        print("Executing operator with inner arguments...")
        try:
            final_result = phase1_result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"Execution of inner operator failed: {e}")
            traceback.print_exc()
            sys.exit(1)
            
    else:
        # Scenario A: Simple Function
        print("No inner path detected. Treating Phase 1 result as final output.")
        final_result = phase1_result
        expected_result = outer_expected

    # 5. COMPARISON
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_result, final_result)
    except Exception as e:
        print(f"Verification process failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_interp2d()