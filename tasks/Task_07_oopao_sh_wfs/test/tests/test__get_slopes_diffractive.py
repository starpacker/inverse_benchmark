import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent__get_slopes_diffractive import _get_slopes_diffractive

# Import verification utility
from verification_utils import recursive_check


def main():
    """
    Test script for _get_slopes_diffractive function.
    Handles both simple function and factory/closure patterns.
    """
    # Define data paths
    data_paths = ['/home/yjh/oopao_sh_sandbox/run_code/std_data/standard_data__get_slopes_diffractive.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        # Check if this is an inner path (contains parent_function pattern)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data__get_slopes_diffractive.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data__get_slopes_diffractive.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and reconstruct operator/result
    try:
        print("\n[Phase 1] Loading outer data...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"  Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"  Number of args: {len(outer_args)}")
        print(f"  Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the target function
    try:
        print("\n[Phase 1] Executing _get_slopes_diffractive...")
        agent_result = _get_slopes_diffractive(*outer_args, **outer_kwargs)
        print(f"  Execution successful")
        
        # Check if result is callable (factory pattern)
        is_callable = callable(agent_result) and not isinstance(agent_result, (np.ndarray, type))
        print(f"  Result is callable: {is_callable}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute _get_slopes_diffractive: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execution & Verification
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("\n[Phase 2] Detected factory/closure pattern (inner data exists)")
        
        for inner_path in inner_paths:
            try:
                print(f"\n  Loading inner data from: {os.path.basename(inner_path)}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"    Inner function name: {inner_data.get('func_name', 'unknown')}")
                print(f"    Number of inner args: {len(inner_args)}")
                print(f"    Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner arguments
                if not callable(agent_result):
                    print("ERROR: Agent result is not callable but inner data exists")
                    sys.exit(1)
                
                print("  Executing agent operator with inner arguments...")
                actual_result = agent_result(*inner_args, **inner_kwargs)
                
                # Verify
                print("  Verifying results...")
                passed, msg = recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Verification passed for {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR: Failed during inner execution/verification: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function (no inner data)
        print("\n[Phase 2] Simple function pattern (no inner data)")
        
        try:
            expected = outer_output
            actual_result = agent_result
            
            print("  Verifying results against outer data output...")
            passed, msg = recursive_check(expected, actual_result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("  Verification passed")
                
        except Exception as e:
            print(f"ERROR: Failed during verification: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()