import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent__plot_results import _plot_results
from verification_utils import recursive_check

def main():
    """Main test function for _plot_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/ezyrb_rom_sandbox_sandbox/run_code/std_data/standard_data__plot_results.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data__plot_results.pkl':
            outer_path = path
    
    # Scenario A: Simple function (no inner paths)
    if outer_path is None:
        print("[ERROR] No outer data file found.")
        sys.exit(1)
    
    # Phase 1: Load outer data and run function
    print(f"[INFO] Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"[INFO] Outer data loaded successfully.")
    print(f"[INFO] Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"[INFO] Number of args: {len(outer_args)}")
    print(f"[INFO] Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Check if this is Scenario B (factory/closure pattern)
    if inner_paths:
        # Scenario B: Factory pattern
        print(f"[INFO] Detected factory/closure pattern with {len(inner_paths)} inner data file(s).")
        
        # Phase 1: Create the operator/closure
        try:
            print("[INFO] Phase 1: Creating operator from outer data...")
            agent_operator = _plot_results(*outer_args, **outer_kwargs)
            print(f"[INFO] Operator created. Type: {type(agent_operator)}")
        except Exception as e:
            print(f"[ERROR] Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print(f"[ERROR] Created operator is not callable. Type: {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            print(f"[INFO] Loading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"[INFO] Inner data loaded. Function: {inner_data.get('func_name', 'unknown')}")
            
            try:
                print("[INFO] Phase 2: Executing operator with inner data...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"[INFO] Execution completed. Result type: {type(result)}")
            except Exception as e:
                print(f"[ERROR] Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Comparison
            try:
                print("[INFO] Comparing results...")
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"[FAIL] Verification failed: {msg}")
                    sys.exit(1)
                else:
                    print(f"[INFO] Inner data verification passed.")
            except Exception as e:
                print(f"[ERROR] Comparison failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    else:
        # Scenario A: Simple function call
        print("[INFO] Detected simple function pattern (no inner data).")
        
        try:
            print("[INFO] Executing _plot_results with outer data...")
            result = _plot_results(*outer_args, **outer_kwargs)
            print(f"[INFO] Execution completed. Result type: {type(result)}")
        except Exception as e:
            print(f"[ERROR] Failed to execute function: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Comparison
        try:
            print("[INFO] Comparing results...")
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"[FAIL] Verification failed: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Comparison failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()